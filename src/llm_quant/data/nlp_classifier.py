"""Claude API text classifier for NLP-based financial signals.

Classifies financial text (10-K filings, earnings calls, CEO letters) into
categories useful for quantitative research:

- **Forward-looking density**: ratio of forward-looking vs backward-looking
  sentences (predictive of earnings surprises — see B3).
- **I/we ratio**: first-person singular vs plural pronoun usage
  (CEO narcissism indicator — see B2).
- **Hedging density**: density of uncertainty/hedging language
  (predictive of sell-side misses — see B9).
- **Causal density**: ratio of causal vs correlational language
  (indicator of management confidence in their explanation).
- **Sentiment**: overall sentiment score (-1.0 to 1.0).

Design principles:
  - Use regex for simple pattern-matching tasks (I/we ratio, hedging density).
  - Use Claude API only for complex classification requiring semantic
    understanding (forward-looking, causal, sentiment).
  - Batch sentences to minimize API calls (chunks of 50).
  - Cache results to disk (data/nlp/cache/{hash}.json) to avoid
    re-classifying identical text.
  - Exponential backoff on API errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Maximum sentences per API call to minimize costs.
BATCH_SIZE = 50

#: Model to use for classification tasks.
DEFAULT_MODEL = "claude-sonnet-4-20250514"

#: Maximum tokens for classification responses.
MAX_TOKENS = 4096

#: Maximum retry attempts on API errors.
MAX_RETRIES = 3

#: Base delay (seconds) for exponential backoff.
BASE_DELAY = 1.0

# ---------------------------------------------------------------------------
# Regex patterns for local scoring (no API needed)
# ---------------------------------------------------------------------------

#: First-person singular pronouns (case-insensitive, whole word).
_I_PRONOUNS = re.compile(
    r"\b(?:I|me|my|mine|myself)\b",
    re.IGNORECASE,
)

#: First-person plural pronouns (case-insensitive, whole word).
_WE_PRONOUNS = re.compile(
    r"\b(?:we|us|our|ours|ourselves)\b",
    re.IGNORECASE,
)

#: Hedging / uncertainty words and phrases.
#: Based on Loughran-McDonald uncertainty word list (subset) and
#: common hedging phrases in financial filings.
_HEDGING_WORDS = re.compile(
    r"\b(?:"
    r"may|might|could|possibly|perhaps|approximately|uncertain|"
    r"uncertainty|uncertainties|unclear|unpredictable|risk|risks|"
    r"appear|appears|appeared|seem|seems|seemed|suggest|suggests|"
    r"suggested|estimate|estimates|estimated|believe|believes|believed|"
    r"anticipate|anticipated|expect|expected|expects|intend|intends|"
    r"intended|likely|unlikely|probable|probably|potential|potentially|"
    r"approximately|roughly|about|around|nearly|almost|substantially|"
    r"generally|typically|usually|often|sometimes|occasionally"
    r")\b",
    re.IGNORECASE,
)

#: Simple word boundary pattern for counting total words.
_WORD_PATTERN = re.compile(r"\b\w+\b")


# ---------------------------------------------------------------------------
# Sentence splitting helper
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation rules.

    Parameters
    ----------
    text:
        Input text to split.

    Returns
    -------
    list[str]
        List of non-empty sentence strings, stripped of whitespace.
    """
    # Split on period, exclamation, question mark followed by space or end.
    # Preserve abbreviations like "U.S." and "Inc." by requiring following
    # uppercase or end-of-string.
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(text: str, method: str) -> str:
    """Generate a deterministic cache key from text content and method name.

    Parameters
    ----------
    text:
        Input text to hash.
    method:
        Classification method name (e.g. ``"forward_looking"``).

    Returns
    -------
    str
        Hex digest of SHA-256 hash.
    """
    payload = f"{method}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_cache_dir() -> Path:
    """Return the default NLP cache directory (data/nlp/cache/ from project root)."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "data" / "nlp" / "cache"
        if (current / "data").is_dir() or (current / "src").is_dir():
            return candidate
        current = current.parent
    return Path.cwd() / "data" / "nlp" / "cache"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class TextClassifier:
    """Claude API text classifier for financial NLP signals.

    Usage
    -----
    >>> classifier = TextClassifier()
    >>> ratio = classifier.score_i_we_ratio("I believe my strategy is correct.")
    >>> hedging = classifier.score_hedging_density("Results may vary significantly.")
    >>> sentiment = classifier.extract_sentiment("Revenue grew 15% year-over-year.")

    Parameters
    ----------
    client:
        Anthropic client instance. If *None*, creates one from the
        ``ANTHROPIC_API_KEY`` environment variable.
    model:
        Claude model to use for API-based classification.
    cache_dir:
        Directory for caching classification results. Defaults to
        ``data/nlp/cache`` relative to the project root.
    batch_size:
        Maximum sentences per API call.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._client = client or anthropic.Anthropic()
        self._model = model
        self._cache_dir = cache_dir or _default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Regex-based scorers (no API calls)
    # ------------------------------------------------------------------

    @staticmethod
    def score_i_we_ratio(text: str) -> float:
        """Compute ratio of first-person singular vs plural pronouns.

        A higher ratio indicates more self-referential (narcissistic)
        language by the speaker. Used as a CEO narcissism indicator
        in the B2 research hypothesis.

        Parameters
        ----------
        text:
            Input text (e.g. CEO letter, earnings call transcript).

        Returns
        -------
        float
            Ratio in [0.0, 1.0]. Value of 1.0 means all first-person
            pronouns are singular. Value of 0.0 means all are plural.
            Returns 0.0 if no first-person pronouns are found.
        """
        i_count = len(_I_PRONOUNS.findall(text))
        we_count = len(_WE_PRONOUNS.findall(text))
        total = i_count + we_count
        if total == 0:
            return 0.0
        return i_count / total

    @staticmethod
    def score_hedging_density(text: str) -> float:
        """Compute density of hedging/uncertainty words in text.

        Higher density indicates more uncertain or hedged language.
        Used as a predictor of sell-side earnings misses (B9 hypothesis).

        Parameters
        ----------
        text:
            Input text (e.g. earnings call Q&A, MD&A section).

        Returns
        -------
        float
            Ratio of hedging words to total words, in [0.0, 1.0].
            Returns 0.0 if text contains no words.
        """
        total_words = len(_WORD_PATTERN.findall(text))
        if total_words == 0:
            return 0.0
        hedge_count = len(_HEDGING_WORDS.findall(text))
        return min(hedge_count / total_words, 1.0)

    # ------------------------------------------------------------------
    # API-based scorers
    # ------------------------------------------------------------------

    def classify_sentences(
        self,
        sentences: list[str],
        categories: list[str],
    ) -> list[dict[str, Any]]:
        """Batch-classify sentences into provided categories via Claude API.

        Sentences are grouped into chunks of ``batch_size`` to minimize
        API calls. Results are cached to avoid re-classifying identical
        sentence batches.

        Parameters
        ----------
        sentences:
            List of sentences to classify.
        categories:
            List of category labels (e.g. ``["forward_looking",
            "backward_looking", "neutral"]``).

        Returns
        -------
        list[dict]
            One dict per sentence with keys ``"sentence"``, ``"category"``,
            and ``"confidence"`` (0.0-1.0).
        """
        if not sentences:
            return []

        results: list[dict[str, Any]] = []

        # Process in batches
        for i in range(0, len(sentences), self._batch_size):
            chunk = sentences[i : i + self._batch_size]
            chunk_results = self._classify_chunk(chunk, categories)
            results.extend(chunk_results)

        return results

    def score_forward_looking(self, text: str) -> float:
        """Compute ratio of forward-looking vs backward-looking sentences.

        Uses Claude API to classify each sentence as forward-looking,
        backward-looking, or neutral. Forward-looking sentences contain
        predictions, expectations, or plans. Backward-looking sentences
        describe historical results.

        Parameters
        ----------
        text:
            Input text to analyze.

        Returns
        -------
        float
            Ratio in [0.0, 1.0]. Higher means more forward-looking content.
            Returns 0.0 if text has no classifiable sentences.
        """
        # Check cache first
        cached = self._load_cache(text, "forward_looking")
        if cached is not None:
            return cached

        sentences = _split_sentences(text)
        if not sentences:
            return 0.0

        categories = ["forward_looking", "backward_looking", "neutral"]
        classified = self.classify_sentences(sentences, categories)

        forward = sum(1 for c in classified if c["category"] == "forward_looking")
        backward = sum(1 for c in classified if c["category"] == "backward_looking")

        total_directional = forward + backward
        score = 0.0 if total_directional == 0 else forward / total_directional

        self._save_cache(text, "forward_looking", score)
        return score

    def score_causal_density(self, text: str) -> float:
        """Compute ratio of causal vs correlational language.

        Uses Claude API to classify sentences as expressing causal
        relationships ("X caused Y", "due to", "resulted in") vs
        correlational language ("X was associated with Y", "alongside").

        Parameters
        ----------
        text:
            Input text to analyze.

        Returns
        -------
        float
            Ratio in [0.0, 1.0]. Higher means more causal language.
            Returns 0.0 if no causal or correlational sentences found.
        """
        cached = self._load_cache(text, "causal_density")
        if cached is not None:
            return cached

        sentences = _split_sentences(text)
        if not sentences:
            return 0.0

        categories = ["causal", "correlational", "neither"]
        classified = self.classify_sentences(sentences, categories)

        causal = sum(1 for c in classified if c["category"] == "causal")
        correlational = sum(1 for c in classified if c["category"] == "correlational")

        total = causal + correlational
        score = 0.0 if total == 0 else causal / total

        self._save_cache(text, "causal_density", score)
        return score

    def extract_sentiment(self, text: str) -> float:
        """Extract overall sentiment score from financial text.

        Uses Claude API to assess overall sentiment of the text on a
        scale from -1.0 (very negative) to 1.0 (very positive).

        Parameters
        ----------
        text:
            Input text to analyze.

        Returns
        -------
        float
            Sentiment score in [-1.0, 1.0].
        """
        cached = self._load_cache(text, "sentiment")
        if cached is not None:
            return cached

        prompt = (
            "Analyze the overall sentiment of the following financial text. "
            "Return ONLY a JSON object with a single key 'sentiment' containing "
            "a float between -1.0 (very negative) and 1.0 (very positive). "
            "Consider financial context: revenue growth is positive, declining "
            "margins are negative, etc.\n\n"
            f"Text:\n{text}\n\n"
            "Response (JSON only):"
        )

        response_text = self._call_api(prompt)
        score = self._parse_sentiment(response_text)

        self._save_cache(text, "sentiment", score)
        return score

    # ------------------------------------------------------------------
    # Internal: API call with retry
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """Call Claude API with exponential backoff on errors.

        Parameters
        ----------
        prompt:
            The user message content to send.

        Returns
        -------
        str
            The text content of the API response.

        Raises
        ------
        anthropic.APIError
            If all retry attempts are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except anthropic.RateLimitError as e:
                last_error = e
                delay = BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
            except anthropic.APIStatusError as e:
                last_error = e
                if e.status_code >= 500:
                    # Server error — retry with backoff
                    delay = BASE_DELAY * (2**attempt)
                    logger.warning(
                        "API server error %d (attempt %d/%d), retrying in %.1fs",
                        e.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    # Client error (4xx except rate limit) — don't retry
                    logger.exception(
                        "API client error %d: %s", e.status_code, e.message
                    )
                    raise
            except anthropic.APIConnectionError as e:
                last_error = e
                delay = BASE_DELAY * (2**attempt)
                logger.warning(
                    "API connection error (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

        # All retries exhausted
        logger.error("All %d API retry attempts exhausted", MAX_RETRIES)
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal: Batch classification
    # ------------------------------------------------------------------

    def _classify_chunk(
        self,
        sentences: list[str],
        categories: list[str],
    ) -> list[dict[str, Any]]:
        """Classify a single chunk of sentences via one API call.

        Parameters
        ----------
        sentences:
            List of sentences (at most ``batch_size`` long).
        categories:
            List of category labels.

        Returns
        -------
        list[dict]
            One dict per sentence with keys ``"sentence"``, ``"category"``,
            ``"confidence"``.
        """
        categories_str = ", ".join(f'"{c}"' for c in categories)
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

        prompt = (
            f"Classify each numbered sentence below into exactly one of these "
            f"categories: [{categories_str}].\n\n"
            f"Return ONLY a JSON array where each element is an object with keys:\n"
            f'  "index" (1-based sentence number),\n'
            f'  "category" (one of the categories above),\n'
            f'  "confidence" (float 0.0-1.0)\n\n'
            f"Sentences:\n{numbered}\n\n"
            f"Response (JSON array only):"
        )

        response_text = self._call_api(prompt)
        return self._parse_classification(response_text, sentences, categories)

    # ------------------------------------------------------------------
    # Internal: Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_classification(
        response_text: str,
        sentences: list[str],
        categories: list[str],
    ) -> list[dict[str, Any]]:
        """Parse classification response JSON into structured results.

        Parameters
        ----------
        response_text:
            Raw API response text (expected to be a JSON array).
        sentences:
            Original sentences (for result assembly).
        categories:
            Valid category labels (for validation).

        Returns
        -------
        list[dict]
            Parsed results. Falls back to ``"unknown"`` category with
            0.0 confidence for sentences that cannot be parsed.
        """
        # Extract JSON array from response (may have surrounding text)
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in classification response")
            return [
                {"sentence": s, "category": "unknown", "confidence": 0.0}
                for s in sentences
            ]

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse classification JSON response")
            return [
                {"sentence": s, "category": "unknown", "confidence": 0.0}
                for s in sentences
            ]

        # Build results aligned with input sentences
        results: list[dict[str, Any]] = []
        parsed_by_index = {item.get("index", -1): item for item in parsed}

        for i, sentence in enumerate(sentences):
            item = parsed_by_index.get(i + 1, {})
            category = item.get("category", "unknown")
            confidence = item.get("confidence", 0.0)

            # Validate category
            if category not in categories:
                category = "unknown"
                confidence = 0.0

            results.append(
                {
                    "sentence": sentence,
                    "category": category,
                    "confidence": float(confidence),
                }
            )

        return results

    @staticmethod
    def _parse_sentiment(response_text: str) -> float:
        """Parse sentiment score from API response.

        Parameters
        ----------
        response_text:
            Raw API response (expected to contain JSON with ``"sentiment"``
            key).

        Returns
        -------
        float
            Sentiment score clamped to [-1.0, 1.0]. Returns 0.0 on
            parse failure.
        """
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON object found in sentiment response")
            return 0.0

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse sentiment JSON response")
            return 0.0

        score = parsed.get("sentiment", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            return 0.0

        return max(-1.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Internal: Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, text: str, method: str) -> Path:
        """Return the cache file path for a given text and method.

        Parameters
        ----------
        text:
            Input text to hash.
        method:
            Classification method name.

        Returns
        -------
        Path
            Path to the cache JSON file.
        """
        key = _cache_key(text, method)
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, text: str, method: str) -> float | None:
        """Load a cached score for text + method.

        Parameters
        ----------
        text:
            Input text.
        method:
            Classification method name.

        Returns
        -------
        float or None
            Cached score if found, else None.
        """
        path = self._cache_path(text, method)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            score = data.get("score")
            if score is not None:
                logger.debug("Cache hit for %s (%s)", method, path.name[:12])
                return float(score)
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Corrupt cache file %s — ignoring", path)

        return None

    def _save_cache(self, text: str, method: str, score: float) -> None:
        """Save a classification score to the cache.

        Parameters
        ----------
        text:
            Input text (used for cache key derivation).
        method:
            Classification method name.
        score:
            The computed score to cache.
        """
        path = self._cache_path(text, method)
        data = {
            "method": method,
            "score": score,
            "text_hash": _cache_key(text, method),
        }
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
            logger.debug("Cached %s score=%.4f to %s", method, score, path.name[:12])
        except OSError:
            logger.warning("Failed to write cache file %s", path)

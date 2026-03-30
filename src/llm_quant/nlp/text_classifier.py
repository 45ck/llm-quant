"""Claude API text classifier for NLP signals.

Classifies 10-K/earnings call sentences using Claude API:
- forward_looking vs backward_looking
- causal vs correlational
- I/we first-person ratio

Results are cached in DuckDB to minimise API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

_DDL_NLP_CLASSIFICATIONS = """
CREATE TABLE IF NOT EXISTS nlp_classifications (
    doc_id        VARCHAR NOT NULL,
    sentence_hash VARCHAR NOT NULL,
    sentence_text VARCHAR NOT NULL,
    forward_looking BOOLEAN,
    causal          BOOLEAN,
    confidence      DOUBLE,
    classified_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, sentence_hash)
)
"""

_BATCH_SIZE = 20

_CLASSIFICATION_PROMPT_TEMPLATE = """\
You are a financial NLP classifier. For each sentence below, output a JSON array \
where each element has:
  - "forward_looking": true if the sentence makes a prediction, forecast, or \
forward-looking statement; false if it describes past or current facts.
  - "causal": true if the sentence asserts a causal relationship (X causes/drives/leads \
to Y); false if it merely describes correlation or co-movement.
  - "confidence": a float in [0, 1] reflecting classification certainty.

Return ONLY the JSON array, no additional text.

Sentences (one per line, numbered):
{numbered_sentences}
"""


def _sentence_hash(sentence: str) -> str:
    """Return a short SHA-256 hex digest for a sentence string."""
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on period / exclamation / question marks."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


class TextClassifier:
    """Classify financial text sentences using Claude API with DuckDB caching.

    Parameters
    ----------
    db_conn:
        An open DuckDB connection.  The classifier will create the
        ``nlp_classifications`` table if it does not exist.
    model:
        Claude model identifier to use for classification.
    """

    def __init__(
        self,
        db_conn: duckdb.DuckDBPyConnection,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._db = db_conn
        self._model = model
        self._db.execute(_DDL_NLP_CLASSIFICATIONS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_sentences(
        self, sentences: list[str], doc_id: str
    ) -> list[dict[str, Any]]:
        """Batch-classify sentences, using DuckDB cache where possible.

        Parameters
        ----------
        sentences:
            List of sentence strings to classify.
        doc_id:
            Identifier for the source document (e.g. ticker + filing year).

        Returns
        -------
        list[dict]
            One dict per sentence with keys:
            ``sentence``, ``forward_looking``, ``causal``, ``confidence``.
        """
        if not sentences:
            return []

        hashes = [_sentence_hash(s) for s in sentences]

        # --- load cached results -----------------------------------------
        cached: dict[str, dict[str, Any]] = {}
        placeholders = ", ".join("?" * len(hashes))
        rows = self._db.execute(
            f"""
            SELECT sentence_hash, forward_looking, causal, confidence
            FROM   nlp_classifications
            WHERE  doc_id = ?
              AND  sentence_hash IN ({placeholders})
            """,
            [doc_id, *hashes],
        ).fetchall()
        for row in rows:
            cached[row[0]] = {
                "forward_looking": row[1],
                "causal": row[2],
                "confidence": row[3],
            }

        # --- identify uncached sentences ---------------------------------
        uncached_indices = [
            i for i, h in enumerate(hashes) if h not in cached
        ]

        # --- call Claude in batches of _BATCH_SIZE -----------------------
        if uncached_indices:
            for batch_start in range(0, len(uncached_indices), _BATCH_SIZE):
                batch_idx = uncached_indices[batch_start : batch_start + _BATCH_SIZE]
                batch_sentences = [sentences[i] for i in batch_idx]
                batch_hashes = [hashes[i] for i in batch_idx]
                results = self._call_claude(batch_sentences)
                self._cache_results(doc_id, batch_sentences, batch_hashes, results)
                for h, r in zip(batch_hashes, results):
                    cached[h] = r

        # --- assemble output in original order ---------------------------
        output: list[dict[str, Any]] = []
        for sentence, h in zip(sentences, hashes):
            entry = cached.get(h, {})
            output.append(
                {
                    "sentence": sentence,
                    "forward_looking": entry.get("forward_looking"),
                    "causal": entry.get("causal"),
                    "confidence": entry.get("confidence"),
                }
            )
        return output

    def get_iwe_ratio(self, text: str) -> float:
        """Return the I/(I+we) first-person pronoun ratio.

        Returns 0.5 when neither pronoun is present.
        """
        lower = text.lower()
        # word-boundary matches to avoid partial matches (e.g. "indeed")
        i_count = len(re.findall(r"\bi\b", lower))
        we_count = len(re.findall(r"\bwe\b", lower))
        total = i_count + we_count
        if total == 0:
            return 0.5
        return i_count / total

    def get_forward_looking_density(self, text: str) -> float:
        """Fraction of sentences classified as forward-looking.

        Requires a ``doc_id``-less cache lookup, so results are fetched
        fresh from the model if not already cached under the text hash.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return 0.0
        doc_id = f"_density_{_sentence_hash(text)}"
        results = self.classify_sentences(sentences, doc_id)
        forward = sum(1 for r in results if r.get("forward_looking"))
        return forward / len(results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_claude(self, sentences: list[str]) -> list[dict[str, Any]]:
        """Call Claude API and parse JSON classification response.

        Returns a list of dicts (one per sentence).  On error, returns
        dicts with ``None`` values so callers can continue.
        """
        import anthropic  # imported lazily to avoid hard dependency at import time

        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        prompt = _CLASSIFICATION_PROMPT_TEMPLATE.format(numbered_sentences=numbered)

        client = anthropic.Anthropic()
        try:
            message = client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got: {type(parsed)}")
            # Pad or truncate to match sentence count
            while len(parsed) < len(sentences):
                parsed.append({"forward_looking": None, "causal": None, "confidence": None})
            return parsed[: len(sentences)]
        except Exception as exc:
            logger.warning("Claude classification failed: %s", exc)
            return [
                {"forward_looking": None, "causal": None, "confidence": None}
                for _ in sentences
            ]

    def _cache_results(
        self,
        doc_id: str,
        sentences: list[str],
        hashes: list[str],
        results: list[dict[str, Any]],
    ) -> None:
        """Insert classification results into DuckDB cache."""
        rows = [
            (
                doc_id,
                h,
                s,
                r.get("forward_looking"),
                r.get("causal"),
                r.get("confidence"),
            )
            for s, h, r in zip(sentences, hashes, results)
        ]
        self._db.executemany(
            """
            INSERT OR REPLACE INTO nlp_classifications
                (doc_id, sentence_hash, sentence_text, forward_looking, causal, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

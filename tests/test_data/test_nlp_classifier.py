"""Tests for Claude API text classifier.

All tests use mocking — no live API calls to Anthropic.
Regex-based scorers are tested with real text samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quant.data.nlp_classifier import (
    BATCH_SIZE,
    TextClassifier,
    _cache_key,
    _split_sentences,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Return a temporary cache directory."""
    cache = tmp_path / "nlp_cache"
    cache.mkdir()
    return cache


@pytest.fixture
def mock_client() -> MagicMock:
    """Return a mock Anthropic client."""
    return MagicMock()


@pytest.fixture
def classifier(mock_client: MagicMock, tmp_cache: Path) -> TextClassifier:
    """Return a TextClassifier with mocked client and temporary cache."""
    return TextClassifier(
        client=mock_client,
        cache_dir=tmp_cache,
    )


def _make_api_response(text: str) -> MagicMock:
    """Create a mock API response with the given text content."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]
    return response


# ---------------------------------------------------------------------------
# Tests: Sentence splitting
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    """Tests for the sentence splitting helper."""

    def test_basic_splitting(self) -> None:
        """Splits on period-space boundaries."""
        text = "Revenue grew 10%. Operating margin expanded. Cash flow was strong."
        result = _split_sentences(text)
        assert len(result) == 3
        assert result[0] == "Revenue grew 10%."

    def test_single_sentence(self) -> None:
        """Returns a single sentence when no split points exist."""
        text = "Revenue grew 10%"
        result = _split_sentences(text)
        assert len(result) == 1

    def test_empty_string(self) -> None:
        """Returns empty list for empty input."""
        assert _split_sentences("") == []

    def test_whitespace_only(self) -> None:
        """Returns empty list for whitespace-only input."""
        assert _split_sentences("   ") == []

    def test_question_and_exclamation(self) -> None:
        """Splits on question marks and exclamation points."""
        text = "Is this a question? Yes it is! Great."
        result = _split_sentences(text)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: I/We ratio scorer (regex-based, no API)
# ---------------------------------------------------------------------------


class TestIWeRatio:
    """Tests for the I/we pronoun ratio scorer."""

    def test_all_singular(self) -> None:
        """Text with only singular pronouns returns 1.0."""
        text = "I believe my strategy is correct. I will execute my plan myself."
        score = TextClassifier.score_i_we_ratio(text)
        assert score == 1.0

    def test_all_plural(self) -> None:
        """Text with only plural pronouns returns 0.0."""
        text = "We believe our strategy is correct. We will execute our plan ourselves."
        score = TextClassifier.score_i_we_ratio(text)
        assert score == 0.0

    def test_mixed_pronouns(self) -> None:
        """Mixed pronouns return a ratio between 0 and 1."""
        text = "I think we should act. My view is that our team will succeed."
        score = TextClassifier.score_i_we_ratio(text)
        assert 0.0 < score < 1.0

    def test_no_pronouns(self) -> None:
        """Text with no first-person pronouns returns 0.0."""
        text = "The company reported strong earnings. Revenue increased 15%."
        score = TextClassifier.score_i_we_ratio(text)
        assert score == 0.0

    def test_empty_text(self) -> None:
        """Empty text returns 0.0."""
        assert TextClassifier.score_i_we_ratio("") == 0.0

    def test_case_insensitive(self) -> None:
        """Pronoun matching is case-insensitive."""
        text1 = "I think MY plan is good."
        text2 = "i think my plan is good."
        assert TextClassifier.score_i_we_ratio(
            text1
        ) == TextClassifier.score_i_we_ratio(text2)

    def test_ceo_narcissism_sample(self) -> None:
        """Realistic CEO letter with high I-density scores high."""
        narcissistic = (
            "I am proud of what I have accomplished. My vision drove our success. "
            "I personally oversaw every decision. I believe my leadership was key."
        )
        collaborative = (
            "We are proud of what our team accomplished. Our vision drove success. "
            "We collectively oversaw every decision. We believe our leadership was key."
        )
        assert TextClassifier.score_i_we_ratio(narcissistic) > 0.7
        assert TextClassifier.score_i_we_ratio(collaborative) < 0.1


# ---------------------------------------------------------------------------
# Tests: Hedging density scorer (regex-based, no API)
# ---------------------------------------------------------------------------


class TestHedgingDensity:
    """Tests for the hedging/uncertainty word density scorer."""

    def test_no_hedging(self) -> None:
        """Text with no hedging words returns 0.0."""
        text = "Revenue grew 15% year-over-year. Margins expanded to 30%."
        score = TextClassifier.score_hedging_density(text)
        assert score == 0.0

    def test_heavy_hedging(self) -> None:
        """Text with many hedging words returns high score."""
        text = (
            "Results may possibly be uncertain. We believe approximately "
            "half could potentially be at risk, though it seems likely."
        )
        score = TextClassifier.score_hedging_density(text)
        assert score > 0.2

    def test_moderate_hedging(self) -> None:
        """Typical financial text has moderate hedging density."""
        text = (
            "Revenue grew 15% year-over-year. We expect continued growth "
            "in the next quarter, though risks remain. Operating margins "
            "may be affected by supply chain uncertainties."
        )
        score = TextClassifier.score_hedging_density(text)
        assert 0.0 < score < 0.5

    def test_empty_text(self) -> None:
        """Empty text returns 0.0."""
        assert TextClassifier.score_hedging_density("") == 0.0

    def test_score_bounded(self) -> None:
        """Score is bounded to [0.0, 1.0]."""
        # Even with dense hedging words, score should not exceed 1.0
        text = "may could perhaps possibly uncertain risk likely"
        score = TextClassifier.score_hedging_density(text)
        assert 0.0 <= score <= 1.0

    def test_earnings_call_qa_sample(self) -> None:
        """Realistic earnings call Q&A text produces reasonable score."""
        confident_answer = (
            "Revenue increased 20%. Gross margin expanded 300 basis points. "
            "Free cash flow doubled compared to the prior year period. "
            "Operating expenses declined 5% through disciplined cost management."
        )
        hedged_answer = (
            "We believe revenue may increase approximately 10-15%. "
            "We anticipate that margins could potentially improve, though "
            "risks and uncertainties remain. It seems likely that growth "
            "will generally be around our estimates, but outcomes are uncertain."
        )
        assert TextClassifier.score_hedging_density(
            confident_answer
        ) < TextClassifier.score_hedging_density(hedged_answer)


# ---------------------------------------------------------------------------
# Tests: Cache key generation
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Tests for cache key generation."""

    def test_deterministic(self) -> None:
        """Same text + method always produces the same key."""
        key1 = _cache_key("hello world", "sentiment")
        key2 = _cache_key("hello world", "sentiment")
        assert key1 == key2

    def test_different_text(self) -> None:
        """Different text produces different keys."""
        key1 = _cache_key("hello", "sentiment")
        key2 = _cache_key("goodbye", "sentiment")
        assert key1 != key2

    def test_different_method(self) -> None:
        """Same text with different method produces different keys."""
        key1 = _cache_key("hello", "sentiment")
        key2 = _cache_key("hello", "forward_looking")
        assert key1 != key2

    def test_key_is_hex_string(self) -> None:
        """Cache key is a valid hex string."""
        key = _cache_key("test", "method")
        assert len(key) == 64  # SHA-256 hex digest
        int(key, 16)  # Should not raise


# ---------------------------------------------------------------------------
# Tests: Cache logic
# ---------------------------------------------------------------------------


class TestCacheLogic:
    """Tests for file-based cache read/write."""

    def test_cache_miss_returns_none(self, classifier: TextClassifier) -> None:
        """_load_cache returns None when no cache file exists."""
        result = classifier._load_cache("uncached text", "sentiment")
        assert result is None

    def test_cache_round_trip(
        self, classifier: TextClassifier, tmp_cache: Path
    ) -> None:
        """Saving and loading a cache entry returns the original score."""
        classifier._save_cache("test text", "sentiment", 0.75)
        result = classifier._load_cache("test text", "sentiment")
        assert result == 0.75

    def test_corrupt_cache_ignored(
        self, classifier: TextClassifier, tmp_cache: Path
    ) -> None:
        """Corrupt cache files are ignored (return None)."""
        key = _cache_key("test", "sentiment")
        cache_path = tmp_cache / f"{key}.json"
        cache_path.write_text("not valid json {{{", encoding="utf-8")

        result = classifier._load_cache("test", "sentiment")
        assert result is None

    def test_cache_prevents_api_call(
        self,
        classifier: TextClassifier,
        mock_client: MagicMock,
    ) -> None:
        """Cached results prevent API calls for forward_looking scorer."""
        # Pre-populate cache
        classifier._save_cache("test text", "forward_looking", 0.6)

        result = classifier.score_forward_looking("test text")
        assert result == 0.6
        mock_client.messages.create.assert_not_called()

    def test_cache_prevents_api_call_sentiment(
        self,
        classifier: TextClassifier,
        mock_client: MagicMock,
    ) -> None:
        """Cached results prevent API calls for sentiment scorer."""
        classifier._save_cache("test text", "sentiment", -0.3)

        result = classifier.extract_sentiment("test text")
        assert result == -0.3
        mock_client.messages.create.assert_not_called()

    def test_cache_prevents_api_call_causal(
        self,
        classifier: TextClassifier,
        mock_client: MagicMock,
    ) -> None:
        """Cached results prevent API calls for causal_density scorer."""
        classifier._save_cache("test text", "causal_density", 0.8)

        result = classifier.score_causal_density("test text")
        assert result == 0.8
        mock_client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Batching logic
# ---------------------------------------------------------------------------


class TestBatching:
    """Tests for sentence batching in classify_sentences."""

    def test_empty_input(self, classifier: TextClassifier) -> None:
        """Empty sentence list returns empty results."""
        result = classifier.classify_sentences([], ["cat_a", "cat_b"])
        assert result == []

    def test_single_batch(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Small input uses a single API call."""
        sentences = ["Revenue grew.", "Margins declined."]
        response_json = json.dumps(
            [
                {"index": 1, "category": "positive", "confidence": 0.9},
                {"index": 2, "category": "negative", "confidence": 0.8},
            ]
        )
        mock_client.messages.create.return_value = _make_api_response(response_json)

        results = classifier.classify_sentences(sentences, ["positive", "negative"])
        assert len(results) == 2
        assert results[0]["category"] == "positive"
        assert results[1]["category"] == "negative"
        assert mock_client.messages.create.call_count == 1

    def test_multiple_batches(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Input exceeding batch_size is split into multiple API calls."""
        # Use a small batch size for testing
        classifier._batch_size = 3
        sentences = [f"Sentence {i}." for i in range(7)]

        def make_response(*_args, **kwargs):
            # Parse the prompt to figure out how many sentences are in this batch
            prompt = kwargs.get("messages", [{}])[0].get("content", "")
            # Count numbered lines in the prompt
            count = len(
                [line for line in prompt.split("\n") if line.strip()[:1].isdigit()]
            )
            items = [
                {"index": j + 1, "category": "neutral", "confidence": 0.5}
                for j in range(count)
            ]
            return _make_api_response(json.dumps(items))

        mock_client.messages.create.side_effect = make_response

        results = classifier.classify_sentences(sentences, ["neutral", "other"])
        assert len(results) == 7
        # With batch_size=3, 7 sentences should require ceil(7/3) = 3 API calls
        assert mock_client.messages.create.call_count == 3

    def test_exact_batch_size(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Input exactly equal to batch_size uses one API call."""
        classifier._batch_size = 5
        sentences = [f"Sentence {i}." for i in range(5)]

        items = [
            {"index": j + 1, "category": "cat_a", "confidence": 0.7} for j in range(5)
        ]
        mock_client.messages.create.return_value = _make_api_response(json.dumps(items))

        results = classifier.classify_sentences(sentences, ["cat_a", "cat_b"])
        assert len(results) == 5
        assert mock_client.messages.create.call_count == 1

    def test_default_batch_size_is_50(self) -> None:
        """Default batch size constant is 50."""
        assert BATCH_SIZE == 50


# ---------------------------------------------------------------------------
# Tests: Classification response parsing
# ---------------------------------------------------------------------------


class TestClassificationParsing:
    """Tests for parsing API classification responses."""

    def test_valid_json_response(self) -> None:
        """Correctly parses well-formed JSON array response."""
        response = json.dumps(
            [
                {"index": 1, "category": "forward_looking", "confidence": 0.95},
                {"index": 2, "category": "backward_looking", "confidence": 0.88},
            ]
        )
        sentences = ["We expect growth.", "Revenue was $1B."]
        categories = ["forward_looking", "backward_looking", "neutral"]

        results = TextClassifier._parse_classification(response, sentences, categories)
        assert len(results) == 2
        assert results[0]["category"] == "forward_looking"
        assert results[0]["confidence"] == 0.95
        assert results[1]["category"] == "backward_looking"

    def test_json_with_surrounding_text(self) -> None:
        """Extracts JSON array even with surrounding explanation text."""
        response = (
            "Here are the classifications:\n"
            '[{"index": 1, "category": "neutral", "confidence": 0.7}]\n'
            "Hope this helps!"
        )
        sentences = ["The sky is blue."]
        categories = ["neutral", "positive"]

        results = TextClassifier._parse_classification(response, sentences, categories)
        assert len(results) == 1
        assert results[0]["category"] == "neutral"

    def test_invalid_json_returns_unknown(self) -> None:
        """Invalid JSON returns unknown category for all sentences."""
        response = "I cannot process this request"
        sentences = ["A sentence.", "Another one."]
        categories = ["cat_a"]

        results = TextClassifier._parse_classification(response, sentences, categories)
        assert len(results) == 2
        assert all(r["category"] == "unknown" for r in results)
        assert all(r["confidence"] == 0.0 for r in results)

    def test_invalid_category_becomes_unknown(self) -> None:
        """Categories not in the allowed list are replaced with unknown."""
        response = json.dumps(
            [{"index": 1, "category": "invented_category", "confidence": 0.9}]
        )
        sentences = ["A sentence."]
        categories = ["cat_a", "cat_b"]

        results = TextClassifier._parse_classification(response, sentences, categories)
        assert results[0]["category"] == "unknown"
        assert results[0]["confidence"] == 0.0

    def test_missing_sentences_in_response(self) -> None:
        """Missing indices in response get unknown category."""
        response = json.dumps([{"index": 1, "category": "cat_a", "confidence": 0.9}])
        sentences = ["Sentence 1.", "Sentence 2."]
        categories = ["cat_a", "cat_b"]

        results = TextClassifier._parse_classification(response, sentences, categories)
        assert len(results) == 2
        assert results[0]["category"] == "cat_a"
        assert results[1]["category"] == "unknown"


# ---------------------------------------------------------------------------
# Tests: Sentiment parsing
# ---------------------------------------------------------------------------


class TestSentimentParsing:
    """Tests for parsing API sentiment responses."""

    def test_valid_sentiment(self) -> None:
        """Correctly parses well-formed sentiment JSON."""
        response = '{"sentiment": 0.75}'
        score = TextClassifier._parse_sentiment(response)
        assert score == 0.75

    def test_negative_sentiment(self) -> None:
        """Correctly parses negative sentiment."""
        response = '{"sentiment": -0.5}'
        score = TextClassifier._parse_sentiment(response)
        assert score == -0.5

    def test_sentiment_clamped_high(self) -> None:
        """Sentiment above 1.0 is clamped."""
        response = '{"sentiment": 1.5}'
        score = TextClassifier._parse_sentiment(response)
        assert score == 1.0

    def test_sentiment_clamped_low(self) -> None:
        """Sentiment below -1.0 is clamped."""
        response = '{"sentiment": -2.0}'
        score = TextClassifier._parse_sentiment(response)
        assert score == -1.0

    def test_invalid_json_returns_zero(self) -> None:
        """Invalid JSON returns 0.0."""
        response = "Not a JSON response at all"
        score = TextClassifier._parse_sentiment(response)
        assert score == 0.0

    def test_missing_sentiment_key(self) -> None:
        """JSON without sentiment key returns 0.0."""
        response = '{"emotion": "happy"}'
        score = TextClassifier._parse_sentiment(response)
        assert score == 0.0

    def test_sentiment_with_surrounding_text(self) -> None:
        """Extracts JSON from response with surrounding text."""
        response = 'Analysis complete: {"sentiment": 0.42} end.'
        score = TextClassifier._parse_sentiment(response)
        assert score == 0.42


# ---------------------------------------------------------------------------
# Tests: Forward-looking scorer (API-based)
# ---------------------------------------------------------------------------


class TestForwardLooking:
    """Tests for the forward-looking sentence ratio scorer."""

    def test_all_forward_looking(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Text with all forward-looking sentences returns 1.0."""
        text = "We expect revenue growth. We anticipate margin expansion."
        response = json.dumps(
            [
                {"index": 1, "category": "forward_looking", "confidence": 0.9},
                {"index": 2, "category": "forward_looking", "confidence": 0.85},
            ]
        )
        mock_client.messages.create.return_value = _make_api_response(response)

        score = classifier.score_forward_looking(text)
        assert score == 1.0

    def test_all_backward_looking(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Text with all backward-looking sentences returns 0.0."""
        text = "Revenue was $1B last year. Margins contracted in Q3."
        response = json.dumps(
            [
                {"index": 1, "category": "backward_looking", "confidence": 0.9},
                {"index": 2, "category": "backward_looking", "confidence": 0.85},
            ]
        )
        mock_client.messages.create.return_value = _make_api_response(response)

        score = classifier.score_forward_looking(text)
        assert score == 0.0

    def test_mixed_sentences(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Mixed text returns ratio of forward/(forward+backward)."""
        text = "Revenue was $1B. We expect growth. The sky is blue."
        response = json.dumps(
            [
                {"index": 1, "category": "backward_looking", "confidence": 0.9},
                {"index": 2, "category": "forward_looking", "confidence": 0.85},
                {"index": 3, "category": "neutral", "confidence": 0.7},
            ]
        )
        mock_client.messages.create.return_value = _make_api_response(response)

        score = classifier.score_forward_looking(text)
        assert score == 0.5  # 1 forward / (1 forward + 1 backward)

    def test_empty_text(self, classifier: TextClassifier) -> None:
        """Empty text returns 0.0 without API call."""
        score = classifier.score_forward_looking("")
        assert score == 0.0

    def test_result_is_cached(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Forward-looking score is cached after first computation."""
        text = "We expect growth."
        response = json.dumps(
            [{"index": 1, "category": "forward_looking", "confidence": 0.9}]
        )
        mock_client.messages.create.return_value = _make_api_response(response)

        # First call — hits API
        score1 = classifier.score_forward_looking(text)
        assert mock_client.messages.create.call_count == 1

        # Second call — hits cache
        score2 = classifier.score_forward_looking(text)
        assert mock_client.messages.create.call_count == 1  # No additional API call
        assert score1 == score2


# ---------------------------------------------------------------------------
# Tests: Causal density scorer (API-based)
# ---------------------------------------------------------------------------


class TestCausalDensity:
    """Tests for the causal vs correlational language scorer."""

    def test_all_causal(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Text with all causal language returns 1.0."""
        text = "Rate hikes caused the slowdown. Strong demand led to price increases."
        response = json.dumps(
            [
                {"index": 1, "category": "causal", "confidence": 0.9},
                {"index": 2, "category": "causal", "confidence": 0.85},
            ]
        )
        mock_client.messages.create.return_value = _make_api_response(response)

        score = classifier.score_causal_density(text)
        assert score == 1.0

    def test_empty_text(self, classifier: TextClassifier) -> None:
        """Empty text returns 0.0 without API call."""
        score = classifier.score_causal_density("")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: Sentiment scorer (API-based)
# ---------------------------------------------------------------------------


class TestSentiment:
    """Tests for the sentiment extraction scorer."""

    def test_positive_sentiment(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Positive financial text returns positive score."""
        text = "Revenue grew 15% year-over-year. Record quarterly earnings."
        mock_client.messages.create.return_value = _make_api_response(
            '{"sentiment": 0.8}'
        )

        score = classifier.extract_sentiment(text)
        assert score == 0.8

    def test_negative_sentiment(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Negative financial text returns negative score."""
        text = "Revenue declined 20%. Margins collapsed. Guidance withdrawn."
        mock_client.messages.create.return_value = _make_api_response(
            '{"sentiment": -0.7}'
        )

        score = classifier.extract_sentiment(text)
        assert score == -0.7


# ---------------------------------------------------------------------------
# Tests: Error handling and retries
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for API error handling with exponential backoff."""

    def test_rate_limit_retry(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Rate limit errors trigger retries with backoff."""
        import anthropic as anthropic_mod

        # First call raises rate limit, second succeeds
        mock_client.messages.create.side_effect = [
            anthropic_mod.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            ),
            _make_api_response('{"sentiment": 0.5}'),
        ]

        with patch("llm_quant.data.nlp_classifier.time.sleep"):
            score = classifier.extract_sentiment("Test text")

        assert score == 0.5
        assert mock_client.messages.create.call_count == 2

    def test_server_error_retry(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """5xx server errors trigger retries."""
        import anthropic as anthropic_mod

        mock_response = MagicMock(status_code=500, headers={})
        mock_client.messages.create.side_effect = [
            anthropic_mod.APIStatusError(
                message="server error",
                response=mock_response,
                body=None,
            ),
            _make_api_response('{"sentiment": 0.3}'),
        ]

        with patch("llm_quant.data.nlp_classifier.time.sleep"):
            score = classifier.extract_sentiment("Test text")

        assert score == 0.3

    def test_client_error_no_retry(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """4xx client errors (except rate limit) are not retried."""
        import anthropic as anthropic_mod

        mock_response = MagicMock(status_code=400, headers={})
        mock_client.messages.create.side_effect = anthropic_mod.APIStatusError(
            message="bad request",
            response=mock_response,
            body=None,
        )

        with pytest.raises(anthropic_mod.APIStatusError):
            classifier.extract_sentiment("Test text")

        assert mock_client.messages.create.call_count == 1

    def test_connection_error_retry(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Connection errors trigger retries."""
        import anthropic as anthropic_mod

        mock_client.messages.create.side_effect = [
            anthropic_mod.APIConnectionError(request=MagicMock()),
            _make_api_response('{"sentiment": 0.1}'),
        ]

        with patch("llm_quant.data.nlp_classifier.time.sleep"):
            score = classifier.extract_sentiment("Test text")

        assert score == 0.1

    def test_all_retries_exhausted(
        self, classifier: TextClassifier, mock_client: MagicMock
    ) -> None:
        """Raises error when all retry attempts are exhausted."""
        import anthropic as anthropic_mod

        mock_client.messages.create.side_effect = anthropic_mod.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        with (
            patch("llm_quant.data.nlp_classifier.time.sleep"),
            pytest.raises(anthropic_mod.RateLimitError),
        ):
            classifier.extract_sentiment("Test text")

        # Should have attempted MAX_RETRIES times
        assert mock_client.messages.create.call_count == 3


# ---------------------------------------------------------------------------
# Tests: Integration (mocked end-to-end)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end tests with mocked API calls."""

    def test_full_pipeline_forward_looking(
        self,
        classifier: TextClassifier,
        mock_client: MagicMock,
    ) -> None:
        """Full pipeline: text -> split -> classify -> score -> cache."""
        text = (
            "Revenue grew 10% last year. We expect continued growth next quarter. "
            "The board approved a dividend increase."
        )
        response = json.dumps(
            [
                {"index": 1, "category": "backward_looking", "confidence": 0.9},
                {"index": 2, "category": "forward_looking", "confidence": 0.95},
                {"index": 3, "category": "neutral", "confidence": 0.7},
            ]
        )
        mock_client.messages.create.return_value = _make_api_response(response)

        score = classifier.score_forward_looking(text)
        assert score == 0.5  # 1 forward / (1 forward + 1 backward)

        # Verify cached
        cached = classifier._load_cache(text, "forward_looking")
        assert cached == 0.5

    def test_regex_scorers_need_no_api(
        self,
        classifier: TextClassifier,
        mock_client: MagicMock,
    ) -> None:
        """Regex scorers work without any API calls."""
        text = (
            "I believe we may see uncertain results. My team expects growth "
            "but our outlook could potentially change. I remain cautious."
        )

        i_we = classifier.score_i_we_ratio(text)
        hedging = classifier.score_hedging_density(text)

        assert 0.0 < i_we < 1.0
        assert hedging > 0.0
        mock_client.messages.create.assert_not_called()

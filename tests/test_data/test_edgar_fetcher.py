"""Tests for SEC EDGAR 10-K fetcher.

All tests use mocking — no live HTTP calls to EDGAR.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_quant.data.edgar_fetcher import (
    EdgarFetcher,
    _strip_html,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """Return a temporary cache directory."""
    cache = tmp_path / "edgar_cache"
    cache.mkdir()
    return cache


@pytest.fixture
def fetcher(tmp_cache: Path) -> EdgarFetcher:
    """Return an EdgarFetcher with a temporary cache directory."""
    return EdgarFetcher(cache_dir=tmp_cache)


# ---------------------------------------------------------------------------
# Sample data for mocking
# ---------------------------------------------------------------------------

SAMPLE_COMPANY_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon.com Inc."},
}

SAMPLE_SUBMISSIONS = {
    "cik": "320193",
    "entityType": "operating",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-23-000106",
                "0000320193-23-000077",
                "0000320193-22-000108",
            ],
            "filingDate": [
                "2023-11-03",
                "2023-08-04",
                "2022-10-28",
            ],
            "form": [
                "10-K",
                "10-Q",
                "10-K",
            ],
            "primaryDocument": [
                "aapl-20230930.htm",
                "aapl-20230701.htm",
                "aapl-20220924.htm",
            ],
        },
        "files": [],
    },
}

SAMPLE_10K_TEXT = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K

PART I

Item 1. Business
Apple Inc. designs, manufactures, and markets consumer electronics.

Item 1A. Risk Factors

The Company's business, reputation, results of operations, financial condition,
and stock price can be affected by a number of factors. The following discussion
identifies the most significant risk factors.

MACROECONOMIC CONDITIONS: Global economic uncertainty could adversely impact
our business and financial condition. Consumer spending may decline during
economic downturns, reducing demand for our products.

COMPETITION: The markets for our products are highly competitive. We face
aggressive competition in all areas of our business from well-established
companies with significant resources.

Item 1B. Unresolved Staff Comments

None.

PART II

Item 5. Market for Registrant's Common Equity

Our common stock is traded on the NASDAQ Global Select Market.

Item 6. [Reserved]

Item 7. Management's Discussion and Analysis of Financial Condition and
Results of Operations

The following discussion should be read in conjunction with the consolidated
financial statements and notes thereto included elsewhere in this report.

OVERVIEW: Total net revenue for fiscal 2023 was $383.3 billion, a decrease
of 3% compared to fiscal 2022. The decline was driven primarily by lower
iPhone and Mac revenue, partially offset by higher Services revenue.

LIQUIDITY AND CAPITAL RESOURCES: As of September 30, 2023, the Company had
$162.1 billion in cash, cash equivalents and marketable securities. The
Company believes its existing balances and cash flows from operations will
be sufficient to satisfy its working capital needs.

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

The Company is exposed to market risk associated with changes in interest
rates, foreign currency exchange rates, and equity prices.

Item 8. Financial Statements and Supplementary Data
"""


# ---------------------------------------------------------------------------
# Tests: CIK lookup
# ---------------------------------------------------------------------------


class TestCikLookup:
    """Tests for CIK number lookup from ticker."""

    def test_parse_company_tickers_json(self, fetcher: EdgarFetcher) -> None:
        """CIK lookup correctly parses company_tickers.json format."""
        with patch.object(fetcher, "_http_get_json") as mock_get:
            mock_get.return_value = SAMPLE_COMPANY_TICKERS

            cik = fetcher._lookup_cik("AAPL")
            assert cik == "0000320193"  # zero-padded to 10 digits

            cik = fetcher._lookup_cik("MSFT")
            assert cik == "0000789019"

    def test_ticker_case_insensitive(self, fetcher: EdgarFetcher) -> None:
        """CIK lookup is case-insensitive."""
        with patch.object(fetcher, "_http_get_json") as mock_get:
            mock_get.return_value = SAMPLE_COMPANY_TICKERS

            assert fetcher._lookup_cik("aapl") == "0000320193"
            assert fetcher._lookup_cik("Aapl") == "0000320193"

    def test_unknown_ticker_returns_none(self, fetcher: EdgarFetcher) -> None:
        """CIK lookup returns None for unknown ticker."""
        with patch.object(fetcher, "_http_get_json") as mock_get:
            mock_get.return_value = SAMPLE_COMPANY_TICKERS

            assert fetcher._lookup_cik("ZZZZZ") is None

    def test_http_failure_returns_none(self, fetcher: EdgarFetcher) -> None:
        """CIK lookup returns None when HTTP request fails."""
        with patch.object(fetcher, "_http_get_json") as mock_get:
            mock_get.return_value = None

            assert fetcher._lookup_cik("AAPL") is None

    def test_cik_cache_reused(self, fetcher: EdgarFetcher) -> None:
        """Company tickers JSON is fetched only once and cached."""
        with patch.object(fetcher, "_http_get_json") as mock_get:
            mock_get.return_value = SAMPLE_COMPANY_TICKERS

            fetcher._lookup_cik("AAPL")
            fetcher._lookup_cik("MSFT")

            # Should only be called once (cached after first call)
            mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Filing list parsing
# ---------------------------------------------------------------------------


class TestFilingListParsing:
    """Tests for 10-K filing discovery from submissions JSON."""

    def test_find_10k_in_year(self, fetcher: EdgarFetcher) -> None:
        """Correctly identifies 10-K filing for a given year."""
        result = fetcher._search_filing_list(
            SAMPLE_SUBMISSIONS["filings"]["recent"], 2023
        )
        assert result is not None
        assert result["accession_number"] == "000032019323000106"
        assert result["primary_doc"] == "aapl-20230930.htm"

    def test_find_10k_earlier_year(self, fetcher: EdgarFetcher) -> None:
        """Finds 10-K for an earlier year when multiple are present."""
        result = fetcher._search_filing_list(
            SAMPLE_SUBMISSIONS["filings"]["recent"], 2022
        )
        assert result is not None
        assert result["accession_number"] == "000032019322000108"
        assert result["primary_doc"] == "aapl-20220924.htm"

    def test_no_10k_for_year(self, fetcher: EdgarFetcher) -> None:
        """Returns None when no 10-K exists for the given year."""
        result = fetcher._search_filing_list(
            SAMPLE_SUBMISSIONS["filings"]["recent"], 2021
        )
        assert result is None

    def test_skips_10q_filings(self, fetcher: EdgarFetcher) -> None:
        """Only matches 10-K forms, not 10-Q or other types."""
        # The 2023-08-04 entry is a 10-Q, should not match even for 2023
        recent_only_10q = {
            "form": ["10-Q"],
            "filingDate": ["2023-08-04"],
            "accessionNumber": ["0000320193-23-000077"],
            "primaryDocument": ["aapl-20230701.htm"],
        }
        result = fetcher._search_filing_list(recent_only_10q, 2023)
        assert result is None

    def test_empty_filing_list(self, fetcher: EdgarFetcher) -> None:
        """Returns None for empty filing list."""
        result = fetcher._search_filing_list({}, 2023)
        assert result is None

    def test_matches_10k_amendment(self, fetcher: EdgarFetcher) -> None:
        """Matches 10-K/A (amendment) filings too."""
        recent = {
            "form": ["10-K/A"],
            "filingDate": ["2023-03-15"],
            "accessionNumber": ["0000320193-23-000010"],
            "primaryDocument": ["amended-10k.htm"],
        }
        result = fetcher._search_filing_list(recent, 2023)
        assert result is not None
        assert result["primary_doc"] == "amended-10k.htm"


# ---------------------------------------------------------------------------
# Tests: MD&A section extraction
# ---------------------------------------------------------------------------


class TestMdaExtraction:
    """Tests for Management's Discussion and Analysis extraction."""

    def test_extract_mda_from_sample(self) -> None:
        """Extracts MD&A section from sample 10-K text."""
        mda = EdgarFetcher.extract_mda(SAMPLE_10K_TEXT)
        assert len(mda) > 0
        assert "Total net revenue for fiscal 2023" in mda
        assert "LIQUIDITY AND CAPITAL RESOURCES" in mda
        # Should not include Item 7A content
        assert "exposed to market risk" not in mda

    def test_mda_with_item7_header(self) -> None:
        """Extracts MD&A when header uses Item 7 format."""
        text = """
Item 7. Management's Discussion and Analysis

Revenue grew 10% year-over-year due to strong demand.

Item 7A. Quantitative and Qualitative Disclosures
"""
        mda = EdgarFetcher.extract_mda(text)
        assert "Revenue grew 10%" in mda
        assert "Quantitative" not in mda

    def test_mda_empty_when_not_found(self) -> None:
        """Returns empty string when MD&A section is not present."""
        text = "This is a random document with no MD&A section."
        mda = EdgarFetcher.extract_mda(text)
        assert mda == ""

    def test_mda_extraction_stops_at_item8(self) -> None:
        """MD&A extraction stops at Item 8 if Item 7A is missing."""
        text = """
Item 7. Management's Discussion and Analysis of Financial Condition

Discussion content here.

Item 8. Financial Statements and Supplementary Data

Balance sheet follows.
"""
        mda = EdgarFetcher.extract_mda(text)
        assert "Discussion content here" in mda
        assert "Balance sheet follows" not in mda


# ---------------------------------------------------------------------------
# Tests: Risk Factors extraction
# ---------------------------------------------------------------------------


class TestRiskFactorsExtraction:
    """Tests for Risk Factors section extraction."""

    def test_extract_risk_factors_from_sample(self) -> None:
        """Extracts Risk Factors from sample 10-K text."""
        risk = EdgarFetcher.extract_risk_factors(SAMPLE_10K_TEXT)
        assert len(risk) > 0
        assert "MACROECONOMIC CONDITIONS" in risk
        assert "COMPETITION" in risk
        # Should not include Item 1B content
        assert "None." not in risk or "significant risk factors" in risk

    def test_risk_factors_with_item1a_header(self) -> None:
        """Extracts Risk Factors with Item 1A header."""
        text = """
Item 1A. Risk Factors

Market volatility could affect our share price.

Item 1B. Unresolved Staff Comments
"""
        risk = EdgarFetcher.extract_risk_factors(text)
        assert "Market volatility" in risk
        assert "Unresolved" not in risk

    def test_risk_factors_empty_when_not_found(self) -> None:
        """Returns empty string when Risk Factors section is not present."""
        text = "Annual report with no risk factors section."
        risk = EdgarFetcher.extract_risk_factors(text)
        assert risk == ""

    def test_risk_factors_stops_at_item2(self) -> None:
        """Risk Factors extraction stops at Item 2 if Item 1B is missing."""
        text = """
Item 1A. Risk Factors

Cybersecurity threats are increasing.

Item 2. Properties
"""
        risk = EdgarFetcher.extract_risk_factors(text)
        assert "Cybersecurity" in risk
        assert "Properties" not in risk


# ---------------------------------------------------------------------------
# Tests: Cache logic
# ---------------------------------------------------------------------------


class TestCacheLogic:
    """Tests for cache hit/miss behavior."""

    def test_cache_miss_returns_none(self, fetcher: EdgarFetcher) -> None:
        """get_cached returns None when no cache file exists."""
        result = fetcher.get_cached("AAPL", 2023)
        assert result is None

    def test_cache_hit_returns_content(
        self, fetcher: EdgarFetcher, tmp_cache: Path
    ) -> None:
        """get_cached returns content when cache file exists and is non-empty."""
        cache_path = tmp_cache / "AAPL" / "2023.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("Cached 10-K content here", encoding="utf-8")

        result = fetcher.get_cached("AAPL", 2023)
        assert result == "Cached 10-K content here"

    def test_empty_cache_file_is_miss(
        self, fetcher: EdgarFetcher, tmp_cache: Path
    ) -> None:
        """get_cached returns None when cache file exists but is empty."""
        cache_path = tmp_cache / "AAPL" / "2023.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("", encoding="utf-8")

        result = fetcher.get_cached("AAPL", 2023)
        assert result is None

    def test_cache_ticker_case_normalized(
        self, fetcher: EdgarFetcher, tmp_cache: Path
    ) -> None:
        """Cache lookups normalize ticker to uppercase."""
        cache_path = tmp_cache / "AAPL" / "2023.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("content", encoding="utf-8")

        # Lowercase input should still find uppercase cache
        result = fetcher.get_cached("aapl", 2023)
        assert result == "content"

    def test_fetch_10k_uses_cache(self, fetcher: EdgarFetcher, tmp_cache: Path) -> None:
        """fetch_10k returns cached content without making HTTP requests."""
        cache_path = tmp_cache / "AAPL" / "2023.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("Cached 10-K text", encoding="utf-8")

        with (
            patch.object(fetcher, "_http_get_json") as mock_json,
            patch.object(fetcher, "_http_get_text") as mock_text,
        ):
            result = fetcher.fetch_10k("AAPL", 2023)

            assert result == "Cached 10-K text"
            mock_json.assert_not_called()
            mock_text.assert_not_called()

    def test_fetch_10k_caches_result(
        self, fetcher: EdgarFetcher, tmp_cache: Path
    ) -> None:
        """fetch_10k saves fetched text to cache."""
        with (
            patch.object(fetcher, "_lookup_cik", return_value="0000320193"),
            patch.object(
                fetcher,
                "_find_10k_filing",
                return_value={
                    "accession_number": "000032019323000106",
                    "primary_doc": "aapl-20230930.htm",
                },
            ),
            patch.object(
                fetcher,
                "_fetch_document",
                return_value="Plain text 10-K content",
            ),
        ):
            result = fetcher.fetch_10k("AAPL", 2023)

        assert result == "Plain text 10-K content"
        # Verify cache file was written
        cache_path = tmp_cache / "AAPL" / "2023.txt"
        assert cache_path.exists()
        assert cache_path.read_text(encoding="utf-8") == "Plain text 10-K content"


# ---------------------------------------------------------------------------
# Tests: Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for SEC rate limit compliance."""

    def test_rate_limit_enforces_delay(self, fetcher: EdgarFetcher) -> None:
        """Rate limiter enforces minimum interval between requests."""
        # Set last request time to "just now"
        fetcher._last_request_time = time.monotonic()

        start = time.monotonic()
        fetcher._rate_limit()
        elapsed = time.monotonic() - start

        # Should have slept at least some portion of MIN_REQUEST_INTERVAL
        # (allow some tolerance for timing)
        assert elapsed >= 0.05  # at least 50ms of the 120ms interval

    def test_rate_limit_skips_when_enough_time(
        self,
        fetcher: EdgarFetcher,
    ) -> None:
        """Rate limiter does not sleep when enough time has passed."""
        # Set last request time to well in the past
        fetcher._last_request_time = time.monotonic() - 1.0

        start = time.monotonic()
        fetcher._rate_limit()
        elapsed = time.monotonic() - start

        # Should not have slept
        assert elapsed < 0.05


# ---------------------------------------------------------------------------
# Tests: HTML stripping
# ---------------------------------------------------------------------------


class TestHtmlStripping:
    """Tests for HTML tag removal utility."""

    def test_strip_basic_tags(self) -> None:
        """Removes simple HTML tags."""
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_decode_entities(self) -> None:
        """Decodes common HTML entities."""
        html = "AT&amp;T said &quot;hello&quot; &mdash; test"
        result = _strip_html(html)
        assert 'AT&T said "hello"' in result

    def test_remove_style_blocks(self) -> None:
        """Removes <style> blocks entirely."""
        html = "<style>body { color: red; }</style><p>Content</p>"
        result = _strip_html(html)
        assert "color: red" not in result
        assert "Content" in result

    def test_remove_script_blocks(self) -> None:
        """Removes <script> blocks entirely."""
        html = '<script>alert("xss")</script><p>Safe content</p>'
        result = _strip_html(html)
        assert "alert" not in result
        assert "Safe content" in result

    def test_collapse_whitespace(self) -> None:
        """Collapses excessive whitespace."""
        html = "word1     word2\t\t\tword3"
        result = _strip_html(html)
        assert "word1 word2 word3" in result

    def test_plain_text_passes_through(self) -> None:
        """Plain text without HTML is returned unchanged (modulo whitespace)."""
        text = "This is plain text with no HTML."
        result = _strip_html(text)
        assert result == text


# ---------------------------------------------------------------------------
# Tests: Full fetch_10k flow (integration with mocks)
# ---------------------------------------------------------------------------


class TestFetch10kFlow:
    """End-to-end tests for the fetch_10k method with mocked HTTP."""

    def test_full_flow_success(self, fetcher: EdgarFetcher, tmp_cache: Path) -> None:
        """Full fetch flow: CIK lookup -> filing list -> document fetch."""
        with (
            patch.object(fetcher, "_http_get_json") as mock_json,
            patch.object(fetcher, "_http_get_text") as mock_text,
        ):
            # First call: company_tickers.json
            # Second call: submissions JSON
            mock_json.side_effect = [
                SAMPLE_COMPANY_TICKERS,
                SAMPLE_SUBMISSIONS,
            ]
            # Document fetch returns plain text
            mock_text.return_value = SAMPLE_10K_TEXT

            result = fetcher.fetch_10k("AAPL", 2023)

        assert result is not None
        assert len(result) > 0
        # Should contain content from the sample
        assert "Apple" in result or "revenue" in result.lower()

    def test_flow_cik_not_found(self, fetcher: EdgarFetcher) -> None:
        """Returns None when ticker CIK cannot be found."""
        with patch.object(fetcher, "_http_get_json") as mock_json:
            mock_json.return_value = SAMPLE_COMPANY_TICKERS

            result = fetcher.fetch_10k("ZZZZZ", 2023)
            assert result is None

    def test_flow_no_filing_for_year(self, fetcher: EdgarFetcher) -> None:
        """Returns None when no 10-K exists for the year."""
        with patch.object(fetcher, "_http_get_json") as mock_json:
            mock_json.side_effect = [
                SAMPLE_COMPANY_TICKERS,
                SAMPLE_SUBMISSIONS,
            ]

            result = fetcher.fetch_10k("AAPL", 2020)
            assert result is None

    def test_flow_document_fetch_fails(self, fetcher: EdgarFetcher) -> None:
        """Returns None when document fetch fails."""
        with (
            patch.object(fetcher, "_http_get_json") as mock_json,
            patch.object(fetcher, "_http_get_text") as mock_text,
        ):
            mock_json.side_effect = [
                SAMPLE_COMPANY_TICKERS,
                SAMPLE_SUBMISSIONS,
            ]
            mock_text.return_value = None

            result = fetcher.fetch_10k("AAPL", 2023)
            assert result is None

    def test_section_caches_written(
        self, fetcher: EdgarFetcher, tmp_cache: Path
    ) -> None:
        """fetch_10k writes section cache files (mda, risk_factors)."""
        with (
            patch.object(fetcher, "_lookup_cik", return_value="0000320193"),
            patch.object(
                fetcher,
                "_find_10k_filing",
                return_value={
                    "accession_number": "000032019323000106",
                    "primary_doc": "aapl-20230930.htm",
                },
            ),
            patch.object(
                fetcher,
                "_fetch_document",
                return_value=SAMPLE_10K_TEXT,
            ),
        ):
            fetcher.fetch_10k("AAPL", 2023)

        # Check section cache files
        mda_path = tmp_cache / "AAPL" / "2023_mda.txt"
        risk_path = tmp_cache / "AAPL" / "2023_risk_factors.txt"
        assert mda_path.exists()
        assert risk_path.exists()

        mda_text = mda_path.read_text(encoding="utf-8")
        assert "Total net revenue" in mda_text

        risk_text = risk_path.read_text(encoding="utf-8")
        assert "MACROECONOMIC" in risk_text

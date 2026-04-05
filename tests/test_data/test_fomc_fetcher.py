"""Tests for FOMC transcript fetcher and hedging language scorer.

All tests use sample text — no live HTTP calls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from llm_quant.data.fomc_fetcher import (
    CONFIDENCE_WORDS,
    HEDGING_WORDS,
    FomcFetcher,
    FomcTranscript,
    HedgingScore,
    HedgingScorer,
    _scores_to_dataframe,
    score_all_transcripts,
)

# ---------------------------------------------------------------------------
# Sample texts for testing
# ---------------------------------------------------------------------------

SAMPLE_FOMC_PARAGRAPH = (
    "The Committee judges that the outlook for economic activity is uncertain. "
    "Risks to the outlook are roughly balanced, but inflation could remain "
    "somewhat elevated. The labor market is strong and robust, with solid "
    "job gains. We are cautious about the path forward and will gradually "
    "adjust the stance of monetary policy as conditions may warrant. "
    "It is possible that further tightening might be appropriate, "
    "perhaps at a later date, depending on the data. "
    "The risks are clearly contingent on how the economy evolves."
)

PURE_HEDGING_TEXT = (
    "uncertain uncertainty approximately roughly might could may "
    "possible possibly perhaps somewhat relatively likely unlikely "
    "risk risks cautious cautiously gradual gradually tentative "
    "tentatively contingent depends"
)

PURE_CONFIDENCE_TEXT = (
    "confident strong robust solid firmly certainly clearly decisive determined"
)

NEUTRAL_TEXT = (
    "The committee met today to discuss the economic outlook. "
    "Members reviewed recent data on employment and inflation. "
    "The next meeting is scheduled for six weeks from now."
)


# ---------------------------------------------------------------------------
# HedgingScorer tests
# ---------------------------------------------------------------------------


class TestHedgingScorer:
    """Tests for the HedgingScorer class."""

    def setup_method(self) -> None:
        self.scorer = HedgingScorer()

    def test_hedging_word_detection(self) -> None:
        """All hedging words are detected when present."""
        score = self.scorer.score(PURE_HEDGING_TEXT)
        # All 24 hedging words should be detected
        assert score.hedging_count == len(HEDGING_WORDS)
        assert score.confidence_count == 0
        # With only hedging words, score should be 1000 per-mille
        assert score.hedging_score == pytest.approx(1000.0)
        assert score.net_hedging_score == pytest.approx(1000.0)

    def test_confidence_word_detection(self) -> None:
        """All confidence words are detected when present."""
        score = self.scorer.score(PURE_CONFIDENCE_TEXT)
        assert score.confidence_count == len(CONFIDENCE_WORDS)
        assert score.hedging_count == 0
        # With only confidence words, confidence score should be 1000
        assert score.confidence_score == pytest.approx(1000.0)
        assert score.net_hedging_score == pytest.approx(-1000.0)

    def test_net_score_computation(self) -> None:
        """Net score = hedging_score - confidence_score."""
        score = self.scorer.score(SAMPLE_FOMC_PARAGRAPH)
        expected_net = score.hedging_score - score.confidence_score
        assert score.net_hedging_score == pytest.approx(expected_net)

    def test_fomc_paragraph_scoring(self) -> None:
        """Score a realistic FOMC-like paragraph."""
        score = self.scorer.score(SAMPLE_FOMC_PARAGRAPH)

        # Should find hedging words
        assert score.hedging_count > 0
        # Should find confidence words ("strong", "robust", "solid", "clearly")
        assert score.confidence_count > 0
        # Total words should be reasonable
        assert score.total_words > 50
        # Per-mille rates should be positive
        assert score.hedging_score > 0
        assert score.confidence_score > 0
        # Hedging should dominate in this dovish paragraph
        assert score.net_hedging_score > 0

    def test_hedging_words_in_paragraph(self) -> None:
        """Verify specific hedging words are found in the sample paragraph."""
        score = self.scorer.score(SAMPLE_FOMC_PARAGRAPH)
        top = score.top_hedging_words
        # These words appear in SAMPLE_FOMC_PARAGRAPH
        assert "uncertain" in top
        assert "risks" in top
        assert "could" in top
        assert "cautious" in top
        assert "gradually" in top
        assert "possible" in top
        assert "might" in top
        assert "perhaps" in top
        assert "contingent" in top

    def test_confidence_words_in_paragraph(self) -> None:
        """Confidence words are correctly counted in the sample paragraph."""
        score = self.scorer.score(SAMPLE_FOMC_PARAGRAPH)
        # "strong", "robust", "solid", "clearly" are in the sample
        assert score.confidence_count >= 4

    def test_empty_text(self) -> None:
        """Empty text returns zero scores."""
        score = self.scorer.score("")
        assert score.hedging_count == 0
        assert score.confidence_count == 0
        assert score.total_words == 0
        assert score.hedging_score == 0.0
        assert score.confidence_score == 0.0
        assert score.net_hedging_score == 0.0
        assert score.top_hedging_words == {}

    def test_no_hedging_words(self) -> None:
        """Text with no hedging or confidence words scores zero."""
        score = self.scorer.score(NEUTRAL_TEXT)
        assert score.hedging_count == 0
        assert score.confidence_count == 0
        assert score.total_words > 0
        assert score.hedging_score == 0.0
        assert score.confidence_score == 0.0
        assert score.net_hedging_score == 0.0

    def test_all_hedging_words(self) -> None:
        """Text composed entirely of hedging words scores 1000 per-mille."""
        score = self.scorer.score(PURE_HEDGING_TEXT)
        assert score.hedging_score == pytest.approx(1000.0)
        assert score.total_words == len(HEDGING_WORDS)

    def test_case_insensitivity(self) -> None:
        """Scoring is case-insensitive."""
        score = self.scorer.score("UNCERTAIN Risks PERHAPS Cautious")
        assert score.hedging_count == 4

    def test_date_assignment(self) -> None:
        """Score date is correctly assigned when provided."""
        d = date(2024, 3, 20)
        score = self.scorer.score("uncertain risks", score_date=d)
        assert score.date == d

    def test_date_default(self) -> None:
        """Score date defaults to date.min when not provided."""
        score = self.scorer.score("uncertain")
        assert score.date == date.min

    def test_top_hedging_words_sorted(self) -> None:
        """Top hedging words are sorted by frequency descending."""
        text = "risk risk risk uncertain uncertain might"
        score = self.scorer.score(text)
        words = list(score.top_hedging_words.items())
        # risk(3) should come before uncertain(2) should come before might(1)
        assert words[0] == ("risk", 3)
        assert words[1] == ("uncertain", 2)
        assert words[2] == ("might", 1)

    def test_per_mille_rate_calculation(self) -> None:
        """Per-mille rate is hedging_count / total_words * 1000."""
        text = "uncertain possible the economy is growing well"
        score = self.scorer.score(text)
        expected_rate = score.hedging_count / score.total_words * 1000
        assert score.hedging_score == pytest.approx(expected_rate)

    def test_custom_word_lists(self) -> None:
        """Custom hedging and confidence word lists are respected."""
        custom_scorer = HedgingScorer(
            hedging_words=["foo", "bar"],
            confidence_words=["baz"],
        )
        score = custom_scorer.score("foo bar baz qux")
        assert score.hedging_count == 2
        assert score.confidence_count == 1
        assert score.total_words == 4


# ---------------------------------------------------------------------------
# HedgingScore dataclass tests
# ---------------------------------------------------------------------------


class TestHedgingScore:
    """Tests for the HedgingScore dataclass."""

    def test_dataclass_fields(self) -> None:
        """HedgingScore has all required fields."""
        score = HedgingScore(
            date=date(2024, 1, 31),
            hedging_count=50,
            confidence_count=10,
            total_words=5000,
            hedging_score=10.0,
            confidence_score=2.0,
            net_hedging_score=8.0,
            top_hedging_words={"risk": 15, "uncertain": 10},
        )
        assert score.date == date(2024, 1, 31)
        assert score.hedging_count == 50
        assert score.confidence_count == 10
        assert score.total_words == 5000
        assert score.hedging_score == 10.0
        assert score.confidence_score == 2.0
        assert score.net_hedging_score == 8.0
        assert score.top_hedging_words == {"risk": 15, "uncertain": 10}

    def test_default_top_hedging_words(self) -> None:
        """top_hedging_words defaults to empty dict."""
        score = HedgingScore(
            date=date(2024, 1, 31),
            hedging_count=0,
            confidence_count=0,
            total_words=0,
            hedging_score=0.0,
            confidence_score=0.0,
            net_hedging_score=0.0,
        )
        assert score.top_hedging_words == {}


# ---------------------------------------------------------------------------
# FomcTranscript dataclass tests
# ---------------------------------------------------------------------------


class TestFomcTranscript:
    """Tests for the FomcTranscript dataclass."""

    def test_dataclass_fields(self) -> None:
        """FomcTranscript has all required fields."""
        t = FomcTranscript(
            date=date(2024, 3, 20),
            text="Full transcript text here.",
            url="https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20240320.pdf",
            speaker="Powell",
        )
        assert t.date == date(2024, 3, 20)
        assert t.text == "Full transcript text here."
        assert "FOMCpresconf20240320" in t.url
        assert t.speaker == "Powell"


# ---------------------------------------------------------------------------
# DataFrame conversion tests
# ---------------------------------------------------------------------------


class TestScoresToDataFrame:
    """Tests for _scores_to_dataframe."""

    def test_conversion(self) -> None:
        """Scores are correctly converted to a Polars DataFrame."""
        scores = [
            HedgingScore(
                date=date(2024, 1, 31),
                hedging_count=50,
                confidence_count=10,
                total_words=5000,
                hedging_score=10.0,
                confidence_score=2.0,
                net_hedging_score=8.0,
                top_hedging_words={"risk": 15},
            ),
            HedgingScore(
                date=date(2024, 3, 20),
                hedging_count=30,
                confidence_count=20,
                total_words=4000,
                hedging_score=7.5,
                confidence_score=5.0,
                net_hedging_score=2.5,
                top_hedging_words={"uncertain": 8},
            ),
        ]
        df = _scores_to_dataframe(scores)
        assert len(df) == 2
        assert set(df.columns) == {
            "date",
            "hedging_count",
            "confidence_count",
            "total_words",
            "hedging_score",
            "confidence_score",
            "net_hedging_score",
        }
        assert df["date"][0] == date(2024, 1, 31)
        assert df["hedging_score"][1] == pytest.approx(7.5)

    def test_empty_scores(self) -> None:
        """Empty scores list produces empty DataFrame."""
        df = _scores_to_dataframe([])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# FomcFetcher cache tests (no HTTP)
# ---------------------------------------------------------------------------


class TestFomcFetcherCache:
    """Tests for FomcFetcher cache operations (no network calls)."""

    def test_get_cached_empty_dir(self, tmp_path: Path) -> None:
        """get_cached returns empty list for empty cache dir."""
        fetcher = FomcFetcher(cache_dir=tmp_path)
        result = fetcher.get_cached()
        assert result == []

    def test_get_cached_with_files(self, tmp_path: Path) -> None:
        """get_cached reads transcript files correctly."""
        # Create sample cached transcripts
        (tmp_path / "2024-01-31.txt").write_text(
            "The economy is uncertain.", encoding="utf-8"
        )
        (tmp_path / "2024-03-20.txt").write_text(
            "Growth is strong and robust.", encoding="utf-8"
        )

        fetcher = FomcFetcher(cache_dir=tmp_path)
        result = fetcher.get_cached()

        assert len(result) == 2
        assert result[0].date == date(2024, 1, 31)
        assert result[1].date == date(2024, 3, 20)
        assert "uncertain" in result[0].text
        assert result[0].speaker == "Powell"

    def test_get_cached_skips_empty_files(self, tmp_path: Path) -> None:
        """get_cached skips empty transcript files."""
        (tmp_path / "2024-01-31.txt").write_text("", encoding="utf-8")
        (tmp_path / "2024-03-20.txt").write_text("Some content.", encoding="utf-8")

        fetcher = FomcFetcher(cache_dir=tmp_path)
        result = fetcher.get_cached()
        assert len(result) == 1
        assert result[0].date == date(2024, 3, 20)

    def test_get_cached_skips_non_date_files(self, tmp_path: Path) -> None:
        """get_cached skips files that don't match date pattern."""
        (tmp_path / "README.txt").write_text("Not a transcript.", encoding="utf-8")
        (tmp_path / "2024-03-20.txt").write_text("Real transcript.", encoding="utf-8")

        fetcher = FomcFetcher(cache_dir=tmp_path)
        result = fetcher.get_cached()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Batch scoring tests (no HTTP)
# ---------------------------------------------------------------------------


class TestBatchScoring:
    """Tests for score_all_transcripts batch processing."""

    def test_score_all_with_cached(self, tmp_path: Path) -> None:
        """score_all_transcripts scores all cached files."""
        cache_dir = tmp_path / "transcripts"
        cache_dir.mkdir()
        output_path = tmp_path / "scores.parquet"

        # Create sample transcripts
        (cache_dir / "2024-01-31.txt").write_text(
            "The outlook is uncertain and risks are elevated. "
            "We are cautious about the path forward.",
            encoding="utf-8",
        )
        (cache_dir / "2024-03-20.txt").write_text(
            "The economy is strong and robust. We are confident "
            "in the solid growth trajectory.",
            encoding="utf-8",
        )

        scores = score_all_transcripts(cache_dir=cache_dir, output_path=output_path)

        assert len(scores) == 2
        assert scores[0].date == date(2024, 1, 31)
        assert scores[1].date == date(2024, 3, 20)

        # First transcript is more hedging
        assert scores[0].net_hedging_score > scores[1].net_hedging_score

        # Parquet file written
        assert output_path.exists()

        # Read back and verify
        import polars as pl

        df = pl.read_parquet(output_path)
        assert len(df) == 2
        assert "hedging_score" in df.columns
        assert "net_hedging_score" in df.columns

    def test_score_all_empty_cache(self, tmp_path: Path) -> None:
        """score_all_transcripts returns empty list for empty cache."""
        scores = score_all_transcripts(cache_dir=tmp_path)
        assert scores == []


# ---------------------------------------------------------------------------
# Chair mapping tests
# ---------------------------------------------------------------------------


class TestChairMapping:
    """Tests for chair identification by date."""

    def test_bernanke_era(self) -> None:
        """Bernanke is identified for 2012-2014 dates."""
        from llm_quant.data.fomc_fetcher import _chair_for_date

        assert _chair_for_date(date(2012, 6, 20)) == "Bernanke"
        assert _chair_for_date(date(2013, 12, 18)) == "Bernanke"

    def test_yellen_era(self) -> None:
        """Yellen is identified for 2014-2018 dates."""
        from llm_quant.data.fomc_fetcher import _chair_for_date

        assert _chair_for_date(date(2014, 3, 19)) == "Yellen"
        assert _chair_for_date(date(2017, 12, 13)) == "Yellen"

    def test_powell_era(self) -> None:
        """Powell is identified for 2018+ dates."""
        from llm_quant.data.fomc_fetcher import _chair_for_date

        assert _chair_for_date(date(2018, 3, 21)) == "Powell"
        assert _chair_for_date(date(2024, 3, 20)) == "Powell"


# ---------------------------------------------------------------------------
# Parse presconf dates tests
# ---------------------------------------------------------------------------


class TestParsePresconfDates:
    """Tests for HTML parsing of press conference dates."""

    def test_parse_dates_from_html(self) -> None:
        """Press conference dates are extracted from calendar HTML."""
        html = """
        <a href="/monetarypolicy/fomcpresconf20240131.htm">Press Conference</a>
        <a href="/monetarypolicy/fomcpresconf20240320.htm">Press Conference</a>
        <a href="/monetarypolicy/fomcpresconf20240612.htm">Press Conference</a>
        """
        fetcher = FomcFetcher(cache_dir=Path())
        dates = fetcher._parse_presconf_dates(html, 2024)
        assert len(dates) == 3
        assert dates[0] == date(2024, 1, 31)
        assert dates[1] == date(2024, 3, 20)
        assert dates[2] == date(2024, 6, 12)

    def test_parse_filters_by_year(self) -> None:
        """Only dates matching the target year are returned."""
        html = """
        <a href="/monetarypolicy/fomcpresconf20230920.htm">Press Conference</a>
        <a href="/monetarypolicy/fomcpresconf20240131.htm">Press Conference</a>
        """
        fetcher = FomcFetcher(cache_dir=Path())
        dates = fetcher._parse_presconf_dates(html, 2024)
        assert len(dates) == 1
        assert dates[0] == date(2024, 1, 31)

    def test_parse_deduplicates(self) -> None:
        """Duplicate links are deduplicated."""
        html = """
        <a href="/monetarypolicy/fomcpresconf20240131.htm">Press Conference</a>
        <a href="/monetarypolicy/fomcpresconf20240131.htm">Transcript</a>
        """
        fetcher = FomcFetcher(cache_dir=Path())
        dates = fetcher._parse_presconf_dates(html, 2024)
        assert len(dates) == 1

    def test_parse_no_matches(self) -> None:
        """Empty list when no press conference links found."""
        html = "<html><body>No press conferences here.</body></html>"
        fetcher = FomcFetcher(cache_dir=Path())
        dates = fetcher._parse_presconf_dates(html, 2024)
        assert len(dates) == 0

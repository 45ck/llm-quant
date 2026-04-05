"""Fetch FOMC press conference transcripts and score hedging language density.

Downloads press conference transcripts from the Federal Reserve website,
caches them as text files, and computes hedging/confidence language scores.

Transcripts are available since 2012 (when Bernanke started regular press
conferences). PDFs are downloaded from the Fed's media center and converted
to text using pypdf.

Data source:
    https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf
"""

from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FED_BASE_URL = "https://www.federalreserve.gov"
TRANSCRIPT_URL_TEMPLATE = FED_BASE_URL + "/mediacenter/files/FOMCpresconf{date_str}.pdf"
CALENDAR_URL = FED_BASE_URL + "/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL_TEMPLATE = FED_BASE_URL + "/monetarypolicy/fomchistorical{year}.htm"

USER_AGENT = "llm-quant/1.0 (research; non-commercial)"

# Hedging and confidence word lists — fixed, not to be optimised.
HEDGING_WORDS: list[str] = [
    "uncertain",
    "uncertainty",
    "approximately",
    "roughly",
    "might",
    "could",
    "may",
    "possible",
    "possibly",
    "perhaps",
    "somewhat",
    "relatively",
    "likely",
    "unlikely",
    "risk",
    "risks",
    "cautious",
    "cautiously",
    "gradual",
    "gradually",
    "tentative",
    "tentatively",
    "contingent",
    "depends",
]

CONFIDENCE_WORDS: list[str] = [
    "confident",
    "strong",
    "robust",
    "solid",
    "firmly",
    "certainly",
    "clearly",
    "decisive",
    "determined",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FomcTranscript:
    """A single FOMC press conference transcript."""

    date: date
    text: str
    url: str
    speaker: str


@dataclass
class HedgingScore:
    """Hedging language score for a single transcript."""

    date: date
    hedging_count: int
    confidence_count: int
    total_words: int
    hedging_score: float
    confidence_score: float
    net_hedging_score: float
    top_hedging_words: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chair mapping — used to tag transcripts by speaker
# ---------------------------------------------------------------------------

#: FOMC Chair terms for tagging transcripts.
_CHAIR_TERMS: list[tuple[date, date, str]] = [
    (date(2006, 2, 1), date(2014, 1, 31), "Bernanke"),
    (date(2014, 2, 3), date(2018, 2, 3), "Yellen"),
    (date(2018, 2, 5), date(2026, 12, 31), "Powell"),
]


def _chair_for_date(meeting_date: date) -> str:
    """Return the Fed Chair name for a given meeting date."""
    for start, end, name in _CHAIR_TERMS:
        if start <= meeting_date <= end:
            return name
    return "Unknown"


# ---------------------------------------------------------------------------
# FomcFetcher — download and cache transcripts
# ---------------------------------------------------------------------------


class FomcFetcher:
    """Fetch and cache FOMC press conference transcripts.

    Usage
    -----
    >>> fetcher = FomcFetcher()
    >>> transcripts = fetcher.fetch_transcripts(start_year=2020)
    >>> cached = fetcher.get_cached()

    Parameters
    ----------
    cache_dir:
        Directory for cached transcript text files. Defaults to
        ``data/nlp/fomc/transcripts/`` from the project root.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_transcripts(self, start_year: int = 2012) -> list[FomcTranscript]:
        """Fetch all available press conference transcripts.

        Discovers FOMC meeting dates from the Fed calendar pages, downloads
        transcript PDFs, extracts text, and caches as ``{YYYY-MM-DD}.txt``.

        Parameters
        ----------
        start_year:
            Earliest year to fetch. Press conferences started in 2011
            (Bernanke era). Defaults to 2012.

        Returns
        -------
        list[FomcTranscript]
            All successfully fetched transcripts, sorted by date.
        """
        meeting_dates = self._discover_meeting_dates(start_year)
        logger.info(
            "Discovered %d FOMC press conference dates from %d onward",
            len(meeting_dates),
            start_year,
        )

        transcripts: list[FomcTranscript] = []
        for meeting_date in meeting_dates:
            cache_file = self._cache_dir / f"{meeting_date.isoformat()}.txt"

            # Use cache if available
            if cache_file.exists():
                text = cache_file.read_text(encoding="utf-8")
                if text.strip():
                    url = TRANSCRIPT_URL_TEMPLATE.format(
                        date_str=meeting_date.strftime("%Y%m%d")
                    )
                    transcripts.append(
                        FomcTranscript(
                            date=meeting_date,
                            text=text,
                            url=url,
                            speaker=_chair_for_date(meeting_date),
                        )
                    )
                    continue

            # Fetch and cache
            transcript = self._fetch_single(meeting_date)
            if transcript is not None:
                cache_file.write_text(transcript.text, encoding="utf-8")
                transcripts.append(transcript)
                logger.info("Fetched transcript for %s", meeting_date)
            else:
                logger.warning("No transcript available for %s", meeting_date)

        transcripts.sort(key=lambda t: t.date)
        logger.info("Total transcripts fetched/cached: %d", len(transcripts))
        return transcripts

    def get_cached(self) -> list[FomcTranscript]:
        """Read all cached transcripts from disk.

        Returns
        -------
        list[FomcTranscript]
            Cached transcripts sorted by date. Empty list if no cache.
        """
        transcripts: list[FomcTranscript] = []
        for txt_file in sorted(self._cache_dir.glob("*.txt")):
            stem = txt_file.stem  # e.g. "2024-03-20"
            try:
                meeting_date = date.fromisoformat(stem)
            except ValueError:
                logger.warning("Skipping non-date file: %s", txt_file.name)
                continue

            text = txt_file.read_text(encoding="utf-8")
            if not text.strip():
                continue

            url = TRANSCRIPT_URL_TEMPLATE.format(
                date_str=meeting_date.strftime("%Y%m%d")
            )
            transcripts.append(
                FomcTranscript(
                    date=meeting_date,
                    text=text,
                    url=url,
                    speaker=_chair_for_date(meeting_date),
                )
            )

        return transcripts

    # ------------------------------------------------------------------
    # Internal: discover meeting dates
    # ------------------------------------------------------------------

    def _discover_meeting_dates(self, start_year: int) -> list[date]:
        """Discover FOMC meeting dates with press conferences.

        Scrapes the Fed calendar and historical pages to find meeting dates
        that have associated press conference links.
        """
        current_year = date.today().year
        all_dates: list[date] = []

        for year in range(start_year, current_year + 1):
            try:
                dates = self._scrape_year_dates(year)
                all_dates.extend(dates)
                logger.debug("Found %d press conference dates for %d", len(dates), year)
            except Exception:
                logger.exception("Failed to scrape FOMC dates for year %d", year)

        all_dates.sort()
        return all_dates

    def _scrape_year_dates(self, year: int) -> list[date]:
        """Scrape press conference dates for a single year.

        For 2021+, uses the main calendar page. For older years, uses
        the historical year page.
        """
        if year >= 2021:
            url = CALENDAR_URL
        else:
            url = HISTORICAL_URL_TEMPLATE.format(year=year)

        html = self._http_get(url)
        if html is None:
            return []

        return self._parse_presconf_dates(html, year)

    def _parse_presconf_dates(self, html: str, year: int) -> list[date]:
        """Extract press conference dates from calendar HTML.

        Looks for links matching the pattern:
            fomcpresconf{YYYYMMDD}.htm
        """
        # Match presconf links: fomcpresconf20250319.htm
        pattern = r"fomcpresconf(\d{8})\.htm"
        matches = re.findall(pattern, html)

        dates: list[date] = []
        for date_str in matches:
            try:
                meeting_date = date(
                    int(date_str[:4]),
                    int(date_str[4:6]),
                    int(date_str[6:8]),
                )
                if meeting_date.year == year:
                    dates.append(meeting_date)
            except ValueError:
                logger.warning("Invalid date in presconf link: %s", date_str)

        # Deduplicate (same link may appear multiple times)
        return sorted(set(dates))

    # ------------------------------------------------------------------
    # Internal: fetch single transcript
    # ------------------------------------------------------------------

    def _fetch_single(self, meeting_date: date) -> FomcTranscript | None:
        """Download and extract text from a single transcript PDF."""
        date_str = meeting_date.strftime("%Y%m%d")
        url = TRANSCRIPT_URL_TEMPLATE.format(date_str=date_str)

        pdf_bytes = self._http_get_bytes(url)
        if pdf_bytes is None:
            return None

        text = self._extract_text_from_pdf(pdf_bytes)
        if not text or not text.strip():
            logger.warning("Empty text extracted from PDF for %s", meeting_date)
            return None

        return FomcTranscript(
            date=meeting_date,
            text=text,
            url=url,
            speaker=_chair_for_date(meeting_date),
        )

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from a PDF byte stream using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.exception("pypdf not installed — run: pip install pypdf")
            return ""

        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            pages_text: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            return "\n".join(pages_text)
        except Exception:
            logger.exception("Failed to extract text from PDF")
            return ""

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _http_get(self, url: str) -> str | None:
        """Fetch a URL and return the response as a string."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            logger.exception("HTTP GET failed: %s", url)
            return None

    def _http_get_bytes(self, url: str) -> bytes | None:
        """Fetch a URL and return the response as bytes."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception:
            logger.exception("HTTP GET (bytes) failed: %s", url)
            return None


# ---------------------------------------------------------------------------
# HedgingScorer — score transcripts for hedging language
# ---------------------------------------------------------------------------


class HedgingScorer:
    """Score FOMC transcripts for hedging vs. confidence language density.

    The hedging score measures the per-mille (parts per thousand) rate
    of hedging language in a transcript. A higher net score means more
    hedging relative to confidence — historically associated with dovish
    uncertainty about the economic outlook.

    Usage
    -----
    >>> scorer = HedgingScorer()
    >>> score = scorer.score("The outlook is uncertain and risks are elevated.")
    >>> print(score.net_hedging_score)

    Parameters
    ----------
    hedging_words:
        Override the default hedging word list.
    confidence_words:
        Override the default confidence word list.
    """

    def __init__(
        self,
        hedging_words: list[str] | None = None,
        confidence_words: list[str] | None = None,
    ) -> None:
        self._hedging_words = hedging_words or HEDGING_WORDS
        self._confidence_words = confidence_words or CONFIDENCE_WORDS

    def score(self, text: str, score_date: date | None = None) -> HedgingScore:
        """Score a text for hedging language density.

        Parameters
        ----------
        text:
            The transcript text to score.
        score_date:
            Date to assign to the score. Defaults to ``date.min`` if
            not provided (caller should set from transcript metadata).

        Returns
        -------
        HedgingScore
            Complete hedging analysis including word counts and rates.
        """
        if score_date is None:
            score_date = date.min

        # Tokenise: lowercase, split on non-alpha characters
        words = re.findall(r"[a-z]+", text.lower())
        total_words = len(words)

        if total_words == 0:
            return HedgingScore(
                date=score_date,
                hedging_count=0,
                confidence_count=0,
                total_words=0,
                hedging_score=0.0,
                confidence_score=0.0,
                net_hedging_score=0.0,
                top_hedging_words={},
            )

        # Count hedging words
        hedging_counts: dict[str, int] = {}
        for word in self._hedging_words:
            count = words.count(word)
            if count > 0:
                hedging_counts[word] = count

        hedging_total = sum(hedging_counts.values())

        # Count confidence words
        confidence_total = 0
        for word in self._confidence_words:
            confidence_total += words.count(word)

        # Compute per-mille rates
        hedging_rate = hedging_total / total_words * 1000
        confidence_rate = confidence_total / total_words * 1000
        net_rate = hedging_rate - confidence_rate

        # Sort top hedging words by frequency (descending)
        top_words = dict(
            sorted(hedging_counts.items(), key=lambda x: x[1], reverse=True)
        )

        return HedgingScore(
            date=score_date,
            hedging_count=hedging_total,
            confidence_count=confidence_total,
            total_words=total_words,
            hedging_score=hedging_rate,
            confidence_score=confidence_rate,
            net_hedging_score=net_rate,
            top_hedging_words=top_words,
        )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def score_all_transcripts(
    cache_dir: Path | None = None,
    output_path: Path | None = None,
) -> list[HedgingScore]:
    """Score all cached FOMC transcripts for hedging language.

    Parameters
    ----------
    cache_dir:
        Transcript cache directory. Defaults to standard location.
    output_path:
        Path for the output parquet file. Defaults to
        ``data/nlp/fomc/hedging_scores.parquet``.

    Returns
    -------
    list[HedgingScore]
        All scores sorted by date.
    """
    fetcher = FomcFetcher(cache_dir=cache_dir)
    transcripts = fetcher.get_cached()

    if not transcripts:
        logger.warning("No cached transcripts found — run fetch_transcripts() first")
        return []

    scorer = HedgingScorer()
    scores: list[HedgingScore] = []

    for transcript in transcripts:
        score = scorer.score(transcript.text, score_date=transcript.date)
        scores.append(score)
        logger.info(
            "Scored %s: hedging=%.1f, confidence=%.1f, net=%.1f (words=%d)",
            transcript.date,
            score.hedging_score,
            score.confidence_score,
            score.net_hedging_score,
            score.total_words,
        )

    scores.sort(key=lambda s: s.date)

    # Save as Polars DataFrame to parquet
    if scores:
        df = _scores_to_dataframe(scores)
        if output_path is None:
            output_path = _default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        logger.info("Saved %d hedging scores to %s", len(scores), output_path)

    return scores


def _scores_to_dataframe(scores: list[HedgingScore]) -> pl.DataFrame:
    """Convert a list of HedgingScore to a Polars DataFrame."""
    return pl.DataFrame(
        {
            "date": [s.date for s in scores],
            "hedging_count": [s.hedging_count for s in scores],
            "confidence_count": [s.confidence_count for s in scores],
            "total_words": [s.total_words for s in scores],
            "hedging_score": [s.hedging_score for s in scores],
            "confidence_score": [s.confidence_score for s in scores],
            "net_hedging_score": [s.net_hedging_score for s in scores],
        }
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Return the default FOMC transcript cache directory."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "data" / "nlp" / "fomc" / "transcripts"
        if (current / "data").is_dir():
            return candidate
        current = current.parent
    return Path.cwd() / "data" / "nlp" / "fomc" / "transcripts"


def _default_output_path() -> Path:
    """Return the default output path for hedging scores parquet."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "data" / "nlp" / "fomc" / "hedging_scores.parquet"
        if (current / "data").is_dir():
            return candidate
        current = current.parent
    return Path.cwd() / "data" / "nlp" / "fomc" / "hedging_scores.parquet"

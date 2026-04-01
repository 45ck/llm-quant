"""FOMC transcript fetcher and hedging language scorer.

Fetches Federal Reserve FOMC meeting minutes from the Fed website,
scores hedging language density, and caches results in DuckDB.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date

import duckdb
import polars as pl
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DDL_FOMC_MINUTES = """
CREATE TABLE IF NOT EXISTS fomc_minutes (
    meeting_date    DATE NOT NULL PRIMARY KEY,
    raw_text        VARCHAR,
    hedging_score   DOUBLE,
    word_count      INTEGER,
    fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Hedging / uncertainty vocabulary
HEDGING_WORDS: list[str] = [
    "uncertain",
    "uncertainty",
    "risk",
    "could",
    "may",
    "might",
    "if",
    "whether",
    "potential",
    "possible",
    "concern",
    "cautious",
]

_HEDGING_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in HEDGING_WORDS) + r")\b",
    re.IGNORECASE,
)

_REQUEST_DELAY_SECS = 0.5

_HEADERS = {
    "User-Agent": "llm-quant research bot (research@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Fed publishes minutes at URLs of the form:
#   https://www.federalreserve.gov/monetarypolicy/fomcminutes{YYYYMMDD}.htm
# Meeting months: Jan, Mar, Apr/May, Jun, Jul, Sep, Oct/Nov, Dec
# We build candidate URLs by trying the most common meeting days per month.
_FED_MINUTES_URL_TEMPLATE = (
    "https://www.federalreserve.gov/monetarypolicy/fomcminutes{date}.htm"
)

# Typical FOMC meeting days by month (approximate; real dates vary by year).
_TYPICAL_MEETING_DAYS: dict[int, list[int]] = {
    1: [28, 29, 30, 31, 27, 26],
    2: [],  # no Feb meeting in most years
    3: [19, 20, 21, 22, 18, 17],
    4: [],  # no standalone Apr meeting (but sometimes Apr/May)
    5: [1, 2, 3, 4, 5, 6, 7],
    6: [11, 12, 13, 14, 15, 10],
    7: [29, 30, 31, 28, 27],
    8: [],
    9: [17, 18, 19, 20, 16],
    10: [],
    11: [1, 2, 3, 4, 5, 6, 7],
    12: [10, 11, 12, 13, 14, 9],
}


def _get(url: str) -> requests.Response | None:
    """GET with error handling and courtesy delay."""
    try:
        time.sleep(_REQUEST_DELAY_SECS)
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.debug("HTTP request failed for %s: %s", url, exc)
        return None


def _candidate_urls(year: int, month: int) -> list[str]:
    """Generate candidate Fed minutes URLs for the given year/month."""
    days = _TYPICAL_MEETING_DAYS.get(month, [])
    urls = []
    for day in days:
        date_str = f"{year}{month:02d}{day:02d}"
        urls.append(_FED_MINUTES_URL_TEMPLATE.format(date=date_str))
    return urls


def _extract_text_from_html(html: str) -> str:
    """Strip HTML tags and return clean text."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


class FomcFetcher:
    """Fetch and score FOMC meeting minutes.

    Parameters
    ----------
    db_conn:
        An open DuckDB connection.  The fetcher will create the
        ``fomc_minutes`` table if it does not exist.
    """

    FED_MINUTES_BASE = "https://www.federalreserve.gov/monetarypolicy/fomcminutes"

    def __init__(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        self._db = db_conn
        self._db.execute(_DDL_FOMC_MINUTES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_minutes(self, year: int, month: int) -> str | None:
        """Fetch FOMC minutes for the given meeting year/month.

        Checks DuckDB cache first.  On cache miss, tries candidate URLs
        derived from typical FOMC meeting schedules, saves the result, and
        returns the raw text.

        Parameters
        ----------
        year:
            Four-digit calendar year (e.g. ``2024``).
        month:
            Meeting month as integer 1-12.

        Returns
        -------
        str | None
            Raw text of the minutes, or ``None`` if unavailable.
        """
        # Try cache with approximate meeting date (first of month as key)
        cached = self._load_cache_by_month(year, month)
        if cached is not None:
            return cached

        logger.info("Fetching FOMC minutes: %d-%02d", year, month)

        for url in _candidate_urls(year, month):
            resp = _get(url)
            if resp is None:
                continue
            text = _extract_text_from_html(resp.text)
            if not text:
                continue
            # Extract approximate meeting date from URL
            meeting_date = self._parse_date_from_url(url)
            score = self.score_hedging_language(text)
            word_count = len(text.split())
            self._save_cache(meeting_date, text, score, word_count)
            logger.info(
                "Fetched FOMC minutes %s: %d words, hedging=%.4f",
                meeting_date,
                word_count,
                score,
            )
            return text

        logger.warning("No FOMC minutes found for %d-%02d", year, month)
        return None

    def score_hedging_language(self, text: str) -> float:
        """Return hedging word density (hedging_word_count / total_words).

        Parameters
        ----------
        text:
            Plain text to score.

        Returns
        -------
        float
            Value in ``[0, 1]``.  Returns ``0.0`` for empty/whitespace input.
        """
        words = text.split()
        if not words:
            return 0.0
        hedging_count = len(_HEDGING_PATTERN.findall(text))
        return hedging_count / len(words)

    def get_hedging_series(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Return a time series of hedging scores between two dates.

        Parameters
        ----------
        start_date:
            ISO date string ``"YYYY-MM-DD"`` (inclusive).
        end_date:
            ISO date string ``"YYYY-MM-DD"`` (inclusive).

        Returns
        -------
        pl.DataFrame
            Columns: ``meeting_date`` (Date), ``hedging_score`` (Float64),
            ``word_count`` (Int32).  Sorted ascending by meeting date.
        """
        rows = self._db.execute(
            """
            SELECT meeting_date, hedging_score, word_count
            FROM   fomc_minutes
            WHERE  meeting_date BETWEEN ? AND ?
            ORDER  BY meeting_date ASC
            """,
            [start_date, end_date],
        ).fetchall()

        if not rows:
            return pl.DataFrame(
                schema={
                    "meeting_date": pl.Date,
                    "hedging_score": pl.Float64,
                    "word_count": pl.Int32,
                }
            )

        meeting_dates = [r[0] for r in rows]
        hedging_scores = [r[1] for r in rows]
        word_counts = [r[2] for r in rows]

        return pl.DataFrame(
            {
                "meeting_date": meeting_dates,
                "hedging_score": hedging_scores,
                "word_count": word_counts,
            }
        ).with_columns(pl.col("meeting_date").cast(pl.Date))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache_by_month(self, year: int, month: int) -> str | None:
        """Return cached raw text for any meeting in year/month."""
        row = self._db.execute(
            """
            SELECT raw_text FROM fomc_minutes
            WHERE  YEAR(meeting_date)  = ?
              AND  MONTH(meeting_date) = ?
            LIMIT 1
            """,
            [year, month],
        ).fetchone()
        return row[0] if row else None

    def _save_cache(
        self,
        meeting_date: date,
        raw_text: str,
        hedging_score: float,
        word_count: int,
    ) -> None:
        self._db.execute(
            """
            INSERT OR REPLACE INTO fomc_minutes
                (meeting_date, raw_text, hedging_score, word_count)
            VALUES (?, ?, ?, ?)
            """,
            [meeting_date.isoformat(), raw_text, hedging_score, word_count],
        )

    @staticmethod
    def _parse_date_from_url(url: str) -> date:
        """Extract meeting date from Fed minutes URL."""
        match = re.search(r"fomcminutes(\d{8})\.htm", url)
        if match:
            ds = match.group(1)
            return date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        return date.today()

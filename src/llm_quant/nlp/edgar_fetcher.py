"""EDGAR 10-K text fetcher.

Fetches 10-K annual reports from SEC EDGAR, parses MD&A and Risk Factors
sections, and caches raw text in DuckDB.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import duckdb
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DDL_EDGAR_FILINGS = """
CREATE TABLE IF NOT EXISTS edgar_filings (
    ticker              VARCHAR NOT NULL,
    year                INTEGER NOT NULL,
    filing_date         DATE,
    accession_number    VARCHAR,
    mda_text            VARCHAR,
    risk_factors_text   VARCHAR,
    fetched_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, year)
)
"""

# Courtesy delay between HTTP requests to avoid hammering SEC servers.
_REQUEST_DELAY_SECS = 0.5

# SEC EDGAR full-text search endpoint
_EDGAR_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{ticker}%22"
    "&dateRange=custom"
    "&startdt={year}-01-01"
    "&enddt={year}-12-31"
    "&forms=10-K"
)

# SEC EDGAR company submissions API (preferred for ticker → CIK lookup)
_EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

_HEADERS = {
    "User-Agent": "llm-quant research bot (research@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

# Regex patterns for section detection in 10-K HTML/text
_MDA_HEADERS = re.compile(
    r"(?i)(management.{0,10}s?\s+discussion\s+and\s+analysis|item\s+7\.?\s*[:\-–—]?\s*management.{0,10}discussion)",
    re.IGNORECASE,
)
_RISK_HEADERS = re.compile(
    r"(?i)(risk\s+factors|item\s+1a\.?\s*[:\-–—]?\s*risk\s+factors)",
    re.IGNORECASE,
)
_NEXT_ITEM = re.compile(
    r"(?i)(item\s+\d+[a-z]?\s*[:\-–—.\s])",
)


def _get(url: str, **kwargs: Any) -> requests.Response | None:
    """GET with error handling and courtesy delay."""
    try:
        time.sleep(_REQUEST_DELAY_SECS)
        resp = requests.get(url, headers=_HEADERS, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("HTTP request failed for %s: %s", url, exc)
        return None


def _extract_section(
    soup: BeautifulSoup, header_pattern: re.Pattern[str]
) -> str | None:
    """Extract a named section from a parsed 10-K HTML document.

    Finds the first tag whose text matches ``header_pattern``, then
    accumulates text until the next Item heading or end of document.
    """
    # Try to find the header node
    header_node = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "b", "strong", "p", "div"]):
        text = tag.get_text(separator=" ", strip=True)
        if header_pattern.search(text) and len(text) < 200:
            header_node = tag
            break

    if header_node is None:
        return None

    chunks: list[str] = []
    for sibling in header_node.find_next_siblings():
        sib_text = sibling.get_text(separator=" ", strip=True)
        # Stop at next item heading
        if _NEXT_ITEM.match(sib_text) and sib_text != header_node.get_text(strip=True):
            # Only stop if it looks like a *different* item heading
            if not header_pattern.search(sib_text):
                break
        chunks.append(sib_text)
        # Reasonable cap to avoid runaway extraction
        if sum(len(c) for c in chunks) > 200_000:
            break

    text = " ".join(chunks).strip()
    return text or None


class EdgarFetcher:
    """Fetch and cache 10-K filings from SEC EDGAR.

    Parameters
    ----------
    db_conn:
        An open DuckDB connection.  The fetcher will create the
        ``edgar_filings`` table if it does not exist.
    """

    BASE_URL = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        self._db = db_conn
        self._db.execute(_DDL_EDGAR_FILINGS)
        self._cik_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_10k(self, ticker: str, year: int) -> str | None:
        """Fetch and parse 10-K for ticker/year; return MD&A text.

        Checks DuckDB cache first.  On cache miss, fetches from EDGAR,
        parses MD&A and Risk Factors, and stores both in the cache.

        Parameters
        ----------
        ticker:
            US equity ticker symbol (e.g. ``"AAPL"``).
        year:
            The fiscal/calendar year of the annual report (e.g. ``2023``).

        Returns
        -------
        str | None
            MD&A text, or ``None`` if unavailable.
        """
        cached = self._load_cache(ticker, year)
        if cached is not None:
            logger.debug("Cache hit: %s %d", ticker, year)
            return cached.get("mda_text")

        logger.info("Fetching 10-K from EDGAR: %s %d", ticker, year)
        cik = self._get_cik(ticker)
        if cik is None:
            logger.warning("Could not resolve CIK for ticker: %s", ticker)
            return None

        filing_info = self._find_10k_filing(cik, year)
        if filing_info is None:
            logger.warning("No 10-K filing found for %s %d", ticker, year)
            return None

        accession = filing_info["accession_number"]
        filing_date = filing_info.get("filing_date")
        doc_url = self._get_10k_document_url(cik, accession)
        if doc_url is None:
            return None

        mda_text, risk_text = self._parse_10k_document(doc_url)
        self._save_cache(ticker, year, filing_date, accession, mda_text, risk_text)
        return mda_text

    def get_mda_text(self, ticker: str, year: int) -> str | None:
        """Return cached MD&A text, fetching from EDGAR if not cached."""
        return self.fetch_10k(ticker, year)

    def get_risk_factors(self, ticker: str, year: int) -> str | None:
        """Return cached Risk Factors text, fetching from EDGAR if not cached."""
        self.fetch_10k(ticker, year)  # ensures cache is populated
        cached = self._load_cache(ticker, year)
        if cached:
            return cached.get("risk_factors_text")
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self, ticker: str, year: int) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT mda_text, risk_factors_text FROM edgar_filings WHERE ticker = ? AND year = ?",
            [ticker.upper(), year],
        ).fetchone()
        if row is None:
            return None
        return {"mda_text": row[0], "risk_factors_text": row[1]}

    def _save_cache(
        self,
        ticker: str,
        year: int,
        filing_date: str | None,
        accession_number: str | None,
        mda_text: str | None,
        risk_factors_text: str | None,
    ) -> None:
        self._db.execute(
            """
            INSERT OR REPLACE INTO edgar_filings
                (ticker, year, filing_date, accession_number, mda_text, risk_factors_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ticker.upper(),
                year,
                filing_date,
                accession_number,
                mda_text,
                risk_factors_text,
            ],
        )

    def _get_cik(self, ticker: str) -> str | None:
        """Resolve ticker to CIK using EDGAR company tickers JSON."""
        ticker_upper = ticker.upper()
        if ticker_upper in self._cik_cache:
            return self._cik_cache[ticker_upper]

        resp = _get(_EDGAR_COMPANY_TICKERS_URL)
        if resp is None:
            return None
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = str(entry["cik_str"]).zfill(10)
                self._cik_cache[ticker_upper] = cik
                return cik
        return None

    def _find_10k_filing(self, cik: str, year: int) -> dict[str, Any] | None:
        """Find the most recent 10-K accession number filed for the given year."""
        url = _EDGAR_SUBMISSIONS_URL.format(cik=cik)
        resp = _get(url)
        if resp is None:
            return None

        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        for form, date, accession in zip(forms, dates, accessions, strict=False):
            if form not in ("10-K", "10-K/A"):
                continue
            try:
                filing_year = int(date[:4])
            except (ValueError, TypeError):
                continue
            # Accept filings from the target year or Jan of the following year
            # (companies with Dec fiscal year often file in early next year)
            if filing_year in (year, year + 1):
                return {
                    "accession_number": accession.replace("-", ""),
                    "filing_date": date,
                }
        return None

    def _get_10k_document_url(self, cik: str, accession: str) -> str | None:
        """Get the URL of the primary 10-K HTML/text document."""
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
            f"/{accession}/{accession}-index.htm"
        )
        resp = _get(index_url)
        if resp is None:
            # Try JSON index as fallback
            json_url = (
                f"https://data.sec.gov/submissions/CIK{cik}/filings/{accession}.json"
            )
            resp = _get(json_url)
            if resp is None:
                return None

        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for a link to the primary document (10-K htm/html)
        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            if href.endswith((".htm", ".html")) and "10k" in href.lower().replace(
                "-", ""
            ):
                return f"https://www.sec.gov{href}" if href.startswith("/") else href
        # Fallback: return the index itself for section extraction
        return index_url

    def _parse_10k_document(self, url: str) -> tuple[str | None, str | None]:
        """Fetch 10-K HTML and extract MD&A + Risk Factors sections."""
        resp = _get(url)
        if resp is None:
            return None, None

        soup = BeautifulSoup(resp.text, "html.parser")
        mda_text = _extract_section(soup, _MDA_HEADERS)
        risk_text = _extract_section(soup, _RISK_HEADERS)
        return mda_text, risk_text

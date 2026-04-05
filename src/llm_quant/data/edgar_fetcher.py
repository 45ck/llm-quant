"""Fetch and cache SEC EDGAR 10-K filings for NLP research pipeline.

Downloads full 10-K filings from SEC EDGAR, caches them locally, and extracts
key sections (MD&A, Risk Factors) for downstream NLP analysis.

EDGAR API endpoints used:
  - CIK lookup: https://www.sec.gov/files/company_tickers.json
  - Filing list: https://data.sec.gov/submissions/CIK{cik}.json
  - Document: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}

Rate limit: max 10 requests/second per SEC fair access guidelines.
User-Agent must include identifying contact info per SEC policy.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: SEC requires a descriptive User-Agent with contact info.
DEFAULT_USER_AGENT = "llm-quant/1.0 (research; non-commercial) admin@example.com"

#: Minimum interval between requests (seconds) to respect SEC rate limits.
#: SEC allows max 10 req/sec; we use 0.12s (≈8/sec) for safety margin.
MIN_REQUEST_INTERVAL = 0.12

#: CIK lookup endpoint — returns JSON mapping of all company tickers to CIKs.
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

#: Filing submissions endpoint template (zero-padded 10-digit CIK).
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

#: Document download URL template.
ARCHIVES_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
)


# ---------------------------------------------------------------------------
# Section extraction patterns
# ---------------------------------------------------------------------------

#: Patterns for locating MD&A section boundaries.
MDA_START_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:ITEM\s*7[\.\s:—\-]*)"
        r"\s*MANAGEMENT[''\u2019]?S\s+DISCUSSION\s+AND\s+ANALYSIS",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*MANAGEMENT[''\u2019]?S\s+DISCUSSION\s+AND\s+ANALYSIS"
        r"\s+OF\s+FINANCIAL\s+CONDITION",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*ITEM\s*7[\.\s:—\-]+",
        re.IGNORECASE,
    ),
]

MDA_END_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:ITEM\s*7A[\.\s:—\-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:ITEM\s*8[\.\s:—\-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES"
        r"\s+ABOUT\s+MARKET\s+RISK",
        re.IGNORECASE,
    ),
]

#: Patterns for locating Risk Factors section boundaries.
RISK_START_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:ITEM\s*1A[\.\s:—\-]*)\s*RISK\s+FACTORS",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*ITEM\s*1A[\.\s:—\-]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*RISK\s+FACTORS\s*\n",
        re.IGNORECASE,
    ),
]

RISK_END_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:ITEM\s*1B[\.\s:—\-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:ITEM\s*2[\.\s:—\-]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*UNRESOLVED\s+STAFF\s+COMMENTS",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class EdgarFetcher:
    """Fetch and cache SEC EDGAR 10-K filings.

    Usage
    -----
    >>> fetcher = EdgarFetcher()
    >>> text = fetcher.fetch_10k("AAPL", 2023)
    >>> mda = fetcher.extract_mda(text)
    >>> risk = fetcher.extract_risk_factors(text)

    Parameters
    ----------
    cache_dir:
        Root directory for cached filings.  Defaults to
        ``data/nlp/edgar`` relative to the project root.
    user_agent:
        User-Agent header sent with all SEC requests.  SEC requires
        identifying contact information.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._cache_dir = cache_dir or _default_cache_dir()
        self._user_agent = user_agent
        self._last_request_time: float = 0.0
        self._cik_cache: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_10k(self, ticker: str, year: int) -> str | None:
        """Fetch full 10-K text for *ticker* filed in *year*.

        Returns cached text if available and non-empty, otherwise
        fetches from EDGAR.

        Parameters
        ----------
        ticker:
            Stock ticker symbol (e.g. ``"AAPL"``).
        year:
            Filing year to search for.

        Returns
        -------
        str or None
            Full 10-K document text, or None if not found.
        """
        ticker = ticker.upper()

        # Check cache first
        cached = self.get_cached(ticker, year)
        if cached is not None:
            return cached

        # Look up CIK
        cik = self._lookup_cik(ticker)
        if cik is None:
            logger.warning("Could not find CIK for ticker %s", ticker)
            return None

        # Find 10-K filing for the given year
        filing_info = self._find_10k_filing(cik, year)
        if filing_info is None:
            logger.warning(
                "No 10-K filing found for %s (CIK=%s) in year %d",
                ticker,
                cik,
                year,
            )
            return None

        accession_number = filing_info["accession_number"]
        primary_doc = filing_info["primary_doc"]

        # Fetch the document
        text = self._fetch_document(cik, accession_number, primary_doc)
        if text is None:
            return None

        # Strip HTML tags if present (many 10-Ks are filed as HTML)
        text = _strip_html(text)

        # Cache the full text
        self._save_to_cache(ticker, year, text)

        # Also extract and cache sections
        mda = self.extract_mda(text)
        if mda:
            self._save_section_cache(ticker, year, "mda", mda)

        risk = self.extract_risk_factors(text)
        if risk:
            self._save_section_cache(ticker, year, "risk_factors", risk)

        logger.info(
            "Fetched and cached 10-K for %s year %d (%d chars)",
            ticker,
            year,
            len(text),
        )
        return text

    def get_cached(self, ticker: str, year: int) -> str | None:
        """Read cached 10-K text from disk.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        year:
            Filing year.

        Returns
        -------
        str or None
            Cached text if file exists and is non-empty, else None.
        """
        ticker = ticker.upper()
        cache_path = self._cache_path(ticker, year)
        if cache_path.exists() and cache_path.stat().st_size > 0:
            logger.debug("Cache hit for %s %d at %s", ticker, year, cache_path)
            return cache_path.read_text(encoding="utf-8")
        return None

    @staticmethod
    def extract_mda(text: str) -> str:
        """Extract Management's Discussion and Analysis section from 10-K text.

        Parameters
        ----------
        text:
            Full 10-K document text.

        Returns
        -------
        str
            Extracted MD&A section text.  Returns empty string if section
            boundaries cannot be identified.
        """
        return _extract_section(text, MDA_START_PATTERNS, MDA_END_PATTERNS)

    @staticmethod
    def extract_risk_factors(text: str) -> str:
        """Extract Risk Factors section from 10-K text.

        Parameters
        ----------
        text:
            Full 10-K document text.

        Returns
        -------
        str
            Extracted Risk Factors section text.  Returns empty string if
            section boundaries cannot be identified.
        """
        return _extract_section(text, RISK_START_PATTERNS, RISK_END_PATTERNS)

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    def _lookup_cik(self, ticker: str) -> str | None:
        """Look up CIK number for a ticker using SEC company tickers JSON.

        Returns the CIK as a zero-padded 10-digit string, or None if not found.
        """
        if self._cik_cache is None:
            self._cik_cache = self._fetch_company_tickers()
            if self._cik_cache is None:
                return None

        return self._cik_cache.get(ticker.upper())

    def _fetch_company_tickers(self) -> dict[str, str] | None:
        """Fetch and parse the SEC company_tickers.json file.

        Returns a dict mapping ticker (uppercase) to zero-padded CIK string.
        """
        data = self._http_get_json(COMPANY_TICKERS_URL)
        if data is None:
            return None

        result: dict[str, str] = {}
        for entry in data.values():
            tick = entry.get("ticker", "").upper()
            cik_int = entry.get("cik_str", 0)
            if tick and cik_int:
                result[tick] = str(cik_int).zfill(10)

        logger.debug("Loaded %d ticker-to-CIK mappings from SEC", len(result))
        return result

    # ------------------------------------------------------------------
    # Filing discovery
    # ------------------------------------------------------------------

    def _find_10k_filing(self, cik: str, year: int) -> dict[str, str] | None:
        """Find the 10-K filing for *cik* in *year*.

        Searches the SEC submissions JSON for 10-K filings whose filing
        date falls within the specified year.

        Returns
        -------
        dict or None
            Dict with keys ``accession_number`` and ``primary_doc``,
            or None if no matching filing is found.
        """
        url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
        data = self._http_get_json(url)
        if data is None:
            return None

        recent = data.get("filings", {}).get("recent", {})
        return self._search_filing_list(recent, year)

    @staticmethod
    def _search_filing_list(recent: dict[str, Any], year: int) -> dict[str, str] | None:
        """Search a filing list for the 10-K filed in *year*.

        Parameters
        ----------
        recent:
            The ``filings.recent`` dict from the SEC submissions JSON,
            containing parallel arrays for form, filingDate,
            accessionNumber, primaryDocument.
        year:
            Target filing year.

        Returns
        -------
        dict or None
            Dict with ``accession_number`` and ``primary_doc``.
        """
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form not in ("10-K", "10-K/A"):
                continue
            if i >= len(dates) or i >= len(accessions) or i >= len(primary_docs):
                continue

            filing_date = dates[i]
            # filing_date is "YYYY-MM-DD" string
            filing_year = int(filing_date[:4])
            if filing_year == year:
                # Convert accession number: remove dashes for URL path
                accession_raw = accessions[i]
                accession_path = accession_raw.replace("-", "")
                return {
                    "accession_number": accession_path,
                    "primary_doc": primary_docs[i],
                }

        return None

    # ------------------------------------------------------------------
    # Document fetching
    # ------------------------------------------------------------------

    def _fetch_document(
        self, cik: str, accession_number: str, primary_doc: str
    ) -> str | None:
        """Fetch a filing document from EDGAR Archives.

        Parameters
        ----------
        cik:
            Zero-padded 10-digit CIK.
        accession_number:
            Accession number (dashes removed) for the URL path.
        primary_doc:
            Primary document filename (e.g. ``"aapl-20230930.htm"``).

        Returns
        -------
        str or None
            Document text content, or None on failure.
        """
        # CIK in the archives URL is NOT zero-padded
        cik_unpadded = cik.lstrip("0") or "0"
        url = ARCHIVES_URL_TEMPLATE.format(
            cik=cik_unpadded,
            accession=accession_number,
            doc=primary_doc,
        )
        return self._http_get_text(url)

    # ------------------------------------------------------------------
    # HTTP helpers with rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce minimum interval between HTTP requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    def _http_get_json(self, url: str) -> dict[str, Any] | None:
        """HTTP GET returning parsed JSON, or None on failure."""
        text = self._http_get_text(url)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.exception("JSON parse failed for %s", url)
            return None

    def _http_get_text(self, url: str) -> str | None:
        """HTTP GET returning text content, with rate limiting.

        Respects SEC fair access guidelines: max 10 req/sec,
        User-Agent with contact info.
        """
        self._rate_limit()

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self._user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Handle gzip encoding
                encoding = resp.headers.get("Content-Encoding", "")
                if encoding == "gzip":
                    import gzip

                    raw = gzip.decompress(raw)
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError:
            logger.exception("HTTP error fetching %s", url)
            return None
        except Exception:
            logger.exception("HTTP request failed for %s", url)
            return None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, year: int) -> Path:
        """Return the cache file path for a ticker/year."""
        return self._cache_dir / ticker / f"{year}.txt"

    def _save_to_cache(self, ticker: str, year: int, text: str) -> None:
        """Save full 10-K text to cache."""
        path = self._cache_path(ticker, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.debug("Cached 10-K to %s (%d chars)", path, len(text))

    def _save_section_cache(
        self, ticker: str, year: int, section: str, text: str
    ) -> None:
        """Save an extracted section to cache.

        Parameters
        ----------
        ticker:
            Stock ticker.
        year:
            Filing year.
        section:
            Section identifier: ``"mda"`` or ``"risk_factors"``.
        text:
            Section text content.
        """
        path = self._cache_dir / ticker / f"{year}_{section}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logger.debug("Cached %s section to %s (%d chars)", section, path, len(text))


# ---------------------------------------------------------------------------
# Section extraction helper
# ---------------------------------------------------------------------------


def _extract_section(
    text: str,
    start_patterns: list[re.Pattern[str]],
    end_patterns: list[re.Pattern[str]],
) -> str:
    """Extract a section from 10-K text using start/end regex patterns.

    Tries each start pattern in order; uses the first match. Then tries each
    end pattern after the start position; uses the first match.

    Parameters
    ----------
    text:
        Full 10-K document text.
    start_patterns:
        Ordered list of compiled regex patterns for section start.
    end_patterns:
        Ordered list of compiled regex patterns for section end.

    Returns
    -------
    str
        Extracted section text, stripped of leading/trailing whitespace.
        Returns empty string if section boundaries cannot be identified.
    """
    # Find section start
    start_pos = -1
    for pattern in start_patterns:
        match = pattern.search(text)
        if match:
            start_pos = match.end()
            break

    if start_pos < 0:
        return ""

    # Find section end (searching after the start position)
    end_pos = len(text)
    for pattern in end_patterns:
        match = pattern.search(text, pos=start_pos)
        if match:
            end_pos = match.start()
            break

    return text[start_pos:end_pos].strip()


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common HTML entities.

    Uses simple regex-based stripping. Not a full HTML parser — sufficient
    for SEC filings where we need text content only.

    Parameters
    ----------
    text:
        Raw document text (may contain HTML markup).

    Returns
    -------
    str
        Text with HTML tags removed and common entities decoded.
    """
    # Remove style and script blocks
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    entity_map = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&#39;": "'",
        "&nbsp;": " ",
        "&#160;": " ",
        "&mdash;": "\u2014",
        "&ndash;": "\u2013",
        "&rsquo;": "\u2019",
        "&lsquo;": "\u2018",
        "&rdquo;": "\u201d",
        "&ldquo;": "\u201c",
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)

    # Decode numeric entities (&#123; style)
    text = re.sub(
        r"&#(\d+);",
        lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x10000 else "",
        text,
    )

    # Collapse excessive whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Return the default EDGAR cache directory (data/nlp/edgar/ from project root)."""
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "data" / "nlp" / "edgar"
        if (current / "data").is_dir() or (current / "src").is_dir():
            return candidate
        current = current.parent
    # Fallback
    return Path.cwd() / "data" / "nlp" / "edgar"

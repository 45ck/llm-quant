"""Polymarket Gamma API client -- read-only market discovery and metadata.

The Gamma API at gamma-api.polymarket.com provides paginated access to
Polymarket's market and event catalog. No authentication is required.
Rate limit is ~4,000 requests/10 seconds.

Supports automatic failover to the Polymarket US API (api.polymarket.us)
if the Gamma API is geo-blocked (returns Azure Front Door 404 from US IPs).

Key endpoints:
  GET /markets          -- paginated list (limit, offset, active, closed)
  GET /markets/{id}     -- single market by condition ID
  GET /events           -- paginated event listing (includes nested markets)
  GET /events/{id}      -- single event by ID

Identifier hierarchy (critical for bridging to CLOB API):
  - id:            Gamma's internal market ID (string)
  - conditionId:   CTF condition on-chain (hex string)
  - clobTokenIds:  CLOB token IDs for YES/NO outcomes -- these are what
                   you pass to CLOB API calls (GET /book, /midpoint, etc.)
  - questionId:    used in UMA oracle requests

NegRisk markets have negRisk=True. For these, each outcome is a separate
binary market within an event container. The sum of all YES outcome prices
should equal 1.0. When sum < 1.0, buying all YES outcomes is profitable.

Gamma API docs: https://docs.polymarket.com/developers/gamma-markets-api/overview
CLOB API docs:  https://docs.polymarket.com/developers/CLOB/introduction
US API docs:    https://polymarket.us/developer
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# -- API base URLs --
# Gamma API: market discovery and metadata (may be geo-blocked from US)
GAMMA_BASE = "https://gamma-api.polymarket.com"
# CLOB API: orderbook, pricing, trade execution (may be geo-blocked from US)
CLOB_BASE = "https://clob.polymarket.com"
# Polymarket US: CFTC-regulated, works from US IPs
US_API_BASE = "https://api.polymarket.us"

_DEFAULT_TIMEOUT = 15  # seconds
_PAGE_SIZE = 100
_RATE_LIMIT_SLEEP = 0.25  # 4 req/s -- well within 4000 req/10s limit

_DEFAULT_SSL_VERIFY = True

# HTTP status codes for retry/fallback logic
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds, doubled each retry


@dataclass
class ConditionPrice:
    condition_id: str
    question: str
    outcome_yes: float
    outcome_no: float
    volume_24h: float
    open_interest: float
    clob_token_ids: list[str] = field(default_factory=list)

    @property
    def spread(self) -> float:
        """YES + NO - 1.0. Negative = arb opportunity."""
        return self.outcome_yes + self.outcome_no - 1.0

    @property
    def is_rebalance_arb(self) -> bool:
        """Single condition: buy YES+NO for less than $1."""
        return self.spread < -0.02  # >2 cent spread after fees


@dataclass
class Market:
    market_id: str
    slug: str
    question: str
    active: bool
    is_negrisk: bool
    category: str
    end_date: str | None
    conditions: list[ConditionPrice] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def sum_yes(self) -> float:
        """Sum of YES prices for NegRisk markets. < 1.0 = opportunity."""
        return sum(c.outcome_yes for c in self.conditions)

    @property
    def negrisk_complement(self) -> float:
        """1.0 - sum_yes. Positive = buy all NO outcomes, collect complement."""
        return 1.0 - self.sum_yes

    @property
    def is_negrisk_arb(self) -> bool:
        """NegRisk: complement > 0.05 after 2% fee = net >3%."""
        return self.is_negrisk and self.negrisk_complement > 0.05


class GammaClient:
    """Polymarket Gamma API client with US API fallback.

    Endpoint hierarchy:
      1. Gamma API (gamma-api.polymarket.com) -- full market discovery,
         paginated, no auth. May be geo-blocked from US IPs.
      2. US API (api.polymarket.us) -- CFTC-regulated fallback. Returns max
         20 markets without API key auth; full access requires Ed25519 key.

    SSL verification is enabled by default. Pass ssl_verify=False explicitly
    if you encounter certificate issues (common on Windows).

    Rate limit handling: 429 responses trigger exponential backoff with up
    to 3 retries. Geo-block detection: 404 with HTML content-type raises
    ConnectionError for clean fallback to the US API.
    """

    def __init__(
        self,
        base_url: str = GAMMA_BASE,
        timeout: int = _DEFAULT_TIMEOUT,
        ssl_verify: bool = _DEFAULT_SSL_VERIFY,
        us_api_key: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._ssl_verify = ssl_verify
        self._us_api_key = us_api_key
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "llm-quant-arb-scanner/1.0"})
        if not ssl_verify:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ------------------------------------------------------------------
    # Low-level HTTP with retry and rate-limit handling
    # ------------------------------------------------------------------

    def _get(
        self,
        path: str,
        params: dict | None = None,
        base_url: str | None = None,
    ) -> Any:
        """GET with retry on rate-limit (429) and transient server errors.

        Raises ConnectionError on geo-block (HTML 404 from Azure Front Door).
        Raises requests.HTTPError on non-retryable 4xx/5xx.
        """
        url = f"{base_url or self._base}{path}"
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout,
                    verify=self._ssl_verify,
                )
            except requests.ConnectionError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    backoff = _RETRY_BACKOFF * (2**attempt)
                    logger.debug(
                        "Connection error on %s (attempt %d/%d), retrying in %.1fs",
                        path,
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                raise

            # Geo-block detection: Azure Front Door returns 404 with HTML
            if resp.status_code == 404:
                ct = resp.headers.get("content-type", "")
                if "text/html" in ct:
                    raise requests.ConnectionError(
                        f"Gamma API geo-blocked (HTTP 404 with HTML from "
                        f"{resp.headers.get('x-azure-ref', 'Azure Front Door')}). "
                        f"International Polymarket APIs are blocked from US IPs."
                    )
                # Real 404 (e.g. market not found) -- raise immediately
                resp.raise_for_status()

            # Rate limit or transient server error -- retry with backoff
            if resp.status_code in _RETRYABLE_STATUS:
                backoff = _RETRY_BACKOFF * (2**attempt)
                # Respect Retry-After header if provided
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    with contextlib.suppress(ValueError):
                        backoff = max(backoff, float(retry_after))
                logger.warning(
                    "HTTP %d on %s (attempt %d/%d), retrying in %.1fs",
                    resp.status_code,
                    path,
                    attempt + 1,
                    _MAX_RETRIES,
                    backoff,
                )
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                time.sleep(backoff)
                continue

            # Non-retryable error
            resp.raise_for_status()

            # Success
            return resp.json()

        # All retries exhausted
        if last_exc:
            raise last_exc
        msg = f"GET {path} failed after {_MAX_RETRIES} retries"
        raise requests.HTTPError(msg)

    def _get_us(self, path: str, params: dict | None = None) -> Any:
        """GET against Polymarket US API (api.polymarket.us)."""
        url = f"{US_API_BASE}{path}"
        headers = {}
        if self._us_api_key:
            headers["Authorization"] = f"Bearer {self._us_api_key}"
        resp = self._session.get(
            url,
            params=params,
            timeout=self._timeout,
            verify=self._ssl_verify,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def fetch_markets_page(
        self,
        offset: int = 0,
        limit: int = _PAGE_SIZE,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict]:
        """Fetch one page of markets from Gamma API.

        Parameters
        ----------
        offset : int
            Number of markets to skip (for pagination).
        limit : int
            Max markets to return per page (default 100).
        active : bool
            If True, only return active markets.
        closed : bool
            If True, include closed/resolved markets.

        Returns
        -------
        list[dict]
            Raw market dicts from the Gamma API.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }
        data = self._get("/markets", params=params)
        # Gamma returns a list directly
        if isinstance(data, list):
            return data
        # Defensive: handle wrapped response
        return data.get("data", data.get("markets", []))

    def _fetch_us_markets(self) -> list[dict]:
        """Fetch markets from Polymarket US API.

        Without an API key, the US API returns at most ~20 markets (no
        pagination/filtering). With an API key (Ed25519), full access
        is available including offset/limit params.
        """
        data = self._get_us("/v1/markets")
        if isinstance(data, list):
            return data
        return data.get("markets", [])

    def fetch_all_active_markets(self, max_markets: int = 5000) -> list[dict]:
        """Paginate through all active markets.

        Tries Gamma API first (gamma-api.polymarket.com); falls back to
        US API (api.polymarket.us) if Gamma is geo-blocked or unreachable.

        Note: The US API fallback returns limited data without an API key
        (~20 markets). For full access from US IPs, provide a us_api_key
        or use a non-US proxy for the Gamma API.
        """
        # Try Gamma API first
        try:
            return self._fetch_all_gamma(max_markets)
        except (requests.HTTPError, requests.ConnectionError) as exc:
            logger.warning(
                "Gamma API unreachable (%s), falling back to US API. "
                "Note: gamma-api.polymarket.com may be geo-blocked from US IPs. "
                "Use a non-US proxy or provide a us_api_key for full access.",
                exc,
            )

        # Fallback: Polymarket US API
        try:
            markets = self._fetch_us_markets()
        except (requests.HTTPError, requests.ConnectionError) as exc:
            logger.exception("Both Gamma and US APIs failed: %s", exc)
            return []
        else:
            logger.info(
                "Fetched %d markets from Polymarket US API%s",
                len(markets),
                " (limited -- no API key)" if not self._us_api_key else "",
            )
            return markets

    def _fetch_all_gamma(self, max_markets: int) -> list[dict]:
        """Paginate through Gamma API /markets endpoint."""
        markets: list[dict] = []
        offset = 0
        while len(markets) < max_markets:
            page = self.fetch_markets_page(offset=offset, limit=_PAGE_SIZE)
            if not page:
                break
            markets.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
            time.sleep(_RATE_LIMIT_SLEEP)
        logger.info("Fetched %d active markets from Gamma API", len(markets))
        return markets

    def fetch_market(self, market_id: str) -> dict:
        """Fetch single market by condition ID.

        Tries Gamma API first, falls back to US API on failure.
        """
        try:
            return self._get(f"/markets/{market_id}")
        except (requests.HTTPError, requests.ConnectionError) as exc:
            logger.warning(
                "Gamma API failed for market %s (%s), trying US API",
                market_id,
                exc,
            )
            return self._get_us(f"/v1/markets/{market_id}")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def fetch_events_page(
        self,
        offset: int = 0,
        limit: int = _PAGE_SIZE,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict]:
        """Fetch one page of events from Gamma API.

        Events contain nested markets. This is the preferred way to
        discover NegRisk multi-outcome markets, where each outcome
        is a separate market within an event container.

        Parameters
        ----------
        offset : int
            Number of events to skip (for pagination).
        limit : int
            Max events to return per page (default 100).
        active : bool
            If True, only return active events.
        closed : bool
            If True, include closed/resolved events.

        Returns
        -------
        list[dict]
            Raw event dicts from the Gamma API. Each event may contain
            a 'markets' key with nested market dicts.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }
        data = self._get("/events", params=params)
        if isinstance(data, list):
            return data
        return data.get("data", data.get("events", []))

    def fetch_all_active_events(self, max_events: int = 2000) -> list[dict]:
        """Paginate through all active events.

        Events include nested markets, making this efficient for
        NegRisk scanning (one API call per event page instead of
        separate market queries).
        """
        events: list[dict] = []
        offset = 0
        while len(events) < max_events:
            page = self.fetch_events_page(offset=offset, limit=_PAGE_SIZE)
            if not page:
                break
            events.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
            time.sleep(_RATE_LIMIT_SLEEP)
        logger.info("Fetched %d active events from Gamma API", len(events))
        return events

    def fetch_event(self, event_id: str) -> dict:
        """Fetch a single event by ID, including nested markets."""
        return self._get(f"/events/{event_id}")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_market(raw: dict) -> Market | None:  # noqa: C901, PLR0912
        """Parse raw market dict into Market dataclass.

        Handles both Gamma API and US API response formats.

        The Gamma API returns outcomes/outcomePrices as JSON strings
        (e.g. '["Yes","No"]', '["0.65","0.35"]'). The CLOB token IDs
        are in clobTokenIds (also a JSON string). The negRisk field
        (boolean) indicates NegRisk multi-outcome markets.
        """
        import json as _json

        try:
            market_id = str(raw.get("id") or raw.get("condition_id", ""))
            if not market_id:
                return None

            yes_price = 0.0
            no_price = 0.0

            # Approach 1: outcomes/outcomePrices (JSON string or list)
            outcomes_raw = raw.get("outcomes", "")
            outcome_prices_raw = raw.get("outcomePrices", "")

            if isinstance(outcomes_raw, str) and outcomes_raw:
                try:
                    outcomes = _json.loads(outcomes_raw)
                    prices = (
                        _json.loads(outcome_prices_raw) if outcome_prices_raw else []
                    )
                    for i, o in enumerate(outcomes):
                        if str(o).lower() == "yes" and i < len(prices):
                            yes_price = float(prices[i])
                        elif str(o).lower() == "no" and i < len(prices):
                            no_price = float(prices[i])
                except (ValueError, TypeError):
                    logger.debug("Failed to parse outcomes JSON %s", market_id)
            elif isinstance(outcomes_raw, list):
                prices_list = (
                    outcome_prices_raw if isinstance(outcome_prices_raw, list) else []
                )
                for i, o in enumerate(outcomes_raw):
                    if str(o).lower() == "yes" and i < len(prices_list):
                        yes_price = float(prices_list[i])
                    elif str(o).lower() == "no" and i < len(prices_list):
                        no_price = float(prices_list[i])

            # Approach 2: CLOB tokens (overrides approach 1 -- more accurate)
            tokens = raw.get("tokens", [])
            if isinstance(tokens, list):
                for tok in tokens:
                    if tok.get("outcome", "").lower() == "yes":
                        yes_price = float(tok.get("price") or yes_price)
                    elif tok.get("outcome", "").lower() == "no":
                        no_price = float(tok.get("price") or no_price)

            # Approach 3: bestBid/bestAsk from Gamma API (available on /markets)
            if yes_price == 0.0:
                best_ask = raw.get("bestAsk")
                if best_ask is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        yes_price = float(best_ask)
            if no_price == 0.0 and yes_price > 0:
                no_price = 1.0 - yes_price

            # Extract CLOB token IDs for bridging to CLOB API
            clob_token_ids: list[str] = []
            clob_raw = raw.get("clobTokenIds", "")
            if isinstance(clob_raw, str) and clob_raw:
                try:
                    parsed = _json.loads(clob_raw)
                    if isinstance(parsed, list):
                        clob_token_ids = [str(t) for t in parsed]
                except (ValueError, TypeError):
                    pass
            elif isinstance(clob_raw, list):
                clob_token_ids = [str(t) for t in clob_raw]

            # Use category from API if provided, else infer from text
            category = raw.get("category", "")
            if not category:
                category = _infer_category(
                    raw.get("question", "") + " " + raw.get("slug", "")
                )

            # Volume: try multiple field names (Gamma uses volume24hr, volumeNum)
            volume_24h = float(raw.get("volume24hr") or raw.get("volumeNum24hr") or 0.0)

            cond = ConditionPrice(
                condition_id=raw.get("conditionId", market_id),
                question=raw.get("question", ""),
                outcome_yes=yes_price,
                outcome_no=no_price,
                volume_24h=volume_24h,
                open_interest=float(raw.get("openInterest") or 0.0),
                clob_token_ids=clob_token_ids,
            )

            # NegRisk: Gamma API uses 'negRisk' (boolean), not 'isNegRisk'
            is_negrisk = bool(
                raw.get("negRisk")
                or raw.get("isNegRisk")
                or raw.get("is_neg_risk", False)
            )

            return Market(
                market_id=market_id,
                slug=raw.get("slug") or raw.get("market_slug", ""),
                question=raw.get("question") or raw.get("title", ""),
                active=bool(raw.get("active", True)),
                is_negrisk=is_negrisk,
                category=category,
                end_date=raw.get("endDate") or raw.get("end_date"),
                conditions=[cond],
                raw=raw,
            )
        except Exception as exc:
            logger.debug("Failed to parse market %s: %s", raw.get("id", "?"), exc)
            return None

    def parse_all_markets(self, raw_list: list[dict]) -> list[Market]:
        """Parse list of raw market dicts, skipping failures."""
        markets = []
        for raw in raw_list:
            m = self.parse_market(raw)
            if m and m.question:
                markets.append(m)
        logger.info("Parsed %d/%d markets successfully", len(markets), len(raw_list))
        return markets


# ------------------------------------------------------------------
# Category inference (simple keyword heuristic)
# ------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "sports": [
        "nba",
        "nfl",
        "nhl",
        "mlb",
        "soccer",
        "tennis",
        "golf",
        "ufc",
        "mma",
        "superbowl",
        "world cup",
        "olympics",
        "championship",
        "game 7",
        "series",
        "playoff",
        "win the",
        "beat the",
        "score",
        "points",
        "match",
    ],
    "politics": [
        "president",
        "election",
        "senate",
        "congress",
        "vote",
        "ballot",
        "democrat",
        "republican",
        "trump",
        "biden",
        "harris",
        "party",
        "governor",
        "mayor",
        "legislation",
        "bill pass",
    ],
    "crypto": [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "sol",
        "price above",
        "price below",
        "crypto",
        "defi",
        "nft",
        "token",
        "blockchain",
    ],
    "finance": [
        "fed",
        "rate",
        "cpi",
        "gdp",
        "inflation",
        "recession",
        "market",
        "s&p",
        "nasdaq",
        "dow",
        "earnings",
        "ipo",
    ],
    "geopolitics": [
        "war",
        "ceasefire",
        "invasion",
        "nato",
        "un",
        "sanctions",
        "ukraine",
        "russia",
        "china",
        "taiwan",
        "middle east",
    ],
}


def _infer_category(text: str) -> str:
    import re

    text_lower = text.lower()
    # Use word-boundary matching to avoid substring false positives
    # (e.g. "eth" inside "something", "sol" inside "resolution")
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            # Multi-word phrases: simple substring match
            # Single short tokens (<=3 chars): require word boundary
            if len(kw) <= 3 or " " not in kw:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text_lower):
                    return cat
            elif kw in text_lower:
                return cat
    return "other"

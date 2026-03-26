"""Polymarket market data client — read-only, no authentication required.

Supports two backends:
  1. Gamma API (gamma-api.polymarket.com) — full market data, geo-blocked in US
  2. Polymarket US API (api.polymarket.us) — US-regulated, limited without auth

The client tries Gamma first and falls back to the US API on failure.

Gamma API docs: https://docs.polymarket.com/
US API docs: https://polymarket.us/developer

Key endpoints used:
  Gamma:  GET /markets          — paginated list of all markets
          GET /markets/{id}     — single market detail
  US API: GET /v1/markets       — market listing (limited without API key)

NegRisk markets have is_neg_risk=True. For these, the sum of all YES
outcome prices should equal 1.0 (one outcome must occur). When sum < 1.0,
buying all NO outcomes is profitable (buy N outcomes, collect 1 outcome).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
US_API_BASE = "https://api.polymarket.us"
_DEFAULT_TIMEOUT = 15  # seconds
_PAGE_SIZE = 100
_RATE_LIMIT_SLEEP = 0.25  # 4 req/s — well within public limits


@dataclass
class ConditionPrice:
    condition_id: str
    question: str
    outcome_yes: float
    outcome_no: float
    volume_24h: float
    open_interest: float

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
    """Polymarket market data client with Gamma + US API fallback."""

    def __init__(
        self,
        base_url: str = GAMMA_BASE,
        timeout: int = _DEFAULT_TIMEOUT,
        ssl_verify: bool = True,
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
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(
            url, params=params, timeout=self._timeout, verify=self._ssl_verify
        )
        resp.raise_for_status()
        return resp.json()

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
        """Fetch one page of markets from Gamma API."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }
        data = self._get("/markets", params=params)
        # Gamma returns list directly or wrapped — handle both
        if isinstance(data, list):
            return data
        return data.get("data", data.get("markets", []))

    def _fetch_us_markets(self) -> list[dict]:
        """Fetch markets from Polymarket US API (limited without auth)."""
        data = self._get_us("/v1/markets")
        return data.get("markets", [])

    def fetch_all_active_markets(self, max_markets: int = 5000) -> list[dict]:
        """Paginate through all active markets.

        Tries Gamma API first; falls back to US API if Gamma is unreachable
        (e.g. geo-blocked from US IPs).
        """
        # Try Gamma API first
        try:
            return self._fetch_all_gamma(max_markets)
        except (requests.HTTPError, requests.ConnectionError) as exc:
            logger.warning("Gamma API unreachable (%s), falling back to US API", exc)

        # Fallback: Polymarket US API
        try:
            markets = self._fetch_us_markets()
        except (requests.HTTPError, requests.ConnectionError) as exc:
            logger.exception("Both Gamma and US APIs failed: %s", exc)
            return []
        else:
            logger.info("Fetched %d markets from Polymarket US API", len(markets))
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
        """Fetch single market by ID."""
        return self._get(f"/markets/{market_id}")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_market(raw: dict) -> Market | None:  # noqa: C901, PLR0912
        """Parse raw market dict into Market dataclass.

        Handles both Gamma API and US API response formats.
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

            # Approach 2: CLOB tokens (overrides approach 1 — more accurate)
            tokens = raw.get("tokens", [])
            if isinstance(tokens, list):
                for tok in tokens:
                    if tok.get("outcome", "").lower() == "yes":
                        yes_price = float(tok.get("price") or yes_price)
                    elif tok.get("outcome", "").lower() == "no":
                        no_price = float(tok.get("price") or no_price)

            # Approach 3: US API marketSides — extract prices if available
            sides = raw.get("marketSides", [])
            if isinstance(sides, list) and sides and not tokens:
                # US API doesn't provide per-side prices in the public response,
                # but outcomePrices are already parsed above in Approach 1.
                pass

            # Use category from API if provided (US API has it), else infer
            category = raw.get("category", "")
            if not category:
                category = _infer_category(
                    raw.get("question", "") + " " + raw.get("slug", "")
                )

            cond = ConditionPrice(
                condition_id=raw.get("conditionId", market_id),
                question=raw.get("question", ""),
                outcome_yes=yes_price,
                outcome_no=no_price,
                volume_24h=float(
                    raw.get("volumeNum24hr") or raw.get("volume24hr") or 0.0
                ),
                open_interest=float(raw.get("openInterest") or 0.0),
            )

            return Market(
                market_id=market_id,
                slug=raw.get("slug") or raw.get("market_slug", ""),
                question=raw.get("question") or raw.get("title", ""),
                active=bool(raw.get("active", True)),
                is_negrisk=bool(raw.get("isNegRisk") or raw.get("is_neg_risk", False)),
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

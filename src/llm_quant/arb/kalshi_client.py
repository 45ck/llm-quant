"""Kalshi CFTC-regulated prediction market client.

Kalshi is the US-legal prediction market exchange (CFTC-regulated).
No authentication required for public market data.

API base: https://api.elections.kalshi.com/trade-api/v2
Key endpoints:
  GET /events?status=open     — list of events
  GET /markets?event_ticker=X — markets within an event
  GET /markets?status=open    — all open markets

Kalshi fee: 3% taker fee on winning positions (vs Polymarket's 2%).

NegRisk equivalent: events with mutually_exclusive=True.
  Strategy: sum(YES_ask) < 1.0 → buy all YES, one pays $1.
  Net profit: (1 - sum_yes_ask) - 0.03 (fee on winning leg).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_DEFAULT_TIMEOUT = 15
_PAGE_SIZE = 200
_RATE_LIMIT_SLEEP = 0.20  # 5 req/s

# Kalshi uses "active" for tradeable markets (not "open")
_ACTIVE_STATUS = "open"

# Kalshi fee on winning position
KALSHI_WIN_FEE = 0.03


@dataclass
class KalshiCondition:
    """A single Kalshi binary market (YES/NO contract)."""

    ticker: str
    event_ticker: str
    title: str
    yes_ask: float  # price to buy YES (what you pay)
    yes_bid: float  # price someone will pay for YES
    no_ask: float  # price to buy NO
    volume_24h: float
    volume_total: float
    is_open: bool = True

    @property
    def mid_yes(self) -> float:
        """Mid-price for YES."""
        return (self.yes_ask + self.yes_bid) / 2.0 if self.yes_bid > 0 else self.yes_ask

    @property
    def bid_ask_spread(self) -> float:
        """YES bid-ask spread — proxy for liquidity."""
        return self.yes_ask - self.yes_bid


@dataclass
class KalshiEvent:
    """A Kalshi event — may contain multiple mutually exclusive markets."""

    event_ticker: str
    series_ticker: str
    title: str
    category: str
    mutually_exclusive: bool
    markets: list[KalshiCondition] = field(default_factory=list)

    @property
    def sum_yes_ask(self) -> float:
        """Sum of YES ask prices across all conditions."""
        return sum(c.yes_ask for c in self.markets)

    @property
    def negrisk_complement(self) -> float:
        """1 - sum_yes_ask. Positive when there's an arb opportunity."""
        return 1.0 - self.sum_yes_ask

    @property
    def net_spread(self) -> float:
        """Net spread after 3% winning fee."""
        return self.negrisk_complement - KALSHI_WIN_FEE

    @property
    def is_negrisk_arb(self) -> bool:
        """True when buying all YES positions yields net positive EV."""
        return self.mutually_exclusive and self.net_spread > 0

    @property
    def total_volume_24h(self) -> float:
        return sum(c.volume_24h for c in self.markets)

    @property
    def min_condition_volume(self) -> float:
        """Minimum volume across conditions — bottleneck for fill."""
        if not self.markets:
            return 0.0
        return min(c.volume_24h for c in self.markets)


class KalshiClient:
    """Thin wrapper around the Kalshi public REST API."""

    def __init__(
        self, base_url: str = KALSHI_BASE, timeout: int = _DEFAULT_TIMEOUT
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "llm-quant-arb-scanner/1.0",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self._base}{path}"
        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def fetch_events_page(
        self,
        cursor: str | None = None,
        limit: int = _PAGE_SIZE,
        status: str = "open",
    ) -> dict[str, Any]:
        """Fetch one page of events."""
        params: dict[str, Any] = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params=params)

    def fetch_all_open_events(self) -> list[dict[str, Any]]:
        """Paginate through all open events."""
        events: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = self.fetch_events_page(cursor=cursor)
            batch = page.get("events", [])
            events.extend(batch)
            cursor = page.get("cursor")
            if not cursor or len(batch) < _PAGE_SIZE:
                break
            time.sleep(_RATE_LIMIT_SLEEP)
        logger.info("Fetched %d events from Kalshi", len(events))
        return events

    def fetch_mutually_exclusive_events(self) -> list[dict[str, Any]]:
        """Return only mutually exclusive events — the NegRisk analogue."""
        all_events = self.fetch_all_open_events()
        me = [e for e in all_events if e.get("mutually_exclusive")]
        logger.info(
            "Mutually exclusive events: %d / %d total", len(me), len(all_events)
        )
        return me

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def fetch_markets_for_event(
        self, event_ticker: str, status: str = _ACTIVE_STATUS
    ) -> list[dict[str, Any]]:
        """Fetch all active markets within an event."""
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "event_ticker": event_ticker,
                "status": status,
                "limit": _PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor
            page = self._get("/markets", params=params)
            batch = page.get("markets", [])
            markets.extend(batch)
            cursor = page.get("cursor")
            if not cursor or len(batch) < _PAGE_SIZE:
                break
            time.sleep(_RATE_LIMIT_SLEEP)
        return markets

    def fetch_all_open_markets(self, limit: int = 50_000) -> list[dict[str, Any]]:
        """Paginate through all active markets (Kalshi status='active')."""
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        while len(markets) < limit:
            params: dict[str, Any] = {"status": _ACTIVE_STATUS, "limit": _PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            page = self._get("/markets", params=params)
            batch = page.get("markets", [])
            markets.extend(batch)
            cursor = page.get("cursor")
            if not cursor or len(batch) < _PAGE_SIZE:
                break
            time.sleep(_RATE_LIMIT_SLEEP)
        logger.info("Fetched %d active markets from Kalshi", len(markets))
        return markets

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_condition(raw: dict) -> KalshiCondition | None:
        """Parse a raw Kalshi market dict into KalshiCondition."""
        try:
            ticker = raw.get("ticker", "")
            if not ticker:
                return None
            return KalshiCondition(
                ticker=ticker,
                event_ticker=raw.get("event_ticker", ""),
                title=raw.get("title", raw.get("yes_sub_title", "")),
                yes_ask=float(raw.get("yes_ask_dollars") or 0),
                yes_bid=float(raw.get("yes_bid_dollars") or 0),
                no_ask=float(raw.get("no_ask_dollars") or 0),
                volume_24h=float(raw.get("volume_24h_fp") or 0),
                volume_total=float(raw.get("volume_fp") or 0),
                is_open=raw.get("status", "open") == "open",
            )
        except Exception as exc:
            logger.debug("Failed to parse Kalshi market %s: %s", raw.get("ticker"), exc)
            return None

    @staticmethod
    def parse_event(raw_event: dict, raw_markets: list[dict]) -> KalshiEvent:
        """Parse a Kalshi event with its markets.

        Filters out empty-book conditions (YES_ask >= 0.97, YES_bid == 0)
        which are default placeholder prices, not real market prices.
        """
        conditions = []
        for m in raw_markets:
            c = KalshiClient.parse_condition(m)
            if c is None or c.yes_ask <= 0:
                continue
            # Skip empty-book conditions: default ask=0.97-1.0 with no active bid
            if c.yes_ask >= 0.97 and c.yes_bid == 0.0:
                continue
            conditions.append(c)

        return KalshiEvent(
            event_ticker=raw_event.get("event_ticker", ""),
            series_ticker=raw_event.get("series_ticker", ""),
            title=raw_event.get("title", ""),
            category=raw_event.get("category", "other"),
            mutually_exclusive=bool(raw_event.get("mutually_exclusive")),
            markets=conditions,
        )

    # ------------------------------------------------------------------
    # High-level scan helpers
    # ------------------------------------------------------------------

    def fetch_negrisk_events(  # noqa: C901
        self,
        max_events: int = 500,
        use_bulk: bool = True,
        bulk_limit: int = 200_000,
    ) -> list[KalshiEvent]:
        """
        Fetch mutually exclusive events with their market prices.

        Two strategies:
          bulk (default): Fetch all open events + all markets in parallel sweeps,
            join in memory. Fast if market count < bulk_limit.
          per_event: Fetch markets for each ME event individually.
            Slower (N×API calls) but complete. Used as fallback.

        Args:
            max_events: Maximum ME events to return.
            use_bulk: If True, attempt bulk-fetch first.
            bulk_limit: Max markets to fetch in bulk mode. Increase if elections
                markets are missed (they can be far in the pagination order).
        """
        logger.info("Fetching Kalshi NegRisk events (bulk=%s)...", use_bulk)

        # Step 1: All events indexed by ticker
        raw_events = self.fetch_all_open_events()
        event_index: dict[str, dict] = {
            e["event_ticker"]: e for e in raw_events if e.get("event_ticker")
        }
        me_tickers = {t for t, e in event_index.items() if e.get("mutually_exclusive")}
        logger.info(
            "Kalshi: %d total events, %d mutually exclusive",
            len(event_index),
            len(me_tickers),
        )

        if use_bulk:
            # Step 2: Bulk fetch all markets, join in memory
            raw_markets = self.fetch_all_open_markets(limit=bulk_limit)
            logger.info("Kalshi: fetched %d total markets in bulk", len(raw_markets))

            by_event: dict[str, list[dict]] = {}
            for m in raw_markets:
                evt_ticker = m.get("event_ticker", "")
                if evt_ticker and evt_ticker in me_tickers:
                    by_event.setdefault(evt_ticker, []).append(m)

            # For ME events not found in bulk, fall back to per-event fetch
            missing = [t for t in me_tickers if t not in by_event]
            if missing:
                logger.info(
                    "Bulk missed %d ME events — fetching per-event (cap=%d)",
                    len(missing),
                    max_events,
                )
                for ticker in list(missing)[:max_events]:
                    try:
                        markets = self.fetch_markets_for_event(ticker)
                        time.sleep(_RATE_LIMIT_SLEEP)
                        if len(markets) >= 2:
                            by_event[ticker] = markets
                    except Exception as exc:
                        logger.debug("Per-event fetch failed for %s: %s", ticker, exc)
        else:
            # Pure per-event approach
            by_event = {}
            for ticker in list(me_tickers)[:max_events]:
                try:
                    markets = self.fetch_markets_for_event(ticker)
                    time.sleep(_RATE_LIMIT_SLEEP)
                    if len(markets) >= 2:
                        by_event[ticker] = markets
                except Exception as exc:
                    logger.debug("Per-event fetch failed for %s: %s", ticker, exc)

        # Parse into KalshiEvent objects
        kalshi_events: list[KalshiEvent] = []
        for evt_ticker, markets in by_event.items():
            if len(markets) < 2:
                continue
            raw_evt = event_index.get(evt_ticker, {"event_ticker": evt_ticker})
            evt = self.parse_event(raw_evt, markets)
            if len(evt.markets) >= 2:
                kalshi_events.append(evt)

        logger.info(
            "Loaded %d multi-condition ME events from Kalshi", len(kalshi_events)
        )
        return kalshi_events

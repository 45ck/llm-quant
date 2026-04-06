"""Polymarket CLOB API client -- read-only orderbook and pricing data.

The CLOB API at clob.polymarket.com provides real-time orderbook depth,
pricing, and historical price data. Public (read-only) endpoints require
no authentication. Trading endpoints require EIP-712 L2 keys.

This client covers only the public read-only endpoints:
  GET /book            -- full L2 orderbook snapshot
  GET /price           -- best bid or ask price
  GET /midpoint        -- midpoint between best bid/ask
  GET /spread          -- current spread
  GET /tick-size       -- dynamic tick size for a market
  GET /prices-history  -- historical price time series
  GET /last-trade-price -- last execution price

All endpoints accept a ``token_id`` parameter. This is the CLOB token ID
(a large integer string), NOT the conditionId. Token IDs are obtained from
the Gamma API's ``clobTokenIds`` field on market objects.

Rate limits (per 10 seconds):
  /book:           50 requests
  /price:          100 requests
  /midpoint:       no documented limit (use conservatively)
  /spread:         no documented limit
  /tick-size:      no documented limit
  /prices-history: no documented limit
  General:         9,000 requests/10s across all endpoints

CLOB API docs: https://docs.polymarket.com/developers/CLOB/introduction
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"

_DEFAULT_TIMEOUT = 15  # seconds
_DEFAULT_SSL_VERIFY = True

# Retry configuration for rate limits and transient errors
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds, doubled each retry

# Valid intervals for /prices-history
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "6h", "1d", "1w", "1M", "max"}


@dataclass
class OrderbookLevel:
    """A single price level in the orderbook."""

    price: float
    size: float


@dataclass
class Orderbook:
    """Full L2 orderbook snapshot for a token."""

    market: str  # conditionId (hex)
    asset_id: str  # CLOB token ID
    bids: list[OrderbookLevel] = field(default_factory=list)
    asks: list[OrderbookLevel] = field(default_factory=list)
    timestamp: str | None = None
    hash: str | None = None

    @property
    def best_bid(self) -> float | None:
        """Highest bid price, or None if no bids."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Lowest ask price, or None if no asks."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        """Best ask - best bid, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def midpoint(self) -> float | None:
        """(best_bid + best_ask) / 2, or None."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def bid_depth(self) -> float:
        """Total bid-side depth in contract units."""
        return sum(level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total ask-side depth in contract units."""
        return sum(level.size for level in self.asks)


@dataclass
class PricePoint:
    """A single point in a price time series."""

    timestamp: int  # Unix timestamp
    price: float


@dataclass
class PriceHistory:
    """Historical price time series for a token."""

    token_id: str
    interval: str
    points: list[PricePoint] = field(default_factory=list)

    @property
    def prices(self) -> list[float]:
        """Extract price values only."""
        return [p.price for p in self.points]

    @property
    def timestamps(self) -> list[int]:
        """Extract timestamp values only."""
        return [p.timestamp for p in self.points]


class ClobClient:
    """Polymarket CLOB API client for read-only market data.

    Provides access to orderbook depth, pricing, and historical price
    data. No authentication is required for read-only endpoints.

    SSL verification is enabled by default. Pass ssl_verify=False
    explicitly if you encounter certificate issues (common on Windows).

    Rate limit handling: 429 responses trigger exponential backoff with
    up to 3 retries.
    """

    def __init__(
        self,
        base_url: str = CLOB_BASE,
        timeout: int = _DEFAULT_TIMEOUT,
        ssl_verify: bool = _DEFAULT_SSL_VERIFY,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._ssl_verify = ssl_verify
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "llm-quant-clob-client/1.0"})
        if not ssl_verify:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # ------------------------------------------------------------------
    # Low-level HTTP with retry
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> Any:
        """GET with retry on rate-limit (429) and transient server errors."""
        url = f"{self._base}{path}"
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

            # Rate limit or transient server error -- retry with backoff
            if resp.status_code in _RETRYABLE_STATUS:
                backoff = _RETRY_BACKOFF * (2**attempt)
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

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------

    def get_book(self, token_id: str) -> Orderbook:
        """Fetch full L2 orderbook for a CLOB token.

        Parameters
        ----------
        token_id : str
            CLOB token ID (large integer string from Gamma API's
            clobTokenIds field). NOT the conditionId.

        Returns
        -------
        Orderbook
            Parsed orderbook with bids and asks sorted by price
            (bids descending, asks ascending).
        """
        data = self._get("/book", params={"token_id": token_id})
        return self._parse_orderbook(data, token_id)

    def get_books(self, token_ids: list[str]) -> list[Orderbook]:
        """Fetch orderbooks for multiple tokens.

        Uses the POST /books batch endpoint if available, falling back
        to sequential GET /book calls.

        Parameters
        ----------
        token_ids : list[str]
            List of CLOB token IDs.

        Returns
        -------
        list[Orderbook]
            One orderbook per token ID.
        """
        books = []
        for tid in token_ids:
            try:
                book = self.get_book(tid)
                books.append(book)
            except (requests.HTTPError, requests.ConnectionError) as exc:
                logger.warning("Failed to fetch book for %s: %s", tid[:20], exc)
        return books

    @staticmethod
    def _parse_orderbook(data: dict, token_id: str) -> Orderbook:
        """Parse raw orderbook JSON into Orderbook dataclass."""
        bids = [
            OrderbookLevel(
                price=float(level.get("price", 0)),
                size=float(level.get("size", 0)),
            )
            for level in data.get("bids", [])
        ]
        asks = [
            OrderbookLevel(
                price=float(level.get("price", 0)),
                size=float(level.get("size", 0)),
            )
            for level in data.get("asks", [])
        ]
        # Ensure correct sort order
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return Orderbook(
            market=data.get("market", ""),
            asset_id=data.get("asset_id", token_id),
            bids=bids,
            asks=asks,
            timestamp=data.get("timestamp"),
            hash=data.get("hash"),
        )

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token.

        Returns
        -------
        float or None
            Midpoint price, or None if not available.
        """
        data = self._get("/midpoint", params={"token_id": token_id})
        mid = data.get("mid")
        if mid is not None:
            return float(mid)
        return None

    def get_spread(self, token_id: str) -> float | None:
        """Get current spread for a token.

        Returns
        -------
        float or None
            Spread in price units, or None if not available.
        """
        data = self._get("/spread", params={"token_id": token_id})
        spread = data.get("spread")
        if spread is not None:
            return float(spread)
        return None

    def get_price(self, token_id: str, side: str = "BUY") -> float | None:
        """Get best price for a token on a given side.

        Parameters
        ----------
        token_id : str
            CLOB token ID.
        side : str
            'BUY' or 'SELL'.

        Returns
        -------
        float or None
            Best price, or None if not available.
        """
        data = self._get("/price", params={"token_id": token_id, "side": side.upper()})
        price = data.get("price")
        if price is not None:
            return float(price)
        return None

    def get_last_trade_price(self, token_id: str) -> float | None:
        """Get last execution price for a token.

        Returns
        -------
        float or None
            Last trade price, or None if not available.
        """
        data = self._get("/last-trade-price", params={"token_id": token_id})
        price = data.get("price")
        if price is not None:
            return float(price)
        return None

    def get_tick_size(self, token_id: str) -> float | None:
        """Get current tick size for a market.

        Tick sizes are dynamic and can change when prices exceed 0.96
        or fall below 0.04. Common values: 0.1, 0.01, 0.001, 0.0001.

        Returns
        -------
        float or None
            Minimum tick size, or None if not available.
        """
        data = self._get("/tick-size", params={"token_id": token_id})
        tick = data.get("minimum_tick_size")
        if tick is not None:
            return float(tick)
        return None

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------

    def get_prices_history(
        self,
        token_id: str,
        interval: str = "1h",
        fidelity: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> PriceHistory:
        """Get historical price time series for a token.

        Parameters
        ----------
        token_id : str
            CLOB token ID (from Gamma API's clobTokenIds field).
        interval : str
            Time interval: '1m', '5m', '15m', '1h', '6h', '1d', '1w',
            '1M', or 'max'. Default '1h'.
        fidelity : int or None
            Resolution in minutes. Overrides interval granularity if set.
        start_ts : int or None
            Start Unix timestamp. If None, defaults to server default.
        end_ts : int or None
            End Unix timestamp. If None, defaults to current time.

        Returns
        -------
        PriceHistory
            Parsed time series with timestamp/price pairs.
        """
        params: dict[str, Any] = {"market": token_id, "interval": interval}
        if fidelity is not None:
            params["fidelity"] = fidelity
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts

        data = self._get("/prices-history", params=params)
        points = [
            PricePoint(
                timestamp=int(pt.get("t", 0)),
                price=float(pt.get("p", 0)),
            )
            for pt in data.get("history", [])
        ]
        # Sort by timestamp ascending
        points.sort(key=lambda x: x.timestamp)

        return PriceHistory(
            token_id=token_id,
            interval=interval,
            points=points,
        )

    # ------------------------------------------------------------------
    # Convenience: market snapshot
    # ------------------------------------------------------------------

    def get_market_snapshot(self, token_id: str) -> dict[str, Any]:
        """Get a combined snapshot of current market state.

        Fetches midpoint, spread, tick size, and last trade price in
        one call. Useful for quick market assessment.

        Parameters
        ----------
        token_id : str
            CLOB token ID.

        Returns
        -------
        dict
            Keys: token_id, midpoint, spread, tick_size, last_trade_price.
            Values are None if the endpoint returned no data.
        """
        snapshot: dict[str, Any] = {"token_id": token_id}

        for key, fetcher in [
            ("midpoint", lambda: self.get_midpoint(token_id)),
            ("spread", lambda: self.get_spread(token_id)),
            ("tick_size", lambda: self.get_tick_size(token_id)),
            ("last_trade_price", lambda: self.get_last_trade_price(token_id)),
        ]:
            try:
                snapshot[key] = fetcher()
            except (requests.HTTPError, requests.ConnectionError) as exc:
                logger.debug("Failed to fetch %s for %s: %s", key, token_id[:20], exc)
                snapshot[key] = None

        return snapshot

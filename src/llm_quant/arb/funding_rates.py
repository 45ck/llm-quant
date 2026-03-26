"""Crypto perpetual funding rate data pipeline.

Fetches historical and current funding rates from Binance, OKX, and Bybit
via CCXT (public endpoints, no auth required). Stores results in DuckDB.

Track C — Niche Arbitrage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import duckdb
import polars as pl

logger = logging.getLogger(__name__)

# Default symbols and exchanges
DEFAULT_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
DEFAULT_EXCHANGES = ["binance", "okx", "bybit"]

# Annualization: 3 funding periods per day * 365 days
PERIODS_PER_YEAR = 3 * 365

# Rate limit delay between API calls (seconds)
API_DELAY_SECS = 0.5

# DuckDB schema
_FUNDING_RATES_DDL = """
CREATE TABLE IF NOT EXISTS funding_rates (
    timestamp TIMESTAMPTZ NOT NULL,
    exchange VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    funding_rate DOUBLE NOT NULL,
    annualized_rate DOUBLE NOT NULL,
    mark_price DOUBLE,
    PRIMARY KEY (timestamp, exchange, symbol)
);
"""

_FUNDING_RATES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol_ts
    ON funding_rates (symbol, exchange, timestamp DESC);
"""


def annualize_funding_rate(rate_per_8h: float) -> float:
    """Convert an 8-hour funding rate to annualized rate.

    Formula: rate_per_8h * 3 * 365
    """
    return rate_per_8h * PERIODS_PER_YEAR


def init_funding_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create funding_rates table if it does not exist."""
    conn.execute(_FUNDING_RATES_DDL)
    conn.execute(_FUNDING_RATES_INDEX)
    logger.info("Funding rates schema initialized.")


def get_funding_connection(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and ensure funding tables exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    init_funding_schema(conn)
    return conn


@dataclass
class FundingRecord:
    """A single funding rate observation."""

    timestamp: datetime
    exchange: str
    symbol: str
    funding_rate: float
    annualized_rate: float
    mark_price: float | None = None


@dataclass
class FundingCollector:
    """Fetches funding rate data from crypto exchanges via CCXT."""

    exchanges: list[str] = field(default_factory=lambda: list(DEFAULT_EXCHANGES))
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    api_delay: float = API_DELAY_SECS

    def _create_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Instantiate a CCXT exchange object with safe defaults."""
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )

    def _normalize_symbol(self, symbol: str, exchange: ccxt.Exchange) -> str | None:
        """Map a canonical symbol to the exchange-specific symbol, or None."""
        try:
            exchange.load_markets()
        except Exception as exc:
            logger.warning("Failed to load markets for %s: %s", exchange.id, exc)
            return None

        if symbol in exchange.markets:
            return symbol

        # Try without the settlement suffix (e.g. BTC/USDT:USDT -> BTC/USDT)
        base = symbol.split(":", maxsplit=1)[0] if ":" in symbol else symbol
        for variant in [symbol, base, f"{base}:USDT"]:
            if variant in exchange.markets:
                return variant
        return None

    def fetch_current_rates(self) -> list[FundingRecord]:
        """Fetch current (latest) funding rates across all exchanges/symbols."""
        records: list[FundingRecord] = []

        for exch_id in self.exchanges:
            try:
                exchange = self._create_exchange(exch_id)
            except AttributeError:
                logger.warning("Exchange '%s' not supported by CCXT.", exch_id)
                continue

            for symbol in self.symbols:
                mapped = self._normalize_symbol(symbol, exchange)
                if mapped is None:
                    logger.debug(
                        "Symbol %s not available on %s, skipping.", symbol, exch_id
                    )
                    continue

                try:
                    result = exchange.fetch_funding_rate(mapped)
                    rate = result.get("fundingRate")
                    if rate is None:
                        continue

                    ts_ms = result.get("fundingTimestamp") or result.get("timestamp")
                    ts = (
                        datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                        if ts_ms
                        else datetime.now(UTC)
                    )
                    mark = result.get("markPrice")

                    # Normalize symbol to short form for storage (BTC/USDT)
                    display_symbol = symbol.split(":")[0] if ":" in symbol else symbol

                    records.append(
                        FundingRecord(
                            timestamp=ts,
                            exchange=exch_id,
                            symbol=display_symbol,
                            funding_rate=rate,
                            annualized_rate=annualize_funding_rate(rate),
                            mark_price=float(mark) if mark is not None else None,
                        )
                    )
                    logger.info(
                        "%s %s: rate=%.6f ann=%.2f%%",
                        exch_id,
                        display_symbol,
                        rate,
                        annualize_funding_rate(rate) * 100,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch funding rate %s on %s: %s",
                        symbol,
                        exch_id,
                        exc,
                    )

                time.sleep(self.api_delay)

        return records

    def fetch_history(
        self,
        days: int = 30,
    ) -> list[FundingRecord]:
        """Fetch historical funding rates for the last N days.

        Not all exchanges support historical funding rate queries.
        We handle gracefully if an exchange lacks this endpoint.
        """
        records: list[FundingRecord] = []
        since_ms = int((datetime.now(UTC).timestamp() - days * 86400) * 1000)

        for exch_id in self.exchanges:
            try:
                exchange = self._create_exchange(exch_id)
            except AttributeError:
                logger.warning("Exchange '%s' not supported by CCXT.", exch_id)
                continue

            for symbol in self.symbols:
                mapped = self._normalize_symbol(symbol, exchange)
                if mapped is None:
                    logger.debug(
                        "Symbol %s not available on %s, skipping.", symbol, exch_id
                    )
                    continue

                display_symbol = symbol.split(":")[0] if ":" in symbol else symbol
                logger.info(
                    "Fetching %d-day history for %s on %s...",
                    days,
                    display_symbol,
                    exch_id,
                )

                try:
                    history = self._fetch_history_paginated(exchange, mapped, since_ms)
                    for entry in history:
                        rate = entry.get("fundingRate")
                        if rate is None:
                            continue

                        ts_ms = entry.get("timestamp")
                        ts = (
                            datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                            if ts_ms
                            else datetime.now(UTC)
                        )
                        mark = entry.get("markPrice")

                        records.append(
                            FundingRecord(
                                timestamp=ts,
                                exchange=exch_id,
                                symbol=display_symbol,
                                funding_rate=rate,
                                annualized_rate=annualize_funding_rate(rate),
                                mark_price=float(mark) if mark is not None else None,
                            )
                        )

                    logger.info(
                        "  %s %s: fetched %d historical records.",
                        exch_id,
                        display_symbol,
                        len(
                            [
                                r
                                for r in records
                                if r.exchange == exch_id and r.symbol == display_symbol
                            ]
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "Historical funding rates not available for %s on %s: %s",
                        symbol,
                        exch_id,
                        exc,
                    )

                time.sleep(self.api_delay)

        return records

    def _fetch_history_paginated(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        since_ms: int,
    ) -> list[dict]:
        """Paginate through funding rate history to avoid API limits."""
        all_entries: list[dict] = []
        current_since = since_ms
        max_iterations = 100  # Safety limit

        for _ in range(max_iterations):
            try:
                batch = exchange.fetch_funding_rate_history(
                    symbol, since=current_since, limit=100
                )
            except (ccxt.NotSupported, ccxt.BadRequest, AttributeError):
                logger.debug(
                    "fetch_funding_rate_history not supported on %s for %s",
                    exchange.id,
                    symbol,
                )
                break
            except Exception as exc:
                logger.warning("Pagination error on %s: %s", exchange.id, exc)
                break

            if not batch:
                break

            all_entries.extend(batch)

            # Advance the cursor past the last timestamp
            last_ts = batch[-1].get("timestamp")
            if last_ts is None or last_ts <= current_since:
                break
            current_since = last_ts + 1

            time.sleep(self.api_delay)

        return all_entries


def persist_records(
    conn: duckdb.DuckDBPyConnection,
    records: list[FundingRecord],
) -> int:
    """Insert funding rate records into DuckDB. Returns count of rows inserted."""
    if not records:
        return 0

    df = pl.DataFrame(
        {
            "timestamp": [r.timestamp for r in records],
            "exchange": [r.exchange for r in records],
            "symbol": [r.symbol for r in records],
            "funding_rate": [r.funding_rate for r in records],
            "annualized_rate": [r.annualized_rate for r in records],
            "mark_price": [r.mark_price for r in records],
        }
    )

    # Deduplicate within the batch
    df = df.unique(subset=["timestamp", "exchange", "symbol"], keep="last")

    # Use INSERT OR REPLACE to handle conflicts
    conn.execute(
        """
        INSERT OR REPLACE INTO funding_rates
            (timestamp, exchange, symbol, funding_rate, annualized_rate, mark_price)
        SELECT timestamp, exchange, symbol, funding_rate, annualized_rate, mark_price
        FROM df
        """
    )
    conn.commit()

    count = len(df)
    logger.info("Persisted %d funding rate records to DuckDB.", count)
    return count


def load_rates(
    conn: duckdb.DuckDBPyConnection,
    symbol: str | None = None,
    exchange: str | None = None,
    days: int | None = None,
) -> pl.DataFrame:
    """Load funding rates from DuckDB as a Polars DataFrame."""
    query = "SELECT * FROM funding_rates WHERE 1=1"
    params: list = []

    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if exchange:
        query += " AND exchange = ?"
        params.append(exchange)
    if days:
        query += " AND timestamp >= NOW() - INTERVAL ? DAY"
        params.append(days)

    query += " ORDER BY timestamp DESC"

    result = conn.execute(query, params).fetchall()
    if not result:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "exchange": pl.Utf8,
                "symbol": pl.Utf8,
                "funding_rate": pl.Float64,
                "annualized_rate": pl.Float64,
                "mark_price": pl.Float64,
            }
        )

    columns = [
        "timestamp",
        "exchange",
        "symbol",
        "funding_rate",
        "annualized_rate",
        "mark_price",
    ]
    return pl.DataFrame(
        {col: [row[i] for row in result] for i, col in enumerate(columns)}
    )

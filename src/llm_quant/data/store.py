"""DuckDB read/write layer for market data."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from datetime import date as date_type

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

# Canonical column order matching the market_data_daily table schema.
_ALL_COLUMNS: list[str] = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "sma_20",
    "sma_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
]

_OHLCV_COLUMNS: list[str] = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
]


def upsert_market_data(
    conn: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
) -> int:
    """Insert or replace rows into ``market_data_daily``.

    The incoming DataFrame must contain at least the OHLCV columns
    (``symbol, date, open, high, low, close, volume, adj_close``).
    Indicator columns (``sma_20``, ``rsi_14``, etc.) are optional; when
    absent they will be inserted as ``NULL``.

    Parameters
    ----------
    conn:
        An open DuckDB connection with the schema already initialised.
    df:
        Polars DataFrame with market data rows.

    Returns
    -------
    int
        Number of rows upserted.
    """
    if df is None or len(df) == 0:
        logger.info("Empty DataFrame — nothing to upsert")
        return 0

    # Ensure all table columns exist in the DataFrame (fill missing with null)
    for col in _ALL_COLUMNS:
        if col not in df.columns:
            if col == "symbol":
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
            elif col == "date":
                df = df.with_columns(pl.lit(None).cast(pl.Date).alias(col))
            elif col == "volume":
                df = df.with_columns(pl.lit(None).cast(pl.Int64).alias(col))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Select columns in canonical order
    df = df.select(_ALL_COLUMNS)

    # Coerce date column to Date type if needed
    if df.schema["date"] != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))

    row_count = len(df)

    # Use row-by-row parameterized INSERT to avoid pyarrow dependency
    try:
        cols = ", ".join(_ALL_COLUMNS)
        placeholders = ", ".join(["?"] * len(_ALL_COLUMNS))
        stmt = (
            f"INSERT OR REPLACE INTO market_data_daily ({cols}) VALUES ({placeholders})"
        )
        rows = df.rows()
        conn.executemany(stmt, rows)
        conn.commit()
    except duckdb.Error:
        logger.exception("Failed to upsert %d rows into market_data_daily", row_count)
        raise

    logger.info("Upserted %d rows into market_data_daily", row_count)
    return row_count


def get_market_data(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start_date: date_type | str,
    end_date: date_type | str | None = None,
) -> pl.DataFrame:
    """Read market data from the database for given symbols and date range.

    Parameters
    ----------
    conn:
        An open DuckDB connection.
    symbols:
        List of ticker symbols to retrieve.
    start_date:
        Earliest date (inclusive).  Accepts :class:`datetime.date` or an
        ISO-format string (``"2024-01-15"``).
    end_date:
        Latest date (inclusive).  When *None*, defaults to today.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with all columns from ``market_data_daily``,
        sorted by ``(symbol, date)``.  Empty DataFrame (with correct schema)
        when no rows match.
    """
    if not symbols:
        logger.warning("No symbols provided — returning empty DataFrame")
        return pl.DataFrame(
            schema={
                c: (
                    pl.Utf8 if c == "symbol" else pl.Date if c == "date" else pl.Float64
                )
                for c in _ALL_COLUMNS
            }
        )

    if end_date is None:
        end_date = datetime.now(tz=UTC).date()

    start_str = str(start_date)
    end_str = str(end_date)

    # Build parameterised query
    placeholders = ", ".join(["?"] * len(symbols))
    query = (
        f"SELECT * FROM market_data_daily "
        f"WHERE symbol IN ({placeholders}) "
        f"AND date >= ? AND date <= ? "
        f"ORDER BY symbol, date"
    )
    params = [*symbols, start_str, end_str]

    try:
        df = conn.execute(query, params).pl()
    except duckdb.Error:
        logger.exception(
            "Failed to query market_data_daily for symbols=%s, range=[%s, %s]",
            symbols,
            start_str,
            end_str,
        )
        raise

    logger.info(
        "Retrieved %d rows for %d symbols from %s to %s",
        len(df),
        len(symbols),
        start_str,
        end_str,
    )
    return df


def get_latest_date(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
) -> date_type | None:
    """Return the most recent date stored for *symbol*, or None.

    Useful for determining where an incremental fetch should start.

    Parameters
    ----------
    conn:
        An open DuckDB connection.
    symbol:
        The ticker symbol to look up.

    Returns
    -------
    datetime.date | None
        The latest date, or ``None`` if the symbol has no data.
    """
    try:
        result = conn.execute(
            "SELECT MAX(date) AS latest FROM market_data_daily WHERE symbol = ?",
            [symbol],
        ).fetchone()
    except duckdb.Error:
        logger.exception("Failed to query latest date for %s", symbol)
        raise

    if result is None or result[0] is None:
        logger.debug("No data found for symbol %s", symbol)
        return None

    latest = result[0]

    # DuckDB may return a datetime.date, datetime.datetime, or string
    if isinstance(latest, str):
        latest = datetime.strptime(latest, "%Y-%m-%d").replace(tzinfo=UTC).date()
    elif hasattr(latest, "date"):
        # datetime object
        latest = latest.date()

    logger.debug("Latest date for %s: %s", symbol, latest)
    return latest

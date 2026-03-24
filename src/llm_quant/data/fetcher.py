"""Fetch OHLCV data from Yahoo Finance via *yfinance*."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbols: list[str],
    lookback_days: int = 252,
    timeout: int = 30,
) -> pl.DataFrame:
    """Download daily OHLCV data for the given symbols.

    Uses :func:`yfinance.download` with ``group_by='ticker'`` so that a
    single request is made for all tickers.  Symbols that fail to download
    are logged and silently skipped.

    Parameters
    ----------
    symbols:
        Ticker symbols to fetch (e.g. ``["SPY", "QQQ"]``).
    lookback_days:
        Number of calendar days of history to request.  Defaults to 252
        (roughly one trading year).
    timeout:
        HTTP timeout in seconds passed to *yfinance*.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:
        ``symbol, date, open, high, low, close, volume, adj_close``.
        Sorted by ``(symbol, date)``.  Returns an empty DataFrame with
        the correct schema when no data could be fetched.
    """
    empty_schema = {
        "symbol": pl.Utf8,
        "date": pl.Date,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "adj_close": pl.Float64,
    }

    if not symbols:
        logger.warning("No symbols provided — returning empty DataFrame")
        return pl.DataFrame(schema=empty_schema)

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    logger.info(
        "Fetching OHLCV for %d symbols from %s to %s (timeout=%ds)",
        len(symbols),
        start_date,
        end_date,
        timeout,
    )

    try:
        raw = yf.download(
            tickers=symbols,
            start=str(start_date),
            end=str(end_date),
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            timeout=timeout,
        )
    except Exception:
        logger.exception("yfinance.download() raised an unexpected error")
        return pl.DataFrame(schema=empty_schema)

    if raw is None or raw.empty:
        logger.warning("yfinance returned no data for %s", symbols)
        return pl.DataFrame(schema=empty_schema)

    frames: list[pl.DataFrame] = []

    # Single-symbol download: columns are flat (Open, High, …)
    if len(symbols) == 1:
        symbol = symbols[0]
        try:
            df = _pandas_to_polars(raw, symbol)
            if df is not None and len(df) > 0:
                frames.append(df)
            else:
                logger.warning("No rows returned for %s", symbol)
        except Exception:
            logger.exception("Failed to process data for %s", symbol)
    else:
        # Multi-symbol download: top-level columns are (ticker, field) MultiIndex
        for symbol in symbols:
            try:
                if symbol not in raw.columns.get_level_values(0):
                    logger.warning("Symbol %s not found in downloaded data", symbol)
                    continue
                subset = raw[symbol].copy()
                # Drop rows that are entirely NaN (symbol had no data on that date)
                subset = subset.dropna(how="all")
                if subset.empty:
                    logger.warning("No rows returned for %s", symbol)
                    continue
                df = _pandas_to_polars(subset, symbol)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception:
                logger.exception("Failed to process data for %s", symbol)

    if not frames:
        logger.warning("No valid data after processing — returning empty DataFrame")
        return pl.DataFrame(schema=empty_schema)

    result = pl.concat(frames, how="vertical").sort(["symbol", "date"])
    logger.info("Fetched %d total rows for %d symbols", len(result), len(frames))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pandas_to_polars(pandas_df: object, symbol: str) -> pl.DataFrame | None:
    """Convert a single-symbol pandas OHLCV DataFrame to Polars.

    Handles column-name normalisation (Yahoo may capitalise differently)
    and adds the ``symbol`` column.

    Returns ``None`` when the conversion fails.
    """
    import pandas as pd  # local import; only used for conversion

    if not isinstance(pandas_df, pd.DataFrame):
        return None

    pdf: pd.DataFrame = pandas_df.copy()
    pdf = pdf.reset_index()

    # Normalise column names to lowercase
    pdf.columns = [str(c).lower().strip() for c in pdf.columns]

    # Map possible column names from yfinance to our canonical names
    rename_map: dict[str, str] = {
        "adj close": "adj_close",
        "adjclose": "adj_close",
    }
    pdf.rename(columns=rename_map, inplace=True)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(pdf.columns)
    if missing:
        logger.warning("Symbol %s: missing columns %s — skipping", symbol, missing)
        return None

    # Ensure adj_close exists (fall back to close)
    if "adj_close" not in pdf.columns:
        pdf["adj_close"] = pdf["close"]

    pdf["symbol"] = symbol

    # Select the columns we need in canonical order
    pdf = pdf[["symbol", "date", "open", "high", "low", "close", "volume", "adj_close"]]

    # Drop rows with NaN in critical price columns
    pdf = pdf.dropna(subset=["close"])

    # Convert to Polars
    df = pl.from_pandas(pdf)

    # Coerce types
    df = df.with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        pl.col("adj_close").cast(pl.Float64),
        pl.col("symbol").cast(pl.Utf8),
    )

    return df

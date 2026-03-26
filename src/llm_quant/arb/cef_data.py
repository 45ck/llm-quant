"""CEF (Closed-End Fund) data pipeline.

Fetches CEF market prices from yfinance and estimates NAV using
benchmark ETF proxies. Stores results in a DuckDB table.

CEFs trade at market prices that deviate from their Net Asset Value (NAV).
Since yfinance does not reliably provide NAV for all CEFs, we estimate
NAV by tracking the CEF's benchmark ETF and applying a regression-based
mapping calibrated on the historical price-to-benchmark ratio.
"""

from __future__ import annotations

import logging
from datetime import date

import duckdb
import polars as pl

from llm_quant.data.fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

# CEF tickers and their benchmark ETF proxies for NAV estimation.
# The proxy ETF tracks a similar index/asset class as the CEF's holdings.
CEF_BENCHMARK_MAP: dict[str, str] = {
    # Bond CEFs → AGG (aggregate bond)
    "NEA": "AGG",
    "NAD": "AGG",
    "PDI": "HYG",
    "PTY": "HYG",
    "HYT": "HYG",
    "EHI": "HYG",
    "AWF": "AGG",
    "BGT": "AGG",
    # Muni CEFs → MUB (muni bond)
    "NVG": "MUB",
    "VPV": "MUB",
    "BLE": "MUB",
    "MQY": "MUB",
    # Equity CEFs → SPY (broad equity)
    "ADX": "SPY",
    "GAM": "SPY",
    "GDV": "SPY",
}

# Default CEF universe
DEFAULT_CEF_TICKERS: list[str] = list(CEF_BENCHMARK_MAP.keys())


def fetch_cef_data(
    cef_tickers: list[str] | None = None,
    lookback_days: int = 252 * 5,
) -> pl.DataFrame:
    """Fetch CEF price data and estimate NAV from benchmark proxies.

    Parameters
    ----------
    cef_tickers:
        CEF tickers to fetch. Defaults to DEFAULT_CEF_TICKERS.
    lookback_days:
        Calendar days of history to request.

    Returns
    -------
    pl.DataFrame
        Columns: date, ticker, price, nav_estimate, discount_pct, volume
    """
    if cef_tickers is None:
        cef_tickers = DEFAULT_CEF_TICKERS

    # Collect unique benchmark ETFs needed
    benchmarks = set()
    for ticker in cef_tickers:
        bm = CEF_BENCHMARK_MAP.get(ticker)
        if bm:
            benchmarks.add(bm)

    all_symbols = list(set(cef_tickers) | benchmarks)
    logger.info(
        "Fetching %d CEFs + %d benchmarks (%d total)",
        len(cef_tickers),
        len(benchmarks),
        len(all_symbols),
    )

    raw_df = fetch_ohlcv(all_symbols, lookback_days=lookback_days)
    if len(raw_df) == 0:
        logger.warning("No data fetched for CEFs")
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "ticker": pl.Utf8,
                "price": pl.Float64,
                "nav_estimate": pl.Float64,
                "discount_pct": pl.Float64,
                "volume": pl.Int64,
            }
        )

    frames: list[pl.DataFrame] = []
    for ticker in cef_tickers:
        benchmark = CEF_BENCHMARK_MAP.get(ticker)
        if benchmark is None:
            logger.warning("No benchmark for %s, skipping NAV estimation", ticker)
            continue

        cef_df = _estimate_nav_for_cef(raw_df, ticker, benchmark)
        if cef_df is not None and len(cef_df) > 0:
            frames.append(cef_df)

    if not frames:
        logger.warning("No CEF data produced after NAV estimation")
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "ticker": pl.Utf8,
                "price": pl.Float64,
                "nav_estimate": pl.Float64,
                "discount_pct": pl.Float64,
                "volume": pl.Int64,
            }
        )

    return pl.concat(frames, how="vertical").sort(["ticker", "date"])


def _estimate_nav_for_cef(
    raw_df: pl.DataFrame,
    cef_ticker: str,
    benchmark_ticker: str,
) -> pl.DataFrame | None:
    """Estimate NAV for a single CEF using its benchmark ETF.

    Method: Use the rolling median of (CEF_price / benchmark_price) over
    the first 60 days as a calibration ratio. Then:
        NAV_estimate(t) = benchmark_price(t) * calibration_ratio

    This captures the structural relationship while allowing the market
    price to deviate (the discount/premium we want to trade).
    """
    cef_data = (
        raw_df.filter(pl.col("symbol") == cef_ticker)
        .sort("date")
        .select(["date", "close", "volume"])
        .rename({"close": "price"})
    )
    bm_data = (
        raw_df.filter(pl.col("symbol") == benchmark_ticker)
        .sort("date")
        .select(["date", "close"])
        .rename({"close": "bm_price"})
    )

    if len(cef_data) < 60 or len(bm_data) < 60:
        logger.warning(
            "Insufficient data for %s or %s (need 60 days)",
            cef_ticker,
            benchmark_ticker,
        )
        return None

    # Join on date
    joined = cef_data.join(bm_data, on="date", how="inner").sort("date")
    if len(joined) < 60:
        logger.warning("Insufficient overlapping dates for %s", cef_ticker)
        return None

    # Compute rolling ratio and calibration
    joined = joined.with_columns(
        (pl.col("price") / pl.col("bm_price")).alias("ratio"),
    )

    # Use expanding median of the ratio as the "fair" ratio.
    # This adapts over time as the fund's NAV/benchmark relationship drifts.
    joined = joined.with_columns(
        pl.col("ratio")
        .rolling_median(window_size=252, min_samples=60)
        .alias("fair_ratio"),
    )

    # NAV estimate = benchmark_price * fair_ratio
    joined = joined.with_columns(
        (pl.col("bm_price") * pl.col("fair_ratio")).alias("nav_estimate"),
    )

    joined = joined.with_columns(
        ((pl.col("price") - pl.col("nav_estimate")) / pl.col("nav_estimate")).alias(
            "discount_pct"
        ),
    )

    # Filter out rows where nav_estimate is null (warmup period)
    result = (
        joined.filter(pl.col("nav_estimate").is_not_null())
        .with_columns(pl.lit(cef_ticker).alias("ticker"))
        .select(["date", "ticker", "price", "nav_estimate", "discount_pct", "volume"])
    )

    logger.info(
        "CEF %s: %d rows, avg discount %.2f%%",
        cef_ticker,
        len(result),
        result["discount_pct"].mean() * 100 if len(result) > 0 else 0,
    )

    return result


def store_cef_data(
    conn: duckdb.DuckDBPyConnection,
    cef_df: pl.DataFrame,
) -> int:
    """Store CEF data in DuckDB table cef_daily.

    Creates the table if it does not exist. Upserts on (date, ticker).

    Returns the number of rows stored.
    """
    if len(cef_df) == 0:
        return 0

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cef_daily (
            date DATE NOT NULL,
            ticker VARCHAR NOT NULL,
            price DOUBLE NOT NULL,
            nav_estimate DOUBLE,
            discount_pct DOUBLE,
            volume BIGINT,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Use INSERT OR REPLACE for upsert semantics
    conn.execute("""
        INSERT OR REPLACE INTO cef_daily
        SELECT date, ticker, price, nav_estimate, discount_pct, volume
        FROM cef_df
    """)

    row_count = len(cef_df)
    logger.info("Stored %d CEF rows in cef_daily", row_count)
    return row_count


def load_cef_data(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str] | None = None,
    start_date: date | None = None,
) -> pl.DataFrame:
    """Load CEF data from DuckDB.

    Parameters
    ----------
    conn:
        DuckDB connection.
    tickers:
        Filter to specific tickers. None = all.
    start_date:
        Filter to dates >= start_date. None = all.

    Returns
    -------
    pl.DataFrame
        CEF daily data.
    """
    query = "SELECT * FROM cef_daily WHERE 1=1"
    params: list = []

    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)

    query += " ORDER BY ticker, date"

    return conn.execute(query, params).pl()

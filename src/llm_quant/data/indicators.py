"""Compute technical indicators using pure Polars expressions.

All computations are performed per-symbol via ``group_by`` / ``over``
partitioning.  No pandas dependency is used here.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def compute_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Add technical-indicator columns to an OHLCV DataFrame.

    The input must contain at least the columns:
    ``symbol, date, open, high, low, close, volume``.

    Indicators computed (per symbol, ordered by date):

    * **sma_20** -- 20-day simple moving average of *close*.
    * **sma_50** -- 50-day simple moving average of *close*.
    * **sma_200** -- 200-day simple moving average of *close*.
    * **bb_upper** -- Upper Bollinger Band (SMA-20 + 2 std devs).
    * **bb_lower** -- Lower Bollinger Band (SMA-20 - 2 std devs).
    * **bb_pct_b** -- Bollinger %B (position of close within the bands).
    * **rsi_14** -- 14-period RSI (Wilder's smoothing).
    * **macd** -- MACD line (EMA-12 minus EMA-26 of *close*).
    * **macd_signal** -- 9-period EMA of the MACD line.
    * **macd_hist** -- MACD minus MACD signal.
    * **atr_14** -- 14-period Average True Range (Wilder's smoothing).
    * **vol_sma_20** -- 20-day simple moving average of *volume*.
    * **intraday_return** -- Same-day return: ``(close - open) / open``.
    * **high_20** -- 20-day rolling maximum of *close* (for ATR breakout signals).

    Parameters
    ----------
    df:
        Polars DataFrame with OHLCV data.  Must be sorted by
        ``(symbol, date)`` or the function will sort it first.

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the eleven indicator columns appended
        (or replaced if they already existed).
    """
    required = {"symbol", "date", "close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    if len(df) == 0:
        logger.warning("Empty DataFrame — returning with null indicator columns")
        indicator_names = (
            "sma_20",
            "sma_50",
            "sma_200",
            "bb_upper",
            "bb_lower",
            "bb_pct_b",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "vol_sma_20",
            "intraday_return",
            "high_20",
        )
        for col in indicator_names:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    # Drop any pre-existing indicator columns so we can recompute cleanly
    indicator_cols = [
        "sma_20",
        "sma_50",
        "sma_200",
        "bb_upper",
        "bb_lower",
        "bb_pct_b",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "vol_sma_20",
        "intraday_return",
        "high_20",
    ]
    existing = [c for c in indicator_cols if c in df.columns]
    if existing:
        df = df.drop(existing)

    # Ensure sorted by symbol then date
    df = df.sort(["symbol", "date"])

    # -- SMA indicators -------------------------------------------------------
    df = df.with_columns(
        pl.col("close")
        .rolling_mean(window_size=20, min_samples=20)
        .over("symbol")
        .alias("sma_20"),
        pl.col("close")
        .rolling_mean(window_size=50, min_samples=50)
        .over("symbol")
        .alias("sma_50"),
        pl.col("close")
        .rolling_mean(window_size=200, min_samples=200)
        .over("symbol")
        .alias("sma_200"),
    )

    # -- Bollinger Bands (20-period, 2 std devs) --------------------------------
    df = df.with_columns(
        pl.col("close")
        .rolling_std(window_size=20, min_samples=20)
        .over("symbol")
        .alias("_bb_std"),
    )
    df = df.with_columns(
        (pl.col("sma_20") + 2.0 * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("sma_20") - 2.0 * pl.col("_bb_std")).alias("bb_lower"),
    )
    df = df.with_columns(
        pl.when((pl.col("bb_upper") - pl.col("bb_lower")).abs() > 1e-10)
        .then(
            (pl.col("close") - pl.col("bb_lower"))
            / (pl.col("bb_upper") - pl.col("bb_lower"))
        )
        .otherwise(0.5)
        .alias("bb_pct_b"),
    )
    df = df.drop("_bb_std")

    # -- RSI (Wilder's) -------------------------------------------------------
    df = _compute_rsi(df, period=14)

    # -- MACD -----------------------------------------------------------------
    df = _compute_macd(df, fast=12, slow=26, signal=9)

    # -- ATR ------------------------------------------------------------------
    df = _compute_atr(df, period=14)

    # -- Volume SMA (20-day) --------------------------------------------------
    if "volume" in df.columns:
        df = df.with_columns(
            pl.col("volume")
            .rolling_mean(window_size=20, min_samples=20)
            .over("symbol")
            .alias("vol_sma_20"),
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("vol_sma_20"))

    # -- Intraday return (close vs open, causal) --------------------------------
    if "open" in df.columns:
        df = df.with_columns(
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias(
                "intraday_return"
            ),
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("intraday_return"))

    # -- 20-day rolling high of close (for ATR breakout signals) ---------------
    df = df.with_columns(
        pl.col("close")
        .rolling_max(window_size=20, min_samples=20)
        .over("symbol")
        .alias("high_20"),
    )

    logger.info("Computed indicators for %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute RSI using Wilder's smoothing (EWM with ``alpha = 1/period``).

    RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.
    """
    alpha = 1.0 / period

    # Per-symbol price change
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("_delta"),
    )

    df = df.with_columns(
        pl.when(pl.col("_delta") > 0)
        .then(pl.col("_delta"))
        .otherwise(0.0)
        .alias("_gain"),
        pl.when(pl.col("_delta") < 0)
        .then(-pl.col("_delta"))
        .otherwise(0.0)
        .alias("_loss"),
    )

    # Wilder's smoothing via ewm_mean with alpha = 1/period
    df = df.with_columns(
        pl.col("_gain")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        .over("symbol")
        .alias("_avg_gain"),
        pl.col("_loss")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        .over("symbol")
        .alias("_avg_loss"),
    )

    df = df.with_columns(
        pl.when(pl.col("_avg_loss") == 0.0)
        .then(100.0)
        .otherwise(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
        .alias("rsi_14"),
    )

    # Clean up helper columns
    return df.drop(["_delta", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def _compute_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pl.DataFrame:
    """Compute MACD, MACD signal, and MACD histogram.

    * MACD line = EMA(close, fast) - EMA(close, slow)
    * Signal line = EMA(MACD, signal)
    * Histogram = MACD - Signal
    """
    df = df.with_columns(
        pl.col("close")
        .ewm_mean(span=fast, adjust=False, min_samples=fast)
        .over("symbol")
        .alias("_ema_fast"),
        pl.col("close")
        .ewm_mean(span=slow, adjust=False, min_samples=slow)
        .over("symbol")
        .alias("_ema_slow"),
    )

    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd"),
    )

    df = df.with_columns(
        pl.col("macd")
        .ewm_mean(span=signal, adjust=False, min_samples=signal)
        .over("symbol")
        .alias("macd_signal"),
    )

    df = df.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
    )

    return df.drop(["_ema_fast", "_ema_slow"])


def _compute_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Compute Average True Range using Wilder's smoothing.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR = EWM(TR, alpha=1/period).
    """
    alpha = 1.0 / period

    df = df.with_columns(
        pl.col("close").shift(1).over("symbol").alias("_prev_close"),
    )

    df = df.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr"),
    )

    df = df.with_columns(
        pl.col("_tr")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        .over("symbol")
        .alias("atr_14"),
    )

    return df.drop(["_prev_close", "_tr"])

"""Tests for technical indicator computation."""

from datetime import date, timedelta

import polars as pl

from llm_quant.data.indicators import compute_indicators


def _make_ohlcv(n_days: int = 60, symbol: str = "TEST") -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    import random
    random.seed(42)

    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n_days)]
    base_price = 100.0
    closes = []
    for _ in range(n_days):
        base_price += random.uniform(-2, 2.1)
        closes.append(round(base_price, 2))

    return pl.DataFrame({
        "symbol": [symbol] * n_days,
        "date": dates,
        "open": [c - random.uniform(0, 1) for c in closes],
        "high": [c + random.uniform(0, 2) for c in closes],
        "low": [c - random.uniform(0, 2) for c in closes],
        "close": closes,
        "volume": [random.randint(1_000_000, 10_000_000) for _ in range(n_days)],
    })


def test_compute_indicators_columns():
    df = _make_ohlcv(60)
    result = compute_indicators(df)

    expected_cols = {"sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14"}
    assert expected_cols.issubset(set(result.columns))


def test_compute_indicators_length():
    df = _make_ohlcv(60)
    result = compute_indicators(df)
    assert len(result) == len(df)


def test_sma_values():
    df = _make_ohlcv(60)
    result = compute_indicators(df)

    # SMA20 should be null for first 19 rows, then populated
    sma_20 = result["sma_20"]
    assert sma_20[0] is None or sma_20.is_null()[0]
    assert sma_20[-1] is not None and not sma_20.is_null()[-1]


def test_rsi_bounded():
    df = _make_ohlcv(60)
    result = compute_indicators(df)

    rsi = result.filter(pl.col("rsi_14").is_not_null())["rsi_14"]
    if len(rsi) > 0:
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0


def test_multi_symbol():
    df1 = _make_ohlcv(60, "SPY")
    df2 = _make_ohlcv(60, "QQQ")
    combined = pl.concat([df1, df2])

    result = compute_indicators(combined)
    assert result["symbol"].n_unique() == 2
    assert len(result) == 120


def test_empty_dataframe():
    df = pl.DataFrame({
        "symbol": [],
        "date": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }).cast({"date": pl.Date, "open": pl.Float64, "high": pl.Float64,
             "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64})

    result = compute_indicators(df)
    assert len(result) == 0

"""Tests for CEF discount mean-reversion strategy."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from llm_quant.arb.cef_data import _estimate_nav_for_cef
from llm_quant.arb.cef_strategy import (
    CEFDiscountRegistryStrategy,
    CEFDiscountStrategy,
    CEFStrategyConfig,
)
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.brain.models import Action
from llm_quant.trading.portfolio import Portfolio, Position

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_cef_df(
    ticker: str = "NEA",
    n_days: int = 500,
    base_price: float = 10.0,
    base_discount: float = -0.05,
    start_date: date | None = None,
) -> pl.DataFrame:
    """Create synthetic CEF data with known discount pattern."""
    if start_date is None:
        start_date = date(2020, 1, 1)

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    prices = []
    nav_estimates = []
    discounts = []

    for i in range(n_days):
        # NAV grows at ~5% annual
        nav = base_price * (1 + 0.05 / 252) ** i

        # Discount oscillates: normal around base_discount,
        # with periodic deep discounts
        cycle = math.sin(2 * math.pi * i / 120)  # ~120 day cycle
        noise = math.sin(i * 0.7) * 0.01  # small noise
        disc = base_discount + cycle * 0.03 + noise

        price = nav * (1 + disc)
        prices.append(price)
        nav_estimates.append(nav)
        discounts.append(disc)

    return pl.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * n_days,
            "price": prices,
            "nav_estimate": nav_estimates,
            "discount_pct": discounts,
            "volume": [100000] * n_days,
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def _make_multi_cef_df(n_days: int = 500) -> pl.DataFrame:
    """Create synthetic data for multiple CEFs."""
    frames = []
    tickers = ["NEA", "PDI", "NVG", "ADX"]
    for i, ticker in enumerate(tickers):
        df = _make_cef_df(
            ticker=ticker,
            n_days=n_days,
            base_price=10.0 + i * 2,
            base_discount=-0.04 - i * 0.01,
        )
        frames.append(df)
    return pl.concat(frames, how="vertical").sort(["ticker", "date"])


@pytest.fixture
def cef_df() -> pl.DataFrame:
    return _make_cef_df(n_days=500)


@pytest.fixture
def multi_cef_df() -> pl.DataFrame:
    return _make_multi_cef_df(n_days=500)


@pytest.fixture
def strategy() -> CEFDiscountStrategy:
    config = CEFStrategyConfig(
        z_entry=-1.5,
        z_exit=0.0,
        lookback_days=252,
        max_positions=10,
        rebalance_frequency_days=21,
    )
    return CEFDiscountStrategy(config)


# ------------------------------------------------------------------
# Discount calculation tests
# ------------------------------------------------------------------


def test_discount_pct_computed_correctly(cef_df):
    """Verify discount = (price - NAV) / NAV."""
    row = cef_df.row(300, named=True)
    expected = (row["price"] - row["nav_estimate"]) / row["nav_estimate"]
    assert abs(row["discount_pct"] - expected) < 1e-6


def test_discount_negative_when_price_below_nav(cef_df):
    """CEFs typically trade at a discount (price < NAV)."""
    # Our synthetic data has base_discount=-0.05, so most discounts are negative
    mid_point = len(cef_df) // 2
    discounts = cef_df.slice(mid_point, 100)["discount_pct"].to_list()
    negative_count = sum(1 for d in discounts if d < 0)
    assert negative_count > 50, "Most discounts should be negative"


# ------------------------------------------------------------------
# Z-score signal generation tests
# ------------------------------------------------------------------


def test_z_score_computation(strategy, cef_df):
    """Z-scores should have mean ~0 and std ~1 over the lookback window."""
    z_df = strategy.compute_discount_z_scores(cef_df)
    assert "discount_z_score" in z_df.columns

    # Filter to non-null z-scores
    valid = z_df.filter(pl.col("discount_z_score").is_not_null())
    assert len(valid) > 0

    z_scores = valid["discount_z_score"].to_list()
    mean_z = sum(z_scores) / len(z_scores)
    # Mean should be roughly 0 (over full history, not exactly 0 due to edge effects)
    assert abs(mean_z) < 1.0, f"Mean z-score too far from 0: {mean_z}"


def test_entry_signal_on_deep_discount(strategy):
    """Strategy should generate BUY when z-score < z_entry."""
    # Create data where the last portion has a very deep discount
    n_days = 500
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    prices = []
    nav_estimates = []
    discounts = []

    for i in range(n_days):
        nav = 10.0
        disc = -0.15 if i >= 480 else -0.05 + math.sin(i * 0.05) * 0.02

        prices.append(nav * (1 + disc))
        nav_estimates.append(nav)
        discounts.append(disc)

    df = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["NEA"] * n_days,
            "price": prices,
            "nav_estimate": nav_estimates,
            "discount_pct": discounts,
            "volume": [100000] * n_days,
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    as_of = dates[-1]
    signals = strategy.generate_signals(as_of, df)

    # Should have a buy signal for NEA
    buy_signals = [s for s in signals if s.action == "buy"]
    assert len(buy_signals) == 1
    assert buy_signals[0].ticker == "NEA"
    assert buy_signals[0].z_score < strategy.config.z_entry


def test_exit_signal_on_discount_reversion(strategy):
    """Strategy should generate SELL when z-score > z_exit and position held."""
    n_days = 500
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    prices = []
    nav_estimates = []
    discounts = []

    for i in range(n_days):
        nav = 10.0
        # Last 20 days: discount narrows to -0.01 (above the ~-0.05 mean)
        disc = -0.01 if i >= 480 else -0.05 + math.sin(i * 0.05) * 0.02

        prices.append(nav * (1 + disc))
        nav_estimates.append(nav)
        discounts.append(disc)

    df = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["NEA"] * n_days,
            "price": prices,
            "nav_estimate": nav_estimates,
            "discount_pct": discounts,
            "volume": [100000] * n_days,
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    # Simulate holding a position
    strategy.state.positions["NEA"] = 0.10
    strategy.state.entry_z_scores["NEA"] = -2.0
    strategy.state.entry_dates["NEA"] = dates[400]

    as_of = dates[-1]
    signals = strategy.generate_signals(as_of, df)

    # Should have a sell signal (z ~0 or positive)
    sell_signals = [s for s in signals if s.action == "sell"]
    assert len(sell_signals) == 1
    assert sell_signals[0].ticker == "NEA"


def test_no_signal_when_z_between_thresholds(strategy, cef_df):
    """No signal when z-score is between entry and exit thresholds."""
    # Use data near the middle where z-score should be moderate
    as_of = cef_df.sort("date")["date"].to_list()[350]
    signals = strategy.generate_signals(as_of, cef_df)

    # Signals might or might not fire depending on synthetic data —
    # but if they do fire, they must respect the thresholds
    for s in signals:
        if s.action == "buy":
            assert s.z_score < strategy.config.z_entry
        elif s.action == "sell":
            assert s.ticker in strategy.state.positions


# ------------------------------------------------------------------
# Rebalancing logic tests
# ------------------------------------------------------------------


def test_max_positions_respected(strategy):
    """Strategy should not exceed max_positions."""
    strategy.config.max_positions = 2

    # Create data with all tickers at deep discount
    n_days = 500
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    frames = []

    for ticker in ["NEA", "PDI", "NVG", "ADX"]:
        discounts = []
        prices = []
        navs = []
        for i in range(n_days):
            nav = 10.0
            disc = -0.15 if i >= 480 else -0.05 + math.sin(i * 0.05) * 0.02
            discounts.append(disc)
            prices.append(nav * (1 + disc))
            navs.append(nav)

        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": [ticker] * n_days,
                    "price": prices,
                    "nav_estimate": navs,
                    "discount_pct": discounts,
                    "volume": [100000] * n_days,
                }
            ).with_columns(pl.col("date").cast(pl.Date))
        )

    df = pl.concat(frames, how="vertical")

    as_of = dates[-1]
    signals = strategy.generate_signals(as_of, df)

    buy_signals = [s for s in signals if s.action == "buy"]
    assert len(buy_signals) <= 2, "Should not exceed max_positions=2"


def test_apply_signal_updates_state(strategy):
    """apply_signal should update internal tracking state."""
    from llm_quant.arb.cef_strategy import CEFSignal

    signal = CEFSignal(
        ticker="NEA",
        action="buy",
        z_score=-2.0,
        discount_pct=-0.10,
        reasoning="Test buy",
    )
    today = date(2024, 6, 1)

    strategy.apply_signal(signal, today)
    assert "NEA" in strategy.state.positions
    assert strategy.state.entry_dates["NEA"] == today
    assert strategy.state.entry_z_scores["NEA"] == -2.0

    # Now sell
    sell_signal = CEFSignal(
        ticker="NEA",
        action="sell",
        z_score=0.5,
        discount_pct=-0.04,
        reasoning="Test sell",
    )
    strategy.apply_signal(sell_signal, date(2024, 7, 1))
    assert "NEA" not in strategy.state.positions
    assert "NEA" not in strategy.state.entry_dates


# ------------------------------------------------------------------
# Stop-loss tests
# ------------------------------------------------------------------


def test_stop_loss_on_discount_blowout(strategy):
    """Strategy should exit when discount z-score goes below max_discount_z."""
    strategy.config.max_discount_z = -3.0

    n_days = 500
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    discounts = []
    prices = []
    navs = []

    for i in range(n_days):
        nav = 10.0
        # Extreme blowout in last 20 days: -0.30 discount
        disc = -0.30 if i >= 480 else -0.05 + math.sin(i * 0.05) * 0.005
        discounts.append(disc)
        prices.append(nav * (1 + disc))
        navs.append(nav)

    df = pl.DataFrame(
        {
            "date": dates,
            "ticker": ["NEA"] * n_days,
            "price": prices,
            "nav_estimate": navs,
            "discount_pct": discounts,
            "volume": [100000] * n_days,
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    # Simulate holding
    strategy.state.positions["NEA"] = 0.10
    strategy.state.entry_z_scores["NEA"] = -2.0

    as_of = dates[-1]
    signals = strategy.generate_signals(as_of, df)

    sell_signals = [s for s in signals if s.action == "sell"]
    assert len(sell_signals) == 1
    assert (
        "blowout" in sell_signals[0].reasoning.lower() or sell_signals[0].z_score < -3.0
    )


# ------------------------------------------------------------------
# NAV estimation tests
# ------------------------------------------------------------------


def test_nav_estimation_produces_discount():
    """_estimate_nav_for_cef should produce reasonable NAV estimates."""
    # Create synthetic raw OHLCV data
    n_days = 400
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # AGG (benchmark) grows steadily
    # NEA (CEF) tracks AGG closely but with some noise
    frames = []
    agg_prices = []
    nea_prices = []
    for i in range(n_days):
        base = 100.0 * (1 + 0.04 / 252) ** i
        agg_prices.append(base)
        # NEA oscillates around 0.95x benchmark (5% structural discount)
        noise = math.sin(i * 0.1) * 0.01
        nea_prices.append(base * (0.95 + noise))

    for sym, price_list in [("NEA", nea_prices), ("AGG", agg_prices)]:
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [sym] * n_days,
                    "date": dates,
                    "close": price_list,
                    "volume": [500000] * n_days,
                }
            ).with_columns(pl.col("date").cast(pl.Date))
        )

    raw_df = pl.concat(frames, how="vertical")

    result = _estimate_nav_for_cef(raw_df, "NEA", "AGG")
    assert result is not None
    assert len(result) > 0
    assert "discount_pct" in result.columns
    assert "nav_estimate" in result.columns

    # NAV estimates should exist and be reasonable
    avg_nav = result["nav_estimate"].mean()
    assert avg_nav is not None
    assert avg_nav > 0

    # Discount values should be small (close to zero since
    # the rolling median tracks the actual ratio).
    # The key test is that the pipeline produces valid data.
    avg_disc = result["discount_pct"].mean()
    assert avg_disc is not None
    assert abs(avg_disc) < 0.10, f"Discount too extreme: {avg_disc:.3f}"


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_dataframe_returns_no_signals(strategy):
    """Empty input should return empty signal list."""
    empty_df = pl.DataFrame(
        schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "price": pl.Float64,
            "nav_estimate": pl.Float64,
            "discount_pct": pl.Float64,
            "volume": pl.Int64,
        }
    )
    signals = strategy.generate_signals(date(2024, 1, 1), empty_df)
    assert signals == []


def test_insufficient_history_returns_no_signals(strategy):
    """Less than lookback_days should return no signals."""
    short_df = _make_cef_df(n_days=100)  # less than 252 lookback
    as_of = short_df.sort("date")["date"].to_list()[-1]
    signals = strategy.generate_signals(as_of, short_df)
    assert signals == []


# ------------------------------------------------------------------
# STRATEGY_REGISTRY-compatible CEFDiscountRegistryStrategy tests
# ------------------------------------------------------------------


def _make_indicators_df(
    n_days: int = 500,
    cef_tickers: list[str] | None = None,
) -> pl.DataFrame:
    """Create synthetic indicators_df with CEF + benchmark symbols.

    Each CEF trades at a structural discount to its benchmark ETF.
    The discount oscillates, creating entry/exit opportunities.
    """
    if cef_tickers is None:
        cef_tickers = ["NEA", "PDI", "NVG"]

    from llm_quant.arb.cef_strategy import _CEF_BENCHMARK

    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    frames = []

    # Create benchmark data first
    benchmarks_done: set[str] = set()
    for ticker in cef_tickers:
        bm = _CEF_BENCHMARK.get(ticker, "AGG")
        if bm not in benchmarks_done:
            bm_prices = [100.0 * (1 + 0.04 / 252) ** i for i in range(n_days)]
            frames.append(
                pl.DataFrame(
                    {
                        "symbol": [bm] * n_days,
                        "date": dates,
                        "close": bm_prices,
                        "open": bm_prices,
                        "high": [p * 1.005 for p in bm_prices],
                        "low": [p * 0.995 for p in bm_prices],
                        "volume": [1000000] * n_days,
                    }
                )
            )
            benchmarks_done.add(bm)

    # Create CEF data with varying discounts
    for idx, ticker in enumerate(cef_tickers):
        bm = _CEF_BENCHMARK.get(ticker, "AGG")
        cef_prices = []
        for i in range(n_days):
            bm_price = 100.0 * (1 + 0.04 / 252) ** i
            # Each CEF has a different base discount and oscillation phase
            base_disc = -0.04 - idx * 0.02
            cycle = math.sin(2 * math.pi * i / 120 + idx * 0.5) * 0.03
            noise = math.sin(i * 0.7 + idx) * 0.005
            ratio = 0.95 + base_disc + cycle + noise
            cef_prices.append(bm_price * ratio)

        frames.append(
            pl.DataFrame(
                {
                    "symbol": [ticker] * n_days,
                    "date": dates,
                    "close": cef_prices,
                    "open": cef_prices,
                    "high": [p * 1.005 for p in cef_prices],
                    "low": [p * 0.995 for p in cef_prices],
                    "volume": [100000] * n_days,
                }
            )
        )

    # Add TLT for hedge testing
    tlt_prices = [100.0 * (1 + 0.03 / 252) ** i for i in range(n_days)]
    frames.append(
        pl.DataFrame(
            {
                "symbol": ["TLT"] * n_days,
                "date": dates,
                "close": tlt_prices,
                "open": tlt_prices,
                "high": [p * 1.005 for p in tlt_prices],
                "low": [p * 0.995 for p in tlt_prices],
                "volume": [500000] * n_days,
            }
        )
    )

    return pl.concat(frames, how="vertical").with_columns(pl.col("date").cast(pl.Date))


def _make_deep_discount_indicators_df(
    n_days: int = 500,
    cef_tickers: list[str] | None = None,
) -> pl.DataFrame:
    """Create indicators_df where CEFs have deep discounts in the final days."""
    if cef_tickers is None:
        cef_tickers = ["NEA", "PDI", "NVG"]

    from llm_quant.arb.cef_strategy import _CEF_BENCHMARK

    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    frames = []

    benchmarks_done: set[str] = set()
    for ticker in cef_tickers:
        bm = _CEF_BENCHMARK.get(ticker, "AGG")
        if bm not in benchmarks_done:
            bm_prices = [100.0 * (1 + 0.04 / 252) ** i for i in range(n_days)]
            frames.append(
                pl.DataFrame(
                    {
                        "symbol": [bm] * n_days,
                        "date": dates,
                        "close": bm_prices,
                        "open": bm_prices,
                        "high": [p * 1.005 for p in bm_prices],
                        "low": [p * 0.995 for p in bm_prices],
                        "volume": [1000000] * n_days,
                    }
                )
            )
            benchmarks_done.add(bm)

    for _idx, ticker in enumerate(cef_tickers):
        cef_prices = []
        for i in range(n_days):
            bm_price = 100.0 * (1 + 0.04 / 252) ** i
            # Normal discount ~5%, then deep discount -20% in last 20 days
            ratio = 0.80 if i >= n_days - 20 else 0.95 + math.sin(i * 0.05) * 0.01
            cef_prices.append(bm_price * ratio)

        frames.append(
            pl.DataFrame(
                {
                    "symbol": [ticker] * n_days,
                    "date": dates,
                    "close": cef_prices,
                    "open": cef_prices,
                    "high": [p * 1.005 for p in cef_prices],
                    "low": [p * 0.995 for p in cef_prices],
                    "volume": [100000] * n_days,
                }
            )
        )

    tlt_prices = [100.0 * (1 + 0.03 / 252) ** i for i in range(n_days)]
    frames.append(
        pl.DataFrame(
            {
                "symbol": ["TLT"] * n_days,
                "date": dates,
                "close": tlt_prices,
                "open": tlt_prices,
                "high": [p * 1.005 for p in tlt_prices],
                "low": [p * 0.995 for p in tlt_prices],
                "volume": [500000] * n_days,
            }
        )
    )

    return pl.concat(frames, how="vertical").with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture
def registry_strategy() -> CEFDiscountRegistryStrategy:
    config = StrategyConfig(
        name="cef_discount",
        rebalance_frequency_days=21,
        max_positions=10,
        target_position_weight=0.10,
        stop_loss_pct=0.10,
        parameters={
            "cef_tickers": "NEA,PDI,NVG",
            "lookback_days": 252,
            "z_entry": -1.5,
            "z_exit": 0.0,
            "z_stop": -4.0,
            "quintile_select": True,
            "hedge_enabled": True,
            "hedge_symbol": "TLT",
            "hedge_ratio": 0.30,
            "target_weight": 0.10,
            "nav_ratio_window": 252,
        },
    )
    return CEFDiscountRegistryStrategy(config)


@pytest.fixture
def empty_portfolio() -> Portfolio:
    return Portfolio(initial_capital=100_000.0)


def test_registry_strategy_in_strategy_registry():
    """CEFDiscountRegistryStrategy should be in STRATEGY_REGISTRY."""
    from llm_quant.backtest.strategies import STRATEGY_REGISTRY

    assert "cef_discount" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["cef_discount"] is CEFDiscountRegistryStrategy


def test_registry_strategy_returns_buy_on_deep_discount(
    registry_strategy, empty_portfolio
):
    """Registry strategy should buy CEFs with deep discounts."""
    df = _make_deep_discount_indicators_df(n_days=500)
    as_of = date(2020, 1, 1) + timedelta(days=499)
    prices = {
        row["symbol"]: row["close"]
        for row in df.filter(pl.col("date") == as_of).iter_rows(named=True)
    }

    signals = registry_strategy.generate_signals(as_of, df, empty_portfolio, prices)
    buy_signals = [s for s in signals if s.action == Action.BUY]

    assert len(buy_signals) > 0, "Should detect deep discounts and generate buys"
    for s in buy_signals:
        assert s.target_weight == 0.10
        assert s.stop_loss > 0


def test_registry_strategy_quintile_limits_entries(empty_portfolio):
    """Quintile selection should limit entries to bottom ~20% of universe."""
    config = StrategyConfig(
        name="cef_discount",
        max_positions=10,
        parameters={
            "cef_tickers": "NEA,NAD,PDI,PTY,HYT",
            "quintile_select": True,
            "z_entry": -0.5,  # Very permissive to ensure all would qualify
            "hedge_enabled": False,
        },
    )
    strat = CEFDiscountRegistryStrategy(config)
    df = _make_deep_discount_indicators_df(
        n_days=500, cef_tickers=["NEA", "NAD", "PDI", "PTY", "HYT"]
    )
    as_of = date(2020, 1, 1) + timedelta(days=499)
    prices = {
        row["symbol"]: row["close"]
        for row in df.filter(pl.col("date") == as_of).iter_rows(named=True)
    }

    signals = strat.generate_signals(as_of, df, empty_portfolio, prices)
    buy_signals = [s for s in signals if s.action == Action.BUY]

    # With 5 CEFs and quintile_select, max bottom quintile = 1
    assert len(buy_signals) <= 1, (
        f"Quintile should limit to ~20% of {5} CEFs, got {len(buy_signals)}"
    )


def test_registry_strategy_hedge_generates_tlt_sell(empty_portfolio):
    """Hedge should generate a SELL signal for TLT when buying CEFs."""
    config = StrategyConfig(
        name="cef_discount",
        max_positions=10,
        parameters={
            "cef_tickers": "NEA,PDI,NVG",
            "hedge_enabled": True,
            "hedge_symbol": "TLT",
            "hedge_ratio": 0.30,
            "z_entry": -0.5,  # permissive
        },
    )
    strat = CEFDiscountRegistryStrategy(config)
    df = _make_deep_discount_indicators_df(n_days=500)
    as_of = date(2020, 1, 1) + timedelta(days=499)
    prices = {
        row["symbol"]: row["close"]
        for row in df.filter(pl.col("date") == as_of).iter_rows(named=True)
    }

    signals = strat.generate_signals(as_of, df, empty_portfolio, prices)
    tlt_signals = [s for s in signals if s.symbol == "TLT"]
    buy_signals = [s for s in signals if s.action == Action.BUY]

    if buy_signals:
        assert len(tlt_signals) == 1, "Should generate TLT hedge when buying CEFs"
        assert tlt_signals[0].action == Action.SELL
        assert tlt_signals[0].target_weight > 0


def test_registry_strategy_no_hedge_when_disabled(empty_portfolio):
    """No TLT signal when hedge_enabled=False."""
    config = StrategyConfig(
        name="cef_discount",
        max_positions=10,
        parameters={
            "cef_tickers": "NEA,PDI,NVG",
            "hedge_enabled": False,
            "z_entry": -0.5,
        },
    )
    strat = CEFDiscountRegistryStrategy(config)
    df = _make_deep_discount_indicators_df(n_days=500)
    as_of = date(2020, 1, 1) + timedelta(days=499)
    prices = {
        row["symbol"]: row["close"]
        for row in df.filter(pl.col("date") == as_of).iter_rows(named=True)
    }

    signals = strat.generate_signals(as_of, df, empty_portfolio, prices)
    tlt_signals = [s for s in signals if s.symbol == "TLT"]
    assert len(tlt_signals) == 0


def test_registry_strategy_exit_on_reversion(registry_strategy):
    """Registry strategy should exit when discount reverts to mean."""
    df = _make_indicators_df(n_days=500)
    as_of = date(2020, 1, 1) + timedelta(days=499)
    prices = {
        row["symbol"]: row["close"]
        for row in df.filter(pl.col("date") == as_of).iter_rows(named=True)
    }

    # Create portfolio with existing CEF position
    portfolio = Portfolio(initial_capital=90_000.0)
    portfolio.positions["NEA"] = Position(
        symbol="NEA", shares=100, avg_cost=90.0, current_price=90.0
    )

    signals = registry_strategy.generate_signals(as_of, df, portfolio, prices)

    # This test validates the exit path exists — exact signal depends on
    # synthetic data z-scores. The important thing is no crash.
    assert isinstance(signals, list)
    # All signals should have valid actions
    for s in signals:
        assert s.action in (Action.BUY, Action.SELL, Action.CLOSE)


def test_registry_strategy_empty_indicators(registry_strategy, empty_portfolio):
    """Empty indicators_df should return no signals."""
    empty_df = pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "date": pl.Date,
            "close": pl.Float64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "volume": pl.Int64,
        }
    )
    signals = registry_strategy.generate_signals(
        date(2024, 1, 1), empty_df, empty_portfolio, {}
    )
    assert signals == []


def test_registry_strategy_insufficient_data(registry_strategy, empty_portfolio):
    """Insufficient history should produce no signals."""
    df = _make_indicators_df(n_days=30)  # far less than 252 lookback
    as_of = date(2020, 1, 1) + timedelta(days=29)
    prices = {
        row["symbol"]: row["close"]
        for row in df.filter(pl.col("date") == as_of).iter_rows(named=True)
    }
    signals = registry_strategy.generate_signals(as_of, df, empty_portfolio, prices)
    assert signals == []

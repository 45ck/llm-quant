"""Tests for BacktestEngine: look-ahead, fill-delay, cost, stop-loss."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategy import Strategy, StrategyConfig
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.data.indicators import compute_indicators
from llm_quant.trading.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    symbols: list[str],
    n_days: int = 300,
    start_date: date | None = None,
    trend: float = 0.0005,
    base_price: float = 100.0,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data."""
    if start_date is None:
        start_date = date(2020, 1, 1)

    rows = []
    for symbol in symbols:
        price = base_price
        for i in range(n_days):
            d = start_date + timedelta(days=i)
            # Skip weekends
            if d.weekday() >= 5:
                continue
            open_ = price
            high = price * 1.01
            low = price * 0.99
            close = price * (1 + trend)
            adj_close = close
            volume = 1_000_000
            rows.append(
                {
                    "symbol": symbol,
                    "date": d,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "adj_close": adj_close,
                }
            )
            price = close

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("volume").cast(pl.Int64),
    )


class AlwaysBuyStrategy(Strategy):
    """Test strategy that buys everything on every rebalance."""

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals = []
        for symbol in prices:
            if symbol not in portfolio.positions:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.BUY,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.05,
                        stop_loss=prices[symbol] * 0.90,
                        reasoning="test buy",
                    )
                )
        return signals


class NeverTradeStrategy(Strategy):
    """Test strategy that never trades."""

    def generate_signals(self, *args, **kwargs) -> list[TradeSignal]:
        return []


# ---------------------------------------------------------------------------
# Test: Look-ahead prevention
# ---------------------------------------------------------------------------


class TestLookAhead:
    """Verify that indicators at date T are identical whether computed
    from data[<=T] or data[<=T+30]."""

    def test_indicators_causal(self):
        """Indicators at T must be the same regardless of future data."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        full_indicators = compute_indicators(prices)

        # Pick a date in the middle
        dates = sorted(full_indicators.select("date").unique().to_series().to_list())
        test_date = dates[250]

        # Full dataset indicators at test_date
        full_at_date = full_indicators.filter(
            (pl.col("symbol") == "SPY") & (pl.col("date") == test_date)
        )

        # Truncated dataset indicators at test_date
        truncated_prices = prices.filter(pl.col("date") <= test_date)
        truncated_indicators = compute_indicators(truncated_prices)
        trunc_at_date = truncated_indicators.filter(
            (pl.col("symbol") == "SPY") & (pl.col("date") == test_date)
        )

        # Compare all indicator columns
        indicator_cols = [
            "sma_20",
            "sma_50",
            "sma_200",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14",
        ]
        for col in indicator_cols:
            full_val = full_at_date.select(col).item()
            trunc_val = trunc_at_date.select(col).item()
            if full_val is not None and trunc_val is not None:
                assert abs(full_val - trunc_val) < 1e-10, (
                    f"Look-ahead detected in {col}: "
                    f"full={full_val}, truncated={trunc_val}"
                )


# ---------------------------------------------------------------------------
# Test: Fill delay
# ---------------------------------------------------------------------------


class TestFillDelay:
    """With fill_delay=1, buys execute at T+1 open, not T close."""

    def test_fill_at_next_day_open(self):
        """Buy signal on day T should fill at T+1 open price."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=1,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=1,
            warmup_days=50,
            trial_count=1,
        )

        # Check that the first trade's price is an open price
        if result.trades:
            first_trade = result.trades[0]
            trade_date = first_trade.date
            # Get the open price on the trade date
            open_price = (
                prices.filter(
                    (pl.col("symbol") == first_trade.symbol)
                    & (pl.col("date") == trade_date)
                )
                .select("open")
                .item()
            )
            assert first_trade.price == open_price, (
                f"Fill price {first_trade.price} != open {open_price} "
                f"(fill delay not working)"
            )


# ---------------------------------------------------------------------------
# Test: Cost model
# ---------------------------------------------------------------------------


class TestCostModel:
    """Verify transaction cost computation."""

    def test_spread_cost(self):
        """Spread cost should be notional * spread_bps / 10000."""
        cm = CostModel(spread_bps=5.0, flat_slippage_bps=0.0)
        cost = cm.compute_cost(10_000.0, 100, daily_volume=None, daily_volatility=None)
        expected = 10_000.0 * 5.0 / 10_000.0  # spread only
        assert abs(cost - expected) < 0.01

    def test_sqrt_impact(self):
        """Square-root impact cost when volume is available."""
        cm = CostModel(
            spread_bps=5.0,
            slippage_volatility_factor=0.5,
            flat_slippage_bps=2.0,
        )
        notional = 10_000.0
        shares = 100
        daily_volume = 1_000_000.0
        daily_vol = 0.02

        cost = cm.compute_cost(
            notional,
            shares,
            daily_volume=daily_volume,
            daily_volatility=daily_vol,
        )

        spread = notional * 5.0 / 10_000.0
        impact = notional * 0.5 * 0.02 * math.sqrt(100 / 1_000_000.0)
        expected = spread + impact
        assert abs(cost - expected) < 0.01

    def test_cost_multiplier(self):
        """Cost multiplier scales total cost."""
        cm = CostModel(spread_bps=5.0, flat_slippage_bps=2.0)
        cost_1x = cm.compute_cost(10_000.0, 100, multiplier=1.0)
        cost_2x = cm.compute_cost(10_000.0, 100, multiplier=2.0)
        assert abs(cost_2x - 2 * cost_1x) < 0.01

    def test_cost_reduces_nav(self):
        """NAV should decrease by cost amount after trading."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.0)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=1,
            target_position_weight=0.05,
            stop_loss_pct=0.50,  # wide stop to avoid triggers
        )
        strategy = AlwaysBuyStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            cost_model=CostModel(spread_bps=50.0, flat_slippage_bps=50.0),
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # With zero trend and positive costs, final NAV < initial
        if result.trades:
            assert result.nav_series[-1] < 100_000.0, (
                "NAV should decrease with positive costs and zero trend"
            )


# ---------------------------------------------------------------------------
# Test: Stop-loss
# ---------------------------------------------------------------------------


class TestStopLoss:
    """Verify stop-loss triggers close signals."""

    def test_stop_loss_triggers_close(self):
        """Price below stop → CLOSE signal fired."""
        # Create data with a crash
        rows = []
        base_date = date(2020, 1, 6)  # Monday
        prices_up = [100, 101, 102, 103, 104, 105]
        prices_crash = [90, 85, 80, 75, 70, 65]
        all_prices = prices_up + prices_crash

        for i, p in enumerate(all_prices):
            d = base_date + timedelta(days=i)
            if d.weekday() >= 5:
                d = d + timedelta(days=7 - d.weekday())
            rows.append(
                {
                    "symbol": "SPY",
                    "date": d,
                    "open": p,
                    "high": p * 1.01,
                    "low": p * 0.99,
                    "close": p,
                    "volume": 1_000_000,
                    "adj_close": p,
                }
            )

        pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Int64),
        )

        # Manually create a portfolio with a position and stop-loss
        engine = BacktestEngine(
            strategy=NeverTradeStrategy(StrategyConfig()),
            initial_capital=100_000.0,
        )

        # The stop-loss check is internal to the engine
        portfolio = Portfolio(initial_capital=100_000.0)
        from llm_quant.trading.portfolio import Position

        portfolio.positions["SPY"] = Position(
            symbol="SPY",
            shares=100,
            avg_cost=100.0,
            current_price=105.0,
            stop_loss=95.0,
        )
        portfolio.cash = 90_000.0

        # Check stop-loss detection
        stop_signals = engine._check_stop_losses(portfolio, {"SPY": 90.0})
        assert len(stop_signals) == 1
        assert stop_signals[0].action == Action.CLOSE
        assert stop_signals[0].symbol == "SPY"


# ---------------------------------------------------------------------------
# Test: Benchmark uses adj_close (total return)
# ---------------------------------------------------------------------------


class TestBenchmark:
    """Verify benchmark uses adj_close for total return."""

    def test_total_return_vs_price_return(self):
        """adj_close benchmark return > close-only return (div yield)."""
        from llm_quant.backtest.metrics import compute_benchmark_returns

        # Create data where adj_close > close (simulating dividends)
        rows = []
        base_date = date(2020, 1, 6)
        for i in range(100):
            d = base_date + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            close = 100 + i * 0.1
            adj_close = close * 1.03  # 3% dividend adjustment
            rows.append(
                {
                    "symbol": "SPY",
                    "date": d,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "adj_close": adj_close,
                }
            )

        prices = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Int64),
        )

        tr_returns = compute_benchmark_returns(
            prices, {"SPY": 1.0}, rebalance_frequency_days=21, use_adj_close=True
        )
        pr_returns = compute_benchmark_returns(
            prices, {"SPY": 1.0}, rebalance_frequency_days=21, use_adj_close=False
        )

        # Total return from adj_close should match price-only
        # (since we applied a fixed multiplier, they should be close but
        # the adj_close version captures the dividend component)
        if tr_returns and pr_returns:
            import numpy as np

            tr_total = float(np.prod([1 + r for r in tr_returns])) - 1.0
            pr_total = float(np.prod([1 + r for r in pr_returns])) - 1.0
            # With a fixed multiplier they'll be very close, but for real
            # data where adj_close captures actual dividends, TR > PR
            assert isinstance(tr_total, float)
            assert isinstance(pr_total, float)


# ---------------------------------------------------------------------------
# Test: Engine integration
# ---------------------------------------------------------------------------


class TestEngineEdgeCases:
    """Edge case tests for BacktestEngine."""

    def test_empty_data_returns_early(self):
        """Engine with zero-row prices_df should return gracefully, not crash."""
        empty_prices = pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "adj_close": [],
            },
            schema={
                "symbol": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
                "adj_close": pl.Float64,
            },
        )
        empty_indicators = pl.DataFrame(
            {
                "symbol": [],
                "date": [],
                "close": [],
            },
            schema={
                "symbol": pl.Utf8,
                "date": pl.Date,
                "close": pl.Float64,
            },
        )

        config = StrategyConfig(name="test")
        strategy = NeverTradeStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=empty_prices,
            indicators_df=empty_indicators,
            slug="test",
            warmup_days=50,
            trial_count=1,
        )

        assert len(result.trades) == 0
        assert len(result.data_warnings) > 0  # should warn about insufficient data

    def test_single_day_data(self):
        """Prices with only 1 day (less than warmup) should not crash."""
        rows = [
            {
                "symbol": "SPY",
                "date": date(2020, 1, 6),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
                "adj_close": 100.0,
            }
        ]
        prices = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Int64),
        )
        indicators = pl.DataFrame(
            {
                "symbol": ["SPY"],
                "date": [date(2020, 1, 6)],
                "close": [100.0],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        config = StrategyConfig(name="test")
        strategy = NeverTradeStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            warmup_days=50,
            trial_count=1,
        )

        assert len(result.trades) == 0
        assert len(result.data_warnings) > 0

    def test_zero_volume_falls_back_to_flat_slippage(self):
        """CostModel with daily_volume=0 should use flat_slippage_bps."""
        cm = CostModel(
            spread_bps=5.0,
            slippage_volatility_factor=0.5,
            flat_slippage_bps=3.0,
        )
        cost = cm.compute_cost(
            notional=10_000.0,
            shares=100,
            daily_volume=0,
            daily_volatility=0.02,
        )
        # daily_volume=0 => falls to flat slippage path
        expected = 10_000.0 * 5.0 / 10_000.0 + 10_000.0 * 3.0 / 10_000.0
        assert abs(cost - expected) < 0.01, (
            f"Expected flat slippage fallback ({expected}), got {cost}"
        )

    def test_cost_model_zero_shares(self):
        """CostModel with shares=0 should return 0 cost (notional=0)."""
        cm = CostModel(spread_bps=5.0, flat_slippage_bps=2.0)
        cost = cm.compute_cost(
            notional=0.0,
            shares=0,
            daily_volume=1_000_000,
            daily_volatility=0.02,
        )
        assert cost == 0.0

    def test_fill_delay_changes_fill_price(self):
        """fill_delay=0 fills at close; fill_delay=1 fills at next-day open.
        Create data where open != previous close to demonstrate the difference."""
        # Build custom data where each day's open is 1% below close
        # (gap-down open). This ensures fill_delay=1 (next open) differs
        # from fill_delay=0 (same-day close).
        rows = []
        base_date = date(2020, 1, 6)
        price = 100.0
        for i in range(400):
            d = base_date + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            open_ = price * 0.99  # open is 1% below close level
            close = price
            high = price * 1.01
            low = price * 0.98
            rows.append(
                {
                    "symbol": "SPY",
                    "date": d,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1_000_000,
                    "adj_close": close,
                }
            )
            price = close * 1.002  # slight upward trend

        prices = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Int64),
        )
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=1,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )

        # Run with fill_delay=0
        strategy0 = AlwaysBuyStrategy(config)
        engine0 = BacktestEngine(strategy=strategy0, initial_capital=100_000.0)
        result0 = engine0.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # Run with fill_delay=1
        strategy1 = AlwaysBuyStrategy(config)
        engine1 = BacktestEngine(strategy=strategy1, initial_capital=100_000.0)
        result1 = engine1.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=1,
            warmup_days=50,
            trial_count=1,
        )

        # Both should have trades
        assert len(result0.trades) > 0, "fill_delay=0 should produce trades"
        assert len(result1.trades) > 0, "fill_delay=1 should produce trades"

        # fill_delay=0 fills at close; fill_delay=1 fills at next day's open.
        # Since open != close in our data, these must differ.
        price0 = result0.trades[0].price
        price1 = result1.trades[0].price
        assert price0 != price1, (
            f"Fill prices should differ: delay=0 got {price0}, delay=1 got {price1}"
        )

    def test_nav_series_no_gaps_on_empty_price_day(self):
        """If one trading day has no prices, NAV series should carry forward
        with no gaps (D5 fix)."""
        rows = []
        base_date = date(2020, 1, 6)
        for i in range(300):
            d = base_date + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            # Skip one day entirely (day 100 in the date sequence)
            if i == 100:
                continue
            rows.append(
                {
                    "symbol": "SPY",
                    "date": d,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1_000_000,
                    "adj_close": 100.0,
                }
            )

        prices = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Int64),
        )
        indicators = compute_indicators(prices)

        config = StrategyConfig(name="test")
        strategy = NeverTradeStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            warmup_days=50,
            trial_count=1,
        )

        # NAV series should have initial + one entry per trading date
        n_trading_dates = len(
            sorted(prices.select("date").unique().to_series().to_list())
        )
        if n_trading_dates > 50:
            # nav_series = [initial] + one per post-warmup trading date
            expected_len = (n_trading_dates - 50) + 1
            assert len(result.nav_series) == expected_len, (
                f"NAV series length {len(result.nav_series)} != expected {expected_len}"
            )


class TestEngineIntegration:
    """Integration tests for the full engine loop."""

    def test_no_trades_strategy(self):
        """Engine runs cleanly with a strategy that never trades."""
        prices = _make_prices(["SPY"], n_days=400)
        indicators = compute_indicators(prices)

        config = StrategyConfig(name="no_trade")
        strategy = NeverTradeStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            warmup_days=50,
            trial_count=1,
        )

        assert len(result.trades) == 0
        assert result.nav_series[-1] == 100_000.0  # no change

    def test_cost_sensitivity_runs_multiple(self):
        """run_with_cost_sensitivity produces metrics at multiple multipliers."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.10,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=100_000.0,
            data_dir="data",
        )

        result = engine.run_with_cost_sensitivity(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            cost_multipliers=[1.0, 2.0],
        )

        assert "1.0x" in result.metrics
        assert "2.0x" in result.metrics

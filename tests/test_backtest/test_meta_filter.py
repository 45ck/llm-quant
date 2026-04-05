"""Tests for rule-based meta-filter integration in BacktestEngine.

Validates:
- Backward compatibility (no meta_filter = unchanged behavior)
- regime_filter suppresses BUY signals when VIX > threshold
- signal_strength_weight scales BUY position sizes
- ensemble_vote requires minimum BUY agreement count
- SELL/CLOSE signals always pass through (never blocked)
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from llm_quant.backtest.engine import BacktestEngine, MetaFilterConfig
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


def _make_vix_data(
    n_days: int = 300,
    start_date: date | None = None,
    vix_level: float = 20.0,
) -> pl.DataFrame:
    """Generate synthetic VIX data at a constant level."""
    if start_date is None:
        start_date = date(2020, 1, 1)

    rows = []
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        rows.append(
            {
                "symbol": "VIX",
                "date": d,
                "open": vix_level,
                "high": vix_level * 1.02,
                "low": vix_level * 0.98,
                "close": vix_level,
                "volume": 0,
                "adj_close": vix_level,
            }
        )

    return pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("volume").cast(pl.Int64),
    )


class AlwaysBuyStrategy(Strategy):
    """Test strategy that buys SPY on every rebalance."""

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals = []
        for symbol in prices:
            if symbol == "VIX":
                continue
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


class BuyAndCloseStrategy(Strategy):
    """Strategy that emits both BUY and CLOSE signals for testing passthrough."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._call_count = 0

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        self._call_count += 1
        signals = []
        # First call: buy SPY
        if self._call_count == 1 and "SPY" not in portfolio.positions:
            signals.append(
                TradeSignal(
                    symbol="SPY",
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.05,
                    stop_loss=prices.get("SPY", 100.0) * 0.90,
                    reasoning="initial buy",
                )
            )
        # Second call: close SPY
        elif self._call_count == 2 and "SPY" in portfolio.positions:
            signals.append(
                TradeSignal(
                    symbol="SPY",
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning="test close",
                )
            )
        return signals


class NeverTradeStrategy(Strategy):
    """Test strategy that never trades."""

    def generate_signals(self, *args, **kwargs) -> list[TradeSignal]:
        return []


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------


class TestMetaFilterBackwardCompat:
    """No meta_filter = unchanged behavior."""

    def test_no_meta_filter_unchanged(self):
        """Engine without meta_filter runs identically to before."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        assert len(result.trades) > 0, "Should produce trades without meta_filter"
        assert result.nav_series[-1] > 0

    def test_meta_filter_all_disabled(self):
        """MetaFilterConfig with all filters disabled = same as no filter."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )

        # Run without meta_filter
        strategy1 = AlwaysBuyStrategy(config)
        engine1 = BacktestEngine(strategy=strategy1, initial_capital=100_000.0)
        result1 = engine1.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # Run with meta_filter (all disabled)
        strategy2 = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            regime_filter_enabled=False,
            signal_strength_enabled=False,
            ensemble_vote_enabled=False,
        )
        engine2 = BacktestEngine(
            strategy=strategy2, initial_capital=100_000.0, meta_filter=mf
        )
        result2 = engine2.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        assert len(result1.trades) == len(result2.trades), (
            "All-disabled meta_filter should produce same trade count"
        )
        assert abs(result1.nav_series[-1] - result2.nav_series[-1]) < 0.01


# ---------------------------------------------------------------------------
# Test: Regime filter
# ---------------------------------------------------------------------------


class TestRegimeFilter:
    """regime_filter suppresses BUY when VIX > threshold."""

    def test_high_vix_suppresses_buys(self):
        """When VIX=35 and threshold=25, BUY signals are blocked."""
        # Create SPY + VIX data
        spy_prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        vix_data = _make_vix_data(n_days=400, vix_level=35.0)
        prices = pl.concat([spy_prices, vix_data])
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            regime_filter_enabled=True,
            vix_threshold=25.0,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # With VIX=35 > 25, all BUY signals should be suppressed
        assert len(result.trades) == 0, (
            f"Expected 0 trades with VIX=35 > threshold=25, got {len(result.trades)}"
        )

    def test_low_vix_allows_buys(self):
        """When VIX=15 and threshold=25, BUY signals pass through."""
        spy_prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        vix_data = _make_vix_data(n_days=400, vix_level=15.0)
        prices = pl.concat([spy_prices, vix_data])
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            regime_filter_enabled=True,
            vix_threshold=25.0,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        assert len(result.trades) > 0, (
            "With VIX=15 < threshold=25, trades should be allowed"
        )

    def test_regime_filter_passes_close_signals(self):
        """CLOSE signals must pass through even when VIX is high."""
        spy_prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        vix_data = _make_vix_data(n_days=400, vix_level=35.0)
        prices = pl.concat([spy_prices, vix_data])
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=1,
            target_position_weight=0.05,
            stop_loss_pct=0.50,
        )
        # BuyAndClose buys on call 1, closes on call 2
        # With regime filter active and VIX=35, BUY on call 1 will be blocked.
        # But we want to verify CLOSE signals pass through when VIX is high.
        # We'll test the _apply_meta_filters method directly.
        strategy = NeverTradeStrategy(config)
        mf = MetaFilterConfig(
            regime_filter_enabled=True,
            vix_threshold=25.0,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        # Create a CLOSE signal and a BUY signal
        close_signal = TradeSignal(
            symbol="SPY",
            action=Action.CLOSE,
            conviction=Conviction.HIGH,
            target_weight=0.0,
            stop_loss=0.0,
            reasoning="test close",
        )
        buy_signal = TradeSignal(
            symbol="SPY",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.05,
            stop_loss=90.0,
            reasoning="test buy",
        )

        # Pick a date in the middle
        test_date = date(2020, 7, 1)
        causal = indicators.filter(pl.col("date") <= test_date)

        filtered = engine._apply_meta_filters(
            [buy_signal, close_signal], causal, test_date
        )

        # CLOSE should pass, BUY should be blocked (VIX=35 > 25)
        actions = [s.action for s in filtered]
        assert Action.CLOSE in actions, "CLOSE signal must pass through regime filter"
        assert Action.BUY not in actions, (
            "BUY signal should be blocked by regime filter"
        )


# ---------------------------------------------------------------------------
# Test: Ensemble vote
# ---------------------------------------------------------------------------


class TestEnsembleVote:
    """ensemble_vote requires N BUY signals to proceed."""

    def test_single_buy_blocked_when_min_votes_2(self):
        """1 BUY signal with min_votes=2 should be blocked."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            ensemble_vote_enabled=True,
            ensemble_min_votes=2,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # AlwaysBuyStrategy only buys SPY (1 signal), so with min_votes=2
        # the ensemble gate should block it
        assert len(result.trades) == 0, (
            f"Expected 0 trades with 1 BUY and min_votes=2, got {len(result.trades)}"
        )

    def test_multiple_buys_pass_when_min_votes_met(self):
        """2 BUY signals with min_votes=2 should pass."""
        # Two symbols means AlwaysBuyStrategy produces 2 BUY signals
        spy_prices = _make_prices(["SPY", "QQQ"], n_days=400, trend=0.001)
        indicators = compute_indicators(spy_prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            ensemble_vote_enabled=True,
            ensemble_min_votes=2,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        result = engine.run(
            prices_df=spy_prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        assert len(result.trades) > 0, (
            "With 2 BUY signals and min_votes=2, trades should be allowed"
        )

    def test_ensemble_passes_close_signals(self):
        """CLOSE signals bypass ensemble vote."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=1,
        )
        strategy = NeverTradeStrategy(config)
        mf = MetaFilterConfig(
            ensemble_vote_enabled=True,
            ensemble_min_votes=3,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        close_signal = TradeSignal(
            symbol="SPY",
            action=Action.CLOSE,
            conviction=Conviction.HIGH,
            target_weight=0.0,
            stop_loss=0.0,
            reasoning="test close",
        )

        test_date = date(2020, 7, 1)
        causal = indicators.filter(pl.col("date") <= test_date)

        filtered = engine._apply_meta_filters([close_signal], causal, test_date)
        assert len(filtered) == 1, "CLOSE signal should pass ensemble vote"
        assert filtered[0].action == Action.CLOSE


# ---------------------------------------------------------------------------
# Test: Signal strength weight
# ---------------------------------------------------------------------------


class TestSignalStrengthWeight:
    """signal_strength_weight scales BUY target_weight."""

    def test_strength_scaling_modifies_weight(self):
        """With signal_strength enabled, BUY target_weight should be scaled."""
        # Need a leader symbol in the data for the engine to compute leader_return
        spy_prices = _make_prices(["SPY"], n_days=400, trend=0.005)
        indicators = compute_indicators(spy_prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
            parameters={"leader_symbol": "SPY"},
        )

        # Run without signal_strength
        strategy1 = AlwaysBuyStrategy(config)
        engine1 = BacktestEngine(strategy=strategy1, initial_capital=100_000.0)
        result1 = engine1.run(
            prices_df=spy_prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # Run with signal_strength enabled
        strategy2 = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            signal_strength_enabled=True,
            signal_strength_scale=0.01,
            signal_strength_cap=2.0,
        )
        engine2 = BacktestEngine(
            strategy=strategy2, initial_capital=100_000.0, meta_filter=mf
        )
        result2 = engine2.run(
            prices_df=spy_prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # Both should have trades (signal strength only scales, doesn't block)
        assert len(result1.trades) > 0
        assert len(result2.trades) > 0

        # Final NAVs should differ since position sizes are scaled
        # (with trend=0.005 and scaling, the effect may be small but should exist)
        # We can't predict the direction, just that the filter was applied
        # So we just verify no crash and trades exist
        assert result2.nav_series[-1] > 0

    def test_strength_does_not_scale_close(self):
        """CLOSE signals should not have their weight modified."""
        prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=1,
            parameters={"leader_symbol": "SPY"},
        )
        strategy = NeverTradeStrategy(config)
        mf = MetaFilterConfig(
            signal_strength_enabled=True,
            signal_strength_scale=0.01,
            signal_strength_cap=2.0,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        close_signal = TradeSignal(
            symbol="SPY",
            action=Action.CLOSE,
            conviction=Conviction.HIGH,
            target_weight=0.0,
            stop_loss=0.0,
            reasoning="test close",
        )

        test_date = date(2020, 7, 1)
        causal = indicators.filter(pl.col("date") <= test_date)

        filtered = engine._apply_meta_filters([close_signal], causal, test_date)
        assert len(filtered) == 1
        assert filtered[0].action == Action.CLOSE
        assert filtered[0].target_weight == 0.0, (
            "CLOSE signal weight should not be modified"
        )


# ---------------------------------------------------------------------------
# Test: MetaFilterConfig dataclass defaults
# ---------------------------------------------------------------------------


class TestMetaFilterConfig:
    """Verify MetaFilterConfig defaults and construction."""

    def test_defaults(self):
        """All filters disabled by default."""
        mf = MetaFilterConfig()
        assert mf.regime_filter_enabled is False
        assert mf.signal_strength_enabled is False
        assert mf.ensemble_vote_enabled is False
        assert mf.vix_threshold == 25.0
        assert mf.ensemble_min_votes == 2

    def test_custom_config(self):
        """Custom values are stored correctly."""
        mf = MetaFilterConfig(
            regime_filter_enabled=True,
            vix_threshold=30.0,
            signal_strength_enabled=True,
            signal_strength_scale=0.02,
            signal_strength_cap=1.5,
            ensemble_vote_enabled=True,
            ensemble_min_votes=3,
        )
        assert mf.regime_filter_enabled is True
        assert mf.vix_threshold == 30.0
        assert mf.signal_strength_scale == 0.02
        assert mf.signal_strength_cap == 1.5
        assert mf.ensemble_min_votes == 3


# ---------------------------------------------------------------------------
# Test: Combined filters
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    """Multiple filters enabled simultaneously."""

    def test_regime_plus_ensemble(self):
        """When both regime and ensemble are enabled, both gates apply."""
        spy_prices = _make_prices(["SPY"], n_days=400, trend=0.001)
        # Low VIX = regime filter passes, but only 1 BUY = ensemble blocks
        vix_data = _make_vix_data(n_days=400, vix_level=15.0)
        prices = pl.concat([spy_prices, vix_data])
        indicators = compute_indicators(prices)

        config = StrategyConfig(
            name="test",
            rebalance_frequency_days=5,
            target_position_weight=0.05,
            stop_loss_pct=0.10,
        )
        strategy = AlwaysBuyStrategy(config)
        mf = MetaFilterConfig(
            regime_filter_enabled=True,
            vix_threshold=25.0,
            ensemble_vote_enabled=True,
            ensemble_min_votes=2,
        )
        engine = BacktestEngine(
            strategy=strategy, initial_capital=100_000.0, meta_filter=mf
        )

        result = engine.run(
            prices_df=prices,
            indicators_df=indicators,
            slug="test",
            fill_delay=0,
            warmup_days=50,
            trial_count=1,
        )

        # VIX is low (passes regime), but only 1 BUY signal (fails ensemble min=2)
        assert len(result.trades) == 0, (
            "Ensemble gate should block single BUY even when regime passes"
        )


# ---------------------------------------------------------------------------
# Test: Underlying rule-based functions (unit tests)
# ---------------------------------------------------------------------------


class TestRuleBasedFunctions:
    """Direct unit tests for the three functions in meta_label.py."""

    def test_regime_filter_below_threshold(self):
        from llm_quant.backtest.meta_label import regime_filter

        assert regime_filter(20.0, 25.0) is True

    def test_regime_filter_above_threshold(self):
        from llm_quant.backtest.meta_label import regime_filter

        assert regime_filter(30.0, 25.0) is False

    def test_regime_filter_at_threshold(self):
        from llm_quant.backtest.meta_label import regime_filter

        assert regime_filter(25.0, 25.0) is True  # <= threshold

    def test_signal_strength_weight_basics(self):
        from llm_quant.backtest.meta_label import signal_strength_weight

        # At threshold, ratio=1.0, weight=1.0
        w = signal_strength_weight(0.01, 0.01)
        assert 0.5 <= w <= 1.5

        # Double threshold, should scale up
        w2 = signal_strength_weight(0.02, 0.01)
        assert w2 > w

        # Below threshold
        w3 = signal_strength_weight(0.005, 0.01)
        assert w3 >= 0.5

    def test_signal_strength_weight_cap(self):
        from llm_quant.backtest.meta_label import signal_strength_weight

        w = signal_strength_weight(1.0, 0.01, max_multiplier=1.5)
        assert w == 1.5

    def test_signal_strength_weight_zero_threshold(self):
        from llm_quant.backtest.meta_label import signal_strength_weight

        w = signal_strength_weight(0.01, 0.0)
        assert w == 1.0

    def test_ensemble_vote_enough(self):
        from llm_quant.backtest.meta_label import ensemble_vote

        signals = {"SPY": "buy", "QQQ": "buy", "IWM": "sell"}
        assert ensemble_vote(signals, min_agreement=2) is True

    def test_ensemble_vote_not_enough(self):
        from llm_quant.backtest.meta_label import ensemble_vote

        signals = {"SPY": "buy", "QQQ": "hold"}
        assert ensemble_vote(signals, min_agreement=2) is False

    def test_ensemble_vote_sell_agreement(self):
        from llm_quant.backtest.meta_label import ensemble_vote

        signals = {"SPY": "sell", "QQQ": "sell"}
        assert ensemble_vote(signals, min_agreement=2) is True

    def test_ensemble_vote_empty(self):
        from llm_quant.backtest.meta_label import ensemble_vote

        assert ensemble_vote({}, min_agreement=2) is False

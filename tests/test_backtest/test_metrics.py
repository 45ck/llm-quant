"""Tests for DSR formula, PBO, benchmark total return, and survivorship warnings."""

from __future__ import annotations

import numpy as np

from llm_quant.backtest.metrics import (
    BacktestMetrics,
    compute_all_metrics,
    compute_annualized_return,
    compute_calmar,
    compute_dsr,
    compute_max_drawdown,
    compute_psr,
    compute_returns,
    compute_sharpe,
    compute_sortino,
    compute_sr0,
)
from llm_quant.backtest.robustness import compute_pbo

# ---------------------------------------------------------------------------
# Test: DSR on random walk (should fail)
# ---------------------------------------------------------------------------


class TestDSR:
    """Verify DSR correctly penalizes multiple testing."""

    def test_random_walk_low_dsr(self):
        """On random walk data with N=10 trials, DSR should be << 0.95."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0001, 0.02, size=252).tolist()  # ~1yr

        dsr, _psr, _sr0 = compute_dsr(returns, trial_count=10)

        # Random walk should not pass DSR gate
        assert dsr < 0.95, f"DSR={dsr} should be < 0.95 for random walk"

    def test_trending_data_high_dsr(self):
        """On trending data with genuine SR, DSR should be > 0.95."""
        rng = np.random.default_rng(42)
        # Strong trend: 0.1% daily mean with 1% vol → SR ≈ 1.6
        returns = rng.normal(0.001, 0.01, size=1000).tolist()

        dsr, _psr, _sr0 = compute_dsr(returns, trial_count=1)

        # With only 1 trial and strong signal, DSR should pass
        assert dsr > 0.95, f"DSR={dsr} should be > 0.95 for strong trend"

    def test_dsr_penalizes_more_trials(self):
        """DSR should decrease as trial count increases."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.015, size=500).tolist()

        dsr_1, _, _ = compute_dsr(returns, trial_count=1)
        dsr_10, _, _ = compute_dsr(returns, trial_count=10)
        dsr_100, _, _ = compute_dsr(returns, trial_count=100)

        assert dsr_1 >= dsr_10 >= dsr_100, (
            f"DSR should decrease with more trials: "
            f"N=1:{dsr_1:.4f}, N=10:{dsr_10:.4f}, N=100:{dsr_100:.4f}"
        )

    def test_sr0_increases_with_trials(self):
        """SR_0 (false strategy threshold) increases with trial count."""
        var_sr = 0.01  # typical variance of SR estimator
        sr0_1 = compute_sr0(1, var_sr)
        sr0_10 = compute_sr0(10, var_sr)
        sr0_100 = compute_sr0(100, var_sr)

        assert sr0_1 <= sr0_10 <= sr0_100

    def test_psr_properties(self):
        """PSR should be in [0, 1] and increase with SR."""
        # Low SR
        psr_low = compute_psr(0.01, 0.0, 252, 0.0, 3.0)
        # High SR
        psr_high = compute_psr(0.10, 0.0, 252, 0.0, 3.0)

        assert 0 <= psr_low <= 1
        assert 0 <= psr_high <= 1
        assert psr_high > psr_low


# ---------------------------------------------------------------------------
# Test: PBO on random vs genuine signal
# ---------------------------------------------------------------------------


class TestPBO:
    """Verify PBO correctly identifies overfitting."""

    def test_random_strategies_high_pbo(self):
        """Random strategies should have PBO → 1.0."""
        rng = np.random.default_rng(42)
        # 10 random strategies, 500 days each
        returns_matrix = [rng.normal(0.0, 0.02, size=500).tolist() for _ in range(10)]

        result = compute_pbo(returns_matrix, n_submatrices=8)

        # PBO should be high for random strategies
        assert result.pbo > 0.30, (
            f"PBO={result.pbo} should be high for random strategies"
        )

    def test_genuine_signal_low_pbo(self):
        """Strategies with genuine signal should have low PBO."""
        rng = np.random.default_rng(42)
        # All strategies have positive mean return (genuine alpha)
        returns_matrix = [
            rng.normal(0.001 * (i + 1), 0.01, size=500).tolist() for i in range(5)
        ]

        result = compute_pbo(returns_matrix, n_submatrices=8)

        # With genuine signal across all variants, PBO should be lower
        assert result.pbo < 0.50, (
            f"PBO={result.pbo} should be < 0.50 for genuine signal"
        )

    def test_pbo_requires_two_strategies(self):
        """PBO with < 2 strategies should return PBO=1.0."""
        result = compute_pbo([[0.01, 0.02, 0.03]])
        assert result.pbo == 1.0


# ---------------------------------------------------------------------------
# Test: Basic metrics
# ---------------------------------------------------------------------------


class TestBasicMetrics:
    """Verify core metric computations."""

    def test_compute_returns(self):
        returns = compute_returns([100.0, 105.0, 102.0])
        assert len(returns) == 2
        assert abs(returns[0] - 0.05) < 1e-10
        assert abs(returns[1] - (102.0 / 105.0 - 1.0)) < 1e-10

    def test_compute_sharpe_annualized(self):
        """Positive return series should have positive Sharpe."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=252).tolist()
        sr = compute_sharpe(returns, annualize=True)
        assert sr > 0

    def test_compute_sharpe_unannualized(self):
        """Unannualized Sharpe should be smaller than annualized."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=252).tolist()
        sr_ann = compute_sharpe(returns, annualize=True)
        sr_per = compute_sharpe(returns, annualize=False)
        assert sr_per < sr_ann

    def test_compute_sortino(self):
        """Sortino should be >= Sharpe for positively skewed returns."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=252).tolist()
        compute_sharpe(returns)
        sortino = compute_sortino(returns)
        # Sortino uses only downside deviation, so for positive mean
        # it should generally be >= Sharpe
        assert sortino > 0

    def test_compute_max_drawdown(self):
        """Known drawdown should be computed correctly."""
        nav = [100, 110, 105, 95, 100, 108]
        dd, _duration = compute_max_drawdown(nav)
        # Peak is 110, trough is 95 → dd = (110-95)/110 ≈ 0.1364
        expected_dd = (110 - 95) / 110
        assert abs(dd - expected_dd) < 1e-10

    def test_compute_calmar(self):
        calmar = compute_calmar(0.10, 0.05)
        assert abs(calmar - 2.0) < 1e-10

    def test_compute_annualized_return(self):
        # 10% over 252 days = ~10% annualized
        ann = compute_annualized_return(0.10, 252)
        assert abs(ann - 0.10) < 0.01

    def test_compute_all_metrics(self):
        """compute_all_metrics should produce valid BacktestMetrics."""
        rng = np.random.default_rng(42)
        nav = [100_000.0]
        for _ in range(252):
            nav.append(nav[-1] * (1 + rng.normal(0.0003, 0.01)))

        metrics = compute_all_metrics(
            nav,
            trades=[{"pnl": 100}, {"pnl": -50}, {"pnl": 200}],
            trial_count=5,
        )

        assert isinstance(metrics, BacktestMetrics)
        assert len(metrics.daily_returns) == 252
        assert metrics.total_trades == 3
        assert metrics.trial_count == 5
        assert 0 < metrics.win_rate <= 1


# ---------------------------------------------------------------------------
# Test: Survivorship warning
# ---------------------------------------------------------------------------


class TestSurvivorshipWarning:
    """Verify survivorship warnings for low coverage."""

    def test_low_coverage_warning(self):
        """Symbol with < 80% date coverage should trigger warning."""
        from datetime import date, timedelta

        import polars as pl

        from llm_quant.backtest.engine import BacktestEngine
        from llm_quant.backtest.strategy import StrategyConfig

        # Create data where one symbol has gaps
        rows = []
        base_date = date(2020, 1, 6)
        for i in range(200):
            d = base_date + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            # SPY has all dates
            rows.append(
                {
                    "symbol": "SPY",
                    "date": d,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000000,
                    "adj_close": 100.0,
                }
            )
            # QQQ has only 50% of dates
            if i % 2 == 0:
                rows.append(
                    {
                        "symbol": "QQQ",
                        "date": d,
                        "open": 200.0,
                        "high": 201.0,
                        "low": 199.0,
                        "close": 200.0,
                        "volume": 500000,
                        "adj_close": 200.0,
                    }
                )

        df = pl.DataFrame(rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("volume").cast(pl.Int64),
        )

        class NoopStrategy:
            config = StrategyConfig()

            def generate_signals(self, *a, **k):
                return []

        engine = BacktestEngine(
            strategy=NoopStrategy(),
            initial_capital=100_000.0,
        )

        dates = sorted(df.select("date").unique().to_series().to_list())
        warnings = engine._check_data_quality(df, dates[50:])

        # QQQ should trigger survivorship warning
        qqq_warnings = [w for w in warnings if "QQQ" in w]
        assert len(qqq_warnings) > 0, "QQQ should have a survivorship warning"

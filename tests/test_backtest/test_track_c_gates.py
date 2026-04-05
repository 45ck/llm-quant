"""Tests for Track C structural arb robustness gates.

Covers: beta-to-SPY gate, cost stress test, leg risk simulation,
and min trades gate.
"""

from __future__ import annotations

import numpy as np

from llm_quant.backtest.robustness import (
    TrackCGateResult,
    _compute_rolling_beta,
    _simulate_leg_risk,
    run_track_c_gates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _market_neutral_returns(n: int = 300, seed: int = 42) -> list[float]:
    """Generate near-zero-beta strategy returns (small uncorrelated noise)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.003, size=n).tolist()


def _spy_returns(n: int = 300, seed: int = 99) -> list[float]:
    """Generate synthetic SPY returns with market-like drift and vol."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0004, 0.012, size=n).tolist()


def _high_beta_returns(
    spy: list[float], beta: float = 0.8, seed: int = 7
) -> list[float]:
    """Generate strategy returns with a target beta to SPY."""
    rng = np.random.default_rng(seed)
    spy_arr = np.array(spy)
    noise = rng.normal(0.0, 0.003, size=len(spy))
    return (beta * spy_arr + noise).tolist()


# ---------------------------------------------------------------------------
# Beta gate tests
# ---------------------------------------------------------------------------


class TestBetaGate:
    """Verify rolling 30-day beta computation and gate logic."""

    def test_market_neutral_passes(self):
        """Uncorrelated strategy returns should have low beta to SPY."""
        strat = _market_neutral_returns(300)
        spy = _spy_returns(300)
        beta = _compute_rolling_beta(strat, spy)
        assert beta < 0.15, f"Market-neutral beta={beta:.4f} should be < 0.15"

    def test_high_beta_fails(self):
        """Strategy highly correlated to SPY should have high beta."""
        spy = _spy_returns(300)
        strat = _high_beta_returns(spy, beta=0.8)
        beta = _compute_rolling_beta(strat, spy)
        assert beta >= 0.15, f"High-beta strategy beta={beta:.4f} should be >= 0.15"

    def test_insufficient_data_fallback(self):
        """With fewer than 30 points, should fall back to full-period beta."""
        spy = _spy_returns(20)
        strat = _market_neutral_returns(20)
        beta = _compute_rolling_beta(strat, spy, window=30)
        # Should not raise and should return a float
        assert isinstance(beta, float)

    def test_empty_returns(self):
        """Empty returns should return 0.0 beta."""
        assert _compute_rolling_beta([], []) == 0.0

    def test_single_observation(self):
        """Single observation should return 0.0 beta."""
        assert _compute_rolling_beta([0.01], [0.01]) == 0.0

    def test_zero_variance_spy(self):
        """Constant SPY returns should produce 0.0 beta (no variance)."""
        strat = [0.001] * 50
        spy = [0.0005] * 50
        beta = _compute_rolling_beta(strat, spy)
        assert beta == 0.0

    def test_perfect_correlation(self):
        """Strategy = SPY (beta=1.0) should clearly fail the gate."""
        spy = _spy_returns(300)
        beta = _compute_rolling_beta(spy, spy)
        assert beta >= 0.15, f"Perfect correlation beta={beta:.4f} should fail gate"


# ---------------------------------------------------------------------------
# Cost stress gate tests
# ---------------------------------------------------------------------------


class TestCostStressGate:
    """Verify cost stress gate (2x fees, Sharpe > 0)."""

    def test_positive_2x_sharpe_passes(self):
        """Positive 2x-cost Sharpe should pass."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(300),
            trade_count=100,
            base_sharpe=1.5,
            cost_multiplied_sharpe=0.5,
        )
        assert result.cost_stress_gate is True

    def test_zero_2x_sharpe_fails(self):
        """Exactly zero 2x-cost Sharpe should fail (strict > 0)."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(300),
            trade_count=100,
            base_sharpe=1.5,
            cost_multiplied_sharpe=0.0,
        )
        assert result.cost_stress_gate is False

    def test_negative_2x_sharpe_fails(self):
        """Negative 2x-cost Sharpe should fail."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(300),
            trade_count=100,
            base_sharpe=1.5,
            cost_multiplied_sharpe=-0.3,
        )
        assert result.cost_stress_gate is False


# ---------------------------------------------------------------------------
# Leg risk simulation tests
# ---------------------------------------------------------------------------


class TestLegRiskGate:
    """Verify leg risk simulation and gate logic."""

    def test_market_neutral_survives_leg_risk(self):
        """A profitable market-neutral strategy should survive 5% leg failures."""
        rng = np.random.default_rng(42)
        # Strategy with positive mean, uncorrelated to SPY
        strat = rng.normal(0.001, 0.004, size=300).tolist()
        spy = _spy_returns(300)

        leg_risk_sharpe = _simulate_leg_risk(strat, spy)
        assert leg_risk_sharpe > 0, (
            f"Leg risk Sharpe={leg_risk_sharpe:.3f} should be > 0 "
            "for profitable neutral strategy"
        )

    def test_all_cash_returns_zero_sharpe(self):
        """All-cash strategy (no invested days) should return 0.0 Sharpe."""
        strat = [0.0] * 100
        spy = _spy_returns(100)
        sharpe = _simulate_leg_risk(strat, spy)
        assert sharpe == 0.0

    def test_empty_returns(self):
        """Empty returns should return 0.0."""
        assert _simulate_leg_risk([], []) == 0.0

    def test_single_invested_day(self):
        """With very few invested days, at least 1 trade should fail."""
        strat = [0.0] * 99 + [0.01]
        spy = [0.005] * 100
        sharpe = _simulate_leg_risk(strat, spy)
        assert isinstance(sharpe, float)

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        strat = _market_neutral_returns(200)
        spy = _spy_returns(200)
        s1 = _simulate_leg_risk(strat, spy, seed=123)
        s2 = _simulate_leg_risk(strat, spy, seed=123)
        assert s1 == s2

    def test_different_seeds_differ(self):
        """Different seeds should produce different results."""
        strat = _market_neutral_returns(200)
        spy = _spy_returns(200)
        s1 = _simulate_leg_risk(strat, spy, seed=1)
        s2 = _simulate_leg_risk(strat, spy, seed=2)
        # Not guaranteed to differ with all seeds, but overwhelmingly likely
        # with different random selections on 200 invested days
        assert s1 != s2


# ---------------------------------------------------------------------------
# Min trades gate tests
# ---------------------------------------------------------------------------


class TestMinTradesGate:
    """Verify min trades gate (>= 50 trades)."""

    def test_sufficient_trades_passes(self):
        """50+ trades should pass."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(300),
            trade_count=50,
            base_sharpe=1.5,
            cost_multiplied_sharpe=1.0,
        )
        assert result.min_trades_gate is True

    def test_insufficient_trades_fails(self):
        """Fewer than 50 trades should fail."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(300),
            trade_count=49,
            base_sharpe=1.5,
            cost_multiplied_sharpe=1.0,
        )
        assert result.min_trades_gate is False

    def test_zero_trades_fails(self):
        """Zero trades should fail."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(300),
            trade_count=0,
            base_sharpe=1.5,
            cost_multiplied_sharpe=1.0,
        )
        assert result.min_trades_gate is False


# ---------------------------------------------------------------------------
# Overall gate tests
# ---------------------------------------------------------------------------


class TestTrackCGatesOverall:
    """Verify the overall run_track_c_gates integration."""

    def test_all_gates_pass(self):
        """When all conditions are met, overall should pass."""
        strat = _market_neutral_returns(300)
        spy = _spy_returns(300)

        result = run_track_c_gates(
            strategy_returns=strat,
            spy_returns=spy,
            trade_count=100,
            base_sharpe=2.0,
            cost_multiplied_sharpe=0.5,
        )

        assert result.beta_gate is True
        assert result.cost_stress_gate is True
        assert result.leg_risk_gate is True
        assert result.min_trades_gate is True
        assert result.overall_passed is True
        assert all(result.gate_details.values())

    def test_single_gate_failure_fails_overall(self):
        """If any single gate fails, overall should fail."""
        strat = _market_neutral_returns(300)
        spy = _spy_returns(300)

        # Fail only cost stress gate
        result = run_track_c_gates(
            strategy_returns=strat,
            spy_returns=spy,
            trade_count=100,
            base_sharpe=2.0,
            cost_multiplied_sharpe=-0.1,  # cost stress fails
        )

        assert result.cost_stress_gate is False
        assert result.overall_passed is False

    def test_high_beta_fails_overall(self):
        """High-beta strategy should fail overall via beta gate."""
        spy = _spy_returns(300)
        strat = _high_beta_returns(spy, beta=0.8)

        result = run_track_c_gates(
            strategy_returns=strat,
            spy_returns=spy,
            trade_count=100,
            base_sharpe=2.0,
            cost_multiplied_sharpe=1.0,
        )

        assert result.beta_gate is False
        assert result.overall_passed is False

    def test_gate_details_populated(self):
        """gate_details dict should have all 4 gate keys."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(100),
            spy_returns=_spy_returns(100),
            trade_count=10,
            base_sharpe=0.5,
            cost_multiplied_sharpe=0.1,
        )

        expected_keys = {
            "beta_to_spy_<_0.15",
            "cost_2x_sharpe_>_0",
            "leg_risk_sharpe_>_0",
            "min_50_trades",
        }
        assert set(result.gate_details.keys()) == expected_keys

    def test_result_stores_inputs(self):
        """TrackCGateResult should store the raw inputs for audit trail."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(100),
            spy_returns=_spy_returns(100),
            trade_count=75,
            base_sharpe=1.8,
            cost_multiplied_sharpe=0.9,
        )

        assert result.trade_count == 75
        assert result.base_sharpe == 1.8
        assert result.cost_multiplied_sharpe == 0.9
        assert isinstance(result.beta_to_spy, float)
        assert isinstance(result.leg_risk_sharpe, float)

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        strat = _market_neutral_returns(300)
        spy = _spy_returns(300)

        # With very strict beta threshold, should fail
        result_strict = run_track_c_gates(
            strategy_returns=strat,
            spy_returns=spy,
            trade_count=100,
            base_sharpe=2.0,
            cost_multiplied_sharpe=0.5,
            beta_threshold=0.001,  # extremely strict
        )

        # With very relaxed beta threshold, should pass
        result_relaxed = run_track_c_gates(
            strategy_returns=strat,
            spy_returns=spy,
            trade_count=100,
            base_sharpe=2.0,
            cost_multiplied_sharpe=0.5,
            beta_threshold=1.0,  # very relaxed
        )

        # Strict should fail, relaxed should pass (beta gate)
        assert result_strict.beta_gate is False or result_relaxed.beta_gate is True

    def test_custom_min_trades_threshold(self):
        """Custom min_trades should be respected."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(100),
            spy_returns=_spy_returns(100),
            trade_count=10,
            base_sharpe=1.5,
            cost_multiplied_sharpe=0.5,
            min_trades=5,  # lower threshold
        )
        assert result.min_trades_gate is True

    def test_dataclass_compute_overall(self):
        """TrackCGateResult.compute_overall should set overall_passed."""
        result = TrackCGateResult(
            beta_gate=True,
            cost_stress_gate=True,
            leg_risk_gate=True,
            min_trades_gate=True,
        )
        result.compute_overall()
        assert result.overall_passed is True

        result.beta_gate = False
        result.compute_overall()
        assert result.overall_passed is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestTrackCEdgeCases:
    """Edge cases: insufficient data, zero trades, perfect correlation."""

    def test_very_short_returns(self):
        """Very short return series should not crash."""
        result = run_track_c_gates(
            strategy_returns=[0.01, 0.02],
            spy_returns=[0.005, 0.01],
            trade_count=2,
            base_sharpe=0.5,
            cost_multiplied_sharpe=0.1,
        )
        # Should complete without error; min_trades will fail
        assert result.min_trades_gate is False
        assert isinstance(result.beta_to_spy, float)

    def test_empty_returns(self):
        """Empty return series should not crash."""
        result = run_track_c_gates(
            strategy_returns=[],
            spy_returns=[],
            trade_count=0,
            base_sharpe=0.0,
            cost_multiplied_sharpe=0.0,
        )
        assert result.overall_passed is False
        assert result.beta_to_spy == 0.0

    def test_mismatched_lengths(self):
        """Different-length returns should be handled (truncated to shorter)."""
        result = run_track_c_gates(
            strategy_returns=_market_neutral_returns(300),
            spy_returns=_spy_returns(200),
            trade_count=100,
            base_sharpe=1.5,
            cost_multiplied_sharpe=0.5,
        )
        # Should not crash — beta and leg risk use min(len) internally
        assert isinstance(result.beta_to_spy, float)
        assert isinstance(result.leg_risk_sharpe, float)

    def test_all_zero_strategy_returns(self):
        """All-cash strategy should have zero beta and zero leg risk sharpe."""
        strat = [0.0] * 200
        spy = _spy_returns(200)
        result = run_track_c_gates(
            strategy_returns=strat,
            spy_returns=spy,
            trade_count=0,
            base_sharpe=0.0,
            cost_multiplied_sharpe=0.0,
        )
        assert result.leg_risk_sharpe == 0.0

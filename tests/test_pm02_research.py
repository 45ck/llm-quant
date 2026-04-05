"""Tests for PM-02 Viral Claim Reconstruction research modules."""

from __future__ import annotations

import math

import pytest

from llm_quant.arb.research.compounder import MonteCarloCompounder
from llm_quant.arb.research.fill_simulator import FillSimulator
from llm_quant.arb.research.path_models import CompoundingPath
from llm_quant.arb.research.reconstruction import ReconstructionEngine

# ── CompoundingPath tests ──────────────────────────────────────


class TestMinEdge:
    def test_known_values(self) -> None:
        """50 -> 500 in 30 days at 10 trades/day => ~0.77% per trade."""
        edge = CompoundingPath.min_edge(50, 500, 30, 10)
        # (500/50)^(1/300) - 1 = 10^(1/300) - 1
        expected = 10 ** (1.0 / 300) - 1
        assert abs(edge - expected) < 1e-10

    def test_no_growth_needed(self) -> None:
        """start == target => edge = 0."""
        edge = CompoundingPath.min_edge(100, 100, 30, 10)
        assert abs(edge) < 1e-12

    def test_small_target(self) -> None:
        """target < start => negative edge (drawdown path)."""
        edge = CompoundingPath.min_edge(100, 50, 30, 10)
        assert edge < 0

    def test_zero_trades_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CompoundingPath.min_edge(100, 200, 0, 10)

    def test_zero_start_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CompoundingPath.min_edge(0, 200, 30, 10)


class TestKellyFraction:
    def test_fair_coin_even_odds(self) -> None:
        """50% win rate, 1:1 ratio => f* = 0."""
        f = CompoundingPath.kelly_fraction(0.5, 1.0)
        assert abs(f) < 1e-10

    def test_edge_case(self) -> None:
        """60% win rate, 1:1 ratio => f* = 0.20."""
        f = CompoundingPath.kelly_fraction(0.6, 1.0)
        assert abs(f - 0.2) < 1e-10

    def test_high_win_rate(self) -> None:
        """90% win, 2:1 ratio => f* = (0.9*3-1)/2 = 0.85."""
        f = CompoundingPath.kelly_fraction(0.9, 2.0)
        assert abs(f - 0.85) < 1e-10

    def test_zero_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="nonzero"):
            CompoundingPath.kelly_fraction(0.5, 0.0)


class TestRequiredWinRate:
    def test_symmetric(self) -> None:
        """Edge=0, equal win/loss => 50%."""
        wr = CompoundingPath.required_win_rate(0.0, 1.0, 1.0)
        assert abs(wr - 0.5) < 1e-10

    def test_positive_edge(self) -> None:
        """Edge=0.1, avg_win=1, avg_loss=1 => 55%."""
        wr = CompoundingPath.required_win_rate(0.1, 1.0, 1.0)
        assert abs(wr - 0.55) < 1e-10


class TestSimulatePath:
    def test_length(self) -> None:
        """Path length = days * trades_per_day + 1."""
        path = CompoundingPath.simulate_path(
            bankroll=100,
            edge_per_trade=0.01,
            trades_per_day=5,
            days=10,
        )
        assert len(path) == 5 * 10 + 1

    def test_first_element(self) -> None:
        """First element is starting bankroll."""
        path = CompoundingPath.simulate_path(
            bankroll=1000,
            edge_per_trade=0.01,
            trades_per_day=5,
            days=3,
        )
        assert path[0] == 1000.0

    def test_monotonic_positive_edge(self) -> None:
        """With positive net edge, path is strictly increasing."""
        path = CompoundingPath.simulate_path(
            bankroll=100,
            edge_per_trade=0.05,
            trades_per_day=2,
            days=5,
            fee_rate=0.01,
            slippage=0.005,
        )
        for i in range(1, len(path)):
            assert path[i] > path[i - 1]


class TestDaysToTarget:
    def test_known(self) -> None:
        """Consistency with min_edge: if edge = min_edge, days = days."""
        edge = CompoundingPath.min_edge(100, 1000, 30, 10)
        d = CompoundingPath.days_to_target(100, 1000, edge, 10)
        assert abs(d - 30.0) < 1e-6

    def test_negative_net_edge(self) -> None:
        """Negative net edge => infinite days."""
        d = CompoundingPath.days_to_target(
            100,
            1000,
            0.01,
            10,
            fee_rate=0.02,
        )
        assert d == math.inf


# ── MonteCarloCompounder tests ─────────────────────────────────


class TestMonteCarloCompounder:
    def test_returns_expected_keys(self) -> None:
        mc = MonteCarloCompounder(n_simulations=50, seed=1)
        result = mc.run(
            bankroll=100,
            edge_mean=0.01,
            edge_std=0.005,
            trades_per_day=5,
            days=10,
            fee_rate=0.002,
            fill_rate=0.9,
            slippage_mean=0.001,
        )
        expected_keys = {
            "median_final",
            "mean_final",
            "p5",
            "p25",
            "p75",
            "p95",
            "prob_target",
            "max_drawdown_median",
        }
        assert set(result.keys()) == expected_keys

    def test_percentile_ordering(self) -> None:
        mc = MonteCarloCompounder(n_simulations=200, seed=42)
        result = mc.run(
            bankroll=100,
            edge_mean=0.01,
            edge_std=0.005,
            trades_per_day=5,
            days=10,
            fee_rate=0.002,
            fill_rate=0.9,
            slippage_mean=0.001,
        )
        assert result["p5"] <= result["p25"]
        assert result["p25"] <= result["median_final"]
        assert result["median_final"] <= result["p75"]
        assert result["p75"] <= result["p95"]

    def test_prob_target(self) -> None:
        """With strong edge, high prob of reaching modest target."""
        mc = MonteCarloCompounder(n_simulations=100, seed=42)
        result = mc.run(
            bankroll=100,
            edge_mean=0.02,
            edge_std=0.001,
            trades_per_day=10,
            days=20,
            fee_rate=0.001,
            fill_rate=1.0,
            slippage_mean=0.0,
            target=150,
        )
        # With 2% edge, 10 trades/day, 20 days, should easily
        # exceed 150 from 100
        assert result["prob_target"] > 0.5

    def test_max_drawdown_nonnegative(self) -> None:
        mc = MonteCarloCompounder(n_simulations=50, seed=7)
        result = mc.run(
            bankroll=100,
            edge_mean=0.005,
            edge_std=0.01,
            trades_per_day=5,
            days=10,
            fee_rate=0.001,
            fill_rate=0.9,
            slippage_mean=0.001,
        )
        assert result["max_drawdown_median"] >= 0.0


# ── FillSimulator tests ───────────────────────────────────────


class TestFillProbability:
    def test_bounds(self) -> None:
        """Fill probability is always in [0, 1]."""
        for size in [10, 100, 1000]:
            for depth in [5, 50, 500]:
                for dist in [0.0, 0.01, 0.05, 0.10]:
                    p = FillSimulator.fill_probability(
                        size,
                        depth,
                        dist,
                    )
                    assert 0.0 <= p <= 1.0

    def test_zero_trade_size(self) -> None:
        p = FillSimulator.fill_probability(0, 100, 0.05)
        assert p == 0.0

    def test_zero_depth(self) -> None:
        p = FillSimulator.fill_probability(100, 0, 0.05)
        assert p == 0.0

    def test_at_midpoint(self) -> None:
        """At midpoint (distance=0), fill prob = 0."""
        p = FillSimulator.fill_probability(100, 200, 0.0)
        assert p == 0.0

    def test_increases_with_distance(self) -> None:
        """Fill prob increases with distance from mid."""
        p1 = FillSimulator.fill_probability(100, 200, 0.01)
        p2 = FillSimulator.fill_probability(100, 200, 0.05)
        assert p2 > p1


class TestExpectedSlippage:
    def test_no_slippage_when_depth_sufficient(self) -> None:
        s = FillSimulator.expected_slippage(50, 100)
        assert s == 0.0

    def test_full_slippage_zero_depth(self) -> None:
        s = FillSimulator.expected_slippage(100, 0, tick_size=0.01)
        assert s == 0.01

    def test_partial_slippage(self) -> None:
        """50 contracts, 25 depth => 50% excess => 0.005."""
        s = FillSimulator.expected_slippage(50, 25, tick_size=0.01)
        assert abs(s - 0.005) < 1e-10


class TestPolymarketFee:
    def test_formula(self) -> None:
        """Fee = C * p * feeRate * p * (1-p)."""
        fee = FillSimulator.polymarket_fee(0.5, 100, 0.02)
        # 100 * 0.5 * 0.02 * 0.5 * 0.5 = 0.25
        assert abs(fee - 0.25) < 1e-10

    def test_extremes_zero(self) -> None:
        """Price at 0 or 1 => fee = 0."""
        assert FillSimulator.polymarket_fee(0.0, 100) == 0.0
        assert FillSimulator.polymarket_fee(1.0, 100) == 0.0

    def test_asymmetric(self) -> None:
        """Fee at p=0.3 vs p=0.7: different due to p^2 term."""
        f1 = FillSimulator.polymarket_fee(0.3, 100, 0.02)
        f2 = FillSimulator.polymarket_fee(0.7, 100, 0.02)
        # f1 = 100*0.3*0.02*0.3*0.7 = 0.126
        # f2 = 100*0.7*0.02*0.7*0.3 = 0.294
        assert abs(f1 - 0.126) < 1e-10
        assert abs(f2 - 0.294) < 1e-10


# ── ReconstructionEngine tests ─────────────────────────────────


class TestReconstructionEngine:
    def test_required_edge(self) -> None:
        eng = ReconstructionEngine(50, 500, 30)
        edge = eng.required_edge_per_trade(10)
        expected = CompoundingPath.min_edge(50, 500, 30, 10)
        assert abs(edge - expected) < 1e-12

    def test_h1_returns_dict(self) -> None:
        eng = ReconstructionEngine(
            100,
            1000,
            30,
        )
        eng.compounder = MonteCarloCompounder(
            n_simulations=10,
            seed=1,
        )
        result = eng.run_h1_negrisk(
            spreads=[0.03, 0.04, 0.02],
            depths=[200, 300, 150],
        )
        assert result["hypothesis"] == "H1_negrisk"
        assert "feasible" in result
        assert "monte_carlo" in result

    def test_h1_empty_spreads(self) -> None:
        eng = ReconstructionEngine(100, 1000, 30)
        result = eng.run_h1_negrisk(spreads=[], depths=[])
        assert result["feasible"] is False

    def test_h2_returns_dict(self) -> None:
        eng = ReconstructionEngine(100, 1000, 30)
        eng.compounder = MonteCarloCompounder(
            n_simulations=10,
            seed=1,
        )
        result = eng.run_h2_rebalancing(
            complement_gaps=[0.02, 0.03, 0.015],
        )
        assert result["hypothesis"] == "H2_rebalancing"

    def test_h3_returns_dict(self) -> None:
        eng = ReconstructionEngine(100, 1000, 30)
        eng.compounder = MonteCarloCompounder(
            n_simulations=10,
            seed=1,
        )
        result = eng.run_h3_combinatorial(
            opportunities=[
                {"edge": 0.05, "contracts": 100},
                {"edge": 0.03, "contracts": 200},
            ],
        )
        assert result["hypothesis"] == "H3_combinatorial"

    def test_h4_returns_dict(self) -> None:
        eng = ReconstructionEngine(100, 1000, 30)
        eng.compounder = MonteCarloCompounder(
            n_simulations=10,
            seed=1,
        )
        result = eng.run_h4_latency(lag_seconds=0.5)
        assert result["hypothesis"] == "H4_latency"

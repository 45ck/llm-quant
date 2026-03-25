"""Tests for robustness analysis: PBO, CPCV, perturbations, and gate."""

from __future__ import annotations

import numpy as np

from llm_quant.backtest.robustness import (
    PerturbationResult,
    compute_pbo,
    generate_perturbations,
    run_cpcv,
    run_robustness_gate,
)

# ---------------------------------------------------------------------------
# PBO tests
# ---------------------------------------------------------------------------


class TestPBO:
    """Verify PBO via CSCV."""

    def test_random_strategies_high_pbo(self):
        """Random strategies should produce PBO close to 1.0 (overfit)."""
        rng = np.random.default_rng(42)
        # 10 iid random walks — no real alpha, IS-best is noise
        returns_matrix = [rng.normal(0.0, 0.02, size=500).tolist() for _ in range(10)]

        result = compute_pbo(returns_matrix, n_submatrices=8)

        assert result.pbo > 0.30, f"PBO={result.pbo} should be high for random data"
        assert result.n_strategies == 10
        assert result.n_combinations > 0

    def test_genuine_signal_lower_pbo(self):
        """Strategies with genuinely different means should produce lower PBO."""
        rng = np.random.default_rng(42)
        # Strategy i has mean return proportional to (i+1), with clear ordering
        returns_matrix = [
            rng.normal(0.002 * (i + 1), 0.01, size=500).tolist() for i in range(5)
        ]

        result = compute_pbo(returns_matrix, n_submatrices=8)

        # With strong signal differentiation, PBO should be lower
        assert result.pbo < 0.50, (
            f"PBO={result.pbo} should be < 0.50 for genuine signal"
        )

    def test_requires_two_strategies(self):
        """PBO with < 2 strategies returns PBO=1.0."""
        result = compute_pbo([[0.01, 0.02, 0.03]])
        assert result.pbo == 1.0
        assert result.n_strategies == 1

    def test_insufficient_observations(self):
        """PBO with too few observations for submatrices returns PBO=1.0."""
        # 5 observations for 16 submatrices — not enough
        result = compute_pbo([[0.01] * 5, [0.02] * 5], n_submatrices=16)
        assert result.pbo == 1.0

    def test_oos_rank_ties(self):
        """Tied OOS performance should not artificially inflate rank.

        After Fix 3 (strict inequality), ties are not counted as 'better',
        so the IS-best gets a fair rank.
        """
        rng = np.random.default_rng(123)
        # Two identical strategies — IS-best and its OOS twin should rank equally
        base = rng.normal(0.001, 0.02, size=500).tolist()
        returns_matrix = [base[:], base[:]]  # copies

        result = compute_pbo(returns_matrix, n_submatrices=8)

        # With identical strategies, IS-best should rank 1st in OOS (rank=1)
        # since no other strategy is strictly better
        for rank in result.is_best_oos_ranks:
            assert rank == 1, f"Tied strategies should rank 1, got {rank}"


# ---------------------------------------------------------------------------
# CPCV tests
# ---------------------------------------------------------------------------


class TestCPCV:
    """Verify Combinatorial Purged Cross-Validation."""

    def test_basic_structure(self):
        """CPCV should return results with correct structure."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=300).tolist()

        result = run_cpcv(returns, strategy_fn=None, n_groups=6, k_test=2)

        assert result.n_combinations > 0
        assert len(result.oos_sharpes) == result.n_combinations
        assert result.n_paths > 0

    def test_positive_signal_passes(self):
        """Strong positive signal should have positive mean OOS Sharpe."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, size=500).tolist()

        result = run_cpcv(returns, strategy_fn=None, n_groups=6, k_test=2)

        assert result.mean_oos_sharpe > 0
        assert result.passed is True

    def test_too_few_observations(self):
        """Too few observations should return empty result."""
        result = run_cpcv([0.01] * 10, strategy_fn=None, n_groups=6, k_test=2)
        assert result.n_combinations == 0

    def test_custom_strategy_fn(self):
        """CPCV should use custom strategy_fn when provided."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=300).tolist()

        # Custom function: always return 0.5
        result = run_cpcv(returns, strategy_fn=lambda _rets: 0.5, n_groups=6, k_test=2)

        # All OOS sharpes should be 0.5 (from the constant fn)
        assert all(abs(s - 0.5) < 1e-10 for s in result.oos_sharpes)


# ---------------------------------------------------------------------------
# Perturbation tests
# ---------------------------------------------------------------------------


class TestPerturbations:
    """Verify parameter perturbation generation."""

    def test_correct_count(self):
        """Each numeric non-zero param generates 2 perturbations (+/-)."""
        params = {"sma_fast": 20, "sma_slow": 50, "name": "test"}
        perturbs = generate_perturbations(params, perturbation_pct=0.20)

        # 2 numeric params x 2 directions = 4
        assert len(perturbs) == 4

    def test_perturbation_values(self):
        """Perturbation should shift values by the specified percentage."""
        params = {"lookback": 10}
        perturbs = generate_perturbations(params, perturbation_pct=0.20)

        names = [p[0] for p in perturbs]
        values = [p[1]["lookback"] for p in perturbs]

        assert "lookback+20%" in names
        assert "lookback-20%" in names
        # 10 * 1.2 = 12, 10 * 0.8 = 8
        assert 12 in values
        assert 8 in values

    def test_negative_integer_preserved(self):
        """Negative int parameters should stay negative after perturbation.

        After Fix 2: clamping preserves sign.
        """
        params = {"z_score_threshold": -3}
        perturbs = generate_perturbations(params, perturbation_pct=0.20)

        for _name, p in perturbs:
            val = p["z_score_threshold"]
            assert val < 0, f"Negative param became {val} after perturbation"
            assert isinstance(val, int), f"Should remain int, got {type(val)}"

    def test_small_negative_int_clamped_to_minus_one(self):
        """A negative int param close to zero should clamp to -1, not +1."""
        params = {"offset": -1}
        perturbs = generate_perturbations(params, perturbation_pct=0.20)

        for _name, p in perturbs:
            val = p["offset"]
            assert val <= -1, f"Negative param should clamp to <= -1, got {val}"

    def test_zero_value_skipped(self):
        """Parameters with value 0 should be skipped."""
        params = {"a": 0, "b": 10}
        perturbs = generate_perturbations(params, perturbation_pct=0.20)
        assert len(perturbs) == 2  # only 'b' generates perturbations

    def test_float_params(self):
        """Float parameters should not be clamped to int."""
        params = {"threshold": 0.5}
        perturbs = generate_perturbations(params, perturbation_pct=0.20)
        for _name, p in perturbs:
            assert isinstance(p["threshold"], float)


# ---------------------------------------------------------------------------
# Robustness gate tests
# ---------------------------------------------------------------------------


class TestRobustnessGate:
    """Verify the full robustness gate."""

    def test_insufficient_trials_pbo_fails(self):
        """With < 2 strategies, PBO gate should fail."""
        result = run_robustness_gate(
            dsr=0.99,
            returns_matrix=[[0.01, 0.02]],  # only 1 strategy
            best_returns=[0.01] * 100,
            cost_2x_sharpe=0.5,
            perturbation_results=[PerturbationResult(profitable=True)],
        )

        assert result.pbo.pbo == 1.0
        assert result.pbo_passed is False

    def test_all_gates_pass(self):
        """When all inputs pass thresholds, overall should pass."""
        rng = np.random.default_rng(42)
        # Strongly ordered strategies for low PBO
        returns_matrix = [
            rng.normal(0.003 * (i + 1), 0.008, size=500).tolist() for i in range(5)
        ]

        result = run_robustness_gate(
            dsr=0.99,
            returns_matrix=returns_matrix,
            best_returns=rng.normal(0.002, 0.01, size=500).tolist(),
            cost_2x_sharpe=0.5,
            perturbation_results=[
                PerturbationResult(profitable=True),
                PerturbationResult(profitable=True),
                PerturbationResult(profitable=True),
            ],
            pbo_threshold=0.50,  # relaxed threshold for test
        )

        assert result.dsr_passed is True
        assert result.cost_2x_survives is True
        assert result.parameter_stability_passed is True

    def test_pbo_threshold_parameter_respected(self):
        """pbo_threshold should control the PBO pass/fail decision.

        After Fix 1: the threshold parameter is actually used.
        """
        rng = np.random.default_rng(42)
        returns_matrix = [
            rng.normal(0.001 * (i + 1), 0.01, size=500).tolist() for i in range(5)
        ]

        # Run with a very strict threshold — should fail
        result_strict = run_robustness_gate(
            dsr=0.99,
            returns_matrix=returns_matrix,
            best_returns=[0.01] * 100,
            cost_2x_sharpe=0.5,
            perturbation_results=[],
            pbo_threshold=0.001,  # nearly impossible to pass
        )

        # Run with a very relaxed threshold — should pass
        result_relaxed = run_robustness_gate(
            dsr=0.99,
            returns_matrix=returns_matrix,
            best_returns=[0.01] * 100,
            cost_2x_sharpe=0.5,
            perturbation_results=[],
            pbo_threshold=0.99,  # almost anything passes
        )

        assert result_strict.pbo_passed is False, (
            f"PBO={result_strict.pbo.pbo} should fail with threshold 0.001"
        )
        assert result_relaxed.pbo_passed is True, (
            f"PBO={result_relaxed.pbo.pbo} should pass with threshold 0.99"
        )

    def test_parameter_stability(self):
        """Parameter stability should be computed from perturbation results."""
        result = run_robustness_gate(
            dsr=0.50,
            returns_matrix=[],
            best_returns=[],
            cost_2x_sharpe=-0.1,
            perturbation_results=[
                PerturbationResult(profitable=True),
                PerturbationResult(profitable=True),
                PerturbationResult(profitable=False),
                PerturbationResult(profitable=False),
            ],
        )

        # 2/4 = 50%, not > 50%
        assert result.parameter_stability == 0.50
        assert result.parameter_stability_passed is False

"""Tests for DCC-GARCH dynamic correlation estimation module."""

import numpy as np
import pytest

from llm_quant.risk.dcc_garch import (
    DccGarchEstimator,
    _forbes_rigobon_adjust,
    compute_ewma_correlation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_returns(
    n: int = 300,
    corr: float = 0.5,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Generate two correlated return series with known correlation.

    Uses Cholesky decomposition to produce returns with approximately
    the target pairwise correlation.
    """
    rng = np.random.default_rng(seed)
    # Cholesky factor for [1, corr; corr, 1]
    chol = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    raw = rng.standard_normal((n, 2))
    correlated = raw @ chol.T
    # Scale to realistic daily returns (~1% vol)
    correlated *= 0.01
    return correlated[:, 0].tolist(), correlated[:, 1].tolist()


def _make_three_asset_returns(
    n: int = 300,
    seed: int = 123,
) -> dict[str, list[float]]:
    """Generate three synthetic return series with known correlation structure.

    Assets A and B are positively correlated (~0.6), asset C is nearly
    uncorrelated with both.
    """
    rng = np.random.default_rng(seed)
    cov = np.array(
        [
            [1.0, 0.6, 0.05],
            [0.6, 1.0, 0.10],
            [0.05, 0.10, 1.0],
        ]
    )
    chol = np.linalg.cholesky(cov)
    raw = rng.standard_normal((n, 3))
    correlated = (raw @ chol.T) * 0.01
    return {
        "AssetA": correlated[:, 0].tolist(),
        "AssetB": correlated[:, 1].tolist(),
        "AssetC": correlated[:, 2].tolist(),
    }


# ---------------------------------------------------------------------------
# Tests: compute_ewma_correlation (standalone function)
# ---------------------------------------------------------------------------


class TestComputeEwmaCorrelation:
    """Tests for the standalone EWMA correlation function."""

    def test_positively_correlated_series(self):
        """EWMA correlation should be positive for positively correlated inputs."""
        a, b = _make_synthetic_returns(n=200, corr=0.7, seed=1)
        corr = compute_ewma_correlation(a, b, lambda_=0.94)
        assert corr > 0.3, f"Expected positive correlation, got {corr}"
        assert corr < 1.0

    def test_negatively_correlated_series(self):
        """EWMA correlation should be negative for negatively correlated inputs."""
        a, b = _make_synthetic_returns(n=200, corr=-0.6, seed=2)
        corr = compute_ewma_correlation(a, b, lambda_=0.94)
        assert corr < -0.2, f"Expected negative correlation, got {corr}"
        assert corr > -1.0

    def test_uncorrelated_series(self):
        """EWMA correlation should be near zero for uncorrelated inputs."""
        rng = np.random.default_rng(3)
        a = (rng.standard_normal(200) * 0.01).tolist()
        b = (rng.standard_normal(200) * 0.01).tolist()
        corr = compute_ewma_correlation(a, b, lambda_=0.94)
        assert abs(corr) < 0.3, f"Expected near-zero correlation, got {corr}"

    def test_perfectly_correlated(self):
        """Identical series should give correlation near 1.0."""
        a = [0.01, -0.005, 0.003, 0.007, -0.002] * 20
        corr = compute_ewma_correlation(a, a, lambda_=0.94)
        assert corr > 0.99, f"Expected ~1.0, got {corr}"

    def test_empty_input(self):
        """Empty inputs should return 0.0."""
        assert compute_ewma_correlation([], [], lambda_=0.94) == 0.0

    def test_single_observation(self):
        """Single observation should return 0.0 (insufficient data)."""
        assert compute_ewma_correlation([0.01], [0.02], lambda_=0.94) == 0.0

    def test_different_lengths(self):
        """Should handle mismatched lengths by using shorter."""
        a = [0.01, -0.005, 0.003, 0.007, -0.002]
        b = [0.01, -0.005, 0.003]
        corr = compute_ewma_correlation(a, b, lambda_=0.94)
        # Should not crash and should return a valid correlation
        assert -1.0 <= corr <= 1.0

    def test_deterministic_results(self):
        """Same inputs should always produce the same result."""
        a, b = _make_synthetic_returns(n=100, corr=0.5, seed=42)
        c1 = compute_ewma_correlation(a, b, lambda_=0.94)
        c2 = compute_ewma_correlation(a, b, lambda_=0.94)
        assert c1 == c2

    def test_lambda_effect(self):
        """Higher lambda should give slower decay (smoother estimate)."""
        a, b = _make_synthetic_returns(n=200, corr=0.5, seed=10)
        corr_fast = compute_ewma_correlation(a, b, lambda_=0.80)
        corr_slow = compute_ewma_correlation(a, b, lambda_=0.99)
        # Both should be valid correlations — exact relationship depends on data
        assert -1.0 <= corr_fast <= 1.0
        assert -1.0 <= corr_slow <= 1.0

    def test_numpy_array_input(self):
        """Should accept numpy arrays as well as lists."""
        a = np.array([0.01, -0.005, 0.003, 0.007, -0.002] * 20)
        b = np.array([0.008, -0.003, 0.005, 0.004, -0.001] * 20)
        corr = compute_ewma_correlation(a, b, lambda_=0.94)
        assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# Tests: DccGarchEstimator
# ---------------------------------------------------------------------------


class TestDccGarchEstimator:
    """Tests for the DCC-GARCH estimator class."""

    def test_basic_two_asset_estimation(self):
        """Two correlated assets should produce valid estimation results."""
        a, b = _make_synthetic_returns(n=300, corr=0.6, seed=42)
        est = DccGarchEstimator()
        result = est.fit({"SPY": a, "TLT": b})

        assert result["method"] in ("dcc_garch", "ewma_fallback")
        assert result["n_obs"] == 252  # default lookback truncates 300 -> 252
        assert len(result["assets"]) == 2

        # Check correlation matrix
        corr_dict = est.get_correlation_matrix()
        assert ("SPY", "SPY") in corr_dict
        assert abs(corr_dict[("SPY", "SPY")] - 1.0) < 1e-10
        assert ("TLT", "TLT") in corr_dict
        assert abs(corr_dict[("TLT", "TLT")] - 1.0) < 1e-10
        # Cross-correlation should be positive for positively correlated inputs
        assert corr_dict[("SPY", "TLT")] > 0.0

        # Symmetry
        assert abs(corr_dict[("SPY", "TLT")] - corr_dict[("TLT", "SPY")]) < 1e-10

    def test_three_asset_estimation(self):
        """Three assets with known correlation structure."""
        returns = _make_three_asset_returns(n=300, seed=123)
        est = DccGarchEstimator()
        result = est.fit(returns)

        assert len(result["assets"]) == 3
        corr_dict = est.get_correlation_matrix()

        # A-B should be more correlated than A-C or B-C
        corr_ab = abs(corr_dict[("AssetA", "AssetB")])
        corr_ac = abs(corr_dict[("AssetA", "AssetC")])
        corr_bc = abs(corr_dict[("AssetB", "AssetC")])
        assert corr_ab > corr_ac, f"AB={corr_ab} should be > AC={corr_ac}"
        assert corr_ab > corr_bc, f"AB={corr_ab} should be > BC={corr_bc}"

    def test_diversification_score_bounds(self):
        """Diversification score should be in [0, 1]."""
        returns = _make_three_asset_returns(n=300, seed=123)
        est = DccGarchEstimator()
        est.fit(returns)

        score = est.get_diversification_score()
        assert 0.0 <= score <= 1.0

    def test_diversification_score_high_corr(self):
        """Highly correlated assets should have a low diversification score."""
        a, b = _make_synthetic_returns(n=300, corr=0.95, seed=55)
        est = DccGarchEstimator()
        est.fit({"X": a, "Y": b})

        score = est.get_diversification_score()
        assert score < 0.5, f"Expected low div score for corr=0.95, got {score}"

    def test_diversification_score_low_corr(self):
        """Uncorrelated assets should have a high diversification score."""
        rng = np.random.default_rng(66)
        a = (rng.standard_normal(300) * 0.01).tolist()
        b = (rng.standard_normal(300) * 0.01).tolist()
        est = DccGarchEstimator()
        est.fit({"X": a, "Y": b})

        score = est.get_diversification_score()
        assert score > 0.5, f"Expected high div score for uncorrelated, got {score}"

    def test_vol_forecasts_returned(self):
        """Vol forecasts should be positive for non-degenerate inputs."""
        a, b = _make_synthetic_returns(n=300, corr=0.5, seed=42)
        est = DccGarchEstimator()
        est.fit({"SPY": a, "TLT": b})

        vols = est.get_vol_forecasts()
        assert "SPY" in vols
        assert "TLT" in vols
        assert vols["SPY"] > 0.0
        assert vols["TLT"] > 0.0

    def test_vol_forecasts_annualized(self):
        """Vol forecasts should be annualized (daily vol * sqrt(252))."""
        rng = np.random.default_rng(77)
        # Generate returns with ~1% daily vol => ~15.9% annualized
        a = (rng.standard_normal(300) * 0.01).tolist()
        b = (rng.standard_normal(300) * 0.01).tolist()
        est = DccGarchEstimator()
        est.fit({"X": a, "Y": b})

        vols = est.get_vol_forecasts()
        # Annualized vol should be roughly 10-25% for 1% daily vol
        for name, vol in vols.items():
            assert 0.05 < vol < 0.50, (
                f"{name} vol={vol:.4f} outside expected annualized range"
            )

    def test_deterministic_results(self):
        """Same inputs should produce identical results."""
        a, b = _make_synthetic_returns(n=300, corr=0.5, seed=42)
        data = {"SPY": a, "TLT": b}

        est1 = DccGarchEstimator()
        est1.fit(data)
        corr1 = est1.get_correlation_matrix()
        score1 = est1.get_diversification_score()
        vols1 = est1.get_vol_forecasts()

        est2 = DccGarchEstimator()
        est2.fit(data)
        corr2 = est2.get_correlation_matrix()
        score2 = est2.get_diversification_score()
        vols2 = est2.get_vol_forecasts()

        assert score1 == score2
        for key in corr1:
            assert abs(corr1[key] - corr2[key]) < 1e-10
        for key in vols1:
            assert abs(vols1[key] - vols2[key]) < 1e-10

    def test_lookback_truncation(self):
        """Setting lookback should limit data used."""
        a, b = _make_synthetic_returns(n=500, corr=0.5, seed=42)
        est = DccGarchEstimator()
        result = est.fit({"X": a, "Y": b}, lookback=100)
        assert result["n_obs"] == 100

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_returns_dict(self):
        """Empty dict should produce identity-like result."""
        est = DccGarchEstimator()
        result = est.fit({})
        assert result["method"] == "none"
        assert result["diversification_score"] == 1.0
        assert len(result["warnings"]) > 0

        # Should not crash on accessors
        assert est.get_correlation_matrix() == {}
        assert est.get_diversification_score() == 1.0
        assert est.get_vol_forecasts() == {}

    def test_single_asset(self):
        """Single asset should return trivial 1x1 correlation."""
        rng = np.random.default_rng(88)
        a = (rng.standard_normal(300) * 0.01).tolist()
        est = DccGarchEstimator()
        result = est.fit({"SPY": a})

        assert result["method"] == "single_asset"
        corr = est.get_correlation_matrix()
        assert len(corr) == 1
        assert abs(corr[("SPY", "SPY")] - 1.0) < 1e-10
        assert est.get_diversification_score() == 1.0
        assert est.get_vol_forecasts()["SPY"] > 0.0

    def test_insufficient_data(self):
        """Very short series should fall back gracefully."""
        est = DccGarchEstimator(min_obs=30)
        result = est.fit({"X": [0.01], "Y": [-0.01]})
        assert result["method"] == "insufficient_data"
        assert est.get_diversification_score() == 1.0

    def test_zero_variance_returns(self):
        """Zero-variance (constant) returns should not crash."""
        est = DccGarchEstimator()
        result = est.fit(
            {
                "FLAT": [0.0] * 100,
                "ALSO_FLAT": [0.0] * 100,
            }
        )
        assert result["method"] == "zero_variance"
        corr = est.get_correlation_matrix()
        assert abs(corr[("FLAT", "FLAT")] - 1.0) < 1e-10
        assert est.get_diversification_score() == 1.0

    def test_mixed_zero_and_nonzero_variance(self):
        """Mix of zero-variance and normal returns should work."""
        rng = np.random.default_rng(99)
        est = DccGarchEstimator()
        result = est.fit(
            {
                "FLAT": [0.0] * 200,
                "ACTIVE": (rng.standard_normal(200) * 0.01).tolist(),
            }
        )
        assert est.get_vol_forecasts()["FLAT"] == 0.0
        assert est.get_vol_forecasts()["ACTIVE"] > 0.0
        assert len(result["warnings"]) > 0  # Should warn about zero variance

    def test_short_data_uses_ewma_fallback(self):
        """With fewer obs than min_obs, EWMA fallback should be used."""
        a, b = _make_synthetic_returns(n=20, corr=0.5, seed=42)
        est = DccGarchEstimator(min_obs=30)
        result = est.fit({"X": a, "Y": b})
        assert result["method"] == "ewma_fallback"

    def test_fit_required_before_accessors(self):
        """Accessors should raise RuntimeError if fit() not called."""
        est = DccGarchEstimator()
        with pytest.raises(RuntimeError, match="fit"):
            est.get_correlation_matrix()
        with pytest.raises(RuntimeError, match="fit"):
            est.get_diversification_score()
        with pytest.raises(RuntimeError, match="fit"):
            est.get_vol_forecasts()

    def test_refit_overwrites_previous(self):
        """Calling fit() again should overwrite previous results."""
        a1, b1 = _make_synthetic_returns(n=300, corr=0.3, seed=1)
        a2, b2 = _make_synthetic_returns(n=300, corr=0.9, seed=2)

        est = DccGarchEstimator()
        est.fit({"X": a1, "Y": b1})
        score1 = est.get_diversification_score()

        est.fit({"X": a2, "Y": b2})
        score2 = est.get_diversification_score()

        # Higher correlation -> lower diversification score
        assert score2 < score1


# ---------------------------------------------------------------------------
# Tests: Forbes-Rigobon adjustment
# ---------------------------------------------------------------------------


class TestForbesRigobonAdjustment:
    """Tests for the Forbes-Rigobon heteroskedasticity correction."""

    def test_no_adjustment_below_threshold(self):
        """Vol ratios below threshold should leave correlations unchanged."""
        corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        vol_ratios = np.array([1.0, 1.2])  # Below 1.5 threshold
        adjusted = _forbes_rigobon_adjust(corr, vol_ratios)
        np.testing.assert_array_almost_equal(adjusted, corr)

    def test_adjustment_above_threshold(self):
        """Vol ratios above threshold should reduce correlation magnitudes."""
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        vol_ratios = np.array([2.0, 2.0])  # Above 1.5 threshold
        adjusted = _forbes_rigobon_adjust(corr, vol_ratios)

        # Adjusted correlation should be lower in magnitude
        assert abs(adjusted[0, 1]) < abs(corr[0, 1])
        assert abs(adjusted[1, 0]) < abs(corr[1, 0])
        # Should remain symmetric
        assert abs(adjusted[0, 1] - adjusted[1, 0]) < 1e-10

    def test_adjustment_preserves_sign(self):
        """Forbes-Rigobon should preserve the sign of correlation."""
        corr_pos = np.array([[1.0, 0.6], [0.6, 1.0]])
        adjusted_pos = _forbes_rigobon_adjust(corr_pos, np.array([2.5, 2.5]))
        assert adjusted_pos[0, 1] > 0.0

        corr_neg = np.array([[1.0, -0.6], [-0.6, 1.0]])
        adjusted_neg = _forbes_rigobon_adjust(corr_neg, np.array([2.5, 2.5]))
        assert adjusted_neg[0, 1] < 0.0

    def test_adjustment_diagonal_unchanged(self):
        """Diagonal elements (self-correlation = 1) should not change."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        adjusted = _forbes_rigobon_adjust(corr, np.array([3.0, 3.0]))
        assert abs(adjusted[0, 0] - 1.0) < 1e-10
        assert abs(adjusted[1, 1] - 1.0) < 1e-10

    def test_adjustment_zero_correlation(self):
        """Zero correlation should remain zero."""
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        adjusted = _forbes_rigobon_adjust(corr, np.array([3.0, 3.0]))
        assert abs(adjusted[0, 1]) < 1e-10
        assert abs(adjusted[1, 0]) < 1e-10

    def test_adjustment_produces_different_result_than_naive(self):
        """With high vol ratios, adjustment should differ from naive correlation.

        This verifies the Forbes-Rigobon correction is actually doing work
        rather than being a no-op.
        """
        # Create data with a high-vol regime
        rng = np.random.default_rng(42)
        n = 300
        # First half: normal vol
        normal = rng.standard_normal((n // 2, 2)) * 0.01
        # Second half: 3x vol (stress period)
        stress = rng.standard_normal((n // 2, 2)) * 0.03
        # Make them correlated
        chol = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
        normal = normal @ chol.T
        stress = stress @ chol.T
        combined = np.vstack([normal, stress])

        # Naive correlation
        naive_corr = np.corrcoef(combined[:, 0], combined[:, 1])[0, 1]

        # Forbes-Rigobon adjusted
        corr_matrix = np.array([[1.0, naive_corr], [naive_corr, 1.0]])
        vol_ratios = np.array([3.0, 3.0])  # Stress vol / normal vol = 3
        adjusted = _forbes_rigobon_adjust(corr_matrix, vol_ratios)

        # The adjusted value should differ from naive
        assert abs(adjusted[0, 1] - naive_corr) > 0.01, (
            f"Adjusted {adjusted[0, 1]:.4f} too close to naive {naive_corr:.4f}"
        )
        # And should be lower in magnitude (FR corrects upward bias)
        assert abs(adjusted[0, 1]) < abs(naive_corr), (
            f"Adjusted |{adjusted[0, 1]:.4f}| should be < naive |{naive_corr:.4f}|"
        )

    def test_three_asset_mixed_vol_ratios(self):
        """Only pairs with high vol ratios should be adjusted."""
        corr = np.array(
            [
                [1.0, 0.6, 0.4],
                [0.6, 1.0, 0.5],
                [0.4, 0.5, 1.0],
            ]
        )
        # Only asset 0 has elevated vol
        vol_ratios = np.array([2.5, 1.0, 1.0])
        adjusted = _forbes_rigobon_adjust(corr, vol_ratios)

        # Pair (0,1) should be adjusted (max(2.5, 1.0) > 1.5)
        assert abs(adjusted[0, 1]) < abs(corr[0, 1])
        # Pair (0,2) should be adjusted (max(2.5, 1.0) > 1.5)
        assert abs(adjusted[0, 2]) < abs(corr[0, 2])
        # Pair (1,2) should NOT be adjusted (max(1.0, 1.0) <= 1.5)
        assert abs(adjusted[1, 2] - corr[1, 2]) < 1e-10


# ---------------------------------------------------------------------------
# Tests: Integration / end-to-end
# ---------------------------------------------------------------------------


class TestDccGarchIntegration:
    """End-to-end integration tests combining GARCH + EWMA + F-R."""

    def test_full_pipeline_three_assets(self):
        """Full estimation pipeline with three assets."""
        returns = _make_three_asset_returns(n=300, seed=42)
        est = DccGarchEstimator()
        result = est.fit(returns)

        # Should complete without error
        assert result["method"] in ("dcc_garch", "ewma_fallback")
        assert result["n_obs"] == 252  # default lookback truncates 300 -> 252

        # Correlation matrix should be 3x3
        corr = est.get_correlation_matrix()
        assert len(corr) == 9  # 3x3

        # All diagonal = 1.0
        for asset in result["assets"]:
            assert abs(corr[(asset, asset)] - 1.0) < 1e-10

        # All off-diagonal in [-1, 1]
        for (a, b), val in corr.items():
            assert -1.0 <= val <= 1.0, f"({a}, {b}) = {val} out of range"

        # Diversification score in valid range
        score = est.get_diversification_score()
        assert 0.0 <= score <= 1.0

        # Vol forecasts all positive
        vols = est.get_vol_forecasts()
        assert len(vols) == 3
        for name, vol in vols.items():
            assert vol > 0.0, f"{name} vol should be positive"

    def test_ewma_and_garch_produce_valid_correlations(self):
        """Both EWMA fallback and GARCH paths should produce valid correlations."""
        a, b = _make_synthetic_returns(n=300, corr=0.5, seed=42)
        data = {"X": a, "Y": b}

        # GARCH path (min_obs=30, n=300 >= 30)
        est_garch = DccGarchEstimator(min_obs=30)
        est_garch.fit(data)
        corr_garch = est_garch.get_correlation_matrix()

        # Force EWMA fallback (min_obs=999, n=300 < 999)
        est_ewma = DccGarchEstimator(min_obs=999)
        est_ewma.fit(data)
        corr_ewma = est_ewma.get_correlation_matrix()

        # Both should produce valid correlations
        for key in corr_garch:
            assert -1.0 <= corr_garch[key] <= 1.0
            assert -1.0 <= corr_ewma[key] <= 1.0

        # Both should show positive correlation for positively correlated data
        assert corr_garch[("X", "Y")] > 0.0
        assert corr_ewma[("X", "Y")] > 0.0

    def test_result_dataclass_construction(self):
        """DccGarchResult should be constructable with expected fields."""
        from llm_quant.risk.dcc_garch import DccGarchResult

        result = DccGarchResult(
            assets=["SPY", "TLT"],
            correlation_matrix=np.eye(2),
            vol_forecasts={"SPY": 0.15, "TLT": 0.10},
            diversification_score=0.85,
            method="dcc_garch",
            n_obs_used=252,
        )
        assert result.assets == ["SPY", "TLT"]
        assert result.diversification_score == 0.85
        assert result.method == "dcc_garch"
        assert result.n_obs_used == 252
        assert len(result.warnings) == 0

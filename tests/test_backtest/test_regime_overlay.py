"""Tests for regime overlay meta-strategy.

Validates:
- Regime classification (all three regimes with various VIX/SMA combos)
- Weight adjustment (verify weights sum to 1.0 after adjustment)
- Risk-off correctly reduces credit lead-lag
- Risk-on correctly reduces trend_following weights
- Transition returns base weights unchanged
- Tilt magnitude of 0.0 returns base weights
- Edge cases (empty weights, single strategy, unknown family)
- Backtest simulation produces sensible metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_quant.backtest.regime_overlay import (
    FAMILY_COMMODITY,
    FAMILY_CREDIT_LEAD_LAG,
    FAMILY_DEFENSIVE,
    FAMILY_PAIRS,
    FAMILY_TREND_FOLLOWING,
    REGIME_RISK_OFF,
    REGIME_RISK_ON,
    REGIME_TRANSITION,
    RegimeClassifier,
    RegimeOverlay,
    backtest_regime_overlay,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_weights() -> dict[str, float]:
    """Representative base HRP weights across families."""
    return {
        "lqd-spy-credit-lead": 0.20,
        "agg-spy-credit-lead": 0.10,
        "skip-month-tsmom-v1": 0.15,
        "gld-slv-mean-reversion-v4": 0.15,
        "vol-regime-v2": 0.10,
        "commodity-carry-v2": 0.10,
        "soxx-qqq-lead-lag": 0.20,
    }


@pytest.fixture
def strategy_families() -> dict[str, str]:
    """Map strategy slugs to family types."""
    return {
        "lqd-spy-credit-lead": FAMILY_CREDIT_LEAD_LAG,
        "agg-spy-credit-lead": FAMILY_CREDIT_LEAD_LAG,
        "skip-month-tsmom-v1": FAMILY_TREND_FOLLOWING,
        "gld-slv-mean-reversion-v4": FAMILY_PAIRS,
        "vol-regime-v2": FAMILY_DEFENSIVE,
        "commodity-carry-v2": FAMILY_COMMODITY,
        "soxx-qqq-lead-lag": FAMILY_PAIRS,
    }


@pytest.fixture
def overlay(strategy_families: dict[str, str]) -> RegimeOverlay:
    """Default overlay instance."""
    return RegimeOverlay(
        strategy_families=strategy_families,
        tilt_magnitude=0.30,
    )


# ---------------------------------------------------------------------------
# Test: RegimeClassifier
# ---------------------------------------------------------------------------


class TestRegimeClassifier:
    """Regime classification with various VIX/SMA combinations."""

    def test_risk_off_high_vix(self):
        """VIX > 25 triggers risk-off regardless of SPY trend."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        assert c.classify(vix=30.0, spy_sma200_ratio=1.05) == REGIME_RISK_OFF

    def test_risk_off_spy_below_sma(self):
        """SPY below 200d SMA triggers risk-off even with low VIX."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        assert c.classify(vix=15.0, spy_sma200_ratio=0.95) == REGIME_RISK_OFF

    def test_risk_off_both_conditions(self):
        """Both VIX high and SPY below SMA: risk-off."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        assert c.classify(vix=35.0, spy_sma200_ratio=0.90) == REGIME_RISK_OFF

    def test_risk_on(self):
        """VIX < 18 AND SPY > SMA200: risk-on."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        assert c.classify(vix=12.0, spy_sma200_ratio=1.10) == REGIME_RISK_ON

    def test_transition_vix_between_thresholds(self):
        """VIX between 18 and 25, SPY above SMA: transition."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        assert c.classify(vix=20.0, spy_sma200_ratio=1.05) == REGIME_TRANSITION

    def test_transition_vix_at_risk_on_boundary(self):
        """VIX exactly at risk-on boundary (18.0), SPY above SMA: transition."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        # VIX=18.0 is NOT < 18.0, so it falls to transition
        assert c.classify(vix=18.0, spy_sma200_ratio=1.05) == REGIME_TRANSITION

    def test_risk_off_vix_at_boundary(self):
        """VIX exactly at risk-off boundary (25.0): NOT risk-off (requires >)."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        # VIX=25.0 is NOT > 25.0, SPY above SMA -> transition
        assert c.classify(vix=25.0, spy_sma200_ratio=1.05) == REGIME_TRANSITION

    def test_risk_off_spy_exactly_at_sma(self):
        """SPY/SMA200 ratio exactly 1.0: NOT below SMA, not risk-off from trend."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        # ratio=1.0 is NOT < 1.0, and VIX=20 is between thresholds
        # but risk-on requires ratio > 1.0 strictly, so transition
        assert c.classify(vix=20.0, spy_sma200_ratio=1.0) == REGIME_TRANSITION

    def test_risk_on_requires_spy_above_sma(self):
        """Risk-on needs SPY strictly above SMA200."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        # VIX=12 (< 18), but ratio=1.0 is not > 1.0 -> transition
        assert c.classify(vix=12.0, spy_sma200_ratio=1.0) == REGIME_TRANSITION

    def test_custom_thresholds(self):
        """Custom VIX thresholds are respected."""
        c = RegimeClassifier(vix_risk_off=30.0, vix_risk_on=15.0)
        # VIX=20 with default would be transition; with new threshold still is
        assert c.classify(vix=20.0, spy_sma200_ratio=1.05) == REGIME_TRANSITION
        # VIX=31 exceeds the higher risk-off threshold
        assert c.classify(vix=31.0, spy_sma200_ratio=1.05) == REGIME_RISK_OFF
        # VIX=14 is below the lower risk-on threshold
        assert c.classify(vix=14.0, spy_sma200_ratio=1.05) == REGIME_RISK_ON

    def test_get_regime_history(self):
        """History method classifies each observation."""
        c = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
        vix = [30.0, 15.0, 20.0, 10.0, 28.0]
        ratios = [1.05, 0.95, 1.03, 1.10, 0.92]
        history = c.get_regime_history(vix, ratios)

        assert len(history) == 5
        assert history[0] == REGIME_RISK_OFF  # VIX > 25
        assert history[1] == REGIME_RISK_OFF  # SPY below SMA
        assert history[2] == REGIME_TRANSITION  # VIX=20, SPY above
        assert history[3] == REGIME_RISK_ON  # VIX=10, SPY above
        assert history[4] == REGIME_RISK_OFF  # VIX > 25 AND SPY below

    def test_get_regime_history_length_mismatch(self):
        """Mismatched series lengths raise ValueError."""
        c = RegimeClassifier()
        with pytest.raises(ValueError, match="length mismatch"):
            c.get_regime_history([20.0, 15.0], [1.0])

    def test_get_regime_history_empty(self):
        """Empty series returns empty history."""
        c = RegimeClassifier()
        assert c.get_regime_history([], []) == []


# ---------------------------------------------------------------------------
# Test: RegimeOverlay weight adjustment
# ---------------------------------------------------------------------------


class TestRegimeOverlayWeights:
    """Weight adjustment logic."""

    def test_weights_sum_to_one_risk_off(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Adjusted weights sum to 1.0 in risk-off."""
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10

    def test_weights_sum_to_one_risk_on(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Adjusted weights sum to 1.0 in risk-on."""
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_ON)
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10

    def test_weights_sum_to_one_transition(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Adjusted weights sum to 1.0 in transition."""
        adjusted = overlay.adjust_weights(base_weights, REGIME_TRANSITION)
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10

    def test_transition_preserves_base_weights(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Transition regime returns base weights (normalized) unchanged."""
        adjusted = overlay.adjust_weights(base_weights, REGIME_TRANSITION)
        base_total = sum(base_weights.values())
        for slug, w in adjusted.items():
            expected = base_weights[slug] / base_total
            assert abs(w - expected) < 1e-10, (
                f"{slug}: expected {expected:.6f}, got {w:.6f}"
            )

    def test_risk_off_reduces_credit_lead_lag(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Risk-off reduces credit_lead_lag weights."""
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)

        # Credit lead-lag strategies should have lower weight
        for slug in ["lqd-spy-credit-lead", "agg-spy-credit-lead"]:
            assert adjusted[slug] < base_normalized[slug], (
                f"{slug} weight should decrease in risk-off: "
                f"base={base_normalized[slug]:.4f}, adjusted={adjusted[slug]:.4f}"
            )

    def test_risk_off_increases_trend_and_pairs(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Risk-off increases trend_following and pairs weights."""
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)

        # Trend following and pairs strategies should have higher weight
        for slug in [
            "skip-month-tsmom-v1",
            "gld-slv-mean-reversion-v4",
            "soxx-qqq-lead-lag",
        ]:
            assert adjusted[slug] > base_normalized[slug], (
                f"{slug} weight should increase in risk-off: "
                f"base={base_normalized[slug]:.4f}, adjusted={adjusted[slug]:.4f}"
            )

    def test_risk_on_reduces_trend_following(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Risk-on reduces trend_following weight."""
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_ON)

        # Trend following should have lower weight
        assert adjusted["skip-month-tsmom-v1"] < base_normalized["skip-month-tsmom-v1"]

    def test_risk_on_increases_credit_lead_lag(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Risk-on increases credit_lead_lag weight."""
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_ON)

        # Credit lead-lag strategies should have higher weight
        for slug in ["lqd-spy-credit-lead", "agg-spy-credit-lead"]:
            assert adjusted[slug] > base_normalized[slug], (
                f"{slug} weight should increase in risk-on: "
                f"base={base_normalized[slug]:.4f}, adjusted={adjusted[slug]:.4f}"
            )

    def test_risk_on_asymmetric_tilt(
        self,
        strategy_families: dict[str, str],
        base_weights: dict[str, float],
    ):
        """Risk-on tilt is weaker than risk-off tilt (asymmetric 0.5x factor)."""
        overlay = RegimeOverlay(
            strategy_families=strategy_families,
            tilt_magnitude=0.30,
        )
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }

        risk_off = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)
        risk_on = overlay.adjust_weights(base_weights, REGIME_RISK_ON)

        # Credit lead-lag reduction in risk-off should be larger magnitude
        # than trend following reduction in risk-on
        credit_slug = "lqd-spy-credit-lead"
        trend_slug = "skip-month-tsmom-v1"

        credit_reduction = base_normalized[credit_slug] - risk_off[credit_slug]
        trend_reduction = base_normalized[trend_slug] - risk_on[trend_slug]

        # Normalize by base weight to compare relative reductions
        rel_credit_reduction = credit_reduction / base_normalized[credit_slug]
        rel_trend_reduction = trend_reduction / base_normalized[trend_slug]

        # Risk-off relative reduction should be ~2x risk-on
        assert rel_credit_reduction > rel_trend_reduction, (
            "Risk-off tilt should be stronger than risk-on tilt"
        )


# ---------------------------------------------------------------------------
# Test: Zero tilt magnitude
# ---------------------------------------------------------------------------


class TestZeroTiltMagnitude:
    """Tilt magnitude of 0.0 always returns base weights."""

    def test_zero_tilt_risk_off(self, strategy_families, base_weights):
        overlay = RegimeOverlay(
            strategy_families=strategy_families,
            tilt_magnitude=0.0,
        )
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        for slug in adjusted:
            assert abs(adjusted[slug] - base_normalized[slug]) < 1e-10

    def test_zero_tilt_risk_on(self, strategy_families, base_weights):
        overlay = RegimeOverlay(
            strategy_families=strategy_families,
            tilt_magnitude=0.0,
        )
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_ON)
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        for slug in adjusted:
            assert abs(adjusted[slug] - base_normalized[slug]) < 1e-10


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty weights, single strategy, unknown family."""

    def test_empty_weights(self, strategy_families):
        overlay = RegimeOverlay(strategy_families=strategy_families)
        adjusted = overlay.adjust_weights({}, REGIME_RISK_OFF)
        assert adjusted == {}

    def test_single_strategy(self, strategy_families):
        """Single strategy gets weight 1.0 regardless of regime."""
        overlay = RegimeOverlay(strategy_families=strategy_families)
        adjusted = overlay.adjust_weights({"lqd-spy-credit-lead": 1.0}, REGIME_RISK_OFF)
        assert abs(adjusted["lqd-spy-credit-lead"] - 1.0) < 1e-10

    def test_single_strategy_risk_on(self, strategy_families):
        """Single trend strategy: weight stays 1.0 in risk-on (no beneficiaries)."""
        overlay = RegimeOverlay(strategy_families=strategy_families)
        adjusted = overlay.adjust_weights({"skip-month-tsmom-v1": 1.0}, REGIME_RISK_ON)
        assert abs(adjusted["skip-month-tsmom-v1"] - 1.0) < 1e-10

    def test_unknown_family_left_unchanged(self):
        """Strategies with unknown families are not penalized or boosted."""
        families = {
            "known-credit": FAMILY_CREDIT_LEAD_LAG,
            "known-trend": FAMILY_TREND_FOLLOWING,
            "unknown-strat": "some_unknown_family",
        }
        overlay = RegimeOverlay(
            strategy_families=families,
            tilt_magnitude=0.30,
        )
        weights = {
            "known-credit": 0.40,
            "known-trend": 0.30,
            "unknown-strat": 0.30,
        }
        adjusted = overlay.adjust_weights(weights, REGIME_RISK_OFF)

        # Weights should still sum to 1.0
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10

        # Unknown strategy should not be directly penalized
        # (its weight may change slightly due to normalization, but it
        # should not be in the penalized or beneficiary sets)
        base_normalized = {k: v / sum(weights.values()) for k, v in weights.items()}
        # Credit is penalized (reduced), trend is beneficiary (increased)
        assert adjusted["known-credit"] < base_normalized["known-credit"]
        assert adjusted["known-trend"] > base_normalized["known-trend"]

    def test_no_family_mapping(self):
        """If strategy_families is empty, no tilting occurs (nothing is penalized)."""
        overlay = RegimeOverlay(
            strategy_families={},
            tilt_magnitude=0.50,
        )
        weights = {"a": 0.5, "b": 0.5}
        adjusted = overlay.adjust_weights(weights, REGIME_RISK_OFF)
        # No strategies match any family, so nothing changes
        assert abs(adjusted["a"] - 0.5) < 1e-10
        assert abs(adjusted["b"] - 0.5) < 1e-10

    def test_all_weights_zero(self, strategy_families):
        """All zero weights: returned as-is (no division by zero)."""
        overlay = RegimeOverlay(strategy_families=strategy_families)
        weights = {"lqd-spy-credit-lead": 0.0, "skip-month-tsmom-v1": 0.0}
        adjusted = overlay.adjust_weights(weights, REGIME_RISK_OFF)
        # Total is 0; normalize returns as-is
        assert sum(adjusted.values()) == 0.0

    def test_invalid_regime_raises(self, overlay, base_weights):
        """Invalid regime string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regime"):
            overlay.adjust_weights(base_weights, "invalid_regime")

    def test_only_penalized_families(self):
        """All penalized, no beneficiaries: weight redistributes back."""
        families = {
            "a": FAMILY_CREDIT_LEAD_LAG,
            "b": FAMILY_CREDIT_LEAD_LAG,
        }
        overlay = RegimeOverlay(
            strategy_families=families,
            tilt_magnitude=0.30,
        )
        weights = {"a": 0.6, "b": 0.4}
        adjusted = overlay.adjust_weights(weights, REGIME_RISK_OFF)
        # All penalized, no beneficiaries -> weight comes back to them
        # Should still sum to 1.0
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10

    def test_no_penalized_families_risk_off(self):
        """If no credit_lead_lag strategies exist, risk-off is a no-op."""
        families = {
            "a": FAMILY_TREND_FOLLOWING,
            "b": FAMILY_PAIRS,
        }
        overlay = RegimeOverlay(
            strategy_families=families,
            tilt_magnitude=0.50,
        )
        weights = {"a": 0.6, "b": 0.4}
        adjusted = overlay.adjust_weights(weights, REGIME_RISK_OFF)
        assert abs(adjusted["a"] - 0.6) < 1e-10
        assert abs(adjusted["b"] - 0.4) < 1e-10


# ---------------------------------------------------------------------------
# Test: adjust_weights_from_market convenience method
# ---------------------------------------------------------------------------


class TestAdjustFromMarket:
    """Test the combined classify + adjust method."""

    def test_convenience_matches_manual(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Convenience method gives same result as classify + adjust."""
        vix = 30.0
        ratio = 1.05

        adjusted_manual = overlay.adjust_weights(
            base_weights,
            overlay.classifier.classify(vix, ratio),
        )
        adjusted_conv, regime = overlay.adjust_weights_from_market(
            base_weights, vix, ratio
        )

        assert regime == REGIME_RISK_OFF
        for slug in adjusted_manual:
            assert abs(adjusted_manual[slug] - adjusted_conv[slug]) < 1e-10


# ---------------------------------------------------------------------------
# Test: backtest_regime_overlay simulation
# ---------------------------------------------------------------------------


class TestBacktestRegimeOverlay:
    """Backtest simulation tests."""

    def _make_returns(
        self,
        n_days: int = 500,
        n_strategies: int = 5,
        seed: int = 42,
    ) -> tuple[dict[str, list[float]], dict[str, float], dict[str, str]]:
        """Generate synthetic returns, weights, and families."""
        rng = np.random.default_rng(seed)
        slugs = [f"strat-{i}" for i in range(n_strategies)]
        families_list = [
            FAMILY_CREDIT_LEAD_LAG,
            FAMILY_TREND_FOLLOWING,
            FAMILY_PAIRS,
            FAMILY_DEFENSIVE,
            FAMILY_COMMODITY,
        ]

        returns = {}
        families = {}
        weights = {}
        for i, slug in enumerate(slugs):
            # Small positive drift with noise
            returns[slug] = (rng.normal(0.0003, 0.01, n_days)).tolist()
            families[slug] = families_list[i % len(families_list)]
            weights[slug] = 1.0 / n_strategies

        return returns, weights, families

    def test_backtest_returns_all_metrics(self):
        """Backtest returns all expected metric keys."""
        returns, weights, families = self._make_returns()
        regime_history = [REGIME_TRANSITION] * len(returns["strat-0"])

        result = backtest_regime_overlay(
            base_returns=returns,
            base_weights=weights,
            regime_history=regime_history,
            strategy_families=families,
        )

        expected_keys = {
            "base_sharpe",
            "overlay_sharpe",
            "base_max_dd",
            "overlay_max_dd",
            "base_cagr",
            "overlay_cagr",
            "regime_counts",
            "n_days",
        }
        assert set(result.keys()) == expected_keys

    def test_backtest_transition_only_matches_base(self):
        """All-transition regime means overlay = base (no tilting)."""
        returns, weights, families = self._make_returns()
        regime_history = [REGIME_TRANSITION] * len(returns["strat-0"])

        result = backtest_regime_overlay(
            base_returns=returns,
            base_weights=weights,
            regime_history=regime_history,
            strategy_families=families,
        )

        assert abs(result["base_sharpe"] - result["overlay_sharpe"]) < 1e-6
        assert abs(result["base_max_dd"] - result["overlay_max_dd"]) < 1e-6
        assert abs(result["base_cagr"] - result["overlay_cagr"]) < 1e-6

    def test_backtest_regime_counts(self):
        """Regime counts match the input history."""
        returns, weights, families = self._make_returns(n_days=300)
        regime_history = (
            [REGIME_RISK_OFF] * 100 + [REGIME_TRANSITION] * 100 + [REGIME_RISK_ON] * 100
        )

        result = backtest_regime_overlay(
            base_returns=returns,
            base_weights=weights,
            regime_history=regime_history,
            strategy_families=families,
        )

        assert result["regime_counts"][REGIME_RISK_OFF] == 100
        assert result["regime_counts"][REGIME_TRANSITION] == 100
        assert result["regime_counts"][REGIME_RISK_ON] == 100
        assert result["n_days"] == 300

    def test_backtest_empty_returns(self):
        """Empty returns dict produces zero metrics."""
        result = backtest_regime_overlay(
            base_returns={},
            base_weights={},
            regime_history=[],
            strategy_families={},
        )
        assert result["n_days"] == 0
        assert result["base_sharpe"] == 0.0

    def test_backtest_length_mismatch_raises(self):
        """Mismatched return series length raises ValueError."""
        returns = {
            "a": [0.01] * 100,
            "b": [0.01] * 99,  # wrong length
        }
        weights = {"a": 0.5, "b": 0.5}
        regime_history = [REGIME_TRANSITION] * 100

        with pytest.raises(ValueError, match="length mismatch"):
            backtest_regime_overlay(
                base_returns=returns,
                base_weights=weights,
                regime_history=regime_history,
                strategy_families={},
            )

    def test_backtest_overlay_differs_from_base_with_regime_changes(self):
        """When regime changes, overlay should produce different results from base."""
        returns, weights, families = self._make_returns(n_days=500, seed=123)

        # Create a regime history with mixed regimes
        regime_history = (
            [REGIME_RISK_ON] * 200 + [REGIME_RISK_OFF] * 200 + [REGIME_RISK_ON] * 100
        )

        result = backtest_regime_overlay(
            base_returns=returns,
            base_weights=weights,
            regime_history=regime_history,
            strategy_families=families,
            tilt_magnitude=0.50,
        )

        # With mixed regimes and non-zero tilt, overlay should differ from base
        # (It's theoretically possible they match exactly, but extremely unlikely
        # with random returns)
        sharpe_diff = abs(result["base_sharpe"] - result["overlay_sharpe"])
        cagr_diff = abs(result["base_cagr"] - result["overlay_cagr"])

        # At least one metric should differ meaningfully
        assert sharpe_diff > 1e-6 or cagr_diff > 1e-6, (
            "Overlay should produce different metrics when regimes change"
        )

    def test_backtest_max_drawdown_non_negative(self):
        """Max drawdown should always be non-negative."""
        returns, weights, families = self._make_returns()
        regime_history = [REGIME_RISK_OFF] * len(returns["strat-0"])

        result = backtest_regime_overlay(
            base_returns=returns,
            base_weights=weights,
            regime_history=regime_history,
            strategy_families=families,
        )

        assert result["base_max_dd"] >= 0.0
        assert result["overlay_max_dd"] >= 0.0

    def test_backtest_zero_tilt_matches_base(self):
        """Zero tilt magnitude means overlay always matches base."""
        returns, weights, families = self._make_returns()
        regime_history = [REGIME_RISK_OFF] * len(returns["strat-0"])

        result = backtest_regime_overlay(
            base_returns=returns,
            base_weights=weights,
            regime_history=regime_history,
            strategy_families=families,
            tilt_magnitude=0.0,
        )

        assert abs(result["base_sharpe"] - result["overlay_sharpe"]) < 1e-6
        assert abs(result["base_cagr"] - result["overlay_cagr"]) < 1e-6


# ---------------------------------------------------------------------------
# Test: Full tilt magnitude (1.0)
# ---------------------------------------------------------------------------


class TestFullTiltMagnitude:
    """Full tilt (1.0) should zero out penalized families."""

    def test_full_tilt_risk_off_zeros_credit(self, strategy_families, base_weights):
        """With tilt=1.0, credit_lead_lag weight goes to zero in risk-off."""
        overlay = RegimeOverlay(
            strategy_families=strategy_families,
            tilt_magnitude=1.0,
        )
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)

        for slug in ["lqd-spy-credit-lead", "agg-spy-credit-lead"]:
            assert adjusted[slug] < 1e-10, (
                f"{slug} should be ~0 with full tilt risk-off"
            )
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10

    def test_full_tilt_risk_on_halves_trend(self, strategy_families, base_weights):
        """With tilt=1.0, risk-on uses 0.5 scale, so trend weight halves."""
        overlay = RegimeOverlay(
            strategy_families=strategy_families,
            tilt_magnitude=1.0,
        )
        base_normalized = {
            k: v / sum(base_weights.values()) for k, v in base_weights.items()
        }
        adjusted = overlay.adjust_weights(base_weights, REGIME_RISK_ON)

        # Trend should be reduced by 50% (scale=0.5), then renormalized
        # The exact value depends on normalization, but it should be reduced
        assert adjusted["skip-month-tsmom-v1"] < base_normalized["skip-month-tsmom-v1"]
        assert abs(sum(adjusted.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Overlay should be deterministic."""

    def test_same_inputs_same_outputs(
        self,
        overlay: RegimeOverlay,
        base_weights: dict[str, float],
    ):
        """Calling adjust_weights twice with same inputs gives same result."""
        result1 = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)
        result2 = overlay.adjust_weights(base_weights, REGIME_RISK_OFF)

        for slug in result1:
            assert abs(result1[slug] - result2[slug]) < 1e-15

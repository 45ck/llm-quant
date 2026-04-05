"""Regime overlay meta-strategy for dynamic portfolio weight allocation.

Adjusts HRP portfolio weights based on the current market regime (risk-on,
risk-off, or transition) using VIX level and SPY trend relative to its 200-day
SMA.  This is a weight-tilting layer that sits on top of the base HRP
allocation -- it does NOT generate trade signals or modify individual
strategy logic.

Regime classification:
    risk_off  -- VIX > vix_risk_off OR SPY < 200d SMA
    risk_on   -- VIX < vix_risk_on AND SPY > 200d SMA
    transition -- everything else (hold base HRP weights)

Weight adjustment rules:
    risk_off   -- reduce credit_lead_lag weights, redistribute to
                  trend_following + pairs (defensive tilt)
    risk_on    -- reduce trend_following weights, redistribute to
                  credit_lead_lag (risk-on tilt)
    transition -- return base weights unchanged

Usage:
    from llm_quant.backtest.regime_overlay import RegimeClassifier, RegimeOverlay

    classifier = RegimeClassifier(vix_risk_off=25.0, vix_risk_on=18.0)
    regime = classifier.classify(vix=30.0, spy_sma200_ratio=0.97)
    # => "risk_off"

    overlay = RegimeOverlay(
        strategy_families={"lqd-spy-credit-lead": "credit_lead_lag", ...},
        tilt_magnitude=0.30,
    )
    adjusted = overlay.adjust_weights(base_weights, regime)
    # => dict with tilted weights summing to 1.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Strategy family types recognized by the overlay
FAMILY_CREDIT_LEAD_LAG = "credit_lead_lag"
FAMILY_TREND_FOLLOWING = "trend_following"
FAMILY_PAIRS = "pairs"
FAMILY_DEFENSIVE = "defensive"
FAMILY_COMMODITY = "commodity"

# Regime labels
REGIME_RISK_ON = "risk_on"
REGIME_RISK_OFF = "risk_off"
REGIME_TRANSITION = "transition"

VALID_REGIMES = {REGIME_RISK_ON, REGIME_RISK_OFF, REGIME_TRANSITION}

# Families that receive weight in risk-off redistribution
RISK_OFF_BENEFICIARIES = {FAMILY_TREND_FOLLOWING, FAMILY_PAIRS, FAMILY_DEFENSIVE}

# Families that receive weight in risk-on redistribution
RISK_ON_BENEFICIARIES = {FAMILY_CREDIT_LEAD_LAG}


@dataclass
class RegimeClassifier:
    """Classify market regime from VIX level and SPY/SMA200 ratio.

    Parameters
    ----------
    vix_risk_off:
        VIX threshold above which regime is classified as risk-off
        (default: 25.0).
    vix_risk_on:
        VIX threshold below which (combined with SPY above SMA200)
        regime is classified as risk-on (default: 18.0).
    """

    vix_risk_off: float = 25.0
    vix_risk_on: float = 18.0

    def classify(self, vix: float, spy_sma200_ratio: float) -> str:
        """Classify the current market regime.

        Parameters
        ----------
        vix:
            Current VIX level.
        spy_sma200_ratio:
            Ratio of SPY price to its 200-day SMA.  Values > 1.0 mean
            SPY is above the SMA; values < 1.0 mean below.

        Returns
        -------
        str
            One of ``"risk_on"``, ``"risk_off"``, or ``"transition"``.
        """
        # Risk-off: VIX elevated OR SPY below 200d SMA
        if vix > self.vix_risk_off or spy_sma200_ratio < 1.0:
            return REGIME_RISK_OFF

        # Risk-on: VIX calm AND SPY above 200d SMA
        if vix < self.vix_risk_on and spy_sma200_ratio > 1.0:
            return REGIME_RISK_ON

        # Everything else: transition
        return REGIME_TRANSITION

    def get_regime_history(
        self,
        vix_series: list[float],
        spy_sma_ratios: list[float],
    ) -> list[str]:
        """Classify regime for each observation in a time series.

        Parameters
        ----------
        vix_series:
            List of VIX values (chronological order).
        spy_sma_ratios:
            List of SPY/SMA200 ratios (same length as vix_series).

        Returns
        -------
        list[str]
            Regime classification for each observation.

        Raises
        ------
        ValueError
            If the two series have different lengths.
        """
        if len(vix_series) != len(spy_sma_ratios):
            raise ValueError(
                f"Series length mismatch: vix_series has {len(vix_series)} "
                f"elements, spy_sma_ratios has {len(spy_sma_ratios)}"
            )
        return [
            self.classify(vix, ratio)
            for vix, ratio in zip(vix_series, spy_sma_ratios, strict=True)
        ]


@dataclass
class RegimeOverlay:
    """Adjust portfolio weights based on market regime.

    Takes base HRP weights and tilts them according to the current regime.
    Strategies are classified by family type, and the overlay shifts weight
    from penalized families to beneficiary families.

    Parameters
    ----------
    strategy_families:
        Maps strategy slug to family type.  Recognized family types:
        ``credit_lead_lag``, ``trend_following``, ``pairs``, ``defensive``,
        ``commodity``.  Unknown families are left unchanged.
    tilt_magnitude:
        How aggressively to tilt weights (0.0 = no tilt, 1.0 = full tilt).
        Default: 0.30.
    vix_risk_off:
        VIX threshold for risk-off regime (passed to RegimeClassifier).
    vix_risk_on:
        VIX threshold for risk-on regime (passed to RegimeClassifier).
    """

    strategy_families: dict[str, str] = field(default_factory=dict)
    tilt_magnitude: float = 0.30
    vix_risk_off: float = 25.0
    vix_risk_on: float = 18.0

    def __post_init__(self) -> None:
        self._classifier = RegimeClassifier(
            vix_risk_off=self.vix_risk_off,
            vix_risk_on=self.vix_risk_on,
        )

    @property
    def classifier(self) -> RegimeClassifier:
        """Access the underlying regime classifier."""
        return self._classifier

    def adjust_weights(
        self,
        base_weights: dict[str, float],
        regime: str,
    ) -> dict[str, float]:
        """Adjust portfolio weights based on regime.

        Parameters
        ----------
        base_weights:
            Dict mapping strategy slug to HRP weight (should sum to ~1.0).
        regime:
            Current regime: ``"risk_on"``, ``"risk_off"``, or ``"transition"``.

        Returns
        -------
        dict[str, float]
            Adjusted weights, normalized to sum to 1.0.

        Raises
        ------
        ValueError
            If regime is not one of the three valid values.
        """
        if regime not in VALID_REGIMES:
            raise ValueError(
                f"Invalid regime '{regime}'. Must be one of {VALID_REGIMES}"
            )

        # Empty weights or zero tilt: return normalized base
        if not base_weights:
            return {}

        if self.tilt_magnitude == 0.0 or regime == REGIME_TRANSITION:
            return _normalize(dict(base_weights))

        if regime == REGIME_RISK_OFF:
            return self._apply_risk_off_tilt(base_weights)
        # REGIME_RISK_ON
        return self._apply_risk_on_tilt(base_weights)

    def adjust_weights_from_market(
        self,
        base_weights: dict[str, float],
        vix: float,
        spy_sma200_ratio: float,
    ) -> tuple[dict[str, float], str]:
        """Classify regime and adjust weights in one call.

        Parameters
        ----------
        base_weights:
            Base HRP weights.
        vix:
            Current VIX level.
        spy_sma200_ratio:
            SPY price / SMA200.

        Returns
        -------
        tuple[dict[str, float], str]
            (adjusted_weights, regime_label).
        """
        regime = self._classifier.classify(vix, spy_sma200_ratio)
        adjusted = self.adjust_weights(base_weights, regime)
        return adjusted, regime

    def _apply_risk_off_tilt(self, base_weights: dict[str, float]) -> dict[str, float]:
        """Risk-off: reduce credit_lead_lag, redistribute to trend/pairs/defensive."""
        adjusted = dict(base_weights)
        penalized_families = {FAMILY_CREDIT_LEAD_LAG}
        beneficiary_families = RISK_OFF_BENEFICIARIES

        return self._redistribute(adjusted, penalized_families, beneficiary_families)

    def _apply_risk_on_tilt(self, base_weights: dict[str, float]) -> dict[str, float]:
        """Risk-on: reduce trend_following, redistribute to credit_lead_lag.

        Uses tilt_magnitude * 0.5 for a more conservative risk-on shift
        (asymmetric -- we tilt harder into defense than into offense).
        """
        adjusted = dict(base_weights)
        penalized_families = {FAMILY_TREND_FOLLOWING}
        beneficiary_families = RISK_ON_BENEFICIARIES

        # Risk-on uses half tilt magnitude (asymmetric)
        return self._redistribute(
            adjusted, penalized_families, beneficiary_families, scale=0.5
        )

    def _redistribute(
        self,
        weights: dict[str, float],
        penalized_families: set[str],
        beneficiary_families: set[str],
        scale: float = 1.0,
    ) -> dict[str, float]:
        """Shift weight from penalized families to beneficiary families.

        Parameters
        ----------
        weights:
            Current weights (will be modified in place then normalized).
        penalized_families:
            Family types whose weights are reduced.
        beneficiary_families:
            Family types that receive the redistributed weight.
        scale:
            Multiplier on tilt_magnitude (default 1.0 for risk-off,
            0.5 for risk-on).
        """
        effective_tilt = self.tilt_magnitude * scale

        # Identify penalized and beneficiary strategies
        penalized_slugs = [
            s for s in weights if self.strategy_families.get(s) in penalized_families
        ]
        beneficiary_slugs = [
            s for s in weights if self.strategy_families.get(s) in beneficiary_families
        ]

        # Compute weight to redistribute
        weight_to_redistribute = 0.0
        for slug in penalized_slugs:
            reduction = weights[slug] * effective_tilt
            weights[slug] -= reduction
            weight_to_redistribute += reduction

        # Distribute to beneficiaries proportionally to their base weights
        if beneficiary_slugs and weight_to_redistribute > 0:
            beneficiary_total = sum(weights[s] for s in beneficiary_slugs)
            if beneficiary_total > 0:
                for slug in beneficiary_slugs:
                    share = weights[slug] / beneficiary_total
                    weights[slug] += weight_to_redistribute * share
            else:
                # Equal distribution if all beneficiaries have zero weight
                per_strategy = weight_to_redistribute / len(beneficiary_slugs)
                for slug in beneficiary_slugs:
                    weights[slug] += per_strategy
        elif not beneficiary_slugs and weight_to_redistribute > 0:
            # No beneficiaries exist: redistribute back to all non-penalized
            non_penalized = [s for s in weights if s not in penalized_slugs]
            if non_penalized:
                total_non_penalized = sum(weights[s] for s in non_penalized)
                if total_non_penalized > 0:
                    for slug in non_penalized:
                        share = weights[slug] / total_non_penalized
                        weights[slug] += weight_to_redistribute * share
                else:
                    per_strategy = weight_to_redistribute / len(non_penalized)
                    for slug in non_penalized:
                        weights[slug] += per_strategy
            else:
                # All strategies are penalized; add back evenly
                per_strategy = weight_to_redistribute / len(penalized_slugs)
                for slug in penalized_slugs:
                    weights[slug] += per_strategy

        return _normalize(weights)


def backtest_regime_overlay(
    base_returns: dict[str, list[float]],
    base_weights: dict[str, float],
    regime_history: list[str],
    strategy_families: dict[str, str],
    tilt_magnitude: float = 0.30,
) -> dict:
    """Simulate regime overlay on historical data.

    Runs a day-by-day simulation where portfolio weights are adjusted
    based on the regime classification for each day.

    Parameters
    ----------
    base_returns:
        Dict mapping strategy slug to list of daily returns (all same
        length, chronological order).
    base_weights:
        Base HRP weights (dict slug -> weight).
    regime_history:
        Regime label for each day (same length as return series).
    strategy_families:
        Maps strategy slug to family type.
    tilt_magnitude:
        Tilt magnitude for the overlay.

    Returns
    -------
    dict
        Performance metrics including:
        - ``base_sharpe``: Sharpe of the un-tilted portfolio
        - ``overlay_sharpe``: Sharpe of the regime-tilted portfolio
        - ``base_max_dd``: Max drawdown without overlay
        - ``overlay_max_dd``: Max drawdown with overlay
        - ``base_cagr``: CAGR without overlay
        - ``overlay_cagr``: CAGR with overlay
        - ``regime_counts``: dict of regime -> day count
        - ``n_days``: number of trading days
    """
    slugs = sorted(base_returns.keys())
    if not slugs:
        return {
            "base_sharpe": 0.0,
            "overlay_sharpe": 0.0,
            "base_max_dd": 0.0,
            "overlay_max_dd": 0.0,
            "base_cagr": 0.0,
            "overlay_cagr": 0.0,
            "regime_counts": {},
            "n_days": 0,
        }

    n_days = len(regime_history)
    # Validate all return series have the same length
    for slug in slugs:
        if len(base_returns[slug]) != n_days:
            raise ValueError(
                f"Return series length mismatch: '{slug}' has "
                f"{len(base_returns[slug])} elements, expected {n_days}"
            )

    overlay = RegimeOverlay(
        strategy_families=strategy_families,
        tilt_magnitude=tilt_magnitude,
    )

    # Convert to numpy for performance
    returns_matrix = np.column_stack([np.array(base_returns[s]) for s in slugs])

    # Base weights vector (static)
    base_w = np.array([base_weights.get(s, 0.0) for s in slugs])
    base_w_sum = base_w.sum()
    if base_w_sum > 0:
        base_w = base_w / base_w_sum

    # Compute base portfolio returns (no overlay)
    base_port_returns = returns_matrix @ base_w

    # Compute overlay portfolio returns (regime-adjusted each day)
    overlay_port_returns = np.zeros(n_days)
    regime_counts: dict[str, int] = {
        REGIME_RISK_ON: 0,
        REGIME_RISK_OFF: 0,
        REGIME_TRANSITION: 0,
    }

    # Pre-compute adjusted weight vectors for each regime
    regime_weight_cache: dict[str, np.ndarray] = {}
    for regime in VALID_REGIMES:
        adj = overlay.adjust_weights(dict(base_weights), regime)
        regime_weight_cache[regime] = np.array([adj.get(s, 0.0) for s in slugs])

    for day_idx in range(n_days):
        regime = regime_history[day_idx]
        if regime not in VALID_REGIMES:
            regime = REGIME_TRANSITION
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        w = regime_weight_cache[regime]
        overlay_port_returns[day_idx] = float(returns_matrix[day_idx] @ w)

    trading_days_per_year = 252

    return {
        "base_sharpe": _sharpe(base_port_returns, trading_days_per_year),
        "overlay_sharpe": _sharpe(overlay_port_returns, trading_days_per_year),
        "base_max_dd": _max_drawdown(base_port_returns),
        "overlay_max_dd": _max_drawdown(overlay_port_returns),
        "base_cagr": _cagr(base_port_returns, trading_days_per_year),
        "overlay_cagr": _cagr(overlay_port_returns, trading_days_per_year),
        "regime_counts": regime_counts,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1.0.  Returns empty dict if total is zero."""
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def _sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio (excess return over risk-free assumed 0)."""
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(trading_days)


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a return series."""
    if len(returns) == 0:
        return 0.0
    nav = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    return float(np.max(dd))


def _cagr(returns: np.ndarray, trading_days: int = 252) -> float:
    """Compound annual growth rate from daily returns."""
    if len(returns) == 0:
        return 0.0
    nav = np.prod(1.0 + returns)
    if nav <= 0:
        return -1.0
    years = len(returns) / trading_days
    if years == 0:
        return 0.0
    return float(nav ** (1.0 / years) - 1.0)

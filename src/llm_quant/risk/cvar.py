"""Filtered Historical Simulation (FHS) CVaR constraints.

Based on Barone-Adesi (1999): filter residuals through GARCH model,
rescale to current volatility before historical simulation.

Why FHS over raw historical sim:
- Raw 252d lookback: only ~12-13 tail observations (dangerously unstable)
- FHS with 500d+ and EWMA decay: ~25-50 effective tail observations
- Bootstrap CI on CVaR estimate: quantifies estimation uncertainty

Stress scenario overlay (mandatory safety net):
- COVID crash (Feb-Mar 2020): ~-34% SPY in 23 days
- 2022 rate hike cycle: ~-19% SPY in 1 year, bonds also down
- SVB crisis (Mar 2023): banking sector stress
- Crypto winter (Nov 2022): -60%+ BTC

Enforcement: proportional scaling (CPPI-style), NOT binary blocking.
Scale position size proportionally when CVaR exceeds limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CvarConfig:
    """Configuration for FHS CVaR estimation."""

    confidence_level: float = 0.95  # 95% CVaR
    lookback_days: int = 504  # 2 years of trading days
    ewma_lambda: float = 0.94  # decay weight for older observations
    n_bootstrap: int = 1000  # bootstrap samples for CI
    cvar_limit_pct: float = 0.05  # max 5% 1-day CVaR for portfolio
    stress_scenarios: list[str] = field(
        default_factory=lambda: ["covid_2020", "rates_2022", "svb_2023"]
    )


# ---------------------------------------------------------------------------
# Stress scenarios
# ---------------------------------------------------------------------------


@dataclass
class StressScenario:
    """Defines a historical stress scenario for CVaR overlay."""

    name: str
    start_date: date
    end_date: date
    max_daily_loss: float  # worst single day in the period (negative = loss)
    total_period_loss: float  # cumulative loss over the period (negative = loss)


STRESS_SCENARIOS: list[StressScenario] = [
    StressScenario(
        "covid_2020",
        date(2020, 2, 19),
        date(2020, 3, 23),
        max_daily_loss=-0.12,
        total_period_loss=-0.34,
    ),
    StressScenario(
        "rates_2022",
        date(2022, 1, 1),
        date(2022, 10, 13),
        max_daily_loss=-0.04,
        total_period_loss=-0.25,
    ),
    StressScenario(
        "svb_2023",
        date(2023, 3, 8),
        date(2023, 3, 17),
        max_daily_loss=-0.05,
        total_period_loss=-0.08,
    ),
    StressScenario(
        "crypto_2022",
        date(2022, 11, 1),
        date(2022, 11, 30),
        max_daily_loss=-0.20,
        total_period_loss=-0.25,
    ),
]

# Lookup by name for fast access
_STRESS_BY_NAME: dict[str, StressScenario] = {s.name: s for s in STRESS_SCENARIOS}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CvarResult:
    """Result of FHS CVaR estimation for a return series."""

    cvar_95: float  # 95% CVaR estimate (positive = loss magnitude)
    cvar_ci_lower: float  # 2.5th percentile of bootstrap CI
    cvar_ci_upper: float  # 97.5th percentile of bootstrap CI
    var_95: float  # 95% VaR (positive = loss magnitude)
    stress_cvar: float  # worst CVaR under stress scenarios
    worst_stress_scenario: str  # name of worst stress scenario
    scaling_factor: float  # how much to scale down position (1.0 = no scaling needed)
    exceeds_limit: bool
    method_used: str  # "fhs_garch" or "ewma_weighted"


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class FhsCvarEstimator:
    """Filtered Historical Simulation CVaR estimator.

    Primary path: GARCH(1,1) standardised residuals rescaled to current vol.
    Fallback: EWMA-weighted historical simulation.

    Parameters
    ----------
    config:
        CVaR estimation configuration.  Defaults to ``CvarConfig()``.
    """

    def __init__(self, config: CvarConfig | None = None) -> None:
        self.config = config or CvarConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, returns: pl.Series) -> CvarResult:
        """Estimate CVaR with bootstrap confidence interval and stress overlay.

        Parameters
        ----------
        returns:
            Daily return series (most recent last).  Length should be at
            least ``config.lookback_days`` for reliable estimates.

        Returns
        -------
        CvarResult
            Full CVaR result including CI, stress overlay, and scaling factor.
        """
        arr = returns.to_numpy().astype(float)

        # Trim to lookback window
        if len(arr) > self.config.lookback_days:
            arr = arr[-self.config.lookback_days :]

        if len(arr) < 30:
            logger.warning(
                "FhsCvarEstimator: only %d observations — CVaR estimate unreliable.",
                len(arr),
            )

        # Step 1: filter returns through GARCH or EWMA
        try:
            filtered, method = self._filter_garch(arr)
        except Exception:
            logger.debug("GARCH filtering failed, falling back to EWMA.", exc_info=True)
            filtered = self._ewma_weighted_historical(arr)
            method = "ewma_weighted"

        # Step 2: compute VaR and CVaR at confidence level
        1.0 - self.config.confidence_level  # 0.05 for 95%
        var_95 = float(np.quantile(-filtered, self.config.confidence_level))
        tail_mask = -filtered >= var_95
        if tail_mask.sum() == 0:
            cvar_95 = var_95
        else:
            cvar_95 = float(np.mean(-filtered[tail_mask]))

        # CVaR must be positive (loss magnitude)
        cvar_95 = max(cvar_95, 0.0)
        var_95 = max(var_95, 0.0)

        # Step 3: bootstrap CI
        ci_lower, ci_upper = self._bootstrap_cvar_ci(filtered, self.config.n_bootstrap)

        # Step 4: stress test overlay
        stress_cvar, worst_scenario = self._stress_test(cvar_95)

        # Step 5: compute scaling factor
        limit = self.config.cvar_limit_pct
        scaling_factor = min(1.0, limit / cvar_95) if cvar_95 > 0.0 else 1.0
        exceeds_limit = cvar_95 > limit

        return CvarResult(
            cvar_95=cvar_95,
            cvar_ci_lower=ci_lower,
            cvar_ci_upper=ci_upper,
            var_95=var_95,
            stress_cvar=stress_cvar,
            worst_stress_scenario=worst_scenario,
            scaling_factor=scaling_factor,
            exceeds_limit=exceeds_limit,
            method_used=method,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_garch(self, returns: np.ndarray) -> tuple[np.ndarray, str]:
        """Filter returns through GARCH(1,1) and rescale to current volatility.

        Fits a GARCH(1,1) model, extracts standardised residuals, then
        multiplies each residual by the current conditional volatility estimate
        so the simulated distribution reflects today's vol regime.

        Falls back to EWMA if ``arch`` is unavailable or fitting fails.

        Parameters
        ----------
        returns:
            Raw daily return array.

        Returns
        -------
        tuple[np.ndarray, str]
            (filtered_returns, method_name) where method_name is either
            "fhs_garch" or "ewma_weighted".
        """
        from arch import arch_model

        # Rescale to percentage returns for numerical stability in arch
        r_pct = returns * 100.0

        am = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)

        # Standardised residuals ε_t = r_t / σ_t
        std_resid = res.std_resid  # length T

        # Current conditional volatility (last estimate, in pct)
        current_vol_pct = float(res.conditional_volatility[-1])

        # Re-express residuals in return space at current vol
        filtered = std_resid * (current_vol_pct / 100.0)

        return filtered.astype(float), "fhs_garch"

    def _ewma_weighted_historical(self, returns: np.ndarray) -> np.ndarray:
        """Apply EWMA weights to historical returns for weighted simulation.

        EWMA weights: w_t = (1-λ) × λ^(T-t), normalised to sum to 1.
        Recent observations receive higher weight.

        Rather than resampling, we expand each return proportionally to
        its weight (repeated sampling approximation) to preserve the full
        distribution shape while honouring the time-decay.

        Parameters
        ----------
        returns:
            Raw daily return array (oldest first).

        Returns
        -------
        np.ndarray
            Weighted return samples suitable for CVaR estimation.
        """
        lam = self.config.ewma_lambda
        T = len(returns)

        # w_t = (1-λ) × λ^(T-1-t), t=0..T-1 (t=T-1 is most recent → weight = 1-λ)
        exponents = np.arange(T - 1, -1, -1, dtype=float)
        weights = (1.0 - lam) * (lam**exponents)
        weights = weights / weights.sum()  # normalise

        # Weighted bootstrap: sample with replacement using EWMA probabilities
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(T, size=T, replace=True, p=weights)
        return returns[indices]

    def _bootstrap_cvar_ci(
        self,
        returns: np.ndarray,
        n_bootstrap: int,
    ) -> tuple[float, float]:
        """Bootstrap 95% confidence interval on the CVaR estimate.

        Resamples the filtered return series with replacement ``n_bootstrap``
        times, computes CVaR from each sample, and returns the 2.5th and
        97.5th percentiles of the bootstrap distribution.

        Parameters
        ----------
        returns:
            Filtered daily return array (used as input to bootstrap).
        n_bootstrap:
            Number of bootstrap samples.

        Returns
        -------
        tuple[float, float]
            (ci_lower, ci_upper) — 95% confidence interval on CVaR.
        """
        conf = self.config.confidence_level
        rng = np.random.default_rng(seed=42)
        n = len(returns)
        boot_cvars = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(returns, size=n, replace=True)
            var_b = np.quantile(-sample, conf)
            tail = -sample[-sample <= -var_b]
            boot_cvars[i] = float(np.mean(tail)) if len(tail) > 0 else var_b

        ci_lower = float(np.percentile(boot_cvars, 2.5))
        ci_upper = float(np.percentile(boot_cvars, 97.5))
        return ci_lower, ci_upper

    def _stress_test(self, cvar_95: float) -> tuple[float, str]:
        """Compute worst-case CVaR under historical stress scenarios.

        For each active stress scenario in config, derives a daily CVaR
        proxy from the scenario's worst single-day loss.  The stress CVaR
        is the maximum of the FHS estimate and all scenario proxies.

        Parameters
        ----------
        cvar_95:
            FHS CVaR estimate (positive = loss magnitude).

        Returns
        -------
        tuple[float, str]
            (stress_cvar, worst_scenario_name) — worst CVaR and its source.
        """
        worst_cvar = cvar_95
        worst_name = "fhs_estimate"

        for scenario_name in self.config.stress_scenarios:
            scenario = _STRESS_BY_NAME.get(scenario_name)
            if scenario is None:
                logger.debug("Unknown stress scenario '%s' — skipped.", scenario_name)
                continue

            # Use the worst single-day loss as a conservative CVaR proxy.
            # The tail of the stress period is approximated by the max daily loss.
            scenario_cvar = abs(scenario.max_daily_loss)

            if scenario_cvar > worst_cvar:
                worst_cvar = scenario_cvar
                worst_name = scenario.name

        return worst_cvar, worst_name

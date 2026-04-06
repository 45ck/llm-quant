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

Two estimator classes:
- ``SingleAssetFhsCvar``: original single-series estimator (used by risk limits).
- ``FhsCvarEstimator``: multi-asset portfolio-level FHS CVaR with named stress
  scenarios, bootstrap CI, and EWMA decay weighting (Barone-Adesi 1999).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Minimum observations for reliable GARCH fitting
_MIN_OBS_GARCH = 30


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
# Stress scenarios (legacy dataclass for SingleAssetFhsCvar)
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
# Default stress scenarios for multi-asset FhsCvarEstimator
# ---------------------------------------------------------------------------

# Per-asset single-day returns during stress events.  Negative = loss.
_DEFAULT_STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "covid_crash": {
        "SPY": -0.12,
        "QQQ": -0.13,
        "TLT": 0.05,
        "GLD": -0.04,
        "BTC-USD": -0.40,
        "LQD": -0.06,
        "HYG": -0.08,
        "XLE": -0.20,
    },
    "2022_rates": {
        "SPY": -0.04,
        "QQQ": -0.05,
        "TLT": -0.04,
        "GLD": -0.02,
        "BTC-USD": -0.10,
        "LQD": -0.03,
        "HYG": -0.03,
        "XLE": 0.02,
    },
    "crypto_winter": {
        "SPY": -0.02,
        "QQQ": -0.03,
        "TLT": 0.01,
        "GLD": 0.00,
        "BTC-USD": -0.25,
        "LQD": -0.01,
        "HYG": -0.02,
        "XLE": -0.01,
    },
    "svb_crisis": {
        "SPY": -0.05,
        "QQQ": -0.04,
        "TLT": 0.03,
        "GLD": 0.02,
        "BTC-USD": -0.08,
        "LQD": -0.03,
        "HYG": -0.04,
        "XLE": -0.06,
    },
}


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
# Standalone fallback function
# ---------------------------------------------------------------------------


def compute_historical_cvar(
    returns: list[float],
    confidence: float = 0.95,
) -> float:
    """Compute simple historical CVaR (Expected Shortfall) from a return series.

    This is a non-parametric fallback when GARCH fitting is not needed or not
    feasible.  No volatility filtering, no EWMA weighting — just the empirical
    tail mean of losses.

    Parameters
    ----------
    returns:
        Daily return series (decimal, e.g. -0.02 for a 2% loss).
    confidence:
        Confidence level (e.g. 0.95 for 95% CVaR).

    Returns
    -------
    float
        CVaR estimate as a positive number (loss magnitude).
        Returns 0.0 for empty or all-positive return series.
    """
    arr = np.asarray(returns, dtype=np.float64)

    # Remove NaNs
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return 0.0

    # VaR as quantile of losses (negate returns so losses are positive)
    losses = -arr
    var = float(np.quantile(losses, confidence))

    # CVaR = mean of losses exceeding VaR
    tail = losses[losses >= var]
    if len(tail) == 0:
        return max(var, 0.0)

    cvar = float(np.mean(tail))
    return max(cvar, 0.0)


# ---------------------------------------------------------------------------
# Single-asset estimator (legacy — used by risk limits)
# ---------------------------------------------------------------------------


class SingleAssetFhsCvar:
    """Single-asset Filtered Historical Simulation CVaR estimator.

    Primary path: GARCH(1,1) standardised residuals rescaled to current vol.
    Fallback: EWMA-weighted historical simulation.

    This is the original estimator used by ``risk.limits.check_cvar_limit()``.

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
                "SingleAssetFhsCvar: only %d observations — CVaR estimate unreliable.",
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
        """Filter returns through GARCH(1,1) and rescale to current volatility."""
        from arch import arch_model

        r_pct = returns * 100.0
        am = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)

        std_resid = res.std_resid
        current_vol_pct = float(res.conditional_volatility[-1])
        filtered = std_resid * (current_vol_pct / 100.0)

        return filtered.astype(float), "fhs_garch"

    def _ewma_weighted_historical(self, returns: np.ndarray) -> np.ndarray:
        """Apply EWMA weights to historical returns for weighted simulation."""
        lam = self.config.ewma_lambda
        T = len(returns)

        exponents = np.arange(T - 1, -1, -1, dtype=float)
        weights = (1.0 - lam) * (lam**exponents)
        weights = weights / weights.sum()

        rng = np.random.default_rng(seed=0)
        indices = rng.choice(T, size=T, replace=True, p=weights)
        return returns[indices]

    def _bootstrap_cvar_ci(
        self,
        returns: np.ndarray,
        n_bootstrap: int,
    ) -> tuple[float, float]:
        """Bootstrap 95% confidence interval on the CVaR estimate."""
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
        """Compute worst-case CVaR under historical stress scenarios."""
        worst_cvar = cvar_95
        worst_name = "fhs_estimate"

        for scenario_name in self.config.stress_scenarios:
            scenario = _STRESS_BY_NAME.get(scenario_name)
            if scenario is None:
                logger.debug("Unknown stress scenario '%s' — skipped.", scenario_name)
                continue

            scenario_cvar = abs(scenario.max_daily_loss)
            if scenario_cvar > worst_cvar:
                worst_cvar = scenario_cvar
                worst_name = scenario.name

        return worst_cvar, worst_name


# ---------------------------------------------------------------------------
# Multi-asset portfolio FHS CVaR estimator (Barone-Adesi 1999)
# ---------------------------------------------------------------------------


class FhsCvarEstimator:
    """Multi-asset Filtered Historical Simulation CVaR estimator.

    Implements the Barone-Adesi (1999) procedure for portfolio-level VaR/CVaR:

    1. Fit GARCH(1,1) to each asset's returns using the ``arch`` package.
    2. Extract standardized residuals (innovations).
    3. Apply EWMA decay weighting (lambda=0.94) to residuals so recent
       observations receive higher weight.
    4. Rescale each asset's residuals to its current conditional volatility
       forecast.
    5. Compute portfolio-level returns using position weights.
    6. Estimate VaR and CVaR from the rescaled portfolio distribution.

    This is a PROPORTIONAL scaling signal (CPPI-style), not a binary blocker.

    Named stress scenarios can be added to augment the tail distribution.
    Four default scenarios are pre-loaded: COVID crash, 2022 rates, crypto
    winter, and SVB crisis.

    Parameters
    ----------
    ewma_lambda:
        Decay factor for EWMA weighting of residuals (default 0.94).
    """

    def __init__(self, ewma_lambda: float = 0.94) -> None:
        self.ewma_lambda = ewma_lambda

        # State populated by fit()
        self._fitted: bool = False
        self._portfolio_returns: np.ndarray | None = None
        self._method: str = ""
        self._assets: list[str] = []
        self._weights: dict[str, float] = {}

        # Named stress scenarios: name -> per-asset returns dict
        self._stress_scenarios: dict[str, dict[str, float]] = dict(
            _DEFAULT_STRESS_SCENARIOS
        )

    # ------------------------------------------------------------------
    # Public API: fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        returns_dict: dict[str, list[float]],
        weights: dict[str, float],
        lookback: int = 500,
    ) -> None:
        """Fit GARCH(1,1) to each asset and build FHS portfolio distribution.

        Parameters
        ----------
        returns_dict:
            Mapping of asset name to daily return series (decimal, oldest
            first).  All series must have the same length.
        weights:
            Mapping of asset name to portfolio weight (e.g. {"SPY": 0.6,
            "TLT": 0.4}).  Weights should sum to ~1.0 but this is not
            enforced (allows short positions).
        lookback:
            Maximum number of recent observations to use.  Default 500.

        Raises
        ------
        ValueError
            If ``returns_dict`` is empty, weights reference unknown assets,
            or series lengths are inconsistent.
        """
        if not returns_dict:
            msg = "returns_dict must not be empty."
            raise ValueError(msg)

        # Validate that all weight keys exist in returns_dict
        missing = set(weights.keys()) - set(returns_dict.keys())
        if missing:
            msg = f"Weight keys not found in returns_dict: {sorted(missing)}"
            raise ValueError(msg)

        # If weights reference a subset, only use those assets
        assets = sorted(weights.keys())
        if not assets:
            msg = "weights must not be empty."
            raise ValueError(msg)

        self._assets = assets
        self._weights = dict(weights)

        # Build aligned return matrix
        raw_lengths = [len(returns_dict[a]) for a in assets]
        min_len = min(raw_lengths)
        n_use = min(min_len, lookback)

        if n_use == 0:
            msg = "All return series are empty."
            raise ValueError(msg)

        returns_matrix = np.column_stack(
            [np.asarray(returns_dict[a], dtype=np.float64)[-n_use:] for a in assets]
        )

        # Handle NaN values — replace with 0.0 and warn
        nan_mask = np.isnan(returns_matrix)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logger.warning(
                "FhsCvarEstimator.fit: %d NaN values replaced with 0.0.", n_nan
            )
            returns_matrix = np.where(nan_mask, 0.0, returns_matrix)

        # Weight vector aligned with assets
        w = np.array([weights[a] for a in assets], dtype=np.float64)

        # FHS procedure: fit GARCH per asset, extract filtered returns
        filtered_matrix, method = self._fit_garch_all(returns_matrix, assets)

        # Apply EWMA decay weighting to residuals
        filtered_matrix = self._apply_ewma_weights(filtered_matrix)

        # Compute portfolio-level returns from weighted sum
        portfolio_returns = filtered_matrix @ w
        self._portfolio_returns = portfolio_returns
        self._method = method
        self._fitted = True

    # ------------------------------------------------------------------
    # Public API: VaR / CVaR
    # ------------------------------------------------------------------

    def compute_var(self, confidence: float = 0.95) -> float:
        """Compute Value at Risk from the FHS distribution.

        Parameters
        ----------
        confidence:
            Confidence level (e.g. 0.95 for 95% VaR).

        Returns
        -------
        float
            VaR as a positive number representing loss magnitude.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted("compute_var")
        assert self._portfolio_returns is not None

        losses = -self._portfolio_returns
        var = float(np.quantile(losses, confidence))
        return max(var, 0.0)

    def compute_cvar(self, confidence: float = 0.95) -> float:
        """Compute Conditional VaR (Expected Shortfall) from FHS distribution.

        CVaR is the expected loss given that the loss exceeds VaR — i.e. the
        mean of the tail beyond VaR.  CVaR >= VaR by construction.

        Parameters
        ----------
        confidence:
            Confidence level (e.g. 0.95 for 95% CVaR).

        Returns
        -------
        float
            CVaR as a positive number representing expected tail loss.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted("compute_cvar")
        assert self._portfolio_returns is not None

        losses = -self._portfolio_returns
        var = float(np.quantile(losses, confidence))
        tail = losses[losses >= var]

        if len(tail) == 0:
            return max(var, 0.0)

        cvar = float(np.mean(tail))
        return max(cvar, 0.0)

    def compute_bootstrap_ci(
        self,
        confidence: float = 0.95,
        n_bootstraps: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval on the CVaR estimate.

        Resamples the FHS portfolio return distribution ``n_bootstraps``
        times and returns the 2.5th and 97.5th percentiles of the CVaR
        bootstrap distribution.

        Parameters
        ----------
        confidence:
            Confidence level for CVaR computation (e.g. 0.95).
        n_bootstraps:
            Number of bootstrap resamples.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        tuple[float, float]
            (ci_lower, ci_upper) — 95% bootstrap confidence interval on CVaR.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted("compute_bootstrap_ci")
        assert self._portfolio_returns is not None

        rng = np.random.default_rng(seed=seed)
        n = len(self._portfolio_returns)
        boot_cvars = np.empty(n_bootstraps)

        for i in range(n_bootstraps):
            sample = rng.choice(self._portfolio_returns, size=n, replace=True)
            losses = -sample
            var_b = float(np.quantile(losses, confidence))
            tail = losses[losses >= var_b]
            boot_cvars[i] = float(np.mean(tail)) if len(tail) > 0 else var_b

        ci_lower = float(np.percentile(boot_cvars, 2.5))
        ci_upper = float(np.percentile(boot_cvars, 97.5))
        return ci_lower, ci_upper

    # ------------------------------------------------------------------
    # Public API: stress scenarios
    # ------------------------------------------------------------------

    def add_stress_scenario(self, name: str, returns: dict[str, float]) -> None:
        """Add a named stress scenario.

        The scenario specifies per-asset single-day returns during a stress
        event.  These are injected into the FHS distribution when computing
        stressed CVaR.

        Parameters
        ----------
        name:
            Unique scenario identifier (e.g. "covid_crash", "2022_rates").
        returns:
            Mapping of asset name to single-day return during the stress
            event (decimal, e.g. -0.12 for a 12% loss).
        """
        self._stress_scenarios[name] = dict(returns)

    def compute_stressed_cvar(self, confidence: float = 0.95) -> float:
        """Compute CVaR with stress scenarios injected into the distribution.

        For each named stress scenario, computes the portfolio return using
        current weights and the scenario's per-asset returns.  These stress
        returns are appended to the FHS distribution before computing CVaR,
        ensuring the tail captures extreme but plausible events.

        Parameters
        ----------
        confidence:
            Confidence level (e.g. 0.95 for 95% CVaR).

        Returns
        -------
        float
            Stressed CVaR as a positive number representing expected tail loss.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        self._check_fitted("compute_stressed_cvar")
        assert self._portfolio_returns is not None

        # Compute portfolio return for each stress scenario
        stress_returns: list[float] = []
        for scenario_rets in self._stress_scenarios.values():
            port_ret = 0.0
            for asset, weight in self._weights.items():
                # If scenario doesn't specify this asset, assume 0 return
                asset_ret = scenario_rets.get(asset, 0.0)
                port_ret += weight * asset_ret
            stress_returns.append(port_ret)

        # Augment the FHS distribution with stress returns
        if stress_returns:
            augmented = np.concatenate(
                [self._portfolio_returns, np.array(stress_returns, dtype=np.float64)]
            )
        else:
            augmented = self._portfolio_returns

        # Compute CVaR on augmented distribution
        losses = -augmented
        var = float(np.quantile(losses, confidence))
        tail = losses[losses >= var]

        if len(tail) == 0:
            return max(var, 0.0)

        cvar = float(np.mean(tail))
        return max(cvar, 0.0)

    # ------------------------------------------------------------------
    # Internal: GARCH fitting
    # ------------------------------------------------------------------

    def _fit_garch_all(
        self,
        returns_matrix: np.ndarray,
        assets: list[str],
    ) -> tuple[np.ndarray, str]:
        """Fit GARCH(1,1) to each asset and return filtered returns matrix.

        For each asset column:
        1. Fit GARCH(1,1) to extract standardized residuals.
        2. Rescale residuals to current conditional volatility.

        If GARCH fails for any asset, falls back to EWMA volatility
        rescaling for that asset.

        Parameters
        ----------
        returns_matrix:
            T x N matrix of raw daily returns.
        assets:
            Asset names aligned with columns.

        Returns
        -------
        tuple[np.ndarray, str]
            (filtered_matrix, method) where filtered_matrix is T x N.
        """
        n_obs, n_assets = returns_matrix.shape
        filtered = np.zeros_like(returns_matrix)
        garch_succeeded = 0
        garch_failed = 0

        for i in range(n_assets):
            series = returns_matrix[:, i]

            # Check for zero variance
            if np.std(series) < 1e-12:
                filtered[:, i] = 0.0
                continue

            if n_obs >= _MIN_OBS_GARCH:
                try:
                    filt_col = self._fit_single_garch(series)
                    filtered[:, i] = filt_col
                    garch_succeeded += 1
                    continue
                except Exception:
                    logger.debug(
                        "GARCH fit failed for %s — using EWMA fallback.",
                        assets[i],
                    )
                    garch_failed += 1

            # Fallback: rescale by EWMA vol estimate
            filtered[:, i] = self._ewma_vol_rescale(series)

        if garch_succeeded > 0 and garch_failed == 0:
            method = "fhs_garch"
        elif garch_succeeded > 0:
            method = "fhs_garch_partial"
        else:
            method = "ewma_fallback"

        return filtered, method

    @staticmethod
    def _fit_single_garch(series: np.ndarray) -> np.ndarray:
        """Fit GARCH(1,1) to a single asset and return filtered returns.

        Standardized residuals are rescaled to the current conditional
        volatility so the distribution reflects today's vol regime.

        Parameters
        ----------
        series:
            Daily return array for one asset (decimal).

        Returns
        -------
        np.ndarray
            Filtered returns: std_resid * current_vol.
        """
        from arch import arch_model

        r_pct = series * 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)

        std_resid = res.std_resid
        current_vol_pct = float(res.conditional_volatility[-1])

        # Rescale to decimal return space
        filtered = std_resid * (current_vol_pct / 100.0)
        return filtered.astype(np.float64)

    def _ewma_vol_rescale(self, series: np.ndarray) -> np.ndarray:
        """Rescale returns by ratio of current EWMA vol to historical vol.

        Fallback when GARCH fitting is not available or fails.

        Parameters
        ----------
        series:
            Daily return array for one asset (decimal).

        Returns
        -------
        np.ndarray
            Rescaled returns reflecting current vol regime.
        """
        lam = self.ewma_lambda
        n = len(series)

        # EWMA variance
        var_ewma = series[0] ** 2
        for t in range(1, n):
            var_ewma = lam * var_ewma + (1.0 - lam) * series[t - 1] ** 2

        current_vol = np.sqrt(max(var_ewma, 1e-16))
        hist_vol = max(float(np.std(series)), 1e-16)

        # Rescale: multiply by ratio of current vol to historical vol
        return series * (current_vol / hist_vol)

    def _apply_ewma_weights(self, filtered_matrix: np.ndarray) -> np.ndarray:
        """Apply EWMA decay weighting to filtered returns matrix.

        More recent observations receive higher weight via importance
        sampling: each row is multiplied by its EWMA weight (normalized),
        then resampled to build a weighted distribution.

        This makes CVaR estimates more reactive to recent market conditions
        rather than giving equal weight to all historical observations.

        Parameters
        ----------
        filtered_matrix:
            T x N matrix of filtered returns.

        Returns
        -------
        np.ndarray
            T x N matrix of EWMA-weighted resampled returns.
        """
        lam = self.ewma_lambda
        n_obs = filtered_matrix.shape[0]

        if n_obs <= 1:
            return filtered_matrix

        # EWMA weights: w_t = (1-lambda) * lambda^(T-1-t), most recent = highest
        exponents = np.arange(n_obs - 1, -1, -1, dtype=np.float64)
        weights = (1.0 - lam) * (lam**exponents)
        weights = weights / weights.sum()

        # Weighted resampling preserving row structure (correlated assets)
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(n_obs, size=n_obs, replace=True, p=weights)
        return filtered_matrix[indices]

    def _check_fitted(self, method_name: str) -> None:
        """Raise RuntimeError if fit() has not been called."""
        if not self._fitted:
            msg = f"Must call fit() before {method_name}()."
            raise RuntimeError(msg)

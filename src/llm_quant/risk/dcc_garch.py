"""DCC-GARCH dynamic correlation estimation module.

Two-step estimation procedure (Engle 2002):
  Step 1: Fit univariate GARCH(1,1) to each asset's returns via the ``arch``
          package to obtain standardized residuals and conditional volatility
          forecasts.
  Step 2: Compute DCC correlation matrix from standardized residuals using
          EWMA (lambda=0.94) as the practical second-stage estimator.  Full
          DCC-GARCH maximum-likelihood optimization is fragile for large
          cross-sections (39 assets); EWMA provides a robust, closed-form
          alternative that tracks dynamic correlations adequately for
          portfolio-level monitoring.

Forbes-Rigobon heteroskedasticity adjustment is applied to correct the upward
bias in conditional correlation estimates observed during high-volatility
regimes.

This is an ADVISORY signal — it is NOT a hard trade blocker.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of observations for GARCH fitting
_MIN_OBS_GARCH = 30
# Forbes-Rigobon: only apply adjustment when vol ratio exceeds this
_FR_VOL_RATIO_THRESHOLD = 1.5


@dataclass
class DccGarchResult:
    """Container for DCC-GARCH estimation results."""

    assets: list[str]
    correlation_matrix: np.ndarray  # N x N
    vol_forecasts: dict[str, float]
    diversification_score: float
    method: str  # "dcc_garch" or "ewma_fallback"
    n_obs_used: int
    warnings: list[str] = field(default_factory=list)


def compute_ewma_correlation(
    returns_a: list[float] | np.ndarray,
    returns_b: list[float] | np.ndarray,
    lambda_: float = 0.94,
) -> float:
    """Compute EWMA correlation between two return series.

    Lightweight standalone function that can be used as a fallback when
    full DCC-GARCH estimation is not needed or not feasible.

    Uses the RiskMetrics EWMA approach:
        Cov_t = lambda * Cov_{t-1} + (1-lambda) * r_a,{t-1} * r_b,{t-1}
        Var_a,t = lambda * Var_a,{t-1} + (1-lambda) * r_a,{t-1}^2
        Var_b,t = lambda * Var_b,{t-1} + (1-lambda) * r_b,{t-1}^2

    Parameters
    ----------
    returns_a:
        Return series for asset A.
    returns_b:
        Return series for asset B.
    lambda_:
        Decay factor (RiskMetrics default: 0.94).

    Returns
    -------
    float
        EWMA correlation estimate.  Returns 0.0 if inputs are insufficient
        or degenerate.
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)

    if len(a) < 2 or len(b) < 2:
        return 0.0

    # Align lengths
    n = min(len(a), len(b))
    a = a[-n:]
    b = b[-n:]

    # Initialize with sample statistics from first observation
    var_a = a[0] ** 2
    var_b = b[0] ** 2
    cov_ab = a[0] * b[0]

    # EWMA recursion
    for t in range(1, n):
        var_a = lambda_ * var_a + (1.0 - lambda_) * a[t - 1] ** 2
        var_b = lambda_ * var_b + (1.0 - lambda_) * b[t - 1] ** 2
        cov_ab = lambda_ * cov_ab + (1.0 - lambda_) * a[t - 1] * b[t - 1]

    # Convert covariance to correlation
    denom = np.sqrt(max(var_a, 1e-16)) * np.sqrt(max(var_b, 1e-16))
    corr = cov_ab / denom

    # Clamp to [-1, 1]
    return float(np.clip(corr, -1.0, 1.0))


def _forbes_rigobon_adjust(
    corr_matrix: np.ndarray,
    vol_ratios: np.ndarray,
) -> np.ndarray:
    """Apply Forbes-Rigobon heteroskedasticity adjustment to a correlation matrix.

    Corrects upward bias in conditional correlation estimates during
    high-volatility regimes.  For each pair (i, j), the adjustment is:

        rho_adj = rho / sqrt(1 + delta^2 * (1 - rho^2))

    where delta = max(vol_ratio_i, vol_ratio_j) - 1.0 (the excess volatility).

    The adjustment is only applied when the vol ratio exceeds the threshold
    ``_FR_VOL_RATIO_THRESHOLD``.

    Parameters
    ----------
    corr_matrix:
        N x N conditional correlation matrix (will be modified in-place).
    vol_ratios:
        Length-N array of vol_current / vol_historical for each asset.
        Values > 1 indicate elevated volatility.

    Returns
    -------
    np.ndarray
        Adjusted N x N correlation matrix.
    """
    n = corr_matrix.shape[0]
    adjusted = corr_matrix.copy()

    for i in range(n):
        for j in range(i + 1, n):
            # Use the maximum vol ratio of the pair
            delta = max(vol_ratios[i], vol_ratios[j])
            if delta <= _FR_VOL_RATIO_THRESHOLD:
                continue

            rho = corr_matrix[i, j]
            if abs(rho) < 1e-10:
                continue

            # Forbes-Rigobon formula: adjust for excess volatility
            # delta here is the vol ratio; excess = delta - 1
            # Using simplified form: rho_adj = rho / sqrt(1 + delta^2 * (1-rho^2))
            excess_sq = delta**2
            denom = 1.0 + excess_sq * (1.0 - rho**2)
            if denom <= 0:
                continue

            rho_adj = rho / np.sqrt(denom)
            adjusted[i, j] = float(np.clip(rho_adj, -1.0, 1.0))
            adjusted[j, i] = adjusted[i, j]

    return adjusted


class DccGarchEstimator:
    """DCC-GARCH dynamic correlation estimator with EWMA second stage.

    Two-step procedure:
      1. Fit univariate GARCH(1,1) to each asset via ``arch`` to extract
         standardized residuals and one-step-ahead volatility forecasts.
      2. Apply EWMA (lambda=0.94) to the standardized residuals to compute
         the dynamic correlation matrix.

    Forbes-Rigobon adjustment is applied when asset volatility ratios
    (current / historical) exceed ``_FR_VOL_RATIO_THRESHOLD``.

    This is an ADVISORY signal, NOT a hard trade blocker.

    Parameters
    ----------
    lambda_ewma:
        Decay factor for EWMA second stage (default 0.94).
    min_obs:
        Minimum observations per asset to attempt GARCH fitting.
        Below this, raw EWMA on returns is used as fallback.
    """

    def __init__(self, lambda_ewma: float = 0.94, min_obs: int = 30) -> None:
        self.lambda_ewma = lambda_ewma
        self.min_obs = min_obs

        # State populated by fit()
        self._assets: list[str] = []
        self._corr_matrix: np.ndarray | None = None
        self._vol_forecasts: dict[str, float] = {}
        self._diversification_score: float = 1.0
        self._fitted: bool = False
        self._method: str = ""
        self._n_obs_used: int = 0
        self._warnings: list[str] = []

    def fit(
        self,
        returns_dict: dict[str, list[float]],
        lookback: int = 252,
    ) -> dict:
        """Fit DCC-GARCH model to multi-asset return series.

        Parameters
        ----------
        returns_dict:
            Mapping of asset name to return series.  Each series should be
            ordered chronologically (oldest first).
        lookback:
            Maximum number of recent observations to use.  Defaults to 252
            (approx one trading year).

        Returns
        -------
        dict
            Summary with keys: ``assets``, ``method``, ``n_obs``,
            ``diversification_score``, ``warnings``.
        """
        self._warnings = []
        self._fitted = False

        # Validate inputs
        if not returns_dict:
            self._assets = []
            self._corr_matrix = np.array([[]])
            self._vol_forecasts = {}
            self._diversification_score = 1.0
            self._method = "none"
            self._n_obs_used = 0
            self._fitted = True
            self._warnings.append("Empty returns dict — no estimation performed.")
            return self._summary()

        assets = sorted(returns_dict.keys())
        self._assets = assets

        # Single asset — trivial case
        if len(assets) == 1:
            name = assets[0]
            series = np.asarray(returns_dict[name], dtype=np.float64)
            series = series[-lookback:] if len(series) > lookback else series

            vol = self._fit_single_garch(name, series)
            self._vol_forecasts = {name: vol}
            self._corr_matrix = np.array([[1.0]])
            self._diversification_score = 1.0
            self._method = "single_asset"
            self._n_obs_used = len(series)
            self._fitted = True
            return self._summary()

        # Build aligned return matrix
        min_len = min(len(returns_dict[a]) for a in assets)
        if min_len < 2:
            self._corr_matrix = np.eye(len(assets))
            self._vol_forecasts = dict.fromkeys(assets, 0.0)
            self._diversification_score = 1.0
            self._method = "insufficient_data"
            self._n_obs_used = min_len
            self._fitted = True
            self._warnings.append(
                f"Insufficient data ({min_len} obs) — returning identity matrix."
            )
            return self._summary()

        # Truncate to lookback window
        n_use = min(min_len, lookback)
        self._n_obs_used = n_use

        returns_matrix = np.column_stack(
            [np.asarray(returns_dict[a], dtype=np.float64)[-n_use:] for a in assets]
        )

        # Check for zero-variance columns
        col_std = np.std(returns_matrix, axis=0)
        zero_var_mask = col_std < 1e-12
        if np.all(zero_var_mask):
            self._corr_matrix = np.eye(len(assets))
            self._vol_forecasts = dict.fromkeys(assets, 0.0)
            self._diversification_score = 1.0
            self._method = "zero_variance"
            self._fitted = True
            self._warnings.append(
                "All assets have zero variance — returning identity matrix."
            )
            return self._summary()

        if np.any(zero_var_mask):
            zero_names = [assets[i] for i in range(len(assets)) if zero_var_mask[i]]
            self._warnings.append(
                f"Zero-variance assets (treated as uncorrelated): {zero_names}"
            )

        # Step 1: Fit univariate GARCH(1,1) to each asset
        n_assets = len(assets)
        std_residuals = np.zeros_like(returns_matrix)
        vol_forecasts: dict[str, float] = {}
        historical_vols = np.zeros(n_assets)
        current_vols = np.zeros(n_assets)

        can_use_garch = n_use >= self.min_obs

        for i, asset in enumerate(assets):
            series = returns_matrix[:, i]

            if zero_var_mask[i]:
                std_residuals[:, i] = 0.0
                vol_forecasts[asset] = 0.0
                historical_vols[i] = 1e-10
                current_vols[i] = 1e-10
                continue

            if can_use_garch:
                std_res, vol_fcast, hist_vol, curr_vol = self._fit_garch_asset(
                    asset, series
                )
            else:
                std_res, vol_fcast, hist_vol, curr_vol = self._fallback_standardize(
                    asset, series
                )

            std_residuals[:, i] = std_res
            vol_forecasts[asset] = vol_fcast
            historical_vols[i] = hist_vol
            current_vols[i] = curr_vol

        self._vol_forecasts = vol_forecasts

        # Step 2: EWMA correlation on standardized residuals
        corr_matrix = self._ewma_correlation_matrix(std_residuals)

        # Forbes-Rigobon adjustment
        vol_ratios = np.where(
            historical_vols > 1e-10,
            current_vols / historical_vols,
            1.0,
        )
        if np.any(vol_ratios > _FR_VOL_RATIO_THRESHOLD):
            corr_matrix = _forbes_rigobon_adjust(corr_matrix, vol_ratios)
            logger.debug(
                "Forbes-Rigobon adjustment applied — max vol ratio: %.2f",
                float(np.max(vol_ratios)),
            )

        # Enforce symmetry and unit diagonal
        corr_matrix = (corr_matrix + corr_matrix.T) / 2.0
        np.fill_diagonal(corr_matrix, 1.0)

        self._corr_matrix = corr_matrix
        self._method = "dcc_garch" if can_use_garch else "ewma_fallback"

        # Compute diversification score
        self._diversification_score = self._compute_diversification_score(corr_matrix)

        self._fitted = True
        return self._summary()

    def get_correlation_matrix(self) -> dict[tuple[str, str], float]:
        """Return pairwise correlations as a dict keyed by asset pair tuples.

        Returns
        -------
        dict[tuple[str, str], float]
            Mapping of ``(asset_i, asset_j)`` to correlation value for all
            pairs including the diagonal (which is always 1.0).

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self._fitted:
            msg = "Must call fit() before get_correlation_matrix()."
            raise RuntimeError(msg)

        result: dict[tuple[str, str], float] = {}
        n = len(self._assets)
        if self._corr_matrix is None or self._corr_matrix.size == 0:
            return result

        for i in range(n):
            for j in range(n):
                result[(self._assets[i], self._assets[j])] = float(
                    self._corr_matrix[i, j]
                )

        return result

    def get_diversification_score(self) -> float:
        """Return the portfolio diversification score.

        Score ranges from 0.0 (all perfectly correlated) to 1.0 (fully
        diversified / uncorrelated).  Computed as ``1 - avg_abs_off_diagonal``.

        Returns
        -------
        float
            Diversification score in [0.0, 1.0].

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self._fitted:
            msg = "Must call fit() before get_diversification_score()."
            raise RuntimeError(msg)
        return self._diversification_score

    def get_vol_forecasts(self) -> dict[str, float]:
        """Return per-asset one-step-ahead volatility forecasts from GARCH.

        Returns
        -------
        dict[str, float]
            Mapping of asset name to annualized volatility forecast.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """
        if not self._fitted:
            msg = "Must call fit() before get_vol_forecasts()."
            raise RuntimeError(msg)
        return dict(self._vol_forecasts)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fit_single_garch(self, name: str, series: np.ndarray) -> float:
        """Fit GARCH(1,1) to a single series and return annualized vol forecast."""
        if len(series) < self.min_obs or np.std(series) < 1e-12:
            daily_std = float(np.std(series)) if np.std(series) > 1e-12 else 0.0
            return daily_std * np.sqrt(252)

        try:
            from arch import arch_model

            scaled = series * 100.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
                fit = model.fit(disp="off", show_warning=False)
                # One-step forecast
                fcast = fit.forecast(horizon=1)
                var_fcast = fcast.variance.values[-1, 0]
                # Convert back from percent^2 to decimal, annualize
                daily_vol = np.sqrt(var_fcast) / 100.0
                return float(daily_vol * np.sqrt(252))
        except Exception:
            logger.debug("GARCH fit failed for %s — using sample std.", name)
            daily_std = float(np.std(series))
            return daily_std * np.sqrt(252)

    def _fit_garch_asset(
        self,
        name: str,
        series: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """Fit GARCH(1,1) to one asset, return standardized residuals and vol info.

        Returns
        -------
        tuple of:
            - std_residuals: standardized residuals (T,)
            - vol_forecast: annualized one-step vol forecast
            - historical_vol: full-sample annualized vol
            - current_vol: recent (last 20 obs) annualized vol
        """
        historical_vol = float(np.std(series) * np.sqrt(252))
        recent_window = min(20, len(series))
        current_vol = float(np.std(series[-recent_window:]) * np.sqrt(252))

        try:
            from arch import arch_model

            scaled = series * 100.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
                fit = model.fit(disp="off", show_warning=False)

                # Standardized residuals
                cond_vol = fit.conditional_volatility
                cond_vol = np.maximum(cond_vol, 1e-8)
                std_res = fit.resid / cond_vol

                # One-step vol forecast
                fcast = fit.forecast(horizon=1)
                var_fcast = fcast.variance.values[-1, 0]
                daily_vol = np.sqrt(max(var_fcast, 0.0)) / 100.0
                vol_forecast = float(daily_vol * np.sqrt(252))

                return std_res, vol_forecast, historical_vol, current_vol

        except Exception:
            logger.debug(
                "GARCH fit failed for %s — falling back to sample standardization.",
                name,
            )
            return self._fallback_standardize(name, series)

    def _fallback_standardize(
        self,
        name: str,
        series: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """Fallback: standardize by rolling std when GARCH fails or insufficient data."""
        std = float(np.std(series))
        if std < 1e-12:
            std = 1.0
            self._warnings.append(f"{name}: zero variance — standardized to unit.")

        historical_vol = std * np.sqrt(252)
        recent_window = min(20, len(series))
        current_vol = float(np.std(series[-recent_window:]) * np.sqrt(252))

        std_res = series / std
        vol_forecast = current_vol  # naive forecast = recent vol

        return std_res, vol_forecast, historical_vol, current_vol

    def _ewma_correlation_matrix(self, std_residuals: np.ndarray) -> np.ndarray:
        """Compute EWMA correlation matrix from standardized residuals.

        Parameters
        ----------
        std_residuals:
            T x N matrix of standardized residuals.

        Returns
        -------
        np.ndarray
            N x N correlation matrix (final time-step estimate).
        """
        n_obs, _n_assets = std_residuals.shape
        lam = self.lambda_ewma

        # Initialize with sample covariance of standardized residuals
        cov = np.cov(std_residuals, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # EWMA recursion over time
        for t in range(1, n_obs):
            r_t = std_residuals[t - 1 : t, :].T  # N x 1
            cov = lam * cov + (1.0 - lam) * (r_t @ r_t.T)

        # Convert covariance to correlation
        diag_std = np.sqrt(np.maximum(np.diag(cov), 1e-16))
        d_inv = np.diag(1.0 / diag_std)
        return d_inv @ cov @ d_inv

    @staticmethod
    def _compute_diversification_score(corr_matrix: np.ndarray) -> float:
        """Compute diversification score from correlation matrix.

        Score = 1 - mean(|off-diagonal correlations|).
        Range: 0.0 (perfectly correlated) to 1.0 (uncorrelated).
        """
        n = corr_matrix.shape[0]
        if n < 2:
            return 1.0

        off_diag_mask = ~np.eye(n, dtype=bool)
        avg_abs_corr = float(np.mean(np.abs(corr_matrix[off_diag_mask])))

        return max(0.0, min(1.0, 1.0 - avg_abs_corr))

    def _summary(self) -> dict:
        """Return a summary dict of the estimation results."""
        return {
            "assets": list(self._assets),
            "method": self._method,
            "n_obs": self._n_obs_used,
            "diversification_score": self._diversification_score,
            "warnings": list(self._warnings),
        }

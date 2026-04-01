"""DCC-GARCH dynamic correlation estimation.

Primary: DCC-GARCH (Engle 2002) via arch package.
Fallback: EWMA (lambda=0.94) when arch unavailable or insufficient data.

Per issue spec: correlation scoring is ADVISORY only — not a hard trade blocker.
The blocking mechanism for novel correlation patterns is untested.

Forbes-Rigobon heteroskedasticity adjustment applied when estimating correlations
from high-vol regimes (adjust by volatility ratio).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Warning thresholds
_WARN_AVG_CORR = 0.70
_CRITICAL_AVG_CORR = 0.85

# DCC scalar parameters (Engle 2002 typical starting values)
_DCC_A = 0.05
_DCC_B = 0.93

# Forbes-Rigobon: apply adjustment when vol ratio exceeds this
_FR_VOL_RATIO_THRESHOLD = 1.5


class CorrelationMethod(Enum):
    DCC_GARCH = "dcc_garch"
    EWMA = "ewma"
    ROLLING = "rolling"


@dataclass
class CorrelationResult:
    """Result of a dynamic correlation estimation."""

    method_used: CorrelationMethod
    correlation_matrix: np.ndarray  # N×N current correlation estimate
    avg_pairwise_corr: float  # mean of off-diagonal elements
    max_pairwise_corr: float  # max off-diagonal
    diversification_score: float  # 1 - avg_pairwise_corr (higher = more diversified)
    warning: str | None  # advisory message if high correlation detected


class DccGarchEstimator:
    """DCC-GARCH dynamic correlation estimator with EWMA fallback.

    Uses the scalar DCC model (Engle 2002):
        Q_t = (1-a-b)*Q_bar + a*e_{t-1}e_{t-1}' + b*Q_{t-1}
        R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

    Falls back to EWMA when arch is unavailable or insufficient data exists.

    Parameters
    ----------
    lambda_ewma:
        Decay factor for EWMA fallback (RiskMetrics default: 0.94).
    min_obs:
        Minimum observations required to attempt DCC-GARCH.
        Below this threshold, EWMA fallback is used.
    """

    def __init__(self, lambda_ewma: float = 0.94, min_obs: int = 60) -> None:
        self.lambda_ewma = lambda_ewma
        self.min_obs = min_obs

    def estimate(self, returns_df: pl.DataFrame) -> CorrelationResult:
        """Estimate the current correlation matrix from a returns DataFrame.

        Parameters
        ----------
        returns_df:
            Polars DataFrame where each column is a return series for one asset.
            Rows are observations (most recent last), values are returns (not prices).

        Returns
        -------
        CorrelationResult
            Contains the current correlation matrix and summary statistics.
        """
        if returns_df.is_empty() or returns_df.width < 2:
            n = returns_df.width if not returns_df.is_empty() else 1
            identity = np.eye(n)
            return CorrelationResult(
                method_used=CorrelationMethod.EWMA,
                correlation_matrix=identity,
                avg_pairwise_corr=0.0,
                max_pairwise_corr=0.0,
                diversification_score=1.0,
                warning="Insufficient columns for pairwise correlation (need >= 2).",
            )

        returns = returns_df.to_numpy().astype(float)
        n_obs, n_assets = returns.shape

        # Attempt DCC-GARCH if sufficient data
        method = CorrelationMethod.EWMA
        corr_matrix: np.ndarray | None = None

        if n_obs >= self.min_obs:
            try:
                corr_matrix = self._estimate_dcc(returns)
                method = CorrelationMethod.DCC_GARCH
                logger.debug(
                    "DCC-GARCH estimated on %d obs × %d assets.", n_obs, n_assets
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "DCC-GARCH failed — falling back to EWMA.", exc_info=True
                )
                corr_matrix = None

        if corr_matrix is None:
            corr_matrix = self._estimate_ewma(returns)
            method = CorrelationMethod.EWMA
            logger.debug(
                "EWMA correlation estimated on %d obs × %d assets.", n_obs, n_assets
            )

        # Enforce symmetry and unit diagonal (numerical cleanup)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2.0
        np.fill_diagonal(corr_matrix, 1.0)

        # Summary statistics from off-diagonal elements
        off_diag_mask = ~np.eye(n_assets, dtype=bool)
        off_diag = corr_matrix[off_diag_mask]

        avg_corr = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
        max_corr = float(np.max(off_diag)) if len(off_diag) > 0 else 0.0
        div_score = 1.0 - avg_corr

        # Advisory warning
        warning: str | None = None
        if avg_corr > _CRITICAL_AVG_CORR:
            warning = (
                f"CRITICAL: Correlation spike ({avg_corr:.2f}) — potential contagion"
            )
        elif avg_corr > _WARN_AVG_CORR:
            warning = (
                f"WARNING: High portfolio correlation ({avg_corr:.2f}) "
                "— diversification degraded"
            )

        return CorrelationResult(
            method_used=method,
            correlation_matrix=corr_matrix,
            avg_pairwise_corr=avg_corr,
            max_pairwise_corr=max_corr,
            diversification_score=div_score,
            warning=warning,
        )

    def _estimate_dcc(self, returns: np.ndarray) -> np.ndarray:
        """Estimate current correlation via scalar DCC-GARCH.

        Fits univariate GARCH(1,1) to each return series, extracts standardized
        residuals, then applies the scalar DCC recursion to obtain the
        current correlation matrix R_T.

        Parameters
        ----------
        returns:
            T×N array of returns.

        Returns
        -------
        np.ndarray
            N×N current correlation matrix.
        """
        try:
            from arch import arch_model
        except ImportError as e:
            msg = "arch package required for DCC-GARCH estimation"
            raise ImportError(msg) from e

        n_obs, n_assets = returns.shape
        std_residuals = np.zeros_like(returns)

        # Step 1: Fit univariate GARCH(1,1) to each series; extract std residuals
        for i in range(n_assets):
            series = returns[:, i]
            # Scale to percent returns to improve numerical stability
            scaled = series * 100.0
            try:
                model = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
                fit = model.fit(disp="off", show_warning=False)
                cond_vol = fit.conditional_volatility
                # Avoid division by near-zero volatility
                cond_vol = np.maximum(cond_vol, 1e-8)
                std_residuals[:, i] = scaled / cond_vol
            except Exception:  # noqa: BLE001
                # If GARCH fails for this series, standardize by sample std
                std_dev = np.std(series)
                std_residuals[:, i] = series / (std_dev if std_dev > 1e-10 else 1.0)
                logger.debug("GARCH fit failed for asset %d — using sample std.", i)

        # Step 2: Scalar DCC recursion
        # Q_bar = unconditional covariance of standardized residuals
        q_bar = np.cov(std_residuals, rowvar=False)
        if q_bar.ndim == 0:
            q_bar = np.array([[float(q_bar)]])

        # Initialize Qt = Q_bar
        q_t = q_bar.copy()
        a, b = _DCC_A, _DCC_B

        for t in range(1, n_obs):
            e_t = std_residuals[t - 1 : t, :].T  # N×1
            q_t = (1.0 - a - b) * q_bar + a * (e_t @ e_t.T) + b * q_t

        # Step 3: Extract correlation matrix R_T
        d_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(q_t), 1e-12)))
        r_t = d_inv @ q_t @ d_inv

        return r_t

    def _estimate_ewma(self, returns: np.ndarray) -> np.ndarray:
        """Estimate current correlation via EWMA (RiskMetrics approach).

        Initializes with sample covariance, then applies:
            Cov_t = lambda * Cov_{t-1} + (1-lambda) * r_{t-1} r_{t-1}'

        Parameters
        ----------
        returns:
            T×N array of returns.

        Returns
        -------
        np.ndarray
            N×N current correlation matrix.
        """
        n_obs, n_assets = returns.shape
        lam = self.lambda_ewma

        # Initialize with sample covariance
        cov = np.cov(returns, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # EWMA update over time
        for t in range(1, n_obs):
            r_t = returns[t - 1 : t, :].T  # N×1
            cov = lam * cov + (1.0 - lam) * (r_t @ r_t.T)

        # Convert covariance to correlation
        std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        d_inv = np.diag(1.0 / std)
        corr = d_inv @ cov @ d_inv

        return corr

    def _forbes_rigobon_adjust(self, corr: float, vol_ratio: float) -> float:
        """Apply Forbes-Rigobon heteroskedasticity adjustment.

        Corrects upward bias in conditional correlation estimates during
        high-volatility regimes. The unconditional (adjusted) correlation is:

            rho_adjusted = rho / sqrt(1 + delta^2 * (1 - rho^2) / rho^2)

        where delta = sigma_high / sigma_normal (the vol ratio).

        Only applied when vol_ratio > _FR_VOL_RATIO_THRESHOLD (default 1.5).

        Parameters
        ----------
        corr:
            Conditional (biased) correlation estimate from the high-vol period.
        vol_ratio:
            Ratio of high-vol sigma to normal-vol sigma (delta = sigma_high / sigma_normal).

        Returns
        -------
        float
            Forbes-Rigobon adjusted (unconditional) correlation.
        """
        if vol_ratio <= _FR_VOL_RATIO_THRESHOLD:
            return corr

        # Guard against zero correlation (formula is undefined at rho=0)
        if abs(corr) < 1e-8:
            return corr

        delta_sq = vol_ratio**2
        rho_sq = corr**2
        denominator = 1.0 + delta_sq * (1.0 - rho_sq) / rho_sq
        if denominator <= 0.0:
            return corr

        adjusted = corr / np.sqrt(denominator)
        logger.debug(
            "Forbes-Rigobon: rho=%.4f, vol_ratio=%.2f -> rho_adj=%.4f",
            corr,
            vol_ratio,
            adjusted,
        )
        return float(adjusted)

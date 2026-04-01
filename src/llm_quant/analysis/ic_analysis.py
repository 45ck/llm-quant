"""Information Coefficient (IC) analysis for signal validation.

Uses alphalens-reloaded to compute:
- IC (Information Coefficient): Spearman correlation between signal and forward returns
- ICIR (IC Information Ratio): IC mean / IC std — measures signal consistency
- IC decay: IC at 1d, 5d, 10d, 21d horizons — how long the signal persists

Threshold guidance (not hard gates — advisory):
- |IC| > 0.05: signal worth investigating
- ICIR > 0.5: consistent enough to trade
- IC decay half-life > 5d: signal persists enough to trade with T+1 execution

Reference: Grinold & Kahn (2000), Chincarini & Kim (2006)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_ALPHALENS_AVAILABLE: bool | None = None


def _check_alphalens() -> bool:
    global _ALPHALENS_AVAILABLE  # noqa: PLW0603
    if _ALPHALENS_AVAILABLE is None:
        try:
            import alphalens  # noqa: F401

            _ALPHALENS_AVAILABLE = True
        except ImportError:
            _ALPHALENS_AVAILABLE = False
            logger.warning(
                "alphalens-reloaded not installed. Falling back to manual IC computation. "
                "Install with: pip install alphalens-reloaded"
            )
    return _ALPHALENS_AVAILABLE


@dataclass
class IcAnalysisConfig:
    forward_periods: list[int] = field(default_factory=lambda: [1, 5, 10, 21])
    ic_threshold: float = 0.05  # minimum |IC| for advisory pass
    icir_threshold: float = 0.5
    lookback_days: int = 504  # 2 years


@dataclass
class IcResult:
    ic_mean: float  # mean IC across all dates (at 1d horizon)
    ic_std: float  # std of IC time series
    icir: float  # ic_mean / ic_std
    ic_by_period: dict[int, float]  # {1: 0.05, 5: 0.04, 10: 0.03, 21: 0.02}
    ic_decay_halflife: float  # days for IC to decay to half its 1d value
    passes_ic_threshold: bool  # |ic_mean| > ic_threshold
    passes_icir_threshold: bool  # |icir| > icir_threshold
    advisory_grade: str  # "STRONG", "WEAK", "NOISE"
    n_observations: int


class IcAnalyzer:
    def __init__(self, config: IcAnalysisConfig | None = None) -> None:
        self.config = config or IcAnalysisConfig()

    def analyze(
        self,
        signal_series: pl.Series,
        price_series: pl.Series,
        dates: pl.Series,
    ) -> IcResult:
        """Compute IC analysis for a signal against forward returns.

        Args:
            signal_series: Factor/signal values (one per observation).
            price_series: Price or return series (used as forward return proxy if
                          already returns, or to compute them).
            dates: Date or integer index series aligned with signal/price.

        Returns:
            IcResult with IC, ICIR, decay and advisory grade.
        """
        signal = signal_series.to_numpy().astype(float)
        prices = price_series.to_numpy().astype(float)
        n = min(len(signal), len(prices))
        signal = signal[:n]
        prices = prices[:n]

        # Compute IC by period using manual approach (single-asset use case)
        # For multi-asset use, alphalens path is preferred
        if _check_alphalens():
            try:
                return self._compute_ic_alphalens_single(signal, prices, n)
            except Exception as exc:
                logger.warning("alphalens path failed (%s), using manual fallback", exc)

        return self._compute_ic_manual_all_periods(signal, prices, n)

    def _compute_ic_alphalens_single(
        self,
        signal: np.ndarray,
        prices: np.ndarray,
        n: int,
    ) -> IcResult:
        """Use alphalens for a single-asset factor — wraps into required MultiIndex format."""
        import pandas as pd

        config = self.config

        # Build a synthetic single-asset price DataFrame that alphalens expects:
        # columns = assets, index = dates
        date_idx = pd.date_range("2020-01-01", periods=n, freq="B")
        price_df = pd.DataFrame({"ASSET": prices}, index=date_idx)

        # Factor: MultiIndex (date, asset) → factor value
        factor_index = pd.MultiIndex.from_arrays(
            [date_idx, ["ASSET"] * n], names=["date", "asset"]
        )
        factor_series = pd.Series(signal, index=factor_index, name="factor")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import alphalens

            try:
                factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
                    factor_series,
                    price_df,
                    periods=config.forward_periods,
                    filter_zscore=None,
                    quantiles=None,
                    bins=5,
                )
                ic_df = alphalens.performance.factor_information_coefficient(
                    factor_data, group_adjust=False
                )
            except Exception:
                # Fall back to manual if alphalens internal format issues
                return self._compute_ic_manual_all_periods(signal, prices, n)

        ic_by_period: dict[int, float] = {}
        for period in config.forward_periods:
            col = f"{period}D" if f"{period}D" in ic_df.columns else period
            if col in ic_df.columns:
                ic_by_period[period] = float(ic_df[col].mean())
            else:
                # fallback for this period
                ic_by_period[period] = self._compute_ic_manual(
                    signal[: n - period], _forward_returns(prices, period)[: n - period]
                )

        return self._build_result(ic_by_period, n)

    def _compute_ic_manual_all_periods(
        self,
        signal: np.ndarray,
        prices: np.ndarray,
        n: int,
    ) -> IcResult:
        """Manual IC computation using scipy.stats.spearmanr."""
        ic_by_period: dict[int, float] = {}
        for period in self.config.forward_periods:
            fwd_ret = _forward_returns(prices, period)
            max_obs = min(len(signal), len(fwd_ret))
            if max_obs < 10:
                ic_by_period[period] = 0.0
                continue
            ic_by_period[period] = self._compute_ic_manual(
                signal[:max_obs], fwd_ret[:max_obs]
            )
        return self._build_result(ic_by_period, n)

    def _compute_ic_manual(
        self,
        signal: np.ndarray,
        forward_returns: np.ndarray,
    ) -> float:
        """Compute Spearman rank correlation between signal and forward returns."""
        from scipy import stats

        mask = np.isfinite(signal) & np.isfinite(forward_returns)
        if mask.sum() < 5:
            return 0.0
        corr, _ = stats.spearmanr(signal[mask], forward_returns[mask])
        return float(corr) if np.isfinite(corr) else 0.0

    def _build_result(
        self,
        ic_by_period: dict[int, float],
        n: int,
    ) -> IcResult:
        """Construct IcResult from ic_by_period dict."""
        config = self.config

        # Use 1d IC as primary (or smallest available period)
        primary_period = min(ic_by_period)
        ic_1d = ic_by_period.get(1, ic_by_period[primary_period])

        # For IC std, we'd need a time-series of rolling ICs; approximate with
        # cross-sectional std from available periods as a decay proxy
        ic_values = np.array(
            [ic_by_period.get(p, np.nan) for p in config.forward_periods]
        )
        ic_mean = float(ic_1d)

        # IC std: use std across periods (proxy when single asset)
        finite_ic = ic_values[np.isfinite(ic_values)]
        ic_std = float(np.std(finite_ic)) if len(finite_ic) > 1 else abs(ic_mean) * 0.3

        icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

        # IC decay half-life: fit exponential IC(t) = IC(1) * 0.5^(t/hl)
        ic_decay_halflife = _estimate_halflife(ic_by_period, config.forward_periods)

        passes_ic = abs(ic_mean) > config.ic_threshold
        passes_icir = abs(icir) > config.icir_threshold

        if abs(ic_mean) > 0.07 and abs(icir) > 0.7:
            grade = "STRONG"
        elif passes_ic and passes_icir:
            grade = "WEAK"
        else:
            grade = "NOISE"

        return IcResult(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_by_period=ic_by_period,
            ic_decay_halflife=ic_decay_halflife,
            passes_ic_threshold=passes_ic,
            passes_icir_threshold=passes_icir,
            advisory_grade=grade,
            n_observations=n,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forward_returns(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute period-ahead returns from a price/return series.

    If prices look like raw prices (values > 5 on average), compute log returns.
    If prices look like returns already (mean near 0), treat as returns and shift.
    """
    if np.nanmean(np.abs(prices)) > 1.0:
        log_px = np.log(np.where(prices > 0, prices, np.nan))
        fwd = np.roll(log_px, -period) - log_px
        fwd[-period:] = np.nan
    else:
        # Already returns — just shift
        fwd = np.roll(prices, -period).astype(float)
        fwd[-period:] = np.nan
    return fwd


def _estimate_halflife(ic_by_period: dict[int, float], periods: list[int]) -> float:
    """Fit exponential decay to IC across horizons and return half-life in days.

    Uses log-linear regression: log|IC(t)| = log|IC(1)| - lambda * t
    Half-life = ln(2) / lambda
    """
    xs = []
    ys = []
    ic_1d = ic_by_period.get(min(periods), 0.0)
    if abs(ic_1d) < 1e-8:
        return float("inf")

    for p in periods:
        ic_p = ic_by_period.get(p, np.nan)
        if not np.isfinite(ic_p) or abs(ic_p) < 1e-8:
            continue
        xs.append(p)
        ys.append(np.log(abs(ic_p)))

    if len(xs) < 2:
        return float("inf")

    # Linear regression in log space
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    slope, _ = np.polyfit(xs_arr, ys_arr, 1)

    if slope >= 0:
        return float("inf")  # IC not decaying

    halflife = -np.log(2.0) / slope
    return float(max(halflife, 0.5))


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_ic_report(result: IcResult) -> str:
    """Format IC analysis result as a human-readable report.

    Returns a markdown-style text report showing all metrics, IC decay table,
    PASS/FAIL for each threshold, and a final recommendation.
    """
    lines: list[str] = []

    def _pass(condition: bool) -> str:
        return "PASS" if condition else "FAIL"

    lines.append("=" * 60)
    lines.append("IC ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Observations : {result.n_observations}")
    lines.append(f"Advisory grade: {result.advisory_grade}")
    lines.append("")

    lines.append("--- Core Metrics ---")
    lines.append(
        f"IC mean (1d)  : {result.ic_mean:+.4f}  [{_pass(result.passes_ic_threshold)}]  (|IC| > 0.05)"
    )
    lines.append(f"IC std        : {result.ic_std:.4f}")
    lines.append(
        f"ICIR          : {result.icir:+.4f}  [{_pass(result.passes_icir_threshold)}]  (|ICIR| > 0.5)"
    )
    hl = result.ic_decay_halflife
    hl_str = f"{hl:.1f}d" if np.isfinite(hl) else "∞ (non-decaying)"
    hl_pass = hl > 5.0 or not np.isfinite(hl)
    lines.append(
        f"IC half-life  : {hl_str}  [{_pass(hl_pass)}]  (> 5d for T+1 execution)"
    )
    lines.append("")

    lines.append("--- IC Decay Table ---")
    lines.append(f"{'Horizon':>10}  {'IC':>8}  {'vs 1d':>8}")
    lines.append("-" * 32)
    ic_1d = result.ic_by_period.get(1, result.ic_mean)
    for period, ic_val in sorted(result.ic_by_period.items()):
        ratio_str = f"{ic_val / ic_1d:.2f}x" if abs(ic_1d) > 1e-8 else "n/a"
        lines.append(f"{period:>9}d  {ic_val:>+8.4f}  {ratio_str:>8}")
    lines.append("")

    lines.append("--- Threshold Gates (advisory, not hard gates) ---")
    lines.append(
        f"IC threshold  : {_pass(result.passes_ic_threshold)}  |IC| = {abs(result.ic_mean):.4f} (min 0.05)"
    )
    lines.append(
        f"ICIR threshold: {_pass(result.passes_icir_threshold)}  ICIR = {abs(result.icir):.4f} (min 0.5)"
    )
    lines.append("")

    lines.append("--- Recommendation ---")
    if result.advisory_grade == "STRONG":
        lines.append("Worth backtesting — signal shows consistent IC and ICIR.")
    elif result.advisory_grade == "WEAK":
        lines.append(
            "Weak signal — proceed with caution. Backtest with tight OOS discipline."
        )
    else:
        lines.append("Insufficient signal — do not backtest. IC below noise threshold.")
    lines.append("=" * 60)

    return "\n".join(lines)

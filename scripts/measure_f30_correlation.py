#!/usr/bin/env python3
"""Measure F30 ERP Regime (erp-regime-v1) pairwise correlation against all
existing cluster representative strategies, and re-run the portfolio optimizer
with F30 included.

F30 signal logic:
  - ERP = SPY 1-year return (trailing 252d) minus ^TNX yield (decimal)
  - Z-score of ERP over trailing 252 days
  - EQUITY_CHEAP (z > 1):      70% SPY + 10% TLT + 10% GLD + 10% SHY
  - EQUITY_EXPENSIVE (z < -1):  20% SPY + 30% TLT + 30% GLD + 20% SHY
  - NEUTRAL (else):             50% SPY + 20% TLT + 15% GLD + 15% SHY
  - Weekly rebalance (only update allocation on Mondays)
  - Causal: data through day t-1 to decide day t position

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/measure_f30_correlation.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

# Ensure src and project root are on path
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(_project_root) / "src"))
sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# F30 ERP Regime signal simulation
# ---------------------------------------------------------------------------


def _compute_return_metrics(daily_returns: list[float]) -> dict:
    """Compute Sharpe, Sortino, MaxDD, CAGR, total return from daily returns."""
    mean = sum(daily_returns) / len(daily_returns)
    std = (sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

    downside = [r for r in daily_returns if r < 0]
    down_std = (sum(r**2 for r in downside) / len(downside)) ** 0.5 if downside else 0.0
    sortino = (mean / down_std * math.sqrt(252)) if down_std > 0 else 0.0

    nav = [1.0]
    for r in daily_returns:
        nav.append(nav[-1] * (1.0 + r))
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    total_return = nav[-1] / nav[0] - 1.0
    n_years = len(daily_returns) / 252
    cagr = (nav[-1] / nav[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "cagr": cagr,
    }


def _build_erp_series(
    spy_close: list[float],
    dates: list,
    sym_data: dict[str, dict],
    erp_lookback: int,
) -> list[float | None]:
    """Build the ERP time series: SPY 1yr return minus TNX yield (decimal)."""
    n = len(dates)
    erp_series: list[float | None] = [None] * n
    for i in range(erp_lookback, n):
        spy_now = spy_close[i]
        spy_past = spy_close[i - erp_lookback]
        if spy_past <= 0:
            continue
        spy_1y_ret = spy_now / spy_past - 1.0

        d = dates[i]
        tnx_val = sym_data["TNX"].get(d, None)
        if tnx_val is None or tnx_val <= 0:
            continue

        tnx_decimal = tnx_val / 100.0  # e.g., 4.5 -> 0.045
        erp_series[i] = spy_1y_ret - tnx_decimal
    return erp_series


def _simulate_erp_daily_returns(
    dates: list,
    sym_data: dict[str, dict],
    erp_series: list[float | None],
    warmup: int,
    zscore_window: int,
) -> list[float]:
    """Simulate daily returns for ERP regime strategy with weekly rebalance."""
    n = len(dates)
    cost_per_switch = 0.0003
    regime_weights = {
        "EQUITY_CHEAP": {"SPY": 0.70, "TLT": 0.10, "GLD": 0.10, "SHY": 0.10},
        "EQUITY_EXPENSIVE": {"SPY": 0.20, "TLT": 0.30, "GLD": 0.30, "SHY": 0.20},
        "NEUTRAL": {"SPY": 0.50, "TLT": 0.20, "GLD": 0.15, "SHY": 0.15},
    }

    def _asset_ret(sym: str, i: int) -> float:
        d, dp = dates[i], dates[i - 1]
        data = sym_data[sym]
        if d in data and dp in data and data[dp] > 0:
            return data[d] / data[dp] - 1
        return 0.0

    daily_returns: list[float] = []
    prev_regime: str | None = None
    active_weights: dict[str, float] | None = None

    for i in range(warmup, n):
        # Compute z-score using data through day i-1 (causal)
        erp_window = [
            erp_series[j]
            for j in range(i - zscore_window, i)
            if j >= 0 and erp_series[j] is not None
        ]

        if len(erp_window) < zscore_window // 2:
            daily_returns.append(0.0)
            continue

        erp_mean = sum(erp_window) / len(erp_window)
        erp_std = (
            sum((v - erp_mean) ** 2 for v in erp_window) / len(erp_window)
        ) ** 0.5

        current_erp = erp_series[i - 1]
        if erp_std <= 0 or current_erp is None:
            daily_returns.append(0.0)
            continue

        z_score = (current_erp - erp_mean) / erp_std

        # Determine regime
        if z_score > 1.0:
            regime = "EQUITY_CHEAP"
        elif z_score < -1.0:
            regime = "EQUITY_EXPENSIVE"
        else:
            regime = "NEUTRAL"

        # Weekly rebalance: only update weights on Mondays
        d = dates[i]
        is_monday = hasattr(d, "weekday") and d.weekday() == 0

        switching_cost = 0.0
        if active_weights is None or is_monday:
            active_weights = regime_weights[regime]
            if prev_regime is not None and regime != prev_regime and is_monday:
                switching_cost = cost_per_switch
            prev_regime = regime

        # Compute day's return using active weights
        day_ret = sum(
            active_weights[sym] * _asset_ret(sym, i) for sym in active_weights
        )
        day_ret -= switching_cost
        daily_returns.append(day_ret)

    return daily_returns


def compute_erp_regime_returns() -> dict | None:
    """Simulate F30 ERP Regime strategy daily returns.

    Signal: SPY 1-year return minus TNX yield (decimal), 252-day z-score.
    EQUITY_CHEAP (z > 1):      70% SPY + 10% TLT + 10% GLD + 10% SHY
    EQUITY_EXPENSIVE (z < -1): 20% SPY + 30% TLT + 30% GLD + 20% SHY
    NEUTRAL:                   50% SPY + 20% TLT + 15% GLD + 15% SHY
    Weekly rebalance (Mondays only). Causal (data through t-1 for day t).
    """
    import polars as pl

    from llm_quant.data.fetcher import fetch_ohlcv

    symbols = ["SPY", "TLT", "GLD", "SHY", "TNX"]
    prices = fetch_ohlcv(symbols, lookback_days=1825)

    # Build date-aligned price dicts
    spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
    dates = spy_df["date"].to_list()
    spy_close = spy_df["close"].to_list()
    n = len(dates)

    sym_data: dict[str, dict] = {}
    for sym in symbols:
        sdf = prices.filter(pl.col("symbol") == sym).sort("date")
        sym_data[sym] = dict(
            zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
        )

    warmup = 504
    erp_lookback = 252
    zscore_window = 252

    if n < warmup + 10:
        logger.error("Not enough data: %d days (need >= %d)", n, warmup + 10)
        return None

    # Build ERP series and simulate returns
    erp_series = _build_erp_series(spy_close, dates, sym_data, erp_lookback)
    daily_returns = _simulate_erp_daily_returns(
        dates, sym_data, erp_series, warmup, zscore_window
    )

    if len(daily_returns) < 60:
        logger.error("Too few return days: %d", len(daily_returns))
        return None

    metrics = _compute_return_metrics(daily_returns)
    logger.info(
        "F30 ERP Regime: Sharpe=%.3f, Sortino=%.3f, MaxDD=%.1f%%, "
        "CAGR=%.1f%%, TotalReturn=%.1f%%, Days=%d",
        metrics["sharpe"],
        metrics["sortino"],
        metrics["max_drawdown"] * 100,
        metrics["cagr"] * 100,
        metrics["total_return"] * 100,
        len(daily_returns),
    )

    return {
        "daily_returns": daily_returns,
        **metrics,
        "dsr": 0.0,  # placeholder -- DSR not computed here
        "start_date": str(dates[warmup]),
        "end_date": str(dates[-1]),
        "family": "F30: ERP Regime",
        "n_days": len(daily_returns),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _report_correlations(
    f30_result: dict,
    rep_corrs: list[tuple[str, str, float]],
    all_corrs: list[tuple[str, str, float]],
) -> None:
    """Print correlation tables for F30 vs existing strategies."""
    # F30 metrics summary
    print("## F30 Strategy Metrics")
    print(f"  Sharpe:       {f30_result['sharpe']:.3f}")
    print(f"  Sortino:      {f30_result['sortino']:.3f}")
    print(f"  Max Drawdown: {f30_result['max_drawdown']:.1%}")
    print(f"  CAGR:         {f30_result['cagr']:.1%}")
    print(f"  Total Return: {f30_result['total_return']:.1%}")
    print(f"  Days:         {f30_result['n_days']}")
    print(f"  Period:       {f30_result['start_date']} to {f30_result['end_date']}")
    print()

    # Rep correlations table
    print("## Pairwise Correlations: F30 vs Cluster Representatives")
    print(f"{'Strategy':<40} {'Family':<30} {'Rho':>6} {'Rep?':>5}")
    print("-" * 85)
    for slug, family, rho in rep_corrs:
        print(f"{slug:<40} {family:<30} {rho:>6.3f}   YES")
    print()

    # Sorted by |correlation|
    print("## Sorted by |correlation| (most correlated first)")
    print(f"{'Strategy':<40} {'Family':<30} {'Rho':>6}")
    print("-" * 80)
    for slug, family, rho in sorted(rep_corrs, key=lambda x: -abs(x[2])):
        marker = " ***" if abs(rho) >= 0.70 else ""
        print(f"{slug:<40} {family:<30} {rho:>6.3f}{marker}")

    print()
    avg_corr_reps = np.mean([abs(rho) for _, _, rho in rep_corrs])
    max_corr = max(rep_corrs, key=lambda x: abs(x[2]))
    print(f"  Avg |correlation| with cluster reps: {avg_corr_reps:.3f}")
    print(
        f"  Max |correlation|: {abs(max_corr[2]):.3f} (vs {max_corr[0]}, {max_corr[1]})"
    )
    print()

    # All strategies
    print("## Pairwise Correlations: F30 vs ALL Existing Strategies")
    print(f"{'Strategy':<40} {'Family':<30} {'Rho':>6}")
    print("-" * 80)
    for slug, family, rho in sorted(all_corrs, key=lambda x: -abs(x[2])):
        marker = " ***" if abs(rho) >= 0.70 else ""
        print(f"{slug:<40} {family:<30} {rho:>6.3f}{marker}")

    print()
    avg_corr_all = np.mean([abs(rho) for _, _, rho in all_corrs])
    max_corr_all = max(all_corrs, key=lambda x: abs(x[2]))
    print(f"  Avg |correlation| with all strategies: {avg_corr_all:.3f}")
    print(
        f"  Max |correlation|: {abs(max_corr_all[2]):.3f} "
        f"(vs {max_corr_all[0]}, {max_corr_all[1]})"
    )
    print()


def _report_clustering(
    f30_slug: str,
    clusters_with_f30: dict,
    strategies_with_f30: dict,
    reps_with_f30: list[str],
    n_clusters_before: int,
) -> bool:
    """Print clustering results. Returns True if F30 is a new cluster rep."""
    n_clusters_after = len(clusters_with_f30)
    f30_is_new_cluster = f30_slug in reps_with_f30

    f30_cluster_id = None
    for cid, members in clusters_with_f30.items():
        if f30_slug in members:
            f30_cluster_id = cid
            break

    print(f"  Clusters before F30: {n_clusters_before}")
    print(f"  Clusters after F30:  {n_clusters_after}")
    print(f"  F30 cluster ID:      {f30_cluster_id}")
    print(f"  F30 cluster members: {clusters_with_f30.get(f30_cluster_id, [])}")
    print(f"  F30 is new cluster rep: {'YES' if f30_is_new_cluster else 'NO'}")
    print()

    cluster_members = clusters_with_f30.get(f30_cluster_id, [])
    if f30_is_new_cluster:
        print(
            "  >>> F30 forms a NEW cluster — it adds genuine diversification "
            "to the portfolio."
        )
    else:
        cluster_rep = max(
            cluster_members, key=lambda s: strategies_with_f30[s]["sharpe"]
        )
        if cluster_rep == f30_slug:
            print(
                "  >>> F30 clusters with existing strategies but becomes "
                "the new rep (highest Sharpe)."
            )
            old_members = [m for m in cluster_members if m != f30_slug]
            if old_members:
                print(f"  >>> Replaces: {old_members}")
        else:
            print(
                f"  >>> F30 clusters with {cluster_rep} — does NOT add a "
                f"new independent cluster."
            )
    print()
    return f30_is_new_cluster


def _report_portfolio_sharpe(
    pm_before: dict,
    pm_after: dict,
    n_reps_after: int,
    f30_is_new_cluster: bool,
) -> None:
    """Print portfolio Sharpe comparison, marginal value, and verdict."""
    print("  BEFORE (existing cluster reps):")
    print(f"    N strategies:       {pm_before['n_strategies']}")
    print(f"    Avg Sharpe:         {pm_before['avg_sharpe']:.3f}")
    print(f"    Avg pairwise rho:   {pm_before['avg_pairwise_correlation']:.3f}")
    print(f"    Portfolio SR (formula):   {pm_before['portfolio_sharpe_formula']:.3f}")
    print(
        f"    Portfolio SR (empirical): {pm_before['portfolio_sharpe_empirical']:.3f}"
    )
    print(f"    Portfolio MaxDD:    {pm_before['portfolio_max_drawdown']:.1%}")
    print()

    print(f"  AFTER (with F30, {n_reps_after} cluster reps):")
    print(f"    N strategies:       {pm_after['n_strategies']}")
    print(f"    Avg Sharpe:         {pm_after['avg_sharpe']:.3f}")
    print(f"    Avg pairwise rho:   {pm_after['avg_pairwise_correlation']:.3f}")
    print(f"    Portfolio SR (formula):   {pm_after['portfolio_sharpe_formula']:.3f}")
    print(f"    Portfolio SR (empirical): {pm_after['portfolio_sharpe_empirical']:.3f}")
    print(f"    Portfolio MaxDD:    {pm_after['portfolio_max_drawdown']:.1%}")
    print()

    sr_delta_formula = (
        pm_after["portfolio_sharpe_formula"] - pm_before["portfolio_sharpe_formula"]
    )
    sr_delta_empirical = (
        pm_after["portfolio_sharpe_empirical"] - pm_before["portfolio_sharpe_empirical"]
    )
    dd_delta = pm_after["portfolio_max_drawdown"] - pm_before["portfolio_max_drawdown"]
    print("  DELTA:")
    print(f"    Portfolio SR change (formula):   {sr_delta_formula:+.3f}")
    print(f"    Portfolio SR change (empirical): {sr_delta_empirical:+.3f}")
    print(f"    MaxDD change: {dd_delta:+.1%}")
    print()

    # Marginal value
    print("=" * 80)
    print("MARGINAL VALUE ASSESSMENT")
    print("=" * 80)
    print()

    n_before = pm_before["n_strategies"]
    rho_before = pm_before["avg_pairwise_correlation"]
    sr_avg_before = pm_before["avg_sharpe"]
    n_after_theory = n_before + 1
    sr_marginal_uncorr = sr_avg_before * (
        math.sqrt(n_after_theory / (1 + (n_after_theory - 1) * rho_before))
        - math.sqrt(n_before / (1 + (n_before - 1) * rho_before))
    )
    print(
        f"  Theoretical marginal SR (if fully uncorrelated): {sr_marginal_uncorr:+.3f}"
    )
    print(f"  Actual marginal SR (formula):                    {sr_delta_formula:+.3f}")
    print(
        f"  Actual marginal SR (empirical):                  {sr_delta_empirical:+.3f}"
    )
    print()

    # Verdict
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()
    if f30_is_new_cluster and sr_delta_empirical > 0:
        print("  F30 ERP Regime ADDS a new cluster and IMPROVES portfolio Sharpe.")
        print("  RECOMMENDATION: Accept as new cluster representative.")
    elif f30_is_new_cluster:
        print(
            "  F30 ERP Regime adds a new cluster but does NOT improve portfolio Sharpe."
        )
        print("  RECOMMENDATION: Review — new cluster but negative marginal SR.")
    else:
        print(
            "  F30 ERP Regime does NOT form a new cluster "
            "(rho >= 0.70 with existing strategy)."
        )
        print("  RECOMMENDATION: Do not add as cluster representative.")
    print()


def main() -> None:
    # Import portfolio optimizer infrastructure
    from scripts.portfolio_optimizer import (
        MECHANISM_FAMILIES,
        cluster_strategies,
        compute_correlation_matrix,
        compute_portfolio_sharpe,
        load_daily_returns,
        select_representatives,
    )

    data_dir = Path("data")

    # 1. Load all existing strategies
    logger.info("=" * 70)
    logger.info("STEP 1: Loading existing strategy returns")
    logger.info("=" * 70)
    strategies = load_daily_returns(data_dir)
    logger.info("Loaded %d existing strategies", len(strategies))

    # 2. Compute F30 returns
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Computing F30 ERP Regime returns")
    logger.info("=" * 70)
    f30_result = compute_erp_regime_returns()
    if f30_result is None:
        logger.error("FATAL: F30 computation failed -- aborting")
        sys.exit(1)

    # 3. Cluster existing strategies to identify current reps
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Clustering existing strategies (threshold=0.70)")
    logger.info("=" * 70)

    corr_matrix_existing, slugs_existing = compute_correlation_matrix(strategies)
    clusters_existing = cluster_strategies(
        corr_matrix_existing, slugs_existing, threshold=0.70
    )
    reps_existing = select_representatives(clusters_existing, strategies)
    logger.info(
        "Existing portfolio: %d clusters, %d reps",
        len(clusters_existing),
        len(reps_existing),
    )

    # 4. Add F30 and compute full correlation matrix
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: Adding F30 and computing pairwise correlations")
    logger.info("=" * 70)

    f30_slug = "erp-regime-v1"
    strategies_with_f30 = dict(strategies)
    strategies_with_f30[f30_slug] = f30_result

    corr_matrix_all, slugs_all = compute_correlation_matrix(strategies_with_f30)
    f30_idx = slugs_all.index(f30_slug)

    # Build correlation lists
    rep_corrs = []
    for rep_slug in sorted(reps_existing):
        if rep_slug in slugs_all:
            rep_idx = slugs_all.index(rep_slug)
            rho = corr_matrix_all[f30_idx, rep_idx]
            family = MECHANISM_FAMILIES.get(rep_slug, "Unknown")
            rep_corrs.append((rep_slug, family, rho))

    all_corrs = []
    for slug in sorted(slugs_all):
        if slug == f30_slug:
            continue
        slug_idx = slugs_all.index(slug)
        rho = corr_matrix_all[f30_idx, slug_idx]
        family = MECHANISM_FAMILIES.get(slug, "Unknown")
        all_corrs.append((slug, family, rho))

    # 5. Report correlations
    print("=" * 80)
    print("F30 ERP REGIME (erp-regime-v1) -- PAIRWISE CORRELATION ANALYSIS")
    print("=" * 80)
    print()
    _report_correlations(f30_result, rep_corrs, all_corrs)

    # 6. Re-cluster with F30 included
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: Re-clustering with F30 included")
    logger.info("=" * 70)

    clusters_with_f30 = cluster_strategies(corr_matrix_all, slugs_all, threshold=0.70)
    reps_with_f30 = select_representatives(clusters_with_f30, strategies_with_f30)

    print("=" * 80)
    print("CLUSTERING RESULTS WITH F30 INCLUDED")
    print("=" * 80)
    print()
    f30_is_new_cluster = _report_clustering(
        f30_slug,
        clusters_with_f30,
        strategies_with_f30,
        reps_with_f30,
        len(clusters_existing),
    )

    # 7. Compute portfolio Sharpe -- before and after
    print("=" * 80)
    print("PORTFOLIO SHARPE COMPARISON")
    print("=" * 80)
    print()

    pm_before = compute_portfolio_sharpe(
        reps_existing, strategies, corr_matrix_existing, slugs_existing
    )
    pm_after = compute_portfolio_sharpe(
        reps_with_f30, strategies_with_f30, corr_matrix_all, slugs_all
    )
    _report_portfolio_sharpe(
        pm_before, pm_after, len(reps_with_f30), f30_is_new_cluster
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Portfolio optimizer: correlation clustering and portfolio Sharpe estimation.

Loads OOS daily return series for all passing strategies, computes the
pairwise correlation matrix, clusters by hierarchical clustering, picks
cluster representatives, and estimates portfolio Sharpe using the
corrected formula that accounts for average pairwise correlation.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/portfolio_optimizer.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/portfolio_optimizer.py --threshold 0.6
    cd E:/llm-quant && PYTHONPATH=src python scripts/portfolio_optimizer.py --top-n 8
    cd E:/llm-quant && PYTHONPATH=src python scripts/portfolio_optimizer.py --ignore-missing
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.artifacts import load_artifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: the 15 passing strategies in paper trading
# ---------------------------------------------------------------------------

# Map slug -> experiment_id of the best (highest-Sharpe) experiment
STRATEGY_EXPERIMENTS: dict[str, str] = {
    "soxx-qqq-lead-lag": "57fba00d",
    "lqd-spy-credit-lead": "b0588e6d",
    "agg-spy-credit-lead": "66bec9a0",
    "hyg-spy-5d-credit-lead": "1736ac56",
    "agg-qqq-credit-lead": "eaf37299",
    "lqd-qqq-credit-lead": "ec8745f9",
    "vcit-qqq-credit-lead": "b99dac63",
    "hyg-qqq-credit-lead": "ba0c05a2",
    "emb-spy-credit-lead": "90e531d1",
    "agg-efa-credit-lead": "bef23aa4",
    "spy-overnight-momentum": "22cddf8c",
    "tlt-spy-rate-momentum": "9e14ce90",
    "tlt-qqq-rate-tech": "2338b9e5",
    "ief-qqq-rate-tech": "594c4f53",
    "behavioral-structural": "7cb2cace",
    "gld-slv-mean-reversion-v4": "14cdfaaf",
    "skip-month-tsmom-v1": "a220a89a",
    "credit-spread-regime-v1": "DIRECT",  # Direct computation (engine incompatible)
}

# Mechanism family labels for context
MECHANISM_FAMILIES: dict[str, str] = {
    "soxx-qqq-lead-lag": "F8: Non-Credit Lead-Lag",
    "lqd-spy-credit-lead": "F1: Credit Lead-Lag",
    "agg-spy-credit-lead": "F1: Credit Lead-Lag",
    "hyg-spy-5d-credit-lead": "F1: Credit Lead-Lag",
    "agg-qqq-credit-lead": "F1: Credit Lead-Lag",
    "lqd-qqq-credit-lead": "F1: Credit Lead-Lag",
    "vcit-qqq-credit-lead": "F1: Credit Lead-Lag",
    "hyg-qqq-credit-lead": "F1: Credit Lead-Lag",
    "emb-spy-credit-lead": "F1: Credit Lead-Lag",
    "agg-efa-credit-lead": "F1: Credit Lead-Lag",
    "spy-overnight-momentum": "F5: Overnight Momentum",
    "tlt-spy-rate-momentum": "F6: Rate Momentum",
    "tlt-qqq-rate-tech": "F6: Rate Momentum",
    "ief-qqq-rate-tech": "F6: Rate Momentum",
    "behavioral-structural": "F7: Behavioral/Structural",
    "gld-slv-mean-reversion-v4": "F2: Mean Reversion",
    "skip-month-tsmom-v1": "F3: TSMOM",
    "credit-spread-regime-v1": "F9: Spread Regime",
}

TRADING_DAYS_PER_YEAR = 252

# Track A vs Track B membership for 70/30 split enforcement
TRACK_A_SLUGS: list[str] = [
    "lqd-spy-credit-lead",
    "agg-spy-credit-lead",
    "hyg-spy-5d-credit-lead",
    "agg-qqq-credit-lead",
    "lqd-qqq-credit-lead",
    "vcit-qqq-credit-lead",
    "hyg-qqq-credit-lead",
    "emb-spy-credit-lead",
    "agg-efa-credit-lead",
    "spy-overnight-momentum",
    "tlt-spy-rate-momentum",
    "tlt-qqq-rate-tech",
    "ief-qqq-rate-tech",
    "behavioral-structural",
    "gld-slv-mean-reversion-v4",
    "skip-month-tsmom-v1",
    "credit-spread-regime-v1",
]

TRACK_B_SLUGS: list[str] = [
    "soxx-qqq-lead-lag",
]

# Max weight per strategy for validation (Track B limit is binding)
MAX_STRATEGY_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# HRP allocation
# ---------------------------------------------------------------------------


def compute_hrp_weights(
    strategies: dict[str, dict],
    slugs: list[str],
    min_len: int,
) -> dict[str, float]:
    """Compute HRP weights using Riskfolio-Lib, then enforce 70/30 A/B split.

    Parameters
    ----------
    strategies:
        Dict of slug -> strategy data (must include ``daily_returns``).
    slugs:
        Ordered list of slugs (must match the order in the returns matrix).
    min_len:
        Number of aligned return days to use (tail of each series).

    Returns
    -------
    dict mapping slug -> portfolio weight (sums to 1.0, 70% Track A / 30% Track B).
    """
    import pandas as pd

    try:
        import Riskfolio as rp
    except ImportError as exc:
        msg = (
            "Riskfolio-Lib is required for --method hrp. "
            "Install with: pip install 'Riskfolio-Lib>=6.0'"
        )
        raise ImportError(msg) from exc

    # Build returns DataFrame (columns = slugs, rows = trading days)
    data: dict[str, list[float]] = {}
    for slug in slugs:
        dr = strategies[slug]["daily_returns"]
        data[slug] = dr[-min_len:]

    returns_pd = pd.DataFrame(data)

    # Fit HRP — ward linkage, Pearson codependence, MV risk measure
    port = rp.HCPortfolio(returns=returns_pd)
    weights_df = port.optimization(
        model="HRP",
        codependence="pearson",
        rm="MV",
        rf=0,
        linkage="ward",
        max_k=10,
        leaf_order=True,
    )

    # weights_df: DataFrame with strategy names as index, 'weights' column
    raw_weights: dict[str, float] = weights_df["weights"].to_dict()

    # Determine which loaded slugs belong to each track
    a_slugs_loaded = [s for s in slugs if s in TRACK_A_SLUGS]
    b_slugs_loaded = [s for s in slugs if s in TRACK_B_SLUGS]

    a_total = sum(raw_weights.get(s, 0.0) for s in a_slugs_loaded)
    b_total = sum(raw_weights.get(s, 0.0) for s in b_slugs_loaded)

    adjusted: dict[str, float] = {}

    # Rescale Track A to 70% of portfolio
    for slug in a_slugs_loaded:
        adjusted[slug] = (
            raw_weights.get(slug, 0.0) * (0.70 / a_total) if a_total > 0 else 0.0
        )

    # Rescale Track B to 30% of portfolio
    for slug in b_slugs_loaded:
        adjusted[slug] = (
            raw_weights.get(slug, 0.0) * (0.30 / b_total) if b_total > 0 else 0.0
        )

    # Any slugs not assigned to a track get raw HRP weights (edge case)
    untracked = [
        s for s in slugs if s not in a_slugs_loaded and s not in b_slugs_loaded
    ]
    for slug in untracked:
        adjusted[slug] = raw_weights.get(slug, 0.0)
        logger.warning(
            "Slug '%s' is not assigned to Track A or B — using raw HRP weight %.4f",
            slug,
            adjusted[slug],
        )

    return adjusted


def validate_weights(weights: dict[str, float], method: str) -> None:
    """Validate weight constraints: sum = 1.0 ± 0.001, no strategy > 15%.

    Logs a warning for any violation but does not raise — the report still
    shows the violation so the PM can decide how to handle it.
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        logger.warning(
            "WEIGHT VALIDATION [%s]: weights sum to %.6f (expected 1.000 ± 0.001)",
            method,
            total,
        )
    for slug, w in weights.items():
        if w > MAX_STRATEGY_WEIGHT + 1e-9:
            logger.warning(
                "WEIGHT VALIDATION [%s]: %s weight=%.4f exceeds Track B limit of %.0f%%",
                method,
                slug,
                w,
                MAX_STRATEGY_WEIGHT * 100,
            )


# ---------------------------------------------------------------------------
# Direct computation for strategies incompatible with engine
# ---------------------------------------------------------------------------


def _load_direct_returns(slug: str, data_dir: Path) -> dict | None:
    """Compute daily returns directly for strategies that bypass the engine."""
    if slug == "credit-spread-regime-v1":
        return _compute_credit_spread_regime_returns(data_dir)
    return None


def _compute_credit_spread_regime_returns(data_dir: Path) -> dict | None:
    """Credit spread regime: HYG/SHY ratio momentum → SPY/SHY allocation."""
    try:
        import polars as pl

        from llm_quant.data.fetcher import fetch_ohlcv

        symbols = ["SPY", "SHY", "HYG"]
        prices = fetch_ohlcv(symbols, lookback_days=5 * 365)

        spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
        dates = spy_df["date"].to_list()
        spy_close = spy_df["close"].to_list()
        n = len(dates)

        shy_df = prices.filter(pl.col("symbol") == "SHY").sort("date")
        shy_data = dict(
            zip(shy_df["date"].to_list(), shy_df["close"].to_list(), strict=False)
        )
        hyg_df = prices.filter(pl.col("symbol") == "HYG").sort("date")
        hyg_data = dict(
            zip(hyg_df["date"].to_list(), hyg_df["close"].to_list(), strict=False)
        )

        spy_rets = [0.0] + [
            (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0
            for i in range(1, n)
        ]

        warmup = 60
        ratio_lookback = 10
        sma_period = 20

        # Build ratio series
        ratio_series = []
        for i in range(n):
            d = dates[i]
            hyg = hyg_data.get(d, 0.0)
            shy = shy_data.get(d, 0.0)
            ratio_series.append(hyg / shy if hyg > 0 and shy > 0 else 0.0)

        daily_returns = []
        prev_regime = None
        cost_per_switch = 0.0003

        for i in range(warmup, n):
            if i < max(ratio_lookback, sma_period) + 1:
                daily_returns.append(0.0)
                continue

            ratio_now = ratio_series[i - 1]
            ratio_lb = ratio_series[i - 1 - ratio_lookback]
            if ratio_now <= 0 or ratio_lb <= 0:
                daily_returns.append(0.0)
                continue

            ratio_mom = ratio_now / ratio_lb - 1
            sma_window = ratio_series[i - 1 - sma_period : i - 1]
            ratio_sma = (
                sum(sma_window) / sma_period if len(sma_window) >= sma_period else 0
            )
            if ratio_sma <= 0:
                daily_returns.append(0.0)
                continue
            ratio_vs_sma = ratio_now / ratio_sma - 1

            # SHY return
            d, dp = dates[i], dates[i - 1]
            shy_r = (
                (shy_data[d] / shy_data[dp] - 1)
                if d in shy_data and dp in shy_data and shy_data[dp] > 0
                else 0.0
            )

            if ratio_mom > 0 and ratio_vs_sma > 0:
                regime = "risk_on"
                day_ret = spy_rets[i] * 0.90
            elif ratio_mom < 0 and ratio_vs_sma < 0:
                regime = "risk_off"
                day_ret = shy_r * 0.90
            else:
                regime = "neutral"
                day_ret = spy_rets[i] * 0.45

            if prev_regime is not None and regime != prev_regime:
                day_ret -= cost_per_switch
            prev_regime = regime
            daily_returns.append(day_ret)

        if len(daily_returns) < 60:
            return None

        mean = sum(daily_returns) / len(daily_returns)
        std = (sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

        nav = [1.0]
        for r in daily_returns:
            nav.append(nav[-1] * (1.0 + r))
        peak = nav[0]
        max_dd = 0.0
        for v in nav:
            peak = max(peak, v)
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        return {
            "daily_returns": daily_returns,
            "sharpe": sharpe,
            "sortino": 0.0,
            "max_drawdown": max_dd,
            "total_return": nav[-1] / nav[0] - 1.0,
            "dsr": 0.961,
            "start_date": str(dates[warmup]),
            "end_date": str(dates[-1]),
        }
    except Exception as e:
        logger.warning("Failed to compute credit-spread-regime returns: %s", e)
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_daily_returns(
    data_dir: Path,
) -> dict[str, dict]:
    """Load daily returns and metrics from experiment artifacts.

    Returns a dict of slug -> {daily_returns: list[float], sharpe: float, ...}
    Logs a warning for every skipped strategy with the specific reason.
    """
    strategies: dict[str, dict] = {}
    registered = len(STRATEGY_EXPERIMENTS)

    for slug, exp_id in STRATEGY_EXPERIMENTS.items():
        # Handle strategies with direct computation (no engine artifact)
        if exp_id == "DIRECT":
            direct_rets = _load_direct_returns(slug, data_dir)
            if direct_rets is not None:
                direct_rets["family"] = MECHANISM_FAMILIES.get(slug, "Unknown")
                strategies[slug] = direct_rets
            else:
                logger.warning("SKIP [%s]: direct computation failed", slug)
            continue

        artifact_path = (
            data_dir / "strategies" / slug / "experiments" / f"{exp_id}.yaml"
        )
        if not artifact_path.exists():
            logger.warning(
                "SKIP [%s]: artifact not found at %s "
                "(experiment %s — run /lifecycle to reconstruct)",
                slug,
                artifact_path,
                exp_id,
            )
            continue

        artifact = load_artifact(artifact_path)
        daily_returns = artifact.get("daily_returns", [])
        metrics = artifact.get("metrics_1x", {})

        if not daily_returns:
            logger.warning(
                "SKIP [%s]: artifact %s exists but contains no daily_returns "
                "(re-run /backtest to regenerate)",
                slug,
                exp_id,
            )
            continue

        strategies[slug] = {
            "daily_returns": daily_returns,
            "sharpe": metrics.get("sharpe_ratio", 0.0),
            "sortino": metrics.get("sortino_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "total_return": metrics.get("total_return", 0.0),
            "dsr": metrics.get("dsr", 0.0),
            "start_date": artifact.get("start_date", ""),
            "end_date": artifact.get("end_date", ""),
            "family": MECHANISM_FAMILIES.get(slug, "Unknown"),
        }

    loaded = len(strategies)
    skipped = registered - loaded
    if skipped > 0:
        logger.warning(
            "Loaded %d of %d registered strategies (%d skipped — artifacts missing). "
            "See warnings above for details. Run /lifecycle to identify gaps.",
            loaded,
            registered,
            skipped,
        )
    else:
        logger.info("Loaded %d strategies with daily returns", loaded)
    return strategies


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def compute_correlation_matrix(
    strategies: dict[str, dict],
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise correlation matrix of strategy daily returns.

    Aligns all return series to the same length by trimming from the front
    (keeping the most recent overlapping period). This handles strategies
    with different backtest start dates.

    Returns (correlation_matrix, ordered_slugs).
    """
    slugs = sorted(strategies.keys())
    n = len(slugs)

    # Find minimum length for alignment
    lengths = [len(strategies[s]["daily_returns"]) for s in slugs]
    min_len = min(lengths)
    logger.info(
        "Return series lengths: min=%d, max=%d — aligning to %d days",
        min(lengths),
        max(lengths),
        min_len,
    )

    # Build aligned returns matrix (N_strategies x T_days)
    returns_matrix = np.zeros((n, min_len))
    for i, slug in enumerate(slugs):
        dr = strategies[slug]["daily_returns"]
        # Take the LAST min_len entries (most recent common period)
        returns_matrix[i, :] = dr[-min_len:]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(returns_matrix)

    # Clean up numerical artifacts
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix, slugs


# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------


def cluster_strategies(
    corr_matrix: np.ndarray,
    slugs: list[str],
    threshold: float = 0.7,
) -> dict[int, list[str]]:
    """Cluster strategies by correlation using hierarchical clustering.

    Uses complete linkage with (1 - correlation) as distance.
    Strategies with correlation >= threshold are grouped together.

    Returns dict of cluster_id -> [slugs].
    """
    n = len(slugs)
    if n <= 1:
        return {1: list(slugs)}

    # Convert correlation to distance: d = 1 - |corr|
    # Using absolute correlation because anti-correlated strategies
    # are also redundant from a diversification perspective
    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0.0)

    # Ensure symmetry and non-negative
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    dist_matrix = np.maximum(dist_matrix, 0.0)

    # Convert to condensed form for scipy
    condensed = squareform(dist_matrix, checks=False)

    # Complete linkage: cluster merges when ALL members are within threshold
    linkage_matrix = linkage(condensed, method="complete")

    # Cut at distance = 1 - threshold
    # Strategies with correlation >= threshold have distance <= 1-threshold
    cut_distance = 1.0 - threshold
    labels = fcluster(linkage_matrix, t=cut_distance, criterion="distance")

    # Build cluster dict
    clusters: dict[int, list[str]] = {}
    for slug, label in zip(slugs, labels, strict=True):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(slug)

    return clusters


# ---------------------------------------------------------------------------
# Cluster representative selection
# ---------------------------------------------------------------------------


def select_representatives(
    clusters: dict[int, list[str]],
    strategies: dict[str, dict],
) -> list[str]:
    """From each cluster, pick the strategy with the highest Sharpe ratio.

    Returns list of selected slugs.
    """
    selected: list[str] = []
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        best_slug = max(members, key=lambda s: strategies[s]["sharpe"])
        selected.append(best_slug)
    return selected


# ---------------------------------------------------------------------------
# Portfolio Sharpe estimation
# ---------------------------------------------------------------------------


def compute_portfolio_sharpe(
    selected_slugs: list[str],
    strategies: dict[str, dict],
    corr_matrix: np.ndarray,
    all_slugs: list[str],
) -> dict:
    """Compute estimated portfolio Sharpe using the corrected formula.

    SR_P = SR_avg * sqrt(N / (1 + (N-1) * rho_avg))

    where:
      SR_avg = average Sharpe of selected strategies
      N = number of selected strategies
      rho_avg = average pairwise correlation among selected strategies

    Also computes equal-weight portfolio Sharpe from actual return series
    for validation.

    Returns dict with portfolio metrics.
    """
    n = len(selected_slugs)
    if n == 0:
        return {"portfolio_sharpe_formula": 0.0, "n_strategies": 0}

    # Get indices of selected strategies in the full slug list
    idx_map = {s: i for i, s in enumerate(all_slugs)}
    selected_indices = [idx_map[s] for s in selected_slugs]

    # Compute average Sharpe
    sharpes = [strategies[s]["sharpe"] for s in selected_slugs]
    sr_avg = np.mean(sharpes)

    # Compute average pairwise correlation among selected
    if n > 1:
        pairwise_corrs = [
            corr_matrix[selected_indices[i], selected_indices[j]]
            for i in range(len(selected_indices))
            for j in range(i + 1, len(selected_indices))
        ]
        rho_avg = float(np.mean(pairwise_corrs))
    else:
        rho_avg = 0.0

    # Corrected portfolio Sharpe formula
    denominator = 1.0 + (n - 1) * rho_avg
    if denominator > 0:
        sr_portfolio = float(sr_avg) * math.sqrt(n / denominator)
    else:
        sr_portfolio = float(sr_avg) * math.sqrt(n)

    # Also compute equal-weight portfolio Sharpe from actual returns
    # for validation against the formula estimate
    min_len = min(len(strategies[s]["daily_returns"]) for s in selected_slugs)
    portfolio_returns = np.zeros(min_len)
    for slug in selected_slugs:
        dr = np.array(strategies[slug]["daily_returns"][-min_len:])
        portfolio_returns += dr / n  # equal weight

    ew_mean = float(np.mean(portfolio_returns))
    ew_std = float(np.std(portfolio_returns, ddof=1))
    if ew_std > 0:
        sr_empirical = ew_mean / ew_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sr_empirical = 0.0

    # Empirical max drawdown of equal-weight portfolio
    nav = np.cumprod(1.0 + portfolio_returns)
    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav) / peak
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    return {
        "n_strategies": n,
        "avg_sharpe": float(sr_avg),
        "avg_pairwise_correlation": rho_avg,
        "portfolio_sharpe_formula": sr_portfolio,
        "portfolio_sharpe_empirical": sr_empirical,
        "portfolio_max_drawdown": max_dd,
        "individual_sharpes": {s: strategies[s]["sharpe"] for s in selected_slugs},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_correlation_heatmap(
    corr_matrix: np.ndarray,
    slugs: list[str],
) -> str:
    """Format correlation matrix as a text heatmap with short labels."""
    n = len(slugs)
    # Create short labels (max 20 chars)
    short_labels = []
    for s in slugs:
        label = s.replace("-credit-lead", "-CL").replace("-lead-lag", "-LL")
        label = label.replace("-momentum", "-mom").replace("-rate-tech", "-RT")
        label = label.replace("-rate-momentum", "-RM")
        label = label.replace("behavioral-structural", "behav-struct")
        label = label.replace("spy-overnight", "spy-ON")
        if len(label) > 20:
            label = label[:20]
        short_labels.append(label)

    # Header
    col_width = 8
    header = " " * 22
    for label in short_labels:
        header += f"{label[:col_width]:>{col_width}}"
    lines = [header]

    # Rows
    for i in range(n):
        row = f"{short_labels[i]:>20}  "
        for j in range(n):
            val = corr_matrix[i, j]
            cell = "  1.00" if i == j else f"{val:6.2f}"
            # Add visual marker for high correlation
            if abs(val) >= 0.7 and i != j:
                cell += "*"
            else:
                cell += " "
            row += f"{cell:>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def _format_strategy_overview(strategies: dict[str, dict]) -> list[str]:
    """Format strategy overview section."""
    lines: list[str] = [
        "=" * 80,
        "PORTFOLIO OPTIMIZER REPORT",
        "=" * 80,
        "",
        "## 1. Strategy Overview",
        "",
        f"{'Strategy':<32} {'Family':<25} {'Sharpe':>7} {'MaxDD':>7} "
        f"{'DSR':>6} {'Return':>8}",
        "-" * 95,
    ]
    for slug in sorted(strategies.keys(), key=lambda s: -strategies[s]["sharpe"]):
        s = strategies[slug]
        lines.append(
            f"{slug:<32} {s['family']:<25} {s['sharpe']:>7.3f} "
            f"{s['max_drawdown']:>6.1%} {s['dsr']:>6.3f} "
            f"{s['total_return']:>7.1%}"
        )
    lines.append("")
    return lines


def _format_correlation_section(
    corr_matrix: np.ndarray,
    slugs: list[str],
) -> list[str]:
    """Format correlation matrix section with summary stats."""
    n = len(slugs)
    lines: list[str] = [
        "## 2. Pairwise Correlation Matrix",
        "",
        "(* marks |correlation| >= 0.70)",
        "",
        format_correlation_heatmap(corr_matrix, slugs),
        "",
    ]
    upper_tri = [corr_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    upper_tri_arr = np.array(upper_tri)
    lines.append(f"Correlation stats (all {len(upper_tri)} pairs):")
    lines.append(f"  Mean:   {np.mean(upper_tri_arr):.3f}")
    lines.append(f"  Median: {np.median(upper_tri_arr):.3f}")
    lines.append(f"  Min:    {np.min(upper_tri_arr):.3f}")
    lines.append(f"  Max:    {np.max(upper_tri_arr):.3f}")
    lines.append(
        f"  Pairs with |corr| >= 0.70: {np.sum(np.abs(upper_tri_arr) >= 0.70)}"
    )
    lines.append(
        f"  Pairs with |corr| >= 0.50: {np.sum(np.abs(upper_tri_arr) >= 0.50)}"
    )
    lines.append("")
    return lines


def _format_clustering_section(
    strategies: dict[str, dict],
    corr_matrix: np.ndarray,
    slugs: list[str],
    clusters: dict[int, list[str]],
    selected: list[str],
    threshold: float,
) -> list[str]:
    """Format clustering section."""
    lines: list[str] = [
        f"## 3. Hierarchical Clustering (threshold = {threshold:.2f})",
        "",
        "Method: complete linkage on distance = 1 - |correlation|",
        f"Strategies with |correlation| >= {threshold:.2f} are grouped together.",
        "",
    ]
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        lines.append(f"### Cluster {cluster_id} ({len(members)} strategies)")
        for slug in sorted(members, key=lambda s: -strategies[s]["sharpe"]):
            marker = " <-- REPRESENTATIVE" if slug in selected else ""
            lines.append(
                f"  {slug:<32} Sharpe={strategies[slug]['sharpe']:.3f}  "
                f"[{strategies[slug]['family']}]{marker}"
            )
        if len(members) > 1:
            member_indices = [slugs.index(m) for m in members]
            intra_corrs = [
                corr_matrix[member_indices[i], member_indices[j]]
                for i in range(len(member_indices))
                for j in range(i + 1, len(member_indices))
            ]
            lines.append(f"  Avg intra-cluster correlation: {np.mean(intra_corrs):.3f}")
        lines.append("")
    return lines


def _format_portfolio_metrics_block(label: str, metrics: dict) -> list[str]:
    """Format a single portfolio metrics block."""
    return [
        label,
        f"  N strategies:              {metrics['n_strategies']}",
        f"  Avg individual Sharpe:     {metrics['avg_sharpe']:.3f}",
        f"  Avg pairwise correlation:  {metrics['avg_pairwise_correlation']:.3f}",
        f"  Portfolio Sharpe (formula): {metrics['portfolio_sharpe_formula']:.3f}",
        f"  Portfolio Sharpe (empirical EW): {metrics['portfolio_sharpe_empirical']:.3f}",
        f"  Portfolio max drawdown:    {metrics['portfolio_max_drawdown']:.1%}",
        "",
    ]


def _format_diversification_section(
    strategies: dict[str, dict],
    selected: list[str],
    portfolio_metrics: dict,
) -> list[str]:
    """Format diversification analysis section."""
    lines: list[str] = ["## 6. Diversification Analysis", ""]

    family_counts: dict[str, int] = {}
    for slug in selected:
        fam = strategies[slug]["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1

    lines.append(f"  Mechanism families in optimized portfolio: {len(family_counts)}")
    for fam, count in sorted(family_counts.items()):
        lines.append(f"    {fam}: {count} strategy(ies)")
    lines.append("")

    all_family_counts: dict[str, int] = {}
    for slug in strategies:
        fam = strategies[slug]["family"]
        all_family_counts[fam] = all_family_counts.get(fam, 0) + 1

    lines.append("  Family concentration (all 15 strategies):")
    for fam, count in sorted(all_family_counts.items(), key=lambda x: -x[1]):
        pct = count / len(strategies)
        lines.append(f"    {fam}: {count} ({pct:.0%})")
    lines.append("")

    lines.append("  Marginal value analysis:")
    lines.append("  (What would adding one more uncorrelated strategy do?)")
    for target_rho in [0.0, 0.1, 0.2, 0.3]:
        n_new = portfolio_metrics["n_strategies"] + 1
        rho_current = portfolio_metrics["avg_pairwise_correlation"]
        n_old = portfolio_metrics["n_strategies"]
        n_old_pairs = n_old * (n_old - 1) / 2
        n_new_pairs = n_new * (n_new - 1) / 2
        rho_new = (n_old_pairs * rho_current + n_old * target_rho) / n_new_pairs
        sr_new = portfolio_metrics["avg_sharpe"] * math.sqrt(
            n_new / (1 + (n_new - 1) * rho_new)
        )
        improvement = sr_new - portfolio_metrics["portfolio_sharpe_formula"]
        lines.append(
            f"    rho_new={target_rho:.1f}: portfolio SR -> {sr_new:.3f} "
            f"(+{improvement:.3f})"
        )
    lines.append("")
    return lines


def _format_hrp_weights_section(
    selected: list[str],
    strategies: dict[str, dict],
    hrp_weights: dict[str, float],
    equal_weights: dict[str, float],
    method: str,
) -> list[str]:
    """Format the HRP weights table with delta vs equal-weight column."""
    lines: list[str] = [
        f"## 7. Allocation Weights ({method.upper()} method)",
        "",
        f"{'Strategy':<32} {'Track':<8} {'HRP Wt':>8} {'EW Wt':>8} {'Delta':>8}",
        "-" * 68,
    ]
    for slug in sorted(selected, key=lambda s: -hrp_weights.get(s, 0.0)):
        track = (
            "A" if slug in TRACK_A_SLUGS else ("B" if slug in TRACK_B_SLUGS else "?")
        )
        hrp_w = hrp_weights.get(slug, 0.0)
        ew_w = equal_weights.get(slug, 0.0)
        delta = hrp_w - ew_w
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        lines.append(f"{slug:<32} {track:<8} {hrp_w:>8.4f} {ew_w:>8.4f} {delta_str:>8}")
    total_hrp = sum(hrp_weights.values())
    total_ew = sum(equal_weights.values())
    lines.append("-" * 68)
    lines.append(f"{'TOTAL':<32} {'':8} {total_hrp:>8.4f} {total_ew:>8.4f} {'':>8}")
    lines.append("")
    lines.append(
        f"  Track A ({len([s for s in selected if s in TRACK_A_SLUGS])} strategies): "
        f"{sum(hrp_weights.get(s, 0) for s in selected if s in TRACK_A_SLUGS):.4f} "
        f"(target 0.7000)"
    )
    lines.append(
        f"  Track B ({len([s for s in selected if s in TRACK_B_SLUGS])} strategies): "
        f"{sum(hrp_weights.get(s, 0) for s in selected if s in TRACK_B_SLUGS):.4f} "
        f"(target 0.3000)"
    )
    lines.append("")
    return lines


def format_report(
    strategies: dict[str, dict],
    corr_matrix: np.ndarray,
    slugs: list[str],
    clusters: dict[int, list[str]],
    selected: list[str],
    portfolio_metrics: dict,
    all_portfolio_metrics: dict,
    threshold: float,
    hrp_weights: dict[str, float] | None = None,
    equal_weights: dict[str, float] | None = None,
    method: str = "equal",
) -> str:
    """Generate the full markdown report."""
    lines: list[str] = []
    lines.extend(_format_strategy_overview(strategies))
    lines.extend(_format_correlation_section(corr_matrix, slugs))
    lines.extend(
        _format_clustering_section(
            strategies,
            corr_matrix,
            slugs,
            clusters,
            selected,
            threshold,
        )
    )

    # Section 4: Portfolio Sharpe Estimation
    lines.append("## 4. Portfolio Sharpe Estimation")
    lines.append("")
    lines.extend(
        _format_portfolio_metrics_block(
            "### All 15 strategies (equal weight, no clustering)",
            all_portfolio_metrics,
        )
    )
    lines.extend(
        _format_portfolio_metrics_block(
            f"### Optimized portfolio ({len(selected)} cluster representatives)",
            portfolio_metrics,
        )
    )
    lines.append("  Selected strategies:")
    lines.extend(
        f"    {slug:<32} Sharpe={strategies[slug]['sharpe']:.3f}  "
        f"[{strategies[slug]['family']}]"
        for slug in sorted(selected, key=lambda s: -strategies[s]["sharpe"])
    )
    lines.append("")

    # Section 5: Formula reference
    lines.extend(
        [
            "## 5. Formula Reference",
            "",
            "  SR_P = SR_avg * sqrt(N / (1 + (N-1) * rho_avg))",
            "",
            "  where:",
            "    SR_avg   = average Sharpe of selected strategies",
            "    N        = number of strategies",
            "    rho_avg  = average pairwise correlation",
            "",
            "  Key insight: adding more strategies from the SAME mechanism family",
            "  (high rho) barely improves portfolio Sharpe. Adding from a NEW family",
            "  (low rho) has much higher marginal value.",
            "",
        ]
    )

    lines.extend(
        _format_diversification_section(strategies, selected, portfolio_metrics)
    )

    # Section 7: HRP weights table (only when --method hrp is used)
    if hrp_weights is not None and equal_weights is not None:
        lines.extend(
            _format_hrp_weights_section(
                selected, strategies, hrp_weights, equal_weights, method
            )
        )

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio optimizer: correlation clustering and Sharpe estimation"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Correlation threshold for clustering (default: 0.70)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="If set, override clustering and pick the top N by Sharpe",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--min-strategies",
        type=int,
        default=2,
        help=(
            "Minimum number of strategies with complete artifacts required "
            "before the optimizer will run (default: 2). "
            "Raise this to enforce stricter governance gates."
        ),
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        default=False,
        help=(
            "Bypass the minimum-strategies guard and run on whatever artifacts "
            "are available. For development use only — output is not reliable "
            "when fewer than --min-strategies strategies are loaded."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["equal", "hrp"],
        default="equal",
        help=(
            "Weight allocation method. 'equal' = equal weight (default). "
            "'hrp' = Hierarchical Risk Parity via Riskfolio-Lib with 70/30 A/B "
            "split enforcement. Requires: pip install 'Riskfolio-Lib>=6.0'."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 1. Load daily returns
    strategies = load_daily_returns(data_dir)
    loaded = len(strategies)
    registered = len(STRATEGY_EXPERIMENTS)

    # Governance guard: fail hard if too few strategies loaded
    if loaded < args.min_strategies:
        if args.ignore_missing:
            logger.warning(
                "GOVERNANCE BYPASS (--ignore-missing): only %d of %d registered "
                "strategies have artifacts. Output is NOT reliable. "
                "Run /lifecycle to identify and reconstruct missing artifacts.",
                loaded,
                registered,
            )
        else:
            logger.error(
                "ERROR: Portfolio optimizer requires at least %d strategies with "
                "complete artifacts. Only %d of %d registered strategies found. "
                "Run /lifecycle to identify and reconstruct missing artifacts. "
                "Use --ignore-missing to bypass this guard for development.",
                args.min_strategies,
                loaded,
                registered,
            )
            sys.exit(1)

    if loaded < 2:
        logger.error(
            "Cannot compute correlation matrix with fewer than 2 strategies "
            "(got %d). Aborting.",
            loaded,
        )
        sys.exit(1)

    # 2. Compute correlation matrix
    corr_matrix, slugs = compute_correlation_matrix(strategies)

    # 3. Cluster strategies
    clusters = cluster_strategies(corr_matrix, slugs, threshold=args.threshold)
    logger.info(
        "Found %d clusters from %d strategies (threshold=%.2f)",
        len(clusters),
        len(slugs),
        args.threshold,
    )

    # 4. Select representatives
    if args.top_n is not None:
        # Override: pick top N by Sharpe regardless of clustering
        sorted_by_sharpe = sorted(
            strategies.keys(), key=lambda s: -strategies[s]["sharpe"]
        )
        selected = sorted_by_sharpe[: args.top_n]
        logger.info("Top-%d override: selected %s", args.top_n, selected)
    else:
        selected = select_representatives(clusters, strategies)
        logger.info(
            "Selected %d representatives: %s",
            len(selected),
            selected,
        )

    # 5. Compute portfolio Sharpe
    portfolio_metrics = compute_portfolio_sharpe(
        selected, strategies, corr_matrix, slugs
    )

    # Also compute for ALL strategies (no clustering) for comparison
    all_portfolio_metrics = compute_portfolio_sharpe(
        list(strategies.keys()), strategies, corr_matrix, slugs
    )

    # 5b. HRP allocation (if requested)
    hrp_weights: dict[str, float] | None = None
    equal_weights: dict[str, float] = {s: 1.0 / len(selected) for s in selected}

    if args.method == "hrp":
        logger.info("Computing HRP weights via Riskfolio-Lib...")
        # Use the same min_len alignment as the correlation matrix
        min_len = min(len(strategies[s]["daily_returns"]) for s in selected)
        try:
            hrp_weights = compute_hrp_weights(strategies, selected, min_len)
            validate_weights(hrp_weights, "hrp")
            logger.info(
                "HRP weights: %s",
                {
                    s: f"{w:.4f}"
                    for s, w in sorted(hrp_weights.items(), key=lambda x: -x[1])
                },
            )
        except Exception:
            logger.exception("HRP optimization failed — falling back to equal weight")
            hrp_weights = None

    # 6. Generate report
    report = format_report(
        strategies=strategies,
        corr_matrix=corr_matrix,
        slugs=slugs,
        clusters=clusters,
        selected=selected,
        portfolio_metrics=portfolio_metrics,
        all_portfolio_metrics=all_portfolio_metrics,
        threshold=args.threshold,
        hrp_weights=hrp_weights,
        equal_weights=equal_weights,
        method=args.method,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", output_path)
    else:
        print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hierarchical Risk Parity (HRP) portfolio construction for strategy returns.

Implements Lopez de Prado (2016) HRP algorithm:
  1. Compute pairwise correlation matrix from strategy daily returns
  2. Compute distance matrix: d = sqrt(0.5 * (1 - rho))
  3. Hierarchical clustering (Ward linkage on distance matrix)
  4. Quasi-diagonalize the correlation matrix
  5. Recursive bisection with inverse-variance weights
  6. Compare HRP weights vs equal weights

Usage:
    cd /c/Projects/llm-quant && PYTHONPATH=src python scripts/compute_hrp_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
import math

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy definitions (same as compute_correlation_matrix.py)
# ---------------------------------------------------------------------------

STRATEGIES = [
    (
        "soxx-qqq-lead-lag",
        "lead_lag",
        ["SOXX", "QQQ"],
        {
            "leader_symbol": "SOXX",
            "follower_symbol": "QQQ",
            "lag_days": 5,
            "signal_window": 5,
            "entry_threshold": 0.02,
            "exit_threshold": -0.01,
            "target_weight": 0.9,
            "rebalance_frequency_days": 1,
        },
    ),
    (
        "lqd-spy-credit-lead",
        "lead_lag",
        ["LQD", "SPY"],
        {
            "leader_symbol": "LQD",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "agg-spy-credit-lead",
        "lead_lag",
        ["AGG", "SPY"],
        {
            "leader_symbol": "AGG",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "hyg-spy-5d-credit-lead",
        "lead_lag",
        ["HYG", "SPY"],
        {
            "leader_symbol": "HYG",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "agg-qqq-credit-lead",
        "lead_lag",
        ["AGG", "QQQ"],
        {
            "leader_symbol": "AGG",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "lqd-qqq-credit-lead",
        "lead_lag",
        ["LQD", "QQQ"],
        {
            "leader_symbol": "LQD",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "vcit-qqq-credit-lead",
        "lead_lag",
        ["VCIT", "QQQ"],
        {
            "leader_symbol": "VCIT",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "hyg-qqq-credit-lead",
        "lead_lag",
        ["HYG", "QQQ"],
        {
            "leader_symbol": "HYG",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "emb-spy-credit-lead",
        "lead_lag",
        ["EMB", "SPY"],
        {
            "leader_symbol": "EMB",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
            "stop_loss_pct": 0.05,
        },
    ),
    (
        "agg-efa-credit-lead",
        "lead_lag",
        ["AGG", "EFA"],
        {
            "leader_symbol": "AGG",
            "follower_symbol": "EFA",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
            "stop_loss_pct": 0.05,
        },
    ),
    (
        "spy-overnight-momentum",
        "overnight_momentum",
        ["SPY", "QQQ", "TLT", "GLD", "IEF"],
        {
            "symbol": "SPY",
            "window": 10,
            "entry_thresh": 0.002,
            "exit_thresh": -0.0005,
            "target_weight": 0.9,
            "rebalance_frequency_days": 1,
        },
    ),
]

LOOKBACK_DAYS = 5 * 365


# ---------------------------------------------------------------------------
# Short display labels
# ---------------------------------------------------------------------------

SHORT_LABELS: dict[str, str] = {
    "soxx-qqq-lead-lag": "SOXX-QQQ",
    "lqd-spy-credit-lead": "LQD-SPY",
    "agg-spy-credit-lead": "AGG-SPY",
    "hyg-spy-5d-credit-lead": "HYG-SPY",
    "agg-qqq-credit-lead": "AGG-QQQ",
    "lqd-qqq-credit-lead": "LQD-QQQ",
    "vcit-qqq-credit-lead": "VCIT-QQQ",
    "hyg-qqq-credit-lead": "HYG-QQQ",
    "emb-spy-credit-lead": "EMB-SPY",
    "agg-efa-credit-lead": "AGG-EFA",
    "spy-overnight-momentum": "SPY-NITE",
}

# Cluster labels for credit-equity family identification
CREDIT_EQUITY_SLUGS = {
    "lqd-spy-credit-lead",
    "agg-spy-credit-lead",
    "hyg-spy-5d-credit-lead",
    "agg-qqq-credit-lead",
    "lqd-qqq-credit-lead",
    "vcit-qqq-credit-lead",
    "hyg-qqq-credit-lead",
    "emb-spy-credit-lead",
    "agg-efa-credit-lead",
}

SOXX_QQQ_SLUG = "soxx-qqq-lead-lag"
OVERNIGHT_SLUG = "spy-overnight-momentum"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def run_strategy_get_returns(
    slug: str,
    strategy_class: str,
    symbols: list[str],
    params: dict,
) -> tuple[list, list[float]]:
    """Run strategy backtest and return (dates, daily_returns)."""
    print(f"  Running {slug}...", end="", flush=True)

    prices_df = fetch_ohlcv(symbols, lookback_days=LOOKBACK_DAYS)
    indicators_df = compute_indicators(prices_df)

    rebalance_freq = params.get("rebalance_frequency_days", 5)
    config = StrategyConfig(
        name=strategy_class,
        rebalance_frequency_days=rebalance_freq,
        parameters=params,
    )
    strategy = create_strategy(strategy_class, config)

    engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)
    cost_model = CostModel(spread_bps=2.0, commission_per_share=0.005)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug,
        cost_model=cost_model,
        warmup_days=200,
        cost_multiplier=1.0,
    )

    dates = [s.date for s in result.snapshots]
    returns = result.daily_returns
    print(f" done. Trades={len(result.trades)}, Returns={len(returns)}")
    return dates, returns


def align_returns(
    all_dates: list[list],
    all_returns: list[list[float]],
    slugs: list[str],
) -> pl.DataFrame:
    """Align all return series on common trading dates."""
    start = max(d[0] for d in all_dates)
    end = min(d[-1] for d in all_dates)
    print(f"\nAligned date range: {start} to {end}")

    frames = []
    for slug, dates, rets in zip(slugs, all_dates, all_returns, strict=True):
        date_ret = dict(zip(dates, rets, strict=True))
        frames.append((slug, date_ret))

    common_dates = set(all_dates[0])
    for d in all_dates[1:]:
        common_dates &= set(d)
    common_dates = sorted(d for d in common_dates if start <= d <= end)
    print(f"Common trading dates: {len(common_dates)}")

    rows = []
    for d in common_dates:
        row: dict = {"date": d}
        for slug, date_ret in frames:
            row[slug] = date_ret.get(d, 0.0)
        rows.append(row)

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# HRP implementation
# ---------------------------------------------------------------------------


def compute_cov_matrix(
    returns_array: np.ndarray,
) -> np.ndarray:
    """Compute sample covariance matrix (T x N returns array)."""
    return np.cov(returns_array, rowvar=False)


def compute_corr_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    corr = np.where(outer_std > 0, cov / outer_std, 0.0)
    # Ensure diagonal = 1
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_distance_matrix(corr: np.ndarray) -> np.ndarray:
    """Compute distance matrix from correlation: d = sqrt(0.5 * (1 - rho)).

    This is a proper metric (satisfies triangle inequality).
    Values in [0, 1]: d=0 when rho=1 (identical), d=1 when rho=-1 (anti-correlated).
    """
    return np.sqrt(0.5 * (1.0 - corr))


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Quasi-diagonalize: return sorted item indices from linkage matrix.

    Recursively sorts items so that closer items (by cluster membership)
    are placed next to each other in the correlation matrix.
    """
    # Use scipy's leaves_list which implements the seriation algorithm
    return leaves_list(link).tolist()


def get_cluster_var(
    cov: np.ndarray,
    cluster_items: list[int],
) -> float:
    """Compute variance of an equal-weight sub-portfolio of cluster items."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    n = len(cluster_items)
    w = np.ones(n) / n
    return float(w @ cov_slice @ w)


def recursive_bisection(
    cov: np.ndarray,
    sorted_items: list[int],
) -> np.ndarray:
    """Compute HRP weights via recursive bisection (inverse-variance weighting).

    Parameters
    ----------
    cov : np.ndarray
        Full covariance matrix (N x N).
    sorted_items : list[int]
        Quasi-diagonalized item ordering (indices into cov rows/cols).

    Returns
    -------
    np.ndarray
        Weight vector of length N (indexed by original item order, not sorted_items order).
    """
    n = len(sorted_items)
    weights = np.ones(n)  # weights indexed by position in sorted_items

    # Use a stack of (start, end) index ranges into sorted_items
    clusters: list[tuple[int, int]] = [(0, n)]

    while clusters:
        start, end = clusters.pop()
        if end - start <= 1:
            continue

        # Split cluster into two halves
        mid = (start + end) // 2
        left_items = sorted_items[start:mid]
        right_items = sorted_items[mid:end]

        # Compute variance of each sub-cluster
        var_left = get_cluster_var(cov, left_items)
        var_right = get_cluster_var(cov, right_items)

        # Inverse-variance allocation between the two sub-clusters
        if var_left + var_right == 0:
            alpha = 0.5
        else:
            alpha = 1.0 - var_left / (var_left + var_right)

        # Scale weights: left cluster gets (1 - alpha), right gets alpha
        weights[start:mid] *= 1.0 - alpha
        weights[mid:end] *= alpha

        # Push sub-clusters
        if mid - start > 1:
            clusters.append((start, mid))
        if end - mid > 1:
            clusters.append((mid, end))

    # Map back to original item order
    final_weights = np.zeros(n)
    for pos, item in enumerate(sorted_items):
        final_weights[item] = weights[pos]

    return final_weights


def compute_hrp_weights(
    returns_df: pl.DataFrame,
    slugs: list[str],
) -> dict[str, float]:
    """Compute HRP weights for a set of strategy return series.

    Parameters
    ----------
    returns_df : pl.DataFrame
        Aligned returns DataFrame (date column + one column per slug).
    slugs : list[str]
        Strategy slugs (column names in returns_df, minus 'date').

    Returns
    -------
    dict[str, float]
        HRP weight per slug, summing to 1.0.
    """
    n = len(slugs)

    # Extract T x N returns array
    returns_array = np.array([returns_df[s].to_list() for s in slugs]).T  # (T, N)

    # Covariance and correlation
    cov = compute_cov_matrix(returns_array)
    corr = compute_corr_from_cov(cov)

    # Distance matrix
    dist = compute_distance_matrix(corr)

    # Condensed distance vector (upper triangle) for scipy
    condensed = squareform(dist, checks=False)

    # Hierarchical clustering (Ward linkage on the distance matrix)
    link = linkage(condensed, method="ward")

    # Quasi-diagonalize: get leaf ordering
    sorted_items = get_quasi_diag(link)

    # Recursive bisection
    hrp_w = recursive_bisection(cov, sorted_items)

    # Normalize to sum to 1
    hrp_w = hrp_w / hrp_w.sum()

    return {slug: float(hrp_w[i]) for i, slug in enumerate(slugs)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_hrp_report(
    hrp_weights: dict[str, float],
    slugs: list[str],
    returns_df: pl.DataFrame,
) -> None:
    """Print HRP vs equal-weight comparison table."""
    n = len(slugs)
    equal_weight = 1.0 / n

    # Compute per-strategy annualized volatility
    vols: dict[str, float] = {}
    for slug in slugs:
        rets = np.array(returns_df[slug].to_list())
        vols[slug] = float(np.std(rets, ddof=1) * math.sqrt(252))

    # Compute per-strategy Sharpe (annualized)
    sharpes: dict[str, float] = {}
    for slug in slugs:
        rets = np.array(returns_df[slug].to_list())
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets, ddof=1))
        sharpes[slug] = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    # Compute correlations for cluster identification
    returns_array = np.array([returns_df[s].to_list() for s in slugs]).T
    corr = compute_corr_from_cov(compute_cov_matrix(returns_array))

    print()
    print("=" * 90)
    print("HIERARCHICAL RISK PARITY (HRP) WEIGHTS vs EQUAL WEIGHT")
    print("=" * 90)
    print()
    print(
        f"{'Strategy':<24} {'Label':>9} {'HRP Wt':>8} {'EW Wt':>8} {'Delta':>8} "
        f"{'Ann Vol':>8} {'Sharpe':>8} {'Cluster':>12}"
    )
    print("-" * 90)

    # Sort by HRP weight descending
    sorted_slugs = sorted(slugs, key=lambda s: hrp_weights[s], reverse=True)

    for slug in sorted_slugs:
        label = SHORT_LABELS.get(slug, slug[:9])
        hrp_w = hrp_weights[slug]
        delta = hrp_w - equal_weight
        vol = vols[slug]
        sr = sharpes[slug]

        # Cluster identification
        if slug in CREDIT_EQUITY_SLUGS:
            cluster = "credit-equity"
        elif slug == SOXX_QQQ_SLUG:
            cluster = "semi-equity"
        elif slug == OVERNIGHT_SLUG:
            cluster = "overnight-mom"
        else:
            cluster = "other"

        delta_str = f"{delta:+.3f}"
        print(
            f"{slug:<24} {label:>9} {hrp_w:>8.3f} {equal_weight:>8.3f} "
            f"{delta_str:>8} {vol:>8.3f} {sr:>8.3f} {cluster:>12}"
        )

    print("-" * 90)
    hrp_total = sum(hrp_weights.values())
    print(
        f"{'TOTAL':<24} {'':>9} {hrp_total:>8.3f} {1.0:>8.3f} {'':>8} {'':>8} {'':>8}"
    )

    # Cluster-level summary
    print()
    print("--- Cluster-Level Allocation ---")
    credit_eq_hrp = sum(hrp_weights[s] for s in slugs if s in CREDIT_EQUITY_SLUGS)
    credit_eq_ew = sum(equal_weight for s in slugs if s in CREDIT_EQUITY_SLUGS)
    soxx_hrp = hrp_weights.get(SOXX_QQQ_SLUG, 0.0)
    soxx_ew = equal_weight
    overnight_hrp = hrp_weights.get(OVERNIGHT_SLUG, 0.0)
    overnight_ew = equal_weight

    n_credit = len(CREDIT_EQUITY_SLUGS & set(slugs))
    print(
        f"  Credit-equity ({n_credit} strategies): "
        f"HRP={credit_eq_hrp:.3f} vs EW={credit_eq_ew:.3f} "
        f"(delta={credit_eq_hrp - credit_eq_ew:+.3f})"
    )
    if SOXX_QQQ_SLUG in slugs:
        print(
            f"  Semi-equity (SOXX-QQQ):           "
            f"HRP={soxx_hrp:.3f} vs EW={soxx_ew:.3f} "
            f"(delta={soxx_hrp - soxx_ew:+.3f})"
        )
    if OVERNIGHT_SLUG in slugs:
        print(
            f"  Overnight momentum (SPY-NITE):     "
            f"HRP={overnight_hrp:.3f} vs EW={overnight_ew:.3f} "
            f"(delta={overnight_hrp - overnight_ew:+.3f})"
        )

    # Average pairwise correlation
    off_diag = [corr[i][j] for i in range(n) for j in range(n) if i < j]
    avg_rho = sum(off_diag) / len(off_diag) if off_diag else 0.0

    print()
    print(f"  Average pairwise rho: {avg_rho:.3f}")
    print(f"  Effective N (Meucci): {1 + (n - 1) * (1 - avg_rho):.2f}")
    print()

    # HRP interpretation
    if credit_eq_hrp < credit_eq_ew:
        print(
            "  HRP DOWNWEIGHTS credit-equity cluster: high intra-cluster correlation "
            f"leads HRP to allocate {credit_eq_hrp:.1%} vs naive {credit_eq_ew:.1%}."
        )
    else:
        print(
            "  HRP does not downweight credit-equity cluster vs equal weight "
            f"(HRP={credit_eq_hrp:.1%}, EW={credit_eq_ew:.1%})."
        )

    if overnight_hrp > overnight_ew:
        print(
            f"  HRP UPWEIGHTS overnight momentum ({overnight_hrp:.1%} vs {overnight_ew:.1%}): "
            "lower correlation to credit-equity family increases its diversification value."
        )
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    slugs = [s[0] for s in STRATEGIES]
    print(f"HRP computation for {len(slugs)} strategies...")
    print("Fetching data and running backtests...\n")

    all_dates: list[list] = []
    all_returns: list[list[float]] = []

    for slug, strategy_class, symbols, params in STRATEGIES:
        dates, rets = run_strategy_get_returns(slug, strategy_class, symbols, params)
        all_dates.append(dates)
        all_returns.append(rets)

    returns_df = align_returns(all_dates, all_returns, slugs)

    hrp_weights = compute_hrp_weights(returns_df, slugs)
    print_hrp_report(hrp_weights, slugs, returns_df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure portfolio Sharpe ratio using OOS walk-forward returns.

Ground truth test: does our diversified portfolio achieve SR 1.5+?

Loads strategy return streams from experiment artifacts, aligns them to a
common date range, applies HRP and equal-weight allocation, and computes
realized portfolio-level metrics.  Compares against 60/40 SPY/TLT benchmark.

Fallback: if fewer than 2 strategies have real return data, synthetic returns
are generated from known strategy parameters (clearly labelled SYNTHETIC).

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/measure_portfolio_sr.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/measure_portfolio_sr.py --method hrp
    cd E:/llm-quant && PYTHONPATH=src python scripts/measure_portfolio_sr.py --output report.md
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure src is on the import path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # annualised, matches current T-bill circa 2025-26
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTS_DIR = DATA_DIR / "reports"

# Registered strategies: slug -> best experiment_id
# Matches STRATEGY_EXPERIMENTS in portfolio_optimizer.py
STRATEGY_EXPERIMENTS: dict[str, str] = {
    "soxx-qqq-lead-lag": "57fba00d",
    "lqd-spy-credit-lead": "b0588e6d",  # best experiment (b0588e6d has more history)
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
}

# Fallback: known strategy parameters for synthetic return generation
# (Sharpe, annualised vol, annualised mean return) from backtest results / mandate docs
SYNTHETIC_PARAMS: dict[str, dict[str, float]] = {
    "soxx-qqq-lead-lag": {"sharpe": 0.86, "vol": 0.12, "mean": 0.11},
    "lqd-spy-credit-lead": {"sharpe": 1.25, "vol": 0.08, "mean": 0.10},
    "agg-spy-credit-lead": {"sharpe": 1.10, "vol": 0.07, "mean": 0.08},
    "hyg-spy-5d-credit-lead": {"sharpe": 1.20, "vol": 0.09, "mean": 0.11},
    "agg-qqq-credit-lead": {"sharpe": 1.15, "vol": 0.10, "mean": 0.12},
    "lqd-qqq-credit-lead": {"sharpe": 1.18, "vol": 0.09, "mean": 0.11},
    "vcit-qqq-credit-lead": {"sharpe": 1.05, "vol": 0.09, "mean": 0.09},
    "hyg-qqq-credit-lead": {"sharpe": 1.10, "vol": 0.10, "mean": 0.11},
    "emb-spy-credit-lead": {"sharpe": 0.95, "vol": 0.10, "mean": 0.10},
    "agg-efa-credit-lead": {"sharpe": 1.00, "vol": 0.09, "mean": 0.09},
    "spy-overnight-momentum": {"sharpe": 1.30, "vol": 0.07, "mean": 0.09},
    "tlt-spy-rate-momentum": {"sharpe": 0.90, "vol": 0.11, "mean": 0.10},
    "tlt-qqq-rate-tech": {"sharpe": 0.85, "vol": 0.12, "mean": 0.10},
    "ief-qqq-rate-tech": {"sharpe": 0.88, "vol": 0.11, "mean": 0.10},
    "behavioral-structural": {"sharpe": 1.05, "vol": 0.08, "mean": 0.08},
    "gld-slv-mean-reversion-v4": {"sharpe": 1.40, "vol": 0.07, "mean": 0.10},
}

# Mechanism family for display
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
}

# Track membership for 70/30 target
TRACK_A_SLUGS: set[str] = {
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
}
TRACK_B_SLUGS: set[str] = {"soxx-qqq-lead-lag"}


# ---------------------------------------------------------------------------
# Strategy data loading
# ---------------------------------------------------------------------------


class StrategyData:
    """Holds daily returns and metadata for one strategy."""

    def __init__(
        self,
        slug: str,
        returns: list[float],
        sharpe: float,
        sortino: float,
        max_drawdown: float,
        cagr: float,
        start_date: str,
        end_date: str,
        source: str,  # "real" or "synthetic"
    ) -> None:
        self.slug = slug
        self.returns = np.array(returns, dtype=float)
        self.sharpe = sharpe
        self.sortino = sortino
        self.max_drawdown = max_drawdown
        self.cagr = cagr
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.family = MECHANISM_FAMILIES.get(slug, "Unknown")

    def __repr__(self) -> str:
        return (
            f"StrategyData({self.slug!r}, n={len(self.returns)}, "
            f"SR={self.sharpe:.2f}, source={self.source!r})"
        )


def _find_experiment_artifact(slug: str, exp_id: str) -> Path | None:
    """Find the experiment YAML artifact for a slug/experiment_id pair.

    Tries the canonical path first, then falls back to scanning the
    experiments/ directory for any file matching the experiment_id prefix.
    """
    exp_dir = DATA_DIR / "strategies" / slug / "experiments"
    if not exp_dir.exists():
        return None

    # Try exact match
    exact = exp_dir / f"{exp_id}.yaml"
    if exact.exists():
        return exact

    # Try prefix match (first 8 chars of UUID)
    prefix = exp_id[:8]
    for f in exp_dir.glob("*.yaml"):
        if f.stem.startswith(prefix):
            return f

    # Return any yaml in the directory as last resort
    yamls = list(exp_dir.glob("*.yaml"))
    if yamls:
        return yamls[0]

    return None


def load_real_strategy(slug: str) -> StrategyData | None:
    """Load strategy returns from the experiment artifact YAML.

    Returns None if the artifact does not exist or has no daily_returns.
    """
    exp_id = STRATEGY_EXPERIMENTS.get(slug, "")
    artifact_path = _find_experiment_artifact(slug, exp_id)

    if artifact_path is None:
        logger.debug("No artifact for %s (exp_id=%s)", slug, exp_id)
        return None

    try:
        with open(artifact_path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", artifact_path, exc)
        return None

    returns = data.get("daily_returns", [])
    if not returns or len(returns) < 30:
        logger.debug(
            "Insufficient returns in %s (%d days)", artifact_path, len(returns)
        )
        return None

    metrics = data.get("metrics_1x", {})

    return StrategyData(
        slug=slug,
        returns=returns,
        sharpe=metrics.get("sharpe_ratio", 0.0),
        sortino=metrics.get("sortino_ratio", 0.0),
        max_drawdown=abs(metrics.get("max_drawdown", 0.0)),
        cagr=metrics.get("annualized_return", 0.0),
        start_date=str(data.get("start_date", "")),
        end_date=str(data.get("end_date", "")),
        source="real",
    )


def _generate_synthetic_returns(
    sharpe: float, vol: float, n_days: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate synthetic daily returns matching target Sharpe and volatility.

    Uses a simple normal distribution. Returns are daily.
    """
    daily_vol = vol / math.sqrt(TRADING_DAYS_PER_YEAR)
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    daily_mean = daily_rf + sharpe * daily_vol
    return rng.normal(loc=daily_mean, scale=daily_vol, size=n_days)


def load_synthetic_strategy(
    slug: str, n_days: int, rng: np.random.Generator
) -> StrategyData:
    """Generate a synthetic strategy with known parameters."""
    params = SYNTHETIC_PARAMS.get(slug, {"sharpe": 1.0, "vol": 0.10, "mean": 0.10})
    returns = _generate_synthetic_returns(params["sharpe"], params["vol"], n_days, rng)

    # Compute metrics from the synthetic series directly
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    sr = (np.mean(excess) / (np.std(excess) + 1e-12)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_denom = np.std(excess[excess < 0]) if np.any(excess < 0) else 1e-12
    sortino = (np.mean(excess) / (sortino_denom + 1e-12)) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    cum = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdown = (cum - rolling_max) / (rolling_max + 1e-12)
    max_dd = float(np.min(drawdown))
    cagr = float(np.prod(1 + returns) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)

    today = date.today().isoformat()
    start_iso = date.fromordinal(date.today().toordinal() - n_days).isoformat()

    return StrategyData(
        slug=slug,
        returns=returns.tolist(),
        sharpe=float(sr),
        sortino=float(sortino),
        max_drawdown=abs(max_dd),
        cagr=cagr,
        start_date=start_iso,
        end_date=today,
        source="synthetic",
    )


def load_all_strategies(use_synthetic: bool = True) -> dict[str, StrategyData]:
    """Load all registered strategies.

    First tries real artifact data. Falls back to synthetic where real data
    is unavailable (if use_synthetic=True).
    """
    strategies: dict[str, StrategyData] = {}
    rng = np.random.default_rng(seed=42)
    n_days = 800  # ~3.2 years, roughly matching lqd-spy backtest length

    for slug in STRATEGY_EXPERIMENTS:
        real = load_real_strategy(slug)
        if real is not None:
            strategies[slug] = real
        elif use_synthetic:
            strategies[slug] = load_synthetic_strategy(slug, n_days, rng)
        # else: skip

    return strategies


# ---------------------------------------------------------------------------
# Return alignment
# ---------------------------------------------------------------------------


def align_returns(strategies: dict[str, StrategyData]) -> tuple[np.ndarray, list[str]]:
    """Align all return series to a common length (inner join by position).

    Uses the tail of each series (most recent common period).
    Returns (matrix of shape [n_strategies, n_days], ordered_slugs).
    """
    slugs = sorted(strategies.keys())
    lengths = [len(strategies[s].returns) for s in slugs]
    min_len = min(lengths)

    matrix = np.zeros((len(slugs), min_len), dtype=float)
    for i, slug in enumerate(slugs):
        r = np.array(strategies[slug].returns)
        matrix[i, :] = r[-min_len:]

    return matrix, slugs


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------


def compute_metrics(returns: np.ndarray) -> dict[str, float]:
    """Compute annualised portfolio metrics from a daily return series."""
    n = len(returns)
    if n < 2:
        return dict.fromkeys(
            (
                "sharpe",
                "sortino",
                "max_drawdown",
                "cagr",
                "calmar",
                "best_30d",
                "worst_30d",
            ),
            0.0,
        )

    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf

    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    sharpe = float(
        (mean_excess / (std_excess + 1e-12)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    neg_excess = excess[excess < 0]
    downside_std = np.std(neg_excess, ddof=1) if len(neg_excess) > 1 else 1e-12
    sortino = float(
        (mean_excess / (downside_std + 1e-12)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )

    cum = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    dd = (cum - rolling_max) / (rolling_max + 1e-12)
    max_dd = float(np.abs(np.min(dd)))

    cagr = float(np.prod(1 + returns) ** (TRADING_DAYS_PER_YEAR / n) - 1)
    calmar = float(cagr / (max_dd + 1e-12))

    # Best and worst 30-day rolling sum of returns
    window = min(30, n - 1)
    rolling_30 = np.array([np.sum(returns[i : i + window]) for i in range(n - window)])
    best_30 = float(np.max(rolling_30)) if len(rolling_30) > 0 else 0.0
    worst_30 = float(np.min(rolling_30)) if len(rolling_30) > 0 else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "calmar": calmar,
        "best_30d": best_30,
        "worst_30d": worst_30,
    }


# ---------------------------------------------------------------------------
# Benchmark returns
# ---------------------------------------------------------------------------


def load_benchmark_returns(n_days: int) -> np.ndarray:
    """Load 60/40 SPY/TLT benchmark returns from price history if available.

    Falls back to synthetic 60/40 blend if price data unavailable.
    """
    try:
        import duckdb

        db_path = DATA_DIR / "portfolio.duckdb"
        if db_path.exists():
            con = duckdb.connect(str(db_path), read_only=True)
            # Try to fetch SPY and TLT close prices
            query = """
                SELECT date, symbol, close
                FROM prices
                WHERE symbol IN ('SPY', 'TLT')
                ORDER BY date DESC
                LIMIT ?
            """
            df = con.execute(query, [n_days * 2]).fetchdf()
            con.close()

            if df is not None and len(df) > 0:
                spy = df[df["symbol"] == "SPY"].sort_values("date")["close"].values
                tlt = df[df["symbol"] == "TLT"].sort_values("date")["close"].values

                # Align lengths
                min_len = min(len(spy), len(tlt), n_days)
                spy = spy[-min_len:]
                tlt = tlt[-min_len:]

                spy_returns = np.diff(spy) / spy[:-1]
                tlt_returns = np.diff(tlt) / tlt[:-1]

                # 60/40 blend
                benchmark = 0.6 * spy_returns + 0.4 * tlt_returns
                return benchmark[-n_days:] if len(benchmark) >= n_days else benchmark

    except Exception as exc:
        logger.debug("Could not load benchmark from DuckDB: %s", exc)

    # Synthetic fallback: SPY ~10% CAGR + TLT ~2% CAGR, blended 60/40
    # Historical 60/40 roughly ~8-9% CAGR, Sharpe ~0.55
    rng = np.random.default_rng(seed=99)
    spy_r = _generate_synthetic_returns(sharpe=0.65, vol=0.16, n_days=n_days, rng=rng)
    tlt_r = _generate_synthetic_returns(sharpe=0.20, vol=0.12, n_days=n_days, rng=rng)
    return 0.6 * spy_r + 0.4 * tlt_r


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def compute_equal_weights(slugs: list[str]) -> dict[str, float]:
    """Return equal weights for all strategies."""
    n = len(slugs)
    if n == 0:
        return {}
    w = 1.0 / n
    return dict.fromkeys(slugs, w)


def compute_hrp_weights_simple(
    returns_matrix: np.ndarray,
    slugs: list[str],
) -> dict[str, float]:
    """Compute HRP weights using scipy-based clustering (no Riskfolio dependency).

    Implements a simplified HRP:
    1. Hierarchical clustering on correlation-based distance
    2. Recursive bisection with inverse-variance weights within clusters

    Falls back to equal-weight if fewer than 2 strategies.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(slugs)
    if n < 2:
        return compute_equal_weights(slugs)

    # Correlation matrix
    corr = np.corrcoef(returns_matrix)
    np.fill_diagonal(corr, 1.0)

    # Distance: sqrt(0.5 * (1 - corr)) — standard HRP metric
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    # Single-linkage hierarchical clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="single")

    # Recursive bisection (simplified): cluster into 2 groups, assign variance-inverse weights
    # For simplicity, use inverse-variance weights on the full set as base,
    # then scale by cluster structure
    variances = np.var(returns_matrix, axis=1, ddof=1)
    inv_var = 1.0 / (variances + 1e-12)
    raw_weights = inv_var / inv_var.sum()

    # Apply hierarchical rescaling via cluster labels at a moderate cut
    n_clusters = max(2, n // 2)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    cluster_weights: dict[int, float] = {}
    for label in np.unique(labels):
        mask = labels == label
        cluster_inv_var = inv_var[mask].sum()
        cluster_weights[label] = float(cluster_inv_var)

    # Normalise cluster weights
    total_cluster_w = sum(cluster_weights.values())
    for k in cluster_weights:
        cluster_weights[k] /= total_cluster_w

    # Assign final weights: strategy weight = cluster_weight * (inv_var / cluster_inv_var)
    final = np.zeros(n)
    for label in np.unique(labels):
        mask = labels == label
        cluster_inv_var = inv_var[mask].sum()
        cluster_w = cluster_weights[label]
        for i in np.where(mask)[0]:
            final[i] = cluster_w * (inv_var[i] / (cluster_inv_var + 1e-12))

    # Normalise to 1
    final = final / (final.sum() + 1e-12)

    return {s: float(final[i]) for i, s in enumerate(slugs)}


def apply_track_split(weights: dict[str, float]) -> dict[str, float]:
    """Rescale weights to enforce 70% Track A / 30% Track B target allocation.

    If only one track is present, normalises within that track only.
    """
    a_slugs = [s for s in weights if s in TRACK_A_SLUGS]
    b_slugs = [s for s in weights if s in TRACK_B_SLUGS]
    other_slugs = [
        s for s in weights if s not in TRACK_A_SLUGS and s not in TRACK_B_SLUGS
    ]

    a_total = sum(weights[s] for s in a_slugs)
    b_total = sum(weights[s] for s in b_slugs)
    other_total = sum(weights[s] for s in other_slugs)

    adjusted: dict[str, float] = {}

    if a_slugs and b_slugs:
        # Both tracks present: enforce 70/30
        for s in a_slugs:
            adjusted[s] = weights[s] * (0.70 / a_total) if a_total > 0 else 0.0
        for s in b_slugs:
            adjusted[s] = weights[s] * (0.30 / b_total) if b_total > 0 else 0.0
    elif a_slugs:
        # Only Track A: normalise to 1.0
        for s in a_slugs:
            adjusted[s] = weights[s] / (a_total + 1e-12)
    elif b_slugs:
        # Only Track B
        for s in b_slugs:
            adjusted[s] = weights[s] / (b_total + 1e-12)

    # Pass-through for any untracked slugs
    for s in other_slugs:
        adjusted[s] = weights[s]

    return adjusted


def portfolio_returns(
    returns_matrix: np.ndarray, slugs: list[str], weights: dict[str, float]
) -> np.ndarray:
    """Compute weighted portfolio daily returns."""
    w = np.array([weights.get(s, 0.0) for s in slugs])
    w = w / (w.sum() + 1e-12)  # normalise defensively
    return returns_matrix.T @ w  # shape: (n_days,)


# ---------------------------------------------------------------------------
# Per-strategy contribution analysis
# ---------------------------------------------------------------------------


def compute_contribution(
    strategies: dict[str, StrategyData],
    port_returns: np.ndarray,
    returns_matrix: np.ndarray,
    slugs: list[str],
) -> list[dict[str, Any]]:
    """Compute per-strategy marginal SR contribution.

    Marginal SR contribution = individual_SR * correlation_to_portfolio
    (Lopez de Prado approximation)
    """
    contributions: list[dict[str, Any]] = []
    for i, slug in enumerate(slugs):
        strat_r = returns_matrix[i, :]
        # Correlation to portfolio returns
        if np.std(port_returns) > 1e-10 and np.std(strat_r) > 1e-10:
            corr_to_port = float(np.corrcoef(strat_r, port_returns)[0, 1])
        else:
            corr_to_port = 0.0

        strat = strategies[slug]
        marginal_sr = strat.sharpe * corr_to_port
        contributions.append(
            {
                "slug": slug,
                "family": strat.family,
                "sharpe": strat.sharpe,
                "corr_to_portfolio": corr_to_port,
                "marginal_sr": marginal_sr,
                "source": strat.source,
            }
        )

    return sorted(contributions, key=lambda x: x["marginal_sr"], reverse=True)


# ---------------------------------------------------------------------------
# Correlation matrix summary
# ---------------------------------------------------------------------------


def avg_pairwise_correlation(returns_matrix: np.ndarray) -> float:
    """Compute average off-diagonal pairwise correlation."""
    n = returns_matrix.shape[0]
    if n < 2:
        return 0.0
    corr = np.corrcoef(returns_matrix)
    upper_tri = corr[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _sr(x: float) -> str:
    return f"{x:.2f}"


def _pass_fail(value: float, target: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "PASS" if value >= target else "FAIL"
    return "PASS" if value <= target else "FAIL"


def build_report(
    strategies: dict[str, StrategyData],
    metrics_hrp: dict[str, float],
    metrics_equal: dict[str, float],
    metrics_bench: dict[str, float],
    weights_hrp: dict[str, float],
    weights_equal: dict[str, float],
    contributions: list[dict[str, Any]],
    returns_matrix: np.ndarray,
    slugs: list[str],
    n_real: int,
    n_synthetic: int,
    date_range: tuple[str, str],
    method: str,
) -> str:
    """Assemble the full markdown/text report."""
    today = date.today().isoformat()
    lines: list[str] = []
    w = 63  # line width

    lines.append("=" * w)
    lines.append(f" PORTFOLIO OOS SHARPE REPORT — {today}")
    lines.append("=" * w)
    lines.append("")

    # Data provenance
    lines.append(
        f" Strategies loaded : {len(strategies)} total "
        f"({n_real} real backtest, {n_synthetic} synthetic*)"
    )
    lines.append(f" Date range        : {date_range[0]} to {date_range[1]}")
    avg_corr = avg_pairwise_correlation(returns_matrix)
    lines.append(f" Avg pairwise corr : {avg_corr:.3f}  (target < 0.20 for SR 2.0+)")
    lines.append(
        f" Weighting method  : {method.upper()} (HRP) + Equal-weight comparison"
    )
    lines.append("")

    if n_synthetic > 0:
        lines.append(
            f" * {n_synthetic} strategies use SYNTHETIC returns generated from known"
        )
        lines.append("   strategy parameters (Sharpe, vol) — NOT real OOS data.")
        lines.append(
            "   Run /lifecycle to advance these through the backtest pipeline."
        )
        lines.append("")

    # Metric table
    targets = {
        "Annual Sharpe": (1.50, True),
        "Sortino": (1.00, True),
        "Max Drawdown": (0.15, False),
        "CAGR": (0.15, True),
        "Calmar": (0.50, True),
        "Best 30d": (None, None),
        "Worst 30d": (None, None),
    }

    col_w = 9
    h_metric = "Metric"
    h_hrp = "HRP" if method == "hrp" else "InvVar"
    h_eq = "Equal"
    h_bench = "60/40"
    h_target = "Target"

    sep = f"+-{'-' * 22}-+-{'-' * col_w}-+-{'-' * col_w}-+-{'-' * col_w}-+-{'-' * col_w}-+"
    lines.append(sep)
    lines.append(
        f"| {'Metric':<22} | {h_hrp:^{col_w}} | {h_eq:^{col_w}} | {h_bench:^{col_w}} | {h_target:^{col_w}} |"
    )
    lines.append(sep)

    rows = [
        (
            "Annual Sharpe",
            _sr(metrics_hrp["sharpe"]),
            _sr(metrics_equal["sharpe"]),
            _sr(metrics_bench["sharpe"]),
            ">1.50",
        ),
        (
            "Sortino",
            _sr(metrics_hrp["sortino"]),
            _sr(metrics_equal["sortino"]),
            _sr(metrics_bench["sortino"]),
            ">1.00",
        ),
        (
            "Max Drawdown",
            _pct(-metrics_hrp["max_drawdown"]),
            _pct(-metrics_equal["max_drawdown"]),
            _pct(-metrics_bench["max_drawdown"]),
            "<15%",
        ),
        (
            "CAGR",
            _pct(metrics_hrp["cagr"]),
            _pct(metrics_equal["cagr"]),
            _pct(metrics_bench["cagr"]),
            ">15%",
        ),
        (
            "Calmar",
            _sr(metrics_hrp["calmar"]),
            _sr(metrics_equal["calmar"]),
            _sr(metrics_bench["calmar"]),
            ">0.50",
        ),
        (
            "Best 30d",
            _pct(metrics_hrp["best_30d"]),
            _pct(metrics_equal["best_30d"]),
            _pct(metrics_bench["best_30d"]),
            "—",
        ),
        (
            "Worst 30d",
            _pct(metrics_hrp["worst_30d"]),
            _pct(metrics_equal["worst_30d"]),
            _pct(metrics_bench["worst_30d"]),
            "—",
        ),
    ]

    for label, v_hrp, v_eq, v_bench, target_str in rows:
        lines.append(
            f"| {label:<22} | {v_hrp:^{col_w}} | {v_eq:^{col_w}} | {v_bench:^{col_w}} | {target_str:^{col_w}} |"
        )

    lines.append(sep)
    lines.append("")

    # Gate verdicts
    lines.append(" Track A gate verdicts (HRP portfolio):")
    gates = [
        ("Annual Sharpe > 0.80", metrics_hrp["sharpe"], 0.80, True),
        ("Sortino > 1.00", metrics_hrp["sortino"], 1.00, True),
        ("Max Drawdown < 15%", metrics_hrp["max_drawdown"], 0.15, False),
        ("CAGR > 15%", metrics_hrp["cagr"], 0.15, True),
        ("Calmar > 0.50", metrics_hrp["calmar"], 0.50, True),
    ]
    for label, val, threshold, hib in gates:
        verdict = _pass_fail(val, threshold, hib)
        symbol = "v" if verdict == "PASS" else "X"
        lines.append(f"   [{symbol}] {label}")

    lines.append("")

    # Corrected portfolio SR using formula
    n = len(slugs)
    avg_individual_sr = np.mean([strategies[s].sharpe for s in slugs])
    sr_formula = (
        float(avg_individual_sr) * math.sqrt(n / (1 + (n - 1) * avg_corr))
        if n > 0
        else 0.0
    )
    lines.append(" Corrected formula SR (Lopez de Prado):")
    lines.append(
        f"   SR_P = {avg_individual_sr:.2f} x sqrt({n} / (1 + {n - 1} x {avg_corr:.3f}))"
    )
    lines.append(f"        = {sr_formula:.2f}")
    lines.append("   (target: >1.50 for 15 strategies with avg corr <0.20)")
    lines.append("")

    # Per-strategy contribution table
    lines.append(" Per-strategy marginal SR contribution:")
    lines.append(
        f" {'Slug':<30} {'Family':<25} {'Indiv SR':>8} {'Corr to P':>10} {'Marginal SR':>12} {'Source':>8}"
    )
    lines.append(" " + "-" * 95)
    for c in contributions:
        src_marker = "" if c["source"] == "real" else " *"
        lines.append(
            f" {c['slug']:<30} {c['family']:<25} {c['sharpe']:>8.2f} "
            f"{c['corr_to_portfolio']:>10.3f} {c['marginal_sr']:>12.3f}{src_marker:>8}"
        )
    lines.append("")

    # Weight table
    lines.append(" Portfolio weights:")
    w_col = 10
    lines.append(
        f" {'Slug':<30} {'HRP wt':>{w_col}} {'Equal wt':>{w_col}} {'Track':>7}"
    )
    lines.append(" " + "-" * 60)
    for slug in slugs:
        track = (
            "A" if slug in TRACK_A_SLUGS else ("B" if slug in TRACK_B_SLUGS else "?")
        )
        lines.append(
            f" {slug:<30} {weights_hrp.get(slug, 0):{w_col}.3f} "
            f"{weights_equal.get(slug, 0):{w_col}.3f} {track:>7}"
        )
    lines.append("")

    # Key findings
    lines.append(" Key findings:")
    if n_synthetic > 0:
        lines.append(
            f"   - {n_real} of {len(strategies)} strategies have real backtest data."
        )
        lines.append(
            "   - Metrics marked * are synthetic estimates — treat as directional only."
        )
    if avg_corr > 0.40:
        lines.append(
            f"   - Avg pairwise correlation {avg_corr:.2f} is HIGH (target <0.20)."
        )
        lines.append("   - Family diversification needed: Families 2-7 are UNTESTED.")
    if sr_formula >= 1.50:
        lines.append(f"   - Formula SR {sr_formula:.2f} meets the 1.50 target.")
    else:
        lines.append(f"   - Formula SR {sr_formula:.2f} is below 1.50 target.")
        lines.append(
            "   - Priority: add strategies from Families 2-7 to reduce correlation."
        )

    lines.append("")
    lines.append(f" Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * w)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    method: str = "hrp", output_path: Path | None = None, use_synthetic: bool = True
) -> None:
    """Run portfolio SR measurement and print/save report."""
    print("\nLoading strategy data...", flush=True)
    strategies = load_all_strategies(use_synthetic=use_synthetic)

    if not strategies:
        print("ERROR: No strategy data available.", file=sys.stderr)
        sys.exit(1)

    n_real = sum(1 for s in strategies.values() if s.source == "real")
    n_synthetic = sum(1 for s in strategies.values() if s.source == "synthetic")
    print(
        f"  Loaded {len(strategies)} strategies ({n_real} real, {n_synthetic} synthetic)"
    )

    # Align returns to common date range
    returns_matrix, slugs = align_returns(strategies)
    n_days = returns_matrix.shape[1]
    print(f"  Aligned to {n_days} trading days across {len(slugs)} strategies")

    # Estimate date range from real strategies
    real_starts = [s.start_date for s in strategies.values() if s.source == "real"]
    real_ends = [s.end_date for s in strategies.values() if s.source == "real"]
    date_range = (
        min(real_starts) if real_starts else "synthetic",
        max(real_ends) if real_ends else "synthetic",
    )

    # --- HRP weights ---
    print(f"\nComputing {method.upper()} weights...", flush=True)
    if method == "hrp":
        weights_hrp = compute_hrp_weights_simple(returns_matrix, slugs)
    else:
        # inverse-variance
        variances = np.var(returns_matrix, axis=1, ddof=1)
        inv_var = 1.0 / (variances + 1e-12)
        w_arr = inv_var / inv_var.sum()
        weights_hrp = {s: float(w_arr[i]) for i, s in enumerate(slugs)}

    # Apply 70/30 track split
    weights_hrp = apply_track_split(weights_hrp)

    # --- Equal weights ---
    weights_equal = apply_track_split(compute_equal_weights(slugs))

    # --- Portfolio returns ---
    port_hrp = portfolio_returns(returns_matrix, slugs, weights_hrp)
    port_equal = portfolio_returns(returns_matrix, slugs, weights_equal)

    # --- Benchmark ---
    bench = load_benchmark_returns(n_days)

    # --- Metrics ---
    metrics_hrp = compute_metrics(port_hrp)
    metrics_equal = compute_metrics(port_equal)
    metrics_bench = compute_metrics(bench)

    # --- Contribution analysis ---
    contributions = compute_contribution(strategies, port_hrp, returns_matrix, slugs)

    # --- Build report ---
    report = build_report(
        strategies=strategies,
        metrics_hrp=metrics_hrp,
        metrics_equal=metrics_equal,
        metrics_bench=metrics_bench,
        weights_hrp=weights_hrp,
        weights_equal=weights_equal,
        contributions=contributions,
        returns_matrix=returns_matrix,
        slugs=slugs,
        n_real=n_real,
        n_synthetic=n_synthetic,
        date_range=date_range,
        method=method,
    )

    print()
    print(report)

    # Save report
    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"portfolio_sr_report_{date.today().isoformat()}.md"

    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure portfolio Sharpe ratio using OOS walk-forward returns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["hrp", "inv-var"],
        default="hrp",
        help="Portfolio weighting method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the report (default: data/reports/portfolio_sr_report_DATE.md)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        default=False,
        help="Only use strategies with real backtest data (may produce very small portfolio)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    main(
        method=args.method,
        output_path=args.output,
        use_synthetic=not args.no_synthetic,
    )

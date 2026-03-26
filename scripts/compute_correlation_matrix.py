#!/usr/bin/env python3
"""Compute pairwise correlation matrix for all passing strategies.

Runs each strategy through the backtest engine and aligns daily returns
on trading dates. Outputs a full correlation matrix + effective N.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
import math

import polars as pl

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Strategy definitions: (slug, strategy_class, symbols, params)
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
    (
        "tlt-spy-rate-momentum",
        "lead_lag",
        ["TLT", "SPY"],
        {
            "leader_symbol": "TLT",
            "follower_symbol": "SPY",
            "lag_days": 5,
            "signal_window": 10,
            "entry_threshold": 0.01,
            "exit_threshold": -0.01,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "tlt-qqq-rate-tech",
        "lead_lag",
        ["TLT", "QQQ"],
        {
            "leader_symbol": "TLT",
            "follower_symbol": "QQQ",
            "lag_days": 5,
            "signal_window": 10,
            "entry_threshold": 0.01,
            "exit_threshold": -0.01,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
    (
        "ief-qqq-rate-tech",
        "lead_lag",
        ["IEF", "QQQ"],
        {
            "leader_symbol": "IEF",
            "follower_symbol": "QQQ",
            "lag_days": 5,
            "signal_window": 10,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    ),
]

LOOKBACK_DAYS = 5 * 365  # match run_backtest.py


def run_strategy_get_returns(
    slug: str,
    strategy_class: str,
    symbols: list[str],
    params: dict,
) -> tuple[list, list[float]]:
    """Run strategy backtest and return (dates, daily_returns)."""
    print(f"  Running {slug}...", end="", flush=True)

    # Fetch data
    prices_df = fetch_ohlcv(symbols, lookback_days=LOOKBACK_DAYS)
    indicators_df = compute_indicators(prices_df)

    # Build strategy config
    rebalance_freq = params.get("rebalance_frequency_days", 5)
    config = StrategyConfig(
        name=strategy_class,
        rebalance_frequency_days=rebalance_freq,
        parameters=params,
    )
    strategy = create_strategy(strategy_class, config)

    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=100_000.0,
    )
    cost_model = CostModel(spread_bps=2.0, commission_per_share=0.005)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug,
        cost_model=cost_model,
        warmup_days=200,
        cost_multiplier=1.0,
    )

    # Get dates from snapshots
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
    # Find common date range
    start = max(d[0] for d in all_dates)
    end = min(d[-1] for d in all_dates)
    print(f"\nAligned date range: {start} to {end}")

    # Build per-strategy dict
    frames = []
    for slug, dates, rets in zip(slugs, all_dates, all_returns, strict=True):
        date_ret = dict(zip(dates, rets, strict=True))
        frames.append((slug, date_ret))

    # Find common dates (intersection)
    common_dates = set(all_dates[0])
    for d in all_dates[1:]:
        common_dates &= set(d)
    common_dates = sorted(d for d in common_dates if start <= d <= end)
    print(f"Common trading dates: {len(common_dates)}")

    # Build aligned DataFrame
    rows = []
    for d in common_dates:
        row = {"date": d}
        for slug, date_ret in frames:
            row[slug] = date_ret.get(d, 0.0)
        rows.append(row)

    return pl.DataFrame(rows)


def compute_correlation_matrix(df: pl.DataFrame, slugs: list[str]) -> None:
    """Compute and print the full pairwise correlation matrix."""
    n = len(slugs)

    # Short labels for display
    labels = {
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
        "tlt-spy-rate-momentum": "TLT-SPY",
        "tlt-qqq-rate-tech": "TLT-QQQ",
        "ief-qqq-rate-tech": "IEF-QQQ",
    }

    # Extract return arrays
    return_arrays = []
    for s in slugs:
        arr = df[s].to_list()
        return_arrays.append(arr)

    n_obs = len(return_arrays[0])

    # Compute correlation matrix
    def mean(arr: list[float]) -> float:
        return sum(arr) / len(arr)

    def std(arr: list[float], m: float) -> float:
        return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))

    def corr(a: list[float], b: list[float]) -> float:
        ma, mb = mean(a), mean(b)
        sa, sb = std(a, ma), std(b, mb)
        if sa == 0 or sb == 0:
            return 0.0
        return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / (
            (n_obs - 1) * sa * sb
        )

    # Build matrix
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 1.0
            elif i < j:
                c = corr(return_arrays[i], return_arrays[j])
                mat[i][j] = c
                mat[j][i] = c

    # Print matrix
    short_labels = [labels[s] for s in slugs]
    col_w = 10

    print("\n" + "=" * 80)
    print("PAIRWISE CORRELATION MATRIX (11 strategies, 5-year daily returns)")
    print("=" * 80)

    # Header
    header = f"{'':12}" + "".join(f"{lbl:>{col_w}}" for lbl in short_labels)
    print(header)
    print("-" * len(header))

    for i, label in enumerate(short_labels):
        row = f"{label:<12}"
        for j in range(n):
            val = mat[i][j]
            if i == j:
                row += f"{'1.000':>{col_w}}"
            else:
                row += f"{val:>{col_w}.3f}"
        print(row)

    _print_stats(mat, slugs, labels, n)


SHARPES = {
    "soxx-qqq-lead-lag": 0.861,
    "lqd-spy-credit-lead": 1.250,
    "agg-spy-credit-lead": 1.145,
    "hyg-spy-5d-credit-lead": 0.913,
    "agg-qqq-credit-lead": 1.080,
    "lqd-qqq-credit-lead": 1.023,
    "vcit-qqq-credit-lead": 1.037,
    "hyg-qqq-credit-lead": 0.867,
    "emb-spy-credit-lead": 1.005,
    "agg-efa-credit-lead": 0.860,
    "spy-overnight-momentum": 1.043,
}


def _print_stats(
    mat: list[list[float]],
    slugs: list[str],
    labels: dict[str, str],
    n: int,
) -> None:
    """Print summary stats for the correlation matrix."""
    off_diag = [mat[i][j] for i in range(n) for j in range(n) if i < j]
    avg_rho = sum(off_diag) / len(off_diag)
    eff_n = 1 + (n - 1) * (1 - avg_rho)

    print(f"\nAverage pairwise rho: {avg_rho:.3f}")
    print(f"Effective independent N (Meucci): {eff_n:.2f}")

    # C7 correlations
    c7_idx = slugs.index("spy-overnight-momentum")
    print("\n--- C7 (SPY Overnight Momentum) vs. credit-equity family ---")
    c7_corrs = []
    for i, s in enumerate(slugs):
        if s != "spy-overnight-momentum":
            c = mat[c7_idx][i]
            c7_corrs.append(c)
            print(f"  vs {labels[s]:12}: rho = {c:.3f}")
    print(
        f"\n  Average C7 rho with credit-equity family: {sum(c7_corrs) / len(c7_corrs):.3f}"
    )

    # Credit-equity only
    credit_idxs = [i for i, s in enumerate(slugs) if s != "spy-overnight-momentum"]
    credit_off = [mat[i][j] for i in credit_idxs for j in credit_idxs if i < j]
    avg_credit_rho = sum(credit_off) / len(credit_off) if credit_off else 0.0
    eff_n_credit = 1 + (len(credit_idxs) - 1) * (1 - avg_credit_rho)
    print("\n--- Credit-equity family only (10 strategies) ---")
    print(f"  Average pairwise rho: {avg_credit_rho:.3f}")
    print(f"  Effective N: {eff_n_credit:.2f}")

    # Full portfolio
    avg_sharpe = sum(SHARPES[s] for s in slugs) / n
    portfolio_sharpe = avg_sharpe * math.sqrt(eff_n)
    print("\n--- Full 11-strategy portfolio ---")
    print(f"  Average pairwise rho: {avg_rho:.3f}")
    print(f"  Effective N: {eff_n:.2f}")
    print(f"  Avg individual Sharpe: {avg_sharpe:.3f}")
    print(f"  Equal-weight portfolio Sharpe (approx): {portfolio_sharpe:.2f}")
    print("=" * 80)


def main() -> None:
    slugs = [s[0] for s in STRATEGIES]
    print(f"Computing correlation matrix for {len(slugs)} strategies...")
    print("Fetching data and running backtests...\n")

    all_dates = []
    all_returns = []

    for slug, strategy_class, symbols, params in STRATEGIES:
        dates, rets = run_strategy_get_returns(slug, strategy_class, symbols, params)
        all_dates.append(dates)
        all_returns.append(rets)

    df = align_returns(all_dates, all_returns, slugs)
    compute_correlation_matrix(df, slugs)


if __name__ == "__main__":
    main()

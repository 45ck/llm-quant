#!/usr/bin/env python3
"""Track D Weight Analysis: CAGR potential at higher weight allocations.

For each of the top 4 Track D strategies (TLT-TQQQ, AGG-TQQQ, LQD-SOXL, TLT-UPRO),
runs backtests at weights 30%-70% and computes combined portfolio metrics.

Key question: At what weight level does the combined Track D portfolio hit 50% CAGR,
and what's the MaxDD at that level?

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_weight_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Strategy definitions (from frozen research specs) ─────────────────────
STRATEGIES = {
    "TLT-TQQQ": {
        "leader_symbol": "TLT",
        "follower_symbol": "TQQQ",
        "symbols": ["TLT", "TQQQ"],
    },
    "AGG-TQQQ": {
        "leader_symbol": "AGG",
        "follower_symbol": "TQQQ",
        "symbols": ["AGG", "TQQQ"],
    },
    "LQD-SOXL": {
        "leader_symbol": "LQD",
        "follower_symbol": "SOXL",
        "symbols": ["LQD", "SOXL"],
    },
    "TLT-UPRO": {
        "leader_symbol": "TLT",
        "follower_symbol": "UPRO",
        "symbols": ["TLT", "UPRO"],
    },
}

# Common parameters (from frozen specs — all four strategies use identical params)
COMMON_PARAMS = {
    "lag_days": 3,
    "signal_window": 10,
    "entry_threshold": 0.01,
    "exit_threshold": -0.005,
    "inverse": False,
    "rebalance_frequency_days": 1,
}

COST_MODEL = CostModel(
    spread_bps=10.0,
    flat_slippage_bps=5.0,
    slippage_volatility_factor=0.2,
)

WEIGHTS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
YEARS = 5
INITIAL_CAPITAL = 100_000.0
WARMUP_DAYS = 30


def run_single_backtest(
    name: str,
    strat_def: dict,
    target_weight: float,
    prices_df,
    indicators_df,
) -> dict:
    """Run a single backtest for one strategy at one weight level.

    Returns dict with Sharpe, MaxDD, CAGR, Sortino, Calmar, daily_returns.
    """
    params = {
        **COMMON_PARAMS,
        "leader_symbol": strat_def["leader_symbol"],
        "follower_symbol": strat_def["follower_symbol"],
        "target_weight": target_weight,
    }

    config = StrategyConfig(
        name="lead_lag",
        rebalance_frequency_days=1,
        max_positions=10,
        target_position_weight=0.05,
        stop_loss_pct=0.05,
        parameters=params,
    )

    strategy = create_strategy("lead_lag", config)
    engine = BacktestEngine(
        strategy=strategy,
        data_dir="data",
        initial_capital=INITIAL_CAPITAL,
    )

    slug = f"{name.lower()}-weight-{int(target_weight * 100)}"
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug,
        cost_model=COST_MODEL,
        fill_delay=1,
        warmup_days=WARMUP_DAYS,
        cost_multiplier=1.0,
        benchmark_weights={"SPY": 0.60, "TLT": 0.40},
    )

    m = result.metrics.get("1.0x")
    if m is None:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
            "trades": 0,
            "daily_returns": [],
            "nav_series": [],
        }

    return {
        "sharpe": m.sharpe_ratio,
        "max_dd": m.max_drawdown,
        "cagr": m.annualized_return,
        "sortino": m.sortino_ratio,
        "calmar": m.calmar_ratio,
        "total_return": m.total_return,
        "trades": m.total_trades,
        "daily_returns": result.daily_returns,
        "nav_series": result.nav_series,
    }


def compute_combined_metrics(
    strategy_daily_returns: dict[str, list[float]],
    strategy_nav_series: dict[str, list[float]],
    n_strategies: int,
) -> dict:
    """Compute equal-weight combined portfolio metrics from daily returns.

    Each strategy contributes 1/n_strategies of portfolio weight.
    """
    # Find the minimum length across all return series
    names = list(strategy_daily_returns.keys())
    min_len = min(len(strategy_daily_returns[n]) for n in names)

    if min_len < 10:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
        }

    # Equal-weight combined daily returns
    combined_returns = np.zeros(min_len)
    for name in names:
        rets = np.array(strategy_daily_returns[name][:min_len])
        combined_returns += rets / n_strategies

    # Compute NAV series from combined returns
    nav = np.ones(min_len + 1) * INITIAL_CAPITAL
    for i, r in enumerate(combined_returns):
        nav[i + 1] = nav[i] * (1 + r)

    # Total return
    total_return = nav[-1] / nav[0] - 1.0

    # CAGR
    years = min_len / 252.0
    if years > 0 and nav[-1] > 0:
        cagr = (nav[-1] / nav[0]) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Sharpe (annualized)
    mean_daily = np.mean(combined_returns)
    std_daily = np.std(combined_returns, ddof=1)
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

    # Sortino
    downside = combined_returns[combined_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_daily / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    max_dd = abs(float(np.min(dd)))

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "cagr": float(cagr),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "total_return": float(total_return),
    }


def main() -> None:  # noqa: PLR0912
    print("=" * 90)
    print("TRACK D WEIGHT ANALYSIS — CAGR Potential at Higher Allocations")
    print("=" * 90)
    print()

    # ── Fetch all needed data once ────────────────────────────────────────
    all_symbols = set()
    for s in STRATEGIES.values():
        all_symbols.update(s["symbols"])
    all_symbols = sorted(all_symbols)

    print(f"Fetching {YEARS}-year data for: {', '.join(all_symbols)}")
    prices_df = fetch_ohlcv(all_symbols, lookback_days=YEARS * 365)
    if len(prices_df) == 0:
        print("ERROR: No data fetched. Aborting.")
        sys.exit(1)

    print("Computing indicators...")
    indicators_df = compute_indicators(prices_df)
    print()

    # ── Run backtests ────────────────────────────────────────────────────
    # results[strategy_name][weight] = metrics dict
    results: dict[str, dict[float, dict]] = {}

    for strat_name, strat_def in STRATEGIES.items():
        results[strat_name] = {}
        for weight in WEIGHTS:
            print(f"  Running {strat_name} @ {int(weight * 100)}% weight...", end=" ")
            r = run_single_backtest(
                strat_name, strat_def, weight, prices_df, indicators_df
            )
            results[strat_name][weight] = r
            print(
                f"CAGR={r['cagr']:.1%}  Sharpe={r['sharpe']:.3f}  "
                f"MaxDD={r['max_dd']:.1%}  Trades={r['trades']}"
            )
        print()

    # ── Individual Strategy Table ────────────────────────────────────────
    print()
    print("=" * 90)
    print("INDIVIDUAL STRATEGY RESULTS")
    print("=" * 90)
    print()
    print(
        f"{'Strategy':<14} {'Weight':>6} {'CAGR':>8} {'TotalRet':>10} "
        f"{'Sharpe':>7} {'MaxDD':>7} {'Sortino':>8} {'Calmar':>7} {'Trades':>7}"
    )
    print("-" * 90)

    for strat_name in STRATEGIES:
        for weight in WEIGHTS:
            r = results[strat_name][weight]
            print(
                f"{strat_name:<14} {weight:>5.0%} "
                f"{r['cagr']:>8.1%} {r['total_return']:>10.1%} "
                f"{r['sharpe']:>7.3f} {r['max_dd']:>7.1%} "
                f"{r['sortino']:>8.3f} {r['calmar']:>7.3f} {r['trades']:>7d}"
            )
        print()

    # ── Combined Portfolio Analysis ──────────────────────────────────────
    print()
    print("=" * 90)
    print("COMBINED PORTFOLIO (Equal-Weight Across Top 4 Strategies)")
    print("=" * 90)
    print()
    print(
        "Each strategy gets 1/4 of portfolio. Weight shown is per-strategy "
        "target_weight parameter."
    )
    print()
    print(
        f"{'Weight':>6} {'CAGR':>8} {'TotalRet':>10} "
        f"{'Sharpe':>7} {'MaxDD':>7} {'Sortino':>8} {'Calmar':>7}"
    )
    print("-" * 70)

    for weight in WEIGHTS:
        daily_rets = {}
        nav_series = {}
        for strat_name in STRATEGIES:
            r = results[strat_name][weight]
            daily_rets[strat_name] = r["daily_returns"]
            nav_series[strat_name] = r["nav_series"]

        combined = compute_combined_metrics(daily_rets, nav_series, 4)
        print(
            f"{weight:>5.0%} "
            f"{combined['cagr']:>8.1%} {combined['total_return']:>10.1%} "
            f"{combined['sharpe']:>7.3f} {combined['max_dd']:>7.1%} "
            f"{combined['sortino']:>8.3f} {combined['calmar']:>7.3f}"
        )

    # ── Top-3 Combined (exclude weakest) ─────────────────────────────────
    print()
    print("=" * 90)
    print("COMBINED PORTFOLIO (Equal-Weight Top 3: TLT-TQQQ, AGG-TQQQ, LQD-SOXL)")
    print("=" * 90)
    print()
    print(
        f"{'Weight':>6} {'CAGR':>8} {'TotalRet':>10} "
        f"{'Sharpe':>7} {'MaxDD':>7} {'Sortino':>8} {'Calmar':>7}"
    )
    print("-" * 70)

    top3_names = ["TLT-TQQQ", "AGG-TQQQ", "LQD-SOXL"]
    for weight in WEIGHTS:
        daily_rets = {}
        nav_series = {}
        for strat_name in top3_names:
            r = results[strat_name][weight]
            daily_rets[strat_name] = r["daily_returns"]
            nav_series[strat_name] = r["nav_series"]

        combined = compute_combined_metrics(daily_rets, nav_series, 3)
        print(
            f"{weight:>5.0%} "
            f"{combined['cagr']:>8.1%} {combined['total_return']:>10.1%} "
            f"{combined['sharpe']:>7.3f} {combined['max_dd']:>7.1%} "
            f"{combined['sortino']:>8.3f} {combined['calmar']:>7.3f}"
        )

    # ── Sweet Spot Analysis ──────────────────────────────────────────────
    print()
    print("=" * 90)
    print("SWEET SPOT ANALYSIS — Finding CAGR > 50% with MaxDD < 40%")
    print("=" * 90)
    print()

    # Check all combos for 50% CAGR gate
    print("Checking all weight levels for the CAGR > 50% / MaxDD < 40% sweet spot:")
    print()

    # Individual strategies hitting 50% CAGR
    print("--- Individual Strategies ---")
    for strat_name in STRATEGIES:
        for weight in WEIGHTS:
            r = results[strat_name][weight]
            if r["cagr"] >= 0.50:
                gate = "PASS" if r["max_dd"] < 0.40 else "FAIL (MaxDD)"
                print(
                    f"  {strat_name} @ {weight:.0%}: "
                    f"CAGR={r['cagr']:.1%}, MaxDD={r['max_dd']:.1%} -> {gate}"
                )

    # Combined top-4 hitting 50% CAGR
    print()
    print("--- Combined Top-4 Portfolio ---")
    for weight in WEIGHTS:
        daily_rets = {}
        nav_series = {}
        for strat_name in STRATEGIES:
            r = results[strat_name][weight]
            daily_rets[strat_name] = r["daily_returns"]
            nav_series[strat_name] = r["nav_series"]
        combined = compute_combined_metrics(daily_rets, nav_series, 4)
        if combined["cagr"] >= 0.30:  # show anything close
            gate = (
                "PASS"
                if combined["cagr"] >= 0.50 and combined["max_dd"] < 0.40
                else "MISS"
            )
            print(
                f"  Top-4 @ {weight:.0%}: "
                f"CAGR={combined['cagr']:.1%}, MaxDD={combined['max_dd']:.1%}, "
                f"Sharpe={combined['sharpe']:.3f} -> {gate}"
            )

    # Combined top-3 hitting 50% CAGR
    print()
    print("--- Combined Top-3 Portfolio (TLT-TQQQ, AGG-TQQQ, LQD-SOXL) ---")
    for weight in WEIGHTS:
        daily_rets = {}
        nav_series = {}
        for strat_name in top3_names:
            r = results[strat_name][weight]
            daily_rets[strat_name] = r["daily_returns"]
            nav_series[strat_name] = r["nav_series"]
        combined = compute_combined_metrics(daily_rets, nav_series, 3)
        if combined["cagr"] >= 0.30:  # show anything close
            gate = (
                "PASS"
                if combined["cagr"] >= 0.50 and combined["max_dd"] < 0.40
                else "MISS"
            )
            print(
                f"  Top-3 @ {weight:.0%}: "
                f"CAGR={combined['cagr']:.1%}, MaxDD={combined['max_dd']:.1%}, "
                f"Sharpe={combined['sharpe']:.3f} -> {gate}"
            )

    # ── Correlation Matrix at 30% Weight ─────────────────────────────────
    print()
    print("=" * 90)
    print("PAIRWISE CORRELATION (at 30% weight)")
    print("=" * 90)
    print()

    strat_names = list(STRATEGIES.keys())
    n = len(strat_names)
    base_weight = 0.30

    # Build correlation matrix
    min_len = min(len(results[s][base_weight]["daily_returns"]) for s in strat_names)
    ret_matrix = np.zeros((n, min_len))
    for i, s in enumerate(strat_names):
        ret_matrix[i] = np.array(results[s][base_weight]["daily_returns"][:min_len])

    corr = np.corrcoef(ret_matrix)

    # Print header
    print(f"{'':>14}", end="")
    for s in strat_names:
        print(f"{s:>14}", end="")
    print()
    for i, s in enumerate(strat_names):
        print(f"{s:>14}", end="")
        for j in range(n):
            print(f"{corr[i, j]:>14.3f}", end="")
        print()

    avg_corr = (np.sum(corr) - n) / (n * (n - 1))
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")

    # ── Recommendation ───────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)
    print()

    # Collect ALL configurations that hit 50% CAGR with MaxDD < 40%
    passing_configs: list[dict] = []

    # Individual strategies
    for strat_name in STRATEGIES:
        for weight in WEIGHTS:
            r = results[strat_name][weight]
            if r["cagr"] >= 0.50 and r["max_dd"] < 0.40:
                passing_configs.append(
                    {
                        "config": f"{strat_name} (solo)",
                        "weight": weight,
                        "cagr": r["cagr"],
                        "max_dd": r["max_dd"],
                        "sharpe": r["sharpe"],
                        "sortino": r["sortino"],
                        "calmar": r["calmar"],
                    }
                )

    # Combined portfolios
    for combo_name, combo_names, combo_n in [
        ("Top-3", top3_names, 3),
        ("Top-4", list(STRATEGIES.keys()), 4),
    ]:
        for weight in WEIGHTS:
            daily_rets = {}
            nav_series = {}
            for s in combo_names:
                r = results[s][weight]
                daily_rets[s] = r["daily_returns"]
                nav_series[s] = r["nav_series"]
            c = compute_combined_metrics(daily_rets, nav_series, combo_n)
            if c["cagr"] >= 0.50 and c["max_dd"] < 0.40:
                passing_configs.append(
                    {
                        "config": f"{combo_name} combined",
                        "weight": weight,
                        **c,
                    }
                )

    if passing_configs:
        print("Configurations achieving CAGR > 50% with MaxDD < 40%:")
        print()
        print(
            f"{'Configuration':<24} {'Weight':>6} {'CAGR':>8} "
            f"{'MaxDD':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7}"
        )
        print("-" * 80)
        for cfg in sorted(passing_configs, key=lambda x: x["sharpe"], reverse=True):
            print(
                f"{cfg['config']:<24} {cfg['weight']:>5.0%} "
                f"{cfg['cagr']:>8.1%} {cfg['max_dd']:>7.1%} "
                f"{cfg['sharpe']:>7.3f} {cfg['sortino']:>8.3f} {cfg['calmar']:>7.3f}"
            )
        print()
        best = max(passing_configs, key=lambda x: x["sharpe"])
        print(f"BEST RISK-ADJUSTED: {best['config']} @ {best['weight']:.0%} weight")
        print(
            f"  CAGR={best['cagr']:.1%}, MaxDD={best['max_dd']:.1%}, "
            f"Sharpe={best['sharpe']:.3f}"
        )
    else:
        print("No configuration achieves CAGR > 50% with MaxDD < 40%.")

    # Always show best combined portfolio near 50% CAGR
    print()
    print("--- Best combined portfolio options (MaxDD < 40%) ---")
    for combo_name, combo_names, combo_n in [
        ("Top-3", top3_names, 3),
        ("Top-4", list(STRATEGIES.keys()), 4),
    ]:
        best_cagr = 0
        best_info = ""
        for weight in WEIGHTS:
            daily_rets = {}
            nav_series = {}
            for s in combo_names:
                r = results[s][weight]
                daily_rets[s] = r["daily_returns"]
                nav_series[s] = r["nav_series"]
            c = compute_combined_metrics(daily_rets, nav_series, combo_n)
            if c["max_dd"] < 0.40 and c["cagr"] > best_cagr:
                best_cagr = c["cagr"]
                best_info = (
                    f"{combo_name} @ {weight:.0%}: CAGR={c['cagr']:.1%}, "
                    f"MaxDD={c['max_dd']:.1%}, Sharpe={c['sharpe']:.3f}"
                )
        if best_info:
            print(f"  {best_info}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()

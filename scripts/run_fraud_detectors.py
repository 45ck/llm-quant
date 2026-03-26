#!/usr/bin/env python3
"""Run fraud detectors on all passing strategies.

Implements:
1. Shuffled Signal Test (permutation test for timing alpha)
2. Mechanism Inversion Test (inverted signal should lose money)
3. Time-in-Market Analysis (>80% invested = likely beta capture)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging
from typing import Any

import yaml

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.robustness import (
    mechanism_inversion_test,
    shuffled_signal_test,
    time_in_market,
)
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# All 15 passing strategies
# ---------------------------------------------------------------------------

STRATEGIES: list[dict] = [
    {
        "slug": "lqd-spy-credit-lead",
        "cls": "lead_lag",
        "syms": ["LQD", "SPY"],
        "params": {
            "leader_symbol": "LQD",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "agg-spy-credit-lead",
        "cls": "lead_lag",
        "syms": ["AGG", "SPY"],
        "params": {
            "leader_symbol": "AGG",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "hyg-spy-5d-credit-lead",
        "cls": "lead_lag",
        "syms": ["HYG", "SPY"],
        "params": {
            "leader_symbol": "HYG",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "agg-qqq-credit-lead",
        "cls": "lead_lag",
        "syms": ["AGG", "QQQ"],
        "params": {
            "leader_symbol": "AGG",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "lqd-qqq-credit-lead",
        "cls": "lead_lag",
        "syms": ["LQD", "QQQ"],
        "params": {
            "leader_symbol": "LQD",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "vcit-qqq-credit-lead",
        "cls": "lead_lag",
        "syms": ["VCIT", "QQQ"],
        "params": {
            "leader_symbol": "VCIT",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "hyg-qqq-credit-lead",
        "cls": "lead_lag",
        "syms": ["HYG", "QQQ"],
        "params": {
            "leader_symbol": "HYG",
            "follower_symbol": "QQQ",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "emb-spy-credit-lead",
        "cls": "lead_lag",
        "syms": ["EMB", "SPY"],
        "params": {
            "leader_symbol": "EMB",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "agg-efa-credit-lead",
        "cls": "lead_lag",
        "syms": ["AGG", "EFA"],
        "params": {
            "leader_symbol": "AGG",
            "follower_symbol": "EFA",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "soxx-qqq-lead-lag",
        "cls": "lead_lag",
        "syms": ["SOXX", "QQQ"],
        "params": {
            "leader_symbol": "SOXX",
            "follower_symbol": "QQQ",
            "lag_days": 5,
            "signal_window": 5,
            "entry_threshold": 0.02,
            "exit_threshold": -0.01,
            "target_weight": 0.9,
            "rebalance_frequency_days": 1,
        },
    },
    {
        "slug": "spy-overnight-momentum",
        "cls": "overnight_momentum",
        "syms": ["SPY"],
        "params": {
            "symbol": "SPY",
            "window": 10,
            "entry_thresh": 0.002,
            "exit_thresh": -0.0005,
            "target_weight": 0.9,
        },
    },
    {
        "slug": "tlt-spy-rate-momentum",
        "cls": "lead_lag",
        "syms": ["TLT", "SPY"],
        "params": {
            "leader_symbol": "TLT",
            "follower_symbol": "SPY",
            "lag_days": 5,
            "signal_window": 10,
            "entry_threshold": 0.01,
            "exit_threshold": -0.01,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "tlt-qqq-rate-tech",
        "cls": "lead_lag",
        "syms": ["TLT", "QQQ"],
        "params": {
            "leader_symbol": "TLT",
            "follower_symbol": "QQQ",
            "lag_days": 5,
            "signal_window": 10,
            "entry_threshold": 0.01,
            "exit_threshold": -0.01,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
    {
        "slug": "ief-qqq-rate-tech",
        "cls": "lead_lag",
        "syms": ["IEF", "QQQ"],
        "params": {
            "leader_symbol": "IEF",
            "follower_symbol": "QQQ",
            "lag_days": 5,
            "signal_window": 10,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
]


def get_follower_symbol(strat_def: dict) -> str:
    """Extract the follower/traded symbol from strategy definition."""
    params = strat_def["params"]
    if strat_def["cls"] == "lead_lag":
        return str(params.get("follower_symbol", "SPY"))
    if strat_def["cls"] == "overnight_momentum":
        return str(params.get("symbol", "SPY"))
    return "SPY"


def get_asset_daily_returns(symbol: str, prices_df: Any) -> list[float]:
    """Get daily returns for a single asset from price data."""
    import polars as pl

    sym_data = prices_df.filter(pl.col("symbol") == symbol).sort("date").select("close")
    closes = sym_data["close"].to_list()
    if len(closes) < 2:
        return []
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


def run_backtest(
    strat_def: dict,
) -> tuple[list[float], list[float]]:
    """Run backtest. Return (strategy_returns, asset_returns)."""
    prices = fetch_ohlcv(strat_def["syms"], lookback_days=5 * 365)
    indicators = compute_indicators(prices)

    follower = get_follower_symbol(strat_def)
    asset_rets = get_asset_daily_returns(follower, prices)

    config = StrategyConfig(
        name=strat_def["cls"],
        rebalance_frequency_days=strat_def["params"].get("rebalance_frequency_days", 5),
        parameters=strat_def["params"],
    )
    strategy = create_strategy(strat_def["cls"], config)
    engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)
    result = engine.run(
        prices_df=prices,
        indicators_df=indicators,
        slug=strat_def["slug"],
        cost_model=CostModel(),
        warmup_days=30,
        cost_multiplier=1.0,
    )

    # Align lengths: strategy may have fewer days due to warmup
    strat_rets = result.daily_returns
    n = min(len(strat_rets), len(asset_rets))
    # Align from the END (both end on the same date)
    return strat_rets[-n:], asset_rets[-n:]


def run_inverted_backtest(strat_def: dict) -> list[float]:
    """Run backtest with inverted signal direction.

    For lead_lag: negate both thresholds so the strategy enters
    when the leader FALLS and exits when the leader RISES.
    """
    params = dict(strat_def["params"])

    if strat_def["cls"] == "lead_lag":
        # Negate thresholds: entry on leader decline, exit on rise
        orig_entry = params.get("entry_threshold", 0.005)
        orig_exit = params.get("exit_threshold", -0.005)
        params["entry_threshold"] = -abs(orig_entry)
        params["exit_threshold"] = abs(orig_exit)
    elif strat_def["cls"] == "overnight_momentum":
        orig_entry = params.get("entry_thresh", 0.002)
        orig_exit = params.get("exit_thresh", -0.0005)
        params["entry_thresh"] = -abs(orig_entry)
        params["exit_thresh"] = abs(orig_exit)
    else:
        return []

    prices = fetch_ohlcv(strat_def["syms"], lookback_days=5 * 365)
    indicators = compute_indicators(prices)

    config = StrategyConfig(
        name=strat_def["cls"],
        rebalance_frequency_days=params.get("rebalance_frequency_days", 5),
        parameters=params,
    )
    strategy = create_strategy(strat_def["cls"], config)
    engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)
    result = engine.run(
        prices_df=prices,
        indicators_df=indicators,
        slug=strat_def["slug"] + "-inverted",
        cost_model=CostModel(),
        warmup_days=30,
        cost_multiplier=1.0,
    )
    return result.daily_returns


def main() -> None:
    print("=" * 80)
    print("FRAUD DETECTOR SUITE — All Passing Strategies")
    print("=" * 80)

    all_results: list[dict] = []

    for i, strat in enumerate(STRATEGIES):
        slug = strat["slug"]
        print(f"\n{'—' * 70}")
        print(f"[{i + 1}/{len(STRATEGIES)}] {slug}")
        print(f"{'—' * 70}")

        # 1. Run base backtest
        print("  Running base backtest...", end="", flush=True)
        returns, asset_rets = run_backtest(strat)
        print(f" {len(returns)} days")

        # 2. Shuffled Signal Test
        print("  Running shuffled signal test (1000 permutations)...", end="")
        shuffle_result = shuffled_signal_test(returns, asset_rets, n_shuffles=1000)
        status = "PASS" if shuffle_result.passed else "FAIL"
        print(
            f" {status} (real={shuffle_result.real_sharpe:.3f}, "
            f"95th={shuffle_result.shuffled_95th:.3f}, "
            f"p={shuffle_result.p_value:.4f})"
        )

        # 3. Time-in-Market
        tim_result = time_in_market(returns)
        tim_status = "PASS" if tim_result.passed else "FAIL"
        print(
            f"  Time-in-market: {tim_result.pct_invested:.1%} "
            f"({tim_result.invested_days}/{tim_result.total_days} days) "
            f"— {tim_status}"
        )

        # 4. Mechanism Inversion
        print("  Running inverted signal backtest...", end="", flush=True)
        inv_returns = run_inverted_backtest(strat)
        if inv_returns:
            inv_result = mechanism_inversion_test(returns, inv_returns)
            inv_status = "PASS" if inv_result.passed else "FAIL"
            print(
                f" {inv_status} (orig={inv_result.original_sharpe:.3f}, "
                f"inv={inv_result.inverted_sharpe:.3f}, "
                f"diff={inv_result.sharpe_differential:+.3f})"
            )
        else:
            inv_result = None
            print(" SKIPPED (unsupported strategy class)")

        result = {
            "slug": slug,
            "shuffle_real_sharpe": round(shuffle_result.real_sharpe, 4),
            "shuffle_95th": round(shuffle_result.shuffled_95th, 4),
            "shuffle_p_value": round(shuffle_result.p_value, 4),
            "shuffle_passed": shuffle_result.passed,
            "time_in_market_pct": round(tim_result.pct_invested, 4),
            "time_in_market_passed": tim_result.passed,
        }
        if inv_result:
            result["inversion_original_sharpe"] = round(inv_result.original_sharpe, 4)
            result["inversion_inverted_sharpe"] = round(inv_result.inverted_sharpe, 4)
            result["inversion_differential"] = round(inv_result.sharpe_differential, 4)
            result["inversion_passed"] = inv_result.passed

        all_results.append(result)

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    header = (
        f"{'Strategy':<30} {'Shuffle':>8} {'p-val':>7} "
        f"{'TiM%':>6} {'Inv.SR':>8} {'All':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        shuf = "PASS" if r["shuffle_passed"] else "FAIL"
        tim = f"{r['time_in_market_pct']:.0%}"
        inv_sr = (
            f"{r.get('inversion_inverted_sharpe', 0):.3f}"
            if "inversion_inverted_sharpe" in r
            else "N/A"
        )
        inv_pass = r.get("inversion_passed", False)
        tim_pass = r["time_in_market_passed"]
        all_pass = r["shuffle_passed"] and tim_pass and inv_pass
        all_str = "PASS" if all_pass else "FAIL"
        print(
            f"{r['slug']:<30} {shuf:>8} {r['shuffle_p_value']:>7.4f} "
            f"{tim:>6} {inv_sr:>8} {all_str:>5}"
        )

    # Save results
    out_path = Path("data/strategies/fraud-detector-results.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {out_path}")

    # Count passes
    n_shuffle = sum(1 for r in all_results if r["shuffle_passed"])
    n_tim = sum(1 for r in all_results if r["time_in_market_passed"])
    n_inv = sum(1 for r in all_results if r.get("inversion_passed", False))
    n_all = sum(
        1
        for r in all_results
        if r["shuffle_passed"]
        and r["time_in_market_passed"]
        and r.get("inversion_passed", False)
    )
    n = len(all_results)
    print(f"\nShuffled signal: {n_shuffle}/{n} pass")
    print(f"Time-in-market:  {n_tim}/{n} pass (<80%)")
    print(f"Inversion test:  {n_inv}/{n} pass (inverted < 0)")
    print(f"ALL 3 detectors: {n_all}/{n} pass")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robustness analysis for btc-spy-risk-signal."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "btc-spy-risk-signal"
STRATEGY = "lead_lag"
SYMBOLS = ["BTC-USD", "SPY"]
BASE_PARAMS = {
    "leader_symbol": "BTC-USD",
    "follower_symbol": "SPY",
    "lag_days": 3,
    "signal_window": 5,
    "entry_threshold": 0.05,
    "exit_threshold": -0.03,
    "target_weight": 0.80,
    "rebalance_frequency_days": 5,
}

print("Fetching data...")
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365)
indicators_df = compute_indicators(prices_df)


def run_single(params):
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=params.get("rebalance_frequency_days", 5),
        max_positions=1,
        target_position_weight=params.get("target_weight", 0.80),
        stop_loss_pct=0.05,
        parameters=dict(params),
    )
    strategy = create_strategy(STRATEGY, config)
    engine = BacktestEngine(strategy, initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=CostModel(),
        warmup_days=200,
        cost_multiplier=1.0,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "total_return": m.total_return if m else 0.0,
        "total_trades": m.total_trades if m else 0,
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(returns, n_groups=6, k=2, purge=5):
    from itertools import combinations

    n = len(returns)
    if n < n_groups:
        return 0.0, 0.0, 0, []
    group_size = n // n_groups
    oos_sharpes = []
    for test_idx in combinations(range(n_groups), k):
        test_rets = []
        for i in test_idx:
            s, e = i * group_size + purge, (i + 1) * group_size - purge
            if s < e:
                test_rets.extend(returns[s:e])
        if len(test_rets) < 20:
            continue
        mean = sum(test_rets) / len(test_rets)
        std = (sum((r - mean) ** 2 for r in test_rets) / len(test_rets)) ** 0.5
        if std > 0:
            oos_sharpes.append(mean / std * math.sqrt(252))
    if not oos_sharpes:
        return 0.0, 0.0, 0, []
    m = sum(oos_sharpes) / len(oos_sharpes)
    s = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    return m, s, len(oos_sharpes), oos_sharpes


# ---- Base backtest ----
print("Running base backtest...")
base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std, n_folds, oos_sharpes = cpcv_sharpe(base["daily_returns"])
oos_is = cpcv_mean / (base["sharpe"] + 1e-8)
pct_positive = (
    sum(1 for x in oos_sharpes if x > 0) / len(oos_sharpes) * 100 if oos_sharpes else 0
)

print(f"Base Sharpe: {base['sharpe']:.4f}, Max DD: {base['max_dd'] * 100:.2f}%")
print(
    f"Total return: {base['total_return'] * 100:.2f}%, Trades: {base['total_trades']}"
)
print(f"\nCPCV OOS Sharpe: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS ratio: {oos_is:.4f}")
print(f"CPCV folds: {n_folds}, % positive: {pct_positive:.1f}%")

# ---- Perturbations ----
perturbations = [
    (
        "lag_days=2",
        {**BASE_PARAMS, "lag_days": 2},
    ),
    (
        "lag_days=5",
        {**BASE_PARAMS, "lag_days": 5},
    ),
    (
        "entry_threshold=0.03",
        {**BASE_PARAMS, "entry_threshold": 0.03, "exit_threshold": -0.02},
    ),
    (
        "signal_window=3",
        {**BASE_PARAMS, "signal_window": 3, "rebalance_frequency_days": 3},
    ),
    (
        "target_weight=0.90",
        {**BASE_PARAMS, "target_weight": 0.90},
    ),
]

print("\nPerturbation results:")
stable_count = 0
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = "STABLE" if abs(pct) <= 25 else "UNSTABLE"
    if stable == "STABLE":
        stable_count += 1
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} "
        f"max_dd={r['max_dd'] * 100:.1f}% change={pct:+.1f}% {stable}"
    )
print(f"\nPerturbation stability: {stable_count}/5 ({stable_count / 5 * 100:.0f}%)")

# ---- DSR from experiment registry ----
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
dsr_value = exps[-1].get("dsr", 0.0)
trial_count = len(exps)
print(f"\nDSR: {dsr_value:.4f} (trial_count={trial_count})")

# ---- Gate Summary ----
g1 = base["sharpe"] > 0.80
g2 = base["max_dd"] < 0.15
g3 = dsr_value >= 0.95
g4 = cpcv_mean > 0
g5 = (stable_count / 5) >= 0.60  # 60% = 3/5 stable

print("\n" + "=" * 60)
print("ROBUSTNESS GATE SUMMARY")
print("=" * 60)
print(f"  1. Sharpe > 0.80:        {'PASS' if g1 else 'FAIL'}  ({base['sharpe']:.4f})")
print(
    f"  2. MaxDD < 15%:          {'PASS' if g2 else 'FAIL'}  ({base['max_dd'] * 100:.2f}%)"
)
print(f"  3. DSR >= 0.95:          {'PASS' if g3 else 'FAIL'}  ({dsr_value:.4f})")
print(f"  4. CPCV OOS Sharpe > 0:  {'PASS' if g4 else 'FAIL'}  ({cpcv_mean:.4f})")
print(
    f"  5. Perturbation >= 60%:  {'PASS' if g5 else 'FAIL'}  ({stable_count}/5 = {stable_count / 5 * 100:.0f}%)"
)
print("-" * 60)

all_pass = g1 and g2 and g3 and g4 and g5
print(f"  OVERALL: {'>>> ALL GATES PASSED <<<' if all_pass else '*** REJECTED ***'}")
print("=" * 60)

# ---- Save robustness.yaml ----
robustness_data = {
    "strategy_slug": SLUG,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"] * 100, 2),
    "base_total_return": round(base["total_return"] * 100, 2),
    "base_trades": base["total_trades"],
    "dsr": round(dsr_value, 4),
    "dsr_passed": g3,
    "cpcv_mean_oos_sharpe": round(cpcv_mean, 4),
    "cpcv_std_oos_sharpe": round(cpcv_std, 4),
    "cpcv_oos_is_ratio": round(oos_is, 4),
    "cpcv_n_folds": n_folds,
    "cpcv_pct_positive": round(pct_positive, 1),
    "cpcv_passed": g4,
    "perturbation_stable_count": stable_count,
    "perturbation_total": 5,
    "perturbation_pct": round(stable_count / 5 * 100, 0),
    "perturbation_passed": g5,
    "gates": {
        "sharpe_gt_0.80": g1,
        "maxdd_lt_15pct": g2,
        "dsr_gte_0.95": g3,
        "cpcv_oos_sharpe_gt_0": g4,
        "perturbation_gte_60pct": g5,
    },
    "overall_passed": all_pass,
}

out_path = Path(f"data/strategies/{SLUG}/robustness.yaml")
with out_path.open("w") as f:
    yaml.dump(robustness_data, f, default_flow_style=False, sort_keys=False)

print(f"\nSaved robustness results to {out_path}")

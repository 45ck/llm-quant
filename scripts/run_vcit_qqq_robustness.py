#!/usr/bin/env python3
"""Robustness analysis for vcit-qqq-credit-lead."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "vcit-qqq-credit-lead"
STRATEGY = "lead_lag"
SYMBOLS = ["VCIT", "QQQ"]
BASE_PARAMS = {
    "leader_symbol": "VCIT",
    "follower_symbol": "QQQ",
    "lag_days": 1,
    "signal_window": 5,
    "entry_threshold": 0.005,
    "exit_threshold": -0.005,
    "target_weight": 0.80,
    "rebalance_frequency_days": 5,
}

print("Fetching data...")
prices_df = fetch_ohlcv(
    SYMBOLS, lookback_days=5 * 365
)  # match official run_backtest.py (no +30)
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
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(returns, n_groups=6, k=2, purge=5):
    from itertools import combinations

    n = len(returns)
    if n < n_groups:
        return 0.0, 0.0
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
        return 0.0, 0.0
    m = sum(oos_sharpes) / len(oos_sharpes)
    s = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    return m, s


print("Running base backtest...")
base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std = cpcv_sharpe(base["daily_returns"])
oos_is = cpcv_mean / (base["sharpe"] + 1e-8)
print(f"Base Sharpe: {base['sharpe']:.4f}, Max DD: {base['max_dd'] * 100:.2f}%")
print(f"CPCV OOS Sharpe: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS ratio: {oos_is:.4f}")

perturbations = [
    (
        "signal_window=3",
        {**BASE_PARAMS, "signal_window": 3, "rebalance_frequency_days": 3},
    ),
    (
        "signal_window=7",
        {**BASE_PARAMS, "signal_window": 7, "rebalance_frequency_days": 7},
    ),
    (
        "entry_threshold=0.003",
        {**BASE_PARAMS, "entry_threshold": 0.003, "exit_threshold": -0.003},
    ),
    ("exit_threshold=-0.010", {**BASE_PARAMS, "exit_threshold": -0.010}),
    ("target_weight=0.90", {**BASE_PARAMS, "target_weight": 0.90}),
]
print("\nPerturbation results:")
stable_count = 0
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = "STABLE" if abs(pct) <= 30 else "UNSTABLE"
    if stable == "STABLE":
        stable_count += 1
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} "
        f"max_dd={r['max_dd'] * 100:.1f}% ({pct:+.1f}%) {stable}"
    )
print(f"\nPerturbation stability: {stable_count}/5")

# DSR from registry
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
dsr_value = exps[-1].get("dsr", 0.0)
trial_count = len(exps)
print(f"\nDSR: {dsr_value:.4f} (trial_count={trial_count})")
print("\n--- GATE SUMMARY ---")
print(
    f"Sharpe gate (>0.80):  {'PASS' if base['sharpe'] > 0.80 else 'FAIL'}  ({base['sharpe']:.4f})"
)
print(
    f"MaxDD gate (<15.0%):  {'PASS' if base['max_dd'] < 0.15 else 'FAIL'}  ({base['max_dd'] * 100:.2f}%)"
)
print(
    f"DSR gate (>=0.95):    {'PASS' if dsr_value >= 0.95 else 'FAIL'}  ({dsr_value:.4f})"
)
print(f"CPCV OOS/IS (>=0.5):  {'PASS' if oos_is >= 0.5 else 'FAIL'}  ({oos_is:.4f})")
print(
    f"Perturbation (>=3/5): {'PASS' if stable_count >= 3 else 'FAIL'}  ({stable_count}/5)"
)

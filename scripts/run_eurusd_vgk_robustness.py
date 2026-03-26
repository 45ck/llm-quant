#!/usr/bin/env python3
"""Robustness analysis for eurusd-vgk-fx-lead."""

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

SLUG = "eurusd-vgk-fx-lead"
STRATEGY = "lead_lag"
SYMBOLS = ["EURUSD=X", "VGK"]
DD_THRESHOLD = 0.15
BASE_PARAMS = {
    "leader_symbol": "EURUSD=X",
    "follower_symbol": "VGK",
    "lag_days": 2,
    "signal_window": 5,
    "entry_threshold": 0.008,
    "exit_threshold": -0.004,
    "target_weight": 0.80,
    "rebalance_frequency_days": 5,
}

prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365)
indicators_df = compute_indicators(prices_df)


def run_single(params):
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=int(params.get("rebalance_frequency_days", 5)),
        max_positions=2,
        target_position_weight=0.80,
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
        warmup_days=30,
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
        return 0.0, 0.0, 0.0
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
        return 0.0, 0.0, 0.0
    m = sum(oos_sharpes) / len(oos_sharpes)
    s = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    pct_positive = sum(1 for x in oos_sharpes if x > 0) / len(oos_sharpes)
    return m, s, pct_positive


# --- Run base case ---
print("=" * 60)
print(f"ROBUSTNESS ANALYSIS: {SLUG}")
print("=" * 60)

base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std, cpcv_pct_pos = cpcv_sharpe(base["daily_returns"])

# Compute OOS/IS ratio: run full sample Sharpe vs CPCV OOS
oos_is_ratio = cpcv_mean / base["sharpe"] if base["sharpe"] != 0 else 0.0

print("\n--- BASE RESULTS ---")
print(f"Base Sharpe: {base['sharpe']:.4f}")
print(f"Max DD:      {base['max_dd']:.4f}")
print("\n--- CPCV RESULTS ---")
print(f"CPCV OOS Mean Sharpe: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS Ratio:    {oos_is_ratio:.4f}")
print(f"CPCV % Positive Folds: {cpcv_pct_pos:.1%}")

# --- Perturbation analysis ---
perturbations = [
    ("lag_days=1", {**BASE_PARAMS, "lag_days": 1}),
    ("lag_days=3", {**BASE_PARAMS, "lag_days": 3}),
    ("entry_threshold=0.005", {**BASE_PARAMS, "entry_threshold": 0.005}),
    ("entry_threshold=0.012", {**BASE_PARAMS, "entry_threshold": 0.012}),
    ("target_weight=0.60", {**BASE_PARAMS, "target_weight": 0.60}),
]
print("\n--- PERTURBATION RESULTS ---")
perturbation_results = []
stable_count = 0
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = "STABLE" if abs(pct) <= 25 else "UNSTABLE"
    if abs(pct) <= 25:
        stable_count += 1
    perturbation_results.append(
        {
            "variant": name,
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "change_pct": round(pct, 1),
            "status": stable,
        }
    )
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} "
        f"max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {stable}"
    )

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# --- DSR from registry ---
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
dsr_value = 0.0
if registry_path.exists():
    with registry_path.open() as f:
        exps = [json.loads(line) for line in f if line.strip()]
    if exps:
        dsr_value = exps[-1].get("dsr", 0.0)

print("\n--- DSR ---")
print(f"DSR: {dsr_value:.4f}")

# --- Gate Assessment ---
print(f"\n{'=' * 60}")
print("GATE ASSESSMENT")
print(f"{'=' * 60}")

gate1 = base["sharpe"] > 0.80
gate2 = base["max_dd"] < DD_THRESHOLD
gate3 = dsr_value >= 0.95
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 60

gates = [
    ("Gate 1: Sharpe > 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 15%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.95", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    ("Gate 5: Perturbation >= 60% stable", gate5, f"{pct_stable:.0f}%"),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

all_pass = all(g[1] for g in gates)
verdict = "PASS -- ALL GATES CLEARED" if all_pass else "FAIL"
print(f"\n  VERDICT: {verdict}")

# --- Save robustness.yaml ---
output = {
    "strategy_slug": SLUG,
    "strategy_type": STRATEGY,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "dsr": round(dsr_value, 4),
    "cpcv": {
        "oos_mean_sharpe": round(cpcv_mean, 4),
        "oos_std": round(cpcv_std, 4),
        "oos_is_ratio": round(oos_is_ratio, 4),
        "pct_positive_folds": round(cpcv_pct_pos, 4),
    },
    "perturbation": {
        "variants": perturbation_results,
        "pct_stable": round(pct_stable, 1),
    },
    "gates": {
        "sharpe_gt_0.80": gate1,
        "maxdd_lt_15pct": gate2,
        "dsr_gte_0.95": gate3,
        "cpcv_oos_positive": gate4,
        "perturbation_gte_60pct": gate5,
    },
    "verdict": "PASS" if all_pass else "FAIL",
}

out_path = Path(f"data/strategies/{SLUG}/robustness.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")

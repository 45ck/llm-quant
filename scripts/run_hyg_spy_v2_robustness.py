#!/usr/bin/env python3
"""Robustness analysis for hyg-spy-lead-lag-v2."""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "src")

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "hyg-spy-lead-lag-v2"
SYMBOLS = ["HYG", "SPY"]
YEARS = 5
STRAT = "lead_lag"
COST = CostModel(spread_bps=5.0, flat_slippage_bps=2.0, slippage_volatility_factor=0.1)

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
base_reg = exps[-1]

print(f"=== {SLUG} ===")
print(f"Sharpe: {base_reg['sharpe_ratio']:.4f}")
print(f"MaxDD:  {base_reg['max_drawdown']:.4f}")
print(f"DSR:    {base_reg['dsr']:.4f}")
print(f"Trades: {base_reg['total_trades']}")

prices_df = fetch_ohlcv(SYMBOLS, lookback_days=YEARS * 365 + 60)
indicators_df = compute_indicators(prices_df)
base_params = base_reg["parameters"]


def run_single(params):
    cfg = StrategyConfig(
        name=STRAT,
        rebalance_frequency_days=params.get("rebalance_frequency_days", 1),
        max_positions=2,
        target_position_weight=params.get("target_weight", 0.70),
        stop_loss_pct=0.05,
        parameters=dict(params),
    )
    s = create_strategy(STRAT, cfg)
    engine = BacktestEngine(strategy=s, data_dir="data", initial_capital=100000)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=COST,
        fill_delay=1,
        warmup_days=30,
        benchmark_weights={"SPY": 1.0},
        trial_count=1,
        cost_multiplier=1.0,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(returns, n_groups=6, k=2, purge=5):
    n = len(returns)
    if n < n_groups * 20:
        return 0.0, 0.0
    group_size = n // n_groups
    oos_sharpes = []
    for test_idx in combinations(range(n_groups), k):
        test_rets = []
        for i in test_idx:
            s = i * group_size + purge
            e = (i + 1) * group_size - purge
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


# Run base for CPCV
base_result = run_single(base_params)

print("\n=== PERTURBATION ANALYSIS ===")
perturbations = [
    ("signal_window=3", {**base_params, "signal_window": 3}),
    ("signal_window=10", {**base_params, "signal_window": 10}),
    ("entry_thresh=0.001", {**base_params, "entry_threshold": 0.001}),
    ("entry_thresh=0.005", {**base_params, "entry_threshold": 0.005}),
    ("weight=0.50", {**base_params, "target_weight": 0.50}),
]

n_stable = 0
for name, params in perturbations:
    r = run_single(params)
    pct = (
        (r["sharpe"] - base_reg["sharpe_ratio"])
        / (abs(base_reg["sharpe_ratio"]) + 1e-8)
        * 100
    )
    stable = abs(pct) <= 30
    if stable:
        n_stable += 1
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )

print(f"\nPerturbation: {n_stable}/5 stable ({'PASS' if n_stable >= 3 else 'FAIL'})")

print("\n=== CPCV ===")
cpcv_mean, cpcv_std = cpcv_sharpe(base_result["daily_returns"])
print(
    f"  Mean OOS Sharpe: {cpcv_mean:.4f}  Std: {cpcv_std:.4f}  ({'PASS' if cpcv_mean > 0 else 'FAIL'})"
)

print("\n=== GATE SUMMARY ===")
print(
    f"  DSR:         {base_reg['dsr']:.4f} ({'PASS' if base_reg['dsr'] >= 0.95 else 'FAIL'} >= 0.95)"
)
print(
    f"  MaxDD:       {base_reg['max_drawdown'] * 100:.2f}% ({'PASS' if base_reg['max_drawdown'] < 0.15 else 'FAIL'} < 15%)"
)
print(f"  Perturbation: {n_stable}/5 ({'PASS' if n_stable >= 3 else 'FAIL'})")
print(f"  CPCV OOS:    {cpcv_mean:.4f} ({'PASS' if cpcv_mean > 0 else 'FAIL'} > 0)")

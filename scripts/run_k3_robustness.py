#!/usr/bin/env python3
"""Robustness analysis for K3 usmv-spy-low-vol-rotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.robustness import run_cpcv
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "usmv-spy-low-vol-rotation"
SYMBOLS = ["USMV", "SPY"]
YEARS = 5
STRAT = "asset_rotation"

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
base = exps[-1]
print(f"=== {SLUG} ===")
print(
    f"Sharpe: {base['sharpe_ratio']:.4f}  MaxDD: {base['max_drawdown']:.4f}  DSR: {base['dsr']:.4f}  Trades: {base['total_trades']}"
)

lookback_days = YEARS * 365 + 60
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=lookback_days)
indicators_df = compute_indicators(prices_df)
cost = CostModel(spread_bps=3.0, flat_slippage_bps=1.0, slippage_volatility_factor=0.05)
base_params = base["parameters"]


def run_variant(params):
    cfg = StrategyConfig(
        name=STRAT,
        rebalance_frequency_days=int(params.get("rebalance_frequency_days", 5)),
        max_positions=int(params.get("top_k", 1)),
        target_position_weight=float(params.get("target_weight", 0.90)),
        stop_loss_pct=0.07,
        parameters=params,
    )
    s = create_strategy(STRAT, cfg)
    engine = BacktestEngine(strategy=s, data_dir="data", initial_capital=100000)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=cost,
        fill_delay=1,
        warmup_days=30,
        benchmark_weights={"SPY": 1.0},
        trial_count=1,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0,
        "max_dd": m.max_drawdown if m else 0,
        "daily_returns": result.daily_returns,
    }


perturbations = [
    ("lookback=10", {**base_params, "lookback_days": 10}),
    ("lookback=60", {**base_params, "lookback_days": 60}),
    ("rebalance=1", {**base_params, "rebalance_frequency_days": 1}),
    ("rebalance=20", {**base_params, "rebalance_frequency_days": 20}),
    ("weight=0.70", {**base_params, "target_weight": 0.70}),
]

n_stable = 0
for name, params in perturbations:
    r = run_variant(params)
    pct = (
        (r["sharpe"] - base["sharpe_ratio"]) / (abs(base["sharpe_ratio"]) + 1e-8) * 100
    )
    stable = abs(pct) <= 30
    if stable:
        n_stable += 1
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )
print(f"Perturbation: {n_stable}/5 stable ({'PASS' if n_stable >= 3 else 'FAIL'})")

# Run base to get daily returns for CPCV
base_result = run_variant(base_params)
base_daily_returns = base_result["daily_returns"]

try:
    cpcv = run_cpcv(
        returns=base_daily_returns,
        strategy_fn=None,
        n_groups=6,
        k_test=2,
        purge_days=5,
    )
    mean_s = cpcv.mean_oos_sharpe
    std_s = cpcv.std_oos_sharpe
    print(
        f"CPCV mean OOS Sharpe: {mean_s:.4f}  std: {std_s:.4f}  ({'PASS' if mean_s > 0 else 'FAIL'})"
    )
except Exception as e:
    print(f"CPCV error: {e}")
    import traceback

    traceback.print_exc()

print(f"DSR: {base['dsr']:.4f} ({'PASS' if base['dsr'] >= 0.95 else 'FAIL'} >= 0.95)")
print(
    f"MaxDD: {base['max_drawdown'] * 100:.2f}% ({'PASS' if base['max_drawdown'] < 0.15 else 'FAIL'} < 15%)"
)

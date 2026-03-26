#!/usr/bin/env python3
"""Robustness analysis for xlu-spy-inverse-lead-lag-v2."""

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

SLUG = "xlu-spy-inverse-lead-lag-v2"
SYMBOLS = ["XLU", "SPY"]
YEARS = 5
STRAT = "lead_lag"

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
base = exps[-1]
print(f"=== {SLUG} ===")
print(f"Sharpe: {base['sharpe_ratio']:.4f}")
print(f"MaxDD:  {base['max_drawdown']:.4f}")
print(f"DSR:    {base['dsr']:.4f}")
print(f"Trades: {base['total_trades']}")

lookback_days = YEARS * 365 + 60
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=lookback_days)
indicators_df = compute_indicators(prices_df)
cost = CostModel(spread_bps=5.0, flat_slippage_bps=2.0, slippage_volatility_factor=0.1)
base_params = base["parameters"]


def run_variant(params):
    cfg = StrategyConfig(name=STRAT, parameters=params)
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


print("\n=== PERTURBATION ANALYSIS ===")
perturbations = [
    ("lag_days=1", {**base_params, "lag_days": 1}),
    ("lag_days=3", {**base_params, "lag_days": 3}),
    ("entry_thresh=0.005", {**base_params, "entry_threshold": 0.005}),
    ("entry_thresh=0.012", {**base_params, "entry_threshold": 0.012}),
    ("weight=0.30", {**base_params, "target_weight": 0.30}),
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
        f"  {name}: sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )

print(f"\nPerturbation: {n_stable}/5 stable ({'PASS' if n_stable >= 3 else 'FAIL'})")

print("\n=== CPCV ===")
# Get base returns for CPCV
base_result = run_variant(base_params)
daily_returns = base_result["daily_returns"]
print(f"  Using {len(daily_returns)} daily returns for CPCV")
try:
    cpcv = run_cpcv(
        returns=daily_returns,
        strategy_fn=None,
        n_groups=6,
        k_test=2,
        purge_days=5,
    )
    mean_s = cpcv.mean_oos_sharpe
    std_s = cpcv.std_oos_sharpe
    passed = cpcv.passed
    print(
        f"  Mean OOS Sharpe: {mean_s:.4f}  Std: {std_s:.4f}  ({'PASS' if passed else 'FAIL'})"
    )
    if hasattr(cpcv, "oos_sharpes") and cpcv.oos_sharpes:
        print(f"  OOS Sharpe distribution: {[round(x, 3) for x in cpcv.oos_sharpes]}")
except Exception as e:
    import traceback

    print(f"  CPCV failed: {e}")
    traceback.print_exc()

print("\n=== GATE SUMMARY ===")
print(
    f"  DSR:          {base['dsr']:.4f} ({'PASS' if base['dsr'] >= 0.95 else 'FAIL'} >= 0.95)"
)
print(
    f"  MaxDD:        {base['max_drawdown'] * 100:.2f}% ({'PASS' if base['max_drawdown'] < 0.15 else 'FAIL'} < 15%)"
)
print(f"  Perturbation: {n_stable}/5 ({'PASS' if n_stable >= 3 else 'FAIL'})")

#!/usr/bin/env python3
"""Robustness for vov-spy-defensive after VoV percentile logic fix."""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "src")

from llm_quant.backtest.cost import CostModel

from llm_quant.backtest.engine import BacktestEngine
from llm_quant.backtest.robustness import run_cpcv
from llm_quant.backtest.strategies import StrategyConfig, create_strategy
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "vov-spy-defensive"
SYMBOLS = ["SPY", "VIX"]
YEARS = 5
STRAT = "vix_regime"

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as fh:
    exps = [json.loads(line) for line in fh if line.strip()]
if not exps:
    print("ERROR: No experiments found")
    sys.exit(1)

base = exps[-1]
print(f"=== {SLUG} ===")
sharpe = base["sharpe_ratio"]
max_dd = base["max_drawdown"]
dsr = base["dsr"]
trades = base["total_trades"]
print(f"Sharpe={sharpe:.4f}  MaxDD={max_dd:.4f}  DSR={dsr:.4f}  Trades={trades}")

lookback_days = YEARS * 365 + 60
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=lookback_days)
indicators_df = compute_indicators(prices_df)
cost = CostModel(spread_bps=3.0, flat_slippage_bps=1.0, slippage_volatility_factor=0.05)
base_params = base["parameters"]


def run_variant(params: dict) -> dict:
    cfg = StrategyConfig(
        name=STRAT,
        rebalance_frequency_days=1,
        max_positions=2,
        target_position_weight=0.90,
        stop_loss_pct=0.05,
        parameters=dict(params),
    )
    s = create_strategy(STRAT, cfg)
    engine = BacktestEngine(strategy=s, initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=cost,
        warmup_days=50,
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
    st = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    return m, st


print("\n=== PERTURBATION ANALYSIS ===")
perturbations = [
    ("vov_window=20", {**base_params, "vov_window": 20}),
    ("vov_window=45", {**base_params, "vov_window": 45}),
    ("vov_percentile=0.70", {**base_params, "vov_percentile": 0.70}),
    ("vov_percentile=0.90", {**base_params, "vov_percentile": 0.90}),
    ("target_weight=0.70", {**base_params, "target_weight": 0.70}),
]
n_stable = 0
for name, params in perturbations:
    r = run_variant(params)
    pct = (r["sharpe"] - sharpe) / (abs(sharpe) + 1e-8) * 100
    stable = abs(pct) <= 30
    if stable:
        n_stable += 1
    tag = "STABLE" if stable else "UNSTABLE"
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {tag}"
    )
print(f"Perturbation: {n_stable}/5 ({'PASS' if n_stable >= 3 else 'FAIL'})")

print("\n=== CPCV ===")
cfg = StrategyConfig(parameters=base_params)
s = create_strategy(STRAT, cfg)
try:
    cpcv = run_cpcv(
        strategy=s,
        prices_df=prices_df,
        indicators_df=indicators_df,
        cost_model=cost,
        n_groups=6,
        n_test_groups=2,
        purge_days=5,
        initial_capital=100000,
        benchmark_weights={"SPY": 1.0},
    )
    sharpes = [f.sharpe_ratio for f in cpcv.fold_metrics if f.sharpe_ratio is not None]
    if sharpes:
        m = sum(sharpes) / len(sharpes)
        sd = (sum((x - m) ** 2 for x in sharpes) / len(sharpes)) ** 0.5
        print(
            f"  Mean OOS Sharpe={m:.4f}  Std={sd:.4f}  ({'PASS' if m > 0 else 'FAIL'})"
        )
except Exception as e:
    print(f"  CPCV failed: {e}")

print("\n=== GATE SUMMARY ===")
print(f"  DSR={dsr:.4f} ({'PASS' if dsr >= 0.95 else 'FAIL'} >= 0.95)")
print(f"  MaxDD={max_dd * 100:.2f}% ({'PASS' if max_dd < 0.12 else 'FAIL'} < 12%)")
print(f"  Perturbation: {n_stable}/5 ({'PASS' if n_stable >= 3 else 'FAIL'})")

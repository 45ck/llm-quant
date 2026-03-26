"""Robustness analysis for F3 turn-of-month-spy."""

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

SLUG = "turn-of-month-spy"
STRATEGY = "calendar_event"
SYMBOLS = ["SPY"]
BASE_PARAMS = {
    "mode": "month_end",
    "target_symbol": "SPY",
    "end_days": 1,
    "start_days": 1,
    "target_weight": 0.95,
    "rebalance_frequency_days": 1,
}

prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365 + 30)
indicators_df = compute_indicators(prices_df)


def run_single(params: dict) -> dict:
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=1,
        max_positions=1,
        target_position_weight=params.get("target_weight", 0.95),
        stop_loss_pct=0.05,
        parameters=dict(params),
    )
    engine = BacktestEngine(create_strategy(STRATEGY, config), initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=CostModel(),
        warmup_days=5,
        trial_count=1,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "total_return": m.total_return if m else 0.0,
        "total_trades": m.total_trades if m else 0,
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(
    returns: list[float], n_groups: int = 6, k: int = 2, purge: int = 2
) -> tuple[float, float]:
    from itertools import combinations

    n = len(returns)
    if n < n_groups:
        return 0.0, 0.0
    group_size = n // n_groups
    oos = []
    for test_idx in combinations(range(n_groups), k):
        tr = []
        for i in test_idx:
            s, e = i * group_size + purge, (i + 1) * group_size - purge
            if s < e:
                tr.extend(returns[s:e])
        if len(tr) < 20:
            continue
        mu = sum(tr) / len(tr)
        std = (sum((r - mu) ** 2 for r in tr) / len(tr)) ** 0.5
        if std > 0:
            oos.append(mu / std * math.sqrt(252))
    if not oos:
        return 0.0, 0.0
    m = sum(oos) / len(oos)
    s = (sum((x - m) ** 2 for x in oos) / len(oos)) ** 0.5
    return m, s


# Base run
base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std = cpcv_sharpe(base["daily_returns"])
print(
    f"Base: Sharpe={base['sharpe']:.4f} MaxDD={base['max_dd']:.4f} Return={base['total_return']:.4f} Trades={base['total_trades']}"
)
print(f"CPCV: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")

# Perturbations
perturbations = [
    ("end_days=2", {**BASE_PARAMS, "end_days": 2}),
    ("end_days=0", {**BASE_PARAMS, "end_days": 0}),
    ("start_days=2", {**BASE_PARAMS, "start_days": 2}),
    ("start_days=0", {**BASE_PARAMS, "start_days": 0}),
    ("weight=0.80", {**BASE_PARAMS, "target_weight": 0.80}),
]
perturb_results = []
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = abs(pct) <= 30
    print(
        f"  {name}: {r['sharpe']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )
    perturb_results.append(
        {"name": name, "sharpe": r["sharpe"], "pct_change": pct, "stable": stable}
    )

# DSR from experiment registry
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
dsr_value = 0.0
if registry_path.exists():
    with open(registry_path) as f:
        exps = [json.loads(line) for line in f if line.strip()]
    if exps:
        dsr_value = exps[-1].get("dsr", 0.0)

n_stable = sum(1 for r in perturb_results if r["stable"])
print(f"\nDSR: {dsr_value:.4f}, n_stable: {n_stable}/{len(perturb_results)}")
print(f"CPCV OOS Sharpe: {cpcv_mean:.4f}")
print(f"Max DD: {base['max_dd']:.2f}%")

# Gate decisions
dsr_pass = dsr_value >= 0.95
cpcv_pass = cpcv_mean > 0
perturb_pass = n_stable >= len(perturb_results) * 0.6
dd_pass = base["max_dd"] < 15.0
overall = dsr_pass and cpcv_pass

print("\n--- GATE RESULTS ---")
print(f"DSR >= 0.95: {'PASS' if dsr_pass else 'FAIL'} ({dsr_value:.4f})")
print(f"CPCV > 0: {'PASS' if cpcv_pass else 'FAIL'} ({cpcv_mean:.4f})")
print(
    f"Perturbation >= 60%: {'PASS' if perturb_pass else 'FAIL'} ({n_stable}/{len(perturb_results)})"
)
print(f"Max DD < 15%: {'PASS' if dd_pass else 'FAIL'} ({base['max_dd']:.2f}%)")
print(f"Overall: {'PASS' if overall else 'FAIL'}")

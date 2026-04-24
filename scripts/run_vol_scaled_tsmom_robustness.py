#!/usr/bin/env python3
"""Robustness analysis for vol-scaled-tsmom (H3.1).

Modeled on scripts/robustness_m3.py. Loads the base experiment from the
registry, re-runs the strategy on +/-20% perturbations, and runs CPCV on
the daily-return series stored in the experiment yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.robustness import run_cpcv
from llm_quant.backtest.strategies import StrategyConfig, create_strategy
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "vol-scaled-tsmom"
STRAT = "vol_scaled_tsmom"
SYMBOLS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "AGG",
    "HYG", "GLD", "SLV", "USO", "DBC", "UUP",
]
YEARS = 5

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]

# Use trial 1 (largest universe, highest sharpe) as the base
base = exps[0]
print(f"=== Robustness: {SLUG} (base experiment {base['experiment_id']}) ===")
print(
    f"Sharpe: {base['sharpe_ratio']:.4f}  MaxDD: {base['max_drawdown']:.4f}  "
    f"DSR: {base['dsr']:.4f}  Trades: {base['total_trades']}"
)

# Load daily returns from the experiment yaml
exp_yaml_path = Path(
    f"data/strategies/{SLUG}/experiments/{base['experiment_id']}.yaml"
)
with exp_yaml_path.open() as f:
    exp_data = yaml.safe_load(f)
base_daily_returns = exp_data.get("daily_returns") or []
print(f"Loaded {len(base_daily_returns)} daily returns from experiment yaml")

base_params = base["parameters"]


def run_variant(params, symbols=None):
    syms = symbols or SYMBOLS
    cfg = StrategyConfig(name=STRAT, parameters=params)
    s = create_strategy(STRAT, cfg)
    engine = BacktestEngine(strategy=s, data_dir="data", initial_capital=100000)
    lookback_days = YEARS * 365 + 60
    prices_df = fetch_ohlcv(syms, lookback_days=lookback_days)
    indicators_df = compute_indicators(prices_df)
    cost = CostModel(
        spread_bps=base_params.get("spread_bps", 5.0)
        if False
        else 5.0,
        flat_slippage_bps=3.0,
        slippage_volatility_factor=0.2,
    )
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=cost,
        fill_delay=1,
        warmup_days=300,
        benchmark_weights={"SPY": 0.6, "TLT": 0.4},
        trial_count=1,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "daily_returns": result.daily_returns,
    }


# Perturbation suite (+/-20% on key parameters)
perturbations = [
    ("vol_target=0.08", {**base_params, "vol_target": 0.08}),
    ("vol_target=0.12", {**base_params, "vol_target": 0.12}),
    ("vol_window=101", {**base_params, "vol_window": 101}),
    ("vol_window=151", {**base_params, "vol_window": 151}),
    ("flat_threshold=0.16", {**base_params, "flat_threshold": 0.16}),
    ("flat_threshold=0.24", {**base_params, "flat_threshold": 0.24}),
    ("max_vol_scalar=1.6", {**base_params, "max_vol_scalar": 1.6}),
    ("max_vol_scalar=2.4", {**base_params, "max_vol_scalar": 2.4}),
]

print("\n--- Parameter Perturbation (+/-20%) ---")
n_stable = 0
perturbation_results = []
for name, params in perturbations:
    try:
        r = run_variant(params)
        pct = (
            (r["sharpe"] - base["sharpe_ratio"])
            / (abs(base["sharpe_ratio"]) + 1e-8)
            * 100
        )
        stable = abs(pct) <= 30
        if stable:
            n_stable += 1
        perturbation_results.append(
            {"name": name, "sharpe": r["sharpe"], "pct": pct, "stable": stable}
        )
        flag = "STABLE" if stable else "UNSTABLE"
        print(f"  {name}: sharpe={r['sharpe']:.4f} ({pct:+.1f}%) {flag}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")
        perturbation_results.append(
            {"name": name, "sharpe": None, "pct": None, "stable": False, "error": str(e)}
        )

print(f"\nPerturbation stability: {n_stable}/{len(perturbations)} stable")

# CPCV from stored daily returns
print("\n--- CPCV (6 groups, 2 test, 5-day purge) ---")
cpcv_mean = None
cpcv_std = None
cpcv_passed = False
try:
    cpcv = run_cpcv(
        returns=base_daily_returns,
        strategy_fn=None,
        n_groups=6,
        k_test=2,
        purge_days=5,
    )
    cpcv_mean = cpcv.mean_oos_sharpe
    cpcv_std = cpcv.std_oos_sharpe
    cpcv_passed = cpcv.passed
    print(
        f"CPCV mean OOS Sharpe: {cpcv_mean:.4f}  std: {cpcv_std:.4f}  "
        f"n_paths: {cpcv.n_paths}  n_combos: {cpcv.n_combinations}"
    )
    print(f"Fold Sharpes: {[f'{s:.3f}' for s in (cpcv.oos_sharpes or [])]}")
    print(f"CPCV passed (mean > 0): {cpcv_passed}")
except Exception as e:
    print(f"CPCV error: {e}")
    import traceback
    traceback.print_exc()

# Gate evaluation
print("\n--- Robustness Gates (Track A) ---")
sharpe_gate = base["sharpe_ratio"] >= 0.80
maxdd_gate = base["max_drawdown"] < 0.15
dsr_gate = base["dsr"] >= 0.95
cpcv_gate = cpcv_passed and (cpcv_mean is not None and cpcv_mean > 0)
stability_gate = (n_stable / len(perturbations)) > 0.5

gates = {
    "sharpe>=0.80": (sharpe_gate, base["sharpe_ratio"]),
    "maxdd<0.15": (maxdd_gate, base["max_drawdown"]),
    "dsr>=0.95": (dsr_gate, base["dsr"]),
    "cpcv mean>0": (cpcv_gate, cpcv_mean),
    "perturb stability>50%": (stability_gate, f"{n_stable}/{len(perturbations)}"),
}

for name, (passed, val) in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {val}  [{status}]")

overall = all(p for p, _ in gates.values())
print(f"\nOVERALL ROBUSTNESS GATE: {'PASS' if overall else 'FAIL'}")

summary = {
    "slug": SLUG,
    "base_experiment_id": base["experiment_id"],
    "base_sharpe": base["sharpe_ratio"],
    "base_max_dd": base["max_drawdown"],
    "base_dsr": base["dsr"],
    "base_trades": base["total_trades"],
    "perturbation_stable": n_stable,
    "perturbation_total": len(perturbations),
    "perturbation_results": perturbation_results,
    "cpcv_mean_sharpe": cpcv_mean,
    "cpcv_std_sharpe": cpcv_std,
    "gates": {k: {"passed": p, "value": v} for k, (p, v) in gates.items()},
    "overall_gate": "pass" if overall else "fail",
}
out_path = Path(f"data/strategies/{SLUG}/robustness-result.yaml")
with out_path.open("w") as f:
    yaml.safe_dump(summary, f, sort_keys=False)
print(f"\nWrote {out_path}")

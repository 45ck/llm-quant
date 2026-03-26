#!/usr/bin/env python3
"""Robustness analysis for M3: spy-tlt-corr-sign-change"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.robustness import run_cpcv
from llm_quant.backtest.strategies import StrategyConfig, create_strategy
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "spy-tlt-corr-sign-change"
SYMBOLS = ["SPY", "TLT"]
YEARS = 5
STRAT = "correlation_regime"

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
base = exps[-1]
print(f"=== M3: {SLUG} ===")
print(
    f"Sharpe: {base['sharpe_ratio']:.4f}  MaxDD: {base['max_drawdown']:.4f}  DSR: {base['dsr']:.4f}  Trades: {base['total_trades']}"
)

lookback_days = YEARS * 365 + 60
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=lookback_days)
indicators_df = compute_indicators(prices_df)
cost = CostModel(spread_bps=3.0, flat_slippage_bps=1.0, slippage_volatility_factor=0.05)
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


# Run base to get daily returns for CPCV
print("\nRunning base strategy for daily returns...")
base_result = run_variant(base_params)
base_daily_returns = base_result["daily_returns"]

perturbations = [
    ("corr_window=10", {**base_params, "corr_window": 10}),
    ("corr_window=30", {**base_params, "corr_window": 30}),
    (
        "exit_thresh=0.05",
        {**base_params, "corr_exit_threshold": 0.05, "corr_entry_threshold": 0.05},
    ),
    (
        "exit_thresh=-0.05",
        {**base_params, "corr_exit_threshold": -0.05, "corr_entry_threshold": -0.05},
    ),
    ("weight=0.75", {**base_params, "spy_weight_risk_on": 0.75}),
]

print("\n--- Perturbation Analysis ---")
n_stable = 0
perturbation_results = []
for name, params in perturbations:
    r = run_variant(params)
    pct = (
        (r["sharpe"] - base["sharpe_ratio"]) / (abs(base["sharpe_ratio"]) + 1e-8) * 100
    )
    stable = abs(pct) <= 30
    if stable:
        n_stable += 1
    perturbation_results.append(
        {"name": name, "sharpe": r["sharpe"], "pct": pct, "stable": stable}
    )
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )
print(f"Perturbation stability: {n_stable}/5 stable")

print("\n--- CPCV (6 groups, 2 test, 5-day purge) ---")
cpcv_mean = None
cpcv_std = None
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
    if cpcv_mean is not None:
        print(
            f"CPCV mean OOS Sharpe: {cpcv_mean:.4f}  std: {cpcv_std:.4f}  n_paths: {cpcv.n_paths}  n_combos: {cpcv.n_combinations}"
        )
        print(f"Fold Sharpes: {[f'{s:.3f}' for s in (cpcv.oos_sharpes or [])]}")
        print(f"CPCV passed: {cpcv.passed}")
    else:
        print("CPCV: insufficient data (no result)")
except Exception as e:
    print(f"CPCV error: {e}")
    import traceback

    traceback.print_exc()

# Output summary JSON for results file
summary = {
    "slug": SLUG,
    "base_sharpe": base["sharpe_ratio"],
    "base_max_dd": base["max_drawdown"],
    "base_dsr": base["dsr"],
    "base_trades": base["total_trades"],
    "perturbation_stable": n_stable,
    "perturbation_results": perturbation_results,
    "cpcv_mean_sharpe": cpcv_mean,
    "cpcv_std_sharpe": cpcv_std,
}
with open(f"data/strategies/{SLUG}/robustness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to data/strategies/{SLUG}/robustness_summary.json")

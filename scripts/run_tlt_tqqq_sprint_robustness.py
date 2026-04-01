#!/usr/bin/env python3
"""Robustness analysis for tlt-tqqq-sprint (Track D).

Hypothesis: TLT 10-day momentum (with 3-day lag) predicts TQQQ direction.
When TLT has risen >= 1% over a 10-day window observed 3 days ago, go long TQQQ.

Track D gates: Sharpe >= 0.80, MaxDD < 40%, DSR >= 0.90, CPCV OOS Sharpe > 0.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "tlt-tqqq-sprint"
STRATEGY = "lead_lag"
SYMBOLS = ["TLT", "TQQQ"]
DD_THRESHOLD = 0.40  # Track D: relaxed to 40%
SHARPE_THRESHOLD = 0.80  # Track D minimum
DSR_THRESHOLD = 0.90  # Track D minimum (vs 0.95 for Track A)

BASE_PARAMS = {
    "leader_symbol": "TLT",
    "follower_symbol": "TQQQ",
    "lag_days": 3,
    "signal_window": 10,
    "entry_threshold": 0.01,
    "exit_threshold": -0.005,
    "target_weight": 0.30,
    "inverse": False,
    "rebalance_frequency_days": 1,
}

# Higher cost model for leveraged ETF (wider spreads)
cost_model = CostModel(
    spread_bps=10.0, flat_slippage_bps=5.0, slippage_volatility_factor=0.2
)

print("Fetching data...")
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365 + 30)
print("Computing indicators...")
indicators_df = compute_indicators(prices_df)


def run_single(params: dict) -> dict:
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=params.get("rebalance_frequency_days", 1),
        max_positions=2,
        target_position_weight=params.get("target_weight", 0.30),
        stop_loss_pct=0.10,
        parameters=dict(params),
    )
    strategy = create_strategy(STRATEGY, config)
    engine = BacktestEngine(strategy, initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=cost_model,
        warmup_days=30,
        cost_multiplier=1.0,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "total_return": m.total_return if m else 0.0,
        "total_trades": m.total_trades if m else 0,
        "dsr": m.dsr if m else 0.0,
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(
    returns: list[float], n_groups: int = 6, k: int = 2, purge: int = 5
) -> tuple[float, float, float]:
    from itertools import combinations

    n = len(returns)
    if n < n_groups:
        return 0.0, 0.0, 0.0
    group_size = n // n_groups
    oos_sharpes: list[float] = []
    for test_idx in combinations(range(n_groups), k):
        test_rets: list[float] = []
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


print("=" * 60)
print(f"ROBUSTNESS ANALYSIS: {SLUG} (Track D)")
print("=" * 60)

# --- Base run ---
print("\nRunning base configuration...")
base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std, cpcv_pct_pos = cpcv_sharpe(base["daily_returns"])
oos_is_ratio = cpcv_mean / base["sharpe"] if base["sharpe"] != 0 else 0.0

print("\n--- BASE RESULTS ---")
print(f"Base Sharpe:    {base['sharpe']:.4f}")
print(f"Max DD:         {base['max_dd']:.4f} ({base['max_dd'] * 100:.1f}%)")
print(f"Total Return:   {base['total_return']:.4f} ({base['total_return'] * 100:.1f}%)")
print(f"DSR:            {base['dsr']:.4f}")
print(f"Total Trades:   {base['total_trades']}")

print("\n--- CPCV RESULTS ---")
print(f"CPCV OOS Mean Sharpe: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS Ratio:    {oos_is_ratio:.4f}")
print(f"CPCV % Positive Folds: {cpcv_pct_pos:.1%}")

# --- Higher weight variants (as requested in issue) ---
print("\n--- HIGHER WEIGHT VARIANTS ---")
weight_variants = [
    ("target_weight=0.30 (base)", {**BASE_PARAMS, "target_weight": 0.30}),
    ("target_weight=0.50", {**BASE_PARAMS, "target_weight": 0.50}),
    ("target_weight=0.70", {**BASE_PARAMS, "target_weight": 0.70}),
]
weight_results = []
for name, params in weight_variants:
    r = run_single(params)
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({r['max_dd'] * 100:.1f}%) return={r['total_return'] * 100:.1f}%"
    )
    weight_results.append(
        {
            "variant": name,
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "total_return": round(r["total_return"], 4),
            "passes_track_d": bool(
                r["sharpe"] >= SHARPE_THRESHOLD and r["max_dd"] < DD_THRESHOLD
            ),
        }
    )

# --- Perturbation analysis ---
perturbations = [
    ("lag_days=2", {**BASE_PARAMS, "lag_days": 2}),
    ("lag_days=4", {**BASE_PARAMS, "lag_days": 4}),
    ("lag_days=5", {**BASE_PARAMS, "lag_days": 5}),
    ("signal_window=8", {**BASE_PARAMS, "signal_window": 8}),
    ("signal_window=12", {**BASE_PARAMS, "signal_window": 12}),
    ("entry_threshold=0.008", {**BASE_PARAMS, "entry_threshold": 0.008}),
    ("entry_threshold=0.012", {**BASE_PARAMS, "entry_threshold": 0.012}),
    ("entry_threshold=0.015", {**BASE_PARAMS, "entry_threshold": 0.015}),
]

print("\n--- PERTURBATION RESULTS ---")
perturbation_results = []
stable_count = 0
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = abs(pct) <= 30
    if stable:
        stable_count += 1
    perturbation_results.append(
        {
            "variant": name,
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "change_pct": round(pct, 1),
            "status": "STABLE" if stable else "UNSTABLE",
        }
    )
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# --- DSR from registry ---
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
dsr_value = base["dsr"]
if registry_path.exists():
    with registry_path.open() as f:
        exps = [json.loads(line) for line in f if line.strip()]
    if exps:
        dsr_value = exps[-1].get("dsr", dsr_value)

print("\n--- DSR ---")
print(f"DSR (from base run): {base['dsr']:.4f}")
print(f"DSR (from registry): {dsr_value:.4f}")

# --- Gate Assessment (Track D) ---
print(f"\n{'=' * 60}")
print("GATE ASSESSMENT (Track D)")
print(f"{'=' * 60}")

gate1 = base["sharpe"] >= SHARPE_THRESHOLD
gate2 = base["max_dd"] < DD_THRESHOLD
gate3 = base["dsr"] >= DSR_THRESHOLD
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 60

gates = [
    (f"Gate 1: Sharpe >= {SHARPE_THRESHOLD}", gate1, f"{base['sharpe']:.4f}"),
    (
        f"Gate 2: MaxDD < {DD_THRESHOLD * 100:.0f}%",
        gate2,
        f"{base['max_dd'] * 100:.1f}%",
    ),
    (f"Gate 3: DSR >= {DSR_THRESHOLD}", gate3, f"{base['dsr']:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    ("Gate 5: Perturbation >= 60% stable", gate5, f"{pct_stable:.0f}%"),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

all_pass = all(g[1] for g in gates)
verdict = "PASS — ALL TRACK D GATES CLEARED" if all_pass else "FAIL"
print(f"\n  VERDICT: {verdict}")

# --- Save robustness results ---
output = {
    "strategy_slug": SLUG,
    "strategy_type": STRATEGY,
    "track": "D",
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "dsr": round(base["dsr"], 4),
    "cpcv": {
        "oos_mean_sharpe": round(cpcv_mean, 4),
        "oos_std": round(cpcv_std, 4),
        "oos_is_ratio": round(oos_is_ratio, 4),
        "pct_positive_folds": round(cpcv_pct_pos, 4),
    },
    "weight_variants": weight_results,
    "perturbation": {
        "variants": perturbation_results,
        "pct_stable": round(pct_stable, 1),
    },
    "gates": {
        f"sharpe_gte_{SHARPE_THRESHOLD}": gate1,
        f"maxdd_lt_{DD_THRESHOLD * 100:.0f}pct": gate2,
        f"dsr_gte_{DSR_THRESHOLD}": gate3,
        "cpcv_oos_positive": gate4,
        "perturbation_gte_60pct": gate5,
    },
    "verdict": "PASS" if all_pass else "FAIL",
}

out_yaml = Path(f"data/strategies/{SLUG}/robustness.yaml")
out_yaml.parent.mkdir(parents=True, exist_ok=True)
with open(out_yaml, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)

out_json = Path(f"data/strategies/{SLUG}/robustness_results.json")
with open(out_json, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved YAML to {out_yaml}")
print(f"Saved JSON to {out_json}")

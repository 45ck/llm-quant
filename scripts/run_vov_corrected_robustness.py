#!/usr/bin/env python3
"""Robustness for vov-spy-defensive after VoV percentile logic fix."""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import yaml as _yaml

sys.path.insert(0, "src")

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import StrategyConfig, create_strategy
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "vov-spy-defensive"
SYMBOLS = ["SPY", "VIX"]
YEARS = 5
STRAT = "vix_regime"
DD_THRESHOLD = 0.12  # mandate: < 12%

registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as fh:
    exps = [json.loads(line) for line in fh if line.strip()]
if not exps:
    print("ERROR: No experiments found")
    sys.exit(1)

base_exp = exps[-1]
print(f"=== {SLUG} ===")
sharpe = base_exp["sharpe_ratio"]
max_dd = base_exp["max_drawdown"]
dsr = base_exp["dsr"]
trades = base_exp["total_trades"]
print(f"Sharpe={sharpe:.4f}  MaxDD={max_dd:.4f}  DSR={dsr:.4f}  Trades={trades}")

lookback_days = YEARS * 365  # match run_backtest.py
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=lookback_days)
indicators_df = compute_indicators(prices_df)
cost = CostModel(spread_bps=3.0, flat_slippage_bps=1.0, slippage_volatility_factor=0.05)
base_params = base_exp["parameters"]
exp_id = base_exp["experiment_id"]

# Load daily_returns from saved artifact (avoids re-running base strategy)
_artifact_path = Path(f"data/strategies/{SLUG}/experiments/{exp_id}.yaml")
with _artifact_path.open() as _f:
    _artifact = _yaml.safe_load(_f)
base_daily_returns = _artifact.get("daily_returns", [])
print(f"Loaded {len(base_daily_returns)} daily returns from artifact {exp_id}")


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
        fill_delay=1,
        warmup_days=50,
        cost_multiplier=1.0,
        trial_count=1,
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


# base daily returns loaded from artifact above

print()
print("=== PERTURBATION ANALYSIS ===")
perturbations = [
    ("vov_window=20", {**base_params, "vov_window": 20}),
    ("vov_window=45", {**base_params, "vov_window": 45}),
    ("vov_percentile=0.70", {**base_params, "vov_percentile": 0.70}),
    ("vov_percentile=0.90", {**base_params, "vov_percentile": 0.90}),
    ("target_weight=0.70", {**base_params, "target_weight": 0.70}),
]
n_stable = 0
perturbation_results = []
for name, params in perturbations:
    r = run_variant(params)
    pct = (r["sharpe"] - sharpe) / (abs(sharpe) + 1e-8) * 100
    tag = "STABLE" if abs(pct) <= 30 else "UNSTABLE"
    if abs(pct) <= 30:
        n_stable += 1
    perturbation_results.append((name, r["sharpe"], r["max_dd"], pct, tag))
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {tag}"
    )
perturb_pass = n_stable >= 3
print(f"Perturbation: {n_stable}/5 ({'PASS' if perturb_pass else 'FAIL'})")
print()
print("=== CPCV (6 groups, 2 test groups, 5-day purge) ===")
cpcv_mean, cpcv_std = cpcv_sharpe(base_daily_returns)
cpcv_pass = cpcv_mean > 0
print(f"  Mean OOS Sharpe: {cpcv_mean:.4f}  Std: {cpcv_std:.4f}")

dsr_pass = dsr >= 0.95
dd_pass = max_dd < DD_THRESHOLD
print()
print("=== DSR ===")
print(f"  DSR: {dsr:.4f} (threshold >= 0.95)")
print()
print("=== MAX DRAWDOWN CHECK ===")
dd_label = "PASS" if dd_pass else "FAIL"
print(f"  MaxDD: {max_dd * 100:.2f}% (mandate threshold < 12.0%) -> {dd_label}")
gates = [
    ("Perturbation stability >= 60%", perturb_pass),
    ("CPCV mean OOS Sharpe > 0", cpcv_pass),
    ("DSR >= 0.95", dsr_pass),
    ("MaxDD < 12%", dd_pass),
]
all_pass = all(v for _, v in gates)
print()
print("=== OVERALL VERDICT ===")
for gate_name, gate_pass_v in gates:
    status = "PASS" if gate_pass_v else "FAIL"
    print(f"  [{status}] {gate_name}")
verdict = "PASS -- all 4 gates met" if all_pass else "FAIL -- one or more gates not met"
print(f"  Overall: {verdict}")

notes_path = Path(f"data/strategies/{SLUG}/robustness-v2-notes.txt")
out = []
out.append("VOV-SPY-DEFENSIVE: Corrected Robustness Analysis (v2)")
out.append("=" * 60)
out.append("Date: 2026-03-26")
out.append(f"Experiment ID: {base_exp['experiment_id']}")
out.append("Implementation fix: vix_symbol=VIX (was ^VIX), VoV tail limit removed")
out.append("")
out.append("BASE METRICS")
out.append("-" * 40)
out.append(f"Sharpe Ratio:    {sharpe:.4f}")
out.append(f"Max Drawdown:    {max_dd * 100:.2f}%")
out.append(f"DSR:             {dsr:.4f}")
out.append(f"Total Trades:    {trades}")
out.append(f"Total Return:    {base_exp['total_return'] * 100:.2f}%")
out.append("")
out.append("PERTURBATION ANALYSIS")
out.append("-" * 40)
for pname, sh, md, pct_v, tag_v in perturbation_results:
    out.append(f"  {pname}: sharpe={sh:.4f} max_dd={md:.4f} ({pct_v:+.1f}%) {tag_v}")
plabel = "PASS" if perturb_pass else "FAIL"
out.append(f"  Result: {n_stable}/5 stable ({plabel} >= 3/5)")
out.append("")
out.append("CPCV (6 groups, 2 test, 5-day purge)")
out.append("-" * 40)
out.append(f"  Mean OOS Sharpe: {cpcv_mean:.4f}")
out.append(f"  Std OOS Sharpe:  {cpcv_std:.4f}")
clabel = "PASS" if cpcv_pass else "FAIL"
out.append(f"  Result: {clabel} (mean OOS > 0)")
out.append("")
out.append("GATE RESULTS")
out.append("-" * 40)
for gate_name, gate_pass_v in gates:
    status = "PASS" if gate_pass_v else "FAIL"
    out.append(f"  [{status}] {gate_name}")
out.append("")
out.append("OVERALL VERDICT")
out.append("-" * 40)
if all_pass:
    out.append("PASS -- all 4 robustness gates met.")
else:
    failing = [g for g, v in gates if not v]
    out.append("FAIL -- gates not met: " + ", ".join(failing))
    out.append("")
    if not dd_pass:
        out.append(
            f"PRIMARY FAILURE: MaxDD={max_dd * 100:.2f}% exceeds mandate threshold of 12%."
        )
        out.append(f"VoV signal functional: Sharpe={sharpe:.3f}, trades={trades}")
        out.append("Strategy does not reduce drawdown vs buy-and-hold SPY.")
        out.append(
            "Remedies: (1) tighten stop-loss, (2) add hedge, (3) revise mandate, (4) retire."
        )
    if not dsr_pass:
        out.append(f"DSR FAILURE: {dsr:.4f} below 0.95 threshold.")
sep = chr(10)
notes_path.write_text(sep.join(out), encoding="utf-8")
print(f"Notes written to: {notes_path}")

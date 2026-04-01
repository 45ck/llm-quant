#!/usr/bin/env python3
"""Robustness analysis for bond-curve-trade-v1 (F20 Bond Curve Trade).

Mechanism: TLT/IEF price ratio z-score mean-reversion.
  - Z > 1.0 (TLT expensive, curve too flat): 60% IEF + 20% TLT
  - Z < -1.0 (TLT cheap, curve too steep): 60% TLT + 20% IEF
  - Else (neutral): 40% TLT + 40% IEF

This is a pure fixed income relative value trade with ZERO equity exposure.
The benchmark for shuffled signal test is 50/50 TLT/IEF (equal weight bond blend).

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_bond_curve_trade_robustness.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, "src")

import polars as pl

from llm_quant.backtest.robustness import shuffled_signal_test
from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "bond-curve-trade-v1"
SYMBOLS = ["TLT", "IEF"]
DD_THRESHOLD = 0.15
LOOKBACK_DAYS = 5 * 365
WARMUP = 90

BASE_PARAMS = {
    "bb_window": 60,
    "bb_std": 1.0,
    "overweight": 0.60,
    "underweight": 0.20,
    "neutral_weight": 0.40,
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series — use TLT as the date spine
tlt_df = prices.filter(pl.col("symbol") == "TLT").sort("date")
dates = tlt_df["date"].to_list()
n = len(dates)
print(f"Trading days: {n}")

sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )


def asset_ret(sym, i):
    """Get daily return for asset at day i."""
    d, dp = dates[i], dates[i - 1]
    data = sym_data[sym]
    if d in data and dp in data and data[dp] > 0:
        return data[d] / data[dp] - 1
    return 0.0


# Compute 50/50 TLT/IEF benchmark returns (for shuffled signal test)
benchmark_rets = [0.0] + [
    0.5 * asset_ret("TLT", i) + 0.5 * asset_ret("IEF", i) for i in range(1, n)
]


def _compute_metrics(daily_returns):
    """Compute Sharpe, MaxDD, total return from daily returns."""
    if not daily_returns or len(daily_returns) < 60:
        return {"sharpe": 0.0, "max_dd": 0.0, "total_return": 0.0, "daily_returns": []}

    nav = [1.0]
    for r in daily_returns:
        nav.append(nav[-1] * (1.0 + r))

    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    mean = sum(daily_returns) / len(daily_returns)
    std = (sum((r - mean) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0
    total_ret = nav[-1] / nav[0] - 1.0

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_ret,
        "daily_returns": daily_returns,
    }


def run_single(params):
    """Run a single backtest with the given parameters."""
    bb_window = int(params.get("bb_window", 60))
    bb_std_mult = float(params.get("bb_std", 1.0))
    overweight = float(params.get("overweight", 0.60))
    underweight = float(params.get("underweight", 0.20))
    neutral_weight = float(params.get("neutral_weight", 0.40))

    daily_returns = []
    cost_per_switch = 0.0003  # 3 bps

    # Pre-compute TLT/IEF ratio series aligned to dates
    ratios = []
    for i in range(n):
        d = dates[i]
        tlt_p = sym_data["TLT"].get(d, 0.0)
        ief_p = sym_data["IEF"].get(d, 0.0)
        if tlt_p > 0 and ief_p > 0:
            ratios.append(tlt_p / ief_p)
        else:
            ratios.append(None)

    prev_regime = None
    for i in range(WARMUP, n):
        if i < bb_window + 1:
            daily_returns.append(0.0)
            continue

        # Z-score of TLT/IEF ratio at close i-1 (lag-1, no look-ahead)
        # Use the window ending at i-1
        window_ratios = []
        for j in range(i - bb_window, i):
            if ratios[j] is not None:
                window_ratios.append(ratios[j])

        if len(window_ratios) < bb_window // 2:
            daily_returns.append(0.0)
            continue

        current_ratio = ratios[i - 1]
        if current_ratio is None:
            daily_returns.append(0.0)
            continue

        sma = sum(window_ratios) / len(window_ratios)
        std = (sum((r - sma) ** 2 for r in window_ratios) / len(window_ratios)) ** 0.5

        if std < 1e-10:
            daily_returns.append(0.0)
            continue

        z_score = (current_ratio - sma) / std

        # Determine regime and weights based on z-score
        if z_score > bb_std_mult:
            # TLT expensive (curve too flat) -> overweight IEF, underweight TLT
            regime = "flat"
            day_ret = (
                asset_ret("IEF", i) * overweight + asset_ret("TLT", i) * underweight
            )
        elif z_score < -bb_std_mult:
            # TLT cheap (curve too steep) -> overweight TLT, underweight IEF
            regime = "steep"
            day_ret = (
                asset_ret("TLT", i) * overweight + asset_ret("IEF", i) * underweight
            )
        else:
            # Neutral zone -> equal-ish weight
            regime = "neutral"
            day_ret = (
                asset_ret("TLT", i) * neutral_weight
                + asset_ret("IEF", i) * neutral_weight
            )

        # Apply switching cost
        if prev_regime is not None and regime != prev_regime:
            day_ret -= cost_per_switch
        prev_regime = regime

        daily_returns.append(day_ret)

    return _compute_metrics(daily_returns)


def cpcv_sharpe(returns, n_groups=6, k=2, purge=5):
    """Combinatorial Purged Cross-Validation (inline implementation)."""
    from itertools import combinations

    n_r = len(returns)
    if n_r < n_groups:
        return 0.0, 0.0, 0.0
    group_size = n_r // n_groups
    oos_sharpes = []
    for test_idx in combinations(range(n_groups), k):
        test_rets = []
        for g in test_idx:
            s, e = g * group_size + purge, (g + 1) * group_size - purge
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


# ==============================================================================
# RUN BASE
# ==============================================================================
print("=" * 70)
print(f"ROBUSTNESS ANALYSIS: {SLUG}")
print("=" * 70)
print("Base parameters:")
for k, v in BASE_PARAMS.items():
    print(f"  {k} = {v}")

print("\n--- RUNNING BASE BACKTEST ---")
base = run_single(BASE_PARAMS)
print(f"Base Sharpe: {base['sharpe']:.4f}")
print(f"Base MaxDD:  {base['max_dd']:.4f}")
print(f"Base Return: {base['total_return']:.4f}")

# ==============================================================================
# CPCV
# ==============================================================================
print("\n--- CPCV (Combinatorial Purged Cross-Validation) ---")
cpcv_mean, cpcv_std, cpcv_pct_pos = cpcv_sharpe(base["daily_returns"])
oos_is_ratio = cpcv_mean / base["sharpe"] if base["sharpe"] != 0 else 0.0
print(f"CPCV OOS Mean Sharpe:  {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS Ratio:     {oos_is_ratio:.4f}")
print(f"CPCV % Positive Folds: {cpcv_pct_pos:.1%}")

# ==============================================================================
# PERTURBATION TESTS
# ==============================================================================
perturbations = [
    ("bb_window=45", {**BASE_PARAMS, "bb_window": 45}),
    ("bb_window=90", {**BASE_PARAMS, "bb_window": 90}),
    ("bb_std=0.75", {**BASE_PARAMS, "bb_std": 0.75}),
    ("bb_std=1.5", {**BASE_PARAMS, "bb_std": 1.5}),
    ("overweight=0.50", {**BASE_PARAMS, "overweight": 0.50}),
    ("overweight=0.70", {**BASE_PARAMS, "overweight": 0.70}),
]

print("\n--- PERTURBATION RESULTS ---")
perturbation_results = []
stable_count = 0
for name, params in perturbations:
    print(f"  Running: {name}...", end=" ", flush=True)
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
    print(f"sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {stable}")

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# ==============================================================================
# SHUFFLED SIGNAL TEST
# ==============================================================================
print("\n--- SHUFFLED SIGNAL TEST ---")
# Use 50/50 TLT/IEF benchmark as the asset returns for shuffled test
bench_daily_rets = benchmark_rets[WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(bench_daily_rets))
aligned_strat = strat_returns[-n_min:]
aligned_bench = bench_daily_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  Benchmark (50/50 TLT/IEF) returns: {len(aligned_bench)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_bench,
    n_shuffles=1000,
    seed=42,
)
print(f"  Real Sharpe:     {shuffled_result.real_sharpe:.4f}")
print(f"  Shuffled Mean:   {shuffled_result.shuffled_mean:.4f}")
print(f"  Shuffled 95th:   {shuffled_result.shuffled_95th:.4f}")
print(f"  Shuffled 99th:   {shuffled_result.shuffled_99th:.4f}")
print(f"  p-value:         {shuffled_result.p_value:.4f}")
print(f"  PASSED:          {shuffled_result.passed}")

# ==============================================================================
# DSR
# ==============================================================================
print("\n--- DSR ---")
sr = base["sharpe"]
T = len(base["daily_returns"])

T_years = T / 252
se_sr = math.sqrt((1 + sr**2 / 2) / T_years) if T_years > 0 else 1.0
dsr_value = float(stats.norm.cdf(sr / se_sr)) if se_sr > 0 else 0.0
print(f"  DSR (computed inline): {dsr_value:.4f}")

# ==============================================================================
# GATE ASSESSMENT
# ==============================================================================
print(f"\n{'=' * 70}")
print("GATE ASSESSMENT")
print(f"{'=' * 70}")

gate1 = base["sharpe"] > 0.80
gate2 = base["max_dd"] < DD_THRESHOLD
gate3 = dsr_value >= 0.95
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 60
gate6 = shuffled_result.passed

gates = [
    ("Gate 1: Sharpe > 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 15%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.95", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    ("Gate 5: Perturbation >= 60% stable", gate5, f"{pct_stable:.0f}%"),
    (
        "Gate 6: Shuffled Signal p < 0.05",
        gate6,
        f"p={shuffled_result.p_value:.4f}",
    ),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

all_pass = all(g[1] for g in gates)
verdict = "PASS - ALL GATES CLEARED" if all_pass else "FAIL"
print(f"\n  VERDICT: {verdict}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output = {
    "strategy_slug": SLUG,
    "strategy_type": "bond_curve_trade",
    "mechanism_family": "F20_bond_curve_trade",
    "base_params": BASE_PARAMS,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
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
    "shuffled_signal": {
        "real_sharpe": round(shuffled_result.real_sharpe, 4),
        "shuffled_mean": round(shuffled_result.shuffled_mean, 4),
        "shuffled_95th": round(shuffled_result.shuffled_95th, 4),
        "shuffled_99th": round(shuffled_result.shuffled_99th, 4),
        "p_value": round(shuffled_result.p_value, 4),
        "n_shuffles": shuffled_result.n_shuffles,
        "passed": shuffled_result.passed,
    },
    "gates": {
        "sharpe_gt_0.80": gate1,
        "maxdd_lt_15pct": gate2,
        "dsr_gte_0.95": gate3,
        "cpcv_oos_positive": gate4,
        "perturbation_gte_60pct": gate5,
        "shuffled_signal_passed": gate6,
    },
    "verdict": "PASS" if all_pass else "FAIL",
}

out_path = Path(f"data/strategies/{SLUG}/robustness.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")

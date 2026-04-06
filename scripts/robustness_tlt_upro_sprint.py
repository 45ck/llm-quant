#!/usr/bin/env python3
"""Robustness analysis for tlt-upro-sprint (Track D -- Sprint Alpha).

Mechanism: TLT (20+ year Treasury) 10-day return as lead signal
-> UPRO (3x leveraged S&P 500) follower. Treasury duration signal
leads equity via the rate-sensitivity channel: falling yields (rising TLT)
signal risk-on, rising yields (falling TLT) signal risk-off.

Signal logic:
  - TLT 10-day return >= entry_threshold (1.0%) -> entry (buy UPRO at target_weight)
  - TLT 10-day return <= exit_threshold (-0.5%) -> exit to SHY (cash proxy)
  - Use signal from lag_days (3) ago -- no look-ahead
  - VIX crash filter: VIX > threshold (30) -> 100% SHY (override all signals)
  - Daily rebalancing (rebalance_frequency=1 per research-spec)
  - Cost per switch: 20 bps round-trip

Track D gates (leveraged):
  - Gate 1: Sharpe >= 0.80
  - Gate 2: MaxDD < 40%
  - Gate 3: DSR >= 0.90
  - Gate 4: CPCV OOS > 0
  - Gate 5: Perturbation >= 40% stable
  - Gate 6: Shuffled signal p < 0.05

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/robustness_tlt_upro_sprint.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, "src")

import polars as pl

from llm_quant.backtest.robustness import shuffled_signal_test
from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "tlt-upro-sprint"
SYMBOLS = ["TLT", "UPRO", "SHY", "VIX"]
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

# Base parameters from experiment-registry.jsonl and research-spec.yaml
BASE_PARAMS = {
    "entry_threshold": 0.01,  # TLT 10-day return >= 1.0% -> buy UPRO
    "exit_threshold": -0.005,  # TLT 10-day return <= -0.5% -> exit
    "lag_days": 3,  # Use signal from 3 days ago
    "signal_window": 10,  # 10-day return lookback for TLT
    "target_weight": 0.30,  # 30% position in UPRO
    "rebalance_frequency": 1,  # Daily rebalancing (per research-spec)
    "vix_crash_threshold": 30,  # VIX > 30 -> 100% SHY
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series -- use UPRO as the date backbone
upro_df = prices.filter(pl.col("symbol") == "UPRO").sort("date")
dates = upro_df["date"].to_list()
upro_close = upro_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# Precompute UPRO daily returns
upro_rets = [0.0] + [
    (upro_close[i] / upro_close[i - 1] - 1) if upro_close[i - 1] > 0 else 0
    for i in range(1, n)
]


def asset_ret(sym: str, i: int) -> float:
    """Get daily return for asset at day i."""
    d, dp = dates[i], dates[i - 1]
    data = sym_data[sym]
    if d in data and dp in data and data[dp] > 0:
        return data[d] / data[dp] - 1
    return 0.0


def get_vix(i: int) -> float:
    """Get VIX close value at day i."""
    d = dates[i]
    return sym_data["VIX"].get(d, 0.0)


def _compute_metrics(daily_returns: list[float]) -> dict:
    """Compute Sharpe, MaxDD, total return, CAGR from daily returns."""
    if not daily_returns or len(daily_returns) < 60:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "daily_returns": [],
        }

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
    years = len(daily_returns) / 252
    cagr = ((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_ret,
        "cagr": cagr,
        "daily_returns": daily_returns,
    }


def run_single(params: dict) -> dict:
    """Run a single backtest with the given parameters.

    Signal logic (with lag + rebalancing + VIX crash filter):
    - Compute TLT return over signal_window days
    - Use the signal from lag_days ago (causal)
    - Evaluate signal every rebalance_frequency days
    - VIX crash filter: VIX > vix_crash_threshold -> 100% SHY (override)
    - If TLT return >= entry_threshold -> hold UPRO at target_weight, rest in SHY
    - If TLT return <= exit_threshold -> exit to SHY (100%)
    - Otherwise hold current position
    """
    entry_thresh = float(params.get("entry_threshold", 0.01))
    exit_thresh = float(params.get("exit_threshold", -0.005))
    lag = int(params.get("lag_days", 3))
    window = int(params.get("signal_window", 10))
    tw = float(params.get("target_weight", 0.30))
    rebal_freq = int(params.get("rebalance_frequency", 1))
    vix_thresh = float(params.get("vix_crash_threshold", 30))
    cost_per_switch = 0.0020  # 20 bps round-trip

    daily_returns: list[float] = []
    in_position = False
    n_trades = 0
    min_lookback = WARMUP + window + lag
    days_since_rebal = 0

    for i in range(WARMUP, n):
        if i < min_lookback:
            daily_returns.append(asset_ret("SHY", i))
            continue

        prev_position = in_position

        # VIX crash filter: override everything when VIX is extreme
        vix_val = get_vix(i)
        if vix_val > vix_thresh:
            in_position = False
        else:
            # Only evaluate signal on rebalance days
            days_since_rebal += 1
            if days_since_rebal >= rebal_freq:
                days_since_rebal = 0

                # Signal date: lag_days ago
                signal_idx = i - lag
                if signal_idx < window:
                    daily_returns.append(asset_ret("SHY", i))
                    continue

                # TLT return over signal_window ending at signal_idx
                d_signal = dates[signal_idx]
                d_signal_lb = dates[signal_idx - window]

                tlt_now = sym_data["TLT"].get(d_signal, 0.0)
                tlt_lb = sym_data["TLT"].get(d_signal_lb, 0.0)

                if tlt_now <= 0 or tlt_lb <= 0:
                    daily_returns.append(asset_ret("SHY", i))
                    continue

                tlt_ret = tlt_now / tlt_lb - 1.0

                # Position logic -- only update on rebalance days
                if tlt_ret >= entry_thresh:
                    in_position = True
                elif tlt_ret <= exit_thresh:
                    in_position = False
                # else: hold current state

        # Compute daily return based on current position
        if in_position:
            day_ret = upro_rets[i] * tw + asset_ret("SHY", i) * (1.0 - tw)
        else:
            day_ret = asset_ret("SHY", i)

        # Apply switching cost on state change
        if prev_position != in_position:
            day_ret -= cost_per_switch
            n_trades += 1

        daily_returns.append(day_ret)

    result = _compute_metrics(daily_returns)
    result["n_trades"] = n_trades
    return result


def cpcv_sharpe(
    returns: list[float], n_groups: int = 15, k: int = 3, purge: int = 5
) -> tuple[float, float, float]:
    """Combinatorial Purged Cross-Validation (inline implementation).

    Uses 15 groups, 3 test groups, 5-day purge -- consistent with other sprint scripts.
    """
    from itertools import combinations

    n_r = len(returns)
    if n_r < n_groups * 10:
        print(f"  WARNING: Only {n_r} returns, need {n_groups * 10} for CPCV")
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
print(f"Base Sharpe:  {base['sharpe']:.4f}")
print(f"Base MaxDD:   {base['max_dd']:.4f}")
print(f"Base Return:  {base['total_return']:.4f}")
print(f"Base CAGR:    {base['cagr']:.4f}")
print(f"Trades:       {base['n_trades']}")

# ==============================================================================
# CPCV (15 groups, 3 test, 5-day purge)
# ==============================================================================
print("\n--- CPCV (Combinatorial Purged Cross-Validation) ---")
print("  Config: n_groups=15, k=3, purge=5")
cpcv_mean, cpcv_std, cpcv_pct_pos = cpcv_sharpe(
    base["daily_returns"], n_groups=15, k=3, purge=5
)
oos_is_ratio = cpcv_mean / base["sharpe"] if base["sharpe"] != 0 else 0.0
print(f"CPCV OOS Mean Sharpe:  {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS Ratio:     {oos_is_ratio:.4f}")
print(f"CPCV % Positive Folds: {cpcv_pct_pos:.1%}")

# ==============================================================================
# PERTURBATION TESTS (22 variants covering all 7 parameter axes)
# ==============================================================================
# Per user spec:
#   lag_days: [1, 2, 3, 7, 10] -- base is 3, so test 1, 2, 7, 10
#   signal_window: [5, 7, 15, 20] -- base is 10
#   entry_threshold: +/-50% from base 0.01 -> 0.005, 0.015
#   exit_threshold: +/-50% from base -0.005 -> -0.0025, -0.0075
#   target_weight: [0.20, 0.25, 0.40, 0.50] -- base is 0.30
#   rebalance_frequency: [1, 3, 5, 7] -- base is 1, so test 3, 5, 7
#   vix_crash_threshold: [25, 35, 40] -- base is 30
perturbations = [
    # lag_days variants (base=3)
    ("lag=1", {**BASE_PARAMS, "lag_days": 1}),
    ("lag=2", {**BASE_PARAMS, "lag_days": 2}),
    ("lag=7", {**BASE_PARAMS, "lag_days": 7}),
    ("lag=10", {**BASE_PARAMS, "lag_days": 10}),
    # signal_window variants (base=10)
    ("window=5", {**BASE_PARAMS, "signal_window": 5}),
    ("window=7", {**BASE_PARAMS, "signal_window": 7}),
    ("window=15", {**BASE_PARAMS, "signal_window": 15}),
    ("window=20", {**BASE_PARAMS, "signal_window": 20}),
    # entry_threshold variants (base=0.01, +/-50%)
    ("entry=0.5%", {**BASE_PARAMS, "entry_threshold": 0.005}),
    ("entry=1.5%", {**BASE_PARAMS, "entry_threshold": 0.015}),
    # exit_threshold variants (base=-0.005, +/-50%)
    ("exit=-0.25%", {**BASE_PARAMS, "exit_threshold": -0.0025}),
    ("exit=-0.75%", {**BASE_PARAMS, "exit_threshold": -0.0075}),
    # target_weight variants (base=0.30)
    ("weight=0.20", {**BASE_PARAMS, "target_weight": 0.20}),
    ("weight=0.25", {**BASE_PARAMS, "target_weight": 0.25}),
    ("weight=0.40", {**BASE_PARAMS, "target_weight": 0.40}),
    ("weight=0.50", {**BASE_PARAMS, "target_weight": 0.50}),
    # rebalance_frequency variants (base=1)
    ("rebal=3", {**BASE_PARAMS, "rebalance_frequency": 3}),
    ("rebal=5", {**BASE_PARAMS, "rebalance_frequency": 5}),
    ("rebal=7", {**BASE_PARAMS, "rebalance_frequency": 7}),
    # vix_crash_threshold variants (base=30)
    ("vix_crash=25", {**BASE_PARAMS, "vix_crash_threshold": 25}),
    ("vix_crash=35", {**BASE_PARAMS, "vix_crash_threshold": 35}),
    ("vix_crash=40", {**BASE_PARAMS, "vix_crash_threshold": 40}),
]

print("\n--- PERTURBATION RESULTS ---")
print(
    f"  {'Variant':<25} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} "
    f"{'Change%':>9} {'Status':>10}"
)
print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 10}")
perturbation_results = []
stable_count = 0
for name, params in perturbations:
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
            "cagr": round(r["cagr"], 4),
            "change_pct": round(pct, 1),
            "status": stable,
        }
    )
    print(
        f"  {name:<25} {r['sharpe']:>8.4f} {r['max_dd']:>8.4f} "
        f"{r['cagr']:>8.4f} {pct:>+8.1f}% {stable:>10}"
    )

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# ==============================================================================
# SHUFFLED SIGNAL TEST (1000 shuffles)
# ==============================================================================
print("\n--- SHUFFLED SIGNAL TEST (1000 shuffles) ---")
# Use UPRO returns as the asset baseline for shuffled test
upro_daily_rets = upro_rets[WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(upro_daily_rets))
aligned_strat = strat_returns[-n_min:]
aligned_upro = upro_daily_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  UPRO returns:     {len(aligned_upro)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_upro,
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
# GATE ASSESSMENT -- Track D gates
# ==============================================================================
print(f"\n{'=' * 70}")
print("GATE ASSESSMENT (Track D -- Leveraged Sprint)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] >= 0.80
gate2 = base["max_dd"] < DD_THRESHOLD  # 40% for Track D
gate3 = dsr_value >= 0.90
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 40  # Track D minimum
gate6 = shuffled_result.passed

gates = [
    ("Gate 1: Sharpe >= 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    (
        "Gate 5: Perturbation >= 40% stable",
        gate5,
        f"{pct_stable:.0f}% ({stable_count}/{len(perturbations)})",
    ),
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
    "strategy_type": "leveraged_lead_lag",
    "track": "D",
    "mechanism": "TLT (20+ year Treasury) 10-day return -> UPRO (3x leveraged S&P 500)",
    "key_differentiator": "Treasury duration signal leads equity via rate-sensitivity channel; VIX crash filter for tail risk",
    "parent_strategy": "tlt-spy-lead-lag (Treasury-equity lead-lag family)",
    "base_params": BASE_PARAMS,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "base_cagr": round(base["cagr"], 4),
    "n_trades": base["n_trades"],
    "dsr": round(dsr_value, 4),
    "cpcv": {
        "config": "n_groups=15, k=3, purge=5",
        "oos_mean_sharpe": round(cpcv_mean, 4),
        "oos_std": round(cpcv_std, 4),
        "oos_is_ratio": round(oos_is_ratio, 4),
        "pct_positive_folds": round(cpcv_pct_pos, 4),
    },
    "perturbation": {
        "variants": perturbation_results,
        "pct_stable": round(pct_stable, 1),
        "stable_count": stable_count,
        "total_variants": len(perturbations),
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
        "sharpe_gte_0.80": gate1,
        "maxdd_lt_40pct": gate2,
        "dsr_gte_0.90": gate3,
        "cpcv_oos_positive": gate4,
        "perturbation_gte_40pct": gate5,
        "shuffled_signal_passed": gate6,
    },
    "verdict": "PASS" if all_pass else "FAIL",
}

out_path = Path(f"data/strategies/{SLUG}/robustness.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved robustness results to {out_path}")

# Also save as JSON for programmatic access
json_path = Path(f"data/strategies/{SLUG}/robustness_results.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON results to {json_path}")

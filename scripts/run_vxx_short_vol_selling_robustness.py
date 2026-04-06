#!/usr/bin/env python3
"""Robustness analysis for vxx-short-vol-selling-v2 (Track B).

Mechanism: VIX term structure (VIX3M/VIX ratio) as entry/exit signal for SVXY
(inverse VIX short-term ETF, equivalent to short VXX). Harvests VIX futures
contango roll yield in normal markets, exits to SHY during backwardation.

Signal logic (with 1-day lag for causality):
  - VIX3M/VIX ratio (lagged) > entry_threshold (1.05) -> hold SVXY at target_weight
  - VIX3M/VIX ratio (lagged) < exit_threshold (1.00) -> exit to 100% SHY
  - Between thresholds -> hold current position (hysteresis)
  - Cost per switch: 3 bps

Track B gates:
  - Gate 1: Sharpe > 1.0
  - Gate 2: MaxDD < 30%
  - Gate 3: DSR >= 0.95
  - Gate 4: CPCV OOS > 0
  - Gate 5: Perturbation >= 60% stable
  - Gate 6: Shuffled signal p < 0.05

References: Simon & Campasano 2014, Whaley 2013, Carr & Wu 2009.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_vxx_short_vol_selling_robustness.py
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

SLUG = "vxx-short-vol-selling-v2"
TRADE_SYMBOLS = ["SVXY", "SHY"]
SIGNAL_SYMBOLS = ["VIX", "VIX3M"]
ALL_SYMBOLS = TRADE_SYMBOLS + SIGNAL_SYMBOLS
DD_THRESHOLD = 0.30  # Track B: 30% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

BASE_PARAMS = {
    "entry_threshold": 1.05,  # VIX3M/VIX > 1.05 -> confirmed contango, hold SVXY
    "exit_threshold": 1.00,  # VIX3M/VIX < 1.00 -> backwardation, exit to SHY
    "lag_days": 1,  # Use signal from 1 day ago (causality)
    "target_weight": 0.50,  # 50% in SVXY when active, rest in SHY
}

# ==============================================================================
# FETCH DATA
# ==============================================================================
print("Fetching data...")
prices = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series -- use SVXY as the date backbone
svxy_df = prices.filter(pl.col("symbol") == "SVXY").sort("date")
dates = svxy_df["date"].to_list()
svxy_close = svxy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

sym_data: dict[str, dict] = {}
for sym in ALL_SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# Precompute SVXY daily returns
svxy_rets = [0.0] + [
    (svxy_close[i] / svxy_close[i - 1] - 1) if svxy_close[i - 1] > 0 else 0
    for i in range(1, n)
]

# Precompute VIX3M/VIX ratio for each date
vix3m_vix_ratio: dict[int, float | None] = {}
for i in range(n):
    d = dates[i]
    vix_val = sym_data["VIX"].get(d)
    vix3m_val = sym_data["VIX3M"].get(d)
    if vix_val and vix3m_val and vix_val > 0:
        vix3m_vix_ratio[i] = vix3m_val / vix_val
    else:
        vix3m_vix_ratio[i] = None

# Report term structure stats
valid_ratios = [v for v in vix3m_vix_ratio.values() if v is not None]
if valid_ratios:
    contango_days = sum(1 for v in valid_ratios if v > 1.0)
    pct_contango = contango_days / len(valid_ratios) * 100
    print(f"VIX3M/VIX ratio: {len(valid_ratios)} valid days")
    print(
        f"  Mean={sum(valid_ratios) / len(valid_ratios):.4f}, "
        f"Min={min(valid_ratios):.4f}, Max={max(valid_ratios):.4f}"
    )
    print(
        f"  Contango days (ratio > 1.0): {contango_days}/{len(valid_ratios)} ({pct_contango:.1f}%)"
    )
    confirmed_contango = sum(1 for v in valid_ratios if v > 1.05)
    print(
        f"  Confirmed contango (ratio > 1.05): {confirmed_contango}/{len(valid_ratios)} ({confirmed_contango / len(valid_ratios) * 100:.1f}%)"
    )
else:
    print("WARNING: No valid VIX3M/VIX ratios found!")


def asset_ret(sym, i):
    """Get daily return for asset at day i."""
    d, dp = dates[i], dates[i - 1]
    data = sym_data[sym]
    if d in data and dp in data and data[dp] > 0:
        return data[d] / data[dp] - 1
    return 0.0


def _compute_metrics(daily_returns):
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


def run_single(params):
    """Run a single backtest with the given parameters.

    Signal logic (with lag):
    - Compute VIX3M/VIX ratio
    - Use the ratio from lag_days ago (causal)
    - If ratio > entry_threshold -> hold SVXY at target_weight, rest in SHY
    - If ratio < exit_threshold -> exit to 100% SHY (backwardation = danger)
    - Otherwise hold current position (hysteresis)
    """
    entry_thresh = float(params.get("entry_threshold", 1.05))
    exit_thresh = float(params.get("exit_threshold", 1.00))
    lag = int(params.get("lag_days", 1))
    tw = float(params.get("target_weight", 0.50))
    cost_per_switch = 0.0003  # 3 bps for liquid ETFs

    daily_returns = []
    in_position = False

    for i in range(WARMUP, n):
        if i < 1:
            daily_returns.append(0.0)
            continue

        # Signal date: lag_days ago
        signal_idx = i - lag
        if signal_idx < 0:
            daily_returns.append(asset_ret("SHY", i))
            continue

        # Get the VIX3M/VIX ratio from the signal date
        ratio = vix3m_vix_ratio.get(signal_idx)

        # Position logic with hysteresis
        prev_position = in_position

        if ratio is not None:
            if ratio > entry_thresh:
                in_position = True
            elif ratio < exit_thresh:
                in_position = False
            # else: hold current state (hysteresis)

        if in_position:
            day_ret = svxy_rets[i] * tw + asset_ret("SHY", i) * (1.0 - tw)
        else:
            day_ret = asset_ret("SHY", i)

        # Apply switching cost on state change
        if prev_position != in_position:
            day_ret -= cost_per_switch

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
print(f"Base Sharpe:  {base['sharpe']:.4f}")
print(f"Base MaxDD:   {base['max_dd']:.4f}")
print(f"Base Return:  {base['total_return']:.4f}")
print(f"Base CAGR:    {base['cagr']:.4f}")

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
    (
        "entry_threshold=1.03 (more permissive)",
        {**BASE_PARAMS, "entry_threshold": 1.03},
    ),
    ("entry_threshold=1.08 (stricter)", {**BASE_PARAMS, "entry_threshold": 1.08}),
    ("exit_threshold=0.98 (earlier exit)", {**BASE_PARAMS, "exit_threshold": 0.98}),
    (
        "exit_threshold=1.02 (exit on narrow contango)",
        {**BASE_PARAMS, "exit_threshold": 1.02},
    ),
    ("target_weight=0.30 (reduced position)", {**BASE_PARAMS, "target_weight": 0.30}),
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
# Use SVXY returns as the asset baseline for shuffled test
svxy_daily_rets = svxy_rets[WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(svxy_daily_rets))
aligned_strat = strat_returns[-n_min:]
aligned_svxy = svxy_daily_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  SVXY returns:     {len(aligned_svxy)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_svxy,
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
print(f"  SR = {sr:.4f}, T = {T} days ({T_years:.2f} years)")
print(f"  SE(SR) = {se_sr:.4f}")
print(f"  DSR (computed inline): {dsr_value:.4f}")

# ==============================================================================
# GATE ASSESSMENT -- Track B gates
# ==============================================================================
print(f"\n{'=' * 70}")
print("GATE ASSESSMENT (Track B -- Aggressive Alpha)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] > 1.0
gate2 = base["max_dd"] < DD_THRESHOLD  # 30% for Track B
gate3 = dsr_value >= 0.95
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 60  # 3/5 minimum for Track B
gate6 = shuffled_result.passed

gates = [
    ("Gate 1: Sharpe > 1.0", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 30%", gate2, f"{base['max_dd']:.4f}"),
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
    "strategy_type": "carry_vol_selling",
    "track": "B",
    "family": "F4",
    "mechanism": "VIX term structure (VIX3M/VIX ratio) -> SVXY carry (short vol)",
    "base_params": BASE_PARAMS,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "base_cagr": round(base["cagr"], 4),
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
        "sharpe_gt_1.0": gate1,
        "maxdd_lt_30pct": gate2,
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

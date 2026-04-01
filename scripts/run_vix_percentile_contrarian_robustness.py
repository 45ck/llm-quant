#!/usr/bin/env python3
"""Robustness analysis for vix-percentile-contrarian-v1 (F23 VIX Mean-Reversion Contrarian).

Hypothesis: VIX percentile rank relative to its own 252-day rolling history
provides an adaptive fear/complacency signal for regime-based allocation.

This is DIFFERENT from previous VIX strategies:
  - VIX term structure (contango/backwardation as equity timing -- failed)
  - VIX spike contrarian (absolute level thresholds -- too few trades)
  - VoV (volatility-of-volatility -- falsified)
This uses VIX PERCENTILE RANK -- adaptive threshold relative to own history.

Signal: VIX 252-day rolling percentile rank.
  - VIX percentile > 80 (fear elevated): 80% SPY (contrarian buy)
  - VIX percentile < 20 (complacency): 50% GLD + 20% SHY
  - 20 <= VIX percentile <= 80 (normal): 50% SPY

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_vix_percentile_contrarian_robustness.py
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

SLUG = "vix-percentile-contrarian-v1"
SYMBOLS = ["SPY", "GLD", "SHY", "VIX"]  # VIX maps to ^VIX in fetcher
DD_THRESHOLD = 0.15
LOOKBACK_DAYS = 5 * 365
WARMUP = 252  # Need 252 days for VIX percentile window

BASE_PARAMS = {
    "vix_window": 252,
    "fear_threshold": 80,  # percentile
    "complacency_threshold": 20,  # percentile
    "spy_fear_weight": 0.80,
    "spy_neutral_weight": 0.50,
    "gld_complacency_weight": 0.50,
    "shy_complacency_weight": 0.20,
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series -- use SPY as the date spine
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Build lookup dicts for GLD, SHY, VIX
sym_data: dict[str, dict] = {}
for sym in ["GLD", "SHY", "VIX"]:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# Daily returns for SPY
spy_rets = [0.0] + [
    (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0
    for i in range(1, n)
]

# Build close arrays for GLD and SHY aligned to SPY dates
gld_closes = [sym_data["GLD"].get(dates[i], 0.0) for i in range(n)]
gld_rets = [0.0] + [
    (gld_closes[i] / gld_closes[i - 1] - 1) if gld_closes[i - 1] > 0 else 0
    for i in range(1, n)
]

shy_closes = [sym_data["SHY"].get(dates[i], 0.0) for i in range(n)]
shy_rets = [0.0] + [
    (shy_closes[i] / shy_closes[i - 1] - 1) if shy_closes[i - 1] > 0 else 0
    for i in range(1, n)
]

# Build VIX close array aligned to SPY dates
vix_closes = [sym_data["VIX"].get(dates[i], 0.0) for i in range(n)]

print(f"VIX data points: {sum(1 for v in vix_closes if v > 0)} / {n}")


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
    vix_window = int(params.get("vix_window", 252))
    fear_thresh = float(params.get("fear_threshold", 80))
    complacency_thresh = float(params.get("complacency_threshold", 20))
    spy_fear_w = float(params.get("spy_fear_weight", 0.80))
    spy_neutral_w = float(params.get("spy_neutral_weight", 0.50))
    gld_comp_w = float(params.get("gld_complacency_weight", 0.50))
    shy_comp_w = float(params.get("shy_complacency_weight", 0.20))

    daily_returns = []
    cost_per_switch = 0.0003  # 3 bps spread each way

    prev_regime = None
    for i in range(WARMUP, n):
        # Need at least vix_window days of VIX history
        if i < vix_window:
            daily_returns.append(0.0)
            continue

        # Signal based on YESTERDAY's data (lag-1, no look-ahead)
        # Get current VIX value (yesterday's close = dates[i-1])
        vix_current = vix_closes[i - 1]
        if vix_current <= 0:
            daily_returns.append(0.0)
            continue

        # Get VIX values over the past vix_window days (ending at i-1)
        vix_history = []
        for j in range(i - vix_window, i):
            if j >= 0 and vix_closes[j] > 0:
                vix_history.append(vix_closes[j])

        if len(vix_history) < vix_window * 0.8:
            # Not enough VIX data for reliable percentile
            daily_returns.append(0.0)
            continue

        # Compute percentile rank of current VIX within its rolling window
        count_below = sum(1 for v in vix_history if v < vix_current)
        percentile = (count_below / len(vix_history)) * 100

        # Determine regime based on VIX percentile
        if percentile > fear_thresh:
            # Fear elevated: VIX is high relative to its own history
            # Contrarian: BUY SPY (fear is overdone, expect mean-reversion)
            regime = "fear"
            day_ret = spy_rets[i] * spy_fear_w
        elif percentile < complacency_thresh:
            # Complacency: VIX is low relative to its own history
            # Defensive: hold GLD + SHY (low-vol complacency often precedes correction)
            regime = "complacency"
            day_ret = gld_rets[i] * gld_comp_w + shy_rets[i] * shy_comp_w
        else:
            # Normal regime: moderate SPY exposure
            regime = "neutral"
            day_ret = spy_rets[i] * spy_neutral_w

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
    ("vix_window=126", {**BASE_PARAMS, "vix_window": 126}),
    ("vix_window=504", {**BASE_PARAMS, "vix_window": 504}),
    ("fear_threshold=75", {**BASE_PARAMS, "fear_threshold": 75}),
    ("fear_threshold=85", {**BASE_PARAMS, "fear_threshold": 85}),
    ("gld_complacency_weight=0.40", {**BASE_PARAMS, "gld_complacency_weight": 0.40}),
    ("gld_complacency_weight=0.60", {**BASE_PARAMS, "gld_complacency_weight": 0.60}),
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
spy_daily_rets = spy_rets[WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(spy_daily_rets))
aligned_strat = strat_returns[-n_min:]
aligned_spy = spy_daily_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  SPY returns: {len(aligned_spy)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_spy,
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
    "strategy_type": "vix_percentile_contrarian",
    "mechanism": "VIX 252-day rolling percentile rank -> adaptive fear/complacency threshold",
    "hypothesis": (
        "When VIX is in top 20th percentile (elevated fear), buy SPY (contrarian). "
        "When VIX is in bottom 20th percentile (complacency), hold GLD+SHY (defensive)."
    ),
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

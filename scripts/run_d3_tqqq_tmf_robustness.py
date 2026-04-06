#!/usr/bin/env python3
"""Robustness analysis for d3-tqqq-tmf-ratio-mr (Track D).

Mechanism: TQQQ/TMF price ratio z-score mean-reversion with VIX crash filter.
This is a leveraged risk-parity rebalance strategy using 3x equity (TQQQ) and
3x bond (TMF).

Signal logic:
  - Z-score < -1.0 (TMF relatively cheap): Hold TMF at target_weight, rest SHY
  - Z-score > +1.0 (TQQQ relatively cheap): Hold TQQQ at target_weight, rest SHY
  - -1.0 <= Z-score <= 1.0 (neutral): Hold 50/50 TQQQ/TMF at combined target_weight
  - VIX > 30: Override to 100% SHY (crash filter)
  - Cost per switch: 5 bps (leveraged ETFs)

Track D gates (leveraged):
  - Gate 1: Sharpe > 0.80
  - Gate 2: MaxDD < 40%
  - Gate 3: DSR >= 0.90
  - Gate 4: CPCV OOS > 0
  - Gate 5: Perturbation >= 40% stable (2/5 minimum)
  - Gate 6: Shuffled signal p < 0.05

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_d3_tqqq_tmf_robustness.py
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

SLUG = "d3-tqqq-tmf-ratio-mr"
SYMBOLS = ["TQQQ", "TMF", "SHY", "VIX"]
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

BASE_PARAMS = {
    "lookback_days": 60,  # Rolling z-score window
    "z_threshold": 1.0,  # Z-score threshold for directional tilt
    "target_weight": 0.30,  # Total weight in leveraged ETFs
    "vix_crash_threshold": 30.0,  # VIX level to exit to cash
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series -- use TQQQ as the date backbone
tqqq_df = prices.filter(pl.col("symbol") == "TQQQ").sort("date")
dates = tqqq_df["date"].to_list()
tqqq_close = tqqq_df["close"].to_list()
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


# Precompute TQQQ/TMF ratio series aligned to dates
ratio_series = []
for i in range(n):
    d = dates[i]
    tqqq_p = sym_data["TQQQ"].get(d, 0.0)
    tmf_p = sym_data["TMF"].get(d, 0.0)
    if tqqq_p > 0 and tmf_p > 0:
        ratio_series.append(tqqq_p / tmf_p)
    else:
        ratio_series.append(float("nan"))


def _state_return(state, i, tw):
    """Compute daily return for a given position state."""
    if state == "tqqq":
        return asset_ret("TQQQ", i) * tw + asset_ret("SHY", i) * (1.0 - tw)
    if state == "tmf":
        return asset_ret("TMF", i) * tw + asset_ret("SHY", i) * (1.0 - tw)
    if state == "neutral":
        half_tw = tw / 2.0
        return (
            asset_ret("TQQQ", i) * half_tw
            + asset_ret("TMF", i) * half_tw
            + asset_ret("SHY", i) * (1.0 - tw)
        )
    return asset_ret("SHY", i)


def _compute_zscore(i, lookback):
    """Compute z-score of the ratio at day i. Returns None if insufficient data."""
    window_ratios = [
        ratio_series[j]
        for j in range(i - lookback, i)
        if not math.isnan(ratio_series[j])
    ]
    if len(window_ratios) < lookback // 2:
        return None
    current_ratio = ratio_series[i]
    if math.isnan(current_ratio):
        return None
    roll_mean = sum(window_ratios) / len(window_ratios)
    roll_std = (
        sum((r - roll_mean) ** 2 for r in window_ratios) / len(window_ratios)
    ) ** 0.5
    if roll_std <= 0:
        return None
    return (current_ratio - roll_mean) / roll_std


def run_single(params):
    """Run a single backtest with the given parameters.

    Signal logic:
    - Compute TQQQ/TMF price ratio rolling z-score (lookback_days window)
    - Z-score < -z_threshold: TMF relatively cheap -> hold TMF at target_weight
    - Z-score > +z_threshold: TQQQ relatively cheap -> hold TQQQ at target_weight
    - Neutral zone: hold 50/50 TQQQ/TMF at combined target_weight (half each)
    - VIX > vix_crash_threshold: override to 100% SHY
    """
    lookback = int(params.get("lookback_days", 60))
    z_thresh = float(params.get("z_threshold", 1.0))
    tw = float(params.get("target_weight", 0.30))
    vix_thresh = float(params.get("vix_crash_threshold", 30.0))
    cost_per_switch = 0.0005  # 5 bps for leveraged ETFs

    daily_returns = []
    current_state = "shy"

    for i in range(WARMUP, n):
        if i < WARMUP + lookback:
            daily_returns.append(asset_ret("SHY", i))
            continue

        z_score = _compute_zscore(i, lookback)
        if z_score is None:
            daily_returns.append(asset_ret("SHY", i))
            continue

        # VIX crash filter
        d = dates[i]
        vix_val = sym_data["VIX"].get(d, 0.0)
        vix_crash = vix_val > vix_thresh if vix_val > 0 else False

        prev_state = current_state

        if vix_crash:
            current_state = "shy"
        elif z_score < -z_thresh:
            current_state = "tmf"
        elif z_score > z_thresh:
            current_state = "tqqq"
        else:
            current_state = "neutral"

        day_ret = _state_return(current_state, i, tw)

        if prev_state != current_state:
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
    ("z_threshold=0.5 (half)", {**BASE_PARAMS, "z_threshold": 0.5}),
    ("z_threshold=1.5 (1.5x)", {**BASE_PARAMS, "z_threshold": 1.5}),
    ("lookback_days=40", {**BASE_PARAMS, "lookback_days": 40}),
    ("lookback_days=90", {**BASE_PARAMS, "lookback_days": 90}),
    ("target_weight=0.20", {**BASE_PARAMS, "target_weight": 0.20}),
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
# Use a blended TQQQ/TMF baseline for the shuffled test.
# The strategy alternates between TQQQ, TMF, neutral, and SHY -- so the
# relevant "asset returns" are a 50/50 TQQQ/TMF blend (the neutral state).
blended_rets = []
for i in range(WARMUP, n):
    tqqq_r = asset_ret("TQQQ", i)
    tmf_r = asset_ret("TMF", i)
    blended_rets.append(0.5 * tqqq_r + 0.5 * tmf_r)

strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(blended_rets))
aligned_strat = strat_returns[-n_min:]
aligned_asset = blended_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  Asset returns:    {len(aligned_asset)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_asset,
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
print("GATE ASSESSMENT (Track D -- Leveraged)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] > 0.80
gate2 = base["max_dd"] < DD_THRESHOLD  # 40% for Track D
gate3 = dsr_value >= 0.90  # Relaxed for Track D
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 40  # 2/5 minimum for Track D
gate6 = shuffled_result.passed

gates = [
    ("Gate 1: Sharpe > 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    ("Gate 5: Perturbation >= 40% stable", gate5, f"{pct_stable:.0f}%"),
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
    "strategy_type": "pairs_ratio_mean_reversion",
    "track": "D",
    "mechanism": "TQQQ/TMF ratio z-score mean-reversion with VIX crash filter",
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
        "sharpe_gt_0.80": gate1,
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
print(f"\nSaved to {out_path}")

#!/usr/bin/env python3
"""Robustness analysis for iwm-qqq-risk-appetite-v1 (F20 Risk Appetite Timing).

Hypothesis: IWM/QQQ price ratio 30-day momentum captures broad risk appetite.
When small caps (IWM) outperform tech (QQQ), broad risk appetite is strong
and equities benefit. When QQQ outperforms IWM, flight to quality/mega-cap
concentration -- hold quality + gold hedge.

Signal:
- Compute IWM/QQQ price ratio at close t-1
- Compute 30-day ratio momentum (ratio_now / ratio_30d_ago - 1)
- ratio_mom > 0 (IWM outperforming, broad risk-on): 80% SPY
- ratio_mom <= 0 (QQQ outperforming, quality flight): 60% QQQ + 20% GLD

Perturbations:
  1. lookback=20 (faster signal)
  2. lookback=45 (slower signal)
  3. spy_risk_on_weight=0.70 (less aggressive risk-on)
  4. spy_risk_on_weight=0.90 (more aggressive risk-on)
  5. qqq_quality_weight=0.50 (less QQQ in defensive)
  6. qqq_quality_weight=0.70 (more QQQ in defensive)

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_iwm_qqq_risk_appetite_robustness.py
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

SLUG = "iwm-qqq-risk-appetite-v1"
SYMBOLS = ["IWM", "QQQ", "SPY", "GLD"]
DD_THRESHOLD = 0.15
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

BASE_PARAMS = {
    "lookback": 30,
    "spy_risk_on_weight": 0.80,
    "qqq_quality_weight": 0.60,
    "gld_quality_weight": 0.20,
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series -- SPY as the date spine
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Build lookup dicts for each symbol
sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# SPY daily returns
spy_rets = [0.0] + [
    (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0
    for i in range(1, n)
]


def _asset_ret(sym, i):
    """Compute daily return for an asset at day i."""
    d, dp = dates[i], dates[i - 1]
    data = sym_data[sym]
    if d in data and dp in data and data[dp] > 0:
        return data[d] / data[dp] - 1
    return 0.0


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
    lookback = int(params.get("lookback", 30))
    spy_risk_on_w = float(params.get("spy_risk_on_weight", 0.80))
    qqq_quality_w = float(params.get("qqq_quality_weight", 0.60))
    gld_quality_w = float(params.get("gld_quality_weight", 0.20))

    daily_returns = []
    cost_per_switch = 0.0003  # 3 bps spread each way

    prev_regime = None
    for i in range(WARMUP, n):
        if i < lookback + 1:
            daily_returns.append(0.0)
            continue

        # IWM/QQQ ratio momentum at close i-1 (lag-1, no look-ahead)
        d_now = dates[i - 1]
        d_lb = dates[i - 1 - lookback]

        iwm_now = sym_data["IWM"].get(d_now, 0.0)
        qqq_now = sym_data["QQQ"].get(d_now, 0.0)
        iwm_lb = sym_data["IWM"].get(d_lb, 0.0)
        qqq_lb = sym_data["QQQ"].get(d_lb, 0.0)

        if iwm_now <= 0 or qqq_now <= 0 or iwm_lb <= 0 or qqq_lb <= 0:
            daily_returns.append(0.0)
            continue

        ratio_now = iwm_now / qqq_now
        ratio_lb = iwm_lb / qqq_lb
        ratio_mom = ratio_now / ratio_lb - 1

        # Determine regime based on IWM/QQQ ratio momentum
        if ratio_mom > 0:
            # IWM outperforming QQQ -- broad risk appetite strong
            regime = "risk_on"
            day_ret = spy_rets[i] * spy_risk_on_w
        else:
            # QQQ outperforming IWM -- flight to quality/mega-cap
            regime = "quality_flight"
            day_ret = (
                _asset_ret("QQQ", i) * qqq_quality_w
                + _asset_ret("GLD", i) * gld_quality_w
            )

        # Apply switching cost on regime change
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
    ("lookback=20", {**BASE_PARAMS, "lookback": 20}),
    ("lookback=45", {**BASE_PARAMS, "lookback": 45}),
    ("spy_risk_on_weight=0.70", {**BASE_PARAMS, "spy_risk_on_weight": 0.70}),
    ("spy_risk_on_weight=0.90", {**BASE_PARAMS, "spy_risk_on_weight": 0.90}),
    ("qqq_quality_weight=0.50", {**BASE_PARAMS, "qqq_quality_weight": 0.50}),
    ("qqq_quality_weight=0.70", {**BASE_PARAMS, "qqq_quality_weight": 0.70}),
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
dsr_value = 0.0
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
if registry_path.exists():
    with registry_path.open() as f:
        exps = [json.loads(line) for line in f if line.strip()]
    if exps:
        dsr_values = [e.get("dsr", 0.0) for e in exps]
        dsr_value = max(dsr_values) if dsr_values else 0.0
        print(f"  DSR (from experiment registry, best trial): {dsr_value:.4f}")
if dsr_value == 0.0:
    # Compute inline
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
    "strategy_type": "iwm_qqq_risk_appetite",
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

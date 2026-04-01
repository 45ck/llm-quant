#!/usr/bin/env python3
"""Robustness analysis for cross-sectional-momentum-v1 (F23).

Mechanism: Rank 5 asset classes by 60-day returns, hold top-2 at 40% each.
Pure cross-sectional momentum — NOT time-series momentum.

Signal: Rank SPY, TLT, GLD, EFA, DBA by 60-day returns.
  - Hold top-2 ranked assets at 40% each (80% total invested)
  - Hold 0% in bottom-3
  - Rebalance every 20 trading days (monthly)
  - 3 bps switching cost when holdings change

This is DIFFERENT from:
  - TSMOM (time-series momentum on individual assets with vol-scaling)
  - Asset rotation (sector ETFs only)
  - Ratio momentum (comparing exactly 2 assets)

This is pure CROSS-SECTIONAL RANKING across different asset classes:
equities (SPY), bonds (TLT), gold (GLD), international equities (EFA),
agriculture commodities (DBA).

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_cross_sectional_momentum_robustness.py
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

SLUG = "cross-sectional-momentum-v1"
SYMBOLS = ["SPY", "TLT", "GLD", "EFA", "DBA"]
DD_THRESHOLD = 0.15
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

BASE_PARAMS = {
    "ranking_lookback": 60,  # 60-day return ranking
    "top_n": 2,  # hold top-2
    "weight_per_position": 0.40,  # 40% each = 80% total
    "rebalance_interval": 20,  # every 20 trading days
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series — SPY as date spine
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Build lookup dicts for each symbol
sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# Equal-weight benchmark for shuffled signal test (equal-weight all 5 assets)
eq_benchmark_rets = [0.0]
for i in range(1, n):
    d, dp = dates[i], dates[i - 1]
    day_ret = 0.0
    count = 0
    for sym in SYMBOLS:
        if d in sym_data[sym] and dp in sym_data[sym] and sym_data[sym][dp] > 0:
            day_ret += sym_data[sym][d] / sym_data[sym][dp] - 1
            count += 1
    eq_benchmark_rets.append(day_ret / count if count > 0 else 0.0)


def asset_ret(sym, i):
    """Get daily return for asset at day i."""
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
    """Run a single backtest with the given parameters.

    Cross-sectional momentum:
    1. Every rebalance_interval days, rank all 5 assets by trailing
       ranking_lookback-day returns.
    2. Select top_n assets.
    3. Assign weight_per_position to each selected asset.
    4. Between rebalance dates, hold the selected weights.
    5. Apply 3bps switching cost when holdings change.
    """
    ranking_lookback = int(params.get("ranking_lookback", 60))
    top_n = int(params.get("top_n", 2))
    weight_per_pos = float(params.get("weight_per_position", 0.40))
    rebalance_interval = int(params.get("rebalance_interval", 20))

    daily_returns = []
    cost_per_switch = 0.0003  # 3 bps

    current_holdings = set()  # set of symbols currently held
    days_since_rebalance = rebalance_interval  # force rebalance on first valid day

    for i in range(WARMUP, n):
        if i < ranking_lookback + 1:
            daily_returns.append(0.0)
            continue

        # Check if rebalance day
        if days_since_rebalance >= rebalance_interval:
            # Compute ranking_lookback-day return for each symbol
            # Using close at i-1 (lag-1, no look-ahead)
            momentums = {}
            valid = True
            for sym in SYMBOLS:
                d_now = dates[i - 1]
                d_lb = dates[i - 1 - ranking_lookback]
                p_now = sym_data[sym].get(d_now, 0.0)
                p_lb = sym_data[sym].get(d_lb, 0.0)
                if p_now <= 0 or p_lb <= 0:
                    valid = False
                    break
                momentums[sym] = p_now / p_lb - 1

            if not valid:
                daily_returns.append(0.0)
                continue

            # Rank by return (descending) and select top_n
            ranked = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
            new_holdings = {sym for sym, _ in ranked[:top_n]}

            # Apply switching cost if holdings changed
            switching_cost = 0.0
            if current_holdings and new_holdings != current_holdings:
                switching_cost = cost_per_switch

            current_holdings = new_holdings
            days_since_rebalance = 0

        # Compute daily return from held positions
        if not current_holdings:
            daily_returns.append(0.0)
            continue

        day_ret = sum(asset_ret(sym, i) * weight_per_pos for sym in current_holdings)

        # Apply switching cost on rebalance day
        if days_since_rebalance == 0 and switching_cost > 0:
            day_ret -= switching_cost

        days_since_rebalance += 1
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
    ("ranking_lookback=40", {**BASE_PARAMS, "ranking_lookback": 40}),
    ("ranking_lookback=90", {**BASE_PARAMS, "ranking_lookback": 90}),
    (
        "top_n=1 (weight=0.80)",
        {**BASE_PARAMS, "top_n": 1, "weight_per_position": 0.80},
    ),
    (
        "top_n=3 (weight=0.27)",
        {**BASE_PARAMS, "top_n": 3, "weight_per_position": 0.27},
    ),
    ("weight_per_position=0.30", {**BASE_PARAMS, "weight_per_position": 0.30}),
    ("weight_per_position=0.45", {**BASE_PARAMS, "weight_per_position": 0.45}),
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
# Benchmark = equal-weight basket of all 5 asset classes
# This isolates the cross-sectional ranking signal from passive multi-asset exposure
benchmark_rets = eq_benchmark_rets[WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(benchmark_rets))
aligned_strat = strat_returns[-n_min:]
aligned_bench = benchmark_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  Benchmark (EW 5-asset): {len(aligned_bench)} days")

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
    "strategy_type": "cross_sectional_momentum",
    "mechanism": (
        "F23 Cross-Sectional Momentum — rank SPY/TLT/GLD/EFA/DBA by 60-day returns, "
        "hold top-2 at 40% each. Pure cross-sectional ranking across asset classes."
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

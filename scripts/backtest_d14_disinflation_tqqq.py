#!/usr/bin/env python3
"""Backtest + robustness for D14: TLT/GLD Disinflation -> TQQQ (Track D — Sprint Alpha).

Mechanism: TLT/GLD 30-day momentum as a disinflation/inflation regime signal.
When TLT outperforms GLD (disinflation regime, ratio_mom > 0), allocate to TQQQ.
When GLD outperforms TLT (inflation regime, ratio_mom < 0), rotate to defensive
(GLD + DBA + SHY). VIX crash filter overrides to 100% SHY.

This re-expresses the proven F19 signal (Track A Sharpe=1.313) through 3x leverage.

Signal logic:
  - Compute TLT/GLD price ratio daily
  - ratio_mom = ratio_today / ratio_{lookback}d_ago - 1
  - Disinflation (ratio_mom > 0): target_weight% TQQQ + rest SHY
  - Inflation (ratio_mom < 0): 40% GLD + 10% DBA + 50% SHY
  - VIX > crash_threshold: Override to 100% SHY
  - Rebalance every rebalance_days trading days
  - Cost: 20bps per trade

Track D gates (leveraged):
  - Gate 1: Sharpe >= 0.80
  - Gate 2: MaxDD < 40%
  - Gate 3: DSR >= 0.90
  - Gate 4: CPCV OOS > 0
  - Gate 5: Perturbation >= 40% stable
  - Gate 6: Shuffled signal p < 0.05

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_d14_disinflation_tqqq.py
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

SLUG = "d14-disinflation-tqqq"
SYMBOLS = ["TLT", "GLD", "TQQQ", "SHY", "VIX", "DBA"]
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

BASE_PARAMS = {
    "lookback": 30,  # 30-day momentum of TLT/GLD ratio
    "target_weight": 0.50,  # TQQQ allocation in disinflation regime
    "vix_crash_threshold": 30,  # VIX override to 100% SHY
    "gld_inflation_weight": 0.40,  # GLD weight in inflation regime
    "dba_inflation_weight": 0.10,  # DBA weight in inflation regime
    "rebalance_days": 5,  # Rebalance every 5 trading days (weekly)
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

# Precompute TQQQ daily returns
tqqq_rets = [0.0] + [
    (tqqq_close[i] / tqqq_close[i - 1] - 1) if tqqq_close[i - 1] > 0 else 0
    for i in range(1, n)
]


def asset_ret(sym: str, i: int) -> float:
    """Get daily return for asset at day i."""
    d, dp = dates[i], dates[i - 1]
    data = sym_data[sym]
    if d in data and dp in data and data[dp] > 0:
        return data[d] / data[dp] - 1
    return 0.0


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

    Signal logic (with periodic rebalancing):
    - Compute TLT/GLD price ratio daily
    - ratio_mom = ratio[today] / ratio[lookback days ago] - 1
    - Every rebalance_days trading days, evaluate regime:
      * VIX > crash_threshold -> 100% SHY (override)
      * ratio_mom > 0 (disinflation) -> target_weight% TQQQ + rest SHY
      * ratio_mom <= 0 (inflation) -> gld_weight% GLD + dba_weight% DBA + rest SHY
    - Between rebalance days, hold current allocation
    """
    lookback = int(params.get("lookback", 30))
    tw = float(params.get("target_weight", 0.50))
    vix_thresh = float(params.get("vix_crash_threshold", 30))
    gld_w = float(params.get("gld_inflation_weight", 0.40))
    dba_w = float(params.get("dba_inflation_weight", 0.10))
    rebal_days = int(params.get("rebalance_days", 5))
    cost_per_switch = 0.0020  # 20 bps round-trip

    daily_returns = []
    # Regime: "cash", "disinflation", "inflation"
    current_regime = "cash"
    n_trades = 0
    min_lookback = WARMUP + lookback
    days_since_rebal = 0

    for i in range(WARMUP, n):
        if i < min_lookback:
            daily_returns.append(asset_ret("SHY", i))
            continue

        days_since_rebal += 1
        prev_regime = current_regime
        is_rebal_day = days_since_rebal >= rebal_days

        if is_rebal_day:
            days_since_rebal = 0

            # Get VIX level
            d_today = dates[i]
            vix_level = sym_data["VIX"].get(d_today, 0.0)

            # Compute TLT/GLD ratio momentum
            d_lb = dates[i - lookback] if (i - lookback) >= 0 else dates[0]
            tlt_now = sym_data["TLT"].get(d_today, 0.0)
            gld_now = sym_data["GLD"].get(d_today, 0.0)
            tlt_lb = sym_data["TLT"].get(d_lb, 0.0)
            gld_lb = sym_data["GLD"].get(d_lb, 0.0)

            if tlt_now <= 0 or gld_now <= 0 or tlt_lb <= 0 or gld_lb <= 0:
                daily_returns.append(asset_ret("SHY", i))
                continue

            ratio_now = tlt_now / gld_now
            ratio_lb = tlt_lb / gld_lb
            ratio_mom = ratio_now / ratio_lb - 1.0

            # Regime classification
            if vix_level > vix_thresh:
                current_regime = "cash"
            elif ratio_mom > 0:
                current_regime = "disinflation"
            else:
                current_regime = "inflation"

        # Compute daily return based on current regime
        if current_regime == "disinflation":
            day_ret = tqqq_rets[i] * tw + asset_ret("SHY", i) * (1.0 - tw)
        elif current_regime == "inflation":
            shy_w = 1.0 - gld_w - dba_w
            day_ret = (
                asset_ret("GLD", i) * gld_w
                + asset_ret("DBA", i) * dba_w
                + asset_ret("SHY", i) * shy_w
            )
        else:  # cash
            day_ret = asset_ret("SHY", i)

        # Apply switching cost on regime change
        if prev_regime != current_regime:
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

    Uses 15 groups, 3 test groups, 5-day purge.
    """
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
print(f"Trades:       {base['n_trades']}")

# Early gate check -- only run full robustness if base Sharpe >= 0.80
if base["sharpe"] < 0.80:
    print(f"\n*** BASE SHARPE {base['sharpe']:.4f} < 0.80 -- FAILING GATE 1 ***")
    print("Running remaining gates for completeness but strategy will FAIL.\n")

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
# PERTURBATION TESTS
# ==============================================================================
perturbations = [
    # lookback perturbations
    ("lookback=20", {**BASE_PARAMS, "lookback": 20}),
    ("lookback=25", {**BASE_PARAMS, "lookback": 25}),
    ("lookback=40", {**BASE_PARAMS, "lookback": 40}),
    ("lookback=45", {**BASE_PARAMS, "lookback": 45}),
    # target_weight perturbations
    ("tw=0.30", {**BASE_PARAMS, "target_weight": 0.30}),
    ("tw=0.40", {**BASE_PARAMS, "target_weight": 0.40}),
    ("tw=0.60", {**BASE_PARAMS, "target_weight": 0.60}),
    ("tw=0.70", {**BASE_PARAMS, "target_weight": 0.70}),
    # VIX crash threshold perturbations
    ("vix=25", {**BASE_PARAMS, "vix_crash_threshold": 25}),
    ("vix=35", {**BASE_PARAMS, "vix_crash_threshold": 35}),
    # GLD inflation weight perturbations
    ("gld_w=0.30", {**BASE_PARAMS, "gld_inflation_weight": 0.30}),
    ("gld_w=0.50", {**BASE_PARAMS, "gld_inflation_weight": 0.50}),
    # Rebalance frequency perturbations
    ("rebal=3", {**BASE_PARAMS, "rebalance_days": 3}),
    ("rebal=7", {**BASE_PARAMS, "rebalance_days": 7}),
    ("rebal=10", {**BASE_PARAMS, "rebalance_days": 10}),
]

print("\n--- PERTURBATION RESULTS ---")
print(
    f"  {'Variant':<25} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'Change%':>9} {'Status':>10}"
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
        f"  {name:<25} {r['sharpe']:>8.4f} {r['max_dd']:>8.4f} {r['cagr']:>8.4f} {pct:>+8.1f}% {stable:>10}"
    )

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# ==============================================================================
# SHUFFLED SIGNAL TEST (1000 shuffles)
# ==============================================================================
print("\n--- SHUFFLED SIGNAL TEST (1000 shuffles) ---")
# Use TQQQ returns as the asset baseline for shuffled test
tqqq_daily_rets = tqqq_rets[WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(tqqq_daily_rets))
aligned_strat = strat_returns[-n_min:]
aligned_tqqq = tqqq_daily_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  TQQQ returns:     {len(aligned_tqqq)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_tqqq,
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
# BENCHMARK COMPARISON
# ==============================================================================
print("\n--- BENCHMARK: Buy-and-Hold TQQQ ---")
bh_returns = tqqq_rets[WARMUP:]
bh = _compute_metrics(bh_returns)
print(f"  TQQQ B&H Sharpe:  {bh['sharpe']:.4f}")
print(f"  TQQQ B&H MaxDD:   {bh['max_dd']:.4f}")
print(f"  TQQQ B&H CAGR:    {bh['cagr']:.4f}")
print(f"  Strategy vs B&H:  Sharpe {base['sharpe']:.4f} vs {bh['sharpe']:.4f}")

# ==============================================================================
# GATE ASSESSMENT -- Track D gates
# ==============================================================================
print(f"\n{'=' * 70}")
print("GATE ASSESSMENT (Track D -- Leveraged)")
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
    "strategy_type": "disinflation_regime_leveraged",
    "track": "D",
    "mechanism": "TLT/GLD 30-day momentum as disinflation/inflation regime signal -> TQQQ (3x leveraged QQQ)",
    "parent_signal": "F19 TLT/GLD disinflation (Track A, Sharpe=1.313)",
    "key_differentiator": "Re-expresses proven F19 disinflation signal through 3x leverage; "
    "defensive rotation to GLD+DBA in inflation avoids equity drawdowns",
    "base_params": BASE_PARAMS,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "base_cagr": round(base["cagr"], 4),
    "n_trades": base["n_trades"],
    "n_days": T,
    "benchmark_tqqq_bh": {
        "sharpe": round(bh["sharpe"], 4),
        "max_dd": round(bh["max_dd"], 4),
        "cagr": round(bh["cagr"], 4),
    },
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

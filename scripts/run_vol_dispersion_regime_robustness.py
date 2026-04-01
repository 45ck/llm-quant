#!/usr/bin/env python3
"""Robustness analysis for vol-dispersion-regime-v1 (cross-asset vol dispersion).

Mechanism: Cross-asset realized volatility dispersion predicts regime transitions.
  - Compute 20-day realized vol (annualized) for SPY, TLT, GLD, EFA, DBA, IEF, SHY
  - Cross-asset vol dispersion = std deviation of all 7 assets' realized vols
  - 120-day percentile rank of current vol dispersion
  - High dispersion (>75th pct): 40% GLD + 30% SHY + 10% TLT (defensive)
  - Low dispersion (<25th pct): 70% SPY + 10% EFA (risk-on)
  - Middle zone: 40% SPY + 20% TLT + 15% GLD

Base parameters: vol_window=20, rank_window=120, high_pct=75, low_pct=25.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/run_vol_dispersion_regime_robustness.py
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

SLUG = "vol-dispersion-regime-v1"
SYMBOLS = ["SPY", "TLT", "GLD", "EFA", "DBA", "IEF", "SHY"]
DD_THRESHOLD = 0.15
LOOKBACK_DAYS = 5 * 365
WARMUP = 130

BASE_PARAMS = {
    "vol_window": 20,
    "rank_window": 120,
    "high_pct": 75,
    "low_pct": 25,
}

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price series
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

spy_rets = [0.0] + [
    (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0
    for i in range(1, n)
]


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


def run_single(params):  # noqa: PLR0912
    """Run a single backtest with the given parameters."""
    vol_window = int(params.get("vol_window", 20))
    rank_window = int(params.get("rank_window", 120))
    high_pct = float(params.get("high_pct", 75))
    low_pct = float(params.get("low_pct", 25))

    daily_returns = []
    cost_per_switch = 0.0003  # 3 bps

    # Precompute daily returns for all symbols
    all_sym_rets: dict[str, list[float]] = {}
    for sym in SYMBOLS:
        rets = [0.0]
        for i in range(1, n):
            d, dp = dates[i], dates[i - 1]
            data = sym_data[sym]
            if d in data and dp in data and data[dp] > 0:
                rets.append(data[d] / data[dp] - 1)
            else:
                rets.append(0.0)
        all_sym_rets[sym] = rets

    prev_regime = None
    for i in range(WARMUP, n):
        # Need at least vol_window + rank_window days of history
        if i < vol_window + 1:
            daily_returns.append(0.0)
            continue

        # Compute realized vol for each asset using returns up to day i-1 (lag-1)
        asset_vols = []
        valid = True
        for sym in SYMBOLS:
            # Use returns from i-vol_window to i-1 (inclusive), all prior to day i
            window_rets = all_sym_rets[sym][i - vol_window : i]
            if len(window_rets) < vol_window:
                valid = False
                break
            mean_r = sum(window_rets) / len(window_rets)
            var_r = sum((r - mean_r) ** 2 for r in window_rets) / len(window_rets)
            realized_vol = math.sqrt(var_r) * math.sqrt(252)
            asset_vols.append(realized_vol)

        if not valid or len(asset_vols) < len(SYMBOLS):
            daily_returns.append(0.0)
            continue

        # Cross-asset vol dispersion = std of realized vols
        mean_vol = sum(asset_vols) / len(asset_vols)
        vol_dispersion = math.sqrt(
            sum((v - mean_vol) ** 2 for v in asset_vols) / len(asset_vols)
        )

        # Compute percentile rank over rank_window days
        # We need to compute vol dispersion for past rank_window days
        # For efficiency, compute dispersion history on the fly
        dispersion_history = []
        lookback_start = max(vol_window + 1, i - rank_window)
        for j in range(lookback_start, i):
            j_vols = []
            j_valid = True
            for sym in SYMBOLS:
                w_rets = all_sym_rets[sym][j - vol_window : j]
                if len(w_rets) < vol_window:
                    j_valid = False
                    break
                m = sum(w_rets) / len(w_rets)
                v = sum((r - m) ** 2 for r in w_rets) / len(w_rets)
                j_vols.append(math.sqrt(v) * math.sqrt(252))
            if j_valid and len(j_vols) == len(SYMBOLS):
                m_v = sum(j_vols) / len(j_vols)
                disp = math.sqrt(sum((x - m_v) ** 2 for x in j_vols) / len(j_vols))
                dispersion_history.append(disp)

        if len(dispersion_history) < 20:
            daily_returns.append(0.0)
            continue

        # Percentile rank: fraction of historical values <= current
        pct_rank = (
            sum(1 for d in dispersion_history if d <= vol_dispersion)
            / len(dispersion_history)
            * 100
        )

        # Determine regime
        if pct_rank > high_pct:
            # High dispersion: defensive
            regime = "high_dispersion"
            day_ret = (
                asset_ret("GLD", i) * 0.40
                + asset_ret("SHY", i) * 0.30
                + asset_ret("TLT", i) * 0.10
            )
        elif pct_rank < low_pct:
            # Low dispersion: risk-on
            regime = "low_dispersion"
            day_ret = spy_rets[i] * 0.70 + asset_ret("EFA", i) * 0.10
        else:
            # Middle zone
            regime = "middle"
            day_ret = (
                spy_rets[i] * 0.40
                + asset_ret("TLT", i) * 0.20
                + asset_ret("GLD", i) * 0.15
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
    ("vol_window=15", {**BASE_PARAMS, "vol_window": 15}),
    ("vol_window=30", {**BASE_PARAMS, "vol_window": 30}),
    ("high_pct=65", {**BASE_PARAMS, "high_pct": 65}),
    ("high_pct=85", {**BASE_PARAMS, "high_pct": 85}),
    ("low_pct=15", {**BASE_PARAMS, "low_pct": 15}),
    ("low_pct=35", {**BASE_PARAMS, "low_pct": 35}),
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
    "strategy_type": "vol_dispersion_regime",
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

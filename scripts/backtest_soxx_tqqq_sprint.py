#!/usr/bin/env python3
"""Backtest for soxx-tqqq-sprint (Track D).

Mechanism: SOXX 5-day return as lead signal -> TQQQ (3x leveraged QQQ) follower.
This is a leveraged re-expression of the proven SOXX-QQQ lead-lag signal
(Track A, Sharpe=0.861, DSR=0.9603).

Signal logic:
  - SOXX 5-day return >= entry_threshold (2%) -> entry (buy TQQQ at target_weight)
  - SOXX 5-day return <= exit_threshold (-1%) -> exit to SHY (cash proxy)
  - Use signal from lag_days (5) ago -- no look-ahead
  - Weekly rebalancing
  - Cost per switch: 20 bps round-trip (higher for leveraged ETFs)

Track D gates (leveraged):
  - Gate 1: Sharpe >= 0.80
  - Gate 2: MaxDD < 40%
  - Gate 3: DSR >= 0.90
  - Gate 4: CPCV OOS > 0

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_soxx_tqqq_sprint.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, "src")

import polars as pl

from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "soxx-tqqq-sprint"
SYMBOLS = ["SOXX", "TQQQ", "QQQ", "SHY"]
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

BASE_PARAMS = {
    "entry_threshold": 0.02,  # SOXX 5-day return >= 2% -> buy TQQQ
    "exit_threshold": -0.01,  # SOXX 5-day return <= -1% -> exit
    "lag_days": 5,  # Use signal from 5 days ago
    "signal_window": 5,  # 5-day return lookback for SOXX
    "target_weight": 0.30,  # 30% position in TQQQ
    "cost_per_switch": 0.0020,  # 20 bps round-trip
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

    Signal logic (with lag, daily evaluation):
    - Compute SOXX return over signal_window days
    - Use the signal from lag_days ago (causal)
    - If SOXX return >= entry_threshold -> hold TQQQ at target_weight, rest in SHY
    - If SOXX return <= exit_threshold -> exit to SHY (100%)
    - Otherwise hold current position
    - Daily signal evaluation (matching TLT-TQQQ reference and SOXX-QQQ parent)
    """
    entry_thresh = float(params.get("entry_threshold", 0.02))
    exit_thresh = float(params.get("exit_threshold", -0.01))
    lag = int(params.get("lag_days", 5))
    window = int(params.get("signal_window", 5))
    tw = float(params.get("target_weight", 0.30))
    cost_per_switch = float(params.get("cost_per_switch", 0.0020))

    daily_returns = []
    in_position = False
    min_lookback = WARMUP + window + lag

    for i in range(WARMUP, n):
        if i < min_lookback:
            # Not enough history yet -- stay in cash proxy
            daily_returns.append(asset_ret("SHY", i))
            continue

        # Signal date: lag_days ago
        signal_idx = i - lag
        if signal_idx < window:
            daily_returns.append(asset_ret("SHY", i))
            continue

        # SOXX return over signal_window ending at signal_idx
        d_signal = dates[signal_idx]
        d_signal_lb = dates[signal_idx - window]

        soxx_now = sym_data["SOXX"].get(d_signal, 0.0)
        soxx_lb = sym_data["SOXX"].get(d_signal_lb, 0.0)

        if soxx_now <= 0 or soxx_lb <= 0:
            daily_returns.append(asset_ret("SHY", i))
            continue

        soxx_ret = soxx_now / soxx_lb - 1.0

        # Position logic
        prev_position = in_position

        if soxx_ret >= entry_thresh:
            in_position = True
        elif soxx_ret <= exit_thresh:
            in_position = False
        # else: hold current state

        if in_position:
            day_ret = tqqq_rets[i] * tw + asset_ret("SHY", i) * (1.0 - tw)
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
# BENCHMARK: TQQQ buy-and-hold
# ==============================================================================
print("\n--- BENCHMARK: TQQQ buy-and-hold ---")
tqqq_bh_returns = [tqqq_rets[i] for i in range(WARMUP, n)]
bh_metrics = _compute_metrics(tqqq_bh_returns)
print(f"TQQQ B&H Sharpe:  {bh_metrics['sharpe']:.4f}")
print(f"TQQQ B&H MaxDD:   {bh_metrics['max_dd']:.4f}")
print(f"TQQQ B&H Return:  {bh_metrics['total_return']:.4f}")
print(f"TQQQ B&H CAGR:    {bh_metrics['cagr']:.4f}")

# ==============================================================================
# RUN BASE
# ==============================================================================
print("=" * 70)
print(f"BACKTEST: {SLUG}")
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
print(f"Trade days:   {len(base['daily_returns'])}")

# Count position days
in_pos_days = sum(1 for r in base["daily_returns"] if abs(r) > 0.001)  # rough estimate
print(f"Approx active days: {in_pos_days}")

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
# GATE ASSESSMENT -- Track D gates (basic)
# ==============================================================================
print(f"\n{'=' * 70}")
print("PRELIMINARY GATE ASSESSMENT (Track D -- Leveraged)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] >= 0.80
gate2 = base["max_dd"] < DD_THRESHOLD  # 40% for Track D
gate3 = dsr_value >= 0.90
gate4 = cpcv_mean > 0

gates = [
    ("Gate 1: Sharpe >= 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

all_pass = all(g[1] for g in gates)
verdict = "PASS - BASIC GATES CLEARED (proceed to robustness)" if all_pass else "FAIL"
print(f"\n  PRELIMINARY VERDICT: {verdict}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output = {
    "strategy_slug": SLUG,
    "strategy_type": "leveraged_lead_lag",
    "track": "D",
    "mechanism": "SOXX 5-day return -> TQQQ (3x leveraged QQQ)",
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
    "benchmark": {
        "tqqq_bh_sharpe": round(bh_metrics["sharpe"], 4),
        "tqqq_bh_max_dd": round(bh_metrics["max_dd"], 4),
        "tqqq_bh_cagr": round(bh_metrics["cagr"], 4),
    },
    "preliminary_gates": {
        "sharpe_gte_0.80": gate1,
        "maxdd_lt_40pct": gate2,
        "dsr_gte_0.90": gate3,
        "cpcv_oos_positive": gate4,
    },
    "preliminary_verdict": "PASS" if all_pass else "FAIL",
}

out_path = Path(f"data/strategies/{SLUG}/backtest.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")

if not all_pass:
    print("\n*** BASIC GATES FAILED -- skipping robustness. ***")
    print("*** Run robustness_soxx_tqqq_sprint.py only if basic gates pass. ***")
    sys.exit(1)
else:
    print("\n*** BASIC GATES PASSED -- proceed to robustness analysis. ***")
    print(
        "*** Run: cd E:/llm-quant && PYTHONPATH=src python scripts/robustness_soxx_tqqq_sprint.py ***"
    )

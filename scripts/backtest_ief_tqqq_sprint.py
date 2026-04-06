#!/usr/bin/env python3
"""Backtest for ief-tqqq-sprint (Track D — Sprint Alpha).

Mechanism: IEF (intermediate 7-year Treasury) 10-day return as lead signal
-> TQQQ (3x leveraged QQQ) follower. IEF reacts faster to Fed rate surprises
than TLT, creating a shorter 2-day lag signal.

Signal logic:
  - IEF 10-day return >= entry_threshold (0.5%) -> entry (buy TQQQ at target_weight)
  - IEF 10-day return <= exit_threshold (-0.3%) -> exit to SHY (cash proxy)
  - Use signal from lag_days (2) ago -- no look-ahead
  - Weekly rebalancing (Monday open)
  - Cost per switch: 20 bps round-trip

Track D gates:
  - Gate 1: Sharpe >= 0.80
  - Gate 2: MaxDD < 40%
  - Gate 3: DSR >= 0.90

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_ief_tqqq_sprint.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import polars as pl
from scipy import stats

from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "ief-tqqq-sprint"
SYMBOLS = ["IEF", "TQQQ", "QQQ", "SHY"]
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365  # 1825 days
WARMUP = 60

BASE_PARAMS = {
    "entry_threshold": 0.005,  # IEF 10-day return >= 0.5% -> buy TQQQ
    "exit_threshold": -0.003,  # IEF 10-day return <= -0.3% -> exit
    "lag_days": 2,  # 2-day lag (shorter than TLT's 3-day)
    "signal_window": 10,  # 10-day return lookback for IEF
    "target_weight": 0.30,  # 30% position in TQQQ
    "rebalance": "weekly",  # Weekly rebalancing (Monday open)
}

# ==============================================================================
# FETCH DATA
# ==============================================================================
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


def is_monday(i: int) -> bool:
    """Check if trading day i falls on a Monday."""
    return dates[i].weekday() == 0


def _compute_metrics(daily_returns: list[float]) -> dict:
    """Compute Sharpe, MaxDD, total return, CAGR from daily returns."""
    if not daily_returns or len(daily_returns) < 60:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "n_trades": 0,
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

    Signal logic (with lag + weekly rebalancing):
    - Compute IEF return over signal_window days
    - Use the signal from lag_days ago (causal)
    - Only act on rebalance days (weekly=Monday, daily=every day, biweekly=every other Monday)
    - If IEF return >= entry_threshold -> hold TQQQ at target_weight, rest in SHY
    - If IEF return <= exit_threshold -> exit to SHY (100%)
    - Otherwise hold current position
    """
    entry_thresh = float(params.get("entry_threshold", 0.005))
    exit_thresh = float(params.get("exit_threshold", -0.003))
    lag = int(params.get("lag_days", 2))
    window = int(params.get("signal_window", 10))
    tw = float(params.get("target_weight", 0.30))
    rebalance = str(params.get("rebalance", "weekly"))
    cost_per_switch = 0.0020  # 20 bps round-trip

    daily_returns = []
    in_position = False
    n_trades = 0
    min_lookback = WARMUP + window + lag
    monday_count = 0

    for i in range(WARMUP, n):
        if i < min_lookback:
            daily_returns.append(asset_ret("SHY", i))
            continue

        # Determine if this is a rebalance day
        is_rebal_day = False
        if rebalance == "daily":
            is_rebal_day = True
        elif rebalance == "weekly":
            is_rebal_day = is_monday(i)
        elif rebalance == "biweekly":
            if is_monday(i):
                monday_count += 1
                is_rebal_day = monday_count % 2 == 1
        else:
            is_rebal_day = is_monday(i)

        prev_position = in_position

        if is_rebal_day:
            # Signal date: lag_days ago
            signal_idx = i - lag
            if signal_idx < window:
                daily_returns.append(asset_ret("SHY", i))
                continue

            # IEF return over signal_window ending at signal_idx
            d_signal = dates[signal_idx]
            d_signal_lb = dates[signal_idx - window]

            ief_now = sym_data["IEF"].get(d_signal, 0.0)
            ief_lb = sym_data["IEF"].get(d_signal_lb, 0.0)

            if ief_now <= 0 or ief_lb <= 0:
                daily_returns.append(asset_ret("SHY", i))
                continue

            ief_ret = ief_now / ief_lb - 1.0

            # Position logic -- only update on rebalance days
            if ief_ret >= entry_thresh:
                in_position = True
            elif ief_ret <= exit_thresh:
                in_position = False
            # else: hold current state

        # Compute daily return based on current position
        if in_position:
            day_ret = tqqq_rets[i] * tw + asset_ret("SHY", i) * (1.0 - tw)
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


# ==============================================================================
# RUN BASE BACKTEST
# ==============================================================================
print("=" * 70)
print(f"BACKTEST: {SLUG}")
print("=" * 70)
print("Base parameters:")
for k, v in BASE_PARAMS.items():
    print(f"  {k} = {v}")

print("\n--- RUNNING BASE BACKTEST ---")
base = run_single(BASE_PARAMS)
print(f"Sharpe:       {base['sharpe']:.4f}")
print(f"MaxDD:        {base['max_dd']:.4f}")
print(f"Total Return: {base['total_return']:.4f}")
print(f"CAGR:         {base['cagr']:.4f}")
print(f"Trades:       {base['n_trades']}")

# ==============================================================================
# BENCHMARK COMPARISON
# ==============================================================================
print("\n--- BENCHMARK: TQQQ Buy-and-Hold ---")
tqqq_daily = [tqqq_rets[i] for i in range(WARMUP, n)]
bm = _compute_metrics(tqqq_daily)
print(f"TQQQ B&H Sharpe:  {bm['sharpe']:.4f}")
print(f"TQQQ B&H MaxDD:   {bm['max_dd']:.4f}")
print(f"TQQQ B&H CAGR:    {bm['cagr']:.4f}")

# QQQ benchmark
qqq_daily = [asset_ret("QQQ", i) for i in range(WARMUP, n)]
qqq_bm = _compute_metrics(qqq_daily)
print(f"\nQQQ B&H Sharpe:   {qqq_bm['sharpe']:.4f}")
print(f"QQQ B&H MaxDD:    {qqq_bm['max_dd']:.4f}")
print(f"QQQ B&H CAGR:     {qqq_bm['cagr']:.4f}")

# ==============================================================================
# DSR
# ==============================================================================
print("\n--- DSR ---")
sr = base["sharpe"]
T = len(base["daily_returns"])
T_years = T / 252
se_sr = math.sqrt((1 + sr**2 / 2) / T_years) if T_years > 0 else 1.0
dsr_value = float(stats.norm.cdf(sr / se_sr)) if se_sr > 0 else 0.0
print(f"  Sharpe:    {sr:.4f}")
print(f"  T (days):  {T}")
print(f"  T (years): {T_years:.2f}")
print(f"  SE(SR):    {se_sr:.4f}")
print(f"  DSR:       {dsr_value:.4f}")

# ==============================================================================
# BASIC GATE ASSESSMENT
# ==============================================================================
print(f"\n{'=' * 70}")
print("BASIC GATE ASSESSMENT (Track D)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] >= 0.80
gate2 = base["max_dd"] < DD_THRESHOLD
gate3 = dsr_value >= 0.90

gates = [
    ("Gate 1: Sharpe >= 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr_value:.4f}"),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

basic_pass = all(g[1] for g in gates)
print(f"\n  BASIC GATES: {'PASS' if basic_pass else 'FAIL'}")
if basic_pass:
    print("  -> Proceed to robustness testing")
else:
    print("  -> STOP. Basic gates failed. Strategy rejected.")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output = {
    "strategy_slug": SLUG,
    "strategy_type": "leveraged_lead_lag",
    "track": "D",
    "mechanism": "IEF (intermediate Treasury) 10-day return -> TQQQ (3x leveraged QQQ)",
    "base_params": BASE_PARAMS,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "base_cagr": round(base["cagr"], 4),
    "n_trades": base["n_trades"],
    "dsr": round(dsr_value, 4),
    "basic_gates": {
        "sharpe_gte_0.80": gate1,
        "maxdd_lt_40pct": gate2,
        "dsr_gte_0.90": gate3,
    },
    "basic_gates_pass": basic_pass,
    "benchmarks": {
        "tqqq_bh": {
            "sharpe": round(bm["sharpe"], 4),
            "max_dd": round(bm["max_dd"], 4),
            "cagr": round(bm["cagr"], 4),
        },
        "qqq_bh": {
            "sharpe": round(qqq_bm["sharpe"], 4),
            "max_dd": round(qqq_bm["max_dd"], 4),
            "cagr": round(qqq_bm["cagr"], 4),
        },
    },
}

out_path = Path(f"data/strategies/{SLUG}/backtest_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved backtest results to {out_path}")

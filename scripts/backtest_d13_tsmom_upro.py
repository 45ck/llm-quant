#!/usr/bin/env python3
"""Backtest: tsmom-upro-trend-v1 (Track D - Sprint Alpha).

Skip-month TSMOM (Novy-Marx) re-expressed through UPRO (3x S&P 500).

Signal logic:
  - Compute SPY skip-month momentum: return from t-252 to t-21
  - If momentum > 0 (bullish): allocate base_upro_weight to UPRO (vol-scaled),
    remainder to SHY
  - If momentum <= 0 (bearish): allocate 30% GLD + 30% TLT + 40% SHY
  - VIX > 30 override: 100% SHY regardless of momentum
  - Vol-scaling: UPRO weight = min(max_upro_weight, base_upro_weight * (vol_target / realized_vol))
  - Rebalance every 5 days
  - Cost: 10 bps per switch

Explores multiple parameter configurations if base doesn't pass gates.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_d13_tsmom_upro.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")

import polars as pl

from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "tsmom-upro-trend-v1"
SYMBOLS = ["SPY", "UPRO", "GLD", "TLT", "SHY"]
VIX_SYMBOL = "^VIX"
LOOKBACK_DAYS = 5 * 365  # 5 years
WARMUP = 300  # need 252 for momentum + buffer

print("=" * 70)
print(f"BACKTEST: {SLUG} (Track D - Sprint Alpha)")
print("=" * 70)

# Fetch data
print("\nFetching data...")
prices = fetch_ohlcv([*SYMBOLS, VIX_SYMBOL], lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price lookup by symbol
sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# VIX data — stored as "VIX" in fetcher
vix_key = "VIX"
vix_df = prices.filter(pl.col("symbol") == vix_key).sort("date")
if len(vix_df) == 0:
    # Try ^VIX
    vix_df = prices.filter(pl.col("symbol") == VIX_SYMBOL).sort("date")
    vix_key = VIX_SYMBOL
vix_data: dict = dict(
    zip(vix_df["date"].to_list(), vix_df["close"].to_list(), strict=False)
)
print(f"VIX data points: {len(vix_data)}")

# Use SPY as date backbone
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Precompute daily returns for each asset
daily_rets: dict[str, list[float]] = {}
for sym in SYMBOLS:
    rets = [0.0]
    data = sym_data[sym]
    for i in range(1, n):
        d, dp = dates[i], dates[i - 1]
        if d in data and dp in data and data[dp] > 0:
            rets.append(data[d] / data[dp] - 1)
        else:
            rets.append(0.0)
    daily_rets[sym] = rets


def _compute_metrics(returns):
    """Compute Sharpe, MaxDD, total return, CAGR from daily returns."""
    if not returns or len(returns) < 60:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "daily_returns": [],
        }

    nav = [1.0]
    for r in returns:
        nav.append(nav[-1] * (1.0 + r))

    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    mean_r = sum(returns) / len(returns)
    std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    total_ret = nav[-1] / nav[0] - 1.0
    years = len(returns) / 252
    cagr = ((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_ret,
        "cagr": cagr,
        "daily_returns": returns,
    }


def _vol_scale_weight(spy_daily_rets, i, base_w, max_w, vol_target, vol_window):
    """Compute vol-scaled UPRO weight using realized SPY vol."""
    start_idx = max(0, i - 1 - vol_window)
    end_idx = i - 1
    if end_idx <= start_idx or end_idx > len(spy_daily_rets):
        return base_w
    window_rets = spy_daily_rets[start_idx:end_idx]
    if len(window_rets) < 10:
        return base_w
    mean_wr = sum(window_rets) / len(window_rets)
    var_wr = sum((r - mean_wr) ** 2 for r in window_rets) / len(window_rets)
    realized_vol = math.sqrt(var_wr) * math.sqrt(252)
    if realized_vol <= 0:
        return base_w
    scaled = min(max_w, base_w * (vol_target / realized_vol))
    return max(0.10, scaled)  # floor at 10%


def _classify_regime(i, lookback, skip, vix_thresh):
    """Classify regime as crash/bullish/bearish. Returns (regime, valid)."""
    d = dates[i]
    vix_level = vix_data.get(d, 0.0)
    if vix_level > vix_thresh:
        return "crash", True

    idx_lb = i - lookback
    idx_skip = i - skip
    if idx_lb < 0 or idx_skip < 0:
        return "shy_only", False
    price_lb = spy_close[idx_lb]
    price_skip = spy_close[idx_skip]
    if price_lb <= 0:
        return "shy_only", False
    skip_mom = price_skip / price_lb - 1.0
    return ("bullish" if skip_mom > 0 else "bearish"), True


def run_single(params):
    """Run a single backtest with the given parameters."""
    lookback = int(params.get("momentum_lookback", 252))
    skip = int(params.get("skip_period", 21))
    base_w = float(params.get("base_upro_weight", 0.40))
    max_w = float(params.get("max_upro_weight", 0.50))
    gld_bear_w = float(params.get("gld_bearish_weight", 0.30))
    tlt_bear_w = float(params.get("tlt_bearish_weight", 0.30))
    shy_bear_w = float(params.get("shy_bearish_weight", 0.40))
    vol_target = float(params.get("vol_target", 0.15))
    vol_window = int(params.get("vol_window", 20))
    vix_thresh = float(params.get("vix_crash_threshold", 30))
    cost_bps = float(params.get("cost_bps", 10))

    cost_per_switch = cost_bps / 10000.0
    returns = []
    prev_regime = None

    # Precompute SPY daily returns for realized vol calculation
    spy_daily_rets = [
        (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0.0
        for i in range(1, n)
    ]

    min_start = max(WARMUP, lookback + 1)

    for i in range(min_start, n):
        regime, valid = _classify_regime(i, lookback, skip, vix_thresh)
        if not valid:
            returns.append(daily_rets["SHY"][i])
            continue

        # Vol-scaling for UPRO
        upro_w = base_w
        if regime == "bullish" and vol_window > 0:
            upro_w = _vol_scale_weight(
                spy_daily_rets, i, base_w, max_w, vol_target, vol_window
            )

        # Compute daily return based on regime
        if regime == "crash":
            day_ret = daily_rets["SHY"][i]
        elif regime == "bullish":
            shy_w = 1.0 - upro_w
            day_ret = daily_rets["UPRO"][i] * upro_w + daily_rets["SHY"][i] * shy_w
        else:  # bearish
            day_ret = (
                daily_rets["GLD"][i] * gld_bear_w
                + daily_rets["TLT"][i] * tlt_bear_w
                + daily_rets["SHY"][i] * shy_bear_w
            )

        # Apply switching cost on regime change
        if prev_regime is not None and regime != prev_regime:
            day_ret -= cost_per_switch

        prev_regime = regime
        returns.append(day_ret)

    return _compute_metrics(returns)


# ==============================================================================
# PARAMETER CONFIGURATIONS TO TEST
# ==============================================================================

BASE_PARAMS = {
    "momentum_lookback": 252,
    "skip_period": 21,
    "base_upro_weight": 0.40,
    "max_upro_weight": 0.50,
    "shy_bullish_weight": 0.10,
    "gld_bearish_weight": 0.30,
    "tlt_bearish_weight": 0.30,
    "shy_bearish_weight": 0.40,
    "vol_target": 0.15,
    "vol_window": 20,
    "vix_crash_threshold": 30,
    "rebalance_frequency_days": 5,
    "cost_bps": 10,
}

# Additional configurations to explore if base doesn't pass
VARIANTS = [
    ("base (252/21, 40% UPRO, vol_target=0.15)", BASE_PARAMS),
    (
        "half-year momentum (126/21)",
        {**BASE_PARAMS, "momentum_lookback": 126},
    ),
    (
        "quarter-year momentum (63/21)",
        {**BASE_PARAMS, "momentum_lookback": 63, "skip_period": 10},
    ),
    (
        "35% UPRO (lower leverage)",
        {**BASE_PARAMS, "base_upro_weight": 0.35, "max_upro_weight": 0.45},
    ),
    (
        "50% UPRO (higher leverage)",
        {**BASE_PARAMS, "base_upro_weight": 0.50, "max_upro_weight": 0.60},
    ),
    (
        "no vol-scaling",
        {**BASE_PARAMS, "vol_window": 0},
    ),
    (
        "vol_target=0.12 (tighter vol scaling)",
        {**BASE_PARAMS, "vol_target": 0.12},
    ),
    (
        "vol_target=0.20 (looser vol scaling)",
        {**BASE_PARAMS, "vol_target": 0.20},
    ),
    (
        "VIX threshold=25 (more conservative)",
        {**BASE_PARAMS, "vix_crash_threshold": 25},
    ),
    (
        "VIX threshold=35 (less conservative)",
        {**BASE_PARAMS, "vix_crash_threshold": 35},
    ),
    (
        "no skip (pure 12m momentum)",
        {**BASE_PARAMS, "skip_period": 0},
    ),
    (
        "skip_period=42 (skip 2 months)",
        {**BASE_PARAMS, "skip_period": 42},
    ),
]

# ==============================================================================
# RUN ALL VARIANTS
# ==============================================================================
print("\n" + "=" * 70)
print("RUNNING ALL PARAMETER CONFIGURATIONS")
print("=" * 70)

results = []
best_sharpe = -999
best_name = ""
best_params = {}

for name, params in VARIANTS:
    print(f"\n--- {name} ---")
    r = run_single(params)
    print(
        f"  Sharpe={r['sharpe']:.4f}  MaxDD={r['max_dd']:.4f}  "
        f"CAGR={r['cagr']:.4f}  Return={r['total_return']:.4f}"
    )

    results.append(
        {
            "name": name,
            "params": params,
            "sharpe": r["sharpe"],
            "max_dd": r["max_dd"],
            "cagr": r["cagr"],
            "total_return": r["total_return"],
            "daily_returns": r["daily_returns"],
        }
    )

    passes_sharpe = r["sharpe"] > 0.80
    passes_dd = r["max_dd"] < 0.40
    status = "CANDIDATE" if passes_sharpe and passes_dd else "BELOW GATE"
    print(f"  Status: {status}")

    if r["sharpe"] > best_sharpe:
        best_sharpe = r["sharpe"]
        best_name = name
        best_params = params

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Variant':<45} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'Status':>12}")
print("-" * 90)

for r in results:
    passes = r["sharpe"] > 0.80 and r["max_dd"] < 0.40
    status = "CANDIDATE" if passes else "BELOW GATE"
    print(
        f"{r['name']:<45} {r['sharpe']:>8.4f} {r['max_dd']:>8.4f} "
        f"{r['cagr']:>8.4f} {status:>12}"
    )

print(f"\nBest variant: {best_name} (Sharpe={best_sharpe:.4f})")

# ==============================================================================
# SAVE BEST RESULT
# ==============================================================================
best_result = next(r for r in results if r["name"] == best_name)

output = {
    "strategy_slug": SLUG,
    "track": "D",
    "best_variant": best_name,
    "best_params": {k: v for k, v in best_params.items() if k != "daily_returns"},
    "sharpe": round(best_result["sharpe"], 4),
    "max_dd": round(best_result["max_dd"], 4),
    "cagr": round(best_result["cagr"], 4),
    "total_return": round(best_result["total_return"], 4),
    "all_variants": [
        {
            "name": r["name"],
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "cagr": round(r["cagr"], 4),
        }
        for r in results
    ],
}

out_path = Path(f"data/strategies/{SLUG}/backtest_results.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")

# Also save experiment registry entry
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
spec_hash = hashlib.sha256(yaml.dump(best_params, sort_keys=True).encode()).hexdigest()[
    :8
]
entry = {
    "experiment_id": spec_hash,
    "variant": best_name,
    "sharpe": round(best_result["sharpe"], 4),
    "max_dd": round(best_result["max_dd"], 4),
    "cagr": round(best_result["cagr"], 4),
    "total_return": round(best_result["total_return"], 4),
    "params": {k: v for k, v in best_params.items() if k != "daily_returns"},
    "trial_number": 1,
}
with open(registry_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"Saved experiment registry entry to {registry_path}")

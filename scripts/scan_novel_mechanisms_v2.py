#!/usr/bin/env python3
"""Corrected scan: fix look-ahead bias in H12.2 and H12.6.

All signals now use close(i-1) information to decide day-i allocation.
This is the correct causal formulation: observe yesterday, act today.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/scan_novel_mechanisms_v2.py
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "src")

import polars as pl

from llm_quant.data.fetcher import fetch_ohlcv

SYMBOLS = [
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "SHY",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLU",
    "XLP",
    "XLB",
    "XLC",
    "XLRE",
    "VIX",
    "EFA",
    "EEM",
    "IEF",
    "DBA",
    "USO",
]
LOOKBACK = 5 * 365
WARMUP = 60

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK)
print(f"Data: {len(prices)} rows, {len(prices['symbol'].unique())} symbols")

spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

shy_df = prices.filter(pl.col("symbol") == "SHY").sort("date")
shy_data = dict(zip(shy_df["date"].to_list(), shy_df["close"].to_list(), strict=False))

# TLT for blended allocation
tlt_df = prices.filter(pl.col("symbol") == "TLT").sort("date")
tlt_data = dict(zip(tlt_df["date"].to_list(), tlt_df["close"].to_list(), strict=False))

# GLD for blended allocation
gld_df = prices.filter(pl.col("symbol") == "GLD").sort("date")
gld_data = dict(zip(gld_df["date"].to_list(), gld_df["close"].to_list(), strict=False))

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


def asset_ret(data_dict, i):
    """Get daily return for asset at day i."""
    d = dates[i]
    d_prev = dates[i - 1]
    if d in data_dict and d_prev in data_dict and data_dict[d_prev] > 0:
        return data_dict[d] / data_dict[d_prev] - 1
    return 0.0


def metrics(daily_rets):
    if len(daily_rets) < 60:
        return 0.0, 0.0, 0.0
    nav = [1.0]
    for r in daily_rets:
        nav.append(nav[-1] * (1.0 + r))
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    mean = sum(daily_rets) / len(daily_rets)
    std = (sum((r - mean) ** 2 for r in daily_rets) / len(daily_rets)) ** 0.5
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0
    total_ret = nav[-1] / nav[0] - 1.0
    return sharpe, max_dd, total_ret


# ============================================================
# H12.6 CORRECTED: Multi-Asset Breadth (lag-1 signal)
# Signal uses close(i-1) to decide day-i allocation
# ============================================================
print("\n" + "=" * 60)
print("H12.6 CORRECTED: Multi-Asset Breadth (lag-1)")
print("=" * 60)
breadth_assets = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "GLD",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "DBA",
    "USO",
]

for lookback_d, bull_thresh, bear_thresh in [
    (20, 0.60, 0.40),
    (20, 0.65, 0.35),
    (20, 0.70, 0.30),
    (40, 0.60, 0.40),
    (10, 0.55, 0.45),
    (10, 0.60, 0.40),
    (5, 0.60, 0.40),
]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        # Signal based on YESTERDAY's close (i-1) vs lookback ago (i-1-lookback_d)
        d_signal = dates[i - 1]  # yesterday
        d_lb = dates[i - 1 - lookback_d]  # lookback from yesterday

        pos_count = 0
        total_count = 0
        for asset in breadth_assets:
            if (
                d_signal in sym_data[asset]
                and d_lb in sym_data[asset]
                and sym_data[asset][d_lb] > 0
            ):
                total_count += 1
                if sym_data[asset][d_signal] / sym_data[asset][d_lb] - 1 > 0:
                    pos_count += 1

        if total_count == 0:
            strat_rets.append(0.0)
            continue

        breadth = pos_count / total_count

        if breadth >= bull_thresh:
            strat_rets.append(spy_rets[i] * 0.90)
        elif breadth <= bear_thresh:
            strat_rets.append(asset_ret(shy_data, i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    bull_days = sum(1 for r in strat_rets if r != 0 and abs(r) > 0.0001)
    print(
        f"  lb={lookback_d} bull={bull_thresh} bear={bear_thresh}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.6b: Breadth with TLT/GLD blend in risk-off
# Instead of SHY, use 50% TLT + 40% GLD when bearish
# ============================================================
print("\n" + "=" * 60)
print("H12.6b: Breadth with TLT/GLD defensive blend")
print("=" * 60)

for lookback_d, bull_thresh, bear_thresh in [
    (20, 0.60, 0.40),
    (20, 0.65, 0.35),
    (10, 0.60, 0.40),
]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        d_signal = dates[i - 1]
        d_lb = dates[i - 1 - lookback_d]

        pos_count = 0
        total_count = 0
        for asset in breadth_assets:
            if (
                d_signal in sym_data[asset]
                and d_lb in sym_data[asset]
                and sym_data[asset][d_lb] > 0
            ):
                total_count += 1
                if sym_data[asset][d_signal] / sym_data[asset][d_lb] - 1 > 0:
                    pos_count += 1

        if total_count == 0:
            strat_rets.append(0.0)
            continue

        breadth = pos_count / total_count

        if breadth >= bull_thresh:
            strat_rets.append(spy_rets[i] * 0.90)
        elif breadth <= bear_thresh:
            # Defensive: 50% TLT + 40% GLD
            tlt_r = asset_ret(tlt_data, i)
            gld_r = asset_ret(gld_data, i)
            strat_rets.append(tlt_r * 0.50 + gld_r * 0.40)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} bull={bull_thresh} bear={bear_thresh}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.2 CORRECTED: Cross-Sector Dispersion (lag-1 signal)
# ============================================================
print("\n" + "=" * 60)
print("H12.2 CORRECTED: Cross-Sector Dispersion (lag-1)")
print("=" * 60)
sector_etfs = [
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLU",
    "XLP",
    "XLB",
    "XLC",
    "XLRE",
]

for lookback_lb, threshold_pct in [
    (10, 60),
    (10, 50),
    (10, 70),
    (20, 60),
    (20, 70),
    (5, 60),
]:
    strat_rets = []
    all_dispersions = []

    for i in range(WARMUP, n):
        if i < lookback_lb + 1:
            all_dispersions.append(0.0)
            strat_rets.append(0.0)
            continue

        # Use YESTERDAY's dispersion (lag-1)
        d_signal = dates[i - 1]
        d_lb = dates[i - 1 - lookback_lb]

        sector_rets = []
        for etf in sector_etfs:
            if (
                d_signal in sym_data[etf]
                and d_lb in sym_data[etf]
                and sym_data[etf][d_lb] > 0
            ):
                sector_rets.append(sym_data[etf][d_signal] / sym_data[etf][d_lb] - 1)

        if len(sector_rets) < 5:
            all_dispersions.append(0.0)
            strat_rets.append(0.0)
            continue

        mean_r = sum(sector_rets) / len(sector_rets)
        dispersion = (
            sum((r - mean_r) ** 2 for r in sector_rets) / len(sector_rets)
        ) ** 0.5
        all_dispersions.append(dispersion)

        hist_window = all_dispersions[max(0, len(all_dispersions) - 252) :]
        if len(hist_window) < 20:
            strat_rets.append(spy_rets[i] * 0.90)
            continue

        pct_rank = (
            sum(1 for h in hist_window if h <= dispersion) / len(hist_window) * 100
        )

        if pct_rank > threshold_pct:
            strat_rets.append(asset_ret(shy_data, i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.90)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_lb} threshold={threshold_pct}%: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.7: Momentum + Breadth combo
# Breadth > 60% AND SPY above SMA50 -> full long
# Otherwise defensive
# ============================================================
print("\n" + "=" * 60)
print("H12.7: Momentum + Breadth Combo")
print("=" * 60)

for lookback_d, sma_period, bull_thresh in [
    (20, 50, 0.60),
    (20, 50, 0.65),
    (10, 50, 0.60),
    (20, 100, 0.60),
]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < max(lookback_d, sma_period) + 1:
            strat_rets.append(0.0)
            continue

        d_signal = dates[i - 1]
        d_lb = dates[i - 1 - lookback_d]

        # Breadth
        pos_count = 0
        total_count = 0
        for asset in breadth_assets:
            if (
                d_signal in sym_data[asset]
                and d_lb in sym_data[asset]
                and sym_data[asset][d_lb] > 0
            ):
                total_count += 1
                if sym_data[asset][d_signal] / sym_data[asset][d_lb] - 1 > 0:
                    pos_count += 1
        breadth = pos_count / total_count if total_count > 0 else 0.5

        # SPY SMA
        sma = sum(spy_close[i - 1 - sma_period : i - 1]) / sma_period
        spy_above_sma = spy_close[i - 1] > sma

        if breadth >= bull_thresh and spy_above_sma:
            strat_rets.append(spy_rets[i] * 0.90)
        elif breadth < 0.35 or not spy_above_sma:
            strat_rets.append(asset_ret(shy_data, i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} sma={sma_period} bull={bull_thresh}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

print("\n" + "=" * 60)
print("CORRECTED SCAN COMPLETE")
print("=" * 60)

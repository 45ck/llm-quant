#!/usr/bin/env python3
"""Quick scan of novel strategy mechanisms for SR 2.0+ sprint.

Tests 4 hypotheses without full strategy class implementation:
  H12.1: Momentum Acceleration (2nd derivative timing)
  H12.2: Cross-sector dispersion (high dispersion → defensive)
  H12.3: Turn-of-month effect (calendar anomaly)
  H12.4: VIX mean-reversion timing (buy VIX spikes, hold for normalization)

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/scan_novel_mechanisms.py
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
print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")

# Get SPY dates as trading days
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# SHY data
shy_df = prices.filter(pl.col("symbol") == "SHY").sort("date")
shy_data = dict(zip(shy_df["date"].to_list(), shy_df["close"].to_list(), strict=False))

# All symbol close data
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


def metrics(daily_rets):
    """Compute Sharpe, MaxDD, total return from daily returns."""
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


def shy_ret(i):
    """Get SHY return for day i."""
    d = dates[i]
    d_prev = dates[i - 1]
    if d in shy_data and d_prev in shy_data and shy_data[d_prev] > 0:
        return shy_data[d] / shy_data[d_prev] - 1
    return 0.0


# ============================================================
# H12.1: Momentum Acceleration (2nd derivative)
# ============================================================
print("\n" + "=" * 60)
print("H12.1: Momentum Acceleration (2nd derivative timing)")
print("=" * 60)
for mom_period, accel_period in [(20, 5), (20, 10), (40, 10), (10, 5)]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < mom_period + accel_period:
            strat_rets.append(0.0)
            continue
        mom_now = spy_close[i - 1] / spy_close[i - 1 - mom_period] - 1
        mom_prev = (
            spy_close[i - 1 - accel_period]
            / spy_close[i - 1 - mom_period - accel_period]
            - 1
        )
        accel = mom_now - mom_prev

        if mom_now > 0 and accel > 0:
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            strat_rets.append(shy_ret(i) * 0.90)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  mom={mom_period} accel={accel_period}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.2: Cross-sector dispersion timing
# ============================================================
print("\n" + "=" * 60)
print("H12.2: Cross-Sector Dispersion Timing")
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

for lookback, threshold_pct in [(20, 50), (20, 70), (40, 50), (10, 60)]:
    strat_rets = []
    # Pre-compute rolling dispersion
    all_dispersions = []
    for i in range(WARMUP, n):
        if i < lookback:
            all_dispersions.append(0.0)
            strat_rets.append(0.0)
            continue

        d = dates[i]
        d_lb = dates[i - lookback]
        sector_rets = []
        for etf in sector_etfs:
            if d in sym_data[etf] and d_lb in sym_data[etf] and sym_data[etf][d_lb] > 0:
                sector_rets.append(sym_data[etf][d] / sym_data[etf][d_lb] - 1)

        if len(sector_rets) < 5:
            all_dispersions.append(0.0)
            strat_rets.append(0.0)
            continue

        mean_r = sum(sector_rets) / len(sector_rets)
        dispersion = (
            sum((r - mean_r) ** 2 for r in sector_rets) / len(sector_rets)
        ) ** 0.5
        all_dispersions.append(dispersion)

        # Percentile vs last 252 days
        hist_window = all_dispersions[max(0, len(all_dispersions) - 252) :]
        if len(hist_window) < 20:
            strat_rets.append(spy_rets[i] * 0.90)
            continue

        pct_rank = (
            sum(1 for h in hist_window if h <= dispersion) / len(hist_window) * 100
        )

        if pct_rank > threshold_pct:  # High dispersion -> defensive
            strat_rets.append(shy_ret(i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.90)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback} threshold={threshold_pct}%: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.3: Turn-of-month effect
# ============================================================
print("\n" + "=" * 60)
print("H12.3: Turn-of-Month Effect")
print("=" * 60)
for pre_days, post_days in [(3, 3), (2, 4), (4, 2), (1, 5)]:
    strat_rets = []
    for i in range(WARMUP, n):
        d = dates[i]
        day = d.day
        near_end = day >= (29 - pre_days)
        near_start = day <= post_days

        if near_start or near_end:
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            strat_rets.append(shy_ret(i) * 0.90)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  pre={pre_days} post={post_days}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.4: VIX mean-reversion timing
# ============================================================
print("\n" + "=" * 60)
print("H12.4: VIX Mean-Reversion Timing")
print("=" * 60)

for entry_pct, exit_pct, hold_min in [
    (80, 40, 5),
    (90, 50, 3),
    (70, 30, 7),
    (85, 45, 5),
]:
    strat_rets = []
    in_trade = False
    hold_count = 0

    for i in range(WARMUP, n):
        d = dates[i]
        # VIX percentile over last 252 days
        vix_hist = []
        for j in range(max(0, i - 252), i):
            dj = dates[j]
            if dj in sym_data["VIX"]:
                vix_hist.append(sym_data["VIX"][dj])

        if d not in sym_data["VIX"] or len(vix_hist) < 60:
            strat_rets.append(0.0)
            continue

        current_vix = sym_data["VIX"][d]
        vix_pct = sum(1 for v in vix_hist if v <= current_vix) / len(vix_hist) * 100

        if not in_trade and vix_pct >= entry_pct:
            in_trade = True
            hold_count = 0

        if in_trade:
            hold_count += 1
            if hold_count >= hold_min and vix_pct <= exit_pct:
                in_trade = False

        if in_trade:
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            strat_rets.append(shy_ret(i) * 0.90)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  entry_pct={entry_pct} exit_pct={exit_pct} hold_min={hold_min}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.5: Dollar strength via USDJPY inverse (strong $ = risk-off)
# ============================================================
print("\n" + "=" * 60)
print("H12.5: Dollar Strength Signal (USDJPY momentum)")
print("=" * 60)
usdjpy_data = sym_data.get("USDJPY=X", {})
if not usdjpy_data:
    # Try alternative key
    for sym, value in sym_data.items():
        if "USDJPY" in sym:
            usdjpy_data = value
            break

for lookback_d, threshold in [(20, 0.02), (20, 0.03), (40, 0.03), (10, 0.015)]:
    strat_rets = []
    for i in range(WARMUP, n):
        d = dates[i]
        if i < lookback_d:
            strat_rets.append(0.0)
            continue
        d_lb = dates[i - lookback_d]
        if d not in usdjpy_data or d_lb not in usdjpy_data or usdjpy_data[d_lb] <= 0:
            strat_rets.append(spy_rets[i] * 0.90)
            continue

        usd_mom = usdjpy_data[d] / usdjpy_data[d_lb] - 1

        if usd_mom > threshold:  # Strong dollar -> risk-off
            strat_rets.append(shy_ret(i) * 0.90)
        elif usd_mom < -threshold:  # Weak dollar -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.50)  # Neutral -> half

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} threshold={threshold}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H12.6: Multi-asset breadth (% of assets with positive 20d momentum)
# ============================================================
print("\n" + "=" * 60)
print("H12.6: Multi-Asset Breadth Signal")
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
    (20, 0.70, 0.30),
    (40, 0.60, 0.40),
    (10, 0.55, 0.45),
]:
    strat_rets = []
    for i in range(WARMUP, n):
        d = dates[i]
        if i < lookback_d:
            strat_rets.append(0.0)
            continue
        d_lb = dates[i - lookback_d]

        pos_count = 0
        total_count = 0
        for asset in breadth_assets:
            if (
                d in sym_data[asset]
                and d_lb in sym_data[asset]
                and sym_data[asset][d_lb] > 0
            ):
                total_count += 1
                if sym_data[asset][d] / sym_data[asset][d_lb] - 1 > 0:
                    pos_count += 1

        if total_count == 0:
            strat_rets.append(0.0)
            continue

        breadth = pos_count / total_count

        if breadth >= bull_thresh:
            strat_rets.append(spy_rets[i] * 0.90)
        elif breadth <= bear_thresh:
            strat_rets.append(shy_ret(i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} bull={bull_thresh} bear={bear_thresh}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

print("\n" + "=" * 60)
print("SCAN COMPLETE")
print("=" * 60)

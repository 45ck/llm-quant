#!/usr/bin/env python3
"""Cross-asset timing signals scan — these actually work in our portfolio.

Pattern: use information from asset class A to time asset class B.
All signals lag-1 (no look-ahead).

H13.1: Credit spread regime (HYG/SHY ratio momentum → equity timing)
H13.2: Bond/equity relative strength (TLT/SPY ratio → regime switch)
H13.3: Commodity/equity ratio (GLD/SPY or DBA/SPY momentum → regime)
H13.4: Cross-asset momentum divergence (when credit + rate + commodity agree)
H13.5: EEM/SPY relative strength (EM leadership → equity regime)
H13.6: Real rate signal (TIP/IEF ratio → equity)

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/scan_crossasset_v3.py
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
    "HYG",
    "LQD",
    "IEF",
    "EFA",
    "EEM",
    "TIP",
    "DBA",
    "USO",
    "AGG",
    "VIX",
    "XLK",
    "XLE",
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
    d = dates[i]
    d_prev = dates[i - 1]
    data = sym_data[sym]
    if d in data and d_prev in data and data[d_prev] > 0:
        return data[d] / data[d_prev] - 1
    return 0.0


def get_close(sym, i):
    d = dates[i]
    return sym_data[sym].get(d, 0.0)


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
# H13.1: Credit Spread Regime (HYG/SHY ratio momentum)
# When HYG outperforms SHY (spread tightening) -> risk-on
# When HYG underperforms SHY (spread widening) -> risk-off
# Different from lead-lag: this is LEVEL-based regime, not return prediction
# ============================================================
print("\n" + "=" * 60)
print("H13.1: Credit Spread Regime (HYG/SHY ratio momentum)")
print("=" * 60)

for lookback_d, sma_period in [
    (10, 20),
    (10, 50),
    (20, 50),
    (5, 20),
    (20, 100),
    (10, 40),
]:
    strat_rets = []
    # Pre-compute HYG/SHY ratio
    ratio_series = []
    for i in range(n):
        hyg = get_close("HYG", i)
        shy = get_close("SHY", i)
        if hyg > 0 and shy > 0:
            ratio_series.append(hyg / shy)
        else:
            ratio_series.append(0.0)

    for i in range(WARMUP, n):
        if i < max(lookback_d, sma_period) + 1:
            strat_rets.append(0.0)
            continue

        # Signal at close i-1 (lag-1)
        ratio_now = ratio_series[i - 1]
        ratio_lb = ratio_series[i - 1 - lookback_d]
        ratio_sma = sum(ratio_series[i - 1 - sma_period : i - 1]) / sma_period

        if ratio_now <= 0 or ratio_lb <= 0 or ratio_sma <= 0:
            strat_rets.append(0.0)
            continue

        ratio_mom = ratio_now / ratio_lb - 1
        ratio_vs_sma = ratio_now / ratio_sma - 1

        # Risk-on: ratio rising AND above SMA (credit improving)
        if ratio_mom > 0 and ratio_vs_sma > 0:
            strat_rets.append(spy_rets[i] * 0.90)
        # Risk-off: ratio falling AND below SMA
        elif ratio_mom < 0 and ratio_vs_sma < 0:
            strat_rets.append(asset_ret("SHY", i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} sma={sma_period}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H13.2: Bond/Equity Relative Strength (TLT/SPY ratio)
# When TLT outperforming SPY (ratio rising) -> defensive
# ============================================================
print("\n" + "=" * 60)
print("H13.2: Bond/Equity Relative Strength (TLT/SPY ratio)")
print("=" * 60)

for lookback_d, threshold in [(20, 0.0), (20, 0.01), (40, 0.0), (10, 0.0), (60, 0.0)]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        # TLT/SPY ratio momentum at close i-1
        tlt_now = get_close("TLT", i - 1)
        spy_now = spy_close[i - 1]
        tlt_lb = get_close("TLT", i - 1 - lookback_d)
        spy_lb = spy_close[i - 1 - lookback_d]

        if tlt_now <= 0 or spy_now <= 0 or tlt_lb <= 0 or spy_lb <= 0:
            strat_rets.append(0.0)
            continue

        ratio_now = tlt_now / spy_now
        ratio_lb = tlt_lb / spy_lb
        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom > threshold:  # Bonds outperforming -> defensive
            strat_rets.append(asset_ret("TLT", i) * 0.50 + asset_ret("GLD", i) * 0.30)
        elif ratio_mom < -threshold:  # Stocks outperforming -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} threshold={threshold}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H13.3: Commodity Leadership (GLD/SPY or DBA/SPY)
# Commodity strength -> late-cycle/inflationary -> defensive equities
# ============================================================
print("\n" + "=" * 60)
print("H13.3: Commodity Leadership Signal")
print("=" * 60)

for commodity, lookback_d, threshold in [
    ("GLD", 20, 0.02),
    ("GLD", 20, 0.03),
    ("GLD", 40, 0.03),
    ("DBA", 20, 0.02),
    ("GLD", 10, 0.01),
]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        com_now = get_close(commodity, i - 1)
        spy_now = spy_close[i - 1]
        com_lb = get_close(commodity, i - 1 - lookback_d)
        spy_lb = spy_close[i - 1 - lookback_d]

        if com_now <= 0 or spy_now <= 0 or com_lb <= 0 or spy_lb <= 0:
            strat_rets.append(0.0)
            continue

        ratio_now = com_now / spy_now
        ratio_lb = com_lb / spy_lb
        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom > threshold:  # Commodity leading -> hedge
            strat_rets.append(
                asset_ret(commodity, i) * 0.50 + asset_ret("SHY", i) * 0.40
            )
        elif ratio_mom < -threshold:  # Equity leading -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.60 + asset_ret(commodity, i) * 0.20)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  {commodity} lb={lookback_d} thresh={threshold}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H13.4: Cross-asset consensus (credit + rate + commodity agreement)
# When ALL three say risk-on -> full long
# When >= 2 say risk-off -> defensive
# ============================================================
print("\n" + "=" * 60)
print("H13.4: Cross-Asset Consensus (3-signal voting)")
print("=" * 60)

for lookback_d in [5, 10, 20]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        signals = []

        # Signal 1: Credit (HYG momentum > 0)
        hyg_now = get_close("HYG", i - 1)
        hyg_lb = get_close("HYG", i - 1 - lookback_d)
        if hyg_now > 0 and hyg_lb > 0:
            signals.append(1 if hyg_now / hyg_lb - 1 > 0 else -1)

        # Signal 2: Rate (TLT momentum > 0 = rates falling = equity positive)
        tlt_now = get_close("TLT", i - 1)
        tlt_lb = get_close("TLT", i - 1 - lookback_d)
        if tlt_now > 0 and tlt_lb > 0:
            signals.append(1 if tlt_now / tlt_lb - 1 > 0 else -1)

        # Signal 3: Commodity (GLD not outpacing SPY = not late cycle)
        gld_now = get_close("GLD", i - 1)
        gld_lb = get_close("GLD", i - 1 - lookback_d)
        spy_lb = spy_close[i - 1 - lookback_d]
        if gld_now > 0 and gld_lb > 0 and spy_lb > 0:
            gld_rel = (gld_now / gld_lb) / (spy_close[i - 1] / spy_lb) - 1
            signals.append(-1 if gld_rel > 0.02 else 1)

        if len(signals) < 2:
            strat_rets.append(0.0)
            continue

        vote = sum(signals)

        if vote >= 2:  # Strong consensus risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        elif vote <= -2:  # Strong consensus risk-off
            strat_rets.append(
                asset_ret("TLT", i) * 0.40
                + asset_ret("GLD", i) * 0.30
                + asset_ret("SHY", i) * 0.20
            )
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(f"  lb={lookback_d}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

# ============================================================
# H13.5: EM/DM Relative Strength (EEM/EFA ratio)
# ============================================================
print("\n" + "=" * 60)
print("H13.5: EM/DM Relative Strength (EEM/EFA)")
print("=" * 60)

for lookback_d, threshold in [(20, 0.0), (20, 0.01), (40, 0.0), (10, 0.0)]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        eem_now = get_close("EEM", i - 1)
        efa_now = get_close("EFA", i - 1)
        eem_lb = get_close("EEM", i - 1 - lookback_d)
        efa_lb = get_close("EFA", i - 1 - lookback_d)

        if eem_now <= 0 or efa_now <= 0 or eem_lb <= 0 or efa_lb <= 0:
            strat_rets.append(0.0)
            continue

        ratio_now = eem_now / efa_now
        ratio_lb = eem_lb / efa_lb
        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom > threshold:  # EM leading -> risk-on (growth sensitive)
            strat_rets.append(spy_rets[i] * 0.90)
        elif ratio_mom < -threshold:  # DM leading -> defensive
            strat_rets.append(asset_ret("SHY", i) * 0.50 + spy_rets[i] * 0.30)
        else:
            strat_rets.append(spy_rets[i] * 0.60)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} thresh={threshold}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

# ============================================================
# H13.6: Real Rate Signal (TIP/IEF ratio)
# Rising real rates (TIP outperforming IEF = breakevens falling) -> equity negative
# ============================================================
print("\n" + "=" * 60)
print("H13.6: Real Rate Signal (TIP/IEF ratio)")
print("=" * 60)

for lookback_d, threshold in [(10, 0.0), (20, 0.0), (20, 0.005), (40, 0.0)]:
    strat_rets = []
    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        tip_now = get_close("TIP", i - 1)
        ief_now = get_close("IEF", i - 1)
        tip_lb = get_close("TIP", i - 1 - lookback_d)
        ief_lb = get_close("IEF", i - 1 - lookback_d)

        if tip_now <= 0 or ief_now <= 0 or tip_lb <= 0 or ief_lb <= 0:
            strat_rets.append(0.0)
            continue

        ratio_now = tip_now / ief_now
        ratio_lb = tip_lb / ief_lb
        ratio_mom = ratio_now / ratio_lb - 1

        # TIP outperforming IEF means real rates falling (inflation expectations rising)
        # This is actually equity-positive in moderate inflation
        if ratio_mom > threshold:  # Inflation rising -> commodity/equity tilt
            strat_rets.append(spy_rets[i] * 0.60 + asset_ret("GLD", i) * 0.30)
        elif ratio_mom < -threshold:  # Real rates rising -> defensive
            strat_rets.append(asset_ret("SHY", i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    print(
        f"  lb={lookback_d} thresh={threshold}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}"
    )

print("\n" + "=" * 60)
print("CROSS-ASSET SCAN COMPLETE")
print("=" * 60)

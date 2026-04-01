#!/usr/bin/env python3
"""Cross-asset timing signals scan v5 — orthogonal hypothesis hunt.

Targeting mechanisms UNCORRELATED with ALL existing portfolio families:
  - Credit-equity lead-lag (HYG, LQD, AGG, VCIT, EMB -> SPY/QQQ)
  - Rate momentum (TLT, IEF -> SPY/QQQ)
  - Overnight momentum (SPY overnight returns)
  - Mean reversion (GLD-SLV ratio)
  - TSMOM (skip-month vol-scaled momentum)
  - Behavioral (structural patterns)
  - Non-credit lead-lag (SOXX -> QQQ)
  - Credit spread regime (HYG/SHY ratio -> SPY/SHY)
  - Commodity cycle (DBA momentum -> SPY/GLD)

Goal: Find ONE more strategy with LOW correlation to push portfolio SR
from 1.84 to 2.0+.

H15.1: FX Dollar Regime (EFA/SPY inverse as USD strength proxy)
H15.2: Sector Rotation Momentum (XLK/XLE growth vs value regime)
H15.3: Small Cap Relative Strength Regime (IWM/SPY breadth signal)
H15.4: Yield Curve Slope via Bond ETFs (TLT/SHY ratio regime)
H15.5: Multi-Asset Volatility Regime (SPY vol vs GLD vol)
H15.6: Cross-Market Momentum Divergence (macro breadth count)
H15.7: EEM Macro Regime (EM absolute momentum as risk proxy)

All signals use lag-1 (close[i-1] for decisions at day i).
Lookback: 5*365 days. Warmup: 60 days.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/scan_crossasset_v5.py
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
    "EFA",
    "EEM",
    "IWM",
    "DBA",
    "XLK",
    "XLE",
    "VIX",
    "IEF",
]
LOOKBACK = 5 * 365
WARMUP = 60

print("=" * 70)
print("CROSS-ASSET HYPOTHESIS SCAN V5 — ORTHOGONAL MECHANISM HUNT")
print("=" * 70)
print()

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK)
print(f"Data: {len(prices)} rows, {len(prices['symbol'].unique())} symbols")

spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Build price lookup by symbol
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


def asset_ret(sym: str, i: int) -> float:
    """Return for symbol on day i."""
    d = dates[i]
    d_prev = dates[i - 1]
    data = sym_data[sym]
    if d in data and d_prev in data and data[d_prev] > 0:
        return data[d] / data[d_prev] - 1
    return 0.0


def get_close(sym: str, i: int) -> float:
    """Close price for symbol on day i."""
    d = dates[i]
    return sym_data[sym].get(d, 0.0)


def metrics(daily_rets: list[float]) -> tuple[float, float, float]:
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


def sma(series: list[float], end_idx: int, window: int) -> float:
    """Simple moving average ending at end_idx (inclusive), looking back window bars."""
    if end_idx < window - 1:
        return 0.0
    segment = series[end_idx - window + 1 : end_idx + 1]
    valid = [x for x in segment if x > 0]
    if len(valid) < window // 2:
        return 0.0
    return sum(valid) / len(valid)


def realized_vol(rets: list[float], end_idx: int, window: int) -> float:
    """Annualized realized vol ending at end_idx."""
    if end_idx < window:
        return 0.0
    segment = rets[end_idx - window + 1 : end_idx + 1]
    if len(segment) < window:
        return 0.0
    m = sum(segment) / len(segment)
    var = sum((r - m) ** 2 for r in segment) / len(segment)
    return var**0.5 * math.sqrt(252)


# Pre-compute price series as lists for fast access
price_lists: dict[str, list[float]] = {}
for sym in SYMBOLS:
    price_lists[sym] = [get_close(sym, i) for i in range(n)]

# Pre-compute daily returns for all symbols
ret_lists: dict[str, list[float]] = {}
for sym in SYMBOLS:
    rets = [0.0]
    for i in range(1, n):
        rets.append(asset_ret(sym, i))
    ret_lists[sym] = rets

# Pre-compute ratio series we'll need multiple times
efa_spy_ratio = []
for i in range(n):
    e = price_lists["EFA"][i]
    s = price_lists["SPY"][i]
    if e > 0 and s > 0:
        efa_spy_ratio.append(e / s)
    else:
        efa_spy_ratio.append(0.0)

iwm_spy_ratio = []
for i in range(n):
    w = price_lists["IWM"][i]
    s = price_lists["SPY"][i]
    if w > 0 and s > 0:
        iwm_spy_ratio.append(w / s)
    else:
        iwm_spy_ratio.append(0.0)

tlt_shy_ratio = []
for i in range(n):
    t = price_lists["TLT"][i]
    sh = price_lists["SHY"][i]
    if t > 0 and sh > 0:
        tlt_shy_ratio.append(t / sh)
    else:
        tlt_shy_ratio.append(0.0)

xlk_xle_ratio = []
for i in range(n):
    k = price_lists["XLK"][i]
    e = price_lists["XLE"][i]
    if k > 0 and e > 0:
        xlk_xle_ratio.append(k / e)
    else:
        xlk_xle_ratio.append(0.0)

results: list[tuple[str, str, float, float, float]] = []


def record(hyp: str, config: str, s: float, dd: float, tr: float) -> None:
    """Record a result and print it."""
    tag = " *** PASS ***" if s > 0.80 and dd < 0.15 else ""
    print(f"  {config}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}")
    results.append((hyp, config, s, dd, tr))


# ============================================================
# H15.1: FX Dollar Regime
# USD strength proxy: inverse of EFA/SPY ratio
# When EFA falling vs SPY -> dollar strengthening -> defensive
# Uses SMA-based regime classification (not raw momentum)
# ============================================================
print("\n" + "=" * 70)
print("H15.1: FX Dollar Regime (EFA/SPY inverse as USD strength)")
print("Mechanism: dollar strength regime via international underperformance")
print("When USD strong (EFA/SPY ratio falling below SMA) -> defensive")
print("When USD weak (EFA/SPY ratio rising above SMA) -> risk-on")
print("=" * 70)

for lookback_d, sma_period in [
    (20, 50),
    (20, 100),
    (40, 50),
    (40, 100),
]:
    strat_rets: list[float] = []
    required = max(lookback_d, sma_period) + 1

    for i in range(WARMUP, n):
        if i < required:
            strat_rets.append(0.0)
            continue

        # Lag-1: use data at i-1
        ratio_now = efa_spy_ratio[i - 1]
        ratio_lb = efa_spy_ratio[i - 1 - lookback_d]
        sma_val = sma(efa_spy_ratio, i - 1, sma_period)

        if ratio_now <= 0 or ratio_lb <= 0 or sma_val <= 0:
            strat_rets.append(0.0)
            continue

        mom = ratio_now / ratio_lb - 1
        above_sma = ratio_now > sma_val

        if mom > 0 and above_sma:
            # USD weakening (EFA outperforming) -> global risk-on
            # Tilt toward international + SPY
            strat_rets.append(asset_ret("EFA", i) * 0.45 + spy_rets[i] * 0.45)
        elif mom < 0 and not above_sma:
            # USD strengthening (EFA underperforming) -> defensive
            strat_rets.append(asset_ret("SHY", i) * 0.50 + spy_rets[i] * 0.40)
        else:
            # Mixed signal -> moderate SPY
            strat_rets.append(spy_rets[i] * 0.60 + asset_ret("SHY", i) * 0.30)

    s, dd, tr = metrics(strat_rets)
    record("H15.1", f"lb={lookback_d} sma={sma_period}", s, dd, tr)


# ============================================================
# H15.2: Sector Rotation Momentum (XLK/XLE)
# Growth vs Value proxy regime signal
# XLK outperforming XLE -> growth/tech regime -> long QQQ
# XLE outperforming XLK -> value/inflation regime -> long GLD+DBA
# ============================================================
print("\n" + "=" * 70)
print("H15.2: Sector Rotation Momentum (XLK/XLE growth vs value)")
print("Mechanism: tech vs energy relative strength as regime signal")
print("Growth regime (XLK leading) -> QQQ; Value regime (XLE leading) -> GLD+DBA")
print("=" * 70)

for lookback_d, sma_period in [
    (10, 20),
    (10, 50),
    (20, 20),
    (20, 50),
    (40, 20),
    (40, 50),
]:
    strat_rets = []
    required = max(lookback_d, sma_period) + 1

    for i in range(WARMUP, n):
        if i < required:
            strat_rets.append(0.0)
            continue

        ratio_now = xlk_xle_ratio[i - 1]
        ratio_lb = xlk_xle_ratio[i - 1 - lookback_d]
        sma_val = sma(xlk_xle_ratio, i - 1, sma_period)

        if ratio_now <= 0 or ratio_lb <= 0 or sma_val <= 0:
            strat_rets.append(0.0)
            continue

        mom = ratio_now / ratio_lb - 1
        above_sma = ratio_now > sma_val

        if mom > 0 and above_sma:
            # Growth/tech regime -> long QQQ
            strat_rets.append(asset_ret("QQQ", i) * 0.90)
        elif mom < 0 and not above_sma:
            # Value/inflation regime -> GLD + DBA
            strat_rets.append(
                asset_ret("GLD", i) * 0.50
                + asset_ret("DBA", i) * 0.20
                + asset_ret("SHY", i) * 0.20
            )
        else:
            # Transition -> diversified
            strat_rets.append(
                spy_rets[i] * 0.45
                + asset_ret("GLD", i) * 0.25
                + asset_ret("SHY", i) * 0.20
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.2", f"lb={lookback_d} sma={sma_period}", s, dd, tr)


# ============================================================
# H15.3: Small Cap Relative Strength Regime (IWM/SPY)
# IWM outperforming SPY -> broad risk appetite -> full equity
# IWM underperforming SPY -> flight to quality -> defensive
# Uses ratio vs SMA for regime classification
# ============================================================
print("\n" + "=" * 70)
print("H15.3: Small Cap Relative Strength Regime (IWM/SPY)")
print("Mechanism: small cap leadership as risk appetite breadth signal")
print("IWM leading SPY -> broad rally -> full SPY")
print("IWM lagging SPY -> narrow/defensive market -> SHY tilt")
print("=" * 70)

for lookback_d, sma_period in [
    (10, 20),
    (10, 50),
    (20, 20),
    (20, 50),
]:
    strat_rets = []
    required = max(lookback_d, sma_period) + 1

    for i in range(WARMUP, n):
        if i < required:
            strat_rets.append(0.0)
            continue

        ratio_now = iwm_spy_ratio[i - 1]
        ratio_lb = iwm_spy_ratio[i - 1 - lookback_d]
        sma_val = sma(iwm_spy_ratio, i - 1, sma_period)

        if ratio_now <= 0 or ratio_lb <= 0 or sma_val <= 0:
            strat_rets.append(0.0)
            continue

        mom = ratio_now / ratio_lb - 1
        above_sma = ratio_now > sma_val

        if mom > 0 and above_sma:
            # Broad risk appetite -> full SPY
            strat_rets.append(spy_rets[i] * 0.90)
        elif mom < 0 and not above_sma:
            # Flight to quality -> defensive
            strat_rets.append(asset_ret("SHY", i) * 0.60 + spy_rets[i] * 0.30)
        else:
            # Mixed -> moderate
            strat_rets.append(spy_rets[i] * 0.60 + asset_ret("SHY", i) * 0.30)

    s, dd, tr = metrics(strat_rets)
    record("H15.3", f"lb={lookback_d} sma={sma_period}", s, dd, tr)


# ============================================================
# H15.4: Yield Curve Slope via Bond ETFs (TLT/SHY ratio)
# TLT/SHY ratio falling -> bear steepening or curve flattening -> defensive
# TLT/SHY ratio rising -> bull steepening -> risk-on
# Different from rate momentum: uses RATIO not absolute TLT
# ============================================================
print("\n" + "=" * 70)
print("H15.4: Yield Curve Slope via Bond ETFs (TLT/SHY ratio)")
print("Mechanism: yield curve shape as macro regime signal")
print("Ratio rising (curve steepening favorably) -> risk-on")
print("Ratio falling (curve flattening/inverting) -> defensive")
print("=" * 70)

for lookback_d, sma_period in [
    (20, 50),
    (20, 100),
    (40, 50),
    (40, 100),
]:
    strat_rets = []
    required = max(lookback_d, sma_period) + 1

    for i in range(WARMUP, n):
        if i < required:
            strat_rets.append(0.0)
            continue

        ratio_now = tlt_shy_ratio[i - 1]
        ratio_lb = tlt_shy_ratio[i - 1 - lookback_d]
        sma_val = sma(tlt_shy_ratio, i - 1, sma_period)

        if ratio_now <= 0 or ratio_lb <= 0 or sma_val <= 0:
            strat_rets.append(0.0)
            continue

        mom = ratio_now / ratio_lb - 1
        above_sma = ratio_now > sma_val

        if mom > 0 and above_sma:
            # Bull steepening / rates falling -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        elif mom < 0 and not above_sma:
            # Bear flattening / rates rising -> defensive
            strat_rets.append(asset_ret("SHY", i) * 0.60 + asset_ret("GLD", i) * 0.30)
        else:
            # Mixed -> moderate
            strat_rets.append(
                spy_rets[i] * 0.50
                + asset_ret("TLT", i) * 0.20
                + asset_ret("SHY", i) * 0.20
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.4", f"lb={lookback_d} sma={sma_period}", s, dd, tr)


# ============================================================
# H15.5: Multi-Asset Volatility Regime (SPY vol vs GLD vol)
# When SPY vol > GLD vol -> equity stress -> tilt GLD
# When GLD vol > SPY vol -> commodity stress -> tilt SPY
# Relative volatility as regime signal — not level-based
# ============================================================
print("\n" + "=" * 70)
print("H15.5: Multi-Asset Volatility Regime (SPY vol vs GLD vol)")
print("Mechanism: relative realized vol as cross-asset regime signal")
print("SPY vol > GLD vol -> equity stress -> tilt GLD")
print("GLD vol > SPY vol -> commodity stress -> tilt SPY")
print("=" * 70)

for vol_window in [10, 20, 30]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < vol_window + 1:
            strat_rets.append(0.0)
            continue

        # Realized vol at i-1 (lag-1)
        spy_vol = realized_vol(ret_lists["SPY"], i - 1, vol_window)
        gld_vol = realized_vol(ret_lists["GLD"], i - 1, vol_window)

        if spy_vol <= 0 or gld_vol <= 0:
            strat_rets.append(0.0)
            continue

        vol_ratio = spy_vol / gld_vol

        if vol_ratio > 1.5:
            # SPY much more volatile -> equity stress -> heavy GLD
            strat_rets.append(asset_ret("GLD", i) * 0.60 + asset_ret("SHY", i) * 0.30)
        elif vol_ratio > 1.0:
            # SPY slightly more volatile -> mild equity stress -> balanced
            strat_rets.append(
                spy_rets[i] * 0.45
                + asset_ret("GLD", i) * 0.30
                + asset_ret("SHY", i) * 0.15
            )
        elif vol_ratio > 0.67:
            # Roughly equal vol -> normal -> SPY tilt
            strat_rets.append(spy_rets[i] * 0.70 + asset_ret("GLD", i) * 0.20)
        else:
            # GLD much more volatile -> commodity stress -> full SPY
            strat_rets.append(spy_rets[i] * 0.90)

    s, dd, tr = metrics(strat_rets)
    record("H15.5a", f"vol_w={vol_window} (4-tier)", s, dd, tr)

# Also test simple binary version with various thresholds
for vol_window, threshold in [
    (10, 1.0),
    (10, 1.2),
    (20, 1.0),
    (20, 1.2),
    (30, 1.0),
    (30, 1.2),
]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < vol_window + 1:
            strat_rets.append(0.0)
            continue

        spy_vol = realized_vol(ret_lists["SPY"], i - 1, vol_window)
        gld_vol = realized_vol(ret_lists["GLD"], i - 1, vol_window)

        if spy_vol <= 0 or gld_vol <= 0:
            strat_rets.append(0.0)
            continue

        vol_ratio = spy_vol / gld_vol

        if vol_ratio > threshold:
            # SPY more volatile -> tilt GLD
            strat_rets.append(
                asset_ret("GLD", i) * 0.50
                + spy_rets[i] * 0.30
                + asset_ret("SHY", i) * 0.10
            )
        else:
            # GLD more volatile or equal -> tilt SPY
            strat_rets.append(spy_rets[i] * 0.80 + asset_ret("GLD", i) * 0.10)

    s, dd, tr = metrics(strat_rets)
    record("H15.5b", f"vol_w={vol_window} thresh={threshold:.1f} (binary)", s, dd, tr)


# ============================================================
# H15.6: Cross-Market Momentum Divergence (macro breadth)
# Count how many of SPY, TLT, GLD, DBA have positive momentum
# >= 3 positive -> coordinated growth -> full SPY
# <= 1 positive -> broad weakness -> defensive SHY + GLD
# 2 positive -> uncertain -> moderate
# ============================================================
print("\n" + "=" * 70)
print("H15.6: Cross-Market Momentum Divergence (macro breadth)")
print("Mechanism: multi-asset momentum breadth as macro regime signal")
print("Count assets with positive momentum: SPY, TLT, GLD, DBA")
print(">= 3 positive -> coordinated growth -> SPY")
print("<= 1 positive -> broad weakness -> SHY+GLD defensive")
print("=" * 70)

breadth_assets = ["SPY", "TLT", "GLD", "DBA"]

for lookback_d in [10, 20, 40, 60]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        # Count positive momentum at i-1 (lag-1)
        pos_count = 0
        valid_count = 0
        for sym in breadth_assets:
            p_now = price_lists[sym][i - 1]
            p_lb = price_lists[sym][i - 1 - lookback_d]
            if p_now > 0 and p_lb > 0:
                valid_count += 1
                if p_now > p_lb:
                    pos_count += 1

        if valid_count < 3:
            strat_rets.append(0.0)
            continue

        if pos_count >= 3:
            # Broad positive momentum -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        elif pos_count <= 1:
            # Broad weakness -> defensive
            strat_rets.append(
                asset_ret("SHY", i) * 0.50
                + asset_ret("GLD", i) * 0.30
                + spy_rets[i] * 0.10
            )
        else:
            # 2 positive -> uncertain -> moderate
            strat_rets.append(
                spy_rets[i] * 0.45
                + asset_ret("GLD", i) * 0.25
                + asset_ret("SHY", i) * 0.20
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.6", f"lb={lookback_d}", s, dd, tr)

# Also test with 5 assets (add IEF for bond breadth)
breadth_assets_5 = ["SPY", "TLT", "GLD", "DBA", "IEF"]
print("  -- 5-asset variant (add IEF) --")
for lookback_d in [10, 20, 40, 60]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        pos_count = 0
        valid_count = 0
        for sym in breadth_assets_5:
            p_now = price_lists[sym][i - 1]
            p_lb = price_lists[sym][i - 1 - lookback_d]
            if p_now > 0 and p_lb > 0:
                valid_count += 1
                if p_now > p_lb:
                    pos_count += 1

        if valid_count < 3:
            strat_rets.append(0.0)
            continue

        # Thresholds adjusted for 5 assets
        if pos_count >= 4:
            strat_rets.append(spy_rets[i] * 0.90)
        elif pos_count <= 1:
            strat_rets.append(
                asset_ret("SHY", i) * 0.50
                + asset_ret("GLD", i) * 0.30
                + spy_rets[i] * 0.10
            )
        else:
            strat_rets.append(
                spy_rets[i] * 0.45
                + asset_ret("GLD", i) * 0.25
                + asset_ret("SHY", i) * 0.20
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.6-5a", f"lb={lookback_d} (5-asset)", s, dd, tr)

# Also test with EEM + EFA for global breadth
breadth_assets_global = ["SPY", "TLT", "GLD", "EEM", "EFA"]
print("  -- Global breadth variant (SPY, TLT, GLD, EEM, EFA) --")
for lookback_d in [10, 20, 40, 60]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        pos_count = 0
        valid_count = 0
        for sym in breadth_assets_global:
            p_now = price_lists[sym][i - 1]
            p_lb = price_lists[sym][i - 1 - lookback_d]
            if p_now > 0 and p_lb > 0:
                valid_count += 1
                if p_now > p_lb:
                    pos_count += 1

        if valid_count < 3:
            strat_rets.append(0.0)
            continue

        if pos_count >= 4:
            strat_rets.append(spy_rets[i] * 0.90)
        elif pos_count <= 1:
            strat_rets.append(
                asset_ret("SHY", i) * 0.50
                + asset_ret("GLD", i) * 0.30
                + spy_rets[i] * 0.10
            )
        else:
            strat_rets.append(
                spy_rets[i] * 0.45
                + asset_ret("GLD", i) * 0.25
                + asset_ret("SHY", i) * 0.20
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.6-global", f"lb={lookback_d} (global)", s, dd, tr)


# ============================================================
# H15.7: EEM Macro Regime (EM absolute momentum as risk proxy)
# EEM rising -> global risk appetite increasing -> equities
# EEM falling -> global risk-off -> defensive
# Different from EFA/SPY relative — this is ABSOLUTE EM momentum
# ============================================================
print("\n" + "=" * 70)
print("H15.7: EEM Macro Regime (EM absolute momentum)")
print("Mechanism: emerging market momentum as global risk appetite proxy")
print("EEM rising -> global risk-on -> SPY")
print("EEM falling -> global risk-off -> SHY+GLD")
print("=" * 70)

for lookback_d in [20, 40, 60]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        eem_now = price_lists["EEM"][i - 1]
        eem_lb = price_lists["EEM"][i - 1 - lookback_d]

        if eem_now <= 0 or eem_lb <= 0:
            strat_rets.append(0.0)
            continue

        eem_mom = eem_now / eem_lb - 1

        if eem_mom > 0:
            # EM positive momentum -> global risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            # EM negative momentum -> global risk-off
            strat_rets.append(
                asset_ret("SHY", i) * 0.50
                + asset_ret("GLD", i) * 0.30
                + spy_rets[i] * 0.10
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.7a", f"lb={lookback_d} (binary)", s, dd, tr)

# Test with SMA confirmation for EEM
for lookback_d, sma_period in [
    (20, 50),
    (20, 100),
    (40, 50),
    (40, 100),
    (60, 100),
]:
    strat_rets = []
    required = max(lookback_d, sma_period) + 1

    for i in range(WARMUP, n):
        if i < required:
            strat_rets.append(0.0)
            continue

        eem_now = price_lists["EEM"][i - 1]
        eem_lb = price_lists["EEM"][i - 1 - lookback_d]
        eem_sma = sma(price_lists["EEM"], i - 1, sma_period)

        if eem_now <= 0 or eem_lb <= 0 or eem_sma <= 0:
            strat_rets.append(0.0)
            continue

        eem_mom = eem_now / eem_lb - 1
        above_sma = eem_now > eem_sma

        if eem_mom > 0 and above_sma:
            # Strong EM momentum -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        elif eem_mom < 0 and not above_sma:
            # Weak EM momentum -> defensive
            strat_rets.append(
                asset_ret("SHY", i) * 0.50
                + asset_ret("GLD", i) * 0.30
                + spy_rets[i] * 0.10
            )
        else:
            # Mixed -> moderate
            strat_rets.append(
                spy_rets[i] * 0.50
                + asset_ret("SHY", i) * 0.25
                + asset_ret("GLD", i) * 0.15
            )

    s, dd, tr = metrics(strat_rets)
    record("H15.7b", f"lb={lookback_d} sma={sma_period} (confirmed)", s, dd, tr)


# ============================================================
# BENCHMARKS
# ============================================================
print("\n" + "=" * 70)
print("BENCHMARKS")
print("=" * 70)

bh_rets = spy_rets[WARMUP:]
s, dd, tr = metrics(bh_rets)
print(f"  SPY buy-hold:  Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

bench_rets = []
for i in range(WARMUP, n):
    bench_rets.append(spy_rets[i] * 0.60 + asset_ret("TLT", i) * 0.40)
s, dd, tr = metrics(bench_rets)
print(f"  60/40 SPY/TLT: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

# GLD buy-hold for reference
gld_rets = ret_lists["GLD"][WARMUP:]
s, dd, tr = metrics(gld_rets)
print(f"  GLD buy-hold:  Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

# EFA buy-hold for reference
efa_rets = ret_lists["EFA"][WARMUP:]
s, dd, tr = metrics(efa_rets)
print(f"  EFA buy-hold:  Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")


# ============================================================
# SUMMARY — top configs passing both gates
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY — TOP CONFIGS")
print("=" * 70)

passing = [(h, c, s, dd, tr) for h, c, s, dd, tr in results if s > 0.80 and dd < 0.15]
passing.sort(key=lambda x: x[2], reverse=True)

if passing:
    print(f"\n  {len(passing)} configs pass BOTH gates (Sharpe > 0.80, MaxDD < 15%):\n")
    for h, c, s, dd, tr in passing:
        print(f"    {h} {c}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")
else:
    print("\n  NO configs pass both gates.\n")

# Show near-misses (Sharpe > 0.65 or MaxDD < 12%)
near = [
    (h, c, s, dd, tr)
    for h, c, s, dd, tr in results
    if (s > 0.65 and dd < 0.18) and not (s > 0.80 and dd < 0.15)
]
near.sort(key=lambda x: x[2], reverse=True)

if near:
    print(f"\n  Near-misses ({len(near)} configs with Sharpe > 0.65, MaxDD < 18%):\n")
    for h, c, s, dd, tr in near[:15]:
        print(f"    {h} {c}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

# Show all results sorted by Sharpe
print(f"\n  All {len(results)} configs ranked by Sharpe:\n")
all_sorted = sorted(results, key=lambda x: x[2], reverse=True)
for h, c, s, dd, tr in all_sorted[:20]:
    tag = " ***" if s > 0.80 and dd < 0.15 else ""
    print(f"    {h} {c}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}")

print("\n" + "=" * 70)
print("CROSS-ASSET SCAN V5 COMPLETE")
print("Target: Sharpe > 0.80, MaxDD < 15%")
print("*** marks configs passing both gates")
print("=" * 70)

#!/usr/bin/env python3
"""Cross-asset timing signals scan v4 — low-correlation hypothesis hunt.

Targeting mechanisms UNCORRELATED with existing portfolio
(credit-equity, rate-equity, TSMOM families).

H14.1: Copper/Gold Ratio Regime (DBA/GLD as proxy for Dr. Copper)
H14.2: High-Yield vs Investment-Grade Spread (HYG/LQD credit quality rotation)
H14.3: International vs Domestic Momentum (EFA/SPY ratio)
H14.4: TLT Volatility Regime (bond vol as equity signal)
H14.5: Commodity Cycle Signal (DBA absolute momentum as inflation timing)
H14.6: Cross-Volatility Regime (equity vol vs bond vol)

All signals use lag-1 (close[i-1] for decisions at day i).
Lookback: 5*365 days. Warmup: 60 days.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/scan_crossasset_v4.py
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
    "HYG",
    "LQD",
    "EFA",
    "EEM",
    "IEF",
    "DBA",
    "VIX",
    "IWM",
    "TIP",
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


def asset_ret(sym: str, i: int) -> float:
    d = dates[i]
    d_prev = dates[i - 1]
    data = sym_data[sym]
    if d in data and d_prev in data and data[d_prev] > 0:
        return data[d] / data[d_prev] - 1
    return 0.0


def get_close(sym: str, i: int) -> float:
    d = dates[i]
    return sym_data[sym].get(d, 0.0)


def metrics(daily_rets: list[float]) -> tuple[float, float, float]:
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


# Pre-compute VIX series for H14.6
vix_series: list[float] = []
for i in range(n):
    vix_series.append(get_close("VIX", i))


# ============================================================
# H14.1: Copper/Gold Ratio Regime (DBA/GLD proxy)
# Dr. Copper thesis: industrial commodities outperforming gold -> growth
# DBA = diversified commodity basket (includes copper, ag, energy)
# When DBA/GLD rising -> macro growth -> risk-on SPY
# When DBA/GLD falling -> safety bid -> risk-off SHY
# ============================================================
print("\n" + "=" * 60)
print("H14.1: Copper/Gold Ratio Regime (DBA/GLD)")
print("Mechanism: industrial commodity vs safe-haven strength")
print("=" * 60)

for ratio_lb, sma_period in [
    (10, 20),
    (10, 50),
    (20, 20),
    (20, 50),
]:
    strat_rets: list[float] = []

    # Pre-compute DBA/GLD ratio
    ratio_series: list[float] = []
    for i in range(n):
        dba = get_close("DBA", i)
        gld = get_close("GLD", i)
        if dba > 0 and gld > 0:
            ratio_series.append(dba / gld)
        else:
            ratio_series.append(0.0)

    for i in range(WARMUP, n):
        if i < max(ratio_lb, sma_period) + 1:
            strat_rets.append(0.0)
            continue

        # Signal at close i-1 (lag-1)
        ratio_now = ratio_series[i - 1]
        ratio_prev = ratio_series[i - 1 - ratio_lb]
        sma_window = ratio_series[i - 1 - sma_period : i - 1]

        if ratio_now <= 0 or ratio_prev <= 0 or len(sma_window) < sma_period:
            strat_rets.append(0.0)
            continue

        sma_val = sum(sma_window) / sma_period
        if sma_val <= 0:
            strat_rets.append(0.0)
            continue

        mom = ratio_now / ratio_prev - 1
        vs_sma = ratio_now / sma_val - 1

        # Growth regime: DBA outperforming GLD and above trend
        if mom > 0 and vs_sma > 0:
            strat_rets.append(spy_rets[i] * 0.90)
        # Safety regime: GLD outperforming DBA and below trend
        elif mom < 0 and vs_sma < 0:
            strat_rets.append(asset_ret("SHY", i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  ratio_lb={ratio_lb} sma={sma_period}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )


# ============================================================
# H14.2: High-Yield vs Investment-Grade Spread (HYG/LQD)
# Credit quality rotation: HY outperforming IG -> risk appetite rising
# Different from H13.1 (HYG/SHY) -- this isolates CREDIT QUALITY
# premium, stripping out duration component
# ============================================================
print("\n" + "=" * 60)
print("H14.2: HY vs IG Credit Quality Rotation (HYG/LQD)")
print("Mechanism: credit quality spread momentum")
print("=" * 60)

for lookback_d, threshold in [
    (5, 0.000),
    (5, 0.005),
    (10, 0.000),
    (10, 0.005),
    (20, 0.000),
    (20, 0.005),
    (20, 0.010),
]:
    strat_rets = []

    # Pre-compute HYG/LQD ratio
    ratio_series = []
    for i in range(n):
        hyg = get_close("HYG", i)
        lqd = get_close("LQD", i)
        if hyg > 0 and lqd > 0:
            ratio_series.append(hyg / lqd)
        else:
            ratio_series.append(0.0)

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        ratio_now = ratio_series[i - 1]
        ratio_lb = ratio_series[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            continue

        ratio_mom = ratio_now / ratio_lb - 1

        # HY outperforming IG -> risk appetite rising -> SPY
        if ratio_mom > threshold:
            strat_rets.append(spy_rets[i] * 0.90)
        # HY underperforming IG -> risk aversion -> SHY
        elif ratio_mom < -threshold:
            strat_rets.append(asset_ret("SHY", i) * 0.90)
        else:
            strat_rets.append(spy_rets[i] * 0.45)

    s, dd, tr = metrics(strat_rets)
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  lb={lookback_d} thresh={threshold:.3f}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )


# ============================================================
# H14.3: International vs Domestic Momentum (EFA/SPY)
# When EFA outperforming SPY -> international cycle leading
# Trade the leader (EFA or SPY) based on relative momentum
# ============================================================
print("\n" + "=" * 60)
print("H14.3: International vs Domestic Momentum (EFA/SPY)")
print("Mechanism: international/domestic relative strength rotation")
print("=" * 60)

for lookback_d, sma_filter in [
    (10, 0),
    (10, 20),
    (20, 0),
    (20, 40),
    (40, 0),
    (40, 60),
]:
    strat_rets = []

    # Pre-compute EFA/SPY ratio
    ratio_series = []
    for i in range(n):
        efa = get_close("EFA", i)
        spy_val = spy_close[i] if i < n else 0.0
        if efa > 0 and spy_val > 0:
            ratio_series.append(efa / spy_val)
        else:
            ratio_series.append(0.0)

    required_history = (
        max(lookback_d, sma_filter) + 1 if sma_filter > 0 else lookback_d + 1
    )

    for i in range(WARMUP, n):
        if i < required_history:
            strat_rets.append(0.0)
            continue

        ratio_now = ratio_series[i - 1]
        ratio_lb = ratio_series[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            continue

        ratio_mom = ratio_now / ratio_lb - 1

        # Optional SMA filter on ratio
        sma_ok = True
        if sma_filter > 0:
            sma_window = ratio_series[i - 1 - sma_filter : i - 1]
            valid = [x for x in sma_window if x > 0]
            if len(valid) < sma_filter // 2:
                strat_rets.append(0.0)
                continue
            sma_val = sum(valid) / len(valid)
            sma_ok = ratio_now > sma_val if ratio_mom > 0 else ratio_now < sma_val

        if ratio_mom > 0 and sma_ok:
            # EFA outperforming -> trade EFA
            strat_rets.append(asset_ret("EFA", i) * 0.90)
        elif ratio_mom < 0 and sma_ok:
            # SPY outperforming -> stay in SPY
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            # Mixed signal -> split
            strat_rets.append(spy_rets[i] * 0.45 + asset_ret("EFA", i) * 0.45)

    s, dd, tr = metrics(strat_rets)
    sma_str = f" sma={sma_filter}" if sma_filter > 0 else ""
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  lb={lookback_d}{sma_str}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )


# ============================================================
# H14.4: TLT Volatility Regime (bond vol as equity signal)
# When TLT realized vol is high -> rate uncertainty -> equity defensive
# When TLT vol is low -> stable rates -> equity risk-on
# Uses percentile ranking against rolling history
# ============================================================
print("\n" + "=" * 60)
print("H14.4: TLT Volatility Regime (bond vol -> equity signal)")
print("Mechanism: rate volatility as equity regime indicator")
print("=" * 60)

# Pre-compute TLT daily returns
tlt_rets: list[float] = [0.0]
for i in range(1, n):
    tlt_rets.append(asset_ret("TLT", i))

for vol_window, pct_thresh, hist_window in [
    (10, 60, 252),
    (10, 70, 252),
    (10, 80, 252),
    (20, 60, 252),
    (20, 70, 252),
    (20, 80, 252),
    (20, 70, 504),
]:
    strat_rets = []

    required_history = hist_window + vol_window + 1

    for i in range(WARMUP, n):
        if i < required_history:
            strat_rets.append(0.0)
            continue

        # Compute current TLT realized vol at i-1 (lag-1)
        window_rets = tlt_rets[i - vol_window : i]
        if len(window_rets) < vol_window:
            strat_rets.append(0.0)
            continue
        mean_r = sum(window_rets) / vol_window
        var_r = sum((r - mean_r) ** 2 for r in window_rets) / vol_window
        current_vol = var_r**0.5

        # Compute historical vol distribution for percentile ranking
        hist_vols: list[float] = []
        for j in range(i - hist_window, i):
            if j < vol_window:
                continue
            w_rets = tlt_rets[j - vol_window + 1 : j + 1]
            if len(w_rets) < vol_window:
                continue
            m = sum(w_rets) / vol_window
            v = sum((r - m) ** 2 for r in w_rets) / vol_window
            hist_vols.append(v**0.5)

        if len(hist_vols) < 50:
            strat_rets.append(0.0)
            continue

        # Percentile rank
        pct_rank = (
            sum(1 for hv in hist_vols if hv <= current_vol) / len(hist_vols) * 100
        )

        if pct_rank >= pct_thresh:
            # High bond vol -> defensive
            strat_rets.append(asset_ret("SHY", i) * 0.90)
        else:
            # Normal/low bond vol -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)

    s, dd, tr = metrics(strat_rets)
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  vol_w={vol_window} pct={pct_thresh} hist={hist_window}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )


# ============================================================
# H14.5: Commodity Cycle Signal (DBA absolute momentum)
# DBA rising -> inflation heating up -> GLD allocation
# DBA falling -> disinflation -> full SPY
# NOT relative to SPY -- pure commodity cycle
# ============================================================
print("\n" + "=" * 60)
print("H14.5: Commodity Cycle Signal (DBA absolute momentum)")
print("Mechanism: commodity cycle as inflation timing indicator")
print("=" * 60)

for lookback_d, risk_on_alloc in [
    (20, "spy_full"),
    (20, "spy_gld"),
    (40, "spy_full"),
    (40, "spy_gld"),
    (60, "spy_full"),
    (60, "spy_gld"),
]:
    strat_rets = []

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            continue

        dba_now = get_close("DBA", i - 1)
        dba_lb = get_close("DBA", i - 1 - lookback_d)

        if dba_now <= 0 or dba_lb <= 0:
            strat_rets.append(0.0)
            continue

        dba_mom = dba_now / dba_lb - 1

        if dba_mom > 0:
            # DBA rising -> inflation heating -> tilt GLD, reduce SPY
            strat_rets.append(asset_ret("GLD", i) * 0.50 + spy_rets[i] * 0.30)
        elif risk_on_alloc == "spy_full":
            # DBA falling -> disinflation -> full equity
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            # DBA falling -> disinflation -> equity + gold hedge
            strat_rets.append(spy_rets[i] * 0.70 + asset_ret("GLD", i) * 0.20)

    s, dd, tr = metrics(strat_rets)
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  lb={lookback_d} alloc={risk_on_alloc}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )

# Also test SMA crossover variant for DBA
print("  -- DBA SMA crossover variant --")
for fast_sma, slow_sma in [(10, 40), (20, 60), (10, 20)]:
    strat_rets = []

    # Pre-compute DBA SMA series
    dba_prices: list[float] = [get_close("DBA", i) for i in range(n)]

    for i in range(WARMUP, n):
        if i < slow_sma + 1:
            strat_rets.append(0.0)
            continue

        # SMA at i-1 (lag-1)
        fast_window = dba_prices[i - fast_sma : i]
        slow_window = dba_prices[i - slow_sma : i]

        fast_valid = [p for p in fast_window if p > 0]
        slow_valid = [p for p in slow_window if p > 0]

        if len(fast_valid) < fast_sma // 2 or len(slow_valid) < slow_sma // 2:
            strat_rets.append(0.0)
            continue

        fast_avg = sum(fast_valid) / len(fast_valid)
        slow_avg = sum(slow_valid) / len(slow_valid)

        if fast_avg > slow_avg:
            # DBA trending up -> inflation -> GLD tilt
            strat_rets.append(asset_ret("GLD", i) * 0.50 + spy_rets[i] * 0.30)
        else:
            # DBA trending down -> disinflation -> SPY
            strat_rets.append(spy_rets[i] * 0.90)

    s, dd, tr = metrics(strat_rets)
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  fast={fast_sma} slow={slow_sma}: "
        f"Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )


# ============================================================
# H14.6: Cross-Volatility Regime (equity vol vs bond vol)
# Uses VIX for equity vol, TLT realized vol as MOVE proxy
# Quadrant approach:
#   High VIX + Low TLT vol -> equity correction (buy dip)
#   High VIX + High TLT vol -> systemic risk (defensive)
#   Low VIX + Low TLT vol -> Goldilocks (risk-on)
#   Low VIX + High TLT vol -> rate shock (cautious)
# ============================================================
print("\n" + "=" * 60)
print("H14.6: Cross-Volatility Regime (VIX vs TLT vol)")
print("Mechanism: equity vol / bond vol quadrant analysis")
print("=" * 60)

for vol_window, vix_pct_thresh, tlt_pct_thresh, hist_window in [
    (20, 60, 60, 252),
    (20, 70, 70, 252),
    (20, 50, 50, 252),
    (20, 60, 60, 504),
    (10, 60, 60, 252),
]:
    strat_rets = []

    required_history = hist_window + vol_window + 1

    for i in range(WARMUP, n):
        if i < required_history:
            strat_rets.append(0.0)
            continue

        # Current VIX level at i-1 (lag-1)
        current_vix = vix_series[i - 1]
        if current_vix <= 0:
            strat_rets.append(0.0)
            continue

        # Historical VIX for percentile ranking
        hist_vix_vals = [
            vix_series[j] for j in range(i - hist_window, i) if vix_series[j] > 0
        ]
        if len(hist_vix_vals) < 50:
            strat_rets.append(0.0)
            continue
        vix_pct = (
            sum(1 for v in hist_vix_vals if v <= current_vix) / len(hist_vix_vals) * 100
        )

        # Current TLT realized vol at i-1
        window_rets = tlt_rets[i - vol_window : i]
        if len(window_rets) < vol_window:
            strat_rets.append(0.0)
            continue
        mean_r = sum(window_rets) / vol_window
        var_r = sum((r - mean_r) ** 2 for r in window_rets) / vol_window
        current_tlt_vol = var_r**0.5

        # Historical TLT vol for percentile ranking
        hist_tlt_vols: list[float] = []
        for j in range(i - hist_window, i):
            if j < vol_window:
                continue
            w_rets = tlt_rets[j - vol_window + 1 : j + 1]
            if len(w_rets) < vol_window:
                continue
            m = sum(w_rets) / vol_window
            v = sum((r - m) ** 2 for r in w_rets) / vol_window
            hist_tlt_vols.append(v**0.5)

        if len(hist_tlt_vols) < 50:
            strat_rets.append(0.0)
            continue
        tlt_pct = (
            sum(1 for tv in hist_tlt_vols if tv <= current_tlt_vol)
            / len(hist_tlt_vols)
            * 100
        )

        high_vix = vix_pct >= vix_pct_thresh
        high_tlt_vol = tlt_pct >= tlt_pct_thresh

        if high_vix and not high_tlt_vol:
            # Equity-specific correction -> buy the dip
            strat_rets.append(spy_rets[i] * 0.90)
        elif high_vix and high_tlt_vol:
            # Systemic risk -> full defensive
            strat_rets.append(asset_ret("SHY", i) * 0.60 + asset_ret("GLD", i) * 0.30)
        elif not high_vix and not high_tlt_vol:
            # Goldilocks -> risk-on
            strat_rets.append(spy_rets[i] * 0.90)
        else:
            # Low VIX + High TLT vol -> rate shock -> cautious
            strat_rets.append(spy_rets[i] * 0.50 + asset_ret("TLT", i) * 0.30)

    s, dd, tr = metrics(strat_rets)
    tag = " *** " if s > 0.80 and dd < 0.15 else ""
    print(
        f"  vol_w={vol_window} vix_pct={vix_pct_thresh} tlt_pct={tlt_pct_thresh} "
        f"hist={hist_window}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}{tag}"
    )


# ============================================================
# BENCHMARK COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("BENCHMARKS")
print("=" * 60)

# Buy-and-hold SPY
bh_rets = spy_rets[WARMUP:]
s, dd, tr = metrics(bh_rets)
print(f"  SPY buy-hold: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

# 60/40 SPY/TLT
bench_rets = []
for i in range(WARMUP, n):
    bench_rets.append(spy_rets[i] * 0.60 + asset_ret("TLT", i) * 0.40)
s, dd, tr = metrics(bench_rets)
print(f"  60/40 SPY/TLT: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")


print("\n" + "=" * 60)
print("CROSS-ASSET SCAN V4 COMPLETE")
print("Target: Sharpe > 0.80, MaxDD < 15%")
print("*** marks configs passing both gates")
print("=" * 60)

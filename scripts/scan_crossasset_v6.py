#!/usr/bin/env python3
"""Cross-asset timing signals scan v6 — novel mechanism hypothesis hunt.

Targeting mechanisms GENUINELY UNCORRELATED with all existing portfolio families:
  F1:  Credit-equity lead-lag (AGG/LQD/HYG/EMB/VCIT -> SPY/QQQ/EFA)
  F2:  GLD-SLV mean reversion (consensus windows)
  F3:  Skip-month TSMOM (vol-scaled multi-asset)
  F5:  SPY overnight momentum
  F6:  Rate-tech (TLT/IEF -> QQQ)
  F7:  VIX spike contrarian behavioral
  F8:  SOXX-QQQ lead-lag
  F9:  Credit spread regime (HYG/SHY ratio momentum)
  F11: DBA commodity cycle
  F12: XLK/XLE sector rotation

NEW hypotheses — novel information channels:

H16.1: Dollar Strength Equity Timing (UUP momentum -> IWM vs EFA)
H16.2: Treasury Curve Steepening/Flattening (TLT/SHY ratio momentum -> SPY vs GLD)
H16.3: Copper/Gold Ratio as Growth Signal (DBA/GLD ratio momentum -> QQQ vs TLT+GLD)
H16.4: High Yield Spread Velocity (HYG/LQD 5d ROC -> QQQ vs SHY)
H16.5: Real Yield Proxy (TIP/TLT ratio momentum -> SPY vs GLD+SHY)
H16.6: Emerging Market Currency Stress (EEM/SPY ratio momentum -> SPY vs EEM)
H16.7: Utilities/Tech Ratio as Risk Barometer (XLU/XLK ratio momentum -> QQQ vs TLT+GLD)

All signals use lag-1 (close[i-1] for decisions at day i).
Lookback: 5*365 days. Warmup: 60 days.
Transaction cost: 3 bps per switch.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/scan_crossasset_v6.py
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
    "SHY",
    "GLD",
    "EFA",
    "IWM",
    "EEM",
    "DBA",
    "HYG",
    "LQD",
    "TIP",
    "XLU",
    "XLK",
    "UUP",
]
LOOKBACK = 5 * 365
WARMUP = 60
TC_BPS = 3  # 3 bps transaction cost per switch

print("=" * 70)
print("CROSS-ASSET HYPOTHESIS SCAN V6 — NOVEL MECHANISM HUNT")
print("=" * 70)
print()

print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK)
print(f"Data: {len(prices)} rows, {len(prices['symbol'].unique())} symbols")

# Check which symbols we actually got
available_symbols = prices["symbol"].unique().to_list()
missing = [s for s in SYMBOLS if s not in available_symbols]
if missing:
    print(f"WARNING: Missing symbols: {missing}")
    print("Strategies using missing symbols will be skipped.")

spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Build price lookup by symbol
sym_data: dict[str, dict] = {}
for sym in available_symbols:
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
    if sym not in sym_data:
        return 0.0
    d = dates[i]
    d_prev = dates[i - 1]
    data = sym_data[sym]
    if d in data and d_prev in data and data[d_prev] > 0:
        return data[d] / data[d_prev] - 1
    return 0.0


def get_close(sym: str, i: int) -> float:
    """Close price for symbol on day i."""
    if sym not in sym_data:
        return 0.0
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


def count_switches(regimes: list[int]) -> int:
    """Count regime switches."""
    switches = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            switches += 1
    return switches


# Pre-compute price series as lists for fast access
price_lists: dict[str, list[float]] = {}
for sym in available_symbols:
    price_lists[sym] = [get_close(sym, i) for i in range(n)]

# Pre-compute daily returns for all symbols
ret_lists: dict[str, list[float]] = {}
for sym in available_symbols:
    rets = [0.0]
    for i in range(1, n):
        rets.append(asset_ret(sym, i))
    ret_lists[sym] = rets


# Pre-compute ratio series
def build_ratio(sym_a: str, sym_b: str) -> list[float]:
    """Build price ratio series sym_a / sym_b."""
    ratio = []
    for i in range(n):
        a = price_lists.get(sym_a, [0.0] * n)[i] if sym_a in price_lists else 0.0
        b = price_lists.get(sym_b, [0.0] * n)[i] if sym_b in price_lists else 0.0
        if a > 0 and b > 0:
            ratio.append(a / b)
        else:
            ratio.append(0.0)
    return ratio


tlt_shy_ratio = build_ratio("TLT", "SHY")
dba_gld_ratio = build_ratio("DBA", "GLD")
hyg_lqd_ratio = build_ratio("HYG", "LQD")
tip_tlt_ratio = build_ratio("TIP", "TLT")
eem_spy_ratio = build_ratio("EEM", "SPY")
xlu_xlk_ratio = build_ratio("XLU", "XLK")

results: list[tuple[str, str, float, float, float, int]] = []


def record(hyp: str, config: str, s: float, dd: float, tr: float, sw: int) -> None:
    """Record a result and print it."""
    tag = " *** PASS ***" if s > 0.80 and dd < 0.15 else ""
    print(
        f"  {config}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f} Switches={sw}{tag}"
    )
    results.append((hyp, config, s, dd, tr, sw))


# ============================================================
# H16.1: Dollar Strength Equity Timing
# Signal: UUP (dollar ETF) 20-day momentum
# Strong dollar -> headwind for multinationals -> underweight EFA, overweight IWM
# When UUP 20d return > 0: 60% IWM + 30% SPY; when < 0: 70% EFA + 20% SPY
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.1: Dollar Strength Equity Timing (UUP momentum)")
print("Mechanism: strong USD -> headwind for multinationals -> domestic small caps")
print("UUP rising -> 60% IWM + 30% SPY; UUP falling -> 70% EFA + 20% SPY")
print("=" * 70)

if "UUP" in available_symbols:
    for lookback_d in [10, 15, 20, 30, 40]:
        strat_rets: list[float] = []
        regimes: list[int] = []
        prev_regime = -1

        for i in range(WARMUP, n):
            if i < lookback_d + 1:
                strat_rets.append(0.0)
                regimes.append(0)
                continue

            # Lag-1: use data at i-1
            uup_now = price_lists["UUP"][i - 1]
            uup_lb = price_lists["UUP"][i - 1 - lookback_d]

            if uup_now <= 0 or uup_lb <= 0:
                strat_rets.append(0.0)
                regimes.append(max(prev_regime, 0))
                continue

            uup_mom = uup_now / uup_lb - 1

            if uup_mom > 0:
                # Strong dollar -> domestic small caps
                regime = 1
                port_ret = asset_ret("IWM", i) * 0.60 + spy_rets[i] * 0.30
            else:
                # Weak dollar -> international
                regime = 2
                port_ret = asset_ret("EFA", i) * 0.70 + spy_rets[i] * 0.20

            # Apply transaction cost on switch
            if prev_regime >= 0 and regime != prev_regime:
                port_ret -= TC_BPS / 10000.0
            prev_regime = regime
            regimes.append(regime)
            strat_rets.append(port_ret)

        s, dd, tr = metrics(strat_rets)
        sw = count_switches(regimes)
        record("H16.1", f"lb={lookback_d}", s, dd, tr, sw)
else:
    print("  SKIPPED — UUP not available")


# ============================================================
# H16.2: Treasury Curve Steepening/Flattening via TLT/SHY Ratio
# Signal: TLT/SHY price ratio 30-day momentum (ratio momentum, NOT yield level)
# Steepening (ratio falling) -> risk-off; Flattening (ratio rising) -> risk-on
# When ratio mom > 0 (flattening): 80% SPY; when < 0 (steepening): 80% GLD
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.2: Treasury Curve Steepening/Flattening (TLT/SHY ratio momentum)")
print("Mechanism: yield curve shape MOMENTUM as macro regime signal")
print("Ratio rising (flattening) -> 80% SPY; Ratio falling (steepening) -> 80% GLD")
print("=" * 70)

for lookback_d in [15, 20, 30, 40, 60]:
    strat_rets = []
    regimes = []
    prev_regime = -1

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            regimes.append(0)
            continue

        ratio_now = tlt_shy_ratio[i - 1]
        ratio_lb = tlt_shy_ratio[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            regimes.append(max(prev_regime, 0))
            continue

        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom > 0:
            # Flattening (long bonds rising vs short) -> risk-on
            regime = 1
            port_ret = spy_rets[i] * 0.80
        else:
            # Steepening (long bonds falling vs short) -> risk-off
            regime = 2
            port_ret = asset_ret("GLD", i) * 0.80

        if prev_regime >= 0 and regime != prev_regime:
            port_ret -= TC_BPS / 10000.0
        prev_regime = regime
        regimes.append(regime)
        strat_rets.append(port_ret)

    s, dd, tr = metrics(strat_rets)
    sw = count_switches(regimes)
    record("H16.2", f"lb={lookback_d}", s, dd, tr, sw)


# ============================================================
# H16.3: Copper/Gold Ratio as Growth Signal
# Signal: DBA/GLD ratio 40-day momentum (proxy for copper/gold via commodities)
# When DBA/GLD ratio mom > 0 (growth): 80% QQQ
# When < 0 (recession): 50% TLT + 30% GLD
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.3: Copper/Gold Ratio as Growth Signal (DBA/GLD ratio momentum)")
print("Mechanism: commodity/gold ratio momentum as growth vs recession signal")
print(
    "Ratio rising (growth) -> 80% QQQ; Ratio falling (recession) -> 50% TLT + 30% GLD"
)
print("=" * 70)

for lookback_d in [20, 30, 40, 50, 60]:
    strat_rets = []
    regimes = []
    prev_regime = -1

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            regimes.append(0)
            continue

        ratio_now = dba_gld_ratio[i - 1]
        ratio_lb = dba_gld_ratio[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            regimes.append(max(prev_regime, 0))
            continue

        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom > 0:
            # Growth signal -> tech/growth
            regime = 1
            port_ret = asset_ret("QQQ", i) * 0.80
        else:
            # Recession signal -> safe haven
            regime = 2
            port_ret = asset_ret("TLT", i) * 0.50 + asset_ret("GLD", i) * 0.30

        if prev_regime >= 0 and regime != prev_regime:
            port_ret -= TC_BPS / 10000.0
        prev_regime = regime
        regimes.append(regime)
        strat_rets.append(port_ret)

    s, dd, tr = metrics(strat_rets)
    sw = count_switches(regimes)
    record("H16.3", f"lb={lookback_d}", s, dd, tr, sw)


# ============================================================
# H16.4: High Yield Spread Velocity
# Signal: HYG/LQD ratio 5-day rate of change (velocity of credit compression)
# Rapid HYG outperformance of LQD = credit compression = risk-on momentum
# When HYG/LQD 5d ROC > +0.2%: 90% QQQ; when < -0.2%: 70% SHY
# Neutral zone (-0.2% to +0.2%): 50% SPY + 30% SHY
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.4: High Yield Spread Velocity (HYG/LQD 5d ROC)")
print("Mechanism: VELOCITY of credit spread compression/expansion")
print("Rapid HYG outperformance -> credit compression -> 90% QQQ")
print("Rapid HYG underperformance -> credit expansion -> 70% SHY")
print("=" * 70)

for lookback_d, threshold_pct in [
    (3, 0.10),
    (3, 0.20),
    (5, 0.10),
    (5, 0.20),
    (5, 0.30),
    (10, 0.10),
    (10, 0.20),
    (10, 0.30),
]:
    strat_rets = []
    regimes = []
    prev_regime = -1
    threshold = threshold_pct / 100.0

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            regimes.append(0)
            continue

        ratio_now = hyg_lqd_ratio[i - 1]
        ratio_lb = hyg_lqd_ratio[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            regimes.append(max(prev_regime, 0))
            continue

        roc = ratio_now / ratio_lb - 1  # rate of change

        if roc > threshold:
            # Credit compression -> risk-on momentum
            regime = 1
            port_ret = asset_ret("QQQ", i) * 0.90
        elif roc < -threshold:
            # Credit expansion -> risk-off
            regime = 2
            port_ret = asset_ret("SHY", i) * 0.70
        else:
            # Neutral zone
            regime = 0
            port_ret = spy_rets[i] * 0.50 + asset_ret("SHY", i) * 0.30

        if prev_regime >= 0 and regime != prev_regime:
            port_ret -= TC_BPS / 10000.0
        prev_regime = regime
        regimes.append(regime)
        strat_rets.append(port_ret)

    s, dd, tr = metrics(strat_rets)
    sw = count_switches(regimes)
    record("H16.4", f"lb={lookback_d} thresh={threshold_pct:.2f}%", s, dd, tr, sw)


# ============================================================
# H16.5: Real Yield Proxy (TIP/TLT Ratio Momentum)
# Signal: TIP/TLT price ratio 20-day momentum
# Rising TIP/TLT = real yields rising = tightening = risk-off
# Falling TIP/TLT = real yields falling = loosening = risk-on
# When ratio mom < 0 (real yields falling): 85% SPY
# When ratio mom > 0 (real yields rising): 60% GLD + 25% SHY
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.5: Real Yield Proxy (TIP/TLT Ratio Momentum)")
print("Mechanism: real yield direction via TIPS vs nominal bond ratio")
print(
    "TIP/TLT falling (easing) -> 85% SPY; TIP/TLT rising (tightening) -> 60% GLD + 25% SHY"
)
print("=" * 70)

if "TIP" in available_symbols:
    for lookback_d in [10, 15, 20, 30, 40]:
        strat_rets = []
        regimes = []
        prev_regime = -1

        for i in range(WARMUP, n):
            if i < lookback_d + 1:
                strat_rets.append(0.0)
                regimes.append(0)
                continue

            ratio_now = tip_tlt_ratio[i - 1]
            ratio_lb = tip_tlt_ratio[i - 1 - lookback_d]

            if ratio_now <= 0 or ratio_lb <= 0:
                strat_rets.append(0.0)
                regimes.append(max(prev_regime, 0))
                continue

            ratio_mom = ratio_now / ratio_lb - 1

            if ratio_mom < 0:
                # Real yields falling -> easing -> risk-on
                regime = 1
                port_ret = spy_rets[i] * 0.85
            else:
                # Real yields rising -> tightening -> risk-off
                regime = 2
                port_ret = asset_ret("GLD", i) * 0.60 + asset_ret("SHY", i) * 0.25

            if prev_regime >= 0 and regime != prev_regime:
                port_ret -= TC_BPS / 10000.0
            prev_regime = regime
            regimes.append(regime)
            strat_rets.append(port_ret)

        s, dd, tr = metrics(strat_rets)
        sw = count_switches(regimes)
        record("H16.5", f"lb={lookback_d}", s, dd, tr, sw)
else:
    print("  SKIPPED — TIP not available")


# ============================================================
# H16.6: Emerging Market Currency Stress
# Signal: EEM/SPY ratio 10-day momentum as proxy for EM stress
# EEM underperforming SPY = EM stress = flight to quality
# When EEM/SPY 10d return < -1%: 80% SPY + 10% GLD
# When EEM/SPY 10d return > +1%: 70% EEM
# Neutral: 50% SPY
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.6: Emerging Market Currency Stress (EEM/SPY ratio momentum)")
print("Mechanism: EM vs US relative performance as global stress indicator")
print("EM stress (EEM lagging) -> 80% SPY + 10% GLD; EM strength -> 70% EEM")
print("=" * 70)

for lookback_d, threshold_pct in [
    (5, 0.50),
    (5, 1.00),
    (10, 0.50),
    (10, 1.00),
    (10, 1.50),
    (20, 1.00),
    (20, 1.50),
    (20, 2.00),
]:
    strat_rets = []
    regimes = []
    prev_regime = -1
    threshold = threshold_pct / 100.0

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            regimes.append(0)
            continue

        ratio_now = eem_spy_ratio[i - 1]
        ratio_lb = eem_spy_ratio[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            regimes.append(max(prev_regime, 0))
            continue

        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom < -threshold:
            # EM stress -> flight to quality
            regime = 1
            port_ret = spy_rets[i] * 0.80 + asset_ret("GLD", i) * 0.10
        elif ratio_mom > threshold:
            # EM strength -> allocate to EEM
            regime = 2
            port_ret = asset_ret("EEM", i) * 0.70
        else:
            # Neutral
            regime = 0
            port_ret = spy_rets[i] * 0.50

        if prev_regime >= 0 and regime != prev_regime:
            port_ret -= TC_BPS / 10000.0
        prev_regime = regime
        regimes.append(regime)
        strat_rets.append(port_ret)

    s, dd, tr = metrics(strat_rets)
    sw = count_switches(regimes)
    record("H16.6", f"lb={lookback_d} thresh={threshold_pct:.2f}%", s, dd, tr, sw)


# ============================================================
# H16.7: Utilities/Tech Ratio as Risk Barometer
# Signal: XLU/XLK price ratio 15-day momentum (DIFFERENT from XLK/XLE!)
# XLU outperforming XLK = defensive rotation = risk-off incoming
# When XLU/XLK rising: 50% TLT + 30% GLD; when falling: 90% QQQ
# Cost: 3 bps per switch
# ============================================================
print("\n" + "=" * 70)
print("H16.7: Utilities/Tech Ratio as Risk Barometer (XLU/XLK ratio momentum)")
print("Mechanism: defensive vs growth sector rotation as risk signal")
print("XLU outperforming XLK (defensive) -> 50% TLT + 30% GLD")
print("XLK outperforming XLU (growth) -> 90% QQQ")
print("=" * 70)

for lookback_d in [10, 15, 20, 30, 40]:
    strat_rets = []
    regimes = []
    prev_regime = -1

    for i in range(WARMUP, n):
        if i < lookback_d + 1:
            strat_rets.append(0.0)
            regimes.append(0)
            continue

        ratio_now = xlu_xlk_ratio[i - 1]
        ratio_lb = xlu_xlk_ratio[i - 1 - lookback_d]

        if ratio_now <= 0 or ratio_lb <= 0:
            strat_rets.append(0.0)
            regimes.append(max(prev_regime, 0))
            continue

        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom > 0:
            # XLU outperforming XLK -> defensive rotation -> risk-off
            regime = 1
            port_ret = asset_ret("TLT", i) * 0.50 + asset_ret("GLD", i) * 0.30
        else:
            # XLK outperforming XLU -> growth rotation -> risk-on
            regime = 2
            port_ret = asset_ret("QQQ", i) * 0.90

        if prev_regime >= 0 and regime != prev_regime:
            port_ret -= TC_BPS / 10000.0
        prev_regime = regime
        regimes.append(regime)
        strat_rets.append(port_ret)

    s, dd, tr = metrics(strat_rets)
    sw = count_switches(regimes)
    record("H16.7", f"lb={lookback_d}", s, dd, tr, sw)


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

gld_rets = ret_lists["GLD"][WARMUP:]
s, dd, tr = metrics(gld_rets)
print(f"  GLD buy-hold:  Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")

eem_rets = ret_lists["EEM"][WARMUP:]
s, dd, tr = metrics(eem_rets)
print(f"  EEM buy-hold:  Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f}")


# ============================================================
# SUMMARY — ranked by Sharpe with gate check
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE — ALL CONFIGS RANKED BY SHARPE")
print("=" * 70)

# Header
print(
    f"\n  {'Hypothesis':<12} {'Config':<30} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Switches':>9} {'Gates':>8}"
)
print(f"  {'-' * 12} {'-' * 30} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 9} {'-' * 8}")

all_sorted = sorted(results, key=lambda x: x[2], reverse=True)
for h, c, s, dd, tr, sw in all_sorted:
    pass_sharpe = "Y" if s > 0.80 else "N"
    pass_dd = "Y" if dd < 0.15 else "N"
    gate_str = f"S:{pass_sharpe} D:{pass_dd}"
    tag = " ***" if s > 0.80 and dd < 0.15 else ""
    print(f"  {h:<12} {c:<30} {s:>7.4f} {dd:>7.4f} {tr:>8.4f} {sw:>9} {gate_str}{tag}")


# Passing configs
passing = [
    (h, c, s, dd, tr, sw) for h, c, s, dd, tr, sw in results if s > 0.80 and dd < 0.15
]
passing.sort(key=lambda x: x[2], reverse=True)

print("\n" + "=" * 70)
if passing:
    print(f"CANDIDATES FOR FULL ROBUSTNESS — {len(passing)} configs pass BOTH gates")
    print("(Sharpe > 0.80 AND MaxDD < 15%)")
    print("=" * 70)
    for h, c, s, dd, tr, sw in passing:
        print(f"  {h} {c}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f} Switches={sw}")
else:
    print("NO configs pass both gates (Sharpe > 0.80 AND MaxDD < 15%)")
    print("=" * 70)

# Near-misses
near = [
    (h, c, s, dd, tr, sw)
    for h, c, s, dd, tr, sw in results
    if (s > 0.65 and dd < 0.18) and not (s > 0.80 and dd < 0.15)
]
near.sort(key=lambda x: x[2], reverse=True)

if near:
    print(f"\n  Near-misses ({len(near)} configs with Sharpe > 0.65, MaxDD < 18%):")
    for h, c, s, dd, tr, sw in near[:10]:
        print(
            f"    {h} {c}: Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f} Switches={sw}"
        )

# Best per hypothesis
print("\n" + "=" * 70)
print("BEST CONFIG PER HYPOTHESIS")
print("=" * 70)
hyp_set = sorted({h for h, _, _, _, _, _ in results})
for hyp in hyp_set:
    hyp_results = [(h, c, s, dd, tr, sw) for h, c, s, dd, tr, sw in results if h == hyp]
    best = max(hyp_results, key=lambda x: x[2])
    h, c, s, dd, tr, sw = best
    tag = " *** PASS ***" if s > 0.80 and dd < 0.15 else ""
    print(
        f"  {h}: {c} -> Sharpe={s:.4f} MaxDD={dd:.4f} Return={tr:.4f} Switches={sw}{tag}"
    )

print("\n" + "=" * 70)
print("CROSS-ASSET SCAN V6 COMPLETE")
print("Target: Sharpe > 0.80, MaxDD < 15%")
print("*** marks configs passing both gates")
print("=" * 70)

#!/usr/bin/env python3
"""Backtest for xlk-xle-soxl-rotation-v1 (D10 — Track D Sprint Alpha).

Mechanism: XLK/XLE ratio momentum (40d) + ratio vs 20d SMA classifies regimes.
  Growth (mom>0 AND ratio>SMA):  40% SOXL + 10% SHY
  Inflation (mom<0 AND ratio<SMA): 30% GLD + 20% DBA
  Neutral:                         25% UPRO + 25% SHY
  VIX > 30 crash filter:          100% SHY

Rebalances every 5 trading days. 10 bps round-trip cost per rebalance.

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_d10_xlk_xle_soxl.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

import polars as pl
import yaml
from scipy import stats

from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "xlk-xle-soxl-rotation-v1"
SYMBOLS = ["XLK", "XLE", "SOXL", "UPRO", "GLD", "DBA", "SHY", "VIX"]
LOOKBACK_DAYS = 5 * 365
WARMUP = 60

# Base parameters (from research spec)
BASE_PARAMS = {
    "lookback": 40,
    "sma_period": 20,
    "rebalance_freq": 5,
    "vix_threshold": 30,
    "growth_soxl_weight": 0.40,
    "growth_shy_weight": 0.10,
    "inflation_gld_weight": 0.30,
    "inflation_dba_weight": 0.20,
    "neutral_upro_weight": 0.25,
    "neutral_shy_weight": 0.25,
}

# ============================================================================
# DATA FETCH
# ============================================================================
print("Fetching data...")
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build date spine from XLK (signal instrument, longest history)
xlk_df = prices.filter(pl.col("symbol") == "XLK").sort("date")
dates = xlk_df["date"].to_list()
n = len(dates)
print(f"Trading days: {n}")


# Build lookup dicts for each symbol
def _build_lookup(sym: str) -> dict:
    df = prices.filter(pl.col("symbol") == sym).sort("date")
    return dict(zip(df["date"].to_list(), df["close"].to_list(), strict=False))


xlk_data = _build_lookup("XLK")
xle_data = _build_lookup("XLE")
soxl_data = _build_lookup("SOXL")
upro_data = _build_lookup("UPRO")
gld_data = _build_lookup("GLD")
dba_data = _build_lookup("DBA")
shy_data = _build_lookup("SHY")
vix_data = _build_lookup("VIX")


# ============================================================================
# HELPERS
# ============================================================================
def _asset_ret(data: dict, i: int) -> float:
    """Compute daily return for an asset from its lookup dict."""
    d, dp = dates[i], dates[i - 1]
    if d in data and dp in data and data[dp] > 0:
        return data[d] / data[dp] - 1
    return 0.0


def _classify_regime(ratio_mom: float, ratio_now: float, ratio_sma: float) -> str:
    """Classify market regime from ratio momentum and SMA position."""
    if ratio_mom > 0 and ratio_now > ratio_sma:
        return "growth"
    if ratio_mom < 0 and ratio_now < ratio_sma:
        return "inflation"
    return "neutral"


def _regime_return(regime: str, i: int, weights: dict) -> float:
    """Compute portfolio return for one day given the regime and weights."""
    if regime == "crash":
        return _asset_ret(shy_data, i) * 1.0
    if regime == "growth":
        return (
            _asset_ret(soxl_data, i) * weights["g_soxl"]
            + _asset_ret(shy_data, i) * weights["g_shy"]
        )
    if regime == "inflation":
        return (
            _asset_ret(gld_data, i) * weights["i_gld"]
            + _asset_ret(dba_data, i) * weights["i_dba"]
        )
    # neutral
    return (
        _asset_ret(upro_data, i) * weights["n_upro"]
        + _asset_ret(shy_data, i) * weights["n_shy"]
    )


def _compute_signal(ratio_series: list, i: int, lookback: int, sma_period: int):
    """Compute ratio momentum and SMA. Returns (ratio_mom, ratio_now, ratio_sma) or None."""
    ratio_now = ratio_series[i - 1]
    ratio_lb = ratio_series[i - 1 - lookback]
    if ratio_now <= 0 or ratio_lb <= 0:
        return None
    ratio_mom = ratio_now / ratio_lb - 1
    sma_start = max(0, i - 1 - sma_period)
    sma_window = ratio_series[sma_start : i - 1]
    if len(sma_window) < sma_period:
        return None
    sma_window = sma_window[-sma_period:]
    ratio_sma = sum(sma_window) / len(sma_window)
    if ratio_sma <= 0:
        return None
    return ratio_mom, ratio_now, ratio_sma


def run_backtest(params: dict, cost_bps: float = 10.0) -> dict:
    """Run a single backtest with the given parameters.

    Returns dict with sharpe, max_dd, cagr, total_return, daily_returns,
    trade_count, regime_counts, nav_series.
    """
    lookback = int(params.get("lookback", 40))
    sma_period = int(params.get("sma_period", 20))
    rebalance_freq = int(params.get("rebalance_freq", 5))
    vix_threshold = float(params.get("vix_threshold", 30))
    weights = {
        "g_soxl": float(params.get("growth_soxl_weight", 0.40)),
        "g_shy": float(params.get("growth_shy_weight", 0.10)),
        "i_gld": float(params.get("inflation_gld_weight", 0.30)),
        "i_dba": float(params.get("inflation_dba_weight", 0.20)),
        "n_upro": float(params.get("neutral_upro_weight", 0.25)),
        "n_shy": float(params.get("neutral_shy_weight", 0.25)),
    }

    cost_per_switch = cost_bps / 10000.0  # Convert bps to decimal

    # Build XLK/XLE ratio series
    ratio_series = []
    for i in range(n):
        d = dates[i]
        xlk = xlk_data.get(d, 0.0)
        xle = xle_data.get(d, 0.0)
        ratio_series.append(xlk / xle if xlk > 0 and xle > 0 else 0.0)

    daily_returns = []
    nav_series = [1.0]
    trade_count = 0
    regime_counts = {"growth": 0, "inflation": 0, "neutral": 0, "crash": 0}

    prev_regime = None
    days_since_rebalance = 0

    for i in range(WARMUP, n):
        if i < max(lookback, sma_period) + 1:
            daily_returns.append(0.0)
            nav_series.append(nav_series[-1])
            continue

        # Compute signal from YESTERDAY's data (lag-1, no look-ahead)
        signal = _compute_signal(ratio_series, i, lookback, sma_period)
        if signal is None:
            daily_returns.append(0.0)
            nav_series.append(nav_series[-1])
            continue

        ratio_mom, ratio_now, ratio_sma = signal

        # VIX crash filter (yesterday's close)
        vix_level = vix_data.get(dates[i - 1], 0.0)

        # Determine regime
        regime = (
            "crash"
            if vix_level > vix_threshold
            else _classify_regime(ratio_mom, ratio_now, ratio_sma)
        )
        regime_counts[regime] += 1
        days_since_rebalance += 1

        # Compute daily return based on regime allocation
        day_ret = _regime_return(regime, i, weights)

        # Apply switching cost on regime change or scheduled rebalance
        if prev_regime is not None and (
            regime != prev_regime or days_since_rebalance >= rebalance_freq
        ):
            day_ret -= cost_per_switch
            trade_count += 1
            days_since_rebalance = 0

        if regime != prev_regime:
            days_since_rebalance = 0
        prev_regime = regime

        daily_returns.append(day_ret)
        nav_series.append(nav_series[-1] * (1.0 + day_ret))

    return _compute_metrics(daily_returns, nav_series, trade_count, regime_counts)


def _compute_metrics(
    daily_returns: list[float],
    nav_series: list[float],
    trade_count: int,
    regime_counts: dict,
) -> dict:
    """Compute performance metrics from daily returns."""
    if not daily_returns or len(daily_returns) < 60:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "total_return": 0.0,
            "daily_returns": [],
            "trade_count": 0,
            "regime_counts": {},
            "nav_series": [],
            "trading_days": 0,
            "sortino": 0.0,
            "calmar": 0.0,
        }

    # MaxDD
    peak = nav_series[0]
    max_dd = 0.0
    for v in nav_series:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Sharpe
    mean_r = sum(daily_returns) / len(daily_returns)
    std_r = (sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    # CAGR
    total_ret = nav_series[-1] / nav_series[0] - 1.0
    years = len(daily_returns) / 252
    cagr = (nav_series[-1] / nav_series[0]) ** (1 / years) - 1 if years > 0 else 0.0

    # Sortino (downside deviation)
    downside = [r for r in daily_returns if r < 0]
    if downside:
        downside_std = (sum(r**2 for r in downside) / len(daily_returns)) ** 0.5
        sortino = (mean_r / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "cagr": cagr,
        "total_return": total_ret,
        "daily_returns": daily_returns,
        "trade_count": trade_count,
        "regime_counts": regime_counts,
        "nav_series": nav_series,
        "trading_days": len(daily_returns),
        "sortino": sortino,
        "calmar": calmar,
    }


# ============================================================================
# RUN BACKTEST
# ============================================================================
print("=" * 70)
print(f"BACKTEST: {SLUG} (Track D Sprint Alpha)")
print("=" * 70)
print("\nBase parameters:")
for k, v in BASE_PARAMS.items():
    print(f"  {k} = {v}")

print("\n--- RUNNING BASE BACKTEST (10 bps costs) ---")
result = run_backtest(BASE_PARAMS, cost_bps=10.0)

print(f"\nSharpe:       {result['sharpe']:.4f}")
print(f"MaxDD:        {result['max_dd']:.4f} ({result['max_dd'] * 100:.1f}%)")
print(f"CAGR:         {result['cagr']:.4f} ({result['cagr'] * 100:.1f}%)")
print(
    f"Total Return: {result['total_return']:.4f} ({result['total_return'] * 100:.1f}%)"
)
print(f"Sortino:      {result['sortino']:.4f}")
print(f"Calmar:       {result['calmar']:.4f}")
print(f"Trading Days: {result['trading_days']}")
print(f"Trade Count:  {result['trade_count']}")
print(f"Final NAV:    {result['nav_series'][-1]:.4f} (from 1.0000)")

print("\nRegime Distribution:")
total_days = sum(result["regime_counts"].values())
for regime, count in sorted(result["regime_counts"].items()):
    pct = count / total_days * 100 if total_days > 0 else 0
    print(f"  {regime:12s}: {count:5d} days ({pct:.1f}%)")

# ============================================================================
# DSR
# ============================================================================
print("\n--- DSR (Deflated Sharpe Ratio) ---")
sr = result["sharpe"]
T = result["trading_days"]
T_years = T / 252
se_sr = math.sqrt((1 + sr**2 / 2) / T_years) if T_years > 0 else 1.0
dsr = float(stats.norm.cdf(sr / se_sr)) if se_sr > 0 else 0.0
print(f"  Annualized Sharpe: {sr:.4f}")
print(f"  T = {T} days ({T_years:.2f} years)")
print(f"  SE(SR) = {se_sr:.4f}")
print(f"  DSR = {dsr:.4f}")

# ============================================================================
# BENCHMARK COMPARISON (TQQQ buy-and-hold)
# ============================================================================
print("\n--- BENCHMARK: TQQQ Buy-and-Hold ---")
# Fetch TQQQ for benchmark
try:
    tqqq_prices = fetch_ohlcv(["TQQQ"], lookback_days=LOOKBACK_DAYS)
    tqqq_df = tqqq_prices.filter(pl.col("symbol") == "TQQQ").sort("date")
    tqqq_data = dict(
        zip(tqqq_df["date"].to_list(), tqqq_df["close"].to_list(), strict=False)
    )
    tqqq_rets = []
    for i in range(WARMUP, n):
        d, dp = dates[i], dates[i - 1]
        if d in tqqq_data and dp in tqqq_data and tqqq_data[dp] > 0:
            tqqq_rets.append(tqqq_data[d] / tqqq_data[dp] - 1)
        else:
            tqqq_rets.append(0.0)
    if tqqq_rets:
        tqqq_nav = [1.0]
        for r in tqqq_rets:
            tqqq_nav.append(tqqq_nav[-1] * (1 + r))
        tqqq_mean = sum(tqqq_rets) / len(tqqq_rets)
        tqqq_std = (
            sum((r - tqqq_mean) ** 2 for r in tqqq_rets) / len(tqqq_rets)
        ) ** 0.5
        tqqq_sharpe = (tqqq_mean / tqqq_std * math.sqrt(252)) if tqqq_std > 0 else 0.0
        tqqq_total = tqqq_nav[-1] / tqqq_nav[0] - 1
        tqqq_years = len(tqqq_rets) / 252
        tqqq_cagr = (
            (tqqq_nav[-1] / tqqq_nav[0]) ** (1 / tqqq_years) - 1
            if tqqq_years > 0
            else 0.0
        )
        tqqq_peak = tqqq_nav[0]
        tqqq_maxdd = 0.0
        for v in tqqq_nav:
            tqqq_peak = max(tqqq_peak, v)
            tqqq_maxdd = max(tqqq_maxdd, (tqqq_peak - v) / tqqq_peak)
        print(f"  TQQQ Sharpe:       {tqqq_sharpe:.4f}")
        print(f"  TQQQ CAGR:         {tqqq_cagr:.4f} ({tqqq_cagr * 100:.1f}%)")
        print(f"  TQQQ MaxDD:        {tqqq_maxdd:.4f} ({tqqq_maxdd * 100:.1f}%)")
        print(f"  TQQQ Total Return: {tqqq_total:.4f} ({tqqq_total * 100:.1f}%)")
except Exception as e:
    print(f"  TQQQ benchmark skipped: {e}")

# ============================================================================
# TRACK D GATE ASSESSMENT
# ============================================================================
print(f"\n{'=' * 70}")
print("TRACK D GATE ASSESSMENT")
print(f"{'=' * 70}")

gate1 = result["sharpe"] >= 0.80
gate2 = result["max_dd"] < 0.40
gate3 = dsr >= 0.90

gates = [
    ("Gate 1: Sharpe >= 0.80", gate1, f"{result['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{result['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr:.4f}"),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

all_pass = all(g[1] for g in gates)
print(f"\n  PRELIMINARY VERDICT: {'PASS' if all_pass else 'FAIL'}")
print("  (Full gates require robustness script: CPCV, perturbation, shuffled signal)")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output = {
    "strategy_slug": SLUG,
    "track": "D",
    "strategy_type": "sector_rotation_leveraged",
    "base_params": BASE_PARAMS,
    "sharpe": round(result["sharpe"], 4),
    "max_dd": round(result["max_dd"], 4),
    "cagr": round(result["cagr"], 4),
    "total_return": round(result["total_return"], 4),
    "sortino": round(result["sortino"], 4),
    "calmar": round(result["calmar"], 4),
    "dsr": round(dsr, 4),
    "trading_days": result["trading_days"],
    "trade_count": result["trade_count"],
    "regime_counts": result["regime_counts"],
    "final_nav": round(result["nav_series"][-1], 4),
    "gates": {
        "sharpe_gte_0.80": gate1,
        "maxdd_lt_40pct": gate2,
        "dsr_gte_0.90": gate3,
    },
    "preliminary_verdict": "PASS" if all_pass else "FAIL",
}

out_path = Path(f"data/strategies/{SLUG}/backtest.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")

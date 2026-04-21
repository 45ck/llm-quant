#!/usr/bin/env python3
"""Paper trading batch runner: generates daily signals for all 37 strategies.

Fetches OHLCV data once (shared across all strategies), computes today's signal
for each strategy, appends to paper-trading.yaml, and prints a summary table.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_paper_batch.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_paper_batch.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import sys
from pathlib import Path

import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 5 * 365
COST_PER_SWITCH = 0.0003
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"
INITIAL_NAV = 100_000.0
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Strategy parameter registry — read from frozen research specs
# ---------------------------------------------------------------------------

# Type 1: Lead-lag strategies
# Format: (leader, follower, signal_window, entry_threshold, exit_threshold, target_weight)
LEAD_LAG_PARAMS: dict[str, tuple[str, str, int, float, float, float]] = {
    "soxx-qqq-lead-lag": ("SOXX", "QQQ", 5, 0.02, -0.01, 0.90),
    "lqd-spy-credit-lead": ("LQD", "SPY", 5, 0.005, -0.005, 0.80),
    "agg-spy-credit-lead": ("AGG", "SPY", 5, 0.003, -0.003, 0.80),
    "hyg-spy-5d-credit-lead": ("HYG", "SPY", 5, 0.005, -0.005, 0.80),
    "agg-qqq-credit-lead": ("AGG", "QQQ", 5, 0.003, -0.003, 0.80),
    "lqd-qqq-credit-lead": ("LQD", "QQQ", 5, 0.005, -0.005, 0.80),
    "vcit-qqq-credit-lead": ("VCIT", "QQQ", 5, 0.004, -0.004, 0.80),
    "hyg-qqq-credit-lead": ("HYG", "QQQ", 5, 0.005, -0.005, 0.75),
    "emb-spy-credit-lead": ("EMB", "SPY", 5, 0.005, -0.005, 0.80),
    "agg-efa-credit-lead": ("AGG", "EFA", 5, 0.003, -0.003, 0.80),
    # Rate momentum (signal_window=10)
    "tlt-spy-rate-momentum": ("TLT", "SPY", 10, 0.01, -0.01, 0.80),
    "tlt-qqq-rate-tech": ("TLT", "QQQ", 10, 0.01, -0.01, 0.80),
    "ief-qqq-rate-tech": ("IEF", "QQQ", 10, 0.005, -0.005, 0.80),
    # Track D: leveraged re-expression
    "tlt-tqqq-leveraged-lead-lag": ("TLT", "TQQQ", 10, 0.01, -0.01, 0.30),
    # Track D Sprint Alpha: TLT->TQQQ with frozen spec params (exit_threshold=-0.005)
    "tlt-tqqq-sprint": ("TLT", "TQQQ", 10, 0.01, -0.005, 0.30),
    # Track D Sprint Alpha batch (llm-quant-e6ju) — frozen spec params
    # Rate-momentum sprints
    "tlt-soxl-sprint": ("TLT", "SOXL", 10, 0.01, -0.005, 0.30),
    "tlt-upro-sprint": ("TLT", "UPRO", 10, 0.01, -0.005, 0.30),
    "ief-tqqq-sprint": ("IEF", "TQQQ", 10, 0.005, -0.003, 0.30),
    # Credit-lead sprints (IG credit into TQQQ/UPRO)
    "agg-tqqq-sprint": ("AGG", "TQQQ", 10, 0.01, -0.005, 0.30),
    "agg-upro-sprint": ("AGG", "UPRO", 10, 0.01, -0.005, 0.30),
    "vcit-tqqq-sprint": ("VCIT", "TQQQ", 10, 0.005, -0.003, 0.30),
    "lqd-tqqq-sprint": ("LQD", "TQQQ", 10, 0.01, -0.005, 0.30),
}

# D3 TQQQ/TMF ratio z-score mean-reversion parameters
D3_PARAMS = {
    "lookback_days": 60,
    "z_threshold": 1.0,
    "target_weight": 0.30,
    "vix_crash_threshold": 30.0,
}

# All symbols needed across all strategies (union)
ALL_SYMBOLS = sorted(
    {
        "SPY",
        "QQQ",
        "TLT",
        "SHY",
        "GLD",
        "SLV",
        "EFA",
        "IEF",
        "TIP",
        "DBA",
        "DBC",
        "USO",
        "XLK",
        "XLE",
        "AGG",
        "HYG",
        "LQD",
        "VCIT",
        "EMB",
        "SOXX",
        "IWM",
        "UUP",
        "GDX",
        "XLB",
        "XLF",
        "XLI",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
        "XLC",
        "XLRE",
        "SPYD",
        "TQQQ",
        "TMF",
        "VIX",
        "TNX",
        # Track D leveraged universe (llm-quant-e6ju)
        "SOXL",
        "UPRO",
    }
)

MECHANISM_FAMILIES: dict[str, str] = {
    "soxx-qqq-lead-lag": "F8",
    "lqd-spy-credit-lead": "F1",
    "agg-spy-credit-lead": "F1",
    "hyg-spy-5d-credit-lead": "F1",
    "agg-qqq-credit-lead": "F1",
    "lqd-qqq-credit-lead": "F1",
    "vcit-qqq-credit-lead": "F1",
    "hyg-qqq-credit-lead": "F1",
    "emb-spy-credit-lead": "F1",
    "agg-efa-credit-lead": "F1",
    "spy-overnight-momentum": "F5",
    "tlt-spy-rate-momentum": "F6",
    "tlt-qqq-rate-tech": "F6",
    "ief-qqq-rate-tech": "F6",
    "behavioral-structural": "F7",
    "gld-slv-mean-reversion-v4": "F2",
    "skip-month-tsmom-v1": "F3",
    "credit-spread-regime-v1": "F9",
    "dba-commodity-cycle-v1": "F11",
    "xlk-xle-sector-rotation-v1": "F12",
    "vol-regime-v2": "F13",
    "tlt-shy-curve-momentum-v1": "F14",
    "tip-tlt-real-yield-v1": "F15",
    "breakeven-inflation-v1": "F16",
    "global-yield-flow-v2": "F17",
    "commodity-carry-v2": "F18",
    "tlt-gld-disinflation-v1": "F19",
    "dbc-spy-commodity-equity-v1": "F21",
    "agg-tlt-duration-rotation-v2": "F22",
    "uso-xle-mean-reversion-v2": "F2",
    "gdx-gld-mean-reversion-v1": "F2",
    "dollar-gold-regime-v1": "F26",
    "erp-regime-v1": "F30",
    "reit-divergence-v2": "F33",
    "dividend-yield-regime-v1": "F42",
    "tlt-tqqq-leveraged-lead-lag": "F6-leveraged",
    "tlt-tqqq-sprint": "F6-D",
    "d3-tqqq-tmf-ratio-mr": "D3",
    # Track D Sprint Alpha batch (llm-quant-e6ju)
    # F6 rate-momentum leveraged variants
    "tlt-soxl-sprint": "F6-D",
    "tlt-upro-sprint": "F6-D",
    "ief-tqqq-sprint": "F6-D",
    # F1 credit lead-lag leveraged variants
    "agg-tqqq-sprint": "F1-D",
    "agg-upro-sprint": "F1-D",
    "vcit-tqqq-sprint": "F1-D",
    "lqd-tqqq-sprint": "F1-D",
    # F8 lead-lag (semis) leveraged variant (D11)
    "soxx-soxl-lead-lag-v1": "F8-D",
    # F12 sector rotation leveraged variant (D10)
    "xlk-xle-soxl-rotation-v1": "F12-D",
    # F13 vol-regime leveraged variant (D15)
    "d15-vol-regime-tqqq": "F13-D",
    # F15 real-yield leveraged variant (D12)
    "tip-tlt-upro-real-yield-v1": "F15-D",
    # F19 disinflation leveraged variant (D14)
    "d14-disinflation-tqqq": "F19-D",
    # F3 TSMOM leveraged variant (D13)
    "tsmom-upro-trend-v1": "F3-D",
}


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------


def load_shared_data() -> tuple[pl.DataFrame, dict[str, dict], list]:
    """Fetch all symbols and build shared price structures.

    Returns (prices_df, sym_data, spy_dates) where sym_data maps
    symbol -> {date: close_price}.
    """
    logger.info("Fetching OHLCV data for %d symbols...", len(ALL_SYMBOLS))
    prices = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS)
    logger.info(
        "Data: %d rows, date range: %s to %s",
        len(prices),
        prices["date"].min(),
        prices["date"].max(),
    )

    sym_data: dict[str, dict] = {}
    for sym in ALL_SYMBOLS:
        sdf = prices.filter(pl.col("symbol") == sym).sort("date")
        sym_data[sym] = dict(
            zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
        )

    # Use SPY dates as the canonical trading calendar
    spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
    spy_dates = spy_df["date"].to_list()

    return prices, sym_data, spy_dates


def get_close(sym_data: dict[str, dict], symbol: str, date) -> float:
    """Get closing price for a symbol on a date, or 0.0 if missing."""
    return sym_data.get(symbol, {}).get(date, 0.0)


def get_return(sym_data: dict[str, dict], symbol: str, date, prev_date) -> float:
    """Compute daily return for a symbol between two dates."""
    p1 = get_close(sym_data, symbol, date)
    p0 = get_close(sym_data, symbol, prev_date)
    if p0 > 0 and p1 > 0:
        return p1 / p0 - 1
    return 0.0


# ---------------------------------------------------------------------------
# Signal generators — one per strategy type
# ---------------------------------------------------------------------------


def signal_lead_lag(
    slug: str,
    sym_data: dict[str, dict],
    dates: list,
) -> dict:
    """Compute today's signal for a lead-lag strategy.

    Uses data up to dates[-2] (yesterday) to decide today's position (CAUSAL).
    Returns dict with signal info.
    """
    leader, follower, window, entry_thresh, exit_thresh, weight = LEAD_LAG_PARAMS[slug]
    n = len(dates)
    if n < window + 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Leader N-day return as of yesterday (causal: data through dates[-2])
    yesterday = dates[-2]
    lookback_date = dates[-2 - window] if (n - 2 - window) >= 0 else dates[0]
    leader_now = get_close(sym_data, leader, yesterday)
    leader_lb = get_close(sym_data, leader, lookback_date)

    if leader_now <= 0 or leader_lb <= 0:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    leader_ret = leader_now / leader_lb - 1

    if leader_ret >= entry_thresh:
        regime = "risk_on"
        position = "long_" + follower.lower()
        alloc = {follower: weight}
    elif leader_ret <= exit_thresh:
        regime = "risk_off"
        position = "flat"
        alloc = {}
    else:
        regime = "neutral"
        position = "hold_prev"
        alloc = {}

    return {
        "position": position,
        "regime": regime,
        "weight": weight if regime == "risk_on" else 0.0,
        "signal_value": leader_ret,
        "signal_desc": f"{leader} {window}d ret={leader_ret:+.4f} "
        f"(entry>={entry_thresh:+.4f}, exit<={exit_thresh:+.4f})",
        "allocation": alloc,
    }


def signal_overnight_momentum(
    sym_data: dict[str, dict],
    dates: list,
    prices_df: pl.DataFrame,
) -> dict:
    """SPY overnight momentum: 10d avg overnight return > 0.002 -> long SPY."""
    n = len(dates)
    if n < 12:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Get SPY open prices from raw data
    spy_df = prices_df.filter(pl.col("symbol") == "SPY").sort("date")
    spy_open_data = dict(
        zip(spy_df["date"].to_list(), spy_df["open"].to_list(), strict=False)
    )

    # Compute 10d average overnight return as of yesterday (causal)
    overnight_rets = []
    for i in range(max(0, n - 12), n - 1):  # Use data up to yesterday
        d = dates[i]
        prev_d = dates[i - 1] if i > 0 else None
        if prev_d is None:
            continue
        open_price = spy_open_data.get(d, 0.0)
        prev_close = get_close(sym_data, "SPY", prev_d)
        if prev_close > 0 and open_price > 0:
            overnight_rets.append(open_price / prev_close - 1)

    if len(overnight_rets) < 5:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    avg_overnight = sum(overnight_rets[-10:]) / min(10, len(overnight_rets[-10:]))

    if avg_overnight > 0.002:
        regime = "risk_on"
        position = "long_spy"
        weight = 0.90
    elif avg_overnight < -0.0005:
        regime = "risk_off"
        position = "flat"
        weight = 0.0
    else:
        regime = "neutral"
        position = "hold_prev"
        weight = 0.0

    return {
        "position": position,
        "regime": regime,
        "weight": weight,
        "signal_value": avg_overnight,
        "signal_desc": f"10d avg overnight ret={avg_overnight:+.6f} "
        f"(entry>=0.002, exit<=-0.0005)",
        "allocation": {"SPY": weight} if regime == "risk_on" else {},
    }


def signal_behavioral_structural(
    sym_data: dict[str, dict],
    dates: list,
) -> dict:
    """Behavioral-structural: low-vol sector + correlation regime + TLT MR."""
    n = len(dates)
    if n < 270:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Sector ETFs for low-vol ranking
    sector_etfs = [
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
        "XLRE",
    ]

    # Compute 63-day realized vol for each sector as of yesterday
    vols = {}
    for etf in sector_etfs:
        rets = []
        for i in range(max(0, n - 64), n - 1):
            r = get_return(sym_data, etf, dates[i], dates[i - 1] if i > 0 else dates[i])
            rets.append(r)
        if len(rets) >= 30:
            mean = sum(rets) / len(rets)
            std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
            vols[etf] = std * math.sqrt(252)
        else:
            vols[etf] = 999.0  # Exclude if insufficient data

    # Bottom 3 by vol = low-vol sectors
    ranked = sorted(vols.items(), key=lambda x: x[1])
    low_vol_sectors = [s for s, _ in ranked[:3]]

    # Correlation regime check: 30d pairwise corr of SPY/QQQ/IWM
    corr_syms = ["SPY", "QQQ", "IWM"]
    corr_rets: dict[str, list[float]] = {}
    for sym in corr_syms:
        rs = []
        for i in range(max(0, n - 31), n - 1):
            r = get_return(sym_data, sym, dates[i], dates[i - 1] if i > 0 else dates[i])
            rs.append(r)
        corr_rets[sym] = rs

    min_len = min(len(v) for v in corr_rets.values())
    if min_len < 20:
        avg_corr = 0.5
    else:
        # Compute average pairwise correlation
        pairs = [("SPY", "QQQ"), ("SPY", "IWM"), ("QQQ", "IWM")]
        corrs = []
        for s1, s2 in pairs:
            r1 = corr_rets[s1][-min_len:]
            r2 = corr_rets[s2][-min_len:]
            m1, m2 = sum(r1) / len(r1), sum(r2) / len(r2)
            cov = sum((a - m1) * (b - m2) for a, b in zip(r1, r2, strict=True)) / len(
                r1
            )
            s1v = (sum((a - m1) ** 2 for a in r1) / len(r1)) ** 0.5
            s2v = (sum((b - m2) ** 2 for b in r2) / len(r2)) ** 0.5
            if s1v > 0 and s2v > 0:
                corrs.append(cov / (s1v * s2v))
        avg_corr = sum(corrs) / len(corrs) if corrs else 0.5

    # High correlation = passive flow concentration -> rotate to alternatives
    high_corr = avg_corr > 0.90

    if high_corr:
        regime = "corr_rotation"
        position = "defensive"
        alloc = {"GLD": 0.20, "TLT": 0.20}
        weight = 0.40
    else:
        regime = "low_vol_sectors"
        position = "low_vol_" + "_".join(s.lower() for s in low_vol_sectors)
        per_sector = 0.20
        alloc = dict.fromkeys(low_vol_sectors, per_sector)
        weight = per_sector * 3

    return {
        "position": position,
        "regime": regime,
        "weight": weight,
        "signal_value": avg_corr,
        "signal_desc": f"avg 30d corr={avg_corr:.4f}, "
        f"low_vol={low_vol_sectors}, high_corr={high_corr}",
        "allocation": alloc,
    }


def signal_gld_slv_mean_reversion(
    sym_data: dict[str, dict],
    dates: list,
) -> dict:
    """GLD/SLV mean reversion v4: consensus Bollinger Bands [60,90,120]."""
    n = len(dates)
    if n < 130:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Build GLD/SLV ratio as of yesterday
    ratio_series = []
    for i in range(n - 1):  # Up to yesterday
        d = dates[i]
        gld = get_close(sym_data, "GLD", d)
        slv = get_close(sym_data, "SLV", d)
        ratio_series.append(gld / slv if gld > 0 and slv > 0 else 0.0)

    if not ratio_series:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    # Consensus vote across windows [60, 90, 120]
    bb_std = 2.0
    votes = {"long_gld": 0, "long_slv": 0, "neutral": 0}

    for window in [60, 90, 120]:
        if len(ratio_series) < window:
            votes["neutral"] += 1
            continue
        w = ratio_series[-window:]
        sma = sum(w) / len(w)
        std = (sum((x - sma) ** 2 for x in w) / len(w)) ** 0.5
        if std <= 0:
            votes["neutral"] += 1
            continue
        z = (ratio_series[-1] - sma) / std
        if z < -bb_std:
            votes["long_gld"] += 1  # Ratio cheap -> long GLD (ratio reverts up)
        elif z > bb_std:
            votes["long_slv"] += 1  # Ratio rich -> long SLV (ratio reverts down)
        else:
            votes["neutral"] += 1

    # Majority vote
    weight = 0.40
    if votes["long_gld"] >= 2:
        regime = "long_gld"
        position = "long_gld"
        alloc = {"GLD": weight}
    elif votes["long_slv"] >= 2:
        regime = "long_slv"
        position = "long_slv"
        alloc = {"SLV": weight}
    else:
        regime = "neutral"
        position = "neutral"
        alloc = {"GLD": 0.20, "SLV": 0.20}
        weight = 0.40

    last_z = 0.0
    if len(ratio_series) >= 90:
        w90 = ratio_series[-90:]
        sma90 = sum(w90) / len(w90)
        std90 = (sum((x - sma90) ** 2 for x in w90) / len(w90)) ** 0.5
        if std90 > 0:
            last_z = (ratio_series[-1] - sma90) / std90

    return {
        "position": position,
        "regime": regime,
        "weight": weight,
        "signal_value": last_z,
        "signal_desc": f"GLD/SLV z(90d)={last_z:+.2f}, votes={dict(votes)}",
        "allocation": alloc,
    }


def _tsmom_asset_momentum(
    sym_data: dict[str, dict], dates: list, asset: str, skip: int, lookback: int
) -> float:
    """Compute skip-month momentum for a single asset (causal)."""
    n = len(dates)
    idx_end = n - 2 - skip
    idx_start = n - 2 - lookback
    if idx_end < 0 or idx_start < 0:
        return 0.0
    p_end = get_close(sym_data, asset, dates[idx_end])
    p_start = get_close(sym_data, asset, dates[idx_start])
    if p_start > 0 and p_end > 0:
        return p_end / p_start - 1
    return 0.0


def _tsmom_vol_weight(
    sym_data: dict[str, dict], dates: list, asset: str, vol_window: int
) -> float:
    """Compute vol-scaled weight for a single TSMOM asset."""
    n = len(dates)
    rets = [
        get_return(sym_data, asset, dates[i], dates[i - 1])
        for i in range(max(1, n - 1 - vol_window), n - 1)
    ]
    if len(rets) < 20:
        return 0.25
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    ann_vol = std * math.sqrt(252)
    if ann_vol <= 0:
        return 0.0
    return min(0.10 / ann_vol, 2.0)


def signal_skip_month_tsmom(
    sym_data: dict[str, dict],
    dates: list,
) -> dict:
    """Skip-month TSMOM: multi-asset momentum with 21d skip, 252d lookback."""
    assets = ["SPY", "TLT", "GLD", "EFA"]
    n = len(dates)
    if n < 280:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    signals = {a: _tsmom_asset_momentum(sym_data, dates, a, 21, 252) for a in assets}

    # Vol-scaled directional weights
    weights: dict[str, float] = {}
    for asset in assets:
        raw_w = _tsmom_vol_weight(sym_data, dates, asset, 63)
        direction = 1.0 if signals[asset] > 0 else -1.0
        weights[asset] = direction * raw_w / len(assets)

    # Normalize: cap total exposure at 1.0
    total_abs = sum(abs(w) for w in weights.values())
    if total_abs > 1.0:
        scale = 1.0 / total_abs
        weights = {k: v * scale for k, v in weights.items()}

    # Determine regime from net direction
    net_long = sum(1 for v in weights.values() if v > 0)
    regime = "risk_on" if net_long >= 3 else ("risk_off" if net_long <= 1 else "mixed")

    # Long-only allocation for paper trading
    alloc: dict[str, float] = {
        a: round(weights[a], 4) for a in assets if weights[a] > 0
    }
    total_weight = sum(alloc.values())
    mom_str = ", ".join(f"{a}={signals[a]:+.3f}" for a in assets)

    return {
        "position": regime,
        "regime": regime,
        "weight": round(total_weight, 4),
        "signal_value": sum(signals.values()) / len(assets),
        "signal_desc": f"skip-TSMOM: {mom_str}",
        "allocation": alloc,
    }


# ---------------------------------------------------------------------------
# Regime strategy signals (Type 2 — DIRECT computation)
# ---------------------------------------------------------------------------


def _ratio_momentum(
    sym_data: dict[str, dict],
    dates: list,
    sym_a: str,
    sym_b: str,
    lookback: int,
) -> float | None:
    """Compute ratio momentum of sym_a/sym_b over lookback days (causal)."""
    n = len(dates)
    if n < lookback + 2:
        return None
    # Use data as of yesterday
    d_now = dates[-2]
    d_lb = dates[-2 - lookback]
    a_now = get_close(sym_data, sym_a, d_now)
    a_lb = get_close(sym_data, sym_a, d_lb)
    b_now = get_close(sym_data, sym_b, d_now)
    b_lb = get_close(sym_data, sym_b, d_lb)
    if a_now <= 0 or a_lb <= 0 or b_now <= 0 or b_lb <= 0:
        return None
    ratio_now = a_now / b_now
    ratio_lb = a_lb / b_lb
    return ratio_now / ratio_lb - 1


def _ratio_vs_sma(
    sym_data: dict[str, dict],
    dates: list,
    sym_a: str,
    sym_b: str,
    sma_period: int,
) -> float | None:
    """Compute ratio vs SMA of sym_a/sym_b (causal)."""
    n = len(dates)
    if n < sma_period + 2:
        return None
    # Build ratio series up to yesterday
    ratio_series = []
    for i in range(max(0, n - 1 - sma_period - 1), n - 1):
        d = dates[i]
        a = get_close(sym_data, sym_a, d)
        b = get_close(sym_data, sym_b, d)
        if a > 0 and b > 0:
            ratio_series.append(a / b)

    if len(ratio_series) < sma_period:
        return None
    ratio_now = ratio_series[-1]
    sma_window = ratio_series[-sma_period:]
    ratio_sma = sum(sma_window) / len(sma_window)
    if ratio_sma <= 0:
        return None
    return ratio_now / ratio_sma - 1


def signal_credit_spread_regime(sym_data: dict[str, dict], dates: list) -> dict:
    """Credit spread regime: HYG/SHY ratio momentum + vs SMA."""
    ratio_mom = _ratio_momentum(sym_data, dates, "HYG", "SHY", 10)
    ratio_sma = _ratio_vs_sma(sym_data, dates, "HYG", "SHY", 20)
    if ratio_mom is None or ratio_sma is None:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    if ratio_mom > 0 and ratio_sma > 0:
        regime = "risk_on"
        alloc = {"SPY": 0.90}
    elif ratio_mom < 0 and ratio_sma < 0:
        regime = "risk_off"
        alloc = {"SHY": 0.90}
    else:
        regime = "neutral"
        alloc = {"SPY": 0.45}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": ratio_mom,
        "signal_desc": f"HYG/SHY mom(10d)={ratio_mom:+.4f}, vs_sma(20d)={ratio_sma:+.4f}",
        "allocation": alloc,
    }


def signal_ratio_regime(
    sym_data: dict[str, dict],
    dates: list,
    sym_a: str,
    sym_b: str,
    lookback: int,
    regime_a_name: str,
    regime_b_name: str,
    alloc_a: dict[str, float],
    alloc_b: dict[str, float],
) -> dict:
    """Generic ratio momentum regime: sym_a/sym_b ratio momentum -> allocation."""
    ratio_mom = _ratio_momentum(sym_data, dates, sym_a, sym_b, lookback)
    if ratio_mom is None:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    if ratio_mom > 0:
        regime = regime_a_name
        alloc = alloc_a
    else:
        regime = regime_b_name
        alloc = alloc_b

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": ratio_mom,
        "signal_desc": f"{sym_a}/{sym_b} mom({lookback}d)={ratio_mom:+.4f}",
        "allocation": alloc,
    }


def signal_xlk_xle_sector_rotation(sym_data: dict[str, dict], dates: list) -> dict:
    """XLK/XLE sector rotation: ratio momentum + vs SMA."""
    ratio_mom = _ratio_momentum(sym_data, dates, "XLK", "XLE", 40)
    ratio_sma = _ratio_vs_sma(sym_data, dates, "XLK", "XLE", 20)
    if ratio_mom is None or ratio_sma is None:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    if ratio_mom > 0 and ratio_sma > 0:
        regime = "growth"
        alloc = {"QQQ": 0.90}
    elif ratio_mom < 0 and ratio_sma < 0:
        regime = "inflation"
        alloc = {"GLD": 0.50, "DBA": 0.30}
    else:
        regime = "neutral"
        alloc = {"SPY": 0.45}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": ratio_mom,
        "signal_desc": f"XLK/XLE mom(40d)={ratio_mom:+.4f}, vs_sma(20d)={ratio_sma:+.4f}",
        "allocation": alloc,
    }


def signal_vol_regime_v2(sym_data: dict[str, dict], dates: list) -> dict:
    """Vol regime v2: SPY 30d vol vs GLD 30d vol."""
    n = len(dates)
    vol_window = 30
    if n < vol_window + 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Compute 30d vol for SPY and GLD (as of yesterday)
    spy_rets, gld_rets = [], []
    for i in range(max(1, n - 1 - vol_window), n - 1):
        spy_rets.append(get_return(sym_data, "SPY", dates[i], dates[i - 1]))
        gld_rets.append(get_return(sym_data, "GLD", dates[i], dates[i - 1]))

    if len(spy_rets) < vol_window or len(gld_rets) < vol_window:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    def ann_vol(rets):
        m = sum(rets) / len(rets)
        s = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
        return s * math.sqrt(252)

    spy_vol = ann_vol(spy_rets[-vol_window:])
    gld_vol = ann_vol(gld_rets[-vol_window:])

    if gld_vol <= 0:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    vol_ratio = spy_vol / gld_vol

    if vol_ratio > 1.0:
        regime = "equity_stress"
        alloc = {"GLD": 0.50, "SPY": 0.20, "SHY": 0.30}
    else:
        regime = "commodity_stress"
        alloc = {"SPY": 0.80, "GLD": 0.10, "SHY": 0.10}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": vol_ratio,
        "signal_desc": f"SPY vol={spy_vol:.1%}, GLD vol={gld_vol:.1%}, "
        f"ratio={vol_ratio:.3f}",
        "allocation": alloc,
    }


def signal_dba_commodity_cycle(sym_data: dict[str, dict], dates: list) -> dict:
    """DBA commodity cycle: 60d absolute momentum."""
    n = len(dates)
    lookback = 60
    if n < lookback + 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    d_now = dates[-2]
    d_lb = dates[-2 - lookback]
    dba_now = get_close(sym_data, "DBA", d_now)
    dba_lb = get_close(sym_data, "DBA", d_lb)
    if dba_now <= 0 or dba_lb <= 0:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    dba_mom = dba_now / dba_lb - 1

    if dba_mom > 0:
        regime = "inflation"
        alloc = {"SPY": 0.30, "GLD": 0.50}
    else:
        regime = "disinflation"
        alloc = {"SPY": 0.90}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": dba_mom,
        "signal_desc": f"DBA 60d momentum={dba_mom:+.4f}",
        "allocation": alloc,
    }


def signal_pair_mr(
    sym_data: dict[str, dict],
    dates: list,
    sym_a: str,
    sym_b: str,
    overweight: float,
    underweight: float,
    neutral_weight: float,
    bb_window: int,
    bb_std: float,
) -> dict:
    """Generic pair mean-reversion z-score signal."""
    n = len(dates)
    if n < bb_window + 10:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Build ratio series up to yesterday
    ratio_series = []
    for i in range(n - 1):
        d = dates[i]
        a = get_close(sym_data, sym_a, d)
        b = get_close(sym_data, sym_b, d)
        if a > 0 and b > 0:
            ratio_series.append(a / b)
        else:
            ratio_series.append(0.0)

    if len(ratio_series) < bb_window:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    window = ratio_series[-bb_window:]
    sma = sum(window) / len(window)
    std = (sum((x - sma) ** 2 for x in window) / len(window)) ** 0.5
    if std <= 0:
        return {"position": "flat", "regime": "no_vol", "weight": 0.0}

    z = (ratio_series[-1] - sma) / std

    if z < -bb_std:
        regime = f"long_{sym_a.lower()}"
        alloc = {sym_a: overweight, sym_b: underweight}
    elif z > bb_std:
        regime = f"long_{sym_b.lower()}"
        alloc = {sym_b: overweight, sym_a: underweight}
    else:
        regime = "neutral"
        alloc = {sym_a: neutral_weight, sym_b: neutral_weight}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": z,
        "signal_desc": f"{sym_a}/{sym_b} z={z:+.2f} (thresholds: +/-{bb_std})",
        "allocation": alloc,
    }


def signal_d3_tqqq_tmf_ratio_mr(
    sym_data: dict[str, dict],
    dates: list,
) -> dict:
    """D3 TQQQ/TMF ratio z-score mean-reversion with VIX crash filter.

    Signal: TQQQ/TMF 60-day ratio z-score.
    Z < -1 (TMF cheap): 30% TMF, 70% SHY.
    Z > 1 (TQQQ cheap): 30% TQQQ, 70% SHY.
    Neutral: 15% TQQQ + 15% TMF + 70% SHY.
    VIX > 30: 100% SHY (crash filter).
    """
    lb = D3_PARAMS["lookback_days"]
    z_thresh = D3_PARAMS["z_threshold"]
    tw = D3_PARAMS["target_weight"]
    vix_crash = D3_PARAMS["vix_crash_threshold"]

    n = len(dates)
    if n < lb + 10:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # VIX crash filter (as of yesterday)
    yesterday = dates[-2]
    vix_price = get_close(sym_data, "VIX", yesterday)
    if vix_price > vix_crash:
        return {
            "position": "crash_protection",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > {vix_crash} -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    # Build TQQQ/TMF ratio series up to yesterday
    ratio_series = []
    for i in range(n - 1):
        d = dates[i]
        tqqq = get_close(sym_data, "TQQQ", d)
        tmf = get_close(sym_data, "TMF", d)
        if tqqq > 0 and tmf > 0:
            ratio_series.append(tqqq / tmf)
        else:
            ratio_series.append(0.0)

    if len(ratio_series) < lb:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Compute z-score
    window = ratio_series[-lb:]
    sma = sum(window) / len(window)
    std = (sum((x - sma) ** 2 for x in window) / len(window)) ** 0.5
    if std <= 0:
        return {"position": "flat", "regime": "no_vol", "weight": 0.0}

    z = (ratio_series[-1] - sma) / std

    if z < -z_thresh:
        # TMF cheap relative to TQQQ -> buy TMF
        regime = "long_tmf"
        alloc = {"TMF": tw, "SHY": 1.0 - tw}
    elif z > z_thresh:
        # TQQQ cheap relative to TMF -> buy TQQQ
        regime = "long_tqqq"
        alloc = {"TQQQ": tw, "SHY": 1.0 - tw}
    else:
        # Neutral: split
        half = tw / 2
        regime = "neutral"
        alloc = {"TQQQ": half, "TMF": half, "SHY": 1.0 - tw}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": z,
        "signal_desc": f"TQQQ/TMF z={z:+.2f} (thresh=+/-{z_thresh}), VIX={vix_price:.1f}",
        "allocation": alloc,
    }


def signal_erp_regime(
    sym_data: dict[str, dict],
    dates: list,
) -> dict:
    """F30 ERP Valuation Regime: SPY 1yr return minus TNX yield, 252d z-score.

    EQUITY_CHEAP (z > 1.0): 70% SPY + 10% TLT + 10% GLD + 10% SHY
    EQUITY_EXPENSIVE (z < -1.0): 20% SPY + 30% TLT + 30% GLD + 20% SHY
    NEUTRAL: 50% SPY + 20% TLT + 15% GLD + 15% SHY
    """
    n = len(dates)
    spy_lookback = 252
    zscore_lookback = 252
    min_data = spy_lookback + zscore_lookback + 2
    if n < min_data:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Build ERP proxy series up to yesterday (causal)
    erp_series = []
    for i in range(spy_lookback, n - 1):
        d = dates[i]
        d_lb = dates[i - spy_lookback]
        spy_now = get_close(sym_data, "SPY", d)
        spy_lb = get_close(sym_data, "SPY", d_lb)
        tnx = get_close(sym_data, "TNX", d)
        if spy_now <= 0 or spy_lb <= 0 or tnx <= 0:
            erp_series.append(None)
            continue
        spy_1y_ret = spy_now / spy_lb - 1
        tnx_yield = tnx / 100.0  # TNX is in percentage points
        erp = spy_1y_ret - tnx_yield
        erp_series.append(erp)

    # Filter valid entries for z-score
    valid_erp = [x for x in erp_series if x is not None]
    if len(valid_erp) < zscore_lookback:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Z-score of most recent ERP value against trailing window
    window = valid_erp[-zscore_lookback:]
    current = valid_erp[-1]
    mean = sum(window) / len(window)
    std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5
    if std <= 0:
        return {"position": "flat", "regime": "no_vol", "weight": 0.0}

    z = (current - mean) / std

    if z > 1.0:
        regime = "equity_cheap"
        alloc = {"SPY": 0.70, "TLT": 0.10, "GLD": 0.10, "SHY": 0.10}
    elif z < -1.0:
        regime = "equity_expensive"
        alloc = {"SPY": 0.20, "TLT": 0.30, "GLD": 0.30, "SHY": 0.20}
    else:
        regime = "neutral"
        alloc = {"SPY": 0.50, "TLT": 0.20, "GLD": 0.15, "SHY": 0.15}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": z,
        "signal_desc": f"ERP z={z:+.2f} (erp={current:+.4f}, spy_1y_ret-tnx_yield)",
        "allocation": alloc,
    }


def signal_reit_divergence_v2(sym_data: dict[str, dict], dates: list) -> dict:
    """F33 REIT Divergence v2: XLRE/SPY ratio z-score + 15d momentum.

    EASING (z > 0.5 AND mom > 0): 80% SPY + 10% QQQ + 10% GLD
    TIGHTENING (z < -0.5 AND mom < 0): 40% GLD + 30% SHY + 30% SPY
    NEUTRAL: 50% SPY + 40% SHY + 10% GLD
    """
    n = len(dates)
    zscore_window = 60
    mom_window = 15
    min_data = zscore_window + mom_window + 2
    if n < min_data:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Build XLRE/SPY ratio series (causal: through yesterday)
    ratios = []
    for i in range(n):
        xlre = get_close(sym_data, "XLRE", dates[i])
        spy = get_close(sym_data, "SPY", dates[i])
        if xlre > 0 and spy > 0:
            ratios.append(xlre / spy)
        else:
            ratios.append(None)

    # Use yesterday's values (causal)
    idx = n - 2  # yesterday

    # Compute z-score of ratio
    window_vals = [
        ratios[j]
        for j in range(idx - zscore_window + 1, idx + 1)
        if ratios[j] is not None
    ]
    if len(window_vals) < zscore_window // 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    mean_r = sum(window_vals) / len(window_vals)
    var_r = sum((v - mean_r) ** 2 for v in window_vals) / len(window_vals)
    std_r = var_r**0.5
    if std_r < 1e-10:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    z = (ratios[idx] - mean_r) / std_r if ratios[idx] is not None else 0.0

    # Compute momentum
    if ratios[idx] is not None and ratios[idx - mom_window] is not None:
        mom = ratios[idx] / ratios[idx - mom_window] - 1
    else:
        mom = 0.0

    # Regime classification
    if z > 0.5 and mom > 0:
        regime = "easing"
        alloc = {"SPY": 0.80, "QQQ": 0.10, "GLD": 0.10}
    elif z < -0.5 and mom < 0:
        regime = "tightening"
        alloc = {"GLD": 0.40, "SHY": 0.30, "SPY": 0.30}
    else:
        regime = "neutral"
        alloc = {"SPY": 0.50, "SHY": 0.40, "GLD": 0.10}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": z,
        "signal_desc": f"XLRE/SPY z={z:+.2f} mom={mom:+.4f}",
        "allocation": alloc,
    }


def signal_dividend_yield_regime(sym_data: dict[str, dict], dates: list) -> dict:
    """F42 Dividend Yield Regime: SPYD/SPY ratio 20d momentum + 40d SMA.

    INCOME_PREFERENCE (mom > 0 AND ratio > SMA): 40% GLD + 20% SHY + 20% TLT + 20% SPY
    GROWTH_PREFERENCE (mom < 0 AND ratio < SMA): 70% SPY + 15% QQQ + 15% SHY
    NEUTRAL: 50% SPY + 30% SHY + 10% GLD + 10% TLT
    """
    n = len(dates)
    mom_lookback = 20
    sma_period = 40
    min_data = sma_period + mom_lookback + 2
    if n < min_data:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Build SPYD/SPY ratio series
    ratios: list[float | None] = []
    for i in range(n):
        spyd = get_close(sym_data, "SPYD", dates[i])
        spy = get_close(sym_data, "SPY", dates[i])
        if spyd > 0 and spy > 0:
            ratios.append(spyd / spy)
        else:
            ratios.append(None)

    # Use yesterday's values (causal)
    idx = n - 2

    # Compute 20-day momentum
    if (
        ratios[idx] is not None
        and idx >= mom_lookback
        and ratios[idx - mom_lookback] is not None
        and ratios[idx - mom_lookback] > 0
    ):
        mom = ratios[idx] / ratios[idx - mom_lookback] - 1
    else:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Compute 40-day SMA of ratio
    window_vals = [
        ratios[j] for j in range(idx - sma_period + 1, idx + 1) if ratios[j] is not None
    ]
    if len(window_vals) < sma_period // 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    sma = sum(window_vals) / len(window_vals)
    current_ratio = ratios[idx]

    # Regime classification
    if mom > 0 and current_ratio > sma:
        regime = "income_preference"
        alloc = {"SPY": 0.20, "GLD": 0.40, "SHY": 0.20, "TLT": 0.20}
    elif mom < 0 and current_ratio < sma:
        regime = "growth_preference"
        alloc = {"SPY": 0.70, "QQQ": 0.15, "SHY": 0.15}
    else:
        regime = "neutral"
        alloc = {"SPY": 0.50, "SHY": 0.30, "GLD": 0.10, "TLT": 0.10}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": mom,
        "signal_desc": f"SPYD/SPY mom={mom:+.4f} ratio={current_ratio:.4f} sma={sma:.4f}",
        "allocation": alloc,
    }


# ---------------------------------------------------------------------------
# Track D regime/rotation signals — new in llm-quant-e6ju registration batch
# All include mandatory VIX > threshold crash filter (-> 100% SHY).
# ---------------------------------------------------------------------------


def _vix_crash(
    sym_data: dict[str, dict], dates: list, threshold: float = 30.0
) -> float | None:
    """Return yesterday's VIX close if VIX > threshold, else None.

    Used as shared risk-off override for all Track D regime strategies.
    """
    if len(dates) < 2:
        return None
    vix = get_close(sym_data, "VIX", dates[-2])
    if vix > threshold:
        return vix
    return None


def signal_d10_xlk_xle_soxl_rotation(sym_data: dict[str, dict], dates: list) -> dict:
    """D10: XLK/XLE ratio momentum + SMA -> SOXL (3x semis) for growth regime.

    Frozen params from research-spec.yaml:
      lookback=40, sma_period=20, growth_soxl=0.40, growth_shy=0.10,
      inflation_gld=0.30, inflation_dba=0.20, neutral_upro=0.25,
      neutral_shy=0.25, vix_crash=30.
    """
    vix_price = _vix_crash(sym_data, dates, 30.0)
    if vix_price is not None:
        return {
            "position": "vix_crash",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > 30 -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    ratio_mom = _ratio_momentum(sym_data, dates, "XLK", "XLE", 40)
    ratio_sma = _ratio_vs_sma(sym_data, dates, "XLK", "XLE", 20)
    if ratio_mom is None or ratio_sma is None:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    if ratio_mom > 0 and ratio_sma > 0:
        regime = "growth"
        alloc = {"SOXL": 0.40, "SHY": 0.10}
    elif ratio_mom < 0 and ratio_sma < 0:
        regime = "inflation"
        alloc = {"GLD": 0.30, "DBA": 0.20}
    else:
        regime = "neutral"
        alloc = {"UPRO": 0.25, "SHY": 0.25}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": ratio_mom,
        "signal_desc": f"XLK/XLE mom(40d)={ratio_mom:+.4f}, "
        f"vs_sma(20d)={ratio_sma:+.4f}",
        "allocation": alloc,
    }


def signal_d11_soxx_soxl(sym_data: dict[str, dict], dates: list) -> dict:
    """D11: SOXX 7-day return leads SOXL (3x semis), with VIX>35 crash filter.

    Frozen params (v2 post-scan): signal_window=7, entry_threshold=0.03,
    exit_threshold=-0.01, soxl_weight=0.50, shy_weight_risk_on=0.50,
    vix_crash_threshold=35.
    """
    vix_price = _vix_crash(sym_data, dates, 35.0)
    if vix_price is not None:
        return {
            "position": "vix_crash",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > 35 -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    n = len(dates)
    window = 7
    if n < window + 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    yesterday = dates[-2]
    lookback = dates[-2 - window]
    soxx_now = get_close(sym_data, "SOXX", yesterday)
    soxx_lb = get_close(sym_data, "SOXX", lookback)
    if soxx_now <= 0 or soxx_lb <= 0:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    soxx_ret = soxx_now / soxx_lb - 1

    if soxx_ret >= 0.03:
        regime = "risk_on"
        alloc = {"SOXL": 0.50, "SHY": 0.50}
    elif soxx_ret <= -0.01:
        regime = "risk_off"
        alloc = {"SHY": 1.00}
    else:
        regime = "neutral"
        alloc = {"SHY": 1.00}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": soxx_ret,
        "signal_desc": f"SOXX 7d ret={soxx_ret:+.4f} (entry>=0.030, exit<=-0.010)",
        "allocation": alloc,
    }


def signal_d12_tip_tlt_upro(sym_data: dict[str, dict], dates: list) -> dict:
    """D12: TIP/TLT 15-day ratio momentum -> UPRO real-yield regime.

    Loosening (ratio_mom >= 0): 35% UPRO + 65% SHY.
    Tightening (ratio_mom < 0): 40% GLD + 10% TLT + 50% SHY.
    VIX > 30 crash filter -> 100% SHY.
    """
    vix_price = _vix_crash(sym_data, dates, 30.0)
    if vix_price is not None:
        return {
            "position": "vix_crash",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > 30 -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    ratio_mom = _ratio_momentum(sym_data, dates, "TIP", "TLT", 15)
    if ratio_mom is None:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    if ratio_mom >= 0:
        regime = "loosening"
        alloc = {"UPRO": 0.35, "SHY": 0.65}
    else:
        regime = "tightening"
        alloc = {"GLD": 0.40, "TLT": 0.10, "SHY": 0.50}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": ratio_mom,
        "signal_desc": f"TIP/TLT mom(15d)={ratio_mom:+.4f}",
        "allocation": alloc,
    }


def signal_d13_tsmom_upro(sym_data: dict[str, dict], dates: list) -> dict:
    """D13: Skip-month TSMOM on SPY -> UPRO (3x SPY).

    Frozen params: momentum_lookback=252, skip_period=21,
    base_upro_weight=0.40, vol_target=0.15, vol_window=20,
    vix_crash_threshold=25 (best variant per backtest_results.yaml),
    shy_bullish=0.10, gld_bearish=0.30, tlt_bearish=0.30, shy_bearish=0.40.
    """
    vix_price = _vix_crash(sym_data, dates, 25.0)
    if vix_price is not None:
        return {
            "position": "vix_crash",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > 25 -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    n = len(dates)
    lookback = 252
    skip = 21
    if n < lookback + skip + 3:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    # Skip-month momentum: return from (t-lookback) to (t-skip), causal as of yesterday
    idx_end = n - 2 - skip
    idx_start = n - 2 - lookback
    if idx_start < 0:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    spy_end = get_close(sym_data, "SPY", dates[idx_end])
    spy_start = get_close(sym_data, "SPY", dates[idx_start])
    if spy_start <= 0 or spy_end <= 0:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    mom = spy_end / spy_start - 1

    # Vol-scaling: scale UPRO weight by vol_target / realized_vol
    vol_window = 20
    rets = [
        get_return(sym_data, "SPY", dates[i], dates[i - 1])
        for i in range(max(1, n - 1 - vol_window), n - 1)
    ]
    if len(rets) >= 10:
        mean = sum(rets) / len(rets)
        std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
        ann_vol = std * math.sqrt(252)
        if ann_vol > 0:
            vol_scale = min(0.15 / ann_vol, 1.25)
        else:
            vol_scale = 1.0
    else:
        vol_scale = 1.0

    base_upro = 0.40
    max_upro = 0.50
    scaled_upro = min(base_upro * vol_scale, max_upro)

    if mom > 0:
        regime = "bullish"
        alloc = {"UPRO": round(scaled_upro, 3), "SHY": round(1.0 - scaled_upro, 3)}
    else:
        regime = "bearish"
        alloc = {"GLD": 0.30, "TLT": 0.30, "SHY": 0.40}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": mom,
        "signal_desc": f"SPY skip-mom(252-21)={mom:+.4f}, vol_scale={vol_scale:.2f}",
        "allocation": alloc,
    }


def signal_d14_disinflation_tqqq(sym_data: dict[str, dict], dates: list) -> dict:
    """D14: TLT/GLD 30-day ratio momentum -> TQQQ disinflation regime.

    Disinflation (ratio_mom > 0): 50% TQQQ + 50% SHY.
    Inflation (ratio_mom < 0): 40% GLD + 10% DBA + 50% SHY.
    VIX > 30 crash filter -> 100% SHY.
    Frozen params: lookback=30, target_weight=0.50, vix_crash=30,
    gld_inflation=0.40, dba_inflation=0.10, rebalance_days=5.
    """
    vix_price = _vix_crash(sym_data, dates, 30.0)
    if vix_price is not None:
        return {
            "position": "vix_crash",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > 30 -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    ratio_mom = _ratio_momentum(sym_data, dates, "TLT", "GLD", 30)
    if ratio_mom is None:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    if ratio_mom > 0:
        regime = "disinflation"
        alloc = {"TQQQ": 0.50, "SHY": 0.50}
    else:
        regime = "inflation"
        alloc = {"GLD": 0.40, "DBA": 0.10, "SHY": 0.50}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": ratio_mom,
        "signal_desc": f"TLT/GLD mom(30d)={ratio_mom:+.4f}",
        "allocation": alloc,
    }


def signal_d15_vol_regime_tqqq(sym_data: dict[str, dict], dates: list) -> dict:
    """D15: SPY/GLD realized vol regime -> TQQQ (3x Nasdaq).

    Frozen params (phase2 recentered): vol_window=45, target_weight=0.50,
    vix_crash=30, gld_defensive=0.40, tlt_defensive=0.30, shy_defensive=0.30,
    rebalance_days=5.

    Equity calm (GLD vol > SPY vol): 50% TQQQ + 50% SHY.
    Equity stress (SPY vol > GLD vol): 40% GLD + 30% TLT + 30% SHY.
    """
    vix_price = _vix_crash(sym_data, dates, 30.0)
    if vix_price is not None:
        return {
            "position": "vix_crash",
            "regime": "vix_crash",
            "weight": 1.0,
            "signal_value": vix_price,
            "signal_desc": f"VIX={vix_price:.1f} > 30 -> 100% SHY",
            "allocation": {"SHY": 1.0},
        }

    n = len(dates)
    vol_window = 45
    if n < vol_window + 2:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    spy_rets, gld_rets = [], []
    for i in range(max(1, n - 1 - vol_window), n - 1):
        spy_rets.append(get_return(sym_data, "SPY", dates[i], dates[i - 1]))
        gld_rets.append(get_return(sym_data, "GLD", dates[i], dates[i - 1]))

    if len(spy_rets) < vol_window or len(gld_rets) < vol_window:
        return {"position": "flat", "regime": "insufficient_data", "weight": 0.0}

    def _ann_vol(rets: list[float]) -> float:
        m = sum(rets) / len(rets)
        s = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
        return s * math.sqrt(252)

    spy_vol = _ann_vol(spy_rets[-vol_window:])
    gld_vol = _ann_vol(gld_rets[-vol_window:])

    if gld_vol <= 0:
        return {"position": "flat", "regime": "missing_data", "weight": 0.0}

    vol_ratio = spy_vol / gld_vol

    if vol_ratio <= 1.0:
        regime = "equity_calm"
        alloc = {"TQQQ": 0.50, "SHY": 0.50}
    else:
        regime = "equity_stress"
        alloc = {"GLD": 0.40, "TLT": 0.30, "SHY": 0.30}

    return {
        "position": regime,
        "regime": regime,
        "weight": sum(alloc.values()),
        "signal_value": vol_ratio,
        "signal_desc": f"SPY vol(45d)={spy_vol:.1%}, "
        f"GLD vol(45d)={gld_vol:.1%}, ratio={vol_ratio:.3f}",
        "allocation": alloc,
    }


# ---------------------------------------------------------------------------
# Compute today's daily return for a signal
# ---------------------------------------------------------------------------


def compute_daily_return(
    alloc: dict[str, float],
    sym_data: dict[str, dict],
    dates: list,
) -> float:
    """Compute today's weighted return from allocation dict.

    Uses today's return (dates[-1] vs dates[-2]) weighted by allocation.
    """
    if not alloc or len(dates) < 2:
        return 0.0
    today = dates[-1]
    yesterday = dates[-2]
    day_ret = 0.0
    for sym, weight in alloc.items():
        r = get_return(sym_data, sym, today, yesterday)
        day_ret += r * weight
    return day_ret


# ---------------------------------------------------------------------------
# Paper trading YAML management
# ---------------------------------------------------------------------------


def load_paper_yaml(slug: str) -> dict:
    """Load existing paper-trading.yaml or create default structure."""
    path = DATA_DIR / slug / "paper-trading.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is not None:
            return data

    return {
        "strategy_slug": slug,
        "start_date": str(datetime.date.today()),
        "status": "active",
        "performance": {
            "initial_nav": INITIAL_NAV,
            "current_nav": INITIAL_NAV,
            "peak_nav": INITIAL_NAV,
            "current_sharpe": None,
            "current_drawdown": 0.0,
            "total_return": 0.0,
            "total_trades": 0,
        },
        "daily_log": [],
    }


def save_paper_yaml(slug: str, data: dict, dry_run: bool = False) -> None:
    """Save paper-trading.yaml for a strategy."""
    if dry_run:
        return
    path = DATA_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    yaml_path = path / "paper-trading.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)


def compute_cumulative_metrics(daily_log: list) -> dict:
    """Compute Sharpe, MaxDD, total return from daily log entries."""
    returns = [
        e.get("daily_return_pct", 0.0) / 100.0
        for e in daily_log
        if e.get("daily_return_pct") is not None
    ]
    if len(returns) < 2:
        return {
            "sharpe": None,
            "max_dd": 0.0,
            "total_return": 0.0,
            "days": len(returns),
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

    mean = sum(returns) / len(returns)
    std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5
    sharpe = (mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0
    total_ret = nav[-1] / nav[0] - 1.0

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "days": len(returns),
    }


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------


def run_all_signals(
    sym_data: dict[str, dict],
    dates: list,
    prices_df: pl.DataFrame,
) -> dict[str, dict]:
    """Compute today's signal for all 35 strategies. Returns slug -> signal dict."""
    results: dict[str, dict] = {}

    # Type 1: Lead-lag strategies
    for slug in LEAD_LAG_PARAMS:
        try:
            results[slug] = signal_lead_lag(slug, sym_data, dates)
        except Exception as e:
            logger.warning("Error computing %s: %s", slug, e)
            results[slug] = {"position": "error", "regime": str(e), "weight": 0.0}

    # Special strategies
    try:
        results["spy-overnight-momentum"] = signal_overnight_momentum(
            sym_data, dates, prices_df
        )
    except Exception as e:
        logger.warning("Error computing spy-overnight-momentum: %s", e)
        results["spy-overnight-momentum"] = {
            "position": "error",
            "regime": str(e),
            "weight": 0.0,
        }

    try:
        results["behavioral-structural"] = signal_behavioral_structural(sym_data, dates)
    except Exception as e:
        logger.warning("Error computing behavioral-structural: %s", e)
        results["behavioral-structural"] = {
            "position": "error",
            "regime": str(e),
            "weight": 0.0,
        }

    try:
        results["gld-slv-mean-reversion-v4"] = signal_gld_slv_mean_reversion(
            sym_data, dates
        )
    except Exception as e:
        logger.warning("Error computing gld-slv-mean-reversion-v4: %s", e)
        results["gld-slv-mean-reversion-v4"] = {
            "position": "error",
            "regime": str(e),
            "weight": 0.0,
        }

    try:
        results["skip-month-tsmom-v1"] = signal_skip_month_tsmom(sym_data, dates)
    except Exception as e:
        logger.warning("Error computing skip-month-tsmom-v1: %s", e)
        results["skip-month-tsmom-v1"] = {
            "position": "error",
            "regime": str(e),
            "weight": 0.0,
        }

    # Type 2: DIRECT computation regime strategies
    regime_configs = [
        ("credit-spread-regime-v1", signal_credit_spread_regime),
        ("xlk-xle-sector-rotation-v1", signal_xlk_xle_sector_rotation),
        ("vol-regime-v2", signal_vol_regime_v2),
        ("dba-commodity-cycle-v1", signal_dba_commodity_cycle),
        ("reit-divergence-v2", signal_reit_divergence_v2),
        ("dividend-yield-regime-v1", signal_dividend_yield_regime),
        # Track D regime/rotation strategies (llm-quant-e6ju)
        ("xlk-xle-soxl-rotation-v1", signal_d10_xlk_xle_soxl_rotation),
        ("soxx-soxl-lead-lag-v1", signal_d11_soxx_soxl),
        ("tip-tlt-upro-real-yield-v1", signal_d12_tip_tlt_upro),
        ("tsmom-upro-trend-v1", signal_d13_tsmom_upro),
        ("d14-disinflation-tqqq", signal_d14_disinflation_tqqq),
        ("d15-vol-regime-tqqq", signal_d15_vol_regime_tqqq),
    ]
    for slug, fn in regime_configs:
        try:
            results[slug] = fn(sym_data, dates)
        except Exception as e:
            logger.warning("Error computing %s: %s", slug, e)
            results[slug] = {"position": "error", "regime": str(e), "weight": 0.0}

    # Ratio momentum regime strategies
    ratio_configs = [
        # (slug, sym_a, sym_b, lookback, regime_a, regime_b, alloc_a, alloc_b)
        (
            "tlt-shy-curve-momentum-v1",
            "TLT",
            "SHY",
            30,
            "flattening",
            "steepening",
            {"SPY": 0.80},
            {"GLD": 0.80},
        ),
        (
            "tip-tlt-real-yield-v1",
            "TIP",
            "TLT",
            20,
            "tightening",
            "loosening",
            {"GLD": 0.50, "SHY": 0.20},
            {"SPY": 0.75},
        ),
        (
            "breakeven-inflation-v1",
            "TIP",
            "IEF",
            30,
            "inflation_rising",
            "inflation_falling",
            {"DBA": 0.40, "GLD": 0.30, "SPY": 0.10},
            {"SPY": 0.70, "SHY": 0.10},
        ),
        (
            "global-yield-flow-v2",
            "TLT",
            "EFA",
            30,
            "us_inflow",
            "intl_preferred",
            {"SPY": 0.80},
            {"EFA": 0.40, "GLD": 0.20},
        ),
        (
            "commodity-carry-v2",
            "USO",
            "DBC",
            20,
            "carry",
            "contango",
            {"USO": 0.20, "GLD": 0.20, "SPY": 0.30},
            {"SPY": 0.70, "SHY": 0.10},
        ),
        (
            "tlt-gld-disinflation-v1",
            "TLT",
            "GLD",
            30,
            "disinflation",
            "inflation",
            {"SPY": 0.80},
            {"GLD": 0.50, "DBA": 0.20, "SPY": 0.10},
        ),
        (
            "dbc-spy-commodity-equity-v1",
            "DBC",
            "SPY",
            30,
            "commodity",
            "equity",
            {"GLD": 0.40, "DBA": 0.30, "SPY": 0.10},
            {"SPY": 0.80},
        ),
        (
            "agg-tlt-duration-rotation-v2",
            "AGG",
            "TLT",
            20,
            "flattening",
            "steepening",
            {"SPY": 0.50, "GLD": 0.20},
            {"SPY": 0.80},
        ),
        (
            "dollar-gold-regime-v1",
            "UUP",
            "GLD",
            30,
            "dollar_strong",
            "gold_strong",
            {"SPY": 0.60},
            {"GLD": 0.40, "DBA": 0.15},
        ),
    ]
    for slug, sym_a, sym_b, lb, ra, rb, aa, ab in ratio_configs:
        try:
            results[slug] = signal_ratio_regime(
                sym_data, dates, sym_a, sym_b, lb, ra, rb, aa, ab
            )
        except Exception as e:
            logger.warning("Error computing %s: %s", slug, e)
            results[slug] = {"position": "error", "regime": str(e), "weight": 0.0}

    # Pair MR strategies
    pair_mr_configs = [
        # (slug, sym_a, sym_b, overweight, underweight, neutral_weight, bb_window, bb_std)
        ("uso-xle-mean-reversion-v2", "USO", "XLE", 0.60, 0.20, 0.35, 60, 1.0),
        ("gdx-gld-mean-reversion-v1", "GDX", "GLD", 0.50, 0.20, 0.35, 60, 1.0),
    ]
    for slug, sa, sb, ow, uw, nw, bbw, bbs in pair_mr_configs:
        try:
            results[slug] = signal_pair_mr(
                sym_data, dates, sa, sb, ow, uw, nw, bbw, bbs
            )
        except Exception as e:
            logger.warning("Error computing %s: %s", slug, e)
            results[slug] = {"position": "error", "regime": str(e), "weight": 0.0}

    # Track D: D3 TQQQ/TMF ratio z-score MR
    try:
        results["d3-tqqq-tmf-ratio-mr"] = signal_d3_tqqq_tmf_ratio_mr(sym_data, dates)
    except Exception as e:
        logger.warning("Error computing d3-tqqq-tmf-ratio-mr: %s", e)
        results["d3-tqqq-tmf-ratio-mr"] = {
            "position": "error",
            "regime": str(e),
            "weight": 0.0,
        }

    # F30: ERP Valuation Regime
    try:
        results["erp-regime-v1"] = signal_erp_regime(sym_data, dates)
    except Exception as e:
        logger.warning("Error computing erp-regime-v1: %s", e)
        results["erp-regime-v1"] = {
            "position": "error",
            "regime": str(e),
            "weight": 0.0,
        }

    return results


def rebuild_performance(paper: dict) -> dict:
    """Rebuild NAV chain and performance aggregates from daily_log.

    Walks daily_log in order, recomputing nav, daily_pnl, day_number,
    peak_nav, drawdown, total_return, total_trades, and current_sharpe
    from stored daily_return_pct values. Used after out-of-order insertion
    to restore chain consistency.
    """
    daily_log = paper.get("daily_log", [])
    perf = paper.get("performance", {})
    initial_nav = perf.get("initial_nav", INITIAL_NAV)

    nav = initial_nav
    peak = initial_nav
    total_trades = 0
    prev_regime = None

    for i, entry in enumerate(daily_log):
        r_pct = entry.get("daily_return_pct")
        r = (r_pct or 0.0) / 100.0
        prev_nav = nav
        nav = round(nav * (1.0 + r), 2)
        peak = max(peak, nav)
        entry["day_number"] = i + 1
        entry["nav"] = nav
        entry["daily_pnl"] = round(nav - prev_nav, 2)
        regime = entry.get("regime")
        if prev_regime is not None and regime != prev_regime:
            total_trades += 1
        prev_regime = regime

    current_dd = round((peak - nav) / peak, 4) if peak > 0 else 0.0
    metrics = compute_cumulative_metrics(daily_log)

    perf["current_nav"] = nav
    perf["peak_nav"] = peak
    perf["current_drawdown"] = current_dd
    perf["total_return"] = round((nav / initial_nav - 1.0) * 100, 4)
    perf["total_trades"] = total_trades
    perf["current_sharpe"] = metrics.get("sharpe")
    paper["performance"] = perf
    paper["daily_log"] = daily_log
    return paper


def main():  # noqa: PLR0912
    parser = argparse.ArgumentParser(description="Paper trading batch runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show signals without writing files",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD) to run for. Defaults to latest available "
        "trading day. Use to backfill missed days; entries inserted in "
        "chronological order with NAV chain rebuilt.",
    )
    args = parser.parse_args()

    today = datetime.date.today()
    today_str = str(today)
    logger.info("Paper trading batch run: %s", today_str)

    # 1. Fetch shared data
    prices_df, sym_data, spy_dates_full = load_shared_data()

    if not spy_dates_full:
        logger.error("No SPY data available. Aborting.")
        sys.exit(1)

    # Resolve target date
    if args.date:
        try:
            target_date = datetime.date.fromisoformat(args.date)
        except ValueError:
            logger.exception("Invalid --date %r, expected YYYY-MM-DD", args.date)
            sys.exit(1)
        spy_dates = [d for d in spy_dates_full if d <= target_date]
        if not spy_dates or spy_dates[-1] != target_date:
            logger.error(
                "Target date %s not in SPY trading calendar (nearest <= %s). Aborting.",
                target_date,
                spy_dates[-1] if spy_dates else "none",
            )
            sys.exit(1)
        logger.info("Backfill mode: target date %s", target_date)
    else:
        spy_dates = spy_dates_full

    latest_date = spy_dates[-1]
    logger.info("Latest data date: %s", latest_date)

    # 2. Compute signals for all strategies (using sliced calendar)
    signals = run_all_signals(sym_data, spy_dates, prices_df)

    # 3. Process each strategy
    summary_rows = []
    backfill = args.date is not None

    for slug in sorted(signals.keys()):
        sig = signals[slug]
        alloc = sig.get("allocation", {})

        # Compute signal-date return from allocation (uses dates[-1] vs dates[-2])
        daily_ret = compute_daily_return(alloc, sym_data, spy_dates)

        # Load existing YAML
        paper = load_paper_yaml(slug)
        daily_log = paper.get("daily_log", [])

        # Idempotency: skip if target date already logged
        existing_dates = {str(e.get("date", "")) for e in daily_log}
        latest_str = str(latest_date)
        if latest_str in existing_dates:
            metrics = compute_cumulative_metrics(daily_log)
            summary_rows.append(
                {
                    "slug": slug,
                    "family": MECHANISM_FAMILIES.get(slug, "?"),
                    "regime": sig.get("regime", "?"),
                    "position": sig.get("position", "?"),
                    "weight": sig.get("weight", 0.0),
                    "daily_ret": 0.0,
                    "sharpe": metrics.get("sharpe"),
                    "max_dd": metrics.get("max_dd", 0.0),
                    "days": metrics.get("days", 0),
                    "status": "SKIP (already logged)",
                }
            )
            continue

        # Find insertion position (chronological); for append-mode this is len(daily_log)
        insert_pos = len(daily_log)
        for i, e in enumerate(daily_log):
            if str(e.get("date", "")) > latest_str:
                insert_pos = i
                break

        # Previous-entry regime for switch cost (the entry that will precede the new one)
        prev_regime = None
        if insert_pos > 0:
            prev_regime = daily_log[insert_pos - 1].get("regime")

        # Apply switch cost
        if prev_regime is not None and sig.get("regime") != prev_regime:
            daily_ret -= COST_PER_SWITCH

        daily_ret_pct = round(daily_ret * 100, 4)

        # Build log entry (nav/day_number/pnl filled in by rebuild)
        log_entry = {
            "date": latest_str,
            "day_number": 0,  # placeholder — rebuild sets this
            "position": sig.get("position", "flat"),
            "regime": sig.get("regime", "unknown"),
            "nav": 0.0,  # placeholder
            "daily_pnl": 0.0,  # placeholder
            "daily_return_pct": daily_ret_pct,
            "signal_desc": sig.get("signal_desc", ""),
            "allocation": alloc,
        }
        daily_log.insert(insert_pos, log_entry)
        paper["daily_log"] = daily_log

        # Rebuild NAV chain and performance block from full daily_log
        paper = rebuild_performance(paper)
        metrics = compute_cumulative_metrics(daily_log)

        paper["updated_at"] = today_str
        if backfill:
            paper.setdefault("backfilled_dates", [])
            if latest_str not in paper["backfilled_dates"]:
                paper["backfilled_dates"].append(latest_str)

        # Save
        save_paper_yaml(slug, paper, dry_run=args.dry_run)

        summary_rows.append(
            {
                "slug": slug,
                "family": MECHANISM_FAMILIES.get(slug, "?"),
                "regime": sig.get("regime", "?"),
                "position": sig.get("position", "?"),
                "weight": sig.get("weight", 0.0),
                "daily_ret": daily_ret_pct,
                "sharpe": metrics.get("sharpe"),
                "max_dd": metrics.get("max_dd", 0.0),
                "days": metrics.get("days", 0),
                "status": "DRY-RUN" if args.dry_run else "OK",
            }
        )

    # 4. Print summary table
    print()
    print("=" * 120)
    print(f"PAPER TRADING BATCH — {today_str} ({len(summary_rows)} strategies)")
    print("=" * 120)
    header = (
        f"{'Strategy':<38} {'Fam':>3} {'Regime':<18} {'Wt':>5} "
        f"{'DayRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'Days':>5} {'Status':<10}"
    )
    print(header)
    print("-" * 120)

    for row in summary_rows:
        sharpe_str = f"{row['sharpe']:.3f}" if row["sharpe"] is not None else "  n/a"
        print(
            f"{row['slug']:<38} {row['family']:>3} {row['regime']:<18} "
            f"{row['weight']:>5.0%} {row['daily_ret']:>+7.2f}% "
            f"{sharpe_str:>7} {row['max_dd']:>6.2%} {row['days']:>5} "
            f"{row['status']:<10}"
        )

    print("-" * 120)

    # Aggregate stats
    active_rets = [
        r["daily_ret"] for r in summary_rows if r["status"] != "SKIP (already logged)"
    ]
    if active_rets:
        avg_ret = sum(active_rets) / len(active_rets)
        risk_on = sum(1 for r in summary_rows if "risk_on" in str(r["regime"]))
        risk_off = sum(1 for r in summary_rows if "risk_off" in str(r["regime"]))
        print(
            f"Avg daily return: {avg_ret:+.3f}% | "
            f"Risk-on: {risk_on} | Risk-off: {risk_off} | "
            f"Total: {len(summary_rows)}"
        )

    if args.dry_run:
        print("\n[DRY RUN — no files were written]")

    print()


if __name__ == "__main__":
    main()

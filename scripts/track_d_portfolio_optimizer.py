#!/usr/bin/env python3
"""Track D Portfolio Optimizer: find optimal combination of passing Track D strategies.

Backtests each strategy individually, computes pairwise correlations, and finds
the portfolio combination that maximizes CAGR while keeping MaxDD < 40%.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_portfolio_optimizer.py
"""

from __future__ import annotations

import itertools
import logging
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 5 * 365
COST_BPS = 20  # 20 bps per trade round-trip
COST_PER_SWITCH = COST_BPS / 10000
INITIAL_NAV = 100_000.0
TRADING_DAYS_PER_YEAR = 252
VIX_CRASH_THRESHOLD = 30.0  # Universal VIX crash filter

# ---------------------------------------------------------------------------
# All symbols needed across all Track D strategies
# ---------------------------------------------------------------------------
ALL_SYMBOLS = sorted(
    {
        "SPY",
        "QQQ",
        "TLT",
        "SHY",
        "GLD",
        "IEF",
        "TIP",
        "DBA",
        "XLK",
        "XLE",
        "AGG",
        "VCIT",
        "SOXX",
        "TQQQ",
        "TMF",
        "UPRO",
        "SOXL",
        "VIX",
    }
)

# ---------------------------------------------------------------------------
# Strategy definitions — each is a callable that returns daily allocations
# ---------------------------------------------------------------------------


def _get_close(sym_data: dict, symbol: str, d) -> float:
    return sym_data.get(symbol, {}).get(d, 0.0)


def _get_return(sym_data: dict, symbol: str, d, prev_d) -> float:
    p1 = _get_close(sym_data, symbol, d)
    p0 = _get_close(sym_data, symbol, prev_d)
    if p0 > 0 and p1 > 0:
        return p1 / p0 - 1
    return 0.0


# ---------------------------------------------------------------------------
# Strategy 1: TLT-TQQQ Sprint (lead-lag, TLT -> TQQQ)
# lag=3, window=10, entry=0.01, exit=-0.005, weight=0.30
# ---------------------------------------------------------------------------
def backtest_tlt_tqqq_sprint(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.30,
) -> list[float]:
    """TLT-TQQQ sprint: TLT 10d return leads TQQQ with 3-day lag."""
    window = 10
    entry_thresh = 0.01
    exit_thresh = -0.005
    follower = "TQQQ"
    leader = "TLT"

    daily_returns = []
    in_position = False

    for i in range(max(window + 5, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        # VIX crash filter
        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            # 100% SHY
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

        # Leader return as of yesterday (causal, lagged 3 days)
        lag_d = dates[i - 1 - 3] if (i - 1 - 3) >= 0 else dates[0]
        lb_d = dates[i - 1 - 3 - window] if (i - 1 - 3 - window) >= 0 else dates[0]
        leader_now = _get_close(sym_data, leader, lag_d)
        leader_lb = _get_close(sym_data, leader, lb_d)

        if leader_now <= 0 or leader_lb <= 0:
            daily_returns.append(0.0)
            continue

        leader_ret = leader_now / leader_lb - 1

        was_in_position = in_position
        if leader_ret >= entry_thresh:
            in_position = True
        elif leader_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(sym_data, follower, d, prev_d) * target_weight
            r += _get_return(sym_data, "SHY", d, prev_d) * (1.0 - target_weight)
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        # Transaction cost on switches
        if in_position != was_in_position:
            r -= COST_PER_SWITCH

        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 2: TLT-TQQQ Leveraged Lead-Lag (lag=5, window=10)
# ---------------------------------------------------------------------------
def backtest_tlt_tqqq_leveraged(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.30,
) -> list[float]:
    """TLT-TQQQ leveraged lead-lag: lag=5, window=10."""
    window = 10
    entry_thresh = 0.01
    exit_thresh = -0.01
    follower = "TQQQ"
    leader = "TLT"

    daily_returns = []
    in_position = False

    for i in range(max(window + 7, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

        lag_d = dates[i - 1 - 5] if (i - 1 - 5) >= 0 else dates[0]
        lb_d = dates[i - 1 - 5 - window] if (i - 1 - 5 - window) >= 0 else dates[0]
        leader_now = _get_close(sym_data, leader, lag_d)
        leader_lb = _get_close(sym_data, leader, lb_d)

        if leader_now <= 0 or leader_lb <= 0:
            daily_returns.append(0.0)
            continue

        leader_ret = leader_now / leader_lb - 1

        was_in_position = in_position
        if leader_ret >= entry_thresh:
            in_position = True
        elif leader_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(sym_data, follower, d, prev_d) * target_weight
            r += _get_return(sym_data, "SHY", d, prev_d) * (1.0 - target_weight)
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        if in_position != was_in_position:
            r -= COST_PER_SWITCH

        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 3: D3 TQQQ/TMF Ratio Mean-Reversion
# ---------------------------------------------------------------------------
def backtest_d3_tqqq_tmf_ratio_mr(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.30,
) -> list[float]:
    """D3: TQQQ/TMF ratio z-score mean-reversion with VIX crash filter."""
    lookback = 60
    z_threshold = 1.0

    # Build ratio series
    ratio_by_date = {}
    for d in dates:
        tqqq = _get_close(sym_data, "TQQQ", d)
        tmf = _get_close(sym_data, "TMF", d)
        if tqqq > 0 and tmf > 0:
            ratio_by_date[d] = tqqq / tmf

    daily_returns = []
    prev_regime = "neutral"

    for i in range(lookback + 10, len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if prev_regime != "crash":
                r -= COST_PER_SWITCH
            prev_regime = "crash"
            daily_returns.append(r)
            continue

        # Compute z-score as of yesterday
        window_ratios = []
        for j in range(i - lookback, i):
            dd = dates[j]
            if dd in ratio_by_date:
                window_ratios.append(ratio_by_date[dd])

        if len(window_ratios) < lookback // 2:
            daily_returns.append(0.0)
            continue

        sma = sum(window_ratios) / len(window_ratios)
        std = (sum((x - sma) ** 2 for x in window_ratios) / len(window_ratios)) ** 0.5
        if std <= 0:
            daily_returns.append(0.0)
            continue

        current_ratio = ratio_by_date.get(prev_d, sma)
        z = (current_ratio - sma) / std

        if z < -z_threshold:
            regime = "long_tmf"
            alloc = {"TMF": target_weight, "SHY": 1.0 - target_weight}
        elif z > z_threshold:
            regime = "long_tqqq"
            alloc = {"TQQQ": target_weight, "SHY": 1.0 - target_weight}
        else:
            half = target_weight / 2
            regime = "neutral"
            alloc = {"TQQQ": half, "TMF": half, "SHY": 1.0 - target_weight}

        r = sum(_get_return(sym_data, sym, d, prev_d) * w for sym, w in alloc.items())

        if regime != prev_regime:
            r -= COST_PER_SWITCH

        prev_regime = regime
        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 4: IEF-TQQQ Sprint (lead-lag, IEF -> TQQQ)
# lag=2, window=10, entry=0.005, exit=-0.003, weight=0.30
# ---------------------------------------------------------------------------
def backtest_ief_tqqq_sprint(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.30,
) -> list[float]:
    """IEF-TQQQ sprint: IEF 10d return leads TQQQ with 2-day lag."""
    window = 10
    entry_thresh = 0.005
    exit_thresh = -0.003
    follower = "TQQQ"
    leader = "IEF"

    daily_returns = []
    in_position = False

    for i in range(max(window + 5, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

        lag_d = dates[i - 1 - 2] if (i - 1 - 2) >= 0 else dates[0]
        lb_d = dates[i - 1 - 2 - window] if (i - 1 - 2 - window) >= 0 else dates[0]
        leader_now = _get_close(sym_data, leader, lag_d)
        leader_lb = _get_close(sym_data, leader, lb_d)

        if leader_now <= 0 or leader_lb <= 0:
            daily_returns.append(0.0)
            continue

        leader_ret = leader_now / leader_lb - 1

        was_in_position = in_position
        if leader_ret >= entry_thresh:
            in_position = True
        elif leader_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(sym_data, follower, d, prev_d) * target_weight
            r += _get_return(sym_data, "SHY", d, prev_d) * (1.0 - target_weight)
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        if in_position != was_in_position:
            r -= COST_PER_SWITCH

        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 5: VCIT-TQQQ Sprint (lead-lag, VCIT -> TQQQ)
# lag=3, window=10, entry=0.005, exit=-0.003, weight=0.30
# ---------------------------------------------------------------------------
def backtest_vcit_tqqq_sprint(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.30,
) -> list[float]:
    """VCIT-TQQQ sprint: VCIT 10d return leads TQQQ with 3-day lag."""
    window = 10
    entry_thresh = 0.005
    exit_thresh = -0.003
    follower = "TQQQ"
    leader = "VCIT"

    daily_returns = []
    in_position = False

    for i in range(max(window + 5, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

        lag_d = dates[i - 1 - 3] if (i - 1 - 3) >= 0 else dates[0]
        lb_d = dates[i - 1 - 3 - window] if (i - 1 - 3 - window) >= 0 else dates[0]
        leader_now = _get_close(sym_data, leader, lag_d)
        leader_lb = _get_close(sym_data, leader, lb_d)

        if leader_now <= 0 or leader_lb <= 0:
            daily_returns.append(0.0)
            continue

        leader_ret = leader_now / leader_lb - 1

        was_in_position = in_position
        if leader_ret >= entry_thresh:
            in_position = True
        elif leader_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(sym_data, follower, d, prev_d) * target_weight
            r += _get_return(sym_data, "SHY", d, prev_d) * (1.0 - target_weight)
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        if in_position != was_in_position:
            r -= COST_PER_SWITCH

        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 6: TSMOM-UPRO Trend (skip-month momentum -> UPRO)
# ---------------------------------------------------------------------------
def backtest_tsmom_upro(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.40,
) -> list[float]:
    """TSMOM -> UPRO: skip-month momentum with vol-scaling."""
    mom_lookback = 252
    skip_period = 21
    max_weight = 0.50
    vol_target = 0.15
    vol_window = 20
    _vix_crash = 25  # This strategy uses 25

    daily_returns = []
    prev_regime = "neutral"

    for i in range(max(mom_lookback + skip_period + 5, 300), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        # VIX crash filter (uses 30 universally per instructions)
        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if prev_regime != "crash":
                r -= COST_PER_SWITCH
            prev_regime = "crash"
            daily_returns.append(r)
            continue

        # Compute skip-month momentum: SPY return from t-252 to t-21
        idx_end = i - 1 - skip_period
        idx_start = i - 1 - mom_lookback
        if idx_end < 0 or idx_start < 0:
            daily_returns.append(0.0)
            continue

        spy_end = _get_close(sym_data, "SPY", dates[idx_end])
        spy_start = _get_close(sym_data, "SPY", dates[idx_start])

        if spy_start <= 0 or spy_end <= 0:
            daily_returns.append(0.0)
            continue

        mom = spy_end / spy_start - 1

        if mom > 0:
            regime = "bullish"
            # Vol-scaling
            rets = []
            for j in range(max(1, i - 1 - vol_window), i - 1):
                rets.append(_get_return(sym_data, "UPRO", dates[j], dates[j - 1]))
            if len(rets) >= 10:
                mean_r = sum(rets) / len(rets)
                vol = (sum((r - mean_r) ** 2 for r in rets) / len(rets)) ** 0.5
                ann_vol = vol * math.sqrt(252)
                if ann_vol > 0:
                    vol_scaled_w = min(
                        target_weight * (vol_target / ann_vol), max_weight
                    )
                else:
                    vol_scaled_w = target_weight
            else:
                vol_scaled_w = target_weight

            upro_w = vol_scaled_w
            shy_w = 1.0 - upro_w
            r = (
                _get_return(sym_data, "UPRO", d, prev_d) * upro_w
                + _get_return(sym_data, "SHY", d, prev_d) * shy_w
            )
        else:
            regime = "bearish"
            # Defensive: 30% GLD + 30% TLT + 40% SHY
            r = (
                _get_return(sym_data, "GLD", d, prev_d) * 0.30
                + _get_return(sym_data, "TLT", d, prev_d) * 0.30
                + _get_return(sym_data, "SHY", d, prev_d) * 0.40
            )

        if regime != prev_regime:
            r -= COST_PER_SWITCH
        prev_regime = regime
        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 7: TIP/TLT -> UPRO Real Yield
# ---------------------------------------------------------------------------
def backtest_tip_tlt_upro(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.35,
) -> list[float]:
    """TIP/TLT real yield regime -> UPRO."""
    lookback = 15

    daily_returns = []
    prev_regime = "neutral"

    for i in range(max(lookback + 5, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if prev_regime != "crash":
                r -= COST_PER_SWITCH
            prev_regime = "crash"
            daily_returns.append(r)
            continue

        # TIP/TLT ratio momentum
        tip_now = _get_close(sym_data, "TIP", prev_d)
        tlt_now = _get_close(sym_data, "TLT", prev_d)
        tip_lb = _get_close(sym_data, "TIP", dates[i - 1 - lookback])
        tlt_lb = _get_close(sym_data, "TLT", dates[i - 1 - lookback])

        if tip_now <= 0 or tlt_now <= 0 or tip_lb <= 0 or tlt_lb <= 0:
            daily_returns.append(0.0)
            continue

        ratio_now = tip_now / tlt_now
        ratio_lb = tip_lb / tlt_lb
        ratio_mom = ratio_now / ratio_lb - 1

        if ratio_mom < 0:
            # Loosening (real yields falling) -> UPRO
            regime = "loosening"
            r = _get_return(sym_data, "UPRO", d, prev_d) * target_weight + _get_return(
                sym_data, "SHY", d, prev_d
            ) * (1.0 - target_weight)
        else:
            # Tightening -> defensive
            regime = "tightening"
            r = (
                _get_return(sym_data, "GLD", d, prev_d) * 0.40
                + _get_return(sym_data, "SHY", d, prev_d) * 0.50
                + _get_return(sym_data, "TLT", d, prev_d) * 0.10
            )

        if regime != prev_regime:
            r -= COST_PER_SWITCH
        prev_regime = regime
        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 8: XLK/XLE -> SOXL Sector Rotation
# ---------------------------------------------------------------------------
def backtest_xlk_xle_soxl_rotation(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.40,
) -> list[float]:
    """XLK/XLE ratio momentum -> SOXL/GLD/UPRO rotation."""
    lookback = 40
    sma_period = 20

    daily_returns = []
    prev_regime = "neutral"

    for i in range(max(lookback + sma_period + 5, 100), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if prev_regime != "crash":
                r -= COST_PER_SWITCH
            prev_regime = "crash"
            daily_returns.append(r)
            continue

        # XLK/XLE ratio and its SMA
        ratios = []
        for j in range(i - lookback - sma_period, i):
            xlk = _get_close(sym_data, "XLK", dates[j])
            xle = _get_close(sym_data, "XLE", dates[j])
            if xlk > 0 and xle > 0:
                ratios.append(xlk / xle)

        if len(ratios) < sma_period + 5:
            daily_returns.append(0.0)
            continue

        current_ratio = ratios[-1]
        sma_val = sum(ratios[-sma_period:]) / sma_period

        # Momentum: current vs lookback-ago
        if len(ratios) >= lookback:
            ratio_mom = ratios[-1] / ratios[-lookback] - 1
        else:
            ratio_mom = 0.0

        if current_ratio > sma_val and ratio_mom > 0:
            # Growth regime -> SOXL
            regime = "growth"
            r = (
                _get_return(sym_data, "SOXL", d, prev_d) * target_weight
                + _get_return(sym_data, "SHY", d, prev_d) * 0.10
                + _get_return(sym_data, "SHY", d, prev_d) * (0.50 - target_weight)
            )
        elif current_ratio < sma_val and ratio_mom < 0:
            # Inflation regime -> GLD + DBA
            regime = "inflation"
            r = (
                _get_return(sym_data, "GLD", d, prev_d) * 0.30
                + _get_return(sym_data, "DBA", d, prev_d) * 0.20
                + _get_return(sym_data, "SHY", d, prev_d) * 0.50
            )
        else:
            # Neutral -> moderate UPRO + SHY
            regime = "neutral"
            r = (
                _get_return(sym_data, "UPRO", d, prev_d) * 0.25
                + _get_return(sym_data, "SHY", d, prev_d) * 0.75
            )

        if regime != prev_regime:
            r -= COST_PER_SWITCH
        prev_regime = regime
        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Strategy 9: SOXX-SOXL Lead-Lag
# ---------------------------------------------------------------------------
def backtest_soxx_soxl_lead_lag(
    sym_data: dict,
    dates: list,
    target_weight: float = 0.50,
) -> list[float]:
    """SOXX -> SOXL lead-lag: 7d return > 3% -> long SOXL."""
    window = 7
    entry_thresh = 0.03
    exit_thresh = -0.01
    _vix_crash = 35.0  # This strategy uses 35, but we override to 30

    daily_returns = []
    in_position = False

    for i in range(max(window + 5, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

        # SOXX 7-day return
        lb_idx = i - 1 - window
        if lb_idx < 0:
            daily_returns.append(0.0)
            continue

        soxx_now = _get_close(sym_data, "SOXX", prev_d)
        soxx_lb = _get_close(sym_data, "SOXX", dates[lb_idx])

        if soxx_now <= 0 or soxx_lb <= 0:
            daily_returns.append(0.0)
            continue

        soxx_ret = soxx_now / soxx_lb - 1

        was_in_position = in_position
        if soxx_ret >= entry_thresh:
            in_position = True
        elif soxx_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(sym_data, "SOXL", d, prev_d) * target_weight + _get_return(
                sym_data, "SHY", d, prev_d
            ) * (1.0 - target_weight)
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        if in_position != was_in_position:
            r -= COST_PER_SWITCH

        daily_returns.append(r)

    return daily_returns


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(daily_returns: list[float]) -> dict:
    """Compute CAGR, MaxDD, Sharpe, Calmar, Sortino from daily returns."""
    if not daily_returns or len(daily_returns) < 20:
        return {
            "cagr": 0,
            "max_dd": 1.0,
            "sharpe": 0,
            "calmar": 0,
            "sortino": 0,
            "total_return": 0,
            "ann_vol": 0,
            "n_days": 0,
        }

    n = len(daily_returns)
    years = n / TRADING_DAYS_PER_YEAR

    # NAV curve
    nav = [1.0]
    for r in daily_returns:
        nav.append(nav[-1] * (1 + r))

    total_return = nav[-1] / nav[0] - 1
    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe
    mean_r = sum(daily_returns) / n
    var = sum((r - mean_r) ** 2 for r in daily_returns) / n
    std = var**0.5
    ann_vol = std * math.sqrt(252)
    sharpe = (mean_r / std) * math.sqrt(252) if std > 0 else 0

    # Sortino (downside deviation)
    downside = [r for r in daily_returns if r < 0]
    if downside:
        ds_var = sum(r**2 for r in downside) / n  # Use full n for proper scaling
        ds_std = ds_var**0.5
        sortino = (mean_r / ds_std) * math.sqrt(252) if ds_std > 0 else 0
    else:
        sortino = sharpe * 2

    calmar = cagr / max_dd if max_dd > 0 else 0

    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "sortino": sortino,
        "total_return": total_return,
        "ann_vol": ann_vol,
        "n_days": n,
    }


def compute_correlation(r1: list[float], r2: list[float]) -> float:
    """Compute Pearson correlation between two return series (aligned by length)."""
    min_len = min(len(r1), len(r2))
    if min_len < 20:
        return 0.0
    # Use the last min_len days (most recent overlap)
    a = r1[-min_len:]
    b = r2[-min_len:]

    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
    std_a = (sum((x - mean_a) ** 2 for x in a) / n) ** 0.5
    std_b = (sum((x - mean_b) ** 2 for x in b) / n) ** 0.5
    if std_a > 0 and std_b > 0:
        return cov / (std_a * std_b)
    return 0.0


def combine_portfolios_equal_weight(
    return_streams: dict[str, list[float]],
    selected: list[str],
) -> list[float]:
    """Combine return streams using equal weight."""
    n_strats = len(selected)
    if n_strats == 0:
        return []

    weight = 1.0 / n_strats

    # Align by using the shortest common length
    min_len = min(len(return_streams[s]) for s in selected)
    combined = []
    for i in range(min_len):
        r = sum(return_streams[s][-min_len + i] * weight for s in selected)
        combined.append(r)

    return combined


def hrp_weights(
    return_streams: dict[str, list[float]],
    selected: list[str],
) -> dict[str, float]:
    """Compute Hierarchical Risk Parity weights (simplified).

    Uses inverse-variance weighting with correlation-based clustering.
    This is a simplified HRP that computes inverse-vol weights
    adjusted for pairwise correlations.
    """
    n = len(selected)
    if n <= 1:
        return dict.fromkeys(selected, 1.0)

    # Compute covariance matrix
    min_len = min(len(return_streams[s]) for s in selected)
    returns = {s: return_streams[s][-min_len:] for s in selected}

    # Inverse variance weights
    inv_var = {}
    for s in selected:
        rets = returns[s]
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        inv_var[s] = 1.0 / var if var > 0 else 1.0

    total = sum(inv_var.values())
    return {s: inv_var[s] / total for s in selected}


def combine_portfolios_hrp(
    return_streams: dict[str, list[float]],
    selected: list[str],
) -> list[float]:
    """Combine return streams using HRP weights."""
    weights = hrp_weights(return_streams, selected)

    min_len = min(len(return_streams[s]) for s in selected)
    combined = []
    for i in range(min_len):
        r = sum(return_streams[s][-min_len + i] * weights[s] for s in selected)
        combined.append(r)

    return combined


# ---------------------------------------------------------------------------
# TQQQ buy-and-hold benchmark
# ---------------------------------------------------------------------------
def backtest_tqqq_buyhold(sym_data: dict, dates: list) -> list[float]:
    """TQQQ buy-and-hold benchmark."""
    daily_returns = []
    for i in range(60, len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]
        r = _get_return(sym_data, "TQQQ", d, prev_d)
        daily_returns.append(r)
    return daily_returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():  # noqa: PLR0912
    print("=" * 80)
    print("TRACK D PORTFOLIO OPTIMIZER")
    print("=" * 80)
    print()

    # --- Load data ---
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

    spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
    dates = spy_df["date"].to_list()
    logger.info(
        "Trading calendar: %d dates from %s to %s", len(dates), dates[0], dates[-1]
    )

    # --- Backtest all strategies at base weight ---
    STRATEGIES = {
        "TLT-TQQQ Sprint": lambda sd, dd: backtest_tlt_tqqq_sprint(sd, dd, 0.30),
        "TLT-TQQQ Leveraged": lambda sd, dd: backtest_tlt_tqqq_leveraged(sd, dd, 0.30),
        "D3 TQQQ/TMF Ratio MR": lambda sd, dd: backtest_d3_tqqq_tmf_ratio_mr(
            sd, dd, 0.30
        ),
        "IEF-TQQQ Sprint": lambda sd, dd: backtest_ief_tqqq_sprint(sd, dd, 0.30),
        "VCIT-TQQQ Sprint": lambda sd, dd: backtest_vcit_tqqq_sprint(sd, dd, 0.30),
        "TSMOM-UPRO Trend": lambda sd, dd: backtest_tsmom_upro(sd, dd, 0.40),
        "TIP/TLT-UPRO RealYld": lambda sd, dd: backtest_tip_tlt_upro(sd, dd, 0.35),
        "XLK/XLE-SOXL Rotation": lambda sd, dd: backtest_xlk_xle_soxl_rotation(
            sd, dd, 0.40
        ),
        "SOXX-SOXL Lead-Lag": lambda sd, dd: backtest_soxx_soxl_lead_lag(sd, dd, 0.50),
    }

    return_streams: dict[str, list[float]] = {}
    individual_metrics: dict[str, dict] = {}

    print("\n" + "=" * 80)
    print("INDIVIDUAL STRATEGY RESULTS (base weight)")
    print("=" * 80)
    header = f"{'Strategy':<28} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sortino':>8} {'TotRet':>8} {'N days':>7}"
    print(header)
    print("-" * len(header))

    for name, backtest_fn in STRATEGIES.items():
        rets = backtest_fn(sym_data, dates)
        return_streams[name] = rets
        m = compute_metrics(rets)
        individual_metrics[name] = m
        print(
            f"{name:<28} {m['sharpe']:>7.3f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} "
            f"{m['calmar']:>7.2f} {m['sortino']:>8.3f} {m['total_return']:>7.1%} {m['n_days']:>7d}"
        )

    # TQQQ buy-and-hold benchmark
    tqqq_bh = backtest_tqqq_buyhold(sym_data, dates)
    tqqq_m = compute_metrics(tqqq_bh)
    print("-" * len(header))
    print(
        f"{'TQQQ Buy & Hold (benchmark)':<28} {tqqq_m['sharpe']:>7.3f} {tqqq_m['cagr']:>7.1%} "
        f"{tqqq_m['max_dd']:>7.1%} {tqqq_m['calmar']:>7.2f} {tqqq_m['sortino']:>8.3f} "
        f"{tqqq_m['total_return']:>7.1%} {tqqq_m['n_days']:>7d}"
    )

    # --- Weight variants ---
    print("\n" + "=" * 80)
    print("WEIGHT VARIANTS (select strategies at higher weights)")
    print("=" * 80)
    WEIGHT_TESTS = {
        "TLT-TQQQ Sprint": (backtest_tlt_tqqq_sprint, [0.30, 0.50, 0.70]),
        "D3 TQQQ/TMF Ratio MR": (backtest_d3_tqqq_tmf_ratio_mr, [0.30, 0.50, 0.70]),
        "TSMOM-UPRO Trend": (backtest_tsmom_upro, [0.40, 0.50, 0.60]),
        "XLK/XLE-SOXL Rotation": (backtest_xlk_xle_soxl_rotation, [0.40, 0.50, 0.60]),
        "SOXX-SOXL Lead-Lag": (backtest_soxx_soxl_lead_lag, [0.30, 0.50, 0.70]),
    }

    header2 = f"{'Strategy':<28} {'Weight':>6} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7}"
    print(header2)
    print("-" * len(header2))

    for name, (fn, weights) in WEIGHT_TESTS.items():
        for w in weights:
            rets = fn(sym_data, dates, w)
            m = compute_metrics(rets)
            dd_flag = " **" if m["max_dd"] > 0.40 else ""
            print(
                f"{name:<28} {w:>5.0%} {m['sharpe']:>7.3f} {m['cagr']:>7.1%} "
                f"{m['max_dd']:>7.1%} {m['calmar']:>7.2f}{dd_flag}"
            )

    # --- Pairwise correlation matrix ---
    print("\n" + "=" * 80)
    print("PAIRWISE CORRELATION MATRIX")
    print("=" * 80)
    strategy_names = list(return_streams.keys())
    short_names = [
        "TLT-TQQQ",
        "TLT-TQQQ2",
        "D3-MR",
        "IEF-TQQQ",
        "VCIT-TQQQ",
        "TSMOM",
        "TIP-UPRO",
        "XLK-SOXL",
        "SOXX-SOXL",
    ]

    # Print header
    print(f"{'':>12}", end="")
    for sn in short_names:
        print(f"{sn:>10}", end="")
    print()

    corr_matrix: dict[tuple[str, str], float] = {}
    for i, s1 in enumerate(strategy_names):
        print(f"{short_names[i]:>12}", end="")
        for _j, s2 in enumerate(strategy_names):
            corr = compute_correlation(return_streams[s1], return_streams[s2])
            corr_matrix[(s1, s2)] = corr
            print(f"{corr:>10.3f}", end="")
        print()

    # Avg pairwise correlation
    off_diag = [
        corr_matrix[(s1, s2)]
        for i, s1 in enumerate(strategy_names)
        for j, s2 in enumerate(strategy_names)
        if i < j
    ]
    avg_corr = sum(off_diag) / len(off_diag) if off_diag else 0
    print(f"\nAverage pairwise correlation: {avg_corr:.4f}")

    # --- Portfolio combinations: equal weight ---
    print("\n" + "=" * 80)
    print("PORTFOLIO COMBINATIONS (Equal Weight)")
    print("=" * 80)

    # Rank strategies by Sharpe for selection
    ranked = sorted(
        individual_metrics.items(),
        key=lambda x: x[1]["sharpe"],
        reverse=True,
    )
    ranked_names = [name for name, _ in ranked]

    print("\nStrategy ranking by Sharpe:")
    for i, (name, m) in enumerate(ranked):
        print(
            f"  {i + 1}. {name}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}"
        )

    # Test top-N combinations
    print(
        f"\n{'Combination':<55} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sortino':>8}"
    )
    print("-" * 100)

    best_cagr = 0
    best_combo = None
    best_metrics = None

    for n_strats in range(2, min(len(strategy_names) + 1, 10)):
        # Try top-N
        top_n = ranked_names[:n_strats]
        combined = combine_portfolios_equal_weight(return_streams, top_n)
        m = compute_metrics(combined)
        label = f"Top {n_strats} EW: {', '.join(s[:10] for s in top_n)}"
        dd_flag = " BREACH" if m["max_dd"] > 0.40 else ""
        print(
            f"{label:<55} {m['sharpe']:>7.3f} {m['cagr']:>7.1%} "
            f"{m['max_dd']:>7.1%} {m['calmar']:>7.2f} {m['sortino']:>8.3f}{dd_flag}"
        )
        if m["max_dd"] < 0.40 and m["cagr"] > best_cagr:
            best_cagr = m["cagr"]
            best_combo = top_n
            best_metrics = m

    # Also try specific hand-picked combinations based on low correlation
    print(f"\n{'Specific Low-Corr Combos:':<55}")
    print("-" * 100)

    CURATED_COMBOS = [
        # Mix different mechanism families
        ["D3 TQQQ/TMF Ratio MR", "TSMOM-UPRO Trend"],
        ["D3 TQQQ/TMF Ratio MR", "XLK/XLE-SOXL Rotation"],
        ["D3 TQQQ/TMF Ratio MR", "TSMOM-UPRO Trend", "XLK/XLE-SOXL Rotation"],
        ["D3 TQQQ/TMF Ratio MR", "TSMOM-UPRO Trend", "SOXX-SOXL Lead-Lag"],
        ["D3 TQQQ/TMF Ratio MR", "TLT-TQQQ Sprint", "TSMOM-UPRO Trend"],
        ["D3 TQQQ/TMF Ratio MR", "TLT-TQQQ Sprint", "XLK/XLE-SOXL Rotation"],
        [
            "D3 TQQQ/TMF Ratio MR",
            "TLT-TQQQ Sprint",
            "TSMOM-UPRO Trend",
            "XLK/XLE-SOXL Rotation",
        ],
        [
            "D3 TQQQ/TMF Ratio MR",
            "TLT-TQQQ Sprint",
            "TSMOM-UPRO Trend",
            "SOXX-SOXL Lead-Lag",
        ],
        ["TLT-TQQQ Sprint", "TSMOM-UPRO Trend", "XLK/XLE-SOXL Rotation"],
        ["TLT-TQQQ Sprint", "TSMOM-UPRO Trend", "TIP/TLT-UPRO RealYld"],
        # All 9
        list(return_streams.keys()),
    ]

    for combo in CURATED_COMBOS:
        valid = [s for s in combo if s in return_streams]
        if len(valid) < 2:
            continue
        combined = combine_portfolios_equal_weight(return_streams, valid)
        m = compute_metrics(combined)
        label = " + ".join(s[:12] for s in valid)
        if len(label) > 54:
            label = label[:51] + "..."
        dd_flag = " BREACH" if m["max_dd"] > 0.40 else ""
        print(
            f"{label:<55} {m['sharpe']:>7.3f} {m['cagr']:>7.1%} "
            f"{m['max_dd']:>7.1%} {m['calmar']:>7.2f} {m['sortino']:>8.3f}{dd_flag}"
        )
        if m["max_dd"] < 0.40 and m["cagr"] > best_cagr:
            best_cagr = m["cagr"]
            best_combo = valid
            best_metrics = m

    # --- HRP portfolio ---
    print("\n" + "=" * 80)
    print("HRP (Inverse-Variance) PORTFOLIOS")
    print("=" * 80)
    header3 = f"{'Combination':<55} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sortino':>8}"
    print(header3)
    print("-" * 100)

    HRP_COMBOS = [
        ranked_names[:3],
        ranked_names[:4],
        ranked_names[:5],
        ranked_names[:6],
        list(return_streams.keys()),
        ["D3 TQQQ/TMF Ratio MR", "TSMOM-UPRO Trend", "XLK/XLE-SOXL Rotation"],
        [
            "D3 TQQQ/TMF Ratio MR",
            "TLT-TQQQ Sprint",
            "TSMOM-UPRO Trend",
            "XLK/XLE-SOXL Rotation",
        ],
    ]

    for combo in HRP_COMBOS:
        valid = [s for s in combo if s in return_streams]
        if len(valid) < 2:
            continue
        weights = hrp_weights(return_streams, valid)
        combined = combine_portfolios_hrp(return_streams, valid)
        m = compute_metrics(combined)
        label = "HRP: " + " + ".join(s[:10] for s in valid)
        if len(label) > 54:
            label = label[:51] + "..."
        dd_flag = " BREACH" if m["max_dd"] > 0.40 else ""
        print(
            f"{label:<55} {m['sharpe']:>7.3f} {m['cagr']:>7.1%} "
            f"{m['max_dd']:>7.1%} {m['calmar']:>7.2f} {m['sortino']:>8.3f}{dd_flag}"
        )
        if m["max_dd"] < 0.40 and m["cagr"] > best_cagr:
            best_cagr = m["cagr"]
            best_combo = valid
            best_metrics = m

        # Print weights
        w_str = ", ".join(f"{s[:10]}={w:.1%}" for s, w in weights.items())
        print(f"  Weights: {w_str}")

    # --- Exhaustive search for max CAGR under MaxDD < 40% ---
    print("\n" + "=" * 80)
    print("EXHAUSTIVE SEARCH: MAX CAGR WITH MaxDD < 40%")
    print("=" * 80)

    best_ew_combos = []
    for n in range(2, min(len(strategy_names) + 1, 10)):
        for combo in itertools.combinations(strategy_names, n):
            combined = combine_portfolios_equal_weight(return_streams, list(combo))
            m = compute_metrics(combined)
            if m["max_dd"] < 0.40:
                best_ew_combos.append((list(combo), m))
                if m["cagr"] > best_cagr:
                    best_cagr = m["cagr"]
                    best_combo = list(combo)
                    best_metrics = m

    # Sort by CAGR descending
    best_ew_combos.sort(key=lambda x: x[1]["cagr"], reverse=True)

    print("\nTop 20 combinations by CAGR (MaxDD < 40%):")
    print(
        f"{'Rank':>4} {'N':>2} {'Combination':<55} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7}"
    )
    print("-" * 95)

    for rank, (combo, m) in enumerate(best_ew_combos[:20], 1):
        label = " + ".join(s[:10] for s in combo)
        if len(label) > 54:
            label = label[:51] + "..."
        print(
            f"{rank:>4} {len(combo):>2} {label:<55} {m['sharpe']:>7.3f} "
            f"{m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.2f}"
        )

    # --- Higher-weight portfolio optimization ---
    print("\n" + "=" * 80)
    print(
        "HIGHER-WEIGHT PORTFOLIO OPTIMIZATION (strategies at elevated internal weights)"
    )
    print("=" * 80)

    # Re-run best strategies at their optimal higher weights
    HIGH_WEIGHT_STRATS = {
        "XLK-SOXL@60%": lambda sd, dd: backtest_xlk_xle_soxl_rotation(sd, dd, 0.60),
        "XLK-SOXL@50%": lambda sd, dd: backtest_xlk_xle_soxl_rotation(sd, dd, 0.50),
        "TLT-TQQQ@70%": lambda sd, dd: backtest_tlt_tqqq_sprint(sd, dd, 0.70),
        "TLT-TQQQ@50%": lambda sd, dd: backtest_tlt_tqqq_sprint(sd, dd, 0.50),
        "TSMOM-UPRO@50%": lambda sd, dd: backtest_tsmom_upro(sd, dd, 0.50),
        "IEF-TQQQ@40%": lambda sd, dd: backtest_ief_tqqq_sprint(sd, dd, 0.40),
        "VCIT-TQQQ@40%": lambda sd, dd: backtest_vcit_tqqq_sprint(sd, dd, 0.40),
        "TIP-UPRO@45%": lambda sd, dd: backtest_tip_tlt_upro(sd, dd, 0.45),
    }

    high_streams: dict[str, list[float]] = {}
    high_metrics: dict[str, dict] = {}

    header_hw = f"{'Strategy@Weight':<28} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Sortino':>8}"
    print(header_hw)
    print("-" * len(header_hw))

    for name, fn in HIGH_WEIGHT_STRATS.items():
        rets = fn(sym_data, dates)
        high_streams[name] = rets
        m = compute_metrics(rets)
        high_metrics[name] = m
        print(
            f"{name:<28} {m['sharpe']:>7.3f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} "
            f"{m['calmar']:>7.2f} {m['sortino']:>8.3f}"
        )

    # Exhaustive search over high-weight combos
    print("\nBest high-weight portfolio combos (MaxDD < 40%, sorted by CAGR):")
    high_names = list(high_streams.keys())
    best_high_combos = []

    for n in range(2, min(len(high_names) + 1, 7)):
        for combo in itertools.combinations(high_names, n):
            combined = combine_portfolios_equal_weight(high_streams, list(combo))
            m = compute_metrics(combined)
            if m["max_dd"] < 0.40 and m["sharpe"] > 0.5:
                best_high_combos.append((list(combo), m))

    best_high_combos.sort(key=lambda x: x[1]["cagr"], reverse=True)

    print(
        f"{'Rank':>4} {'N':>2} {'Combination':<60} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7}"
    )
    print("-" * 100)

    for rank, (combo, m) in enumerate(best_high_combos[:25], 1):
        label = " + ".join(s for s in combo)
        if len(label) > 59:
            label = label[:56] + "..."
        print(
            f"{rank:>4} {len(combo):>2} {label:<60} {m['sharpe']:>7.3f} "
            f"{m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.2f}"
        )
        if m["cagr"] > best_cagr and m["max_dd"] < 0.40:
            best_cagr = m["cagr"]
            best_combo = combo
            best_metrics = m

    # Also try 50% CAGR target: what's the minimum-strategies combo that achieves it?
    print(f"\n{'Combos achieving CAGR > 50% with MaxDD < 40%:'}")
    over_50 = [(c, m) for c, m in best_high_combos if m["cagr"] > 0.50]
    if over_50:
        for combo, m in over_50:
            label = " + ".join(s for s in combo)
            print(
                f"  {label}: CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, Sharpe={m['sharpe']:.3f}"
            )
    else:
        print("  None found. Closest combos:")
        for combo, m in best_high_combos[:3]:
            label = " + ".join(s for s in combo)
            print(
                f"  {label}: CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, Sharpe={m['sharpe']:.3f}"
            )

    # --- Final recommendation ---
    print("\n" + "=" * 80)
    print("OPTIMAL PORTFOLIO RECOMMENDATION")
    print("=" * 80)

    if best_combo and best_metrics:
        print("\nBest combination (max CAGR, MaxDD < 40%):")
        print(f"  Strategies: {best_combo}")
        print(f"  N strategies: {len(best_combo)}")
        print(f"  Weight per strategy: {1 / len(best_combo):.1%}")
        print(f"  CAGR: {best_metrics['cagr']:.2%}")
        print(f"  MaxDD: {best_metrics['max_dd']:.2%}")
        print(f"  Sharpe: {best_metrics['sharpe']:.3f}")
        print(f"  Calmar: {best_metrics['calmar']:.2f}")
        print(f"  Sortino: {best_metrics['sortino']:.3f}")
        print(f"  Total Return: {best_metrics['total_return']:.1%}")
        print(f"  Ann. Vol: {best_metrics['ann_vol']:.2%}")
        print(f"  Backtest Length: {best_metrics['n_days']} days")

        # Print pairwise correlations within the optimal combo
        all_streams = {**return_streams, **high_streams}
        print("\n  Pairwise correlations within optimal portfolio:")
        for i, s1 in enumerate(best_combo):
            for j, s2 in enumerate(best_combo):
                if i < j and s1 in all_streams and s2 in all_streams:
                    corr = compute_correlation(all_streams[s1], all_streams[s2])
                    print(f"    {s1} x {s2}: {corr:.3f}")
    else:
        print("\nNo combination found meeting MaxDD < 40% constraint.")

    # Comparison vs benchmarks
    print("\n  vs TQQQ buy-and-hold:")
    print(
        f"    TQQQ CAGR: {tqqq_m['cagr']:.2%}, MaxDD: {tqqq_m['max_dd']:.2%}, Sharpe: {tqqq_m['sharpe']:.3f}"
    )
    if best_metrics:
        cagr_ratio = best_metrics["cagr"] / tqqq_m["cagr"] if tqqq_m["cagr"] != 0 else 0
        dd_ratio = (
            best_metrics["max_dd"] / tqqq_m["max_dd"] if tqqq_m["max_dd"] != 0 else 0
        )
        print(f"    CAGR ratio (portfolio/TQQQ): {cagr_ratio:.2f}x")
        print(f"    MaxDD ratio (portfolio/TQQQ): {dd_ratio:.2f}x")

    print("\n" + "=" * 80)
    print("D3 TQQQ/TMF RATIO MR -- INVESTIGATION NOTE")
    print("=" * 80)
    print("  The D3 robustness file claims Sharpe=2.21, but independent replication")
    print("  shows negative returns (Sharpe=-1.08). The original backtest engine")
    print("  (tqqq-tmf-ratio-reversion via PairsRatioStrategy) also shows")
    print("  Sharpe=0.08, MaxDD=43.5% -- all gates FAIL. The D3 robustness data")
    print("  is likely from an ad-hoc calculation with a signal direction bug.")
    print()
    print("  RECOMMENDATION: D3 should be RETIRED. Do not include in Track D")
    print("  portfolio until the signal is independently validated.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

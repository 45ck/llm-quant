#!/usr/bin/env python3
"""Track D Portfolio Optimizer — Combine ALL passing Track D strategies.

Reconstructs daily returns for all 12 Track D strategies, computes correlations,
and optimizes portfolio weights using HRP and Monte Carlo to maximize CAGR
while keeping MaxDD < 40%.

Strategies (9 confirmed + 3 untested but promising):
  1. TLT-TQQQ Sprint (lead_lag)       7. IEF-TQQQ Sprint (lead_lag)
  2. AGG-TQQQ Sprint (lead_lag)       8. VCIT-TQQQ Sprint (lead_lag)
  3. D3 TQQQ/TMF Ratio MR             9. TLT-UPRO Sprint (lead_lag)
  4. D10 XLK-XLE-SOXL Rotation       10. AGG-SOXL Sprint (lead_lag)
  5. D11 SOXX-SOXL Lead-Lag          11. TLT-SOXL Sprint (lead_lag)
  6. D12 TIP/TLT-UPRO Real Yield     12. D13 TSMOM-UPRO Trend

Key question: "What is the maximum CAGR achievable while keeping MaxDD under 40%?"

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_portfolio_optimizer.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl

from llm_quant.data.fetcher import fetch_ohlcv

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LOOKBACK_DAYS = 5 * 365  # 5 year backtest window
INITIAL_CAPITAL = 10_000.0  # $10k for growth projections
COST_BPS = 20  # round-trip cost in bps
COST_PER_SWITCH = COST_BPS / 10_000
VIX_CRASH_THRESHOLD = 30.0  # universal crash filter

ALL_SYMBOLS = sorted(
    {
        "AGG",
        "DBA",
        "GLD",
        "IEF",
        "SHY",
        "SOXX",
        "SOXL",
        "SPY",
        "TIP",
        "TLT",
        "TMF",
        "TQQQ",
        "UPRO",
        "VCIT",
        "VIX",
        "XLK",
        "XLE",
    }
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _get_close(sym_data: dict, symbol: str, d) -> float:
    return sym_data.get(symbol, {}).get(d, 0.0)


def _get_return(sym_data: dict, symbol: str, d, prev_d) -> float:
    p1 = _get_close(sym_data, symbol, d)
    p0 = _get_close(sym_data, symbol, prev_d)
    if p0 > 0 and p1 > 0:
        return p1 / p0 - 1
    return 0.0


def compute_metrics(daily_returns: list[float]) -> dict:
    """Compute CAGR, MaxDD, Sharpe, Calmar, Sortino from daily returns."""
    if not daily_returns or len(daily_returns) < 20:
        return {
            "cagr": 0.0,
            "max_dd": 1.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "sortino": 0.0,
            "total_return": 0.0,
            "ann_vol": 0.0,
            "n_days": 0,
        }

    n = len(daily_returns)
    years = n / 252.0

    # NAV curve
    nav = [1.0]
    for r in daily_returns:
        nav.append(nav[-1] * (1 + r))

    total_return = nav[-1] / nav[0] - 1
    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 and nav[-1] > 0 else 0

    # Max drawdown
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Sharpe (population std for consistency with existing scripts)
    mean_r = sum(daily_returns) / n
    var_r = sum((r - mean_r) ** 2 for r in daily_returns) / n
    std_r = var_r**0.5
    ann_vol = std_r * math.sqrt(252)
    sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0

    # Sortino
    downside = [r for r in daily_returns if r < 0]
    if downside:
        ds_var = sum(r**2 for r in downside) / n
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


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY BACKTESTS — self-contained, no registry dependency
# ══════════════════════════════════════════════════════════════════════════════


def _backtest_lead_lag(
    sym_data: dict,
    dates: list,
    leader: str,
    follower: str,
    lag_days: int = 3,
    signal_window: int = 10,
    entry_thresh: float = 0.01,
    exit_thresh: float = -0.005,
    target_weight: float = 0.30,
) -> list[float]:
    """Generic lead-lag backtest: leader N-day return -> follower position."""
    daily_returns: list[float] = []
    in_position = False
    warmup = max(signal_window + lag_days + 5, 60)

    for i in range(warmup, len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        # VIX crash filter
        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > VIX_CRASH_THRESHOLD:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

        # Leader return as of yesterday, lagged lag_days
        lag_idx = i - 1 - lag_days
        lb_idx = lag_idx - signal_window
        if lag_idx < 0 or lb_idx < 0:
            daily_returns.append(0.0)
            continue

        leader_now = _get_close(sym_data, leader, dates[lag_idx])
        leader_lb = _get_close(sym_data, leader, dates[lb_idx])
        if leader_now <= 0 or leader_lb <= 0:
            daily_returns.append(0.0)
            continue

        leader_ret = leader_now / leader_lb - 1

        was_in = in_position
        if leader_ret >= entry_thresh:
            in_position = True
        elif leader_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(
                sym_data, follower, d, prev_d
            ) * target_weight + _get_return(sym_data, "SHY", d, prev_d) * (
                1.0 - target_weight
            )
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        if in_position != was_in:
            r -= COST_PER_SWITCH
        daily_returns.append(r)

    return daily_returns


def backtest_d3_tqqq_tmf(
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

    daily_returns: list[float] = []
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


def backtest_d10_xlk_xle_soxl(
    sym_data: dict,
    dates: list,
    soxl_weight: float = 0.40,
) -> list[float]:
    """D10: XLK/XLE ratio momentum -> SOXL/GLD+DBA/UPRO rotation."""
    lookback = 40
    sma_period = 20

    daily_returns: list[float] = []
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

        # XLK/XLE ratio history
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
        ratio_mom = ratios[-1] / ratios[-lookback] - 1 if len(ratios) >= lookback else 0

        if current_ratio > sma_val and ratio_mom > 0:
            regime = "growth"
            shy_w = max(0.10, 1.0 - soxl_weight)
            r = (
                _get_return(sym_data, "SOXL", d, prev_d) * soxl_weight
                + _get_return(sym_data, "SHY", d, prev_d) * shy_w
            )
        elif current_ratio < sma_val and ratio_mom < 0:
            regime = "inflation"
            r = (
                _get_return(sym_data, "GLD", d, prev_d) * 0.30
                + _get_return(sym_data, "DBA", d, prev_d) * 0.20
                + _get_return(sym_data, "SHY", d, prev_d) * 0.50
            )
        else:
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


def backtest_d11_soxx_soxl(
    sym_data: dict,
    dates: list,
    soxl_weight: float = 0.50,
) -> list[float]:
    """D11: SOXX 7d return -> SOXL position, VIX crash filter at 35."""
    window = 7
    entry_thresh = 0.03
    exit_thresh = -0.01
    vix_thresh = 35.0  # D11 uses 35

    daily_returns: list[float] = []
    in_position = False

    for i in range(max(window + 5, 60), len(dates)):
        d = dates[i]
        prev_d = dates[i - 1]

        vix = _get_close(sym_data, "VIX", prev_d)
        if vix > vix_thresh:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0
            if in_position:
                r -= COST_PER_SWITCH
                in_position = False
            daily_returns.append(r)
            continue

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

        was_in = in_position
        if soxx_ret >= entry_thresh:
            in_position = True
        elif soxx_ret <= exit_thresh:
            in_position = False

        if in_position:
            r = _get_return(sym_data, "SOXL", d, prev_d) * soxl_weight + _get_return(
                sym_data, "SHY", d, prev_d
            ) * (1.0 - soxl_weight)
        else:
            r = _get_return(sym_data, "SHY", d, prev_d) * 1.0

        if in_position != was_in:
            r -= COST_PER_SWITCH
        daily_returns.append(r)

    return daily_returns


def backtest_d12_tip_tlt_upro(
    sym_data: dict,
    dates: list,
    upro_weight: float = 0.35,
) -> list[float]:
    """D12: TIP/TLT real yield regime -> UPRO/GLD+TLT rotation."""
    lookback = 15

    daily_returns: list[float] = []
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
        lb_idx = i - 1 - lookback
        if lb_idx < 0:
            daily_returns.append(0.0)
            continue

        tip_now = _get_close(sym_data, "TIP", prev_d)
        tlt_now = _get_close(sym_data, "TLT", prev_d)
        tip_lb = _get_close(sym_data, "TIP", dates[lb_idx])
        tlt_lb = _get_close(sym_data, "TLT", dates[lb_idx])

        if tip_now <= 0 or tlt_now <= 0 or tip_lb <= 0 or tlt_lb <= 0:
            daily_returns.append(0.0)
            continue

        ratio_mom = (tip_now / tlt_now) / (tip_lb / tlt_lb) - 1

        if ratio_mom < 0:
            # Loosening -> UPRO
            regime = "loosening"
            shy_w = 1.0 - upro_weight
            r = (
                _get_return(sym_data, "UPRO", d, prev_d) * upro_weight
                + _get_return(sym_data, "SHY", d, prev_d) * shy_w
            )
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


def backtest_d13_tsmom_upro(
    sym_data: dict,
    dates: list,
    base_upro_weight: float = 0.40,
) -> list[float]:
    """D13: Skip-month TSMOM -> UPRO with vol-scaling."""
    mom_lookback = 252
    skip_period = 21
    max_weight = 0.50
    vol_target = 0.15
    vol_window = 20

    daily_returns: list[float] = []
    prev_regime = "neutral"

    warmup = max(mom_lookback + skip_period + 5, 300)
    for i in range(warmup, len(dates)):
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

        # Skip-month momentum: SPY return from t-252 to t-21
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
            upro_w = base_upro_weight
            rets = []
            for j in range(max(1, i - 1 - vol_window), i - 1):
                rets.append(_get_return(sym_data, "UPRO", dates[j], dates[j - 1]))
            if len(rets) >= 10:
                mean_r = sum(rets) / len(rets)
                vol = (sum((r - mean_r) ** 2 for r in rets) / len(rets)) ** 0.5
                ann_vol = vol * math.sqrt(252)
                if ann_vol > 0:
                    upro_w = min(base_upro_weight * (vol_target / ann_vol), max_weight)

            shy_w = 1.0 - upro_w
            r = (
                _get_return(sym_data, "UPRO", d, prev_d) * upro_w
                + _get_return(sym_data, "SHY", d, prev_d) * shy_w
            )
        else:
            regime = "bearish"
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


def backtest_tqqq_buyhold(sym_data: dict, dates: list) -> list[float]:
    """TQQQ buy-and-hold benchmark."""
    daily_returns: list[float] = []
    for i in range(60, len(dates)):
        r = _get_return(sym_data, "TQQQ", dates[i], dates[i - 1])
        daily_returns.append(r)
    return daily_returns


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MATH
# ══════════════════════════════════════════════════════════════════════════════


def align_returns(
    return_streams: dict[str, list[float]],
    selected: list[str],
) -> np.ndarray:
    """Align return series to common length, return T x N matrix."""
    min_len = min(len(return_streams[s]) for s in selected)
    matrix = np.zeros((min_len, len(selected)))
    for j, name in enumerate(selected):
        matrix[:, j] = np.array(return_streams[name][-min_len:])
    return matrix


def combine_equal_weight(
    return_streams: dict[str, list[float]],
    selected: list[str],
) -> list[float]:
    """Combine return streams using equal weight."""
    n_strats = len(selected)
    if n_strats == 0:
        return []
    min_len = min(len(return_streams[s]) for s in selected)
    weight = 1.0 / n_strats
    combined = []
    for i in range(min_len):
        r = sum(return_streams[s][-min_len + i] * weight for s in selected)
        combined.append(r)
    return combined


def combine_weighted(
    return_streams: dict[str, list[float]],
    weights: dict[str, float],
) -> list[float]:
    """Combine return streams using specified weights."""
    selected = list(weights.keys())
    min_len = min(len(return_streams[s]) for s in selected)
    combined = []
    for i in range(min_len):
        r = sum(return_streams[s][-min_len + i] * weights[s] for s in selected)
        combined.append(r)
    return combined


def hrp_weights(
    return_streams: dict[str, list[float]],
    selected: list[str],
) -> dict[str, float]:
    """Hierarchical Risk Parity weights using scipy linkage.

    Full De Prado HRP: cluster by correlation distance, then recursive bisection
    with inverse-variance allocation.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    n = len(selected)
    if n <= 1:
        return dict.fromkeys(selected, 1.0)

    matrix = align_returns(return_streams, selected)
    cov = np.cov(matrix, rowvar=False)
    corr = np.corrcoef(matrix, rowvar=False)

    # Correlation distance
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist_condensed = squareform(dist, checks=False)
    dist_condensed = np.nan_to_num(dist_condensed, nan=0.0, posinf=1.0, neginf=0.0)

    link = linkage(dist_condensed, method="single")

    # Quasi-diagonalize
    link_int = link.astype(int)
    sort_ix = [int(link_int[-1, 0]), int(link_int[-1, 1])]
    n_items = int(link_int[-1, 3])
    while True:
        new_ix = []
        found = False
        for ix in sort_ix:
            if ix >= n_items:
                found = True
                row = link_int[ix - n_items]
                new_ix.append(int(row[0]))
                new_ix.append(int(row[1]))
            else:
                new_ix.append(ix)
        sort_ix = new_ix
        if not found:
            break

    # Recursive bisection
    w = np.ones(n)
    c_items = [sort_ix]
    while c_items:
        new_items = []
        for subset in c_items:
            if len(subset) <= 1:
                continue
            half = len(subset) // 2
            left = subset[:half]
            right = subset[half:]

            inv_left = 1.0 / np.diag(cov[np.ix_(left, left)])
            inv_right = 1.0 / np.diag(cov[np.ix_(right, right)])

            var_l = 1.0 / np.sum(inv_left) if np.sum(inv_left) > 0 else 1.0
            var_r = 1.0 / np.sum(inv_right) if np.sum(inv_right) > 0 else 1.0

            alpha = 1.0 - var_l / (var_l + var_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha

            if len(left) > 1:
                new_items.append(left)
            if len(right) > 1:
                new_items.append(right)
        c_items = new_items

    w = w / np.sum(w)
    return {selected[i]: float(w[i]) for i in range(n)}


def monte_carlo_optimize(
    return_streams: dict[str, list[float]],
    selected: list[str],
    objective: str = "sharpe",  # "sharpe" or "cagr"
    max_dd: float = 0.40,
    min_w: float = 0.03,
    max_w: float = 0.50,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[dict[str, float], dict]:
    """Monte Carlo weight optimization with MaxDD constraint."""
    matrix = align_returns(return_streams, selected)
    n_strats = len(selected)
    n_days = matrix.shape[0]
    rng = np.random.default_rng(seed)

    best_obj = -999.0
    best_w: dict[str, float] = {}
    best_m: dict = {}

    for _ in range(n_samples):
        raw = rng.uniform(min_w, max_w, size=n_strats)
        w = raw / np.sum(raw)
        w = np.clip(w, min_w, max_w)
        w = w / np.sum(w)

        combined = matrix @ w

        # NAV and MaxDD
        nav = np.cumprod(1.0 + combined)
        nav = np.insert(nav, 0, 1.0)
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak
        mdd = float(abs(np.min(dd)))
        if mdd > max_dd:
            continue

        mean_r = float(np.mean(combined))
        std_r = float(np.std(combined))
        if std_r <= 0:
            continue

        sharpe = mean_r / std_r * math.sqrt(252)
        years = n_days / 252.0
        cagr = float((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

        obj_val = sharpe if objective == "sharpe" else cagr

        if obj_val > best_obj:
            best_obj = obj_val
            best_w = {selected[i]: float(w[i]) for i in range(n_strats)}
            best_m = {
                "sharpe": sharpe,
                "max_dd": mdd,
                "cagr": cagr,
                "calmar": cagr / mdd if mdd > 0 else 0,
                "sortino": 0.0,
                "total_return": float(nav[-1] / nav[0] - 1.0),
                "ann_vol": float(std_r * math.sqrt(252)),
                "n_days": n_days,
            }

    return best_w, best_m


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def print_growth_table(cagr: float, label: str = "") -> None:
    """Print $10k growth projection at given CAGR."""
    capital = INITIAL_CAPITAL
    prefix = f"  {label} " if label else "  "
    vals = []
    for yr in [1, 2, 3, 5, 10]:
        vals.append(f"{yr}yr=${capital * (1 + cagr) ** yr:,.0f}")
    print(f"{prefix}$10k growth: {', '.join(vals)}")


def print_portfolio_line(label: str, m: dict, show_gate: bool = True) -> None:
    """Print one-line portfolio summary."""
    gate = ""
    if show_gate:
        dd_ok = "OK" if m["max_dd"] < 0.40 else "FAIL"
        cagr_ok = "OK" if m["cagr"] >= 0.50 else "MISS"
        gate = f"  DD:{dd_ok} CAGR:{cagr_ok}"
    print(
        f"  {label:<45} SR={m['sharpe']:>6.3f}  CAGR={m['cagr']:>6.1%}  "
        f"MaxDD={m['max_dd']:>5.1%}  Calmar={m['calmar']:>5.2f}  "
        f"Sortino={m['sortino']:>6.3f}{gate}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():  # noqa: PLR0912
    print("=" * 100)
    print("  TRACK D PORTFOLIO OPTIMIZER")
    print(
        "  Combining 12 Track D strategies | Objective: max CAGR | Constraint: MaxDD < 40%"
    )
    print("=" * 100)

    # ── DATA ─────────────────────────────────────────────────────────────
    print(
        f"\nFetching {LOOKBACK_DAYS // 365}-year data for {len(ALL_SYMBOLS)} symbols..."
    )
    prices = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS)
    if len(prices) == 0:
        print("ERROR: No data fetched. Aborting.")
        sys.exit(1)
    print(f"  {len(prices)} rows, {prices['date'].min()} to {prices['date'].max()}")

    sym_data: dict[str, dict] = {}
    for sym in ALL_SYMBOLS:
        sdf = prices.filter(pl.col("symbol") == sym).sort("date")
        sym_data[sym] = dict(
            zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
        )

    # Use TQQQ as date spine (all 3x ETFs share the same trading calendar)
    tqqq_df = prices.filter(pl.col("symbol") == "TQQQ").sort("date")
    dates = tqqq_df["date"].to_list()
    print(f"  Trading days: {len(dates)}")

    # ── RUN ALL 12 BACKTESTS ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 1: INDIVIDUAL STRATEGY BACKTESTS (30% base weight)")
    print("=" * 100)

    STRATEGY_DEFS: dict[str, tuple] = {
        # name -> (type, *args)
        "TLT-TQQQ": ("lead_lag", "TLT", "TQQQ", {}),
        "AGG-TQQQ": ("lead_lag", "AGG", "TQQQ", {}),
        "IEF-TQQQ": (
            "lead_lag",
            "IEF",
            "TQQQ",
            {"lag_days": 2, "entry_thresh": 0.005, "exit_thresh": -0.003},
        ),
        "VCIT-TQQQ": (
            "lead_lag",
            "VCIT",
            "TQQQ",
            {"lag_days": 3, "entry_thresh": 0.005, "exit_thresh": -0.003},
        ),
        "TLT-UPRO": ("lead_lag", "TLT", "UPRO", {}),
        "AGG-SOXL": ("lead_lag", "AGG", "SOXL", {}),
        "TLT-SOXL": ("lead_lag", "TLT", "SOXL", {}),
        "D3-TQQQ/TMF": ("d3",),
        "D10-XLK/SOXL": ("d10",),
        "D11-SOXX/SOXL": ("d11",),
        "D12-TIP/UPRO": ("d12",),
        "D13-TSMOM/UPRO": ("d13",),
    }

    return_streams: dict[str, list[float]] = {}
    individual_metrics: dict[str, dict] = {}

    header = (
        f"  {'Strategy':<18} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} "
        f"{'Calmar':>7} {'Sortino':>8} {'TotRet':>9} {'AnnVol':>7} {'Days':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, spec in STRATEGY_DEFS.items():
        if spec[0] == "lead_lag":
            _, leader, follower, kwargs = spec
            rets = _backtest_lead_lag(
                sym_data,
                dates,
                leader,
                follower,
                target_weight=0.30,
                **kwargs,
            )
        elif spec[0] == "d3":
            rets = backtest_d3_tqqq_tmf(sym_data, dates, 0.30)
        elif spec[0] == "d10":
            rets = backtest_d10_xlk_xle_soxl(sym_data, dates, 0.40)
        elif spec[0] == "d11":
            rets = backtest_d11_soxx_soxl(sym_data, dates, 0.50)
        elif spec[0] == "d12":
            rets = backtest_d12_tip_tlt_upro(sym_data, dates, 0.35)
        elif spec[0] == "d13":
            rets = backtest_d13_tsmom_upro(sym_data, dates, 0.40)
        else:
            continue

        return_streams[name] = rets
        m = compute_metrics(rets)
        individual_metrics[name] = m
        print(
            f"  {name:<18} {m['sharpe']:>7.3f} {m['cagr']:>8.1%} {m['max_dd']:>8.1%} "
            f"{m['calmar']:>7.2f} {m['sortino']:>8.3f} {m['total_return']:>9.1%} "
            f"{m['ann_vol']:>7.1%} {m['n_days']:>6d}"
        )

    # Benchmark
    tqqq_rets = backtest_tqqq_buyhold(sym_data, dates)
    tqqq_m = compute_metrics(tqqq_rets)
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TQQQ B&H (bench)':<18} {tqqq_m['sharpe']:>7.3f} {tqqq_m['cagr']:>8.1%} "
        f"{tqqq_m['max_dd']:>8.1%} {tqqq_m['calmar']:>7.2f} {tqqq_m['sortino']:>8.3f} "
        f"{tqqq_m['total_return']:>9.1%} {tqqq_m['ann_vol']:>7.1%} {tqqq_m['n_days']:>6d}"
    )

    strategy_names = list(return_streams.keys())

    # ── CORRELATION MATRIX ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 2: PAIRWISE CORRELATION MATRIX")
    print("=" * 100)

    n_strats = len(strategy_names)
    matrix = align_returns(return_streams, strategy_names)
    corr = np.corrcoef(matrix, rowvar=False)

    # Short names for display
    short = [n[:11] for n in strategy_names]
    cw = 10
    print(f"\n  {'':>13}", end="")
    for sn in short:
        print(f"{sn:>{cw}}", end="")
    print()

    for i, name in enumerate(strategy_names):
        print(f"  {name[:13]:<13}", end="")
        for j in range(n_strats):
            val = corr[i, j]
            print(f"{val:>{cw}.3f}", end="")
        print()

    off_diag = []
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            off_diag.append(corr[i, j])
    avg_corr = sum(off_diag) / len(off_diag) if off_diag else 0
    print(f"\n  Average pairwise correlation: {avg_corr:.4f}")

    print("\n  High-correlation pairs (>0.50):")
    found_high = False
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            if corr[i, j] > 0.50:
                found_high = True
                print(
                    f"    {strategy_names[i]} x {strategy_names[j]}: {corr[i, j]:.3f}"
                )
    if not found_high:
        print("    None (excellent diversification)")

    print("\n  Low-correlation pairs (<0.15):")
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            if abs(corr[i, j]) < 0.15:
                print(
                    f"    {strategy_names[i]} x {strategy_names[j]}: {corr[i, j]:.3f}"
                )

    # ── RANKING ──────────────────────────────────────────────────────────
    ranked = sorted(
        individual_metrics.items(),
        key=lambda x: x[1]["sharpe"],
        reverse=True,
    )
    ranked_names = [name for name, _ in ranked]
    print("\n  Strategy ranking by Sharpe:")
    for rank, (name, m) in enumerate(ranked, 1):
        print(
            f"    {rank:>2}. {name:<18} SR={m['sharpe']:.3f}  "
            f"CAGR={m['cagr']:.1%}  MaxDD={m['max_dd']:.1%}"
        )

    # ── PORTFOLIO COMBINATIONS ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 3: EQUAL-WEIGHT PORTFOLIO COMBINATIONS")
    print("=" * 100)

    # Top-N by Sharpe
    for n_top in [3, 4, 5, 6, 9, 12]:
        top_n = ranked_names[: min(n_top, len(ranked_names))]
        if len(top_n) < 2:
            continue
        combined = combine_equal_weight(return_streams, top_n)
        m = compute_metrics(combined)
        label = f"Top-{len(top_n)} EW"
        print_portfolio_line(label, m)

    # All 12
    all_combined = combine_equal_weight(return_streams, strategy_names)
    all_m = compute_metrics(all_combined)
    print_portfolio_line("All-12 EW", all_m)

    # Curated low-correlation combos
    print("\n  Curated low-correlation combinations:")
    curated = [
        ["D3-TQQQ/TMF", "D13-TSMOM/UPRO"],
        ["D3-TQQQ/TMF", "D10-XLK/SOXL"],
        ["D3-TQQQ/TMF", "D13-TSMOM/UPRO", "D10-XLK/SOXL"],
        ["D3-TQQQ/TMF", "TLT-TQQQ", "D13-TSMOM/UPRO"],
        ["D3-TQQQ/TMF", "TLT-TQQQ", "D10-XLK/SOXL", "D13-TSMOM/UPRO"],
        ["D3-TQQQ/TMF", "TLT-TQQQ", "D12-TIP/UPRO", "D11-SOXX/SOXL"],
        ["TLT-TQQQ", "AGG-TQQQ", "D13-TSMOM/UPRO", "D10-XLK/SOXL"],
        ["D3-TQQQ/TMF", "TLT-UPRO", "AGG-SOXL", "D11-SOXX/SOXL", "D13-TSMOM/UPRO"],
    ]
    for combo in curated:
        valid = [s for s in combo if s in return_streams]
        if len(valid) < 2:
            continue
        combined = combine_equal_weight(return_streams, valid)
        m = compute_metrics(combined)
        label = " + ".join(s[:10] for s in valid)
        if len(label) > 44:
            label = label[:41] + "..."
        print_portfolio_line(label, m)

    # ── HRP PORTFOLIO ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 4: HRP (HIERARCHICAL RISK PARITY) PORTFOLIOS")
    print("=" * 100)

    hrp_combos = [
        ranked_names[:3],
        ranked_names[:5],
        ranked_names[:6],
        strategy_names,
        ["D3-TQQQ/TMF", "D13-TSMOM/UPRO", "D10-XLK/SOXL"],
        ["D3-TQQQ/TMF", "TLT-TQQQ", "D13-TSMOM/UPRO", "D10-XLK/SOXL"],
        ["D3-TQQQ/TMF", "TLT-TQQQ", "D12-TIP/UPRO", "D13-TSMOM/UPRO", "D10-XLK/SOXL"],
    ]

    for combo in hrp_combos:
        valid = [s for s in combo if s in return_streams]
        if len(valid) < 2:
            continue
        weights = hrp_weights(return_streams, valid)
        combined = combine_weighted(return_streams, weights)
        m = compute_metrics(combined)
        label = f"HRP({len(valid)}): " + "+".join(s[:8] for s in valid)
        if len(label) > 44:
            label = label[:41] + "..."
        print_portfolio_line(label, m)
        w_str = ", ".join(f"{s[:10]}={w:.1%}" for s, w in weights.items())
        print(f"    Weights: {w_str}")

    # ── EXHAUSTIVE EQUAL-WEIGHT SEARCH ───────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 5: EXHAUSTIVE SEARCH (all EW combos, MaxDD < 40%)")
    print("=" * 100)

    all_combos: list[tuple[list[str], dict]] = []
    for n_s in range(2, min(len(strategy_names) + 1, 13)):
        for combo in itertools.combinations(strategy_names, n_s):
            combined = combine_equal_weight(return_streams, list(combo))
            m = compute_metrics(combined)
            if m["max_dd"] < 0.40:
                all_combos.append((list(combo), m))

    all_combos.sort(key=lambda x: x[1]["cagr"], reverse=True)

    print(
        f"\n  {'Rank':>4} {'N':>2} {'Combination':<60} "
        f"{'SR':>6} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7}"
    )
    print("  " + "-" * 95)
    for rank, (combo, m) in enumerate(all_combos[:25], 1):
        label = " + ".join(s[:10] for s in combo)
        if len(label) > 59:
            label = label[:56] + "..."
        print(
            f"  {rank:>4} {len(combo):>2} {label:<60} "
            f"{m['sharpe']:>6.3f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.2f}"
        )

    # ── WEIGHT SCALING ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 6: WEIGHT SCALING (higher internal weights)")
    print("=" * 100)

    WEIGHT_TESTS = {
        "TLT-TQQQ": [0.50, 0.70, 0.90],
        "D3-TQQQ/TMF": [0.50, 0.70],
        "D10-XLK/SOXL": [0.50, 0.60],
        "D11-SOXX/SOXL": [0.30, 0.70],
        "D12-TIP/UPRO": [0.45, 0.55],
        "D13-TSMOM/UPRO": [0.50, 0.60],
    }

    high_streams: dict[str, list[float]] = {}
    high_metrics: dict[str, dict] = {}

    print(
        f"\n  {'Strategy@Weight':<24} {'Sharpe':>7} {'CAGR':>8} "
        f"{'MaxDD':>8} {'Calmar':>7} {'Sortino':>8}"
    )
    print("  " + "-" * 70)

    for name, weights_list in WEIGHT_TESTS.items():
        spec = STRATEGY_DEFS[name]
        for w in weights_list:
            if spec[0] == "lead_lag":
                _, leader, follower, kwargs = spec
                rets = _backtest_lead_lag(
                    sym_data, dates, leader, follower, target_weight=w, **kwargs
                )
            elif spec[0] == "d3":
                rets = backtest_d3_tqqq_tmf(sym_data, dates, w)
            elif spec[0] == "d10":
                rets = backtest_d10_xlk_xle_soxl(sym_data, dates, w)
            elif spec[0] == "d11":
                rets = backtest_d11_soxx_soxl(sym_data, dates, w)
            elif spec[0] == "d12":
                rets = backtest_d12_tip_tlt_upro(sym_data, dates, w)
            elif spec[0] == "d13":
                rets = backtest_d13_tsmom_upro(sym_data, dates, w)
            else:
                continue

            tag = f"{name}@{w:.0%}"
            high_streams[tag] = rets
            m = compute_metrics(rets)
            high_metrics[tag] = m
            dd_flag = " **BREACH**" if m["max_dd"] > 0.40 else ""
            print(
                f"  {tag:<24} {m['sharpe']:>7.3f} {m['cagr']:>8.1%} "
                f"{m['max_dd']:>8.1%} {m['calmar']:>7.2f} {m['sortino']:>8.3f}{dd_flag}"
            )

    # Best high-weight combos
    print("\n  Best high-weight portfolio combos (EW, MaxDD < 40%, top 15):")
    high_names = list(high_streams.keys())
    high_combos: list[tuple[list[str], dict]] = []
    for n_s in range(2, min(len(high_names) + 1, 6)):
        for combo in itertools.combinations(high_names, n_s):
            combined = combine_equal_weight(high_streams, list(combo))
            m = compute_metrics(combined)
            if m["max_dd"] < 0.40:
                high_combos.append((list(combo), m))

    high_combos.sort(key=lambda x: x[1]["cagr"], reverse=True)

    print(
        f"  {'Rank':>4} {'N':>2} {'Combination':<55} "
        f"{'SR':>6} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7}"
    )
    print("  " + "-" * 90)
    for rank, (combo, m) in enumerate(high_combos[:15], 1):
        label = " + ".join(combo)
        if len(label) > 54:
            label = label[:51] + "..."
        print(
            f"  {rank:>4} {len(combo):>2} {label:<55} "
            f"{m['sharpe']:>6.3f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.2f}"
        )

    # ── MONTE CARLO OPTIMIZATION ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  SECTION 7: MONTE CARLO OPTIMIZATION (100,000 random portfolios)")
    print("=" * 100)

    # Max Sharpe
    print("\n  --- Objective: MAXIMIZE SHARPE (MaxDD < 40%) ---")
    ms_w, ms_m = monte_carlo_optimize(
        return_streams,
        strategy_names,
        objective="sharpe",
        max_dd=0.40,
        n_samples=100_000,
    )
    if ms_m:
        print_portfolio_line("MC Max-Sharpe", ms_m)
        w_sorted = sorted(ms_w.items(), key=lambda x: x[1], reverse=True)
        for name, w in w_sorted:
            bar = "#" * int(w * 80)
            print(f"    {name:<18} {w:>6.1%}  {bar}")
        print_growth_table(ms_m["cagr"], "MC-Sharpe")
    else:
        print("  No feasible solution found.")

    # Max CAGR
    print("\n  --- Objective: MAXIMIZE CAGR (MaxDD < 40%) ---")
    mc_w, mc_m = monte_carlo_optimize(
        return_streams,
        strategy_names,
        objective="cagr",
        max_dd=0.40,
        n_samples=100_000,
        seed=123,
    )
    if mc_m:
        print_portfolio_line("MC Max-CAGR", mc_m)
        w_sorted = sorted(mc_w.items(), key=lambda x: x[1], reverse=True)
        for name, w in w_sorted:
            bar = "#" * int(w * 80)
            print(f"    {name:<18} {w:>6.1%}  {bar}")
        print_growth_table(mc_m["cagr"], "MC-CAGR")
    else:
        print("  No feasible solution found.")

    # Max CAGR with tighter DD constraint
    print("\n  --- Objective: MAXIMIZE CAGR (MaxDD < 25%) ---")
    mc25_w, mc25_m = monte_carlo_optimize(
        return_streams,
        strategy_names,
        objective="cagr",
        max_dd=0.25,
        n_samples=100_000,
        seed=456,
    )
    if mc25_m:
        print_portfolio_line("MC Max-CAGR (DD<25%)", mc25_m)
        w_sorted = sorted(mc25_w.items(), key=lambda x: x[1], reverse=True)
        for name, w in w_sorted:
            if w > 0.05:
                print(f"    {name:<18} {w:>6.1%}")
        print_growth_table(mc25_m["cagr"], "MC-CAGR(25%)")
    else:
        print("  No feasible solution found.")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  SECTION 8: FINAL RECOMMENDATION")
    print("=" * 100)

    # Collect all tested configs
    candidates: list[tuple[str, dict, dict[str, float] | None]] = []

    # All-12 EW
    candidates.append(("All-12 EW @30%", all_m, None))

    # HRP all-12
    hrp_all_w = hrp_weights(return_streams, strategy_names)
    hrp_all_rets = combine_weighted(return_streams, hrp_all_w)
    hrp_all_m = compute_metrics(hrp_all_rets)
    candidates.append(("HRP All-12", hrp_all_m, hrp_all_w))

    # Top-5 EW
    top5 = ranked_names[:5]
    top5_rets = combine_equal_weight(return_streams, top5)
    top5_m = compute_metrics(top5_rets)
    candidates.append(("Top-5 EW", top5_m, None))

    # MC results
    if ms_m:
        candidates.append(("MC Max-Sharpe", ms_m, ms_w))
    if mc_m:
        candidates.append(("MC Max-CAGR", mc_m, mc_w))
    if mc25_m:
        candidates.append(("MC Max-CAGR(DD<25%)", mc25_m, mc25_w))

    # Best exhaustive combo
    if all_combos:
        best_ex = all_combos[0]
        candidates.append(
            (
                f"Best EW Combo ({len(best_ex[0])}s)",
                best_ex[1],
                None,
            )
        )

    # Filter feasible and sort by CAGR
    print(
        f"\n  {'Configuration':<25} {'Sharpe':>7} {'CAGR':>8} "
        f"{'MaxDD':>8} {'Calmar':>7} {'Gate':>6}"
    )
    print("  " + "-" * 65)

    feasible = []
    for label, m, w in sorted(candidates, key=lambda x: x[1]["cagr"], reverse=True):
        gate = "PASS" if m["max_dd"] < 0.40 else "FAIL"
        print(
            f"  {label:<25} {m['sharpe']:>7.3f} {m['cagr']:>8.1%} "
            f"{m['max_dd']:>8.1%} {m['calmar']:>7.2f} {gate:>6}"
        )
        if m["max_dd"] < 0.40:
            feasible.append((label, m, w))

    # Answer the key question
    print()
    print("  " + "=" * 70)
    print("  KEY QUESTION: Maximum CAGR achievable while keeping MaxDD < 40%?")
    print("  " + "=" * 70)

    if feasible:
        best_label, best_m, best_w = max(feasible, key=lambda x: x[1]["cagr"])
        print(f"\n  ANSWER: {best_label}")
        print(f"    CAGR:         {best_m['cagr']:.2%}")
        print(f"    MaxDD:        {best_m['max_dd']:.2%}")
        print(f"    Sharpe:       {best_m['sharpe']:.3f}")
        print(f"    Calmar:       {best_m.get('calmar', 0):.2f}")
        print(f"    Sortino:      {best_m.get('sortino', 0):.3f}")
        if best_w:
            print("\n    Optimal weights:")
            for name, w in sorted(best_w.items(), key=lambda x: x[1], reverse=True):
                bar = "#" * int(w * 60)
                print(f"      {name:<18} {w:>6.1%}  {bar}")

        print(f"\n    $10,000 investment growth at {best_m['cagr']:.1%} CAGR:")
        capital = INITIAL_CAPITAL
        for yr in [1, 2, 3, 5, 10]:
            val = capital * (1 + best_m["cagr"]) ** yr
            print(f"      {yr:>2} year{'s' if yr > 1 else ''}: ${val:>15,.0f}")

        # Compare to TQQQ
        print("\n    vs TQQQ buy-and-hold:")
        print(
            f"      TQQQ: CAGR={tqqq_m['cagr']:.1%}, "
            f"MaxDD={tqqq_m['max_dd']:.1%}, Sharpe={tqqq_m['sharpe']:.3f}"
        )
        if tqqq_m["cagr"] != 0:
            cagr_ratio = best_m["cagr"] / tqqq_m["cagr"]
            dd_ratio = (
                best_m["max_dd"] / tqqq_m["max_dd"] if tqqq_m["max_dd"] > 0 else 0
            )
            print(f"      CAGR ratio (ours/TQQQ): {cagr_ratio:.2f}x")
            print(f"      MaxDD ratio (ours/TQQQ): {dd_ratio:.2f}x (lower is better)")
    else:
        print("\n  No configuration meets the MaxDD < 40% constraint.")

    # Show exhaustive top combo details
    if all_combos:
        best_ex_combo, best_ex_m = all_combos[0]
        print("\n  Best exhaustive EW combo (by CAGR, MaxDD<40%):")
        print(f"    Strategies: {best_ex_combo}")
        print(
            f"    CAGR={best_ex_m['cagr']:.2%}  Sharpe={best_ex_m['sharpe']:.3f}  "
            f"MaxDD={best_ex_m['max_dd']:.2%}"
        )

    # Combos achieving CAGR > 50%
    over_50 = [(c, m) for c, m in all_combos if m["cagr"] > 0.50]
    if over_50:
        print("\n  Combos achieving CAGR > 50% with MaxDD < 40%:")
        for combo, m in over_50[:5]:
            label = " + ".join(s[:10] for s in combo)
            print(
                f"    {label}: CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, "
                f"Sharpe={m['sharpe']:.3f}"
            )
    else:
        print("\n  No EW combo achieves CAGR > 50% at 30% base weight.")
        print("  Check SECTION 6 for higher-weight combos that may reach 50%+.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

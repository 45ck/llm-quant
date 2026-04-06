#!/usr/bin/env python3
"""Backtest F31: Factor ETF Cross-Sectional Momentum v1.

Self-contained backtest that:
- Fetches OHLCV for 7 factor ETFs + SPY + SHY + TLT
- Ranks factor ETFs by trailing 126-day return
- Selects top 2, applies absolute momentum filter
- Simulates monthly rebalancing with $100k starting capital
- Computes Sharpe, MaxDD, CAGR, DSR
- Saves results to data/strategies/factor-xsmom-v1/
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from llm_quant.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_dsr,
    compute_sharpe,
)
from llm_quant.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SLUG = "factor-xsmom-v1"
FACTOR_ETFS = ["VLUE", "MTUM", "QUAL", "USMV", "VTV", "VUG", "IWM"]
BENCHMARK_SYMBOLS = ["SPY", "TLT"]
CASH_ETF = "SHY"
DEFENSIVE_ETF = "USMV"
ALL_SYMBOLS = [*FACTOR_ETFS, *BENCHMARK_SYMBOLS, CASH_ETF]

# Parameters (frozen from research-spec)
LOOKBACK_DAYS = 126  # 6-month trailing return
TOP_K = 2
WEIGHT_PER_POSITION = 0.40
CASH_WEIGHT = 0.20
REBALANCE_FREQ = 21  # trading days
INITIAL_CAPITAL = 100_000
COST_BPS = 5.0  # round-trip cost per trade in basis points
LOOKBACK_CAL_DAYS = 5 * 365  # 1825 calendar days of data


def _last_valid(prices: dict[str, np.ndarray], sym: str, t: int) -> float:
    """Get last valid (non-NaN) price for symbol at or before index t."""
    if sym not in prices:
        return 0.0
    arr = prices[sym]
    for i in range(t, -1, -1):
        if not np.isnan(arr[i]):
            return float(arr[i])
    return 0.0


def _fetch_and_prepare(
    available_out: list[str],
) -> tuple[dict[str, np.ndarray], int] | None:
    """Fetch data and build price arrays. Returns (prices, n_days) or None."""
    logger.info("Fetching data for %d symbols...", len(ALL_SYMBOLS))
    prices_df = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_CAL_DAYS)

    if prices_df.is_empty():
        logger.error("No data fetched. Aborting.")
        return None

    close_wide = prices_df.select(["date", "symbol", "close"]).pivot(
        on="symbol", index="date", values="close"
    )
    close_wide = close_wide.sort("date")

    available = [s for s in FACTOR_ETFS if s in close_wide.columns]
    missing = [s for s in FACTOR_ETFS if s not in close_wide.columns]
    if missing:
        logger.warning("Missing factor ETFs: %s", missing)
    if len(available) < 3:
        logger.error("Need at least 3 factor ETFs, got %d", len(available))
        return None

    for sym in [CASH_ETF, *BENCHMARK_SYMBOLS]:
        if sym not in close_wide.columns:
            logger.warning("Missing symbol: %s", sym)

    dates = close_wide["date"].to_list()
    n_days = len(dates)
    logger.info("Data: %d trading days from %s to %s", n_days, dates[0], dates[-1])

    prices: dict[str, np.ndarray] = {}
    for sym in [*available, CASH_ETF, *BENCHMARK_SYMBOLS]:
        if sym in close_wide.columns:
            prices[sym] = np.array(close_wide[sym].to_list(), dtype=np.float64)

    available_out.clear()
    available_out.extend(available)
    return prices, n_days


def _select_etfs(
    available: list[str],
    prices: dict[str, np.ndarray],
    t: int,
    lookback_days: int,
    top_k: int,
    defensive_etf: str,
) -> list[str]:
    """Rank factor ETFs and select top K with absolute momentum filter."""
    trailing_rets: dict[str, float] = {}
    for sym in available:
        if sym not in prices:
            continue
        p_now = prices[sym][t]
        p_past = prices[sym][t - lookback_days]
        if np.isnan(p_now) or np.isnan(p_past) or p_past <= 0:
            continue
        trailing_rets[sym] = (p_now / p_past) - 1.0

    if len(trailing_rets) < top_k:
        return []

    ranked = sorted(trailing_rets.items(), key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    for sym, ret in ranked[:top_k]:
        if ret < 0:
            if defensive_etf not in selected:
                selected.append(defensive_etf)
        else:
            selected.append(sym)
    return selected


def _rebalance(
    nav: float,
    holdings: dict[str, float],
    selected: list[str],
    prices: dict[str, np.ndarray],
    t: int,
    weight_per_position: float,
    cash_weight: float,
    cost_bps: float,
    top_k: int = TOP_K,
) -> tuple[dict[str, float], float, float]:
    """Execute a rebalance. Returns (new_holdings, cost, turnover)."""
    target_weights: dict[str, float] = {}
    for sym in selected:
        target_weights[sym] = target_weights.get(sym, 0.0) + weight_per_position

    # If fewer ETFs selected than top_k (defensive dedup), redirect to cash
    unallocated = (top_k - len(selected)) * weight_per_position
    effective_cash_weight = cash_weight + unallocated

    # Add cash ETF weight (accumulate if already in target_weights)
    if CASH_ETF in prices:
        target_weights[CASH_ETF] = (
            target_weights.get(CASH_ETF, 0.0) + effective_cash_weight
        )

    new_holdings: dict[str, float] = {}
    turnover = 0.0

    for sym, weight in target_weights.items():
        if sym not in prices or np.isnan(prices[sym][t]):
            continue
        target_value = nav * weight
        target_shares = target_value / prices[sym][t]
        new_holdings[sym] = target_shares

        old_shares = holdings.get(sym, 0.0)
        old_value = old_shares * prices[sym][t]
        turnover += abs(target_value - old_value)

    cost = turnover * (cost_bps / 10000.0)
    adjusted_nav = nav - cost

    if adjusted_nav > 0 and (nav + cost) > 0:
        cost_ratio = 1.0 - (cost / (adjusted_nav + cost))
        for sym in new_holdings:
            new_holdings[sym] *= cost_ratio

    return new_holdings, cost, turnover


def _compute_nav(
    holdings: dict[str, float],
    prices: dict[str, np.ndarray],
    t: int,
    fallback_nav: float,
) -> float:
    """Compute portfolio NAV from holdings at time t."""
    if not holdings:
        return fallback_nav
    port_value = 0.0
    for sym, shares in holdings.items():
        if sym in prices and not np.isnan(prices[sym][t]):
            port_value += shares * prices[sym][t]
        else:
            port_value += shares * _last_valid(prices, sym, t)
    return port_value


def _compute_metrics(
    strategy_returns: list[float],
    daily_navs: list[float],
    prices: dict[str, np.ndarray],
    warmup: int,
    n_days: int,
    rebalance_count: int,
    total_turnover: float,
    nav: float,
    params: dict,
) -> dict:
    """Compute all backtest metrics."""
    sharpe = compute_sharpe(strategy_returns, annualize=True)

    nav_series = np.array(daily_navs)
    running_max = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - running_max) / running_max
    max_dd = abs(float(np.min(drawdowns)))

    trading_days = len(strategy_returns)
    years = trading_days / TRADING_DAYS_PER_YEAR
    total_return = float(
        nav_series[-1] / nav_series[0] - 1.0 if nav_series[0] > 0 else 0.0
    )
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0)

    neg_rets = [r for r in strategy_returns if r < 0]
    downside_dev = float(np.std(neg_rets, ddof=1)) if len(neg_rets) > 1 else 1e-8
    sortino = (
        float(np.mean(strategy_returns))
        / downside_dev
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    calmar = float(cagr / max_dd if max_dd > 0 else 0.0)

    dsr, psr, sr0 = compute_dsr(strategy_returns, trial_count=1)

    bench_returns = _compute_benchmark(prices, warmup, n_days)
    bench_sharpe = compute_sharpe(bench_returns, annualize=True) if bench_returns else 0

    logger.info("=== F31 Factor XS Momentum Backtest Results ===")
    logger.info(
        "Sharpe: %.4f | MaxDD: %.2f%% | CAGR: %.2f%%", sharpe, max_dd * 100, cagr * 100
    )
    logger.info("DSR: %.4f | Sortino: %.4f | Calmar: %.4f", dsr, sortino, calmar)
    logger.info(
        "Rebalances: %d | Benchmark Sharpe: %.4f | Final NAV: $%.2f",
        rebalance_count,
        bench_sharpe,
        nav,
    )

    return {
        "slug": SLUG,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "total_return": total_return,
        "sortino": sortino,
        "calmar": calmar,
        "dsr": dsr,
        "psr": psr,
        "sr0": sr0,
        "rebalance_count": rebalance_count,
        "total_trades": rebalance_count,
        "total_turnover": total_turnover,
        "benchmark_sharpe": bench_sharpe,
        "final_nav": float(nav),
        "trading_days": trading_days,
        "daily_returns": strategy_returns,
        "daily_navs": daily_navs,
        "benchmark_returns": bench_returns,
        "parameters": params,
    }


def _compute_benchmark(
    prices: dict[str, np.ndarray], warmup: int, n_days: int
) -> list[float]:
    """Compute 60/40 SPY/TLT benchmark daily returns."""
    bench_returns: list[float] = []
    if "SPY" not in prices or "TLT" not in prices:
        return bench_returns
    for t_idx in range(warmup + 1, n_days):
        spy_prev = prices["SPY"][t_idx - 1]
        tlt_prev = prices["TLT"][t_idx - 1]
        spy_ret = (prices["SPY"][t_idx] - spy_prev) / spy_prev if spy_prev > 0 else 0.0
        tlt_ret = (prices["TLT"][t_idx] - tlt_prev) / tlt_prev if tlt_prev > 0 else 0.0
        bench_returns.append(0.60 * spy_ret + 0.40 * tlt_ret)
    return bench_returns


def run_backtest(
    lookback_days: int = LOOKBACK_DAYS,
    top_k: int = TOP_K,
    weight_per_position: float = WEIGHT_PER_POSITION,
    rebalance_freq: int = REBALANCE_FREQ,
    defensive_etf: str = DEFENSIVE_ETF,
    cost_bps: float = COST_BPS,
    cash_weight: float | None = None,
) -> dict:
    """Run the cross-sectional momentum backtest.

    Returns a dict with metrics and daily_returns list.
    """
    cash_weight_actual = (
        cash_weight
        if cash_weight is not None
        else max(0.0, 1.0 - top_k * weight_per_position)
    )

    available: list[str] = []
    result = _fetch_and_prepare(available)
    if result is None:
        return {"error": "no_data"}
    prices, n_days = result

    warmup = lookback_days
    if warmup >= n_days:
        logger.error("Warmup (%d) >= data length (%d)", warmup, n_days)
        return {"error": "insufficient_data_warmup"}

    # Simulation loop
    nav = float(INITIAL_CAPITAL)
    holdings: dict[str, float] = {}
    daily_navs: list[float] = []
    daily_returns: list[float] = []
    rebalance_count = 0
    total_turnover = 0.0
    days_since_rebalance = rebalance_freq  # force first rebalance

    for t in range(n_days):
        nav = _compute_nav(holdings, prices, t, nav)
        daily_navs.append(nav)

        if t > 0 and daily_navs[t - 1] > 0:
            daily_returns.append((nav - daily_navs[t - 1]) / daily_navs[t - 1])
        else:
            daily_returns.append(0.0)

        if t < warmup:
            continue

        days_since_rebalance += 1
        if days_since_rebalance < rebalance_freq:
            continue

        days_since_rebalance = 0
        selected = _select_etfs(
            available, prices, t, lookback_days, top_k, defensive_etf
        )
        if not selected:
            continue

        holdings, cost, turnover = _rebalance(
            nav,
            holdings,
            selected,
            prices,
            t,
            weight_per_position,
            cash_weight_actual,
            cost_bps,
            top_k,
        )
        nav -= cost
        total_turnover += turnover
        rebalance_count += 1

    strategy_returns = daily_returns[warmup:]
    if len(strategy_returns) < 30:
        logger.error("Too few returns after warmup: %d", len(strategy_returns))
        return {"error": "insufficient_returns"}

    params = {
        "lookback_days": lookback_days,
        "top_k": top_k,
        "weight_per_position": weight_per_position,
        "rebalance_frequency_days": rebalance_freq,
        "defensive_etf": defensive_etf,
        "cash_etf": CASH_ETF,
        "cash_weight": cash_weight_actual,
        "cost_bps": cost_bps,
    }

    return _compute_metrics(
        strategy_returns,
        daily_navs[warmup:],
        prices,
        warmup,
        n_days,
        rebalance_count,
        total_turnover,
        nav,
        params,
    )


def save_results(result: dict) -> None:
    """Save backtest results and experiment registry entry."""
    import yaml

    strat_dir = Path(f"data/strategies/{SLUG}")
    strat_dir.mkdir(parents=True, exist_ok=True)

    results_yaml = {
        "strategy_slug": SLUG,
        "family": "factor_xsmom",
        "family_trial_number": 1,
        "sharpe_ratio": round(result["sharpe_ratio"], 4),
        "max_drawdown": round(result["max_drawdown"], 4),
        "cagr": round(result["cagr"], 4),
        "total_return": round(result["total_return"], 4),
        "sortino": round(result["sortino"], 4),
        "calmar": round(result["calmar"], 4),
        "dsr": round(result["dsr"], 4),
        "psr": round(result["psr"], 4),
        "benchmark_sharpe": round(result["benchmark_sharpe"], 4),
        "final_nav": round(result["final_nav"], 2),
        "rebalance_count": result["rebalance_count"],
        "trading_days": result["trading_days"],
        "parameters": result["parameters"],
        "gates": {
            "sharpe_ge_0.80": result["sharpe_ratio"] >= 0.80,
            "max_dd_lt_0.15": result["max_drawdown"] < 0.15,
            "dsr_ge_0.95": result["dsr"] >= 0.95,
        },
        "computed_at": datetime.now(UTC).isoformat(),
    }

    results_path = strat_dir / "backtest-results.yaml"
    with results_path.open("w") as f:
        yaml.dump(results_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved backtest results to %s", results_path)

    experiment = {
        "experiment_id": str(uuid.uuid4())[:8],
        "slug": SLUG,
        "family": "factor_xsmom",
        "family_trial_number": 1,
        "sharpe_ratio": round(result["sharpe_ratio"], 4),
        "max_drawdown": round(result["max_drawdown"], 4),
        "cagr": round(result["cagr"], 4),
        "total_return": round(result["total_return"], 4),
        "sortino": round(result["sortino"], 4),
        "calmar": round(result["calmar"], 4),
        "dsr": round(result["dsr"], 4),
        "total_trades": result["total_trades"],
        "parameters": result["parameters"],
        "benchmark_sharpe": round(result["benchmark_sharpe"], 4),
        "computed_at": datetime.now(UTC).isoformat(),
        "hash": hashlib.sha256(
            json.dumps(result["parameters"], sort_keys=True).encode()
        ).hexdigest()[:16],
    }

    registry_path = strat_dir / "experiment-registry.jsonl"
    with registry_path.open("a") as f:
        f.write(json.dumps(experiment) + "\n")
    logger.info("Appended experiment to %s", registry_path)


def main() -> None:
    """Run backtest and save results."""
    result = run_backtest()
    if "error" in result:
        logger.error("Backtest failed: %s", result["error"])
        sys.exit(1)

    save_results(result)

    print("\n" + "=" * 60)
    print("F31 FACTOR XS MOMENTUM -- GATE SUMMARY")
    print("=" * 60)
    gates = {
        "Sharpe >= 0.80": (result["sharpe_ratio"], result["sharpe_ratio"] >= 0.80),
        "MaxDD < 15%": (result["max_drawdown"], result["max_drawdown"] < 0.15),
        "DSR >= 0.95": (result["dsr"], result["dsr"] >= 0.95),
    }
    for name, (val, passed) in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {val:.4f} [{status}]")
    all_pass = all(p for _, p in gates.values())
    print(f"\nOverall: {'ALL GATES PASSED' if all_pass else 'GATES FAILED'}")
    print(f"Benchmark (60/40): Sharpe={result['benchmark_sharpe']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

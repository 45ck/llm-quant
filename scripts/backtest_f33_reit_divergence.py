#!/usr/bin/env python3
"""Backtest F33: REIT Divergence Financial Conditions v1.

Self-contained backtest that:
- Fetches OHLCV for XLRE, SPY, GLD, SHY, TLT
- Computes XLRE/SPY ratio, 60-day z-score, 10-day momentum
- Classifies regime: easing / tightening / neutral
- Allocates across assets based on regime
- Simulates weekly (5-day) rebalancing with $100k starting capital
- Applies 5 bps round-trip cost per trade
- Computes: Sharpe, MaxDD, DSR, CAGR, Sortino, Calmar
- Saves results to data/strategies/reit-divergence-v1/
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
SLUG = "reit-divergence-v1"
SIGNAL_SYMBOLS = ["XLRE", "SPY"]
TRADE_SYMBOLS = ["SPY", "GLD", "SHY"]
BENCHMARK_SYMBOLS = ["SPY", "TLT"]
ALL_SYMBOLS = list(set(SIGNAL_SYMBOLS + TRADE_SYMBOLS + BENCHMARK_SYMBOLS))

# Parameters (frozen from research-spec)
ZSCORE_WINDOW = 60  # 60-day z-score of XLRE/SPY ratio
MOM_WINDOW = 10  # 10-day momentum (pct change) of ratio
ZSCORE_THRESH = 0.5  # z-score threshold for regime classification
SPY_EASING = 0.80  # SPY weight in easing regime
GLD_TIGHTENING = 0.60  # GLD weight in tightening regime
SHY_TIGHTENING = 0.30  # SHY weight in tightening regime
SPY_NEUTRAL = 0.50  # SPY weight in neutral regime
SHY_NEUTRAL = 0.40  # SHY weight in neutral regime
REBALANCE_FREQ = 5  # weekly rebalance (trading days)
INITIAL_CAPITAL = 100_000
COST_BPS = 5.0  # round-trip cost per trade in basis points
LOOKBACK_CAL_DAYS = 1825  # 5 * 365 calendar days of data


def _fetch_and_prepare() -> tuple[dict[str, np.ndarray], int, list] | None:
    """Fetch data and build price arrays. Returns (prices, n_days, dates)."""
    logger.info("Fetching data for %d symbols...", len(ALL_SYMBOLS))
    prices_df = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_CAL_DAYS)

    if prices_df.is_empty():
        logger.error("No data fetched. Aborting.")
        return None

    close_wide = prices_df.select(["date", "symbol", "close"]).pivot(
        on="symbol", index="date", values="close"
    )
    close_wide = close_wide.sort("date")

    # Verify required symbols
    for sym in SIGNAL_SYMBOLS + TRADE_SYMBOLS:
        if sym not in close_wide.columns:
            logger.error("Missing required symbol: %s", sym)
            return None

    dates = close_wide["date"].to_list()
    n_days = len(dates)
    logger.info("Data: %d trading days from %s to %s", n_days, dates[0], dates[-1])

    prices: dict[str, np.ndarray] = {}
    for sym in ALL_SYMBOLS:
        if sym in close_wide.columns:
            prices[sym] = np.array(close_wide[sym].to_list(), dtype=np.float64)

    return prices, n_days, dates


def _compute_ratio_signals(
    prices: dict[str, np.ndarray],
    n_days: int,
    zscore_window: int,
    mom_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute XLRE/SPY ratio, z-score, and momentum.

    Returns (ratio, ratio_zscore, ratio_mom) arrays of length n_days.
    """
    xlre = prices["XLRE"]
    spy = prices["SPY"]

    ratio = np.full(n_days, np.nan)
    ratio_zscore = np.full(n_days, np.nan)
    ratio_mom = np.full(n_days, np.nan)

    # Compute ratio
    for t in range(n_days):
        if not np.isnan(xlre[t]) and not np.isnan(spy[t]) and spy[t] > 0:
            ratio[t] = xlre[t] / spy[t]

    # Ratio z-score: (ratio - mean) / std over zscore_window
    for t in range(zscore_window - 1, n_days):
        window = ratio[t - zscore_window + 1 : t + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= zscore_window // 2:
            mean_val = float(np.mean(valid))
            std_val = float(np.std(valid, ddof=1))
            if std_val > 1e-10:
                ratio_zscore[t] = (ratio[t] - mean_val) / std_val

    # Ratio momentum: ratio[t] / ratio[t-mom_window] - 1
    for t in range(mom_window, n_days):
        if not np.isnan(ratio[t]) and not np.isnan(ratio[t - mom_window]):
            if ratio[t - mom_window] > 0:
                ratio_mom[t] = ratio[t] / ratio[t - mom_window] - 1.0

    return ratio, ratio_zscore, ratio_mom


def _classify_regime(
    zscore_t: float,
    mom_t: float,
    zscore_thresh: float,
) -> str:
    """Classify regime based on XLRE/SPY ratio z-score and momentum.

    Returns 'easing', 'tightening', or 'neutral'.
    """
    if np.isnan(zscore_t) or np.isnan(mom_t):
        return "neutral"

    if zscore_t > zscore_thresh and mom_t > 0:
        return "easing"
    if zscore_t < -zscore_thresh and mom_t < 0:
        return "tightening"
    return "neutral"


def _get_target_weights(
    regime: str,
    spy_easing: float,
    gld_tightening: float,
    shy_tightening: float,
    spy_neutral: float,
    shy_neutral: float,
) -> dict[str, float]:
    """Return target allocation weights based on regime."""
    if regime == "easing":
        return {"SPY": spy_easing}
    if regime == "tightening":
        return {"GLD": gld_tightening, "SHY": shy_tightening}
    return {"SPY": spy_neutral, "SHY": shy_neutral}


def _rebalance(
    nav: float,
    holdings: dict[str, float],
    target_weights: dict[str, float],
    prices: dict[str, np.ndarray],
    t: int,
    cost_bps: float,
) -> tuple[dict[str, float], float, float]:
    """Execute rebalance. Returns (new_holdings, cost, turnover)."""
    new_holdings: dict[str, float] = {}
    turnover = 0.0
    invested_weight = 0.0

    for sym, weight in target_weights.items():
        if sym not in prices or np.isnan(prices[sym][t]):
            continue
        target_value = nav * weight
        target_shares = target_value / prices[sym][t]
        new_holdings[sym] = target_shares
        invested_weight += weight

        old_shares = holdings.get(sym, 0.0)
        old_value = old_shares * prices[sym][t]
        turnover += abs(target_value - old_value)

    # Uninvested portion stays as cash
    cash_weight = max(0.0, 1.0 - invested_weight)
    cash_value = nav * cash_weight

    old_cash = holdings.get("_CASH", 0.0)
    turnover += abs(cash_value - old_cash)

    cost = turnover * (cost_bps / 10000.0)
    cash_value -= cost

    new_holdings["_CASH"] = cash_value

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
    port_value = holdings.get("_CASH", 0.0)
    for sym, shares in holdings.items():
        if sym == "_CASH":
            continue
        if sym in prices and not np.isnan(prices[sym][t]):
            port_value += shares * prices[sym][t]
    return port_value if port_value > 0 else fallback_nav


def _compute_benchmark(
    prices: dict[str, np.ndarray], warmup: int, n_days: int
) -> list[float]:
    """Compute 60/40 SPY/TLT benchmark daily returns."""
    bench_returns: list[float] = []
    if "SPY" not in prices or "TLT" not in prices:
        return bench_returns
    for t in range(warmup + 1, n_days):
        spy_prev = prices["SPY"][t - 1]
        tlt_prev = prices["TLT"][t - 1]
        spy_ret = (
            (prices["SPY"][t] - spy_prev) / spy_prev
            if spy_prev > 0
            and not np.isnan(spy_prev)
            and not np.isnan(prices["SPY"][t])
            else 0.0
        )
        tlt_ret = (
            (prices["TLT"][t] - tlt_prev) / tlt_prev
            if tlt_prev > 0
            and not np.isnan(tlt_prev)
            and not np.isnan(prices["TLT"][t])
            else 0.0
        )
        bench_returns.append(0.60 * spy_ret + 0.40 * tlt_ret)
    return bench_returns


def run_backtest(
    zscore_window: int = ZSCORE_WINDOW,
    mom_window: int = MOM_WINDOW,
    zscore_thresh: float = ZSCORE_THRESH,
    spy_easing: float = SPY_EASING,
    gld_tightening: float = GLD_TIGHTENING,
    shy_tightening: float = SHY_TIGHTENING,
    spy_neutral: float = SPY_NEUTRAL,
    shy_neutral: float = SHY_NEUTRAL,
    rebalance_freq: int = REBALANCE_FREQ,
    cost_bps: float = COST_BPS,
) -> dict:
    """Run the REIT divergence backtest.

    Returns dict with metrics and daily_returns list.
    """
    result = _fetch_and_prepare()
    if result is None:
        return {"error": "no_data"}
    prices, n_days, dates = result

    # Compute signals
    ratio, ratio_zscore, ratio_mom = _compute_ratio_signals(
        prices, n_days, zscore_window, mom_window
    )

    # Warmup = zscore_window to ensure z-score is valid
    warmup = zscore_window
    if warmup >= n_days:
        logger.error("Warmup (%d) >= data length (%d)", warmup, n_days)
        return {"error": "insufficient_data_warmup"}

    # Simulation loop
    nav = float(INITIAL_CAPITAL)
    holdings: dict[str, float] = {}
    daily_navs: list[float] = []
    daily_returns: list[float] = []
    regime_history: list[str] = []
    rebalance_count = 0
    total_turnover = 0.0
    total_cost = 0.0
    days_since_rebalance = rebalance_freq  # force first rebalance

    for t in range(n_days):
        nav = _compute_nav(holdings, prices, t, nav)
        daily_navs.append(nav)

        if t > 0 and daily_navs[t - 1] > 0:
            daily_returns.append((nav - daily_navs[t - 1]) / daily_navs[t - 1])
        else:
            daily_returns.append(0.0)

        if t < warmup:
            regime_history.append("warmup")
            continue

        # Use data through t-1 to decide position on t (causal)
        regime = _classify_regime(ratio_zscore[t - 1], ratio_mom[t - 1], zscore_thresh)
        regime_history.append(regime)

        days_since_rebalance += 1
        if days_since_rebalance < rebalance_freq:
            continue

        days_since_rebalance = 0

        target_weights = _get_target_weights(
            regime,
            spy_easing,
            gld_tightening,
            shy_tightening,
            spy_neutral,
            shy_neutral,
        )

        holdings, cost, turnover = _rebalance(
            nav, holdings, target_weights, prices, t, cost_bps
        )
        nav -= cost
        total_turnover += turnover
        total_cost += cost
        rebalance_count += 1

    # Post-warmup returns only
    strategy_returns = daily_returns[warmup:]
    strategy_navs = daily_navs[warmup:]

    if len(strategy_returns) < 30:
        logger.error("Too few returns after warmup: %d", len(strategy_returns))
        return {"error": "insufficient_returns"}

    # Metrics
    sharpe = compute_sharpe(strategy_returns, annualize=True)

    nav_arr = np.array(strategy_navs)
    running_max = np.maximum.accumulate(nav_arr)
    drawdowns = (nav_arr - running_max) / running_max
    max_dd = abs(float(np.min(drawdowns)))

    trading_days = len(strategy_returns)
    years = trading_days / TRADING_DAYS_PER_YEAR
    total_return = float(nav_arr[-1] / nav_arr[0] - 1.0) if nav_arr[0] > 0 else 0.0
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0

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
    bench_sharpe = (
        compute_sharpe(bench_returns, annualize=True) if bench_returns else 0.0
    )

    # Regime counts (post-warmup)
    post_warmup_regimes = regime_history[warmup:]
    regime_counts = {
        "easing": sum(1 for r in post_warmup_regimes if r == "easing"),
        "tightening": sum(1 for r in post_warmup_regimes if r == "tightening"),
        "neutral": sum(1 for r in post_warmup_regimes if r == "neutral"),
    }

    logger.info("=== F33 REIT Divergence Backtest Results ===")
    logger.info(
        "Sharpe: %.4f | MaxDD: %.2f%% | CAGR: %.2f%%",
        sharpe,
        max_dd * 100,
        cagr * 100,
    )
    logger.info("DSR: %.4f | Sortino: %.4f | Calmar: %.4f", dsr, sortino, calmar)
    logger.info(
        "Rebalances: %d | Total cost: $%.2f | Benchmark Sharpe: %.4f",
        rebalance_count,
        total_cost,
        bench_sharpe,
    )
    logger.info("Regime counts: %s", regime_counts)

    params = {
        "zscore_window": zscore_window,
        "mom_window": mom_window,
        "zscore_thresh": zscore_thresh,
        "spy_easing": spy_easing,
        "gld_tightening": gld_tightening,
        "shy_tightening": shy_tightening,
        "spy_neutral": spy_neutral,
        "shy_neutral": shy_neutral,
        "rebalance_frequency_days": rebalance_freq,
        "cost_bps": cost_bps,
    }

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
        "total_cost": total_cost,
        "benchmark_sharpe": bench_sharpe,
        "final_nav": float(nav),
        "trading_days": trading_days,
        "regime_counts": regime_counts,
        "daily_returns": strategy_returns,
        "daily_navs": strategy_navs,
        "benchmark_returns": bench_returns,
        "parameters": params,
    }


def save_results(result: dict) -> None:
    """Save backtest results and experiment registry entry."""
    import yaml

    strat_dir = Path(f"data/strategies/{SLUG}")
    strat_dir.mkdir(parents=True, exist_ok=True)

    results_yaml = {
        "strategy_slug": SLUG,
        "family": "reit_divergence",
        "family_id": "F33",
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
        "regime_counts": result["regime_counts"],
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
        "family": "reit_divergence",
        "family_id": "F33",
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
    print("F33 REIT DIVERGENCE -- GATE SUMMARY")
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
    print(f"CAGR: {result['cagr'] * 100:.2f}%")
    print(f"Sortino: {result['sortino']:.4f} | Calmar: {result['calmar']:.4f}")
    print(f"Regime counts: {result['regime_counts']}")
    print(f"Final NAV: ${result['final_nav']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

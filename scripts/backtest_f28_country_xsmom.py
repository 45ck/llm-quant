#!/usr/bin/env python3
"""Backtest F28: International Country Cross-Sectional Momentum.

Self-contained backtest script. Does NOT use the strategy registry.

Signal: For each of 8 country ETFs, compute trailing 12-1 month return
(Novy-Marx skip-month). Rank cross-sectionally. Buy top 3 with positive
absolute momentum at 30% each. Hold 10% in SHY. Rebalance monthly.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_f28_country_xsmom.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.metrics import compute_sharpe
from llm_quant.backtest.robustness import compute_min_trl
from llm_quant.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SLUG = "country-xsmom-v1"
COUNTRY_ETFS: list[str] = ["EWJ", "VGK", "EWZ", "EWT", "EWC", "EWA", "EWY", "INDA"]
CASH_ETF = "SHY"
BENCHMARK_SYMBOLS: dict[str, float] = {"SPY": 0.60, "TLT": 0.40}
ALL_SYMBOLS: list[str] = [*COUNTRY_ETFS, CASH_ETF, "SPY", "TLT"]
LOOKBACK_DAYS = 5 * 365  # 1825 calendar days
INITIAL_CAPITAL = 100_000.0
ROUND_TRIP_COST_BPS = 5.0  # 5 bps per round-trip trade

# Strategy parameters
LOOKBACK = 252  # 12 months in trading days
SKIP = 21  # 1 month skip (Novy-Marx)
TOP_K = 3
WEIGHT_PER_POSITION = 0.30
REBALANCE_FREQ = 21  # trading days between rebalances
ABS_MOM_THRESHOLD = 0.0


@dataclass
class BacktestParams:
    """Strategy parameters for the backtest."""

    lookback_days: int = LOOKBACK
    skip_days: int = SKIP
    top_k: int = TOP_K
    weight_per_position: float = WEIGHT_PER_POSITION
    rebalance_frequency_days: int = REBALANCE_FREQ
    absolute_momentum_threshold: float = ABS_MOM_THRESHOLD
    cash_etf: str = CASH_ETF
    round_trip_cost_bps: float = ROUND_TRIP_COST_BPS

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "lookback_days": self.lookback_days,
            "skip_days": self.skip_days,
            "top_k": self.top_k,
            "weight_per_position": self.weight_per_position,
            "rebalance_frequency_days": self.rebalance_frequency_days,
            "absolute_momentum_threshold": self.absolute_momentum_threshold,
            "cash_etf": self.cash_etf,
            "round_trip_cost_bps": self.round_trip_cost_bps,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core backtest logic
# ──────────────────────────────────────────────────────────────────────────────


def run_country_xsmom_backtest(
    params: BacktestParams | None = None,
    return_daily_returns: bool = True,
) -> dict:
    """Run the country cross-sectional momentum backtest.

    Parameters
    ----------
    params : BacktestParams | None
        Strategy parameters. Uses defaults if None.
    return_daily_returns : bool
        If True, include daily_returns in the result dict.

    Returns
    -------
    dict
        Backtest results including Sharpe, MaxDD, CAGR, daily returns.
    """
    if params is None:
        params = BacktestParams()

    # Fetch data
    logger.info(
        "Fetching data for %d symbols (%d days)...", len(ALL_SYMBOLS), LOOKBACK_DAYS
    )
    prices_df = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS)
    if len(prices_df) == 0:
        logger.error("No data fetched -- aborting")
        sys.exit(1)

    return _run_backtest_on_data(prices_df, params, return_daily_returns)


def _run_backtest_on_data(  # noqa: PLR0912
    prices_df: pl.DataFrame,
    params: BacktestParams,
    return_daily_returns: bool = True,
) -> dict:
    """Run backtest on pre-fetched data. Allows reuse across perturbation tests.

    This is the core engine, strictly causal: at each rebalance point t,
    we only use prices up to and including t.
    """
    # Build per-symbol close price series, aligned by date
    symbols_needed = [*COUNTRY_ETFS, params.cash_etf, "SPY", "TLT"]
    close_by_symbol: dict[str, dict[str, float]] = {}

    for sym in symbols_needed:
        sym_data = prices_df.filter(pl.col("symbol") == sym).sort("date")
        if len(sym_data) == 0:
            logger.warning("No data for %s -- skipping", sym)
            continue
        dates = sym_data["date"].to_list()
        closes = sym_data["close"].to_list()
        close_by_symbol[sym] = {str(d): c for d, c in zip(dates, closes, strict=True)}

    # Build aligned date index (intersection of all country ETF dates)
    date_sets = []
    for sym in COUNTRY_ETFS:
        if sym in close_by_symbol:
            date_sets.append(set(close_by_symbol[sym].keys()))
    if not date_sets:
        logger.error("No country ETF data available")
        return {"error": "No data"}

    common_dates = sorted(set.intersection(*date_sets))
    logger.info(
        "Common trading dates: %d (from %s to %s)",
        len(common_dates),
        common_dates[0],
        common_dates[-1],
    )

    # Also need cash ETF and benchmark dates
    for sym in [params.cash_etf, "SPY", "TLT"]:
        if sym in close_by_symbol:
            common_dates = [d for d in common_dates if d in close_by_symbol[sym]]

    if len(common_dates) < params.lookback_days + 100:
        logger.error(
            "Insufficient data: %d dates, need %d + warmup",
            len(common_dates),
            params.lookback_days,
        )
        return {"error": "Insufficient data"}

    # ── Simulation ────────────────────────────────────────────────────────
    warmup = params.lookback_days  # need full lookback before first signal
    nav = INITIAL_CAPITAL
    holdings: dict[str, float] = {}  # symbol -> number of shares (fractional)
    daily_navs: list[float] = []
    daily_returns: list[float] = []
    benchmark_navs: list[float] = []
    total_trades = 0
    rebalance_count = 0
    days_since_rebalance = params.rebalance_frequency_days  # force first rebalance

    cost_per_trade = params.round_trip_cost_bps / 10_000.0

    for i, date_str in enumerate(common_dates):
        if i < warmup:
            # During warmup, hold everything in cash ETF
            if i == 0:
                cash_price = close_by_symbol[params.cash_etf][date_str]
                holdings = {params.cash_etf: INITIAL_CAPITAL / cash_price}
            # Track NAV during warmup
            nav = sum(
                shares * close_by_symbol[sym].get(date_str, 0.0)
                for sym, shares in holdings.items()
            )
            daily_navs.append(nav)
            if i > 0:
                prev_nav = daily_navs[-2]
                daily_returns.append(
                    (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
                )

            # Benchmark tracking
            if i == 0:
                bench_nav = INITIAL_CAPITAL
            else:
                bench_nav = _compute_benchmark_nav(
                    INITIAL_CAPITAL, common_dates, i, close_by_symbol, BENCHMARK_SYMBOLS
                )
            benchmark_navs.append(bench_nav)
            continue

        # Compute current NAV from holdings
        nav = sum(
            shares * close_by_symbol[sym].get(date_str, 0.0)
            for sym, shares in holdings.items()
        )
        daily_navs.append(nav)
        if len(daily_navs) >= 2:
            prev_nav = daily_navs[-2]
            daily_returns.append((nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0)

        # Benchmark
        bench_nav = _compute_benchmark_nav(
            INITIAL_CAPITAL, common_dates, i, close_by_symbol, BENCHMARK_SYMBOLS
        )
        benchmark_navs.append(bench_nav)

        # Check if rebalance day
        days_since_rebalance += 1
        if days_since_rebalance < params.rebalance_frequency_days:
            continue

        days_since_rebalance = 0

        # ── Compute momentum signals (strictly causal) ──────────────────
        # momentum_12_1 = price[t - skip] / price[t - lookback] - 1
        momentum_scores: dict[str, float] = {}
        for sym in COUNTRY_ETFS:
            if sym not in close_by_symbol:
                continue
            # Index for t - skip_days and t - lookback_days
            skip_idx = i - params.skip_days
            lookback_idx = i - params.lookback_days
            if skip_idx < 0 or lookback_idx < 0:
                continue
            skip_date = common_dates[skip_idx]
            lookback_date = common_dates[lookback_idx]
            price_skip = close_by_symbol[sym].get(skip_date)
            price_lookback = close_by_symbol[sym].get(lookback_date)
            if price_skip is None or price_lookback is None or price_lookback <= 0:
                continue
            momentum_scores[sym] = (price_skip / price_lookback) - 1.0

        if len(momentum_scores) < params.top_k:
            continue  # Not enough data to rank

        # Rank by momentum descending
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top K with positive absolute momentum
        selected: list[str] = []
        for sym, mom in ranked:
            if len(selected) >= params.top_k:
                break
            if mom > params.absolute_momentum_threshold:
                selected.append(sym)

        # ── Target allocation ────────────────────────────────────────────
        target_weights: dict[str, float] = {}
        for sym in selected:
            target_weights[sym] = params.weight_per_position

        # Remaining goes to cash ETF
        equity_weight = sum(target_weights.values())
        cash_weight = 1.0 - equity_weight
        target_weights[params.cash_etf] = cash_weight

        # ── Execute rebalance with transaction costs ─────────────────────
        # Compute turnover cost: sum of |target_weight - current_weight|
        # for each symbol. Cost is applied once (round-trip) per unit of
        # turnover. The new holdings are computed from NAV minus total cost.
        current_weights: dict[str, float] = {}
        for sym_h, shares_h in holdings.items():
            price_h = close_by_symbol[sym_h].get(date_str, 0.0)
            current_weights[sym_h] = (shares_h * price_h) / nav if nav > 0 else 0.0

        # Total turnover = sum of buys (which equals sum of sells)
        all_syms = set(target_weights.keys()) | set(current_weights.keys())
        total_turnover_value = 0.0
        trades_this_rebalance = 0
        for sym_t in all_syms:
            tw = target_weights.get(sym_t, 0.0)
            cw = current_weights.get(sym_t, 0.0)
            trade_value = abs(tw - cw) * nav
            if trade_value > nav * 0.001:
                trades_this_rebalance += 1
            total_turnover_value += trade_value

        # Cost is applied to one side of turnover (buy side = sell side)
        total_cost = (total_turnover_value / 2.0) * cost_per_trade
        nav_after_costs = nav - total_cost

        # Allocate new holdings from NAV after costs
        new_holdings = {}
        for sym_t, tw in target_weights.items():
            price_t = close_by_symbol[sym_t].get(date_str, 0.0)
            if price_t <= 0:
                continue
            new_holdings[sym_t] = (nav_after_costs * tw) / price_t

        holdings = new_holdings
        total_trades += trades_this_rebalance
        rebalance_count += 1

    # ── Compute metrics ──────────────────────────────────────────────────────
    if len(daily_returns) < 20:
        logger.error("Too few daily returns: %d", len(daily_returns))
        return {"error": "Insufficient returns"}

    # Trim warmup returns (they're just cash ETF returns)
    strategy_returns = daily_returns[warmup:]
    if len(strategy_returns) < 20:
        strategy_returns = daily_returns

    arr = np.array(strategy_returns)
    sharpe = compute_sharpe(strategy_returns, annualize=True)
    n_days = len(strategy_returns)
    annualized_return = float(np.mean(arr)) * 252
    annualized_vol = float(np.std(arr, ddof=1)) * math.sqrt(252)

    # Sortino
    downside = arr[arr < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
    sortino = (
        annualized_return / (downside_std * math.sqrt(252)) if downside_std > 0 else 0.0
    )

    # Max drawdown
    cumulative = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # CAGR
    total_return = float(cumulative[-1]) - 1.0
    years = n_days / 252.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

    # DSR (trial_count=1 for first trial)
    from llm_quant.backtest.metrics import compute_dsr

    dsr, psr, sr0 = compute_dsr(strategy_returns, trial_count=1)

    # MinTRL
    skew_val = float(scipy_stats.skew(arr, bias=False))
    kurt_val = float(scipy_stats.kurtosis(arr, bias=False))
    min_trl = compute_min_trl(
        sharpe=sharpe,
        skew=skew_val,
        kurtosis=kurt_val,
        n_observations=n_days,
    )

    # Win rate (daily)
    win_rate = float(np.sum(arr > 0)) / len(arr) if len(arr) > 0 else 0.0

    result = {
        "slug": SLUG,
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(abs(max_dd), 4),
        "cagr": round(cagr, 4),
        "total_return": round(total_return, 4),
        "annualized_return": round(annualized_return, 4),
        "annualized_vol": round(annualized_vol, 4),
        "dsr": round(dsr, 4),
        "psr": round(psr, 4),
        "total_trades": total_trades,
        "rebalance_count": rebalance_count,
        "n_days": n_days,
        "win_rate": round(win_rate, 4),
        "min_trl_months": round(min_trl.min_trl_months, 2),
        "min_trl_pass": min_trl.min_trl_pass,
        "skew": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "final_nav": round(daily_navs[-1], 2) if daily_navs else 0.0,
        "parameters": params.to_dict(),
    }

    if return_daily_returns:
        result["daily_returns"] = strategy_returns

    logger.info(
        "Backtest complete: Sharpe=%.4f, MaxDD=%.4f, CAGR=%.4f, Trades=%d, DSR=%.4f",
        sharpe,
        abs(max_dd),
        cagr,
        total_trades,
        dsr,
    )

    return result


def _compute_benchmark_nav(
    initial_capital: float,
    dates: list[str],
    current_idx: int,
    close_by_symbol: dict[str, dict[str, float]],
    benchmark_weights: dict[str, float],
) -> float:
    """Compute benchmark NAV at current_idx using buy-and-hold from day 0."""
    if current_idx == 0:
        return initial_capital

    nav = 0.0
    for sym, weight in benchmark_weights.items():
        if sym not in close_by_symbol:
            continue
        start_price = close_by_symbol[sym].get(dates[0])
        current_price = close_by_symbol[sym].get(dates[current_idx])
        if start_price and current_price and start_price > 0:
            nav += initial_capital * weight * (current_price / start_price)

    return nav if nav > 0 else initial_capital


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the F28 backtest and save results."""
    result = run_country_xsmom_backtest()

    if "error" in result:
        logger.error("Backtest failed: %s", result["error"])
        sys.exit(1)

    # Save backtest results (without daily_returns for YAML readability)
    strat_dir = Path(f"data/strategies/{SLUG}")
    strat_dir.mkdir(parents=True, exist_ok=True)

    results_for_yaml = {k: v for k, v in result.items() if k != "daily_returns"}
    results_path = strat_dir / "backtest-results.yaml"

    import yaml

    with results_path.open("w") as f:
        yaml.dump(results_for_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info("Backtest results saved to %s", results_path)

    # Save experiment to registry
    registry_path = strat_dir / "experiment-registry.jsonl"
    experiment_id = str(uuid.uuid4())[:8]
    entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "slug": SLUG,
        "strategy_type": "country_xsmom",
        "sharpe_ratio": result["sharpe_ratio"],
        "max_drawdown": result["max_drawdown"],
        "cagr": result["cagr"],
        "total_return": result["total_return"],
        "dsr": result["dsr"],
        "total_trades": result["total_trades"],
        "parameters": result["parameters"],
    }
    with registry_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Experiment %s appended to %s", experiment_id, registry_path)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  F28 Country XSMOM Backtest: {SLUG}")
    print("=" * 60)
    print(f"  Sharpe Ratio:     {result['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:    {result['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:     {result['calmar_ratio']:.4f}")
    print(
        f"  Max Drawdown:     {result['max_drawdown']:.4f} ({result['max_drawdown'] * 100:.1f}%)"
    )
    print(f"  CAGR:             {result['cagr']:.4f} ({result['cagr'] * 100:.1f}%)")
    print(
        f"  Total Return:     {result['total_return']:.4f} ({result['total_return'] * 100:.1f}%)"
    )
    print(
        f"  Annualized Vol:   {result['annualized_vol']:.4f} ({result['annualized_vol'] * 100:.1f}%)"
    )
    print(f"  DSR:              {result['dsr']:.4f}")
    print(
        f"  Win Rate:         {result['win_rate']:.4f} ({result['win_rate'] * 100:.1f}%)"
    )
    print(f"  Total Trades:     {result['total_trades']}")
    print(f"  Rebalances:       {result['rebalance_count']}")
    print(f"  Trading Days:     {result['n_days']}")
    print(f"  Final NAV:        ${result['final_nav']:,.2f}")
    print(f"  MinTRL Pass:      {result['min_trl_pass']}")
    print("=" * 60)

    # Gate check
    gates = {
        "Sharpe >= 0.80": result["sharpe_ratio"] >= 0.80,
        "MaxDD < 15%": result["max_drawdown"] < 0.15,
        "DSR >= 0.95": result["dsr"] >= 0.95,
    }
    print("\n  Gate Check:")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate}: {status}")
    print()


if __name__ == "__main__":
    main()

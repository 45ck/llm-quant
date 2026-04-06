#!/usr/bin/env python3
"""Backtest F29: Bond Volatility Regime (TLT Vol Proxy).

Self-contained backtest script. Does NOT use the strategy registry.

Signal: Compute TLT and SPY 30-day realized volatility (annualized).
Form bond_vol_ratio = TLT_vol / SPY_vol. Rank via 90-day rolling percentile.
Three regimes:
  - BOND_STRESS (ratio_pctile > 0.80): 20% SPY + 30% TLT + 30% GLD + 20% SHY
  - EQUITY_OVERPRICED (ratio_pctile < 0.20): 70% SPY + 10% TLT + 10% GLD + 10% SHY
  - NEUTRAL (0.20 to 0.80): 50% SPY + 20% TLT + 15% GLD + 15% SHY
Rebalance every 5 trading days.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_f29_bond_vol_regime.py
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

SLUG = "bond-vol-regime-v1"
TRADEABLE_SYMBOLS: list[str] = ["SPY", "TLT", "GLD", "SHY"]
BENCHMARK_SYMBOLS: dict[str, float] = {"SPY": 0.60, "TLT": 0.40}
ALL_SYMBOLS: list[str] = [*TRADEABLE_SYMBOLS]
LOOKBACK_DAYS = 5 * 365  # 1825 calendar days
INITIAL_CAPITAL = 100_000.0
ROUND_TRIP_COST_BPS = 5.0

# Regime allocation weights
REGIME_ALLOCATIONS: dict[str, dict[str, float]] = {
    "bond_stress": {"SPY": 0.20, "TLT": 0.30, "GLD": 0.30, "SHY": 0.20},
    "equity_overpriced": {"SPY": 0.70, "TLT": 0.10, "GLD": 0.10, "SHY": 0.10},
    "neutral": {"SPY": 0.50, "TLT": 0.20, "GLD": 0.15, "SHY": 0.15},
}


@dataclass
class BacktestParams:
    """Strategy parameters for the backtest."""

    vol_window: int = 30
    ratio_lookback: int = 90
    bond_stress_threshold: float = 0.80
    equity_cheap_threshold: float = 0.20
    spy_bond_stress: float = 0.20
    spy_equity_cheap: float = 0.70
    spy_neutral: float = 0.50
    rebalance_frequency_days: int = 5
    round_trip_cost_bps: float = ROUND_TRIP_COST_BPS

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "vol_window": self.vol_window,
            "ratio_lookback": self.ratio_lookback,
            "bond_stress_threshold": self.bond_stress_threshold,
            "equity_cheap_threshold": self.equity_cheap_threshold,
            "spy_bond_stress": self.spy_bond_stress,
            "spy_equity_cheap": self.spy_equity_cheap,
            "spy_neutral": self.spy_neutral,
            "rebalance_frequency_days": self.rebalance_frequency_days,
            "round_trip_cost_bps": self.round_trip_cost_bps,
        }

    def get_regime_allocations(self) -> dict[str, dict[str, float]]:
        """Build regime allocations from current parameters."""
        # Compute TLT/GLD/SHY weights from SPY weight (complement)
        bs_spy = self.spy_bond_stress
        bs_remain = 1.0 - bs_spy
        ec_spy = self.spy_equity_cheap
        ec_remain = 1.0 - ec_spy
        n_spy = self.spy_neutral
        n_remain = 1.0 - n_spy

        return {
            "bond_stress": {
                "SPY": bs_spy,
                "TLT": bs_remain * 0.375,
                "GLD": bs_remain * 0.375,
                "SHY": bs_remain * 0.25,
            },
            "equity_overpriced": {
                "SPY": ec_spy,
                "TLT": ec_remain / 3.0,
                "GLD": ec_remain / 3.0,
                "SHY": ec_remain / 3.0,
            },
            "neutral": {
                "SPY": n_spy,
                "TLT": n_remain * 0.40,
                "GLD": n_remain * 0.30,
                "SHY": n_remain * 0.30,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core backtest logic
# ──────────────────────────────────────────────────────────────────────────────


def _compute_realized_vol(
    returns: list[float], end_idx: int, window: int
) -> float | None:
    """Compute annualized realized vol from daily returns ending at end_idx.

    Strictly causal: uses returns[end_idx - window + 1 : end_idx + 1].
    """
    start = end_idx - window + 1
    if start < 0:
        return None
    subset = returns[start : end_idx + 1]
    if len(subset) < window:
        return None
    arr = np.array(subset)
    std = float(np.std(arr, ddof=1))
    return std * math.sqrt(252)


def _compute_percentile_rank(
    values: list[float], end_idx: int, lookback: int
) -> float | None:
    """Compute rolling percentile rank of values[end_idx] within the window.

    Strictly causal: uses values[end_idx - lookback + 1 : end_idx + 1].
    Returns a value in [0, 1].
    """
    start = end_idx - lookback + 1
    if start < 0:
        return None
    window = values[start : end_idx + 1]
    if len(window) < lookback:
        return None
    current = window[-1]
    # Fraction of values in window that are <= current value
    arr = np.array(window)
    return float(np.sum(arr <= current)) / len(arr)


def run_bond_vol_regime_backtest(
    params: BacktestParams | None = None,
    return_daily_returns: bool = True,
    prices_df: object = None,
) -> dict:
    """Run the bond volatility regime backtest.

    Parameters
    ----------
    params : BacktestParams | None
        Strategy parameters. Uses defaults if None.
    return_daily_returns : bool
        If True, include daily_returns in the result dict.
    prices_df : pl.DataFrame | None
        Pre-fetched price data. If None, fetches fresh data.

    Returns
    -------
    dict
        Backtest results including Sharpe, MaxDD, CAGR, daily returns.
    """
    if params is None:
        params = BacktestParams()

    if prices_df is None:
        logger.info(
            "Fetching data for %d symbols (%d days)...",
            len(ALL_SYMBOLS),
            LOOKBACK_DAYS,
        )
        prices_df = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS)
        if len(prices_df) == 0:
            logger.error("No data fetched -- aborting")
            return {"error": "No data"}

    return _run_backtest_on_data(prices_df, params, return_daily_returns)


def _run_backtest_on_data(  # noqa: PLR0912
    prices_df: object,
    params: BacktestParams,
    return_daily_returns: bool = True,
) -> dict:
    """Run backtest on pre-fetched data.

    Strictly causal: at each point t, only prices up to and including t
    are used.
    """
    import polars as pl

    assert isinstance(prices_df, pl.DataFrame)

    # Build per-symbol close price series aligned by date
    close_by_symbol: dict[str, dict[str, float]] = {}
    for sym in TRADEABLE_SYMBOLS:
        sym_data = prices_df.filter(pl.col("symbol") == sym).sort("date")
        if len(sym_data) == 0:
            logger.warning("No data for %s -- skipping", sym)
            continue
        dates = sym_data["date"].to_list()
        closes = sym_data["close"].to_list()
        close_by_symbol[sym] = {str(d): c for d, c in zip(dates, closes, strict=True)}

    # Build aligned date index (intersection of all tradeable symbol dates)
    date_sets = [
        set(close_by_symbol[sym].keys())
        for sym in TRADEABLE_SYMBOLS
        if sym in close_by_symbol
    ]
    if not date_sets:
        logger.error("No data available for tradeable symbols")
        return {"error": "No data"}

    common_dates = sorted(set.intersection(*date_sets))
    logger.info(
        "Common trading dates: %d (from %s to %s)",
        len(common_dates),
        common_dates[0],
        common_dates[-1],
    )

    # Warmup needs: vol_window + ratio_lookback to get the first percentile
    warmup = params.vol_window + params.ratio_lookback
    if len(common_dates) < warmup + 100:
        logger.error(
            "Insufficient data: %d dates, need %d + buffer",
            len(common_dates),
            warmup,
        )
        return {"error": "Insufficient data"}

    # Pre-compute daily returns for TLT and SPY (for vol calculation)
    tlt_returns: list[float] = []
    spy_returns: list[float] = []
    for i, date_str in enumerate(common_dates):
        if i == 0:
            tlt_returns.append(0.0)
            spy_returns.append(0.0)
            continue
        prev_date = common_dates[i - 1]
        tlt_prev = close_by_symbol["TLT"].get(prev_date, 0.0)
        tlt_curr = close_by_symbol["TLT"].get(date_str, 0.0)
        spy_prev = close_by_symbol["SPY"].get(prev_date, 0.0)
        spy_curr = close_by_symbol["SPY"].get(date_str, 0.0)
        tlt_ret = (tlt_curr / tlt_prev - 1.0) if tlt_prev > 0 else 0.0
        spy_ret = (spy_curr / spy_prev - 1.0) if spy_prev > 0 else 0.0
        tlt_returns.append(tlt_ret)
        spy_returns.append(spy_ret)

    # Pre-compute bond_vol_ratio at each date for percentile ranking
    bond_vol_ratios: list[float | None] = []
    for i in range(len(common_dates)):
        tlt_vol = _compute_realized_vol(tlt_returns, i, params.vol_window)
        spy_vol = _compute_realized_vol(spy_returns, i, params.vol_window)
        if tlt_vol is not None and spy_vol is not None and spy_vol > 1e-10:
            bond_vol_ratios.append(tlt_vol / spy_vol)
        else:
            bond_vol_ratios.append(None)

    # Get regime allocations
    regime_allocs = params.get_regime_allocations()

    # ── Simulation ────────────────────────────────────────────────────────
    nav = INITIAL_CAPITAL
    holdings: dict[str, float] = {}
    daily_navs: list[float] = []
    daily_returns_out: list[float] = []
    benchmark_navs: list[float] = []
    total_trades = 0
    rebalance_count = 0
    regime_counts: dict[str, int] = {
        "bond_stress": 0,
        "equity_overpriced": 0,
        "neutral": 0,
    }
    days_since_rebalance = params.rebalance_frequency_days  # force first rebalance

    cost_per_trade = params.round_trip_cost_bps / 10_000.0

    for i, date_str in enumerate(common_dates):
        if i < warmup:
            # During warmup, hold everything in SHY (cash proxy)
            if i == 0:
                shy_price = close_by_symbol["SHY"][date_str]
                holdings = {"SHY": INITIAL_CAPITAL / shy_price}
            nav = sum(
                shares * close_by_symbol[sym].get(date_str, 0.0)
                for sym, shares in holdings.items()
            )
            daily_navs.append(nav)
            if i > 0:
                prev_nav = daily_navs[-2]
                daily_returns_out.append(
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
            daily_returns_out.append(
                (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
            )

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

        # ── Compute regime signal (strictly causal) ───────────────────────
        # Need valid bond_vol_ratios for the ratio_lookback window
        # Filter the bond_vol_ratios up to and including index i
        valid_ratios: list[float] = []
        for j in range(max(0, i - params.ratio_lookback + 1), i + 1):
            if bond_vol_ratios[j] is not None:
                valid_ratios.append(bond_vol_ratios[j])

        if len(valid_ratios) < params.ratio_lookback // 2:
            # Not enough data for reliable percentile; stay neutral
            continue

        current_ratio = bond_vol_ratios[i]
        if current_ratio is None:
            continue

        # Percentile rank: fraction of values <= current
        arr_ratios = np.array(valid_ratios)
        ratio_percentile = float(np.sum(arr_ratios <= current_ratio)) / len(arr_ratios)

        # Determine regime
        if ratio_percentile > params.bond_stress_threshold:
            regime = "bond_stress"
        elif ratio_percentile < params.equity_cheap_threshold:
            regime = "equity_overpriced"
        else:
            regime = "neutral"

        regime_counts[regime] += 1
        target_weights = regime_allocs[regime]

        # ── Execute rebalance with transaction costs ─────────────────────
        current_weights: dict[str, float] = {}
        for sym_h, shares_h in holdings.items():
            price_h = close_by_symbol[sym_h].get(date_str, 0.0)
            current_weights[sym_h] = (shares_h * price_h) / nav if nav > 0 else 0.0

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

        total_cost = (total_turnover_value / 2.0) * cost_per_trade
        nav_after_costs = nav - total_cost

        new_holdings: dict[str, float] = {}
        for sym_t, tw in target_weights.items():
            price_t = close_by_symbol[sym_t].get(date_str, 0.0)
            if price_t <= 0:
                continue
            new_holdings[sym_t] = (nav_after_costs * tw) / price_t

        holdings = new_holdings
        total_trades += trades_this_rebalance
        rebalance_count += 1

    # ── Compute metrics ──────────────────────────────────────────────────────
    if len(daily_returns_out) < 20:
        logger.error("Too few daily returns: %d", len(daily_returns_out))
        return {"error": "Insufficient returns"}

    # Trim warmup returns (they're just SHY returns)
    strategy_returns = daily_returns_out[warmup:]
    if len(strategy_returns) < 20:
        strategy_returns = daily_returns_out

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

    result: dict = {
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
        "regime_counts": regime_counts,
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
    """Run the F29 backtest and save results."""
    result = run_bond_vol_regime_backtest()

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
        "strategy_type": "bond_vol_regime",
        "sharpe_ratio": result["sharpe_ratio"],
        "max_drawdown": result["max_drawdown"],
        "cagr": result["cagr"],
        "total_return": result["total_return"],
        "dsr": result["dsr"],
        "total_trades": result["total_trades"],
        "regime_counts": result["regime_counts"],
        "parameters": result["parameters"],
    }
    with registry_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Experiment %s appended to %s", experiment_id, registry_path)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  F29 Bond Vol Regime Backtest: {SLUG}")
    print("=" * 60)
    print(f"  Sharpe Ratio:     {result['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:    {result['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:     {result['calmar_ratio']:.4f}")
    print(
        f"  Max Drawdown:     {result['max_drawdown']:.4f}"
        f" ({result['max_drawdown'] * 100:.1f}%)"
    )
    print(f"  CAGR:             {result['cagr']:.4f} ({result['cagr'] * 100:.1f}%)")
    print(
        f"  Total Return:     {result['total_return']:.4f}"
        f" ({result['total_return'] * 100:.1f}%)"
    )
    print(
        f"  Annualized Vol:   {result['annualized_vol']:.4f}"
        f" ({result['annualized_vol'] * 100:.1f}%)"
    )
    print(f"  DSR:              {result['dsr']:.4f}")
    print(
        f"  Win Rate:         {result['win_rate']:.4f}"
        f" ({result['win_rate'] * 100:.1f}%)"
    )
    print(f"  Total Trades:     {result['total_trades']}")
    print(f"  Rebalances:       {result['rebalance_count']}")
    print(f"  Regime Counts:    {result['regime_counts']}")
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

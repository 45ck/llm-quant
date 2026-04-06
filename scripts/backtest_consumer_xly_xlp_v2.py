#!/usr/bin/env python3
"""Backtest: consumer-xly-xlp-v2 (Consumer Cyclical-Defensive).

Self-contained backtest script. Uses yfinance via fetch_ohlcv for data.

Mechanism: XLY/XLP ratio as economic cycle indicator.
  - XLY outperforming XLP = consumer confidence, risk-on
  - XLP outperforming XLY = defensive rotation, risk-off

Signal (dual confirmation):
  - Compute XLY/XLP price ratio
  - 20-day ratio momentum = ratio[t-1] / ratio[t-1-20] - 1
  - 30-day ratio SMA = mean(ratio[t-30:t])
  - Regime classification at t uses data through t-1 (strictly causal)

Three regimes:
  - risk_on  (mom > 0 AND ratio > SMA): 60% SPY + 10% QQQ + 30% SHY
  - risk_off (mom < 0 AND ratio < SMA): 40% GLD + 30% SHY + 30% SPY
  - neutral  (mixed signals):           50% SPY + 40% SHY + 10% GLD

v2 change from v1: gld_riskoff=0.40, three-regime dual-signal approach.
Based on v1 perturbation analysis showing gld_weight=0.40 reduces MaxDD.

Rebalance every 5 trading days. 3 bps half-spread cost per trade.
Benchmark: 60/40 SPY/TLT buy-and-hold.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_consumer_xly_xlp_v2.py
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

from llm_quant.backtest.metrics import compute_dsr, compute_sharpe
from llm_quant.backtest.robustness import compute_min_trl
from llm_quant.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SLUG = "consumer-xly-xlp-v2"
SIGNAL_SYMBOLS: list[str] = ["XLY", "XLP"]
TRADEABLE_SYMBOLS: list[str] = ["SPY", "QQQ", "GLD", "SHY"]
BENCHMARK_SYMBOLS: dict[str, float] = {"SPY": 0.60, "TLT": 0.40}
ALL_SYMBOLS: list[str] = [*SIGNAL_SYMBOLS, *TRADEABLE_SYMBOLS, "TLT"]
LOOKBACK_DAYS = 5 * 365  # 1825 calendar days
INITIAL_CAPITAL = 100_000.0
FAMILY_TRIAL_NUMBER = 2


@dataclass
class BacktestParams:
    """Strategy parameters for consumer-xly-xlp-v2."""

    # Signal parameters
    mom_lookback: int = 20  # 20-day ratio momentum
    sma_period: int = 30  # 30-day SMA of ratio

    # Risk-on allocation (momentum > 0 AND ratio > SMA)
    spy_riskon: float = 0.60
    qqq_riskon: float = 0.10
    # shy_riskon = remainder to 1.0 (0.30)

    # Risk-off allocation (momentum < 0 AND ratio < SMA)
    gld_riskoff: float = 0.40  # v2 key parameter
    shy_riskoff: float = 0.30
    spy_riskoff: float = 0.30

    # Neutral allocation (mixed signals)
    spy_neutral: float = 0.50
    shy_neutral: float = 0.40
    gld_neutral: float = 0.10

    # Execution
    rebalance_frequency_days: int = 5  # weekly rebalance
    half_spread_bps: float = 3.0  # 3 bps per side

    def get_regime_weights(self, regime: str) -> dict[str, float]:
        """Return target weights for a given regime. All sum to 1.0."""
        if regime == "risk_on":
            shy_w = round(1.0 - self.spy_riskon - self.qqq_riskon, 4)
            return {
                "SPY": self.spy_riskon,
                "QQQ": self.qqq_riskon,
                "GLD": 0.0,
                "SHY": max(0.0, shy_w),
            }
        if regime == "risk_off":
            # Enforce sum to 1.0 via SHY as remainder
            shy_w = round(1.0 - self.gld_riskoff - self.spy_riskoff, 4)
            return {
                "SPY": self.spy_riskoff,
                "QQQ": 0.0,
                "GLD": self.gld_riskoff,
                "SHY": max(0.0, shy_w),
            }
        # neutral
        shy_w = round(1.0 - self.spy_neutral - self.gld_neutral, 4)
        return {
            "SPY": self.spy_neutral,
            "QQQ": 0.0,
            "GLD": self.gld_neutral,
            "SHY": max(0.0, shy_w),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mom_lookback": self.mom_lookback,
            "sma_period": self.sma_period,
            "spy_riskon": self.spy_riskon,
            "qqq_riskon": self.qqq_riskon,
            "gld_riskoff": self.gld_riskoff,
            "shy_riskoff": self.shy_riskoff,
            "spy_riskoff": self.spy_riskoff,
            "spy_neutral": self.spy_neutral,
            "shy_neutral": self.shy_neutral,
            "gld_neutral": self.gld_neutral,
            "rebalance_frequency_days": self.rebalance_frequency_days,
            "half_spread_bps": self.half_spread_bps,
        }


# ------------------------------------------------------------------------------
# Core backtest logic
# ------------------------------------------------------------------------------


def run_backtest(
    params: BacktestParams | None = None,
    return_daily_returns: bool = True,
    prices_df: object | None = None,
    **kwargs: object,
) -> dict:
    """Run the consumer-xly-xlp-v2 backtest.

    Parameters
    ----------
    params : BacktestParams | None
        Strategy parameters. Uses defaults if None.
    return_daily_returns : bool
        If True, include daily_returns in the result dict.
    prices_df : pl.DataFrame | None
        Pre-fetched price data. If None, fetches from yfinance.
    **kwargs
        Additional keyword arguments passed to BacktestParams if params is None.

    Returns
    -------
    dict
        Backtest results including Sharpe, MaxDD, CAGR, daily returns.
    """
    if params is None:
        params = BacktestParams(
            **{k: v for k, v in kwargs.items() if hasattr(BacktestParams, k)}
        )

    if prices_df is None:
        logger.info(
            "Fetching data for %d symbols (%d days)...",
            len(ALL_SYMBOLS),
            LOOKBACK_DAYS,
        )
        import polars as pl

        prices_df_typed: pl.DataFrame = fetch_ohlcv(
            ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS
        )
        if len(prices_df_typed) == 0:
            logger.error("No data fetched -- aborting")
            sys.exit(1)
    else:
        import polars as pl

        prices_df_typed = (
            pl.DataFrame(prices_df)
            if not isinstance(prices_df, pl.DataFrame)
            else prices_df
        )

    return _run_backtest_on_data(prices_df_typed, params, return_daily_returns)


def _run_backtest_on_data(  # noqa: PLR0912
    prices_df: object,
    params: BacktestParams,
    return_daily_returns: bool = True,
) -> dict:
    """Run backtest on pre-fetched data.

    Strictly causal: at each rebalance point t, we only use prices up to
    and including t-1 to decide the position held at t.
    """
    import polars as pl

    df = prices_df if isinstance(prices_df, pl.DataFrame) else pl.DataFrame(prices_df)

    # Build per-symbol close price series, aligned by date
    close_by_symbol: dict[str, dict[str, float]] = {}
    for sym in ALL_SYMBOLS:
        sym_data = df.filter(pl.col("symbol") == sym).sort("date")
        if len(sym_data) == 0:
            logger.warning("No data for %s -- skipping", sym)
            continue
        dates = sym_data["date"].to_list()
        closes = sym_data["close"].to_list()
        close_by_symbol[sym] = {str(d): c for d, c in zip(dates, closes, strict=True)}

    # Verify we have all needed symbols
    needed = set(SIGNAL_SYMBOLS + TRADEABLE_SYMBOLS + ["TLT"])
    missing = [s for s in needed if s not in close_by_symbol]
    if missing:
        logger.error("Missing symbols: %s", missing)
        return {"error": f"Missing symbols: {missing}"}

    # Build aligned date index (intersection of all symbols)
    date_sets = [set(close_by_symbol[sym].keys()) for sym in needed]
    common_dates = sorted(set.intersection(*date_sets))
    logger.info(
        "Common trading dates: %d (from %s to %s)",
        len(common_dates),
        common_dates[0] if common_dates else "N/A",
        common_dates[-1] if common_dates else "N/A",
    )

    # Warmup needs max(sma_period, mom_lookback) + buffer
    warmup = max(params.sma_period, params.mom_lookback) + 10
    if len(common_dates) < warmup + 100:
        logger.error(
            "Insufficient data: %d dates, need %d + warmup",
            len(common_dates),
            warmup,
        )
        return {"error": "Insufficient data"}

    # -- Precompute signals (all causal) --
    xly_closes = [close_by_symbol["XLY"][d] for d in common_dates]
    xlp_closes = [close_by_symbol["XLP"][d] for d in common_dates]

    # XLY/XLP ratio series
    ratio = [
        xly / xlp if xlp > 0 else 0.0
        for xly, xlp in zip(xly_closes, xlp_closes, strict=True)
    ]

    # Ratio momentum (mom_lookback-day)
    ratio_mom: list[float | None] = [None] * len(common_dates)
    for i in range(params.mom_lookback, len(common_dates)):
        past = ratio[i - params.mom_lookback]
        if past > 0:
            ratio_mom[i] = (ratio[i] / past) - 1.0

    # Ratio SMA (sma_period-day)
    ratio_sma: list[float | None] = [None] * len(common_dates)
    for i in range(params.sma_period - 1, len(common_dates)):
        window = ratio[i - params.sma_period + 1 : i + 1]
        ratio_sma[i] = sum(window) / len(window)

    # -- Regime classification (uses data at index i) --
    def classify_regime(i: int) -> str | None:
        """Classify regime at date index i. Returns None if signals unavailable."""
        rm = ratio_mom[i]
        rs = ratio_sma[i]
        if rm is None or rs is None:
            return None

        r = ratio[i]
        if rm > 0 and r > rs:
            return "risk_on"
        if rm < 0 and r < rs:
            return "risk_off"
        return "neutral"

    # -- Simulation --
    nav = INITIAL_CAPITAL
    holdings: dict[str, float] = {}  # symbol -> shares (fractional)
    daily_navs: list[float] = []
    daily_returns: list[float] = []
    benchmark_navs: list[float] = []
    total_trades = 0
    rebalance_count = 0
    days_since_rebalance = params.rebalance_frequency_days  # force first rebalance
    regime_history: list[str] = []

    cost_per_side = params.half_spread_bps / 10_000.0

    for i, date_str in enumerate(common_dates):
        if i < warmup:
            # During warmup, hold everything in SHY (cash equivalent)
            if i == 0:
                shy_price = close_by_symbol["SHY"][date_str]
                holdings = {"SHY": INITIAL_CAPITAL / shy_price}
            nav = sum(
                shares * close_by_symbol[sym].get(date_str, 0.0)
                for sym, shares in holdings.items()
                if sym in TRADEABLE_SYMBOLS
            )
            if nav <= 0:
                nav = INITIAL_CAPITAL
            daily_navs.append(nav)
            if i > 0:
                prev_nav = daily_navs[-2]
                daily_returns.append(
                    (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
                )
            # Benchmark tracking
            bench_nav = _compute_benchmark_nav(
                INITIAL_CAPITAL, common_dates, i, close_by_symbol, BENCHMARK_SYMBOLS
            )
            benchmark_navs.append(bench_nav)
            continue

        # Compute current NAV from holdings
        nav = sum(
            shares * close_by_symbol[sym].get(date_str, 0.0)
            for sym, shares in holdings.items()
            if sym in TRADEABLE_SYMBOLS
        )
        if nav <= 0:
            nav = daily_navs[-1] if daily_navs else INITIAL_CAPITAL
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

        # -- Determine regime (strictly causal: use data through i-1) --
        regime = classify_regime(i - 1)
        if regime is None:
            continue
        regime_history.append(regime)

        # -- Target allocation based on regime --
        target_weights = params.get_regime_weights(regime)

        # -- Execute rebalance with transaction costs --
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

        # Apply half-spread cost to each side of turnover
        total_cost = (total_turnover_value / 2.0) * cost_per_side
        nav_after_costs = nav - total_cost

        new_holdings: dict[str, float] = {}
        for sym_t, tw in target_weights.items():
            if tw <= 0:
                continue
            price_t = close_by_symbol[sym_t].get(date_str, 0.0)
            if price_t <= 0:
                continue
            new_holdings[sym_t] = (nav_after_costs * tw) / price_t

        holdings = new_holdings
        total_trades += trades_this_rebalance
        rebalance_count += 1

    # -- Compute metrics --
    if len(daily_returns) < 20:
        logger.error("Too few daily returns: %d", len(daily_returns))
        return {"error": "Insufficient returns"}

    # Trim warmup returns (they're just SHY returns)
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

    # DSR (trial_count=2 for second trial in family)
    dsr, psr, sr0 = compute_dsr(strategy_returns, trial_count=FAMILY_TRIAL_NUMBER)

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

    # Regime distribution
    regime_counts: dict[str, int] = {}
    for r in regime_history:
        regime_counts[r] = regime_counts.get(r, 0) + 1

    # Benchmark metrics
    bench_returns_full = []
    for idx in range(1, len(benchmark_navs)):
        prev_b = benchmark_navs[idx - 1]
        bench_returns_full.append(
            (benchmark_navs[idx] - prev_b) / prev_b if prev_b > 0 else 0.0
        )
    bench_returns = (
        bench_returns_full[warmup:]
        if len(bench_returns_full) > warmup
        else bench_returns_full
    )
    bench_sharpe = (
        compute_sharpe(bench_returns, annualize=True)
        if len(bench_returns) > 20
        else 0.0
    )

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
        "n_days": n_days,
        "win_rate": round(win_rate, 4),
        "min_trl_months": round(min_trl.min_trl_months, 2),
        "min_trl_pass": min_trl.min_trl_pass,
        "skew": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "final_nav": round(daily_navs[-1], 2) if daily_navs else 0.0,
        "benchmark_sharpe": round(bench_sharpe, 4),
        "regime_counts": regime_counts,
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


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main() -> None:
    """Run the consumer-xly-xlp-v2 backtest and save results."""
    import yaml

    result = run_backtest()

    if "error" in result:
        logger.error("Backtest failed: %s", result["error"])
        sys.exit(1)

    # Save backtest results (without daily_returns for YAML readability)
    strat_dir = Path(f"data/strategies/{SLUG}")
    strat_dir.mkdir(parents=True, exist_ok=True)

    results_for_yaml = {k: v for k, v in result.items() if k != "daily_returns"}
    results_path = strat_dir / "backtest-results.yaml"
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
        "strategy_type": "consumer_cyclical_defensive",
        "family": "consumer_cyclical_defensive",
        "family_trial_number": FAMILY_TRIAL_NUMBER,
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
    print(f"  Consumer XLY/XLP v2: {SLUG}")
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
    print(f"  Trading Days:     {result['n_days']}")
    print(f"  Final NAV:        ${result['final_nav']:,.2f}")
    print(f"  MinTRL Pass:      {result['min_trl_pass']}")
    print(f"  Benchmark Sharpe: {result['benchmark_sharpe']:.4f}")
    print(f"  Regime Counts:    {result['regime_counts']}")
    print("=" * 60)

    # Gate check (preliminary -- full robustness in separate script)
    gates = {
        "Sharpe >= 0.80": result["sharpe_ratio"] >= 0.80,
        "MaxDD < 15%": result["max_drawdown"] < 0.15,
        "DSR >= 0.95": result["dsr"] >= 0.95,
    }
    print("\n  Preliminary Gate Check:")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate}: {status}")
    print()


if __name__ == "__main__":
    main()

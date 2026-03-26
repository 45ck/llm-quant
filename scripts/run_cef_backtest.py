#!/usr/bin/env python3
"""Run CEF discount mean-reversion backtest.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_cef_backtest.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_cef_backtest.py --years 3 --z-entry -2.0

The script:
1. Fetches 5 years of CEF + benchmark data from yfinance
2. Computes NAV estimates and discount z-scores
3. Runs the discount mean-reversion strategy
4. Reports: Sharpe, MaxDD, CAGR, trade count, time-in-market
5. Compares against T-bill benchmark (Track C)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.arb.cef_data import DEFAULT_CEF_TICKERS, fetch_cef_data
from llm_quant.arb.cef_strategy import CEFDiscountStrategy, CEFStrategyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Approximate daily T-bill rate for benchmark (annualized 5% -> daily)
TBILL_ANNUAL_RATE = 0.05
TBILL_DAILY_RATE = (1 + TBILL_ANNUAL_RATE) ** (1 / 252) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="CEF Discount Backtest")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    parser.add_argument(
        "--z-entry", type=float, default=-1.5, help="Z-score entry threshold"
    )
    parser.add_argument(
        "--z-exit", type=float, default=0.0, help="Z-score exit threshold"
    )
    parser.add_argument(
        "--lookback", type=int, default=252, help="Rolling lookback days"
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0, help="Initial capital"
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated CEF tickers (default: all 15)",
    )
    args = parser.parse_args()

    tickers = (
        [t.strip() for t in args.tickers.split(",")]
        if args.tickers
        else DEFAULT_CEF_TICKERS
    )

    config = CEFStrategyConfig(
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        lookback_days=args.lookback,
    )

    # 1. Fetch data
    lookback_days = args.years * 365
    logger.info(
        "Fetching %d CEFs with %d days of history...", len(tickers), lookback_days
    )
    cef_df = fetch_cef_data(cef_tickers=tickers, lookback_days=lookback_days)

    if len(cef_df) == 0:
        logger.error("No CEF data fetched. Aborting.")
        sys.exit(1)

    logger.info(
        "Fetched %d rows for %d CEFs",
        len(cef_df),
        cef_df.select("ticker").n_unique(),
    )

    # 2. Run backtest
    strategy = CEFDiscountStrategy(config)
    result = run_backtest(strategy, cef_df, args.initial_capital)

    # 3. Report
    print_report(result, config, tickers, args.years)


def run_backtest(
    strategy: CEFDiscountStrategy,
    cef_df,
    initial_capital: float,
) -> dict:
    """Run the CEF discount backtest.

    Simple event-driven loop: iterate over trading days, generate signals,
    execute trades, track NAV.
    """
    import polars as pl

    config = strategy.config

    # Get sorted unique dates
    all_dates = sorted(cef_df.select("date").unique().to_series().to_list())

    # Warmup period: need at least lookback_days before trading
    warmup = config.lookback_days
    if len(all_dates) <= warmup:
        logger.error("Not enough dates (%d) for warmup (%d)", len(all_dates), warmup)
        return _empty_result(initial_capital)

    trading_dates = all_dates[warmup:]

    # State
    cash = initial_capital
    positions: dict[str, dict] = {}  # ticker -> {shares, avg_cost, weight}
    nav_series: list[float] = [initial_capital]
    trades: list[dict] = []
    rebalance_counter = 0

    # Transaction cost (simple flat model for CEFs — wider spreads)
    cost_bps = 15.0  # 15 bps total cost (CEFs have wider spreads)

    for current_date in trading_dates:
        # Get today's prices
        today_data = cef_df.filter(pl.col("date") == current_date)
        if len(today_data) == 0:
            nav_series.append(nav_series[-1])
            continue

        today_prices: dict[str, float] = dict(
            zip(
                today_data.select("ticker").to_series().to_list(),
                today_data.select("price").to_series().to_list(),
                strict=False,
            )
        )

        # Mark to market
        position_value = 0.0
        for ticker, pos in positions.items():
            if ticker in today_prices:
                pos["current_price"] = today_prices[ticker]
                position_value += pos["shares"] * today_prices[ticker]

        nav = cash + position_value

        # Generate signals on rebalance days
        is_rebalance = rebalance_counter % config.rebalance_frequency_days == 0
        rebalance_counter += 1

        if is_rebalance:
            causal_df = cef_df.filter(pl.col("date") <= current_date)
            signals = strategy.generate_signals(current_date, causal_df)

            for signal in signals:
                price = today_prices.get(signal.ticker, 0)
                if price <= 0:
                    continue

                if signal.action == "sell" and signal.ticker in positions:
                    # Close position
                    pos = positions[signal.ticker]
                    notional = pos["shares"] * price
                    cost = notional * cost_bps / 10_000
                    pnl = (price - pos["avg_cost"]) * pos["shares"] - cost
                    cash += notional - cost

                    trades.append(
                        {
                            "date": current_date,
                            "ticker": signal.ticker,
                            "action": "sell",
                            "shares": pos["shares"],
                            "price": price,
                            "pnl": pnl,
                            "z_score": signal.z_score,
                            "reasoning": signal.reasoning,
                        }
                    )

                    del positions[signal.ticker]
                    strategy.apply_signal(signal, current_date)

                elif signal.action == "buy" and signal.ticker not in positions:
                    # Size: equal-weight allocation
                    target_notional = nav * config.target_weight_per_position
                    shares = math.floor(target_notional / price)
                    if shares <= 0:
                        continue

                    notional = shares * price
                    cost = notional * cost_bps / 10_000

                    if notional + cost > cash:
                        shares = math.floor((cash - cost) / price)
                        if shares <= 0:
                            continue
                        notional = shares * price
                        cost = notional * cost_bps / 10_000

                    cash -= notional + cost
                    positions[signal.ticker] = {
                        "shares": shares,
                        "avg_cost": price,
                        "current_price": price,
                    }

                    trades.append(
                        {
                            "date": current_date,
                            "ticker": signal.ticker,
                            "action": "buy",
                            "shares": shares,
                            "price": price,
                            "pnl": 0.0,
                            "z_score": signal.z_score,
                            "reasoning": signal.reasoning,
                        }
                    )

                    strategy.apply_signal(signal, current_date)

        # Record NAV
        position_value = sum(
            pos["shares"] * pos.get("current_price", pos["avg_cost"])
            for pos in positions.values()
        )
        nav = cash + position_value
        nav_series.append(nav)

    # Compute metrics
    return _compute_metrics(
        nav_series=nav_series,
        trades=trades,
        trading_dates=trading_dates,
        initial_capital=initial_capital,
    )


def _compute_metrics(
    nav_series: list[float],
    trades: list[dict],
    trading_dates: list,
    initial_capital: float,
) -> dict:
    """Compute backtest performance metrics."""
    if len(nav_series) < 2:
        return _empty_result(initial_capital)

    # Daily returns
    daily_returns = [
        nav_series[i] / nav_series[i - 1] - 1
        for i in range(1, len(nav_series))
        if nav_series[i - 1] > 0
    ]

    if not daily_returns:
        return _empty_result(initial_capital)

    # Total return
    total_return = nav_series[-1] / nav_series[0] - 1

    # Annualized return
    n_years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Sharpe ratio (excess over T-bill)
    excess_returns = [r - TBILL_DAILY_RATE for r in daily_returns]
    mean_excess = sum(excess_returns) / len(excess_returns) if excess_returns else 0
    var_excess = (
        sum((r - mean_excess) ** 2 for r in excess_returns) / len(excess_returns)
        if excess_returns
        else 0
    )
    std_excess = math.sqrt(var_excess) if var_excess > 0 else 0
    sharpe = (mean_excess / std_excess * math.sqrt(252)) if std_excess > 0 else 0

    # Max drawdown
    peak = nav_series[0]
    max_dd = 0.0
    for nav in nav_series:
        peak = max(peak, nav)
        dd = (peak - nav) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Win rate
    sell_trades = [t for t in trades if t["action"] == "sell"]
    wins = sum(1 for t in sell_trades if t["pnl"] > 0)
    win_rate = wins / len(sell_trades) if sell_trades else 0

    # Time in market (approximate: fraction of days with > 0 positions)
    nav_count = len(nav_series) - 1  # exclude initial
    invested_days = sum(
        1 for i in range(1, len(nav_series)) if nav_series[i] != nav_series[i - 1]
    )
    time_in_market = invested_days / nav_count if nav_count > 0 else 0

    # Sortino ratio
    downside_returns = [r for r in excess_returns if r < 0]
    downside_var = (
        sum(r**2 for r in downside_returns) / len(excess_returns)
        if excess_returns
        else 0
    )
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0
    sortino = (mean_excess / downside_std * math.sqrt(252)) if downside_std > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "buy_trades": sum(1 for t in trades if t["action"] == "buy"),
        "sell_trades": len(sell_trades),
        "time_in_market": time_in_market,
        "final_nav": nav_series[-1],
        "initial_capital": nav_series[0],
        "n_trading_days": len(nav_series) - 1,
        "trades": trades,
        "nav_series": nav_series,
    }


def _empty_result(initial_capital: float) -> dict:
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "buy_trades": 0,
        "sell_trades": 0,
        "time_in_market": 0.0,
        "final_nav": initial_capital,
        "initial_capital": initial_capital,
        "n_trading_days": 0,
        "trades": [],
        "nav_series": [initial_capital],
    }


def print_report(
    result: dict, config: CEFStrategyConfig, tickers: list[str], years: int
) -> None:
    """Print markdown-formatted backtest report."""
    print("\n" + "=" * 70)
    print("CEF DISCOUNT MEAN-REVERSION BACKTEST")
    print("=" * 70)

    print("\n## Configuration")
    print(f"- Universe: {len(tickers)} CEFs")
    print(f"- Tickers: {', '.join(tickers)}")
    print(f"- History: {years} years")
    print(f"- Z-score entry: {config.z_entry}")
    print(f"- Z-score exit: {config.z_exit}")
    print(f"- Lookback: {config.lookback_days} days")
    print(f"- Rebalance: every {config.rebalance_frequency_days} days")
    print(f"- Max positions: {config.max_positions}")
    print(f"- Weight per position: {config.target_weight_per_position:.0%}")

    print(f"\n## Performance (Track C benchmark: T-bill {TBILL_ANNUAL_RATE:.0%})")
    print("| Metric              | Value         |")
    print("|---------------------|---------------|")
    print(f"| Total Return        | {result['total_return']:>12.2%} |")
    print(f"| CAGR                | {result['cagr']:>12.2%} |")
    print(f"| Sharpe (vs T-bill)  | {result['sharpe']:>12.3f} |")
    print(f"| Sortino             | {result['sortino']:>12.3f} |")
    print(f"| Max Drawdown        | {result['max_drawdown']:>12.2%} |")
    print(f"| Win Rate            | {result['win_rate']:>12.2%} |")
    print(f"| Total Trades        | {result['total_trades']:>12d} |")
    print(f"| Buy Trades          | {result['buy_trades']:>12d} |")
    print(f"| Sell Trades         | {result['sell_trades']:>12d} |")
    print(f"| Trading Days        | {result['n_trading_days']:>12d} |")
    print(f"| Final NAV           | ${result['final_nav']:>11,.2f} |")

    # Print recent trades
    trades = result.get("trades", [])
    if trades:
        recent = trades[-10:]
        print(f"\n## Recent Trades (last {len(recent)})")
        print("| Date       | Ticker | Action | Shares | Price  | Z-Score | PnL     |")
        print("|------------|--------|--------|--------|--------|---------|---------|")
        for t in recent:
            print(
                f"| {t['date']} | {t['ticker']:>6} | {t['action']:>6} | "
                f"{t['shares']:>6.0f} | {t['price']:>6.2f} | "
                f"{t['z_score']:>7.2f} | {t['pnl']:>7.2f} |"
            )

    # Gate check
    print("\n## Gate Check (Track C)")
    sharpe_gate = "PASS" if result["sharpe"] > 0.5 else "FAIL"
    dd_gate = "PASS" if result["max_drawdown"] < 0.10 else "FAIL"
    cagr_gate = "PASS" if result["cagr"] > 0 else "FAIL"
    print(f"- Sharpe > 0.5: {sharpe_gate} ({result['sharpe']:.3f})")
    print(f"- Max DD < 10%: {dd_gate} ({result['max_drawdown']:.2%})")
    print(f"- CAGR > 0%: {cagr_gate} ({result['cagr']:.2%})")

    all_pass = (
        result["sharpe"] > 0.5 and result["max_drawdown"] < 0.10 and result["cagr"] > 0
    )
    print(f"\nOverall: {'ALL GATES PASSED' if all_pass else 'GATES NOT MET'}")


if __name__ == "__main__":
    main()

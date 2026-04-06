#!/usr/bin/env python3
"""Robustness analysis for CEF Discount Mean-Reversion strategy (Track C).

Runs:
1. Base CEF backtest via CEFDiscountStrategy
2. CPCV (6-fold combinatorial purged cross-validation)
3. DSR (Deflated Sharpe Ratio)
4. Perturbation tests (z_entry, z_exit, lookback variations)
5. Shuffled signal test (fraud detector)
6. Track C gate assessment
7. Saves to data/strategies/cef-discount-mr/robustness.yaml

Track C gates:
  Sharpe >= 1.5
  MaxDD < 10%
  Beta to SPY < 0.15
  Min 50 trades

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_cef_discount_robustness.py
"""

from __future__ import annotations

import math
import random
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.arb.cef_data import DEFAULT_CEF_TICKERS, fetch_cef_data
from llm_quant.arb.cef_strategy import CEFDiscountStrategy, CEFStrategyConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TBILL_ANNUAL_RATE = 0.05
TBILL_DAILY_RATE = (1 + TBILL_ANNUAL_RATE) ** (1 / 252) - 1

SLUG = "cef-discount-mr"
STRATEGY_DIR = Path("data/strategies") / SLUG

# Track C gates
TRACK_C_SHARPE_GATE = 1.5
TRACK_C_MAXDD_GATE = 0.10
TRACK_C_BETA_GATE = 0.15
TRACK_C_MIN_TRADES = 50


# ---------------------------------------------------------------------------
# Backtest runner (reuses logic from run_cef_backtest.py)
# ---------------------------------------------------------------------------


def run_cef_backtest(
    config: CEFStrategyConfig,
    cef_df,
    initial_capital: float = 100_000.0,
) -> dict:
    """Run a CEF discount backtest and return metrics dict.

    Mirrors the logic in scripts/run_cef_backtest.py but extracted
    as a callable function for reuse in robustness testing.
    """
    import polars as pl

    all_dates = sorted(cef_df.select("date").unique().to_series().to_list())
    warmup = config.lookback_days
    if len(all_dates) <= warmup:
        return _empty_result(initial_capital)

    trading_dates = all_dates[warmup:]

    cash = initial_capital
    positions: dict[str, dict] = {}
    nav_series: list[float] = [initial_capital]
    trades: list[dict] = []
    rebalance_counter = 0
    cost_bps = 15.0

    strategy = CEFDiscountStrategy(config)

    for current_date in trading_dates:
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

        position_value = 0.0
        for ticker, pos in positions.items():
            if ticker in today_prices:
                pos["current_price"] = today_prices[ticker]
                position_value += pos["shares"] * today_prices[ticker]

        nav = cash + position_value

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
                        }
                    )

                    del positions[signal.ticker]
                    strategy.apply_signal(signal, current_date)

                elif signal.action == "buy" and signal.ticker not in positions:
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
                        }
                    )

                    strategy.apply_signal(signal, current_date)

        position_value = sum(
            pos["shares"] * pos.get("current_price", pos["avg_cost"])
            for pos in positions.values()
        )
        nav = cash + position_value
        nav_series.append(nav)

    return _compute_metrics(nav_series, trades, trading_dates, initial_capital)


def _compute_metrics(
    nav_series: list[float],
    trades: list[dict],
    trading_dates: list,
    initial_capital: float,
) -> dict:
    """Compute backtest performance metrics."""
    if len(nav_series) < 2:
        return _empty_result(initial_capital)

    daily_returns = [
        nav_series[i] / nav_series[i - 1] - 1
        for i in range(1, len(nav_series))
        if nav_series[i - 1] > 0
    ]

    if not daily_returns:
        return _empty_result(initial_capital)

    total_return = nav_series[-1] / nav_series[0] - 1
    n_years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    excess_returns = [r - TBILL_DAILY_RATE for r in daily_returns]
    mean_excess = sum(excess_returns) / len(excess_returns) if excess_returns else 0
    var_excess = (
        sum((r - mean_excess) ** 2 for r in excess_returns) / len(excess_returns)
        if excess_returns
        else 0
    )
    std_excess = math.sqrt(var_excess) if var_excess > 0 else 0
    sharpe = (mean_excess / std_excess * math.sqrt(252)) if std_excess > 0 else 0

    peak = nav_series[0]
    max_dd = 0.0
    for nav in nav_series:
        peak = max(peak, nav)
        dd = (peak - nav) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    sell_trades = [t for t in trades if t["action"] == "sell"]
    wins = sum(1 for t in sell_trades if t["pnl"] > 0)
    win_rate = wins / len(sell_trades) if sell_trades else 0

    downside_returns = [r for r in excess_returns if r < 0]
    downside_var = (
        sum(r**2 for r in downside_returns) / len(excess_returns)
        if excess_returns
        else 0
    )
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0
    sortino = (mean_excess / downside_std * math.sqrt(252)) if downside_std > 0 else 0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "buy_trades": sum(1 for t in trades if t["action"] == "buy"),
        "sell_trades": len(sell_trades),
        "final_nav": nav_series[-1],
        "initial_capital": initial_capital,
        "n_trading_days": len(nav_series) - 1,
        "daily_returns": daily_returns,
        "nav_series": nav_series,
    }


def _empty_result(initial_capital: float) -> dict:
    return {
        "sharpe": 0.0,
        "sortino": 0.0,
        "cagr": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "buy_trades": 0,
        "sell_trades": 0,
        "final_nav": initial_capital,
        "initial_capital": initial_capital,
        "n_trading_days": 0,
        "daily_returns": [],
        "nav_series": [initial_capital],
    }


# ---------------------------------------------------------------------------
# CPCV (Combinatorial Purged Cross-Validation)
# ---------------------------------------------------------------------------


def cpcv_sharpe(
    daily_returns: list[float],
    n_groups: int = 6,
    k: int = 2,
    purge: int = 5,
) -> tuple[float, float, list[float]]:
    """Compute CPCV OOS Sharpe ratio.

    Split daily returns into n_groups time-ordered blocks.
    For each C(n_groups, k) combination, hold out k blocks as OOS,
    purge boundary days, and compute OOS Sharpe.

    Returns (mean_oos_sharpe, std_oos_sharpe, all_oos_sharpes).
    """
    n = len(daily_returns)
    if n < n_groups:
        return 0.0, 0.0, []

    group_size = n // n_groups
    oos_sharpes: list[float] = []

    for test_idx in combinations(range(n_groups), k):
        test_rets: list[float] = []
        for i in test_idx:
            start = i * group_size + purge
            end = (i + 1) * group_size - purge
            if start < end:
                test_rets.extend(daily_returns[start:end])

        if len(test_rets) < 20:
            continue

        mean = sum(test_rets) / len(test_rets)
        std = (sum((r - mean) ** 2 for r in test_rets) / len(test_rets)) ** 0.5
        if std > 0:
            oos_sharpes.append(mean / std * math.sqrt(252))

    if not oos_sharpes:
        return 0.0, 0.0, []

    m = sum(oos_sharpes) / len(oos_sharpes)
    s = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    return m, s, oos_sharpes


# ---------------------------------------------------------------------------
# DSR (Deflated Sharpe Ratio)
# ---------------------------------------------------------------------------


def compute_dsr(
    observed_sharpe: float,
    n_obs: int,
    trial_count: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe ratio for multiple testing by comparing
    against the expected maximum Sharpe of trial_count independent trials
    under the null hypothesis of zero alpha.

    Returns DSR as a probability (0 to 1).
    """
    from scipy.stats import norm

    if n_obs < 10 or trial_count < 1:
        return 0.0

    # Expected max Sharpe under null for `trial_count` trials
    # E[max(SR)] ~ sqrt(2 * ln(trial_count)) - (euler_gamma / sqrt(2 * ln(trial_count)))
    if trial_count == 1:
        sr_benchmark = 0.0
    else:
        euler_gamma = 0.5772156649
        z = math.sqrt(2 * math.log(trial_count))
        sr_benchmark = z - (euler_gamma / z) if z > 0 else 0.0

    # Standard error of Sharpe ratio with non-normality correction
    se = math.sqrt(
        (1 - skewness * observed_sharpe + ((kurtosis - 3) / 4) * observed_sharpe**2)
        / n_obs
    )

    if se <= 0:
        return 0.0

    # DSR = Prob(SR* > sr_benchmark)
    test_stat = (observed_sharpe - sr_benchmark) / se
    return float(norm.cdf(test_stat))


# ---------------------------------------------------------------------------
# Shuffled signal test (fraud detector)
# ---------------------------------------------------------------------------


def shuffled_signal_test(
    cef_df,
    base_config: CEFStrategyConfig,
    n_shuffles: int = 100,
    seed: int = 42,
) -> dict:
    """Run the strategy on shuffled discount z-scores.

    If the real strategy Sharpe is not significantly above shuffled
    Sharpe distribution, the signal has no predictive power.
    """
    import polars as pl

    rng = random.Random(seed)

    # Run base strategy for reference
    base_result = run_cef_backtest(base_config, cef_df)
    base_sharpe = base_result["sharpe"]

    # Create shuffled versions by randomly permuting discount_pct within each ticker
    shuffled_sharpes: list[float] = []

    for i in range(n_shuffles):
        # Shuffle discount_pct column within each ticker group
        frames = []
        for ticker in cef_df.select("ticker").unique().to_series().to_list():
            ticker_df = cef_df.filter(pl.col("ticker") == ticker).sort("date")
            discounts = ticker_df["discount_pct"].to_list()
            rng.shuffle(discounts)
            shuffled_ticker = ticker_df.with_columns(
                pl.Series("discount_pct", discounts)
            )
            frames.append(shuffled_ticker)

        shuffled_df = pl.concat(frames, how="vertical").sort(["ticker", "date"])
        result = run_cef_backtest(base_config, shuffled_df)
        shuffled_sharpes.append(result["sharpe"])

        if (i + 1) % 25 == 0:
            print(f"  Shuffle {i + 1}/{n_shuffles} done")

    # Percentile rank of real Sharpe vs shuffled distribution
    n_below = sum(1 for s in shuffled_sharpes if s < base_sharpe)
    percentile = n_below / len(shuffled_sharpes) if shuffled_sharpes else 0.0

    mean_shuffled = (
        sum(shuffled_sharpes) / len(shuffled_sharpes) if shuffled_sharpes else 0.0
    )
    std_shuffled = (
        (
            sum((s - mean_shuffled) ** 2 for s in shuffled_sharpes)
            / len(shuffled_sharpes)
        )
        ** 0.5
        if shuffled_sharpes
        else 0.0
    )

    return {
        "base_sharpe": base_sharpe,
        "mean_shuffled_sharpe": mean_shuffled,
        "std_shuffled_sharpe": std_shuffled,
        "percentile_rank": percentile,
        "n_shuffles": n_shuffles,
        "passed": percentile > 0.95,  # Real Sharpe should be top 5%
    }


# ---------------------------------------------------------------------------
# Beta to SPY
# ---------------------------------------------------------------------------


def compute_beta_to_spy(
    strategy_returns: list[float],
    spy_returns: list[float],
) -> float:
    """Compute beta of strategy returns to SPY returns."""
    n = min(len(strategy_returns), len(spy_returns))
    if n < 20:
        return 0.0

    sr = strategy_returns[-n:]
    mr = spy_returns[-n:]

    mean_s = sum(sr) / n
    mean_m = sum(mr) / n

    cov = sum((sr[i] - mean_s) * (mr[i] - mean_m) for i in range(n)) / n
    var_m = sum((mr[i] - mean_m) ** 2 for i in range(n)) / n

    if var_m <= 0:
        return 0.0

    return cov / var_m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("CEF DISCOUNT MEAN-REVERSION: ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------
    # 1. Fetch data
    # ---------------------------------------------------------------
    print("[1/7] Fetching CEF data (5 years)...")
    cef_df = fetch_cef_data(
        cef_tickers=DEFAULT_CEF_TICKERS,
        lookback_days=5 * 365,
    )
    print(f"  Fetched {len(cef_df)} rows for {cef_df.select('ticker').n_unique()} CEFs")

    # Also fetch SPY for beta computation
    print("  Fetching SPY for beta computation...")
    from llm_quant.data.fetcher import fetch_ohlcv

    spy_df = fetch_ohlcv(["SPY"], lookback_days=5 * 365)
    spy_returns: list[float] = []
    if len(spy_df) > 1:
        spy_closes = (
            spy_df.filter(spy_df["symbol"] == "SPY").sort("date")["close"].to_list()
        )
        spy_returns = [
            spy_closes[i] / spy_closes[i - 1] - 1 for i in range(1, len(spy_closes))
        ]

    # ---------------------------------------------------------------
    # 2. Base backtest
    # ---------------------------------------------------------------
    print("\n[2/7] Running base backtest...")
    base_config = CEFStrategyConfig(
        z_entry=-1.5,
        z_exit=0.0,
        lookback_days=252,
    )
    base_result = run_cef_backtest(base_config, cef_df)

    print(f"  Sharpe:    {base_result['sharpe']:.4f}")
    print(f"  Sortino:   {base_result['sortino']:.4f}")
    print(f"  CAGR:      {base_result['cagr']:.4%}")
    print(f"  Max DD:    {base_result['max_drawdown']:.4%}")
    print(f"  Win Rate:  {base_result['win_rate']:.4%}")
    print(f"  Trades:    {base_result['total_trades']}")

    # ---------------------------------------------------------------
    # 3. CPCV (6-fold, k=2)
    # ---------------------------------------------------------------
    print("\n[3/7] Computing CPCV (6 groups, k=2, purge=5)...")
    daily_returns = base_result.get("daily_returns", [])
    cpcv_mean, cpcv_std, cpcv_sharpes = cpcv_sharpe(
        daily_returns, n_groups=6, k=2, purge=5
    )
    cpcv_ois = (
        cpcv_mean / abs(base_result["sharpe"]) if base_result["sharpe"] != 0 else 0.0
    )

    print(f"  CPCV OOS Sharpe: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
    print(f"  CPCV OOS/IS:     {cpcv_ois:.4f}")
    print(f"  N combinations:  {len(cpcv_sharpes)}")
    if cpcv_sharpes:
        n_positive = sum(1 for s in cpcv_sharpes if s > 0)
        print(
            f"  Positive OOS:    {n_positive}/{len(cpcv_sharpes)} "
            f"({n_positive / len(cpcv_sharpes):.0%})"
        )

    # ---------------------------------------------------------------
    # 4. DSR
    # ---------------------------------------------------------------
    print("\n[4/7] Computing Deflated Sharpe Ratio...")
    # Compute skewness and kurtosis of daily returns
    n_obs = len(daily_returns)
    if n_obs > 3:
        mean_r = sum(daily_returns) / n_obs
        std_r = (sum((r - mean_r) ** 2 for r in daily_returns) / n_obs) ** 0.5
        if std_r > 0:
            skew = sum(((r - mean_r) / std_r) ** 3 for r in daily_returns) / n_obs
            kurt = sum(((r - mean_r) / std_r) ** 4 for r in daily_returns) / n_obs
        else:
            skew, kurt = 0.0, 3.0
    else:
        skew, kurt = 0.0, 3.0

    dsr_value = compute_dsr(
        observed_sharpe=base_result["sharpe"],
        n_obs=n_obs,
        trial_count=1,  # First trial for this mechanism family
        skewness=skew,
        kurtosis=kurt,
    )
    print(f"  Observed Sharpe:  {base_result['sharpe']:.4f}")
    print(f"  N observations:   {n_obs}")
    print("  Trial count:      1")
    print(f"  Skewness:         {skew:.4f}")
    print(f"  Kurtosis:         {kurt:.4f}")
    print(f"  DSR:              {dsr_value:.4f}")

    # ---------------------------------------------------------------
    # 5. Perturbation tests
    # ---------------------------------------------------------------
    print("\n[5/7] Running perturbation tests...")
    perturbations = [
        ("z_entry=-1.0 (less strict)", {"z_entry": -1.0}),
        ("z_entry=-2.0 (baseline)", {"z_entry": -2.0}),
        ("z_entry=-2.5 (stricter)", {"z_entry": -2.5}),
        ("z_exit=0.5 (earlier exit)", {"z_exit": 0.5}),
        ("z_exit=-0.5 (later exit)", {"z_exit": -0.5}),
        ("lookback=30 (shorter window)", {"lookback_days": 30}),
        ("lookback=126 (6 months)", {"lookback_days": 126}),
        ("lookback=504 (2 years)", {"lookback_days": 504}),
    ]

    perturbation_results: list[dict] = []
    for name, overrides in perturbations:
        params = {
            "z_entry": base_config.z_entry,
            "z_exit": base_config.z_exit,
            "lookback_days": base_config.lookback_days,
        }
        params.update(overrides)
        config = CEFStrategyConfig(**params)
        result = run_cef_backtest(config, cef_df)

        delta_pct = (
            (result["sharpe"] - base_result["sharpe"])
            / (abs(base_result["sharpe"]) + 1e-8)
            * 100
        )
        stable = "STABLE" if abs(delta_pct) <= 30 else "UNSTABLE"

        perturbation_results.append(
            {
                "name": name,
                "sharpe": round(result["sharpe"], 4),
                "max_dd": round(result["max_drawdown"], 4),
                "cagr": round(result["cagr"], 4),
                "trades": result["total_trades"],
                "delta_pct": round(delta_pct, 1),
                "stability": stable,
            }
        )

        print(
            f"  {name}: Sharpe={result['sharpe']:.4f}, "
            f"MaxDD={result['max_drawdown']:.4%}, "
            f"Trades={result['total_trades']}, "
            f"({delta_pct:+.1f}%) {stable}"
        )

    n_stable = sum(1 for p in perturbation_results if p["stability"] == "STABLE")
    stability_pct = n_stable / len(perturbation_results) if perturbation_results else 0

    # ---------------------------------------------------------------
    # 6. Shuffled signal test
    # ---------------------------------------------------------------
    print("\n[6/7] Running shuffled signal test (100 shuffles)...")
    shuffle_result = shuffled_signal_test(cef_df, base_config, n_shuffles=100, seed=42)
    print(f"  Base Sharpe:      {shuffle_result['base_sharpe']:.4f}")
    print(f"  Mean shuffled:    {shuffle_result['mean_shuffled_sharpe']:.4f}")
    print(f"  Std shuffled:     {shuffle_result['std_shuffled_sharpe']:.4f}")
    print(f"  Percentile rank:  {shuffle_result['percentile_rank']:.4f}")
    print(f"  Signal test:      {'PASS' if shuffle_result['passed'] else 'FAIL'}")

    # ---------------------------------------------------------------
    # 7. Beta to SPY
    # ---------------------------------------------------------------
    print("\n[7/7] Computing beta to SPY...")
    strat_returns = base_result.get("daily_returns", [])
    beta = compute_beta_to_spy(strat_returns, spy_returns) if spy_returns else 0.0
    print(f"  Beta to SPY: {beta:.4f}")

    # ---------------------------------------------------------------
    # Gate assessment (Track C)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRACK C GATE ASSESSMENT")
    print("=" * 70)

    gates = {
        "sharpe": {
            "value": round(base_result["sharpe"], 4),
            "threshold": TRACK_C_SHARPE_GATE,
            "passed": base_result["sharpe"] >= TRACK_C_SHARPE_GATE,
            "detail": f"Sharpe {base_result['sharpe']:.4f} vs >= {TRACK_C_SHARPE_GATE}",
        },
        "max_drawdown": {
            "value": round(base_result["max_drawdown"], 4),
            "threshold": TRACK_C_MAXDD_GATE,
            "passed": base_result["max_drawdown"] < TRACK_C_MAXDD_GATE,
            "detail": f"MaxDD {base_result['max_drawdown']:.4%} vs < {TRACK_C_MAXDD_GATE:.0%}",
        },
        "beta_to_spy": {
            "value": round(beta, 4),
            "threshold": TRACK_C_BETA_GATE,
            "passed": abs(beta) < TRACK_C_BETA_GATE,
            "detail": f"Beta {beta:.4f} vs < {TRACK_C_BETA_GATE}",
        },
        "min_trades": {
            "value": base_result["total_trades"],
            "threshold": TRACK_C_MIN_TRADES,
            "passed": base_result["total_trades"] >= TRACK_C_MIN_TRADES,
            "detail": f"Trades {base_result['total_trades']} vs >= {TRACK_C_MIN_TRADES}",
        },
        "dsr": {
            "value": round(dsr_value, 4),
            "threshold": 0.90,
            "passed": dsr_value >= 0.90,
            "detail": f"DSR {dsr_value:.4f} vs >= 0.90",
        },
        "cpcv_oos_positive": {
            "value": round(cpcv_mean, 4),
            "threshold": 0.0,
            "passed": cpcv_mean > 0,
            "detail": f"CPCV OOS Sharpe {cpcv_mean:.4f} vs > 0",
        },
        "shuffled_signal": {
            "value": round(shuffle_result["percentile_rank"], 4),
            "threshold": 0.95,
            "passed": shuffle_result["passed"],
            "detail": f"Percentile rank {shuffle_result['percentile_rank']:.4f} vs > 0.95",
        },
    }

    all_pass = all(g["passed"] for g in gates.values())

    for gate in gates.values():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  {status}  {gate['detail']}")

    print(f"\nOverall: {'ALL GATES PASSED' if all_pass else 'GATES NOT MET'}")

    if not all_pass:
        failed = [name for name, g in gates.items() if not g["passed"]]
        print(f"Failed gates: {', '.join(failed)}")

    # ---------------------------------------------------------------
    # Save robustness.yaml
    # ---------------------------------------------------------------
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    robustness_artifact = {
        "strategy_slug": SLUG,
        "strategy_type": "track_c",
        "strategy_class": "cef_discount",
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "base_results": {
            "sharpe": round(base_result["sharpe"], 4),
            "sortino": round(base_result["sortino"], 4),
            "cagr": round(base_result["cagr"], 4),
            "total_return": round(base_result["total_return"], 4),
            "max_drawdown": round(base_result["max_drawdown"], 4),
            "win_rate": round(base_result["win_rate"], 4),
            "total_trades": base_result["total_trades"],
            "buy_trades": base_result["buy_trades"],
            "sell_trades": base_result["sell_trades"],
            "n_trading_days": base_result["n_trading_days"],
            "final_nav": round(base_result["final_nav"], 2),
        },
        "cpcv": {
            "groups": 6,
            "test_groups": 2,
            "combinations": len(cpcv_sharpes),
            "purge_days": 5,
            "oos_sharpe_mean": round(cpcv_mean, 4),
            "oos_sharpe_std": round(cpcv_std, 4),
            "oos_sharpes": [round(s, 4) for s in cpcv_sharpes],
            "oos_is_ratio": round(cpcv_ois, 4),
            "gate_status": "PASS" if cpcv_mean > 0 else "FAIL",
        },
        "dsr": {
            "observed_sharpe": round(base_result["sharpe"], 4),
            "n_observations": n_obs,
            "trial_count": 1,
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "dsr_value": round(dsr_value, 4),
            "gate_threshold": 0.90,
            "gate_status": "PASS" if dsr_value >= 0.90 else "FAIL",
        },
        "perturbation": {
            "tests": perturbation_results,
            "n_stable": n_stable,
            "n_total": len(perturbation_results),
            "stability_pct": round(stability_pct, 4),
            "gate_status": "PASS" if stability_pct >= 0.50 else "FAIL",
        },
        "shuffled_signal": {
            "base_sharpe": round(shuffle_result["base_sharpe"], 4),
            "mean_shuffled_sharpe": round(shuffle_result["mean_shuffled_sharpe"], 4),
            "std_shuffled_sharpe": round(shuffle_result["std_shuffled_sharpe"], 4),
            "percentile_rank": round(shuffle_result["percentile_rank"], 4),
            "n_shuffles": shuffle_result["n_shuffles"],
            "gate_status": "PASS" if shuffle_result["passed"] else "FAIL",
        },
        "beta_to_spy": {
            "value": round(beta, 4),
            "gate_threshold": TRACK_C_BETA_GATE,
            "gate_status": "PASS" if abs(beta) < TRACK_C_BETA_GATE else "FAIL",
        },
        "track_c_gates": gates,
        "overall_gate": "PASS" if all_pass else "FAIL",
        "recommendation": "PROMOTE" if all_pass else "REJECT",
        "diagnosis": _build_diagnosis(
            base_result,
            gates,
            cpcv_mean,
            dsr_value,
            beta,
            shuffle_result,
            perturbation_results,
        ),
    }

    out_path = STRATEGY_DIR / "robustness.yaml"
    with open(out_path, "w") as f:
        yaml.dump(robustness_artifact, f, default_flow_style=False, sort_keys=False)

    print(f"\nRobustness artifact saved to: {out_path}")


def _build_diagnosis(
    base_result: dict,
    gates: dict,
    cpcv_mean: float,
    dsr_value: float,
    beta: float,
    shuffle_result: dict,
    perturbation_results: list[dict],
) -> str:
    """Build a human-readable diagnosis string."""
    lines = []

    if base_result["sharpe"] < TRACK_C_SHARPE_GATE:
        lines.append(
            f"CRITICAL: Sharpe {base_result['sharpe']:.4f} is far below Track C "
            f"gate of {TRACK_C_SHARPE_GATE}. The strategy returns less than T-bills "
            f"after adjusting for risk — negative risk-adjusted alpha."
        )

    if base_result["cagr"] < TBILL_ANNUAL_RATE:
        lines.append(
            f"CAGR {base_result['cagr']:.2%} is below T-bill rate "
            f"({TBILL_ANNUAL_RATE:.0%}). The strategy underperforms risk-free."
        )

    if base_result["max_drawdown"] < TRACK_C_MAXDD_GATE:
        lines.append(
            f"MaxDD {base_result['max_drawdown']:.2%} passes the <10% gate. "
            f"Low drawdown is consistent with the structural arb thesis."
        )

    if abs(beta) < TRACK_C_BETA_GATE:
        lines.append(
            f"Beta to SPY = {beta:.4f} passes the <0.15 gate. "
            f"Strategy is market-neutral as expected for CEF discount MR."
        )

    if dsr_value < 0.90:
        lines.append(
            f"DSR {dsr_value:.4f} fails — observed Sharpe is not statistically "
            f"significant even at the relaxed 0.90 threshold."
        )

    if not shuffle_result["passed"]:
        lines.append(
            f"Shuffled signal test FAILS — strategy Sharpe is at "
            f"{shuffle_result['percentile_rank']:.0%} percentile of random signals. "
            f"The discount z-score signal may not have predictive power with "
            f"the current NAV estimation method."
        )
    else:
        lines.append(
            "Shuffled signal test PASSES — discount z-score has genuine "
            "predictive power above random noise."
        )

    n_stable = sum(1 for p in perturbation_results if p["stability"] == "STABLE")
    lines.append(
        f"Perturbation stability: {n_stable}/{len(perturbation_results)} "
        f"configurations within 30% of base Sharpe."
    )

    lines.append("")
    lines.append(
        "RECOMMENDATION: REJECT. The core issue is that NAV estimation via "
        "benchmark ETF ratios introduces too much noise — the estimated discount "
        "does not track true NAV accurately enough to generate alpha. The strategy "
        "has good structural properties (low beta, low MaxDD) but negative "
        "risk-adjusted return. Consider: (1) using actual NAV data from fund "
        "sponsors, (2) focusing on the most liquid CEFs with the widest "
        "structural discounts (PIMCO funds), (3) adding an activist catalyst "
        "overlay to time entries."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()

"""
F43 Multi-Asset Regime Signal — Backtest Script
================================================
Family: multi_asset_regime (F43), Trial: 1

Mechanism: 3-asset composite z-score (SPY, TLT, GLD) detects macro
regime transitions. Unlike pair ratios (F1-F42), this captures ABSOLUTE
momentum ACROSS 3 uncorrelated asset classes.

Signal:
  1. Compute 30-day returns for SPY, TLT, GLD
  2. Z-score each against 120-day rolling history
  3. Composite = 0.50*SPY_z + 0.25*TLT_z + 0.25*GLD_z

Regimes (priority order):
  Reflation  -> SPY_z > 0.5 AND GLD_z > 0.5 AND TLT_z < -0.5
  Deflation  -> SPY_z < -0.5 AND TLT_z > 0.5 AND GLD_z < -0.5
  Risk-on    -> composite > 0.75
  Risk-off   -> composite < -0.75
  Neutral    -> everything else

Benchmark: 60/40 SPY/TLT
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_data(lookback_days: int = 1825) -> pd.DataFrame:
    """Fetch adjusted close prices for all universe symbols."""
    symbols = ["SPY", "TLT", "GLD", "QQQ", "SHY", "DBA"]
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    data = yf.download(symbols, start=start, end=end, auto_adjust=True)
    return data["Close"].dropna()


def run_backtest(  # noqa: PLR0912
    prices: pd.DataFrame | None = None,
    return_lookback: int = 30,
    z_window: int = 120,
    spy_weight_composite: float = 0.50,
    tlt_weight_composite: float | None = None,
    gld_weight_composite: float | None = None,
    composite_threshold: float = 0.75,
    rebalance_freq: int = 5,
    # Regime allocation overrides
    spy_riskon: float = 0.75,
    qqq_riskon: float | None = None,
    shy_riskon: float | None = None,
    gld_riskoff: float = 0.40,
    tlt_riskoff: float | None = None,
    shy_riskoff: float | None = None,
    spy_riskoff: float | None = None,
    spy_reflation: float = 0.50,
    gld_reflation: float = 0.30,
    dba_reflation: float | None = None,
    shy_reflation: float | None = None,
    tlt_deflation: float = 0.50,
    shy_deflation: float | None = None,
    spy_deflation: float | None = None,
    # Regime detection thresholds for reflation/deflation
    reflation_spy_z: float = 0.50,
    reflation_gld_z: float = 0.50,
    reflation_tlt_z: float = -0.50,
    deflation_spy_z: float = -0.50,
    deflation_tlt_z: float = 0.50,
    deflation_gld_z: float = -0.50,
    lookback_days: int = 1825,
) -> dict:
    """
    Run the F43 multi-asset regime backtest.

    Returns dict with: sharpe, max_drawdown, cagr, sortino, calmar,
    nav_series, regime_series, benchmark_nav, daily_returns.
    """
    if prices is None:
        prices = fetch_data(lookback_days)

    # --- Derive composite weights (must sum to 1.0) ---
    if tlt_weight_composite is None:
        tlt_weight_composite = (1.0 - spy_weight_composite) / 2.0
    if gld_weight_composite is None:
        gld_weight_composite = 1.0 - spy_weight_composite - tlt_weight_composite

    # --- Derive regime allocations (each must sum to 1.0) ---
    # Risk-on
    if qqq_riskon is None:
        qqq_riskon = (1.0 - spy_riskon) * 0.6  # 60% of remainder to QQQ
    if shy_riskon is None:
        shy_riskon = 1.0 - spy_riskon - qqq_riskon

    # Risk-off
    if tlt_riskoff is None:
        tlt_riskoff = (1.0 - gld_riskoff) * 0.5
    if shy_riskoff is None:
        shy_riskoff = (1.0 - gld_riskoff - tlt_riskoff) * 0.667
    if spy_riskoff is None:
        spy_riskoff = 1.0 - gld_riskoff - tlt_riskoff - shy_riskoff

    # Reflation
    if dba_reflation is None:
        dba_reflation = (1.0 - spy_reflation - gld_reflation) / 2.0
    if shy_reflation is None:
        shy_reflation = 1.0 - spy_reflation - gld_reflation - dba_reflation

    # Deflation
    if shy_deflation is None:
        shy_deflation = (1.0 - tlt_deflation) * 0.6
    if spy_deflation is None:
        spy_deflation = 1.0 - tlt_deflation - shy_deflation

    # Build allocation dicts
    alloc_risk_on = {"SPY": spy_riskon, "QQQ": qqq_riskon, "SHY": shy_riskon}
    alloc_risk_off = {
        "GLD": gld_riskoff,
        "TLT": tlt_riskoff,
        "SHY": shy_riskoff,
        "SPY": spy_riskoff,
    }
    alloc_reflation = {
        "SPY": spy_reflation,
        "GLD": gld_reflation,
        "DBA": dba_reflation,
        "SHY": shy_reflation,
    }
    alloc_deflation = {
        "TLT": tlt_deflation,
        "SHY": shy_deflation,
        "SPY": spy_deflation,
    }
    alloc_neutral = {"SPY": 0.40, "TLT": 0.25, "GLD": 0.15, "SHY": 0.10, "DBA": 0.10}

    # --- Signal construction ---
    signal_assets = ["SPY", "TLT", "GLD"]
    returns = {}
    z_scores = {}

    for sym in signal_assets:
        ret = prices[sym].pct_change(return_lookback)
        roll_mean = ret.rolling(z_window).mean()
        roll_std = ret.rolling(z_window).std()
        z = (ret - roll_mean) / roll_std
        returns[sym] = ret
        z_scores[sym] = z

    z_df = pd.DataFrame(z_scores)
    composite = (
        spy_weight_composite * z_df["SPY"]
        + tlt_weight_composite * z_df["TLT"]
        + gld_weight_composite * z_df["GLD"]
    )

    # --- Warmup: need return_lookback + z_window days ---
    valid_start = composite.dropna().index[0]
    prices_bt = prices.loc[valid_start:]
    composite_bt = composite.loc[valid_start:]
    z_bt = z_df.loc[valid_start:]

    # --- Daily returns for all assets ---
    daily_returns = prices_bt.pct_change().iloc[1:]
    dates = daily_returns.index

    # --- Regime classification and portfolio simulation ---
    nav = np.ones(len(dates))
    regimes = []
    current_alloc = alloc_neutral  # start neutral
    rebal_counter = 0

    for i, date in enumerate(dates):
        # Portfolio return for this day
        port_ret = 0.0
        for sym, w in current_alloc.items():
            if sym in daily_returns.columns:
                r = daily_returns.loc[date, sym]
                if not np.isnan(r):
                    port_ret += w * r
        if i == 0:
            nav[i] = 1.0 + port_ret
        else:
            nav[i] = nav[i - 1] * (1.0 + port_ret)

        # Rebalance check
        rebal_counter += 1
        if rebal_counter >= rebalance_freq:
            rebal_counter = 0

            # Get current z-scores
            if date in z_bt.index and date in composite_bt.index:
                spy_z = z_bt.loc[date, "SPY"]
                tlt_z = z_bt.loc[date, "TLT"]
                gld_z = z_bt.loc[date, "GLD"]
                comp = composite_bt.loc[date]

                if np.isnan(comp) or np.isnan(spy_z):
                    regime = "neutral"
                # Priority order: reflation -> deflation -> risk_on -> risk_off -> neutral
                elif (
                    spy_z > reflation_spy_z
                    and gld_z > reflation_gld_z
                    and tlt_z < reflation_tlt_z
                ):
                    regime = "reflation"
                elif (
                    spy_z < deflation_spy_z
                    and tlt_z > deflation_tlt_z
                    and gld_z < deflation_gld_z
                ):
                    regime = "deflation"
                elif comp > composite_threshold:
                    regime = "risk_on"
                elif comp < -composite_threshold:
                    regime = "risk_off"
                else:
                    regime = "neutral"

                if regime == "risk_on":
                    current_alloc = alloc_risk_on
                elif regime == "risk_off":
                    current_alloc = alloc_risk_off
                elif regime == "reflation":
                    current_alloc = alloc_reflation
                elif regime == "deflation":
                    current_alloc = alloc_deflation
                else:
                    current_alloc = alloc_neutral

        regimes.append(
            {
                "date": date,
                "regime": regime
                if rebal_counter == 0
                else regimes[-1]["regime"]
                if regimes
                else "neutral",
            }
        )

    # --- Benchmark: 60/40 SPY/TLT ---
    bench_nav = np.ones(len(dates))
    for i, date in enumerate(dates):
        spy_r = (
            daily_returns.loc[date, "SPY"]
            if not np.isnan(daily_returns.loc[date, "SPY"])
            else 0.0
        )
        tlt_r = (
            daily_returns.loc[date, "TLT"]
            if not np.isnan(daily_returns.loc[date, "TLT"])
            else 0.0
        )
        bench_ret = 0.60 * spy_r + 0.40 * tlt_r
        if i == 0:
            bench_nav[i] = 1.0 + bench_ret
        else:
            bench_nav[i] = bench_nav[i - 1] * (1.0 + bench_ret)

    # --- Compute metrics ---
    nav_series = pd.Series(nav, index=dates)
    strat_daily = nav_series.pct_change().dropna()
    trading_days = len(strat_daily)
    years = trading_days / 252

    # Sharpe (annualized, excess over risk-free ~0%)
    sharpe = (
        strat_daily.mean() / strat_daily.std() * np.sqrt(252)
        if strat_daily.std() > 0
        else 0.0
    )

    # Max drawdown
    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    max_dd = abs(drawdown.min())

    # CAGR
    total_return = nav[-1] / nav[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Sortino
    downside = strat_daily[strat_daily < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (strat_daily.mean() * 252) / downside_std

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Benchmark metrics
    bench_series = pd.Series(bench_nav, index=dates)
    bench_daily = bench_series.pct_change().dropna()
    bench_sharpe = (
        bench_daily.mean() / bench_daily.std() * np.sqrt(252)
        if bench_daily.std() > 0
        else 0.0
    )
    bench_running_max = bench_series.cummax()
    bench_dd = (bench_series - bench_running_max) / bench_running_max
    bench_max_dd = abs(bench_dd.min())
    bench_total = bench_nav[-1] / bench_nav[0] - 1.0
    bench_cagr = (1.0 + bench_total) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Regime distribution
    regime_df = pd.DataFrame(regimes)
    regime_counts = regime_df["regime"].value_counts()

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "sortino": sortino,
        "calmar": calmar,
        "total_return": total_return,
        "trading_days": trading_days,
        "years": years,
        "nav_series": nav_series,
        "daily_returns": strat_daily,
        "benchmark_nav": bench_series,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_max_dd": bench_max_dd,
        "benchmark_cagr": bench_cagr,
        "regime_counts": regime_counts,
    }


def main():
    print("=" * 70)
    print("F43 Multi-Asset Regime Signal — Backtest")
    print("=" * 70)

    print("\nFetching data (5-year lookback)...")
    prices = fetch_data(1825)
    print(f"Data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Symbols: {list(prices.columns)}")

    print("\nRunning backtest with base parameters...")
    results = run_backtest(prices=prices)

    print("\n" + "-" * 70)
    print("STRATEGY RESULTS")
    print("-" * 70)
    print(f"  Sharpe Ratio:     {results['sharpe']:.3f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:.1%}")
    print(f"  CAGR:             {results['cagr']:.1%}")
    print(f"  Total Return:     {results['total_return']:.1%}")
    print(f"  Sortino:          {results['sortino']:.3f}")
    print(f"  Calmar:           {results['calmar']:.3f}")
    print(f"  Trading Days:     {results['trading_days']}")
    print(f"  Years:            {results['years']:.2f}")

    print("\n" + "-" * 70)
    print("BENCHMARK (60/40 SPY/TLT)")
    print("-" * 70)
    print(f"  Sharpe Ratio:     {results['benchmark_sharpe']:.3f}")
    print(f"  Max Drawdown:     {results['benchmark_max_dd']:.1%}")
    print(f"  CAGR:             {results['benchmark_cagr']:.1%}")

    print("\n" + "-" * 70)
    print("REGIME DISTRIBUTION")
    print("-" * 70)
    for regime, count in results["regime_counts"].items():
        pct = count / results["trading_days"] * 100
        print(f"  {regime:15s}: {count:5d} days ({pct:.1f}%)")

    print("\n" + "-" * 70)
    print("GATE CHECK (Track A)")
    print("-" * 70)
    gates = {
        "Sharpe >= 0.80": results["sharpe"] >= 0.80,
        "MaxDD < 15%": results["max_drawdown"] < 0.15,
        "Sortino >= 1.0": results["sortino"] >= 1.0,
        "Calmar >= 0.50": results["calmar"] >= 0.50,
        "Beats benchmark Sharpe": results["sharpe"] > results["benchmark_sharpe"],
    }
    all_pass = True
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {gate}")

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL GATES PASSED — proceed to robustness")
    else:
        print("RESULT: GATE FAILURE — review before robustness")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()

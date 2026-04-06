#!/usr/bin/env python3
"""Robustness analysis for F35: Cyclical-Defensive Sector Breadth.

Runs the following robustness tests:
1. Base backtest (from backtest script)
2. Shuffled signal test (1000 shuffles)
3. CPCV (15 folds, ~70 trading days each)
4. Perturbation tests across all key parameters
5. Gate check against Track A thresholds

Gates:
  - Sharpe >= 0.80
  - MaxDD < 15%
  - DSR >= 0.95
  - CPCV OOS/IS > 0
  - Perturbation stability >= 60%
  - Shuffled signal p < 0.05

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/robustness_f35_cyclical_defensive.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.metrics import compute_sharpe
from llm_quant.backtest.robustness import run_cpcv
from llm_quant.data.fetcher import fetch_ohlcv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from backtest_f35_cyclical_defensive import (  # noqa: E402
    ALL_SYMBOLS,
    INITIAL_CAPITAL,
    LOOKBACK_DAYS,
    SLUG,
    TRADEABLE_SYMBOLS,
    BacktestParams,
    _run_backtest_on_data,
)

STRAT_DIR = Path(f"data/strategies/{SLUG}")


# ------------------------------------------------------------------------------
# Shuffled signal test
# ------------------------------------------------------------------------------


def run_shuffled_signal_test(
    prices_df: pl.DataFrame,
    base_sharpe: float,
    n_shuffles: int = 1000,
) -> dict:
    """Test whether the regime signal is genuine by shuffling regime labels.

    At each rebalance point, instead of using the real regime classification,
    we randomly assign one of {expansion, contraction, transition}. If the
    real signal is genuine, the shuffled Sharpe distribution should be lower.

    Returns
    -------
    dict
        p_value, shuffled_mean, shuffled_std, base_sharpe, n_shuffles, passed
    """
    logger.info("Running shuffled signal test (%d shuffles)...", n_shuffles)

    rng = np.random.default_rng(42)
    shuffled_sharpes: list[float] = []

    for shuffle_i in range(n_shuffles):
        result = _run_shuffled_backtest(prices_df, rng)
        if result is not None:
            shuffled_sharpes.append(result)

        if (shuffle_i + 1) % 100 == 0:
            logger.info(
                "  Shuffle %d/%d done (mean Sharpe: %.4f)",
                shuffle_i + 1,
                n_shuffles,
                np.mean(shuffled_sharpes) if shuffled_sharpes else 0.0,
            )

    if len(shuffled_sharpes) < 10:
        logger.warning("Too few successful shuffles: %d", len(shuffled_sharpes))
        return {
            "p_value": 1.0,
            "shuffled_mean": 0.0,
            "shuffled_std": 0.0,
            "base_sharpe": base_sharpe,
            "n_shuffles": len(shuffled_sharpes),
            "passed": False,
        }

    arr = np.array(shuffled_sharpes)
    p_value = float(np.mean(arr >= base_sharpe))

    return {
        "p_value": round(p_value, 6),
        "shuffled_mean": round(float(np.mean(arr)), 4),
        "shuffled_std": round(float(np.std(arr)), 4),
        "base_sharpe": round(base_sharpe, 4),
        "n_shuffles": len(shuffled_sharpes),
        "passed": p_value < 0.05,
    }


def _run_shuffled_backtest(  # noqa: PLR0912
    prices_df: pl.DataFrame,
    rng: np.random.Generator,
) -> float | None:
    """Run a single backtest with shuffled regime assignments.

    We override the regime classification by randomly choosing among the three
    regimes at each rebalance, while keeping everything else identical.
    """
    params = BacktestParams()

    # Build per-symbol close price series
    close_by_symbol: dict[str, dict[str, float]] = {}

    for sym in ALL_SYMBOLS:
        sym_data = prices_df.filter(pl.col("symbol") == sym).sort("date")
        if len(sym_data) == 0:
            continue
        dates = sym_data["date"].to_list()
        closes = sym_data["close"].to_list()
        close_by_symbol[sym] = {str(d): c for d, c in zip(dates, closes, strict=True)}

    missing = [s for s in ALL_SYMBOLS if s not in close_by_symbol]
    if missing:
        return None

    date_sets = [set(close_by_symbol[sym].keys()) for sym in ALL_SYMBOLS]
    common_dates = sorted(set.intersection(*date_sets))

    warmup = max(params.sma_period, params.ratio_lookback, params.xli_lookback) + 5
    if len(common_dates) < warmup + 100:
        return None

    nav = INITIAL_CAPITAL
    holdings: dict[str, float] = {}
    daily_navs: list[float] = []
    daily_returns: list[float] = []
    days_since_rebalance = params.rebalance_frequency_days
    cost_per_trade = params.round_trip_cost_bps / 10_000.0

    regimes = ["expansion", "contraction", "transition"]

    for i, date_str in enumerate(common_dates):
        if i < warmup:
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
            continue

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

        days_since_rebalance += 1
        if days_since_rebalance < params.rebalance_frequency_days:
            continue
        days_since_rebalance = 0

        # SHUFFLED: randomly choose regime
        regime = str(rng.choice(regimes))
        target_weights = params.get_regime_weights(regime)

        # Execute rebalance with costs
        current_weights: dict[str, float] = {}
        for sym_h, shares_h in holdings.items():
            price_h = close_by_symbol[sym_h].get(date_str, 0.0)
            current_weights[sym_h] = (shares_h * price_h) / nav if nav > 0 else 0.0

        all_syms = set(target_weights.keys()) | set(current_weights.keys())
        total_turnover_value = 0.0
        for sym_t in all_syms:
            tw = target_weights.get(sym_t, 0.0)
            cw = current_weights.get(sym_t, 0.0)
            total_turnover_value += abs(tw - cw) * nav

        total_cost = (total_turnover_value / 2.0) * cost_per_trade
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

    strategy_returns = daily_returns[warmup:]
    if len(strategy_returns) < 20:
        return None

    return compute_sharpe(strategy_returns, annualize=True)


# ------------------------------------------------------------------------------
# Perturbation tests
# ------------------------------------------------------------------------------

PERTURBATION_CONFIGS: list[tuple[str, dict]] = [
    # ratio lookback
    ("ratio_lookback=15", {"ratio_lookback": 15}),
    ("ratio_lookback=25", {"ratio_lookback": 25}),
    # SMA period
    ("sma_period=30", {"sma_period": 30}),
    ("sma_period=50", {"sma_period": 50}),
    # XLI lookback
    ("xli_lookback=15", {"xli_lookback": 15}),
    ("xli_lookback=25", {"xli_lookback": 25}),
    # SPY expansion weight
    ("spy_expansion=0.70", {"spy_expansion": 0.70}),
    ("spy_expansion=0.90", {"spy_expansion": 0.90}),
    # GLD contraction weight
    ("gld_contraction=0.30", {"gld_contraction": 0.30}),
    ("gld_contraction=0.50", {"gld_contraction": 0.50}),
    # TLT contraction weight
    ("tlt_contraction=0.30", {"tlt_contraction": 0.30}),
    ("tlt_contraction=0.50", {"tlt_contraction": 0.50}),
    # SPY transition weight
    ("spy_transition=0.40", {"spy_transition": 0.40}),
    ("spy_transition=0.60", {"spy_transition": 0.60}),
    # GLD transition weight
    ("gld_transition=0.20", {"gld_transition": 0.20}),
    ("gld_transition=0.40", {"gld_transition": 0.40}),
]


def run_perturbation_tests(
    prices_df: pl.DataFrame,
    base_sharpe: float,
    stability_threshold: float = 0.25,
) -> dict:
    """Run perturbation tests: vary each parameter, check Sharpe stability.

    A perturbation is UNSTABLE if the Sharpe changes by more than
    stability_threshold (25%) from the base.

    Returns
    -------
    dict
        n_stable, n_total, stability_pct, results list, passed
    """
    logger.info("Running %d perturbation tests...", len(PERTURBATION_CONFIGS))

    results: list[dict] = []
    n_stable = 0

    for name, overrides in PERTURBATION_CONFIGS:
        params = BacktestParams(**overrides)
        bt_result = _run_backtest_on_data(prices_df, params, return_daily_returns=False)

        if "error" in bt_result:
            logger.warning("Perturbation '%s' failed: %s", name, bt_result["error"])
            results.append(
                {"name": name, "sharpe": 0.0, "pct_change": -100.0, "stable": False}
            )
            continue

        perturbed_sharpe = bt_result["sharpe_ratio"]
        pct_change = (perturbed_sharpe - base_sharpe) / (abs(base_sharpe) + 1e-8) * 100
        stable = abs(pct_change) <= stability_threshold * 100
        if stable:
            n_stable += 1

        results.append(
            {
                "name": name,
                "sharpe": round(perturbed_sharpe, 4),
                "max_dd": bt_result.get("max_drawdown", 0.0),
                "pct_change": round(pct_change, 1),
                "stable": stable,
            }
        )
        status = "STABLE" if stable else "UNSTABLE"
        logger.info(
            "  %s: Sharpe=%.4f (%+.1f%%) %s",
            name,
            perturbed_sharpe,
            pct_change,
            status,
        )

    n_total = len(results)
    stability_pct = n_stable / n_total * 100 if n_total > 0 else 0.0

    return {
        "n_stable": n_stable,
        "n_total": n_total,
        "stability_pct": round(stability_pct, 1),
        "results": results,
        "passed": stability_pct >= 60.0,
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main() -> None:
    """Run full robustness analysis for F35."""
    import yaml

    # Step 1: Fetch data once (reused across all tests)
    logger.info("Fetching data for all symbols...")
    prices_df = fetch_ohlcv(ALL_SYMBOLS, lookback_days=LOOKBACK_DAYS)
    if len(prices_df) == 0:
        logger.error("No data fetched -- aborting")
        sys.exit(1)

    # Step 2: Run base backtest
    logger.info("=" * 60)
    logger.info("Step 1: Base backtest")
    logger.info("=" * 60)
    base_params = BacktestParams()
    base_result = _run_backtest_on_data(
        prices_df, base_params, return_daily_returns=True
    )
    if "error" in base_result:
        logger.error("Base backtest failed: %s", base_result["error"])
        sys.exit(1)

    base_sharpe = base_result["sharpe_ratio"]
    base_max_dd = base_result["max_drawdown"]
    base_dsr = base_result["dsr"]
    base_daily_returns = base_result["daily_returns"]
    base_cagr = base_result["cagr"]

    print(
        f"\n  Base: Sharpe={base_sharpe:.4f}, MaxDD={base_max_dd:.4f}, "
        f"CAGR={base_cagr:.4f}, DSR={base_dsr:.4f}, "
        f"Trades={base_result['total_trades']}"
    )

    # Step 3: Shuffled signal test (1000 shuffles)
    logger.info("=" * 60)
    logger.info("Step 2: Shuffled signal test (1000 shuffles)")
    logger.info("=" * 60)
    shuffle_result = run_shuffled_signal_test(prices_df, base_sharpe, n_shuffles=1000)
    print(
        f"\n  Shuffled: p={shuffle_result['p_value']:.4f}, "
        f"shuffled_mean={shuffle_result['shuffled_mean']:.4f}, "
        f"shuffled_std={shuffle_result['shuffled_std']:.4f}"
    )
    print(f"  Shuffled signal test: {'PASS' if shuffle_result['passed'] else 'FAIL'}")

    # Step 4: CPCV (15 folds)
    logger.info("=" * 60)
    logger.info("Step 3: CPCV (15 folds, purge=5)")
    logger.info("=" * 60)

    # Use 15 groups with k_test=2 for CPCV (15 folds, ~70 trading days each)
    n_groups = 15
    k_test = 2
    cpcv_result = run_cpcv(
        returns=base_daily_returns,
        strategy_fn=None,
        n_groups=n_groups,
        k_test=k_test,
        purge_days=5,
    )
    cpcv_mean = cpcv_result.mean_oos_sharpe
    cpcv_passed = cpcv_result.passed

    # Compute in-sample Sharpe for OOS/IS ratio
    is_sharpe = compute_sharpe(base_daily_returns, annualize=False)
    oos_is_ratio = cpcv_mean / is_sharpe if abs(is_sharpe) > 1e-8 else 0.0

    # Compute % positive folds
    n_positive = (
        sum(1 for s in cpcv_result.oos_sharpes if s > 0)
        if cpcv_result.oos_sharpes
        else 0
    )
    n_folds = len(cpcv_result.oos_sharpes) if cpcv_result.oos_sharpes else 1
    pct_positive = n_positive / n_folds * 100

    print(f"\n  CPCV mean OOS Sharpe: {cpcv_mean:.4f}")
    print(f"  CPCV std:             {cpcv_result.std_oos_sharpe:.4f}")
    print(f"  CPCV n_combos:        {cpcv_result.n_combinations}")
    print(f"  IS Sharpe (daily):    {is_sharpe:.4f}")
    print(f"  OOS/IS ratio:         {oos_is_ratio:.4f}")
    print(f"  Positive folds:       {n_positive}/{n_folds} ({pct_positive:.0f}%)")
    print(f"  CPCV passed:          {'PASS' if cpcv_passed else 'FAIL'}")

    if cpcv_result.oos_sharpes:
        print(f"  Fold Sharpes: {[f'{s:.3f}' for s in cpcv_result.oos_sharpes[:20]]}")

    # Step 5: Perturbation tests
    logger.info("=" * 60)
    logger.info("Step 4: Perturbation tests")
    logger.info("=" * 60)
    perturbation_result = run_perturbation_tests(prices_df, base_sharpe)
    print(
        f"\n  Perturbation: "
        f"{perturbation_result['n_stable']}/{perturbation_result['n_total']} "
        f"stable ({perturbation_result['stability_pct']:.0f}%)"
    )
    print(
        f"  Perturbation passed: {'PASS' if perturbation_result['passed'] else 'FAIL'}"
    )

    # -- Final gate summary --
    print("\n" + "=" * 60)
    print("  F35 ROBUSTNESS GATE SUMMARY")
    print("=" * 60)

    gates = {
        "Sharpe >= 0.80": (base_sharpe >= 0.80, f"{base_sharpe:.4f}"),
        "MaxDD < 15%": (
            base_max_dd < 0.15,
            f"{base_max_dd:.4f} ({base_max_dd * 100:.1f}%)",
        ),
        "DSR >= 0.95": (base_dsr >= 0.95, f"{base_dsr:.4f}"),
        "CPCV OOS > 0": (cpcv_passed, f"{cpcv_mean:.4f}"),
        "Perturbation >= 60%": (
            perturbation_result["passed"],
            f"{perturbation_result['stability_pct']:.0f}%",
        ),
        "Shuffled p < 0.05": (
            shuffle_result["passed"],
            f"p={shuffle_result['p_value']:.4f}",
        ),
    }

    all_passed = True
    for gate_name, (passed, value) in gates.items():
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(f"  {status}  {gate_name}: {value}")

    print(f"\n  OVERALL: {'ALL GATES PASSED' if all_passed else 'FAILED'}")
    print("=" * 60)

    # -- Save robustness results --
    robustness_yaml: dict = {
        "slug": SLUG,
        "timestamp": datetime.now().isoformat(),
        "family": "F35_cyclical_defensive_breadth",
        "base_metrics": {
            "sharpe_ratio": base_sharpe,
            "max_drawdown": base_max_dd,
            "cagr": base_cagr,
            "dsr": base_dsr,
            "total_trades": base_result["total_trades"],
            "sortino_ratio": base_result["sortino_ratio"],
            "calmar_ratio": base_result["calmar_ratio"],
            "regime_counts": base_result.get("regime_counts", {}),
        },
        "shuffled_signal": {
            "p_value": shuffle_result["p_value"],
            "shuffled_mean_sharpe": shuffle_result["shuffled_mean"],
            "shuffled_std_sharpe": shuffle_result["shuffled_std"],
            "n_shuffles": shuffle_result["n_shuffles"],
            "passed": shuffle_result["passed"],
        },
        "cpcv": {
            "mean_oos_sharpe": round(cpcv_mean, 4),
            "std_oos_sharpe": round(cpcv_result.std_oos_sharpe, 4),
            "n_combinations": cpcv_result.n_combinations,
            "oos_is_ratio": round(oos_is_ratio, 4),
            "pct_positive_folds": round(pct_positive, 1),
            "passed": cpcv_passed,
        },
        "perturbation": {
            "n_stable": perturbation_result["n_stable"],
            "n_total": perturbation_result["n_total"],
            "stability_pct": perturbation_result["stability_pct"],
            "passed": perturbation_result["passed"],
            "results": perturbation_result["results"],
        },
        "gates": {name: passed for name, (passed, _) in gates.items()},
        "overall_passed": all_passed,
    }

    STRAT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = STRAT_DIR / "robustness-result.yaml"
    with results_path.open("w") as f:
        yaml.dump(robustness_yaml, f, default_flow_style=False, sort_keys=False)
    logger.info("Robustness results saved to %s", results_path)


if __name__ == "__main__":
    main()

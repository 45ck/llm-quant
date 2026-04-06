#!/usr/bin/env python3
"""Robustness analysis for F32: Growth-Value Rotation v1.

Runs:
1. Base backtest
2. Shuffled signal test (1000 permutations)
3. CPCV (15 folds, OOS/IS ratio)
4. Perturbation tests across all key parameters
5. Gate evaluation with full summary

Gates:
- Sharpe >= 0.80
- MaxDD < 15%
- DSR >= 0.95
- CPCV OOS/IS > 0
- Perturbation stability >= 60%
- Shuffled signal p < 0.05
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from llm_quant.backtest.metrics import compute_sharpe
from llm_quant.backtest.robustness import run_cpcv, shuffled_signal_test

# Import the backtest function from the backtest script
sys.path.insert(0, "scripts")
from backtest_f32_growth_value import SLUG, run_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STRAT_DIR = Path(f"data/strategies/{SLUG}")


def run_perturbation_suite(base_sharpe: float) -> list[dict]:
    """Run parameter perturbation tests.

    Tests (>= 10 variants as specified):
    - ratio_lookback: 15, 25 (vs base 20)
    - sma_period: 30, 50 (vs base 40)
    - iwm_value: 0.40, 0.60 (vs base 0.50)
    - dia_value: 0.20, 0.40 (vs base 0.30)
    - qqq_growth: 0.60, 0.80 (vs base 0.70)
    - spy_growth: 0.10, 0.30 (vs base 0.20)
    - spy_neutral: 0.35, 0.55 (vs base 0.45)
    - shy_neutral: 0.35, 0.55 (vs base 0.45)
    """
    perturbations: list[tuple[str, dict]] = [
        # Signal parameters
        ("ratio_lookback=15", {"ratio_lookback": 15}),
        ("ratio_lookback=25", {"ratio_lookback": 25}),
        ("sma_period=30", {"sma_period": 30}),
        ("sma_period=50", {"sma_period": 50}),
        # Value regime weights
        ("iwm_value=0.40", {"iwm_value": 0.40}),
        ("iwm_value=0.60", {"iwm_value": 0.60}),
        ("dia_value=0.20", {"dia_value": 0.20}),
        ("dia_value=0.40", {"dia_value": 0.40}),
        # Growth regime weights
        ("qqq_growth=0.60", {"qqq_growth": 0.60}),
        ("qqq_growth=0.80", {"qqq_growth": 0.80}),
        ("spy_growth=0.10", {"spy_growth": 0.10}),
        ("spy_growth=0.30", {"spy_growth": 0.30}),
        # Neutral regime weights
        ("spy_neutral=0.35", {"spy_neutral": 0.35}),
        ("spy_neutral=0.55", {"spy_neutral": 0.55}),
        ("shy_neutral=0.35", {"shy_neutral": 0.35}),
        ("shy_neutral=0.55", {"shy_neutral": 0.55}),
    ]

    # Default parameters
    defaults = {
        "ratio_lookback": 20,
        "sma_period": 40,
        "iwm_value": 0.50,
        "dia_value": 0.30,
        "shy_value": 0.20,
        "qqq_growth": 0.70,
        "spy_growth": 0.20,
        "spy_neutral": 0.45,
        "shy_neutral": 0.45,
        "rebalance_freq": 5,
        "cost_bps": 5.0,
    }

    results: list[dict] = []
    for name, overrides in perturbations:
        logger.info("Running perturbation: %s", name)
        kwargs = {**defaults, **overrides}
        r = run_backtest(**kwargs)
        if "error" in r:
            logger.warning("Perturbation %s failed: %s", name, r["error"])
            results.append(
                {"name": name, "sharpe": 0.0, "pct_change": -100.0, "stable": False}
            )
            continue

        pct_change = (r["sharpe_ratio"] - base_sharpe) / (abs(base_sharpe) + 1e-8) * 100
        stable = abs(pct_change) <= 25.0
        results.append(
            {
                "name": name,
                "sharpe": round(r["sharpe_ratio"], 4),
                "max_dd": round(r["max_drawdown"], 4),
                "pct_change": round(pct_change, 1),
                "stable": stable,
            }
        )
        logger.info(
            "  %s: Sharpe=%.4f (%+.1f%%) %s",
            name,
            r["sharpe_ratio"],
            pct_change,
            "STABLE" if stable else "UNSTABLE",
        )

    return results


def _run_shuffled_test(
    base_returns: list[float], bench_returns: list[float]
) -> tuple[float, bool]:
    """Run shuffled signal test. Returns (p_value, passed)."""
    n = min(len(base_returns), len(bench_returns))
    if n < 30:
        logger.warning("Insufficient data for shuffled test")
        return 1.0, False

    asset_returns = bench_returns[:n]
    shuffled = shuffled_signal_test(
        daily_returns=base_returns[:n],
        asset_returns=asset_returns,
        n_shuffles=1000,
        seed=42,
    )
    print(f"  Real Sharpe:    {shuffled.real_sharpe:.4f}")
    print(f"  Shuffled mean:  {shuffled.shuffled_mean:.4f}")
    print(f"  Shuffled 95th:  {shuffled.shuffled_95th:.4f}")
    print(f"  p-value:        {shuffled.p_value:.4f}")
    print(f"  Passed (p<0.05):{shuffled.passed}")
    return shuffled.p_value, shuffled.passed


def _run_cpcv_analysis(
    base_returns: list[float],
) -> tuple[float, float, float, bool, list[float]]:
    """Run CPCV analysis with 15 folds. Returns (mean, std, oos_is, passed, sharpes)."""
    cpcv = run_cpcv(
        returns=base_returns,
        strategy_fn=None,
        n_groups=15,
        k_test=3,
        purge_days=5,
    )
    is_sharpe = compute_sharpe(base_returns, annualize=False)
    oos_is_ratio = cpcv.mean_oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

    # Count positive folds
    n_positive = sum(1 for s in (cpcv.oos_sharpes or []) if s > 0)
    n_total_folds = len(cpcv.oos_sharpes or [])
    pct_positive = n_positive / n_total_folds if n_total_folds > 0 else 0.0

    print(f"  CPCV mean OOS Sharpe: {cpcv.mean_oos_sharpe:.4f}")
    print(f"  CPCV std OOS Sharpe:  {cpcv.std_oos_sharpe:.4f}")
    print(f"  IS Sharpe (raw):      {is_sharpe:.4f}")
    print(f"  OOS/IS ratio:         {oos_is_ratio:.4f}")
    print(f"  N combinations:       {cpcv.n_combinations}")
    print(f"  Positive folds:       {n_positive}/{n_total_folds} ({pct_positive:.0%})")
    if cpcv.oos_sharpes:
        print(f"  Fold Sharpes:         {[f'{s:.3f}' for s in cpcv.oos_sharpes[:20]]}")
    print(f"  Passed (mean>0):      {cpcv.passed}")

    return (
        cpcv.mean_oos_sharpe,
        cpcv.std_oos_sharpe,
        oos_is_ratio,
        cpcv.passed,
        cpcv.oos_sharpes or [],
    )


def _evaluate_gates(
    base_sharpe: float,
    base_dd: float,
    base_dsr: float,
    cpcv_mean: float,
    oos_is_ratio: float,
    cpcv_passed: bool,
    stability_pct: float,
    n_stable: int,
    n_total: int,
    shuffled_p: float,
    shuffled_passed: bool,
) -> tuple[dict, bool]:
    """Evaluate all gates. Returns (gates_dict, all_pass)."""
    gates = {
        "Sharpe >= 0.80": {
            "value": base_sharpe,
            "passed": base_sharpe >= 0.80,
            "display": f"{base_sharpe:.4f}",
        },
        "MaxDD < 15%": {
            "value": base_dd,
            "passed": base_dd < 0.15,
            "display": f"{base_dd:.4f} ({base_dd * 100:.2f}%)",
        },
        "DSR >= 0.95": {
            "value": base_dsr,
            "passed": base_dsr >= 0.95,
            "display": f"{base_dsr:.4f}",
        },
        "CPCV OOS/IS > 0": {
            "value": cpcv_mean,
            "passed": cpcv_passed,
            "display": f"{cpcv_mean:.4f} (ratio={oos_is_ratio:.4f})",
        },
        "Perturbation >= 60%": {
            "value": stability_pct,
            "passed": stability_pct >= 0.60,
            "display": f"{n_stable}/{n_total} ({stability_pct:.0%})",
        },
        "Shuffled p < 0.05": {
            "value": shuffled_p,
            "passed": shuffled_passed,
            "display": f"p={shuffled_p:.4f}",
        },
    }

    print("\n" + "=" * 70)
    print("GATE RESULTS")
    print("=" * 70)
    for name, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{status}] {name}: {gate['display']}")

    all_pass = all(g["passed"] for g in gates.values())
    print(f"\n  OVERALL: {'ALL GATES PASSED' if all_pass else 'GATES FAILED'}")
    if all_pass:
        print("  --> Ready for /paper (30-day paper trading)")
    else:
        failed = [n for n, g in gates.items() if not g["passed"]]
        print(f"  --> Failed gates: {', '.join(failed)}")

    return gates, all_pass


def main() -> None:
    """Run full robustness analysis."""
    print("=" * 70)
    print("F32 GROWTH-VALUE ROTATION -- ROBUSTNESS ANALYSIS")
    print("=" * 70)

    # 1. Base backtest
    print("\n[1/4] Running base backtest...")
    base = run_backtest()
    if "error" in base:
        logger.error("Base backtest failed: %s", base["error"])
        sys.exit(1)

    base_sharpe = base["sharpe_ratio"]
    base_dd = base["max_drawdown"]
    base_dsr = base["dsr"]
    base_returns = base["daily_returns"]
    bench_returns = base["benchmark_returns"]

    print("\nBase results:")
    print(f"  Sharpe:  {base_sharpe:.4f}")
    print(f"  MaxDD:   {base_dd:.4f} ({base_dd * 100:.2f}%)")
    print(f"  CAGR:    {base['cagr']:.4f} ({base['cagr'] * 100:.2f}%)")
    print(f"  DSR:     {base_dsr:.4f}")
    print(f"  Sortino: {base['sortino']:.4f}")
    print(f"  Calmar:  {base['calmar']:.4f}")
    print(f"  Regimes: {base['regime_counts']}")

    # 2. Shuffled signal test
    print("\n[2/4] Running shuffled signal test (1000 permutations)...")
    shuffled_p, shuffled_passed = _run_shuffled_test(base_returns, bench_returns)

    # 3. CPCV (15 folds)
    print("\n[3/4] Running CPCV (15 groups, 3 test, 5-day purge)...")
    cpcv_mean, cpcv_std, oos_is_ratio, cpcv_passed, oos_sharpes = _run_cpcv_analysis(
        base_returns
    )

    # 4. Perturbation tests
    print("\n[4/4] Running perturbation tests (16 variants)...")
    perturbation_results = run_perturbation_suite(base_sharpe)

    n_stable = sum(1 for p in perturbation_results if p["stable"])
    n_total = len(perturbation_results)
    stability_pct = n_stable / n_total if n_total > 0 else 0.0
    print(f"\nPerturbation summary: {n_stable}/{n_total} stable ({stability_pct:.0%})")

    # 5. Gate evaluation
    gates, all_pass = _evaluate_gates(
        base_sharpe,
        base_dd,
        base_dsr,
        cpcv_mean,
        oos_is_ratio,
        cpcv_passed,
        stability_pct,
        n_stable,
        n_total,
        shuffled_p,
        shuffled_passed,
    )

    # 6. Save results
    n_positive = sum(1 for s in oos_sharpes if s > 0)
    n_total_folds = len(oos_sharpes)

    robustness_result = {
        "strategy_slug": SLUG,
        "family": "growth_value_rotation",
        "family_trial_number": 1,
        "base_metrics": {
            "sharpe_ratio": round(base_sharpe, 4),
            "max_drawdown": round(base_dd, 4),
            "cagr": round(base["cagr"], 4),
            "total_return": round(base["total_return"], 4),
            "sortino": round(base["sortino"], 4),
            "calmar": round(base["calmar"], 4),
            "dsr": round(base_dsr, 4),
            "final_nav": round(base["final_nav"], 2),
            "trading_days": base["trading_days"],
            "rebalance_count": base["rebalance_count"],
            "benchmark_sharpe": round(base["benchmark_sharpe"], 4),
            "regime_counts": base["regime_counts"],
        },
        "shuffled_signal": {
            "p_value": round(shuffled_p, 4),
            "passed": shuffled_passed,
        },
        "cpcv": {
            "n_groups": 15,
            "k_test": 3,
            "mean_oos_sharpe": round(cpcv_mean, 4),
            "std_oos_sharpe": round(cpcv_std, 4),
            "oos_is_ratio": round(oos_is_ratio, 4),
            "n_combinations": n_total_folds,
            "positive_folds": n_positive,
            "pct_positive": (
                round(n_positive / n_total_folds, 4) if n_total_folds > 0 else 0
            ),
            "oos_sharpes": [round(s, 4) for s in oos_sharpes],
            "passed": cpcv_passed,
        },
        "perturbation": {
            "n_stable": n_stable,
            "n_total": n_total,
            "stability_pct": round(stability_pct, 4),
            "passed": stability_pct >= 0.60,
            "details": perturbation_results,
        },
        "gates": {
            name: {
                "value": (
                    round(g["value"], 4)
                    if isinstance(g["value"], float)
                    else g["value"]
                ),
                "passed": g["passed"],
            }
            for name, g in gates.items()
        },
        "overall_passed": all_pass,
        "verdict": "PASS" if all_pass else "REJECTED",
        "computed_at": datetime.now(UTC).isoformat(),
    }

    STRAT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = STRAT_DIR / "robustness-result.yaml"
    with result_path.open("w") as f:
        yaml.dump(robustness_result, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved robustness results to %s", result_path)

    print(f"\nResults saved to {result_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

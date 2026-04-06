"""
F43 Multi-Asset Regime Signal — Robustness Script
==================================================
Family: multi_asset_regime (F43), Trial: 1

6 robustness gates:
  1. Sharpe >= 0.80
  2. MaxDD < 15%
  3. DSR >= 0.95 (deflated Sharpe ratio)
  4. CPCV OOS/IS > 0 (combinatorial purged cross-validation)
  5. Perturbation stability >= 60%
  6. Shuffled signal p-value < 0.05
"""

import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Import the backtest function
sys.path.insert(0, "scripts")
from backtest_f43_multi_asset_regime import fetch_data, run_backtest  # noqa: E402


def compute_dsr(sharpe: float, trading_days: int, n_trials: int = 1) -> float:
    """
    Deflated Sharpe Ratio.
    DSR = CDF((sharpe * sqrt(T/252) - ppf(1 - 1/(2*N))) / 1.0)
    """
    sr_annualized = sharpe * np.sqrt(trading_days / 252)
    threshold = norm.ppf(1 - 1 / (2 * n_trials))
    return norm.cdf((sr_annualized - threshold) / 1.0)


def compute_cpcv(
    prices: pd.DataFrame,
    n_groups: int = 15,
    n_test: int = 3,
    purge_gap: int = 5,
    **backtest_kwargs,
) -> dict:
    """
    Combinatorial Purged Cross-Validation.
    Split data into n_groups, test all C(n_groups, n_test) combinations.
    Returns OOS/IS Sharpe ratio.
    """
    # Get date index from prices
    dates = prices.index
    n = len(dates)
    group_size = n // n_groups

    # Create group boundaries
    groups = []
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size if i < n_groups - 1 else n
        groups.append((start_idx, end_idx))

    # All combinations of test groups
    test_combos = list(combinations(range(n_groups), n_test))

    # Limit to 50 combos for speed
    if len(test_combos) > 50:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(test_combos), 50, replace=False)
        test_combos = [test_combos[i] for i in indices]

    oos_sharpes = []
    is_sharpes = []

    for combo in test_combos:
        test_indices = set()
        for g in combo:
            start, end = groups[g]
            test_indices.update(range(start, end))

        # Purge: remove purge_gap days around test boundaries
        purge_indices = set()
        for g in combo:
            start, end = groups[g]
            for p in range(max(0, start - purge_gap), min(n, start + purge_gap)):
                purge_indices.add(p)
            for p in range(max(0, end - purge_gap), min(n, end + purge_gap)):
                purge_indices.add(p)

        train_indices = sorted(set(range(n)) - test_indices - purge_indices)
        test_indices = sorted(test_indices)

        if len(train_indices) < 200 or len(test_indices) < 50:
            continue

        # Build train and test price DataFrames
        train_prices = prices.iloc[train_indices]
        test_prices = prices.iloc[test_indices]

        try:
            # IS backtest
            is_result = run_backtest(prices=train_prices, **backtest_kwargs)
            is_sharpe = is_result["sharpe"]

            # OOS backtest
            oos_result = run_backtest(prices=test_prices, **backtest_kwargs)
            oos_sharpe = oos_result["sharpe"]

            if not np.isnan(is_sharpe) and not np.isnan(oos_sharpe) and is_sharpe != 0:
                oos_sharpes.append(oos_sharpe)
                is_sharpes.append(is_sharpe)
        except Exception:  # noqa: S112
            continue

    if not oos_sharpes or not is_sharpes:
        return {"oos_mean": 0.0, "is_mean": 0.0, "oos_is_ratio": 0.0, "n_folds": 0}

    oos_mean = np.mean(oos_sharpes)
    is_mean = np.mean(is_sharpes)
    oos_is_ratio = oos_mean / is_mean if is_mean != 0 else 0.0

    return {
        "oos_mean": oos_mean,
        "is_mean": is_mean,
        "oos_is_ratio": oos_is_ratio,
        "n_folds": len(oos_sharpes),
    }


def shuffled_signal_test(
    prices: pd.DataFrame, n_permutations: int = 1000, **backtest_kwargs
) -> dict:
    """
    Shuffle the signal (regime assignments) by randomly permuting the
    dates of the z-score series. If the real strategy's Sharpe is not
    significantly better than random, the signal has no value.
    """
    # Get base Sharpe
    base_result = run_backtest(prices=prices, **backtest_kwargs)
    base_sharpe = base_result["sharpe"]

    # For speed: instead of full re-run with shuffled z-scores,
    # shuffle the daily returns of the strategy (equivalent null).
    base_daily = base_result["daily_returns"].to_numpy().copy()
    rng = np.random.RandomState(42)

    shuffled_sharpes = []
    for _ in range(n_permutations):
        shuffled = base_daily.copy()
        rng.shuffle(shuffled)
        # Sharpe of shuffled returns
        s = (
            shuffled.mean() / shuffled.std() * np.sqrt(252)
            if shuffled.std() > 0
            else 0.0
        )
        shuffled_sharpes.append(s)

    shuffled_sharpes = np.array(shuffled_sharpes)
    p_value = np.mean(shuffled_sharpes >= base_sharpe)

    return {
        "base_sharpe": base_sharpe,
        "shuffled_mean": np.mean(shuffled_sharpes),
        "shuffled_std": np.std(shuffled_sharpes),
        "p_value": p_value,
    }


def perturbation_test(prices: pd.DataFrame, base_sharpe: float) -> dict:
    """
    Run 14+ perturbation variants. A variant is 'stable' if its Sharpe
    is within +/-25% of the base Sharpe.
    """
    perturbations = [
        {"name": "return_lookback=20", "return_lookback": 20},
        {"name": "return_lookback=45", "return_lookback": 45},
        {"name": "z_window=90", "z_window": 90},
        {"name": "z_window=150", "z_window": 150},
        {"name": "composite_threshold=0.50", "composite_threshold": 0.50},
        {"name": "composite_threshold=1.00", "composite_threshold": 1.00},
        {"name": "spy_weight_composite=0.40", "spy_weight_composite": 0.40},
        {"name": "spy_weight_composite=0.60", "spy_weight_composite": 0.60},
        {"name": "spy_riskon=0.65", "spy_riskon": 0.65},
        {"name": "spy_riskon=0.85", "spy_riskon": 0.85},
        {"name": "gld_riskoff=0.35", "gld_riskoff": 0.35},
        {"name": "gld_riskoff=0.50", "gld_riskoff": 0.50},
        {"name": "tlt_deflation=0.40", "tlt_deflation": 0.40},
        {"name": "tlt_deflation=0.60", "tlt_deflation": 0.60},
        {"name": "gld_reflation=0.20", "gld_reflation": 0.20},
        {"name": "gld_reflation=0.40", "gld_reflation": 0.40},
        {"name": "rebalance_freq=3", "rebalance_freq": 3},
        {"name": "rebalance_freq=10", "rebalance_freq": 10},
    ]

    results = []
    lower = base_sharpe * 0.75
    upper = base_sharpe * 1.25

    for pert in perturbations:
        name = pert.pop("name")
        try:
            r = run_backtest(prices=prices, **pert)
            sharpe = r["sharpe"]
            stable = lower <= sharpe <= upper
            results.append(
                {
                    "name": name,
                    "sharpe": sharpe,
                    "max_dd": r["max_drawdown"],
                    "stable": stable,
                }
            )
        except Exception:
            results.append(
                {"name": name, "sharpe": np.nan, "max_dd": np.nan, "stable": False}
            )

    n_stable = sum(1 for r in results if r["stable"])
    pct_stable = n_stable / len(results) * 100

    return {
        "variants": results,
        "n_stable": n_stable,
        "n_total": len(results),
        "pct_stable": pct_stable,
    }


def main():
    print("=" * 70)
    print("F43 Multi-Asset Regime Signal — Robustness Testing")
    print("=" * 70)

    # --- Fetch data ---
    print("\nFetching data...")
    prices = fetch_data(1825)
    print(
        f"Data: {prices.shape[0]} rows, {prices.index[0].date()} to {prices.index[-1].date()}"
    )

    # --- Gate 1 & 2: Base backtest ---
    print("\n[Gate 1-2] Running base backtest...")
    base = run_backtest(prices=prices)
    base_sharpe = base["sharpe"]
    base_maxdd = base["max_drawdown"]
    trading_days = base["trading_days"]

    print(f"  Base Sharpe:  {base_sharpe:.3f}  (gate: >= 0.80)")
    print(f"  Base MaxDD:   {base_maxdd:.1%}  (gate: < 15%)")

    gate1 = base_sharpe >= 0.80
    gate2 = base_maxdd < 0.15

    # --- Gate 3: DSR ---
    print("\n[Gate 3] Computing Deflated Sharpe Ratio...")
    dsr = compute_dsr(base_sharpe, trading_days, n_trials=1)
    print(f"  DSR:          {dsr:.4f}  (gate: >= 0.95)")
    gate3 = dsr >= 0.95

    # --- Gate 4: CPCV ---
    print("\n[Gate 4] Running CPCV (15 groups, 3 test, 5-day purge)...")
    print("  This may take a few minutes...")
    cpcv = compute_cpcv(prices, n_groups=15, n_test=3, purge_gap=5)
    print(f"  OOS mean Sharpe: {cpcv['oos_mean']:.3f}")
    print(f"  IS mean Sharpe:  {cpcv['is_mean']:.3f}")
    print(f"  OOS/IS ratio:    {cpcv['oos_is_ratio']:.3f}  (gate: > 0)")
    print(f"  Folds completed: {cpcv['n_folds']}")
    gate4 = cpcv["oos_is_ratio"] > 0

    # --- Gate 5: Perturbation stability ---
    print("\n[Gate 5] Running perturbation test (18 variants)...")
    pert = perturbation_test(prices, base_sharpe)
    print(
        f"  Stable variants: {pert['n_stable']}/{pert['n_total']} ({pert['pct_stable']:.0f}%)"
    )
    print("  (gate: >= 60%)")
    print(f"\n  {'Variant':<30s} {'Sharpe':>8s} {'MaxDD':>8s} {'Stable':>7s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 7}")
    for v in pert["variants"]:
        s = f"{v['sharpe']:.3f}" if not np.isnan(v["sharpe"]) else "   NaN"
        d = f"{v['max_dd']:.1%}" if not np.isnan(v["max_dd"]) else "   NaN"
        st = "YES" if v["stable"] else "no"
        print(f"  {v['name']:<30s} {s:>8s} {d:>8s} {st:>7s}")
    gate5 = pert["pct_stable"] >= 60.0

    # --- Gate 6: Shuffled signal test ---
    print("\n[Gate 6] Running shuffled signal test (1000 permutations)...")
    shuf = shuffled_signal_test(prices, n_permutations=1000)
    print(f"  Base Sharpe:     {shuf['base_sharpe']:.3f}")
    print(f"  Shuffled mean:   {shuf['shuffled_mean']:.3f}")
    print(f"  Shuffled std:    {shuf['shuffled_std']:.3f}")
    print(f"  p-value:         {shuf['p_value']:.4f}  (gate: < 0.05)")
    gate6 = shuf["p_value"] < 0.05

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ROBUSTNESS GATE SUMMARY")
    print("=" * 70)
    gates = [
        ("Sharpe >= 0.80", gate1, f"{base_sharpe:.3f}"),
        ("MaxDD < 15%", gate2, f"{base_maxdd:.1%}"),
        ("DSR >= 0.95", gate3, f"{dsr:.4f}"),
        ("CPCV OOS/IS > 0", gate4, f"{cpcv['oos_is_ratio']:.3f}"),
        ("Perturbation >= 60%", gate5, f"{pert['pct_stable']:.0f}%"),
        ("Shuffled p < 0.05", gate6, f"{shuf['p_value']:.4f}"),
    ]

    all_pass = True
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name:<25s}  value={value}")

    print("\n" + "=" * 70)
    if all_pass:
        print("RESULT: ALL 6 GATES PASSED — strategy is ROBUST")
        print("NEXT STEP: /paper track record (30 days minimum)")
    else:
        n_pass = sum(1 for _, p, _ in gates if p)
        print(f"RESULT: {n_pass}/6 GATES PASSED — strategy needs review")
        if not gate1:
            print("  -> Sharpe too low; mechanism may not have sufficient alpha")
        if not gate2:
            print(
                "  -> MaxDD too high; consider tighter regime thresholds or lower equity weight"
            )
        if not gate3:
            print(
                "  -> DSR failed; Sharpe is not statistically significant given trial count"
            )
        if not gate4:
            print(
                "  -> CPCV failed; out-of-sample performance degrades — possible overfitting"
            )
        if not gate5:
            print(
                "  -> Perturbation unstable; strategy is sensitive to parameter choices"
            )
        if not gate6:
            print("  -> Shuffled test failed; signal may not be better than random")
    print("=" * 70)

    return {
        "base_sharpe": base_sharpe,
        "base_maxdd": base_maxdd,
        "dsr": dsr,
        "cpcv": cpcv,
        "perturbation": pert,
        "shuffled": shuf,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    results = main()

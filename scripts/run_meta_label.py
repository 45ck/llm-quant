#!/usr/bin/env python3
"""Run meta-labeling analysis on LQD-SPY (strongest strategy).

Tests whether XGBoost meta-labeling adds value on top of the
hypothesis-driven lead-lag signal. Also runs feature importance
to understand WHAT predicts strategy success.

Pre-registered design (no hyperparameter search):
- XGBoost: max_depth=3, n_estimators=100, min_child_weight=5
- Triple barrier: +3% profit, -5% stop, 10-day max hold
- Train: 2022-01 to 2024-12, Test: 2025-01 to 2026-03
- 10-day embargo between train and test
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import logging

import polars as pl
import yaml

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.meta_label import (
    apply_triple_barrier,
    extract_features,
    train_meta_labeler,
)
from llm_quant.backtest.metrics import compute_sharpe
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Strategy definitions to test
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "slug": "lqd-spy-credit-lead",
        "cls": "lead_lag",
        "syms": ["LQD", "SPY"],
        "leader": "LQD",
        "follower": "SPY",
        "params": {
            "leader_symbol": "LQD",
            "follower_symbol": "SPY",
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
            "target_weight": 0.8,
            "rebalance_frequency_days": 5,
        },
    },
]

CUTOFF = date(2025, 1, 1)
EMBARGO_DAYS = 10


def run_analysis(strat_def: dict) -> dict:  # noqa: PLR0915
    """Run full meta-labeling analysis for one strategy."""
    slug = strat_def["slug"]
    leader = strat_def["leader"]
    follower = strat_def["follower"]

    print(f"\n{'=' * 70}")
    print(f"META-LABELING ANALYSIS: {slug}")
    print(f"{'=' * 70}")

    # 1. Fetch data
    print("\n1. Fetching data...", end="", flush=True)
    # Include VIX for regime features
    all_syms = list(set(strat_def["syms"] + ["VIX"]))
    prices_df = fetch_ohlcv(all_syms, lookback_days=5 * 365)
    indicators_df = compute_indicators(prices_df)
    print(f" done ({len(prices_df)} rows)")

    # 2. Run base strategy backtest to find signal dates
    print("2. Running base strategy backtest...", end="", flush=True)
    config = StrategyConfig(
        name=strat_def["cls"],
        rebalance_frequency_days=strat_def["params"].get("rebalance_frequency_days", 5),
        parameters=strat_def["params"],
    )
    strategy = create_strategy(strat_def["cls"], config)
    engine = BacktestEngine(strategy=strategy, initial_capital=100_000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug,
        cost_model=CostModel(),
        warmup_days=30,
        cost_multiplier=1.0,
    )
    base_sharpe = compute_sharpe(result.daily_returns, annualize=True)
    print(f" Sharpe={base_sharpe:.3f}, Trades={len(result.trades)}")

    # 3. Extract signal entry dates from trades
    print("3. Extracting entry signals...", end="", flush=True)
    entry_trades = [
        t for t in result.trades if t.action.lower() == "buy" and t.symbol == follower
    ]
    print(f" {len(entry_trades)} entries found")

    if len(entry_trades) < 20:
        print("   INSUFFICIENT ENTRIES — cannot train meta-labeler")
        return {"slug": slug, "status": "insufficient_data"}

    # 4. Build labeled dataset
    print("4. Building labeled dataset (triple-barrier)...")
    features_list: list[dict[str, float]] = []
    labels: list[int] = []
    signal_dates: list[date] = []
    trade_history: list[dict] = []

    for trade in entry_trades:
        trade_date = trade.date

        # Compute leader return at signal time (approximate)
        leader_data = (
            indicators_df.filter(
                (pl.col("symbol") == leader) & (pl.col("date") <= trade_date)
            )
            .sort("date")
            .tail(6)
        )
        if len(leader_data) < 2:
            continue
        lc = leader_data["close"].to_list()
        leader_ret = lc[-1] / lc[0] - 1.0 if lc[0] > 0 else 0.0

        # Extract features
        feats = extract_features(
            signal_date=trade_date,
            leader_symbol=leader,
            follower_symbol=follower,
            leader_return=leader_ret,
            indicators_df=indicators_df,
            prices_df=prices_df,
            trade_history=trade_history,
        )

        # Apply triple-barrier label
        label = apply_triple_barrier(
            entry_date=trade_date,
            symbol=follower,
            prices_df=prices_df,
            upper_pct=0.03,
            lower_pct=0.05,
            max_holding_days=10,
        )

        features_list.append(feats)
        labels.append(label)
        signal_dates.append(trade_date)

        # Update trade history for rolling features
        trade_history.append({"date": trade_date, "pnl": trade.pnl or 0.0})

    print(
        f"   {len(labels)} labeled entries "
        f"({sum(labels)} wins, {len(labels) - sum(labels)} losses, "
        f"win rate={sum(labels) / len(labels):.1%})"
    )

    # 5. Train meta-labeler
    print("5. Training XGBoost meta-labeler...")
    print(f"   Cutoff: {CUTOFF}, Embargo: {EMBARGO_DAYS} days")

    model, ml_result = train_meta_labeler(
        features_list=features_list,
        labels=labels,
        signal_dates=signal_dates,
        cutoff_date=CUTOFF,
        embargo_days=EMBARGO_DAYS,
    )

    print("\n   --- TRAINING RESULTS ---")
    print(
        f"   Train samples: {ml_result.n_train} ({ml_result.n_train_positive} positive)"
    )
    print(
        f"   Test samples:  {ml_result.n_test} ({ml_result.n_test_positive} positive)"
    )
    print(f"   Train accuracy: {ml_result.train_accuracy:.1%}")
    print(f"   Test accuracy:  {ml_result.test_accuracy:.1%}")
    print(f"   Test AUC:       {ml_result.test_auc:.3f}")
    print(f"   Test precision: {ml_result.test_precision:.1%}")
    print(f"   Test recall:    {ml_result.test_recall:.1%}")

    # Overfit check
    gap = ml_result.train_accuracy - ml_result.test_accuracy
    if gap > 0.15:
        print(f"   WARNING: train-test gap = {gap:.1%} — LIKELY OVERFIT")
    elif gap > 0.05:
        print(f"   CAUTION: train-test gap = {gap:.1%}")
    else:
        print(f"   OK: train-test gap = {gap:.1%}")

    # 6. Feature importance
    print("\n   --- FEATURE IMPORTANCE (XGBoost gain) ---")
    for fname, score in list(ml_result.feature_importance.items())[:10]:
        print(f"   {fname:<35} {score:.4f}")

    if ml_result.shap_importance:
        print("\n   --- FEATURE IMPORTANCE (SHAP mean|value|) ---")
        for fname, score in list(ml_result.shap_importance.items())[:10]:
            print(f"   {fname:<35} {score:.6f}")

    # 7. Skeptic's minimum validation
    print("\n   --- SKEPTIC'S VALIDATION ---")

    auc_pass = ml_result.test_auc > 0.55
    print(f"   AUC > 0.55: {'PASS' if auc_pass else 'FAIL'} ({ml_result.test_auc:.3f})")

    # Check if top feature is >50% of total
    fi_vals = list(ml_result.feature_importance.values())
    concentrated = fi_vals[0] > 0.5 * sum(fi_vals) if fi_vals else False
    print(f"   Feature concentration: {'WARNING' if concentrated else 'OK'}")

    overfit = gap > 0.15
    print(f"   Overfit check: {'FAIL' if overfit else 'OK'} (gap={gap:.1%})")

    overall = auc_pass and not overfit and not concentrated
    print(f"\n   OVERALL: {'PROCEED WITH CAUTION' if overall else 'KILL'}")

    # 8. Save results
    return {
        "slug": slug,
        "base_sharpe": round(base_sharpe, 4),
        "total_entries": len(entry_trades),
        "labeled_entries": len(labels),
        "label_win_rate": round(sum(labels) / len(labels), 4),
        "cutoff_date": str(CUTOFF),
        "ml_result": {
            "train_accuracy": round(ml_result.train_accuracy, 4),
            "test_accuracy": round(ml_result.test_accuracy, 4),
            "test_auc": round(ml_result.test_auc, 4),
            "test_precision": round(ml_result.test_precision, 4),
            "test_recall": round(ml_result.test_recall, 4),
            "n_train": ml_result.n_train,
            "n_test": ml_result.n_test,
            "train_test_gap": round(gap, 4),
            "feature_importance_gain": ml_result.feature_importance,
            "feature_importance_shap": ml_result.shap_importance,
        },
        "validation": {
            "auc_pass": auc_pass,
            "overfit_pass": not overfit,
            "concentration_pass": not concentrated,
            "overall": overall,
        },
    }


def main() -> None:

    all_results = []
    for strat in STRATEGIES:
        result = run_analysis(strat)
        all_results.append(result)

    # Save
    out_path = Path("data/strategies/meta-label-results.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

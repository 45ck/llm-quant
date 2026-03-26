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

import yaml

from llm_quant.backtest.meta_label import (
    MetaLabelConfig,
    MetaLabelFilter,
    replay_lead_lag_signals,
)
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Strategy definitions to test
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "slug": "lqd-spy-credit-lead",
        "syms": ["LQD", "SPY"],
        "leader": "LQD",
        "follower": "SPY",
        "params": {
            "lag_days": 1,
            "signal_window": 5,
            "entry_threshold": 0.005,
            "exit_threshold": -0.005,
        },
    },
]

CUTOFF = date(2025, 1, 1)
EMBARGO_DAYS = 10


def run_analysis(strat_def: dict) -> dict:
    """Run full meta-labeling analysis for one strategy."""
    slug = strat_def["slug"]
    leader = strat_def["leader"]
    follower = strat_def["follower"]
    params = strat_def["params"]

    print(f"\n{'=' * 70}")
    print(f"META-LABELING ANALYSIS: {slug}")
    print(f"{'=' * 70}")

    # 1. Fetch data
    print("\n1. Fetching data...", end="", flush=True)
    all_syms = list(set(strat_def["syms"] + ["VIX"]))
    prices_df = fetch_ohlcv(all_syms, lookback_days=5 * 365)
    indicators_df = compute_indicators(prices_df)
    print(f" done ({len(prices_df)} rows)")

    # 2. Replay strategy signals (lightweight, no BacktestEngine)
    print("2. Replaying lead-lag signals...", end="", flush=True)
    all_signals = replay_lead_lag_signals(
        indicators_df=indicators_df,
        leader_symbol=leader,
        follower_symbol=follower,
        lag_days=params["lag_days"],
        signal_window=params["signal_window"],
        entry_threshold=params["entry_threshold"],
        exit_threshold=params["exit_threshold"],
    )
    entry_signals = [s for s in all_signals if s["action"] == "BUY"]
    print(f" {len(entry_signals)} BUY signals, {len(all_signals)} total")

    if len(entry_signals) < 20:
        print("   INSUFFICIENT ENTRIES -- cannot train meta-labeler")
        return {"slug": slug, "status": "insufficient_data"}

    # 3. Build labeled dataset
    print("3. Building labeled dataset (triple-barrier with high/low)...")
    config = MetaLabelConfig(embargo_days=EMBARGO_DAYS)
    mlf = MetaLabelFilter(config)

    dataset = mlf.build_labeled_dataset(
        indicators_df=indicators_df,
        prices_df=prices_df,
        entry_signals=entry_signals,
        leader_symbol=leader,
        follower_symbol=follower,
    )

    labels = dataset["labels"]
    n_labels = len(labels)
    n_wins = sum(labels)
    print(
        f"   {n_labels} labeled entries "
        f"({n_wins} wins, {n_labels - n_wins} losses, "
        f"win rate={n_wins / n_labels:.1%})"
    )

    # 4. Train meta-labeler
    print(f"4. Training XGBoost meta-labeler (cutoff={CUTOFF})...")
    ml_result = mlf.train(dataset, cutoff_date=CUTOFF)

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
        print(f"   WARNING: train-test gap = {gap:.1%} -- LIKELY OVERFIT")
    elif gap > 0.05:
        print(f"   CAUTION: train-test gap = {gap:.1%}")
    else:
        print(f"   OK: train-test gap = {gap:.1%}")

    # 5. Feature importance
    print("\n   --- FEATURE IMPORTANCE (XGBoost gain) ---")
    for fname, score in list(ml_result.feature_importance.items())[:10]:
        print(f"   {fname:<35} {score:.4f}")

    if ml_result.shap_importance:
        print("\n   --- FEATURE IMPORTANCE (SHAP mean|value|) ---")
        for fname, score in list(ml_result.shap_importance.items())[:10]:
            print(f"   {fname:<35} {score:.6f}")

    # 6. Skeptic's minimum validation
    print("\n   --- SKEPTIC'S VALIDATION ---")

    auc_pass = ml_result.test_auc > 0.55
    print(f"   AUC > 0.55: {'PASS' if auc_pass else 'FAIL'} ({ml_result.test_auc:.3f})")

    fi_vals = list(ml_result.feature_importance.values())
    concentrated = fi_vals[0] > 0.5 * sum(fi_vals) if fi_vals else False
    print(f"   Feature concentration: {'WARNING' if concentrated else 'OK'}")

    overfit = gap > 0.15
    print(f"   Overfit check: {'FAIL' if overfit else 'OK'} (gap={gap:.1%})")

    overall = auc_pass and not overfit and not concentrated
    print(f"\n   OVERALL: {'PROCEED WITH CAUTION' if overall else 'KILL'}")

    # 7. Save model artifact
    artifact_dir = Path(f"data/strategies/{slug}/meta-label")
    mlf.save(artifact_dir)
    print(f"\n   Model saved to {artifact_dir}")

    # 8. Build result dict
    return {
        "slug": slug,
        "total_signals": len(all_signals),
        "entry_signals": len(entry_signals),
        "labeled_entries": n_labels,
        "label_win_rate": round(n_wins / n_labels, 4),
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

"""Reproduce a bounded recorded-model audit and all annual evaluation phases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .__main__ import write_report
from .data import Snapshot, aligned_returns, require
from .evidence import assess_report, audit_track_d_summary, source_record
from .models import Simulation
from .study import audit_snapshot
from .validation import walk_forward

SOURCE_COMMIT = "a08cac5ee98c96e526b824ac649788d6e1b51bd7"
SOURCES = {
    "track_d": ("docs/research/results/track-d-robustness-results-2026-04-24.md",
                "4b8596c512f5a99af0d589ab7a947317988d870f"),
    "latest_daily": ("reports/daily/2026-11-06.md",
                     "3967217fa5d6d54f46b1b6b521d60efa20f00488"),
}


def phase_diagnostics(snapshot: Snapshot, config: Simulation, min_train: int) -> list[dict[str, Any]]:
    """Show every phase, never optimize a phase or pool overlapping outcomes."""
    require(isinstance(min_train, int) and not isinstance(min_train, bool),
            "Training size must be an integer")
    require(1 <= config.horizon <= 12, "Phase audit is limited to 12 native periods")
    returns = aligned_returns([snapshot], portfolio=False)
    config.validate(min_train)
    require(min_train + 2 * config.horizon - 1 <= len(returns),
            "Need one complete outcome in every phase")
    work = config.paths * config.horizon * (len(returns) - min_train)
    require(work <= 50_000_000, "Phase audit exceeds work budget")
    rows = []
    for offset in range(config.horizon):
        result = walk_forward(returns, np.ones(1), config, min_train=min_train + offset)
        folds = result["folds"]
        # Recompute the score from probabilities and outcomes, not the stored score field.
        model = sum((fold["predicted_loss_probability"] - fold["loss_observed"]) ** 2
                    for fold in folds) / len(folds)
        baseline = sum((fold["training_only_base_rate"] - fold["loss_observed"]) ** 2
                       for fold in folds) / len(folds)
        require(abs(model - result["brier_score"]) < 1e-12, "Model score failed independent arithmetic check")
        rows.append({
            "offset_periods": offset,
            "first_origin_label": snapshot.dates[min_train + offset],
            "folds": len(folds),
            "loss_events": sum(fold["loss_observed"] for fold in folds),
            "model_brier": model, "baseline_brier": baseline,
            "model_minus_baseline_brier": model - baseline,
            "coverage_95_count": sum(fold["central_intervals"]["0.95"]["covered"] for fold in folds),
            "coverage_95": result["central_interval_diagnostics"]["0.95"]["observed_coverage"],
        })
    return rows


def build_audit(root: Path, snapshot_path: Path, as_of: date) -> dict[str, Any]:
    refs = {key: source_record(root, path, blob) for key, (path, blob) in SOURCES.items()}
    strategy = audit_track_d_summary((root / SOURCES["track_d"][0]).read_text(encoding="utf-8"))
    daily = assess_report((root / SOURCES["latest_daily"][0]).read_text(encoding="utf-8"))
    snapshot, evidence = audit_snapshot(snapshot_path, as_of, 200)
    require(snapshot.contract.series_id == "QAUR628BIS" and snapshot.contract.frequency == "Q"
            and snapshot.contract.basis == "real_price_index"
            and snapshot.contract.evidence == "observed", "Expected observed quarterly real housing index")
    config = Simulation(horizon=4, paths=2000, block_length=4, seed=20261111)
    phases = phase_diagnostics(snapshot, config, 80)
    return {
        "schema_version": 1,
        "status": "EVIDENCE_AUDIT_NO_VALIDATED_INVESTMENT_EDGE",
        "as_of": as_of.isoformat(), "source_commit": SOURCE_COMMIT,
        "sources": refs, "track_d_recorded_cohort": strategy,
        "latest_committed_daily_report": daily,
        "housing": {
            "source_audit": evidence, "config": asdict(config), "min_train": 80,
            "all_start_phases": phases,
            "conclusion": "No stable advantage over the baseline across all phases; sparse severe-loss outcomes.",
            "limitations": [
                "Reproduction of the earlier browser table capture, not a fresh market-data download.",
                "Eight-city Australian real price index; not Sydney or investable total return.",
                "Within-phase outcomes do not overlap; outcomes overlap across phases and must not be pooled.",
                "Zero or one severe-loss outcome per phase cannot establish tail calibration.",
                "Latest-vintage historical diagnostic, not an untouched holdout or forward paper validation.",
                "Aggregate metrics only; raw provider observations are not included in this public audit.",
            ],
        },
        "strategy_backtests_reexecuted": False,
        "live_trading_authorized": False,
        "full_repository_tests": "NOT_RUN_IN_RESTRICTED_RUNTIME",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("research-local/decision-lab/model-audit.json"))
    args = parser.parse_args()
    try:
        write_report(build_audit(args.root, args.snapshot, args.as_of), args.out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Audit failed closed: {error}\n")


if __name__ == "__main__":
    main()

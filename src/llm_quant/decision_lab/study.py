"""Historical sensitivity studies with evidence and freshness checks, not advice."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .__main__ import write_report
from .data import Snapshot, aligned_returns, load_snapshot, require
from .models import Simulation, risk_report
from .validation import walk_forward


def audit_snapshot(path: Path, as_of: date, max_age_days: int) -> tuple[Snapshot, dict]:
    require(max_age_days >= 0, "Maximum data age must be nonnegative")
    snapshot = load_snapshot(path)
    last = date.fromisoformat(snapshot.dates[-1])
    captured = datetime.fromisoformat(snapshot.contract.retrieved_at).date()
    require(last <= as_of and captured <= as_of, "Future observations or capture")
    raw = path.parent / "raw" / f"{snapshot.contract.raw_sha256}.raw"
    raw_present = raw.is_file()
    if snapshot.contract.evidence == "observed":
        require(raw_present, "Observed evidence requires the retained source payload")
    if raw_present:
        require(raw.stat().st_size <= 10_000_000, "Source exceeds verification budget")
        require(hashlib.sha256(raw.read_bytes()).hexdigest() == snapshot.contract.raw_sha256,
                "Retained source payload hash mismatch")
    age = (as_of - last).days
    return snapshot, {
        "snapshot_sha256": snapshot.digest(),
        "retained_source_verified": raw_present,
        "source": asdict(snapshot.contract),
        "observations": len(snapshot.dates),
        "first_observation": snapshot.dates[0],
        "last_observation": snapshot.dates[-1],
        "age_from_observation_label_days": age,
        "maximum_age_days": max_age_days,
        "freshness_gate": "PASS" if age <= max_age_days else "STALE",
        "freshness_convention": "Age from source period label, not release date.",
        "current_forecast_authorized": False,
    }


def historical_study(
    snapshot: Snapshot, config: Simulation, *, blocks: list[int],
    windows: list[int], min_train: int,
) -> dict:
    snapshot.validate()
    require(1 <= len(blocks) <= 5 and len(set(blocks)) == len(blocks),
            "Use 1..5 distinct block lengths")
    require(1 <= len(windows) <= 5 and len(set(windows)) == len(windows),
            "Use 1..5 distinct history windows")
    returns = aligned_returns([snapshot], portfolio=False)
    require(isinstance(min_train, int) and not isinstance(min_train, bool),
            "Training size must be an integer")
    require(min_train >= max(24, 3 * config.horizon), "Insufficient training history")
    require(min_train + config.horizon <= len(returns), "No complete validation fold")
    for window in windows:
        require(isinstance(window, int) and not isinstance(window, bool)
                and max(24, 3 * config.horizon) <= window <= len(returns),
                "History window exceeds data or is too short")
        for block in blocks:
            replace(config, block_length=block).validate(window)
    folds = (len(returns) - min_train) // config.horizon
    workload = config.paths * config.horizon * (len(blocks) * len(windows) + folds)
    require(workload <= 50_000_000, "Study exceeds offline work budget")
    scenarios = []
    for window in windows:
        for block in blocks:
            setting = replace(config, block_length=block)
            model = risk_report(returns[-window:], np.ones(1), setting)
            scenarios.append({
                "history_periods": window, "block_length": block,
                "history_start": snapshot.dates[len(returns) - window],
                "results": model["results"],
            })
    diagnostic = walk_forward(returns, np.ones(1), config, min_train=min_train)
    for fold in diagnostic["folds"]:
        fold["origin_observation_label"] = snapshot.dates[fold["origin_return_index"]]
        fold["outcome_observation_label"] = snapshot.dates[fold["test_end_exclusive"]]
    return {
        "schema_version": 1,
        "status": ("SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE"
                   if snapshot.contract.evidence == "synthetic"
                   else "HISTORICAL_RESEARCH_NOT_CURRENT_FORECAST"),
        "config": asdict(config),
        "native_frequency": snapshot.contract.frequency,
        "snapshot_sha256": snapshot.digest(),
        "source": asdict(snapshot.contract),
        "sensitivity": scenarios,
        "validation": diagnostic,
        "interpretation": [
            "No best configuration is selected from the sensitivity grid.",
            "All scenarios are conditional on the supplied history, not current odds.",
            "A housing index is not a tradable portfolio and excludes rent and leverage.",
            "Source freshness does not establish model validity or authorize capital use.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--max-age-days", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    parser.add_argument("--windows", type=int, nargs="+", required=True)
    parser.add_argument("--validation-block", type=int, required=True)
    parser.add_argument("--min-train", type=int, required=True)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--out", type=Path,
                        default=Path("research-local/decision-lab/historical-study.json"))
    args = parser.parse_args()
    try:
        snapshot, audit = audit_snapshot(args.snapshot, args.as_of, args.max_age_days)
        config = Simulation(args.horizon, args.paths, args.validation_block, args.seed)
        result = historical_study(snapshot, config, blocks=args.blocks,
                                  windows=args.windows, min_train=args.min_train)
        result["audit"] = audit
        result["as_of"] = args.as_of.isoformat()
        if audit["freshness_gate"] == "STALE":
            result["status"] = "HISTORICAL_RESEARCH_STALE_NOT_CURRENT_FORECAST"
        if snapshot.contract.evidence == "synthetic":
            result["status"] = "SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE"
        write_report(result, args.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Research rejected: {error}\n")


if __name__ == "__main__":
    main()

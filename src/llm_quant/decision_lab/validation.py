"""Retrospective rolling-origin checks with non-overlapping realized labels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import require
from .models import Simulation, bootstrap_paths, validate_returns


def walk_forward(
    returns: np.ndarray, weights: np.ndarray, config: Simulation,
    *, min_train: int, loss: float = 0.10,
) -> dict:
    """Do not call this point-in-time validation of a revised historical snapshot.

    Each fit sees returns strictly before the origin. Evaluation origins advance
    by the forecast horizon, so realized evaluation labels do not overlap.
    """
    data, allocations = validate_returns(returns, weights)
    require(0 < loss < 1, "Loss threshold must be in (0, 1)")
    config.validate(min_train)
    require(len(data) >= min_train + config.horizon, "No complete validation fold")
    folds = []
    for origin in range(min_train, len(data) - config.horizon + 1, config.horizon):
        training = data[:origin]
        sample = bootstrap_paths(
            training, allocations, replace(config, seed=config.seed + origin),
        )
        predicted = float(np.mean(sample[:, -1] <= 1 - loss))
        realized = float(
            np.prod(1 + data[origin:origin + config.horizon], axis=0) @ allocations - 1,
        )
        historical = []
        for start in range(0, origin - config.horizon + 1, config.horizon):
            gross = np.prod(1 + training[start:start + config.horizon], axis=0)
            historical.append(float(gross @ allocations) <= 1 - loss)
        base_rate = float(np.mean(historical))
        outcome = int(realized + 1 <= 1 - loss)
        folds.append({
            "origin_return_index": origin,
            "train_end_exclusive": origin,
            "test_end_exclusive": origin + config.horizon,
            "predicted_loss_probability": predicted,
            "training_only_base_rate": base_rate,
            "realized_return": realized,
            "loss_observed": outcome,
            "brier": (predicted - outcome) ** 2,
            "baseline_brier": (base_rate - outcome) ** 2,
        })
    score = float(np.mean([fold["brier"] for fold in folds]))
    baseline = float(np.mean([fold["baseline_brier"] for fold in folds]))
    return {
        "status": "RETROSPECTIVE_DIAGNOSTIC_NOT_POINT_IN_TIME",
        "fold_count": len(folds),
        "brier_score": score,
        "training_base_rate_brier": baseline,
        "brier_skill_vs_base_rate": None if baseline == 0 else 1 - score / baseline,
        "folds": folds,
        "warnings": [
            "Latest-vintage revisions prevent a claim of historical tradability.",
            "Non-overlapping labels do not guarantee independent forecast errors.",
            "No automatic promotion: calibration, coverage and an untouched holdout remain.",
        ] + (["Fewer than 20 evaluation folds: weak calibration evidence."]
             if len(folds) < 20 else []),
    }

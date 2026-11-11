"""Retrospective rolling-origin checks with non-overlapping realized labels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data import require
from .models import Simulation, bootstrap_paths, validate_returns


def calibration_bins(folds: list[dict], bins: int = 5) -> list[dict]:
    """Descriptive reliability bins, not confidence intervals or independent trials."""
    require(isinstance(bins, int) and not isinstance(bins, bool) and 1 <= bins <= 20,
            "Use 1..20 reliability bins")
    result = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [fold for fold in folds
                    if lower <= fold["predicted_loss_probability"] < upper
                    or (index == bins - 1 and fold["predicted_loss_probability"] == 1)]
        result.append({
            "lower": lower,
            "upper": upper,
            "count": len(selected),
            "mean_predicted": (float(np.mean([item["predicted_loss_probability"]
                                              for item in selected]))
                               if selected else None),
            "observed_frequency": (float(np.mean([item["loss_observed"]
                                                  for item in selected]))
                                   if selected else None),
        })
    return result


def walk_forward(
    returns: np.ndarray, weights: np.ndarray, config: Simulation,
    *, min_train: int, loss: float = 0.10,
) -> dict:
    """Fit strictly before each origin; do not claim point-in-time vintage evidence."""
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
        terminal = sample[:, -1] - 1
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
        intervals = {}
        for coverage in (0.5, 0.8, 0.95):
            alpha = 1 - coverage
            lower, upper = np.quantile(terminal, [alpha / 2, 1 - alpha / 2])
            score = upper - lower
            if realized < lower:
                score += 2 / alpha * (lower - realized)
            elif realized > upper:
                score += 2 / alpha * (realized - upper)
            intervals[str(coverage)] = {
                "lower": float(lower), "upper": float(upper),
                "covered": bool(lower <= realized <= upper),
                "interval_score": float(score),
            }
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
            "central_intervals": intervals,
        })
    score = float(np.mean([fold["brier"] for fold in folds]))
    baseline = float(np.mean([fold["baseline_brier"] for fold in folds]))
    events = sum(fold["loss_observed"] for fold in folds)
    interval_summary = {}
    for key in ("0.5", "0.8", "0.95"):
        intervals = [fold["central_intervals"][key] for fold in folds]
        interval_summary[key] = {
            "target_coverage": float(key),
            "observed_coverage": float(np.mean([item["covered"] for item in intervals])),
            "mean_interval_score": float(np.mean([item["interval_score"] for item in intervals])),
            "mean_width": float(np.mean([item["upper"] - item["lower"] for item in intervals])),
        }
    return {
        "status": "RETROSPECTIVE_DIAGNOSTIC_NOT_POINT_IN_TIME",
        "fold_count": len(folds),
        "loss_event_count": events,
        "loss_threshold": loss,
        "brier_score": score,
        "training_base_rate_brier": baseline,
        "brier_skill_vs_base_rate": None if baseline == 0 else 1 - score / baseline,
        "reliability_bins": calibration_bins(folds),
        "central_interval_diagnostics": interval_summary,
        "folds": folds,
        "warnings": [
            "Latest-vintage revisions prevent a claim of historical tradability.",
            "Non-overlapping labels do not guarantee independent forecast errors.",
            "Parameters are not selected on these results; this is exploratory validation.",
            "No automatic promotion: independent replication and an untouched holdout remain.",
        ] + (["Fewer than 20 evaluation folds: weak calibration evidence."]
             if len(folds) < 20 else [])
          + (["Fewer than 5 realized loss events: tail calibration evidence is sparse."]
             if events < 5 else []),
    }

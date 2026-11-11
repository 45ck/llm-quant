"""Use upstream allocators in the existing cost-aware, held-out evaluation engine."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..benchmark import EvaluationPlan, compare_baselines, performance, simulate
from ..data import require
from .allocation import cvxpy_min_cvar, skfolio_min_risk, sklearn_inverse_volatility
from .contracts import TrainingBatch, load

ALLOCATORS = ("sklearn_inverse_volatility", "skfolio_variance", "skfolio_cvar", "cvxpy_cvar")


def fit(batch: TrainingBatch, name: str) -> dict:
    require(name in ALLOCATORS, "Unknown allocation method")
    if name == "sklearn_inverse_volatility":
        return sklearn_inverse_volatility(batch)
    if name == "cvxpy_cvar":
        return cvxpy_min_cvar(batch)
    return skfolio_min_risk(batch, measure=name.removeprefix("skfolio_"))


def evaluate(batch: TrainingBatch, plan: EvaluationPlan, methods: tuple[str, ...]) -> dict:
    require(bool(methods) and len(methods) <= len(ALLOCATORS)
            and len(set(methods)) == len(methods)
            and all(name in ALLOCATORS for name in methods), "Invalid allocation selection")
    data = batch.returns
    plan.validate(len(data))
    boundary = len(data) - plan.holdout_size
    splitter = load("sklearn", "model_selection").TimeSeriesSplit(
        n_splits=plan.splits, test_size=plan.test_size, gap=plan.gap)
    folds = list(splitter.split(data[:boundary]))
    start = int(folds[0][1][0])
    outputs = {}
    for name in methods:
        schedules, receipts = {}, []
        for training, testing in folds:
            train = TrainingBatch(data[training], tuple(batch.dates[int(i)] for i in training),
                                  batch.dates[int(testing[0]) - 1], batch.currency, batch.evidence)
            fitted = fit(train, name)
            schedules[int(testing[0]) - start] = np.asarray(fitted["output"]["weights"])
            receipts.append(fitted)
        cutoff = boundary - plan.gap
        heldout_fit = fit(TrainingBatch(data[:cutoff], batch.dates[:cutoff],
                                       batch.dates[boundary - 1], batch.currency, batch.evidence), name)
        development = simulate(data[start:boundary], schedules, plan.transaction_bps)
        holdout = simulate(data[boundary:], {0: np.asarray(heldout_fit["output"]["weights"])},
                           plan.transaction_bps)
        outputs[name] = {
            "development": {**development, "metrics": performance(development["nav"], plan.periods_per_year)},
            "reserved_holdout": {**holdout, "metrics": performance(holdout["nav"], plan.periods_per_year)},
            "fit_receipts": receipts, "holdout_fit_receipt": heldout_fit,
        }
    # The same accounting and evaluation periods are used for the reference baselines.
    baseline = compare_baselines(data, plan)
    return {
        "status": ("SYNTHETIC_INTEGRATION_CHECK" if batch.evidence == "synthetic"
                   else "RETROSPECTIVE_LIBRARY_COMPARISON_NOT_POINT_IN_TIME"),
        "data_sha256": batch.digest(), "plan": asdict(plan),
        "holdout_start_index": boundary, "results": outputs, "baseline_reference": baseline,
        "selected_winner": None,
        "limitations": ["Do not tune methods or parameters after inspecting this holdout.",
                        "Latest-vintage history does not establish historical tradability.",
                        "Bar-return accounting is not intraday execution or market-impact modeling."],
    }

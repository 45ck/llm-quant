"""Training-only allocation adapters; costs are applied by the evaluation engine."""

from __future__ import annotations

import math

import numpy as np

from ..data import require
from .contracts import TrainingBatch, cap_check, checked_weights, load, result


def sklearn_inverse_volatility(batch: TrainingBatch) -> dict:
    covariance = load("sklearn", "covariance").LedoitWolf(store_precision=False).fit(
        batch.returns.copy()).covariance_
    diagonal = np.diag(covariance)
    require(bool(np.all(np.isfinite(covariance))) and bool(np.all(diagonal > 0)),
            "Degenerate shrinkage covariance")
    inverse = 1 / np.sqrt(diagonal)
    output = checked_weights(inverse / inverse.sum(), len(diagonal), 1)
    output["covariance"] = covariance.tolist()
    return result(batch, "sklearn", "ledoit_wolf_inverse_volatility", output,
                  ["Long only; fully invested; not equal-risk-contribution optimization."])


def skfolio_min_risk(batch: TrainingBatch, *, measure: str = "variance",
                     cap: float = 1.0, alpha: float = 0.95) -> dict:
    cap_check(batch, cap)
    require(measure in {"variance", "cvar"}, "Unsupported risk measure")
    require(math.isfinite(alpha) and 0.5 < alpha < 1, "Invalid CVaR confidence level")
    opt = load("skfolio", "optimization")
    risk = load("skfolio").RiskMeasure
    estimator = opt.MeanRisk(
        objective_function=opt.ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk.VARIANCE if measure == "variance" else risk.CVAR,
        min_weights=0.0, max_weights=cap, budget=1.0, cvar_beta=alpha,
        solver="CLARABEL", raise_on_failure=True,
    )
    estimator.fit(batch.returns.copy())
    output = checked_weights(estimator.weights_, batch.returns.shape[1], cap)
    output.update({"risk_measure": measure, "alpha": alpha, "solver": "CLARABEL"})
    return result(batch, "skfolio", "mean_risk", output,
                  ["Sample prior; no held-out values are supplied to fit.",
                   "Risk estimate and alpha are assumptions, not future loss probabilities."])


def cvxpy_min_cvar(batch: TrainingBatch, *, cap: float = 1.0,
                    alpha: float = 0.95) -> dict:
    cap_check(batch, cap)
    require(math.isfinite(alpha) and 0.5 < alpha < 1, "Invalid CVaR confidence level")
    cp = load("cvxpy")
    require("CLARABEL" in cp.installed_solvers(), "CLARABEL solver unavailable")
    rows, assets = batch.returns.shape
    weights, threshold, excess = cp.Variable(assets), cp.Variable(), cp.Variable(rows)
    objective = threshold + cp.sum(excess) / (rows * (1 - alpha))
    problem = cp.Problem(cp.Minimize(objective), [
        weights >= 0, weights <= cap, cp.sum(weights) == 1, excess >= 0,
        excess >= -batch.returns @ weights - threshold,
    ])
    problem.solve(solver="CLARABEL", max_iter=200)
    require(problem.status == cp.OPTIMAL, "CVaR optimizer did not reach optimal status")
    require(problem.value is not None and math.isfinite(float(problem.value)),
            "Invalid optimizer objective")
    output = checked_weights(weights.value, assets, cap)
    output.update({"historical_cvar_loss": float(problem.value), "alpha": alpha,
                   "solver": "CLARABEL", "solver_status": str(problem.status)})
    return result(batch, "cvxpy", "historical_cvar_linear_program", output,
                  ["CVaR uses in-sample empirical loss scenarios; no distribution guarantee.",
                   "Negative historical loss values are possible for positive-only scenarios."])

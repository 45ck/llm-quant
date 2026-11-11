"""Dependent bootstrap, volatility forecasts and HAC statistics via upstream APIs."""

from __future__ import annotations

import math

import numpy as np

from ..data import require
from .contracts import TrainingBatch, load, result


def arch_mean_interval(batch: TrainingBatch, *, block: int = 10,
                       repetitions: int = 1000, seed: int = 17) -> dict:
    require(type(block) is int and 1 <= block <= len(batch.returns), "Invalid block size")
    require(type(repetitions) is int and 100 <= repetitions <= 10000,
            "Bootstrap budget is 100..10000 repetitions")
    require(type(seed) is int and seed >= 0, "Invalid random seed")
    bs = load("arch", "bootstrap").StationaryBootstrap(
        block, batch.returns.copy(), seed=seed)
    interval = np.asarray(bs.conf_int(lambda values: np.mean(values, axis=0),
                                     reps=repetitions, method="percentile", size=0.95))
    require(interval.shape == (2, batch.returns.shape[1])
            and bool(np.all(np.isfinite(interval)))
            and bool(np.all(interval[0] <= interval[1])), "Invalid bootstrap interval")
    return result(batch, "arch", "stationary_bootstrap_mean", {
        "mean_return": batch.returns.mean(axis=0).tolist(),
        "lower": interval[0].tolist(), "upper": interval[1].tolist(),
        "nominal_coverage": 0.95, "block": block,
        "repetitions": repetitions, "seed": seed,
    }, ["Interval for a stationary process mean, NOT a future-return prediction interval.",
        "Block size must be predeclared and sensitivity-tested; structural breaks matter."])


def arch_garch_variance(batch: TrainingBatch) -> dict:
    require(batch.returns.shape[1] == 1 and len(batch.returns) >= 100,
            "GARCH requires one series and at least 100 observations")
    require(float(np.std(batch.returns)) > 1e-12, "Constant returns cannot fit GARCH")
    estimator = load("arch").arch_model(100 * batch.returns[:, 0], mean="Zero",
                                        vol="GARCH", p=1, q=1, dist="t", rescale=False)
    fitted = estimator.fit(disp="off", options={"maxiter": 300})
    require(fitted.convergence_flag == 0, "GARCH convergence failed")
    forecast = fitted.forecast(horizon=1, reindex=False)
    variance = float(forecast.variance.iloc[-1, 0]) / 10000
    require(math.isfinite(variance) and variance > 0, "Invalid GARCH variance")
    return result(batch, "arch", "garch_1_1_student_t_zero_mean", {
        "one_period_variance": variance, "one_period_volatility": math.sqrt(variance),
        "fit_observations": int(fitted.nobs), "convergence_flag": int(fitted.convergence_flag),
    }, ["Input is simple-return fractions; fitting uses percent units, output reconverts.",
        "Conditional variance is not an expected return, crash probability or calibrated interval."])


def statsmodels_hac_mean(batch: TrainingBatch, *, maxlags: int = 5) -> dict:
    require(batch.returns.shape[1] == 1, "HAC mean requires one series")
    require(type(maxlags) is int and 0 <= maxlags < len(batch.returns) // 2,
            "HAC lags must be explicit and less than half the sample")
    api = load("statsmodels", "api")
    fitted = api.OLS(batch.returns[:, 0], np.ones((len(batch.returns), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}, use_t=True)
    bounds = np.asarray(fitted.conf_int(alpha=0.05))[0]
    output = {"mean_return": float(fitted.params[0]), "standard_error": float(fitted.bse[0]),
              "p_value_zero_mean": float(fitted.pvalues[0]),
              "lower": float(bounds[0]), "upper": float(bounds[1]), "maxlags": maxlags}
    require(all(math.isfinite(value) for value in output.values()), "Degenerate HAC inference")
    return result(batch, "statsmodels", "intercept_only_ols_hac", output,
                  ["This is a predeclared mean test, not a causal model or predictive edge.",
                   "HAC is asymptotic; selection/multiple-testing corrections remain necessary."])

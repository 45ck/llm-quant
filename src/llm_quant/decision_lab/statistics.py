"""Established statistical routines with explicit applicability limits."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from .data import require


def event_rate_interval(events: int, observations: int, *,
                        independent_bernoulli_assumption: bool = False,
                        confidence: float = .95) -> dict:
    """Clopper-Pearson interval, not a calibrated future-event probability.

    Refuse unless independence/constant-rate assumptions are explicitly accepted.
    In particular, overlapping housing windows do not satisfy them automatically.
    """
    require(independent_bernoulli_assumption is True,
            "Independent constant-rate Bernoulli assumption must be explicit")
    require(type(events) is int and type(observations) is int
            and observations > 0 and 0 <= events <= observations, "Invalid event counts")
    require(math.isfinite(confidence) and 0 < confidence < 1, "Invalid confidence level")
    interval = binomtest(events, observations).proportion_ci(confidence, method="exact")
    return {"method": "scipy_clopper_pearson", "events": events,
            "observations": observations, "sample_rate": events / observations,
            "lower": float(interval.low), "upper": float(interval.high),
            "confidence_level": confidence,
            "status": "DESCRIPTIVE_INTERVAL_UNDER_EXPLICIT_BERNOULLI_ASSUMPTION"}


def holm_family(pvalues: list[float], *, declared_trials: int, alpha: float = .05) -> dict:
    """Correct the complete declared family, not only its surviving strategies.

    Valid input p-values and a complete family are still researcher responsibilities.
    This does not replace financial SPA/Reality Check/DSR procedures or prove profit.
    """
    values = np.asarray(pvalues, dtype=float)
    require(values.ndim == 1 and len(values) > 0, "Nonempty p-value vector required")
    require(type(declared_trials) is int and declared_trials == len(values),
            "Include all declared trials, not only winners")
    require(bool(np.all(np.isfinite(values) & (values >= 0) & (values <= 1))),
            "P-values must lie in [0, 1]")
    require(math.isfinite(alpha) and 0 < alpha < 1, "Invalid familywise error level")
    rejected, adjusted, _, _ = multipletests(values, alpha=alpha, method="holm")
    return {"method": "statsmodels_holm", "declared_trials": declared_trials,
            "alpha": alpha, "adjusted_pvalues": adjusted.tolist(),
            "reject_null": rejected.tolist(),
            "status": "MULTIPLE_TESTING_DIAGNOSTIC_NOT_INVESTMENT_APPROVAL"}

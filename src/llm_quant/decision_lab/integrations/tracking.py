"""Qlib experiment records restricted to local summary-only storage."""

from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import date

from ..data import require
from .contracts import load, versions


def record_qlib_summary(summary: dict, directory: Path, *, experiment: str = "decision-lab") -> dict:
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", experiment) is not None,
            "Invalid experiment name")
    root = (Path.cwd() / "research-local").resolve()
    target = directory.resolve()
    require(not directory.is_symlink() and target.is_relative_to(root) and target != root,
            "Tracking must be inside a research-local subdirectory")
    require(not os.environ.get("MLFLOW_TRACKING_URI") and not os.environ.get("MLFLOW_REGISTRY_URI"),
            "Clear external MLflow URI environment variables before local recording")
    require(summary.get("status") in {"SYNTHETIC_INTEGRATION_CHECK", "UNVALIDATED_LIBRARY_RESEARCH"},
            "Only validated summary shapes may be recorded")
    # Allowlist scalar metadata only: no models, pickles, raw frames or arbitrary artifacts.
    filtered = {name: summary[name] for name in (
        "status", "backend", "method", "training_rows", "training_end", "as_of", "currency",
        "training_data_sha256", "implementation_sha256")}
    require(all(isinstance(value, (str, int)) and not isinstance(value, bool)
                for value in filtered.values()), "Summary metadata must be scalar")
    require(all(len(str(value)) <= 128 for value in filtered.values()), "Metadata value too long")
    for name in ("training_data_sha256", "implementation_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", filtered[name]) is not None, "Invalid summary hash")
    require(filtered["currency"] in {"AUD", "USD", "EUR", "GBP"}, "Invalid summary currency")
    require(filtered["backend"] in {"sklearn", "statsmodels", "skfolio", "arch", "cvxpy"},
            "Unregistered summary backend")
    require(filtered["method"] in {"ledoit_wolf_inverse_volatility", "intercept_only_ols_hac",
            "mean_risk", "historical_cvar_linear_program", "stationary_bootstrap_mean",
            "garch_1_1_student_t_zero_mean"}, "Unregistered summary method")
    require(type(filtered["training_rows"]) is int and 20 <= filtered["training_rows"] <= 50000,
            "Invalid summary observation count")
    require(date.fromisoformat(filtered["training_end"]) <= date.fromisoformat(filtered["as_of"]),
            "Future summary observation")
    workflow = load("qlib", "workflow")
    expm = load("qlib", "workflow.expm")
    target.mkdir(parents=True, exist_ok=True)
    recorder = workflow.QlibRecorder(expm.MLflowExpManager(
        uri=target.as_uri(), default_exp_name=experiment))
    with recorder.start(experiment_name=experiment, recorder_name="summary-only"):
        recorder.log_params(**filtered)
        identifier = str(recorder.get_recorder().id)
    return {"status": "LOCAL_EXPERIMENT_RECORDED_NOT_EXTERNALLY_ATTESTED",
            "backend": "qlib", "libraries": versions("qlib"), "recorder_id": identifier,
            "uri": target.as_uri(), "summary": filtered,
            "limitations": ["No Qlib data provider, training pipeline or trading service was initialized.",
                            "The record is local and mutable; it is not independent timestamp evidence."]}

"""Shared validation, provenance and explicit optional-dependency boundaries."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ..data import require

LIBRARIES = {
    "sklearn": ("scikit-learn", "sklearn", "core"),
    "statsmodels": ("statsmodels", "statsmodels", "core"),
    "skfolio": ("skfolio", "skfolio", "skfolio"),
    "arch": ("arch", "arch", "arch"),
    "backtesting": ("backtesting", "backtesting", "backtesting"),
    "qlib": ("pyqlib", "qlib", "qlib"),
    "cvxpy": ("cvxpy", "cvxpy", "cvxpy"),
    "pandera": ("pandera", "pandera", "pandera"),
    "hypothesis": ("hypothesis", "hypothesis", "hypothesis"),
}


class MissingIntegration(ImportError):
    """The requested integration is not installed; never substitute another model."""


def load(key: str, submodule: str = "") -> Any:
    require(key in LIBRARIES, "Unregistered integration")
    distribution, module, profile = LIBRARIES[key]
    require(not submodule or all(part.isidentifier() for part in submodule.split(".")),
            "Invalid submodule")
    try:
        return importlib.import_module(module + ("." + submodule if submodule else ""))
    except ModuleNotFoundError as error:
        if error.name == module or (error.name and error.name.startswith(module + ".")):
            raise MissingIntegration(
                f"Install {distribution} using requirements/integrations/{profile}.txt; "
                "no automatic installation or fallback was performed"
            ) from error
        # A broken transitive dependency is not equivalent to an absent optional extra.
        raise


def versions(*keys: str) -> dict[str, str]:
    result = {}
    for key in keys:
        name = LIBRARIES[key][0]
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def doctor() -> dict[str, Any]:
    """Metadata only: no imports of optional backends, network or installation."""
    return {
        "status": "DEPENDENCY_INVENTORY_NOT_EXECUTION_EVIDENCE",
        "libraries": versions(*LIBRARIES),
        "warning": "Installed does not mean compatible, tested or statistically validated.",
    }


@dataclass(frozen=True)
class TrainingBatch:
    returns: np.ndarray
    dates: tuple[str, ...]
    as_of: str
    currency: str
    evidence: str

    def __post_init__(self) -> None:
        values = np.array(self.returns, dtype=np.float64, copy=True)
        require(values.ndim == 2 and 20 <= len(values) <= 50000
                and 1 <= values.shape[1] <= 200, "Training shape outside research budget")
        require(bool(np.all(np.isfinite(values) & (values > -1))),
                "Training simple returns must be finite and greater than -1")
        dates = tuple(date.fromisoformat(item) for item in self.dates)
        require(len(dates) == len(values), "One date per training observation required")
        require(all(a < b for a, b in zip(dates[:-1], dates[1:], strict=True)),
                "Training dates must be unique and ordered")
        require(dates[-1] <= date.fromisoformat(self.as_of), "Future training observation")
        require(self.currency in {"AUD", "USD", "EUR", "GBP"}, "Explicit supported currency required")
        require(self.evidence in {"observed", "synthetic"}, "Explicit evidence kind required")
        values.setflags(write=False)
        object.__setattr__(self, "returns", values)
        object.__setattr__(self, "dates", tuple(item.isoformat() for item in dates))

    def digest(self) -> str:
        metadata = {"dates": self.dates, "as_of": self.as_of,
                    "currency": self.currency, "evidence": self.evidence}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(self.returns.astype("<f8").tobytes())
        return digest.hexdigest()


def result(batch: TrainingBatch, backend: str, method: str, output: dict,
           assumptions: list[str]) -> dict:
    # JSON rejects NaNs; a serialization check is not a numerical correctness proof.
    json.dumps(output, allow_nan=False)
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code.update(path.name.encode() + b"\0" + path.read_bytes())
    return {
        "schema_version": 1,
        "status": ("SYNTHETIC_INTEGRATION_CHECK" if batch.evidence == "synthetic"
                   else "UNVALIDATED_LIBRARY_RESEARCH"),
        "backend": backend, "method": method, "libraries": versions(backend),
        "training_rows": len(batch.returns), "training_end": batch.dates[-1],
        "as_of": batch.as_of, "currency": batch.currency,
        "training_data_sha256": batch.digest(), "implementation_sha256": code.hexdigest(),
        "output": output, "assumptions": assumptions,
        "limitations": [
            "Training labels do not prove point-in-time availability or data rights.",
            "No current-price forecast, investment edge, trade or promotion is established.",
        ],
    }


def cap_check(batch: TrainingBatch, cap: float) -> None:
    require(math.isfinite(cap) and 0 < cap <= 1, "Invalid weight cap")
    require(cap * batch.returns.shape[1] >= 1 - 1e-12, "Infeasible fully invested weight cap")


def checked_weights(raw: Any, assets: int, cap: float) -> dict:
    values = np.asarray(raw, dtype=float)
    require(values.shape == (assets,) and bool(np.all(np.isfinite(values))),
            "Missing or invalid optimizer weights")
    require(bool(np.all(values >= -1e-7)) and bool(np.all(values <= cap + 1e-7))
            and abs(float(values.sum()) - 1) <= 1e-7, "Optimizer violated constraints")
    clean = np.maximum(values, 0)
    clean /= clean.sum()
    adjustment = float(np.abs(clean - values).sum())
    require(adjustment <= 1e-6 and bool(np.all(clean <= cap + 1e-7)),
            "Numerical correction exceeds tolerance")
    return {"weights": clean.tolist(), "numerical_projection_l1": adjustment,
            "weight_cap": cap}

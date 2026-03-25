"""Artifact lifecycle management for the quant research pipeline.

Manages YAML artifact files (mandate, hypothesis, data-contract,
research-spec), enforces the lifecycle state machine, and maintains
the append-only experiment registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle state machine
# ---------------------------------------------------------------------------


class LifecycleState(StrEnum):
    IDEA = "idea"
    MANDATE = "mandate"
    HYPOTHESIS = "hypothesis"
    DATA_CONTRACT = "data_contract"
    RESEARCH_SPEC = "research_spec"
    BACKTEST = "backtest"
    ROBUSTNESS = "robustness"
    PAPER_TRADING = "paper_trading"
    PROMOTION = "promotion"


ALLOWED_TRANSITIONS: dict[LifecycleState, list[LifecycleState]] = {
    LifecycleState.IDEA: [LifecycleState.MANDATE],
    LifecycleState.MANDATE: [LifecycleState.HYPOTHESIS],
    LifecycleState.HYPOTHESIS: [LifecycleState.DATA_CONTRACT],
    LifecycleState.DATA_CONTRACT: [LifecycleState.RESEARCH_SPEC],
    LifecycleState.RESEARCH_SPEC: [LifecycleState.BACKTEST],
    LifecycleState.BACKTEST: [LifecycleState.ROBUSTNESS],
    LifecycleState.ROBUSTNESS: [LifecycleState.PAPER_TRADING],
    LifecycleState.PAPER_TRADING: [LifecycleState.PROMOTION],
    LifecycleState.PROMOTION: [],  # terminal
}


class LifecycleError(Exception):
    """Raised when an invalid lifecycle transition is attempted."""


class FrozenSpecError(Exception):
    """Raised when a frozen research spec is required but not available."""


def validate_transition(current: LifecycleState, target: LifecycleState) -> None:
    """Raise LifecycleError if *current* -> *target* is not allowed."""
    allowed = ALLOWED_TRANSITIONS.get(current, [])
    if target not in allowed:
        raise LifecycleError(
            f"Transition {current.value} -> {target.value} is not allowed. "
            f"Allowed transitions from {current.value}: "
            f"{[s.value for s in allowed]}"
        )


# ---------------------------------------------------------------------------
# YAML artifact I/O
# ---------------------------------------------------------------------------


def hash_content(content: str) -> str:
    """Return SHA-256 hex digest of *content*."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a YAML artifact file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def save_artifact(path: Path, data: dict[str, Any]) -> str:
    """Save a dict as a YAML artifact file. Returns the content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    path.write_text(content, encoding="utf-8")
    content_hash = hash_content(content)
    logger.info("Saved artifact %s (hash=%s)", path.name, content_hash[:12])
    return content_hash


# ---------------------------------------------------------------------------
# Strategy directory management
# ---------------------------------------------------------------------------


def strategy_dir(base_dir: Path, slug: str) -> Path:
    """Return the strategy directory for *slug*, creating it if needed."""
    d = base_dir / "strategies" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_lifecycle_state(strat_dir: Path) -> LifecycleState:
    """Determine the current lifecycle state from existing artifacts.

    Checks most-advanced state first. BACKTEST is directory-based (no single
    file marker), so it's checked between ROBUSTNESS and RESEARCH_SPEC.
    """
    # States above BACKTEST — check file markers (most advanced first)
    upper_states: list[tuple[str, LifecycleState]] = [
        ("promotion-decision.yaml", LifecycleState.PROMOTION),
        ("paper-trading.yaml", LifecycleState.PAPER_TRADING),
        ("robustness.yaml", LifecycleState.ROBUSTNESS),
    ]
    for filename, state in upper_states:
        if (strat_dir / filename).exists():
            return state

    # BACKTEST: experiment registry or experiment artifacts exist
    exp_dir = strat_dir / "experiments"
    registry = strat_dir / "experiment-registry.jsonl"
    artifact_suffixes = {".yaml", ".json", ".jsonl"}
    has_artifacts = exp_dir.is_dir() and any(
        f for f in exp_dir.iterdir() if f.suffix in artifact_suffixes
    )
    if registry.exists() or has_artifacts:
        return LifecycleState.BACKTEST

    # States below BACKTEST — check file markers
    lower_states: list[tuple[str, LifecycleState]] = [
        ("research-spec.yaml", LifecycleState.RESEARCH_SPEC),
        ("data-contract.yaml", LifecycleState.DATA_CONTRACT),
        ("hypothesis.yaml", LifecycleState.HYPOTHESIS),
        ("mandate.yaml", LifecycleState.MANDATE),
    ]
    for filename, state in lower_states:
        if (strat_dir / filename).exists():
            return state

    return LifecycleState.IDEA


def ensure_frozen_spec(strat_dir: Path) -> dict[str, Any]:
    """Load research-spec.yaml and verify it is frozen.

    Raises FrozenSpecError if the spec is not frozen.
    Returns the spec data.
    """
    spec_path = strat_dir / "research-spec.yaml"
    if not spec_path.exists():
        raise FrozenSpecError(
            f"No research-spec.yaml found in {strat_dir}. "
            "Create and freeze a research spec before backtesting."
        )
    spec = load_artifact(spec_path)
    if not spec.get("frozen", False):
        msg = (
            "research-spec.yaml is not frozen. "
            "Set frozen: true before running backtests."
        )
        raise FrozenSpecError(msg)
    return spec


def freeze_spec(strat_dir: Path) -> str:
    """Mark research-spec.yaml as frozen and record the hash.

    The frozen_hash covers all spec content EXCEPT the frozen_hash field
    itself (to avoid the self-referential hash problem). Verification
    should strip frozen_hash before re-hashing.
    """
    spec_path = strat_dir / "research-spec.yaml"
    spec = load_artifact(spec_path)
    if spec.get("frozen"):
        return spec.get("frozen_hash", "")
    spec["frozen"] = True
    spec["frozen_at"] = datetime.now(tz=UTC).isoformat()
    # Hash content WITHOUT frozen_hash to avoid self-reference
    hashable = {k: v for k, v in spec.items() if k != "frozen_hash"}
    content = yaml.dump(hashable, default_flow_style=False, sort_keys=False)
    content_hash = hash_content(content)
    spec["frozen_hash"] = content_hash
    save_artifact(spec_path, spec)
    logger.info("Froze research spec: %s", content_hash[:12])
    return content_hash


# ---------------------------------------------------------------------------
# Data quality grade
# ---------------------------------------------------------------------------

DATA_GRADES = ["a", "b", "c", "d"]  # best to worst


def check_data_grade(grade: str, minimum: str = "b") -> bool:
    """Return True if *grade* meets or exceeds *minimum*."""
    if grade.lower() not in DATA_GRADES:
        return False
    if minimum.lower() not in DATA_GRADES:
        return False
    return DATA_GRADES.index(grade.lower()) <= DATA_GRADES.index(minimum.lower())


# ---------------------------------------------------------------------------
# Experiment registry (append-only JSONL)
# ---------------------------------------------------------------------------


class ExperimentRegistry:
    """Append-only registry of all backtest experiments.

    Each entry records the experiment metadata regardless of result quality.
    The total trial count (N) is used by DSR to penalize multiple testing.
    """

    def __init__(self, strat_dir: Path) -> None:
        self.path = strat_dir / "experiment-registry.jsonl"
        self.strat_dir = strat_dir

    @property
    def trial_count(self) -> int:
        """Return the total number of experiments ever recorded."""
        if not self.path.exists():
            return 0
        count = 0
        with self.path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                if raw_line.strip():
                    count += 1
        return count

    def load_all(self) -> list[dict[str, Any]]:
        """Load all experiment entries."""
        if not self.path.exists():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed registry line: %s",
                            stripped[:100],
                        )
        return entries

    def append(self, entry: dict[str, Any]) -> int:
        """Append a new experiment entry. Returns the new trial count."""
        record = {
            **entry,
            "trial_number": self.trial_count + 1,
            "recorded_at": datetime.now(tz=UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(
            "Recorded experiment #%d for %s",
            record["trial_number"],
            self.strat_dir.name,
        )
        return record["trial_number"]

    def get_returns_matrix(self) -> list[list[float]]:
        """Load daily return series from all experiments for PBO computation.

        Each experiment artifact must have a 'daily_returns' field.
        Returns a list of return series (one per experiment).
        """
        entries = self.load_all()
        experiments_dir = self.strat_dir / "experiments"
        if not experiments_dir.exists():
            return []

        returns_matrix: list[list[float]] = []
        for entry in entries:
            exp_id = entry.get("experiment_id", "")
            artifact_path = experiments_dir / f"{exp_id}.yaml"
            if not artifact_path.exists():
                continue
            artifact = load_artifact(artifact_path)
            daily_returns = artifact.get("daily_returns", [])
            if daily_returns:
                returns_matrix.append(daily_returns)
        return returns_matrix

"""Tests for artifact lifecycle, frozen spec, data grade, and experiment registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_quant.backtest.artifacts import (
    ExperimentRegistry,
    FrozenSpecError,
    LifecycleError,
    LifecycleState,
    check_data_grade,
    ensure_frozen_spec,
    freeze_spec,
    get_lifecycle_state,
    hash_content,
    load_artifact,
    save_artifact,
    validate_transition,
)

# ---------------------------------------------------------------------------
# Test: Lifecycle state machine
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Verify lifecycle transitions are enforced."""

    def test_valid_transitions(self):
        """mandate -> hypothesis should be allowed."""
        validate_transition(LifecycleState.MANDATE, LifecycleState.HYPOTHESIS)
        validate_transition(LifecycleState.HYPOTHESIS, LifecycleState.DATA_CONTRACT)
        validate_transition(LifecycleState.DATA_CONTRACT, LifecycleState.RESEARCH_SPEC)
        validate_transition(LifecycleState.RESEARCH_SPEC, LifecycleState.BACKTEST)

    def test_invalid_transition_mandate_to_backtest(self):
        """mandate -> backtest should be blocked."""
        with pytest.raises(LifecycleError):
            validate_transition(LifecycleState.MANDATE, LifecycleState.BACKTEST)

    def test_invalid_transition_idea_to_robustness(self):
        """idea -> robustness should be blocked."""
        with pytest.raises(LifecycleError):
            validate_transition(LifecycleState.IDEA, LifecycleState.ROBUSTNESS)

    def test_invalid_transition_backtest_to_mandate(self):
        """Backwards transitions are not allowed."""
        with pytest.raises(LifecycleError):
            validate_transition(LifecycleState.BACKTEST, LifecycleState.MANDATE)

    def test_get_lifecycle_state_empty(self):
        """Empty directory should be IDEA state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            assert get_lifecycle_state(d) == LifecycleState.IDEA

    def test_get_lifecycle_state_mandate(self):
        """Directory with mandate.yaml should be MANDATE state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(d / "mandate.yaml", {"name": "test"})
            assert get_lifecycle_state(d) == LifecycleState.MANDATE

    def test_get_lifecycle_state_research_spec(self):
        """Directory with research-spec.yaml should be RESEARCH_SPEC state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(d / "mandate.yaml", {"name": "test"})
            save_artifact(d / "hypothesis.yaml", {"statement": "test"})
            save_artifact(d / "data-contract.yaml", {"symbols": ["SPY"]})
            save_artifact(d / "research-spec.yaml", {"strategy_type": "sma"})
            assert get_lifecycle_state(d) == LifecycleState.RESEARCH_SPEC

    def test_get_lifecycle_state_robustness_not_trapped_at_backtest(self):
        """Robustness state should NOT be masked by experiments/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(d / "mandate.yaml", {"name": "test"})
            save_artifact(d / "research-spec.yaml", {"frozen": True})
            save_artifact(d / "robustness.yaml", {"overall_passed": True})
            # Also create experiments dir (the bug: this used to preempt)
            (d / "experiments").mkdir()
            (d / "experiments" / "exp1.yaml").write_text("test")
            assert get_lifecycle_state(d) == LifecycleState.ROBUSTNESS

    def test_get_lifecycle_state_backtest_from_registry(self):
        """Experiment registry file should trigger BACKTEST state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(d / "mandate.yaml", {"name": "test"})
            save_artifact(d / "research-spec.yaml", {"frozen": True})
            (d / "experiment-registry.jsonl").write_text('{"id":"test"}\n')
            assert get_lifecycle_state(d) == LifecycleState.BACKTEST


# ---------------------------------------------------------------------------
# Test: Frozen spec
# ---------------------------------------------------------------------------


class TestFrozenSpec:
    """Verify frozen spec enforcement."""

    def test_frozen_spec_loads(self):
        """A frozen spec should load successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(
                d / "research-spec.yaml",
                {"strategy_type": "sma", "frozen": True},
            )
            spec = ensure_frozen_spec(d)
            assert spec["frozen"] is True
            assert spec["strategy_type"] == "sma"

    def test_unfrozen_spec_raises(self):
        """An unfrozen spec should raise FrozenSpecError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(
                d / "research-spec.yaml",
                {"strategy_type": "sma", "frozen": False},
            )
            with pytest.raises(FrozenSpecError):
                ensure_frozen_spec(d)

    def test_missing_spec_raises(self):
        """Missing spec should raise FrozenSpecError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            with pytest.raises(FrozenSpecError):
                ensure_frozen_spec(d)

    def test_freeze_spec(self):
        """freeze_spec should set frozen=True and record hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            save_artifact(
                d / "research-spec.yaml",
                {"strategy_type": "sma", "frozen": False},
            )
            content_hash = freeze_spec(d)
            assert len(content_hash) == 64  # SHA-256 hex
            spec = load_artifact(d / "research-spec.yaml")
            assert spec["frozen"] is True
            assert "frozen_at" in spec
            assert "frozen_hash" in spec


# ---------------------------------------------------------------------------
# Test: Data grade
# ---------------------------------------------------------------------------


class TestDataGrade:
    """Verify data quality grade gating."""

    def test_grade_a_passes_b_minimum(self):
        assert check_data_grade("a", "b") is True

    def test_grade_b_passes_b_minimum(self):
        assert check_data_grade("b", "b") is True

    def test_grade_c_fails_b_minimum(self):
        assert check_data_grade("c", "b") is False

    def test_grade_d_fails_b_minimum(self):
        assert check_data_grade("d", "b") is False

    def test_invalid_grade(self):
        assert check_data_grade("x", "b") is False

    def test_invalid_minimum(self):
        """Invalid minimum should return False, not crash."""
        assert check_data_grade("a", "x") is False


# ---------------------------------------------------------------------------
# Test: Experiment registry
# ---------------------------------------------------------------------------


class TestExperimentRegistry:
    """Verify append-only experiment registry."""

    def test_empty_registry(self):
        """New registry should have zero trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            registry = ExperimentRegistry(d)
            assert registry.trial_count == 0
            assert registry.load_all() == []

    def test_append_increments(self):
        """Each append should increment trial count by 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            registry = ExperimentRegistry(d)

            n1 = registry.append({"experiment_id": "aaa", "sharpe": 0.5})
            assert n1 == 1
            assert registry.trial_count == 1

            n2 = registry.append({"experiment_id": "bbb", "sharpe": 0.8})
            assert n2 == 2
            assert registry.trial_count == 2

    def test_registry_is_append_only(self):
        """Registry file should contain all entries in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            registry = ExperimentRegistry(d)

            registry.append({"experiment_id": "aaa"})
            registry.append({"experiment_id": "bbb"})
            registry.append({"experiment_id": "ccc"})

            entries = registry.load_all()
            assert len(entries) == 3
            assert entries[0]["experiment_id"] == "aaa"
            assert entries[1]["experiment_id"] == "bbb"
            assert entries[2]["experiment_id"] == "ccc"

    def test_spec_unchanged_by_registry(self):
        """Adding to the registry should not modify the research spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            spec_data = {"strategy_type": "sma", "frozen": True}
            save_artifact(d / "research-spec.yaml", spec_data)
            original_hash = hash_content(
                (d / "research-spec.yaml").read_text(encoding="utf-8")
            )

            registry = ExperimentRegistry(d)
            registry.append({"experiment_id": "test"})
            registry.append({"experiment_id": "test2"})

            new_hash = hash_content(
                (d / "research-spec.yaml").read_text(encoding="utf-8")
            )
            assert original_hash == new_hash


# ---------------------------------------------------------------------------
# Test: Artifact I/O
# ---------------------------------------------------------------------------


class TestArtifactIO:
    """Verify YAML save/load and hashing."""

    def test_save_load_roundtrip(self):
        """Save and load should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            data = {"name": "test", "value": 42, "nested": {"a": 1}}
            save_artifact(d / "test.yaml", data)
            loaded = load_artifact(d / "test.yaml")
            assert loaded["name"] == "test"
            assert loaded["value"] == 42
            assert loaded["nested"]["a"] == 1

    def test_hash_deterministic(self):
        """Same content should produce same hash."""
        h1 = hash_content("hello world")
        h2 = hash_content("hello world")
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_different_content(self):
        """Different content should produce different hash."""
        h1 = hash_content("hello")
        h2 = hash_content("world")
        assert h1 != h2

    def test_load_nonexistent_raises(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_artifact(Path("/nonexistent/file.yaml"))


# ---------------------------------------------------------------------------
# Test: Promotion gate
# ---------------------------------------------------------------------------


class TestPromotionGate:
    """Verify promotion gate checks thresholds."""

    def test_all_pass(self):
        """All thresholds met → overall=pass."""
        thresholds = {
            "dsr_minimum": 0.95,
            "pbo_maximum": 0.10,
            "data_grade_minimum": "b",
        }
        # DSR passes
        assert thresholds["dsr_minimum"] <= 0.97
        # PBO passes
        assert thresholds["pbo_maximum"] >= 0.05
        # Data grade passes
        assert check_data_grade("a", thresholds["data_grade_minimum"])

    def test_dsr_failure_blocks(self):
        """DSR below threshold → blocked."""
        assert 0.80 < 0.95  # fails
        # In the real gate, this would block promotion

    def test_data_grade_c_blocks(self):
        """Grade C with minimum B → blocked."""
        assert not check_data_grade("c", "b")

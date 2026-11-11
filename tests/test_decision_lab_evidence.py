"""Evidence classification, source integrity and phase-sensitivity regression tests."""

import hashlib
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from llm_quant.decision_lab.audit import phase_diagnostics
from llm_quant.decision_lab.data import Contract, Snapshot
from llm_quant.decision_lab.evidence import (
    assess_report, audit_track_d_summary, changed_reports, git_blob_sha, main, source_record,
)
from llm_quant.decision_lab.models import Simulation

TABLE = """| Strategy | Sharpe | MaxDD | DSR | CPCV OOS | Perturb % | Shuffled p | Verdict |
|---|---|---|---|---|---|---|---|
| a-sprint | 1.078 | 10.6% | 0.969 | 1.091 ratio | 56% | 0.0000 | ADVANCE |
| b-sprint | 0.731 | 20.6% | 0.921 | 1.091 ratio | 48% | 0.527 | KILL |
| c-sprint | 0.778 | 10.5% | 0.931 | 0.664 ratio | 45% | 0.0030 | KILL |
"""


def test_recorded_pass_is_never_validated_profitability():
    result = audit_track_d_summary(TABLE)
    assert result["recorded_advances"] == 1
    assert result["recorded_failures"] == 2
    assert result["rows"][0]["reported_verdict"] == "RECORDED_ADVANCE_NOT_VALIDATED"
    assert len(result["rows"][0]["rechecked_thresholds"]) == 5
    assert result["rows"][0]["cpcv_as_reported_ambiguous"] == "1.091 ratio"
    assert result["rows"][1]["failed_rechecked_thresholds"] == [
        "sharpe_at_least_0_80", "shuffled_p_below_0_05",
    ]


def test_contradictory_advance_is_flagged():
    result = audit_track_d_summary(TABLE.replace("0.0030 | KILL", "0.0030 | ADVANCE"))
    assert result["rows"][-1]["advance_contradicted_by_checked_metrics"]


@pytest.mark.parametrize("text", ["", TABLE.replace("Strategy", "Model"),
    TABLE.replace("1.078", "NaN"), TABLE.replace("0.969", "1.5"),
    TABLE.replace("b-sprint", "a-sprint"), TABLE.replace("ADVANCE", "UNKNOWN"),
    TABLE.replace("1.091 ratio | 56%", "56%")])
def test_invalid_summary_fails_closed(text):
    with pytest.raises(ValueError):
        audit_track_d_summary(text)


@pytest.mark.parametrize(("old", "new", "failed"), [
    ("1.078", "0.800", None), ("1.078", "0.799", "sharpe_at_least_0_80"),
    ("10.6%", "40%", "drawdown_below_0_40"),
    ("0.0000", "0.0500", "shuffled_p_below_0_05"),
    ("56%", "40%", None), ("0.969", "0.900", None),
])
def test_exact_gate_boundaries(old, new, failed):
    row = audit_track_d_summary(TABLE.replace(old, new))["rows"][0]
    if failed:
        assert failed in row["failed_rechecked_thresholds"]
    else:
        assert not row["failed_rechecked_thresholds"]


@pytest.mark.parametrize("marker", ["No portfolio snapshot for this date.",
    "No snapshot data available.", "No portfolio snapshots found."])
def test_missing_evidence_overrides_zeros_and_pass_footer(marker):
    report = marker + "\n| NAV | $100,000.00 |\n| Total Return | 0.0% |\nHash chain verified: [PASS]"
    result = assess_report(report)
    assert result["status"] == "MISSING_PORTFOLIO_EVIDENCE"
    assert not result["publication_allowed"]
    assert result["return_if_missing"] is None
    assert not result["performance_verified"]


def test_reported_nav_is_not_independent_verification():
    result = assess_report("| NAV | $100,000.00 |\nNo trades executed today.")
    assert result["publication_allowed"]
    assert not result["performance_verified"]


@pytest.mark.parametrize("text", ["", "| NAV | $0.00 |", "| NAV | $nan |",
                                  "| NAV | $inf |", "Hash chain verified: [PASS]"])
def test_unknown_or_invalid_nav_does_not_pass(text):
    assert not assess_report(text)["publication_allowed"]


def test_source_hash_and_mutation(tmp_path):
    (tmp_path / "source.md").write_bytes(b"original\n")
    expected = git_blob_sha(b"original\n")
    assert source_record(tmp_path, "source.md", expected)["sha256"] == hashlib.sha256(b"original\n").hexdigest()
    (tmp_path / "source.md").write_bytes(b"edited\n")
    with pytest.raises(ValueError, match="Source changed"):
        source_record(tmp_path, "source.md", expected)
    with pytest.raises(ValueError, match="inside"):
        source_record(tmp_path, "../outside", expected)


def test_git_empty_blob_identity():
    assert git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_evidence_cli_rejects_missing_report(tmp_path, monkeypatch):
    path = tmp_path / "daily.md"
    path.write_text("No portfolio snapshot for this date.")
    monkeypatch.setattr("sys.argv", ["evidence", str(path)])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2


def fixture_history():
    rng = np.random.default_rng(17)
    data = np.r_[100., 100 * np.cumprod(1 + rng.normal(.003, .025, 120))]
    dates = tuple(date(1990 + i // 4, 1 + 3 * (i % 4), 1).isoformat() for i in range(121))
    contract = Contract("TEST", "Synthetic", "https://example.invalid/data",
        "https://example.invalid/terms", "Fictitious", "INDEX", "Q", "real_price_index",
        "2020-01-01T00:00:00+00:00", "synthetic", "synthetic", "0" * 64)
    return Snapshot(contract, dates, tuple(data))


def test_all_phases_are_exposed_not_selected_or_pooled():
    rows = phase_diagnostics(fixture_history(), Simulation(horizon=4, paths=100), 80)
    assert [row["offset_periods"] for row in rows] == [0, 1, 2, 3]
    assert [row["folds"] for row in rows] == [10, 9, 9, 9]
    assert [row["first_origin_label"] for row in rows] == [
        "2010-01-01", "2010-04-01", "2010-07-01", "2010-10-01",
    ]
    for row in rows:
        assert row["model_minus_baseline_brier"] == pytest.approx(row["model_brier"] - row["baseline_brier"])
        assert 0 <= row["coverage_95_count"] <= row["folds"]


@pytest.mark.parametrize("min_train", [True, 10, 118])
def test_phase_training_requirements(min_train):
    with pytest.raises(ValueError):
        phase_diagnostics(fixture_history(), Simulation(horizon=4, paths=100), min_train)


def test_phase_work_budget_and_repeatability():
    snapshot = fixture_history()
    config = Simulation(horizon=4, paths=100)
    assert phase_diagnostics(snapshot, config, 80) == phase_diagnostics(snapshot, config, 80)
    with pytest.raises(ValueError, match="limited"):
        phase_diagnostics(snapshot, replace(config, horizon=13), 80)


def test_changed_reports_uses_no_shell_and_excludes_other_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "new file.md").write_text("No snapshot data available.")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert "shell" not in kwargs
        return SimpleNamespace(stdout=b"reports/new file.md\0")
    monkeypatch.setattr("llm_quant.decision_lab.evidence.subprocess.run", run)
    assert changed_reports(tmp_path) == [tmp_path / "reports" / "new file.md"]
    assert len(calls) == 2


def test_report_workflow_no_empty_database_or_swallowed_errors():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/reports.yml").read_text()
    assert "init_schema" not in workflow
    assert "|| true" not in workflow
    assert "[ ! -s data/llm_quant.duckdb ]" in workflow
    assert "--changed-reports" in workflow
    assert workflow.index("Require an existing evidence database") < workflow.index("Install dependencies")
    assert workflow.index("--changed-reports") < workflow.index("git add reports/")
    assert "REQUESTED_DATE:" in workflow


def test_focused_workflow_keeps_read_only_permissions_and_no_schedule():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/decision-lab.yml").read_text()
    assert "contents: read" in workflow
    assert "schedule:" not in workflow
    assert "timeout-minutes: 3" in workflow
    assert "tests/test_decision_lab*.py" in workflow

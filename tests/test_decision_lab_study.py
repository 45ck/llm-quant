"""Offline regression checks for capture receipts, historical studies and providers."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm_quant.decision_lab import providers
from llm_quant.decision_lab.data import Contract, Snapshot, load_snapshot
from llm_quant.decision_lab.fred_table import import_table_export, parse_table
from llm_quant.decision_lab.models import Simulation, summarize, validate_returns
from llm_quant.decision_lab.study import audit_snapshot, historical_study
from llm_quant.decision_lab.validation import calibration_bins, walk_forward


def table_html() -> bytes:
    rows = [
        ("Series ID", "QAUR628BIS"),
        ("Source", "Bank for International Settlements"),
        ("Frequency", "Quarterly"), ("Units", "Index 2010=100"),
        ("Date Range", "2020-01-01 to 2020-04-01"),
        ("Last Updated", "2020-08-01"), ("DATE", "VALUE"),
        ("2020-01-01", "100"), ("2020-04-01", "102"),
    ]
    return ("<table>" + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>"
                               for a, b in rows) + "</table>").encode()


def make_export(tmp_path: Path) -> tuple[Path, Path]:
    raw = b"observation_date,QAUR628BIS\n2020-01-01,100\n2020-04-01,102\n"
    receipt = {
        "series_id": "QAUR628BIS", "acquisition": "browser_table_export",
        "source_url": "https://fred.stlouisfed.org/data/QAUR628BIS",
        "units": "Index 2010=100", "frequency": "Quarterly",
        "scope": "complete_series", "source_last_updated": "2020-08-01",
        "captured_at": "2020-09-01T00:00:00+00:00",
        "export_sha256": hashlib.sha256(raw).hexdigest(), "row_count": 2,
        "first_date": "2020-01-01", "last_date": "2020-04-01",
        "first_value": 100.0, "last_value": 102.0,
    }
    csv_path = tmp_path / "export.csv"
    receipt_path = tmp_path / "receipt.json"
    csv_path.write_bytes(raw)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return csv_path, receipt_path


def test_html_source_parser_has_exact_metadata_and_values() -> None:
    result = parse_table(table_html(), "QAUR628BIS")
    assert result.dates == ("2020-01-01", "2020-04-01")
    assert result.values == (100.0, 102.0)
    assert result.updated == "2020-08-01"


@pytest.mark.parametrize(("before", "after"), [
    (b"QAUR628BIS", b"QAUN628BIS"), (b"Quarterly", b"Monthly"),
    (b"Index 2010=100", b"Index 2020=100"),
    (b"Bank for International Settlements", b"Another source"),
    (b"Last Updated", b"Unknown field"),
    (b"2020-01-01 to 2020-04-01", b"2020-01-01 to 2020-07-01"),
    (b"<td>102</td>", b"<td>.</td>"),
])
def test_html_source_drift_is_rejected(before: bytes, after: bytes) -> None:
    with pytest.raises(ValueError):
        parse_table(table_html().replace(before, after), "QAUR628BIS")


def test_error_page_and_script_rows_are_not_observations() -> None:
    with pytest.raises(ValueError):
        parse_table(b"<h1>404</h1>", "QAUR628BIS")
    hostile = b"<script><tr><td>2020-07-01</td><td>99999</td></tr></script>"
    result = parse_table(table_html() + hostile, "QAUR628BIS")
    assert len(result.dates) == 2


def test_capture_import_retains_source_and_receipt(tmp_path: Path) -> None:
    csv_path, receipt_path = make_export(tmp_path)
    path = import_table_export(csv_path, receipt_path, tmp_path / "snapshots")
    snapshot = load_snapshot(path)
    assert snapshot.contract.evidence == "observed"
    assert "browser table export" in snapshot.contract.provider
    assert len(list((path.parent / "receipts").iterdir())) == 1
    loaded, audit = audit_snapshot(path, date(2021, 1, 1), 100)
    assert loaded == snapshot
    assert audit["freshness_gate"] == "STALE"
    assert audit["retained_source_verified"] is True
    assert audit["current_forecast_authorized"] is False
    with pytest.raises(ValueError, match="Stale"):
        snapshot.check_as_of(date(2021, 1, 1), 100)


@pytest.mark.parametrize(("field", "value"), [
    ("series_id", "UNKNOWN"), ("acquisition", "original_http_csv"),
    ("source_url", "https://example.invalid"), ("units", "AUD"),
    ("frequency", "Daily"), ("scope", "selected_rows"),
    ("source_last_updated", ""), ("captured_at", "2020-09-01T00:00:00"),
    ("captured_at", "2999-01-01T00:00:00+00:00"), ("export_sha256", "0" * 64),
    ("row_count", 3), ("first_date", "2019-01-01"),
    ("last_date", "2020-07-01"), ("first_value", 90), ("last_value", 101),
])
def test_capture_receipt_mismatch_rejected(tmp_path: Path, field: str, value: object) -> None:
    csv_path, receipt_path = make_export(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        import_table_export(csv_path, receipt_path, tmp_path / "snapshots")
    assert not (tmp_path / "snapshots").exists()


def test_observed_snapshots_require_intact_source_payload(tmp_path: Path) -> None:
    csv_path, receipt_path = make_export(tmp_path)
    path = import_table_export(csv_path, receipt_path, tmp_path / "snapshots")
    snapshot = load_snapshot(path)
    raw = path.parent / "raw" / f"{snapshot.contract.raw_sha256}.raw"
    raw.write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_snapshot(path, date(2021, 1, 1), 100)
    raw.unlink()
    with pytest.raises(ValueError, match="retained source"):
        audit_snapshot(path, date(2021, 1, 1), 100)


def test_capture_origin_and_negative_freshness_rejected(tmp_path: Path) -> None:
    csv_path, receipt_path = make_export(tmp_path)
    path = import_table_export(csv_path, receipt_path, tmp_path / "snapshots")
    with pytest.raises(ValueError, match="Future"):
        audit_snapshot(path, date(2020, 8, 1), 100)
    with pytest.raises(ValueError, match="nonnegative"):
        audit_snapshot(path, date(2021, 1, 1), -1)


def synthetic_history() -> Snapshot:
    rng = np.random.default_rng(11)
    values = tuple(float(value) for value in
                   np.r_[100.0, 100 * np.cumprod(1 + rng.normal(0.002, 0.025, 120))])
    dates = tuple(f"{1990 + i // 4}-{1 + (i % 4) * 3:02d}-01" for i in range(121))
    contract = Contract(
        "SYNTHETIC", "fixture", "https://example.invalid", "https://example.invalid",
        "Fictitious", "INDEX", "Q", "real_price_index", "2020-01-01T00:00:00+00:00",
        "synthetic", "synthetic", hashlib.sha256(b"fixture").hexdigest(),
    )
    return Snapshot(contract, dates, values)


def test_historical_study_preserves_units_provenance_and_all_grid_cases() -> None:
    snapshot = synthetic_history()
    config = Simulation(horizon=4, paths=100, block_length=4)
    result = historical_study(snapshot, config, blocks=[1, 4], windows=[40, 120], min_train=80)
    assert result["status"] == "SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE"
    assert result["native_frequency"] == "Q"
    assert result["snapshot_sha256"] == snapshot.digest()
    assert len(result["sensitivity"]) == 4
    assert result["validation"]["fold_count"] == 10
    assert result["validation"]["folds"][0]["origin_observation_label"] == snapshot.dates[80]
    assert result["validation"]["folds"][-1]["outcome_observation_label"] == snapshot.dates[120]


@pytest.mark.parametrize(("blocks", "windows", "min_train"), [
    ([], [40], 80), ([1, 1], [40], 80), ([1], [], 80),
    ([1], [121], 80), ([1], [20], 80), ([1], [40, 40], 80),
    ([41], [40], 80), ([1], [40], 119), ([1], [40], 4),
])
def test_invalid_study_specs(blocks: list, windows: list, min_train: int) -> None:
    with pytest.raises(ValueError):
        historical_study(synthetic_history(), Simulation(horizon=4, paths=100),
                         blocks=blocks, windows=windows, min_train=min_train)


def test_validation_intervals_are_training_only_and_have_proper_scores() -> None:
    data = np.random.default_rng(12).normal(0, .03, (120, 1))
    config = Simulation(horizon=4, paths=100, block_length=4)
    before = walk_forward(data, np.ones(1), config, min_train=80)
    changed = data.copy()
    changed[80:] = 0.25
    after = walk_forward(changed, np.ones(1), config, min_train=80)
    for key in ("0.5", "0.8", "0.95"):
        first = before["folds"][0]["central_intervals"][key]
        second = after["folds"][0]["central_intervals"][key]
        assert first["lower"] == second["lower"]
        assert first["upper"] == second["upper"]
        assert first["interval_score"] >= first["upper"] - first["lower"]
    assert sum(item["count"] for item in before["reliability_bins"]) == before["fold_count"]
    assert before["loss_event_count"] == sum(fold["loss_observed"] for fold in before["folds"])


def test_reliability_bins_handle_zero_and_one_without_double_counting() -> None:
    folds = [{"predicted_loss_probability": p, "loss_observed": int(p > .5)}
             for p in [0., .2, .4, .6, .8, 1.]]
    bins = calibration_bins(folds)
    assert [item["count"] for item in bins] == [1, 1, 1, 1, 2]
    assert bins[-1]["observed_frequency"] == 1
    assert calibration_bins([])[0]["observed_frequency"] is None
    with pytest.raises(ValueError):
        calibration_bins(folds, 0)


def test_empty_wealth_is_rejected_and_weights_use_absolute_tolerance() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        summarize(np.empty((0, 2)))
    with pytest.raises(ValueError, match="sum"):
        validate_returns(np.zeros((100, 1)), np.array([1.0000000005]))


def test_network_failure_does_not_create_fake_observations(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("source unavailable")
    monkeypatch.setattr(providers, "urlopen", unavailable)
    with pytest.raises(OSError, match="source unavailable"):
        providers.fetch_fred("QAUR628BIS", tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_yahoo_requests_missing_rows_and_no_silent_repairs(monkeypatch: pytest.MonkeyPatch,
                                                          tmp_path: Path) -> None:
    calls = {}
    class Index(list):
        is_unique = True
    class Frame:
        columns = ["Adj Close"]
        empty = False
        index = Index([datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC)])
        def __contains__(self, key: str) -> bool:
            return key == "Adj Close"
        def __getitem__(self, key: str) -> list:
            return [100., 101.]
        def to_csv(self) -> str:
            return "Date,Adj Close\n2020-01-02,100\n2020-01-03,101\n"
    class Ticker:
        def history(self, **kwargs: object) -> Frame:
            calls.update(kwargs)
            return Frame()
        def get_history_metadata(self) -> dict:
            return {"currency": "AUD"}
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: Ticker()))
    path = providers.fetch_yahoo("VAS.AX", "2020-01-01", "2020-01-04", tmp_path)
    assert load_snapshot(path).contract.currency == "AUD"
    assert calls["keepna"] is True
    assert calls["repair"] is False
    assert calls["raise_errors"] is True
    assert calls["auto_adjust"] is False

"""Tests for MarketDataRecorder using mock GammaClient responses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_quant.arb.gamma_client import GammaClient, Market
from llm_quant.arb.research.recorder import (
    MarketDataRecorder,
    _compute_days_to_resolution,
    _generate_snapshot_id,
    _init_experiment_schema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_market(
    market_id: str = "123",
    question: str = "Will X happen?",
    yes_price: float = 0.65,
    no_price: float = 0.37,
    *,
    active: bool = True,
    is_negrisk: bool = False,
    category: str = "politics",
    volume_24h: float = 50000.0,
    open_interest: float = 120000.0,
    end_date: str | None = "2026-12-31T23:59:59Z",
) -> dict:
    """Build a raw market dict mimicking Gamma API response."""
    return {
        "id": market_id,
        "conditionId": f"cond_{market_id}",
        "slug": f"will-x-happen-{market_id}",
        "question": question,
        "active": active,
        "negRisk": is_negrisk,
        "isNegRisk": is_negrisk,
        "category": category,
        "endDate": end_date,
        "outcomes": '["Yes","No"]',
        "outcomePrices": f'["{yes_price}","{no_price}"]',
        "tokens": [
            {"outcome": "Yes", "price": str(yes_price)},
            {"outcome": "No", "price": str(no_price)},
        ],
        "volumeNum24hr": str(volume_24h),
        "openInterest": str(open_interest),
    }


def _make_mock_client(
    raw_markets: list[dict] | None = None,
    fetch_raises: Exception | None = None,
) -> GammaClient:
    """Create a GammaClient mock that returns canned data."""
    client = MagicMock(spec=GammaClient)

    if fetch_raises:
        client.fetch_all_active_markets.side_effect = fetch_raises
    else:
        raw = raw_markets if raw_markets is not None else [_make_raw_market()]
        client.fetch_all_active_markets.return_value = raw

    # parse_all_markets delegates to the real static method
    def _parse_all(raw_list: list[dict]) -> list[Market]:
        markets = []
        for r in raw_list:
            m = GammaClient.parse_market(r)
            if m and m.question:
                markets.append(m)
        return markets

    client.parse_all_markets.side_effect = _parse_all
    return client


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_recorder.duckdb"


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_generate_snapshot_id_format(self):
        from datetime import UTC, datetime

        ts = datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC)
        sid = _generate_snapshot_id("live_scan", ts)
        assert sid.startswith("live_scan_20260406T120000Z_")
        # 8-char hex suffix
        suffix = sid.split("_")[-1]
        assert len(suffix) == 8
        int(suffix, 16)  # must be valid hex

    def test_days_to_resolution_iso(self):
        from datetime import UTC, datetime

        now = datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)
        result = _compute_days_to_resolution("2026-04-16T00:00:00Z", now)
        assert result is not None
        assert abs(result - 10.0) < 0.01

    def test_days_to_resolution_date_only(self):
        from datetime import UTC, datetime

        now = datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)
        result = _compute_days_to_resolution("2026-04-11", now)
        assert result is not None
        assert abs(result - 5.0) < 0.01

    def test_days_to_resolution_past(self):
        from datetime import UTC, datetime

        now = datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)
        result = _compute_days_to_resolution("2026-01-01T00:00:00Z", now)
        assert result == 0.0  # clamped to zero

    def test_days_to_resolution_none(self):
        from datetime import UTC, datetime

        now = datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)
        assert _compute_days_to_resolution(None, now) is None

    def test_days_to_resolution_invalid(self):
        from datetime import UTC, datetime

        now = datetime(2026, 4, 6, 0, 0, 0, tzinfo=UTC)
        assert _compute_days_to_resolution("not-a-date", now) is None


# ---------------------------------------------------------------------------
# Integration tests — MarketDataRecorder
# ---------------------------------------------------------------------------


class TestRecorderBasic:
    def test_record_single_snapshot(self, tmp_db: Path):
        client = _make_mock_client()
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)

        sid = recorder.record_snapshot()
        assert sid is not None
        assert recorder.snapshot_count() == 1

    def test_record_multiple_snapshots(self, tmp_db: Path):
        client = _make_mock_client()
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)

        sid1 = recorder.record_snapshot()
        sid2 = recorder.record_snapshot()
        assert sid1 != sid2
        assert recorder.snapshot_count() == 2

    def test_get_snapshot_returns_header_and_markets(self, tmp_db: Path):
        raw = [
            _make_raw_market("m1", "Will A happen?", 0.60, 0.42),
            _make_raw_market("m2", "Will B happen?", 0.80, 0.22, is_negrisk=True),
        ]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)

        sid = recorder.record_snapshot()
        snap = recorder.get_snapshot(sid)

        assert snap is not None
        assert "header" in snap
        assert "markets" in snap
        assert snap["header"]["snapshot_id"] == sid
        assert len(snap["markets"]) == 2

    def test_market_state_fields(self, tmp_db: Path):
        raw = [
            _make_raw_market(
                "m1",
                "Will inflation rise?",
                yes_price=0.55,
                no_price=0.47,
                is_negrisk=False,
                category="finance",
                volume_24h=100000.0,
                open_interest=250000.0,
            ),
        ]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot()

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        m = snap["markets"][0]

        assert m["market_id"] == "m1"
        assert m["question"] == "Will inflation rise?"
        assert m["category"] == "finance"
        assert m["yes_price"] == 0.55
        assert m["no_price"] == 0.47
        assert abs(m["spread"] - 0.02) < 1e-6  # 0.55 + 0.47 - 1.0
        assert m["volume_24h"] == 100000.0
        assert m["open_interest"] == 250000.0
        assert m["is_negrisk"] is False
        assert m["active"] is True

    def test_negrisk_market_flagged(self, tmp_db: Path):
        raw = [_make_raw_market("ng1", "Who will win?", 0.30, 0.72, is_negrisk=True)]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot()

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["markets"][0]["is_negrisk"] is True

    def test_list_snapshots(self, tmp_db: Path):
        client = _make_mock_client()
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        recorder.record_snapshot()
        recorder.record_snapshot()
        recorder.record_snapshot()

        result = recorder.list_snapshots(limit=2)
        assert len(result) == 2

        result_all = recorder.list_snapshots(limit=100)
        assert len(result_all) == 3

    def test_get_nonexistent_snapshot(self, tmp_db: Path):
        client = _make_mock_client()
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        assert recorder.get_snapshot("does_not_exist") is None

    def test_snapshot_count_empty(self, tmp_db: Path):
        client = _make_mock_client()
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        assert recorder.snapshot_count() == 0


class TestRecorderEdgeCases:
    def test_empty_market_list(self, tmp_db: Path):
        client = _make_mock_client(raw_markets=[])
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)

        sid = recorder.record_snapshot()
        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["header"]["active_market_count"] == 0
        assert snap["header"]["data_quality"] == "degraded"
        assert len(snap["markets"]) == 0

    def test_api_failure_records_degraded_snapshot(self, tmp_db: Path):
        client = _make_mock_client(fetch_raises=ConnectionError("timeout"))
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)

        sid = recorder.record_snapshot()
        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["header"]["data_quality"] == "degraded"
        assert snap["header"]["notes"] == "API fetch failed"
        assert len(snap["markets"]) == 0

    def test_market_without_end_date(self, tmp_db: Path):
        raw = [_make_raw_market("no_end", "Perpetual?", 0.50, 0.50, end_date=None)]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot()

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        m = snap["markets"][0]
        assert m["days_to_resolution"] is None

    def test_partial_quality_few_markets(self, tmp_db: Path):
        # Fewer than 10 markets → partial quality
        raw = [_make_raw_market(f"m{i}", f"Q{i}?", 0.50, 0.50) for i in range(5)]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot()

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["header"]["data_quality"] == "partial"

    def test_complete_quality_many_markets(self, tmp_db: Path):
        raw = [_make_raw_market(f"m{i}", f"Q{i}?", 0.50, 0.50) for i in range(15)]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot()

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["header"]["data_quality"] == "complete"

    def test_custom_source(self, tmp_db: Path):
        client = _make_mock_client()
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot(source="historical_replay")

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["header"]["source"] == "historical_replay"
        assert sid.startswith("historical_replay_")

    def test_volume_aggregation(self, tmp_db: Path):
        raw = [
            _make_raw_market("m1", "Q1?", 0.50, 0.50, volume_24h=10000.0),
            _make_raw_market("m2", "Q2?", 0.50, 0.50, volume_24h=20000.0),
            _make_raw_market("m3", "Q3?", 0.50, 0.50, volume_24h=30000.0),
        ]
        client = _make_mock_client(raw_markets=raw)
        recorder = MarketDataRecorder(db_path=tmp_db, client=client)
        sid = recorder.record_snapshot()

        snap = recorder.get_snapshot(sid)
        assert snap is not None
        assert snap["header"]["total_volume_24h"] == 60000.0


class TestRecorderCLI:
    """Test the CLI script entry point."""

    def test_cli_main_success(self, tmp_db: Path):
        from scripts.run_pm_recorder import main

        with (
            patch("llm_quant.arb.gamma_client.GammaClient") as mock_gc_cls,
            patch("llm_quant.arb.research.recorder.MarketDataRecorder") as mock_rec_cls,
        ):
            mock_client = MagicMock()
            mock_gc_cls.return_value = mock_client

            mock_recorder = MagicMock()
            mock_recorder.record_snapshot.return_value = "test_snap_001"
            mock_recorder.get_snapshot.return_value = {
                "header": {
                    "snapshot_id": "test_snap_001",
                    "timestamp": "2026-04-06T12:00:00Z",
                    "source": "live_scan",
                    "data_quality": "complete",
                    "scan_duration_ms": 1200,
                    "total_volume_24h": 50000.0,
                },
                "markets": [
                    {"is_negrisk": True, "spread": -0.03},
                    {"is_negrisk": False, "spread": 0.02},
                ],
            }
            mock_recorder.snapshot_count.return_value = 5
            mock_rec_cls.return_value = mock_recorder

            ret = main(["--db-path", str(tmp_db)])
            assert ret == 0

    def test_cli_main_snapshot_not_found(self, tmp_db: Path):
        from scripts.run_pm_recorder import main

        with (
            patch("llm_quant.arb.gamma_client.GammaClient") as mock_gc_cls,
            patch("llm_quant.arb.research.recorder.MarketDataRecorder") as mock_rec_cls,
        ):
            mock_gc_cls.return_value = MagicMock()
            mock_recorder = MagicMock()
            mock_recorder.record_snapshot.return_value = "gone"
            mock_recorder.get_snapshot.return_value = None
            mock_rec_cls.return_value = mock_recorder

            ret = main(["--db-path", str(tmp_db)])
            assert ret == 1


class TestSchemaInit:
    """Test that DDL is idempotent."""

    def test_init_twice(self, tmp_db: Path):
        import duckdb

        conn = duckdb.connect(str(tmp_db))
        _init_experiment_schema(conn)
        _init_experiment_schema(conn)  # must not raise

        # Verify tables exist
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        assert "n_experiment_snapshots" in tables
        assert "n_experiment_market_states" in tables
        conn.close()

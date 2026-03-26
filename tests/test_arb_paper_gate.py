"""Tests for the PM arb 30-day paper validation gate."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from llm_quant.arb.paper_gate import GateResult, PaperArbGate, PaperGateReport
from llm_quant.arb.schema import init_arb_schema

# ---------------------------------------------------------------------------
# Helpers — in-memory DB builder
# ---------------------------------------------------------------------------


def _make_db() -> duckdb.DuckDBPyConnection:
    """Create a fresh in-memory DuckDB with the arb schema."""
    conn = duckdb.connect(":memory:")
    init_arb_schema(conn)
    return conn


def _insert_scan(
    conn: duckdb.DuckDBPyConnection,
    started_at: datetime,
    opps_found: int,
    source: str = "kalshi",
) -> None:
    """Insert a single scan-log row."""
    conn.execute(
        """
        INSERT INTO pm_scan_log
        (scan_id, scan_type, source, markets_scanned, conditions_scanned,
         opps_found, pairs_detected, duration_secs, started_at, completed_at, error)
        VALUES (?, 'negrisk', ?, 10, 30, ?, 0, 1.5, ?, ?, NULL)
        """,
        [
            str(uuid.uuid4()),
            source,
            opps_found,
            started_at.isoformat(),
            (started_at + timedelta(seconds=2)).isoformat(),
        ],
    )


def _insert_opp(
    conn: duckdb.DuckDBPyConnection,
    total_volume: float,
    source: str = "kalshi",
) -> None:
    """Insert a single opportunity row."""
    conn.execute(
        """
        INSERT INTO pm_arb_opportunities
        (opp_id, arb_type, source, market_id, condition_ids,
         spread_pct, net_spread_pct, kelly_fraction, total_volume,
         detected_at, status, notes)
        VALUES (?, 'negrisk_buy_yes', ?, 'mkt-1', ['c1','c2'],
                0.08, 0.05, 0.047, ?, NOW(), 'open', '')
        """,
        [str(uuid.uuid4()), source, total_volume],
    )


def _make_gate(conn: duckdb.DuckDBPyConnection, source: str = "kalshi") -> PaperArbGate:
    """Return a PaperArbGate whose internal connection is the given in-memory conn."""
    gate = PaperArbGate(db_path=":memory:", source=source)
    # Bypass lazy-init: inject the already-initialized connection directly.
    gate._conn = conn
    return gate


# ---------------------------------------------------------------------------
# Test 1 — insufficient days → Days Elapsed gate fails → CONTINUE_PAPER
# ---------------------------------------------------------------------------


def test_gate_insufficient_days():
    """5 days of data with good quality gates → CONTINUE_PAPER, not PROMOTE."""
    conn = _make_db()

    base = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    # 10 scans over 5 days — 8 have opps (80% persistence, above 50% threshold)
    for i in range(10):
        dt = base + timedelta(hours=i * 12)
        opps = 1 if i < 8 else 0
        _insert_scan(conn, dt, opps)

    # Good volume — all opps fillable, capacity fine
    for _ in range(8):
        _insert_opp(conn, total_volume=50_000.0)

    gate = _make_gate(conn)
    report = gate.run_gate()

    # Days elapsed: 5 days < 30 → Days gate fails
    days_gate = next(g for g in report.gates if g.gate_name == "Days Elapsed")
    assert not days_gate.passed
    assert days_gate.value == pytest.approx(4.5, abs=1)  # 0-indexed from base

    # Quality gates should pass
    persistence = next(g for g in report.gates if g.gate_name == "Persistence")
    fill = next(g for g in report.gates if g.gate_name == "Fill Rate")
    capacity = next(g for g in report.gates if g.gate_name == "Capacity")
    assert persistence.passed
    assert fill.passed
    assert capacity.passed

    assert report.recommendation == "CONTINUE_PAPER"
    assert not report.overall_pass


# ---------------------------------------------------------------------------
# Test 2 — low persistence → Persistence gate fails → REJECT
# ---------------------------------------------------------------------------


def test_gate_low_persistence():
    """2/20 scans had opps (10% persistence < 50% threshold) → REJECT."""
    conn = _make_db()

    base = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
    # 20 scans over 35+ days — only 2 have opps
    for i in range(20):
        dt = base + timedelta(days=i * 2)  # every 2 days → 38-day window
        opps = 1 if i < 2 else 0
        _insert_scan(conn, dt, opps)

    # Insert 2 opps with good volumes
    for _ in range(2):
        _insert_opp(conn, total_volume=20_000.0)

    gate = _make_gate(conn)
    report = gate.run_gate()

    persistence = next(g for g in report.gates if g.gate_name == "Persistence")
    assert not persistence.passed
    assert persistence.value == pytest.approx(2 / 20)

    days_gate = next(g for g in report.gates if g.gate_name == "Days Elapsed")
    assert days_gate.passed  # 38 days >= 30

    assert report.recommendation == "REJECT"
    assert not report.overall_pass


# ---------------------------------------------------------------------------
# Test 3 — all gates pass → PROMOTE
# ---------------------------------------------------------------------------


def test_gate_all_pass():
    """35 days of scans, 60% with opps, high volume → all gates pass → PROMOTE."""
    conn = _make_db()

    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    # 35 daily scans — 21 have opps (60% persistence, above 50%)
    for i in range(35):
        dt = base + timedelta(days=i)
        opps = 1 if i < 21 else 0
        _insert_scan(conn, dt, opps)

    # 21 opps with very high volume → capacity and fill rate both pass
    for _ in range(21):
        _insert_opp(conn, total_volume=100_000.0)

    gate = _make_gate(conn)
    report = gate.run_gate()

    assert report.overall_pass
    assert report.recommendation == "PROMOTE"
    assert report.total_scans == 35
    assert report.total_opps_detected == 21

    # Verify each gate individually
    for g in report.gates:
        assert g.passed, f"Expected gate '{g.gate_name}' to pass but got: {g}"

    # Capacity check: $2k / $100k = 0.02 → well under 0.10 threshold
    capacity = next(g for g in report.gates if g.gate_name == "Capacity")
    assert capacity.value == pytest.approx(0.02, abs=1e-6)

    # Persistence check: 21/35 = 0.60 >= 0.50
    persistence = next(g for g in report.gates if g.gate_name == "Persistence")
    assert persistence.value == pytest.approx(0.60, abs=1e-6)

    # Days elapsed: 34 days (day 0 → day 34) >= 30
    days_gate = next(g for g in report.gates if g.gate_name == "Days Elapsed")
    assert days_gate.value >= 30


# ---------------------------------------------------------------------------
# Test 4 — low volume → Capacity gate fails → REJECT
# ---------------------------------------------------------------------------


def test_gate_capacity_fail():
    """All opps have volume $500 → $2000/$500 = 0.40 > 0.10 → capacity fails."""
    conn = _make_db()

    base = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    # 35 daily scans, all with opps → persistence passes (100%), days passes (34d)
    for i in range(35):
        dt = base + timedelta(days=i)
        _insert_scan(conn, dt, opps_found=1)

    # Each opp has exactly $500 total volume → position_pct = $2000/$500 = 0.40
    for _ in range(35):
        _insert_opp(conn, total_volume=500.0)

    gate = _make_gate(conn)
    report = gate.run_gate()

    capacity = next(g for g in report.gates if g.gate_name == "Capacity")
    assert not capacity.passed
    assert capacity.value == pytest.approx(
        PaperArbGate.MAX_POSITION_USD / 500.0, abs=1e-6
    )
    # $2000 / $500 = 4.0 >> 0.10
    assert capacity.value == pytest.approx(4.0, abs=1e-6)

    assert report.recommendation == "REJECT"
    assert not report.overall_pass

    # Other gates should pass
    persistence = next(g for g in report.gates if g.gate_name == "Persistence")
    days_gate = next(g for g in report.gates if g.gate_name == "Days Elapsed")
    assert persistence.passed
    assert days_gate.passed


# ---------------------------------------------------------------------------
# Test 5 — GateResult __str__ formatting
# ---------------------------------------------------------------------------


def test_gate_result_str_pass():
    g = GateResult(
        gate_name="Persistence",
        passed=True,
        value=0.650,
        threshold=0.500,
        detail="13 of 20 scans",
    )
    s = str(g)
    assert "[PASS]" in s
    assert "Persistence" in s
    assert "0.650" in s
    assert "0.500" in s


def test_gate_result_str_fail():
    g = GateResult(
        gate_name="Capacity",
        passed=False,
        value=0.400,
        threshold=0.100,
        detail="$2000 / $5000 = 40%",
    )
    s = str(g)
    assert "[FAIL]" in s
    assert "Capacity" in s


# ---------------------------------------------------------------------------
# Test 6 — empty DB → all gates fail, REJECT
# ---------------------------------------------------------------------------


def test_gate_empty_db():
    """With no data at all every gate fails and report is REJECT."""
    conn = _make_db()
    gate = _make_gate(conn)
    report = gate.run_gate()

    assert not report.overall_pass
    assert report.recommendation == "REJECT"
    assert report.total_scans == 0
    assert report.total_opps_detected == 0
    for g in report.gates:
        assert not g.passed


# ---------------------------------------------------------------------------
# Test 7 — PaperGateReport fields populated correctly
# ---------------------------------------------------------------------------


def test_report_is_dataclass():
    """PaperGateReport should be a proper dataclass with all expected fields."""
    r = PaperGateReport(
        scan_start="2026-01-01",
        scan_end="2026-02-15",
        days_elapsed=45,
        total_scans=45,
        total_opps_detected=30,
        gates=[],
        overall_pass=True,
        recommendation="PROMOTE",
        summary="All good.",
    )
    assert r.scan_start == "2026-01-01"
    assert r.recommendation == "PROMOTE"
    assert r.overall_pass is True

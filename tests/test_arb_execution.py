"""Tests for Kalshi NegRisk paper execution engine."""

from __future__ import annotations

import json

import pytest

from llm_quant.arb.execution import KalshiArbExecution
from llm_quant.arb.kalshi_client import KalshiCondition, KalshiEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_condition(
    ticker: str,
    yes_ask: float = 0.30,
    volume_24h: float = 500.0,
) -> KalshiCondition:
    """Build a minimal KalshiCondition for testing."""
    return KalshiCondition(
        ticker=ticker,
        event_ticker="TEST-EVENT",
        title=f"Condition {ticker}",
        yes_ask=yes_ask,
        yes_bid=yes_ask - 0.02,
        no_ask=1.0 - yes_ask + 0.01,
        volume_24h=volume_24h,
        volume_total=volume_24h * 10,
        is_open=True,
    )


def _make_event(
    *,
    mutually_exclusive: bool = True,
    yes_asks: list[float] | None = None,
    volumes: list[float] | None = None,
    event_ticker: str = "TEST-EVENT-001",
) -> KalshiEvent:
    """Build a KalshiEvent for testing with configurable parameters."""
    if yes_asks is None:
        yes_asks = [0.30, 0.28, 0.25]  # sum=0.83, complement=0.17, net=0.14
    if volumes is None:
        volumes = [500.0] * len(yes_asks)

    markets = [
        _make_condition(f"COND-{i}", yes_ask=p, volume_24h=v)
        for i, (p, v) in enumerate(zip(yes_asks, volumes, strict=False))
    ]
    return KalshiEvent(
        event_ticker=event_ticker,
        series_ticker="TEST-SERIES",
        title="Test Event",
        category="politics",
        mutually_exclusive=mutually_exclusive,
        markets=markets,
    )


@pytest.fixture
def engine(tmp_path) -> KalshiArbExecution:
    return KalshiArbExecution(db_path=tmp_path / "test_exec.db")


@pytest.fixture
def valid_event() -> KalshiEvent:
    """Event with sum_yes=0.83, complement=0.17, net=0.14 — strong arb."""
    return _make_event(yes_asks=[0.30, 0.28, 0.25], volumes=[500.0, 600.0, 400.0])


# ---------------------------------------------------------------------------
# Test 1 — low volume on one condition → go=False
# ---------------------------------------------------------------------------


def test_evaluate_no_go_low_volume(engine: KalshiArbExecution) -> None:
    """A single condition with volume=0 should block execution (non-atomic risk)."""
    event = _make_event(
        yes_asks=[0.30, 0.28, 0.25],
        volumes=[500.0, 0.0, 400.0],  # middle condition has zero volume
    )
    decision = engine.evaluate(event)

    assert decision.go is False
    assert decision.position_usd == 0.0
    assert decision.expected_pnl == 0.0
    # The failed list should mention condition volume
    assert any("min_condition_volume" in f for f in decision.checks_failed)


# ---------------------------------------------------------------------------
# Test 2 — negative net_spread → go=False
# ---------------------------------------------------------------------------


def test_evaluate_no_go_negative_spread(engine: KalshiArbExecution) -> None:
    """Sum YES > 0.97 means the 3% fee consumes the entire complement → no-go."""
    # sum = 0.33 + 0.34 + 0.33 = 1.00, complement = 0, net = -0.03
    event = _make_event(
        yes_asks=[0.33, 0.34, 0.33],
        volumes=[500.0, 500.0, 500.0],
    )
    decision = engine.evaluate(event)

    assert decision.go is False
    assert any("net_spread" in f for f in decision.checks_failed)


# ---------------------------------------------------------------------------
# Test 3 — valid opportunity → go=True with correct Kelly
# ---------------------------------------------------------------------------


def test_evaluate_go_valid_opportunity(engine: KalshiArbExecution) -> None:
    """A well-formed event should pass all checks and produce correct sizing."""
    # sum_yes = 0.83, complement = 0.17, net = 0.14
    event = _make_event(yes_asks=[0.30, 0.28, 0.25], volumes=[500.0, 600.0, 400.0])
    decision = engine.evaluate(event)

    net_spread = event.net_spread
    kelly_raw = net_spread / (1.0 + net_spread)
    expected_kelly = min(kelly_raw, engine.MAX_KELLY_FRACTION)
    expected_position = expected_kelly * engine.NAV_USD
    expected_pnl = expected_position * net_spread

    assert decision.go is True
    assert decision.checks_failed == []
    assert len(decision.checks_passed) >= 5

    assert abs(decision.kelly_fraction - expected_kelly) < 1e-9
    assert abs(decision.position_usd - expected_position) < 1e-6
    assert abs(decision.expected_pnl - expected_pnl) < 1e-6

    # With a 14% net spread and MAX_KELLY_FRACTION=0.02, kelly is always capped
    assert decision.kelly_fraction <= engine.MAX_KELLY_FRACTION
    assert decision.position_usd <= engine.MAX_KELLY_FRACTION * engine.NAV_USD


# ---------------------------------------------------------------------------
# Test 4 — execute_paper roundtrip → record appears in open executions
# ---------------------------------------------------------------------------


def test_execute_paper_roundtrip(
    engine: KalshiArbExecution, valid_event: KalshiEvent
) -> None:
    """After execute_paper, the record should appear in get_open_executions."""
    decision = engine.evaluate(valid_event)
    assert decision.go, f"Expected go=True, got: {decision.reason}"

    record = engine.execute_paper(valid_event, decision)

    # Basic record fields
    assert record.exec_id  # non-empty UUID
    assert record.event_ticker == valid_event.event_ticker
    assert record.status == "open"
    assert record.actual_pnl is None
    assert record.resolved_dt is None
    assert abs(record.net_spread - valid_event.net_spread) < 1e-9
    assert abs(record.kelly_fraction - decision.kelly_fraction) < 1e-9
    assert abs(record.position_usd - decision.position_usd) < 1e-6

    # conditions_json should be valid JSON with ticker / yes_ask / volume_24h
    conditions = json.loads(record.conditions_json)
    assert len(conditions) == len(valid_event.markets)
    assert "ticker" in conditions[0]
    assert "yes_ask" in conditions[0]
    assert "volume_24h" in conditions[0]

    # The record should appear in open executions query
    open_execs = engine.get_open_executions()
    assert len(open_execs) == 1
    found = open_execs[0]
    assert found.exec_id == record.exec_id
    assert found.status == "open"


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


def test_execute_paper_raises_on_no_go(engine: KalshiArbExecution) -> None:
    """execute_paper should raise ValueError if decision.go is False."""
    event = _make_event(yes_asks=[0.50, 0.50, 0.50], volumes=[500.0, 0.0, 500.0])
    decision = engine.evaluate(event)
    assert not decision.go

    with pytest.raises(ValueError, match=r"decision\.go=False"):
        engine.execute_paper(event, decision)


def test_mark_resolved_updates_pnl(
    engine: KalshiArbExecution, valid_event: KalshiEvent
) -> None:
    """mark_resolved should set status=resolved and compute correct actual_pnl."""
    decision = engine.evaluate(valid_event)
    record = engine.execute_paper(valid_event, decision)

    winning_ticker = valid_event.markets[0].ticker
    actual_pnl = engine.mark_resolved(record.exec_id, winning_ticker)

    # PnL = position_usd * net_spread
    expected_pnl = record.position_usd * record.net_spread
    assert abs(actual_pnl - expected_pnl) < 1e-6

    # Should no longer appear in open executions
    open_execs = engine.get_open_executions()
    assert len(open_execs) == 0


def test_mark_resolved_raises_if_not_found(engine: KalshiArbExecution) -> None:
    """mark_resolved should raise ValueError for unknown exec_id."""
    with pytest.raises(ValueError, match="not found"):
        engine.mark_resolved("00000000-0000-0000-0000-000000000000", "SOME-TICKER")


def test_mark_resolved_raises_if_already_resolved(
    engine: KalshiArbExecution, valid_event: KalshiEvent
) -> None:
    """Double-resolving should raise ValueError."""
    decision = engine.evaluate(valid_event)
    record = engine.execute_paper(valid_event, decision)
    engine.mark_resolved(record.exec_id, valid_event.markets[0].ticker)

    with pytest.raises(ValueError, match="already resolved"):
        engine.mark_resolved(record.exec_id, valid_event.markets[0].ticker)


def test_pnl_summary_after_trades(
    engine: KalshiArbExecution, valid_event: KalshiEvent
) -> None:
    """get_pnl_summary should reflect executed and resolved trades correctly."""
    # One open trade
    d1 = engine.evaluate(valid_event)
    r1 = engine.execute_paper(valid_event, d1)

    # Second event — different ticker
    event2 = _make_event(
        yes_asks=[0.25, 0.27, 0.30],
        volumes=[200.0, 300.0, 250.0],
        event_ticker="TEST-EVENT-002",
    )
    d2 = engine.evaluate(event2)
    r2 = engine.execute_paper(event2, d2)

    # Resolve one
    engine.mark_resolved(r1.exec_id, valid_event.markets[0].ticker)

    summary = engine.get_pnl_summary()
    assert summary["total_trades"] == 2
    assert summary["open_trades"] == 1
    assert summary["resolved_trades"] == 1
    assert summary["win_rate"] == 1.0  # arb always wins
    assert summary["total_pnl"] > 0.0
    assert summary["avg_net_spread"] > 0.0
    assert summary["total_position_usd"] > 0.0

    _ = r2  # referenced to suppress unused warning


def test_not_mutually_exclusive_fails(engine: KalshiArbExecution) -> None:
    """An event with mutually_exclusive=False should fail the first check."""
    event = _make_event(mutually_exclusive=False, yes_asks=[0.30, 0.28, 0.25])
    decision = engine.evaluate(event)
    assert decision.go is False
    assert any("mutually_exclusive=False" in f for f in decision.checks_failed)


def test_kelly_cap_applied(engine: KalshiArbExecution) -> None:
    """Kelly is always capped at MAX_KELLY_FRACTION regardless of spread size."""
    # Extreme complement: sum_yes = 0.10 → complement = 0.90, net ≈ 0.87
    event = _make_event(
        yes_asks=[0.05, 0.03, 0.02],
        volumes=[500.0, 500.0, 500.0],
    )
    decision = engine.evaluate(event)
    if decision.go:
        assert decision.kelly_fraction == engine.MAX_KELLY_FRACTION
        assert decision.position_usd == engine.MAX_KELLY_FRACTION * engine.NAV_USD


def test_schema_init_idempotent(tmp_path) -> None:
    """Creating KalshiArbExecution twice on the same DB should not error."""
    db = tmp_path / "idempotent.db"
    eng1 = KalshiArbExecution(db_path=db)
    eng2 = KalshiArbExecution(db_path=db)
    # Both should be functional
    summary1 = eng1.get_pnl_summary()
    summary2 = eng2.get_pnl_summary()
    assert summary1["total_trades"] == 0
    assert summary2["total_trades"] == 0

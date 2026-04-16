"""Tests for whipsaw guards R2 (cash floor) and R3 (snapshot-gap soft block).

See docs/investigations/apr01-whipsaw.md for the root-cause analysis that
motivates these guards. R2 lives in scripts/execute_decision.py; R3 lives in
scripts/build_context.py.

R2 — reject decisions that project cash above 40% of NAV during risk_on /
transition regimes without explicit high-confidence risk_off override.

R3 — inject a snapshot_staleness advisory into the governance block when the
last ``portfolio_snapshots`` row is >=3 days old (warning) or >=7 days
(halt).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

# Make scripts/ importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from build_context import (
    SNAPSHOT_GAP_HALT_DAYS,
    SNAPSHOT_GAP_WARNING_DAYS,
    _check_snapshot_staleness,
)
from execute_decision import (
    CASH_FLOOR_PCT,
    check_cash_floor_guard,
    project_cash_after_signals,
)

from llm_quant.brain.models import (
    Action,
    Conviction,
    MarketRegime,
    TradeSignal,
    TradingDecision,
)
from llm_quant.db.schema import init_schema
from llm_quant.trading.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    symbol: str,
    action: Action,
    target_weight: float = 0.0,
    conviction: Conviction = Conviction.MEDIUM,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        action=action,
        conviction=conviction,
        target_weight=target_weight,
        stop_loss=0.0,
        reasoning="test",
    )


def _make_decision(
    signals: list[TradeSignal],
    regime: MarketRegime = MarketRegime.RISK_ON,
    confidence: float = 0.8,
) -> TradingDecision:
    return TradingDecision(
        date=date(2026, 4, 17),
        market_regime=regime,
        regime_confidence=confidence,
        regime_reasoning="test",
        signals=signals,
        portfolio_commentary="test",
    )


def _make_portfolio_51pct_cash() -> tuple[Portfolio, dict[str, float]]:
    """Return the pre-trade book from 2026-04-01: 49% gross across 8 lines.

    Matches the state the LLM saw when it fired the whipsaw cascade. Cash is
    $51k, NAV $100k. Closing even two positions pushes cash above 60%.
    """
    p = Portfolio(initial_capital=100_000.0)
    p.cash = 51_000.0
    prices = {
        "SPY": 100.0,
        "QQQ": 100.0,
        "XLF": 100.0,
        "XLI": 100.0,
        "XLC": 100.0,
        "XLRE": 100.0,
        "USO": 100.0,
        "DBA": 100.0,
    }
    for sym in prices:
        p.positions[sym] = Position(
            symbol=sym,
            shares=61.25,  # 8 positions * $61.25 * $100 = $49k
            avg_cost=100.0,
            current_price=100.0,
            stop_loss=90.0,
        )
    # Total gross = 8 * 6125 = 49,000. NAV = 51,000 + 49,000 = 100,000.
    assert abs(p.nav - 100_000.0) < 1.0
    return p, prices


# ---------------------------------------------------------------------------
# R2 — project_cash_after_signals
# ---------------------------------------------------------------------------


class TestProjectCashAfterSignals:
    """The projection helper must mirror executor logic for BUY/SELL/CLOSE."""

    def test_no_signals_returns_current_cash(self):
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision([])
        proj_cash, nav = project_cash_after_signals(p, decision, prices)
        assert proj_cash == pytest.approx(51_000.0)
        assert nav == pytest.approx(100_000.0)

    def test_close_increases_projected_cash(self):
        """Closing 4 of 8 positions releases ~$24.5k into cash."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
                _make_signal("XLRE", Action.CLOSE),
            ]
        )
        proj_cash, nav = project_cash_after_signals(p, decision, prices)
        # Cash: 51k + 4 * 6125 = 75,500. NAV unchanged in projection (uses pre-trade).
        assert proj_cash == pytest.approx(75_500.0, abs=1.0)
        assert proj_cash / nav > 0.7

    def test_buy_decreases_projected_cash(self):
        """BUY signal spends cash up to target_weight * nav_before."""
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 100_000.0
        prices = {"SPY": 100.0}
        decision = _make_decision([_make_signal("SPY", Action.BUY, target_weight=0.05)])
        proj_cash, _nav = project_cash_after_signals(p, decision, prices)
        # Spend 5% of $100k = $5000.
        assert proj_cash == pytest.approx(95_000.0)

    def test_sell_partial_toward_target_weight(self):
        """SELL reduces position notional to target_weight * nav_before."""
        p, prices = _make_portfolio_51pct_cash()
        # SPY currently $6,125 notional (~6.1% weight). Reduce to 2% ($2k).
        decision = _make_decision(
            [_make_signal("SPY", Action.SELL, target_weight=0.02)]
        )
        proj_cash, _ = project_cash_after_signals(p, decision, prices)
        # Sold $4,125 of SPY.
        assert proj_cash == pytest.approx(55_125.0, abs=1.0)

    def test_skips_signal_without_price(self):
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision([_make_signal("UNKNOWN_SYM", Action.CLOSE)])
        proj_cash, _ = project_cash_after_signals(p, decision, prices)
        # No price → skipped, cash unchanged.
        assert proj_cash == pytest.approx(51_000.0)


# ---------------------------------------------------------------------------
# R2 — check_cash_floor_guard
# ---------------------------------------------------------------------------


class TestCashFloorGuard:
    """Rejection logic for mass-close attempts in non-risk-off regimes."""

    def test_allows_small_cash_projection(self):
        """Well below the 40% floor — always allowed."""
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 20_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=800,
            avg_cost=100.0,
            current_price=100.0,
            stop_loss=90.0,
        )
        prices = {"SPY": 100.0}
        decision = _make_decision([])  # no signals

        allowed, err, dbg = check_cash_floor_guard(p, decision, prices)
        assert allowed is True
        assert err is None
        assert dbg["projected_cash_pct"] == pytest.approx(0.20)

    def test_rejects_mass_close_in_risk_on(self):
        """The 2026-04-01 whipsaw scenario: 7 closes in risk_on → rejected."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("SPY", Action.CLOSE),
                _make_signal("QQQ", Action.CLOSE),
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
                _make_signal("XLRE", Action.CLOSE),
                _make_signal("USO", Action.CLOSE),
            ],
            regime=MarketRegime.RISK_ON,
            confidence=0.8,
        )

        allowed, err, dbg = check_cash_floor_guard(p, decision, prices)
        assert allowed is False
        assert err is not None
        assert "40% floor" in err
        assert "risk_on" in err
        assert dbg["projected_cash_pct"] > CASH_FLOOR_PCT

    def test_rejects_mass_close_in_transition(self):
        """Transition regime treated same as risk_on."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
                _make_signal("XLRE", Action.CLOSE),
            ],
            regime=MarketRegime.TRANSITION,
            confidence=0.6,
        )

        allowed, err, _ = check_cash_floor_guard(p, decision, prices)
        assert allowed is False
        assert err is not None

    def test_allows_mass_close_in_risk_off_with_high_confidence(self):
        """Explicit risk_off @ conf>=0.70 opens the floor override."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("SPY", Action.CLOSE),
                _make_signal("QQQ", Action.CLOSE),
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
            ],
            regime=MarketRegime.RISK_OFF,
            confidence=0.85,
        )

        allowed, err, _ = check_cash_floor_guard(p, decision, prices)
        assert allowed is True
        assert err is None

    def test_rejects_risk_off_with_low_confidence(self):
        """Low-confidence risk_off must NOT override the floor."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("SPY", Action.CLOSE),
                _make_signal("QQQ", Action.CLOSE),
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
            ],
            regime=MarketRegime.RISK_OFF,
            confidence=0.50,
        )

        allowed, err, _ = check_cash_floor_guard(p, decision, prices)
        assert allowed is False
        assert err is not None

    def test_allows_when_governance_halt(self):
        """Halt state: sells-only gate already in force, don't double-block."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("SPY", Action.CLOSE),
                _make_signal("QQQ", Action.CLOSE),
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
            ],
            regime=MarketRegime.RISK_ON,
            confidence=0.80,
        )

        allowed, err, _ = check_cash_floor_guard(
            p, decision, prices, governance_severity="halt"
        )
        assert allowed is True
        assert err is None

    def test_warning_severity_does_not_override_guard(self):
        """Only 'halt' severity overrides; 'warning' is not enough."""
        p, prices = _make_portfolio_51pct_cash()
        decision = _make_decision(
            [
                _make_signal("SPY", Action.CLOSE),
                _make_signal("QQQ", Action.CLOSE),
                _make_signal("XLF", Action.CLOSE),
                _make_signal("XLI", Action.CLOSE),
                _make_signal("XLC", Action.CLOSE),
            ],
            regime=MarketRegime.RISK_ON,
            confidence=0.80,
        )

        allowed, _, _ = check_cash_floor_guard(
            p, decision, prices, governance_severity="warning"
        )
        assert allowed is False

    def test_borderline_projection_at_floor(self):
        """Exactly 40% projected cash passes (<=, not <)."""
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 40_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=600,
            avg_cost=100.0,
            current_price=100.0,
            stop_loss=90.0,
        )
        prices = {"SPY": 100.0}
        # No-op decision: projected cash == current cash == 40%.
        decision = _make_decision([])
        allowed, err, dbg = check_cash_floor_guard(p, decision, prices)
        assert allowed is True
        assert dbg["projected_cash_pct"] == pytest.approx(0.40)
        assert err is None


# ---------------------------------------------------------------------------
# R3 — _check_snapshot_staleness
# ---------------------------------------------------------------------------


@pytest.fixture
def staleness_db(tmp_path):
    """DuckDB with schema initialised, used for staleness tests."""
    db_path = tmp_path / "staleness.duckdb"
    conn = init_schema(db_path)
    yield conn
    conn.close()


def _insert_snapshot(conn, snapshot_id: int, snap_date: date, pod_id: str = "default"):
    conn.execute(
        """
        INSERT INTO portfolio_snapshots
            (snapshot_id, date, pod_id, nav, cash,
             gross_exposure, net_exposure, total_pnl)
        VALUES (?, ?, ?, 100000.0, 50000.0, 50000.0, 50000.0, 0.0)
        """,
        [snapshot_id, snap_date, pod_id],
    )


class TestSnapshotStaleness:
    """R3 snapshot-gap soft block."""

    def test_no_snapshot_returns_none(self, staleness_db):
        """Empty portfolio_snapshots → no advisory (fresh system)."""
        result = _check_snapshot_staleness(staleness_db, "default", date(2026, 4, 17))
        assert result is None

    def test_fresh_snapshot_returns_none(self, staleness_db):
        """Snapshot from today → no advisory."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, today)
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is None

    def test_2_day_gap_returns_none(self, staleness_db):
        """Below warning threshold (3 days) → no advisory."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, date(2026, 4, 15))  # 2 days old
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is None

    def test_3_day_gap_warning(self, staleness_db):
        """At warning threshold → warning advisory."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, date(2026, 4, 14))  # 3 days old
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is not None
        assert result["severity"] == "warning"
        assert result["detector"] == "snapshot_staleness"
        assert result["days_stale"] == 3
        assert "Reconcile" in result["message"] or "reconcile" in result["message"]

    def test_6_day_gap_still_warning(self, staleness_db):
        """6 days old (the Apr 01 scenario) → warning, not halt yet."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, date(2026, 4, 11))  # 6 days old
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is not None
        assert result["severity"] == "warning"
        assert result["days_stale"] == 6

    def test_7_day_gap_halts(self, staleness_db):
        """At halt threshold → halt advisory, sells-only gate."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, date(2026, 4, 10))  # 7 days old
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is not None
        assert result["severity"] == "halt"
        assert result["days_stale"] == 7
        assert "sells-only" in result["message"]

    def test_14_day_gap_halts(self, staleness_db):
        """Well past halt threshold → halt advisory."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, date(2026, 4, 3))  # 14 days old
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is not None
        assert result["severity"] == "halt"
        assert result["days_stale"] == 14

    def test_per_pod_isolation(self, staleness_db):
        """Staleness for 'default' ignores fresh 'momo' snapshots."""
        today = date(2026, 4, 17)
        _insert_snapshot(staleness_db, 1, date(2026, 4, 10), pod_id="default")
        _insert_snapshot(staleness_db, 2, today, pod_id="momo")

        result_default = _check_snapshot_staleness(staleness_db, "default", today)
        result_momo = _check_snapshot_staleness(staleness_db, "momo", today)

        assert result_default is not None
        assert result_default["severity"] == "halt"
        assert result_momo is None

    def test_message_includes_last_snapshot_date(self, staleness_db):
        """Advisory carries the last_snapshot_date for observability."""
        today = date(2026, 4, 17)
        last = date(2026, 4, 10)
        _insert_snapshot(staleness_db, 1, last)
        result = _check_snapshot_staleness(staleness_db, "default", today)
        assert result is not None
        assert result["last_snapshot_date"] == last.isoformat()

    def test_thresholds_match_docs(self):
        """Sanity-check the constants match the R3 spec in the investigation."""
        assert SNAPSHOT_GAP_WARNING_DAYS == 3
        assert SNAPSHOT_GAP_HALT_DAYS == 7

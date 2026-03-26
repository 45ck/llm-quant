"""Kalshi NegRisk paper execution engine.

Evaluates arbitrage opportunities from the scanner, applies pre-trade checks,
sizes positions via Kelly criterion, and records paper trades to DuckDB.

Trade logic (buy-all-YES NegRisk):
  - Pay sum(YES_ask) across all mutually exclusive conditions.
  - Receive $1 from the single winning condition.
  - Net profit = (1 - sum_yes_ask) - 0.03 (Kalshi 3% fee on winning leg).
  - Position size = min(kelly_fraction, MAX_KELLY_FRACTION) * NAV_USD.

Non-atomic risk:
  If we cannot fill ALL conditions simultaneously, unfilled legs create
  directional exposure (we hold YES on some but not all outcomes).
  The MIN_CONDITION_VOLUME guard ensures each leg has enough liquidity
  to fill at least a minimal position.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from llm_quant.arb.kalshi_client import KalshiEvent
from llm_quant.arb.schema import init_arb_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExecutionDecision:
    """Pre-trade evaluation result for a Kalshi NegRisk event."""

    go: bool
    reason: str
    kelly_fraction: float  # f* = net_spread / (1 + net_spread)
    position_usd: float  # kelly_fraction * NAV, capped at MAX_KELLY_USD
    expected_pnl: float  # position_usd * net_spread
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass
class ExecutionRecord:
    """A single paper execution persisted to pm_executions."""

    exec_id: str  # UUID
    event_ticker: str
    event_title: str
    exec_dt: str  # ISO timestamp
    conditions_json: str  # JSON list of {ticker, yes_ask, volume_24h}
    sum_yes_ask: float
    gross_complement: float
    net_spread: float
    kelly_fraction: float
    position_usd: float
    expected_pnl: float
    status: str  # 'open' | 'resolved' | 'expired'
    actual_pnl: float | None
    resolved_dt: str | None
    notes: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class KalshiArbExecution:
    """Paper execution engine for Kalshi NegRisk arbitrage.

    All trades are paper-only.  No real orders are sent.
    """

    NAV_USD: float = 100_000.0  # paper portfolio NAV
    MAX_KELLY_FRACTION: float = 0.02  # 2% of NAV max per trade
    MIN_CONDITION_VOLUME: float = 100.0  # min $100 24h volume per condition
    MIN_CONDITIONS: int = 2  # need at least 2 mutually exclusive outcomes

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(str(self._db_path))
        init_arb_schema(self._conn)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, event: KalshiEvent) -> ExecutionDecision:
        """Kelly-size the opportunity with pre-trade checks.

        Pre-trade checks (any failure → go=False):
          1. mutually_exclusive is True
          2. net_spread > 0 (positive EV after 3% fee)
          3. min_condition_volume >= MIN_CONDITION_VOLUME
          4. len(markets) >= MIN_CONDITIONS
          5. All YES ask prices valid (0 < yes_ask < 1)

        Returns:
            ExecutionDecision with go flag, sizing, and check details.
        """
        passed: list[str] = []
        failed: list[str] = []

        # Check 1 — mutually exclusive structure
        if event.mutually_exclusive:
            passed.append("mutually_exclusive=True")
        else:
            failed.append("mutually_exclusive=False (not a NegRisk event)")

        # Check 2 — minimum conditions
        n = len(event.markets)
        if n >= self.MIN_CONDITIONS:
            passed.append(f"conditions={n} >= {self.MIN_CONDITIONS}")
        else:
            failed.append(f"conditions={n} < {self.MIN_CONDITIONS} required")

        # Check 3 — valid YES ask prices on every condition
        invalid_prices = [
            c.ticker for c in event.markets if not (0.0 < c.yes_ask < 1.0)
        ]
        if invalid_prices:
            failed.append(f"invalid yes_ask on: {invalid_prices}")
        else:
            passed.append("all yes_ask prices in (0, 1)")

        # Check 4 — minimum liquidity per condition (non-atomic risk guard)
        min_vol = event.min_condition_volume
        if min_vol >= self.MIN_CONDITION_VOLUME:
            passed.append(
                f"min_condition_volume={min_vol:.0f} >= {self.MIN_CONDITION_VOLUME:.0f}"
            )
        else:
            failed.append(
                f"min_condition_volume={min_vol:.0f} < {self.MIN_CONDITION_VOLUME:.0f}"
            )

        # Check 5 — positive EV after fee
        net_spread = event.net_spread
        if net_spread > 0.0:
            passed.append(f"net_spread={net_spread:.4f} > 0")
        else:
            failed.append(f"net_spread={net_spread:.4f} <= 0 (fee exceeds complement)")

        # If any check failed, return no-go
        if failed:
            return ExecutionDecision(
                go=False,
                reason=f"Pre-trade checks failed: {'; '.join(failed)}",
                kelly_fraction=0.0,
                position_usd=0.0,
                expected_pnl=0.0,
                checks_passed=passed,
                checks_failed=failed,
            )

        # All checks passed — compute Kelly sizing
        kelly_raw = net_spread / (1.0 + net_spread)
        kelly_capped = min(kelly_raw, self.MAX_KELLY_FRACTION)
        position_usd = kelly_capped * self.NAV_USD
        expected_pnl = position_usd * net_spread

        return ExecutionDecision(
            go=True,
            reason=(
                f"All checks passed — net_spread={net_spread:.4f}, "
                f"kelly={kelly_capped:.4f}, position=${position_usd:.2f}"
            ),
            kelly_fraction=kelly_capped,
            position_usd=position_usd,
            expected_pnl=expected_pnl,
            checks_passed=passed,
            checks_failed=failed,
        )

    # ------------------------------------------------------------------
    # Paper execution
    # ------------------------------------------------------------------

    def execute_paper(
        self, event: KalshiEvent, decision: ExecutionDecision
    ) -> ExecutionRecord:
        """Record a paper execution to pm_executions.

        Should only be called when decision.go is True.  Raises ValueError
        if decision.go is False to prevent accidental no-go executions.
        """
        if not decision.go:
            msg = f"Cannot execute: decision.go=False — {decision.reason}"
            raise ValueError(msg)

        exec_id = str(uuid.uuid4())
        exec_dt = datetime.now(UTC).isoformat()

        conditions_data = [
            {
                "ticker": c.ticker,
                "yes_ask": c.yes_ask,
                "volume_24h": c.volume_24h,
            }
            for c in event.markets
        ]
        conditions_json = json.dumps(conditions_data)

        notes = (
            f"N={len(event.markets)} conditions, "
            f"category={event.category}, "
            f"kelly_raw={event.net_spread / (1.0 + event.net_spread):.4f}"
        )

        record = ExecutionRecord(
            exec_id=exec_id,
            event_ticker=event.event_ticker,
            event_title=event.title,
            exec_dt=exec_dt,
            conditions_json=conditions_json,
            sum_yes_ask=event.sum_yes_ask,
            gross_complement=event.negrisk_complement,
            net_spread=event.net_spread,
            kelly_fraction=decision.kelly_fraction,
            position_usd=decision.position_usd,
            expected_pnl=decision.expected_pnl,
            status="open",
            actual_pnl=None,
            resolved_dt=None,
            notes=notes,
        )

        self._conn.execute(
            """
            INSERT INTO pm_executions
            (exec_id, event_ticker, event_title, exec_dt,
             conditions_json, sum_yes_ask, gross_complement,
             net_spread, kelly_fraction, position_usd, expected_pnl,
             status, actual_pnl, resolved_dt, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, NULL, ?)
            """,
            [
                record.exec_id,
                record.event_ticker,
                record.event_title,
                record.exec_dt,
                record.conditions_json,
                record.sum_yes_ask,
                record.gross_complement,
                record.net_spread,
                record.kelly_fraction,
                record.position_usd,
                record.expected_pnl,
                record.notes,
            ],
        )
        logger.info(
            "Paper execution recorded: %s | %s | position=$%.2f | expected_pnl=$%.2f",
            exec_id,
            event.event_ticker,
            record.position_usd,
            record.expected_pnl,
        )
        return record

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def mark_resolved(self, exec_id: str, winning_ticker: str) -> float:
        """Mark a paper execution as resolved.

        The NegRisk arb is deterministic: we bought all YES positions, paid
        sum_yes_ask, and receive $1 from the winning condition.  Actual PnL
        equals the net_spread scaled by position_usd (the arb profit does
        not depend on WHICH condition wins, only that exactly one does).

        PnL = position_usd * net_spread

        Args:
            exec_id: UUID of the execution to resolve.
            winning_ticker: Ticker of the condition that resolved YES.

        Returns:
            Actual PnL (USD).

        Raises:
            ValueError: If exec_id not found or already resolved.
        """
        row = self._conn.execute(
            "SELECT position_usd, net_spread, status "
            "FROM pm_executions WHERE exec_id = ?",
            [exec_id],
        ).fetchone()

        if row is None:
            msg = f"Execution {exec_id} not found"
            raise ValueError(msg)

        position_usd, net_spread, status = row

        if status != "open":
            msg = f"Execution {exec_id} is already {status}"
            raise ValueError(msg)

        actual_pnl = position_usd * net_spread
        resolved_dt = datetime.now(UTC).isoformat()

        self._conn.execute(
            """
            UPDATE pm_executions
            SET status = 'resolved',
                actual_pnl = ?,
                resolved_dt = ?,
                notes = notes || ?
            WHERE exec_id = ?
            """,
            [
                actual_pnl,
                resolved_dt,
                f" | resolved: winning_ticker={winning_ticker}",
                exec_id,
            ],
        )
        logger.info(
            "Execution %s resolved — winner=%s actual_pnl=$%.2f",
            exec_id,
            winning_ticker,
            actual_pnl,
        )
        return actual_pnl

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_executions(self) -> list[ExecutionRecord]:
        """Return all open paper executions."""
        rows = self._conn.execute(
            """
            SELECT exec_id, event_ticker, event_title, exec_dt,
                   conditions_json, sum_yes_ask, gross_complement,
                   net_spread, kelly_fraction, position_usd, expected_pnl,
                   status, actual_pnl, resolved_dt, notes
            FROM pm_executions
            WHERE status = 'open'
            ORDER BY exec_dt DESC
            """
        ).fetchall()

        return [_row_to_record(r) for r in rows]

    def get_pnl_summary(self) -> dict:
        """Summary statistics across all paper executions.

        Returns:
            Dict with keys: total_trades, open_trades, resolved_trades,
            win_rate, total_pnl, avg_net_spread, total_position_usd.
        """
        total_row = self._conn.execute(
            "SELECT COUNT(*), COUNT(actual_pnl), SUM(actual_pnl), AVG(net_spread), "
            "SUM(position_usd) FROM pm_executions"
        ).fetchone()

        wins_row = self._conn.execute(
            "SELECT COUNT(*) FROM pm_executions WHERE actual_pnl > 0"
        ).fetchone()

        open_row = self._conn.execute(
            "SELECT COUNT(*) FROM pm_executions WHERE status = 'open'"
        ).fetchone()

        if total_row is None:
            return {}

        total, resolved, total_pnl, avg_spread, total_pos = total_row
        wins = wins_row[0] if wins_row else 0
        open_count = open_row[0] if open_row else 0

        win_rate = (wins / resolved) if resolved and resolved > 0 else 0.0

        return {
            "total_trades": total or 0,
            "open_trades": open_count or 0,
            "resolved_trades": resolved or 0,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl or 0.0, 2),
            "avg_net_spread": round(avg_spread or 0.0, 4),
            "total_position_usd": round(total_pos or 0.0, 2),
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: tuple) -> ExecutionRecord:
    """Map a raw DB row to ExecutionRecord."""
    (
        exec_id,
        event_ticker,
        event_title,
        exec_dt,
        conditions_json,
        sum_yes_ask,
        gross_complement,
        net_spread,
        kelly_fraction,
        position_usd,
        expected_pnl,
        status,
        actual_pnl,
        resolved_dt,
        notes,
    ) = row
    return ExecutionRecord(
        exec_id=exec_id,
        event_ticker=event_ticker,
        event_title=event_title or "",
        exec_dt=str(exec_dt) if exec_dt is not None else "",
        conditions_json=conditions_json or "[]",
        sum_yes_ask=float(sum_yes_ask or 0.0),
        gross_complement=float(gross_complement or 0.0),
        net_spread=float(net_spread or 0.0),
        kelly_fraction=float(kelly_fraction or 0.0),
        position_usd=float(position_usd or 0.0),
        expected_pnl=float(expected_pnl or 0.0),
        status=status or "open",
        actual_pnl=float(actual_pnl) if actual_pnl is not None else None,
        resolved_dt=str(resolved_dt) if resolved_dt is not None else None,
        notes=notes or "",
    )

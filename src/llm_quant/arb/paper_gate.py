"""30-day paper validation gate for prediction market arbitrage strategies.

Replaces DSR (Deflated Sharpe Ratio) for PM arb strategies because the profit
mechanism is deductive (logical identity), not discovered by fitting data.
DSR tests whether a backtest Sharpe is genuine vs lucky — arb profit requires
no luck, just execution quality.

Gate logic:
  Gate 1 — Persistence:  ≥50% of scan windows had ≥1 opportunity
  Gate 2 — Fill Rate:    ≥80% of opportunities estimated fillable
  Gate 3 — Capacity:     Kelly-sized position ($2k) is <10% of avg volume
  Gate 4 — Days Elapsed: ≥30 days of scan history before promotion

Recommendation:
  PROMOTE          — all 4 gates pass
  CONTINUE_PAPER   — only Days Elapsed gate fails (not enough history yet)
  REJECT           — any quality gate (1-3) fails after sufficient data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from llm_quant.arb.schema import init_arb_schema

logger = logging.getLogger(__name__)

# Fill simulation thresholds — a position is "fillable" when both legs have
# enough resting liquidity that a $2k order won't walk the book materially.
_MIN_CONDITION_VOL_USD = 100.0  # minimum per-condition volume to fill one leg
_MIN_TOTAL_VOL_USD = 500.0  # minimum total-market volume for the full trade


@dataclass
class GateResult:
    """Result of a single validation gate."""

    gate_name: str
    passed: bool
    value: float
    threshold: float
    detail: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"  [{status}] {self.gate_name}: {self.value:.3f} "
            f"(threshold: {self.threshold:.3f}) — {self.detail}"
        )


@dataclass
class PaperGateReport:
    """Aggregated result of the full 30-day paper validation run."""

    scan_start: str  # ISO date of earliest scan
    scan_end: str  # ISO date of most recent scan
    days_elapsed: int
    total_scans: int
    total_opps_detected: int
    gates: list[GateResult]
    overall_pass: bool  # ALL gates must pass
    recommendation: str  # "PROMOTE" | "REJECT" | "CONTINUE_PAPER"
    summary: str  # human-readable paragraph


class PaperArbGate:
    """Runs all 4 paper validation gates for a PM arb strategy.

    Reads from pm_scan_log and pm_arb_opportunities tables that were written
    by ArbScanner.  Filtered to a single ``source`` (e.g. 'kalshi') so that
    Polymarket and Kalshi strategies are evaluated independently.
    """

    NAV_USD: float = 100_000.0
    MAX_POSITION_USD: float = 2_000.0  # 2% Kelly cap on a $100k book

    # Gate thresholds
    PERSISTENCE_THRESHOLD: float = 0.50  # ≥50% of scan windows had ≥1 opp
    FILL_RATE_THRESHOLD: float = 0.80  # ≥80% of opportunities fillable
    CAPACITY_THRESHOLD: float = 0.10  # position must be <10% of avg volume
    DAYS_THRESHOLD: float = 30.0  # ≥30 calendar days of data

    def __init__(self, db_path: str | Path, source: str = "kalshi") -> None:
        self.db_path = Path(db_path)
        self.source = source
        self._conn: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            init_arb_schema(self._conn)
        return self._conn

    def close(self) -> None:
        """Close the DuckDB connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Shared stat block (single DB round-trip for scan-level numbers)
    # ------------------------------------------------------------------

    def _get_scan_stats(self) -> dict:
        """Query pm_scan_log for scan history statistics filtered to source.

        Returns a dict with keys:
          total_scans, scans_with_opps, first_scan_dt, last_scan_dt, days_elapsed
        """
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                        AS total_scans,
                COUNT(*) FILTER (WHERE opps_found > 0)         AS scans_with_opps,
                MIN(CAST(started_at AS DATE))                   AS first_scan_dt,
                MAX(CAST(started_at AS DATE))                   AS last_scan_dt
            FROM pm_scan_log
            WHERE source = ?
              AND error IS NULL
            """,
            [self.source],
        ).fetchone()

        if row is None or row[0] == 0:
            return {
                "total_scans": 0,
                "scans_with_opps": 0,
                "first_scan_dt": None,
                "last_scan_dt": None,
                "days_elapsed": 0,
            }

        total_scans, scans_with_opps, first_dt, last_dt = row
        days_elapsed = 0
        if first_dt is not None and last_dt is not None:
            # DuckDB may return date objects or strings depending on driver version
            if isinstance(first_dt, str):
                first_dt = date.fromisoformat(first_dt)
            if isinstance(last_dt, str):
                last_dt = date.fromisoformat(last_dt)
            days_elapsed = (last_dt - first_dt).days

        return {
            "total_scans": total_scans,
            "scans_with_opps": scans_with_opps,
            "first_scan_dt": first_dt,
            "last_scan_dt": last_dt,
            "days_elapsed": days_elapsed,
        }

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------

    def check_persistence(self) -> GateResult:
        """Fraction of scan windows that found ≥1 opportunity.

        Source: pm_scan_log.opps_found > 0 filtered by source.
        Threshold: 0.50 (half of all scan sessions had opportunities).
        """
        stats = self._get_scan_stats()
        total = stats["total_scans"]

        if total == 0:
            return GateResult(
                gate_name="Persistence",
                passed=False,
                value=0.0,
                threshold=self.PERSISTENCE_THRESHOLD,
                detail=f"No scan records found for source='{self.source}'",
            )

        with_opps = stats["scans_with_opps"]
        fraction = with_opps / total
        passed = fraction >= self.PERSISTENCE_THRESHOLD

        detail = f"{with_opps} of {total} scans found ≥1 opportunity"
        return GateResult(
            gate_name="Persistence",
            passed=passed,
            value=fraction,
            threshold=self.PERSISTENCE_THRESHOLD,
            detail=detail,
        )

    def check_fill_rate(self) -> GateResult:
        """Fraction of detected opportunities estimated to be fillable.

        Fill model: an opportunity is fillable when BOTH conditions hold:
          - total_volume >= $500  (enough total market liquidity)
        We proxy per-condition volume as total_volume (scanner already
        applied min_condition_vol filter at detection time).

        Threshold: 0.80 (80% of opps must be fillable).
        """
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                                  AS total_opps,
                COUNT(*) FILTER (WHERE total_volume >= ?)  AS fillable_opps
            FROM pm_arb_opportunities
            WHERE source = ?
            """,
            [_MIN_TOTAL_VOL_USD, self.source],
        ).fetchone()

        total_opps = row[0] if row else 0
        fillable_opps = row[1] if row else 0

        if total_opps == 0:
            return GateResult(
                gate_name="Fill Rate",
                passed=False,
                value=0.0,
                threshold=self.FILL_RATE_THRESHOLD,
                detail=f"No opportunities found for source='{self.source}'",
            )

        fraction = fillable_opps / total_opps
        passed = fraction >= self.FILL_RATE_THRESHOLD

        detail = (
            f"{fillable_opps} of {total_opps} opps fillable "
            f"(total_vol ≥ ${_MIN_TOTAL_VOL_USD:,.0f})"
        )
        return GateResult(
            gate_name="Fill Rate",
            passed=passed,
            value=fraction,
            threshold=self.FILL_RATE_THRESHOLD,
            detail=detail,
        )

    def check_capacity(self) -> GateResult:
        """Average position-as-fraction-of-volume across all opportunities.

        Formula: MAX_POSITION_USD / avg(total_volume).
        Gate passes when the ratio is BELOW the threshold (position is a
        small fraction of available liquidity → won't move the market).

        Threshold: 0.10 (position ≤ 10% of average market volume).
        """
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT AVG(total_volume) AS avg_vol
            FROM pm_arb_opportunities
            WHERE source = ?
              AND total_volume > 0
            """,
            [self.source],
        ).fetchone()

        avg_vol = row[0] if row and row[0] is not None else 0.0

        if avg_vol <= 0:
            return GateResult(
                gate_name="Capacity",
                passed=False,
                value=float("inf"),
                threshold=self.CAPACITY_THRESHOLD,
                detail=f"No volume data for source='{self.source}'",
            )

        position_pct = self.MAX_POSITION_USD / avg_vol
        # Gate passes when position is a SMALL fraction of volume (< threshold)
        passed = position_pct <= self.CAPACITY_THRESHOLD

        detail = (
            f"${self.MAX_POSITION_USD:,.0f} position / "
            f"${avg_vol:,.0f} avg vol = {position_pct:.1%}"
        )
        return GateResult(
            gate_name="Capacity",
            passed=passed,
            value=position_pct,
            threshold=self.CAPACITY_THRESHOLD,
            detail=detail,
        )

    def check_days_elapsed(self) -> GateResult:
        """Days between first and last scan must be >= 30.

        Rationale: 30-calendar-day track record required before promotion.
        """
        stats = self._get_scan_stats()
        days = stats["days_elapsed"]
        passed = days >= int(self.DAYS_THRESHOLD)

        first_dt = stats["first_scan_dt"]
        last_dt = stats["last_scan_dt"]

        if first_dt is None:
            detail = "No scan records found — cannot compute elapsed days"
        else:
            detail = f"First scan: {first_dt}  Last scan: {last_dt}"

        return GateResult(
            gate_name="Days Elapsed",
            passed=passed,
            value=float(days),
            threshold=self.DAYS_THRESHOLD,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def run_gate(self) -> PaperGateReport:
        """Run all 4 gates and return a consolidated PaperGateReport."""
        stats = self._get_scan_stats()

        # Fetch total opp count for the source
        conn = self._get_conn()
        opp_row = conn.execute(
            "SELECT COUNT(*) FROM pm_arb_opportunities WHERE source = ?",
            [self.source],
        ).fetchone()
        total_opps = opp_row[0] if opp_row else 0

        gates = [
            self.check_persistence(),
            self.check_fill_rate(),
            self.check_capacity(),
            self.check_days_elapsed(),
        ]

        overall_pass = all(g.passed for g in gates)

        # Determine recommendation
        days_gate = next(g for g in gates if g.gate_name == "Days Elapsed")
        quality_gates = [g for g in gates if g.gate_name != "Days Elapsed"]
        quality_all_pass = all(g.passed for g in quality_gates)

        if overall_pass:
            recommendation = "PROMOTE"
        elif not days_gate.passed and quality_all_pass:
            # Quality looks good, just needs more calendar time
            recommendation = "CONTINUE_PAPER"
        else:
            recommendation = "REJECT"

        # Format dates
        first_dt = stats["first_scan_dt"]
        last_dt = stats["last_scan_dt"]
        scan_start = str(first_dt) if first_dt is not None else "N/A"
        scan_end = str(last_dt) if last_dt is not None else "N/A"

        # Build human-readable summary
        failed_names = [g.gate_name for g in gates if not g.passed]
        if recommendation == "PROMOTE":
            summary = (
                f"All 4 gates passed for source='{self.source}' over "
                f"{stats['days_elapsed']} days ({stats['total_scans']} scans). "
                f"Strategy is ready for promotion to live paper trading."
            )
        elif recommendation == "CONTINUE_PAPER":
            remaining = int(self.DAYS_THRESHOLD) - stats["days_elapsed"]
            summary = (
                f"Quality gates (Persistence, Fill Rate, Capacity) all passed. "
                f"Continue paper scanning for {remaining} more day(s) to satisfy "
                f"the 30-day track record requirement before promotion."
            )
        else:
            summary = (
                f"Strategy REJECTED: gate(s) failed — {', '.join(failed_names)}. "
                f"Investigate execution quality before re-running paper track."
            )

        return PaperGateReport(
            scan_start=scan_start,
            scan_end=scan_end,
            days_elapsed=stats["days_elapsed"],
            total_scans=stats["total_scans"],
            total_opps_detected=total_opps,
            gates=gates,
            overall_pass=overall_pass,
            recommendation=recommendation,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_report(self, report: PaperGateReport) -> None:
        """Print a formatted gate report to stdout."""
        divider = "=" * 60
        print(divider)
        print(f"  PM ARB PAPER GATE — source: {self.source}")
        print(divider)
        print(f"  Scan window : {report.scan_start}  →  {report.scan_end}")
        print(f"  Days elapsed: {report.days_elapsed}")
        print(f"  Total scans : {report.total_scans}")
        print(f"  Total opps  : {report.total_opps_detected}")
        print()
        print("  Gate Results:")
        for gate in report.gates:
            print(str(gate))
        print()
        status_line = "OVERALL: PASS" if report.overall_pass else "OVERALL: FAIL"
        print(f"  {status_line}")
        print(f"  Recommendation: {report.recommendation}")
        print()
        print(f"  {report.summary}")
        print(divider)

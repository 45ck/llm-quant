"""Generate Markdown reports from DuckDB portfolio data.

Usage::

    cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py \\
        [daily|weekly|monthly] [--date YYYY-MM-DD]
"""

# ruff: noqa: PLR0912, PLR0915, C901, PERF401

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.config import load_config
from llm_quant.db.integrity import verify_chain
from llm_quant.db.schema import get_connection, init_schema
from llm_quant.surveillance.scanner import SurveillanceScanner
from llm_quant.trading.performance import compute_performance

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Annualisation factor
_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_money(value: float) -> str:
    """Format as $XX,XXX.XX."""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


def _fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a ratio (0.05) as '5.0%'."""
    return f"{value * 100:.{decimals}f}%"


def _fmt_pct_raw(value: float, decimals: int = 2) -> str:
    """Format a value already in percentage form (e.g. -2.5 -> '-2.50%')."""
    return f"{value:.{decimals}f}%"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _resolve_db_path(config) -> str:
    """Resolve db_path from config, making relative paths absolute to project root."""
    db_path = config.general.db_path
    project_root = Path(__file__).resolve().parent.parent
    if not Path(db_path).is_absolute():
        db_path = str(project_root / db_path)
    return db_path


def _ensure_db(db_path: str) -> None:
    """Initialize DB schema if it doesn't exist."""
    path = Path(db_path)
    if not path.exists():
        init_schema(db_path)


def _get_snapshot_for_date(
    conn: duckdb.DuckDBPyConnection, target_date: date
) -> dict | None:
    """Get the latest portfolio snapshot for a given date."""
    row = conn.execute(
        """
        SELECT snapshot_id, date, nav, cash, gross_exposure,
               net_exposure, total_pnl, daily_pnl
        FROM portfolio_snapshots
        WHERE date = ?
        ORDER BY snapshot_id DESC
        LIMIT 1
        """,
        [target_date],
    ).fetchone()
    if row is None:
        return None
    return {
        "snapshot_id": row[0],
        "date": row[1],
        "nav": float(row[2]),
        "cash": float(row[3]),
        "gross_exposure": float(row[4]),
        "net_exposure": float(row[5]),
        "total_pnl": float(row[6]),
        "daily_pnl": float(row[7]) if row[7] is not None else 0.0,
    }


def _get_positions_for_snapshot(
    conn: duckdb.DuckDBPyConnection, snapshot_id: int
) -> list[dict]:
    """Get positions for a specific snapshot."""
    rows = conn.execute(
        """
        SELECT symbol, shares, avg_cost, current_price,
               market_value, unrealized_pnl, weight, stop_loss
        FROM positions
        WHERE snapshot_id = ?
        ORDER BY abs(market_value) DESC
        """,
        [snapshot_id],
    ).fetchall()
    return [
        {
            "symbol": r[0],
            "shares": float(r[1]),
            "avg_cost": float(r[2]),
            "current_price": float(r[3]),
            "market_value": float(r[4]),
            "unrealized_pnl": float(r[5]),
            "weight": float(r[6]),
            "stop_loss": float(r[7]) if r[7] is not None else None,
        }
        for r in rows
    ]


def _get_trades_for_date(
    conn: duckdb.DuckDBPyConnection, target_date: date
) -> list[dict]:
    """Get all trades for a given date."""
    rows = conn.execute(
        """
        SELECT symbol, action, shares, price, notional, conviction, reasoning
        FROM trades
        WHERE date = ?
        ORDER BY trade_id ASC
        """,
        [target_date],
    ).fetchall()
    return [
        {
            "symbol": r[0],
            "action": r[1],
            "shares": float(r[2]),
            "price": float(r[3]),
            "notional": float(r[4]),
            "conviction": r[5] or "",
            "reasoning": r[6] or "",
        }
        for r in rows
    ]


def _get_trades_for_range(
    conn: duckdb.DuckDBPyConnection, start_date: date, end_date: date
) -> list[dict]:
    """Get all trades within a date range (inclusive)."""
    rows = conn.execute(
        """
        SELECT date, symbol, action, shares, price, notional, conviction, reasoning
        FROM trades
        WHERE date >= ? AND date <= ?
        ORDER BY date ASC, trade_id ASC
        """,
        [start_date, end_date],
    ).fetchall()
    return [
        {
            "date": r[0],
            "symbol": r[1],
            "action": r[2],
            "shares": float(r[3]),
            "price": float(r[4]),
            "notional": float(r[5]),
            "conviction": r[6] or "",
            "reasoning": r[7] or "",
        }
        for r in rows
    ]


def _get_regime_for_date(
    conn: duckdb.DuckDBPyConnection, target_date: date
) -> dict | None:
    """Get the LLM decision regime for a date."""
    row = conn.execute(
        """
        SELECT market_regime, regime_confidence
        FROM llm_decisions
        WHERE date = ?
        ORDER BY decision_id DESC
        LIMIT 1
        """,
        [target_date],
    ).fetchone()
    if row is None:
        return None
    return {
        "regime": row[0] or "unknown",
        "confidence": float(row[1]) if row[1] is not None else 0.0,
    }


def _get_snapshots_for_range(
    conn: duckdb.DuckDBPyConnection, start_date: date, end_date: date
) -> list[dict]:
    """Get daily snapshots within a date range, one per date (latest snapshot)."""
    rows = conn.execute(
        """
        SELECT date, nav, cash, gross_exposure, net_exposure,
               total_pnl, daily_pnl
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY date ORDER BY snapshot_id DESC
            ) AS rn
            FROM portfolio_snapshots
            WHERE date >= ? AND date <= ?
        )
        WHERE rn = 1
        ORDER BY date ASC
        """,
        [start_date, end_date],
    ).fetchall()
    return [
        {
            "date": r[0],
            "nav": float(r[1]),
            "cash": float(r[2]),
            "gross_exposure": float(r[3]),
            "net_exposure": float(r[4]),
            "total_pnl": float(r[5]),
            "daily_pnl": float(r[6]) if r[6] is not None else 0.0,
        }
        for r in rows
    ]


def _get_regimes_for_range(
    conn: duckdb.DuckDBPyConnection, start_date: date, end_date: date
) -> list[dict]:
    """Get regime decisions within a date range."""
    rows = conn.execute(
        """
        SELECT date, market_regime, regime_confidence
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY date ORDER BY decision_id DESC
            ) AS rn
            FROM llm_decisions
            WHERE date >= ? AND date <= ?
        )
        WHERE rn = 1
        ORDER BY date ASC
        """,
        [start_date, end_date],
    ).fetchall()
    return [
        {
            "date": r[0],
            "regime": r[1] or "unknown",
            "confidence": float(r[2]) if r[2] is not None else 0.0,
        }
        for r in rows
    ]


def _compute_sortino(conn: duckdb.DuckDBPyConnection) -> float:
    """Compute annualized Sortino ratio from portfolio snapshots."""
    rows = conn.execute("""
        SELECT nav FROM portfolio_snapshots
        ORDER BY date ASC, snapshot_id ASC
        """).fetchall()
    if len(rows) < 2:
        return 0.0
    navs = [float(r[0]) for r in rows]
    returns = [(navs[i] / navs[i - 1]) - 1.0 for i in range(1, len(navs))]
    if not returns:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    downside_var = sum(r**2 for r in downside) / len(returns)
    downside_std = math.sqrt(downside_var)
    if downside_std == 0:
        return 0.0
    return (mean_ret / downside_std) * math.sqrt(_TRADING_DAYS)


def _compute_calmar(conn: duckdb.DuckDBPyConnection, initial_capital: float) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    rows = conn.execute("""
        SELECT date, nav FROM portfolio_snapshots
        ORDER BY date ASC, snapshot_id ASC
        """).fetchall()
    if len(rows) < 2:
        return 0.0
    navs = [float(r[1]) for r in rows]
    first_date = rows[0][0]
    last_date = rows[-1][0]
    days = (last_date - first_date).days
    if days <= 0:
        return 0.0
    total_return = (navs[-1] / initial_capital) - 1.0
    ann_return = total_return * (365.0 / days)
    # Max drawdown
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        peak = max(peak, nav)
        dd = (nav - peak) / peak
        max_dd = min(max_dd, dd)
    if max_dd == 0.0:
        return 0.0
    return ann_return / abs(max_dd)


def _compute_benchmark_return(
    conn: duckdb.DuckDBPyConnection,
) -> float | None:
    """Compute 60/40 SPY/TLT benchmark return over snapshot period."""
    snap = conn.execute(
        "SELECT MIN(date), MAX(date) FROM portfolio_snapshots"
    ).fetchone()
    if snap is None or snap[0] is None:
        return None
    start_date, end_date = snap[0], snap[1]

    q_start = (
        "SELECT close FROM market_data_daily"
        " WHERE symbol=? AND date >= ?"
        " ORDER BY date ASC LIMIT 1"
    )
    q_end = (
        "SELECT close FROM market_data_daily"
        " WHERE symbol=? AND date <= ?"
        " ORDER BY date DESC LIMIT 1"
    )
    spy_start = conn.execute(q_start, ["SPY", start_date]).fetchone()
    spy_end = conn.execute(q_end, ["SPY", end_date]).fetchone()
    tlt_start = conn.execute(q_start, ["TLT", start_date]).fetchone()
    tlt_end = conn.execute(q_end, ["TLT", end_date]).fetchone()

    if not all((spy_start, spy_end, tlt_start, tlt_end)):
        return None

    spy_ret = (float(spy_end[0]) / float(spy_start[0])) - 1.0
    tlt_ret = (float(tlt_end[0]) / float(tlt_start[0])) - 1.0
    return 0.6 * spy_ret + 0.4 * tlt_ret


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


def generate_daily_report(
    conn: duckdb.DuckDBPyConnection,
    target_date: date,
    initial_capital: float,
) -> str:
    """Generate a daily Markdown report."""
    lines: list[str] = []
    lines.append(f"# Daily Report — {target_date.isoformat()}")
    lines.append("")

    # Market Regime
    regime = _get_regime_for_date(conn, target_date)
    lines.append("## Market Regime")
    if regime:
        lines.append(
            f"- Regime: {regime['regime']} (confidence: {regime['confidence']:.2f})"
        )
    else:
        lines.append("- Regime: N/A (no LLM decision recorded)")
    lines.append("")

    # Portfolio Summary
    snap = _get_snapshot_for_date(conn, target_date)
    lines.append("## Portfolio Summary")
    lines.append("")
    if snap:
        nav = snap["nav"]
        cash = snap["cash"]
        cash_pct = (cash / nav * 100) if nav else 0.0
        gross_exp = (snap["gross_exposure"] / nav * 100) if nav else 0.0
        net_exp = (snap["net_exposure"] / nav * 100) if nav else 0.0
        daily_pnl = snap["daily_pnl"]
        total_pnl = snap["total_pnl"]
        total_return = (nav / initial_capital - 1.0) if initial_capital else 0.0

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| NAV | {_fmt_money(nav)} |")
        lines.append(f"| Cash | {_fmt_money(cash)} ({cash_pct:.1f}%) |")
        lines.append(f"| Gross Exposure | {gross_exp:.1f}% |")
        lines.append(f"| Net Exposure | {net_exp:.1f}% |")
        lines.append(f"| Daily P&L | {_fmt_money(daily_pnl)} |")
        lines.append(f"| Total P&L | {_fmt_money(total_pnl)} |")
        lines.append(f"| Total Return | {_fmt_pct(total_return)} |")
    else:
        lines.append("No portfolio snapshot for this date.")
    lines.append("")

    # Positions
    lines.append("## Positions")
    lines.append("")
    if snap:
        positions = _get_positions_for_snapshot(conn, snap["snapshot_id"])
        if positions:
            lines.append(
                "| Symbol | Shares | Avg Cost | Current | P&L % | Weight | Stop Loss |"
            )
            lines.append(
                "|--------|--------|----------|---------|-------|--------|-----------|"
            )
            for p in positions:
                pnl_pct = (
                    ((p["current_price"] - p["avg_cost"]) / p["avg_cost"] * 100)
                    if p["avg_cost"]
                    else 0.0
                )
                stop = _fmt_money(p["stop_loss"]) if p["stop_loss"] else "—"
                lines.append(
                    f"| {p['symbol']} "
                    f"| {p['shares']:.2f} "
                    f"| {_fmt_money(p['avg_cost'])} "
                    f"| {_fmt_money(p['current_price'])} "
                    f"| {pnl_pct:+.2f}% "
                    f"| {p['weight'] * 100:.1f}% "
                    f"| {stop} |"
                )
        else:
            lines.append("No open positions.")
    else:
        lines.append("No snapshot data available.")
    lines.append("")

    # Trades
    lines.append("## Trades")
    lines.append("")
    trades = _get_trades_for_date(conn, target_date)
    if trades:
        lines.append(
            "| Symbol | Action | Shares | Price | Notional | Conviction | Reasoning |"
        )
        lines.append(
            "|--------|--------|--------|-------|----------|------------|-----------|"
        )
        for t in trades:
            reasoning_short = (
                t["reasoning"][:60] + "..."
                if len(t["reasoning"]) > 60
                else t["reasoning"]
            )
            lines.append(
                f"| {t['symbol']} "
                f"| {t['action']} "
                f"| {t['shares']:.2f} "
                f"| {_fmt_money(t['price'])} "
                f"| {_fmt_money(t['notional'])} "
                f"| {t['conviction']} "
                f"| {reasoning_short} |"
            )
    else:
        lines.append("No trades executed today.")
    lines.append("")

    # Performance Metrics
    metrics = compute_performance(conn, initial_capital)
    sortino = _compute_sortino(conn)
    calmar = _compute_calmar(conn, initial_capital)

    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value | Target |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} | > 0.80 |")
    lines.append(f"| Sortino Ratio | {sortino:.2f} | > 1.00 |")
    lines.append(f"| Calmar Ratio | {calmar:.2f} | > 0.50 |")
    lines.append(f"| Max Drawdown | {_fmt_pct(metrics['max_drawdown'])} | < -15% |")
    lines.append(f"| Win Rate | {_fmt_pct(metrics['win_rate'])} | — |")
    lines.append("")

    # Benchmark Comparison
    lines.append("## Benchmark Comparison")
    lines.append("")
    bench_return = _compute_benchmark_return(conn)
    portfolio_return = metrics["total_return"]
    lines.append("| | Portfolio | 60/40 SPY/TLT | Alpha |")
    lines.append("|--|-----------|---------------|-------|")
    if bench_return is not None:
        alpha = portfolio_return - bench_return
        lines.append(
            f"| Total Return | {_fmt_pct(portfolio_return)} "
            f"| {_fmt_pct(bench_return)} "
            f"| {_fmt_pct(alpha)} |"
        )
    else:
        lines.append(f"| Total Return | {_fmt_pct(portfolio_return)} | N/A | N/A |")
    lines.append("")

    # Governance Status
    lines.append("## Governance Status")
    lines.append("")
    try:
        scanner = SurveillanceScanner(load_config())
        report = scanner.run_full_scan(conn)
        severity = report.overall_severity.value.upper()
        lines.append(
            f"- **Overall**: {severity} "
            f"({len(report.checks)} checks, "
            f"{len(report.halt_checks)} halts, "
            f"{len(report.warning_checks)} warnings)"
        )
        if report.halt_checks:
            lines.append("")
            lines.append("**Halt triggers:**")
            for c in report.halt_checks:
                lines.append(f"- [{c.detector}] {c.message}")
        if report.warning_checks:
            lines.append("")
            lines.append("**Warnings:**")
            for c in report.warning_checks:
                lines.append(f"- [{c.detector}] {c.message}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- Governance scan unavailable: {exc}")
    lines.append("")

    # Footer
    ok, _, _msg = verify_chain(conn)
    chain_status = "PASS" if ok else "FAIL"
    lines.append("---")
    lines.append(
        "*Generated automatically from llm-quant DuckDB."
        f" Hash chain verified: [{chain_status}]*"
    )
    lines.append("")

    return "\n".join(lines)


def generate_weekly_report(
    conn: duckdb.DuckDBPyConnection,
    target_date: date,
    initial_capital: float,
) -> str:
    """Generate a weekly Markdown report.

    The week is ISO week containing target_date (Monday to Sunday).
    """
    # Compute ISO week boundaries
    iso_year, iso_week, _ = target_date.isocalendar()
    monday = target_date - timedelta(days=target_date.weekday())
    sunday = monday + timedelta(days=6)

    lines: list[str] = []
    lines.append(f"# Weekly Report — {iso_year}-W{iso_week:02d}")
    lines.append(f"*{monday.isoformat()} to {sunday.isoformat()}*")
    lines.append("")

    # Weekly snapshots
    snapshots = _get_snapshots_for_range(conn, monday, sunday)

    lines.append("## Weekly Summary")
    lines.append("")
    if snapshots:
        first_snap = snapshots[0]
        last_snap = snapshots[-1]
        weekly_return = (
            (last_snap["nav"] / first_snap["nav"] - 1.0) if first_snap["nav"] else 0.0
        )
        weekly_pnl = (
            last_snap["total_pnl"] - first_snap["total_pnl"] + first_snap["daily_pnl"]
        )

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Starting NAV | {_fmt_money(first_snap['nav'])} |")
        lines.append(f"| Ending NAV | {_fmt_money(last_snap['nav'])} |")
        lines.append(f"| Weekly Return | {_fmt_pct(weekly_return)} |")
        lines.append(f"| Weekly P&L | {_fmt_money(weekly_pnl)} |")
        lines.append(
            f"| Total P&L (cumulative) | {_fmt_money(last_snap['total_pnl'])} |"
        )
        cum_ret = last_snap["nav"] / initial_capital - 1.0
        lines.append(f"| Total Return (cumulative) | {_fmt_pct(cum_ret)} |")
    else:
        lines.append("No portfolio snapshots for this week.")
    lines.append("")

    # Daily breakdown
    lines.append("## Daily Breakdown")
    lines.append("")
    if snapshots:
        lines.append("| Date | NAV | Daily P&L | Cumulative P&L |")
        lines.append("|------|-----|-----------|----------------|")
        for s in snapshots:
            lines.append(
                f"| {s['date']} "
                f"| {_fmt_money(s['nav'])} "
                f"| {_fmt_money(s['daily_pnl'])} "
                f"| {_fmt_money(s['total_pnl'])} |"
            )
    else:
        lines.append("No data available.")
    lines.append("")

    # Trades for the week
    trades = _get_trades_for_range(conn, monday, sunday)
    lines.append("## Trades")
    lines.append("")
    if trades:
        lines.append(
            "| Date | Symbol | Action | Shares | Price | Notional | Conviction |"
        )
        lines.append(
            "|------|--------|--------|--------|-------|----------|------------|"
        )
        for t in trades:
            lines.append(
                f"| {t['date']} "
                f"| {t['symbol']} "
                f"| {t['action']} "
                f"| {t['shares']:.2f} "
                f"| {_fmt_money(t['price'])} "
                f"| {_fmt_money(t['notional'])} "
                f"| {t['conviction']} |"
            )
    else:
        lines.append("No trades executed this week.")
    lines.append("")

    # Position changes
    lines.append("## Position Changes")
    lines.append("")
    if snapshots:
        q_snap_id = (
            "SELECT snapshot_id FROM portfolio_snapshots"
            " WHERE date = ? ORDER BY snapshot_id DESC LIMIT 1"
        )
        first_id = conn.execute(q_snap_id, [snapshots[0]["date"]]).fetchone()
        last_id = conn.execute(q_snap_id, [snapshots[-1]["date"]]).fetchone()
        if first_id and last_id:
            start_pos = {
                p["symbol"]: p for p in _get_positions_for_snapshot(conn, first_id[0])
            }
            end_pos = {
                p["symbol"]: p for p in _get_positions_for_snapshot(conn, last_id[0])
            }
            all_symbols = sorted(set(start_pos.keys()) | set(end_pos.keys()))
            if all_symbols:
                lines.append("| Symbol | Start Weight | End Weight | Change |")
                lines.append("|--------|-------------|------------|--------|")
                for sym in all_symbols:
                    sw = start_pos.get(sym, {}).get("weight", 0.0) * 100
                    ew = end_pos.get(sym, {}).get("weight", 0.0) * 100
                    change = ew - sw
                    lines.append(f"| {sym} | {sw:.1f}% | {ew:.1f}% | {change:+.1f}% |")
            else:
                lines.append("No positions held during this period.")
        else:
            lines.append("No snapshot data for position comparison.")
    else:
        lines.append("No data available.")
    lines.append("")

    # Regime history
    regimes = _get_regimes_for_range(conn, monday, sunday)
    lines.append("## Regime History")
    lines.append("")
    if regimes:
        lines.append("| Date | Regime | Confidence |")
        lines.append("|------|--------|------------|")
        for r in regimes:
            lines.append(f"| {r['date']} | {r['regime']} | {r['confidence']:.2f} |")
    else:
        lines.append("No regime data for this week.")
    lines.append("")

    # Footer
    ok, _, _msg = verify_chain(conn)
    chain_status = "PASS" if ok else "FAIL"
    lines.append("---")
    lines.append(
        "*Generated automatically from llm-quant DuckDB."
        f" Hash chain verified: [{chain_status}]*"
    )
    lines.append("")

    return "\n".join(lines)


def generate_monthly_report(
    conn: duckdb.DuckDBPyConnection,
    target_date: date,
    initial_capital: float,
) -> str:
    """Generate a monthly Markdown report for the month of target_date."""
    year = target_date.year
    month = target_date.month
    first_day = date(year, month, 1)
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    lines: list[str] = []
    lines.append(f"# Monthly Report — {year}-{month:02d}")
    lines.append(f"*{first_day.isoformat()} to {last_day.isoformat()}*")
    lines.append("")

    # Monthly Summary
    snapshots = _get_snapshots_for_range(conn, first_day, last_day)
    metrics = compute_performance(conn, initial_capital)

    lines.append("## Monthly Summary")
    lines.append("")
    if snapshots:
        first_snap = snapshots[0]
        last_snap = snapshots[-1]
        monthly_return = (
            (last_snap["nav"] / first_snap["nav"] - 1.0) if first_snap["nav"] else 0.0
        )
        cumulative_return = (
            (last_snap["nav"] / initial_capital - 1.0) if initial_capital else 0.0
        )

        # YTD return
        jan1 = date(year, 1, 1)
        ytd_snap = conn.execute(
            "SELECT nav FROM portfolio_snapshots"
            " WHERE date >= ?"
            " ORDER BY date ASC, snapshot_id ASC LIMIT 1",
            [jan1],
        ).fetchone()
        ytd_start_nav = float(ytd_snap[0]) if ytd_snap else initial_capital
        ytd_return = (last_snap["nav"] / ytd_start_nav - 1.0) if ytd_start_nav else 0.0

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Starting NAV | {_fmt_money(first_snap['nav'])} |")
        lines.append(f"| Ending NAV | {_fmt_money(last_snap['nav'])} |")
        lines.append(f"| Monthly Return | {_fmt_pct(monthly_return)} |")
        lines.append(f"| Cumulative Return | {_fmt_pct(cumulative_return)} |")
        lines.append(f"| YTD Return | {_fmt_pct(ytd_return)} |")
        lines.append(f"| Total P&L | {_fmt_money(last_snap['total_pnl'])} |")
    else:
        lines.append("No portfolio snapshots for this month.")
    lines.append("")

    # Full Metrics Dashboard
    sortino = _compute_sortino(conn)
    calmar = _compute_calmar(conn, initial_capital)

    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value | Target |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| Sharpe Ratio | {metrics['sharpe_ratio']:.2f} | > 0.80 |")
    lines.append(f"| Sortino Ratio | {sortino:.2f} | > 1.00 |")
    lines.append(f"| Calmar Ratio | {calmar:.2f} | > 0.50 |")
    lines.append(f"| Max Drawdown | {_fmt_pct(metrics['max_drawdown'])} | < -15% |")
    lines.append(f"| Win Rate | {_fmt_pct(metrics['win_rate'])} | — |")
    lines.append(f"| Total Trades | {metrics['total_trades']} | — |")
    lines.append(f"| Avg Trade P&L | {_fmt_money(metrics['avg_trade_pnl'])} | — |")
    lines.append("")

    # Trade Statistics by Conviction
    trades = _get_trades_for_range(conn, first_day, last_day)
    lines.append("## Trade Statistics by Conviction")
    lines.append("")
    if trades:
        conviction_groups: dict[str, list[dict]] = {}
        for t in trades:
            conv = t["conviction"] or "unknown"
            conviction_groups.setdefault(conv, []).append(t)

        lines.append("| Conviction | Trades | Total Notional | Avg Notional |")
        lines.append("|------------|--------|----------------|--------------|")
        for conv in sorted(conviction_groups.keys()):
            group = conviction_groups[conv]
            total_not = sum(t["notional"] for t in group)
            avg_not = total_not / len(group) if group else 0.0
            lines.append(
                f"| {conv} | {len(group)} "
                f"| {_fmt_money(total_not)} "
                f"| {_fmt_money(avg_not)} |"
            )
    else:
        lines.append("No trades executed this month.")
    lines.append("")

    # All Trades
    lines.append("## Trades")
    lines.append("")
    if trades:
        lines.append(
            "| Date | Symbol | Action | Shares | Price | Notional | Conviction |"
        )
        lines.append(
            "|------|--------|--------|--------|-------|----------|------------|"
        )
        for t in trades:
            lines.append(
                f"| {t['date']} "
                f"| {t['symbol']} "
                f"| {t['action']} "
                f"| {t['shares']:.2f} "
                f"| {_fmt_money(t['price'])} "
                f"| {_fmt_money(t['notional'])} "
                f"| {t['conviction']} |"
            )
    else:
        lines.append("No trades executed this month.")
    lines.append("")

    # Top/Bottom Performers
    lines.append("## Top/Bottom Performers")
    lines.append("")
    if snapshots:
        last_snap_id = conn.execute(
            "SELECT snapshot_id FROM portfolio_snapshots"
            " WHERE date = ?"
            " ORDER BY snapshot_id DESC LIMIT 1",
            [snapshots[-1]["date"]],
        ).fetchone()
        if last_snap_id:
            positions = _get_positions_for_snapshot(conn, last_snap_id[0])
            if positions:
                sorted_by_pnl = sorted(
                    positions,
                    key=lambda p: (
                        (p["current_price"] - p["avg_cost"]) / p["avg_cost"]
                        if p["avg_cost"]
                        else 0.0
                    ),
                    reverse=True,
                )
                lines.append("| Symbol | P&L % | Weight | Market Value |")
                lines.append("|--------|-------|--------|--------------|")
                for p in sorted_by_pnl:
                    pnl_pct = (
                        ((p["current_price"] - p["avg_cost"]) / p["avg_cost"] * 100)
                        if p["avg_cost"]
                        else 0.0
                    )
                    lines.append(
                        f"| {p['symbol']} "
                        f"| {pnl_pct:+.2f}% "
                        f"| {p['weight'] * 100:.1f}% "
                        f"| {_fmt_money(p['market_value'])} |"
                    )
            else:
                lines.append("No positions at month end.")
        else:
            lines.append("No snapshot data for month end.")
    else:
        lines.append("No data available.")
    lines.append("")

    # Regime Breakdown
    regimes = _get_regimes_for_range(conn, first_day, last_day)
    lines.append("## Regime Breakdown")
    lines.append("")
    if regimes:
        regime_counts: dict[str, int] = {}
        for r in regimes:
            regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
        total_days = sum(regime_counts.values())
        lines.append("| Regime | Days | % of Month |")
        lines.append("|--------|------|------------|")
        for regime, count in sorted(regime_counts.items()):
            pct = count / total_days * 100 if total_days else 0.0
            lines.append(f"| {regime} | {count} | {pct:.1f}% |")
    else:
        lines.append("No regime data for this month.")
    lines.append("")

    # Benchmark Comparison
    lines.append("## Benchmark Comparison")
    lines.append("")
    bench_return = _compute_benchmark_return(conn)
    portfolio_return = metrics["total_return"]
    lines.append("| | Portfolio | 60/40 SPY/TLT | Alpha |")
    lines.append("|--|-----------|---------------|-------|")
    if bench_return is not None:
        alpha = portfolio_return - bench_return
        lines.append(
            f"| Total Return | {_fmt_pct(portfolio_return)} "
            f"| {_fmt_pct(bench_return)} "
            f"| {_fmt_pct(alpha)} |"
        )
    else:
        lines.append(f"| Total Return | {_fmt_pct(portfolio_return)} | N/A | N/A |")
    lines.append("")

    # Footer
    ok, _, _msg = verify_chain(conn)
    chain_status = "PASS" if ok else "FAIL"
    lines.append("---")
    lines.append(
        "*Generated automatically from llm-quant DuckDB."
        f" Hash chain verified: [{chain_status}]*"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_report(content: str, output_path: Path) -> None:
    """Write report content to file, creating directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown reports from llm-quant DuckDB."
    )
    parser.add_argument(
        "period",
        choices=["daily", "weekly", "monthly"],
        help="Report period type.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (default: today).",
    )
    args = parser.parse_args()

    # Parse date
    target_date = (
        date.fromisoformat(args.date) if args.date else datetime.now(tz=UTC).date()
    )

    # Load config
    config = load_config()
    db_path = _resolve_db_path(config)
    initial_capital = config.general.initial_capital

    _ensure_db(db_path)
    conn = get_connection(db_path)

    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"

    try:
        if args.period == "daily":
            content = generate_daily_report(conn, target_date, initial_capital)
            filename = f"{target_date.isoformat()}.md"
            output_path = reports_dir / "daily" / filename

        elif args.period == "weekly":
            iso_year, iso_week, _ = target_date.isocalendar()
            content = generate_weekly_report(conn, target_date, initial_capital)
            filename = f"{iso_year}-W{iso_week:02d}.md"
            output_path = reports_dir / "weekly" / filename

        elif args.period == "monthly":
            content = generate_monthly_report(conn, target_date, initial_capital)
            filename = f"{target_date.year}-{target_date.month:02d}.md"
            output_path = reports_dir / "monthly" / filename

        _write_report(content, output_path)
        print(str(output_path))

    finally:
        conn.close()


if __name__ == "__main__":
    main()

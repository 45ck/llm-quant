"""Build market context for Claude Code to analyze.

Fetches data if stale, computes indicators, loads portfolio,
and prints a formatted markdown context to stdout.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/build_context.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/build_context.py --pod momo
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Fix Windows cp1252 encoding crashes when printing Polars DataFrames or
# yfinance progress bars that contain characters outside cp1252. Only rewire
# when executed as a script — importing build_context from tests must not
# replace the pytest-owned stdout/stderr wrappers.
if (
    __name__ == "__main__"
    and sys.platform == "win32"
    and os.environ.get("PYTHONIOENCODING") is None
):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl

from llm_quant.brain.context import build_market_context
from llm_quant.brain.prompts import load_system_prompt, render_decision_prompt
from llm_quant.config import load_config_for_pod
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators
from llm_quant.data.store import get_latest_date, upsert_market_data
from llm_quant.data.universe import get_all_fetch_symbols, get_tradeable_symbols
from llm_quant.db.schema import get_connection, init_schema
from llm_quant.surveillance.scanner import SurveillanceScanner
from llm_quant.trading.portfolio import Portfolio

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)


def _ensure_db(db_path: str) -> None:
    """Initialize DB schema if it doesn't exist."""
    path = Path(db_path)
    if not path.exists():
        init_schema(db_path)


def _data_is_stale(conn, symbols: list[str]) -> bool:
    """Check if market data needs refreshing (>1 trading day old)."""
    today = datetime.now(tz=UTC).date()
    # On weekends, latest data from Friday is acceptable
    if today.weekday() == 5:  # Saturday
        threshold = today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        threshold = today - timedelta(days=2)
    else:
        threshold = today - timedelta(days=1)

    # Check a representative symbol
    for check_symbol in ("SPY", symbols[0] if symbols else "SPY"):
        latest = get_latest_date(conn, check_symbol)
        if latest is None:
            return True
        if latest < threshold:
            return True
    return False


def _fetch_and_store(conn, config) -> None:
    """Fetch fresh data from Yahoo Finance, compute indicators, store."""
    symbols = get_all_fetch_symbols(config)
    print("Fetching market data...", file=sys.stderr)

    df = fetch_ohlcv(
        symbols=symbols,
        lookback_days=config.data.lookback_days,
        timeout=config.data.fetch_timeout,
    )
    if len(df) == 0:
        print("WARNING: No data fetched from Yahoo Finance", file=sys.stderr)
        return

    print(f"Computing indicators for {len(df)} rows...", file=sys.stderr)
    df = compute_indicators(df)
    upsert_market_data(conn, df)
    print(f"Stored {len(df)} rows with indicators", file=sys.stderr)


COT_STALENESS_THRESHOLD_DAYS = 10  # COT data older than this triggers exclusion

# Whipsaw R3 — snapshot-gap soft block thresholds (see
# docs/investigations/apr01-whipsaw.md). A 6-calendar-day snapshot gap on
# 2026-04-01 was the enabling cause of the mass-close whipsaw: the LLM
# came back online after an absence and acted on a cold book. Advisory at
# >=3 days warns the PM to reconcile; at >=7 days we escalate to halt so
# /trade treats it as a sells-only gate.
SNAPSHOT_GAP_WARNING_DAYS = 3
SNAPSHOT_GAP_HALT_DAYS = 7


def _check_snapshot_staleness(conn, pod_id: str, today) -> dict | None:
    """Return a governance advisory dict if the last snapshot is too old.

    Queries ``portfolio_snapshots`` for the most recent ``date`` tied to
    *pod_id* (falls back to any pod when the column is missing) and compares
    to *today*. Returns ``None`` when no staleness exists or no snapshot has
    ever been written (fresh systems shouldn't block).

    Output shape matches the governance block's ``halt_details`` /
    ``warning_details`` entries so downstream consumers see it alongside
    surveillance scans.
    """
    try:
        cols = [c[0] for c in conn.execute("DESCRIBE portfolio_snapshots").fetchall()]
    except Exception as exc:
        logger.debug("DESCRIBE portfolio_snapshots failed: %s", exc)
        return None

    try:
        if "pod_id" in cols:
            row = conn.execute(
                """
                SELECT MAX(date) FROM portfolio_snapshots WHERE pod_id = ?
                """,
                [pod_id],
            ).fetchone()
        else:
            row = conn.execute("SELECT MAX(date) FROM portfolio_snapshots").fetchone()
    except Exception as exc:
        logger.debug("portfolio_snapshots lookup failed: %s", exc)
        return None

    last_date = row[0] if row else None
    if last_date is None:
        # No snapshot history — nothing to warn about (fresh system).
        return None

    # Normalise to date (DuckDB may return date or string depending on version).
    from datetime import date as date_cls

    if isinstance(last_date, str):
        last_date_obj = date_cls.fromisoformat(last_date[:10])
    elif isinstance(last_date, date_cls):
        last_date_obj = last_date
    else:
        try:
            last_date_obj = date_cls.fromisoformat(str(last_date)[:10])
        except (TypeError, ValueError):
            return None

    days_stale = (today - last_date_obj).days
    if days_stale < SNAPSHOT_GAP_WARNING_DAYS:
        return None

    if days_stale >= SNAPSHOT_GAP_HALT_DAYS:
        severity = "halt"
        message = (
            f"Last snapshot is {days_stale} days old "
            f"(>= {SNAPSHOT_GAP_HALT_DAYS} day halt threshold). "
            "Treat this session as sells-only until portfolio state is "
            "reconciled. Do NOT open new positions on a cold book "
            "(whipsaw R3)."
        )
    else:
        severity = "warning"
        message = (
            f"Last snapshot is {days_stale} days old "
            f"(>= {SNAPSHOT_GAP_WARNING_DAYS} day advisory threshold). "
            "Reconcile portfolio state before trading new signals. "
            "Consider HOLD or review-only cycle (whipsaw R3)."
        )

    return {
        "detector": "snapshot_staleness",
        "severity": severity,
        "days_stale": days_stale,
        "last_snapshot_date": last_date_obj.isoformat(),
        "message": message,
    }


def _vix_direction_of_travel(conn) -> tuple[float | None, float | None]:
    """Return (vix_change_5d, vix_pct_from_20d_high) from market_data_daily.

    Both default to None when insufficient history exists.
    """
    vix_rows: list[tuple] = []
    for vix_symbol in ("^VIX", "VIX"):
        rows = conn.execute(
            """
            SELECT date, close
            FROM market_data_daily
            WHERE symbol = ? AND close IS NOT NULL
            ORDER BY date DESC
            LIMIT 25
            """,
            [vix_symbol],
        ).fetchall()
        if rows:
            vix_rows = rows
            break

    if not vix_rows:
        return None, None

    vix_df = pl.DataFrame(
        vix_rows, schema={"date": pl.Date, "close": pl.Float64}, orient="row"
    ).sort("date", descending=False)
    if vix_df.height < 1:
        return None, None

    current_vix = float(vix_df["close"][-1])
    change_5d: float | None = None
    pct_from_high: float | None = None

    if vix_df.height >= 6:
        change_5d = round(current_vix - float(vix_df["close"][-6]), 2)

    if vix_df.height >= 2:
        max_20d = float(vix_df.tail(20)["close"].max())
        if max_20d > 0:
            pct_from_high = round(100.0 * current_vix / max_20d, 1)

    return change_5d, pct_from_high


def _spy_drawdown_from_20d_high(conn) -> float | None:
    """Return (SPY close / 20d rolling max SPY close) - 1, or None."""
    spy_rows = conn.execute(
        """
        SELECT date, close
        FROM market_data_daily
        WHERE symbol = 'SPY' AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 25
        """
    ).fetchall()
    if not spy_rows:
        return None

    spy_df = pl.DataFrame(
        spy_rows, schema={"date": pl.Date, "close": pl.Float64}, orient="row"
    ).sort("date", descending=False)
    if spy_df.height < 2:
        return None

    current_spy = float(spy_df["close"][-1])
    max_20d = float(spy_df.tail(20)["close"].max())
    if max_20d <= 0:
        return None
    return round(current_spy / max_20d - 1.0, 4)


def _days_since_regime_flip(conn, current_regime: str | None) -> int | None:
    """Count sessions since llm_decisions.market_regime was last different.

    Uses distinct SPY trading dates as the session clock. Returns None when
    there is no regime history or the lookup fails.
    """
    if current_regime is None:
        return None
    try:
        row = conn.execute(
            """
            SELECT MAX(date)
            FROM llm_decisions
            WHERE market_regime IS NOT NULL
              AND market_regime != ?
            """,
            [current_regime],
        ).fetchone()
        last_flip_date = row[0] if row else None

        if last_flip_date is not None:
            count_row = conn.execute(
                """
                SELECT COUNT(DISTINCT date)
                FROM market_data_daily
                WHERE symbol = 'SPY' AND date > ?
                """,
                [last_flip_date],
            ).fetchone()
        else:
            # No differing regime on record — fall back to total regime-labelled
            # sessions so the LLM sees how long the current label has held.
            count_row = conn.execute(
                """
                SELECT COUNT(DISTINCT date)
                FROM llm_decisions
                WHERE market_regime IS NOT NULL
                """
            ).fetchone()

        if count_row and count_row[0] is not None and count_row[0] > 0:
            return int(count_row[0])
        return None
    except Exception as exc:
        logger.debug("days_since_regime_flip lookup failed: %s", exc)
        return None


def _compute_direction_of_travel(conn, current_regime: str | None) -> dict:
    """Compute direction-of-travel fields for the macro context (whipsaw R1).

    Adds four fields that prevent the LLM from misreading a post-spike VIX
    reading as a fresh panic (root cause of the 2026-04-01 whipsaw):

      - vix_change_5d: current VIX - VIX 5 trading days ago. Negative value
        means VIX is receding (spike already peaked); positive means climbing.
      - vix_pct_from_20d_high: 100 * current VIX / 20-day rolling max of VIX.
        Values << 100 mean the spike is in the rearview mirror.
      - spy_drawdown_from_20d_high: (current SPY / 20d max SPY) - 1. Already
        negative => SPY has sold off; a near-zero value with falling VIX
        signals a recovery already underway.
      - days_since_regime_flip: trading sessions since the current market_regime
        label was last different in llm_decisions. Large value means the regime
        call is stable; small value means we just flipped (whipsaw risk).

    All fields default to None on insufficient data; callers must handle None.
    """
    vix_change_5d, vix_pct_from_20d_high = _vix_direction_of_travel(conn)
    return {
        "vix_change_5d": vix_change_5d,
        "vix_pct_from_20d_high": vix_pct_from_20d_high,
        "spy_drawdown_from_20d_high": _spy_drawdown_from_20d_high(conn),
        "days_since_regime_flip": _days_since_regime_flip(conn, current_regime),
    }


def _check_cot_staleness(conn) -> tuple[bool, str | None, int]:
    """Check if the most recent COT record in cot_weekly is stale.

    Returns (cot_stale, last_cot_date_str, days_stale).
    If cot_weekly table is empty or missing, returns (False, None, 0) —
    empty table is handled separately, not treated as stale.
    """
    try:
        row = conn.execute("SELECT MAX(report_date) FROM cot_weekly").fetchone()
    except Exception:
        return False, None, 0

    if row is None or row[0] is None:
        return False, None, 0

    last_cot_date = row[0]
    last_cot_date_str = str(last_cot_date)

    today = datetime.now(tz=UTC).date()
    # DuckDB may return a date object or a string depending on version
    if hasattr(last_cot_date, "timetuple"):
        import datetime as dt_mod

        last_date_obj = (
            last_cot_date
            if isinstance(last_cot_date, dt_mod.date)
            else dt_mod.date.fromisoformat(last_cot_date_str[:10])
        )
    else:
        from datetime import date as date_cls

        last_date_obj = date_cls.fromisoformat(last_cot_date_str[:10])

    days_stale = (today - last_date_obj).days

    if days_stale > COT_STALENESS_THRESHOLD_DAYS:
        logger.warning(
            "COT data stale: last update %s, %d days ago. Excluding COT signals.",
            last_cot_date_str,
            days_stale,
        )
        return True, last_cot_date_str, days_stale

    return False, last_cot_date_str, days_stale


def main() -> None:
    parser = argparse.ArgumentParser(description="Build market context for LLM trading")
    parser.add_argument("--pod", default="default", help="Pod ID to build context for")
    args = parser.parse_args()
    pod_id = args.pod

    config = load_config_for_pod(pod_id)
    db_path = config.general.db_path

    # Resolve relative db_path against project root
    project_root = Path(__file__).resolve().parent.parent
    if not Path(db_path).is_absolute():
        db_path = str(project_root / db_path)

    _ensure_db(db_path)
    conn = get_connection(db_path)

    try:
        symbols = get_tradeable_symbols(config)

        # Fetch if stale
        if _data_is_stale(conn, symbols):
            _fetch_and_store(conn, config)

        # Load portfolio
        portfolio = Portfolio.from_db(
            conn, config.general.initial_capital, pod_id=pod_id
        )

        # Update prices from latest market data
        prices: dict[str, float] = {}
        for symbol in list(portfolio.positions.keys()) + symbols:
            row = conn.execute(
                "SELECT close FROM market_data_daily"
                " WHERE symbol = ? ORDER BY date DESC"
                " LIMIT 1",
                [symbol],
            ).fetchone()
            if row and row[0] is not None:
                prices[symbol] = float(row[0])

        portfolio.update_prices(prices)

        # Check COT data staleness before building context
        cot_stale, last_cot_date, days_stale = _check_cot_staleness(conn)

        # Build context
        portfolio_state = portfolio.to_snapshot_dict()
        context = build_market_context(
            conn, portfolio_state, config, cot_stale=cot_stale
        )

        # Whipsaw R1: direction-of-travel fields so the LLM doesn't misread a
        # post-spike VIX reading as fresh panic (see
        # docs/investigations/apr01-whipsaw.md). Computed BEFORE prompt render
        # so the Jinja template in config/prompts/trader_decision.md can
        # reference them.
        current_regime_str = (
            context.market_regime.value
            if hasattr(context.market_regime, "value")
            else str(context.market_regime)
        )
        direction_of_travel = _compute_direction_of_travel(conn, current_regime_str)
        context.vix_change_5d = direction_of_travel["vix_change_5d"]
        context.vix_pct_from_20d_high = direction_of_travel["vix_pct_from_20d_high"]
        context.spy_drawdown_from_20d_high = direction_of_travel[
            "spy_drawdown_from_20d_high"
        ]
        context.days_since_regime_flip = direction_of_travel["days_since_regime_flip"]

        # Load system prompt
        system_prompt = load_system_prompt()

        # Render decision prompt
        decision_prompt = render_decision_prompt(context)

        # Run lightweight governance scan
        governance_status = {"overall_severity": "ok", "halts": 0, "warnings": 0}
        try:
            scanner = SurveillanceScanner(config)
            report = scanner.run_full_scan(conn)
            scanner.persist_scan(conn, report)
            governance_status = {
                "overall_severity": report.overall_severity.value,
                "halts": len(report.halt_checks),
                "warnings": len(report.warning_checks),
                "total_checks": len(report.checks),
                "halt_details": [
                    {"detector": c.detector, "message": c.message}
                    for c in report.halt_checks
                ],
                "warning_details": [
                    {"detector": c.detector, "message": c.message}
                    for c in report.warning_checks
                ],
            }
        except Exception as exc:
            print(f"WARNING: Governance scan failed: {exc}", file=sys.stderr)

        # Whipsaw R3 — snapshot-gap soft block (see
        # docs/investigations/apr01-whipsaw.md). When the last portfolio
        # snapshot is stale we inject a dedicated advisory into the governance
        # block so the LLM doesn't trade on a cold book.
        today_date = datetime.now(tz=UTC).date()
        staleness = _check_snapshot_staleness(conn, pod_id, today_date)
        if staleness is not None:
            severity = staleness["severity"]
            detector = staleness["detector"]
            message = staleness["message"]
            days_stale = staleness["days_stale"]

            if severity == "halt":
                governance_status.setdefault("halt_details", []).append(
                    {
                        "detector": detector,
                        "message": message,
                        "days_stale": days_stale,
                    }
                )
                governance_status["halts"] = int(governance_status.get("halts", 0)) + 1
                # Halt dominates: escalate overall_severity unless already halt.
                governance_status["overall_severity"] = "halt"
            else:
                governance_status.setdefault("warning_details", []).append(
                    {
                        "detector": detector,
                        "message": message,
                        "days_stale": days_stale,
                    }
                )
                governance_status["warnings"] = (
                    int(governance_status.get("warnings", 0)) + 1
                )
                # Warning escalates "ok" but never downgrades halt.
                if governance_status.get("overall_severity", "ok") == "ok":
                    governance_status["overall_severity"] = "warning"

            # Bump total_checks if present so consumer math stays consistent.
            if "total_checks" in governance_status:
                governance_status["total_checks"] = (
                    int(governance_status["total_checks"]) + 1
                )

            governance_status["snapshot_staleness"] = staleness

        # Output structured data for Claude Code
        output = {
            "pod_id": pod_id,
            "system_prompt": system_prompt,
            "decision_prompt": decision_prompt,
            "portfolio_summary": {
                "nav": context.nav,
                "cash": context.cash,
                "cash_pct": context.cash_pct,
                "positions_count": len(context.positions),
                "gross_exposure_pct": context.gross_exposure_pct,
                "net_exposure_pct": context.net_exposure_pct,
            },
            "macro": {
                "vix": context.vix,
                "yield_spread": context.yield_spread,
                "spy_trend": context.spy_trend,
                # Whipsaw R1 — direction-of-travel fields:
                "vix_change_5d": direction_of_travel["vix_change_5d"],
                "vix_pct_from_20d_high": direction_of_travel["vix_pct_from_20d_high"],
                "spy_drawdown_from_20d_high": direction_of_travel[
                    "spy_drawdown_from_20d_high"
                ],
                "days_since_regime_flip": direction_of_travel["days_since_regime_flip"],
            },
            "governance": governance_status,
            "date": str(context.date),
            "cot_status": {
                "stale": cot_stale,
                "last_update": last_cot_date,
                "days_stale": days_stale,
            },
        }

        print(json.dumps(output, indent=2))

    finally:
        conn.close()


if __name__ == "__main__":
    main()

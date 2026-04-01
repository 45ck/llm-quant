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
# yfinance progress bars that contain characters outside cp1252.
if sys.platform == "win32" and os.environ.get("PYTHONIOENCODING") is None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

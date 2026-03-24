"""Build market context for Claude Code to analyze.

Fetches data if stale, computes indicators, loads portfolio,
and prints a formatted markdown context to stdout.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/build_context.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.config import load_config
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators
from llm_quant.data.store import get_latest_date, upsert_market_data
from llm_quant.data.universe import get_tradeable_symbols
from llm_quant.db.schema import get_connection, init_schema
from llm_quant.brain.context import build_market_context
from llm_quant.brain.prompts import load_system_prompt, render_decision_prompt
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
    today = date.today()
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
    symbols = get_tradeable_symbols(config)
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


def main() -> None:
    config = load_config()
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
        portfolio = Portfolio.from_db(conn, config.general.initial_capital)

        # Update prices from latest market data
        prices: dict[str, float] = {}
        for symbol in list(portfolio.positions.keys()) + symbols:
            row = conn.execute(
                "SELECT close FROM market_data_daily WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                [symbol],
            ).fetchone()
            if row and row[0] is not None:
                prices[symbol] = float(row[0])

        portfolio.update_prices(prices)

        # Build context
        portfolio_state = portfolio.to_snapshot_dict()
        context = build_market_context(conn, portfolio_state, config)

        # Load system prompt
        system_prompt = load_system_prompt()

        # Render decision prompt
        decision_prompt = render_decision_prompt(context)

        # Output structured data for Claude Code
        output = {
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
            "date": str(context.date),
        }

        print(json.dumps(output, indent=2))

    finally:
        conn.close()


if __name__ == "__main__":
    main()

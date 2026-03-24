"""Append-only trade ledger and portfolio snapshot persistence.

All writes go to DuckDB via the connection obtained from
``llm_quant.db.schema.get_connection``.  The module never deletes or
updates existing rows – every call *appends* new records, preserving a
full audit trail.
"""

from __future__ import annotations

import logging
from datetime import date

import duckdb

from llm_quant.db.integrity import compute_trade_hash, get_latest_hash
from llm_quant.trading.executor import ExecutedTrade
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade logging
# ---------------------------------------------------------------------------

def log_trades(
    conn: duckdb.DuckDBPyConnection,
    trades: list[ExecutedTrade],
    trade_date: date,
    decision_id: int | None = None,
) -> list[int]:
    """Persist executed trades to the ``trades`` table.

    Each trade is assigned a new ``trade_id`` from ``seq_trade_id``.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    trades:
        Trades to record.
    trade_date:
        Date on which the trades were executed (session date).
    decision_id:
        Optional FK linking back to the ``llm_decisions`` row that
        produced this batch.

    Returns
    -------
    list[int]
        The ``trade_id`` values assigned to the inserted rows, in the
        same order as *trades*.
    """
    trade_ids: list[int] = []
    prev_hash = get_latest_hash(conn)

    for trade in trades:
        row = conn.execute("SELECT nextval('seq_trade_id')").fetchone()
        assert row is not None
        trade_id: int = row[0]

        # DuckDB DEFAULT fills created_at, so we fetch the timestamp
        # after a dummy-free insert by computing it ourselves.
        conn.execute(
            """
            INSERT INTO trades (
                trade_id, date, symbol, action, shares, price,
                notional, conviction, reasoning, llm_decision_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trade_id,
                trade_date,
                trade.symbol,
                trade.action,
                trade.shares,
                trade.price,
                trade.notional,
                trade.conviction,
                trade.reasoning,
                decision_id,
            ],
        )

        # Retrieve the server-generated created_at, then compute hash
        created_row = conn.execute(
            "SELECT created_at FROM trades WHERE trade_id = ?", [trade_id]
        ).fetchone()
        assert created_row is not None
        created_at = created_row[0]

        row_hash = compute_trade_hash(
            prev_hash, trade_id, trade_date, trade.symbol, trade.action,
            trade.shares, trade.price, trade.notional,
            trade.conviction, trade.reasoning, decision_id, created_at,
        )

        conn.execute(
            "UPDATE trades SET prev_hash = ?, row_hash = ? WHERE trade_id = ?",
            [prev_hash, row_hash, trade_id],
        )

        prev_hash = row_hash
        trade_ids.append(trade_id)
        logger.debug(
            "Logged trade %d: %s %s %.4f shares @ %.4f",
            trade_id,
            trade.action,
            trade.symbol,
            trade.shares,
            trade.price,
        )

    if trade_ids:
        conn.commit()
        logger.info(
            "Persisted %d trade(s) for %s (ids=%s)",
            len(trade_ids),
            trade_date,
            trade_ids,
        )

    return trade_ids


# ---------------------------------------------------------------------------
# Portfolio snapshots
# ---------------------------------------------------------------------------

def save_portfolio_snapshot(
    conn: duckdb.DuckDBPyConnection,
    portfolio: Portfolio,
    trade_date: date,
    daily_pnl: float | None = None,
) -> int:
    """Save the current portfolio state to ``portfolio_snapshots`` and
    ``positions``.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    portfolio:
        Portfolio whose state should be persisted.
    trade_date:
        The trading date for the snapshot.
    daily_pnl:
        Optional daily P&L figure.  If *None*, ``NULL`` is stored.

    Returns
    -------
    int
        The assigned ``snapshot_id``.
    """
    row = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()
    assert row is not None
    snapshot_id: int = row[0]

    nav = portfolio.nav

    conn.execute(
        """
        INSERT INTO portfolio_snapshots (
            snapshot_id, date, nav, cash,
            gross_exposure, net_exposure,
            total_pnl, daily_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            snapshot_id,
            trade_date,
            nav,
            portfolio.cash,
            portfolio.gross_exposure,
            portfolio.net_exposure,
            portfolio.total_pnl,
            daily_pnl,
        ],
    )

    # Persist individual positions
    for pos in portfolio.positions.values():
        weight = (pos.market_value / nav) if nav else 0.0
        conn.execute(
            """
            INSERT INTO positions (
                snapshot_id, symbol, shares, avg_cost,
                current_price, market_value, unrealized_pnl,
                weight, stop_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_id,
                pos.symbol,
                pos.shares,
                pos.avg_cost,
                pos.current_price,
                pos.market_value,
                pos.unrealized_pnl,
                weight,
                pos.stop_loss,
            ],
        )

    conn.commit()
    logger.info(
        "Saved snapshot %d for %s: NAV=%.2f, cash=%.2f, %d position(s)",
        snapshot_id,
        trade_date,
        nav,
        portfolio.cash,
        len(portfolio.positions),
    )
    return snapshot_id


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_recent_trades(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 20,
) -> list[dict]:
    """Return the most recent trades as a list of dicts.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    limit:
        Maximum number of rows to return (most recent first).

    Returns
    -------
    list[dict]
        Each dict mirrors a row in the ``trades`` table.
    """
    result = conn.execute(
        """
        SELECT
            trade_id,
            date,
            symbol,
            action,
            shares,
            price,
            notional,
            conviction,
            reasoning,
            llm_decision_id,
            created_at
        FROM trades
        ORDER BY date DESC, trade_id DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()

    columns = [
        "trade_id",
        "date",
        "symbol",
        "action",
        "shares",
        "price",
        "notional",
        "conviction",
        "reasoning",
        "llm_decision_id",
        "created_at",
    ]

    trades: list[dict] = []
    for row in result:
        trades.append(dict(zip(columns, row)))

    logger.debug("Fetched %d recent trade(s).", len(trades))
    return trades


def get_portfolio_history(
    conn: duckdb.DuckDBPyConnection,
    days: int = 30,
) -> list[dict]:
    """Return portfolio snapshots for the last *days* calendar days.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    days:
        Look-back window in calendar days.

    Returns
    -------
    list[dict]
        Each dict mirrors a row in ``portfolio_snapshots``, ordered by
        date ascending.
    """
    result = conn.execute(
        """
        SELECT
            snapshot_id,
            date,
            nav,
            cash,
            gross_exposure,
            net_exposure,
            total_pnl,
            daily_pnl,
            created_at
        FROM portfolio_snapshots
        WHERE date >= CURRENT_DATE - INTERVAL ? DAY
        ORDER BY date ASC, snapshot_id ASC
        """,
        [days],
    ).fetchall()

    columns = [
        "snapshot_id",
        "date",
        "nav",
        "cash",
        "gross_exposure",
        "net_exposure",
        "total_pnl",
        "daily_pnl",
        "created_at",
    ]

    history: list[dict] = []
    for row in result:
        history.append(dict(zip(columns, row)))

    logger.debug("Fetched %d snapshot(s) over last %d day(s).", len(history), days)
    return history

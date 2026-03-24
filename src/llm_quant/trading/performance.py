"""Performance analytics computed from portfolio history.

Uses Polars for vectorised calculations over the ``portfolio_snapshots``
and ``trades`` tables stored in DuckDB.
"""

from __future__ import annotations

import logging
import math

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

# Annualisation factor (trading days per year)
_TRADING_DAYS: int = 252


def compute_performance(
    conn: duckdb.DuckDBPyConnection,
    initial_capital: float = 100_000.0,
) -> dict:
    """Compute headline performance metrics.

    Reads from ``portfolio_snapshots`` and ``trades`` tables.  If there is
    insufficient data (no snapshots or fewer than two data points) the
    function returns a dict of zero / default values so callers never need
    to guard against missing keys.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    initial_capital:
        Starting capital used to compute total return.

    Returns
    -------
    dict
        Keys: ``total_return``, ``sharpe_ratio``, ``max_drawdown``,
        ``win_rate``, ``total_trades``, ``avg_trade_pnl``,
        ``latest_nav``, ``total_pnl``.
    """
    defaults: dict = {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "avg_trade_pnl": 0.0,
        "latest_nav": initial_capital,
        "total_pnl": 0.0,
    }

    # ------------------------------------------------------------------
    # 1. Load portfolio snapshots into Polars
    # ------------------------------------------------------------------
    snap_rows = conn.execute(
        """
        SELECT date, nav, daily_pnl
        FROM portfolio_snapshots
        ORDER BY date ASC, snapshot_id ASC
        """
    ).fetchall()

    if not snap_rows:
        logger.info("No portfolio snapshots – returning default metrics.")
        return defaults

    snap_df = pl.DataFrame(
        {
            "date": [r[0] for r in snap_rows],
            "nav": [float(r[1]) for r in snap_rows],
            "daily_pnl": [
                float(r[2]) if r[2] is not None else None
                for r in snap_rows
            ],
        }
    )

    # Keep one row per date (last snapshot of each day)
    snap_df = snap_df.group_by("date").agg(
        pl.col("nav").last().alias("nav"),
        pl.col("daily_pnl").last().alias("daily_pnl"),
    ).sort("date")

    latest_nav: float = snap_df["nav"][-1]
    total_pnl: float = latest_nav - initial_capital
    total_return: float = (latest_nav / initial_capital) - 1.0 if initial_capital else 0.0

    # ------------------------------------------------------------------
    # 2. Daily returns & Sharpe ratio
    # ------------------------------------------------------------------
    sharpe_ratio: float = 0.0

    if snap_df.height >= 2:
        returns_df = snap_df.with_columns(
            (pl.col("nav") / pl.col("nav").shift(1) - 1.0).alias("daily_return")
        ).drop_nulls("daily_return")

        if returns_df.height >= 2:
            mean_ret: float = returns_df["daily_return"].mean()  # type: ignore[assignment]
            std_ret: float = returns_df["daily_return"].std()    # type: ignore[assignment]

            if std_ret is not None and std_ret > 0.0:
                sharpe_ratio = (mean_ret / std_ret) * math.sqrt(_TRADING_DAYS)

    # ------------------------------------------------------------------
    # 3. Maximum drawdown
    # ------------------------------------------------------------------
    max_drawdown: float = 0.0

    if snap_df.height >= 2:
        dd_df = snap_df.with_columns(
            pl.col("nav").cum_max().alias("peak")
        ).with_columns(
            ((pl.col("nav") - pl.col("peak")) / pl.col("peak")).alias("drawdown")
        )
        min_dd: float = dd_df["drawdown"].min()  # type: ignore[assignment]
        if min_dd is not None:
            max_drawdown = min_dd  # negative value (or zero)

    # ------------------------------------------------------------------
    # 4. Trade-level statistics
    # ------------------------------------------------------------------
    total_trades: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0

    trade_rows = conn.execute(
        """
        SELECT
            t.symbol,
            t.action,
            t.shares,
            t.price,
            t.notional
        FROM trades t
        ORDER BY t.date ASC, t.trade_id ASC
        """
    ).fetchall()

    total_trades = len(trade_rows)

    if total_trades > 0:
        # Compute per-trade P&L via a simple FIFO approach:
        # For sells / closes, P&L = (sell_price - avg_buy_price) * shares
        # We build a cost-basis tracker per symbol.
        cost_basis: dict[str, float] = {}  # symbol -> avg cost per share
        pnl_list: list[float] = []

        for row in trade_rows:
            symbol: str = row[0]
            action: str = row[1]
            shares: float = float(row[2])
            price: float = float(row[3])

            if action == "buy":
                # Update cost basis (running weighted average)
                prev_cost = cost_basis.get(symbol, 0.0)
                # We don't track cumulative share count here; we use a
                # simplified per-trade P&L model.  For a proper FIFO we
                # would need lot tracking.  Instead, record cost for
                # later sell comparison.
                cost_basis[symbol] = price  # last buy price as proxy
            elif action in ("sell", "close"):
                buy_price = cost_basis.get(symbol, price)
                trade_pnl = (price - buy_price) * shares
                pnl_list.append(trade_pnl)

        if pnl_list:
            wins = sum(1 for p in pnl_list if p > 0)
            win_rate = wins / len(pnl_list)
            avg_trade_pnl = sum(pnl_list) / len(pnl_list)

    # ------------------------------------------------------------------
    # 5. Assemble result
    # ------------------------------------------------------------------
    metrics: dict = {
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "latest_nav": round(latest_nav, 2),
        "total_pnl": round(total_pnl, 2),
    }

    logger.info(
        "Performance: return=%.2f%%, sharpe=%.2f, drawdown=%.2f%%, "
        "trades=%d, win_rate=%.1f%%",
        total_return * 100,
        sharpe_ratio,
        max_drawdown * 100,
        total_trades,
        win_rate * 100,
    )

    return metrics

"""Performance analytics computed from portfolio history.

Uses Polars for vectorised calculations over the ``portfolio_snapshots``
and ``trades`` tables stored in DuckDB.
"""

from __future__ import annotations

import logging
import math
import sys
import types
from datetime import date as date_type

import duckdb
import pandas as pd
import polars as pl

# empyrical imports pandas_datareader at module level for benchmark fetching, but
# pandas_datareader 0.9/0.10 breaks on pandas 3.x (deprecate_kwarg signature change).
# We only use empyrical's stats functions, not its data-fetching helpers, so we
# stub the broken sub-module before importing empyrical to avoid the ImportError.
if "pandas_datareader" not in sys.modules:
    _pdr_stub = types.ModuleType("pandas_datareader")
    _pdr_data_stub = types.ModuleType("pandas_datareader.data")
    sys.modules["pandas_datareader"] = _pdr_stub
    sys.modules["pandas_datareader.data"] = _pdr_data_stub

try:
    import empyrical
except ImportError:
    empyrical = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Annualisation factor (trading days per year)
_TRADING_DAYS: int = 252

# 60/40 benchmark weights
_BENCHMARK_WEIGHTS: dict[str, float] = {"SPY": 0.60, "TLT": 0.40}


def _compute_benchmark_return(
    conn: duckdb.DuckDBPyConnection,
    first_date: date_type,
    last_date: date_type,
) -> float | None:
    """Compute 60/40 SPY/TLT buy-and-hold return over the given period.

    Returns *None* if price data is unavailable for either symbol.
    """
    benchmark_return: float = 0.0

    for symbol, weight in _BENCHMARK_WEIGHTS.items():
        rows = conn.execute(
            """
            SELECT date, close
            FROM market_data_daily
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date ASC
            """,
            [symbol, first_date, last_date],
        ).fetchall()

        if len(rows) < 2:
            logger.warning(
                "Insufficient benchmark data for %s (%d rows) – "
                "cannot compute benchmark return.",
                symbol,
                len(rows),
            )
            return None

        first_close: float = float(rows[0][1])
        last_close: float = float(rows[-1][1])

        if first_close == 0.0:
            return None

        asset_return = (last_close / first_close) - 1.0
        benchmark_return += weight * asset_return

    return benchmark_return


def _get_best_worst_positions(
    conn: duckdb.DuckDBPyConnection,
    n: int = 3,
    pod_id: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return top-*n* best and worst positions by unrealized_pnl.

    Reads from the ``positions`` table for the latest snapshot.
    Returns ``(best, worst)`` where each is a list of dicts with
    keys ``symbol``, ``unrealized_pnl``, ``weight``.

    Parameters
    ----------
    pod_id:
        If provided, only consider snapshots for this pod.
    """
    snap_cols = [c[0] for c in conn.execute("DESCRIBE portfolio_snapshots").fetchall()]
    has_pod_id = "pod_id" in snap_cols

    if has_pod_id and pod_id is not None:
        latest_snap = conn.execute(
            """
            SELECT snapshot_id
            FROM portfolio_snapshots
            WHERE pod_id = ?
            ORDER BY date DESC, snapshot_id DESC
            LIMIT 1
            """,
            [pod_id],
        ).fetchone()
    else:
        latest_snap = conn.execute("""
            SELECT snapshot_id
            FROM portfolio_snapshots
            ORDER BY date DESC, snapshot_id DESC
            LIMIT 1
            """).fetchone()

    if latest_snap is None:
        return [], []

    snapshot_id: int = latest_snap[0]

    rows = conn.execute(
        """
        SELECT symbol, unrealized_pnl, weight
        FROM positions
        WHERE snapshot_id = ?
        ORDER BY unrealized_pnl DESC
        """,
        [snapshot_id],
    ).fetchall()

    positions = [
        {
            "symbol": r[0],
            "unrealized_pnl": round(float(r[1]), 2),
            "weight": round(float(r[2]), 4),
        }
        for r in rows
    ]

    best = positions[:n]
    worst = (
        list(reversed(positions[-n:]))
        if len(positions) >= n
        else list(reversed(positions))
    )

    return best, worst


def compute_performance(  # noqa: C901, PLR0912, PLR0915
    conn: duckdb.DuckDBPyConnection,
    initial_capital: float = 100_000.0,
    pod_id: str | None = None,
) -> dict:
    """Compute headline performance metrics.

    Reads from ``portfolio_snapshots``, ``trades``, ``positions``, and
    ``market_data_daily`` tables.  If there is insufficient data (no
    snapshots or fewer than two data points) the function returns a dict
    of zero / default values so callers never need to guard against
    missing keys.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    initial_capital:
        Starting capital used to compute total return.
    pod_id:
        If provided, only include data for this pod. If *None*, use all
        data (backward compatible).

    Returns
    -------
    dict
        Keys: ``total_return``, ``sharpe_ratio``, ``sortino_ratio``,
        ``calmar_ratio``, ``annualized_return``, ``max_drawdown``,
        ``win_rate``, ``total_trades``, ``avg_trade_pnl``,
        ``latest_nav``, ``total_pnl``, ``benchmark_return``,
        ``excess_return``, ``daily_returns``, ``best_positions``,
        ``worst_positions``.
    """
    defaults: dict = {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": None,
        "calmar_ratio": None,
        "annualized_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "avg_trade_pnl": 0.0,
        "latest_nav": initial_capital,
        "total_pnl": 0.0,
        "benchmark_return": None,
        "excess_return": None,
        "daily_returns": [],
        "best_positions": [],
        "worst_positions": [],
    }

    # ------------------------------------------------------------------
    # 1. Load portfolio snapshots into Polars
    # ------------------------------------------------------------------
    snap_cols = [c[0] for c in conn.execute("DESCRIBE portfolio_snapshots").fetchall()]
    snap_has_pod_id = "pod_id" in snap_cols

    if snap_has_pod_id and pod_id is not None:
        snap_rows = conn.execute(
            """
            SELECT date, nav, daily_pnl
            FROM portfolio_snapshots
            WHERE pod_id = ?
            ORDER BY date ASC, snapshot_id ASC
            """,
            [pod_id],
        ).fetchall()
    else:
        snap_rows = conn.execute("""
            SELECT date, nav, daily_pnl
            FROM portfolio_snapshots
            ORDER BY date ASC, snapshot_id ASC
            """).fetchall()

    if not snap_rows:
        logger.info("No portfolio snapshots – returning default metrics.")
        return defaults

    snap_df = pl.DataFrame(
        {
            "date": [r[0] for r in snap_rows],
            "nav": [float(r[1]) for r in snap_rows],
            "daily_pnl": [float(r[2]) if r[2] is not None else None for r in snap_rows],
        }
    )

    # Keep one row per date (last snapshot of each day)
    snap_df = (
        snap_df.group_by("date")
        .agg(
            pl.col("nav").last().alias("nav"),
            pl.col("daily_pnl").last().alias("daily_pnl"),
        )
        .sort("date")
    )

    latest_nav: float = snap_df["nav"][-1]
    total_pnl: float = latest_nav - initial_capital
    total_return: float = (
        (latest_nav / initial_capital) - 1.0 if initial_capital else 0.0
    )

    trading_days: int = snap_df.height

    # ------------------------------------------------------------------
    # 2. Daily returns & Sharpe ratio
    # ------------------------------------------------------------------
    sharpe_ratio: float = 0.0
    sortino_ratio: float | None = None
    daily_returns_list: list[tuple[date_type, float]] = []

    if snap_df.height >= 2:
        returns_df = snap_df.with_columns(
            (pl.col("nav") / pl.col("nav").shift(1) - 1.0).alias("daily_return")
        ).drop_nulls("daily_return")

        # Build daily_returns list for report generator
        dates_col = returns_df["date"].to_list()
        rets_col = returns_df["daily_return"].to_list()
        daily_returns_list = list(zip(dates_col, rets_col, strict=True))

        if returns_df.height >= 2:
            # Convert Polars Series to pandas Series with datetime index for empyrical
            returns_pd = pd.Series(
                returns_df["daily_return"].to_list(),
                index=pd.to_datetime(returns_df["date"].to_list()),
                dtype=float,
            )

            raw_sharpe = empyrical.sharpe_ratio(
                returns_pd, risk_free=0.0, annualization=_TRADING_DAYS
            )
            sharpe_ratio = (
                float(raw_sharpe)
                if raw_sharpe is not None and not math.isnan(raw_sharpe)
                else 0.0
            )

            raw_sortino = empyrical.sortino_ratio(
                returns_pd, required_return=0.0, annualization=_TRADING_DAYS
            )
            if (
                raw_sortino is not None
                and not math.isnan(raw_sortino)
                and not math.isinf(raw_sortino)
            ):
                sortino_ratio = float(raw_sortino)

    # ------------------------------------------------------------------
    # 3. Maximum drawdown
    # ------------------------------------------------------------------
    max_drawdown: float = 0.0
    _empyrical_returns_pd: pd.Series | None = None  # reused in section 3b

    if snap_df.height >= 2:
        # Build returns series — empyrical expects pandas Series with datetime index
        _dd_df = snap_df.with_columns(
            (pl.col("nav") / pl.col("nav").shift(1) - 1.0).alias("daily_return")
        ).drop_nulls("daily_return")
        _empyrical_returns_pd = pd.Series(
            _dd_df["daily_return"].to_list(),
            index=pd.to_datetime(_dd_df["date"].to_list()),
            dtype=float,
        )
        raw_mdd = empyrical.max_drawdown(_empyrical_returns_pd)
        # empyrical returns a negative value (or zero); preserve sign for downstream
        if raw_mdd is not None and not math.isnan(raw_mdd):
            max_drawdown = float(raw_mdd)

    # ------------------------------------------------------------------
    # 3b. Annualised return & Calmar ratio
    # ------------------------------------------------------------------
    annualized_return: float = 0.0
    calmar_ratio: float | None = None

    if trading_days >= 2 and initial_capital > 0.0:
        annualized_return = (1.0 + total_return) ** (_TRADING_DAYS / trading_days) - 1.0

        if _empyrical_returns_pd is not None and len(_empyrical_returns_pd) >= 2:
            raw_calmar = empyrical.calmar_ratio(
                _empyrical_returns_pd, annualization=_TRADING_DAYS
            )
            if (
                raw_calmar is not None
                and not math.isnan(raw_calmar)
                and not math.isinf(raw_calmar)
            ):
                calmar_ratio = float(raw_calmar)

    # ------------------------------------------------------------------
    # 4. Trade-level statistics
    # ------------------------------------------------------------------
    total_trades: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0

    trade_cols = [c[0] for c in conn.execute("DESCRIBE trades").fetchall()]
    trade_has_pod_id = "pod_id" in trade_cols

    if trade_has_pod_id and pod_id is not None:
        trade_rows = conn.execute(
            """
            SELECT
                t.symbol,
                t.action,
                t.shares,
                t.price,
                t.notional
            FROM trades t
            WHERE t.pod_id = ?
            ORDER BY t.date ASC, t.trade_id ASC
            """,
            [pod_id],
        ).fetchall()
    else:
        trade_rows = conn.execute("""
            SELECT
                t.symbol,
                t.action,
                t.shares,
                t.price,
                t.notional
            FROM trades t
            ORDER BY t.date ASC, t.trade_id ASC
            """).fetchall()

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
    # 5. Benchmark return (60/40 SPY/TLT)
    # ------------------------------------------------------------------
    benchmark_return: float | None = None
    excess_return: float | None = None

    if snap_df.height >= 2:
        first_date = snap_df["date"][0]
        last_date = snap_df["date"][-1]
        benchmark_return = _compute_benchmark_return(conn, first_date, last_date)

        if benchmark_return is not None:
            excess_return = total_return - benchmark_return

    # ------------------------------------------------------------------
    # 6. Best / worst positions
    # ------------------------------------------------------------------
    best_positions, worst_positions = _get_best_worst_positions(conn, pod_id=pod_id)

    # ------------------------------------------------------------------
    # 7. Assemble result
    # ------------------------------------------------------------------
    metrics: dict = {
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4) if sortino_ratio is not None else None,
        "calmar_ratio": round(calmar_ratio, 4) if calmar_ratio is not None else None,
        "annualized_return": round(annualized_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "latest_nav": round(latest_nav, 2),
        "total_pnl": round(total_pnl, 2),
        "benchmark_return": (
            round(benchmark_return, 6) if benchmark_return is not None else None
        ),
        "excess_return": (
            round(excess_return, 6) if excess_return is not None else None
        ),
        "daily_returns": daily_returns_list,
        "best_positions": best_positions,
        "worst_positions": worst_positions,
    }

    logger.info(
        "Performance: return=%.2f%%, sharpe=%.2f, sortino=%s, calmar=%s, "
        "drawdown=%.2f%%, trades=%d, win_rate=%.1f%%, "
        "benchmark=%s, excess=%s",
        total_return * 100,
        sharpe_ratio,
        f"{sortino_ratio:.2f}" if sortino_ratio is not None else "N/A",
        f"{calmar_ratio:.2f}" if calmar_ratio is not None else "N/A",
        max_drawdown * 100,
        total_trades,
        win_rate * 100,
        f"{benchmark_return * 100:.2f}%" if benchmark_return is not None else "N/A",
        f"{excess_return * 100:.2f}%" if excess_return is not None else "N/A",
    )

    return metrics

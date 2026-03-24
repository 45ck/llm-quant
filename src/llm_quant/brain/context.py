"""Assemble MarketContext from the database for the LLM decision prompt."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import duckdb

from llm_quant.brain.models import MarketContext, MarketRow, PositionRow
from llm_quant.config import AppConfig

logger = logging.getLogger(__name__)


def _fetch_latest_market_data(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
) -> list[dict[str, Any]]:
    """Fetch the most recent two trading days of market data for each symbol.

    Returns a list of dicts keyed by column name. Two rows per symbol are
    needed so we can compute daily change_pct.
    """
    if not symbols:
        return []

    placeholders = ", ".join(["?"] * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT
                symbol, date, close, volume,
                sma_20, sma_50, rsi_14, macd, atr_14,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM market_data_daily
            WHERE symbol IN ({placeholders})
        )
        SELECT *
        FROM ranked
        WHERE rn <= 2
        ORDER BY symbol, rn
    """
    rows = conn.execute(query, symbols).fetchall()
    columns = [
        "symbol", "date", "close", "volume",
        "sma_20", "sma_50", "rsi_14", "macd", "atr_14", "rn",
    ]
    return [dict(zip(columns, row)) for row in rows]


def _compute_change_pct(
    raw_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group raw rows by symbol and compute daily change_pct.

    Returns {symbol: {close, change_pct, sma_20, ...}} using the latest row
    and computing change from the previous day's close.
    """
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_symbol.setdefault(row["symbol"], []).append(row)

    result: dict[str, dict[str, Any]] = {}
    for symbol, rows in by_symbol.items():
        # rows are ordered by rn (1=latest, 2=previous)
        latest = next((r for r in rows if r["rn"] == 1), None)
        previous = next((r for r in rows if r["rn"] == 2), None)

        if latest is None:
            continue

        close = latest["close"] or 0.0
        prev_close = previous["close"] if previous else None

        if prev_close and prev_close != 0:
            change_pct = (close - prev_close) / prev_close * 100.0
        else:
            change_pct = 0.0

        result[symbol] = {
            "symbol": symbol,
            "close": close,
            "change_pct": round(change_pct, 4),
            "sma_20": latest["sma_20"] or 0.0,
            "sma_50": latest["sma_50"] or 0.0,
            "rsi_14": latest["rsi_14"] or 0.0,
            "macd": latest["macd"] or 0.0,
            "atr_14": latest["atr_14"] or 0.0,
            "volume": latest["volume"] or 0,
        }
    return result


def _get_vix(conn: duckdb.DuckDBPyConnection) -> float:
    """Fetch the latest VIX close from market data.

    Tries ^VIX first, then VIX.  Returns 0.0 if not found.
    """
    for vix_symbol in ("^VIX", "VIX"):
        row = conn.execute(
            """
            SELECT close
            FROM market_data_daily
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            [vix_symbol],
        ).fetchone()
        if row and row[0] is not None:
            logger.debug("VIX value %.2f from symbol %s", row[0], vix_symbol)
            return float(row[0])

    logger.warning("VIX data not found in market_data_daily; defaulting to 0.0")
    return 0.0


def _get_yield_spread(conn: duckdb.DuckDBPyConnection) -> float:
    """Estimate the yield spread from TLT/SHY ratio.

    Uses latest close of TLT and SHY.  A rough proxy for the long-short
    rate differential.  Falls back to 0.0 when data is unavailable.
    """
    query = """
        SELECT symbol, close
        FROM market_data_daily
        WHERE symbol IN ('TLT', 'SHY')
        ORDER BY date DESC
    """
    rows = conn.execute(query).fetchall()

    prices: dict[str, float] = {}
    for symbol, close in rows:
        if symbol not in prices and close is not None:
            prices[symbol] = float(close)

    tlt = prices.get("TLT")
    shy = prices.get("SHY")
    if tlt and shy and shy != 0:
        # Spread approximation: higher TLT/SHY ratio => steeper curve
        spread = (tlt / shy - 1.0) * 100.0
        logger.debug("Yield spread proxy %.2f bps (TLT=%.2f, SHY=%.2f)", spread, tlt, shy)
        return round(spread, 2)

    logger.warning("TLT/SHY data not available; yield spread defaults to 0.0")
    return 0.0


def _get_spy_trend(conn: duckdb.DuckDBPyConnection) -> str:
    """Determine SPY trend from SMA50 vs SMA200 (or SMA50 if no 200 available).

    Returns one of: "bullish", "bearish", "neutral".
    """
    row = conn.execute(
        """
        SELECT close, sma_20, sma_50
        FROM market_data_daily
        WHERE symbol = 'SPY'
        ORDER BY date DESC
        LIMIT 1
        """,
    ).fetchone()

    if row is None:
        logger.warning("SPY data not found; trend defaults to neutral")
        return "neutral"

    close, sma_20, sma_50 = (float(v) if v is not None else None for v in row)

    # Try to get SMA200 from a wider window (manual check via row count)
    sma200_row = conn.execute(
        """
        SELECT AVG(close)
        FROM (
            SELECT close
            FROM market_data_daily
            WHERE symbol = 'SPY'
            ORDER BY date DESC
            LIMIT 200
        )
        """,
    ).fetchone()

    sma_200: float | None = None
    if sma200_row and sma200_row[0] is not None:
        # Only trust if we have enough data points
        count_row = conn.execute(
            "SELECT COUNT(*) FROM market_data_daily WHERE symbol = 'SPY'"
        ).fetchone()
        if count_row and count_row[0] >= 200:
            sma_200 = float(sma200_row[0])

    if sma_200 is not None and sma_50 is not None:
        if sma_50 > sma_200:
            trend = "bullish"
        elif sma_50 < sma_200:
            trend = "bearish"
        else:
            trend = "neutral"
        logger.debug("SPY trend=%s (SMA50=%.2f vs SMA200=%.2f)", trend, sma_50, sma_200)
    elif sma_50 is not None and close is not None:
        # Fallback: compare price to SMA50
        if close > sma_50:
            trend = "bullish"
        elif close < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"
        logger.debug("SPY trend=%s (close=%.2f vs SMA50=%.2f, no SMA200)", trend, close, sma_50)
    else:
        trend = "neutral"
        logger.debug("SPY trend=neutral (insufficient indicator data)")

    return trend


def _build_position_rows(
    positions: list[dict[str, Any]],
    market_prices: dict[str, dict[str, Any]],
    nav: float,
) -> list[PositionRow]:
    """Convert raw position dicts into PositionRow dataclasses.

    Computes current_price from market data, P&L%, and portfolio weight.
    """
    rows: list[PositionRow] = []
    for pos in positions:
        symbol: str = pos["symbol"]
        shares: float = float(pos.get("shares", 0))
        avg_cost: float = float(pos.get("avg_cost", 0))
        stop_loss: float = float(pos.get("stop_loss", 0))

        # Look up current price from market data
        mkt = market_prices.get(symbol)
        current_price = mkt["close"] if mkt else avg_cost

        # Compute P&L %
        if avg_cost and avg_cost != 0:
            pnl_pct = (current_price - avg_cost) / avg_cost * 100.0
        else:
            pnl_pct = 0.0

        # Compute portfolio weight
        market_value = abs(shares * current_price)
        weight_pct = (market_value / nav * 100.0) if nav > 0 else 0.0

        rows.append(
            PositionRow(
                symbol=symbol,
                shares=shares,
                avg_cost=round(avg_cost, 2),
                current_price=round(current_price, 2),
                pnl_pct=round(pnl_pct, 2),
                weight_pct=round(weight_pct, 2),
                stop_loss=round(stop_loss, 2),
            )
        )
    return rows


def build_market_context(
    conn: duckdb.DuckDBPyConnection,
    portfolio_state: dict[str, Any],
    config: AppConfig,
) -> MarketContext:
    """Build a complete MarketContext from the DB and portfolio state.

    Parameters
    ----------
    conn:
        Active DuckDB connection.
    portfolio_state:
        Dict with keys ``nav``, ``cash``, ``positions`` (list of dicts
        with ``symbol``, ``shares``, ``avg_cost``, ``stop_loss``).
    config:
        Application configuration (used to determine the asset universe).

    Returns
    -------
    MarketContext
        Fully populated context ready for prompt rendering.
    """
    nav: float = float(portfolio_state.get("nav", 0))
    cash: float = float(portfolio_state.get("cash", 0))
    raw_positions: list[dict[str, Any]] = portfolio_state.get("positions", [])

    # Determine universe symbols from config
    universe_symbols: list[str] = [
        asset.symbol for asset in config.universe.assets if asset.tradeable
    ]
    if not universe_symbols:
        logger.warning("No tradeable asset symbols found in universe config")

    # Also include symbols of currently-held positions
    held_symbols = {pos["symbol"] for pos in raw_positions}
    all_symbols = sorted(set(universe_symbols) | held_symbols)

    logger.info(
        "Building market context for %d symbols (%d universe, %d held)",
        len(all_symbols),
        len(universe_symbols),
        len(held_symbols),
    )

    # Fetch market data for the last 2 days
    raw_market = _fetch_latest_market_data(conn, all_symbols)
    market_prices = _compute_change_pct(raw_market)

    # Build MarketRow list sorted by momentum (change_pct descending)
    market_data: list[MarketRow] = []
    for symbol in all_symbols:
        data = market_prices.get(symbol)
        if data is None:
            logger.debug("No market data for %s; skipping", symbol)
            continue
        market_data.append(
            MarketRow(
                symbol=data["symbol"],
                close=round(data["close"], 2),
                change_pct=round(data["change_pct"], 2),
                sma_20=round(data["sma_20"], 2),
                sma_50=round(data["sma_50"], 2),
                rsi_14=round(data["rsi_14"], 2),
                macd=round(data["macd"], 4),
                atr_14=round(data["atr_14"], 2),
                volume=data["volume"],
            )
        )

    market_data.sort(key=lambda r: r.change_pct, reverse=True)

    # Build position rows
    position_rows = _build_position_rows(raw_positions, market_prices, nav)

    # Compute portfolio-level metrics
    cash_pct = (cash / nav * 100.0) if nav > 0 else 100.0

    long_exposure = sum(
        abs(p.shares * p.current_price)
        for p in position_rows
        if p.shares > 0
    )
    short_exposure = sum(
        abs(p.shares * p.current_price)
        for p in position_rows
        if p.shares < 0
    )
    gross_exposure_pct = ((long_exposure + short_exposure) / nav * 100.0) if nav > 0 else 0.0
    net_exposure_pct = ((long_exposure - short_exposure) / nav * 100.0) if nav > 0 else 0.0

    # Macro indicators
    vix = _get_vix(conn)
    yield_spread = _get_yield_spread(conn)
    spy_trend = _get_spy_trend(conn)

    today = date.today()

    context = MarketContext(
        date=today,
        nav=round(nav, 2),
        cash=round(cash, 2),
        cash_pct=round(cash_pct, 2),
        gross_exposure_pct=round(gross_exposure_pct, 2),
        net_exposure_pct=round(net_exposure_pct, 2),
        positions=position_rows,
        market_data=market_data,
        vix=round(vix, 2),
        yield_spread=round(yield_spread, 2),
        spy_trend=spy_trend,
    )

    logger.info(
        "Market context built: date=%s, nav=%.2f, %d positions, %d market rows, "
        "VIX=%.2f, spy_trend=%s",
        context.date,
        context.nav,
        len(context.positions),
        len(context.market_data),
        context.vix,
        context.spy_trend,
    )
    return context

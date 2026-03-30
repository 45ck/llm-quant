"""Assemble MarketContext from the database for the LLM decision prompt."""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

import duckdb

from llm_quant.brain.models import MarketContext, MarketRegime, MarketRow, PositionRow
from llm_quant.config import AppConfig
from llm_quant.risk.limits import get_execution_cost

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
        "symbol",
        "date",
        "close",
        "volume",
        "sma_20",
        "sma_50",
        "rsi_14",
        "macd",
        "atr_14",
        "rn",
    ]
    return [dict(zip(columns, row, strict=True)) for row in rows]


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
        logger.debug(
            "Yield spread proxy %.2f bps (TLT=%.2f, SHY=%.2f)",
            spread,
            tlt,
            shy,
        )
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

    close, _sma_20, sma_50 = (float(v) if v is not None else None for v in row)

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
        logger.debug(
            "SPY trend=%s (close=%.2f vs SMA50=%.2f, no SMA200)",
            trend,
            close,
            sma_50,
        )
    else:
        trend = "neutral"
        logger.debug("SPY trend=neutral (insufficient indicator data)")

    return trend


def _get_credit_spread(
    conn: duckdb.DuckDBPyConnection,
    zscore_window: int = 20,
    silent_stress_zscore_threshold: float = 1.5,
    silent_stress_vix_ceiling: float = 20.0,
    vix: float = 0.0,
) -> tuple[float | None, float | None, bool]:
    """Fetch ICE BofA US Corporate OAS (BAMLC0A0CM) from fred_data_daily.

    Returns (oas_bps, zscore, silent_stress) where:
      - oas_bps: latest OAS value in basis points (or None if unavailable)
      - zscore: 20-day rolling z-score of OAS (or None if < 2 data points)
      - silent_stress: True when zscore > threshold AND VIX is below ceiling
        (credit is pricing in stress that equities haven't yet reflected)
    """
    series_id = "BAMLC0A0CM"
    try:
        rows = conn.execute(
            """
            SELECT value
            FROM fred_data_daily
            WHERE series_id = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            [series_id, zscore_window],
        ).fetchall()
    except Exception:
        logger.warning("fred_data_daily not available; credit spread defaults to None")
        return None, None, False

    if not rows:
        logger.warning(
            "No FRED data for %s; credit spread defaults to None", series_id
        )
        return None, None, False

    values = [float(r[0]) for r in rows if r[0] is not None]
    if not values:
        return None, None, False

    oas = values[0]  # most recent value

    if len(values) < 2:
        return oas, None, False

    import statistics  # noqa: PLC0415

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0.0:
        zscore = 0.0
    else:
        zscore = (oas - mean) / stdev

    zscore = round(zscore, 4)
    silent_stress = (zscore > silent_stress_zscore_threshold) and (
        vix < silent_stress_vix_ceiling
    )

    logger.debug(
        "Credit spread OAS=%.2f bps, z-score=%.2f, silent_stress=%s",
        oas,
        zscore,
        silent_stress,
    )
    return oas, zscore, silent_stress


def _get_vix_regime(
    conn: duckdb.DuckDBPyConnection,
    vix: float,
    percentile_window: int = 252,
) -> tuple[MarketRegime, tuple[float, float]]:
    """Classify market regime using adaptive percentile-based VIX thresholds.

    Loads up to ``percentile_window`` days of VIX history, computes the 33rd
    and 67th percentile (tercile boundaries), then classifies the current VIX:

      risk_on   — VIX < 33rd percentile
      transition — 33rd <= VIX <= 67th percentile
      risk_off  — VIX > 67th percentile

    Falls back to static thresholds (20/25) when insufficient history exists.

    Returns
    -------
    (regime, (lower_threshold, upper_threshold))
    """
    _FALLBACK_LOW = 20.0
    _FALLBACK_HIGH = 25.0

    vix_values: list[float] = []
    for vix_symbol in ("^VIX", "VIX"):
        rows = conn.execute(
            """
            SELECT close
            FROM market_data_daily
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            [vix_symbol, percentile_window],
        ).fetchall()
        if rows:
            vix_values = [float(r[0]) for r in rows if r[0] is not None]
            break

    if len(vix_values) < 10:
        logger.warning(
            "Insufficient VIX history (%d rows); using static thresholds %.1f/%.1f",
            len(vix_values),
            _FALLBACK_LOW,
            _FALLBACK_HIGH,
        )
        lower, upper = _FALLBACK_LOW, _FALLBACK_HIGH
    else:
        sorted_vix = sorted(vix_values)
        n = len(sorted_vix)
        # Percentile index (nearest-rank method)
        idx_33 = max(0, int(round(0.33 * n)) - 1)
        idx_67 = max(0, int(round(0.67 * n)) - 1)
        lower = round(sorted_vix[idx_33], 2)
        upper = round(sorted_vix[idx_67], 2)
        logger.debug(
            "Adaptive VIX thresholds: 33rd pct=%.2f, 67th pct=%.2f (n=%d)",
            lower,
            upper,
            n,
        )

    if vix < lower:
        regime = MarketRegime.RISK_ON
    elif vix > upper:
        regime = MarketRegime.RISK_OFF
    else:
        regime = MarketRegime.TRANSITION

    logger.debug(
        "VIX regime=%s (VIX=%.2f, lower=%.2f, upper=%.2f)",
        regime,
        vix,
        lower,
        upper,
    )
    return regime, (lower, upper)


def _get_vix_percentile_126d(
    conn: duckdb.DuckDBPyConnection,
    vix: float,
) -> float:
    """Compute the percentile rank of current VIX among the last 126 trading days.

    Returns a value on the 0–100 scale.  Returns 50.0 if insufficient data.
    The 126-day window covers approximately 6 calendar months.
    """
    vix_values: list[float] = []
    for vix_symbol in ("^VIX", "VIX"):
        rows = conn.execute(
            """
            SELECT close
            FROM market_data_daily
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 126
            """,
            [vix_symbol],
        ).fetchall()
        if rows:
            vix_values = [float(r[0]) for r in rows if r[0] is not None]
            break

    if len(vix_values) < 2:
        logger.warning(
            "Insufficient VIX history for 126d percentile; defaulting to 50.0"
        )
        return 50.0

    count_below = sum(1 for v in vix_values if v < vix)
    percentile = round(count_below / len(vix_values) * 100.0, 1)
    logger.debug(
        "VIX 126d percentile=%.1f (VIX=%.2f, n=%d)",
        percentile,
        vix,
        len(vix_values),
    )
    return percentile


def _get_hmm_regime(
    conn: duckdb.DuckDBPyConnection,
    lookback_days: int = 130,
) -> tuple[MarketRegime | None, float, bool]:
    """Fit 2-state GaussianHMM on VIX, yield slope, SPY momentum.

    Returns
    -------
    (regime, confidence, fallback_used)
        - regime: MarketRegime from HMM, or None if fallback was used
        - confidence: HMM state probability (0–1); 0.5 if fallback
        - fallback_used: True when the HMM could not produce a result
    """
    try:
        from llm_quant.regime.hmm import HmmRegimeConfig, HmmRegimeDetector  # noqa: PLC0415
        import polars as pl  # noqa: PLC0415

        # VIX series
        vix_rows = None
        for vix_symbol in ("^VIX", "VIX"):
            rows = conn.execute(
                """
                SELECT close
                FROM market_data_daily
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                [vix_symbol, lookback_days],
            ).fetchall()
            if rows:
                vix_rows = rows
                break

        if not vix_rows:
            logger.warning("HMM: no VIX data; skipping HMM regime detection")
            return None, 0.5, True

        vix_series = pl.Series([float(r[0]) for r in reversed(vix_rows) if r[0] is not None])

        # Yield slope: T10Y2Y from fred_data_daily if available, else zeros
        try:
            slope_rows = conn.execute(
                """
                SELECT value
                FROM fred_data_daily
                WHERE series_id = 'T10Y2Y'
                ORDER BY date DESC
                LIMIT ?
                """,
                [lookback_days],
            ).fetchall()
            if slope_rows:
                slope_values = [float(r[0]) if r[0] is not None else 0.0 for r in reversed(slope_rows)]
            else:
                slope_values = [0.0] * len(vix_series)
        except Exception:
            slope_values = [0.0] * len(vix_series)

        # Align length with VIX
        n = len(vix_series)
        if len(slope_values) >= n:
            slope_values = slope_values[-n:]
        else:
            slope_values = [0.0] * (n - len(slope_values)) + slope_values

        yield_slope = pl.Series(slope_values)

        # SPY 20d momentum from market data
        spy_rows = conn.execute(
            """
            SELECT close
            FROM market_data_daily
            WHERE symbol = 'SPY'
            ORDER BY date DESC
            LIMIT ?
            """,
            [lookback_days + 20],
        ).fetchall()

        spy_closes = [float(r[0]) for r in reversed(spy_rows) if r[0] is not None]
        spy_momentum_values: list[float] = []
        for i in range(len(spy_closes)):
            if i < 20:
                spy_momentum_values.append(0.0)
            else:
                pct = (spy_closes[i] - spy_closes[i - 20]) / spy_closes[i - 20]
                spy_momentum_values.append(pct)

        # Trim/pad momentum to match n
        spy_momentum_values = spy_momentum_values[-n:] if len(spy_momentum_values) >= n else [0.0] * (n - len(spy_momentum_values)) + spy_momentum_values
        spy_momentum = pl.Series(spy_momentum_values)

        config = HmmRegimeConfig(lookback_days=lookback_days)
        result = HmmRegimeDetector(config).fit_predict(vix_series, yield_slope, spy_momentum)

        if result.fallback_used:
            return None, 0.5, True

        hmm_regime = (
            MarketRegime.RISK_ON if result.regime == "risk_on" else MarketRegime.RISK_OFF
        )
        return hmm_regime, result.confidence, False

    except Exception:
        logger.warning("HMM regime detection failed; keeping heuristic", exc_info=True)
        return None, 0.5, True


def _get_cot_crowding(
    universe_symbols: list[str],
    cot_stale: bool = False,
) -> dict[str, str] | None:
    """Fetch COT crowding signals for applicable universe symbols.

    Returns a dict mapping symbol -> signal string, or None on any failure.
    Only symbols with a CFTC code mapping are included.

    Friday-published data applied on Monday open (look-ahead bias prevention).
    COT signals are confirmation/warning overlays only — never independent entries.

    Parameters
    ----------
    universe_symbols:
        List of tradeable symbols from the universe config.
    cot_stale:
        If True, COT data in the DB is stale (> 10 days old). Returns None
        immediately with a warning rather than serving outdated signals.
    """
    if cot_stale:
        logger.warning(
            "COT crowding overlay skipped: data is stale (>10 days). "
            "Run pq fetch to refresh COT data.",
        )
        return None

    try:
        from llm_quant.data.cot_fetcher import CotFetcher, SYMBOL_TO_COT_KEY

        cot_symbols = [s for s in universe_symbols if s in SYMBOL_TO_COT_KEY]
        if not cot_symbols:
            return None

        fetcher = CotFetcher()
        crowding: dict[str, str] = {}
        for symbol in cot_symbols:
            try:
                result = fetcher.get_regime_signal(symbol)
                signal = result.get("signal", "neutral")
                if signal != "neutral":
                    crowding[symbol] = signal
            except Exception:
                logger.debug("COT signal fetch failed for %s", symbol, exc_info=True)

        return crowding if crowding else None
    except Exception:
        logger.warning("COT crowding overlay unavailable", exc_info=True)
        return None


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
    cot_stale: bool = False,
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
    cot_stale:
        If True, COT data in cot_weekly is stale (>10 days old). Passes the
        flag to ``_get_cot_crowding`` so it returns None rather than serving
        outdated crowding signals.

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
        abs(p.shares * p.current_price) for p in position_rows if p.shares > 0
    )
    short_exposure = sum(
        abs(p.shares * p.current_price) for p in position_rows if p.shares < 0
    )
    gross_exposure_pct = (
        (long_exposure + short_exposure) / nav * 100.0 if nav > 0 else 0.0
    )
    net_exposure_pct = (
        (long_exposure - short_exposure) / nav * 100.0 if nav > 0 else 0.0
    )

    # Macro indicators
    vix = _get_vix(conn)
    yield_spread = _get_yield_spread(conn)
    spy_trend = _get_spy_trend(conn)

    # ev8: credit spread stress indicator
    vix_regime_percentile_window: int = getattr(
        config, "vix_regime_percentile_window", 252
    )
    credit_spread_oas, credit_spread_zscore, silent_stress = _get_credit_spread(
        conn, vix=vix
    )

    # 4a1: adaptive VIX regime classification
    market_regime, vix_regime_thresholds = _get_vix_regime(
        conn, vix, percentile_window=vix_regime_percentile_window
    )

    # vts: 126-day VIX percentile rank
    vix_percentile_126d = _get_vix_percentile_126d(conn, vix)

    # llm-quant-773: 2-state HMM regime detection
    hmm_lookback: int = 130
    hmm_enabled: bool = True
    try:
        hmm_lookback = getattr(config, "hmm_lookback_days", 130)
        hmm_enabled = getattr(config, "hmm_enabled", True)
    except Exception:
        pass

    hmm_confidence_threshold: float = 0.65
    try:
        hmm_confidence_threshold = getattr(config, "hmm_confidence_threshold", 0.65)
    except Exception:
        pass

    regime_confidence: float = 0.5
    hmm_regime_fallback: bool = True

    if hmm_enabled:
        hmm_regime, regime_confidence, hmm_regime_fallback = _get_hmm_regime(
            conn, lookback_days=hmm_lookback
        )
        if not hmm_regime_fallback and hmm_regime is not None:
            # Only override heuristic when HMM is confident
            if regime_confidence >= hmm_confidence_threshold:
                market_regime = hmm_regime
                logger.info(
                    "HMM regime override: %s (confidence=%.3f)",
                    market_regime,
                    regime_confidence,
                )
            else:
                logger.debug(
                    "HMM confidence %.3f below threshold %.3f; keeping heuristic regime %s",
                    regime_confidence,
                    hmm_confidence_threshold,
                    market_regime,
                )

    # llm-quant-56k: COT crowding overlay for applicable symbols
    # cot_stale=True skips fetching and returns None when DB data is >10 days old
    cot_crowding = _get_cot_crowding(universe_symbols, cot_stale=cot_stale)

    # llm-quant-bbt: execution cost lookup for all symbols in context
    execution_costs: dict[str, float] = {}
    for asset in config.universe.assets:
        execution_costs[asset.symbol] = get_execution_cost(
            asset.symbol, asset.asset_class
        )

    from datetime import datetime

    today = datetime.now(tz=UTC).date()

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
        credit_spread_oas=round(credit_spread_oas, 2) if credit_spread_oas is not None else None,
        credit_spread_zscore=round(credit_spread_zscore, 4) if credit_spread_zscore is not None else None,
        silent_stress=silent_stress,
        vix_regime_thresholds=vix_regime_thresholds,
        market_regime=market_regime,
        vix_percentile_126d=vix_percentile_126d,
        cot_crowding=cot_crowding,
        execution_costs=execution_costs if execution_costs else None,
        regime_confidence=round(regime_confidence, 4),
        hmm_regime_fallback=hmm_regime_fallback,
    )

    logger.info(
        "Market context built: date=%s, nav=%.2f, %d positions, %d market rows, "
        "VIX=%.2f (pct126d=%.1f, regime=%s), spy_trend=%s, "
        "credit_oas=%s, credit_z=%s, silent_stress=%s",
        context.date,
        context.nav,
        len(context.positions),
        len(context.market_data),
        context.vix,
        context.vix_percentile_126d,
        context.market_regime,
        context.spy_trend,
        f"{context.credit_spread_oas:.2f}" if context.credit_spread_oas is not None else "N/A",
        f"{context.credit_spread_zscore:.4f}" if context.credit_spread_zscore is not None else "N/A",
        context.silent_stress,
    )
    return context

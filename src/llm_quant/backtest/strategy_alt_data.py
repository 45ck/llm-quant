"""Alt-Data Signals strategy implementation.

Implements three OHLCV-proxy signals -- flow momentum (AD-004),
flow reversal (AD-005), and attention-price divergence (AD-003) --
with a VIX quality filter.  All computations are causal (backward-looking
only).  No external APIs are called during backtesting; attention and flow
are proxied entirely from volume and price data.

See data/strategies/alt-data-signals/research-spec.yaml for the frozen
research design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import ClassVar

import polars as pl

from llm_quant.backtest.strategy import Strategy
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: extract a single symbol's sorted time-series from indicators_df
# ---------------------------------------------------------------------------


def _sym_series(indicators_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Return the rows for *symbol*, sorted by date."""
    return indicators_df.filter(pl.col("symbol") == symbol).sort("date")


def _latest_row(sym_df: pl.DataFrame) -> dict | None:
    """Return the most recent row as a dict, or None."""
    if len(sym_df) == 0:
        return None
    return sym_df.tail(1).row(0, named=True)


# ---------------------------------------------------------------------------
# Helper: rolling statistics computed inline from plain lists
# ---------------------------------------------------------------------------


def _rolling_mean_list(values: list[float], window: int) -> float | None:
    """Trailing mean of the last *window* values."""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _rolling_std_list(values: list[float], window: int) -> float | None:
    """Trailing std-dev of the last *window* values (population)."""
    if len(values) < window:
        return None
    segment = values[-window:]
    mean = sum(segment) / window
    variance = sum((v - mean) ** 2 for v in segment) / window
    return variance**0.5


def _percentile_rank(values: list[float], window: int) -> float | None:
    """Trailing percentile rank (0-100) of the last value against the
    preceding *window* values (including current)."""
    if len(values) < window:
        return None
    segment = values[-window:]
    current = segment[-1]
    count_below = sum(1 for v in segment if v < current)
    return (count_below / len(segment)) * 100.0


# ---------------------------------------------------------------------------
# Parameter bundle to keep method signatures lean
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AltDataParams:
    """Bundles all strategy parameters extracted from config."""

    flow_sma_period: int
    flow_volume_zscore_window: int
    attention_proxy_period: int
    flow_reversal_percentile: float
    flow_reversal_lookback: int
    stop_atr_multiplier: float
    vix_suppress_threshold: float


@dataclass(frozen=True)
class _VixState:
    """VIX filter state for the current bar."""

    level: float | None
    suppressed: bool


# ---------------------------------------------------------------------------
# Alt-Data Signals Strategy
# ---------------------------------------------------------------------------


SECTOR_ETFS: list[str] = [
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLRE",
    "XLB",
    "XLC",
]


class AltDataSignalsStrategy(Strategy):
    """Alt-data proxy signals: flow momentum, flow reversal, attention divergence.

    Hierarchical signal priority:
      1. VIX quality filter (VIX >= 25 suppresses new entries)
      2. Attention-price divergence (AD-003)
      3. Flow momentum -- sector ETF ranking (AD-004)
      4. Flow reversal -- contrarian at extremes (AD-005)

    All indicators are OHLCV-derived proxies -- no external data needed.
    Required columns: close, volume, sma_20, rsi_14, atr_14.
    VIX symbol required for quality filter.
    """

    VIX_SYMBOL: ClassVar[str] = "VIX"

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        p = self._extract_params()

        # ----- VIX quality filter -----
        vix = self._get_vix_state(indicators_df, p.vix_suppress_threshold)

        # ----- Collect all non-VIX symbols -----
        symbols = indicators_df.select("symbol").unique().to_series().to_list()
        symbols = [s for s in symbols if s != self.VIX_SYMBOL]

        signals: list[TradeSignal] = []

        # AD-003: Attention-price divergence
        signals.extend(
            self._divergence_signals(indicators_df, symbols, portfolio, prices, p, vix)
        )
        # AD-004: Flow momentum -- sector ETFs
        signals.extend(
            self._flow_momentum_signals(indicators_df, portfolio, prices, p, vix)
        )
        # AD-005: Flow reversal -- contrarian at extremes
        signals.extend(
            self._flow_reversal_signals(indicators_df, symbols, portfolio, prices, p)
        )

        return signals

    def _extract_params(self) -> _AltDataParams:
        """Extract strategy parameters from config with defaults."""
        cfg = self.config.parameters
        return _AltDataParams(
            flow_sma_period=cfg.get("flow_sma_period", 20),
            flow_volume_zscore_window=cfg.get("flow_volume_zscore_window", 20),
            attention_proxy_period=cfg.get("attention_proxy_period", 20),
            flow_reversal_percentile=cfg.get("flow_reversal_percentile", 5),
            flow_reversal_lookback=cfg.get("flow_reversal_lookback", 63),
            stop_atr_multiplier=cfg.get("stop_atr_multiplier", 2.0),
            vix_suppress_threshold=cfg.get("vix_suppress_threshold", 25),
        )

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def _get_vix_state(
        self, indicators_df: pl.DataFrame, threshold: float
    ) -> _VixState:
        """Get the current VIX level and suppression state."""
        vix_df = _sym_series(indicators_df, self.VIX_SYMBOL)
        row = _latest_row(vix_df)
        level = row["close"] if row else None
        suppressed = level is not None and level >= threshold
        return _VixState(level=level, suppressed=suppressed)

    # ------------------------------------------------------------------
    # AD-003: Attention-price divergence signals
    # ------------------------------------------------------------------

    def _divergence_signals(
        self,
        indicators_df: pl.DataFrame,
        symbols: list[str],
        portfolio: Portfolio,
        prices: dict[str, float],
        p: _AltDataParams,
        vix: _VixState,
    ) -> list[TradeSignal]:
        """Generate AD-003 divergence entry/exit signals."""
        signals: list[TradeSignal] = []

        for symbol in symbols:
            divergence = _classify_divergence(
                indicators_df,
                symbol,
                p.attention_proxy_period,
                p.flow_volume_zscore_window,
            )
            if divergence is None:
                continue

            close = prices.get(symbol, 0)
            if close <= 0:
                continue

            if divergence == "institutional" and symbol not in portfolio.positions:
                signal = self._institutional_entry(
                    indicators_df, symbol, close, p, portfolio, vix
                )
                if signal is not None:
                    signals.append(signal)

            elif divergence == "retail_fomo" and symbol in portfolio.positions:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            "AD-003 retail FOMO: rising volume + rising price "
                            "(fragile momentum, exit)"
                        ),
                    )
                )

        return signals

    def _institutional_entry(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        close: float,
        p: _AltDataParams,
        portfolio: Portfolio,
        vix: _VixState,
    ) -> TradeSignal | None:
        """Build a BUY signal for institutional accumulation, or None."""
        if vix.suppressed:
            logger.debug(
                "AD-003 BUY suppressed for %s: VIX=%.1f >= %.0f",
                symbol,
                vix.level,
                p.vix_suppress_threshold,
            )
            return None
        if len(portfolio.positions) >= self.config.max_positions:
            return None
        sl = self._compute_stop_loss(
            indicators_df, symbol, close, p.stop_atr_multiplier
        )
        return TradeSignal(
            symbol=symbol,
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=self.config.target_position_weight,
            stop_loss=sl,
            reasoning=(
                "AD-003 institutional accumulation: declining volume + "
                "rising price (attention-price divergence)"
            ),
        )

    # ------------------------------------------------------------------
    # AD-004: Flow momentum ranking
    # ------------------------------------------------------------------

    def _flow_momentum_signals(
        self,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        p: _AltDataParams,
        vix: _VixState,
    ) -> list[TradeSignal]:
        """Generate AD-004 flow momentum entry signals for top quintile."""
        signals: list[TradeSignal] = []
        flow_rankings = self._rank_sector_flow(
            indicators_df, p.flow_sma_period, p.flow_volume_zscore_window
        )
        if not flow_rankings:
            return signals

        top_quintile_count = max(1, len(flow_rankings) // 5)
        top_symbols = [sym for sym, _ in flow_rankings[:top_quintile_count]]

        for symbol in top_symbols:
            if symbol in portfolio.positions:
                continue
            if vix.suppressed:
                logger.debug(
                    "AD-004 BUY suppressed for %s: VIX=%.1f >= %.0f",
                    symbol,
                    vix.level,
                    p.vix_suppress_threshold,
                )
                continue
            if len(portfolio.positions) >= self.config.max_positions:
                break

            signal = self._flow_entry(
                indicators_df,
                symbol,
                prices,
                p.flow_sma_period,
                p.stop_atr_multiplier,
            )
            if signal is not None:
                signals.append(signal)

        return signals

    def _flow_entry(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        prices: dict[str, float],
        sma_period: int,
        stop_mult: float,
    ) -> TradeSignal | None:
        """Build a BUY signal for flow momentum, or None."""
        close = prices.get(symbol, 0)
        if close <= 0:
            return None

        # Confirm price > SMA for flow momentum
        sym_data = _sym_series(indicators_df, symbol)
        latest = _latest_row(sym_data)
        if latest is None:
            return None
        sma_col = f"sma_{sma_period}"
        sma_val = latest.get(sma_col)
        if sma_val is None or close <= sma_val:
            return None

        sl = self._compute_stop_loss(indicators_df, symbol, close, stop_mult)
        return TradeSignal(
            symbol=symbol,
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=self.config.target_position_weight,
            stop_loss=sl,
            reasoning=(
                f"AD-004 flow momentum: top quintile sector flow, "
                f"price > SMA_{sma_period}"
            ),
        )

    def _compute_flow_proxy(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        sma_period: int,
        vol_zscore_window: int,
    ) -> float | None:
        """Compute flow proxy for a single symbol.

        flow_proxy = (close - sma_20) / sma_20 * 100 + rel_volume_zscore * 10
        """
        sym_df = _sym_series(indicators_df, symbol)
        latest = _latest_row(sym_df)
        if latest is None:
            return None

        close = latest.get("close")
        sma_col = f"sma_{sma_period}"
        sma_val = latest.get(sma_col)

        if close is None or sma_val is None or sma_val <= 0:
            return None

        price_deviation = (close - sma_val) / sma_val * 100.0

        # Relative volume z-score
        volumes = sym_df.get_column("volume").to_list()
        if len(volumes) < vol_zscore_window:
            return None

        vol_mean = _rolling_mean_list(volumes, vol_zscore_window)
        vol_std = _rolling_std_list(volumes, vol_zscore_window)
        if vol_mean is None or vol_std is None or vol_std == 0:
            return price_deviation  # Fall back to price-only proxy

        rel_vol_zscore = (volumes[-1] - vol_mean) / vol_std
        return price_deviation + rel_vol_zscore * 10.0

    def _rank_sector_flow(
        self,
        indicators_df: pl.DataFrame,
        sma_period: int,
        vol_zscore_window: int,
    ) -> list[tuple[str, float]]:
        """Rank sector ETFs by flow proxy, descending."""
        rankings: list[tuple[str, float]] = []
        available_symbols = (
            indicators_df.select("symbol").unique().to_series().to_list()
        )

        for symbol in SECTOR_ETFS:
            if symbol not in available_symbols:
                continue
            fp = self._compute_flow_proxy(
                indicators_df, symbol, sma_period, vol_zscore_window
            )
            if fp is not None:
                rankings.append((symbol, fp))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # AD-005: Flow reversal (contrarian at extremes)
    # ------------------------------------------------------------------

    def _flow_reversal_signals(
        self,
        indicators_df: pl.DataFrame,
        symbols: list[str],
        portfolio: Portfolio,
        prices: dict[str, float],
        p: _AltDataParams,
    ) -> list[TradeSignal]:
        """Generate AD-005 flow reversal contrarian entry signals."""
        signals: list[TradeSignal] = []

        for symbol in symbols:
            if symbol in portfolio.positions:
                continue
            if len(portfolio.positions) >= self.config.max_positions:
                break

            if not _is_extreme_outflow(
                indicators_df,
                symbol,
                p.flow_sma_period,
                p.flow_reversal_percentile,
                p.flow_reversal_lookback,
            ):
                continue

            close = prices.get(symbol, 0)
            if close <= 0:
                continue

            sl = self._compute_stop_loss(
                indicators_df, symbol, close, p.stop_atr_multiplier
            )
            half_weight = self.config.target_position_weight * 0.5
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.LOW,
                    target_weight=half_weight,
                    stop_loss=sl,
                    reasoning=(
                        f"AD-005 flow reversal: flow proxy in bottom "
                        f"{p.flow_reversal_percentile:.0f}th percentile "
                        f"(contrarian long, half-size)"
                    ),
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Stop-loss helper
    # ------------------------------------------------------------------

    def _compute_stop_loss(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        close: float,
        stop_mult: float,
    ) -> float:
        """ATR-based stop-loss with percentage fallback."""
        sym_data = _sym_series(indicators_df, symbol)
        if "atr_14" in sym_data.columns and len(sym_data) > 0:
            atr = sym_data.tail(1).row(0, named=True).get("atr_14")
            if atr and atr > 0:
                return close - (stop_mult * atr)
        return close * (1.0 - self.config.stop_loss_pct)


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no self access needed)
# ---------------------------------------------------------------------------


def _classify_divergence(
    indicators_df: pl.DataFrame,
    symbol: str,
    attention_period: int,
    volume_zscore_window: int,
) -> str | None:
    """Classify attention-price divergence for a symbol.

    Returns:
        "institutional" -- declining volume + rising price (BUY signal)
        "retail_fomo"   -- rising volume + rising price (CLOSE signal)
        None            -- no actionable divergence
    """
    sym_df = _sym_series(indicators_df, symbol)
    min_history = max(attention_period, volume_zscore_window, 63)
    if len(sym_df) < min_history:
        return None

    volumes = sym_df.get_column("volume").to_list()
    closes = sym_df.get_column("close").to_list()

    # Guard: need enough data for attention proxy and price return
    if len(volumes) < attention_period + 1 or len(closes) <= attention_period:
        return None

    vol_zscore = _compute_attention_zscore(
        volumes, attention_period, volume_zscore_window
    )
    if vol_zscore is None:
        return None

    price_return = (closes[-1] / closes[-attention_period - 1]) - 1.0
    # Only classify when price is rising (> 1% return threshold)
    if price_return <= 0.01:
        return None

    # Quadrant classification: declining vol = institutional, rising = FOMO
    return _vol_zscore_to_quadrant(vol_zscore)


def _vol_zscore_to_quadrant(vol_zscore: float) -> str | None:
    """Map volume z-score to divergence quadrant label."""
    if vol_zscore < -0.5:
        return "institutional"
    if vol_zscore > 0.5:
        return "retail_fomo"
    return None


def _compute_attention_zscore(
    volumes: list[float],
    attention_period: int,
    zscore_window: int,
) -> float | None:
    """Z-score current relative volume against 63-day history."""
    avg_vol = _rolling_mean_list(volumes[:-1], attention_period)
    if avg_vol is None or avg_vol <= 0:
        return None
    current_rel_vol = volumes[-1] / avg_vol

    rel_vol_history: list[float] = []
    start_idx = max(63, zscore_window)
    for i in range(start_idx, len(volumes)):
        hist_avg = _rolling_mean_list(volumes[:i], attention_period)
        if hist_avg and hist_avg > 0:
            rel_vol_history.append(volumes[i] / hist_avg)

    if len(rel_vol_history) < 20:
        return None

    rv_mean = _rolling_mean_list(rel_vol_history, 63)
    rv_std = _rolling_std_list(rel_vol_history, 63)
    if rv_mean is None or rv_std is None or rv_std == 0:
        return None
    return (current_rel_vol - rv_mean) / rv_std


def _is_extreme_outflow(
    indicators_df: pl.DataFrame,
    symbol: str,
    sma_period: int,
    reversal_percentile: float,
    reversal_lookback: int,
) -> bool:
    """Check if symbol has extreme outflow (bottom Nth percentile).

    Returns True if flow proxy is in the bottom *reversal_percentile*
    of its trailing *reversal_lookback* history.
    """
    sym_df = _sym_series(indicators_df, symbol)
    if len(sym_df) < reversal_lookback:
        return False

    sma_col = f"sma_{sma_period}"
    if sma_col not in sym_df.columns or "volume" not in sym_df.columns:
        return False

    tail = sym_df.tail(reversal_lookback)
    flow_values: list[float] = []

    for row in tail.iter_rows(named=True):
        close = row.get("close")
        sma_val = row.get(sma_col)
        if close is None or sma_val is None or sma_val <= 0:
            continue
        flow_values.append((close - sma_val) / sma_val * 100.0)

    if len(flow_values) < reversal_lookback // 2:
        return False

    pct = _percentile_rank(flow_values, len(flow_values))
    if pct is None:
        return False
    return pct <= reversal_percentile


# ---------------------------------------------------------------------------
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "alt_data_signals": AltDataSignalsStrategy,
}

"""LLM-Alpha strategy implementation.

Implements three OHLCV-proxy signals -- FOMC rate sensitivity (H-LA-002),
macro surprise detection (H-LA-007), and communication complexity vol
overlay (H-LA-013) -- using only price and volume data.  No LLM calls
are made during backtesting; all signals are proxied from OHLCV.

See data/strategies/llm-alpha/research-spec.yaml for the frozen research
design.
"""

from __future__ import annotations

import logging
import math
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
# Helper: rolling statistics from plain lists
# ---------------------------------------------------------------------------


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


def _daily_returns(closes: list[float]) -> list[float]:
    """Compute daily returns from a list of close prices."""
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation between two equal-length lists."""
    n = len(x)
    if n < 5 or len(y) != n:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((v - mean_x) ** 2 for v in x)
    var_y = sum((v - mean_y) ** 2 for v in y)

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


# ---------------------------------------------------------------------------
# Parameter / state bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SizingOverlay:
    """Complexity overlay output: stop multiplier and weight scaling."""

    stop_mult: float
    weight_multiplier: float


# ---------------------------------------------------------------------------
# LLM-Alpha Strategy
# ---------------------------------------------------------------------------


class LLMAlphaStrategy(Strategy):
    """LLM-alpha proxy: FOMC sensitivity, macro surprise, complexity overlay.

    Signal hierarchy:
      1. Macro surprise detector (H-LA-007 proxy): event-driven, immediate
      2. FOMC rate sensitivity (H-LA-002 proxy): TLT volatility and correlation
      3. Complexity overlay (H-LA-013 proxy): ATR-based vol sizing adjustment

    Rebalance is event-driven on surprise days, 21-day baseline otherwise.

    All indicators are OHLCV-derived proxies:
      - FOMC proxy: TLT 20-day avg |daily_return| + TLT-SPY correlation
      - Macro surprise: SPY |daily_return| > 2 * 20-day vol
      - Complexity overlay: ATR_14/close percentile-ranked over 63 days
    """

    # Symbols this strategy trades
    FOMC_SYMBOL: ClassVar[str] = "TLT"
    EQUITY_SYMBOLS: ClassVar[list[str]] = ["SPY", "QQQ"]
    SAFE_HAVEN_SYMBOLS: ClassVar[list[str]] = ["TLT", "GLD"]

    # Volatility baseline multiplier for FOMC proxy activation
    TLT_VOL_ACTIVATION: ClassVar[float] = 1.2

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters
        vol_lookback: int = params.get("vol_lookback", 20)

        # Complexity overlay (H-LA-013 proxy) -- computed first for sizing
        overlay = self._compute_sizing_overlay(indicators_df, params)

        # Macro surprise detector (H-LA-007 proxy) -- event-driven
        surprise = _detect_macro_surprise(indicators_df, vol_lookback)
        if surprise is not None:
            direction, spy_return = surprise
            return self._macro_surprise_signals(
                direction,
                spy_return,
                portfolio,
                prices,
                indicators_df,
                overlay,
            )

        # FOMC rate sensitivity (H-LA-002 proxy)
        fomc_signal = _fomc_sensitivity_signal(indicators_df, vol_lookback)
        if fomc_signal is not None:
            return self._fomc_signals(
                fomc_signal,
                portfolio,
                prices,
                indicators_df,
                overlay,
            )

        return []

    def _compute_sizing_overlay(
        self,
        indicators_df: pl.DataFrame,
        params: dict,
    ) -> _SizingOverlay:
        """Compute complexity overlay for sizing/stops."""
        stop_mult: float = params.get("stop_atr_multiplier", 2.0)
        high_pct: float = params.get("complexity_high_threshold", 75.0)
        sizing_factor: float = params.get("complexity_sizing_factor", 0.5)
        lookback: int = params.get("complexity_lookback", 63)

        state = _complexity_state(indicators_df, lookback, high_pct)
        if state == "HIGH":
            logger.debug("Complexity overlay HIGH: halving sizes, widening stops")
            return _SizingOverlay(
                stop_mult=3.0 * (stop_mult / 2.0),
                weight_multiplier=sizing_factor,
            )
        return _SizingOverlay(stop_mult=stop_mult, weight_multiplier=1.0)

    # ------------------------------------------------------------------
    # H-LA-002 proxy: FOMC signals
    # ------------------------------------------------------------------

    def _fomc_signals(
        self,
        direction: str,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        overlay: _SizingOverlay,
    ) -> list[TradeSignal]:
        """Generate signals from FOMC sensitivity proxy."""
        signals: list[TradeSignal] = []
        weight = self.config.target_position_weight * overlay.weight_multiplier

        if direction == "dovish":
            signal = self._dovish_entry(
                portfolio, prices, indicators_df, overlay.stop_mult, weight
            )
            if signal is not None:
                signals.append(signal)

        elif direction == "hawkish" and self.FOMC_SYMBOL in portfolio.positions:
            signals.append(
                TradeSignal(
                    symbol=self.FOMC_SYMBOL,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        "H-LA-002 FOMC proxy: hawkish signal "
                        "(TLT vol elevated, positive TLT-SPY correlation "
                        "= rate-driven selloff)"
                    ),
                )
            )

        return signals

    def _dovish_entry(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        target_weight: float,
    ) -> TradeSignal | None:
        """Build a BUY TLT signal for dovish FOMC, or None."""
        symbol = self.FOMC_SYMBOL
        if symbol in portfolio.positions:
            return None
        close = prices.get(symbol, 0)
        if close <= 0 or len(portfolio.positions) >= self.config.max_positions:
            return None
        sl = self._compute_stop_loss(indicators_df, symbol, close, stop_mult)
        return TradeSignal(
            symbol=symbol,
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=target_weight,
            stop_loss=sl,
            reasoning=(
                "H-LA-002 FOMC proxy: dovish signal "
                "(TLT vol elevated, negative TLT-SPY correlation "
                "= flight to quality)"
            ),
        )

    # ------------------------------------------------------------------
    # H-LA-007 proxy: Macro surprise signals
    # ------------------------------------------------------------------

    def _macro_surprise_signals(
        self,
        direction: str,
        spy_return: float,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        overlay: _SizingOverlay,
    ) -> list[TradeSignal]:
        """Generate signals on macro surprise days."""
        weight = self.config.target_position_weight * overlay.weight_multiplier

        if direction == "positive":
            return self._positive_surprise(
                spy_return,
                portfolio,
                prices,
                indicators_df,
                overlay.stop_mult,
                weight,
            )
        return self._negative_surprise(
            spy_return,
            portfolio,
            prices,
            indicators_df,
            overlay.stop_mult,
            weight,
        )

    def _positive_surprise(
        self,
        spy_return: float,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        target_weight: float,
    ) -> list[TradeSignal]:
        """Positive macro surprise: BUY SPY/QQQ."""
        signals: list[TradeSignal] = []
        for symbol in self.EQUITY_SYMBOLS:
            if symbol in portfolio.positions:
                continue
            if len(portfolio.positions) >= self.config.max_positions:
                break
            close = prices.get(symbol, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(indicators_df, symbol, close, stop_mult)
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=target_weight,
                    stop_loss=sl,
                    reasoning=(
                        f"H-LA-007 macro surprise: positive "
                        f"(SPY return={spy_return:+.2%} > 2x vol), "
                        f"long equities"
                    ),
                )
            )
        return signals

    def _negative_surprise(
        self,
        spy_return: float,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        target_weight: float,
    ) -> list[TradeSignal]:
        """Negative macro surprise: close equities, BUY TLT/GLD."""
        signals: list[TradeSignal] = []

        # Close equity positions
        signals.extend(
            TradeSignal(
                symbol=symbol,
                action=Action.CLOSE,
                conviction=Conviction.HIGH,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=(
                    f"H-LA-007 macro surprise: negative "
                    f"(SPY return={spy_return:+.2%} > 2x vol), "
                    f"exit equities"
                ),
            )
            for symbol in self.EQUITY_SYMBOLS
            if symbol in portfolio.positions
        )

        # Enter safe havens
        for symbol in self.SAFE_HAVEN_SYMBOLS:
            if symbol in portfolio.positions:
                continue
            if len(portfolio.positions) >= self.config.max_positions:
                break
            close = prices.get(symbol, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(indicators_df, symbol, close, stop_mult)
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.HIGH,
                    target_weight=target_weight,
                    stop_loss=sl,
                    reasoning=(
                        f"H-LA-007 macro surprise: negative "
                        f"(SPY return={spy_return:+.2%} > 2x vol), "
                        f"risk-off safe haven"
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


def _fomc_sensitivity_signal(
    indicators_df: pl.DataFrame,
    vol_lookback: int,
) -> str | None:
    """Compute FOMC directional proxy from TLT behavior.

    Proxy logic:
      - TLT 20-day avg |daily_return| measures rate sensitivity (vol).
      - TLT-SPY 20-day correlation measures hawkish/dovish direction.
      - Only act when TLT vol > 1.2x its baseline (indicating rate event).

    Returns "dovish", "hawkish", or None.
    """
    tlt_df = _sym_series(indicators_df, "TLT")
    spy_df = _sym_series(indicators_df, "SPY")

    if len(tlt_df) < vol_lookback + 1 or len(spy_df) < vol_lookback + 1:
        return None

    tlt_returns = _daily_returns(tlt_df.get_column("close").to_list())
    if len(tlt_returns) < vol_lookback:
        return None

    if not _is_tlt_vol_elevated(tlt_returns, vol_lookback):
        return None

    spy_returns = _daily_returns(spy_df.get_column("close").to_list())
    if len(spy_returns) < vol_lookback:
        return None

    tlt_r = tlt_returns[-vol_lookback:]
    spy_r = spy_returns[-vol_lookback:]
    return _direction_from_correlation(_pearson_correlation(tlt_r, spy_r))


def _is_tlt_vol_elevated(tlt_returns: list[float], vol_lookback: int) -> bool:
    """Return True if TLT recent vol > 1.2x baseline vol."""
    recent_abs = [abs(r) for r in tlt_returns[-vol_lookback:]]
    tlt_avg_abs = sum(recent_abs) / vol_lookback

    all_abs = [abs(r) for r in tlt_returns]
    baseline = sum(all_abs) / len(all_abs)
    if baseline == 0:
        return False
    return tlt_avg_abs >= LLMAlphaStrategy.TLT_VOL_ACTIVATION * baseline


def _direction_from_correlation(correlation: float | None) -> str | None:
    """Map TLT-SPY correlation to dovish/hawkish/None."""
    if correlation is None:
        return None
    if correlation < -0.1:
        return "dovish"
    if correlation > 0.1:
        return "hawkish"
    return None


def _detect_macro_surprise(
    indicators_df: pl.DataFrame,
    vol_lookback: int,
) -> tuple[str, float] | None:
    """Detect macro surprise days from SPY price action.

    A surprise day is when SPY |daily_return| > 2 * 20-day rolling vol.
    Returns ("positive"|"negative", return_value), or None.
    """
    spy_df = _sym_series(indicators_df, "SPY")
    if len(spy_df) < vol_lookback + 2:
        return None

    returns = _daily_returns(spy_df.get_column("close").to_list())
    if len(returns) < vol_lookback + 1:
        return None

    latest_return = returns[-1]
    lookback_returns = returns[-(vol_lookback + 1) : -1]
    vol = _rolling_std_list(lookback_returns, vol_lookback)
    if vol is None or vol == 0:
        return None

    if abs(latest_return) <= 2.0 * vol:
        return None

    direction = "positive" if latest_return > 0 else "negative"
    return (direction, latest_return)


def _complexity_state(
    indicators_df: pl.DataFrame,
    lookback: int,
    high_threshold_pct: float,
) -> str:
    """Compute complexity proxy from ATR_14/close normalized vol.

    Uses SPY as the reference asset for market-wide complexity.
    Returns "HIGH", "LOW", or "NORMAL".
    """
    spy_df = _sym_series(indicators_df, "SPY")
    if len(spy_df) < lookback or "atr_14" not in spy_df.columns:
        return "NORMAL"

    norm_vol_values = _build_norm_vol_series(spy_df, lookback)
    if len(norm_vol_values) < lookback // 2:
        return "NORMAL"

    pct = _percentile_rank(norm_vol_values, len(norm_vol_values))
    if pct is None:
        return "NORMAL"

    if pct >= high_threshold_pct:
        return "HIGH"
    if pct <= (100.0 - high_threshold_pct):
        return "LOW"
    return "NORMAL"


def _build_norm_vol_series(spy_df: pl.DataFrame, lookback: int) -> list[float]:
    """Build ATR/close normalized vol series from the tail."""
    tail = spy_df.tail(lookback)
    result: list[float] = []
    for row in tail.iter_rows(named=True):
        close = row.get("close")
        atr = row.get("atr_14")
        if close is not None and close > 0 and atr is not None and atr > 0:
            result.append(atr / close)
    return result


# ---------------------------------------------------------------------------
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "llm_alpha": LLMAlphaStrategy,
}

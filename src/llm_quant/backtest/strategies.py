"""Additional strategy implementations for backtesting.

Each strategy follows the Strategy ABC contract:
- generate_signals() receives only causal data (up to as_of_date)
- Returns a list of TradeSignals
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from llm_quant.backtest.strategy import SMACrossoverStrategy, Strategy, StrategyConfig
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSI Mean Reversion
# ---------------------------------------------------------------------------


class RSIMeanReversionStrategy(Strategy):
    """Mean-reversion strategy based on RSI extremes.

    Buys when RSI < oversold threshold, sells when RSI > overbought.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        params = self.config.parameters
        oversold = params.get("rsi_oversold", 30)
        overbought = params.get("rsi_overbought", 70)

        symbols = indicators_df.select("symbol").unique().to_series().to_list()

        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < 2 or "rsi_14" not in sym_data.columns:
                continue

            curr = sym_data.tail(1).row(0, named=True)
            rsi = curr.get("rsi_14")
            close = curr["close"]
            if rsi is None:
                continue

            has_position = symbol in portfolio.positions

            if (
                rsi < oversold
                and not has_position
                and len(portfolio.positions) < self.config.max_positions
            ):
                stop_loss = close * (1.0 - self.config.stop_loss_pct)
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.BUY,
                        conviction=Conviction.MEDIUM,
                        target_weight=self.config.target_position_weight,
                        stop_loss=stop_loss,
                        reasoning=f"RSI oversold ({rsi:.1f} < {oversold})",
                    )
                )
            elif rsi > overbought and has_position:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"RSI overbought ({rsi:.1f} > {overbought})",
                    )
                )

        return signals


class MomentumStrategy(Strategy):
    """Cross-sectional momentum: buy top-N performers, sell bottom-N.

    Ranks symbols by trailing return over lookback_days, then buys the
    top_n and exits positions that drop below top_n ranking.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        params = self.config.parameters
        lookback = params.get("lookback_days", 63)
        top_n = params.get("top_n", 5)

        symbols = indicators_df.select("symbol").unique().to_series().to_list()
        momentum_scores: list[tuple[str, float]] = []

        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < lookback:
                continue

            recent = sym_data.tail(lookback)
            first_close = recent.row(0, named=True)["close"]
            last_close = recent.row(-1, named=True)["close"]

            if first_close > 0:
                ret = last_close / first_close - 1.0
                momentum_scores.append((symbol, ret))

        # Rank by momentum
        momentum_scores.sort(key=lambda x: x[1], reverse=True)
        top_symbols = {s for s, _ in momentum_scores[:top_n]}

        # Buy top-N that we don't hold
        for symbol, score in momentum_scores[:top_n]:
            if (
                symbol in portfolio.positions
                or len(portfolio.positions) >= self.config.max_positions
            ):
                continue
            close = prices.get(symbol, 0)
            if close <= 0:
                continue
            stop_loss = close * (1.0 - self.config.stop_loss_pct)
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=self.config.target_position_weight,
                    stop_loss=stop_loss,
                    reasoning=(f"Momentum top-{top_n} ({score:.2%} over {lookback}d)"),
                )
            )

        # Close positions that fell out of top-N
        signals.extend(
            TradeSignal(
                symbol=symbol,
                action=Action.CLOSE,
                conviction=Conviction.LOW,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Dropped from momentum top-{top_n}",
            )
            for symbol in portfolio.positions
            if symbol not in top_symbols
        )

        return signals


# ---------------------------------------------------------------------------
# MACD Trend Following
# ---------------------------------------------------------------------------


class MACDStrategy(Strategy):
    """MACD-based trend following strategy.

    Buys on bullish MACD histogram crossover, sells on bearish.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        symbols = indicators_df.select("symbol").unique().to_series().to_list()

        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < 2:
                continue
            if "macd_hist" not in sym_data.columns:
                continue

            last = sym_data.tail(2)
            prev = last.row(0, named=True)
            curr = last.row(1, named=True)

            prev_hist = prev.get("macd_hist")
            curr_hist = curr.get("macd_hist")
            close = curr["close"]

            if prev_hist is None or curr_hist is None:
                continue

            has_position = symbol in portfolio.positions

            # Bullish crossover: histogram turns positive
            if (
                prev_hist <= 0
                and curr_hist > 0
                and not has_position
                and len(portfolio.positions) < self.config.max_positions
            ):
                stop_loss = close * (1.0 - self.config.stop_loss_pct)
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.BUY,
                        conviction=Conviction.MEDIUM,
                        target_weight=self.config.target_position_weight,
                        stop_loss=stop_loss,
                        reasoning=f"MACD histogram bullish crossover ({curr_hist:.4f})",
                    )
                )
            # Bearish crossover: histogram turns negative
            elif prev_hist >= 0 and curr_hist < 0 and has_position:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"MACD histogram bearish crossover ({curr_hist:.4f})",
                    )
                )

        return signals


# ---------------------------------------------------------------------------
# Regime-Aware Momentum
# ---------------------------------------------------------------------------


class RegimeMomentumStrategy(Strategy):
    """Regime-aware momentum: adjusts exposure based on VIX and SMA200.

    Risk-on regime: full momentum allocation
    Transition: half allocation
    Risk-off: reduce to defensive positions only
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        params = self.config.parameters
        vix_risk_off = params.get("vix_risk_off", 25)
        vix_transition = params.get("vix_transition", 20)
        lookback = params.get("lookback_days", 63)
        top_n = params.get("top_n", 5)
        defensive_symbols = set(params.get("defensive_symbols", ["TLT", "GLD", "IEF"]))

        # Determine regime from VIX (if available)
        vix_data = indicators_df.filter(pl.col("symbol") == "VIX").sort("date")
        regime = "risk_on"
        if len(vix_data) > 0:
            vix_close = vix_data.tail(1).row(0, named=True)["close"]
            if vix_close >= vix_risk_off:
                regime = "risk_off"
            elif vix_close >= vix_transition:
                regime = "transition"

        # Also check SPY SMA200 trend
        spy_data = indicators_df.filter(pl.col("symbol") == "SPY").sort("date")
        if len(spy_data) > 0 and "sma_200" in spy_data.columns:
            spy_row = spy_data.tail(1).row(0, named=True)
            sma200 = spy_row.get("sma_200")
            if sma200 is not None and spy_row["close"] < sma200:
                if regime == "risk_on":
                    regime = "transition"
                elif regime == "transition":
                    regime = "risk_off"

        # Adjust allocation based on regime
        if regime == "risk_off":
            weight_mult = 0.5
            max_pos = min(3, self.config.max_positions)
        elif regime == "transition":
            weight_mult = 0.75
            max_pos = self.config.max_positions
        else:
            weight_mult = 1.0
            max_pos = self.config.max_positions

        target_weight = self.config.target_position_weight * weight_mult

        # Compute momentum scores (excluding VIX)
        symbols = [
            s
            for s in indicators_df.select("symbol").unique().to_series().to_list()
            if s != "VIX"
        ]
        scores: list[tuple[str, float]] = []

        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < lookback:
                continue
            recent = sym_data.tail(lookback)
            first_close = recent.row(0, named=True)["close"]
            last_close = recent.row(-1, named=True)["close"]
            if first_close > 0:
                scores.append((symbol, last_close / first_close - 1.0))

        scores.sort(key=lambda x: x[1], reverse=True)

        # In risk-off, prefer defensive symbols
        if regime == "risk_off":
            defensive = [s for s in scores if s[0] in defensive_symbols]
            offensive = [s for s in scores if s[0] not in defensive_symbols]
            ranked = defensive + offensive
        else:
            ranked = scores

        top_symbols = {s for s, _ in ranked[:top_n]}

        # Generate buy signals
        for symbol, score in ranked[:top_n]:
            if symbol not in portfolio.positions and len(portfolio.positions) < max_pos:
                close = prices.get(symbol, 0)
                if close <= 0:
                    continue
                stop_loss = close * (1.0 - self.config.stop_loss_pct)
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.BUY,
                        conviction=Conviction.MEDIUM,
                        target_weight=target_weight,
                        stop_loss=stop_loss,
                        reasoning=(
                            f"Regime={regime}, momentum rank top-{top_n} "
                            f"({score:.2%} over {lookback}d)"
                        ),
                    )
                )

        # Close positions not in top-N or exceeding regime limits
        signals.extend(
            TradeSignal(
                symbol=symbol,
                action=Action.CLOSE,
                conviction=Conviction.LOW,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Regime={regime}, dropped from top-{top_n}",
            )
            for symbol in portfolio.positions
            if symbol not in top_symbols
        )

        return signals


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "momentum": MomentumStrategy,
    "macd": MACDStrategy,
    "regime_momentum": RegimeMomentumStrategy,
}


def create_strategy(name: str, config: StrategyConfig) -> Strategy:
    """Create a strategy instance by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(config)

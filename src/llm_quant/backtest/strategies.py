"""Additional strategy implementations for backtesting.

Each strategy follows the Strategy ABC contract:
- generate_signals() receives only causal data (up to as_of_date)
- Returns a list of TradeSignals
"""

from __future__ import annotations

import logging
import math
from datetime import date

import polars as pl

from llm_quant.arb.cef_strategy import CEFDiscountRegistryStrategy
from llm_quant.backtest.nlp_signal_strategy import NlpSignalStrategy
from llm_quant.backtest.strategy import SMACrossoverStrategy, Strategy, StrategyConfig
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.signals.tsmom import TsmomCalculator, TsmomConfig
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


def _compute_momentum_scores(
    indicators_df: pl.DataFrame,
    symbols: list[str],
    lookback: int,
) -> list[tuple[str, float]]:
    """Compute trailing-return momentum scores for given symbols."""
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
    return scores


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
        momentum_scores = _compute_momentum_scores(indicators_df, symbols, lookback)
        top_symbols = {s for s, _ in momentum_scores[:top_n]}
        scored_symbols = {s for s, _ in momentum_scores}

        # Buy top-N that we don't hold
        new_positions = 0
        for symbol, score in momentum_scores[:top_n]:
            if (
                symbol in portfolio.positions
                or len(portfolio.positions) + new_positions >= self.config.max_positions
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
            new_positions += 1

        # Close positions that fell out of top-N (only for symbols this strategy scored)
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
            if symbol not in top_symbols and symbol in scored_symbols
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

    def _detect_regime(
        self,
        indicators_df: pl.DataFrame,
        vix_risk_off: float,
        vix_transition: float,
    ) -> str:
        """Classify market regime from VIX level and SPY vs SMA200."""
        regime = "risk_on"
        vix_data = indicators_df.filter(pl.col("symbol") == "VIX").sort("date")
        if len(vix_data) > 0:
            vix_close = vix_data.tail(1).row(0, named=True)["close"]
            if vix_close >= vix_risk_off:
                regime = "risk_off"
            elif vix_close >= vix_transition:
                regime = "transition"

        spy_data = indicators_df.filter(pl.col("symbol") == "SPY").sort("date")
        if len(spy_data) > 0 and "sma_200" in spy_data.columns:
            spy_row = spy_data.tail(1).row(0, named=True)
            sma200 = spy_row.get("sma_200")
            if sma200 is not None and spy_row["close"] < sma200:
                if regime == "risk_on":
                    regime = "transition"
                elif regime == "transition":
                    regime = "risk_off"
        return regime

    def _compute_regime_momentum_scores(
        self,
        indicators_df: pl.DataFrame,
        lookback: int,
    ) -> list[tuple[str, float]]:
        """Compute trailing-return momentum scores, filtered by SMA200."""
        symbols = [
            s
            for s in indicators_df.select("symbol").unique().to_series().to_list()
            if s != "VIX"
        ]
        scores = _compute_momentum_scores(indicators_df, symbols, lookback)

        # Filter out symbols trading below their 200-day SMA
        if "sma_200" in indicators_df.columns:
            filtered: list[tuple[str, float]] = []
            for symbol, ret in scores:
                sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
                if len(sym_data) == 0:
                    continue
                row = sym_data.tail(1).row(0, named=True)
                sma200 = row.get("sma_200")
                if sma200 is None or row["close"] >= sma200:
                    filtered.append((symbol, ret))
            scores = filtered

        return scores

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        params = self.config.parameters
        vix_risk_off = params.get(
            "vix_risk_off_threshold", params.get("vix_risk_off", 25)
        )
        vix_transition = params.get(
            "vix_risk_on_threshold", params.get("vix_transition", 20)
        )
        lookback = params.get("momentum_lookback", params.get("lookback_days", 63))
        top_n = params.get("top_n_momentum", params.get("top_n", 5))
        stop_mult = params.get("stop_atr_multiplier", 2.0)
        defensive_symbols = set(
            params.get(
                "defensive_symbols",
                ["TLT", "IEF", "SHY", "GLD", "TIP", "XLP", "XLU", "XLV"],
            )
        )

        regime = self._detect_regime(indicators_df, vix_risk_off, vix_transition)

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
        scores = self._compute_regime_momentum_scores(indicators_df, lookback)

        # In risk-off, prefer defensive symbols
        if regime == "risk_off":
            defensive = [s for s in scores if s[0] in defensive_symbols]
            offensive = [s for s in scores if s[0] not in defensive_symbols]
            ranked = defensive + offensive
        else:
            ranked = scores

        top_symbols = {s for s, _ in ranked[:top_n]}
        scored_symbols = {s for s, _ in scores}

        # Generate buy signals
        new_positions = 0
        for symbol, score in ranked[:top_n]:
            if (
                symbol not in portfolio.positions
                and len(portfolio.positions) + new_positions < max_pos
            ):
                close = prices.get(symbol, 0)
                if close <= 0:
                    continue

                # ATR-based stop-loss
                sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
                atr_col = "atr_14"
                if atr_col in sym_data.columns and len(sym_data) > 0:
                    atr_val = sym_data.tail(1).row(0, named=True).get(atr_col)
                    if atr_val and atr_val > 0:
                        stop_loss = close - (stop_mult * atr_val)
                    else:
                        stop_loss = close * (1.0 - self.config.stop_loss_pct)
                else:
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
                new_positions += 1

        # Close positions not in top-N (only for symbols this strategy scored)
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
            if symbol not in top_symbols and symbol in scored_symbols
        )

        return signals


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _detect_regime_from_vix(
    indicators_df: pl.DataFrame,
    vix_threshold: float,
) -> str:
    """Classify regime as risk_on or risk_off based on VIX level."""
    vix_data = indicators_df.filter(pl.col("symbol") == "VIX").sort("date")
    if len(vix_data) > 0:
        vix_close = vix_data.tail(1).row(0, named=True)["close"]
        if vix_close >= vix_threshold:
            return "risk_off"
    return "risk_on"


def _get_atr_stop(
    sym_data: pl.DataFrame,
    close: float,
    stop_mult: float,
    fallback_pct: float,
) -> float:
    """Compute ATR-based stop-loss, falling back to percentage-based."""
    if "atr_14" in sym_data.columns and len(sym_data) > 0:
        atr_val = sym_data.tail(1).row(0, named=True).get("atr_14")
        if atr_val and atr_val > 0:
            return close - (stop_mult * atr_val)
    return close * (1.0 - fallback_pct)


def _vol_target_weight(
    sym_data: pl.DataFrame,
    base_weight: float,
    target_vol: float,
) -> float:
    """Compute volatility-targeted weight using ATR as vol proxy.

    realized_vol_proxy = ATR_14 * sqrt(252) / close
    weight = base_weight * (target_vol / realized_vol)
    Clamped to [0.01, base_weight * 2].
    """
    if "atr_14" not in sym_data.columns or len(sym_data) == 0:
        return base_weight
    row = sym_data.tail(1).row(0, named=True)
    atr_val = row.get("atr_14")
    close = row.get("close", 0)
    if not atr_val or atr_val <= 0 or not close or close <= 0:
        return base_weight
    realized_vol = atr_val * math.sqrt(252) / close
    if realized_vol <= 0:
        return base_weight
    weight = base_weight * (target_vol / realized_vol)
    return max(0.01, min(weight, base_weight * 2))


def _trailing_return(sym_data: pl.DataFrame, lookback: int) -> float | None:
    """Compute trailing return over lookback days. Returns None if insufficient data."""
    if len(sym_data) < lookback:
        return None
    recent = sym_data.tail(lookback)
    first_close = recent.row(0, named=True)["close"]
    last_close = recent.row(-1, named=True)["close"]
    if first_close <= 0:
        return None
    return last_close / first_close - 1.0


def _close_above_sma(sym_data: pl.DataFrame, sma_col: str = "sma_200") -> bool:
    """Check if latest close is above the given SMA. True if SMA not available."""
    if sma_col not in sym_data.columns or len(sym_data) == 0:
        return True  # no SMA data = no filter applied
    row = sym_data.tail(1).row(0, named=True)
    sma_val = row.get(sma_col)
    if sma_val is None:
        return True
    return row["close"] >= sma_val


# ---------------------------------------------------------------------------
# Trend Following (time-series momentum)
# ---------------------------------------------------------------------------


class TrendFollowingStrategy(Strategy):
    """Time-series momentum: go long each asset with positive trailing return.

    Unlike cross-sectional momentum, each asset is evaluated independently.
    Long if: 126d return > 0 AND close > SMA_200.
    Flat if: 126d return <= 0 OR close < SMA_200.
    """

    def _evaluate_symbols(
        self,
        indicators_df: pl.DataFrame,
        symbols: list[str],
        portfolio: Portfolio,
        prices: dict[str, float],
        params: dict,
    ) -> list[TradeSignal]:
        """Evaluate each symbol independently for trend-following signals."""
        signals: list[TradeSignal] = []
        lookback = params["lookback"]
        sma_col = params["sma_col"]
        target_vol = params["target_vol"]
        weight_mult_risk_off = params["weight_mult_risk_off"]
        stop_mult = params["stop_mult"]
        regime = params["regime"]

        min_positive = params.get("min_tf_positive", 1)

        new_positions = 0
        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            close = prices.get(symbol, 0)
            if close <= 0 or len(sym_data) < lookback:
                continue

            # Multi-timeframe momentum consensus
            lookbacks = [
                params.get("lookback_short"),
                params.get("lookback_medium"),
                params.get("lookback_long"),
            ]
            lookbacks = [lb for lb in lookbacks if lb is not None]
            if not lookbacks:
                lookbacks = [lookback]  # fallback to single timeframe

            timeframe_returns: list[tuple[int, float]] = []
            for lb in lookbacks:
                ret = _trailing_return(sym_data, lb)
                if ret is not None:
                    timeframe_returns.append((lb, ret))

            positive_count = sum(1 for _, r in timeframe_returns if r > 0)
            above_sma = _close_above_sma(sym_data, sma_col)
            has_position = symbol in portfolio.positions
            is_bullish = positive_count >= min_positive and above_sma

            # Momentum acceleration: short > medium suggests strengthening trend
            has_acceleration = False
            if len(timeframe_returns) >= 2:
                sorted_tfs = sorted(timeframe_returns, key=lambda x: x[0])
                short_ret = sorted_tfs[0][1]
                med_ret = sorted_tfs[len(sorted_tfs) // 2][1]
                if short_ret > med_ret:
                    has_acceleration = True

            if is_bullish and not has_position:
                if (
                    len(portfolio.positions) + new_positions
                    >= self.config.max_positions
                ):
                    continue
                base_weight = self.config.target_position_weight
                if regime == "risk_off":
                    base_weight *= weight_mult_risk_off
                weight = _vol_target_weight(sym_data, base_weight, target_vol)
                stop_loss = _get_atr_stop(
                    sym_data, close, stop_mult, self.config.stop_loss_pct
                )

                if has_acceleration and positive_count == len(timeframe_returns):
                    conviction = Conviction.HIGH
                else:
                    conviction = Conviction.MEDIUM

                tf_summary = ", ".join(f"{lb}d={r:.2%}" for lb, r in timeframe_returns)
                reasoning = (
                    f"Trend-following: [{tf_summary}], "
                    f"{positive_count}/{len(timeframe_returns)} positive, "
                    f"above SMA200, regime={regime}"
                )
                if has_acceleration:
                    reasoning += ", accelerating"

                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.BUY,
                        conviction=conviction,
                        target_weight=weight,
                        stop_loss=stop_loss,
                        reasoning=reasoning,
                    )
                )
                new_positions += 1
            elif not is_bullish and has_position:
                tf_summary = ", ".join(f"{lb}d={r:.2%}" for lb, r in timeframe_returns)
                reason = (
                    f"Trend-following exit: [{tf_summary}], "
                    f"{positive_count}/{len(timeframe_returns)} positive"
                )
                if not above_sma:
                    reason += ", below SMA200"
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=reason,
                    )
                )

        return signals

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters
        lookback = params.get("lookback_days", 126)
        sma_trend = params.get("sma_trend", 200)
        vix_threshold = params.get("vix_threshold", 22)

        regime = _detect_regime_from_vix(indicators_df, vix_threshold)

        symbols = [
            s
            for s in indicators_df.select("symbol").unique().to_series().to_list()
            if s != "VIX"
        ]

        lookback_short = params.get("lookback_short", None)
        lookback_medium = params.get("lookback_medium", lookback)
        lookback_long = params.get("lookback_long", None)
        min_tf_positive = params.get("min_timeframes_positive", 1)

        eval_params: dict = {
            "lookback": lookback,
            "sma_col": f"sma_{sma_trend}",
            "target_vol": params.get("target_vol", 0.12),
            "weight_mult_risk_off": params.get("weight_mult_risk_off", 0.50),
            "stop_mult": params.get("stop_atr_multiplier", 1.5),
            "regime": regime,
            "min_tf_positive": min_tf_positive,
        }
        if lookback_short is not None:
            eval_params["lookback_short"] = lookback_short
        if lookback_medium != lookback:
            eval_params["lookback_medium"] = lookback_medium
        if lookback_long is not None:
            eval_params["lookback_long"] = lookback_long

        return self._evaluate_symbols(
            indicators_df, symbols, portfolio, prices, eval_params
        )


class MultiFactorStrategy(Strategy):
    """Multi-factor strategy: momentum + value + quality composite ranking.

    Combines three uncorrelated signals:
    - Momentum: 126d trailing return (higher = better)
    - Value: RSI_14 inverted (lower RSI = more value/oversold)
    - Quality: inverse realized volatility (lower vol = higher quality)
    """

    def _score_universe(
        self,
        indicators_df: pl.DataFrame,
        symbols: list[str],
        momentum_lookback: int,
        sma_col: str,
    ) -> list[dict]:
        """Score each symbol on momentum, value, and quality factors."""
        scored: list[dict] = []
        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < momentum_lookback:
                continue
            row = sym_data.tail(1).row(0, named=True)
            close = row["close"]
            if close <= 0:
                continue

            mom = _trailing_return(sym_data, momentum_lookback)
            if mom is None:
                continue

            rsi = row.get("rsi_14") if "rsi_14" in sym_data.columns else None
            if rsi is None:
                continue
            value = 100.0 - rsi

            atr_val = row.get("atr_14") if "atr_14" in sym_data.columns else None
            if atr_val and atr_val > 0:
                vol_proxy = atr_val * math.sqrt(252) / close
                quality = 1.0 / vol_proxy if vol_proxy > 0 else 0.0
            else:
                quality = 0.0

            above_sma = _close_above_sma(sym_data, sma_col)

            scored.append(
                {
                    "symbol": symbol,
                    "momentum": mom,
                    "value": value,
                    "quality": quality,
                    "above_sma": above_sma,
                    "close": close,
                    "sym_data": sym_data,
                }
            )
        return scored

    def _normalize_and_rank(
        self,
        scored: list[dict],
        mom_w: float,
        val_w: float,
        qual_w: float,
    ) -> list[dict]:
        """Z-score normalize factors and compute composite score."""
        if len(scored) < 2:
            return scored

        for factor in ("momentum", "value", "quality"):
            vals = [s[factor] for s in scored]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            for s in scored:
                if std > 0:
                    s[f"{factor}_z"] = (s[factor] - mean) / std
                else:
                    s[f"{factor}_z"] = 0.0

        for s in scored:
            s["composite"] = (
                mom_w * s["momentum_z"] + val_w * s["value_z"] + qual_w * s["quality_z"]
            )

        scored.sort(key=lambda x: x["composite"], reverse=True)
        return scored

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        params = self.config.parameters
        momentum_lookback = params.get(
            "momentum_lookback", params.get("lookback_days", 126)
        )
        top_n = params.get("top_n", 7)
        mom_w = params.get("momentum_weight", 0.40)
        val_w = params.get("value_weight", 0.30)
        qual_w = params.get("quality_weight", 0.30)
        target_vol = params.get("target_vol", 0.12)
        stop_mult = params.get("stop_atr_multiplier", 1.5)
        vix_risk_off = params.get("vix_risk_off", 25)
        sma_trend = params.get("sma_trend", 200)

        regime = _detect_regime_from_vix(indicators_df, vix_risk_off)
        sma_col = f"sma_{sma_trend}"

        symbols = [
            s
            for s in indicators_df.select("symbol").unique().to_series().to_list()
            if s != "VIX"
        ]

        scored = self._score_universe(
            indicators_df, symbols, momentum_lookback, sma_col
        )
        scored = self._normalize_and_rank(scored, mom_w, val_w, qual_w)

        # Filter: composite > 0 AND above SMA trend filter
        eligible = [s for s in scored if s["composite"] > 0 and s["above_sma"]]
        top_symbols = {s["symbol"] for s in eligible[:top_n]}
        all_scored_symbols = {s["symbol"] for s in scored}

        # Generate buy signals for top-N
        new_positions = 0
        for entry in eligible[:top_n]:
            symbol = entry["symbol"]
            if symbol in portfolio.positions:
                continue
            if len(portfolio.positions) + new_positions >= self.config.max_positions:
                continue
            close = prices.get(symbol, 0)
            if close <= 0:
                continue

            base_weight = self.config.target_position_weight
            if regime == "risk_off":
                base_weight *= 0.5
            weight = _vol_target_weight(entry["sym_data"], base_weight, target_vol)
            stop_loss = _get_atr_stop(
                entry["sym_data"], close, stop_mult, self.config.stop_loss_pct
            )

            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=weight,
                    stop_loss=stop_loss,
                    reasoning=(
                        f"Multi-factor top-{top_n}: composite={entry['composite']:.2f} "
                        f"(mom={entry['momentum_z']:.2f}, val={entry['value_z']:.2f}, "
                        f"qual={entry['quality_z']:.2f}), regime={regime}"
                    ),
                )
            )
            new_positions += 1

        # Close positions not in top-N (only for symbols we scored)
        signals.extend(
            TradeSignal(
                symbol=symbol,
                action=Action.CLOSE,
                conviction=Conviction.LOW,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Multi-factor: dropped from top-{top_n}",
            )
            for symbol in portfolio.positions
            if symbol not in top_symbols and symbol in all_scored_symbols
        )

        return signals


# ---------------------------------------------------------------------------
# Correlation Regime Strategy (A8: SPY-TLT correlation flip signal)
# ---------------------------------------------------------------------------


class CorrelationRegimeStrategy(Strategy):
    """Rolling correlation regime strategy (stateless).

    Holds equity_symbol when the N-day rolling correlation between equity_symbol
    and hedge_symbol daily returns is below exit_threshold (normal regime).
    Exits to cash when correlation rises above exit_threshold (stress regime).
    Defaults to SPY (equity) / TLT (hedge) for backward compatibility.

    Parameters (via StrategyConfig.parameters):
      equity_symbol (str, default "SPY"): The equity/risk asset to hold.
      hedge_symbol (str, default "TLT"): The hedge/safe-haven asset for correlation.
      corr_window (int, default 10): Rolling window for correlation.
      corr_exit_threshold (float, default 0.0): Exit equity when corr > this.
      corr_entry_threshold (float, default 0.0): Enter equity when corr <= this.
      spy_weight_risk_on (float, default 0.95): Target weight in normal regime.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        equity_sym: str = str(params.get("equity_symbol", "SPY"))
        hedge_sym: str = str(params.get("hedge_symbol", "TLT"))
        corr_window: int = int(params.get("corr_window", 10))
        exit_thresh: float = float(params.get("corr_exit_threshold", 0.0))
        entry_thresh: float = float(params.get("corr_entry_threshold", 0.0))
        risk_on_weight: float = float(params.get("spy_weight_risk_on", 0.95))

        # ── Compute rolling correlation ───────────────────────────────────────
        eq_data = (
            indicators_df.filter(pl.col("symbol") == equity_sym)
            .sort("date")
            .tail(corr_window + 2)
        )
        hedge_data = (
            indicators_df.filter(pl.col("symbol") == hedge_sym)
            .sort("date")
            .tail(corr_window + 2)
        )

        if len(eq_data) < corr_window + 1 or len(hedge_data) < corr_window + 1:
            return []

        eq_prices = eq_data["close"].to_list()
        hedge_prices = hedge_data["close"].to_list()
        min_len = min(len(eq_prices), len(hedge_prices))

        # Compute daily returns
        eq_rets = [eq_prices[i] / eq_prices[i - 1] - 1.0 for i in range(1, min_len)]
        hedge_rets = [
            hedge_prices[i] / hedge_prices[i - 1] - 1.0 for i in range(1, min_len)
        ]

        if len(eq_rets) < corr_window:
            return []

        def _corr(xs: list[float], ys: list[float]) -> float:
            n = len(xs)
            if n < 2:
                return 0.0
            mx = sum(xs) / n
            my = sum(ys) / n
            cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
            sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
            sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
            if sx == 0 or sy == 0:
                return 0.0
            return cov / (sx * sy)

        corr_now = _corr(eq_rets[-corr_window:], hedge_rets[-corr_window:])

        eq_price = prices.get(equity_sym)
        if eq_price is None or eq_price <= 0:
            return []

        has_eq = equity_sym in portfolio.positions

        # ── Stress regime: correlation positive → exit equity ─────────────────
        if corr_now > exit_thresh and has_eq:
            logger.info(
                "CorrelationRegime: EXIT on %s (corr=%.3f > threshold=%.3f)",
                as_of_date,
                corr_now,
                exit_thresh,
            )
            return [
                TradeSignal(
                    symbol=equity_sym,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"Correlation stress: {corr_now:.3f} > {exit_thresh}",
                )
            ]

        # ── Normal regime: correlation low → hold/enter equity ────────────────
        if corr_now <= entry_thresh and not has_eq:
            logger.info(
                "CorrelationRegime: ENTER on %s (corr=%.3f <= threshold=%.3f)",
                as_of_date,
                corr_now,
                entry_thresh,
            )
            return [
                TradeSignal(
                    symbol=equity_sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=risk_on_weight,
                    stop_loss=eq_price * 0.95,
                    reasoning=f"Correlation normal: {corr_now:.3f} <= {entry_thresh}",
                )
            ]

        return []


# ---------------------------------------------------------------------------
# Correlation Surprise Strategy
# ---------------------------------------------------------------------------


class CorrelationSurpriseStrategy(Strategy):
    """SPY-TLT delta-correlation regime strategy (M7).

    Defensive when the N-day rolling correlation between SPY and TLT *changes*
    by more than delta_threshold in delta_window days. Rapid correlation shifts
    signal regime change / elevated systemic risk.

    Parameters (via StrategyConfig.parameters):
      corr_window (int, default 10): Rolling window for correlation.
      delta_window (int, default 5): Days over which to measure delta.
      delta_threshold (float, default 0.3): Exit when delta > this.
      spy_weight_risk_on (float, default 0.95): Target SPY weight in normal regime.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        corr_window: int = int(params.get("corr_window", 10))
        delta_window: int = int(params.get("delta_window", 5))
        delta_threshold: float = float(params.get("delta_threshold", 0.3))
        risk_on_weight: float = float(params.get("spy_weight_risk_on", 0.95))

        lookback = corr_window + delta_window + 2
        spy_data = (
            indicators_df.filter(pl.col("symbol") == "SPY").sort("date").tail(lookback)
        )
        tlt_data = (
            indicators_df.filter(pl.col("symbol") == "TLT").sort("date").tail(lookback)
        )

        if len(spy_data) < lookback - 1 or len(tlt_data) < lookback - 1:
            return []

        spy_prices = spy_data["close"].to_list()
        tlt_prices = tlt_data["close"].to_list()
        min_len = min(len(spy_prices), len(tlt_prices))

        spy_rets = [spy_prices[i] / spy_prices[i - 1] - 1.0 for i in range(1, min_len)]
        tlt_rets = [tlt_prices[i] / tlt_prices[i - 1] - 1.0 for i in range(1, min_len)]

        if len(spy_rets) < corr_window + delta_window:
            return []

        def _corr(xs: list[float], ys: list[float]) -> float:
            n = len(xs)
            if n < 2:
                return 0.0
            mx = sum(xs) / n
            my = sum(ys) / n
            cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
            sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
            sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
            if sx == 0 or sy == 0:
                return 0.0
            return cov / (sx * sy)

        corr_now = _corr(spy_rets[-corr_window:], tlt_rets[-corr_window:])
        corr_past = _corr(
            spy_rets[-(corr_window + delta_window) : -delta_window],
            tlt_rets[-(corr_window + delta_window) : -delta_window],
        )
        corr_delta = corr_now - corr_past

        spy_price = prices.get("SPY")
        if spy_price is None or spy_price <= 0:
            return []

        has_spy = "SPY" in portfolio.positions

        if corr_delta > delta_threshold and has_spy:
            logger.info(
                "CorrelationSurprise: EXIT on %s (delta=%.3f > %.3f)",
                as_of_date,
                corr_delta,
                delta_threshold,
            )
            return [
                TradeSignal(
                    symbol="SPY",
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        f"Correlation surprise: delta={corr_delta:.3f}"
                        f" > {delta_threshold}"
                    ),
                )
            ]

        if corr_delta <= 0.0 and not has_spy:
            logger.info(
                "CorrelationSurprise: ENTER on %s (delta=%.3f <= 0)",
                as_of_date,
                corr_delta,
            )
            return [
                TradeSignal(
                    symbol="SPY",
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=risk_on_weight,
                    stop_loss=spy_price * 0.95,
                    reasoning=f"Correlation stable: delta={corr_delta:.3f} <= 0",
                )
            ]

        return []


# ---------------------------------------------------------------------------
# Calendar Event Strategy
# ---------------------------------------------------------------------------

# FOMC meeting dates 2021-2026 (day of meeting; entry = 1-3 days before)
_FOMC_DATES: frozenset[date] = frozenset(
    date(int(y), int(m), int(d))
    for y, m, d in [
        (2021, 1, 27),
        (2021, 3, 17),
        (2021, 4, 28),
        (2021, 6, 16),
        (2021, 7, 28),
        (2021, 9, 22),
        (2021, 11, 3),
        (2021, 12, 15),
        (2022, 1, 26),
        (2022, 3, 16),
        (2022, 5, 4),
        (2022, 6, 15),
        (2022, 7, 27),
        (2022, 9, 21),
        (2022, 11, 2),
        (2022, 12, 14),
        (2023, 2, 1),
        (2023, 3, 22),
        (2023, 5, 3),
        (2023, 6, 14),
        (2023, 7, 26),
        (2023, 9, 20),
        (2023, 11, 1),
        (2023, 12, 13),
        (2024, 1, 31),
        (2024, 3, 20),
        (2024, 5, 1),
        (2024, 6, 12),
        (2024, 7, 31),
        (2024, 9, 18),
        (2024, 11, 7),
        (2024, 12, 18),
        (2025, 1, 29),
        (2025, 3, 19),
        (2025, 5, 7),
        (2025, 6, 18),
        (2025, 7, 30),
        (2025, 9, 17),
        (2025, 11, 5),
        (2025, 12, 17),
        (2026, 1, 28),
        (2026, 3, 18),
        (2026, 4, 29),
    ]
)


def _is_pre_fomc(d: date, pre_days: int = 3) -> bool:
    """True if d falls within pre_days before an FOMC meeting."""
    from datetime import timedelta

    for offset in range(1, pre_days + 1):
        if d + timedelta(days=offset) in _FOMC_DATES:
            return True
    return False


def _is_month_end_window(d: date, end_days: int = 1, start_days: int = 1) -> bool:
    """True if d is within end_days of month end or start_days of month start."""
    import calendar

    last_day = calendar.monthrange(d.year, d.month)[1]
    if d.day >= last_day - end_days:
        return True
    return d.day <= start_days


class CalendarEventStrategy(Strategy):
    """Entry around predictable calendar events (month-end / pre-FOMC).

    Modes:
      - "month_end": Hold SPY during last N days of month + first N days of next.
      - "pre_fomc": Hold TLT in the 3 days before each FOMC meeting date.

    Parameters (via StrategyConfig.parameters):
      mode (str, default "month_end"): "month_end" or "pre_fomc".
      target_symbol (str): Asset to trade ("SPY" for month_end, "TLT" for pre_fomc).
      pre_days (int, default 3): Days before event to enter.
      target_weight (float, default 0.95): Position weight during event window.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        mode: str = str(params.get("mode", "month_end"))
        pre_days: int = int(params.get("pre_days", 3))
        tgt_weight: float = float(params.get("target_weight", 0.95))

        if mode == "pre_fomc":
            symbol = str(params.get("target_symbol", "TLT"))
            in_window = _is_pre_fomc(as_of_date, pre_days=pre_days)
        else:  # month_end
            symbol = str(params.get("target_symbol", "SPY"))
            end_days = int(params.get("end_days", 1))
            start_days = int(params.get("start_days", 1))
            in_window = _is_month_end_window(as_of_date, end_days, start_days)

        price = prices.get(symbol)
        if price is None or price <= 0:
            return []

        has_pos = symbol in portfolio.positions

        if in_window and not has_pos:
            logger.info("CalendarEvent[%s]: ENTER %s on %s", mode, symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price * 0.97,
                    reasoning=f"Calendar event window ({mode}): enter {symbol}",
                )
            ]

        if not in_window and has_pos:
            logger.info("CalendarEvent[%s]: EXIT %s on %s", mode, symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"Calendar event window closed ({mode}): exit {symbol}",
                )
            ]

        return []


# ---------------------------------------------------------------------------
# Pairs Ratio Mean-Reversion Strategy
# ---------------------------------------------------------------------------


class PairsRatioStrategy(Strategy):
    """Bollinger Band mean-reversion on the ratio of two assets (D7 / O-series).

    Computes ratio = price_a / price_b and fits Bollinger Bands. Trades mean
    reversion: when ratio is stretched above upper band, buy the underperformer
    (symbol_b); when below lower band, buy the outperformer (symbol_a).
    Exits when ratio returns within exit_z sigma of the mean.

    Parameters (via StrategyConfig.parameters):
      symbol_a (str): Numerator asset (default "ETH-USD").
      symbol_b (str): Denominator asset (default "BTC-USD").
      bb_window (int, default 20): Bollinger Band lookback (single-window mode).
      bb_std (float, default 2.0): Band width in standard deviations.
      exit_z (float, default 0.5): Exit when |z| < this threshold.
      target_weight (float, default 0.90): Position weight when in trade.
      consensus_windows (list[int], optional): When set, compute BB z-scores for
        each window and require a majority (>= ceil(N/2)) to agree on direction
        before entering or exiting. Overrides bb_window for multi-window mode.
        Example: [60, 90, 120] — at least 2 of 3 windows must agree.
    """

    @staticmethod
    def _bb_z(ratios: list[float], window: int) -> float | None:
        """Compute z-score of current ratio vs BB for the given window."""
        if len(ratios) < window:
            return None
        w_ratios = ratios[-window:]
        mean_r = sum(w_ratios) / window
        std_r = (sum((r - mean_r) ** 2 for r in w_ratios) / window) ** 0.5
        if std_r == 0:
            return None
        return (ratios[-1] - mean_r) / std_r

    def generate_signals(  # noqa: PLR0911
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        symbol_a: str = str(params.get("symbol_a", "ETH-USD"))
        symbol_b: str = str(params.get("symbol_b", "BTC-USD"))
        bb_window: int = int(params.get("bb_window", 20))
        bb_std: float = float(params.get("bb_std", 2.0))
        tgt_weight: float = float(params.get("target_weight", 0.90))
        exit_z: float = float(params.get("exit_z", 0.5))

        # Multi-window consensus mode: list of window sizes.
        raw_cw = params.get("consensus_windows", None)
        if raw_cw is not None:
            windows: list[int] = [int(w) for w in raw_cw]
        else:
            windows = [bb_window]
        max_window = max(windows)

        a_data = (
            indicators_df.filter(pl.col("symbol") == symbol_a)
            .sort("date")
            .tail(max_window + 2)
        )
        b_data = (
            indicators_df.filter(pl.col("symbol") == symbol_b)
            .sort("date")
            .tail(max_window + 2)
        )

        min_len = min(len(a_data), len(b_data))
        if min_len < max_window:
            return []

        a_prices = a_data["close"].to_list()[-min_len:]
        b_prices = b_data["close"].to_list()[-min_len:]

        ratios = [a_prices[i] / b_prices[i] for i in range(min_len) if b_prices[i] > 0]
        if len(ratios) < max_window:
            return []

        # Compute z-scores and tally votes across all windows.
        z_scores: list[float] = []
        votes_buy_b = 0  # ratio stretched up → buy symbol_b
        votes_buy_a = 0  # ratio stretched down → buy symbol_a
        votes_exit = 0  # ratio reverted → exit

        for w in windows:
            z = self._bb_z(ratios, w)
            if z is None:
                continue
            z_scores.append(z)
            if z > bb_std:
                votes_buy_b += 1
            elif z < -bb_std:
                votes_buy_a += 1
            if abs(z) < exit_z:
                votes_exit += 1

        if not z_scores:
            return []

        n_valid = len(z_scores)
        majority = (n_valid + 1) // 2  # ceil(N/2) — majority threshold

        has_a = symbol_a in portfolio.positions
        has_b = symbol_b in portfolio.positions
        price_a = prices.get(symbol_a, 0)
        price_b = prices.get(symbol_b, 0)
        z_now = z_scores[-1]

        # Ratio stretched up: symbol_a expensive → buy symbol_b
        if votes_buy_b >= majority and not has_b and not has_a:
            if price_b <= 0:
                return []
            logger.info(
                "PairsRatio: BUY %s on %s (z=%.2f, %d/%d windows agree)",
                symbol_b,
                as_of_date,
                z_now,
                votes_buy_b,
                n_valid,
            )
            return [
                TradeSignal(
                    symbol=symbol_b,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price_b * 0.90,
                    reasoning=(
                        f"Ratio stretched up: z={z_now:.2f}"
                        f" ({votes_buy_b}/{n_valid} windows), buy {symbol_b}"
                    ),
                )
            ]

        # Ratio stretched down: symbol_b expensive → buy symbol_a
        if votes_buy_a >= majority and not has_a and not has_b:
            if price_a <= 0:
                return []
            logger.info(
                "PairsRatio: BUY %s on %s (z=%.2f, %d/%d windows agree)",
                symbol_a,
                as_of_date,
                z_now,
                votes_buy_a,
                n_valid,
            )
            return [
                TradeSignal(
                    symbol=symbol_a,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price_a * 0.90,
                    reasoning=(
                        f"Ratio stretched down: z={z_now:.2f}"
                        f" ({votes_buy_a}/{n_valid} windows), buy {symbol_a}"
                    ),
                )
            ]

        # Mean-reversion exit: majority windows within exit_z of mean
        if votes_exit >= majority and (has_a or has_b):
            return [
                TradeSignal(
                    symbol=sym,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        f"Pairs ratio reverted: z={z_now:.2f}"
                        f" ({votes_exit}/{n_valid} windows)"
                    ),
                )
                for sym in [symbol_a, symbol_b]
                if sym in portfolio.positions
            ]

        return []


# ---------------------------------------------------------------------------
# Lead-Lag Strategy
# ---------------------------------------------------------------------------


class LeadLagStrategy(Strategy):
    """Directional signal from a leading asset's lagged return (H-series).

    Observes the N-day return of a *leader* asset (e.g. XLF, HYG, BTC-USD)
    and takes a position in a *follower* asset (e.g. SPY, EEM) when the
    lagged signal is strong enough.

    Parameters:
      leader_symbol (str): Leading asset ticker (e.g. "XLF").
      follower_symbol (str): Follower asset ticker (e.g. "SPY").
      lag_days (int, default 2): How many days ago the leader signal is read.
      signal_window (int, default 3): Return window on the leader.
      entry_threshold (float, default 0.01): Min leader return to go long follower.
      exit_threshold (float, default -0.005): Leader return below which to exit.
      target_weight (float, default 0.90): Weight for follower when in trade.
      inverse (bool-like, default False): If True, leader up → short follower.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        leader: str = str(params.get("leader_symbol", "XLF"))
        follower: str = str(params.get("follower_symbol", "SPY"))
        lag_days: int = int(params.get("lag_days", 2))
        sig_window: int = int(params.get("signal_window", 3))
        entry_thresh: float = float(params.get("entry_threshold", 0.01))
        exit_thresh: float = float(params.get("exit_threshold", -0.005))
        tgt_weight: float = float(params.get("target_weight", 0.90))
        inverse: bool = bool(params.get("inverse", False))

        lookback = sig_window + lag_days + 2
        leader_data = (
            indicators_df.filter(pl.col("symbol") == leader).sort("date").tail(lookback)
        )
        if len(leader_data) < sig_window + lag_days:
            return []

        prices_list = leader_data["close"].to_list()
        # Return computed lag_days ago (from [-(lag_days+sig_window)] to [-lag_days])
        end_idx = len(prices_list) - lag_days
        start_idx = end_idx - sig_window
        if start_idx < 0 or prices_list[start_idx] <= 0:
            return []
        leader_ret = prices_list[end_idx - 1] / prices_list[start_idx] - 1.0

        follower_price = prices.get(follower, 0)
        if follower_price <= 0:
            return []

        has_pos = follower in portfolio.positions
        signal_long = (
            (leader_ret >= entry_thresh)
            if not inverse
            else (leader_ret <= -entry_thresh)
        )
        signal_exit = (
            (leader_ret <= exit_thresh) if not inverse else (leader_ret >= -exit_thresh)
        )

        if signal_long and not has_pos:
            logger.info(
                "LeadLag: ENTER %s on %s (leader=%s ret=%.3f)",
                follower,
                as_of_date,
                leader,
                leader_ret,
            )
            return [
                TradeSignal(
                    symbol=follower,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=follower_price * 0.95,
                    reasoning=f"Lead-lag: {leader} {leader_ret:.3f}>={entry_thresh}",
                )
            ]
        if signal_exit and has_pos:
            logger.info(
                "LeadLag: EXIT %s on %s (leader=%s ret=%.3f)",
                follower,
                as_of_date,
                leader,
                leader_ret,
            )
            return [
                TradeSignal(
                    symbol=follower,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"Lead-lag: {leader} {leader_ret:.3f}<={exit_thresh}",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Asset Rotation Strategy
# ---------------------------------------------------------------------------


class AssetRotationStrategy(Strategy):
    """Rank assets by recent Sharpe or return; hold top-K (A7, O3, K-series).

    Parameters:
      symbols_list (str): Comma-separated list of symbols to rotate among.
      lookback_days (int, default 60): Return window for ranking.
      top_k (int, default 1): How many assets to hold simultaneously.
      rerank_days (int, default 20): Minimum days between rebalances.
      target_weight (float, default 0.90): Weight per held asset.
      rank_by (str, default "return"): "return" or "sharpe".
      absolute_momentum_threshold (float|None): If set, only include assets
        whose trailing return exceeds this value. Assets below threshold are
        excluded before top-K ranking. Enables dual momentum (Antonacci).
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        syms_str: str = str(params.get("symbols_list", "SPY,TLT,GLD"))
        symbols = [s.strip() for s in syms_str.split(",") if s.strip()]
        lookback: int = int(params.get("lookback_days", 60))
        top_k: int = int(params.get("top_k", 1))
        tgt_weight: float = float(params.get("target_weight", 0.90))
        rank_by: str = str(params.get("rank_by", "return"))
        abs_thresh_raw = params.get("absolute_momentum_threshold")
        abs_thresh: float | None = (
            float(abs_thresh_raw) if abs_thresh_raw is not None else None
        )

        # Compute scores
        scores: list[tuple[str, float]] = []
        returns: dict[str, float] = {}
        for sym in symbols:
            sym_data = (
                indicators_df.filter(pl.col("symbol") == sym)
                .sort("date")
                .tail(lookback + 2)
            )
            if len(sym_data) < lookback:
                continue
            p = sym_data["close"].to_list()[-lookback - 1 :]
            rets = [p[i] / p[i - 1] - 1.0 for i in range(1, len(p))]
            if not rets:
                continue
            total_ret = p[-1] / p[0] - 1.0 if p[0] > 0 else 0.0
            returns[sym] = total_ret
            if rank_by == "sharpe":
                mu = sum(rets) / len(rets)
                std = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
                score = (mu / std * (252**0.5)) if std > 0 else 0.0
            else:
                score = total_ret
            scores.append((sym, score))

        if not scores:
            return []

        # Absolute momentum filter: exclude assets with return below threshold
        if abs_thresh is not None:
            scores = [(s, sc) for s, sc in scores if returns.get(s, 0.0) > abs_thresh]
        if not scores:
            return []

        scores.sort(key=lambda x: x[1], reverse=True)
        target_set = {s for s, _ in scores[:top_k] if s in prices and prices[s] > 0}
        current_set = set(portfolio.positions.keys()) & set(symbols)

        # Exit positions no longer in top-K
        signals: list[TradeSignal] = [
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.MEDIUM,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Rotation: {sym} dropped from top-{top_k}",
            )
            for sym in current_set - target_set
        ]
        # Enter new top-K positions
        weight_per = tgt_weight / max(len(target_set), 1)
        for sym in target_set - current_set:
            p = prices.get(sym, 0)
            if p <= 0:
                continue
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=weight_per,
                    stop_loss=p * 0.93,
                    reasoning=f"Rotation: {sym} entered top-{top_k} by {rank_by}",
                )
            )
        return signals


# ---------------------------------------------------------------------------
# VIX Regime Strategy
# ---------------------------------------------------------------------------


class VixRegimeStrategy(Strategy):
    """Defensive equity positioning based on VIX level or volatility-of-vol (C-series).

    Modes:
      - "vov": VIX 30-day rolling std dev above percentile_threshold → exit.
      - "level": VIX level above vix_threshold → defensive.
      - "vix_spike": Single-day VIX % change above spike_threshold → contrarian entry.
      - "term_structure": Long equity when VIX term structure is in contango
        (VIX3M/VIX > contango_threshold). Harvests volatility risk premium.
        Cash when backwardated (VIX3M/VIX < contango_threshold = stressed regime).

    Parameters:
      mode (str, default "level"): "vov", "level", "vix_spike", or "term_structure".
      vix_symbol (str, default "VIX"): Short-term VIX ticker (internal symbol).
      vix3m_symbol (str, default "VIX3M"): 3-month VIX ticker for term_structure mode.
      equity_symbol (str, default "SPY"): Asset to trade.
      vix_threshold (float, default 25.0): VIX level for "level" mode.
      vov_window (int, default 30): Rolling window for VoV.
      vov_percentile (float, default 0.80): Percentile for VoV threshold.
      spike_threshold (float, default 0.20): 1-day VIX % rise for spike mode.
      spike_exit_vix (float, default 20.0): Exit vix_spike position when VIX
        drops below this level (fear subsided, bounce captured).
      contango_threshold (float, default 1.05): VIX3M/VIX ratio above which
        the term structure is considered contango (long equity). Default 1.05
        means 3-month vol must exceed spot vol by 5% to confirm contango.
      target_weight (float, default 0.90): Position weight in favourable regime.
    """

    def generate_signals(  # noqa: PLR0911
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        mode: str = str(params.get("mode", "level"))
        vix_sym: str = str(params.get("vix_symbol", "VIX"))
        equity_sym: str = str(params.get("equity_symbol", "SPY"))
        vix_thresh: float = float(params.get("vix_threshold", 25.0))
        vov_window: int = int(params.get("vov_window", 30))
        vov_pct: float = float(params.get("vov_percentile", 0.80))
        spike_thresh: float = float(params.get("spike_threshold", 0.20))
        spike_exit_vix: float = float(params.get("spike_exit_vix", 20.0))
        tgt_weight: float = float(params.get("target_weight", 0.90))

        vix_data = indicators_df.filter(pl.col("symbol") == vix_sym).sort("date")
        if len(vix_data) < 5:
            return []

        vix_prices = vix_data["close"].to_list()
        vix_now = vix_prices[-1]

        equity_price = prices.get(equity_sym, 0)
        if equity_price <= 0:
            return []
        has_pos = equity_sym in portfolio.positions

        if mode == "level":
            defensive = vix_now > vix_thresh
        elif mode == "vov":
            if len(vix_prices) < vov_window + 1:
                return []

            def _rolling_std(series: list[float], w: int) -> float:
                if len(series) < w:
                    return 0.0
                vals = series[-w:]
                mu = sum(vals) / w
                return (sum((v - mu) ** 2 for v in vals) / w) ** 0.5

            # Current VoV = std dev of last vov_window VIX prices
            vov_now = _rolling_std(vix_prices, vov_window)
            # Historical VoV series: one value per past day (expanding window)
            hist_vov = [
                _rolling_std(vix_prices[: i + 1], vov_window)
                for i in range(vov_window - 1, len(vix_prices))
            ]
            if not hist_vov:
                return []
            # Percentile rank of current VoV in its own history
            n_below = sum(1 for v in hist_vov if v <= vov_now)
            pct = n_below / len(hist_vov)
            # Defensive when VoV is elevated (top percentile of its own history)
            defensive = pct >= vov_pct
        elif mode == "vix_spike":
            if len(vix_prices) < 2 or vix_prices[-2] <= 0:
                return []
            vix_change = vix_prices[-1] / vix_prices[-2] - 1.0
            # Contrarian: spike → BUY equity (bounce play)
            if vix_change >= spike_thresh and not has_pos:
                return [
                    TradeSignal(
                        symbol=equity_sym,
                        action=Action.BUY,
                        conviction=Conviction.MEDIUM,
                        target_weight=tgt_weight,
                        stop_loss=equity_price * 0.95,
                        reasoning=f"VIX spike {vix_change:.1%}>={spike_thresh:.1%}",
                    )
                ]
            # Exit when VIX returns to normal (fear subsided, bounce captured)
            if has_pos and vix_now < spike_exit_vix:
                return [
                    TradeSignal(
                        symbol=equity_sym,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"VIX spike exit: VIX={vix_now:.1f}",
                    )
                ]
            return []
        elif mode == "term_structure":
            # VIX term structure: contango (VIX3M > VIX) → long equity
            # Backwardation (VIX3M <= VIX * threshold) → cash/defensive
            vix3m_sym: str = str(params.get("vix3m_symbol", "VIX3M"))
            contango_thresh: float = float(params.get("contango_threshold", 1.05))
            vix3m_data = indicators_df.filter(pl.col("symbol") == vix3m_sym).sort(
                "date"
            )
            if len(vix3m_data) < 5 or vix_now <= 0:
                return []
            vix3m_now = vix3m_data["close"].to_list()[-1]
            if vix3m_now <= 0:
                return []
            ratio = vix3m_now / vix_now
            # Contango: VIX3M > VIX * threshold → risk-on, long equity
            # Backwardation or flat: VIX3M <= VIX * threshold → risk-off, cash
            defensive = ratio < contango_thresh
            logger.debug(
                "VIX term structure: VIX3M=%.2f VIX=%.2f"
                " ratio=%.3f thresh=%.2f defensive=%s",
                vix3m_now,
                vix_now,
                ratio,
                contango_thresh,
                defensive,
            )
        else:
            return []

        if defensive and has_pos:
            return [
                TradeSignal(
                    symbol=equity_sym,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"VIX[{mode}]: defensive exit VIX={vix_now:.1f}",
                )
            ]
        if not defensive and not has_pos:
            return [
                TradeSignal(
                    symbol=equity_sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=equity_price * 0.95,
                    reasoning=f"VIX regime [{mode}]: normal, enter (VIX={vix_now:.1f})",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Yield Curve Regime Strategy
# ---------------------------------------------------------------------------


class YieldCurveRegimeStrategy(Strategy):
    """Equity/bond positioning based on yield curve slope (N/E-series).

    Uses freely available Yahoo Finance yield tickers (^TNX=10y, ^IRX=3m,
    ^FVX=5y) to compute yield spread as regime signal.

    Modes:
      - "inversion": Hold TLT when 2s10s inverted (^IRX > ^TNX); exit when normal.
      - "momentum": Hold TLT when 63d TNX momentum negative (rates falling).
      - "steepener": Hold SPY when curve steepening (spread rising).

    Parameters:
      mode (str, default "inversion"): Strategy mode.
      short_yield_symbol (str, default "^IRX"): Short-end yield ticker.
      long_yield_symbol (str, default "^TNX"): Long-end yield ticker.
      equity_symbol (str, default "SPY"): Equity asset.
      bond_symbol (str, default "TLT"): Bond asset.
      momentum_window (int, default 63): Days for yield momentum.
      target_weight (float, default 0.90): Position weight.
    """

    def generate_signals(  # noqa: PLR0911
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        mode: str = str(params.get("mode", "momentum"))
        short_sym: str = str(params.get("short_yield_symbol", "^IRX"))
        long_sym: str = str(params.get("long_yield_symbol", "^TNX"))
        equity_sym: str = str(params.get("equity_symbol", "SPY"))
        bond_sym: str = str(params.get("bond_symbol", "TLT"))
        mom_window: int = int(params.get("momentum_window", 63))
        tgt_weight: float = float(params.get("target_weight", 0.90))

        long_data = (
            indicators_df.filter(pl.col("symbol") == long_sym)
            .sort("date")
            .tail(mom_window + 5)
        )
        if len(long_data) < 5:
            return []
        tnx_prices = long_data["close"].to_list()
        tnx_now = tnx_prices[-1]

        if mode == "momentum":
            # Hold TLT when 10y yield trending down (rates falling = bonds rising)
            if len(tnx_prices) < mom_window:
                return []
            tnx_past = tnx_prices[-mom_window]
            rates_falling = tnx_now < tnx_past
            trade_sym = bond_sym
        elif mode == "inversion":
            short_data = (
                indicators_df.filter(pl.col("symbol") == short_sym).sort("date").tail(5)
            )
            if len(short_data) < 2:
                return []
            irx_now = short_data["close"].to_list()[-1]
            inverted = irx_now > tnx_now  # short > long = inverted curve
            rates_falling = inverted  # inverted curve → hold TLT (defensive)
            trade_sym = bond_sym
        elif mode == "steepener":
            short_data = (
                indicators_df.filter(pl.col("symbol") == short_sym)
                .sort("date")
                .tail(mom_window + 5)
            )
            if len(short_data) < mom_window:
                return []
            irx = short_data["close"].to_list()
            spread_now = tnx_now - irx[-1]
            spread_past = tnx_prices[-mom_window] - irx[-mom_window]
            rates_falling = spread_now > spread_past  # steepening → hold equities
            trade_sym = equity_sym
        else:
            return []

        asset_price = prices.get(trade_sym, 0)
        if asset_price <= 0:
            return []
        has_pos = trade_sym in portfolio.positions

        if rates_falling and not has_pos:
            return [
                TradeSignal(
                    symbol=trade_sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=asset_price * 0.95,
                    reasoning=f"YieldCurve [{mode}]: signal active (TNX={tnx_now:.2f})",
                )
            ]
        if not rates_falling and has_pos:
            return [
                TradeSignal(
                    symbol=trade_sym,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"YieldCurve [{mode}]: signal off (TNX={tnx_now:.2f})",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# OHLCV Momentum Strategy (L2 / L4 series)
# ---------------------------------------------------------------------------


class OHLCVMomentumStrategy(Strategy):
    """OHLCV-based momentum on high-volume conviction candles (L2/L4 series).

    Modes:
      - "conviction_candle" (L2): Enter when any of the last ``signal_lookback``
        days had intraday_return > conviction_pct AND volume > vol_multiplier *
        vol_sma_20. Maintains position as long as a recent signal exists; exits
        when no conviction candle in the lookback window.
      - "atr_breakout" (L4): Enter when close > high_20 + atr_mult * atr_14
        AND volume > vol_multiplier * vol_sma_20.

    Parameters (via StrategyConfig.parameters):
      symbol (str, default "SPY"): Asset to trade.
      mode (str, default "conviction_candle"): "conviction_candle" or "atr_breakout".
      conviction_pct (float, default 0.01): Min intraday return for candle signal.
      vol_multiplier (float, default 1.5): Volume must exceed this × vol_sma_20.
      signal_lookback (int, default 5): Days to look back for conviction candle.
      atr_mult (float, default 0.5): ATR multiplier for breakout threshold (L4 only).
      target_weight (float, default 0.90): Position weight when in trade.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        symbol: str = str(params.get("symbol", "SPY"))
        mode: str = str(params.get("mode", "conviction_candle"))
        conviction_pct: float = float(params.get("conviction_pct", 0.01))
        vol_multiplier: float = float(params.get("vol_multiplier", 1.5))
        signal_lookback: int = int(params.get("signal_lookback", 5))
        atr_mult: float = float(params.get("atr_mult", 0.5))
        tgt_weight: float = float(params.get("target_weight", 0.90))

        lookback = signal_lookback + 5
        ind = (
            indicators_df.filter(pl.col("symbol") == symbol).sort("date").tail(lookback)
        )
        price = prices.get(symbol, 0)
        if len(ind) < signal_lookback or price <= 0:
            return []

        has_pos = symbol in portfolio.positions
        in_signal = False

        if mode == "conviction_candle":
            # Any of the last signal_lookback days had a high-volume conviction candle
            if "intraday_return" not in ind.columns or "vol_sma_20" not in ind.columns:
                return []
            recent = ind.tail(signal_lookback)
            for row in recent.iter_rows(named=True):
                ir = row.get("intraday_return")
                vol = row.get("volume")
                vsma = row.get("vol_sma_20")
                if (
                    ir is not None
                    and vol is not None
                    and vsma is not None
                    and vsma > 0
                    and ir > conviction_pct
                    and vol > vol_multiplier * vsma
                ):
                    in_signal = True
                    break

        elif mode == "atr_breakout":
            # Current day: close > high_20 + atr_mult * atr_14 AND high volume
            if (
                "high_20" not in ind.columns
                or "atr_14" not in ind.columns
                or "vol_sma_20" not in ind.columns
            ):
                return []
            row = ind.tail(1).row(0, named=True)
            h20 = row.get("high_20")
            atr = row.get("atr_14")
            vol = row.get("volume")
            vsma = row.get("vol_sma_20")
            close = row.get("close")
            if all(v is not None for v in [h20, atr, vol, vsma, close]) and vsma > 0:
                in_signal = close > h20 + atr_mult * atr and vol > vol_multiplier * vsma

        if in_signal and not has_pos:
            logger.info("OHLCVMomentum[%s]: ENTER %s on %s", mode, symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price * 0.95,
                    reasoning=f"OHLCV {mode}: entry signal triggered",
                )
            ]
        if not in_signal and has_pos:
            logger.info("OHLCVMomentum[%s]: EXIT %s on %s", mode, symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"OHLCV {mode}: signal window expired",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Overnight Momentum Strategy (C7 series)
# ---------------------------------------------------------------------------


class OvernightMomentumStrategy(Strategy):
    """Overnight return decomposition momentum (C7 series).

    Computes the N-day rolling average of overnight returns
    (overnight_return = open_t / close_{t-1} - 1).  When the average overnight
    return exceeds ``entry_thresh``, institutional accumulation is assumed and
    the strategy enters long.  When it drops below ``exit_thresh``, it exits.

    Overnight returns capture institutional order flow executing at market open
    after after-hours research.  Sustained positive overnight gaps signal demand
    from large institutions rebalancing on a bi-weekly (10-day) cycle.

    Parameters (via StrategyConfig.parameters):
      symbol (str, default "SPY"): Asset to trade.
      window (int, default 10): Rolling window for overnight return average.
      entry_thresh (float, default 0.002): Enter when avg_overnight > this.
      exit_thresh (float, default -0.0005): Exit when avg_overnight < this.
      target_weight (float, default 0.90): Position weight when in trade.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        symbol: str = str(params.get("symbol", "SPY"))
        window: int = int(params.get("window", 10))
        entry_thresh: float = float(params.get("entry_thresh", 0.002))
        exit_thresh: float = float(params.get("exit_thresh", -0.0005))
        tgt_weight: float = float(params.get("target_weight", 0.90))

        ind = (
            indicators_df.filter(pl.col("symbol") == symbol)
            .sort("date")
            .tail(window + 3)
        )
        price = prices.get(symbol, 0)
        if len(ind) < window + 1 or price <= 0:
            return []
        if "open" not in ind.columns:
            return []

        closes = ind["close"].to_list()
        opens = ind["open"].to_list()
        n = len(closes)

        overnight_rets = [
            opens[i] / closes[i - 1] - 1 for i in range(1, n) if closes[i - 1] > 0
        ]
        if len(overnight_rets) < window:
            return []

        avg_overnight = sum(overnight_rets[-window:]) / window
        has_pos = symbol in portfolio.positions

        if avg_overnight > entry_thresh and not has_pos:
            logger.info(
                "OvernightMomentum: ENTER %s on %s (avg_overnight=%.4f)",
                symbol,
                as_of_date,
                avg_overnight,
            )
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price * 0.95,
                    reasoning=f"C7: avg_overnight={avg_overnight:.4f} > {entry_thresh}",
                )
            ]
        if avg_overnight < exit_thresh and has_pos:
            logger.info(
                "OvernightMomentum: EXIT %s on %s (avg_overnight=%.4f)",
                symbol,
                as_of_date,
                avg_overnight,
            )
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"C7: avg_overnight={avg_overnight:.4f} < {exit_thresh}",
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Macro Momentum Barometer (H6.4 — Family 6)
# ---------------------------------------------------------------------------


class MacroBarometerStrategy(Strategy):
    """4-barometer macro momentum rotation (AQR-inspired, Brooks 2017).

    Computes z-scores of blended 21d/63d returns for 4 barometer assets
    (SPY, TLT, DBC, EFA). Counts bullish barometers to determine regime:
    - 3+ bullish → risk-on basket
    - 0-1 bullish → risk-off basket
    - 2 bullish → blended (50/50)

    Parameters:
        barometers (list[str]): Barometer symbols, default ["SPY","TLT","DBC","EFA"].
        z_lookback_short (int): Short momentum lookback, default 21.
        z_lookback_long (int): Long momentum lookback, default 63.
        z_window (int): Z-score rolling window, default 63.
        risk_on_threshold (int): Min bullish count for risk-on, default 3.
        risk_off_threshold (int): Max bullish count for risk-off, default 1.
        risk_on_weights (dict): Asset weights for risk-on.
        risk_off_weights (dict): Asset weights for risk-off.
        blend_ratio (float): Risk-on fraction when blended, default 0.50.
    """

    def _compute_barometer_zscore(
        self, closes: list[float], short_lb: int, long_lb: int, z_win: int
    ) -> float | None:
        """Compute z-score of blended momentum for a barometer."""
        if len(closes) < max(long_lb, z_win) + long_lb:
            return None

        # Blended momentum: avg of short and long lookback returns
        def _ret(c: list[float], lb: int) -> float | None:
            if len(c) <= lb:
                return None
            return c[-1] / c[-(lb + 1)] - 1.0

        current_short = _ret(closes, short_lb)
        current_long = _ret(closes, long_lb)
        if current_short is None or current_long is None:
            return None
        current_blend = 0.5 * current_short + 0.5 * current_long

        # Build rolling history of blended returns for z-score
        blend_history: list[float] = []
        for end_idx in range(max(long_lb + 1, z_win), len(closes) + 1):
            sub = closes[:end_idx]
            s = _ret(sub, short_lb)
            lo = _ret(sub, long_lb)
            if s is not None and lo is not None:
                blend_history.append(0.5 * s + 0.5 * lo)

        if len(blend_history) < z_win:
            return None

        recent = blend_history[-z_win:]
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = var ** 0.5
        if std < 1e-10:
            return 0.0
        return (current_blend - mean) / std

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        barometers = params.get("barometers", ["SPY", "TLT", "DBC", "EFA"])
        short_lb = int(params.get("z_lookback_short", 21))
        long_lb = int(params.get("z_lookback_long", 63))
        z_win = int(params.get("z_window", 63))
        risk_on_thresh = int(params.get("risk_on_threshold", 3))
        risk_off_thresh = int(params.get("risk_off_threshold", 1))
        risk_on_wts = params.get(
            "risk_on_weights",
            {"SPY": 0.35, "EFA": 0.25, "DBC": 0.10, "GLD": 0.15, "HYG": 0.15},
        )
        risk_off_wts = params.get(
            "risk_off_weights",
            {"TLT": 0.30, "IEF": 0.25, "GLD": 0.25, "AGG": 0.20},
        )
        blend_ratio = float(params.get("blend_ratio", 0.50))

        # Compute barometer z-scores
        bull_count = 0
        barometer_details: list[str] = []
        for baro in barometers:
            baro_data = indicators_df.filter(pl.col("symbol") == baro).sort("date")
            if len(baro_data) < long_lb + z_win:
                continue
            baro_closes = baro_data["close"].to_list()
            z = self._compute_barometer_zscore(baro_closes, short_lb, long_lb, z_win)
            if z is not None:
                is_bull = z > 0
                if is_bull:
                    bull_count += 1
                barometer_details.append(f"{baro}:z={z:.2f}({'+'if is_bull else '-'})")

        if not barometer_details:
            return []

        # Determine regime
        if bull_count >= risk_on_thresh:
            regime = "risk_on"
            target_weights = dict(risk_on_wts)
        elif bull_count <= risk_off_thresh:
            regime = "risk_off"
            target_weights = dict(risk_off_wts)
        else:
            regime = "blended"
            target_weights = {}
            all_syms = set(risk_on_wts.keys()) | set(risk_off_wts.keys())
            for sym in all_syms:
                on_w = risk_on_wts.get(sym, 0.0) * blend_ratio
                off_w = risk_off_wts.get(sym, 0.0) * (1.0 - blend_ratio)
                target_weights[sym] = on_w + off_w

        signals: list[TradeSignal] = []
        regime_str = f"{regime} (bull={bull_count}/4: {', '.join(barometer_details)})"

        # Close positions not in target basket
        for sym in list(portfolio.positions.keys()):
            if sym not in target_weights or target_weights.get(sym, 0) < 0.01:
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"Macro barometer exit: {regime_str}",
                    )
                )

        # Open/adjust positions in target basket
        for sym, weight in target_weights.items():
            if weight < 0.01:
                continue
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            existing = portfolio.positions.get(sym)
            if existing is not None:
                # Check if rebalancing needed (>20% drift from target)
                current_weight = (existing.shares * price) / portfolio.nav
                if abs(current_weight - weight) < weight * 0.20:
                    continue  # Within tolerance

            stop_loss = price * (1.0 - self.config.stop_loss_pct)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=(
                        Conviction.HIGH if regime != "blended" else Conviction.MEDIUM
                    ),
                    target_weight=weight,
                    stop_loss=stop_loss,
                    reasoning=f"Macro barometer {sym}@{weight:.0%}: {regime_str}",
                )
            )

        return signals


# ---------------------------------------------------------------------------
# VRP Timing (H4.2 — Family 4)
# ---------------------------------------------------------------------------


class VRPTimingStrategy(Strategy):
    """Variance Risk Premium timing: long SPY when VRP is elevated.

    VRP = VIX - realized vol(SPY, 20d). When VRP smoothed crosses above
    entry threshold, go long SPY. Exit when VRP collapses below exit
    threshold, or time stop hits. Reduce position when VIX > vix_max.

    Parameters:
        vix_symbol (str): VIX ticker, default "^VIX".
        equity_symbol (str): Asset to trade, default "SPY".
        rv_window (int): Realized vol window in days, default 20.
        vrp_smoothing (int): SMA smoothing for VRP, default 5.
        vrp_entry_threshold (float): Enter when VRP crosses above this, default 3.0.
        vrp_exit_threshold (float): Exit when VRP drops below this, default 0.0.
        vix_min (float): Min VIX level for entry, default 16.
        vix_max (float): Reduce position above this VIX, default 35.
        reduced_weight (float): Position weight when VIX > vix_max, default 0.45.
        time_stop_days (int): Max hold period, default 20.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_date: date | None = None
        self._prev_vrp: float | None = None

    @staticmethod
    def _realized_vol(closes: list[float], window: int) -> float | None:
        """Compute annualized realized vol from daily closes."""
        if len(closes) < window + 1:
            return None
        returns = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(len(closes) - window, len(closes))
        ]
        n = len(returns)
        if n < 2:
            return None
        mean_ret = sum(returns) / n
        var = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        return (var ** 0.5) * (252 ** 0.5) * 100  # annualized, in VIX-like units

    @staticmethod
    def _sma(values: list[float], period: int) -> float | None:
        """Simple moving average of last `period` values."""
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        vix_sym = str(params.get("vix_symbol", "^VIX"))
        equity_sym = str(params.get("equity_symbol", "SPY"))
        rv_window = int(params.get("rv_window", 20))
        vrp_smooth = int(params.get("vrp_smoothing", 5))
        vrp_entry = float(params.get("vrp_entry_threshold", 3.0))
        vrp_exit = float(params.get("vrp_exit_threshold", 0.0))
        vix_min = float(params.get("vix_min", 16))
        vix_max = float(params.get("vix_max", 35))
        tgt_weight = float(params.get("target_weight", 0.90))
        reduced_weight = float(params.get("reduced_weight", 0.45))
        time_stop = int(params.get("time_stop_days", 20))

        # Get VIX data
        vix_data = indicators_df.filter(pl.col("symbol") == vix_sym).sort("date")
        if len(vix_data) < rv_window + vrp_smooth:
            return []
        vix_closes = vix_data["close"].to_list()
        vix_now = vix_closes[-1]

        # Get SPY data for realized vol
        spy_data = indicators_df.filter(pl.col("symbol") == equity_sym).sort("date")
        if len(spy_data) < rv_window + vrp_smooth + 1:
            return []
        spy_closes = spy_data["close"].to_list()
        equity_price = prices.get(equity_sym, 0)
        if equity_price <= 0:
            return []

        # Compute VRP series for smoothing
        vrp_series: list[float] = []
        for i in range(rv_window + 1, len(spy_closes) + 1):
            rv = self._realized_vol(spy_closes[:i], rv_window)
            # VIX index at corresponding date
            vix_idx = len(vix_closes) - (len(spy_closes) - i) - 1
            if rv is not None and 0 <= vix_idx < len(vix_closes):
                vrp_series.append(vix_closes[vix_idx] - rv)

        if len(vrp_series) < vrp_smooth:
            return []

        vrp_smoothed = self._sma(vrp_series, vrp_smooth)
        if vrp_smoothed is None:
            return []

        prev_vrp = self._prev_vrp
        self._prev_vrp = vrp_smoothed

        has_position = equity_sym in portfolio.positions

        # --- EXIT LOGIC ---
        if has_position:
            should_exit = False
            exit_reason = ""

            # 1. VRP collapsed below exit threshold
            if vrp_smoothed < vrp_exit:
                should_exit = True
                exit_reason = f"VRP collapsed ({vrp_smoothed:.1f} < {vrp_exit})"

            # 2. Time stop
            if not should_exit and self._entry_date is not None:
                days_held = (as_of_date - self._entry_date).days
                if days_held >= time_stop:
                    should_exit = True
                    exit_reason = f"Time stop ({days_held}d >= {time_stop}d)"

            if should_exit:
                self._entry_date = None
                return [
                    TradeSignal(
                        symbol=equity_sym,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"VRP exit: {exit_reason}, VRP={vrp_smoothed:.1f}",
                    )
                ]

            # 3. VIX > vix_max: reduce position (but don't exit)
            existing = portfolio.positions.get(equity_sym)
            if (
                vix_now > vix_max
                and existing is not None
                and existing.shares * equity_price
                > reduced_weight * portfolio.nav * 1.1
            ):
                return [
                    TradeSignal(
                        symbol=equity_sym,
                        action=Action.SELL,
                        conviction=Conviction.MEDIUM,
                        target_weight=reduced_weight,
                        stop_loss=equity_price * 0.92,
                        reasoning=(
                            f"VRP reduce: VIX={vix_now:.1f} > {vix_max}, "
                            f"reducing to {reduced_weight:.0%}"
                        ),
                    )
                ]
            return []

        # --- ENTRY LOGIC ---
        # VRP must cross above entry threshold from below
        crossed_up = (
            prev_vrp is not None
            and prev_vrp < vrp_entry
            and vrp_smoothed >= vrp_entry
        )
        if not crossed_up:
            return []

        # VIX must be above minimum (insurance premium meaningful)
        if vix_now < vix_min:
            return []

        # Determine weight based on VIX regime
        weight = reduced_weight if vix_now > vix_max else tgt_weight
        stop_loss = equity_price * (1.0 - self.config.stop_loss_pct)
        self._entry_date = as_of_date

        return [
            TradeSignal(
                symbol=equity_sym,
                action=Action.BUY,
                conviction=Conviction.HIGH,
                target_weight=weight,
                stop_loss=stop_loss,
                reasoning=(
                    f"VRP entry: VRP={vrp_smoothed:.1f} crossed above {vrp_entry}, "
                    f"VIX={vix_now:.1f}, RV20={vrp_series[-1] + vix_now - vrp_smoothed:.1f}"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Vol-Scaled TSMOM (H3.1 — Family 3)
# ---------------------------------------------------------------------------


class VolScaledTsmomStrategy(Strategy):
    """Multi-asset time-series momentum with vol-scaling (Moskowitz et al. 2012).

    Each asset is evaluated independently: blended 21/63/252-day TSMOM signal
    with vol-scaling (Barroso & Santa-Clara 2015). Goes long assets with
    positive blended signal, short assets with negative signal, flat otherwise.

    Delegates signal computation to TsmomCalculator from signals/tsmom.py.

    Parameters:
        lookbacks (list[int]): Lookback periods, default [21, 63, 252].
        blend_weights (list[float]): Weights per lookback, default [1/3, 1/3, 1/3].
        vol_target (float): Portfolio vol target, default 0.10.
        vol_window (int): Realized vol window, default 126.
        max_vol_scalar (float): Max leverage from vol-scaling, default 2.0.
        flat_threshold (float): Signal threshold for entry, default 0.2.
        allow_short (bool): Whether to take short positions, default True.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        params = config.parameters or {}
        lookbacks = params.get("lookbacks", [21, 63, 252])
        weights = params.get("blend_weights", [1 / 3, 1 / 3, 1 / 3])
        # Normalize weights if they don't sum to 1
        w_sum = sum(weights)
        if abs(w_sum - 1.0) > 1e-6 and w_sum > 0:
            weights = [w / w_sum for w in weights]
        self._calc = TsmomCalculator(
            TsmomConfig(
                lookbacks=lookbacks,
                blend_weights=weights,
                vol_target=float(params.get("vol_target", 0.10)),
                vol_window=int(params.get("vol_window", 126)),
                max_vol_scalar=float(params.get("max_vol_scalar", 2.0)),
                flat_threshold=float(params.get("flat_threshold", 0.2)),
            )
        )
        self._allow_short = bool(params.get("allow_short", True))

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
            if len(sym_data) < 63:  # minimum for shortest useful lookback
                continue

            close = prices.get(symbol, 0)
            if close <= 0:
                continue

            price_series = sym_data["close"]
            tsmom_sig = self._calc.compute(price_series, symbol=symbol)
            has_position = symbol in portfolio.positions
            current_pos = portfolio.positions.get(symbol)
            is_long = current_pos is not None and current_pos.shares > 0
            is_short = current_pos is not None and current_pos.shares < 0

            if tsmom_sig.direction == "long":
                if is_short:
                    # Close short, then go long
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=(
                                f"TSMOM reversal to long: "
                                f"signal={tsmom_sig.scaled_signal:.3f}, "
                                f"agreement={tsmom_sig.signal_agreement:.0%}"
                            ),
                        )
                    )
                if not is_long:
                    # Scale weight by signal strength and vol scalar
                    base_weight = self.config.target_position_weight
                    weight = base_weight * min(
                        abs(tsmom_sig.scaled_signal), 1.0
                    )
                    weight = max(weight, base_weight * 0.5)  # floor at 50%

                    conviction = (
                        Conviction.HIGH
                        if tsmom_sig.signal_agreement >= 1.0
                        else Conviction.MEDIUM
                    )
                    stop_loss = close * (1.0 - self.config.stop_loss_pct)
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=Action.BUY,
                            conviction=conviction,
                            target_weight=weight,
                            stop_loss=stop_loss,
                            reasoning=(
                                f"TSMOM long: signal={tsmom_sig.scaled_signal:.3f}, "
                                f"vol_scalar={tsmom_sig.vol_scalar:.2f}, "
                                f"agreement={tsmom_sig.signal_agreement:.0%}, "
                                f"lookbacks={tsmom_sig.lookback_signals}"
                            ),
                        )
                    )

            elif tsmom_sig.direction == "short" and self._allow_short:
                if is_long:
                    # Close long, then go short
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=(
                                f"TSMOM reversal to short: "
                                f"signal={tsmom_sig.scaled_signal:.3f}, "
                                f"agreement={tsmom_sig.signal_agreement:.0%}"
                            ),
                        )
                    )
                if not is_short:
                    base_weight = self.config.target_position_weight
                    weight = base_weight * min(
                        abs(tsmom_sig.scaled_signal), 1.0
                    )
                    weight = max(weight, base_weight * 0.5)

                    conviction = (
                        Conviction.HIGH
                        if tsmom_sig.signal_agreement >= 1.0
                        else Conviction.MEDIUM
                    )
                    stop_loss = close * (1.0 + self.config.stop_loss_pct)
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=Action.SELL,
                            conviction=conviction,
                            target_weight=-weight,
                            stop_loss=stop_loss,
                            reasoning=(
                                f"TSMOM short: signal={tsmom_sig.scaled_signal:.3f}, "
                                f"vol_scalar={tsmom_sig.vol_scalar:.2f}, "
                                f"agreement={tsmom_sig.signal_agreement:.0%}, "
                                f"lookbacks={tsmom_sig.lookback_signals}"
                            ),
                        )
                    )

            elif tsmom_sig.direction == "flat" and has_position:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            f"TSMOM flat: signal={tsmom_sig.scaled_signal:.3f}, "
                            f"closing position"
                        ),
                    )
                )

        return signals


# ---------------------------------------------------------------------------
# RSI(2) Multi-Asset Contrarian (H7.2 — Family 7)
# ---------------------------------------------------------------------------


class RSI2ContrarianStrategy(Strategy):
    """RSI(2) extreme oversold contrarian with golden cross trend filter.

    Buys when RSI(2) < oversold_threshold AND 50-EMA > 200-EMA (structural
    uptrend confirmed). Exits on RSI recovery, consecutive closes above SMA,
    time stop, or hard stop-loss.

    This strategy computes RSI(2) and EMAs internally from close prices
    rather than relying on the standard RSI(14) indicator column.

    Parameters:
        rsi_period (int): RSI lookback, default 2.
        oversold_threshold (float): Entry when RSI below this, default 10.
        overbought_threshold (float): Exit when RSI above this, default 70.
        ema_short (int): Short EMA for golden cross, default 50.
        ema_long (int): Long EMA for golden cross, default 200.
        time_stop_days (int): Max days to hold, default 10.
        consecutive_up_exit (int): Exit after N closes above SMA, default 2.
        sma_exit_period (int): SMA period for exit rule, default 5.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        # Track entry dates for time stop
        self._entry_dates: dict[str, date] = {}
        # Track consecutive closes above SMA for exit rule
        self._consec_above_sma: dict[str, int] = {}

    @staticmethod
    def _compute_rsi(closes: list[float], period: int) -> float | None:
        """Compute RSI from a list of closing prices."""
        if len(closes) < period + 1:
            return None
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = changes[-(period):]
        gains = [c for c in recent if c > 0]
        losses = [-c for c in recent if c < 0]
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_ema(closes: list[float], period: int) -> float | None:
        """Compute EMA from a list of closing prices."""
        if len(closes) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = sum(closes[:period]) / period  # SMA seed
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _compute_sma(closes: list[float], period: int) -> float | None:
        """Compute SMA from a list of closing prices."""
        if len(closes) < period:
            return None
        return sum(closes[-period:]) / period

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        params = self.config.parameters or {}
        rsi_period = int(params.get("rsi_period", 2))
        oversold = float(params.get("oversold_threshold", 10))
        overbought = float(params.get("overbought_threshold", 70))
        ema_short_period = int(params.get("ema_short", 50))
        ema_long_period = int(params.get("ema_long", 200))
        time_stop = int(params.get("time_stop_days", 10))
        consec_exit = int(params.get("consecutive_up_exit", 2))
        sma_exit_period = int(params.get("sma_exit_period", 5))

        symbols = indicators_df.select("symbol").unique().to_series().to_list()

        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < ema_long_period + 1:
                continue

            closes = sym_data["close"].to_list()
            close = closes[-1]
            if close <= 0:
                continue

            has_position = symbol in portfolio.positions

            # --- EXIT LOGIC (check before entry) ---
            if has_position:
                should_exit = False
                exit_reason = ""

                # 1. RSI overbought exit
                rsi = self._compute_rsi(closes, rsi_period)
                if rsi is not None and rsi > overbought:
                    should_exit = True
                    exit_reason = f"RSI({rsi_period}) overbought ({rsi:.1f} > {overbought})"

                # 2. Time stop
                if not should_exit and symbol in self._entry_dates:
                    days_held = (as_of_date - self._entry_dates[symbol]).days
                    if days_held >= time_stop:
                        should_exit = True
                        exit_reason = f"Time stop ({days_held}d >= {time_stop}d)"

                # 3. Consecutive closes above SMA exit
                if not should_exit:
                    sma = self._compute_sma(closes, sma_exit_period)
                    if sma is not None:
                        if close > sma:
                            self._consec_above_sma[symbol] = (
                                self._consec_above_sma.get(symbol, 0) + 1
                            )
                        else:
                            self._consec_above_sma[symbol] = 0
                        if self._consec_above_sma.get(symbol, 0) >= consec_exit:
                            should_exit = True
                            exit_reason = (
                                f"{consec_exit} consecutive closes above "
                                f"SMA({sma_exit_period})"
                            )

                # 4. Stop-loss is handled by the backtest engine via stop_loss field

                if should_exit:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=f"RSI2 contrarian exit: {exit_reason}",
                        )
                    )
                    self._entry_dates.pop(symbol, None)
                    self._consec_above_sma.pop(symbol, None)
                continue  # Don't enter and exit same symbol same day

            # --- ENTRY LOGIC ---
            if len(portfolio.positions) + len(
                [s for s in signals if s.action == Action.BUY]
            ) >= self.config.max_positions:
                continue

            # Compute RSI(2)
            rsi = self._compute_rsi(closes, rsi_period)
            if rsi is None or rsi >= oversold:
                continue

            # Golden cross filter: 50-EMA > 200-EMA
            ema_short = self._compute_ema(closes, ema_short_period)
            ema_long = self._compute_ema(closes, ema_long_period)
            if ema_short is None or ema_long is None or ema_short <= ema_long:
                continue

            # Entry: RSI(2) < oversold AND golden cross confirmed
            stop_loss = close * (1.0 - self.config.stop_loss_pct)
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=self.config.target_position_weight,
                    stop_loss=stop_loss,
                    reasoning=(
                        f"RSI2 contrarian: RSI({rsi_period})={rsi:.1f} < {oversold}, "
                        f"golden cross (EMA{ema_short_period}={ema_short:.2f} > "
                        f"EMA{ema_long_period}={ema_long:.2f})"
                    ),
                )
            )
            self._entry_dates[symbol] = as_of_date
            self._consec_above_sma[symbol] = 0

        return signals


# ---------------------------------------------------------------------------
# OPEX Week Gamma Strategy (Family 5 — Calendar/Structural)
# ---------------------------------------------------------------------------


def _opex_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month (standard OPEX)."""
    import calendar

    # Find first Friday
    cal = calendar.monthcalendar(year, month)
    # monthcalendar: each week is [Mon..Sun], Friday is index 4
    fridays = [week[4] for week in cal if week[4] != 0]
    return date(year, month, fridays[2])  # 3rd Friday (0-indexed: [2])


class OpexWeekStrategy(Strategy):
    """Buy SPY the Friday before OPEX week, sell Thursday of OPEX week.

    Parameters:
      target_symbol (str, default "SPY"): Asset to trade.
      vix_symbol (str, default "^VIX"): VIX ticker for regime filter.
      vix_threshold (float, default 25): Max VIX level at entry.
      target_weight (float, default 0.95): Position weight.
      exclude_quarterly (bool, default True): Skip Mar/Jun/Sep/Dec OPEX.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        symbol: str = str(params.get("target_symbol", "SPY"))
        vix_sym: str = str(params.get("vix_symbol", "^VIX"))
        vix_thresh: float = float(params.get("vix_threshold", 25))
        tgt_weight: float = float(params.get("target_weight", 0.95))
        exclude_q: bool = bool(params.get("exclude_quarterly", True))

        price = prices.get(symbol)
        if price is None or price <= 0:
            return []

        has_pos = symbol in portfolio.positions

        # Find this month's OPEX Friday
        opex_fri = _opex_friday(as_of_date.year, as_of_date.month)
        # Quarterly OPEX months
        quarterly = {3, 6, 9, 12}

        # Entry window: the Friday before OPEX week (T-7 calendar days before OPEX Fri)
        # through Wednesday of OPEX week (T-2 before OPEX Fri).
        # In trading days: enter on the Friday that is ~1 week before OPEX Friday.
        from datetime import timedelta

        entry_date = opex_fri - timedelta(days=7)  # Friday before OPEX week
        exit_date = opex_fri - timedelta(days=1)  # Thursday of OPEX week

        # Are we in the OPEX window? (entry_date <= as_of_date <= exit_date)
        in_window = entry_date <= as_of_date <= exit_date

        # Skip quarterly OPEX
        if exclude_q and as_of_date.month in quarterly:
            in_window = False

        # VIX filter for entry
        if in_window and not has_pos:
            # Check VIX level
            vix_data = (
                indicators_df.filter(pl.col("symbol") == vix_sym)
                .sort("date")
                .tail(1)
            )
            if len(vix_data) > 0:
                vix_level = vix_data.row(0, named=True)["close"]
                if vix_level >= vix_thresh:
                    logger.info(
                        "OpexWeek: VIX=%.1f >= %.1f, skipping entry on %s",
                        vix_level,
                        vix_thresh,
                        as_of_date,
                    )
                    return []

            logger.info("OpexWeek: ENTER %s on %s (OPEX %s)", symbol, as_of_date, opex_fri)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price * 0.97,
                    reasoning=f"OPEX week gamma: enter {symbol} (OPEX {opex_fri})",
                )
            ]

        if not in_window and has_pos:
            logger.info("OpexWeek: EXIT %s on %s", symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"OPEX week over: exit {symbol}",
                )
            ]

        return []


# ---------------------------------------------------------------------------
# Weekend Drift Strategy (Family 5 — Calendar/Structural)
# ---------------------------------------------------------------------------


class WeekendDriftStrategy(Strategy):
    """Buy BTC-USD on Friday, sell on Monday. Filters by momentum and vol.

    Parameters:
      symbol (str, default "BTC-USD"): Asset to trade.
      entry_day (int, default 4): Day of week to enter (4=Friday).
      exit_day (int, default 0): Day of week to exit (0=Monday).
      momentum_lookback (int, default 5): Lookback for momentum filter.
      momentum_floor (float, default -0.05): Min momentum to enter.
      vol_lookback (int, default 20): Lookback for realized vol.
      vol_ceiling (float, default 0.80): Max annualized vol to enter.
      target_weight (float, default 0.95): Position weight.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        symbol: str = str(params.get("symbol", "BTC-USD"))
        entry_day: int = int(params.get("entry_day", 4))  # Friday
        exit_day: int = int(params.get("exit_day", 0))  # Monday
        mom_lookback: int = int(params.get("momentum_lookback", 5))
        mom_floor: float = float(params.get("momentum_floor", -0.05))
        vol_lookback: int = int(params.get("vol_lookback", 20))
        vol_ceiling: float = float(params.get("vol_ceiling", 0.80))
        tgt_weight: float = float(params.get("target_weight", 0.95))

        price = prices.get(symbol)
        if price is None or price <= 0:
            return []

        dow = as_of_date.weekday()  # 0=Mon, 4=Fri
        has_pos = symbol in portfolio.positions

        # Exit on Monday
        if dow == exit_day and has_pos:
            logger.info("WeekendDrift: EXIT %s on %s (Monday)", symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"Weekend drift exit: Monday close {symbol}",
                )
            ]

        # Entry on Friday
        if dow == entry_day and not has_pos:
            sym_data = (
                indicators_df.filter(pl.col("symbol") == symbol)
                .sort("date")
            )
            if len(sym_data) < vol_lookback:
                return []

            closes = sym_data.tail(vol_lookback + 1)["close"].to_list()

            # 5d momentum filter
            if len(closes) >= mom_lookback + 1:
                mom_ret = closes[-1] / closes[-(mom_lookback + 1)] - 1.0
                if mom_ret < mom_floor:
                    logger.info(
                        "WeekendDrift: momentum=%.3f < %.3f, skip on %s",
                        mom_ret, mom_floor, as_of_date,
                    )
                    return []

            # 20d realized vol filter (annualized)
            if len(closes) >= vol_lookback + 1:
                daily_rets = [
                    closes[j] / closes[j - 1] - 1.0
                    for j in range(1, len(closes))
                ]
                if daily_rets:
                    mean_r = sum(daily_rets) / len(daily_rets)
                    var_r = sum((r - mean_r) ** 2 for r in daily_rets) / len(daily_rets)
                    ann_vol = (var_r ** 0.5) * (365 ** 0.5)  # BTC trades 365 days
                    if ann_vol > vol_ceiling:
                        logger.info(
                            "WeekendDrift: vol=%.3f > %.3f, skip on %s",
                            ann_vol, vol_ceiling, as_of_date,
                        )
                        return []

            logger.info("WeekendDrift: ENTER %s on %s (Friday)", symbol, as_of_date)
            return [
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price * 0.90,
                    reasoning=f"Weekend drift entry: Friday buy {symbol}",
                )
            ]

        return []


# ---------------------------------------------------------------------------
# Crypto Weekend Equity Signal Strategy (Family 5 — Calendar/Structural)
# ---------------------------------------------------------------------------


class CryptoWeekendEquityStrategy(Strategy):
    """Use BTC weekend returns to predict Monday equity direction.

    Buy QQQ on Monday when BTC Fri→last-available return >= threshold.
    Exit Tuesday close. Stay flat when BTC weekend return is deeply negative.

    Parameters:
      signal_symbol (str, default "BTC-USD"): Crypto signal source.
      trade_symbol (str, default "QQQ"): Equity to trade.
      weekend_threshold (float, default -0.02): BTC return below this => stay flat.
      target_weight (float, default 0.95): Position weight.
      hold_days (int, default 2): Hold Mon+Tue (exit signal on Tue).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_date: date | None = None

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        sig_sym: str = str(params.get("signal_symbol", "BTC-USD"))
        trade_sym: str = str(params.get("trade_symbol", "QQQ"))
        threshold: float = float(params.get("weekend_threshold", -0.02))
        tgt_weight: float = float(params.get("target_weight", 0.95))
        hold_days: int = int(params.get("hold_days", 2))

        price = prices.get(trade_sym)
        if price is None or price <= 0:
            return []

        dow = as_of_date.weekday()  # 0=Mon, 4=Fri
        has_pos = trade_sym in portfolio.positions

        # Exit on Tuesday (or after hold_days from entry)
        if has_pos:
            if self._entry_date is not None:
                days_held = (as_of_date - self._entry_date).days
                if days_held >= hold_days or dow == 1:  # Tuesday or hold_days elapsed
                    # Only exit on Tuesday (weekday 1) to avoid exiting on weekend
                    if dow == 1:
                        logger.info(
                            "CryptoWeekendEquity: EXIT %s on %s (Tuesday)",
                            trade_sym, as_of_date,
                        )
                        self._entry_date = None
                        return [
                            TradeSignal(
                                symbol=trade_sym,
                                action=Action.CLOSE,
                                conviction=Conviction.HIGH,
                                target_weight=0.0,
                                stop_loss=0.0,
                                reasoning=f"Crypto weekend equity exit: Tuesday close {trade_sym}",
                            )
                        ]
            return []

        # Entry on Monday
        if dow != 0:
            return []

        # Compute BTC weekend return: Friday close to most recent close before Monday
        btc_data = (
            indicators_df.filter(pl.col("symbol") == sig_sym)
            .sort("date")
        )
        if len(btc_data) < 5:
            return []

        # Find Friday's close (last trading day before weekend)
        # BTC trades every day, so look for the most recent Friday (or closest)
        recent_btc = btc_data.tail(10)
        btc_rows = recent_btc.iter_rows(named=True)
        btc_by_date = {row["date"]: row["close"] for row in btc_rows}

        # Find the most recent Friday close
        from datetime import timedelta

        fri_close = None
        for offset in range(1, 5):
            check_date = as_of_date - timedelta(days=offset)
            if check_date.weekday() == 4:  # Friday
                fri_close = btc_by_date.get(check_date)
                break

        if fri_close is None or fri_close <= 0:
            return []

        # Most recent BTC close before Monday (could be Sunday or Saturday)
        latest_btc_close = None
        for offset in range(1, 4):
            check_date = as_of_date - timedelta(days=offset)
            if check_date in btc_by_date:
                latest_btc_close = btc_by_date[check_date]
                break

        if latest_btc_close is None or latest_btc_close <= 0:
            return []

        weekend_ret = latest_btc_close / fri_close - 1.0

        if weekend_ret < threshold:
            logger.info(
                "CryptoWeekendEquity: BTC weekend ret=%.3f < %.3f, skip on %s",
                weekend_ret, threshold, as_of_date,
            )
            return []

        logger.info(
            "CryptoWeekendEquity: ENTER %s on %s (BTC weekend=%.3f)",
            trade_sym, as_of_date, weekend_ret,
        )
        self._entry_date = as_of_date
        return [
            TradeSignal(
                symbol=trade_sym,
                action=Action.BUY,
                conviction=Conviction.MEDIUM,
                target_weight=tgt_weight,
                stop_loss=price * 0.97,
                reasoning=f"Crypto weekend equity: BTC weekend ret={weekend_ret:.3f} >= {threshold}",
            )
        ]


# ---------------------------------------------------------------------------
# VIX Percentile Spike Mean-Reversion (H7.1 — Family 7)
# ---------------------------------------------------------------------------


class VIXPercentileSpikeStrategy(Strategy):
    """Mean-reversion on SPY after VIX panic exhaustion.

    Entry requires ALL 4 conditions simultaneously:
      1) VIX at 90th+ percentile of trailing 252-day closes
      2) VIX single-day spike > 15%
      3) VIX < 1.5 * VIX 50-day SMA (regime filter: excludes sustained crisis)
      4) SPY RSI(5) < 30

    Exit: SPY closes above 5-day SMA, time stop, or stop-loss.

    Parameters:
        vix_symbol (str): VIX ticker, default "VIX".
        equity_symbol (str): Asset to trade, default "SPY".
        vix_percentile (int): Percentile threshold for VIX, default 90.
        vix_spike_pct (float): Min single-day VIX % change, default 0.15.
        vix_regime_mult (float): Max VIX / VIX_SMA ratio, default 1.5.
        vix_sma_period (int): VIX SMA period for regime filter, default 50.
        rsi_period (int): RSI period for SPY, default 5.
        rsi_threshold (float): RSI oversold threshold, default 30.
        sma_exit_period (int): SMA period for exit signal, default 5.
        time_stop (int): Max holding days, default 10.
        stop_loss (float): Stop-loss pct, default 0.03.
        target_weight (float): Position weight, default 0.90.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_date: date | None = None
        self._entry_price: float | None = None

    @staticmethod
    def _rsi(closes: list[float], period: int) -> float | None:
        """Compute RSI from a list of close prices."""
        if len(closes) < period + 1:
            return None
        gains: list[float] = []
        losses: list[float] = []
        for i in range(len(closes) - period, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        vix_sym = str(params.get("vix_symbol", "VIX"))
        equity_sym = str(params.get("equity_symbol", "SPY"))
        pct_thresh = int(params.get("vix_percentile", 90))
        spike_pct = float(params.get("vix_spike_pct", 0.15))
        regime_mult = float(params.get("vix_regime_mult", 1.5))
        sma_period = int(params.get("vix_sma_period", 50))
        rsi_period = int(params.get("rsi_period", 5))
        rsi_thresh = float(params.get("rsi_threshold", 30))
        sma_exit = int(params.get("sma_exit_period", 5))
        time_stop = int(params.get("time_stop", 10))
        stop_loss_pct = float(params.get("stop_loss", 0.03))
        tgt_weight = float(params.get("target_weight", 0.90))

        # Load VIX data
        vix_data = indicators_df.filter(pl.col("symbol") == vix_sym).sort("date")
        if len(vix_data) < 252:
            return []
        vix_closes = vix_data["close"].to_list()

        # Load SPY data
        spy_data = indicators_df.filter(pl.col("symbol") == equity_sym).sort("date")
        if len(spy_data) < max(rsi_period + 1, sma_exit + 1):
            return []
        spy_closes = spy_data["close"].to_list()

        equity_price = prices.get(equity_sym, 0)
        if equity_price <= 0:
            return []
        has_pos = equity_sym in portfolio.positions

        # --- EXIT LOGIC ---
        if has_pos:
            # Exit 1: SPY closes above 5d SMA
            if len(spy_closes) >= sma_exit:
                sma_val = sum(spy_closes[-sma_exit:]) / sma_exit
                if equity_price > sma_val:
                    self._entry_date = None
                    self._entry_price = None
                    return [
                        TradeSignal(
                            symbol=equity_sym,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=f"VIX spike exit: SPY {equity_price:.2f} > SMA{sma_exit} {sma_val:.2f}",
                        )
                    ]

            # Exit 2: Time stop
            if self._entry_date is not None:
                days_held = (as_of_date - self._entry_date).days
                if days_held >= time_stop:
                    self._entry_date = None
                    self._entry_price = None
                    return [
                        TradeSignal(
                            symbol=equity_sym,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=f"VIX spike exit: time stop ({days_held}d >= {time_stop}d)",
                        )
                    ]

            # Exit 3: Stop-loss
            if self._entry_price is not None:
                loss = (equity_price - self._entry_price) / self._entry_price
                if loss <= -stop_loss_pct:
                    self._entry_date = None
                    self._entry_price = None
                    return [
                        TradeSignal(
                            symbol=equity_sym,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=f"VIX spike exit: stop-loss ({loss:.1%} <= -{stop_loss_pct:.0%})",
                        )
                    ]
            return []

        # --- ENTRY LOGIC ---
        # Condition 1: VIX at pct_thresh percentile of trailing 252d
        vix_now = vix_closes[-1]
        window_252 = vix_closes[-252:]
        n_below = sum(1 for v in window_252 if v <= vix_now)
        vix_pct_rank = (n_below / len(window_252)) * 100
        if vix_pct_rank < pct_thresh:
            return []

        # Condition 2: VIX single-day spike > spike_pct
        if len(vix_closes) < 2 or vix_closes[-2] <= 0:
            return []
        vix_change = vix_closes[-1] / vix_closes[-2] - 1.0
        if vix_change < spike_pct:
            return []

        # Condition 3: VIX < regime_mult * VIX SMA
        if len(vix_closes) < sma_period:
            return []
        vix_sma = sum(vix_closes[-sma_period:]) / sma_period
        if vix_now >= regime_mult * vix_sma:
            return []

        # Condition 4: SPY RSI(rsi_period) < rsi_thresh
        rsi = self._rsi(spy_closes, rsi_period)
        if rsi is None or rsi >= rsi_thresh:
            return []

        # All 4 conditions met — ENTER
        self._entry_date = as_of_date
        self._entry_price = equity_price
        stop_loss_price = equity_price * (1.0 - stop_loss_pct)

        logger.info(
            "VIXPercentileSpike: ENTER %s on %s (VIX=%.1f, pct=%.0f, spike=%.3f, RSI=%.1f)",
            equity_sym,
            as_of_date,
            vix_now,
            vix_pct_rank,
            vix_change,
            rsi,
        )

        return [
            TradeSignal(
                symbol=equity_sym,
                action=Action.BUY,
                conviction=Conviction.HIGH,
                target_weight=tgt_weight,
                stop_loss=stop_loss_price,
                reasoning=(
                    f"VIX percentile spike: VIX={vix_now:.1f} (pct={vix_pct_rank:.0f}), "
                    f"spike={vix_change:.1%}, RSI({rsi_period})={rsi:.1f}"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Leveraged ETF Mean-Reversion (H7.3 — Family 7, Track D)
# ---------------------------------------------------------------------------


class LeveragedMeanRevStrategy(Strategy):
    """Mean-reversion on leveraged ETFs after consecutive down days.

    For each of TQQQ/UPRO/SOXL:
      - 3+ consecutive down days with cumulative decline > min_decline
      - VIX between vix_low and vix_high
      - VIX RSI(5) > vix_rsi_threshold (fear is peaking)
    Buy at close. Exit: first profitable close, time stop, or stop-loss.

    Parameters:
        etf_symbols (str): Comma-separated leveraged ETF tickers.
        vix_symbol (str): VIX ticker, default "VIX".
        min_down_days (int): Minimum consecutive down days, default 3.
        min_decline (float): Minimum cumulative decline (negative), default -0.08.
        vix_low (float): Min VIX for entry, default 20.
        vix_high (float): Max VIX for entry, default 35.
        vix_rsi_period (int): RSI period for VIX, default 5.
        vix_rsi_threshold (float): Min VIX RSI for entry, default 70.
        time_stop (int): Max holding days, default 5.
        stop_loss (float): Stop-loss pct, default 0.05.
        target_weight (float): Per-position weight, default 0.30.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_dates: dict[str, date] = {}
        self._entry_prices: dict[str, float] = {}

    @staticmethod
    def _rsi(closes: list[float], period: int) -> float | None:
        """Compute RSI from a list of close prices."""
        if len(closes) < period + 1:
            return None
        gains: list[float] = []
        losses: list[float] = []
        for i in range(len(closes) - period, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        etf_str = str(params.get("etf_symbols", "TQQQ,UPRO,SOXL"))
        etf_symbols = [s.strip() for s in etf_str.split(",") if s.strip()]
        vix_sym = str(params.get("vix_symbol", "VIX"))
        min_down = int(params.get("min_down_days", 3))
        min_decline = float(params.get("min_decline", -0.08))
        vix_low = float(params.get("vix_low", 20))
        vix_high = float(params.get("vix_high", 35))
        vix_rsi_period = int(params.get("vix_rsi_period", 5))
        vix_rsi_thresh = float(params.get("vix_rsi_threshold", 70))
        time_stop = int(params.get("time_stop", 5))
        stop_loss_pct = float(params.get("stop_loss", 0.05))
        tgt_weight = float(params.get("target_weight", 0.30))

        # Load VIX data
        vix_data = indicators_df.filter(pl.col("symbol") == vix_sym).sort("date")
        if len(vix_data) < vix_rsi_period + 2:
            return []
        vix_closes = vix_data["close"].to_list()
        vix_now = vix_closes[-1]

        # VIX band check
        if vix_now < vix_low or vix_now > vix_high:
            # Still check exits for existing positions
            return self._check_exits(
                as_of_date, etf_symbols, portfolio, prices,
                time_stop, stop_loss_pct,
            )

        # VIX RSI check
        vix_rsi = self._rsi(vix_closes, vix_rsi_period)
        if vix_rsi is None or vix_rsi < vix_rsi_thresh:
            return self._check_exits(
                as_of_date, etf_symbols, portfolio, prices,
                time_stop, stop_loss_pct,
            )

        signals: list[TradeSignal] = []

        # Check exits first
        signals.extend(self._check_exits(
            as_of_date, etf_symbols, portfolio, prices,
            time_stop, stop_loss_pct,
        ))

        # Check entries for each ETF
        for sym in etf_symbols:
            if sym in portfolio.positions:
                continue

            sym_data = indicators_df.filter(pl.col("symbol") == sym).sort("date")
            if len(sym_data) < min_down + 2:
                continue
            closes = sym_data["close"].to_list()
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            # Count consecutive down days
            consec_down = 0
            for i in range(len(closes) - 1, 0, -1):
                if closes[i] < closes[i - 1]:
                    consec_down += 1
                else:
                    break
            if consec_down < min_down:
                continue

            # Cumulative decline over consecutive down period
            start_price = closes[-(consec_down + 1)]
            if start_price <= 0:
                continue
            cum_decline = closes[-1] / start_price - 1.0
            if cum_decline > min_decline:  # min_decline is negative
                continue

            # All conditions met — ENTER
            self._entry_dates[sym] = as_of_date
            self._entry_prices[sym] = price
            stop_loss_price = price * (1.0 - stop_loss_pct)

            logger.info(
                "LeveragedMeanRev: ENTER %s on %s (decline=%.3f, down=%dd, VIX=%.1f, VIX_RSI=%.1f)",
                sym, as_of_date, cum_decline, consec_down, vix_now, vix_rsi,
            )

            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=stop_loss_price,
                    reasoning=(
                        f"Lev MR: {sym} {consec_down}d down, decline={cum_decline:.1%}, "
                        f"VIX={vix_now:.1f}, VIX_RSI={vix_rsi:.1f}"
                    ),
                )
            )

        return signals

    def _check_exits(
        self,
        as_of_date: date,
        etf_symbols: list[str],
        portfolio: Portfolio,
        prices: dict[str, float],
        time_stop: int,
        stop_loss_pct: float,
    ) -> list[TradeSignal]:
        """Check exit conditions for all held positions."""
        signals: list[TradeSignal] = []
        for sym in etf_symbols:
            if sym not in portfolio.positions:
                continue
            price = prices.get(sym, 0)
            if price <= 0:
                continue

            entry_price = self._entry_prices.get(sym)
            entry_date = self._entry_dates.get(sym)

            # Exit 1: First profitable close
            if entry_price is not None and price > entry_price:
                self._entry_dates.pop(sym, None)
                self._entry_prices.pop(sym, None)
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"Lev MR exit: {sym} profitable close ({price:.2f} > {entry_price:.2f})",
                    )
                )
                continue

            # Exit 2: Time stop
            if entry_date is not None:
                days_held = (as_of_date - entry_date).days
                if days_held >= time_stop:
                    self._entry_dates.pop(sym, None)
                    self._entry_prices.pop(sym, None)
                    signals.append(
                        TradeSignal(
                            symbol=sym,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=f"Lev MR exit: {sym} time stop ({days_held}d >= {time_stop}d)",
                        )
                    )
                    continue

            # Exit 3: Stop-loss
            if entry_price is not None:
                loss = (price - entry_price) / entry_price
                if loss <= -stop_loss_pct:
                    self._entry_dates.pop(sym, None)
                    self._entry_prices.pop(sym, None)
                    signals.append(
                        TradeSignal(
                            symbol=sym,
                            action=Action.CLOSE,
                            conviction=Conviction.HIGH,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=f"Lev MR exit: {sym} stop-loss ({loss:.1%})",
                        )
                    )

        return signals


# ---------------------------------------------------------------------------
# Adaptive Dual Momentum Sector Rotation (H3.2 — Family 3)
# ---------------------------------------------------------------------------


class AdaptiveSectorMomentumStrategy(Strategy):
    """VIX-regime-adaptive sector rotation using Accelerating Momentum Score.

    AMS = (1M ret * 12) + (3M ret * 4) + (6M ret * 2) + 12M ret
    Combines recency-weighted momentum across 4 horizons.

    VIX regime gate:
      VIX < 20: top N sectors (equal weight)
      VIX 20-30: top N_mid sectors + GLD
      VIX >= 30: 100% SHY

    Absolute momentum filter: only allocate to sectors if SPY 12M > SHY 12M.

    Parameters:
        sector_symbols (str): Comma-separated sector ETFs.
        spy_symbol (str): SPY for absolute momentum, default "SPY".
        gld_symbol (str): Gold ETF for mid-vol allocation, default "GLD".
        shy_symbol (str): Short treasury for high-vol / abs mom filter, default "SHY".
        vix_symbol (str): VIX ticker, default "VIX".
        top_n_low_vix (int): Sectors to hold in low VIX, default 3.
        top_n_mid_vix (int): Sectors to hold in mid VIX, default 2.
        vix_low (float): Low/mid VIX boundary, default 20.
        vix_high (float): Mid/high VIX boundary, default 30.
        gld_weight_mid (float): GLD weight in mid-VIX regime, default 0.50.
        rebalance_days (int): Min days between rebalances, default 10.
        target_weight (float): Total target allocation, default 0.90.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._last_rebalance: date | None = None

    @staticmethod
    def _trailing_return(closes: list[float], days: int) -> float | None:
        """Compute trailing return over `days` trading days."""
        if len(closes) < days + 1:
            return None
        if closes[-(days + 1)] <= 0:
            return None
        return closes[-1] / closes[-(days + 1)] - 1.0

    def _compute_ams(self, closes: list[float]) -> float | None:
        """Compute Accelerating Momentum Score.

        AMS = (1M ret * 12) + (3M ret * 4) + (6M ret * 2) + 12M ret
        where 1M=21d, 3M=63d, 6M=126d, 12M=252d.
        """
        ret_1m = self._trailing_return(closes, 21)
        ret_3m = self._trailing_return(closes, 63)
        ret_6m = self._trailing_return(closes, 126)
        ret_12m = self._trailing_return(closes, 252)
        if any(r is None for r in [ret_1m, ret_3m, ret_6m, ret_12m]):
            return None
        return ret_1m * 12 + ret_3m * 4 + ret_6m * 2 + ret_12m  # type: ignore[operator]

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        sector_str = str(params.get("sector_symbols", "XLF,XLE,XLK,XLV,XLU,XLI,XLP"))
        sectors = [s.strip() for s in sector_str.split(",") if s.strip()]
        spy_sym = str(params.get("spy_symbol", "SPY"))
        gld_sym = str(params.get("gld_symbol", "GLD"))
        shy_sym = str(params.get("shy_symbol", "SHY"))
        vix_sym = str(params.get("vix_symbol", "VIX"))
        top_n_low = int(params.get("top_n_low_vix", 3))
        top_n_mid = int(params.get("top_n_mid_vix", 2))
        vix_low = float(params.get("vix_low", 20))
        vix_high = float(params.get("vix_high", 30))
        gld_weight = float(params.get("gld_weight_mid", 0.50))
        rebal_days = int(params.get("rebalance_days", 10))
        tgt_weight = float(params.get("target_weight", 0.90))

        # Check rebalance interval
        if self._last_rebalance is not None:
            days_since = (as_of_date - self._last_rebalance).days
            if days_since < rebal_days:
                return []

        # Get VIX level
        vix_data = indicators_df.filter(pl.col("symbol") == vix_sym).sort("date")
        if len(vix_data) < 5:
            return []
        vix_now = vix_data["close"].to_list()[-1]

        # Determine target allocation based on VIX regime
        all_symbols = set(sectors) | {spy_sym, gld_sym, shy_sym}
        target_holdings: dict[str, float] = {}

        if vix_now >= vix_high:
            # High vol: 100% SHY
            if shy_sym in prices and prices[shy_sym] > 0:
                target_holdings[shy_sym] = tgt_weight
        else:
            # Check absolute momentum: SPY 12M > SHY 12M
            spy_data = indicators_df.filter(pl.col("symbol") == spy_sym).sort("date")
            shy_data = indicators_df.filter(pl.col("symbol") == shy_sym).sort("date")
            if len(spy_data) < 253 or len(shy_data) < 253:
                return []
            spy_closes = spy_data["close"].to_list()
            shy_closes = shy_data["close"].to_list()
            spy_12m = self._trailing_return(spy_closes, 252)
            shy_12m = self._trailing_return(shy_closes, 252)

            if spy_12m is None or shy_12m is None or spy_12m <= shy_12m:
                # Absolute momentum fails → 100% SHY
                if shy_sym in prices and prices[shy_sym] > 0:
                    target_holdings[shy_sym] = tgt_weight
            else:
                # Rank sectors by AMS
                scores: list[tuple[str, float]] = []
                for sym in sectors:
                    sym_data = indicators_df.filter(pl.col("symbol") == sym).sort("date")
                    if len(sym_data) < 253:
                        continue
                    closes = sym_data["close"].to_list()
                    ams = self._compute_ams(closes)
                    if ams is not None:
                        scores.append((sym, ams))

                scores.sort(key=lambda x: x[1], reverse=True)

                if vix_now < vix_low:
                    # Low vol: top N sectors, equal weight
                    top_sectors = [s for s, _ in scores[:top_n_low]]
                    if top_sectors:
                        w_per = tgt_weight / len(top_sectors)
                        for s in top_sectors:
                            if s in prices and prices[s] > 0:
                                target_holdings[s] = w_per
                else:
                    # Mid vol: top N_mid sectors + GLD
                    equity_weight = tgt_weight * (1.0 - gld_weight)
                    top_sectors = [s for s, _ in scores[:top_n_mid]]
                    if top_sectors:
                        w_per = equity_weight / len(top_sectors)
                        for s in top_sectors:
                            if s in prices and prices[s] > 0:
                                target_holdings[s] = w_per
                    if gld_sym in prices and prices[gld_sym] > 0:
                        target_holdings[gld_sym] = tgt_weight * gld_weight

        # Generate signals: close positions not in target, open target positions
        signals: list[TradeSignal] = []
        managed_symbols = set(sectors) | {gld_sym, shy_sym}
        current_positions = set(portfolio.positions.keys()) & managed_symbols

        # Close positions not in target
        for sym in current_positions - set(target_holdings.keys()):
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=f"Sector rotation: {sym} no longer in target (VIX={vix_now:.1f})",
                )
            )

        # Open or adjust target positions
        for sym, weight in target_holdings.items():
            if sym not in portfolio.positions:
                p = prices.get(sym, 0)
                if p <= 0:
                    continue
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        action=Action.BUY,
                        conviction=Conviction.MEDIUM,
                        target_weight=weight,
                        stop_loss=p * 0.93,
                        reasoning=f"Sector rotation: {sym} target={weight:.0%} (VIX={vix_now:.1f})",
                    )
                )

        if signals:
            self._last_rebalance = as_of_date

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
    "trend_following": TrendFollowingStrategy,
    "multi_factor": MultiFactorStrategy,
    "correlation_regime": CorrelationRegimeStrategy,
    "correlation_surprise": CorrelationSurpriseStrategy,
    "calendar_event": CalendarEventStrategy,
    "pairs_ratio": PairsRatioStrategy,
    "lead_lag": LeadLagStrategy,
    "asset_rotation": AssetRotationStrategy,
    "vix_regime": VixRegimeStrategy,
    "yield_curve_regime": YieldCurveRegimeStrategy,
    "ohlcv_momentum": OHLCVMomentumStrategy,
    "overnight_momentum": OvernightMomentumStrategy,
    "cef_discount": CEFDiscountRegistryStrategy,
    "nlp_signal": NlpSignalStrategy,
    "rsi2_contrarian": RSI2ContrarianStrategy,
    "vol_scaled_tsmom": VolScaledTsmomStrategy,
    "vrp_timing": VRPTimingStrategy,
    "macro_barometer": MacroBarometerStrategy,
    "opex_week": OpexWeekStrategy,
    "weekend_drift": WeekendDriftStrategy,
    "crypto_weekend_equity": CryptoWeekendEquityStrategy,
    "vix_percentile_spike": VIXPercentileSpikeStrategy,
    "leveraged_mean_rev": LeveragedMeanRevStrategy,
    "adaptive_sector_momentum": AdaptiveSectorMomentumStrategy,
}


def create_strategy(name: str, config: StrategyConfig) -> Strategy:
    """Create a strategy instance by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(config)

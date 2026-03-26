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
from llm_quant.backtest.strategy import SMACrossoverStrategy, Strategy, StrategyConfig
from llm_quant.brain.models import Action, Conviction, TradeSignal
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
}


def create_strategy(name: str, config: StrategyConfig) -> Strategy:
    """Create a strategy instance by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(config)

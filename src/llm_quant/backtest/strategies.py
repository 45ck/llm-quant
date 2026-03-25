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
}


def create_strategy(name: str, config: StrategyConfig) -> Strategy:
    """Create a strategy instance by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(config)

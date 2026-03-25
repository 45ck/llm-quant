"""Crypto Microstructure strategy implementation.

Implements three orthogonal crypto signals -- BTC-SPY correlation regime (CM-005),
ETH/BTC risk appetite barometer (CM-009), and volatility-targeted sizing (CM-012).
All computations are causal (backward-looking only).  See
data/strategies/crypto-microstructure/research-spec.yaml for the frozen research design.
"""

from __future__ import annotations

import logging
import math
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


def _latest_close(sym_df: pl.DataFrame) -> float | None:
    """Return the most recent close price, or None."""
    if len(sym_df) == 0:
        return None
    return sym_df.tail(1).row(0, named=True)["close"]


# ---------------------------------------------------------------------------
# Helper: rolling correlation between two return series
# ---------------------------------------------------------------------------


def _rolling_correlation(
    indicators_df: pl.DataFrame,
    symbol_a: str,
    symbol_b: str,
    window: int,
) -> float | None:
    """Compute trailing *window*-day Pearson correlation of daily returns.

    Returns the most recent correlation value, or None if insufficient data.
    Both series are aligned by date (inner join) and daily returns are computed
    from close prices.
    """
    df_a = _sym_series(indicators_df, symbol_a)
    df_b = _sym_series(indicators_df, symbol_b)
    if len(df_a) == 0 or len(df_b) == 0:
        return None

    close_col = "adj_close" if "adj_close" in df_a.columns else "close"

    a_sel = df_a.select(pl.col("date"), pl.col(close_col).alias("close_a"))
    b_sel = df_b.select(pl.col("date"), pl.col(close_col).alias("close_b"))
    joined = a_sel.join(b_sel, on="date", how="inner").sort("date")

    if len(joined) < window + 1:
        return None

    # Compute daily returns
    joined = joined.with_columns(
        (pl.col("close_a") / pl.col("close_a").shift(1) - 1.0).alias("ret_a"),
        (pl.col("close_b") / pl.col("close_b").shift(1) - 1.0).alias("ret_b"),
    )
    # Drop first row (null from shift)
    joined = joined.drop_nulls(subset=["ret_a", "ret_b"])

    if len(joined) < window:
        return None

    # Take trailing window
    tail = joined.tail(window)
    ret_a = tail["ret_a"].to_list()
    ret_b = tail["ret_b"].to_list()

    # Pearson correlation
    n = len(ret_a)
    mean_a = sum(ret_a) / n
    mean_b = sum(ret_b) / n
    cov = (
        sum((a - mean_a) * (b - mean_b) for a, b in zip(ret_a, ret_b, strict=True)) / n
    )
    std_a = (sum((a - mean_a) ** 2 for a in ret_a) / n) ** 0.5
    std_b = (sum((b - mean_b) ** 2 for b in ret_b) / n) ** 0.5
    if std_a == 0 or std_b == 0:
        return None
    return cov / (std_a * std_b)


# ---------------------------------------------------------------------------
# Helper: price ratio and SMA
# ---------------------------------------------------------------------------


def _price_ratio_with_sma(
    indicators_df: pl.DataFrame,
    numerator_symbol: str,
    denominator_symbol: str,
    sma_period: int,
) -> tuple[float | None, float | None]:
    """Compute the current price ratio and its SMA.

    Returns (current_ratio, sma_of_ratio), either may be None.
    """
    num = _sym_series(indicators_df, numerator_symbol)
    den = _sym_series(indicators_df, denominator_symbol)
    if len(num) == 0 or len(den) == 0:
        return None, None

    close_col = "adj_close" if "adj_close" in num.columns else "close"

    num_sel = num.select(pl.col("date"), pl.col(close_col).alias("num_close"))
    den_sel = den.select(pl.col("date"), pl.col(close_col).alias("den_close"))
    joined = num_sel.join(den_sel, on="date", how="inner").sort("date")

    ratios: list[float] = []
    for row in joined.iter_rows(named=True):
        den_val = row["den_close"]
        if den_val is not None and den_val > 0:
            ratios.append(row["num_close"] / den_val)

    if len(ratios) == 0:
        return None, None

    current_ratio = ratios[-1]
    if len(ratios) < sma_period:
        return current_ratio, None

    sma = sum(ratios[-sma_period:]) / sma_period
    return current_ratio, sma


# ---------------------------------------------------------------------------
# Helper: realized volatility
# ---------------------------------------------------------------------------


def _realized_vol(
    indicators_df: pl.DataFrame,
    symbol: str,
    window: int,
    annualization_factor: int = 365,
) -> float | None:
    """Compute *window*-day realized vol, annualized.

    Uses sqrt(*annualization_factor*) -- 365 for crypto (24/7 trading),
    252 for equities.
    """
    sym_df = _sym_series(indicators_df, symbol)
    if len(sym_df) < window + 1:
        return None

    close_col = "adj_close" if "adj_close" in sym_df.columns else "close"
    closes = sym_df[close_col].to_list()

    # Daily log returns
    log_rets: list[float] = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] is not None and closes[i] is not None and closes[i - 1] > 0
    ]

    if len(log_rets) < window:
        return None

    trailing = log_rets[-window:]
    mean_ret = sum(trailing) / window
    variance = sum((r - mean_ret) ** 2 for r in trailing) / window
    daily_vol = variance**0.5
    return daily_vol * (annualization_factor**0.5)


# ---------------------------------------------------------------------------
# Crypto Microstructure Strategy
# ---------------------------------------------------------------------------


class CryptoMicrostructureStrategy(Strategy):
    """Crypto microstructure: correlation regime + ETH/BTC rotation + vol sizing.

    Hierarchical signal priority:
      1. BTC-SPY correlation regime (CM-005): determines total crypto budget
      2. ETH/BTC risk appetite (CM-009): determines intra-crypto allocation
      3. Vol-targeted sizing (CM-012): determines position sizes

    Required symbols: BTC-USD, ETH-USD, SOL-USD, XRP-USD, ADA-USD, SPY.
    """

    # Crypto universe
    CRYPTO_SYMBOLS: ClassVar[list[str]] = [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "XRP-USD",
        "ADA-USD",
    ]
    ALT_SYMBOLS: ClassVar[list[str]] = ["SOL-USD", "XRP-USD", "ADA-USD"]
    CORRELATION_REFERENCE: ClassVar[str] = "SPY"

    # Max per-position weight for crypto (hard constraint from risk/manager.py)
    MAX_CRYPTO_WEIGHT: ClassVar[float] = 0.05

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters

        # Parameter extraction with defaults from research spec
        corr_window = params.get("btc_spy_corr_window", 30)
        corr_high = params.get("corr_high_threshold", 0.70)
        corr_low = params.get("corr_low_threshold", 0.30)
        eth_btc_sma_period = params.get("eth_btc_sma_period", 14)
        vol_window = params.get("realized_vol_window", 20)
        target_vol = params.get("target_vol_annual", 0.15)
        stop_mult = params.get("stop_atr_multiplier", 2.0)

        # Signal 1: BTC-SPY correlation regime (CM-005)
        budget_scale, regime_label = self._correlation_regime(
            indicators_df,
            corr_window,
            corr_high,
            corr_low,
        )

        # Signal 2: ETH/BTC rotation (CM-009)
        intra_weights, rotation_label = self._eth_btc_rotation(
            indicators_df,
            eth_btc_sma_period,
        )
        logger.info("ETH/BTC rotation: %s", rotation_label)

        # Generate signals: vol-targeted sizing (CM-012)
        signals: list[TradeSignal] = []
        for symbol in self.CRYPTO_SYMBOLS:
            sig = self._symbol_signal(
                symbol,
                indicators_df,
                portfolio,
                prices,
                intra_weights,
                budget_scale,
                vol_window,
                target_vol,
                stop_mult,
                regime_label,
                rotation_label,
            )
            if sig is not None:
                signals.append(sig)

        return signals

    # ------------------------------------------------------------------
    # Signal 1: BTC-SPY correlation regime (CM-005)
    # ------------------------------------------------------------------

    def _correlation_regime(
        self,
        indicators_df: pl.DataFrame,
        corr_window: int,
        corr_high: float,
        corr_low: float,
    ) -> tuple[float, str]:
        """Classify BTC-SPY correlation regime.

        Returns (budget_scale, regime_label).
        """
        btc_spy_corr = _rolling_correlation(
            indicators_df,
            "BTC-USD",
            self.CORRELATION_REFERENCE,
            corr_window,
        )

        if btc_spy_corr is None:
            budget_scale = 1.0
            regime_label = "UNKNOWN (insufficient data)"
        elif btc_spy_corr > corr_high:
            budget_scale = 0.5
            regime_label = f"HIGH_CORR ({btc_spy_corr:.2f} > {corr_high})"
        elif btc_spy_corr < corr_low:
            budget_scale = 1.5
            regime_label = f"LOW_CORR ({btc_spy_corr:.2f} < {corr_low})"
        else:
            budget_scale = 1.0
            regime_label = f"NORMAL ({btc_spy_corr:.2f})"

        logger.info(
            "BTC-SPY corr regime: %s, scale=%.2f",
            regime_label,
            budget_scale,
        )
        return budget_scale, regime_label

    # ------------------------------------------------------------------
    # Signal 2: ETH/BTC rotation (CM-009)
    # ------------------------------------------------------------------

    def _eth_btc_rotation(
        self,
        indicators_df: pl.DataFrame,
        sma_period: int,
    ) -> tuple[dict[str, float], str]:
        """Determine intra-crypto allocation from ETH/BTC ratio.

        Returns (intra_weights, rotation_label).
        """
        ratio, sma = _price_ratio_with_sma(
            indicators_df,
            "ETH-USD",
            "BTC-USD",
            sma_period,
        )

        if ratio is not None and sma is not None:
            if ratio > sma:
                weights = {
                    "BTC-USD": 0.30,
                    "ETH-USD": 0.30,
                    "SOL-USD": 0.15,
                    "XRP-USD": 0.13,
                    "ADA-USD": 0.12,
                }
                label = "RISK_APPETITE_ON (ETH/BTC rising)"
            else:
                weights = {
                    "BTC-USD": 0.60,
                    "ETH-USD": 0.25,
                    "SOL-USD": 0.06,
                    "XRP-USD": 0.05,
                    "ADA-USD": 0.04,
                }
                label = "RISK_APPETITE_OFF (ETH/BTC falling)"
        else:
            weights = {
                "BTC-USD": 0.50,
                "ETH-USD": 0.25,
                "SOL-USD": 0.10,
                "XRP-USD": 0.08,
                "ADA-USD": 0.07,
            }
            label = "DEFAULT (insufficient ETH/BTC data)"

        return weights, label

    # ------------------------------------------------------------------
    # Per-symbol signal generation
    # ------------------------------------------------------------------

    def _symbol_signal(  # noqa: PLR0913
        self,
        symbol: str,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        intra_weights: dict[str, float],
        budget_scale: float,
        vol_window: int,
        target_vol: float,
        stop_mult: float,
        regime_label: str,
        rotation_label: str,
    ) -> TradeSignal | None:
        """Generate a signal for a single crypto symbol."""
        close = prices.get(symbol, 0.0)
        if close <= 0:
            return None

        has_position = symbol in portfolio.positions
        base_weight = intra_weights.get(symbol, 0.0) * budget_scale

        # Vol-targeted sizing
        sized_weight = self._vol_sized_weight(
            indicators_df,
            symbol,
            base_weight,
            vol_window,
            target_vol,
        )

        if sized_weight < 0.005:
            if has_position:
                return TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        f"Allocation too small ({sized_weight:.3f}); "
                        f"corr={regime_label}, "
                        f"rot={rotation_label}"
                    ),
                )
            return None

        if not has_position:
            return self._entry_signal(
                symbol,
                close,
                sized_weight,
                base_weight,
                indicators_df,
                portfolio,
                stop_mult,
                regime_label,
                rotation_label,
            )

        return self._rebalance_signal(
            symbol,
            close,
            sized_weight,
            indicators_df,
            portfolio,
            stop_mult,
            regime_label,
            rotation_label,
        )

    def _vol_sized_weight(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        base_weight: float,
        vol_window: int,
        target_vol: float,
    ) -> float:
        """Apply vol-targeting and cap to base weight."""
        rvol = _realized_vol(
            indicators_df,
            symbol,
            vol_window,
            365,
        )
        if rvol is not None and rvol > 0:
            sized = base_weight * (target_vol / rvol)
        else:
            sized = base_weight
        return min(sized, self.MAX_CRYPTO_WEIGHT)

    def _entry_signal(  # noqa: PLR0913
        self,
        symbol: str,
        close: float,
        sized_weight: float,
        base_weight: float,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        stop_mult: float,
        regime_label: str,
        rotation_label: str,
    ) -> TradeSignal | None:
        """Create an entry signal for a new position."""
        if len(portfolio.positions) >= self.config.max_positions:
            return None
        sl = self._compute_stop_loss(
            indicators_df,
            symbol,
            close,
            stop_mult,
        )
        return TradeSignal(
            symbol=symbol,
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=sized_weight,
            stop_loss=sl,
            reasoning=(
                f"Crypto entry: w={sized_weight:.3f} "
                f"(base={base_weight:.3f}); "
                f"corr={regime_label}, "
                f"rot={rotation_label}"
            ),
        )

    def _rebalance_signal(  # noqa: PLR0913
        self,
        symbol: str,
        close: float,
        sized_weight: float,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        stop_mult: float,
        regime_label: str,
        rotation_label: str,
    ) -> TradeSignal | None:
        """Rebalance an existing position if weight drifted."""
        current_pos = portfolio.positions[symbol]
        current_weight = (
            (current_pos.shares * close) / portfolio.nav if portfolio.nav > 0 else 0
        )
        weight_drift = abs(current_weight - sized_weight)
        if weight_drift <= 0.01:
            return None

        if sized_weight > current_weight:
            sl = self._compute_stop_loss(
                indicators_df,
                symbol,
                close,
                stop_mult,
            )
            return TradeSignal(
                symbol=symbol,
                action=Action.BUY,
                conviction=Conviction.LOW,
                target_weight=sized_weight,
                stop_loss=sl,
                reasoning=(
                    f"Rebalance up: "
                    f"target={sized_weight:.3f} "
                    f"vs current={current_weight:.3f}; "
                    f"corr={regime_label}, "
                    f"rot={rotation_label}"
                ),
            )

        return TradeSignal(
            symbol=symbol,
            action=Action.SELL,
            conviction=Conviction.LOW,
            target_weight=sized_weight,
            stop_loss=0.0,
            reasoning=(
                f"Rebalance down: "
                f"target={sized_weight:.3f} "
                f"vs current={current_weight:.3f}; "
                f"corr={regime_label}, "
                f"rot={rotation_label}"
            ),
        )

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
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "crypto_microstructure": CryptoMicrostructureStrategy,
}

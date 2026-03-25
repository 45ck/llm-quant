"""Cross-Asset Lead-Lag strategy implementation.

Implements three cross-asset information diffusion channels -- bond-equity lead
(LL-H01), credit-equity lead (LL-H02), and JPY carry trade unwind (LL-H05) --
with composite scoring, additive reduction (capped at 60%), and anti-whipsaw
guard.  All computations are causal (backward-looking only).  See
data/strategies/cross-asset-lead-lag/research-spec.yaml for the frozen research
design.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import ClassVar

import polars as pl

from llm_quant.backtest.strategy import Strategy, StrategyConfig
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
# Helper: n-day trailing return
# ---------------------------------------------------------------------------


def _trailing_return(
    indicators_df: pl.DataFrame,
    symbol: str,
    window: int,
) -> float | None:
    """Compute the *window*-day trailing return for *symbol*.

    Returns (close_today / close_window_days_ago) - 1, or None.
    """
    sym_df = _sym_series(indicators_df, symbol)
    if len(sym_df) <= window:
        return None

    close_col = "adj_close" if "adj_close" in sym_df.columns else "close"
    closes = sym_df[close_col].to_list()

    current = closes[-1]
    past = closes[-(window + 1)]

    if past is None or current is None or past == 0:
        return None
    return current / past - 1.0


# ---------------------------------------------------------------------------
# Helper: price-ratio trailing return
# ---------------------------------------------------------------------------


def _ratio_trailing_return(
    indicators_df: pl.DataFrame,
    numerator_symbol: str,
    denominator_symbol: str,
    window: int,
) -> float | None:
    """Compute *window*-day trailing return of a price ratio.

    Returns (ratio_today / ratio_window_ago) - 1, or None.
    """
    num = _sym_series(indicators_df, numerator_symbol)
    den = _sym_series(indicators_df, denominator_symbol)
    if len(num) == 0 or len(den) == 0:
        return None

    close_col = "adj_close" if "adj_close" in num.columns else "close"

    num_sel = num.select(pl.col("date"), pl.col(close_col).alias("num_close"))
    den_sel = den.select(pl.col("date"), pl.col(close_col).alias("den_close"))
    joined = num_sel.join(den_sel, on="date", how="inner").sort("date")

    if len(joined) <= window:
        return None

    ratios: list[float] = []
    for row in joined.iter_rows(named=True):
        den_val = row["den_close"]
        if den_val is not None and den_val > 0:
            ratios.append(row["num_close"] / den_val)

    if len(ratios) <= window:
        return None

    current = ratios[-1]
    past = ratios[-(window + 1)]
    if past == 0:
        return None
    return current / past - 1.0


# ---------------------------------------------------------------------------
# Cross-Asset Lead-Lag Strategy
# ---------------------------------------------------------------------------


class CrossAssetLeadLagStrategy(Strategy):
    """Cross-asset lead-lag: bond, credit, and forex early warning channels.

    Signal hierarchy (additive, not prioritized):
      1. Bond channel (LL-H01): TLT 5-day return < -2% warns for equities
      2. Credit channel (LL-H02): HYG/LQD 10-day return < -1.5% warns for equities
      3. Forex channel (LL-H05): USDJPY 3-day return < -1.5% warns for equities

    Composite logic:
      - All clear: full equity allocation (entry/hold)
      - Any warning: proportional reduction (additive across channels)
      - Total reduction capped at 60% per rebalance

    Anti-whipsaw: minimum 3 trading days between rebalances.

    Required symbols: TLT, HYG, LQD, USDJPY=X, SPY, QQQ, EEM, EFA.
    """

    # Target equity symbols to trade
    EQUITY_TARGETS: ClassVar[list[str]] = ["SPY", "QQQ", "EEM", "EFA"]
    # Safe-haven symbols during stress
    SAFE_HAVENS: ClassVar[list[str]] = ["TLT", "GLD"]

    # Reduction percentages per channel at moderate severity
    BOND_REDUCTION: ClassVar[float] = 0.25
    CREDIT_REDUCTION: ClassVar[float] = 0.20
    FOREX_REDUCTION: ClassVar[float] = 0.30

    # Maximum combined reduction cap
    MAX_REDUCTION: ClassVar[float] = 0.60

    # Anti-whipsaw: minimum days between rebalances
    MIN_REBALANCE_GAP_DAYS: ClassVar[int] = 3

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._last_rebalance_date: date | None = None

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters

        # Parameter extraction with defaults from research spec
        tlt_window = params.get("tlt_return_window", 5)
        tlt_threshold = params.get("tlt_selloff_threshold", -0.02)
        hyg_lqd_window = params.get("hyg_lqd_return_window", 10)
        credit_threshold = params.get("credit_deterioration_threshold", -0.015)
        usdjpy_window = params.get("usdjpy_return_window", 3)
        usdjpy_threshold = params.get("usdjpy_unwind_threshold", -0.015)
        stop_mult = params.get("stop_atr_multiplier", 2.0)
        min_gap = params.get("min_rebalance_gap_days", self.MIN_REBALANCE_GAP_DAYS)

        # ------------------------------------------------------------------
        # Anti-whipsaw check
        # ------------------------------------------------------------------
        if self._last_rebalance_date is not None:
            days_since = (as_of_date - self._last_rebalance_date).days
            if days_since < min_gap:
                logger.info(
                    "Anti-whipsaw: %d days since last rebalance (min %d). Skipping.",
                    days_since,
                    min_gap,
                )
                return []

        # ------------------------------------------------------------------
        # Channel 1: Bond-equity lead (LL-H01)
        # ------------------------------------------------------------------
        tlt_ret = _trailing_return(indicators_df, "TLT", tlt_window)
        bond_warning = False
        bond_reduction = 0.0
        bond_label = "CLEAR"

        if tlt_ret is not None and tlt_ret < tlt_threshold:
            bond_warning = True
            if tlt_ret < tlt_threshold * 1.5:
                # Strong: TLT selloff > 3%
                bond_reduction = self.BOND_REDUCTION * 2.0
                bond_label = f"STRONG (TLT {tlt_window}d ret={tlt_ret:.2%})"
            else:
                bond_reduction = self.BOND_REDUCTION
                bond_label = f"MODERATE (TLT {tlt_window}d ret={tlt_ret:.2%})"

        logger.info(
            "Bond channel: %s, reduction=%.1f%%",
            bond_label,
            bond_reduction * 100,
        )

        # ------------------------------------------------------------------
        # Channel 2: Credit-equity lead (LL-H02)
        # ------------------------------------------------------------------
        hyg_lqd_ret = _ratio_trailing_return(
            indicators_df, "HYG", "LQD", hyg_lqd_window
        )
        credit_warning = False
        credit_reduction = 0.0
        credit_label = "CLEAR"

        if hyg_lqd_ret is not None and hyg_lqd_ret < credit_threshold:
            credit_warning = True
            if hyg_lqd_ret < credit_threshold * 1.67:
                # Strong: HYG/LQD decline > 2.5%
                credit_reduction = self.CREDIT_REDUCTION * 2.0
                credit_label = (
                    f"STRONG (HYG/LQD {hyg_lqd_window}d ret={hyg_lqd_ret:.2%})"
                )
            else:
                credit_reduction = self.CREDIT_REDUCTION
                credit_label = (
                    f"MODERATE (HYG/LQD {hyg_lqd_window}d ret={hyg_lqd_ret:.2%})"
                )

        logger.info(
            "Credit channel: %s, reduction=%.1f%%",
            credit_label,
            credit_reduction * 100,
        )

        # ------------------------------------------------------------------
        # Channel 3: JPY carry unwind (LL-H05)
        # ------------------------------------------------------------------
        usdjpy_ret = _trailing_return(indicators_df, "USDJPY=X", usdjpy_window)
        forex_warning = False
        forex_reduction = 0.0
        forex_label = "CLEAR"

        if usdjpy_ret is not None and usdjpy_ret < usdjpy_threshold:
            forex_warning = True
            if usdjpy_ret < usdjpy_threshold * 1.67:
                # Strong: USDJPY decline > 2.5%
                forex_reduction = self.FOREX_REDUCTION * 1.67
                forex_label = f"STRONG (USDJPY {usdjpy_window}d ret={usdjpy_ret:.2%})"
            else:
                forex_reduction = self.FOREX_REDUCTION
                forex_label = f"MODERATE (USDJPY {usdjpy_window}d ret={usdjpy_ret:.2%})"

        logger.info(
            "Forex channel: %s, reduction=%.1f%%",
            forex_label,
            forex_reduction * 100,
        )

        # ------------------------------------------------------------------
        # Composite: additive reduction, capped at MAX_REDUCTION (60%)
        # ------------------------------------------------------------------
        total_reduction = min(
            bond_reduction + credit_reduction + forex_reduction,
            self.MAX_REDUCTION,
        )
        any_warning = bond_warning or credit_warning or forex_warning
        all_clear = not any_warning

        composite_reason = (
            f"Bond={bond_label}, Credit={credit_label}, Forex={forex_label}; "
            f"total reduction={total_reduction:.0%}"
        )
        logger.info("Composite: %s", composite_reason)

        # ------------------------------------------------------------------
        # Generate signals
        # ------------------------------------------------------------------
        signals: list[TradeSignal] = []
        warning_count = sum(
            [bond_warning, credit_warning, forex_warning],
        )

        if all_clear:
            signals.extend(
                self._all_clear_signals(
                    portfolio,
                    prices,
                    indicators_df,
                    stop_mult,
                    composite_reason,
                )
            )
        else:
            signals.extend(
                self._warning_signals(
                    total_reduction,
                    portfolio,
                    prices,
                    indicators_df,
                    stop_mult,
                    composite_reason,
                    warning_count,
                )
            )

        # Record rebalance date if we generated any signals
        if signals:
            self._last_rebalance_date = as_of_date

        return signals

    # ------------------------------------------------------------------
    # All-clear: entry/hold signals
    # ------------------------------------------------------------------

    def _all_clear_signals(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        reason: str,
    ) -> list[TradeSignal]:
        """All three channels clear -- full equity allocation."""
        signals: list[TradeSignal] = []
        new_pos = 0

        for symbol in self.EQUITY_TARGETS:
            if symbol in portfolio.positions:
                # Already held -- no action needed
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
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
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=f"All-clear entry: {reason}",
                )
            )
            new_pos += 1

        # Close any safe-haven positions (no longer needed)
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.LOW,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Exit safe haven (all clear): {reason}",
            )
            for sym in self.SAFE_HAVENS
            if sym in portfolio.positions
        )

        return signals

    # ------------------------------------------------------------------
    # Warning active: reduce equity, open safe havens
    # ------------------------------------------------------------------

    def _warning_signals(  # noqa: PLR0913
        self,
        total_reduction: float,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        reason: str,
        warning_count: int,
    ) -> list[TradeSignal]:
        """One or more channels warning -- reduce equity."""
        signals: list[TradeSignal] = []

        # Reduce or exit equity positions
        signals.extend(
            self._reduce_equity_signals(
                total_reduction,
                portfolio,
                prices,
                reason,
            )
        )

        # Open safe-haven positions if strong warning
        if warning_count >= 2 or total_reduction >= 0.40:
            signals.extend(
                self._safe_haven_signals(
                    portfolio,
                    prices,
                    indicators_df,
                    stop_mult,
                    reason,
                    warning_count,
                )
            )

        return signals

    def _reduce_equity_signals(
        self,
        total_reduction: float,
        portfolio: Portfolio,
        prices: dict[str, float],
        reason: str,
    ) -> list[TradeSignal]:
        """Reduce or close equity positions."""
        signals: list[TradeSignal] = []

        for symbol in self.EQUITY_TARGETS:
            if symbol not in portfolio.positions:
                continue

            current_pos = portfolio.positions[symbol]
            close = prices.get(symbol, 0)
            if close <= 0:
                continue

            current_weight = (
                (current_pos.shares * close) / portfolio.nav if portfolio.nav > 0 else 0
            )
            reduced_weight = current_weight * (1.0 - total_reduction)

            if reduced_weight < 0.005:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=f"Close on warning: {reason}",
                    )
                )
            else:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.SELL,
                        conviction=Conviction.MEDIUM,
                        target_weight=reduced_weight,
                        stop_loss=0.0,
                        reasoning=(
                            f"Reduce by {total_reduction:.0%}: "
                            f"target={reduced_weight:.3f} "
                            f"vs current={current_weight:.3f}; "
                            f"{reason}"
                        ),
                    )
                )

        return signals

    def _safe_haven_signals(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        reason: str,
        warning_count: int,
    ) -> list[TradeSignal]:
        """Open safe-haven positions during stress."""
        signals: list[TradeSignal] = []
        new_pos = 0
        for sym in self.SAFE_HAVENS:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break
            close = prices.get(sym, 0)
            if close <= 0:
                continue

            sl = self._compute_stop_loss(
                indicators_df,
                sym,
                close,
                stop_mult,
            )
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.HIGH,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=(
                        f"Safe haven entry ({warning_count} channels warning): {reason}"
                    ),
                )
            )
            new_pos += 1

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
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "cross_asset_lead_lag": CrossAssetLeadLagStrategy,
}

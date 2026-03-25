"""Calendar Anomalies strategy implementation.

Implements three well-documented calendar effects:
  CA-H01  Turn-of-Month institutional flow (last 2 + first 3 days)
  CA-H09  Sell-in-May / Halloween effect (equity-bond switching)
  CA-H07  Quarter-end rebalancing reversal in sector ETFs

All computations are causal (backward-looking only).  Calendar
detection uses as_of_date.month and as_of_date.day -- no look-ahead.
See data/strategies/calendar-anomalies/research-spec.yaml for the
frozen research design.
"""

from __future__ import annotations

import calendar
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
# Helpers
# ---------------------------------------------------------------------------


def _sym_series(
    indicators_df: pl.DataFrame,
    symbol: str,
) -> pl.DataFrame:
    """Return the rows for *symbol*, sorted by date."""
    return indicators_df.filter(pl.col("symbol") == symbol).sort("date")


def _latest_close(sym_df: pl.DataFrame) -> float | None:
    """Return the most recent close price, or None."""
    if len(sym_df) == 0:
        return None
    return sym_df.tail(1).row(0, named=True)["close"]


def _latest_indicator(
    sym_df: pl.DataFrame,
    col: str,
) -> float | None:
    """Return the most recent value of *col*, or None."""
    if len(sym_df) == 0 or col not in sym_df.columns:
        return None
    return sym_df.tail(1).row(0, named=True).get(col)


def _quarter_start_month(month: int) -> int:
    """Return the first month of the quarter containing *month*."""
    return ((month - 1) // 3) * 3 + 1


def _is_quarter_end_month(month: int) -> bool:
    """True if *month* is a quarter-end (3, 6, 9, 12)."""
    return month in (3, 6, 9, 12)


def _is_quarter_start_month(month: int) -> bool:
    """True if *month* is a quarter-start (1, 4, 7, 10)."""
    return month in (1, 4, 7, 10)


def _count_weekdays_in_range(
    year: int,
    month: int,
    start_day: int,
    end_day: int,
) -> int:
    """Count weekdays (Mon-Fri) from start_day to end_day inclusive."""
    count = 0
    for d in range(start_day, end_day + 1):
        try:
            dt = date(year, month, d)
            if dt.weekday() < 5:
                count += 1
        except ValueError:
            break
    return count


# ---------------------------------------------------------------------------
# Data classes to bundle sub-strategy params (avoid long arg lists)
# ---------------------------------------------------------------------------


@dataclass
class _TomParams:
    """Turn-of-month parameters."""

    start_offset: int
    end_offset: int
    weight: float
    stop_mult: float


@dataclass
class _SeasonalParams:
    """Sell-in-May parameters."""

    summer_start: int
    summer_end: int
    equity_weight: float
    bond_weight: float
    stop_mult: float


@dataclass
class _QuarterEndParams:
    """Quarter-end reversal parameters."""

    selling_window: int
    reversal_window: int
    sector_count: int
    stop_mult: float


# ---------------------------------------------------------------------------
# Calendar Anomalies Strategy
# ---------------------------------------------------------------------------


class CalendarAnomaliesStrategy(Strategy):
    """Calendar anomaly ensemble: TOM + Sell-in-May + Q-end reversal.

    Enters/exits based on calendar position rather than price signals.
    Each sub-strategy has distinct timing:
      - TOM: 12x/year, 4-day holds (last 2 + first 3 trading days)
      - Sell-in-May: 2 switches/year (Nov-Apr equity, May-Oct bonds)
      - Quarter-end reversal: 4x/year, 5-day holds (buy losers)

    Required symbols: SPY, QQQ, TLT, XLK, XLF, XLE, XLV.
    """

    TOM_SYMBOLS: ClassVar[list[str]] = ["SPY", "QQQ"]
    SEASONAL_EQUITY: ClassVar[list[str]] = ["SPY", "QQQ"]
    SEASONAL_BOND: ClassVar[list[str]] = ["TLT"]
    SECTOR_ETFS: ClassVar[list[str]] = [
        "XLK",
        "XLF",
        "XLE",
        "XLV",
    ]

    VIX_SYMBOL: ClassVar[str] = "VIX"

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters
        stop_mult = params.get("stop_atr_multiplier", 2.5)

        tom = _TomParams(
            start_offset=params.get("tom_start_offset", -1),
            end_offset=params.get("tom_end_offset", 3),
            weight=params.get("tom_position_weight", 0.25),
            stop_mult=stop_mult,
        )
        seasonal = _SeasonalParams(
            summer_start=params.get("summer_start_month", 5),
            summer_end=params.get("summer_end_month", 10),
            equity_weight=params.get("summer_equity_weight", 0.40),
            bond_weight=params.get("summer_bond_weight", 0.30),
            stop_mult=stop_mult,
        )
        qe = _QuarterEndParams(
            selling_window=params.get("qe_selling_window", 5),
            reversal_window=params.get("qe_reversal_window", 5),
            sector_count=params.get("qe_sector_count", 3),
            stop_mult=stop_mult,
        )

        signals: list[TradeSignal] = []
        signals.extend(
            self._tom_signals(
                as_of_date,
                indicators_df,
                portfolio,
                prices,
                tom,
            )
        )
        signals.extend(
            self._seasonal_signals(
                as_of_date,
                indicators_df,
                portfolio,
                prices,
                seasonal,
            )
        )
        signals.extend(
            self._quarter_end_signals(
                as_of_date,
                indicators_df,
                portfolio,
                prices,
                qe,
            )
        )
        return signals

    # ------------------------------------------------------------------
    # CA-H01: Turn-of-Month
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tom_window(
        as_of_date: date,
        tom: _TomParams,
    ) -> bool:
        """Check if as_of_date is in the turn-of-month window.

        TOM window spans the last |start_offset| trading days before
        month-end through the first end_offset trading days of the
        new month.
        """
        month = as_of_date.month
        day = as_of_date.day
        year = as_of_date.year

        days_in_month = calendar.monthrange(year, month)[1]

        # Trading days remaining after today
        trading_remaining = _count_weekdays_in_range(
            year,
            month,
            day + 1,
            days_in_month,
        )

        # Near month-end check
        near_end = trading_remaining <= abs(tom.start_offset)

        # Trading day of month (1-indexed)
        td_of_month = _count_weekdays_in_range(
            year,
            month,
            1,
            day,
        )

        near_start = td_of_month <= tom.end_offset

        return near_end or near_start

    def _tom_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        tom: _TomParams,
    ) -> list[TradeSignal]:
        """Generate TOM entry signals."""
        signals: list[TradeSignal] = []

        # VIX override: skip TOM if VIX(t-1) > 35
        vix_df = _sym_series(indicators_df, self.VIX_SYMBOL)
        if len(vix_df) >= 2:
            vix_prev = vix_df.tail(2).row(0, named=True).get("close")
            if vix_prev is not None and vix_prev > 35:
                logger.info(
                    "TOM entry skipped: VIX(t-1)=%.1f > 35",
                    vix_prev,
                )
                return signals

        if not self._is_tom_window(as_of_date, tom):
            return signals

        weight_per = tom.weight / len(self.TOM_SYMBOLS)
        for sym in self.TOM_SYMBOLS:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) >= self.config.max_positions:
                break

            close = prices.get(sym, 0)
            if close <= 0:
                continue

            # Trend filter: skip if below SMA_50
            sym_df = _sym_series(indicators_df, sym)
            sma_50 = _latest_indicator(sym_df, "sma_50")
            if sma_50 is not None and close < sma_50:
                logger.info(
                    "TOM %s skipped: close=%.2f < SMA50=%.2f",
                    sym,
                    close,
                    sma_50,
                )
                continue

            sl = self._compute_stop_loss(
                indicators_df,
                sym,
                close,
                tom.stop_mult,
            )
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=weight_per,
                    stop_loss=sl,
                    reasoning=(
                        f"CA-H01 TOM entry: "
                        f"{as_of_date.strftime('%b %d')} "
                        f"in turn-of-month window"
                    ),
                )
            )

        return signals

    # ------------------------------------------------------------------
    # CA-H09: Sell-in-May (Halloween Effect)
    # ------------------------------------------------------------------

    def _seasonal_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        sp: _SeasonalParams,
    ) -> list[TradeSignal]:
        """Generate Sell-in-May seasonal switch signals.

        Nov-Apr: hold equities (SPY, QQQ).
        May-Oct: hold bonds (TLT).
        """
        month = as_of_date.month
        day = as_of_date.day

        is_switch_summer = month == sp.summer_start and day <= 5
        switch_strong_month = 1 if sp.summer_end == 12 else sp.summer_end + 1
        is_switch_strong = month == switch_strong_month and day <= 5

        if is_switch_summer:
            return self._summer_switch(
                indicators_df,
                portfolio,
                prices,
                sp,
                month,
            )
        if is_switch_strong:
            return self._strong_switch(
                indicators_df,
                portfolio,
                prices,
                sp,
                month,
            )
        return []

    def _summer_switch(
        self,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        sp: _SeasonalParams,
        month: int,
    ) -> list[TradeSignal]:
        """May: exit equities, enter bonds."""
        signals: list[TradeSignal] = []

        # Exit equities
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.MEDIUM,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=(
                    f"CA-H09 Sell-in-May: exit equities for weak period (month={month})"
                ),
            )
            for sym in self.SEASONAL_EQUITY
            if sym in portfolio.positions
        )

        # Enter bonds
        bond_wt = sp.bond_weight / len(self.SEASONAL_BOND)
        for sym in self.SEASONAL_BOND:
            if sym in portfolio.positions:
                continue
            close = prices.get(sym, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(
                indicators_df,
                sym,
                close,
                sp.stop_mult,
            )
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=bond_wt,
                    stop_loss=sl,
                    reasoning=(
                        f"CA-H09 Sell-in-May: enter bonds "
                        f"for weak period (month={month})"
                    ),
                )
            )

        return signals

    def _strong_switch(
        self,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        sp: _SeasonalParams,
        month: int,
    ) -> list[TradeSignal]:
        """November: exit bonds, enter equities."""
        signals: list[TradeSignal] = []

        # Exit bonds
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.MEDIUM,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=(
                    f"CA-H09 Halloween: exit bonds for strong period (month={month})"
                ),
            )
            for sym in self.SEASONAL_BOND
            if sym in portfolio.positions
        )

        # Enter equities
        eq_wt = sp.equity_weight / len(self.SEASONAL_EQUITY)
        for sym in self.SEASONAL_EQUITY:
            if sym in portfolio.positions:
                continue
            close = prices.get(sym, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(
                indicators_df,
                sym,
                close,
                sp.stop_mult,
            )
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=eq_wt,
                    stop_loss=sl,
                    reasoning=(
                        f"CA-H09 Halloween: enter equities "
                        f"for strong period (month={month})"
                    ),
                )
            )

        return signals

    # ------------------------------------------------------------------
    # CA-H07: Quarter-End Rebalancing Reversal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_near_quarter_start(
        as_of_date: date,
        window: int,
    ) -> bool:
        """True if in the first *window* trading days of a quarter."""
        month = as_of_date.month
        if not _is_quarter_start_month(month):
            return False

        td_count = _count_weekdays_in_range(
            as_of_date.year,
            month,
            1,
            as_of_date.day,
        )
        return td_count <= window

    def _rank_sectors_by_qtd_return(
        self,
        indicators_df: pl.DataFrame,
        as_of_date: date,
    ) -> list[tuple[str, float]]:
        """Rank sector ETFs by prior-quarter return (worst first)."""
        month = as_of_date.month
        year = as_of_date.year

        # Look at the *previous* quarter's performance
        if _is_quarter_start_month(month):
            if month == 1:
                q_start_month = 10
                year -= 1
            else:
                q_start_month = _quarter_start_month(month - 1)
        else:
            q_start_month = _quarter_start_month(month)

        q_start_date = date(year, q_start_month, 1)

        results: list[tuple[str, float]] = []
        for sym in self.SECTOR_ETFS:
            sym_df = _sym_series(indicators_df, sym)
            if len(sym_df) == 0:
                continue

            close_col = "adj_close" if "adj_close" in sym_df.columns else "close"

            q_data = sym_df.filter(
                pl.col("date") >= q_start_date,
            )
            if len(q_data) == 0:
                continue

            start_c = q_data.row(0, named=True)[close_col]
            end_c = sym_df.tail(1).row(0, named=True)[close_col]

            if start_c is None or end_c is None or start_c <= 0:
                continue

            results.append((sym, end_c / start_c - 1.0))

        results.sort(key=lambda x: x[1])
        return results

    def _quarter_end_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        qe: _QuarterEndParams,
    ) -> list[TradeSignal]:
        """Quarter-end reversal: buy prior-quarter losers."""
        signals: list[TradeSignal] = []

        if not self._is_near_quarter_start(
            as_of_date,
            qe.reversal_window,
        ):
            return signals

        ranked = self._rank_sectors_by_qtd_return(
            indicators_df,
            as_of_date,
        )
        if len(ranked) < 2:
            return signals

        losers = ranked[: qe.sector_count]
        weight_per = self.config.target_position_weight

        for sym, qtd_ret in losers:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) >= self.config.max_positions:
                break

            close = prices.get(sym, 0)
            if close <= 0:
                continue

            sl = self._compute_stop_loss(
                indicators_df,
                sym,
                close,
                qe.stop_mult,
            )
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=weight_per,
                    stop_loss=sl,
                    reasoning=(
                        f"CA-H07 Q-end reversal: {sym} loser "
                        f"(QTD={qtd_ret:+.2%}), buy for "
                        f"mean reversion"
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
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "calendar_anomalies": CalendarAnomaliesStrategy,
}

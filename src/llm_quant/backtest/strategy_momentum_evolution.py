"""Momentum Evolution strategy implementation.

Implements three complementary momentum signals -- Sharpe momentum ranking
(ME-H07), momentum decay filter (ME-H03), and dispersion crash hedge (ME-H06)
-- with SMA_200 trend filter and ATR-based stops.  All computations are causal
(backward-looking only).  See data/strategies/momentum-evolution/research-spec.yaml
for the frozen research design.
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
# 11 GICS sector ETFs used for dispersion computation
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sym_series(indicators_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Return the rows for *symbol*, sorted by date."""
    return indicators_df.filter(pl.col("symbol") == symbol).sort("date")


def _latest_close(sym_df: pl.DataFrame) -> float | None:
    """Return the most recent close price, or None."""
    if len(sym_df) == 0:
        return None
    return sym_df.tail(1).row(0, named=True)["close"]


def _close_series(sym_df: pl.DataFrame) -> list[float]:
    """Extract close prices as a plain list, date-sorted."""
    if len(sym_df) == 0:
        return []
    col = "adj_close" if "adj_close" in sym_df.columns else "close"
    return sym_df.select(pl.col(col)).to_series().to_list()


def _daily_returns(closes: list[float]) -> list[float]:
    """Compute simple daily returns from a close-price series."""
    if len(closes) < 2:
        return []
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


# ---------------------------------------------------------------------------
# Momentum Evolution Strategy
# ---------------------------------------------------------------------------


class MomentumEvolutionStrategy(Strategy):
    """Risk-adjusted momentum with decay filtering and crash hedging.

    Signal hierarchy:
      1. Universe filter: SMA_200 trend filter (price > SMA_200)
      2. Sharpe momentum ranking: 63-day return / 63-day std
      3. Decay filter: only buy if 63d return > 126d return (accelerating)
      4. Dispersion crash hedge: halve sizes when cross-sectional sector
         dispersion falls below 20th percentile of 252-day history
      5. Top 5 assets after all filters, ATR-based stops

    Required columns: close (or adj_close), sma_200, atr_14
    """

    SECTOR_ETFS: ClassVar[list[str]] = SECTOR_ETFS

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters

        # Parameter extraction with defaults from research spec
        sharpe_lookback = params.get("sharpe_lookback", 63)
        top_n = params.get("top_n_assets", 5)
        decay_short = params.get("decay_lookback_short", 63)
        decay_long = params.get("decay_lookback_long", 126)
        dispersion_lookback = params.get("dispersion_lookback", 63)
        dispersion_history = params.get("dispersion_history", 252)
        dispersion_threshold_pct = params.get("dispersion_threshold_pct", 20)
        stop_mult = params.get("stop_atr_multiplier", 2.0)

        # ---------------------------------------------------------------
        # Step 1: Compute dispersion crash hedge (position-sizing signal)
        # ---------------------------------------------------------------
        crash_hedge_active = self._is_crash_hedge_active(
            indicators_df,
            dispersion_lookback,
            dispersion_history,
            dispersion_threshold_pct,
        )
        if crash_hedge_active:
            logger.info("Dispersion crash hedge ACTIVE: position sizes halved")

        # ---------------------------------------------------------------
        # Step 2: Rank universe by Sharpe momentum, apply filters
        # ---------------------------------------------------------------
        ranked = self._rank_universe(
            indicators_df,
            sharpe_lookback,
            decay_short,
            decay_long,
        )

        # Select top N
        top_assets = ranked[:top_n]
        top_symbols = {sym for sym, _, _ in top_assets}

        # ---------------------------------------------------------------
        # Step 3: Generate entry/exit signals
        # ---------------------------------------------------------------
        signals: list[TradeSignal] = []

        # Exit signals for positions no longer in top N
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.MEDIUM,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning="Dropped from top Sharpe-momentum ranking",
            )
            for sym in portfolio.positions
            if sym not in top_symbols
        )

        # Entry signals for top assets not yet held
        new_pos = 0
        weight = self.config.target_position_weight
        if crash_hedge_active:
            weight *= 0.5  # Halve position sizes during crash hedge

        for sym, sharpe_val, decay_info in top_assets:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break

            close = prices.get(sym, 0.0)
            if close <= 0:
                continue

            sl = self._compute_stop_loss(indicators_df, sym, close, stop_mult)
            hedge_note = " [crash hedge: half-size]" if crash_hedge_active else ""
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=weight,
                    stop_loss=sl,
                    reasoning=(
                        f"Sharpe momentum rank (sharpe={sharpe_val:.2f}, "
                        f"{decay_info}){hedge_note}"
                    ),
                )
            )
            new_pos += 1

        return signals

    # ------------------------------------------------------------------
    # Sharpe momentum ranking with decay + SMA_200 filters
    # ------------------------------------------------------------------

    def _rank_universe(
        self,
        indicators_df: pl.DataFrame,
        sharpe_lookback: int,
        decay_short: int,
        decay_long: int,
    ) -> list[tuple[str, float, str]]:
        """Rank universe assets by Sharpe momentum, applying filters.

        Returns a list of (symbol, sharpe_value, decay_info) sorted
        descending by Sharpe momentum.  Only assets passing ALL filters
        are included.
        """
        symbols = indicators_df.select("symbol").unique().to_series().to_list()
        candidates: list[tuple[str, float, str]] = []

        for symbol in symbols:
            sym_df = _sym_series(indicators_df, symbol)
            if len(sym_df) < decay_long + 1:
                continue

            closes = _close_series(sym_df)
            if len(closes) < decay_long + 1:
                continue

            # -- SMA_200 trend filter --
            if not self._passes_sma200_filter(sym_df):
                continue

            # -- Sharpe momentum: 63d return / 63d std --
            sharpe_val = self._compute_sharpe_momentum(closes, sharpe_lookback)
            if sharpe_val is None or math.isnan(sharpe_val):
                continue

            # -- Decay filter: 63d return > 126d return (accelerating) --
            ret_short = closes[-1] / closes[-decay_short - 1] - 1.0
            ret_long = closes[-1] / closes[-decay_long - 1] - 1.0

            if ret_short <= ret_long:
                # Momentum is decelerating -- filter out
                continue

            decay_info = f"63d={ret_short:.2%}>126d={ret_long:.2%}"
            candidates.append((symbol, sharpe_val, decay_info))

        # Sort by Sharpe momentum descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _compute_sharpe_momentum(
        self,
        closes: list[float],
        lookback: int,
    ) -> float | None:
        """Compute risk-adjusted momentum: 63d return / 63d std.

        Uses the trailing ``lookback`` daily returns to compute both
        the return and the standard deviation.
        """
        if len(closes) < lookback + 1:
            return None

        window_closes = closes[-(lookback + 1) :]
        rets = _daily_returns(window_closes)

        if len(rets) < lookback:
            return None

        # Total return over the window
        total_return = window_closes[-1] / window_closes[0] - 1.0

        # Standard deviation of daily returns over the window
        mean_ret = sum(rets) / len(rets)
        variance = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
        std = math.sqrt(variance)

        if std == 0:
            return None

        # Annualize: return * (252/lookback) / (std * sqrt(252))
        # Simplifies to: total_return / (std * sqrt(lookback))
        annualized_return = total_return * (252.0 / lookback)
        annualized_vol = std * math.sqrt(252.0)

        return annualized_return / annualized_vol

    def _passes_sma200_filter(self, sym_df: pl.DataFrame) -> bool:
        """Check if the latest close is above SMA_200."""
        if "sma_200" not in sym_df.columns:
            # If SMA_200 not available, pass the filter (don't penalise)
            return True

        last_row = sym_df.tail(1).row(0, named=True)
        close = last_row.get("close")
        sma_200 = last_row.get("sma_200")

        if close is None or sma_200 is None:
            return True  # Insufficient data, don't filter

        return close > sma_200

    # ------------------------------------------------------------------
    # Dispersion crash hedge
    # ------------------------------------------------------------------

    def _is_crash_hedge_active(
        self,
        indicators_df: pl.DataFrame,
        dispersion_lookback: int,
        dispersion_history: int,
        threshold_pct: float,
    ) -> bool:
        """Check if cross-sectional sector dispersion is below threshold.

        Computes cross-sectional std of sector 63d returns.  If the
        current dispersion is below the ``threshold_pct`` percentile of
        its trailing 252-day history, the crash hedge is active.
        """
        # We need enough history for the dispersion series
        # Each dispersion point requires dispersion_lookback days of data,
        # and we need dispersion_history points of dispersion data.

        # Get all available dates
        all_dates = (
            indicators_df.select("date").unique().sort("date").to_series().to_list()
        )

        if len(all_dates) < dispersion_lookback + dispersion_history:
            return False  # Not enough history

        # Compute 63d return for each sector ETF at each date
        # Build a dict: symbol -> {date -> close}
        sector_close_maps: dict[str, dict[date, float]] = {}
        for etf in self.SECTOR_ETFS:
            sym_df = _sym_series(indicators_df, etf)
            if len(sym_df) == 0:
                continue
            col = "adj_close" if "adj_close" in sym_df.columns else "close"
            date_close: dict[date, float] = {}
            for row in sym_df.iter_rows(named=True):
                date_close[row["date"]] = row[col]
            sector_close_maps[etf] = date_close

        if len(sector_close_maps) < 3:
            return False  # Need at least a few sectors

        # For the trailing dispersion_history dates, compute cross-sectional
        # dispersion of 63d returns
        # We evaluate at each date in the trailing window
        eval_dates = all_dates[-(dispersion_history):]
        dispersion_series: list[float] = []

        for eval_date in eval_dates:
            # Find the date index
            date_idx = all_dates.index(eval_date)
            if date_idx < dispersion_lookback:
                continue

            lookback_date = all_dates[date_idx - dispersion_lookback]

            # Compute 63d return for each sector
            sector_returns: list[float] = []
            for close_map in sector_close_maps.values():
                curr_close = close_map.get(eval_date)
                prev_close = close_map.get(lookback_date)
                if curr_close is not None and prev_close is not None and prev_close > 0:
                    sector_returns.append(curr_close / prev_close - 1.0)

            if len(sector_returns) < 3:
                continue

            # Cross-sectional standard deviation
            mean_ret = sum(sector_returns) / len(sector_returns)
            variance = sum((r - mean_ret) ** 2 for r in sector_returns) / len(
                sector_returns
            )
            dispersion_series.append(math.sqrt(variance))

        if len(dispersion_series) < 10:
            return False  # Not enough dispersion history

        # Current dispersion is the last value
        current_dispersion = dispersion_series[-1]

        # Percentile rank of current dispersion within the series
        count_below = sum(1 for d in dispersion_series if d < current_dispersion)
        percentile = (count_below / len(dispersion_series)) * 100.0

        logger.debug(
            "Dispersion percentile: %.1f (threshold: %.1f, current: %.4f)",
            percentile,
            threshold_pct,
            current_dispersion,
        )

        return percentile < threshold_pct

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
    "momentum_evolution": MomentumEvolutionStrategy,
}

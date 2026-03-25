"""Behavioral-Structural strategy implementation.

Implements three orthogonal signals -- low-volatility anomaly in sector ETFs
(H04), passive flow concentration rotation (H02), and TLT anchoring bias
mean-reversion (H12).  All computations are causal (backward-looking only).
See data/strategies/behavioral-structural/research-spec.yaml for the frozen
research design.
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
# 11 GICS sector ETFs
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

# Broad equity ETFs for correlation computation
BROAD_EQUITY_ETFS: list[str] = ["SPY", "QQQ", "IWM"]

# Alternative assets for rotation
ROTATION_TARGETS: list[str] = ["GLD", "TLT"]


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
# Behavioral-Structural Strategy
# ---------------------------------------------------------------------------


class BehavioralStructuralStrategy(Strategy):
    """Low-vol anomaly + passive flow rotation + bond mean-reversion.

    Signal hierarchy:
      1. TLT anchoring (event-driven, checked every bar):
         z-score < -2.0 -> buy TLT for mean-reversion
         z-score > -0.5 -> exit TLT mean-reversion position
      2. Passive flow concentration (regime overlay):
         SPY/QQQ/IWM avg 30d correlation > 90th percentile -> rotate
         to GLD/TLT; exit rotation at < 75th percentile (hysteresis)
      3. Low-vol anomaly (core allocation):
         Rank 11 sector ETFs by 63d realized vol, long bottom 3

    Required symbols: All 11 sector ETFs, SPY, QQQ, IWM, GLD, TLT.
    """

    SECTOR_ETFS: ClassVar[list[str]] = SECTOR_ETFS
    BROAD_EQUITY_ETFS: ClassVar[list[str]] = BROAD_EQUITY_ETFS
    ROTATION_TARGETS: ClassVar[list[str]] = ROTATION_TARGETS

    # Internal state for correlation hysteresis.
    # In a backtest, the engine re-creates the strategy each run, so we
    # track the rotation state via portfolio positions as a proxy:
    # if we hold GLD or TLT from a rotation signal, rotation is active.

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters

        # Parameter extraction with defaults from research spec
        vol_lookback = params.get("vol_ranking_lookback", 63)
        low_vol_count = params.get("low_vol_count", 3)
        corr_lookback = params.get("corr_lookback", 30)
        corr_threshold_pct = params.get("corr_threshold_pct", 90)
        corr_exit_pct = params.get("corr_exit_pct", 75)
        zscore_lookback = params.get("zscore_lookback", 63)
        zscore_entry = params.get("zscore_entry_threshold", 2.0)
        stop_pct = params.get("stop_loss_pct", self.config.stop_loss_pct)

        signals: list[TradeSignal] = []

        # ---------------------------------------------------------------
        # Signal 1 (event-driven): TLT anchoring mean-reversion
        # ---------------------------------------------------------------
        tlt_signals = self._tlt_anchoring_signals(
            indicators_df, portfolio, prices, zscore_lookback, zscore_entry, stop_pct
        )
        signals.extend(tlt_signals)

        # ---------------------------------------------------------------
        # Signal 2 (regime): Passive flow concentration rotation
        # ---------------------------------------------------------------
        rotation_active = self._is_rotation_active(
            indicators_df,
            corr_lookback,
            corr_threshold_pct,
            corr_exit_pct,
            portfolio,
        )

        if rotation_active:
            # Close equity sector positions, rotate to alternatives
            rotation_signals = self._rotation_signals(
                indicators_df, portfolio, prices, stop_pct
            )
            signals.extend(rotation_signals)
            logger.info("Passive flow rotation ACTIVE: rotating to GLD/TLT")
        else:
            # ---------------------------------------------------------------
            # Signal 3 (core): Low-vol anomaly sector selection
            # ---------------------------------------------------------------
            low_vol_signals = self._low_vol_signals(
                indicators_df,
                portfolio,
                prices,
                vol_lookback,
                low_vol_count,
                stop_pct,
            )
            signals.extend(low_vol_signals)

        return signals

    # ------------------------------------------------------------------
    # Signal 1: TLT anchoring mean-reversion (H12)
    # ------------------------------------------------------------------

    def _tlt_anchoring_signals(
        self,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        _prices: dict[str, float],
        zscore_lookback: int,
        zscore_entry: float,
        stop_pct: float,
    ) -> list[TradeSignal]:
        """Generate TLT mean-reversion signals based on z-score.

        Buy when z < -2.0 (oversold), exit when z > -0.5 (reverted).
        """
        signals: list[TradeSignal] = []
        tlt_df = _sym_series(indicators_df, "TLT")
        if len(tlt_df) < zscore_lookback + 1:
            return signals

        closes = _close_series(tlt_df)
        if len(closes) < zscore_lookback + 1:
            return signals

        # Compute 63-day z-score
        window = closes[-zscore_lookback:]
        mean_price = sum(window) / len(window)
        variance = sum((c - mean_price) ** 2 for c in window) / len(window)
        std_price = math.sqrt(variance)

        if std_price == 0:
            return signals

        current_close = closes[-1]
        z = (current_close - mean_price) / std_price

        has_tlt = "TLT" in portfolio.positions

        # Buy when z below negative entry threshold (oversold)
        if z < -zscore_entry and not has_tlt:
            sl = current_close * (1.0 - stop_pct)
            signals.append(
                TradeSignal(
                    symbol="TLT",
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=(
                        f"TLT anchoring mean-reversion: z={z:.2f} < "
                        f"-{zscore_entry:.1f} (oversold)"
                    ),
                )
            )

        # Exit: z > -0.5 (reverted toward mean)
        elif z > -0.5 and has_tlt:
            signals.append(
                TradeSignal(
                    symbol="TLT",
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        f"TLT anchoring exit: z={z:.2f} > -0.5 (reverted to mean)"
                    ),
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Signal 2: Passive flow concentration rotation (H02)
    # ------------------------------------------------------------------

    def _is_rotation_active(
        self,
        indicators_df: pl.DataFrame,
        corr_lookback: int,
        corr_threshold_pct: float,
        corr_exit_pct: float,
        portfolio: Portfolio,
    ) -> bool:
        """Determine if passive flow concentration rotation is active.

        Uses hysteresis: enter rotation at 90th percentile, exit at 75th.
        """
        avg_corr = self._compute_broad_equity_correlation(indicators_df, corr_lookback)
        if avg_corr is None:
            return False

        # Compute correlation percentile against trailing 252-day history
        corr_percentile = self._compute_corr_percentile(
            indicators_df, corr_lookback, avg_corr
        )
        if corr_percentile is None:
            return False

        # Hysteresis: check if currently in rotation mode (holding GLD/TLT
        # from rotation) -- if so, stay in rotation until below exit threshold
        currently_rotated = any(
            sym in portfolio.positions for sym in self.ROTATION_TARGETS
        )

        if currently_rotated:
            # Exit rotation only when below corr_exit_pct
            return corr_percentile >= corr_exit_pct
        # Enter rotation when above corr_threshold_pct
        return corr_percentile >= corr_threshold_pct

    def _compute_broad_equity_correlation(
        self,
        indicators_df: pl.DataFrame,
        corr_lookback: int,
    ) -> float | None:
        """Compute average pairwise 30-day correlation among SPY/QQQ/IWM.

        Returns the average of the three pairwise correlations.
        """
        # Build daily return series for each broad equity ETF
        return_series: dict[str, list[tuple[date, float]]] = {}

        for sym in self.BROAD_EQUITY_ETFS:
            sym_df = _sym_series(indicators_df, sym)
            closes = _close_series(sym_df)
            if len(closes) < corr_lookback + 1:
                return None

            dates = sym_df.select("date").to_series().to_list()
            rets = _daily_returns(closes)
            # Align: rets[i] corresponds to dates[i+1]
            dated_rets = list(zip(dates[1:], rets, strict=False))
            return_series[sym] = dated_rets[-corr_lookback:]

        if len(return_series) < 3:
            return None

        # Compute pairwise correlations
        pairs = [
            ("SPY", "QQQ"),
            ("SPY", "IWM"),
            ("QQQ", "IWM"),
        ]

        correlations: list[float] = []
        for sym_a, sym_b in pairs:
            rets_a = return_series.get(sym_a)
            rets_b = return_series.get(sym_b)
            if rets_a is None or rets_b is None:
                continue
            if len(rets_a) != len(rets_b):
                # Align by date
                dates_a = {d for d, _ in rets_a}
                dates_b = {d for d, _ in rets_b}
                common_dates = sorted(dates_a & dates_b)
                vals_a = dict(rets_a)
                vals_b = dict(rets_b)
                a_vals = [vals_a[d] for d in common_dates]
                b_vals = [vals_b[d] for d in common_dates]
            else:
                a_vals = [r for _, r in rets_a]
                b_vals = [r for _, r in rets_b]

            corr = self._pearson_correlation(a_vals, b_vals)
            if corr is not None:
                correlations.append(corr)

        if len(correlations) == 0:
            return None

        return sum(correlations) / len(correlations)

    def _compute_corr_percentile(
        self,
        indicators_df: pl.DataFrame,
        corr_lookback: int,
        current_corr: float,
    ) -> float | None:
        """Compute the percentile rank of current correlation in trailing history.

        Uses a rolling approach: compute the average pairwise correlation for
        each trailing 30-day window over the last 252 days and rank the current
        value against that history.
        """
        all_dates = (
            indicators_df.select("date").unique().sort("date").to_series().to_list()
        )

        history_window = 252
        if len(all_dates) < corr_lookback + history_window:
            return None

        close_maps = self._build_broad_equity_close_maps(indicators_df)
        if len(close_maps) < 3:
            return None

        eval_dates = all_dates[-history_window:]
        corr_history = self._rolling_corr_history(
            all_dates, eval_dates, close_maps, corr_lookback
        )

        if len(corr_history) < 10:
            return None

        count_below = sum(1 for c in corr_history if c < current_corr)
        return (count_below / len(corr_history)) * 100.0

    def _build_broad_equity_close_maps(
        self,
        indicators_df: pl.DataFrame,
    ) -> dict[str, dict[date, float]]:
        """Build date-indexed close price maps for broad equity ETFs."""
        close_maps: dict[str, dict[date, float]] = {}
        for sym in self.BROAD_EQUITY_ETFS:
            sym_df = _sym_series(indicators_df, sym)
            if len(sym_df) == 0:
                continue
            col = "adj_close" if "adj_close" in sym_df.columns else "close"
            close_maps[sym] = {
                row["date"]: row[col] for row in sym_df.iter_rows(named=True)
            }
        return close_maps

    def _rolling_corr_history(
        self,
        all_dates: list[date],
        eval_dates: list[date],
        close_maps: dict[str, dict[date, float]],
        corr_lookback: int,
    ) -> list[float]:
        """Compute rolling pairwise correlation for each evaluation date."""
        corr_history: list[float] = []
        pairs = [("SPY", "QQQ"), ("SPY", "IWM"), ("QQQ", "IWM")]

        for eval_date in eval_dates:
            date_idx = all_dates.index(eval_date)
            if date_idx < corr_lookback + 1:
                continue

            window_dates = all_dates[date_idx - corr_lookback : date_idx + 1]
            window_returns = self._window_returns(close_maps, window_dates)
            if len(window_returns) < 3:
                continue

            corrs: list[float] = []
            for sym_a, sym_b in pairs:
                ra = window_returns.get(sym_a)
                rb = window_returns.get(sym_b)
                if ra is not None and rb is not None and len(ra) == len(rb):
                    c = self._pearson_correlation(ra, rb)
                    if c is not None:
                        corrs.append(c)

            if len(corrs) > 0:
                corr_history.append(sum(corrs) / len(corrs))

        return corr_history

    def _window_returns(
        self,
        close_maps: dict[str, dict[date, float]],
        window_dates: list[date],
    ) -> dict[str, list[float]]:
        """Build daily return series for each broad equity ETF in a window."""
        window_returns: dict[str, list[float]] = {}
        for sym in self.BROAD_EQUITY_ETFS:
            cm = close_maps.get(sym, {})
            closes_in_window = [cm.get(d) for d in window_dates]
            if any(c is None for c in closes_in_window):
                continue
            rets = _daily_returns(closes_in_window)  # type: ignore[arg-type]
            window_returns[sym] = rets
        return window_returns

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation between two series."""
        n = len(x)
        if n < 3 or len(y) != n:
            return None

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        var_x = sum((xi - mean_x) ** 2 for xi in x) / n
        var_y = sum((yi - mean_y) ** 2 for yi in y) / n

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return None

        return cov / denom

    def _rotation_signals(
        self,
        _indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        stop_pct: float,
    ) -> list[TradeSignal]:
        """Close equity sectors, enter GLD/TLT for rotation."""
        signals: list[TradeSignal] = []

        # Close sector ETF positions
        sector_set = set(self.SECTOR_ETFS)
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.HIGH,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=(
                    "Passive flow rotation: high broad equity correlation, exit sectors"
                ),
            )
            for sym in portfolio.positions
            if sym in sector_set
        )

        # Enter rotation targets if not already held
        new_pos = 0
        for sym in self.ROTATION_TARGETS:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break

            close = prices.get(sym, 0.0)
            if close <= 0:
                continue

            sl = close * (1.0 - stop_pct)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.HIGH,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=(
                        "Passive flow rotation: SPY/QQQ/IWM correlation "
                        "> 90th pct, rotate to alternatives"
                    ),
                )
            )
            new_pos += 1

        return signals

    # ------------------------------------------------------------------
    # Signal 3: Low-vol anomaly (H04)
    # ------------------------------------------------------------------

    def _low_vol_signals(
        self,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
        vol_lookback: int,
        low_vol_count: int,
        stop_pct: float,
    ) -> list[TradeSignal]:
        """Rank sector ETFs by realized vol, long the lowest-vol sectors."""
        signals: list[TradeSignal] = []

        # Compute 63-day realized vol for each sector ETF
        vol_rankings: list[tuple[str, float]] = []
        for etf in self.SECTOR_ETFS:
            vol = self._compute_realized_vol(indicators_df, etf, vol_lookback)
            if vol is not None:
                vol_rankings.append((etf, vol))

        if len(vol_rankings) < low_vol_count:
            return signals

        # Sort ascending by vol (lowest vol first)
        vol_rankings.sort(key=lambda x: x[1])
        low_vol_set = {sym for sym, _ in vol_rankings[:low_vol_count]}

        # Exit positions no longer in low-vol set
        sector_set = set(self.SECTOR_ETFS)
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.MEDIUM,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=(
                    f"Low-vol exit: {sym} vol ranking moved "
                    f"out of bottom {low_vol_count}"
                ),
            )
            for sym in portfolio.positions
            if sym in sector_set and sym not in low_vol_set
        )

        # Enter low-vol sectors not yet held
        new_pos = 0
        for sym, vol in vol_rankings[:low_vol_count]:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break

            close = prices.get(sym, 0.0)
            if close <= 0:
                continue

            sl = close * (1.0 - stop_pct)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=(
                        f"Low-vol anomaly: {sym} ranked in bottom "
                        f"{low_vol_count} by 63d vol ({vol:.2%} annualized)"
                    ),
                )
            )
            new_pos += 1

        return signals

    def _compute_realized_vol(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        lookback: int,
    ) -> float | None:
        """Compute annualized realized volatility from daily returns."""
        sym_df = _sym_series(indicators_df, symbol)
        closes = _close_series(sym_df)

        if len(closes) < lookback + 1:
            return None

        rets = _daily_returns(closes[-(lookback + 1) :])
        if len(rets) < lookback:
            return None

        mean_ret = sum(rets) / len(rets)
        variance = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
        daily_std = math.sqrt(variance)

        # Annualize
        return daily_std * math.sqrt(252.0)


# ---------------------------------------------------------------------------
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "behavioral_structural": BehavioralStructuralStrategy,
}

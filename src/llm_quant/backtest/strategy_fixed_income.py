"""Fixed Income Macro strategy implementation.

Implements three orthogonal macro signals — yield curve slope (FM-001),
credit spread regime (FM-005), and breakeven inflation momentum (FM-011) —
with a VIX defensive override.  All computations are causal (backward-looking
only).  See data/strategies/fixed-income-macro/research-spec.yaml for the
frozen research design.
"""

from __future__ import annotations

import logging
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
# Helper: compute price-ratio indicators inline
# ---------------------------------------------------------------------------


def _price_ratio_series(
    indicators_df: pl.DataFrame,
    numerator_symbol: str,
    denominator_symbol: str,
) -> list[tuple[date, float]]:
    """Compute a price ratio time-series from two symbols.

    Returns a date-sorted list of (date, ratio) tuples.  Only dates
    present in *both* symbols are included.
    """
    num = _sym_series(indicators_df, numerator_symbol)
    den = _sym_series(indicators_df, denominator_symbol)
    if len(num) == 0 or len(den) == 0:
        return []

    close_col = "adj_close" if "adj_close" in num.columns else "close"

    # Join on date to align
    num_sel = num.select(pl.col("date"), pl.col(close_col).alias("num_close"))
    den_sel = den.select(pl.col("date"), pl.col(close_col).alias("den_close"))
    joined = num_sel.join(den_sel, on="date", how="inner").sort("date")

    result: list[tuple[date, float]] = []
    for row in joined.iter_rows(named=True):
        den_val = row["den_close"]
        if den_val is not None and den_val > 0:
            result.append((row["date"], row["num_close"] / den_val))
    return result


def _rolling_mean(values: list[float], window: int) -> float | None:
    """Simple trailing mean of the last *window* values."""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _rate_of_change(values: list[float], window: int) -> float | None:
    """Rate of change: (current / value_window_ago) - 1."""
    if len(values) <= window:
        return None
    old = values[-(window + 1)]
    if old == 0:
        return None
    return values[-1] / old - 1.0


def _zscore(
    values: list[float],
    roc_window: int,
    zscore_lookback: int,
) -> float | None:
    """Z-score the trailing rate-of-change series.

    Computes the RoC for each point in the last *zscore_lookback*
    points, then returns the z-score of the most recent RoC against
    that distribution.
    """
    if len(values) < zscore_lookback + roc_window + 1:
        return None

    roc_series: list[float] = []
    for i in range(zscore_lookback):
        idx = len(values) - zscore_lookback + i
        old_idx = idx - roc_window
        if old_idx < 0 or values[old_idx] == 0:
            continue
        roc_series.append(values[idx] / values[old_idx] - 1.0)

    if len(roc_series) < 2:
        return None

    mean = sum(roc_series) / len(roc_series)
    variance = sum((v - mean) ** 2 for v in roc_series)
    std = (variance / len(roc_series)) ** 0.5
    if std == 0:
        return None
    return (roc_series[-1] - mean) / std


def _momentum(values: list[float], window: int) -> float | None:
    """Momentum as rate-of-change over *window* periods."""
    return _rate_of_change(values, window)


# ---------------------------------------------------------------------------
# Fixed Income Macro Strategy
# ---------------------------------------------------------------------------


class FixedIncomeMacroStrategy(Strategy):
    """Fixed-income macro: yield curve + credit + inflation.

    Hierarchical signal priority:
      1. VIX defensive override (VIX > 30)
      2. Credit stress (HYG/LQD RoC z-score < -2.0)
      3. Normal: curve slope duration + inflation momentum

    Required symbols: IEF, SHY, TLT, HYG, LQD, TIP, GLD, XLE, VIX.
    """

    # Symbols this strategy may trade
    DURATION_LONG: ClassVar[list[str]] = ["TLT", "IEF"]
    DURATION_SHORT: ClassVar[list[str]] = ["SHY"]
    INFLATION_LONG: ClassVar[list[str]] = ["TIP", "GLD", "XLE"]
    DEFLATION_LONG: ClassVar[list[str]] = ["TLT"]
    DEFENSIVE: ClassVar[list[str]] = ["SHY", "GLD"]
    CREDIT_RISK_OFF_CLOSE: ClassVar[list[str]] = ["HYG"]
    VIX_SYMBOL: ClassVar[str] = "VIX"

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters

        # Parameter extraction with defaults from research spec
        curve_slope_lookback = params.get("curve_slope_lookback", 50)
        credit_roc_window = params.get("credit_roc_window", 20)
        credit_zscore_lookback = params.get("credit_zscore_lookback", 252)
        credit_zscore_threshold = params.get("credit_zscore_threshold", -2.0)
        breakeven_momentum_window = params.get("breakeven_momentum_window", 21)
        vix_risk_off_threshold = params.get("vix_risk_off_threshold", 30)
        stop_mult = params.get("stop_atr_multiplier", 2.0)

        # Compute indicators
        vix_level = self._get_vix_level(indicators_df)
        curve_signal = self._compute_curve_signal(indicators_df, curve_slope_lookback)
        credit_z = self._compute_credit_zscore(
            indicators_df, credit_roc_window, credit_zscore_lookback
        )
        infl_mom = self._compute_inflation_momentum(
            indicators_df, breakeven_momentum_window
        )

        # Priority 1: VIX defensive override
        if vix_level is not None and vix_level > vix_risk_off_threshold:
            reason = (
                f"VIX defensive override ({vix_level:.1f} > {vix_risk_off_threshold})"
            )
            return self._defensive_signals(
                portfolio, prices, indicators_df, stop_mult, reason
            )

        # Priority 2: Credit stress
        if credit_z is not None and credit_z < credit_zscore_threshold:
            return self._credit_stress_signals(
                portfolio,
                prices,
                indicators_df,
                stop_mult,
                credit_z,
                credit_zscore_threshold,
            )

        # Priority 3: Normal operation
        return self._normal_signals(
            curve_signal,
            infl_mom,
            portfolio,
            prices,
            indicators_df,
            stop_mult,
        )

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def _get_vix_level(self, indicators_df: pl.DataFrame) -> float | None:
        """Get the latest VIX close level."""
        vix_df = _sym_series(indicators_df, self.VIX_SYMBOL)
        return _latest_close(vix_df)

    # ------------------------------------------------------------------
    # Signal 1: Yield curve slope (FM-001)
    # ------------------------------------------------------------------

    def _compute_curve_signal(
        self,
        indicators_df: pl.DataFrame,
        lookback: int,
    ) -> str | None:
        """Compute yield curve slope signal.

        Returns "steepening" or "flattening", or None.
        """
        ratio_ts = _price_ratio_series(indicators_df, "IEF", "SHY")
        if len(ratio_ts) < lookback:
            return None

        ratio_values = [r for _, r in ratio_ts]
        current_ratio = ratio_values[-1]
        sma = _rolling_mean(ratio_values, lookback)
        if sma is None:
            return None

        if current_ratio < sma:
            return "steepening"
        return "flattening"

    # ------------------------------------------------------------------
    # Signal 2: Credit spread regime (FM-005)
    # ------------------------------------------------------------------

    def _compute_credit_zscore(
        self,
        indicators_df: pl.DataFrame,
        roc_window: int,
        zscore_lookback: int,
    ) -> float | None:
        """Z-score of HYG/LQD 20-day rate of change."""
        ratio_ts = _price_ratio_series(indicators_df, "HYG", "LQD")
        if len(ratio_ts) == 0:
            return None
        ratio_values = [r for _, r in ratio_ts]
        return _zscore(ratio_values, roc_window, zscore_lookback)

    # ------------------------------------------------------------------
    # Signal 3: Breakeven inflation momentum (FM-011)
    # ------------------------------------------------------------------

    def _compute_inflation_momentum(
        self,
        indicators_df: pl.DataFrame,
        momentum_window: int,
    ) -> float | None:
        """21-day momentum of TIP/IEF price ratio."""
        ratio_ts = _price_ratio_series(indicators_df, "TIP", "IEF")
        if len(ratio_ts) == 0:
            return None
        ratio_values = [r for _, r in ratio_ts]
        return _momentum(ratio_values, momentum_window)

    # ------------------------------------------------------------------
    # Defensive override (VIX > 30)
    # ------------------------------------------------------------------

    def _defensive_signals(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        reason: str,
    ) -> list[TradeSignal]:
        """Max-defense: close non-defensive, open defensive."""
        signals: list[TradeSignal] = []
        defensive_set = set(self.DEFENSIVE)

        # Close non-defensive positions
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.HIGH,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Exit non-defensive: {reason}",
            )
            for sym in portfolio.positions
            if sym not in defensive_set
        )

        # Open defensive positions if not held
        new_pos = 0
        for sym in self.DEFENSIVE:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break
            close = prices.get(sym, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(indicators_df, sym, close, stop_mult)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.HIGH,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=f"Defensive allocation: {reason}",
                )
            )
            new_pos += 1

        return signals

    # ------------------------------------------------------------------
    # Credit stress (z < -2.0)
    # ------------------------------------------------------------------

    def _credit_stress_signals(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        credit_z: float,
        threshold: float,
    ) -> list[TradeSignal]:
        """Risk-off signals on credit stress."""
        signals: list[TradeSignal] = []
        reason = f"Credit stress: HYG/LQD RoC z={credit_z:.2f} < {threshold}"

        # Close credit-sensitive positions
        risk_on = {"HYG", "SPY", "IWM", "XLF", "QQQ"}
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.HIGH,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Exit risk-on: {reason}",
            )
            for sym in portfolio.positions
            if sym in risk_on
        )

        # Open safe havens
        safe_havens = ["SHY", "GLD", "TLT"]
        new_pos = 0
        for sym in safe_havens:
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break
            close = prices.get(sym, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(indicators_df, sym, close, stop_mult)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.HIGH,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=f"Safe haven: {reason}",
                )
            )
            new_pos += 1

        return signals

    # ------------------------------------------------------------------
    # Normal operation (curve + inflation)
    # ------------------------------------------------------------------

    def _normal_signals(
        self,
        curve_signal: str | None,
        infl_mom: float | None,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
    ) -> list[TradeSignal]:
        """Signals from curve slope + inflation momentum."""
        signals: list[TradeSignal] = []
        desired: dict[str, str] = {}

        # Build desired set and exit signals
        self._apply_curve_signal(curve_signal, portfolio, desired, signals)
        self._apply_inflation_signal(infl_mom, portfolio, desired, signals)

        # Deferred duration exits (after inflation check)
        if curve_signal == "flattening":
            signals.extend(
                TradeSignal(
                    symbol=sym,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning="Curve flattening: exit long duration",
                )
                for sym in self.DURATION_LONG
                if sym in portfolio.positions and sym not in desired
            )

        # Buy desired symbols not yet held
        new_pos = 0
        for sym, reasoning in desired.items():
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break
            close = prices.get(sym, 0)
            if close <= 0:
                continue
            sl = self._compute_stop_loss(indicators_df, sym, close, stop_mult)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=self.config.target_position_weight,
                    stop_loss=sl,
                    reasoning=reasoning,
                )
            )
            new_pos += 1

        return signals

    def _apply_curve_signal(
        self,
        curve_signal: str | None,
        portfolio: Portfolio,
        desired: dict[str, str],
        signals: list[TradeSignal],
    ) -> None:
        """Add curve-driven entries/exits."""
        if curve_signal == "steepening":
            for sym in self.DURATION_LONG:
                desired[sym] = "Curve steepening (IEF/SHY < SMA): extend duration"
            signals.extend(
                TradeSignal(
                    symbol=sym,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning="Curve steepening: exit short duration (SHY)",
                )
                for sym in self.DURATION_SHORT
                if sym in portfolio.positions
            )
        elif curve_signal == "flattening":
            for sym in self.DURATION_SHORT:
                desired[sym] = "Curve flattening (IEF/SHY > SMA): shorten duration"

    def _apply_inflation_signal(
        self,
        infl_mom: float | None,
        portfolio: Portfolio,
        desired: dict[str, str],
        signals: list[TradeSignal],
    ) -> None:
        """Add inflation-driven entries/exits."""
        if infl_mom is not None and infl_mom > 0:
            for sym in self.INFLATION_LONG:
                reason = (
                    f"Rising inflation "
                    f"(TIP/IEF mom={infl_mom:.4f}): "
                    f"inflation beneficiary"
                )
                if sym in desired:
                    desired[sym] += f"; {reason}"
                else:
                    desired[sym] = reason

        elif infl_mom is not None and infl_mom < 0:
            for sym in self.DEFLATION_LONG:
                reason = (
                    f"Falling inflation "
                    f"(TIP/IEF mom={infl_mom:.4f}): "
                    f"nominal bond beneficiary"
                )
                if sym in desired:
                    desired[sym] += f"; {reason}"
                else:
                    desired[sym] = reason
            # Exit inflation beneficiaries
            signals.extend(
                TradeSignal(
                    symbol=sym,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        f"Falling inflation "
                        f"(TIP/IEF mom={infl_mom:.4f})"
                        ": exit inflation tilt"
                    ),
                )
                for sym in self.INFLATION_LONG
                if sym in portfolio.positions and sym not in desired
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
    "fixed_income_macro": FixedIncomeMacroStrategy,
}

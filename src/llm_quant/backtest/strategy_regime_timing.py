"""Regime Timing strategy implementation.

Implements a 4-regime classification (risk_on, risk_off, transition,
inflationary) via a 5-signal majority vote with a SPY-TLT correlation
overlay.  All computations are causal (backward-looking only).
See data/strategies/regime-timing/research-spec.yaml for the frozen
research design.

Hypotheses implemented:
  RT-H10  Multi-Signal Regime Composite Ensemble (5 votes)
  RT-H02  Yield Curve + Credit Spread classification (2 of the 5 votes)
  RT-H03  SPY-TLT Correlation Regime Switch (inflation overlay)
"""

from __future__ import annotations

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
# Helpers (same pattern as strategy_fixed_income.py)
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

    num_sel = num.select(
        pl.col("date"),
        pl.col(close_col).alias("num_close"),
    )
    den_sel = den.select(
        pl.col("date"),
        pl.col(close_col).alias("den_close"),
    )
    joined = num_sel.join(
        den_sel,
        on="date",
        how="inner",
    ).sort("date")

    result: list[tuple[date, float]] = []
    for row in joined.iter_rows(named=True):
        den_val = row["den_close"]
        if den_val is not None and den_val > 0:
            result.append((row["date"], row["num_close"] / den_val))
    return result


def _rolling_mean(
    values: list[float],
    window: int,
) -> float | None:
    """Simple trailing mean of the last *window* values."""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _daily_returns(
    indicators_df: pl.DataFrame,
    symbol: str,
) -> list[tuple[date, float]]:
    """Compute daily returns for *symbol* from close prices.

    Returns a date-sorted list of (date, return) tuples.
    """
    sym = _sym_series(indicators_df, symbol)
    if len(sym) < 2:
        return []

    close_col = "adj_close" if "adj_close" in sym.columns else "close"
    closes = sym.select("date", close_col).to_dicts()
    result: list[tuple[date, float]] = []
    for i in range(1, len(closes)):
        prev_c = closes[i - 1][close_col]
        curr_c = closes[i][close_col]
        if prev_c is not None and curr_c is not None and prev_c > 0:
            result.append(
                (closes[i]["date"], curr_c / prev_c - 1.0),
            )
    return result


def _rolling_correlation(
    returns_a: list[tuple[date, float]],
    returns_b: list[tuple[date, float]],
    window: int,
) -> float | None:
    """Trailing Pearson correlation over the last *window* aligned returns.

    Aligns by date, then computes correlation over the last *window*
    overlapping observations.  Returns None if insufficient data.
    """
    dict_a = dict(returns_a)
    dict_b = dict(returns_b)
    common_dates = sorted(
        set(dict_a.keys()) & set(dict_b.keys()),
    )

    if len(common_dates) < window:
        return None

    tail_dates = common_dates[-window:]
    xs = [dict_a[d] for d in tail_dates]
    ys = [dict_b[d] for d in tail_dates]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return cov / denom


def _cumulative_spread_return(
    indicators_df: pl.DataFrame,
    symbol_a: str,
    symbol_b: str,
    window: int,
) -> float | None:
    """Cumulative return spread between two symbols over *window* days.

    result = sum(daily_return_A - daily_return_B) over the last
    *window* days.
    """
    rets_a = _daily_returns(indicators_df, symbol_a)
    rets_b = _daily_returns(indicators_df, symbol_b)
    if not rets_a or not rets_b:
        return None

    dict_a = dict(rets_a)
    dict_b = dict(rets_b)
    common_dates = sorted(
        set(dict_a.keys()) & set(dict_b.keys()),
    )

    if len(common_dates) < window:
        return None

    tail = common_dates[-window:]
    return sum(dict_a[d] - dict_b[d] for d in tail)


# ---------------------------------------------------------------------------
# Data class to bundle vote context (avoids long arg lists)
# ---------------------------------------------------------------------------


@dataclass
class _RegimeContext:
    """Bundle of regime classification results."""

    regime: str
    risk_on_votes: int
    risk_off_votes: int
    total_votes: int
    spy_tlt_corr: float | None


# ---------------------------------------------------------------------------
# Regime Timing Strategy
# ---------------------------------------------------------------------------


class RegimeTimingStrategy(Strategy):
    """4-regime via 5-signal majority vote + correlation overlay.

    Regime classification:
      1. risk_on   -- majority of 5 votes are risk-on
      2. risk_off  -- majority of 5 votes are risk-off
      3. transition -- mixed / split votes
      4. inflationary -- correlation overlay: 60d SPY-TLT corr > 0

    Five ensemble votes (RT-H10):
      (1) VIX(t-1): < 20 risk-on, > 30 risk-off, else neutral
      (2) Yield curve: IEF/SHY ratio vs 50-day SMA
      (3) Credit: 20-day cumulative HYG-IEF return spread
      (4) SPY momentum: close vs SMA_50
      (5) GLD momentum: close vs SMA_50 (inverse)

    Correlation overlay (RT-H03):
      If 60-day SPY-TLT correlation > 0.0, override to inflationary.

    Required symbols: SPY, TLT, IEF, SHY, HYG, GLD, QQQ, IWM,
                      VIX, USO, LQD, BTC-USD, EURUSD=X.
    """

    # Regime allocation maps (from research-spec.yaml)
    RISK_ON_ALLOC: ClassVar[dict[str, float]] = {
        "SPY": 0.25,
        "QQQ": 0.15,
        "IWM": 0.10,
        "GLD": 0.05,
        "HYG": 0.05,
        "BTC-USD": 0.05,
        "IEF": 0.10,
    }
    RISK_OFF_ALLOC: ClassVar[dict[str, float]] = {
        "TLT": 0.25,
        "IEF": 0.15,
        "SHY": 0.10,
        "GLD": 0.15,
        "LQD": 0.05,
    }
    TRANSITION_ALLOC: ClassVar[dict[str, float]] = {
        "SPY": 0.15,
        "TLT": 0.15,
        "IEF": 0.10,
        "GLD": 0.10,
        "SHY": 0.10,
    }
    INFLATIONARY_ALLOC: ClassVar[dict[str, float]] = {
        "GLD": 0.20,
        "USO": 0.10,
        "SHY": 0.15,
        "SPY": 0.10,
        "EURUSD=X": 0.05,
    }

    REGIME_ALLOCS: ClassVar[dict[str, dict[str, float]]] = {
        "risk_on": RISK_ON_ALLOC,
        "risk_off": RISK_OFF_ALLOC,
        "transition": TRANSITION_ALLOC,
        "inflationary": INFLATIONARY_ALLOC,
    }

    VIX_SYMBOL: ClassVar[str] = "VIX"

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters
        stop_mult = params.get("stop_atr_multiplier", 2.0)
        target_vol = params.get("target_vol_annual", 0.10)

        ctx = self._classify_regime(indicators_df, params)

        logger.info(
            "Regime=%s (on=%d off=%d total=%d corr=%s)",
            ctx.regime,
            ctx.risk_on_votes,
            ctx.risk_off_votes,
            ctx.total_votes,
            f"{ctx.spy_tlt_corr:.3f}" if ctx.spy_tlt_corr is not None else "N/A",
        )

        return self._build_signals(
            ctx,
            portfolio,
            prices,
            indicators_df,
            stop_mult,
            target_vol,
        )

    # ------------------------------------------------------------------
    # Regime classification (5 votes + overlay)
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        indicators_df: pl.DataFrame,
        params: dict,
    ) -> _RegimeContext:
        """Run the 5-vote ensemble + correlation overlay."""
        vix_risk_on = params.get("vix_risk_on_threshold", 20)
        vix_risk_off = params.get("vix_risk_off_threshold", 30)
        curve_sma = params.get("tlt_ief_ratio_sma", 50)
        credit_win = params.get("credit_spread_momentum", 20)
        spy_sma = params.get("spy_sma_trend", 50)
        gld_sma = params.get("gld_sma_trend", 50)
        corr_lb = params.get("correlation_lookback", 60)
        corr_thr = params.get("correlation_threshold", 0.0)
        majority = params.get("ensemble_majority", 3)

        votes = [
            self._vote_vix(indicators_df, vix_risk_on, vix_risk_off),
            self._vote_yield_curve(indicators_df, curve_sma),
            self._vote_credit(indicators_df, credit_win),
            self._vote_momentum(indicators_df, "SPY", spy_sma),
            self._vote_momentum(
                indicators_df,
                "GLD",
                gld_sma,
                inverse=True,
            ),
        ]

        on = sum(1 for v in votes if v == "risk_on")
        off = sum(1 for v in votes if v == "risk_off")
        total = sum(1 for v in votes if v is not None)

        if total == 0:
            regime = "transition"
        elif on >= majority:
            regime = "risk_on"
        elif off >= majority:
            regime = "risk_off"
        else:
            regime = "transition"

        corr = self._compute_spy_tlt_correlation(
            indicators_df,
            corr_lb,
        )
        if corr is not None and corr > corr_thr:
            regime = "inflationary"

        return _RegimeContext(
            regime=regime,
            risk_on_votes=on,
            risk_off_votes=off,
            total_votes=total,
            spy_tlt_corr=corr,
        )

    # ------------------------------------------------------------------
    # Ensemble vote methods
    # ------------------------------------------------------------------

    def _vote_vix(
        self,
        indicators_df: pl.DataFrame,
        risk_on_threshold: float,
        risk_off_threshold: float,
    ) -> str | None:
        """VIX vote using t-1 close to avoid look-ahead bias."""
        vix_df = _sym_series(indicators_df, self.VIX_SYMBOL)
        if len(vix_df) < 2:
            return None

        prev_row = vix_df.tail(2).row(0, named=True)
        vix_val = prev_row.get("close")
        if vix_val is None:
            return None

        if vix_val < risk_on_threshold:
            return "risk_on"
        if vix_val > risk_off_threshold:
            return "risk_off"
        return "neutral"

    def _vote_yield_curve(
        self,
        indicators_df: pl.DataFrame,
        sma_window: int,
    ) -> str | None:
        """Yield curve: IEF/SHY ratio vs its SMA.

        Above SMA = steepening = risk_on.
        Below SMA = flattening = risk_off.
        """
        ratio_ts = _price_ratio_series(
            indicators_df,
            "IEF",
            "SHY",
        )
        if len(ratio_ts) < sma_window:
            return None

        ratio_values = [r for _, r in ratio_ts]
        current = ratio_values[-1]
        sma = _rolling_mean(ratio_values, sma_window)
        if sma is None:
            return None

        return "risk_on" if current > sma else "risk_off"

    def _vote_credit(
        self,
        indicators_df: pl.DataFrame,
        window: int,
    ) -> str | None:
        """Credit: 20-day cumulative HYG-IEF return spread.

        Positive = tightening = risk_on.
        Negative = widening = risk_off.
        """
        spread = _cumulative_spread_return(
            indicators_df,
            "HYG",
            "IEF",
            window,
        )
        if spread is None:
            return None
        return "risk_on" if spread > 0 else "risk_off"

    def _vote_momentum(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        sma_window: int,
        *,
        inverse: bool = False,
    ) -> str | None:
        """Momentum vote: close vs SMA.

        For SPY: above SMA = risk_on.
        For GLD (inverse): above SMA = risk_off.
        """
        sym_df = _sym_series(indicators_df, symbol)
        if len(sym_df) == 0:
            return None

        close = _latest_close(sym_df)
        if close is None:
            return None

        sma_col = f"sma_{sma_window}"
        sma_val = _latest_indicator(sym_df, sma_col)
        if sma_val is None:
            return None

        above_sma = close > sma_val
        if inverse:
            return "risk_off" if above_sma else "risk_on"
        return "risk_on" if above_sma else "risk_off"

    # ------------------------------------------------------------------
    # Correlation overlay
    # ------------------------------------------------------------------

    def _compute_spy_tlt_correlation(
        self,
        indicators_df: pl.DataFrame,
        lookback: int,
    ) -> float | None:
        """60-day rolling correlation between SPY and TLT returns."""
        spy_rets = _daily_returns(indicators_df, "SPY")
        tlt_rets = _daily_returns(indicators_df, "TLT")
        return _rolling_correlation(spy_rets, tlt_rets, lookback)

    # ------------------------------------------------------------------
    # Signal generation from regime allocation
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        ctx: _RegimeContext,
        portfolio: Portfolio,
        prices: dict[str, float],
        indicators_df: pl.DataFrame,
        stop_mult: float,
        target_vol: float,
    ) -> list[TradeSignal]:
        """Buy/close signals to move toward regime allocation."""
        signals: list[TradeSignal] = []
        alloc = self.REGIME_ALLOCS.get(
            ctx.regime,
            self.TRANSITION_ALLOC,
        )

        conviction = self._conviction_from_votes(ctx)
        reason = self._regime_reason(ctx)

        # Close positions not in target allocation
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=conviction,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=(f"Exit: not in {ctx.regime} allocation. {reason}"),
            )
            for sym in portfolio.positions
            if sym not in alloc
        )

        # Open positions in target allocation not yet held
        new_pos = 0
        for sym, weight in alloc.items():
            if sym in portfolio.positions:
                continue
            if len(portfolio.positions) + new_pos >= self.config.max_positions:
                break

            close = prices.get(sym, 0)
            if close <= 0:
                continue

            scaled = self._vol_scaled_weight(
                indicators_df,
                sym,
                weight,
                target_vol,
            )
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
                    conviction=conviction,
                    target_weight=scaled,
                    stop_loss=sl,
                    reasoning=(f"Enter {ctx.regime} ({weight:.0%}). {reason}"),
                )
            )
            new_pos += 1

        return signals

    @staticmethod
    def _conviction_from_votes(ctx: _RegimeContext) -> Conviction:
        """Derive conviction from vote clarity."""
        if ctx.total_votes > 0:
            margin = abs(ctx.risk_on_votes - ctx.risk_off_votes) / ctx.total_votes
        else:
            margin = 0.0

        if margin >= 0.6:
            return Conviction.HIGH
        if margin >= 0.2:
            return Conviction.MEDIUM
        return Conviction.LOW

    @staticmethod
    def _regime_reason(ctx: _RegimeContext) -> str:
        """Build a human-readable regime reasoning string."""
        corr_str = f"{ctx.spy_tlt_corr:.3f}" if ctx.spy_tlt_corr is not None else "N/A"
        return (
            f"Regime={ctx.regime} "
            f"(on={ctx.risk_on_votes} "
            f"off={ctx.risk_off_votes} "
            f"total={ctx.total_votes} "
            f"corr={corr_str})"
        )

    # ------------------------------------------------------------------
    # Vol-targeted sizing
    # ------------------------------------------------------------------

    def _vol_scaled_weight(
        self,
        indicators_df: pl.DataFrame,
        symbol: str,
        base_weight: float,
        target_vol: float,
    ) -> float:
        """Scale weight by target_vol / realized_vol (from ATR).

        Caps at 2x base weight, floors at 0.25x base weight.
        """
        sym_df = _sym_series(indicators_df, symbol)
        atr = _latest_indicator(sym_df, "atr_14")
        close = _latest_close(sym_df)

        if atr is None or close is None or close <= 0 or atr <= 0:
            return base_weight

        # Annualized vol estimate: (ATR/close) * sqrt(252)
        daily_vol = atr / close
        annual_vol = daily_vol * (252**0.5)

        if annual_vol <= 0:
            return base_weight

        scale = target_vol / annual_vol
        scale = max(0.25, min(scale, 2.0))
        return base_weight * scale

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
    "regime_timing": RegimeTimingStrategy,
}

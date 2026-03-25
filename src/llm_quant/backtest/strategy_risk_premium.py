"""Risk Premium Harvesting strategy implementation.

Implements three layers — risk parity base (H10), VRP + credit carry
overlays (H01, H04), and premium collapse circuit breaker — for
systematic risk premium collection with regime-conditional timing.
All computations are causal (backward-looking only).  See
data/strategies/risk-premium/research-spec.yaml for the frozen design.
"""

from __future__ import annotations

import logging
import math
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


def _sym_series(indicators_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Return the rows for *symbol*, sorted by date."""
    return indicators_df.filter(pl.col("symbol") == symbol).sort("date")


def _latest_close(sym_df: pl.DataFrame) -> float | None:
    """Return the most recent close price, or None."""
    if len(sym_df) == 0:
        return None
    return sym_df.tail(1).row(0, named=True)["close"]


def _realized_vol(
    sym_df: pl.DataFrame,
    window: int,
) -> float | None:
    """Annualized realized vol from log returns over *window* days."""
    if len(sym_df) < window + 1:
        return None

    close_col = "adj_close" if "adj_close" in sym_df.columns else "close"
    recent = sym_df.tail(window + 1)
    closes = recent.select(pl.col(close_col)).to_series().to_list()

    log_rets: list[float] = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev is not None and cur is not None and prev > 0:
            log_rets.append(math.log(cur / prev))

    if len(log_rets) < window:
        return None

    rets = log_rets[-window:]
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
    return math.sqrt(var) * math.sqrt(252)


# ---------------------------------------------------------------------------
# Internal bundles to keep argument counts low
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Execution context passed through signal generation."""

    portfolio: Portfolio
    prices: dict[str, float]
    indicators_df: pl.DataFrame
    stop_mult: float
    max_positions: int
    target_weight: float
    stop_loss_pct: float


@dataclass
class _NormalParams:
    """Groups normal-mode thresholds."""

    vrp_entry: float
    vrp_exit: float
    vix_tail: float
    vix_credit_full: float
    vix_credit_zero: float


# ---------------------------------------------------------------------------
# Risk Premium Strategy
# ---------------------------------------------------------------------------


class RiskPremiumStrategy(Strategy):
    """Risk premium harvesting with regime-conditional timing.

    Three layers:
      Layer 1: Risk parity base (SPY, TLT, GLD, EURUSD=X)
      Layer 2: VRP equity tilt + VIX-conditioned HYG carry
      Layer 3: Premium collapse circuit breaker

    Required: SPY, TLT, GLD, EURUSD=X, HYG, IEF, VIX.
    """

    RISK_PARITY_ASSETS: ClassVar[list[str]] = [
        "SPY",
        "TLT",
        "GLD",
        "EURUSD=X",
    ]
    VIX_SYMBOL: ClassVar[str] = "VIX"

    def generate_signals(
        self,
        _as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters

        vol_win = params.get("realized_vol_window", 21)
        rp_lookback = params.get("risk_parity_vol_lookback", 63)
        stop_mult = params.get("stop_atr_multiplier", 2.0)
        vix_credit_zero = params.get("vix_credit_zero", 25)

        # Build execution context
        ctx = _Ctx(
            portfolio=portfolio,
            prices=prices,
            indicators_df=indicators_df,
            stop_mult=stop_mult,
            max_positions=self.config.max_positions,
            target_weight=self.config.target_position_weight,
            stop_loss_pct=self.config.stop_loss_pct,
        )

        # Core indicators
        vix = self._get_vix(indicators_df)
        vrp = self._compute_vrp(indicators_df, vol_win, vix)
        rp_w = self._risk_parity_weights(indicators_df, rp_lookback)

        # Layer 3: Premium collapse
        if self._premium_collapsed(vrp, vix, vix_credit_zero):
            return self._collapse_signals(rp_w, vrp, vix, ctx)

        # Layer 1 + 2: Normal
        np = _NormalParams(
            vrp_entry=params.get("vrp_entry_threshold", 5.0),
            vrp_exit=params.get("vrp_exit_threshold", 2.0),
            vix_tail=params.get("vix_tail_override", 35),
            vix_credit_full=params.get("vix_credit_full", 20),
            vix_credit_zero=vix_credit_zero,
        )
        return self._normal_signals(rp_w, vrp, vix, np, ctx)

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _get_vix(self, indicators_df: pl.DataFrame) -> float | None:
        """Latest VIX close."""
        vix_df = _sym_series(indicators_df, self.VIX_SYMBOL)
        return _latest_close(vix_df)

    def _compute_vrp(
        self,
        indicators_df: pl.DataFrame,
        vol_window: int,
        vix: float | None,
    ) -> float | None:
        """VRP = VIX - realized vol (annualized, in %)."""
        if vix is None:
            return None
        spy_df = _sym_series(indicators_df, "SPY")
        rv = _realized_vol(spy_df, vol_window)
        if rv is None:
            return None
        return vix - rv * 100

    def _risk_parity_weights(
        self,
        indicators_df: pl.DataFrame,
        vol_lookback: int,
    ) -> dict[str, float]:
        """Inverse-vol weights summing to 1.0."""
        vols: dict[str, float] = {}
        for sym in self.RISK_PARITY_ASSETS:
            sym_df = _sym_series(indicators_df, sym)
            rv = _realized_vol(sym_df, vol_lookback)
            if rv is not None and rv > 0:
                vols[sym] = rv

        if not vols:
            eq_w = 1.0 / len(self.RISK_PARITY_ASSETS)
            return dict.fromkeys(self.RISK_PARITY_ASSETS, eq_w)

        inv = {s: 1.0 / v for s, v in vols.items()}
        total = sum(inv.values())
        return {s: iv / total for s, iv in inv.items()}

    # ------------------------------------------------------------------
    # Premium collapse (Layer 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _premium_collapsed(
        vrp: float | None,
        vix: float | None,
        vix_threshold: float,
    ) -> bool:
        """VRP inverted AND VIX elevated = halt."""
        if vrp is None or vix is None:
            return False
        return vrp < 0 and vix > vix_threshold

    def _collapse_signals(
        self,
        rp_w: dict[str, float],
        vrp: float | None,
        vix: float | None,
        ctx: _Ctx,
    ) -> list[TradeSignal]:
        """Scale RP to 0.8x, close overlays."""
        signals: list[TradeSignal] = []
        scale = 0.8
        vrp_s = f"{vrp:.1f}" if vrp is not None else "N/A"
        vix_s = f"{vix:.1f}" if vix is not None else "N/A"
        reason = f"Premium collapse: VRP={vrp_s}, VIX={vix_s} — {scale}x"

        rp_set = set(self.RISK_PARITY_ASSETS)
        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=(
                    Conviction.HIGH if sym in ("HYG", "IEF") else Conviction.MEDIUM
                ),
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=f"Exit overlay: {reason}",
            )
            for sym in ctx.portfolio.positions
            if sym not in rp_set
        )

        new_pos = 0
        for sym, base_w in rp_w.items():
            w = base_w * scale
            close = ctx.prices.get(sym, 0)
            if close <= 0 or sym in ctx.portfolio.positions:
                continue
            if len(ctx.portfolio.positions) + new_pos >= ctx.max_positions:
                continue
            sl = self._stop_loss(ctx, sym, close)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.LOW,
                    target_weight=w,
                    stop_loss=sl,
                    reasoning=(f"RP base ({scale}x): w={w:.2%}; {reason}"),
                )
            )
            new_pos += 1

        return signals

    # ------------------------------------------------------------------
    # Normal mode (Layer 1 + 2)
    # ------------------------------------------------------------------

    def _normal_signals(
        self,
        rp_w: dict[str, float],
        vrp: float | None,
        vix: float | None,
        np: _NormalParams,
        ctx: _Ctx,
    ) -> list[TradeSignal]:
        """Risk parity base + VRP/credit overlays."""
        if vix is not None and vix > np.vix_tail:
            return self._tail_override(rp_w, vix, np, ctx)

        signals: list[TradeSignal] = []
        tw = dict(rp_w)
        rp_reasons: dict[str, list[str]] = {
            s: [f"RP base: w={w:.2%}"] for s, w in rp_w.items()
        }

        self._apply_vrp_overlay(vrp, np, tw, rp_reasons, ctx, signals)
        self._apply_credit_overlay(vix, np, ctx, signals)
        self._emit_rp_buys(tw, rp_reasons, ctx, signals)

        return signals

    # ------------------------------------------------------------------
    # Sub-methods
    # ------------------------------------------------------------------

    def _tail_override(
        self,
        rp_w: dict[str, float],
        vix: float,
        np: _NormalParams,
        ctx: _Ctx,
    ) -> list[TradeSignal]:
        """VIX > 35: exit equity + HYG, keep non-equity RP."""
        signals: list[TradeSignal] = []
        close_syms = {"SPY", "QQQ", "IWM", "HYG"}
        reason = f"VIX tail ({vix:.1f} > {np.vix_tail}): unconditional exit"

        signals.extend(
            TradeSignal(
                symbol=sym,
                action=Action.CLOSE,
                conviction=Conviction.HIGH,
                target_weight=0.0,
                stop_loss=0.0,
                reasoning=reason,
            )
            for sym in ctx.portfolio.positions
            if sym in close_syms
        )

        new_pos = 0
        for sym, w in rp_w.items():
            if sym == "SPY":
                continue
            close = ctx.prices.get(sym, 0)
            if close <= 0 or sym in ctx.portfolio.positions:
                continue
            if len(ctx.portfolio.positions) + new_pos >= ctx.max_positions:
                continue
            sl = self._stop_loss(ctx, sym, close)
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=w,
                    stop_loss=sl,
                    reasoning=(f"RP base (ex-equity): w={w:.2%}, VIX={vix:.1f}"),
                )
            )
            new_pos += 1

        return signals

    def _apply_vrp_overlay(
        self,
        vrp: float | None,
        np: _NormalParams,
        tw: dict[str, float],
        reasons: dict[str, list[str]],
        ctx: _Ctx,
        signals: list[TradeSignal],
    ) -> None:
        """Adjust SPY target weight based on VRP spread."""
        cap = 0.10
        if vrp is None or "SPY" not in tw:
            return

        if vrp > np.vrp_entry:
            tilt = min((vrp - np.vrp_entry) / 10.0 * cap, cap)
            tw["SPY"] += tilt
            reasons.setdefault("SPY", []).append(
                f"VRP overlay: +{tilt:.2%} (VRP={vrp:.1f})"
            )
        elif vrp < np.vrp_exit:
            tilt = -min(cap, tw.get("SPY", 0) * 0.5)
            tw["SPY"] = max(0.0, tw["SPY"] + tilt)
            reasons.setdefault("SPY", []).append(
                f"VRP exit: {tilt:.2%} (VRP={vrp:.1f} < {np.vrp_exit})"
            )
            spy_w = tw.get("SPY", 0)
            if "SPY" in ctx.portfolio.positions and spy_w <= 0.01:
                signals.append(
                    TradeSignal(
                        symbol="SPY",
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            f"VRP exit: compressed (VRP={vrp:.1f} < {np.vrp_exit})"
                        ),
                    )
                )

    def _apply_credit_overlay(
        self,
        vix: float | None,
        np: _NormalParams,
        ctx: _Ctx,
        signals: list[TradeSignal],
    ) -> None:
        """Add/remove HYG based on VIX regime (H04)."""
        if vix is None:
            return

        base_w = ctx.target_weight
        if vix < np.vix_credit_full:
            self._buy_hyg(
                base_w,
                f"Credit carry: full (VIX={vix:.1f})",
                ctx,
                signals,
            )
        elif vix <= np.vix_credit_zero:
            self._buy_hyg(
                base_w * 0.5,
                f"Credit carry: half (VIX={vix:.1f})",
                ctx,
                signals,
            )
        else:
            self._exit_hyg_sub_ief(vix, np, ctx, signals)

    def _buy_hyg(
        self,
        weight: float,
        reason: str,
        ctx: _Ctx,
        signals: list[TradeSignal],
    ) -> None:
        """Emit HYG buy if not held, close IEF substitute."""
        hyg_close = ctx.prices.get("HYG", 0)
        if (
            hyg_close > 0
            and "HYG" not in ctx.portfolio.positions
            and len(ctx.portfolio.positions) < ctx.max_positions
        ):
            sl = self._stop_loss(ctx, "HYG", hyg_close)
            signals.append(
                TradeSignal(
                    symbol="HYG",
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=weight,
                    stop_loss=sl,
                    reasoning=reason,
                )
            )
        if "IEF" in ctx.portfolio.positions:
            signals.append(
                TradeSignal(
                    symbol="IEF",
                    action=Action.CLOSE,
                    conviction=Conviction.LOW,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning="Closing IEF sub: HYG restored",
                )
            )

    def _exit_hyg_sub_ief(
        self,
        vix: float,
        np: _NormalParams,
        ctx: _Ctx,
        signals: list[TradeSignal],
    ) -> None:
        """Close HYG (VIX too high), open IEF substitute."""
        reason = f"Credit carry: zero (VIX={vix:.1f} > {np.vix_credit_zero})"
        if "HYG" in ctx.portfolio.positions:
            signals.append(
                TradeSignal(
                    symbol="HYG",
                    action=Action.CLOSE,
                    conviction=Conviction.HIGH,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=reason,
                )
            )
        ief_close = ctx.prices.get("IEF", 0)
        n_pos = len(ctx.portfolio.positions)
        if (
            ief_close > 0
            and "IEF" not in ctx.portfolio.positions
            and n_pos < ctx.max_positions
        ):
            sl = self._stop_loss(ctx, "IEF", ief_close)
            signals.append(
                TradeSignal(
                    symbol="IEF",
                    action=Action.BUY,
                    conviction=Conviction.LOW,
                    target_weight=ctx.target_weight * 0.5,
                    stop_loss=sl,
                    reasoning=(f"Credit sub: IEF replaces HYG (VIX={vix:.1f})"),
                )
            )

    def _emit_rp_buys(
        self,
        tw: dict[str, float],
        reasons: dict[str, list[str]],
        ctx: _Ctx,
        signals: list[TradeSignal],
    ) -> None:
        """Emit buy/close signals for risk parity base."""
        new_pos = 0
        for sym, w in tw.items():
            if w <= 0.01:
                if sym in ctx.portfolio.positions:
                    signals.append(
                        TradeSignal(
                            symbol=sym,
                            action=Action.CLOSE,
                            conviction=Conviction.LOW,
                            target_weight=0.0,
                            stop_loss=0.0,
                            reasoning=(f"RP weight negligible ({w:.2%})"),
                        )
                    )
                continue

            close = ctx.prices.get(sym, 0)
            if close <= 0 or sym in ctx.portfolio.positions:
                continue
            if len(ctx.portfolio.positions) + new_pos >= ctx.max_positions:
                continue

            sl = self._stop_loss(ctx, sym, close)
            r = "; ".join(reasons.get(sym, [f"w={w:.2%}"]))
            signals.append(
                TradeSignal(
                    symbol=sym,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=w,
                    stop_loss=sl,
                    reasoning=r,
                )
            )
            new_pos += 1

    # ------------------------------------------------------------------
    # Stop-loss
    # ------------------------------------------------------------------

    def _stop_loss(self, ctx: _Ctx, symbol: str, close: float) -> float:
        """ATR-based stop; HYG uses 4% fixed stop."""
        if symbol == "HYG":
            return close * 0.96

        sym_data = _sym_series(ctx.indicators_df, symbol)
        if "atr_14" in sym_data.columns and len(sym_data) > 0:
            atr = sym_data.tail(1).row(0, named=True).get("atr_14")
            if atr and atr > 0:
                return close - (ctx.stop_mult * atr)
        return close * (1.0 - ctx.stop_loss_pct)


# ---------------------------------------------------------------------------
# Strategy registry entry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[Strategy]] = {
    "risk_premium": RiskPremiumStrategy,
}

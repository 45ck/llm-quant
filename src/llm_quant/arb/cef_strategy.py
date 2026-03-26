"""CEF Discount Mean-Reversion Strategy.

Multi-asset strategy that trades closed-end fund discount anomalies.
When a CEF's discount z-score falls below -1.5 (unusually deep), buy.
When it reverts to the mean (z > 0), sell.

This is a structural arbitrage: CEF discounts are mean-reverting because
activist investors and tender offers create a gravitational pull toward NAV.
The strategy is low-beta by nature since it profits from the discount
narrowing, not from the direction of the underlying assets.

Track C (Niche Arbitrage) — benchmark: T-bill rate.

Two implementations:
  1. CEFDiscountStrategy — standalone class with its own signal type (CEFSignal).
     Used by scripts/run_cef_backtest.py.
  2. CEFDiscountRegistryStrategy — Strategy ABC subclass for STRATEGY_REGISTRY.
     Uses quintile-based cross-sectional selection and optional TLT hedge.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from llm_quant.backtest.strategy import Strategy
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)

# CEF→benchmark mapping (duplicated from cef_data to avoid circular import)
_CEF_BENCHMARK: dict[str, str] = {
    "NEA": "AGG",
    "NAD": "AGG",
    "PDI": "HYG",
    "PTY": "HYG",
    "HYT": "HYG",
    "EHI": "HYG",
    "AWF": "AGG",
    "BGT": "AGG",
    "NVG": "MUB",
    "VPV": "MUB",
    "BLE": "MUB",
    "MQY": "MUB",
    "ADX": "SPY",
    "GAM": "SPY",
    "GDV": "SPY",
}


@dataclass
class CEFStrategyConfig:
    """Configuration for CEF discount mean-reversion strategy."""

    # Z-score thresholds
    z_entry: float = -1.5  # Buy when z-score < this (deep discount)
    z_exit: float = 0.0  # Sell when z-score > this (discount reverted)

    # Rolling window for discount statistics
    lookback_days: int = 252  # 1 year trailing average/std

    # Position sizing
    max_positions: int = 10
    target_weight_per_position: float = 0.10  # Equal-weight

    # Rebalancing
    rebalance_frequency_days: int = 21  # Monthly check

    # Stop-loss on individual positions (based on discount widening further)
    max_discount_z: float = -4.0  # Exit if discount blows out to z < -4

    # Hedge: short TLT proportional to portfolio duration (optional)
    hedge_enabled: bool = False
    hedge_symbol: str = "TLT"
    hedge_ratio: float = 0.30  # Short 30% of CEF notional in TLT


@dataclass
class CEFSignal:
    """A single CEF trade signal."""

    ticker: str
    action: str  # "buy" or "sell"
    z_score: float
    discount_pct: float
    reasoning: str


@dataclass
class CEFPortfolioState:
    """Track which CEFs we currently hold."""

    positions: dict[str, float] = field(default_factory=dict)  # ticker → weight
    entry_dates: dict[str, date] = field(default_factory=dict)
    entry_z_scores: dict[str, float] = field(default_factory=dict)


class CEFDiscountStrategy:
    """CEF Discount Mean-Reversion Strategy.

    Generates signals based on discount z-scores across a universe of CEFs.
    Unlike the single-asset Strategy ABC, this operates on multiple CEFs
    simultaneously with equal-weight allocation.

    The strategy:
    1. Computes rolling discount stats (mean, std) per CEF over lookback_days
    2. Calculates current z-score = (current_discount - rolling_mean) / rolling_std
    3. BUY when z-score < z_entry (unusually deep discount)
    4. SELL when z-score > z_exit (discount narrowed to average)
    """

    def __init__(self, config: CEFStrategyConfig | None = None) -> None:
        self.config = config or CEFStrategyConfig()
        self.state = CEFPortfolioState()

    def generate_signals(
        self,
        as_of_date: date,
        cef_df: pl.DataFrame,
    ) -> list[CEFSignal]:
        """Generate CEF trade signals for the given date.

        Parameters
        ----------
        as_of_date:
            Current trading date.
        cef_df:
            CEF daily data with columns: date, ticker, price, nav_estimate,
            discount_pct, volume. Must contain only data <= as_of_date.

        Returns
        -------
        list[CEFSignal]
            Buy/sell signals for CEFs.
        """
        signals: list[CEFSignal] = []

        # Filter to data up to as_of_date (causal)
        causal_df = cef_df.filter(pl.col("date") <= as_of_date)
        if len(causal_df) == 0:
            return signals

        tickers = causal_df.select("ticker").unique().to_series().to_list()

        pending_buys = 0
        for ticker in tickers:
            ticker_data = causal_df.filter(pl.col("ticker") == ticker).sort("date")

            if len(ticker_data) < self.config.lookback_days:
                continue

            signal = self._evaluate_ticker(ticker, ticker_data, pending_buys)
            if signal is not None:
                signals.append(signal)
                if signal.action == "buy":
                    pending_buys += 1

        return signals

    def _evaluate_ticker(
        self,
        ticker: str,
        ticker_data: pl.DataFrame,
        pending_buys: int = 0,
    ) -> CEFSignal | None:
        """Evaluate a single CEF for entry/exit signals."""
        lookback = self.config.lookback_days
        recent = ticker_data.tail(lookback)

        discounts = recent["discount_pct"].to_list()

        # Filter out None values
        discounts = [d for d in discounts if d is not None]
        if len(discounts) < lookback // 2:
            return None

        # Current discount
        current_discount = discounts[-1]

        # Rolling mean and std of discount
        mean_discount = sum(discounts) / len(discounts)
        variance = sum((d - mean_discount) ** 2 for d in discounts) / len(discounts)
        std_discount = math.sqrt(variance) if variance > 0 else 0.0

        if std_discount < 1e-6:
            return None

        z_score = (current_discount - mean_discount) / std_discount

        is_held = ticker in self.state.positions

        # EXIT: discount reverted to mean or beyond
        if is_held and z_score > self.config.z_exit:
            entry_z = self.state.entry_z_scores.get(ticker, 0.0)
            return CEFSignal(
                ticker=ticker,
                action="sell",
                z_score=z_score,
                discount_pct=current_discount,
                reasoning=(
                    f"Discount reverted: z={z_score:.2f} > {self.config.z_exit} "
                    f"(entry z={entry_z:.2f}, discount={current_discount:.3f})"
                ),
            )

        # STOP-LOSS: discount blew out way beyond entry
        if is_held and z_score < self.config.max_discount_z:
            return CEFSignal(
                ticker=ticker,
                action="sell",
                z_score=z_score,
                discount_pct=current_discount,
                reasoning=(
                    f"Discount blowout stop: z={z_score:.2f}"
                    f" < {self.config.max_discount_z}"
                ),
            )

        # ENTRY: deep discount
        total_positions = len(self.state.positions) + pending_buys
        if (
            not is_held
            and z_score < self.config.z_entry
            and total_positions < self.config.max_positions
        ):
            return CEFSignal(
                ticker=ticker,
                action="buy",
                z_score=z_score,
                discount_pct=current_discount,
                reasoning=(
                    f"Deep discount: z={z_score:.2f} < {self.config.z_entry} "
                    f"(discount={current_discount:.3f}, "
                    f"mean={mean_discount:.3f}, std={std_discount:.3f})"
                ),
            )

        return None

    def apply_signal(self, signal: CEFSignal, as_of_date: date) -> None:
        """Update internal state after a signal is executed."""
        if signal.action == "buy":
            self.state.positions[signal.ticker] = self.config.target_weight_per_position
            self.state.entry_dates[signal.ticker] = as_of_date
            self.state.entry_z_scores[signal.ticker] = signal.z_score
        elif signal.action == "sell":
            self.state.positions.pop(signal.ticker, None)
            self.state.entry_dates.pop(signal.ticker, None)
            self.state.entry_z_scores.pop(signal.ticker, None)

    def compute_discount_z_scores(
        self,
        cef_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute discount z-scores for the full history (for backtesting).

        Returns a DataFrame with columns:
            date, ticker, price, discount_pct, discount_z_score
        """
        lookback = self.config.lookback_days

        result = cef_df.sort(["ticker", "date"]).with_columns(
            pl.col("discount_pct")
            .rolling_mean(window_size=lookback, min_samples=lookback // 2)
            .over("ticker")
            .alias("discount_mean"),
            pl.col("discount_pct")
            .rolling_std(window_size=lookback, min_samples=lookback // 2)
            .over("ticker")
            .alias("discount_std"),
        )

        result = result.with_columns(
            pl.when(pl.col("discount_std") > 1e-6)
            .then(
                (pl.col("discount_pct") - pl.col("discount_mean"))
                / pl.col("discount_std")
            )
            .otherwise(0.0)
            .alias("discount_z_score"),
        )

        return result.select(
            [
                "date",
                "ticker",
                "price",
                "discount_pct",
                "discount_z_score",
            ]
        )


# ---------------------------------------------------------------------------
# STRATEGY_REGISTRY-compatible CEF discount strategy
# ---------------------------------------------------------------------------


class CEFDiscountRegistryStrategy(Strategy):
    """CEF discount mean-reversion for STRATEGY_REGISTRY.

    Estimates NAV from benchmark ETF ratios, computes discount z-scores
    cross-sectionally, and buys the deepest-discount quintile. Sells when
    discounts revert to the mean. Optionally hedges interest rate risk by
    shorting TLT proportional to total CEF exposure.

    Parameters (via StrategyConfig.parameters):
      cef_tickers (str): Comma-separated CEF tickers.
          Default: "NEA,NAD,PDI,PTY,HYT".
      lookback_days (int, default 252): Rolling window for discount stats.
      z_entry (float, default -1.5): Buy when z-score < this.
      z_exit (float, default 0.0): Sell when z-score > this.
      z_stop (float, default -4.0): Stop-loss on blowout.
      quintile_select (bool, default True): If True, buy bottom quintile
          (deepest discounts) rather than all below z_entry.
      hedge_enabled (bool, default True): Short TLT to hedge rates.
      hedge_symbol (str, default "TLT"): Hedge instrument.
      hedge_ratio (float, default 0.30): Hedge 30% of CEF notional.
      target_weight (float, default 0.10): Weight per CEF position.
      nav_ratio_window (int, default 252): Window for NAV ratio calibration.
    """

    def _compute_cef_discounts(
        self,
        indicators_df: pl.DataFrame,
        cef_tickers: list[str],
        nav_ratio_window: int,
    ) -> dict[str, tuple[float, list[float]]]:
        """Compute current discount and discount history for each CEF.

        Returns dict of ticker -> (current_discount, discount_history).
        Discount = (price - NAV_est) / NAV_est where NAV_est is derived
        from the rolling median of (CEF_price / benchmark_price).
        """
        results: dict[str, tuple[float, list[float]]] = {}

        for ticker in cef_tickers:
            benchmark = _CEF_BENCHMARK.get(ticker)
            if benchmark is None:
                continue

            cef_data = (
                indicators_df.filter(pl.col("symbol") == ticker)
                .sort("date")
                .select(["date", "close"])
            )
            bm_data = (
                indicators_df.filter(pl.col("symbol") == benchmark)
                .sort("date")
                .select(["date", "close"])
                .rename({"close": "bm_close"})
            )

            if len(cef_data) < 60 or len(bm_data) < 60:
                continue

            joined = cef_data.join(bm_data, on="date", how="inner").sort("date")
            if len(joined) < 60:
                continue

            # Ratio of CEF price to benchmark price
            joined = joined.with_columns(
                (pl.col("close") / pl.col("bm_close")).alias("ratio"),
            )
            # Rolling median as "fair ratio" (NAV proxy)
            joined = joined.with_columns(
                pl.col("ratio")
                .rolling_median(window_size=nav_ratio_window, min_samples=60)
                .alias("fair_ratio"),
            )
            # NAV estimate and discount
            joined = joined.with_columns(
                (pl.col("bm_close") * pl.col("fair_ratio")).alias("nav_est"),
            )
            joined = joined.with_columns(
                ((pl.col("close") - pl.col("nav_est")) / pl.col("nav_est")).alias(
                    "discount"
                ),
            )

            valid = joined.filter(pl.col("discount").is_not_null())
            if len(valid) < 60:
                continue

            discounts = valid["discount"].to_list()
            results[ticker] = (discounts[-1], discounts)

        return results

    def _compute_z_scores(
        self,
        discount_map: dict[str, tuple[float, list[float]]],
        lookback: int,
    ) -> list[tuple[str, float, float]]:
        """Compute z-score for each CEF's current discount.

        Returns list of (ticker, z_score, current_discount) sorted by z_score
        ascending (deepest discounts first).
        """
        scored: list[tuple[str, float, float]] = []

        for ticker, (current_disc, history) in discount_map.items():
            window = history[-lookback:] if len(history) >= lookback else history
            if len(window) < lookback // 2:
                continue

            mean_d = sum(window) / len(window)
            var_d = sum((d - mean_d) ** 2 for d in window) / len(window)
            std_d = math.sqrt(var_d) if var_d > 0 else 0.0

            if std_d < 1e-6:
                continue

            z = (current_disc - mean_d) / std_d
            scored.append((ticker, z, current_disc))

        scored.sort(key=lambda x: x[1])  # most negative z first
        return scored

    def generate_signals(  # noqa: C901, PLR0912
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        params = self.config.parameters or {}
        cef_str: str = str(params.get("cef_tickers", "NEA,NAD,PDI,PTY,HYT"))
        cef_tickers = [t.strip() for t in cef_str.split(",") if t.strip()]
        lookback: int = int(params.get("lookback_days", 252))
        z_entry: float = float(params.get("z_entry", -1.5))
        z_exit: float = float(params.get("z_exit", 0.0))
        z_stop: float = float(params.get("z_stop", -4.0))
        use_quintile: bool = bool(params.get("quintile_select", True))
        hedge_on: bool = bool(params.get("hedge_enabled", True))
        hedge_sym: str = str(params.get("hedge_symbol", "TLT"))
        hedge_ratio: float = float(params.get("hedge_ratio", 0.30))
        tgt_weight: float = float(params.get("target_weight", 0.10))
        nav_window: int = int(params.get("nav_ratio_window", 252))

        # Filter indicators to causal data
        causal = indicators_df.filter(pl.col("date") <= as_of_date)

        discount_map = self._compute_cef_discounts(causal, cef_tickers, nav_window)
        if not discount_map:
            return []

        scored = self._compute_z_scores(discount_map, lookback)
        if not scored:
            return []

        signals: list[TradeSignal] = []

        # --- EXIT signals (sell reverted or blowout positions) ---
        for ticker, z, disc in scored:
            if ticker not in portfolio.positions:
                continue
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            if z > z_exit:
                signals.append(
                    TradeSignal(
                        symbol=ticker,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            f"CEF discount reverted: z={z:.2f} > {z_exit} "
                            f"(disc={disc:.3f})"
                        ),
                    )
                )
            elif z < z_stop:
                signals.append(
                    TradeSignal(
                        symbol=ticker,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            f"CEF discount blowout: z={z:.2f} < {z_stop} "
                            f"(disc={disc:.3f})"
                        ),
                    )
                )

        # --- ENTRY signals (quintile or threshold) ---
        # Exclude already-held and those being sold
        sold_tickers = {s.symbol for s in signals}
        held_tickers = set(portfolio.positions.keys()) & set(cef_tickers)
        available = [
            (t, z, d)
            for t, z, d in scored
            if t not in held_tickers and t not in sold_tickers
        ]

        if use_quintile:
            # Buy bottom quintile of z-scores (deepest discounts)
            n_quintile = max(1, len(scored) // 5)
            candidates = [(t, z, d) for t, z, d in available if z < z_entry]
            candidates = candidates[:n_quintile]
        else:
            candidates = [(t, z, d) for t, z, d in available if z < z_entry]

        current_pos_count = len(held_tickers - sold_tickers)
        for ticker, z, disc in candidates:
            if current_pos_count >= self.config.max_positions:
                break
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            signals.append(
                TradeSignal(
                    symbol=ticker,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=tgt_weight,
                    stop_loss=price * 0.90,
                    reasoning=(
                        f"CEF deep discount Q1: z={z:.2f} < {z_entry} (disc={disc:.3f})"
                    ),
                )
            )
            current_pos_count += 1

        # --- HEDGE: short TLT proportional to CEF exposure ---
        if hedge_on and signals:
            buy_count = sum(1 for s in signals if s.action == Action.BUY)
            total_cef_weight = (current_pos_count + buy_count) * tgt_weight
            hedge_weight = total_cef_weight * hedge_ratio
            hedge_price = prices.get(hedge_sym, 0)

            if hedge_weight > 0.01 and hedge_price > 0:
                has_hedge = hedge_sym in portfolio.positions
                if not has_hedge:
                    signals.append(
                        TradeSignal(
                            symbol=hedge_sym,
                            action=Action.SELL,
                            conviction=Conviction.MEDIUM,
                            target_weight=hedge_weight,
                            stop_loss=hedge_price * 1.10,
                            reasoning=(
                                f"CEF rate hedge: short {hedge_sym} "
                                f"{hedge_weight:.1%} of NAV "
                                f"({hedge_ratio:.0%} of {total_cef_weight:.1%} CEF)"
                            ),
                        )
                    )

            # Remove hedge when all CEF positions are closed
            if (
                buy_count == 0
                and current_pos_count == 0
                and hedge_sym in portfolio.positions
            ):
                signals.append(
                    TradeSignal(
                        symbol=hedge_sym,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning="CEF rate hedge removed: no CEF positions",
                    )
                )

        return signals

"""Strategy abstract base class and reference implementation.

Strategies produce TradeSignals based on indicators and portfolio state.
They are pure functions of observable data — no look-ahead allowed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import polars as pl

from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    """Configuration for a backtest strategy.

    Parameters are loaded from research-spec.yaml and frozen before
    backtesting begins.
    """

    name: str = "unnamed"
    rebalance_frequency_days: int = 1
    max_positions: int = 10
    target_position_weight: float = 0.05
    stop_loss_pct: float = 0.05
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rebalance_frequency_days": self.rebalance_frequency_days,
            "max_positions": self.max_positions,
            "target_position_weight": self.target_position_weight,
            "stop_loss_pct": self.stop_loss_pct,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Abstract base class for backtest strategies.

    Subclasses must implement ``generate_signals()`` which is called
    on each rebalance day by the BacktestEngine.

    The contract:
    - ``indicators_df`` contains ONLY data up to and including ``as_of_date``
    - No peeking at future prices or indicators
    - Return a list of TradeSignals (may be empty)
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate trade signals for *as_of_date*.

        Parameters
        ----------
        as_of_date:
            The current trading date. All data in ``indicators_df``
            is <= this date.
        indicators_df:
            Polars DataFrame with OHLCV + indicators, filtered to
            dates <= as_of_date.
        portfolio:
            Current portfolio state (marked to market at as_of_date close).
        prices:
            Latest close prices keyed by symbol (at as_of_date).

        Returns
        -------
        list[TradeSignal]
            Signals for the engine to execute (after risk filtering).
        """
        ...


# ---------------------------------------------------------------------------
# Reference implementation: SMA Crossover
# ---------------------------------------------------------------------------


class SMACrossoverStrategy(Strategy):
    """Simple moving average crossover strategy.

    Buys when SMA-20 crosses above SMA-50 and RSI < 70 (not overbought).
    Sells when SMA-20 crosses below SMA-50 or RSI > 80.
    """

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []

        # Get parameters with defaults
        params = self.config.parameters
        sma_fast = params.get("sma_fast", 20)
        sma_slow = params.get("sma_slow", 50)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_exit = params.get("rsi_exit", 80)

        sma_fast_col = f"sma_{sma_fast}"
        sma_slow_col = f"sma_{sma_slow}"

        # Get unique symbols in the data
        symbols = indicators_df.select("symbol").unique().to_series().to_list()

        for symbol in symbols:
            sym_data = indicators_df.filter(pl.col("symbol") == symbol).sort("date")
            if len(sym_data) < 2:
                continue

            # Check required columns exist
            if (
                sma_fast_col not in sym_data.columns
                or sma_slow_col not in sym_data.columns
            ):
                continue

            # Get last two rows for crossover detection
            last = sym_data.tail(2)
            prev_row = last.row(0, named=True)
            curr_row = last.row(1, named=True)

            # Skip if indicators are null
            if (
                curr_row[sma_fast_col] is None
                or curr_row[sma_slow_col] is None
                or curr_row.get("rsi_14") is None
            ):
                continue

            curr_fast = curr_row[sma_fast_col]
            curr_slow = curr_row[sma_slow_col]
            prev_fast = prev_row[sma_fast_col]
            prev_slow = prev_row[sma_slow_col]
            rsi = curr_row.get("rsi_14", 50.0)
            close = curr_row["close"]

            if prev_fast is None or prev_slow is None:
                continue

            has_position = symbol in portfolio.positions

            # Buy signal: SMA fast crosses above slow, RSI not overbought
            if (
                prev_fast <= prev_slow
                and curr_fast > curr_slow
                and rsi < rsi_overbought
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
                        reasoning=(
                            f"SMA{sma_fast} crossed above SMA{sma_slow} (RSI={rsi:.1f})"
                        ),
                    )
                )

            # Sell signal: SMA fast crosses below slow, or RSI extreme
            elif has_position and (
                (prev_fast >= prev_slow and curr_fast < curr_slow) or rsi > rsi_exit
            ):
                reason = (
                    f"SMA{sma_fast} crossed below SMA{sma_slow}"
                    if curr_fast < curr_slow
                    else f"RSI overbought ({rsi:.1f})"
                )
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

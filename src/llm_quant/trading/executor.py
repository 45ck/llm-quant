"""Paper-trade executor.

Translates ``TradeSignal`` objects (produced by the LLM brain after risk
filtering) into concrete position changes on the in-memory ``Portfolio``.
No real orders are sent – this is a simulation layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_quant.brain.models import Action, TradeSignal
from llm_quant.trading.portfolio import Portfolio

if TYPE_CHECKING:
    from llm_quant.trading.portfolio import Position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Executed trade record
# ---------------------------------------------------------------------------


@dataclass
class ExecutedTrade:
    """Immutable record of a single trade that was applied to the portfolio."""

    symbol: str
    action: str  # "buy" / "sell" / "close"
    shares: float
    price: float
    notional: float  # abs(shares * price)
    conviction: str
    reasoning: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_signals(
    portfolio: Portfolio,
    signals: list[TradeSignal],
    prices: dict[str, float],
    nav: float,
) -> list[ExecutedTrade]:
    """Execute a batch of trade signals against *portfolio*.

    The function mutates *portfolio* in place (cash, positions) and returns
    the list of trades that were actually executed.  Signals that cannot be
    executed (e.g. missing price, zero shares) are skipped with a warning.

    Parameters
    ----------
    portfolio:
        Live portfolio object to mutate.
    signals:
        Trade signals already approved by the risk manager.
    prices:
        Latest prices keyed by symbol.
    nav:
        Portfolio NAV **before** this batch (used for weight calculations).

    Returns
    -------
    list[ExecutedTrade]
        Records of every trade that was applied.
    """
    executed: list[ExecutedTrade] = []

    for signal in signals:
        symbol = signal.symbol
        price = prices.get(symbol)

        if price is None or price <= 0.0:
            logger.warning(
                "Skipping %s %s – no valid price available",
                signal.action.value,
                symbol,
            )
            continue

        trade: ExecutedTrade | None = None

        if signal.action == Action.BUY:
            trade = _execute_buy(portfolio, signal, price, nav)
        elif signal.action == Action.SELL:
            trade = _execute_sell(portfolio, signal, price, nav)
        elif signal.action == Action.CLOSE:
            trade = _execute_close(portfolio, signal, price)
        elif signal.action == Action.HOLD:
            logger.debug("HOLD signal for %s – no trade.", symbol)
            continue
        else:
            logger.warning("Unknown action %s for %s", signal.action, symbol)
            continue

        if trade is not None:
            executed.append(trade)
            logger.info(
                "Executed %s %s: %.4f shares @ %.4f (notional=%.2f)",
                trade.action,
                trade.symbol,
                trade.shares,
                trade.price,
                trade.notional,
            )

    return executed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _execute_buy(
    portfolio: Portfolio,
    signal: TradeSignal,
    price: float,
    nav: float,
) -> ExecutedTrade | None:
    """Buy (or add to) a position."""
    target_notional = signal.target_weight * nav
    current_notional = 0.0

    existing = portfolio.positions.get(signal.symbol)
    if existing is not None:
        current_notional = existing.market_value

    additional_notional = target_notional - current_notional
    if additional_notional <= 0.0:
        logger.debug(
            "BUY %s: already at or above target weight – skipping.",
            signal.symbol,
        )
        return None

    shares_to_buy = math.floor(additional_notional / price)
    if shares_to_buy <= 0:
        logger.debug(
            "BUY %s: computed 0 shares (notional=%.2f, price=%.4f) – skipping.",
            signal.symbol,
            additional_notional,
            price,
        )
        return None

    cost = shares_to_buy * price

    # Ensure we have enough cash
    if cost > portfolio.cash:
        # Buy as many as cash allows
        shares_to_buy = math.floor(portfolio.cash / price)
        if shares_to_buy <= 0:
            logger.warning(
                "BUY %s: insufficient cash (need=%.2f, have=%.2f).",
                signal.symbol,
                cost,
                portfolio.cash,
            )
            return None
        cost = shares_to_buy * price

    # Update portfolio
    portfolio.cash -= cost

    if existing is not None:
        # Weighted average cost
        total_shares = existing.shares + shares_to_buy
        existing.avg_cost = (
            existing.shares * existing.avg_cost + shares_to_buy * price
        ) / total_shares
        existing.shares = total_shares
        existing.current_price = price
        existing.stop_loss = signal.stop_loss
    else:
        portfolio.positions[signal.symbol] = _make_position(
            signal.symbol, shares_to_buy, price, signal.stop_loss
        )

    return ExecutedTrade(
        symbol=signal.symbol,
        action="buy",
        shares=shares_to_buy,
        price=price,
        notional=cost,
        conviction=signal.conviction.value,
        reasoning=signal.reasoning,
    )


def _execute_sell(
    portfolio: Portfolio,
    signal: TradeSignal,
    price: float,
    nav: float,
) -> ExecutedTrade | None:
    """Reduce a position toward a target weight."""
    existing = portfolio.positions.get(signal.symbol)
    if existing is None or existing.shares <= 0:
        logger.warning("SELL %s: no position to sell.", signal.symbol)
        return None

    target_notional = signal.target_weight * nav
    current_notional = existing.shares * price
    reduce_notional = current_notional - target_notional

    if reduce_notional <= 0.0:
        logger.debug(
            "SELL %s: position already at or below target weight.",
            signal.symbol,
        )
        return None

    shares_to_sell = math.floor(reduce_notional / price)
    shares_to_sell = min(shares_to_sell, existing.shares)

    if shares_to_sell <= 0:
        return None

    proceeds = shares_to_sell * price
    existing.shares -= shares_to_sell
    existing.current_price = price
    portfolio.cash += proceeds

    # Remove position if fully liquidated
    if existing.shares <= 0:
        del portfolio.positions[signal.symbol]
    # Update stop-loss if signal provides one
    elif signal.stop_loss > 0.0:
        existing.stop_loss = signal.stop_loss

    return ExecutedTrade(
        symbol=signal.symbol,
        action="sell",
        shares=shares_to_sell,
        price=price,
        notional=proceeds,
        conviction=signal.conviction.value,
        reasoning=signal.reasoning,
    )


def _execute_close(
    portfolio: Portfolio,
    signal: TradeSignal,
    price: float,
) -> ExecutedTrade | None:
    """Close an entire position."""
    existing = portfolio.positions.get(signal.symbol)
    if existing is None or existing.shares <= 0:
        logger.warning("CLOSE %s: no position to close.", signal.symbol)
        return None

    shares_to_close = existing.shares
    proceeds = shares_to_close * price
    portfolio.cash += proceeds
    del portfolio.positions[signal.symbol]

    return ExecutedTrade(
        symbol=signal.symbol,
        action="close",
        shares=shares_to_close,
        price=price,
        notional=proceeds,
        conviction=signal.conviction.value,
        reasoning=signal.reasoning,
    )


def _make_position(
    symbol: str,
    shares: float,
    price: float,
    stop_loss: float,
) -> Position:
    """Create a new ``Position`` dataclass.

    Import is deferred to the function body to avoid circular imports
    (executor imports Portfolio at module level via type hints, but
    Position lives in the same module).
    """
    from llm_quant.trading.portfolio import Position  # local import

    return Position(
        symbol=symbol,
        shares=shares,
        avg_cost=price,
        current_price=price,
        stop_loss=stop_loss,
    )

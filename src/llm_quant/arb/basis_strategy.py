"""Crypto cash-and-carry basis trade strategy.

Implements the core strategy logic for capturing the spot-futures basis
spread. The trade is: long spot + short quarterly futures (1:1 ratio).
At expiry, futures converge to spot, and you pocket the premium minus fees.

Key properties:
  - Market-neutral: P&L comes from convergence, not direction
  - Bounded risk: max loss = fees + adverse funding (minimal)
  - Predictable return: known premium at entry, known convergence at expiry
  - Time-bound: every position has a hard exit at futures expiry

Track C — Niche Arbitrage.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum

from llm_quant.arb.basis_scanner import (
    BasisOpportunity,
    annualize_basis,
    compute_fee_adjusted_basis,
    compute_raw_basis,
)

logger = logging.getLogger(__name__)


class PositionStatus(StrEnum):
    """Status of a cash-and-carry basis position."""

    OPEN = "open"
    CLOSED_EXPIRY = "closed_expiry"  # Closed at futures expiry (convergence)
    CLOSED_SIGNAL = "closed_signal"  # Closed early due to basis going negative
    CLOSED_MANUAL = "closed_manual"  # Manually closed


# Default strategy parameters
DEFAULT_ENTRY_THRESHOLD = 0.05  # 5% annualized basis minimum for entry
DEFAULT_EXIT_NEGATIVE_BASIS = True  # Exit if basis goes negative (backwardation)
DEFAULT_MAKER_FEE = 0.0002  # 0.02% maker fee
DEFAULT_TAKER_FEE = 0.0005  # 0.05% taker fee
DEFAULT_MAX_POSITION_USD = 2000.0  # Track C max per trade


@dataclass
class BasisPosition:
    """A single cash-and-carry basis trade position.

    Represents: long spot + short futures (1:1 ratio).
    """

    position_id: str
    symbol: str  # e.g. "BTC"
    exchange: str
    futures_symbol: str  # e.g. "BTC/USDT:USDT-250627"

    # Entry details
    entry_date: date
    entry_spot_price: float
    entry_futures_price: float
    entry_basis: float  # raw basis at entry
    entry_annualized_basis: float  # annualized basis at entry
    expiry_date: date
    position_size_usd: float  # notional size in USD (one leg)
    quantity: float  # number of units (position_size_usd / entry_spot_price)

    # Current state (updated on each mark-to-market)
    current_spot_price: float | None = None
    current_futures_price: float | None = None
    current_basis: float | None = None
    days_to_expiry: int = 0
    status: PositionStatus = PositionStatus.OPEN

    # Exit details (populated on close)
    exit_date: date | None = None
    exit_spot_price: float | None = None
    exit_futures_price: float | None = None
    exit_reason: str | None = None

    # Fee tracking
    maker_fee: float = DEFAULT_MAKER_FEE
    taker_fee: float = DEFAULT_TAKER_FEE

    @property
    def entry_fees_usd(self) -> float:
        """Total fees paid at entry (buy spot taker + sell futures maker)."""
        return self.position_size_usd * (self.taker_fee + self.maker_fee)

    @property
    def exit_fees_usd(self) -> float:
        """Total fees at exit (sell spot taker, futures settle for free)."""
        return self.position_size_usd * self.taker_fee

    @property
    def total_fees_usd(self) -> float:
        """Total round-trip fees."""
        return self.entry_fees_usd + self.exit_fees_usd

    @property
    def gross_pnl_usd(self) -> float:
        """Gross PnL from basis convergence (before fees).

        At expiry, futures price converges to spot:
          Spot leg: (exit_spot - entry_spot) * quantity
          Futures leg: (entry_futures - exit_futures) * quantity  [short]
          Combined: (entry_futures - entry_spot) * quantity  [if converged]

        Mid-life: use current prices for unrealized.
        """
        if self.status == PositionStatus.OPEN:
            # Unrealized: use current prices
            if self.current_spot_price is None or self.current_futures_price is None:
                return 0.0
            spot_pnl = (self.current_spot_price - self.entry_spot_price) * self.quantity
            futures_pnl = (
                self.entry_futures_price - self.current_futures_price
            ) * self.quantity
            return spot_pnl + futures_pnl
        # Realized: use exit prices
        if self.exit_spot_price is None or self.exit_futures_price is None:
            return 0.0
        spot_pnl = (self.exit_spot_price - self.entry_spot_price) * self.quantity
        futures_pnl = (
            self.entry_futures_price - self.exit_futures_price
        ) * self.quantity
        return spot_pnl + futures_pnl

    @property
    def net_pnl_usd(self) -> float:
        """Net PnL after all fees."""
        return self.gross_pnl_usd - self.total_fees_usd

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN


@dataclass
class BasisStrategyConfig:
    """Configuration for the cash-and-carry basis strategy."""

    # Entry: minimum annualized basis to open a position (decimal, e.g. 0.05 = 5%)
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD

    # Exit: close if basis turns negative (backwardation)
    exit_on_negative_basis: bool = DEFAULT_EXIT_NEGATIVE_BASIS

    # Fees
    maker_fee: float = DEFAULT_MAKER_FEE
    taker_fee: float = DEFAULT_TAKER_FEE

    # Position sizing
    max_position_usd: float = DEFAULT_MAX_POSITION_USD

    # Maximum concurrent positions
    max_positions: int = 5


@dataclass
class BasisStrategy:
    """Cash-and-carry basis trade strategy.

    Entry: when annualized basis > threshold
    Position: long spot + short futures (1:1 ratio)
    Exit: at futures expiry (convergence) or if basis goes negative
    PnL: futures premium captured minus fees
    """

    config: BasisStrategyConfig = field(default_factory=BasisStrategyConfig)
    positions: dict[str, BasisPosition] = field(default_factory=dict)

    @property
    def open_positions(self) -> list[BasisPosition]:
        """Return all currently open positions."""
        return [p for p in self.positions.values() if p.is_open]

    @property
    def closed_positions(self) -> list[BasisPosition]:
        """Return all closed positions."""
        return [p for p in self.positions.values() if not p.is_open]

    def evaluate_entry(
        self,
        opportunity: BasisOpportunity,
    ) -> tuple[bool, str]:
        """Evaluate whether to enter a cash-and-carry trade.

        Returns (should_enter, reason).
        """
        # Check max positions
        if len(self.open_positions) >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"

        # Check if we already have a position in this contract
        for pos in self.open_positions:
            if (
                pos.futures_symbol == opportunity.futures_symbol
                and pos.exchange == opportunity.exchange
            ):
                return False, f"Already have position in {opportunity.futures_symbol}"

        # Check minimum annualized basis
        fee_adjusted = compute_fee_adjusted_basis(
            opportunity.raw_basis,
            self.config.maker_fee,
            self.config.taker_fee,
        )
        ann_fee_adjusted = annualize_basis(fee_adjusted, opportunity.days_to_expiry)

        if ann_fee_adjusted < self.config.entry_threshold:
            return (
                False,
                f"Fee-adjusted annualized basis {ann_fee_adjusted:.2%} "
                f"below threshold {self.config.entry_threshold:.2%}",
            )

        # Check days to expiry
        if opportunity.days_to_expiry < 1:
            return False, "Contract expires today or has expired"

        return (
            True,
            f"Entry signal: {ann_fee_adjusted:.2%} annualized "
            f"(raw={opportunity.annualized_basis:.2%}, "
            f"{opportunity.days_to_expiry}d to expiry)",
        )

    def open_position(
        self,
        opportunity: BasisOpportunity,
        position_size_usd: float | None = None,
    ) -> BasisPosition:
        """Open a new cash-and-carry position from a scanned opportunity.

        Parameters
        ----------
        opportunity:
            The basis opportunity to trade.
        position_size_usd:
            Notional size per leg in USD. Defaults to config max.

        Returns
        -------
        BasisPosition
            The newly created position.

        Raises
        ------
        ValueError
            If entry conditions are not met.
        """
        should_enter, reason = self.evaluate_entry(opportunity)
        if not should_enter:
            raise ValueError(f"Entry rejected: {reason}")

        size = min(
            position_size_usd or self.config.max_position_usd,
            self.config.max_position_usd,
        )
        quantity = size / opportunity.spot_price

        position = BasisPosition(
            position_id=str(uuid.uuid4()),
            symbol=opportunity.symbol,
            exchange=opportunity.exchange,
            futures_symbol=opportunity.futures_symbol,
            entry_date=opportunity.timestamp.date()
            if isinstance(opportunity.timestamp, datetime)
            else datetime.now(UTC).date(),
            entry_spot_price=opportunity.spot_price,
            entry_futures_price=opportunity.futures_price,
            entry_basis=opportunity.raw_basis,
            entry_annualized_basis=opportunity.annualized_basis,
            expiry_date=opportunity.expiry_date,
            position_size_usd=size,
            quantity=quantity,
            current_spot_price=opportunity.spot_price,
            current_futures_price=opportunity.futures_price,
            current_basis=opportunity.raw_basis,
            days_to_expiry=opportunity.days_to_expiry,
            maker_fee=self.config.maker_fee,
            taker_fee=self.config.taker_fee,
        )

        self.positions[position.position_id] = position
        logger.info(
            "Opened basis position %s: %s on %s, "
            "spot=%.2f, futures=%.2f, basis=%.3f%%, size=$%.2f",
            position.position_id[:8],
            position.symbol,
            position.exchange,
            position.entry_spot_price,
            position.entry_futures_price,
            position.entry_basis * 100,
            position.position_size_usd,
        )
        return position

    def mark_to_market(
        self,
        position_id: str,
        current_spot: float,
        current_futures: float,
        as_of: date | None = None,
    ) -> BasisPosition:
        """Update a position with current prices.

        Parameters
        ----------
        position_id:
            ID of the position to update.
        current_spot:
            Current spot price.
        current_futures:
            Current futures price.
        as_of:
            Current date for days-to-expiry calculation.

        Returns
        -------
        BasisPosition
            The updated position.

        Raises
        ------
        KeyError
            If position_id not found.
        ValueError
            If position is not open.
        """
        if position_id not in self.positions:
            raise KeyError(f"Position {position_id} not found")

        position = self.positions[position_id]
        if not position.is_open:
            raise ValueError(
                f"Position {position_id} is not open (status={position.status})"
            )

        if as_of is None:
            as_of = datetime.now(UTC).date()

        position.current_spot_price = current_spot
        position.current_futures_price = current_futures
        position.current_basis = compute_raw_basis(current_spot, current_futures)
        position.days_to_expiry = max(0, (position.expiry_date - as_of).days)

        return position

    def evaluate_exit(
        self,
        position: BasisPosition,
        as_of: date | None = None,
    ) -> tuple[bool, str]:
        """Evaluate whether to exit a position.

        Exit conditions:
        1. Futures expiry reached (convergence)
        2. Basis turns negative (backwardation) — optional

        Returns (should_exit, reason).
        """
        if as_of is None:
            as_of = datetime.now(UTC).date()

        # Check expiry
        if as_of >= position.expiry_date:
            return True, "Futures expired — convergence"

        # Check negative basis
        if (
            self.config.exit_on_negative_basis
            and position.current_basis is not None
            and position.current_basis < 0
        ):
            return (
                True,
                f"Basis turned negative ({position.current_basis:.4%})"
                " — backwardation exit",
            )

        return False, "No exit signal"

    def close_position(
        self,
        position_id: str,
        exit_spot_price: float,
        exit_futures_price: float,
        reason: str | None = None,
        as_of: date | None = None,
    ) -> BasisPosition:
        """Close an open position.

        Parameters
        ----------
        position_id:
            ID of the position to close.
        exit_spot_price:
            Spot price at exit.
        exit_futures_price:
            Futures price at exit (should be ~= spot if at expiry).
        reason:
            Why the position was closed.
        as_of:
            Date of exit.

        Returns
        -------
        BasisPosition
            The closed position.

        Raises
        ------
        KeyError
            If position_id not found.
        ValueError
            If position is not open.
        """
        if position_id not in self.positions:
            raise KeyError(f"Position {position_id} not found")

        position = self.positions[position_id]
        if not position.is_open:
            raise ValueError(
                f"Position {position_id} is not open (status={position.status})"
            )

        if as_of is None:
            as_of = datetime.now(UTC).date()

        # Determine exit status
        if as_of >= position.expiry_date:
            status = PositionStatus.CLOSED_EXPIRY
        elif reason and "backwardation" in reason.lower():
            status = PositionStatus.CLOSED_SIGNAL
        elif reason and "manual" in reason.lower():
            status = PositionStatus.CLOSED_MANUAL
        else:
            status = PositionStatus.CLOSED_SIGNAL

        position.exit_date = as_of
        position.exit_spot_price = exit_spot_price
        position.exit_futures_price = exit_futures_price
        position.exit_reason = reason or f"Closed: {status.value}"
        position.status = status
        position.current_spot_price = exit_spot_price
        position.current_futures_price = exit_futures_price
        position.current_basis = compute_raw_basis(exit_spot_price, exit_futures_price)
        position.days_to_expiry = max(0, (position.expiry_date - as_of).days)

        logger.info(
            "Closed basis position %s: %s on %s, "
            "pnl=$%.2f (gross=$%.2f, fees=$%.2f), reason=%s",
            position.position_id[:8],
            position.symbol,
            position.exchange,
            position.net_pnl_usd,
            position.gross_pnl_usd,
            position.total_fees_usd,
            position.exit_reason,
        )
        return position

    def get_portfolio_summary(self) -> dict:
        """Return a summary of the basis portfolio.

        Returns dict with aggregate metrics.
        """
        open_pos = self.open_positions
        closed_pos = self.closed_positions

        total_open_notional = sum(p.position_size_usd for p in open_pos)
        total_unrealized_pnl = sum(p.net_pnl_usd for p in open_pos)
        total_realized_pnl = sum(p.net_pnl_usd for p in closed_pos)
        total_fees = sum(p.total_fees_usd for p in self.positions.values())

        # Average basis for open positions
        open_bases = [p.entry_annualized_basis for p in open_pos]
        avg_open_basis = sum(open_bases) / len(open_bases) if open_bases else 0.0

        # Win rate for closed positions
        wins = sum(1 for p in closed_pos if p.net_pnl_usd > 0)
        win_rate = wins / len(closed_pos) if closed_pos else 0.0

        return {
            "open_positions": len(open_pos),
            "closed_positions": len(closed_pos),
            "total_positions": len(self.positions),
            "total_open_notional_usd": total_open_notional,
            "unrealized_pnl_usd": total_unrealized_pnl,
            "realized_pnl_usd": total_realized_pnl,
            "total_pnl_usd": total_unrealized_pnl + total_realized_pnl,
            "total_fees_usd": total_fees,
            "avg_open_annualized_basis": avg_open_basis,
            "win_rate": win_rate,
        }

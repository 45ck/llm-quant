"""Execution bridge: converts aggregated signals into paper trades per pod.

Takes aggregated target allocations from ``SignalAggregator``, diffs them
against current positions in each track's pod, generates trade orders, runs
them through the ``RiskManager`` pre-trade gate, and executes via the paper
trading executor.

Usage::

    bridge = ExecutionBridge(
        router=router,
        aggregator=aggregator,
        risk_manager=risk_mgr,
        portfolios={"track-a": portfolio_a, "track-b": portfolio_b},
        prices=latest_prices,
    )
    summary = bridge.execute_all_tracks()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.risk.manager import RiskManager
from llm_quant.trading.executor import ExecutedTrade, execute_signals
from llm_quant.trading.portfolio import Portfolio
from llm_quant.trading.signal_aggregator import SignalAggregator, TrackSignal
from llm_quant.trading.track_router import TrackRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Track-letter mapping: track_id (from YAML) -> risk-manager track code
# ---------------------------------------------------------------------------

_TRACK_ID_TO_RISK_TRACK: dict[str, str] = {
    "track-a": "A",
    "track-b": "B",
    "track-c": "C",
    "track-d": "A",  # Track D uses Track A limits as base; see _get_stop_loss_pct
    "discretionary": "A",
}

# Default stop-loss percentages per risk track
_DEFAULT_STOP_LOSS_PCT: dict[str, float] = {
    "A": 0.05,
    "B": 0.08,
    "C": 0.05,
    "D": 0.05,
}

# Default conviction for aggregated signals
_DEFAULT_CONVICTION = Conviction.MEDIUM


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RebalanceOrder:
    """A proposed order to move from current to target allocation."""

    symbol: str
    action: Action
    current_weight: float
    target_weight: float
    delta_weight: float
    reasoning: str


@dataclass
class TrackExecutionResult:
    """Result of executing rebalance orders for a single track."""

    track_id: str
    pod_id: str
    risk_track: str
    n_strategies_active: int
    n_strategies_total: int
    target_allocations: dict[str, float]
    proposed_orders: list[RebalanceOrder]
    approved_signals: list[TradeSignal] = field(default_factory=list)
    rejected_signals: list[dict[str, Any]] = field(default_factory=list)
    executed_trades: list[ExecutedTrade] = field(default_factory=list)
    dry_run: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# ExecutionBridge
# ---------------------------------------------------------------------------


class ExecutionBridge:
    """Bridges aggregated signals to paper trade execution per pod.

    Parameters
    ----------
    router:
        ``TrackRouter`` for strategy-to-track mapping and track metadata.
    aggregator:
        ``SignalAggregator`` that provides per-track target allocations.
    risk_manager:
        ``RiskManager`` for pre-trade risk checks.
    portfolios:
        Mapping of ``pod_id`` to ``Portfolio`` instances.  Each track
        operates on its own pod.
    prices:
        Latest market prices keyed by symbol.  Shared across all tracks.
    atrs:
        Optional mapping of symbol to current ATR value for volatility
        sizing and ATR-calibrated stop-loss checks.
    """

    def __init__(
        self,
        router: TrackRouter,
        aggregator: SignalAggregator,
        risk_manager: RiskManager,
        portfolios: dict[str, Portfolio],
        prices: dict[str, float],
        atrs: dict[str, float] | None = None,
    ) -> None:
        self.router = router
        self.aggregator = aggregator
        self.risk_manager = risk_manager
        self.portfolios = portfolios
        self.prices = prices
        self.atrs = atrs

    # ------------------------------------------------------------------
    # Order computation
    # ------------------------------------------------------------------

    def compute_rebalance_orders(self, track_id: str) -> list[RebalanceOrder]:
        """Diff current positions vs target allocations for a track.

        Returns a list of ``RebalanceOrder`` objects representing the
        trades needed to move from the current portfolio state to the
        aggregated signal targets.  Sells are ordered before buys so
        that cash is freed up before new positions are opened.

        Parameters
        ----------
        track_id:
            The track identifier (e.g. ``"track-a"``).

        Returns
        -------
        list[RebalanceOrder]
            Proposed orders sorted with sells first, then buys.
        """
        track_signal: TrackSignal = self.aggregator.aggregate_track(track_id)
        target_allocs = track_signal.net_allocations

        track_info = self.router.get_track_info(track_id)
        pod_id = track_info.pod_id
        portfolio = self.portfolios.get(pod_id)

        if portfolio is None:
            logger.warning(
                "No portfolio found for pod_id='%s' (track '%s') "
                "-- cannot compute rebalance orders.",
                pod_id,
                track_id,
            )
            return []

        nav = portfolio.nav
        if nav <= 0:
            logger.warning(
                "Portfolio NAV is %.2f for pod '%s' -- skipping rebalance.",
                nav,
                pod_id,
            )
            return []

        orders: list[RebalanceOrder] = []

        # 1. Identify symbols that need to be sold or closed
        #    (currently held but not in target, or over-allocated)
        all_symbols = set(portfolio.positions.keys()) | set(target_allocs.keys())

        for symbol in sorted(all_symbols):
            current_weight = portfolio.get_position_weight(symbol)
            target_weight = target_allocs.get(symbol, 0.0)
            delta = target_weight - current_weight

            # Skip negligible changes (< 0.5% of NAV)
            if abs(delta) < 0.005:
                logger.debug(
                    "Skipping %s: delta_weight=%.4f (below 0.5%% threshold)",
                    symbol,
                    delta,
                )
                continue

            if delta < 0:
                # Need to reduce position
                if target_weight <= 0.001:
                    action = Action.CLOSE
                    reasoning = (
                        f"Close {symbol}: target allocation is 0, "
                        f"current weight {current_weight:.2%}"
                    )
                else:
                    action = Action.SELL
                    reasoning = (
                        f"Reduce {symbol}: current {current_weight:.2%} "
                        f"-> target {target_weight:.2%} "
                        f"(delta {delta:+.2%})"
                    )
            else:
                # Need to increase position
                action = Action.BUY
                reasoning = (
                    f"{'Open' if current_weight < 0.001 else 'Add to'} "
                    f"{symbol}: current {current_weight:.2%} "
                    f"-> target {target_weight:.2%} "
                    f"(delta {delta:+.2%})"
                )

            orders.append(
                RebalanceOrder(
                    symbol=symbol,
                    action=action,
                    current_weight=round(current_weight, 6),
                    target_weight=round(target_weight, 6),
                    delta_weight=round(delta, 6),
                    reasoning=reasoning,
                )
            )

        # Sort: sells/closes first (free cash), then buys
        action_priority = {Action.CLOSE: 0, Action.SELL: 1, Action.BUY: 2}
        orders.sort(key=lambda o: action_priority.get(o.action, 99))

        logger.info(
            "Computed %d rebalance orders for track '%s' (pod='%s', NAV=%.2f)",
            len(orders),
            track_id,
            pod_id,
            nav,
        )
        return orders

    # ------------------------------------------------------------------
    # Single-track execution
    # ------------------------------------------------------------------

    def execute_track(
        self,
        track_id: str,
        *,
        dry_run: bool = False,
    ) -> TrackExecutionResult:
        """Full rebalance cycle for a single track.

        1. Get aggregated signal (target allocations)
        2. Compute rebalance orders (diff current vs target)
        3. Convert orders to ``TradeSignal`` objects
        4. Pass through ``RiskManager.filter_signals``
        5. Execute approved signals via ``execute_signals``

        Parameters
        ----------
        track_id:
            The track identifier (e.g. ``"track-a"``).
        dry_run:
            If True, return proposed orders without executing.

        Returns
        -------
        TrackExecutionResult
            Full summary of what was proposed, approved, rejected,
            and executed.
        """
        track_info = self.router.get_track_info(track_id)
        pod_id = track_info.pod_id
        risk_track = _TRACK_ID_TO_RISK_TRACK.get(track_id, "A")

        # Build result scaffold
        result = TrackExecutionResult(
            track_id=track_id,
            pod_id=pod_id,
            risk_track=risk_track,
            n_strategies_active=0,
            n_strategies_total=0,
            target_allocations={},
            proposed_orders=[],
            dry_run=dry_run,
        )

        # Get aggregated signal
        track_signal = self.aggregator.aggregate_track(track_id)
        result.n_strategies_active = track_signal.n_active_strategies
        result.n_strategies_total = track_signal.n_total_strategies
        result.target_allocations = dict(track_signal.net_allocations)

        # Get portfolio
        portfolio = self.portfolios.get(pod_id)
        if portfolio is None:
            result.error = f"No portfolio for pod_id='{pod_id}'"
            logger.error(result.error)
            return result

        # Update portfolio prices before computing orders
        portfolio.update_prices(self.prices)

        # Compute rebalance orders
        orders = self.compute_rebalance_orders(track_id)
        result.proposed_orders = orders

        if not orders:
            logger.info(
                "No rebalance orders for track '%s' -- portfolio already at target.",
                track_id,
            )
            return result

        # Convert orders to TradeSignal objects
        signals = self._orders_to_signals(orders, risk_track)

        if dry_run:
            logger.info(
                "DRY RUN: %d proposed signals for track '%s' (not executing).",
                len(signals),
                track_id,
            )
            result.approved_signals = signals
            return result

        # Run through risk manager
        approved, rejected = self.risk_manager.filter_signals(
            signals=signals,
            portfolio=portfolio,
            prices=self.prices,
            atrs=self.atrs,
            track=risk_track,
        )
        result.approved_signals = approved
        result.rejected_signals = [
            {
                "signal": {
                    "symbol": sig.symbol,
                    "action": sig.action.value,
                    "target_weight": sig.target_weight,
                },
                "failures": [
                    {"rule": c.rule, "message": c.message}
                    for c in checks
                    if not c.passed
                ],
            }
            for sig, checks in rejected
        ]

        if not approved:
            logger.warning(
                "All %d signals rejected by risk manager for track '%s'.",
                len(signals),
                track_id,
            )
            return result

        # Filter out HOLD signals (they don't produce trades)
        tradeable = [s for s in approved if s.action != Action.HOLD]

        if not tradeable:
            logger.info(
                "No tradeable signals after risk filter for track '%s'.",
                track_id,
            )
            return result

        # Execute
        nav = portfolio.nav
        executed = execute_signals(
            portfolio=portfolio,
            signals=tradeable,
            prices=self.prices,
            nav=nav,
        )
        result.executed_trades = executed

        logger.info(
            "Executed %d/%d trades for track '%s' (pod='%s')",
            len(executed),
            len(tradeable),
            track_id,
            pod_id,
        )
        return result

    # ------------------------------------------------------------------
    # All-track execution
    # ------------------------------------------------------------------

    def execute_all_tracks(
        self,
        *,
        dry_run: bool = False,
    ) -> dict[str, TrackExecutionResult]:
        """Execute rebalance cycle for all tracks.

        Parameters
        ----------
        dry_run:
            If True, return proposed orders without executing.

        Returns
        -------
        dict[str, TrackExecutionResult]
            Mapping of track_id to execution result.
        """
        results: dict[str, TrackExecutionResult] = {}

        for track_id in self.router.track_ids:
            logger.info("--- Executing track '%s' ---", track_id)
            results[track_id] = self.execute_track(track_id, dry_run=dry_run)

        # Summary log
        total_executed = sum(len(r.executed_trades) for r in results.values())
        total_rejected = sum(len(r.rejected_signals) for r in results.values())
        logger.info(
            "Execution bridge complete: %d tracks, %d trades executed, "
            "%d signals rejected.",
            len(results),
            total_executed,
            total_rejected,
        )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _orders_to_signals(
        self,
        orders: list[RebalanceOrder],
        risk_track: str,
    ) -> list[TradeSignal]:
        """Convert ``RebalanceOrder`` objects to ``TradeSignal`` objects.

        Applies track-appropriate default stop-loss percentages for
        buy orders.  Sell and close orders get stop_loss=0.0 (not
        applicable).
        """
        signals: list[TradeSignal] = []
        stop_pct = _DEFAULT_STOP_LOSS_PCT.get(risk_track, 0.05)

        for order in orders:
            price = self.prices.get(order.symbol, 0.0)

            # Compute stop-loss for buy orders
            if order.action == Action.BUY and price > 0:
                stop_loss = round(price * (1.0 - stop_pct), 4)
            else:
                stop_loss = 0.0

            signals.append(
                TradeSignal(
                    symbol=order.symbol,
                    action=order.action,
                    conviction=_DEFAULT_CONVICTION,
                    target_weight=order.target_weight,
                    stop_loss=stop_loss,
                    reasoning=order.reasoning,
                )
            )

        return signals

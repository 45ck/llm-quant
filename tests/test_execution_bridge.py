"""Tests for the execution bridge module.

Covers:
- RebalanceOrder computation (diff current vs target)
- Signal conversion with track-appropriate stop-loss
- Risk manager integration (approved + rejected paths)
- Dry-run mode
- Single-track and all-track execution
- Edge cases: empty portfolios, missing prices, no orders needed
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_quant.brain.models import Action, Conviction
from llm_quant.risk.limits import RiskCheckResult
from llm_quant.trading.execution_bridge import (
    _DEFAULT_STOP_LOSS_PCT,
    _TRACK_ID_TO_RISK_TRACK,
    ExecutionBridge,
    RebalanceOrder,
    TrackExecutionResult,
)
from llm_quant.trading.portfolio import Portfolio, Position
from llm_quant.trading.signal_aggregator import SignalAggregator, TrackSignal
from llm_quant.trading.track_router import StrategyEntry, TrackInfo, TrackRouter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def track_router() -> TrackRouter:
    """Minimal TrackRouter with two tracks (track-a, track-b)."""
    tracks = {
        "track-a": TrackInfo(
            pod_id="track-a",
            display_name="Track A",
            benchmark="60/40 SPY/TLT",
            initial_capital=100_000.0,
            target_allocation_pct=70.0,
            strategies=[
                StrategyEntry(
                    slug="strat-a1",
                    family="F1",
                    mechanism="credit_lead",
                    description="Strategy A1",
                ),
                StrategyEntry(
                    slug="strat-a2",
                    family="F2",
                    mechanism="gold_silver",
                    description="Strategy A2",
                ),
            ],
        ),
        "track-b": TrackInfo(
            pod_id="track-b",
            display_name="Track B",
            benchmark="100% SPY",
            initial_capital=100_000.0,
            target_allocation_pct=30.0,
            strategies=[
                StrategyEntry(
                    slug="strat-b1",
                    family="F5",
                    mechanism="soxx_qqq",
                    description="Strategy B1",
                ),
            ],
        ),
    }
    return TrackRouter(tracks)


@pytest.fixture
def portfolios() -> dict[str, Portfolio]:
    """Fresh portfolios for each track pod."""
    return {
        "track-a": Portfolio(initial_capital=100_000.0, pod_id="track-a"),
        "track-b": Portfolio(initial_capital=100_000.0, pod_id="track-b"),
    }


@pytest.fixture
def prices() -> dict[str, float]:
    """Sample market prices."""
    return {
        "SPY": 500.0,
        "TLT": 90.0,
        "LQD": 110.0,
        "GLD": 200.0,
        "QQQ": 450.0,
    }


@pytest.fixture
def track_a_signal() -> TrackSignal:
    """Aggregated signal for track-a: wants SPY=0.4, TLT=0.3."""
    return TrackSignal(
        track_id="track-a",
        date="2026-04-06",
        n_active_strategies=2,
        n_total_strategies=2,
        net_allocations={"SPY": 0.40, "TLT": 0.30},
    )


@pytest.fixture
def track_b_signal() -> TrackSignal:
    """Aggregated signal for track-b: wants QQQ=0.60."""
    return TrackSignal(
        track_id="track-b",
        date="2026-04-06",
        n_active_strategies=1,
        n_total_strategies=1,
        net_allocations={"QQQ": 0.60},
    )


@pytest.fixture
def mock_aggregator(
    track_a_signal: TrackSignal,
    track_b_signal: TrackSignal,
) -> MagicMock:
    """Mock SignalAggregator that returns pre-built track signals."""
    agg = MagicMock(spec=SignalAggregator)
    signal_map = {
        "track-a": track_a_signal,
        "track-b": track_b_signal,
    }
    agg.aggregate_track.side_effect = lambda tid: signal_map[tid]
    agg.aggregate_all_tracks.return_value = signal_map
    return agg


@pytest.fixture
def mock_risk_manager() -> MagicMock:
    """Mock RiskManager that approves all signals by default."""
    rm = MagicMock()
    # Default: approve everything, reject nothing
    rm.filter_signals.side_effect = lambda signals, **_kw: (signals, [])
    return rm


@pytest.fixture
def bridge(
    track_router: TrackRouter,
    mock_aggregator: MagicMock,
    mock_risk_manager: MagicMock,
    portfolios: dict[str, Portfolio],
    prices: dict[str, float],
) -> ExecutionBridge:
    """ExecutionBridge wired up with mocks."""
    return ExecutionBridge(
        router=track_router,
        aggregator=mock_aggregator,
        risk_manager=mock_risk_manager,
        portfolios=portfolios,
        prices=prices,
    )


# ---------------------------------------------------------------------------
# Tests: compute_rebalance_orders
# ---------------------------------------------------------------------------


class TestComputeRebalanceOrders:
    """Tests for order diff computation."""

    def test_empty_portfolio_generates_buy_orders(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """From a fresh portfolio, all target allocations should generate buys."""
        orders = bridge.compute_rebalance_orders("track-a")

        # Should have 2 buys (SPY, TLT)
        buy_orders = [o for o in orders if o.action == Action.BUY]
        assert len(buy_orders) == 2

        symbols = {o.symbol for o in buy_orders}
        assert symbols == {"SPY", "TLT"}

        # SPY target = 0.40
        spy_order = next(o for o in buy_orders if o.symbol == "SPY")
        assert spy_order.target_weight == pytest.approx(0.40, abs=1e-4)
        assert spy_order.delta_weight > 0

    def test_at_target_generates_no_orders(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """When positions already match target, no orders should be generated."""
        # Set up portfolio to match target allocations
        p = portfolios["track-a"]
        # NAV = 100k. SPY=0.40 -> $40k, TLT=0.30 -> $30k, cash=$30k
        p.cash = 30_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=80.0,
            avg_cost=500.0,
            current_price=500.0,
        )  # 80*500 = 40k
        p.positions["TLT"] = Position(
            symbol="TLT",
            shares=333.33,
            avg_cost=90.0,
            current_price=90.0,
        )  # ~30k

        orders = bridge.compute_rebalance_orders("track-a")
        assert len(orders) == 0

    def test_overweight_generates_sell_orders(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """When position exceeds target, should generate a sell order."""
        p = portfolios["track-a"]
        # SPY at 60% weight (target is 40%) -> needs sell
        p.cash = 10_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=120.0,
            avg_cost=500.0,
            current_price=500.0,
        )  # 60k = 60% of ~100k NAV
        p.positions["TLT"] = Position(
            symbol="TLT",
            shares=333.33,
            avg_cost=90.0,
            current_price=90.0,
        )  # ~30k

        orders = bridge.compute_rebalance_orders("track-a")
        sell_orders = [o for o in orders if o.action == Action.SELL]
        assert len(sell_orders) == 1
        assert sell_orders[0].symbol == "SPY"
        assert sell_orders[0].delta_weight < 0

    def test_unwanted_position_generates_close_order(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """Position not in target should generate a close order."""
        p = portfolios["track-a"]
        # Hold GLD which is not in track-a target allocations
        p.cash = 80_000.0
        p.positions["GLD"] = Position(
            symbol="GLD",
            shares=100.0,
            avg_cost=200.0,
            current_price=200.0,
        )  # 20k = 20% of 100k NAV

        orders = bridge.compute_rebalance_orders("track-a")
        close_orders = [o for o in orders if o.action == Action.CLOSE]
        assert len(close_orders) == 1
        assert close_orders[0].symbol == "GLD"
        assert close_orders[0].target_weight == pytest.approx(0.0, abs=1e-4)

    def test_sells_ordered_before_buys(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """Sells and closes should come before buys to free up cash."""
        p = portfolios["track-a"]
        # Hold GLD (unwanted) and no SPY/TLT (needed)
        p.cash = 80_000.0
        p.positions["GLD"] = Position(
            symbol="GLD",
            shares=100.0,
            avg_cost=200.0,
            current_price=200.0,
        )

        orders = bridge.compute_rebalance_orders("track-a")
        actions = [o.action for o in orders]
        # All CLOSE/SELL indices should be less than all BUY indices
        sell_close = (Action.CLOSE, Action.SELL)
        sell_indices = [i for i, a in enumerate(actions) if a in sell_close]
        buy_indices = [i for i, a in enumerate(actions) if a == Action.BUY]
        assert len(sell_indices) > 0, "Expected at least one sell/close"
        assert len(buy_indices) > 0, "Expected at least one buy"
        assert max(sell_indices) < min(buy_indices)

    def test_small_delta_skipped(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """Deltas below 0.5% threshold should be skipped."""
        p = portfolios["track-a"]
        # SPY at 39.8% (target 40%) -> delta = 0.2% < 0.5% threshold
        p.cash = 30_100.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=79.6,
            avg_cost=500.0,
            current_price=500.0,
        )  # 39.8k ~ 39.8% of ~100k
        p.positions["TLT"] = Position(
            symbol="TLT",
            shares=333.33,
            avg_cost=90.0,
            current_price=90.0,
        )  # ~30k

        orders = bridge.compute_rebalance_orders("track-a")
        spy_orders = [o for o in orders if o.symbol == "SPY"]
        assert len(spy_orders) == 0  # delta too small

    def test_missing_portfolio_returns_empty(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """If no portfolio exists for the pod, return empty list."""
        # Remove track-b portfolio
        del bridge.portfolios["track-b"]
        orders = bridge.compute_rebalance_orders("track-b")
        assert orders == []


# ---------------------------------------------------------------------------
# Tests: _orders_to_signals
# ---------------------------------------------------------------------------


class TestOrdersToSignals:
    """Tests for converting orders to TradeSignal objects."""

    def test_buy_order_gets_stop_loss(
        self,
        bridge: ExecutionBridge,
        prices: dict[str, float],
    ) -> None:
        """Buy orders should have a default stop-loss based on track."""
        orders = [
            RebalanceOrder(
                symbol="SPY",
                action=Action.BUY,
                current_weight=0.0,
                target_weight=0.40,
                delta_weight=0.40,
                reasoning="Open SPY",
            ),
        ]
        signals = bridge._orders_to_signals(orders, "A")
        assert len(signals) == 1
        sig = signals[0]
        assert sig.action == Action.BUY
        assert sig.stop_loss == pytest.approx(
            prices["SPY"] * (1 - _DEFAULT_STOP_LOSS_PCT["A"]),
            abs=0.01,
        )

    def test_sell_order_no_stop_loss(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Sell and close orders should have stop_loss=0.0."""
        orders = [
            RebalanceOrder(
                symbol="SPY",
                action=Action.SELL,
                current_weight=0.60,
                target_weight=0.40,
                delta_weight=-0.20,
                reasoning="Reduce SPY",
            ),
            RebalanceOrder(
                symbol="GLD",
                action=Action.CLOSE,
                current_weight=0.10,
                target_weight=0.0,
                delta_weight=-0.10,
                reasoning="Close GLD",
            ),
        ]
        signals = bridge._orders_to_signals(orders, "A")
        for sig in signals:
            assert sig.stop_loss == 0.0

    def test_track_b_stop_loss_wider(
        self,
        bridge: ExecutionBridge,
        prices: dict[str, float],
    ) -> None:
        """Track B should use wider default stop-loss (8% vs 5%)."""
        orders = [
            RebalanceOrder(
                symbol="QQQ",
                action=Action.BUY,
                current_weight=0.0,
                target_weight=0.60,
                delta_weight=0.60,
                reasoning="Open QQQ",
            ),
        ]
        signals_a = bridge._orders_to_signals(orders, "A")
        signals_b = bridge._orders_to_signals(orders, "B")

        assert signals_a[0].stop_loss > signals_b[0].stop_loss  # B has wider stop

    def test_conviction_is_medium(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """All aggregated signals should have MEDIUM conviction."""
        orders = [
            RebalanceOrder(
                symbol="SPY",
                action=Action.BUY,
                current_weight=0.0,
                target_weight=0.40,
                delta_weight=0.40,
                reasoning="Open SPY",
            ),
        ]
        signals = bridge._orders_to_signals(orders, "A")
        assert signals[0].conviction == Conviction.MEDIUM


# ---------------------------------------------------------------------------
# Tests: execute_track
# ---------------------------------------------------------------------------


class TestExecuteTrack:
    """Tests for single-track execution."""

    def test_execute_track_fresh_portfolio(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Execute on a fresh portfolio should produce buy trades."""
        result = bridge.execute_track("track-a")

        assert isinstance(result, TrackExecutionResult)
        assert result.track_id == "track-a"
        assert result.pod_id == "track-a"
        assert result.risk_track == "A"
        assert result.n_strategies_active == 2
        assert result.target_allocations == {"SPY": 0.40, "TLT": 0.30}
        assert len(result.proposed_orders) == 2
        assert len(result.executed_trades) > 0
        assert result.error is None
        assert not result.dry_run

    def test_execute_track_dry_run(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Dry run should propose orders but not execute."""
        result = bridge.execute_track("track-a", dry_run=True)

        assert result.dry_run is True
        assert len(result.proposed_orders) == 2
        # Dry run puts signals in approved_signals but does not execute
        assert len(result.approved_signals) == 2
        assert len(result.executed_trades) == 0

    def test_execute_track_risk_rejection(
        self,
        bridge: ExecutionBridge,
        mock_risk_manager: MagicMock,
    ) -> None:
        """When risk manager rejects signals, they should not execute."""
        # Configure risk manager to reject all signals
        mock_risk_manager.filter_signals.side_effect = lambda signals, **_kw: (
            [],
            [
                (
                    sig,
                    [
                        RiskCheckResult(
                            passed=False,
                            rule="position_size",
                            message="Trade too large",
                        )
                    ],
                )
                for sig in signals
            ],
        )

        result = bridge.execute_track("track-a")

        assert len(result.approved_signals) == 0
        assert len(result.rejected_signals) == 2
        assert len(result.executed_trades) == 0

    def test_execute_track_missing_portfolio(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Should return error result when portfolio is missing."""
        del bridge.portfolios["track-b"]
        result = bridge.execute_track("track-b")

        assert result.error is not None
        assert "No portfolio" in result.error
        assert len(result.executed_trades) == 0

    def test_execute_track_updates_prices(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """Portfolio prices should be updated before computing orders."""
        p = portfolios["track-a"]
        p.cash = 60_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=80.0,
            avg_cost=400.0,
            current_price=400.0,  # stale price
        )

        bridge.execute_track("track-a")

        # Price should be updated to current
        assert p.positions.get("SPY") is None or (
            p.positions["SPY"].current_price == 500.0
        )

    def test_execute_track_no_orders_needed(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """When portfolio is already at target, no trades should execute."""
        p = portfolios["track-a"]
        p.cash = 30_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=80.0,
            avg_cost=500.0,
            current_price=500.0,
        )
        p.positions["TLT"] = Position(
            symbol="TLT",
            shares=333.33,
            avg_cost=90.0,
            current_price=90.0,
        )

        result = bridge.execute_track("track-a")
        assert len(result.proposed_orders) == 0
        assert len(result.executed_trades) == 0

    def test_execute_track_partial_rejection(
        self,
        bridge: ExecutionBridge,
        mock_risk_manager: MagicMock,
    ) -> None:
        """When risk manager rejects some signals, only approved ones execute."""

        def partial_filter(signals, **_kw):
            # Approve first signal, reject rest
            if len(signals) > 1:
                return (
                    [signals[0]],
                    [
                        (
                            sig,
                            [
                                RiskCheckResult(
                                    passed=False,
                                    rule="sector_concentration",
                                    message="Sector too concentrated",
                                )
                            ],
                        )
                        for sig in signals[1:]
                    ],
                )
            return (signals, [])

        mock_risk_manager.filter_signals.side_effect = partial_filter

        result = bridge.execute_track("track-a")

        assert len(result.approved_signals) == 1
        assert len(result.rejected_signals) >= 1
        # At least one trade should have executed
        assert len(result.executed_trades) >= 1


# ---------------------------------------------------------------------------
# Tests: execute_all_tracks
# ---------------------------------------------------------------------------


class TestExecuteAllTracks:
    """Tests for multi-track execution."""

    def test_executes_all_tracks(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Should execute for every track in the router."""
        results = bridge.execute_all_tracks()

        assert set(results.keys()) == {"track-a", "track-b"}
        for track_id, result in results.items():
            assert result.track_id == track_id
            assert isinstance(result, TrackExecutionResult)

    def test_dry_run_all_tracks(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Dry run should apply to all tracks."""
        results = bridge.execute_all_tracks(dry_run=True)

        for result in results.values():
            assert result.dry_run is True
            assert len(result.executed_trades) == 0

    def test_one_track_fails_others_continue(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """If one track has no portfolio, others should still execute."""
        del bridge.portfolios["track-b"]

        results = bridge.execute_all_tracks()

        assert results["track-b"].error is not None
        assert results["track-a"].error is None
        assert len(results["track-a"].executed_trades) > 0


# ---------------------------------------------------------------------------
# Tests: track routing correctness
# ---------------------------------------------------------------------------


class TestTrackRouting:
    """Ensure correct risk-track mapping per track_id."""

    def test_track_id_to_risk_track_mapping(self) -> None:
        """Verify all expected mappings exist."""
        assert _TRACK_ID_TO_RISK_TRACK["track-a"] == "A"
        assert _TRACK_ID_TO_RISK_TRACK["track-b"] == "B"
        assert _TRACK_ID_TO_RISK_TRACK["track-c"] == "C"
        assert _TRACK_ID_TO_RISK_TRACK["track-d"] == "A"
        assert _TRACK_ID_TO_RISK_TRACK["discretionary"] == "A"

    def test_risk_manager_called_with_correct_track(
        self,
        bridge: ExecutionBridge,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Risk manager should be called with the right track letter."""
        bridge.execute_track("track-a")
        call_kwargs = mock_risk_manager.filter_signals.call_args
        assert call_kwargs.kwargs.get("track") == "A" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "A"
        )

    def test_track_b_uses_b_risk_limits(
        self,
        bridge: ExecutionBridge,
        mock_risk_manager: MagicMock,
    ) -> None:
        """Track B should pass track='B' to the risk manager."""
        bridge.execute_track("track-b")
        call_kwargs = mock_risk_manager.filter_signals.call_args
        assert call_kwargs.kwargs.get("track") == "B" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "B"
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case coverage."""

    def test_zero_nav_portfolio(
        self,
        bridge: ExecutionBridge,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """Portfolio with zero NAV should not produce orders."""
        p = portfolios["track-a"]
        p.cash = 0.0

        orders = bridge.compute_rebalance_orders("track-a")
        assert orders == []

    def test_empty_target_allocations(
        self,
        bridge: ExecutionBridge,
        mock_aggregator: MagicMock,
    ) -> None:
        """When aggregator returns no allocations, no orders needed."""
        empty_signal = TrackSignal(
            track_id="track-a",
            date="2026-04-06",
            n_active_strategies=0,
            n_total_strategies=2,
            net_allocations={},
        )
        mock_aggregator.aggregate_track.side_effect = None
        mock_aggregator.aggregate_track.return_value = empty_signal

        orders = bridge.compute_rebalance_orders("track-a")
        assert orders == []

    def test_empty_target_with_existing_positions(
        self,
        bridge: ExecutionBridge,
        mock_aggregator: MagicMock,
        portfolios: dict[str, Portfolio],
    ) -> None:
        """When target is empty but positions exist, should generate closes."""
        p = portfolios["track-a"]
        p.cash = 80_000.0
        p.positions["SPY"] = Position(
            symbol="SPY",
            shares=40.0,
            avg_cost=500.0,
            current_price=500.0,
        )

        empty_signal = TrackSignal(
            track_id="track-a",
            date="2026-04-06",
            n_active_strategies=0,
            n_total_strategies=2,
            net_allocations={},
        )
        mock_aggregator.aggregate_track.side_effect = None
        mock_aggregator.aggregate_track.return_value = empty_signal

        orders = bridge.compute_rebalance_orders("track-a")
        close_orders = [o for o in orders if o.action == Action.CLOSE]
        assert len(close_orders) == 1
        assert close_orders[0].symbol == "SPY"

    def test_atrs_forwarded_to_risk_manager(
        self,
        track_router: TrackRouter,
        mock_aggregator: MagicMock,
        mock_risk_manager: MagicMock,
        portfolios: dict[str, Portfolio],
        prices: dict[str, float],
    ) -> None:
        """ATR data should be forwarded to the risk manager."""
        atrs = {"SPY": 5.0, "TLT": 1.5}
        bridge = ExecutionBridge(
            router=track_router,
            aggregator=mock_aggregator,
            risk_manager=mock_risk_manager,
            portfolios=portfolios,
            prices=prices,
            atrs=atrs,
        )

        bridge.execute_track("track-a")
        call_kwargs = mock_risk_manager.filter_signals.call_args
        assert call_kwargs.kwargs.get("atrs") == atrs

    def test_missing_price_in_stop_loss(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """Buy order for symbol with missing price should get stop_loss=0."""
        orders = [
            RebalanceOrder(
                symbol="UNKNOWN",
                action=Action.BUY,
                current_weight=0.0,
                target_weight=0.10,
                delta_weight=0.10,
                reasoning="Buy unknown",
            ),
        ]
        signals = bridge._orders_to_signals(orders, "A")
        assert signals[0].stop_loss == 0.0

    def test_result_dataclass_fields(
        self,
        bridge: ExecutionBridge,
    ) -> None:
        """TrackExecutionResult should have all expected fields."""
        result = bridge.execute_track("track-a")
        assert hasattr(result, "track_id")
        assert hasattr(result, "pod_id")
        assert hasattr(result, "risk_track")
        assert hasattr(result, "n_strategies_active")
        assert hasattr(result, "n_strategies_total")
        assert hasattr(result, "target_allocations")
        assert hasattr(result, "proposed_orders")
        assert hasattr(result, "approved_signals")
        assert hasattr(result, "rejected_signals")
        assert hasattr(result, "executed_trades")
        assert hasattr(result, "dry_run")
        assert hasattr(result, "error")

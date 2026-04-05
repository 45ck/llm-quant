"""PM-02 Viral Claim Reconstruction — main orchestrator."""

from __future__ import annotations

import logging

from llm_quant.arb.research.compounder import MonteCarloCompounder
from llm_quant.arb.research.fill_simulator import FillSimulator
from llm_quant.arb.research.path_models import CompoundingPath

logger = logging.getLogger(__name__)


class ReconstructionEngine:
    """PM-02 Viral Claim Reconstruction.

    Tests plausibility of extreme PM compounding claims
    by reconstructing the required conditions under each
    hypothesis and stress-testing them.
    """

    def __init__(
        self,
        bankroll_start: float,
        bankroll_target: float,
        days: int,
    ) -> None:
        self.bankroll_start = bankroll_start
        self.bankroll_target = bankroll_target
        self.days = days
        self.compounder = MonteCarloCompounder()

    def required_edge_per_trade(
        self,
        trades_per_day: int,
    ) -> float:
        """Minimum edge per trade to reach target via compounding."""
        return CompoundingPath.min_edge(
            start=self.bankroll_start,
            target=self.bankroll_target,
            days=self.days,
            trades_per_day=trades_per_day,
        )

    def run_h1_negrisk(
        self,
        spreads: list[float],
        depths: list[float],
        trades_per_day: int = 20,
        fee_rate: float = 0.02,
        slippage: float = 0.005,
    ) -> dict:
        """H1: NegRisk compounding simulation.

        Tests whether capturing NegRisk spreads (buy low / sell
        high on correlated event contracts) can compound bankroll
        from start to target.

        Parameters
        ----------
        spreads : observed spreads per trade (fractional)
        depths : available depth per price level
        trades_per_day : trades attempted per day
        fee_rate : fee per trade
        slippage : slippage per trade
        """
        if not spreads:
            return {"feasible": False, "reason": "no spread data"}

        avg_spread = sum(spreads) / len(spreads)
        avg_depth = sum(depths) / len(depths) if depths else 0.0

        required = self.required_edge_per_trade(trades_per_day)
        net_spread = avg_spread - fee_rate - slippage

        # Fill probability check
        fill_prob = FillSimulator.fill_probability(
            trade_size=100.0,
            depth_at_price=avg_depth,
            distance_from_mid=avg_spread / 2,
        )

        # Monte Carlo with spread-derived edge
        mc_result = self.compounder.run(
            bankroll=self.bankroll_start,
            edge_mean=avg_spread,
            edge_std=avg_spread * 0.5,
            trades_per_day=trades_per_day,
            days=self.days,
            fee_rate=fee_rate,
            fill_rate=fill_prob,
            slippage_mean=slippage,
            target=self.bankroll_target,
        )

        return {
            "hypothesis": "H1_negrisk",
            "feasible": net_spread >= required,
            "required_edge": required,
            "avg_spread": avg_spread,
            "net_spread": net_spread,
            "fill_probability": fill_prob,
            "monte_carlo": mc_result,
        }

    def run_h2_rebalancing(
        self,
        complement_gaps: list[float],
        trades_per_day: int = 10,
        fee_rate: float = 0.02,
        slippage: float = 0.005,
    ) -> dict:
        """H2: YES+NO rebalancing simulation.

        Tests whether buying YES+NO when sum < $1 and
        rebalancing as prices move can generate profit.

        Parameters
        ----------
        complement_gaps : list of (1 - YES - NO) values observed
        trades_per_day : rebalancing trades per day
        fee_rate : fee per trade
        slippage : slippage per trade
        """
        if not complement_gaps:
            return {"feasible": False, "reason": "no gap data"}

        avg_gap = sum(complement_gaps) / len(complement_gaps)
        required = self.required_edge_per_trade(trades_per_day)
        net_gap = avg_gap - 2 * fee_rate - slippage  # 2x fee: buy YES + buy NO

        mc_result = self.compounder.run(
            bankroll=self.bankroll_start,
            edge_mean=avg_gap,
            edge_std=avg_gap * 0.3,
            trades_per_day=trades_per_day,
            days=self.days,
            fee_rate=2 * fee_rate,
            fill_rate=0.85,
            slippage_mean=slippage,
            target=self.bankroll_target,
        )

        return {
            "hypothesis": "H2_rebalancing",
            "feasible": net_gap >= required,
            "required_edge": required,
            "avg_gap": avg_gap,
            "net_gap": net_gap,
            "monte_carlo": mc_result,
        }

    def run_h3_combinatorial(
        self,
        opportunities: list[dict],
        trades_per_day: int = 5,
        fee_rate: float = 0.02,
        slippage: float = 0.005,
    ) -> dict:
        """H3: Combinatorial arb simulation.

        Tests whether multi-outcome mispricing (e.g. sum of
        mutually exclusive outcomes != 1) can be exploited.

        Parameters
        ----------
        opportunities : list of dicts with keys:
            - edge: fractional edge per opportunity
            - contracts: number of contracts available
        trades_per_day : arb opportunities per day
        fee_rate : fee per trade
        slippage : slippage per trade
        """
        if not opportunities:
            return {"feasible": False, "reason": "no opportunities"}

        edges = [o["edge"] for o in opportunities]
        avg_edge = sum(edges) / len(edges)
        required = self.required_edge_per_trade(trades_per_day)
        net_edge = avg_edge - fee_rate - slippage

        mc_result = self.compounder.run(
            bankroll=self.bankroll_start,
            edge_mean=avg_edge,
            edge_std=max(edges) - min(edges) if len(edges) > 1 else avg_edge * 0.3,
            trades_per_day=trades_per_day,
            days=self.days,
            fee_rate=fee_rate,
            fill_rate=0.70,
            slippage_mean=slippage,
            target=self.bankroll_target,
        )

        return {
            "hypothesis": "H3_combinatorial",
            "feasible": net_edge >= required,
            "required_edge": required,
            "avg_edge": avg_edge,
            "net_edge": net_edge,
            "monte_carlo": mc_result,
        }

    def run_h4_latency(
        self,
        lag_seconds: float,
        price_moves_per_second: float = 0.001,
        trades_per_day: int = 50,
        fee_rate: float = 0.02,
        slippage: float = 0.005,
    ) -> dict:
        """H4: Latency capture simulation.

        Tests whether speed advantage (seeing price changes
        before others) can generate systematic edge.

        Parameters
        ----------
        lag_seconds : latency advantage in seconds
        price_moves_per_second : avg price move per second
        trades_per_day : latency trades per day
        fee_rate : fee per trade
        slippage : slippage per trade
        """
        # Edge from latency = lag * price_move_rate
        latency_edge = lag_seconds * price_moves_per_second
        required = self.required_edge_per_trade(trades_per_day)
        net_edge = latency_edge - fee_rate - slippage

        mc_result = self.compounder.run(
            bankroll=self.bankroll_start,
            edge_mean=latency_edge,
            edge_std=latency_edge * 0.5,
            trades_per_day=trades_per_day,
            days=self.days,
            fee_rate=fee_rate,
            fill_rate=0.90,
            slippage_mean=slippage,
            target=self.bankroll_target,
        )

        return {
            "hypothesis": "H4_latency",
            "feasible": net_edge >= required,
            "required_edge": required,
            "latency_edge": latency_edge,
            "net_edge": net_edge,
            "lag_seconds": lag_seconds,
            "monte_carlo": mc_result,
        }

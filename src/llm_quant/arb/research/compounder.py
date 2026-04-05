"""Monte Carlo compounder for testing sensitivity."""

from __future__ import annotations

import numpy as np


class MonteCarloCompounder:
    """Monte Carlo simulation of compounding paths under uncertainty."""

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.seed = seed

    def run(
        self,
        bankroll: float,
        edge_mean: float,
        edge_std: float,
        trades_per_day: int,
        days: int,
        fee_rate: float,
        fill_rate: float,
        slippage_mean: float,
        target: float | None = None,
    ) -> dict:
        """Run Monte Carlo simulation.

        Parameters
        ----------
        bankroll : starting capital
        edge_mean : mean per-trade edge (fractional)
        edge_std : std dev of per-trade edge
        trades_per_day : trades attempted per day
        days : number of trading days
        fee_rate : fee per trade (fractional)
        fill_rate : probability of fill (0-1)
        slippage_mean : mean slippage per trade (fractional)
        target : optional target bankroll for prob_target calc

        Returns
        -------
        dict with keys: median_final, p5, p25, p75, p95,
            prob_target, max_drawdown_median, mean_final
        """
        rng = np.random.default_rng(self.seed)
        total_trades = days * trades_per_day
        finals = np.empty(self.n_simulations)
        max_drawdowns = np.empty(self.n_simulations)

        for i in range(self.n_simulations):
            path = np.empty(total_trades + 1)
            path[0] = bankroll

            # Draw per-trade edges from normal distribution
            edges = rng.normal(edge_mean, edge_std, total_trades)
            # Draw fill outcomes (bernoulli)
            fills = rng.random(total_trades) < fill_rate

            for t in range(total_trades):
                if fills[t]:
                    net = edges[t] - fee_rate - slippage_mean
                    path[t + 1] = path[t] * (1.0 + net)
                else:
                    # No fill — bankroll unchanged
                    path[t + 1] = path[t]

                # Floor at zero (can't go negative)
                if path[t + 1] < 0:
                    path[t + 1] = 0.0

            finals[i] = path[-1]

            # Max drawdown for this path
            running_max = np.maximum.accumulate(path)
            drawdowns = (running_max - path) / np.where(
                running_max > 0, running_max, 1.0
            )
            max_drawdowns[i] = float(np.max(drawdowns))

        prob_target = 0.0
        if target is not None:
            prob_target = float(np.mean(finals >= target))

        return {
            "median_final": float(np.median(finals)),
            "mean_final": float(np.mean(finals)),
            "p5": float(np.percentile(finals, 5)),
            "p25": float(np.percentile(finals, 25)),
            "p75": float(np.percentile(finals, 75)),
            "p95": float(np.percentile(finals, 95)),
            "prob_target": prob_target,
            "max_drawdown_median": float(np.median(max_drawdowns)),
        }

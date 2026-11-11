"""Joint stationary bootstrap: historical scenarios, not calibrated forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .data import require


@dataclass(frozen=True)
class Simulation:
    horizon: int = 12
    paths: int = 10000
    block_length: int = 3
    seed: int = 20261111

    def validate(self, observations: int) -> None:
        require(all(isinstance(value, int) and not isinstance(value, bool)
                    for value in asdict(self).values()), "Simulation settings must be integers")
        require(1 <= self.horizon <= 1260, "Horizon must be 1..1260 native periods")
        require(100 <= self.paths <= 100000, "Use 100..100000 paths")
        require(1 <= self.block_length <= observations, "Invalid block length")
        require(observations >= max(24, self.horizon * 3), "Insufficient history")
        require(self.paths * (self.horizon + 1) <= 10000000, "Memory budget exceeded")
        require(self.seed >= 0, "Seed must be nonnegative")


def validate_returns(returns: np.ndarray, weights: np.ndarray) -> tuple:
    data = np.asarray(returns, dtype=float)
    allocations = np.asarray(weights, dtype=float)
    require(data.ndim == 2 and data.shape[1] > 0, "Returns must be rows x assets")
    require(
        allocations.ndim == 1 and len(allocations) == data.shape[1],
        "One weight per asset is required",
    )
    require(bool(np.all(np.isfinite(data) & (data > -1))), "Invalid simple returns")
    require(
        bool(np.all(np.isfinite(allocations) & (allocations >= 0))),
        "Long-only finite weights required; leverage is not supported",
    )
    require(math.isclose(float(allocations.sum()), 1.0, rel_tol=0.0, abs_tol=1e-10), "Weights sum != 1")
    return data, allocations


def bootstrap_paths(
    returns: np.ndarray, weights: np.ndarray, config: Simulation,
) -> np.ndarray:
    """Buy-and-hold paths; resample synchronized asset rows in geometric blocks.

    Politis and Romano (1994), DOI 10.1080/01621459.1994.10476870.
    The block length is a sensitivity input, not a fitted optimal value.
    No drift override, synthetic normal tails, leverage, or rebalancing.
    """
    data, allocations = validate_returns(returns, weights)
    config.validate(len(data))
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, len(data), size=config.paths)
    holdings = np.broadcast_to(allocations, (config.paths, len(allocations))).copy()
    wealth = np.ones((config.paths, config.horizon + 1))
    for step in range(config.horizon):
        if step:
            restart = rng.random(config.paths) < 1.0 / config.block_length
            indices = np.where(
                restart, rng.integers(0, len(data), size=config.paths),
                (indices + 1) % len(data),
            )
        holdings *= 1.0 + data[indices]
        wealth[:, step + 1] = holdings.sum(axis=1)
    require(bool(np.all(np.isfinite(wealth))), "Numerical overflow in simulated wealth")
    return wealth


def summarize(wealth: np.ndarray) -> dict:
    values = np.asarray(wealth, dtype=float)
    require(
        values.ndim == 2 and values.shape[0] > 0 and values.shape[1] >= 2,
        "Need nonempty initial and future wealth",
    )
    require(bool(np.all(np.isfinite(values) & (values > 0))), "Invalid wealth paths")
    require(bool(np.allclose(values[:, 0], 1.0)), "Initial wealth must equal one")
    terminal = values[:, -1] - 1.0
    peaks = np.maximum.accumulate(values, axis=1)
    tail_size = max(1, math.ceil(0.05 * len(terminal)))
    outcomes = {}
    for loss in (0.05, 0.10, 0.20):
        probability = float(np.mean(values[:, -1] <= 1 - loss))
        outcomes[str(loss)] = {
            "terminal_loss_probability": probability,
            "ever_below_initial_probability": float(np.mean(values.min(axis=1) <= 1 - loss)),
            "peak_to_trough_probability": float(np.mean((values / peaks).min(axis=1) <= 1 - loss)),
            "monte_carlo_standard_error_only": math.sqrt(
                probability * (1 - probability) / len(terminal),
            ),
        }
    return {
        "mean_return": float(np.mean(terminal)),
        "median_return": float(np.median(terminal)),
        "p05_return": float(np.quantile(terminal, 0.05)),
        "p95_return": float(np.quantile(terminal, 0.95)),
        "expected_shortfall_95_loss": -float(np.sort(terminal)[:tail_size].mean()),
        "probability_of_negative_terminal_return": float(np.mean(terminal < 0)),
        "loss_events": outcomes,
    }


def risk_report(returns: np.ndarray, weights: np.ndarray, config: Simulation) -> dict:
    wealth = bootstrap_paths(returns, weights, config)
    return {
        "schema_version": 1,
        "model": "joint_stationary_bootstrap_buy_and_hold_v1",
        "status": "UNCALIBRATED_HISTORICAL_SCENARIOS",
        "config": asdict(config),
        "observations": len(returns),
        "weights": [float(value) for value in weights],
        "results": summarize(wealth),
        "limitations": [
            "Historical distribution is not a current-market forecast or fair value.",
            "Block length, estimation window and structural breaks change results.",
            "Unseen crises and parameter uncertainty are not captured by Monte Carlo SE.",
            "Buy-and-hold; no taxes, transaction costs, liquidity or forced-sale effects.",
            "Provider adjusted closes are only a total-return proxy when so labelled.",
        ],
    }

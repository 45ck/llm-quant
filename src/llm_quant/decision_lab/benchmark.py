"""Cost-aware, time-ordered baselines. Research only; no execution integration.

Uses scikit-learn's TimeSeriesSplit/LedoitWolf and SciPy's numerical solvers.
Returns are equally spaced close-to-close observations, not intraday fills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import TimeSeriesSplit

from .data import aligned_returns, load_snapshot, require

FloatArray = NDArray[np.float64]
METHODS = ("cash", "equal_weight", "inverse_volatility_lw", "minimum_variance_lw")


@dataclass(frozen=True)
class EvaluationPlan:
    """Freeze before evaluation; gap includes one full observation of execution lag."""

    splits: int = 4
    test_size: int = 20
    min_train: int = 60
    holdout_size: int = 20
    gap: int = 1
    label_horizon: int = 1
    periods_per_year: int = 252
    transaction_bps: float = 10.0

    def validate(self, observations: int) -> None:
        counts = (self.splits, self.test_size, self.min_train, self.holdout_size,
                  self.gap, self.label_horizon, self.periods_per_year)
        require(all(type(item) is int and item > 0 for item in counts),
                "Period counts must be positive integers")
        require(self.splits >= 2, "TimeSeriesSplit requires at least two splits")
        require(self.holdout_size >= 2, "Holdout needs at least two returns for metrics")
        require(self.min_train >= 20, "At least 20 training observations required")
        require(self.gap >= self.label_horizon, "Gap must cover the forward label horizon")
        require(math.isfinite(self.transaction_bps) and 0 <= self.transaction_bps <= 500,
                "Transaction costs must be finite and within 0..500 bps")
        first_train = observations - self.holdout_size - self.splits * self.test_size - self.gap
        require(first_train >= self.min_train, "Insufficient history for frozen evaluation plan")


def checked_returns(values: FloatArray) -> FloatArray:
    data = np.asarray(values, dtype=float)
    require(data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 0,
            "Returns must be a nonempty observations-by-assets matrix")
    require(bool(np.all(np.isfinite(data) & (data > -1))), "Invalid simple returns")
    return data


def fit_weights(training: FloatArray, method: str) -> FloatArray:
    """Fit on the supplied historical slice only; never select a winner on test data."""
    data = checked_returns(training)
    assets = data.shape[1]
    require(method in METHODS, "Unregistered allocation method")
    if method == "cash":
        return np.zeros(assets)
    if method == "equal_weight":
        return np.full(assets, 1.0 / assets)
    require(len(data) >= 20, "Covariance fit needs at least 20 observations")
    covariance = LedoitWolf(store_precision=False).fit(data).covariance_
    diagonal = np.diag(covariance)
    require(bool(np.all(np.isfinite(covariance))) and bool(np.all(diagonal > 0)),
            "Degenerate covariance; no silent allocation fallback")
    if method == "inverse_volatility_lw":
        inverse = 1.0 / np.sqrt(diagonal)
        return inverse / inverse.sum()
    # Scaling improves solver tolerances without changing the minimizer.
    scaled = covariance / float(np.mean(diagonal))
    solution = minimize(
        lambda weights: float(weights @ scaled @ weights),
        np.full(assets, 1.0 / assets),
        jac=lambda weights: 2.0 * scaled @ weights,
        bounds=[(0.0, 1.0)] * assets,
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1),
                     "jac": lambda weights: np.ones_like(weights)},
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 200},
    )
    require(solution.success and bool(np.all(np.isfinite(solution.x))),
            "Minimum-variance optimizer did not converge")
    require(bool(np.all(solution.x >= -1e-10)) and abs(float(solution.x.sum()) - 1) < 1e-8,
            "Optimizer violated allocation constraints")
    result = np.maximum(solution.x, 0)
    return result / result.sum()


def rebalance(holdings: FloatArray, cash: float, target: FloatArray,
              cost_rate: float) -> tuple[FloatArray, float, float, float]:
    """Self-financing post-fee weights; each risky dollar bought/sold is charged.

    Solve V_after = V_before - c * sum(abs(target * V_after - holdings)).
    Cash is not charged as another trade leg. No borrowing, shorts or fixed fees.
    """
    current = np.asarray(holdings, dtype=float)
    weights = np.asarray(target, dtype=float)
    require(current.ndim == 1 and weights.shape == current.shape and current.size > 0,
            "Allocation shape mismatch")
    require(bool(np.all(np.isfinite(current) & (current >= 0)))
            and math.isfinite(cash) and cash >= 0, "Invalid holdings")
    require(bool(np.all(np.isfinite(weights) & (weights >= 0)))
            and float(weights.sum()) <= 1 + 1e-12, "Long-only unleveraged weights required")
    require(math.isfinite(cost_rate) and 0 <= cost_rate <= 0.05, "Invalid trading cost")
    total = float(current.sum()) + cash
    require(total > 0, "Positive pre-trade wealth required")
    # Solve using fractions so accuracy does not depend on account size.
    before = current / total
    def equation(fraction: float) -> float:
        return fraction + cost_rate * float(np.abs(weights * fraction - before).sum()) - 1
    fraction = float(brentq(equation, 0, 1, xtol=1e-14)) if cost_rate else 1.0
    after = total * fraction
    new_holdings = weights * after
    traded = float(np.abs(new_holdings - current).sum())
    fee = traded * cost_rate
    new_cash = max(0.0, after - float(new_holdings.sum()))
    require(abs(float(new_holdings.sum()) + new_cash + fee - total) <= 1e-10 * total,
            "Self-financing accounting invariant failed")
    return new_holdings, new_cash, fee, traded


def simulate(data: FloatArray, allocations: dict[int, FloatArray],
             cost_bps: float) -> dict:
    """Bar-close rebalances before each next return; inputs must already be lagged."""
    returns = checked_returns(data)
    require(math.isfinite(cost_bps) and 0 <= cost_bps <= 500, "Invalid trading cost")
    require(all(type(index) is int and 0 <= index < len(returns) for index in allocations),
            "Rebalance outside the evaluation period")
    holdings = np.zeros(returns.shape[1])
    cash = 1.0
    nav, fees, turnover = [1.0], [], []
    for index, observation in enumerate(returns):
        fee, traded = 0.0, 0.0
        if index in allocations:
            holdings, cash, fee, traded = rebalance(
                holdings, cash, allocations[index], cost_bps / 10000,
            )
        holdings = holdings * (1.0 + observation)
        total = float(holdings.sum()) + cash
        require(math.isfinite(total) and total > 0, "Invalid simulated wealth")
        nav.append(total)
        fees.append(fee)
        turnover.append(traded)
    return {"nav": nav, "fees_per_initial_unit": fees,
            "risky_turnover_per_initial_unit": turnover,
            "cash_interest": "zero_by_explicit_assumption"}


def performance(nav: list[float], periods_per_year: int) -> dict:
    values = np.asarray(nav, dtype=float)
    require(values.ndim == 1 and len(values) >= 3
            and bool(np.all(np.isfinite(values) & (values > 0))), "Invalid NAV series")
    require(type(periods_per_year) is int and periods_per_year > 0, "Invalid annualization")
    returns = values[1:] / values[:-1] - 1
    volatility = float(np.std(returns, ddof=1))
    gross = float(values[-1] / values[0])
    return {
        "observations": len(returns), "total_return": gross - 1,
        "annualized_compound_return": gross ** (periods_per_year / len(returns)) - 1,
        "annualized_volatility": volatility * math.sqrt(periods_per_year),
        "sharpe_zero_rate": (None if volatility <= 1e-14 else
                             float(np.mean(returns)) / volatility * math.sqrt(periods_per_year)),
        "max_drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
    }


def _period_result(returns: FloatArray, schedule: dict[int, FloatArray],
                   plan: EvaluationPlan) -> dict:
    simulation = simulate(returns, schedule, plan.transaction_bps)
    return {**simulation, "metrics": performance(simulation["nav"], plan.periods_per_year)}


def implementation_identity(plan: EvaluationPlan) -> dict:
    """Bind results to the implemented numerical/data code and declared plan."""
    digest = hashlib.sha256()
    for filename in ("benchmark.py", "data.py"):
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update(Path(__file__).with_name(filename).read_bytes())
    plan_bytes = json.dumps(asdict(plan), sort_keys=True, allow_nan=False).encode()
    return {
        "implementation_sha256": digest.hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "library_versions": {name: version(name)
                             for name in ("numpy", "scipy", "scikit-learn")},
    }


def compare_baselines(returns: FloatArray, plan: EvaluationPlan) -> dict:
    data = checked_returns(returns)
    plan.validate(len(data))
    holdout_start = len(data) - plan.holdout_size
    development = data[:holdout_start]
    splitter = TimeSeriesSplit(n_splits=plan.splits, test_size=plan.test_size, gap=plan.gap)
    folds = list(splitter.split(development))
    evaluation_start = int(folds[0][1][0])
    trace = [{"train_start": int(train[0]), "train_end_exclusive": int(train[-1]) + 1,
              "test_start": int(test[0]), "test_end_exclusive": int(test[-1]) + 1}
             for train, test in folds]
    results = {}
    for method in METHODS:
        schedule = {int(test[0]) - evaluation_start: fit_weights(data[train].copy(), method)
                    for train, test in folds}
        holdout_weights = fit_weights(data[:holdout_start - plan.gap].copy(), method)
        results[method] = {
            "development": _period_result(data[evaluation_start:holdout_start], schedule, plan),
            "reserved_holdout": _period_result(data[holdout_start:], {0: holdout_weights}, plan),
            "development_allocations": {str(key): val.tolist() for key, val in schedule.items()},
            "holdout_weights": holdout_weights.tolist(),
        }
    equal = np.full(data.shape[1], 1.0 / data.shape[1])
    results["equal_weight_buy_and_hold"] = {
        "development": _period_result(data[evaluation_start:holdout_start], {0: equal}, plan),
        "reserved_holdout": _period_result(data[holdout_start:], {0: equal}, plan),
    }
    return {
        "schema_version": 1, "status": "RETROSPECTIVE_BASELINES_NOT_TRADING_APPROVAL",
        "identity": implementation_identity(plan),
        "plan": asdict(plan), "folds": trace, "evaluation_start": evaluation_start,
        "holdout_start": holdout_start, "holdout_train_end_exclusive": holdout_start - plan.gap,
        "results": results, "automatic_model_selection": False,
        "limitations": [
            "Equally spaced close-to-close return rows; exchange calendars must be checked separately.",
            "At least one complete bar separates training from execution; not next-open OHLC fills.",
            "The reserved holdout is isolated by code, but past human inspection cannot be disproved.",
            "Each segment starts at one unit of cash; no leveraged positions or taxes.",
            "Proportional trading cost includes explicit commission/slippage assumption, not market impact.",
            "Cash earns zero; Sharpe is excess over zero, not a current risk-free rate.",
            "Latest-vintage data is not point-in-time; all methods and failures must be retained.",
        ],
    }


def reference_study() -> dict:
    """One reproducible, MIT-licensed artificial fixture; not market observations."""
    rng = np.random.default_rng(67129)
    common = rng.normal(0.0002, 0.009, size=(240, 1))
    data = common + rng.normal(0, [0.005, 0.012, 0.02], size=(240, 3))
    report = compare_baselines(data, EvaluationPlan())
    report["status"] = "SYNTHETIC_REFERENCE_NOT_MARKET_EVIDENCE"
    report["fixture_sha256"] = hashlib.sha256(data.astype("<f8").tobytes()).hexdigest()
    report["fixture_seed"] = 67129
    report["libraries"] = {key: version(key) for key in ("numpy", "scipy", "scikit-learn")}
    report["cost_sensitivity"] = {
        str(bps): compare_baselines(data, EvaluationPlan(transaction_bps=bps))["results"][
            "equal_weight_buy_and_hold"]["reserved_holdout"]["metrics"]["total_return"]
        for bps in (0, 10, 50)
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--snapshot", type=Path, action="append")
    parser.add_argument("--plan", type=Path, help="Frozen JSON EvaluationPlan")
    parser.add_argument("--out", type=Path,
                        default=Path("research-local/decision-lab/benchmark.json"))
    args = parser.parse_args()
    if args.reference:
        require(not args.snapshot and not args.plan, "Reference cannot be mixed with observed inputs")
        report = reference_study()
    else:
        require(bool(args.snapshot) and args.plan is not None, "Supply snapshots and a frozen plan")
        snapshots = [load_snapshot(path) for path in args.snapshot]
        plan = EvaluationPlan(**json.loads(args.plan.read_text(encoding="utf-8")))
        expected = {"D": 252, "M": 12, "Q": 4}[snapshots[0].contract.frequency]
        require(plan.periods_per_year == expected, "Annualization disagrees with native frequency")
        report = compare_baselines(aligned_returns(snapshots), plan)
        report["snapshot_sha256"] = [snapshot.digest() for snapshot in snapshots]
        report["sources"] = [asdict(snapshot.contract) for snapshot in snapshots]
        report["data_end"] = snapshots[0].dates[-1]
        report["currency"] = snapshots[0].contract.currency
        if snapshots[0].contract.evidence == "synthetic":
            report["status"] = "SYNTHETIC_REFERENCE_NOT_MARKET_EVIDENCE"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

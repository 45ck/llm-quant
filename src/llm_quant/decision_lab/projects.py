"""Business sensitivity arithmetic; never infer sales probability from commits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .data import require


@dataclass(frozen=True)
class ProjectCase:
    fee: float
    direct_cost_if_won: float
    presales_hours: float
    delivery_hours_if_won: float
    probability_of_winning: float
    opportunity_cost_per_hour: float
    fixed_experiment_cost: float = 0.0
    currency: str = "AUD"

    def evaluate(self) -> dict:
        fields = asdict(self)
        amounts = [value for key, value in fields.items() if key != "currency"]
        require(all(math.isfinite(value) and value >= 0 for value in amounts),
                "Financial inputs must be finite and nonnegative")
        require(self.probability_of_winning <= 1, "Probability exceeds one")
        require(bool(self.currency), "Explicit currency required")
        probability = self.probability_of_winning
        margin = self.fee - self.direct_cost_if_won
        cash = probability * margin - self.fixed_experiment_cost
        hours = self.presales_hours + probability * self.delivery_hours_if_won
        require(hours > 0, "Expected founder hours must be positive")
        win_surplus = margin - self.delivery_hours_if_won * self.opportunity_cost_per_hour
        sunk = self.fixed_experiment_cost + self.presales_hours * self.opportunity_cost_per_hour
        break_even = None if win_surplus <= 0 else sunk / win_surplus
        return {
            "status": "ASSUMPTION_DRIVEN_SENSITIVITY_NOT_SUCCESS_FORECAST",
            "inputs": fields,
            "expected_cash_contribution": cash,
            "expected_founder_hours": hours,
            "contribution_per_expected_hour": cash / hours,
            "expected_economic_surplus": cash - hours * self.opportunity_cost_per_hour,
            "economic_break_even_win_probability": break_even,
            "positive_economic_surplus_feasible": (
                win_surplus > sunk and break_even is not None and break_even < 1
            ),
            "caveat": "Before tax; excludes unmodelled support, refunds and overruns. "
                      "The win probability is a user assumption, not learned from GitHub.",
        }


def mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    require(math.isfinite(principal) and principal >= 0, "Invalid principal")
    require(math.isfinite(annual_rate) and annual_rate >= 0, "Invalid interest rate")
    require(isinstance(years, int) and 1 <= years <= 50, "Invalid loan term")
    periods = years * 12
    monthly = annual_rate / 12
    if monthly == 0:
        return principal / periods
    return principal * monthly / (-math.expm1(-periods * math.log1p(monthly)))


def affordability_change(price: float, income: float, target_multiple: float) -> float:
    require(
        all(math.isfinite(value) and value > 0 for value in (price, income, target_multiple)),
        "Price, income and target multiple must be finite and positive",
    )
    return target_multiple * income / price - 1

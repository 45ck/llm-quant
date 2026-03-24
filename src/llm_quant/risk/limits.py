"""Individual risk-limit checks.

Each check function evaluates a single constraint and returns a
``RiskCheckResult``.  Results carry enough detail for the risk manager
to log *why* a trade was approved or rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Outcome of a single pre-trade risk check."""

    passed: bool
    rule: str
    message: str
    current_value: float = 0.0
    limit_value: float = 0.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_position_size(
    trade_notional: float,
    nav: float,
    max_trade_size: float,
) -> RiskCheckResult:
    """Ensure a single trade does not exceed ``max_trade_size`` of NAV.

    Parameters
    ----------
    trade_notional:
        Absolute notional value of the proposed trade.
    nav:
        Current net asset value.
    max_trade_size:
        Maximum fraction of NAV a single trade may represent.
    """
    if nav <= 0.0:
        return RiskCheckResult(
            passed=False,
            rule="position_size",
            message="NAV is zero or negative – cannot evaluate trade size.",
            current_value=trade_notional,
            limit_value=0.0,
        )

    trade_frac = trade_notional / nav
    passed = trade_frac <= max_trade_size
    return RiskCheckResult(
        passed=passed,
        rule="position_size",
        message=(
            f"Trade size {trade_frac:.2%} of NAV "
            f"{'<=' if passed else '>'} limit {max_trade_size:.2%}."
        ),
        current_value=trade_frac,
        limit_value=max_trade_size,
    )


def check_position_weight(
    current_weight: float,
    target_weight: float,
    max_weight: float,
) -> RiskCheckResult:
    """Ensure a position's target weight stays within ``max_weight``.

    Parameters
    ----------
    current_weight:
        Current weight of the position (fraction of NAV).
    target_weight:
        Proposed target weight after the trade.
    max_weight:
        Maximum allowed position weight.
    """
    passed = target_weight <= max_weight
    return RiskCheckResult(
        passed=passed,
        rule="position_weight",
        message=(
            f"Target weight {target_weight:.2%} "
            f"{'<=' if passed else '>'} max {max_weight:.2%} "
            f"(current {current_weight:.2%})."
        ),
        current_value=target_weight,
        limit_value=max_weight,
    )


def check_gross_exposure(
    current_gross: float,
    trade_notional: float,
    nav: float,
    max_gross: float,
) -> RiskCheckResult:
    """Ensure gross exposure after the trade stays within ``max_gross``.

    Parameters
    ----------
    current_gross:
        Current gross exposure as an absolute dollar amount.
    trade_notional:
        Absolute notional of the proposed trade.
    nav:
        Current NAV.
    max_gross:
        Maximum allowed gross exposure as a fraction of NAV.
    """
    if nav <= 0.0:
        return RiskCheckResult(
            passed=False,
            rule="gross_exposure",
            message="NAV is zero or negative.",
            current_value=0.0,
            limit_value=max_gross,
        )

    projected_gross = (current_gross + trade_notional) / nav
    passed = projected_gross <= max_gross
    return RiskCheckResult(
        passed=passed,
        rule="gross_exposure",
        message=(
            f"Projected gross exposure {projected_gross:.2%} "
            f"{'<=' if passed else '>'} limit {max_gross:.2%}."
        ),
        current_value=projected_gross,
        limit_value=max_gross,
    )


def check_net_exposure(
    current_net: float,
    trade_notional: float,
    nav: float,
    max_net: float,
) -> RiskCheckResult:
    """Ensure net exposure after the trade stays within ``max_net``.

    For buys, net exposure increases; for sells it decreases.
    The caller should sign *trade_notional* appropriately (+buy / -sell).

    Parameters
    ----------
    current_net:
        Current signed net exposure in dollars.
    trade_notional:
        Signed notional of the proposed trade (+buy, -sell).
    nav:
        Current NAV.
    max_net:
        Maximum absolute net exposure as a fraction of NAV.
    """
    if nav <= 0.0:
        return RiskCheckResult(
            passed=False,
            rule="net_exposure",
            message="NAV is zero or negative.",
            current_value=0.0,
            limit_value=max_net,
        )

    projected_net = abs(current_net + trade_notional) / nav
    passed = projected_net <= max_net
    return RiskCheckResult(
        passed=passed,
        rule="net_exposure",
        message=(
            f"Projected net exposure {projected_net:.2%} "
            f"{'<=' if passed else '>'} limit {max_net:.2%}."
        ),
        current_value=projected_net,
        limit_value=max_net,
    )


def check_sector_concentration(
    sector_weight: float,
    trade_weight: float,
    max_sector: float,
) -> RiskCheckResult:
    """Ensure a sector's aggregate weight does not exceed ``max_sector``.

    Parameters
    ----------
    sector_weight:
        Current weight of the sector (fraction of NAV).
    trade_weight:
        Additional weight this trade would add to the sector.
    max_sector:
        Maximum allowed sector concentration.
    """
    projected = sector_weight + trade_weight
    passed = projected <= max_sector
    return RiskCheckResult(
        passed=passed,
        rule="sector_concentration",
        message=(
            f"Sector weight after trade {projected:.2%} "
            f"{'<=' if passed else '>'} limit {max_sector:.2%} "
            f"(current {sector_weight:.2%})."
        ),
        current_value=projected,
        limit_value=max_sector,
    )


def check_cash_reserve(
    cash: float,
    trade_notional: float,
    nav: float,
    min_reserve: float,
) -> RiskCheckResult:
    """Ensure minimum cash reserve is maintained after a purchase.

    Parameters
    ----------
    cash:
        Current cash balance.
    trade_notional:
        Absolute notional of the proposed *buy* trade (cash outflow).
    nav:
        Current NAV.
    min_reserve:
        Minimum cash as a fraction of NAV that must be maintained.
    """
    if nav <= 0.0:
        return RiskCheckResult(
            passed=False,
            rule="cash_reserve",
            message="NAV is zero or negative.",
            current_value=0.0,
            limit_value=min_reserve,
        )

    remaining_cash_frac = (cash - trade_notional) / nav
    passed = remaining_cash_frac >= min_reserve
    return RiskCheckResult(
        passed=passed,
        rule="cash_reserve",
        message=(
            f"Cash after trade {remaining_cash_frac:.2%} of NAV "
            f"{'>=' if passed else '<'} reserve {min_reserve:.2%}."
        ),
        current_value=remaining_cash_frac,
        limit_value=min_reserve,
    )


def check_stop_loss(
    has_stop_loss: bool,
    require: bool,
) -> RiskCheckResult:
    """Ensure a stop-loss is set if policy requires it.

    Parameters
    ----------
    has_stop_loss:
        Whether the trade signal includes a non-zero stop-loss.
    require:
        Whether the risk policy mandates a stop-loss on every trade.
    """
    if not require:
        return RiskCheckResult(
            passed=True,
            rule="stop_loss",
            message="Stop-loss not required by policy.",
            current_value=1.0 if has_stop_loss else 0.0,
            limit_value=0.0,
        )

    passed = has_stop_loss
    return RiskCheckResult(
        passed=passed,
        rule="stop_loss",
        message=(
            "Stop-loss is set."
            if passed
            else "Stop-loss required but not provided."
        ),
        current_value=1.0 if has_stop_loss else 0.0,
        limit_value=1.0,
    )

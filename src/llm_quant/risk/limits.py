"""Individual risk-limit checks.

Each check function evaluates a single constraint and returns a
``RiskCheckResult``.  Results carry enough detail for the risk manager
to log *why* a trade was approved or rejected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Execution cost estimates (round-trip, in basis points)
# Source: config/risk.toml [execution_costs] — values are mid-range estimates.
# These are embedded here as fallback constants; config values override them
# when available via AppConfig.
# ---------------------------------------------------------------------------

_EXECUTION_COSTS_BPS: dict[str, float] = {
    "equity": 4.0,  # US equity: 2-6 bps round-trip
    "international": 12.0,  # International developed: 7-18 bps
    "emerging": 22.0,  # Emerging markets: 12-33 bps
    "fixed_income": 6.0,  # Treasury ETFs: 3-9 bps
    "hy_credit": 16.0,  # HY credit: 8-24 bps
    "commodity": 6.0,  # Commodity ETFs (GLD/SLV/USO): ~crypto_etf level
    "crypto": 6.0,  # Crypto ETF proxy: 3-9 bps
    "forex": 17.0,  # Forex at ~$2500 notional: 17-18 bps
    "volatility": 0.0,  # Non-tradeable (VIX reference only)
}

# Sector-level overrides for fixed income sub-types
_HY_CREDIT_SYMBOLS: frozenset[str] = frozenset({"HYG", "JNK"})
_EMERGING_EQUITY_SYMBOLS: frozenset[str] = frozenset({"EEM", "VWO"})
_INTL_EQUITY_SYMBOLS: frozenset[str] = frozenset({"EFA", "VGK", "EWJ"})


def get_execution_cost(symbol: str, asset_class: str) -> float:
    """Return the estimated round-trip execution cost in basis points.

    Uses symbol-level overrides first, then asset_class bucketing.
    Values represent the mid-range of typical market-impact + spread costs
    for a ~$2,500–10,000 notional retail/paper trade.

    Example context: "Entering GLD costs ~6 bps round-trip.
    Expected daily alpha must exceed this."

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"GLD"``, ``"HYG"``, ``"EEM"``).
    asset_class:
        Asset class string from universe config
        (``"equity"``, ``"fixed_income"``, ``"crypto"``, ``"forex"``,
        ``"commodity"``, ``"volatility"``).

    Returns
    -------
    float
        Estimated round-trip cost in basis points.
    """
    # Symbol-level overrides (finer granularity than asset_class)
    if symbol in _HY_CREDIT_SYMBOLS:
        return _EXECUTION_COSTS_BPS["hy_credit"]
    if symbol in _EMERGING_EQUITY_SYMBOLS:
        return _EXECUTION_COSTS_BPS["emerging"]
    if symbol in _INTL_EQUITY_SYMBOLS:
        return _EXECUTION_COSTS_BPS["international"]

    # Asset class bucket
    ac = asset_class.lower()
    if ac in _EXECUTION_COSTS_BPS:
        return _EXECUTION_COSTS_BPS[ac]

    # Unknown asset class → default to US equity cost
    logger.debug(
        "get_execution_cost: unknown asset_class '%s' for %s, defaulting to equity",
        asset_class,
        symbol,
    )
    return _EXECUTION_COSTS_BPS["equity"]


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
            "Stop-loss is set." if passed else "Stop-loss required but not provided."
        ),
        current_value=1.0 if has_stop_loss else 0.0,
        limit_value=1.0,
    )


def check_atr_stop_loss(
    stop_loss_price: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float,
) -> RiskCheckResult:
    """Validate that a stop-loss is at least ``atr_multiplier`` ATRs from entry.

    Replaces static percentage stops with ATR-calibrated stops.  The Turtle
    Traders standard is 2x ATR; crypto and volatile commodities use 2.5-3x.

    Parameters
    ----------
    stop_loss_price:
        Proposed stop-loss price.
    entry_price:
        Proposed entry price.
    atr:
        Current ATR value for the symbol.
    atr_multiplier:
        Minimum required distance as a multiple of ATR (e.g. 2.0 for equities,
        2.5-3.0 for crypto/volatile commodities).
    """
    if entry_price <= 0.0 or atr <= 0.0:
        return RiskCheckResult(
            passed=False,
            rule="atr_stop_loss",
            message="Invalid entry price or ATR — cannot validate ATR stop.",
            current_value=0.0,
            limit_value=atr_multiplier,
        )

    distance = entry_price - stop_loss_price
    distance_in_atrs = distance / atr if atr > 0 else 0.0
    passed = distance_in_atrs >= atr_multiplier
    return RiskCheckResult(
        passed=passed,
        rule="atr_stop_loss",
        message=(
            f"Stop distance {distance_in_atrs:.2f}x ATR "
            f"{'>=' if passed else '<'} required {atr_multiplier:.1f}x ATR "
            f"(stop={stop_loss_price:.4f}, entry={entry_price:.4f}, "
            f"ATR={atr:.4f})."
        ),
        current_value=distance_in_atrs,
        limit_value=atr_multiplier,
    )


def check_volatility_sizing(
    symbol: str,
    atr: float,
    price: float,
    proposed_size: float,
    nav: float,
    target_risk_pct: float,
    deviation_buffer: float = 0.20,
) -> RiskCheckResult:
    """Check that position size is consistent with ATR-based volatility sizing.

    Computes the maximum position size fraction of NAV implied by the
    target risk per trade and current ATR, then alerts if the proposed
    size exceeds that limit by more than ``deviation_buffer``.

    Formula: max_size = target_risk_pct / (atr / price)

    Parameters
    ----------
    symbol:
        Ticker symbol (used in the message only).
    atr:
        Current ATR value for the symbol.
    price:
        Current price of the symbol.
    proposed_size:
        Proposed position size as a fraction of NAV (e.g. 0.05 for 5%).
    nav:
        Current net asset value (used to scale the notional check).
    target_risk_pct:
        Fraction of NAV to risk per trade (e.g. 0.01 for 1%).
    deviation_buffer:
        Fractional buffer before triggering a rebalance alert
        (default 0.20 = 20% above the ATR-implied limit).
    """
    if price <= 0.0 or atr <= 0.0 or nav <= 0.0:
        return RiskCheckResult(
            passed=False,
            rule="volatility_sizing",
            message=(
                f"{symbol}: Invalid price ({price}), ATR ({atr}), or NAV ({nav}) "
                "— cannot compute ATR-based size limit."
            ),
            current_value=proposed_size,
            limit_value=0.0,
        )

    atr_pct = atr / price  # normalised ATR (volatility as fraction of price)
    atr_implied_max = target_risk_pct / atr_pct  # max position fraction of NAV
    alert_threshold = atr_implied_max * (1.0 + deviation_buffer)

    passed = proposed_size <= alert_threshold
    return RiskCheckResult(
        passed=passed,
        rule="volatility_sizing",
        message=(
            f"{symbol}: proposed size {proposed_size:.2%} "
            f"{'<=' if passed else '>'} ATR alert threshold {alert_threshold:.2%} "
            f"(ATR-implied max {atr_implied_max:.2%}, ATR/price={atr_pct:.4f}, "
            f"target_risk={target_risk_pct:.2%}, buffer={deviation_buffer:.0%})."
        ),
        current_value=proposed_size,
        limit_value=alert_threshold,
    )


def check_portfolio_correlation_dcc(
    strategy_returns: pl.DataFrame,
    warn_threshold: float = 0.70,
) -> RiskCheckResult:
    """Advisory correlation check using DCC-GARCH dynamic estimation.

    Uses :class:`~llm_quant.risk.correlation.DccGarchEstimator` to compute
    the current conditional correlation matrix across strategy return streams.
    Returns WARNING when avg pairwise correlation exceeds ``warn_threshold``.

    This check is **advisory only** — it never returns ``passed=False``.
    Per the issue spec, the blocking mechanism for novel correlation patterns
    is not yet validated; correlation scoring is informational.

    Parameters
    ----------
    strategy_returns:
        Polars DataFrame where each column represents a return series for one
        strategy. Rows are daily observations (most recent last).
    warn_threshold:
        Average pairwise correlation threshold above which a warning is emitted
        (default 0.70). The result is always ``passed=True``.

    Returns
    -------
    RiskCheckResult
        Always ``passed=True``.  ``message`` contains the advisory text and
        correlation summary.  ``current_value`` is the avg pairwise correlation.
        ``limit_value`` is the warning threshold.
    """
    import polars as pl

    from llm_quant.risk.correlation import DccGarchEstimator

    if not isinstance(strategy_returns, pl.DataFrame) or strategy_returns.width < 2:
        return RiskCheckResult(
            passed=True,
            rule="portfolio_correlation_dcc",
            message=(
                "DCC correlation check skipped: need >= 2 return columns, "
                f"got {strategy_returns.width if isinstance(strategy_returns, pl.DataFrame) else 0}."
            ),
            current_value=0.0,
            limit_value=warn_threshold,
        )

    try:
        estimator = DccGarchEstimator()
        result = estimator.estimate(strategy_returns)
    except Exception:
        logger.warning(
            "DCC correlation estimation failed — check skipped.", exc_info=True
        )
        return RiskCheckResult(
            passed=True,
            rule="portfolio_correlation_dcc",
            message="DCC correlation estimation failed — check skipped (advisory).",
            current_value=0.0,
            limit_value=warn_threshold,
        )

    avg_corr = result.avg_pairwise_corr
    advisory = (
        result.warning
        or f"Portfolio avg correlation {avg_corr:.2f} (div score {result.diversification_score:.2f})."
    )
    method = result.method_used.value

    return RiskCheckResult(
        passed=True,  # advisory only — never blocks trades
        rule="portfolio_correlation_dcc",
        message=f"[{method}] {advisory}",
        current_value=avg_corr,
        limit_value=warn_threshold,
    )


def check_cvar_limit(
    portfolio_returns: pl.Series,
    proposed_trade_size: float,
) -> RiskCheckResult:
    """Check CVaR constraint using Filtered Historical Simulation.

    Runs :class:`~llm_quant.risk.cvar.FhsCvarEstimator` on the portfolio
    return series and returns a scaling recommendation when CVaR exceeds
    the configured limit.

    Enforcement is **proportional** (CPPI-style), not binary:
    - ``cvar_95 > 2 × cvar_limit_pct``: FAIL (hard block — extreme tail risk)
    - ``cvar_95 > cvar_limit_pct``: WARNING with scaling recommendation
    - ``cvar_95 <= cvar_limit_pct``: PASS

    Parameters
    ----------
    portfolio_returns:
        Polars Series of daily portfolio returns (most recent last).
        Typically 504 days (2 years) for reliable FHS estimates.
    proposed_trade_size:
        Proposed trade size as a fraction of NAV (informational — used in
        the scaling message to give an actionable recommendation).

    Returns
    -------
    RiskCheckResult
        - ``passed=False`` when CVaR exceeds 2× limit (hard block).
        - ``passed=True`` with WARNING message when 1×–2× limit (scale down).
        - ``passed=True`` when within limit (clean pass).
        ``current_value`` carries the CVaR estimate; ``limit_value`` is the
        configured limit.
    """
    import polars as pl

    from llm_quant.risk.cvar import CvarConfig, FhsCvarEstimator

    if not isinstance(portfolio_returns, pl.Series) or len(portfolio_returns) < 30:
        return RiskCheckResult(
            passed=True,
            rule="cvar_limit",
            message=(
                "CVaR check skipped: insufficient return history "
                f"({len(portfolio_returns) if isinstance(portfolio_returns, pl.Series) else 0} obs, need >= 30)."
            ),
            current_value=0.0,
            limit_value=CvarConfig().cvar_limit_pct,
        )

    try:
        config = CvarConfig()
        estimator = FhsCvarEstimator(config)
        result = estimator.estimate(portfolio_returns)
    except Exception:
        logger.warning("CVaR estimation failed — check skipped.", exc_info=True)
        return RiskCheckResult(
            passed=True,
            rule="cvar_limit",
            message="CVaR estimation failed — check skipped (advisory).",
            current_value=0.0,
            limit_value=CvarConfig().cvar_limit_pct,
        )

    limit = config.cvar_limit_pct
    cvar = result.cvar_95
    sf = result.scaling_factor
    scaled_size = proposed_trade_size * sf

    hard_block_threshold = 2.0 * limit

    if cvar > hard_block_threshold:
        return RiskCheckResult(
            passed=False,
            rule="cvar_limit",
            message=(
                f"[{result.method_used}] CVaR {cvar:.2%} exceeds hard limit "
                f"({hard_block_threshold:.2%} = 2× {limit:.2%}). "
                f"Worst stress scenario: {result.worst_stress_scenario} "
                f"({result.stress_cvar:.2%}). Trade blocked."
            ),
            current_value=cvar,
            limit_value=hard_block_threshold,
        )

    if cvar > limit:
        return RiskCheckResult(
            passed=True,
            rule="cvar_limit",
            message=(
                f"[{result.method_used}] CVaR {cvar:.2%} > limit {limit:.2%}. "
                f"Scale position to {sf:.0%} of proposed size "
                f"(proposed {proposed_trade_size:.2%} → recommended {scaled_size:.2%}). "
                f"CI: [{result.cvar_ci_lower:.2%}, {result.cvar_ci_upper:.2%}]. "
                f"Worst stress: {result.worst_stress_scenario} ({result.stress_cvar:.2%})."
            ),
            current_value=cvar,
            limit_value=limit,
        )

    return RiskCheckResult(
        passed=True,
        rule="cvar_limit",
        message=(
            f"[{result.method_used}] CVaR {cvar:.2%} within limit {limit:.2%}. "
            f"CI: [{result.cvar_ci_lower:.2%}, {result.cvar_ci_upper:.2%}]."
        ),
        current_value=cvar,
        limit_value=limit,
    )


def check_exchange_concentration(
    exchange: str,
    exchange_weights: dict[str, float],
    trade_weight: float,
    max_exchange_concentration: float,
) -> RiskCheckResult:
    """Ensure no single exchange exceeds ``max_exchange_concentration`` of NAV.

    Track C structural-arb strategies often trade across multiple venues
    (NYSE, CME, Kalshi, crypto exchanges).  This check limits exposure to
    any one exchange to mitigate venue-concentration risk (exchange outage,
    counterparty failure, etc.).

    Parameters
    ----------
    exchange:
        Exchange identifier for the proposed trade (e.g. ``"NYSE"``,
        ``"CME"``, ``"KALSHI"``).
    exchange_weights:
        Current portfolio weight per exchange as fractions of NAV.
    trade_weight:
        Additional weight this trade would add on the target exchange.
    max_exchange_concentration:
        Maximum allowed weight on a single exchange as a fraction of NAV.
    """
    current_weight = exchange_weights.get(exchange, 0.0)
    projected = current_weight + trade_weight
    passed = projected <= max_exchange_concentration
    return RiskCheckResult(
        passed=passed,
        rule="exchange_concentration",
        message=(
            f"Exchange '{exchange}' weight after trade {projected:.2%} "
            f"{'<=' if passed else '>'} limit {max_exchange_concentration:.2%} "
            f"(current {current_weight:.2%})."
        ),
        current_value=projected,
        limit_value=max_exchange_concentration,
    )


def check_drawdown_limit(
    current_nav: float,
    peak_nav: float,
    max_drawdown_pct: float,
) -> RiskCheckResult:
    """Block new BUY signals when portfolio drawdown approaches the limit.

    Triggers when current drawdown exceeds (max_drawdown_pct - 0.03),
    giving a 3% buffer before hitting the hard limit.

    Parameters
    ----------
    current_nav:
        Current net asset value.
    peak_nav:
        Highest NAV recorded so far.
    max_drawdown_pct:
        Maximum allowed drawdown as a positive fraction (e.g. 0.15 for 15%).
    """
    if peak_nav <= 0.0:
        return RiskCheckResult(
            passed=True,
            rule="drawdown_limit",
            message="No peak NAV recorded yet.",
        )

    current_dd = (current_nav - peak_nav) / peak_nav  # negative when in drawdown
    threshold = -(max_drawdown_pct - 0.03)  # e.g., -0.12 for 15% limit

    passed = current_dd >= threshold
    return RiskCheckResult(
        passed=passed,
        rule="drawdown_limit",
        message=(
            f"Current drawdown {current_dd:.2%} "
            f"{'>=' if passed else '<'} threshold {threshold:.2%} "
            f"(hard limit {-max_drawdown_pct:.2%})."
        ),
        current_value=current_dd,
        limit_value=threshold,
    )

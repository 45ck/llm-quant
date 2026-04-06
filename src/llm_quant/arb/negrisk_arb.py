"""NegRisk complement arbitrage strategy for Polymarket.

On Polymarket, multi-outcome events (e.g. "Which state will be the 51st?",
"Who wins Republican nomination?") use the NegRisk framework. Each outcome
has a YES token. The sum of all YES prices should equal 1.00. When the sum
is significantly below 1.00 (after fees), buying all YES tokens guarantees
profit at resolution because exactly one outcome resolves YES (paying $1.00).

Fee structure:
  - Standard fee: 2% of payout (not cost)
  - Geopolitics/politics events: 0% fee (fee-free)
  - For complement arb: total fee = sum(fee_rate * price_i) for each outcome
  - Net profit = (1.0 - sum(prices)) - total_fee
  - Kelly fraction: f* = net_profit / (1.0 + net_profit)

Position sizing ($100 bankroll):
  - Max per trade: $20 (20% of bankroll)
  - Kelly bet = kelly_fraction * bankroll, capped at $20
  - Min bet: $1.00
  - Liquidity check: suggested_size <= 10% of min 24h volume across outcomes

Track C -- Structural Arbitrage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_quant.arb.clob_client import ClobClient
from llm_quant.arb.gamma_client import GammaClient

logger = logging.getLogger(__name__)

# Fee-free categories on Polymarket
_FEE_FREE_CATEGORIES = {"politics", "geopolitics"}

# Standard winning fee on Polymarket (2% of payout)
_STANDARD_FEE_RATE = 0.02

# Position sizing defaults
_DEFAULT_BANKROLL = 100.0
_MAX_POSITION_PCT = 0.20  # 20% of bankroll
_MIN_BET_USD = 1.0
_LIQUIDITY_PCT = 0.10  # suggested size <= 10% of min 24h volume

# Scanner defaults
_DEFAULT_MIN_PROFIT_PCT = 1.0  # 1% minimum net profit
_DEFAULT_MIN_VOLUME = 5.0  # $5 minimum 24h volume per outcome
_DEFAULT_MAX_OUTCOMES = 50  # skip events with > 50 outcomes (data quality)

# Rate limiting: max 2 requests/sec
_RATE_LIMIT_SLEEP = 0.5  # seconds between API calls


@dataclass
class NegRiskOpportunity:
    """A detected NegRisk complement arbitrage opportunity."""

    condition_id: str
    event_slug: str
    question: str
    n_outcomes: int
    prices: list[float]  # YES price for each outcome
    token_ids: list[str]  # token IDs for each outcome
    total_cost: float  # sum of all YES prices
    gross_profit: float  # 1.0 - total_cost
    fee_rate: float  # category-dependent fee (0.00 for geopolitics, 0.02 standard)
    net_profit: float  # gross_profit - total_fees
    net_profit_pct: float  # net_profit / total_cost * 100
    kelly_fraction: float  # optimal bet size
    suggested_size_usd: float  # kelly * bankroll, capped
    volumes: list[float]  # 24h volume per outcome
    min_volume: float  # minimum volume across outcomes (liquidity constraint)
    detected_at: str  # ISO timestamp
    outcome_labels: list[str] = field(default_factory=list)

    def display(self) -> str:
        """Human-readable summary line."""
        return (
            f"{self.event_slug[:40]:<40s} | "
            f"N={self.n_outcomes:>2d} | "
            f"cost={self.total_cost:.3f} | "
            f"net={self.net_profit_pct:>5.1f}% | "
            f"vol=${self.min_volume:>8,.0f} | "
            f"kelly=${self.suggested_size_usd:>5.1f}"
        )


def get_fee_rate(category: str) -> float:
    """Fee rate by category. Geopolitics/politics = 0%, standard = 2%."""
    if category.lower() in _FEE_FREE_CATEGORIES:
        return 0.0
    return _STANDARD_FEE_RATE


def calculate_fees(prices: list[float], fee_rate: float) -> float:
    """Calculate total fees for buying all YES outcomes.

    Fee is charged on payout, not cost. For complement arb, payout per
    outcome = price_i (you paid price_i and receive $1 for the winner,
    so fee applies to each position's payout value).

    Total fee = sum(fee_rate * price_i) for each outcome bought.
    """
    if fee_rate <= 0:
        return 0.0
    return fee_rate * sum(prices)


def calculate_kelly(net_profit: float) -> float:
    """Kelly fraction for guaranteed-payoff arb.

    Since complement arb is a guaranteed payoff (one outcome always resolves
    YES), the simplified Kelly is: f* = net_profit / (1.0 + net_profit).

    Returns 0.0 if net_profit <= 0.
    """
    if net_profit <= 0:
        return 0.0
    return net_profit / (1.0 + net_profit)


def calculate_position_size(
    kelly_fraction: float,
    bankroll: float,
    min_volume: float,
    max_position_pct: float = _MAX_POSITION_PCT,
    min_bet: float = _MIN_BET_USD,
    liquidity_pct: float = _LIQUIDITY_PCT,
) -> float:
    """Calculate position size with Kelly, bankroll cap, and liquidity check.

    Parameters
    ----------
    kelly_fraction : float
        Kelly optimal fraction (0 to 1).
    bankroll : float
        Total bankroll in USD.
    min_volume : float
        Minimum 24h volume across all outcomes.
    max_position_pct : float
        Maximum fraction of bankroll per trade (default 20%).
    min_bet : float
        Minimum bet size in USD (default $1.00).
    liquidity_pct : float
        Max fraction of min volume for the position (default 10%).

    Returns
    -------
    float
        Suggested position size in USD. Returns 0.0 if below min_bet
        or liquidity constraint is binding to below min_bet.
    """
    # Kelly-sized bet capped at max position
    size = min(kelly_fraction * bankroll, max_position_pct * bankroll)

    # Liquidity cap: don't take more than 10% of minimum volume
    if min_volume > 0:
        liquidity_cap = liquidity_pct * min_volume
        size = min(size, liquidity_cap)

    # Below minimum bet threshold
    if size < min_bet:
        return 0.0

    return round(size, 2)


class NegRiskScanner:
    """Scans Polymarket for NegRisk complement arbitrage opportunities.

    Uses the Gamma API for event/market discovery and the CLOB API for
    live orderbook prices (more accurate than Gamma's cached prices).

    Parameters
    ----------
    gamma_client : GammaClient
        Client for Polymarket Gamma API (market discovery).
    clob_client : ClobClient
        Client for Polymarket CLOB API (live prices).
    bankroll : float
        Total bankroll for position sizing (default $100).
    """

    def __init__(
        self,
        gamma_client: GammaClient,
        clob_client: ClobClient,
        bankroll: float = _DEFAULT_BANKROLL,
    ) -> None:
        self._gamma = gamma_client
        self._clob = clob_client
        self._bankroll = bankroll

    def scan_event(
        self,
        event_slug: str,
        min_profit_pct: float = _DEFAULT_MIN_PROFIT_PCT,
        min_volume: float = _DEFAULT_MIN_VOLUME,
        max_outcomes: int = _DEFAULT_MAX_OUTCOMES,
    ) -> NegRiskOpportunity | None:
        """Scan a specific event for complement arb.

        Parameters
        ----------
        event_slug : str
            Polymarket event slug or event ID.
        min_profit_pct : float
            Minimum net profit percentage to report (default 1%).
        min_volume : float
            Minimum 24h volume per outcome (default $5).
        max_outcomes : int
            Skip events with more than this many outcomes.

        Returns
        -------
        NegRiskOpportunity or None
            Opportunity if found, None otherwise.
        """
        # Fetch event and its nested markets via Gamma API
        try:
            event_data = self._gamma.fetch_event(event_slug)
        except Exception as exc:
            logger.warning("Failed to fetch event %s: %s", event_slug, exc)
            return None

        return self._analyze_event(
            event_data,
            min_profit_pct=min_profit_pct,
            min_volume=min_volume,
            max_outcomes=max_outcomes,
        )

    def scan_all_active(
        self,
        min_profit_pct: float = _DEFAULT_MIN_PROFIT_PCT,
        min_volume: float = _DEFAULT_MIN_VOLUME,
        max_outcomes: int = _DEFAULT_MAX_OUTCOMES,
    ) -> list[NegRiskOpportunity]:
        """Scan all active NegRisk events for complement arb.

        Parameters
        ----------
        min_profit_pct : float
            Minimum net profit percentage to report (default 1%).
        min_volume : float
            Minimum 24h volume per outcome (default $5).
        max_outcomes : int
            Skip events with more than this many outcomes.

        Returns
        -------
        list[NegRiskOpportunity]
            Opportunities sorted by net_profit_pct descending.
        """
        # Fetch all active events (includes nested markets)
        logger.info("Fetching all active events from Gamma API...")
        events = self._gamma.fetch_all_active_events()
        logger.info("Fetched %d events total", len(events))

        # Filter to NegRisk events with multiple markets
        negrisk_events = [
            e
            for e in events
            if e.get("negRisk") or e.get("isNegRisk") or e.get("is_neg_risk")
        ]
        logger.info("NegRisk events: %d", len(negrisk_events))

        opportunities: list[NegRiskOpportunity] = []
        for event_data in negrisk_events:
            opp = self._analyze_event(
                event_data,
                min_profit_pct=min_profit_pct,
                min_volume=min_volume,
                max_outcomes=max_outcomes,
            )
            if opp is not None:
                opportunities.append(opp)
            # Rate limit
            time.sleep(_RATE_LIMIT_SLEEP)

        # Sort by net profit percentage descending
        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)

        logger.info("Found %d NegRisk complement arb opportunities", len(opportunities))
        return opportunities

    def _analyze_event(  # noqa: PLR0911
        self,
        event_data: dict[str, Any],
        min_profit_pct: float,
        min_volume: float,
        max_outcomes: int,
    ) -> NegRiskOpportunity | None:
        """Analyze a single event for complement arb opportunity.

        Parameters
        ----------
        event_data : dict
            Raw event dict from Gamma API (includes nested markets).
        min_profit_pct : float
            Minimum net profit percentage threshold.
        min_volume : float
            Minimum 24h volume per outcome.
        max_outcomes : int
            Maximum number of outcomes to consider.

        Returns
        -------
        NegRiskOpportunity or None
        """
        event_slug = event_data.get("slug", event_data.get("id", "unknown"))
        question = event_data.get("title", event_data.get("question", ""))
        category = event_data.get("category", "")
        if not category:
            category = _infer_category_from_event(event_data)

        # Extract markets (outcomes) from event
        markets = event_data.get("markets", [])
        if not markets or len(markets) < 2:
            logger.debug("Event %s has %d markets, skipping", event_slug, len(markets))
            return None

        if len(markets) > max_outcomes:
            logger.debug(
                "Event %s has %d markets (> %d), skipping",
                event_slug,
                len(markets),
                max_outcomes,
            )
            return None

        # Parse each market's prices and token IDs
        prices: list[float] = []
        token_ids: list[str] = []
        volumes: list[float] = []
        outcome_labels: list[str] = []
        condition_id = event_data.get("conditionId", event_data.get("id", ""))

        for mkt in markets:
            parsed = GammaClient.parse_market(mkt)
            if parsed is None or not parsed.conditions:
                continue

            cond = parsed.conditions[0]
            yes_price = cond.outcome_yes

            # Try to get live price from CLOB if token IDs are available
            if cond.clob_token_ids:
                yes_token_id = cond.clob_token_ids[0]  # first is YES
                live_price = self._get_clob_price(yes_token_id)
                if live_price is not None:
                    yes_price = live_price
                token_ids.append(yes_token_id)
            else:
                token_ids.append(cond.condition_id)

            prices.append(yes_price)
            volumes.append(cond.volume_24h)
            outcome_labels.append(
                mkt.get("question", mkt.get("title", f"outcome_{len(prices)}"))
            )

        if len(prices) < 2:
            return None

        # Calculate arb metrics
        total_cost = sum(prices)
        gross_profit = 1.0 - total_cost

        if gross_profit <= 0:
            return None

        # Fee calculation
        fee_rate = get_fee_rate(category)
        total_fees = calculate_fees(prices, fee_rate)
        net_profit = gross_profit - total_fees

        if net_profit <= 0:
            return None

        net_profit_pct = (net_profit / total_cost) * 100.0 if total_cost > 0 else 0.0

        if net_profit_pct < min_profit_pct:
            return None

        # Volume / liquidity check
        event_min_volume = min(volumes) if volumes else 0.0
        if event_min_volume < min_volume:
            logger.debug(
                "Event %s: min_volume=%.1f < threshold=%.1f, skipping",
                event_slug,
                event_min_volume,
                min_volume,
            )
            return None

        # Kelly sizing
        kelly = calculate_kelly(net_profit)
        suggested_size = calculate_position_size(
            kelly_fraction=kelly,
            bankroll=self._bankroll,
            min_volume=event_min_volume,
        )

        opp = NegRiskOpportunity(
            condition_id=condition_id,
            event_slug=event_slug,
            question=question,
            n_outcomes=len(prices),
            prices=prices,
            token_ids=token_ids,
            total_cost=total_cost,
            gross_profit=gross_profit,
            fee_rate=fee_rate,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
            kelly_fraction=kelly,
            suggested_size_usd=suggested_size,
            volumes=volumes,
            min_volume=event_min_volume,
            detected_at=datetime.now(UTC).isoformat(),
            outcome_labels=outcome_labels,
        )
        logger.info("NegRisk complement arb: %s", opp.display())
        return opp

    def _get_clob_price(self, token_id: str) -> float | None:
        """Get live YES price from CLOB API (best ask for buying).

        Returns None if the price cannot be fetched.
        """
        try:
            return self._clob.get_price(token_id, side="BUY")
        except Exception as exc:
            logger.debug("CLOB price fetch failed for %s: %s", token_id[:20], exc)
            return None


def _infer_category_from_event(event_data: dict[str, Any]) -> str:
    """Infer category from event data if not explicitly provided."""
    text = (
        event_data.get("title", "")
        + " "
        + event_data.get("slug", "")
        + " "
        + event_data.get("description", "")
    )
    from llm_quant.arb.gamma_client import _infer_category

    return _infer_category(text)


# ------------------------------------------------------------------
# Paper logging
# ------------------------------------------------------------------


def log_opportunity_to_paper(
    opp: NegRiskOpportunity,
    log_path: Path,
    action: str = "DETECTED",
) -> None:
    """Append a NegRisk opportunity to the paper trading log.

    Parameters
    ----------
    opp : NegRiskOpportunity
        The detected opportunity.
    log_path : Path
        Path to the JSONL paper log file.
    action : str
        Action label: "DETECTED" or "WOULD_BUY".
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = asdict(opp)
    record["action"] = action
    record["logged_at"] = datetime.now(UTC).isoformat()

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "Paper logged %s: %s (net=%.1f%%)",
        action,
        opp.event_slug,
        opp.net_profit_pct,
    )

"""Crypto cash-and-carry basis trade scanner.

Scans spot vs quarterly futures prices across exchanges to identify
basis spread opportunities. The cash-and-carry trade captures the
spread between spot and futures by going long spot + short futures,
with guaranteed convergence at expiry.

Track C — Niche Arbitrage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import ccxt

logger = logging.getLogger(__name__)

# Supported symbols for basis scanning
DEFAULT_SYMBOLS = ["BTC", "ETH"]

# Supported exchanges with quarterly futures
DEFAULT_EXCHANGES = ["binance", "okx", "bybit"]

# Rate limit delay between API calls (seconds)
API_DELAY_SECS = 0.5

# Filtering thresholds
MIN_ANNUALIZED_BASIS_PCT = 3.0  # 3% annualized minimum
MAX_DAYS_TO_EXPIRY = 90
MIN_DAYS_TO_EXPIRY = 1  # Avoid expiring-today contracts

# Standard quarterly expiry months: March, June, September, December
QUARTERLY_MONTHS = [3, 6, 9, 12]

# Quarterly expiry day (last Friday of the month for most exchanges;
# Binance/OKX/Bybit use the last Friday)
QUARTERLY_EXPIRY_DAY = 27  # Approximate — close enough for annualization


@dataclass
class BasisOpportunity:
    """A detected spot-futures basis spread opportunity."""

    symbol: str  # e.g. "BTC"
    exchange: str  # e.g. "binance"
    spot_price: float  # current spot price
    futures_price: float  # quarterly futures price
    futures_symbol: str  # exchange-specific futures symbol
    expiry_date: date  # futures contract expiry
    days_to_expiry: int  # trading days remaining
    raw_basis: float  # (futures - spot) / spot
    annualized_basis: float  # raw_basis * (365 / days_to_expiry)
    timestamp: datetime  # when this was observed

    @property
    def display_basis(self) -> str:
        return f"{self.raw_basis * 100:.3f}%"

    @property
    def display_annualized(self) -> str:
        return f"{self.annualized_basis * 100:.2f}%"

    @property
    def premium_usd(self) -> float:
        """Absolute premium in USD per unit."""
        return self.futures_price - self.spot_price


def compute_raw_basis(spot_price: float, futures_price: float) -> float:
    """Compute raw basis spread: (futures - spot) / spot.

    A positive basis means futures trade at a premium (contango).
    A negative basis means futures trade at a discount (backwardation).
    """
    if spot_price <= 0:
        return 0.0
    return (futures_price - spot_price) / spot_price


def annualize_basis(raw_basis: float, days_to_expiry: int) -> float:
    """Annualize the raw basis spread.

    Formula: raw_basis * (365 / days_to_expiry)

    This converts the spread earned over the contract's remaining life
    into an equivalent annual rate for comparison.
    """
    if days_to_expiry <= 0:
        return 0.0
    return raw_basis * (365.0 / days_to_expiry)


def compute_fee_adjusted_basis(
    raw_basis: float,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0005,
) -> float:
    """Compute basis after round-trip trading fees.

    Cash-and-carry requires:
      - Buy spot (taker): 1 * taker_fee
      - Sell futures (maker): 1 * maker_fee
      - Close at expiry: futures settle automatically (0 fee on most exchanges)
      - Sell spot at expiry: 1 * taker_fee

    Total fee drag = 2 * taker_fee + maker_fee
    """
    total_fees = 2 * taker_fee + maker_fee
    return raw_basis - total_fees


def get_upcoming_expiries(
    as_of: date | None = None,
    max_days: int = MAX_DAYS_TO_EXPIRY,
) -> list[date]:
    """Return upcoming quarterly expiry dates within max_days.

    Quarterly contracts typically expire on the last Friday of
    March, June, September, December. We approximate with day 27.
    """
    if as_of is None:
        as_of = datetime.now(UTC).date()

    expiries: list[date] = []
    # Check current year and next year
    for year in [as_of.year, as_of.year + 1]:
        for month in QUARTERLY_MONTHS:
            try:
                expiry = date(year, month, QUARTERLY_EXPIRY_DAY)
            except ValueError:
                # Month doesn't have 27 days (shouldn't happen for these months)
                continue

            days_away = (expiry - as_of).days
            if MIN_DAYS_TO_EXPIRY <= days_away <= max_days:
                expiries.append(expiry)

    return sorted(expiries)


def format_expiry_suffix(expiry: date) -> str:
    """Format an expiry date as YYMMDD for exchange symbol construction.

    Example: date(2025, 6, 27) -> "250627"
    """
    return expiry.strftime("%y%m%d")


def build_futures_symbol(
    base_symbol: str,
    expiry: date,
    _exchange_id: str,
) -> str:
    """Build exchange-specific quarterly futures symbol.

    Formats:
      - Binance: BTC/USDT:USDT-250627
      - OKX:     BTC/USDT:USDT-250627
      - Bybit:   BTC/USDT:USDT-250627

    CCXT unified format is the same across these three exchanges.
    """
    suffix = format_expiry_suffix(expiry)
    return f"{base_symbol}/USDT:USDT-{suffix}"


def build_spot_symbol(base_symbol: str) -> str:
    """Build spot symbol for CCXT.

    Returns: "BTC/USDT", "ETH/USDT", etc.
    """
    return f"{base_symbol}/USDT"


@dataclass
class BasisScanner:
    """Scans spot vs quarterly futures basis spreads across exchanges.

    Uses CCXT public API (no authentication required) to fetch spot
    and quarterly futures prices, compute basis spreads, and rank
    opportunities by annualized yield.
    """

    exchanges: list[str] = field(default_factory=lambda: list(DEFAULT_EXCHANGES))
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    api_delay: float = API_DELAY_SECS
    min_annualized_pct: float = MIN_ANNUALIZED_BASIS_PCT
    max_days_to_expiry: int = MAX_DAYS_TO_EXPIRY
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%

    def _create_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """Instantiate a CCXT exchange object with futures enabled."""
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )

    def _fetch_spot_price(
        self,
        exchange: ccxt.Exchange,
        base_symbol: str,
    ) -> float | None:
        """Fetch current spot price for a symbol.

        Creates a separate spot exchange instance since the main one
        is configured for futures.
        """
        spot_symbol = build_spot_symbol(base_symbol)
        try:
            # Create a spot exchange instance
            spot_exchange_class = getattr(ccxt, exchange.id)
            spot_exchange = spot_exchange_class(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            spot_exchange.load_markets()

            if spot_symbol not in spot_exchange.markets:
                logger.debug(
                    "Spot symbol %s not available on %s.",
                    spot_symbol,
                    exchange.id,
                )
                return None

            ticker = spot_exchange.fetch_ticker(spot_symbol)
            last_price = ticker.get("last")
        except Exception as exc:
            logger.warning(
                "Failed to fetch spot price for %s on %s: %s",
                spot_symbol,
                exchange.id,
                exc,
            )
            return None
        else:
            if last_price is not None:
                return float(last_price)
            return None

    def _fetch_futures_price(
        self,
        exchange: ccxt.Exchange,
        futures_symbol: str,
    ) -> float | None:
        """Fetch current futures price for a quarterly contract."""
        try:
            exchange.load_markets()

            if futures_symbol not in exchange.markets:
                logger.debug(
                    "Futures symbol %s not available on %s.",
                    futures_symbol,
                    exchange.id,
                )
                return None

            ticker = exchange.fetch_ticker(futures_symbol)
            last_price = ticker.get("last")
        except Exception as exc:
            logger.warning(
                "Failed to fetch futures price for %s on %s: %s",
                futures_symbol,
                exchange.id,
                exc,
            )
            return None
        else:
            if last_price is not None:
                return float(last_price)
            return None

    def scan(
        self,
        as_of: date | None = None,
    ) -> list[BasisOpportunity]:
        """Scan all exchanges and symbols for basis opportunities.

        Returns opportunities sorted by annualized basis descending,
        filtered by minimum annualized basis and maximum days to expiry.
        """
        if as_of is None:
            as_of = datetime.now(UTC).date()

        expiries = get_upcoming_expiries(as_of, self.max_days_to_expiry)
        if not expiries:
            logger.info(
                "No upcoming quarterly expiries within %d days.",
                self.max_days_to_expiry,
            )
            return []

        opportunities: list[BasisOpportunity] = []
        now = datetime.now(UTC)

        for exch_id in self.exchanges:
            try:
                exchange = self._create_exchange(exch_id)
            except AttributeError:
                logger.warning("Exchange '%s' not supported by CCXT.", exch_id)
                continue

            for base_symbol in self.symbols:
                # Fetch spot price first
                spot_price = self._fetch_spot_price(exchange, base_symbol)
                if spot_price is None:
                    logger.debug(
                        "No spot price for %s on %s, skipping.",
                        base_symbol,
                        exch_id,
                    )
                    time.sleep(self.api_delay)
                    continue

                # Check each upcoming quarterly expiry
                for expiry in expiries:
                    futures_sym = build_futures_symbol(base_symbol, expiry, exch_id)
                    futures_price = self._fetch_futures_price(exchange, futures_sym)

                    if futures_price is None:
                        time.sleep(self.api_delay)
                        continue

                    days_to_expiry = (expiry - as_of).days
                    if days_to_expiry < MIN_DAYS_TO_EXPIRY:
                        continue

                    raw = compute_raw_basis(spot_price, futures_price)
                    ann = annualize_basis(raw, days_to_expiry)

                    # Apply fee adjustment for display/filtering
                    fee_adjusted = compute_fee_adjusted_basis(
                        raw, self.maker_fee, self.taker_fee
                    )
                    ann_fee_adjusted = annualize_basis(fee_adjusted, days_to_expiry)

                    # Filter: minimum annualized basis (after fees)
                    if ann_fee_adjusted * 100 < self.min_annualized_pct:
                        logger.debug(
                            "%s %s %s: ann=%.2f%% (fee-adj=%.2f%%) — below threshold.",
                            exch_id,
                            base_symbol,
                            expiry,
                            ann * 100,
                            ann_fee_adjusted * 100,
                        )
                        continue

                    opp = BasisOpportunity(
                        symbol=base_symbol,
                        exchange=exch_id,
                        spot_price=spot_price,
                        futures_price=futures_price,
                        futures_symbol=futures_sym,
                        expiry_date=expiry,
                        days_to_expiry=days_to_expiry,
                        raw_basis=raw,
                        annualized_basis=ann,
                        timestamp=now,
                    )

                    logger.info(
                        "%s %s %s: spot=%.2f, fut=%.2f, basis=%.3f%%, ann=%.2f%%",
                        exch_id,
                        base_symbol,
                        expiry,
                        spot_price,
                        futures_price,
                        raw * 100,
                        ann * 100,
                    )
                    opportunities.append(opp)

                    time.sleep(self.api_delay)

        # Sort by annualized basis descending (best opportunities first)
        opportunities.sort(key=lambda o: o.annualized_basis, reverse=True)
        return opportunities


def format_basis_report(opportunities: list[BasisOpportunity]) -> str:
    """Format basis scan results as a readable text report."""
    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("=" * 78)
    lines.append(f"  CRYPTO BASIS SCANNER — {now}")
    lines.append("=" * 78)

    lines.append(f"\n  CASH-AND-CARRY OPPORTUNITIES ({len(opportunities)} found)")
    lines.append("-" * 78)

    if not opportunities:
        lines.append("  No basis spreads above threshold.")
    else:
        lines.append(
            f"  {'Symbol':<6} {'Exchange':<10} {'Spot':>10} {'Futures':>10} "
            f"{'Basis':>8} {'Ann.':>8} {'Expiry':<12} {'Days':>5}"
        )
        lines.append("  " + "-" * 74)
        lines.extend(
            f"  {opp.symbol:<6} {opp.exchange:<10} "
            f"${opp.spot_price:>9,.2f} ${opp.futures_price:>9,.2f} "
            f"{opp.raw_basis * 100:>+7.3f}% "
            f"{opp.annualized_basis * 100:>+7.2f}% "
            f"{opp.expiry_date.isoformat():<12} {opp.days_to_expiry:>5d}"
            for opp in opportunities
        )

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)

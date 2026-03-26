"""Funding rate opportunity scanner.

Scans current funding rates across exchanges, identifies:
1. High absolute rates (carry opportunities)
2. Cross-exchange differentials (funding rate arb)

Track C — Niche Arbitrage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from llm_quant.arb.funding_rates import (
    FundingRecord,
    annualize_funding_rate,
)

logger = logging.getLogger(__name__)

# Default threshold: 0.01% per 8h = ~10.95% annualized
DEFAULT_RATE_THRESHOLD = 0.0001
DEFAULT_DIFFERENTIAL_THRESHOLD = 0.00005  # 0.005% per 8h cross-exchange


@dataclass
class FundingOpportunity:
    """A detected funding rate opportunity."""

    opp_type: str  # "high_rate" or "cross_exchange"
    symbol: str
    exchange: str
    funding_rate: float  # raw 8h rate
    annualized_rate: float
    mark_price: float | None
    timestamp: datetime
    # Cross-exchange fields (None for high_rate type)
    counter_exchange: str | None = None
    counter_rate: float | None = None
    counter_annualized: float | None = None
    differential: float | None = None  # raw 8h differential
    differential_annualized: float | None = None

    @property
    def display_rate(self) -> str:
        return f"{self.funding_rate * 100:.4f}%"

    @property
    def display_annualized(self) -> str:
        return f"{self.annualized_rate * 100:.2f}%"


class FundingScanner:
    """Scans funding rates for carry and cross-exchange opportunities."""

    def __init__(
        self,
        rate_threshold: float = DEFAULT_RATE_THRESHOLD,
        differential_threshold: float = DEFAULT_DIFFERENTIAL_THRESHOLD,
    ):
        self.rate_threshold = rate_threshold
        self.differential_threshold = differential_threshold

    def scan_high_rates(
        self,
        records: list[FundingRecord],
    ) -> list[FundingOpportunity]:
        """Identify symbols with funding rate above threshold.

        High positive rate = shorts pay longs (short the perp, long spot).
        High negative rate = longs pay shorts (long the perp, short spot).
        """
        opps = [
            FundingOpportunity(
                opp_type="high_rate",
                symbol=r.symbol,
                exchange=r.exchange,
                funding_rate=r.funding_rate,
                annualized_rate=r.annualized_rate,
                mark_price=r.mark_price,
                timestamp=r.timestamp,
            )
            for r in records
            if abs(r.funding_rate) >= self.rate_threshold
        ]

        # Sort by absolute annualized rate descending
        opps.sort(key=lambda o: abs(o.annualized_rate), reverse=True)
        return opps

    def scan_cross_exchange(
        self,
        records: list[FundingRecord],
    ) -> list[FundingOpportunity]:
        """Identify cross-exchange funding rate differentials.

        If Binance BTC rate is 0.03% and OKX BTC rate is 0.01%,
        the differential is 0.02% — you could short perp on Binance
        (collect higher rate) and long perp on OKX (pay lower rate).
        """
        opps: list[FundingOpportunity] = []

        # Group by symbol
        by_symbol: dict[str, list[FundingRecord]] = {}
        for r in records:
            by_symbol.setdefault(r.symbol, []).append(r)

        for symbol, symbol_records in by_symbol.items():
            if len(symbol_records) < 2:
                continue

            # Compare all pairs
            for i, a in enumerate(symbol_records):
                for b in symbol_records[i + 1 :]:
                    diff = a.funding_rate - b.funding_rate
                    if abs(diff) >= self.differential_threshold:
                        # Report with the higher-rate exchange first
                        if diff > 0:
                            high, low = a, b
                        else:
                            high, low = b, a
                            diff = -diff

                        opps.append(
                            FundingOpportunity(
                                opp_type="cross_exchange",
                                symbol=symbol,
                                exchange=high.exchange,
                                funding_rate=high.funding_rate,
                                annualized_rate=high.annualized_rate,
                                mark_price=high.mark_price,
                                timestamp=high.timestamp,
                                counter_exchange=low.exchange,
                                counter_rate=low.funding_rate,
                                counter_annualized=low.annualized_rate,
                                differential=diff,
                                differential_annualized=annualize_funding_rate(diff),
                            )
                        )

        # Sort by differential descending
        opps.sort(
            key=lambda o: abs(o.differential_annualized or 0),
            reverse=True,
        )
        return opps

    def scan_all(
        self,
        records: list[FundingRecord],
    ) -> list[FundingOpportunity]:
        """Run both high-rate and cross-exchange scans, return combined results."""
        high = self.scan_high_rates(records)
        cross = self.scan_cross_exchange(records)
        return high + cross


def format_scan_report(
    high_rate_opps: list[FundingOpportunity],
    cross_exchange_opps: list[FundingOpportunity],
) -> str:
    """Format scan results as a readable text report."""
    lines: list[str] = []
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("=" * 70)
    lines.append(f"  FUNDING RATE SCANNER — {now}")
    lines.append("=" * 70)

    # High rate opportunities
    lines.append(f"\n  HIGH FUNDING RATES ({len(high_rate_opps)} found)")
    lines.append("-" * 70)

    if not high_rate_opps:
        lines.append("  No rates above threshold.")
    else:
        lines.append(
            f"  {'Symbol':<12} {'Exchange':<10} {'Rate(8h)':<12} "
            f"{'Annualized':<14} {'Direction':<12} {'Mark Price':<12}"
        )
        lines.append("  " + "-" * 66)
        for opp in high_rate_opps:
            direction = "LONGS PAY" if opp.funding_rate > 0 else "SHORTS PAY"
            mark = f"${opp.mark_price:,.2f}" if opp.mark_price else "N/A"
            lines.append(
                f"  {opp.symbol:<12} {opp.exchange:<10} "
                f"{opp.funding_rate * 100:>+.4f}%    "
                f"{opp.annualized_rate * 100:>+.2f}%        "
                f"{direction:<12} {mark:<12}"
            )

    # Cross-exchange differentials
    lines.append(f"\n  CROSS-EXCHANGE DIFFERENTIALS ({len(cross_exchange_opps)} found)")
    lines.append("-" * 70)

    if not cross_exchange_opps:
        lines.append("  No significant cross-exchange differentials.")
    else:
        for opp in cross_exchange_opps:
            diff_ann = opp.differential_annualized or 0
            lines.append(f"\n  {opp.symbol}:")
            lines.append(
                f"    {opp.exchange:>8}: {opp.funding_rate * 100:>+.4f}% "
                f"({opp.annualized_rate * 100:>+.2f}% ann)"
            )
            counter_rate = opp.counter_rate or 0
            counter_ann = opp.counter_annualized or 0
            lines.append(
                f"    {opp.counter_exchange or 'N/A':>8}: {counter_rate * 100:>+.4f}% "
                f"({counter_ann * 100:>+.2f}% ann)"
            )
            lines.append(
                f"    Spread: {(opp.differential or 0) * 100:.4f}% per 8h "
                f"= {diff_ann * 100:.2f}% annualized"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def rates_to_polars(records: list[FundingRecord]) -> pl.DataFrame:
    """Convert a list of FundingRecord to a Polars DataFrame."""
    if not records:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "exchange": pl.Utf8,
                "symbol": pl.Utf8,
                "funding_rate": pl.Float64,
                "annualized_rate": pl.Float64,
                "mark_price": pl.Float64,
            }
        )

    return pl.DataFrame(
        {
            "timestamp": [r.timestamp for r in records],
            "exchange": [r.exchange for r in records],
            "symbol": [r.symbol for r in records],
            "funding_rate": [r.funding_rate for r in records],
            "annualized_rate": [r.annualized_rate for r in records],
            "mark_price": [r.mark_price for r in records],
        }
    )

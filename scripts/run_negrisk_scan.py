#!/usr/bin/env python3
"""NegRisk complement arbitrage scanner CLI.

Scans Polymarket NegRisk events where buying all YES outcomes costs less
than $1.00, guaranteeing profit at resolution.

Usage:
  # Scan all active NegRisk events (default: min 1% net profit)
  python scripts/run_negrisk_scan.py

  # Scan with custom minimum profit threshold
  python scripts/run_negrisk_scan.py --min-profit 2.0

  # Scan with minimum volume filter
  python scripts/run_negrisk_scan.py --min-volume 100

  # Scan a specific event
  python scripts/run_negrisk_scan.py --event will-x-happen

  # Paper trade: log detected opportunities to paper log
  python scripts/run_negrisk_scan.py --paper

  # Combine flags
  python scripts/run_negrisk_scan.py --min-profit 0.5 --min-volume 10 --paper
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_quant.arb.clob_client import ClobClient
from llm_quant.arb.gamma_client import GammaClient
from llm_quant.arb.negrisk_arb import (
    NegRiskOpportunity,
    NegRiskScanner,
    log_opportunity_to_paper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("negrisk_scan")

DEFAULT_PAPER_LOG = Path("data/polymarket/paper_log.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NegRisk complement arbitrage scanner for Polymarket"
    )
    p.add_argument(
        "--min-profit",
        type=float,
        default=1.0,
        help="Minimum net profit %% to report (default: 1.0)",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=5.0,
        help="Minimum 24h volume per outcome in USD (default: 5.0)",
    )
    p.add_argument(
        "--event",
        type=str,
        default=None,
        help="Scan a single event by slug or ID",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="Record opportunities to paper log (data/polymarket/paper_log.jsonl)",
    )
    p.add_argument(
        "--paper-log",
        type=Path,
        default=DEFAULT_PAPER_LOG,
        help=f"Paper log path (default: {DEFAULT_PAPER_LOG})",
    )
    p.add_argument(
        "--bankroll",
        type=float,
        default=100.0,
        help="Bankroll for position sizing in USD (default: 100)",
    )
    p.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification",
    )
    p.add_argument(
        "--clob-prices",
        action="store_true",
        help="Use CLOB API for live prices (slower but more accurate)",
    )
    return p.parse_args()


def print_opportunities(opps: list[NegRiskOpportunity]) -> None:
    """Display opportunities as a formatted table."""
    if not opps:
        print("\n  No NegRisk complement arb opportunities found.\n")
        return

    print(f"\n{'=' * 90}")
    print(f"  NEGRISK COMPLEMENT ARB OPPORTUNITIES: {len(opps)}")
    print(f"{'=' * 90}")

    # Header
    print(
        f"\n  {'Event':<35s} {'N':>3s} {'Cost':>7s} {'Net%':>6s} "
        f"{'Fee':>5s} {'MinVol':>9s} {'Kelly$':>7s}"
    )
    print("  " + "-" * 80)

    for opp in opps:
        fee_label = f"{opp.fee_rate:.0%}" if opp.fee_rate > 0 else "free"
        print(
            f"  {opp.event_slug[:35]:<35s} {opp.n_outcomes:>3d} "
            f"{opp.total_cost:>7.3f} {opp.net_profit_pct:>5.1f}% "
            f"{fee_label:>5s} ${opp.min_volume:>8,.0f} "
            f"${opp.suggested_size_usd:>6.1f}"
        )

    print(f"\n{'=' * 90}")

    # Summary stats
    best = opps[0]
    avg_profit = sum(o.net_profit_pct for o in opps) / len(opps)
    total_kelly = sum(o.suggested_size_usd for o in opps)
    print(f"\n  Best opportunity: {best.event_slug} ({best.net_profit_pct:.1f}% net)")
    print(f"  Average net profit: {avg_profit:.1f}%")
    print(f"  Total Kelly allocation: ${total_kelly:.1f}")
    print()


def print_detail(opp: NegRiskOpportunity) -> None:
    """Print detailed breakdown of a single opportunity."""
    print(f"\n  Event:      {opp.event_slug}")
    print(f"  Question:   {opp.question[:80]}")
    print(f"  Outcomes:   {opp.n_outcomes}")
    print(f"  Total cost: ${opp.total_cost:.4f}")
    print(f"  Gross:      ${opp.gross_profit:.4f} ({opp.gross_profit * 100:.2f}%)")
    print(f"  Fee rate:   {opp.fee_rate:.0%}")
    print(f"  Net profit: ${opp.net_profit:.4f} ({opp.net_profit_pct:.2f}%)")
    print(f"  Kelly f*:   {opp.kelly_fraction:.4f}")
    print(f"  Size:       ${opp.suggested_size_usd:.2f}")
    print(f"  Min volume: ${opp.min_volume:,.0f}")
    print("\n  Outcomes:")
    for i, (price, vol) in enumerate(zip(opp.prices, opp.volumes, strict=False)):
        label = opp.outcome_labels[i][:50] if i < len(opp.outcome_labels) else f"#{i}"
        print(f"    {label:<50s} YES={price:.3f} vol=${vol:,.0f}")


def main() -> int:
    args = parse_args()
    ssl_verify = not args.no_ssl_verify

    print("\n" + "=" * 90)
    print("  NEGRISK COMPLEMENT ARBITRAGE SCANNER")
    print(f"  Min profit: {args.min_profit:.1f}% | Min volume: ${args.min_volume:,.0f}")
    print(f"  Bankroll: ${args.bankroll:,.0f} | Paper: {args.paper}")
    print("=" * 90 + "\n")

    gamma = GammaClient(ssl_verify=ssl_verify)
    clob = ClobClient(ssl_verify=ssl_verify) if args.clob_prices else None
    scanner = NegRiskScanner(
        gamma_client=gamma,
        clob_client=clob,
        bankroll=args.bankroll,
        use_clob_prices=args.clob_prices,
    )

    # Single event scan
    if args.event:
        print(f"Scanning event: {args.event}")
        opp = scanner.scan_event(
            event_slug=args.event,
            min_profit_pct=args.min_profit,
            min_volume=args.min_volume,
        )
        if opp:
            print_detail(opp)
            if args.paper:
                action = "WOULD_BUY" if opp.suggested_size_usd > 0 else "DETECTED"
                log_opportunity_to_paper(opp, args.paper_log, action=action)
                print(f"\n  Paper logged to {args.paper_log}")
        else:
            print(f"\n  No complement arb found for event: {args.event}")
        return 0

    # Full scan
    opps = scanner.scan_all_active(
        min_profit_pct=args.min_profit,
        min_volume=args.min_volume,
    )
    print_opportunities(opps)

    # Paper logging
    if args.paper and opps:
        for opp in opps:
            action = "WOULD_BUY" if opp.suggested_size_usd > 0 else "DETECTED"
            log_opportunity_to_paper(opp, args.paper_log, action=action)
        print(f"  Paper logged {len(opps)} opportunities to {args.paper_log}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

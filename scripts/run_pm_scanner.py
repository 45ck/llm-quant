#!/usr/bin/env python3
"""Prediction market arbitrage scanner CLI.

Sources:
  kalshi      — CFTC-regulated, US-legal (DEFAULT, no auth needed)
  polymarket  — Decentralised, higher volume (may be geo-blocked for US users)

Usage:
  python scripts/run_pm_scanner.py [--source kalshi|polymarket] [--mode negrisk|full]

Examples:
  # Kalshi NegRisk scan (US-legal, no auth required)
  python scripts/run_pm_scanner.py --source kalshi

  # Kalshi scan, elections category only
  python scripts/run_pm_scanner.py --source kalshi --category Elections

  # Polymarket full scan (requires API access)
  python scripts/run_pm_scanner.py --source polymarket --mode full

  # Dry-run: fetch + parse but don't write to DB
  python scripts/run_pm_scanner.py --source kalshi --dry-run

  # Run Claude combinatorial detector on sports markets
  python scripts/run_pm_scanner.py --source polymarket --mode full --detect-combinatorial
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_quant.arb.detector import CombinatorialDetector
from llm_quant.arb.gamma_client import GammaClient
from llm_quant.arb.scanner import ArbScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pm_scanner")

DEFAULT_DB = Path("data/quant.db")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prediction market arb scanner")
    p.add_argument(
        "--source",
        choices=["kalshi", "polymarket"],
        default="kalshi",
        help="Data source: kalshi (US-legal, default) or polymarket",
    )
    p.add_argument(
        "--mode",
        choices=["negrisk", "full"],
        default="negrisk",
        help="Scan mode: negrisk (default) or full (all arb types)",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Filter by category (Kalshi: Elections, Politics, Sports, etc.)",
    )
    p.add_argument(
        "--min-spread",
        type=float,
        default=0.00,
        help="Minimum net arb spread to report (default: 0 = show all with spread > fee)",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Minimum 24h volume per condition in USD (default: 0 = no filter)",
    )
    p.add_argument(
        "--max-markets",
        type=int,
        default=5000,
        help="Maximum markets to fetch (polymarket only, default: 5000)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + parse but do not write to DB",
    )
    p.add_argument(
        "--detect-combinatorial",
        action="store_true",
        help="Run Claude combinatorial detector (polymarket only)",
    )
    p.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (useful on Windows)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"DuckDB path (default: {DEFAULT_DB})",
    )
    return p.parse_args()


def print_opportunities(opps: list) -> None:
    if not opps:
        print("\n  No opportunities found above threshold.\n")
        return

    print(f"\n{'=' * 70}")
    print(f"  ARB OPPORTUNITIES DETECTED: {len(opps)}")
    print(f"{'=' * 70}")
    for i, opp in enumerate(opps, 1):
        print(f"\n  #{i} [{opp.arb_type.upper()}]")
        print(f"     Market:     {opp.market_id}")
        print(f"     Gross:      {opp.spread_pct:.2%}")
        print(f"     Net:        {opp.net_spread_pct:.2%}")
        print(f"     Kelly f*:   {opp.kelly_fraction:.2%}")
        print(f"     Volume 24h: ${opp.total_volume:,.0f}")
        if opp.notes:
            print(f"     Notes:      {opp.notes}")
    print(f"\n{'=' * 70}\n")


def run_combinatorial_detection(
    scanner: ArbScanner, client: GammaClient, db_path: Path
) -> None:
    """Run Claude detector on sports markets grouped by topic."""
    print("\nRunning Claude combinatorial dependency detector on sports markets...")

    raw = client.fetch_all_active_markets(max_markets=2000)
    markets = client.parse_all_markets(raw)
    sports = [m for m in markets if m.category == "sports" and m.conditions]

    if not sports:
        print("  No sports markets found for combinatorial analysis.")
        return

    print(f"  Analyzing {len(sports)} sports markets for logical dependencies...")

    detector = CombinatorialDetector(
        db_path=str(db_path),
        model="claude-haiku-4-5-20251001",  # Fast + cheap for screening
        min_confidence=0.80,
        min_arb_spread=0.03,
    )

    # Group markets by rough topic (first 3 words of question)
    from collections import defaultdict

    topic_groups: dict[str, list] = defaultdict(list)
    for m in sports:
        words = m.question.lower().split()[:3]
        key = " ".join(words)
        topic_groups[key].append(m)

    # Analyze groups with 2+ markets
    total_arb_pairs = 0
    for topic, group in list(topic_groups.items())[:20]:  # Cap at 20 groups
        if len(group) < 2:
            continue
        results = detector.analyze_market_group(group[:5])  # Cap at 5 per group
        arb = [r for r in results if r.is_arb and r.implied_arb_spread >= 0.03]
        if arb:
            print(f"\n  Topic: '{topic}' ({len(group)} markets)")
            for r in arb:
                print(f"    [{r.dependency_type}] confidence={r.claude_confidence:.0%}")
                print(f"      A: {r.question_a[:60]}")
                print(f"      B: {r.question_b[:60]}")
                print(f"      Spread: {r.implied_arb_spread:.2%}")
            total_arb_pairs += len(arb)

    print(f"\n  Combinatorial pairs found: {total_arb_pairs}")


def run_kalshi_dry_run(args: argparse.Namespace) -> int:
    """Dry-run mode for Kalshi: fetch and display without DB writes."""
    from llm_quant.arb.kalshi_client import KalshiClient

    print("DRY RUN — fetching Kalshi mutually exclusive events (no DB write)...")
    client = KalshiClient()
    events = client.fetch_negrisk_events()

    cat_filter = args.category.lower() if args.category else None
    if cat_filter:
        events = [e for e in events if e.category.lower() == cat_filter]

    print(f"\nMutually exclusive events loaded: {len(events)}")
    print(
        f"\n{'Event':<30} {'N':>3} {'SumYES':>8} {'Complement':>10} "
        f"{'Net(3%)':>9} {'Vol24h':>10} Category"
    )
    print("-" * 82)

    opps = []
    for evt in sorted(events, key=lambda e: e.negrisk_complement, reverse=True):
        net = evt.net_spread
        flag = " *** ARB ***" if net > 0 else ""
        print(
            f"{evt.event_ticker:<30} {len(evt.markets):>3} "
            f"{evt.sum_yes_ask:>8.3f} {evt.negrisk_complement:>10.3f} "
            f"{net:>9.3f} {evt.total_volume_24h:>10.0f} "
            f"{evt.category}{flag}"
        )
        print(f"  {evt.title[:70]}")
        if net > 0:
            opps.append(evt)

    print(f"\nTotal ARB opportunities (net > 0): {len(opps)}")
    if opps:
        print("\n--- TOP OPPORTUNITIES ---")
        for evt in opps[:5]:
            print(f"\n  {evt.event_ticker}: {evt.title}")
            print(
                f"  Net spread: {evt.net_spread:.2%} | Conditions: {len(evt.markets)}"
            )
            print("  Conditions:")
            for c in sorted(evt.markets, key=lambda x: -x.yes_ask)[:5]:
                print(
                    f"    {c.ticker}: YES_ask={c.yes_ask:.3f} "
                    f"bid_ask_spread={c.bid_ask_spread:.3f} vol_24h={c.volume_24h:.0f}"
                )
    return 0


def main() -> int:
    args = parse_args()
    category_filter = [args.category] if args.category else None
    source_label = args.source.upper()

    print("\n" + "=" * 70)
    print(f"  PREDICTION MARKET ARB SCANNER — {source_label}")
    print(f"  Mode: {args.mode} | Category: {args.category or 'all'}")
    print(f"  Min spread: {args.min_spread:.0%} | Min volume: ${args.min_volume:,.0f}")
    print("=" * 70 + "\n")

    # ── Kalshi path ──────────────────────────────────────────────────────
    if args.source == "kalshi":
        if args.dry_run:
            return run_kalshi_dry_run(args)

        args.db.parent.mkdir(parents=True, exist_ok=True)
        scanner = ArbScanner(
            db_path=args.db,
            min_spread_pct=args.min_spread,
            min_volume=args.min_volume,
            category_filter=category_filter,
            source="kalshi",
        )
        opps = scanner.run_kalshi_negrisk_scan(
            min_spread=args.min_spread,
            min_volume=args.min_volume,
        )
        print_opportunities(opps)
        summary = scanner.get_scan_summary()
        if summary:
            print(
                f"Scan history: {summary['total_scans']} scans, "
                f"{summary['total_opps_detected']} total opportunities detected"
            )
        return 0

    # ── Polymarket path ──────────────────────────────────────────────────
    ssl_verify = not args.no_ssl_verify
    if args.dry_run:
        print("DRY RUN — fetching Polymarket markets (no DB write)...")
        client = GammaClient(ssl_verify=ssl_verify)
        raw = client.fetch_all_active_markets(max_markets=args.max_markets)
        markets = client.parse_all_markets(raw)

        negrisk = [m for m in markets if m.is_negrisk]
        by_cat: dict[str, int] = {}
        for m in markets:
            by_cat[m.category] = by_cat.get(m.category, 0) + 1

        print(f"\nMarkets fetched: {len(markets)}")
        print(f"NegRisk markets: {len(negrisk)}")
        print("\nBy category:")
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat:20s} {n:5d}")

        candidates = [m for m in negrisk if m.negrisk_complement > args.min_spread]
        if category_filter:
            candidates = [m for m in candidates if m.category in category_filter]
        print(
            f"\nNegRisk candidates (complement > {args.min_spread:.0%}): {len(candidates)}"
        )
        for m in candidates[:10]:
            print(
                f"  {m.market_id[:20]} | complement={m.negrisk_complement:.3f}"
                f" | {m.question[:50]}"
            )
        return 0

    args.db.parent.mkdir(parents=True, exist_ok=True)
    scanner = ArbScanner(
        db_path=args.db,
        min_spread_pct=args.min_spread,
        min_volume=args.min_volume,
        category_filter=category_filter,
        source="polymarket",
    )

    if args.mode == "negrisk":
        opps = scanner.run_negrisk_scan(max_markets=args.max_markets)
    else:
        opps = scanner.run_scan(max_markets=args.max_markets)

    print_opportunities(opps)

    summary = scanner.get_scan_summary()
    if summary:
        print(
            f"Scan history: {summary['total_scans']} scans, "
            f"{summary['total_opps_detected']} total opportunities detected"
        )

    if args.detect_combinatorial:
        client = GammaClient(ssl_verify=ssl_verify)
        run_combinatorial_detection(scanner, client, args.db)

    return 0


if __name__ == "__main__":
    sys.exit(main())

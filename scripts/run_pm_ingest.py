#!/usr/bin/env python3
"""Polymarket data ingestion CLI.

Fetches market metadata, price history, and orderbook snapshots from
Polymarket APIs and stores them as Parquet files.

Usage:
  python scripts/run_pm_ingest.py markets              # Fetch market metadata
  python scripts/run_pm_ingest.py prices               # Fetch price history
  python scripts/run_pm_ingest.py orderbooks           # Snapshot orderbooks
  python scripts/run_pm_ingest.py all                  # Run all three

Options:
  --limit N       Limit number of markets processed (for testing)
  --dry-run       Show what would be fetched without writing
  --no-ssl-verify Disable SSL verification (useful on Windows)

Examples:
  # Quick test: fetch 5 markets and their prices
  python scripts/run_pm_ingest.py all --limit 5

  # Dry run to see what would be fetched
  python scripts/run_pm_ingest.py markets --dry-run

  # Full ingestion
  python scripts/run_pm_ingest.py all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from llm_quant.arb.clob_client import ClobClient
from llm_quant.arb.gamma_client import GammaClient
from llm_quant.arb.pm_data import (
    get_data_summary,
    ingest_all_orderbooks,
    ingest_all_prices,
    ingest_markets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pm_ingest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Polymarket data ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "command",
        choices=["markets", "prices", "orderbooks", "all"],
        help="What to ingest: markets, prices, orderbooks, or all",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of markets processed (for testing)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but do not write to disk",
    )
    p.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (useful on Windows)",
    )
    p.add_argument(
        "--max-markets",
        type=int,
        default=5000,
        help="Maximum markets to fetch from Gamma API (default: 5000)",
    )
    return p.parse_args()


def cmd_markets(
    gamma: GammaClient,
    max_markets: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Fetch and store market metadata."""
    effective_max = limit if limit is not None else max_markets
    print(f"\n  Fetching market metadata (max={effective_max}, dry_run={dry_run})...")
    df = ingest_markets(gamma, max_markets=effective_max, dry_run=dry_run)
    print(f"  Markets ingested: {len(df)}")

    if not df.is_empty():
        # Show category breakdown
        by_cat = df.group_by("category").len().sort("len", descending=True)
        print("\n  By category:")
        for row in by_cat.iter_rows(named=True):
            print(f"    {row['category'] or 'unknown':20s} {row['len']:5d}")

        neg_risk_count = df.filter(pl.col("is_neg_risk")).height
        print(f"\n  NegRisk markets: {neg_risk_count}")


def cmd_prices(
    gamma: GammaClient,
    clob: ClobClient,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Fetch price history for all tokens."""
    print(f"\n  Fetching price history (limit={limit}, dry_run={dry_run})...")
    summary = ingest_all_prices(gamma, clob, limit=limit, dry_run=dry_run)
    print(f"  Markets processed: {summary['markets_processed']}")
    print(f"  Tokens fetched:    {summary['tokens_fetched']}")
    print(f"  Errors:            {summary['errors']}")


def cmd_orderbooks(
    gamma: GammaClient,
    clob: ClobClient,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Snapshot current orderbooks."""
    print(f"\n  Snapshotting orderbooks (limit={limit}, dry_run={dry_run})...")
    summary = ingest_all_orderbooks(gamma, clob, limit=limit, dry_run=dry_run)
    print(f"  Markets processed: {summary['markets_processed']}")
    print(f"  Snapshots taken:   {summary['snapshots_taken']}")
    print(f"  Errors:            {summary['errors']}")


def print_summary() -> None:
    """Print current data summary."""
    summary = get_data_summary()
    print("\n" + "=" * 50)
    print("  DATA SUMMARY")
    print("=" * 50)
    print(f"  Markets:            {summary['markets_count']}")
    print(f"  Price tokens:       {summary['price_tokens_count']}")
    print(f"  Orderbook files:    {summary['orderbook_snapshots_count']}")
    if summary["markets_last_modified"]:
        print(f"  Markets updated:    {summary['markets_last_modified']}")
    if summary["oldest_price_ts"]:
        print(f"  Price range:        {summary['oldest_price_ts']}")
        print(f"                   to {summary['newest_price_ts']}")
    print("=" * 50 + "\n")


def main() -> int:
    args = parse_args()
    ssl_verify = not args.no_ssl_verify

    print("\n" + "=" * 60)
    print("  POLYMARKET DATA INGESTION")
    print(f"  Command: {args.command}")
    print(f"  Limit: {args.limit or 'none'} | Dry run: {args.dry_run}")
    print("=" * 60)

    gamma = GammaClient(ssl_verify=ssl_verify)
    clob = ClobClient(ssl_verify=ssl_verify)

    t0 = time.time()

    try:
        if args.command in ("markets", "all"):
            cmd_markets(gamma, args.max_markets, args.limit, args.dry_run)

        if args.command in ("prices", "all"):
            cmd_prices(gamma, clob, args.limit, args.dry_run)

        if args.command in ("orderbooks", "all"):
            cmd_orderbooks(gamma, clob, args.limit, args.dry_run)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        return 1
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        return 1

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    # Always show summary at the end
    print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())

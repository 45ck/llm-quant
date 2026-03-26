#!/usr/bin/env python3
"""Crypto perpetual funding rate scanner CLI.

Fetches and analyzes funding rates from Binance, OKX, and Bybit.
Public endpoints only — no API keys required.

Usage:
  # Scan current funding rates and show opportunities
  python scripts/run_funding_scanner.py --scan

  # Fetch last 30 days of funding rate history
  python scripts/run_funding_scanner.py --fetch-history

  # Fetch history for specific exchanges and symbols
  python scripts/run_funding_scanner.py --fetch-history --exchange binance,okx --symbol BTC,ETH

  # Scan with custom threshold (0.005% per 8h = ~5.5% annualized)
  python scripts/run_funding_scanner.py --scan --threshold 0.00005

  # Dry-run: fetch rates but don't write to DB
  python scripts/run_funding_scanner.py --scan --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_quant.arb.funding_rates import (
    DEFAULT_EXCHANGES,
    DEFAULT_SYMBOLS,
    FundingCollector,
    get_funding_connection,
    persist_records,
)
from llm_quant.arb.funding_scanner import (
    FundingScanner,
    format_scan_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("funding_scanner")

DEFAULT_DB = Path("data/quant.db")


def _parse_list(value: str, canonical_map: dict[str, str]) -> list[str]:
    """Parse a comma-separated list, mapping short names to canonical CCXT forms."""
    items = [s.strip().upper() for s in value.split(",")]
    result = []
    for item in items:
        mapped = canonical_map.get(item, item)
        result.append(mapped)
    return result


# Short name -> CCXT symbol format
SYMBOL_MAP = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
}

# Short name -> CCXT exchange id
EXCHANGE_MAP = {
    "BINANCE": "binance",
    "OKX": "okx",
    "BYBIT": "bybit",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crypto perpetual funding rate scanner")
    p.add_argument(
        "--scan",
        action="store_true",
        help="Scan current funding rates and show opportunities",
    )
    p.add_argument(
        "--fetch-history",
        action="store_true",
        help="Fetch historical funding rates (last N days)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history to fetch (default: 30)",
    )
    p.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="Comma-separated exchanges: binance,okx,bybit (default: all)",
    )
    p.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Comma-separated symbols: BTC,ETH,SOL (default: all)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0001,
        help="Min funding rate per 8h for high-rate detection (default: 0.0001 = 0.01%%)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and display rates without writing to DB",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"DuckDB path (default: {DEFAULT_DB})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.scan and not args.fetch_history:
        print("Error: specify --scan and/or --fetch-history")
        return 1

    # Parse exchange and symbol filters
    exchanges = (
        _parse_list(args.exchange, EXCHANGE_MAP)
        if args.exchange
        else list(DEFAULT_EXCHANGES)
    )
    symbols = (
        _parse_list(args.symbol, SYMBOL_MAP) if args.symbol else list(DEFAULT_SYMBOLS)
    )

    display_exchanges = ", ".join(exchanges)
    display_symbols = ", ".join(s.split("/")[0] if "/" in s else s for s in symbols)

    print("\n" + "=" * 70)
    print("  CRYPTO FUNDING RATE SCANNER")
    print(f"  Exchanges: {display_exchanges}")
    print(f"  Symbols:   {display_symbols}")
    print(f"  Threshold: {args.threshold * 100:.4f}% per 8h")
    print(f"  Dry-run:   {args.dry_run}")
    print("=" * 70 + "\n")

    collector = FundingCollector(exchanges=exchanges, symbols=symbols)

    # Fetch history
    if args.fetch_history:
        print(f"Fetching {args.days}-day funding rate history...\n")
        history_records = collector.fetch_history(days=args.days)
        print(f"\nFetched {len(history_records)} historical records.")

        if history_records and not args.dry_run:
            conn = get_funding_connection(args.db)
            count = persist_records(conn, history_records)
            print(f"Persisted {count} records to {args.db}")
            conn.close()
        elif history_records:
            print("(dry-run — skipping DB write)")

    # Scan current rates
    if args.scan:
        print("\nFetching current funding rates...\n")
        current_records = collector.fetch_current_rates()

        if not current_records:
            print("No funding rate data available.")
            return 0

        if not args.dry_run:
            conn = get_funding_connection(args.db)
            persist_records(conn, current_records)
            conn.close()

        scanner = FundingScanner(
            rate_threshold=args.threshold,
            differential_threshold=args.threshold / 2,
        )
        high_rate = scanner.scan_high_rates(current_records)
        cross_exchange = scanner.scan_cross_exchange(current_records)

        report = format_scan_report(high_rate, cross_exchange)
        print(report)

        total = len(high_rate) + len(cross_exchange)
        print(f"\nTotal opportunities: {total}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

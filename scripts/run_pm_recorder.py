#!/usr/bin/env python3
"""CLI script — record a timestamped Polymarket market snapshot.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_pm_recorder.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_pm_recorder.py --db-path data/custom.duckdb
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_pm_recorder.py --no-ssl-verify
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record a timestamped Polymarket market snapshot."
    )
    parser.add_argument(
        "--db-path",
        default="data/pm_research.duckdb",
        help="Path to the DuckDB database file (default: data/pm_research.duckdb)",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",
        default=False,
        help="Disable SSL certificate verification (use if behind a corporate proxy)",
    )
    args = parser.parse_args(argv)

    # -- late import to keep startup fast and testable --
    from llm_quant.arb.gamma_client import GammaClient
    from llm_quant.arb.research.recorder import MarketDataRecorder

    client = GammaClient(ssl_verify=not args.no_ssl_verify)
    recorder = MarketDataRecorder(db_path=args.db_path, client=client)

    logger.info("Recording Polymarket snapshot to %s ...", args.db_path)
    snapshot_id = recorder.record_snapshot()

    # -- retrieve and summarise --
    snap = recorder.get_snapshot(snapshot_id)
    if snap is None:
        logger.error("Snapshot %s not found after recording.", snapshot_id)
        return 1

    header = snap["header"]
    markets = snap["markets"]

    total_markets = len(markets)
    negrisk_count = sum(1 for m in markets if m.get("is_negrisk"))
    arb_count = sum(1 for m in markets if (m.get("spread") or 0.0) < -0.02)

    print()
    print("=" * 60)
    print(f"  Snapshot ID  : {snapshot_id}")
    print(f"  Timestamp    : {header.get('timestamp')}")
    print(f"  Source       : {header.get('source')}")
    print(f"  Data quality : {header.get('data_quality')}")
    print(f"  Scan time    : {header.get('scan_duration_ms')} ms")
    print("-" * 60)
    print(f"  Markets      : {total_markets}")
    print(f"  NegRisk      : {negrisk_count}")
    print(f"  Arb opps     : {arb_count}  (spread < -2 cents)")
    print(f"  24h volume   : ${header.get('total_volume_24h', 0):,.0f}")
    print("=" * 60)
    print(f"  Total snapshots in DB: {recorder.snapshot_count()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

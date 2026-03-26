#!/usr/bin/env python3
"""Kalshi NegRisk paper execution engine CLI.

Scans Kalshi for mutually exclusive events with a positive net spread,
applies pre-trade checks, sizes via Kelly criterion, and records paper
trades to DuckDB.

Usage:
  python scripts/run_arb_execution.py
  python scripts/run_arb_execution.py --min-spread 0.02 --min-volume 500
  python scripts/run_arb_execution.py --dry-run
  python scripts/run_arb_execution.py --db data/custom.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_quant.arb.execution import KalshiArbExecution
from llm_quant.arb.kalshi_client import KalshiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arb_execution")

DEFAULT_DB = Path("data/quant.db")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kalshi NegRisk paper execution engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=["kalshi"],
        default="kalshi",
        help="Data source (only kalshi supported)",
    )
    p.add_argument(
        "--min-spread",
        type=float,
        default=0.0,
        help="Minimum net spread to execute (0 = any positive spread)",
    )
    p.add_argument(
        "--min-volume",
        type=float,
        default=100.0,
        help="Minimum 24h volume per condition in USD",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show decisions without recording to DB",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="DuckDB file path",
    )
    p.add_argument(
        "--max-markets",
        type=int,
        default=50_000,
        help="Maximum markets to fetch from Kalshi",
    )
    return p.parse_args()


def _print_header(args: argparse.Namespace) -> None:
    mode_label = "DRY RUN" if args.dry_run else "PAPER EXECUTION"
    print("\n" + "=" * 70)
    print(f"  KALSHI NEGRISK PAPER EXECUTION ENGINE — {mode_label}")
    print(
        f"  Min spread: {args.min_spread:.1%} | "
        f"Min volume: ${args.min_volume:,.0f} | "
        f"DB: {args.db}"
    )
    print("=" * 70 + "\n")


def _print_decision_table(decisions: list[dict]) -> None:
    if not decisions:
        print("  No opportunities found.\n")
        return

    go_decisions = [d for d in decisions if d["go"]]
    no_go_decisions = [d for d in decisions if not d["go"]]

    print(f"  Evaluated: {len(decisions)} events")
    print(f"  GO:        {len(go_decisions)}")
    print(f"  NO-GO:     {len(no_go_decisions)}\n")

    if go_decisions:
        print(f"{'=' * 70}")
        print("  GO DECISIONS")
        print(f"{'=' * 70}")
        for d in go_decisions:
            print(f"\n  Event:       {d['event_ticker']}")
            print(f"  Title:       {d['title'][:60]}")
            print(f"  Net spread:  {d['net_spread']:.4f} ({d['net_spread']:.2%})")
            print(
                f"  Kelly f*:    {d['kelly_fraction']:.4f} ({d['kelly_fraction']:.2%})"
            )
            print(f"  Position:    ${d['position_usd']:,.2f}")
            print(f"  Exp. PnL:    ${d['expected_pnl']:,.2f}")
            print(f"  Conditions:  {d['n_conditions']}")
            print(f"  Min Vol/cond: ${d['min_vol']:,.0f}")
            if "exec_id" in d:
                print(f"  Recorded:    {d['exec_id']}")

    if no_go_decisions:
        print(f"\n{'=' * 70}")
        print("  NO-GO DECISIONS (summary)")
        print(f"{'=' * 70}")
        for d in no_go_decisions:
            print(f"  {d['event_ticker']:<35} {d['reason'][:55]}")

    print()


def main() -> int:
    args = parse_args()
    _print_header(args)

    # Fetch Kalshi events
    print("Fetching Kalshi mutually exclusive events...")
    client = KalshiClient()
    try:
        events = client.fetch_negrisk_events(max_markets=args.max_markets)
    except Exception as exc:
        print(f"ERROR: Failed to fetch Kalshi events: {exc}")
        return 1

    print(f"Loaded {len(events)} mutually exclusive events.\n")

    if not events:
        print("No events to process.")
        return 0

    # Override MIN_CONDITION_VOLUME if user supplied --min-volume
    engine: KalshiArbExecution | None = None
    if not args.dry_run:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        engine = KalshiArbExecution(db_path=args.db)
        engine.MIN_CONDITION_VOLUME = args.min_volume

    decisions: list[dict] = []

    for evt in events:
        # Apply min_volume override for dry-run (replicate engine check inline)
        min_cond_vol = evt.min_condition_volume

        if args.dry_run:
            # Evaluate manually without persisting
            net_spread = evt.net_spread
            n = len(evt.markets)

            go = (
                evt.mutually_exclusive
                and net_spread > args.min_spread
                and min_cond_vol >= args.min_volume
                and n >= 2
                and all(0.0 < c.yes_ask < 1.0 for c in evt.markets)
            )

            if go:
                kelly_raw = net_spread / (1.0 + net_spread)
                kelly = min(kelly_raw, KalshiArbExecution.MAX_KELLY_FRACTION)
                position_usd = kelly * KalshiArbExecution.NAV_USD
                expected_pnl = position_usd * net_spread
            else:
                kelly = position_usd = expected_pnl = 0.0

            reason = (
                "go"
                if go
                else (
                    "net_spread <= min_spread"
                    if net_spread <= args.min_spread
                    else f"min_vol={min_cond_vol:.0f} < {args.min_volume:.0f}"
                    if min_cond_vol < args.min_volume
                    else "other check failed"
                )
            )

            decisions.append(
                {
                    "event_ticker": evt.event_ticker,
                    "title": evt.title,
                    "net_spread": net_spread,
                    "n_conditions": n,
                    "min_vol": min_cond_vol,
                    "kelly_fraction": kelly,
                    "position_usd": position_usd,
                    "expected_pnl": expected_pnl,
                    "go": go,
                    "reason": reason,
                }
            )

        else:
            assert engine is not None
            decision = engine.evaluate(evt)

            entry: dict = {
                "event_ticker": evt.event_ticker,
                "title": evt.title,
                "net_spread": evt.net_spread,
                "n_conditions": len(evt.markets),
                "min_vol": min_cond_vol,
                "kelly_fraction": decision.kelly_fraction,
                "position_usd": decision.position_usd,
                "expected_pnl": decision.expected_pnl,
                "go": decision.go,
                "reason": decision.reason,
            }

            if decision.go and evt.net_spread > args.min_spread:
                try:
                    record = engine.execute_paper(evt, decision)
                    entry["exec_id"] = record.exec_id
                    logger.info(
                        "Executed: %s | pos=$%.2f | exp_pnl=$%.2f",
                        evt.event_ticker,
                        record.position_usd,
                        record.expected_pnl,
                    )
                except Exception as exc:
                    logger.warning("Failed to execute %s: %s", evt.event_ticker, exc)
                    entry["go"] = False
                    entry["reason"] = f"execution error: {exc}"

            decisions.append(entry)

    _print_decision_table(decisions)

    # PnL summary (live mode only)
    if not args.dry_run and engine is not None:
        summary = engine.get_pnl_summary()
        if summary:
            print("Portfolio summary:")
            print(f"  Total trades:      {summary['total_trades']}")
            print(f"  Open trades:       {summary['open_trades']}")
            print(f"  Resolved trades:   {summary['resolved_trades']}")
            print(f"  Win rate:          {summary['win_rate']:.1%}")
            print(f"  Total PnL:         ${summary['total_pnl']:,.2f}")
            print(f"  Avg net spread:    {summary['avg_net_spread']:.2%}")
            print(f"  Total position:    ${summary['total_position_usd']:,.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

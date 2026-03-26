#!/usr/bin/env python3
"""Test script: Kalshi combinatorial dependency detector.

Fetches live Kalshi events, groups them by topic keyword, and runs
KalshiCombinatorialDetector on each group.  Caps at 20 Claude API calls
for cost control.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/test_kalshi_combinatorial.py
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from itertools import combinations as _comb
from pathlib import Path

# Add project root to path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_quant.arb.kalshi_client import KalshiClient, KalshiEvent
from llm_quant.arb.kalshi_detector import KalshiCombinatorialDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kalshi_combinatorial_test")

# ── Config ──────────────────────────────────────────────────────────────────

TARGET_CATEGORIES = {"Sports", "Politics", "Elections"}
MAX_EVENTS_TO_FETCH = 50  # per-event market calls so keep small
MAX_CLAUDE_CALLS = 20  # hard cap — each call = ~$0.001 with Haiku

# Topic keywords to look for in event titles (case-insensitive)
TOPIC_KEYWORDS: list[str] = [
    "NBA",
    "NFL",
    "MLB",
    "NHL",
    "NCAA",
    "Stanley Cup",
    "Super Bowl",
    "World Series",
    "championship",
    "playoff",
    "election",
    "Senate",
    "House",
    "president",
    "Pope",
    "Prime Minister",
    "Speaker",
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def fetch_events(client: KalshiClient, target_n: int) -> list[KalshiEvent]:
    """Fetch up to target_n open events in target categories (per-event).

    Uses one API call per event to reliably retrieve markets regardless of
    how the bulk markets endpoint paginates.
    """
    logger.info("Fetching all Kalshi events...")
    raw_events = client.fetch_all_open_events()

    filtered = [
        e
        for e in raw_events
        if e.get("category", "other") in TARGET_CATEGORIES and e.get("event_ticker")
    ]
    logger.info(
        "Events in categories %s: %d / %d total",
        TARGET_CATEGORIES,
        len(filtered),
        len(raw_events),
    )

    filtered = filtered[:target_n]
    logger.info("Fetching markets for %d events (per-event)...", len(filtered))

    events: list[KalshiEvent] = []
    for raw_evt in filtered:
        evt_ticker = raw_evt["event_ticker"]
        try:
            raw_markets = client.fetch_markets_for_event(evt_ticker)
            time.sleep(0.25)  # respect rate limit
        except Exception:
            logger.debug("Failed to fetch markets for %s", evt_ticker)
            continue
        if not raw_markets:
            continue
        evt = client.parse_event(raw_evt, raw_markets)
        if evt.markets:
            events.append(evt)

    logger.info(
        "Loaded %d events with markets in categories %s",
        len(events),
        TARGET_CATEGORIES,
    )
    return events


def group_events_by_keyword(
    events: list[KalshiEvent],
    keywords: list[str],
) -> dict[str, list[KalshiEvent]]:
    """Group events by the first matching keyword in their title."""
    groups: dict[str, list[KalshiEvent]] = defaultdict(list)
    for evt in events:
        title_lower = evt.title.lower()
        for kw in keywords:
            if kw.lower() in title_lower:
                groups[kw].append(evt)
                break  # assign to first matching keyword only

    return {kw: evts for kw, evts in groups.items() if len(evts) >= 2}


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result_table(results: list) -> None:
    if not results:
        print("  (no dependencies above confidence threshold)")
        return

    for r in results:
        arb_flag = " *** ARB ***" if r.is_arb else ""
        print(
            f"\n  [{r.dependency_type.upper()}] conf={r.claude_confidence:.0%}{arb_flag}"
        )
        print(f"    A: {r.condition_a.title[:65]}")
        print(f"       Event: {r.event_a.title[:55]}")
        print(f"       YES ask: {r.price_a:.3f}")
        print(f"    B: {r.condition_b.title[:65]}")
        print(f"       Event: {r.event_b.title[:55]}")
        print(f"       YES ask: {r.price_b:.3f}")
        print(f"    Direction:  {r.expected_direction}")
        print(f"    Constraint: {r.price_constraint}")
        print(f"    Arb spread: {r.implied_arb_spread:.3f}")
        print(f"    Reasoning:  {r.reasoning[:120]}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print_banner("KALSHI COMBINATORIAL DEPENDENCY DETECTOR")
    print(f"  Target categories: {TARGET_CATEGORIES}")
    print(f"  Max events to fetch: {MAX_EVENTS_TO_FETCH}")
    print(f"  Max Claude API calls: {MAX_CLAUDE_CALLS}")
    print(f"  Topic keywords: {TOPIC_KEYWORDS[:8]}...")

    client = KalshiClient()
    events = fetch_events(client, MAX_EVENTS_TO_FETCH)

    if not events:
        print("\n  No events fetched. Check API connectivity.")
        return 1

    print(f"\n  Events fetched and loaded: {len(events)}")
    total_conditions = sum(len(e.markets) for e in events)
    print(f"  Total conditions across events: {total_conditions}")

    # Group by keyword
    groups = group_events_by_keyword(events, TOPIC_KEYWORDS)

    print(f"\n  Topic groups with 2+ events: {len(groups)}")
    for kw, grp in sorted(groups.items(), key=lambda x: -len(x[1])):
        cond_count = sum(len(e.markets) for e in grp)
        print(f"    [{kw}] {len(grp)} events, {cond_count} conditions")

    if not groups:
        # Show ungrouped events so we can diagnose keyword mismatches
        print("\n  No multi-event groups found. Sample event titles:")
        for evt in events[:20]:
            print(f"    [{evt.category}] {evt.title[:70]}")
        return 0

    detector = KalshiCombinatorialDetector(
        db_path=None,
        model="claude-haiku-4-5-20251001",
        min_confidence=0.75,
        min_arb_spread=0.03,
    )

    claude_calls_used = 0
    all_dependencies: list = []
    all_arb_pairs: list = []
    pairs_analyzed_total = 0

    print_banner("RUNNING COMBINATORIAL ANALYSIS")

    for keyword, group_events in sorted(groups.items(), key=lambda x: -len(x[1])):
        if claude_calls_used >= MAX_CLAUDE_CALLS:
            print(f"\n  Claude call cap reached ({MAX_CLAUDE_CALLS}). Stopping.")
            break

        remaining_calls = MAX_CLAUDE_CALLS - claude_calls_used
        max_pairs_this_group = min(remaining_calls, 5)

        print(f"\n  Keyword: '{keyword}' ({len(group_events)} events)")
        for evt in group_events:
            print(
                f"    - {evt.event_ticker}: {evt.title[:55]}"
                f" ({len(evt.markets)} conditions)"
            )

        flat = [(c, e) for e in group_events for c in e.markets]
        n_possible = len(list(_comb(range(len(flat)), 2)))
        n_analyze = min(n_possible, max_pairs_this_group)
        print(f"  Analyzing {n_analyze} of {n_possible} possible pairs...")

        results = detector.analyze_event_group(
            group_events,
            max_pairs=max_pairs_this_group,
        )

        claude_calls_used += n_analyze
        pairs_analyzed_total += n_analyze

        deps = [r for r in results if r.dependency_type != "none"]
        arbs = [r for r in results if r.is_arb]
        all_dependencies.extend(deps)
        all_arb_pairs.extend(arbs)

        if results:
            print_result_table(results)
        else:
            print("  (no results above confidence threshold)")

    print_banner("SUMMARY")
    print(f"  Events fetched:         {len(events)}")
    print(f"  Topic groups:           {len(groups)}")
    print(f"  Pairs analyzed:         {pairs_analyzed_total}")
    print(f"  Claude API calls used:  {claude_calls_used}")
    print(f"  Dependencies found:     {len(all_dependencies)}")
    print(f"  Arb opportunities:      {len(all_arb_pairs)}")

    if all_arb_pairs:
        print_banner("ARB OPPORTUNITIES")
        for r in sorted(all_arb_pairs, key=lambda x: -x.implied_arb_spread):
            print(
                f"\n  [{r.dependency_type.upper()}]"
                f" spread={r.implied_arb_spread:.3f}"
                f" conf={r.claude_confidence:.0%}"
            )
            print(f"    BUY:  {r.condition_b.title[:60]} YES @ {r.price_b:.3f}")
            print(f"    SELL: {r.condition_a.title[:60]} YES @ {r.price_a:.3f}")
            print(f"    Constraint: {r.price_constraint}")
            print(f"    Reasoning: {r.reasoning[:100]}")

    if all_dependencies and not all_arb_pairs:
        print("\n  Dependencies found but no current price violations.")
        print("  These are candidates to monitor for future arb.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

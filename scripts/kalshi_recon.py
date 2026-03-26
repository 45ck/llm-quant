# E:/llm-quant/scripts/kalshi_recon.py
import sys
import time

sys.path.insert(0, "src")
from collections import Counter

from llm_quant.arb.kalshi_client import KalshiClient

client = KalshiClient()

print("Fetching all open ME events via bulk sweep (no per-event calls)...")
t0 = time.time()
results = client.fetch_negrisk_events()
elapsed = time.time() - t0
print(
    f"Bulk fetch complete in {elapsed:.1f}s — {len(results)} valid ME events (>=2 conditions)"
)

# Sort by negrisk_complement descending
results.sort(key=lambda e: e.negrisk_complement, reverse=True)

arb_candidates = [e for e in results if e.net_spread > 0]
liquid_arb = [e for e in arb_candidates if e.min_condition_volume > 50]

print(f"\n{'=' * 80}")
print("KALSHI NEGRISK RECONNAISSANCE — full universe scan")
print(f"{'=' * 80}")
print(f"  Valid ME events (>=2 conditions): {len(results)}")
print(
    f"  Events with negrisk complement > 0: {sum(1 for e in results if e.negrisk_complement > 0)}"
)
print(f"  Net ARB opportunities (net > 0 after 3% fee): {len(arb_candidates)}")
print(f"  LIQUID arb (min_vol > $50): {len(liquid_arb)}")

print("\n--- TOP 30 BY COMPLEMENT (net spread = complement - 3% fee) ---")
print(
    f"{'Event':<32} {'N':>3} {'SumYES':>8} {'Comp':>7} {'Net':>7} {'MinVol':>10}  Category"
)
print("-" * 90)

for evt in results[:30]:
    flag = (
        " *** LIQUID ARB ***"
        if evt.net_spread > 0 and evt.min_condition_volume > 50
        else (" ** ARB **" if evt.net_spread > 0 else "")
    )
    print(
        f"{evt.event_ticker:<32} {len(evt.markets):>3} {evt.sum_yes_ask:>8.3f} "
        f"{evt.negrisk_complement:>7.3f} {evt.net_spread:>7.3f} "
        f"{evt.min_condition_volume:>10.0f}  {evt.category}{flag}"
    )
    print(f"  {evt.title[:75]}")

if liquid_arb:
    print(f"\n{'=' * 80}")
    print("LIQUID ARB OPPORTUNITIES (min_vol > $50, net > 0):")
    print(f"{'=' * 80}")
    for evt in liquid_arb[:10]:
        print(f"\n  {evt.event_ticker}: {evt.title}")
        print(
            f"  Net spread: {evt.net_spread:.2%} | Conditions: {len(evt.markets)} | Min vol: ${evt.min_condition_volume:,.0f}"
        )
        for c in sorted(evt.markets, key=lambda x: -x.yes_ask)[:5]:
            print(
                f"    {c.ticker}: YES_ask={c.yes_ask:.3f} YES_bid={c.yes_bid:.3f} "
                f"bid_ask_spread={c.bid_ask_spread:.3f} vol_24h=${c.volume_24h:,.0f}"
            )
else:
    print("\n  No liquid arb found (min_vol > $50).")
    print("  Showing best net-spread candidates regardless of volume:")
    for evt in arb_candidates[:5]:
        print(f"\n  {evt.event_ticker}: {evt.title}")
        print(
            f"  Net spread: {evt.net_spread:.2%} | Conditions: {len(evt.markets)} | Min vol: ${evt.min_condition_volume:,.0f}"
        )
        for c in sorted(evt.markets, key=lambda x: -x.yes_ask)[:5]:
            print(
                f"    {c.ticker}: YES_ask={c.yes_ask:.3f} YES_bid={c.yes_bid:.3f} "
                f"bid_ask_spread={c.bid_ask_spread:.3f} vol_24h=${c.volume_24h:,.0f}"
            )

# Category breakdown
print("\n--- CATEGORY BREAKDOWN ---")
cat_counts = Counter(e.category for e in results)
for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
    arb_n = sum(1 for e in results if e.category == cat and e.net_spread > 0)
    liq_n = sum(
        1
        for e in results
        if e.category == cat and e.net_spread > 0 and e.min_condition_volume > 50
    )
    print(f"  {cat:<25} {n:>4} events | {arb_n:>3} net-arb | {liq_n:>2} liquid")

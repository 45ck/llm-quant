"""Temporary recon script for Polymarket API analysis."""

import json

with open("data/pm_events_p1.json", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
negrisk = [e for e in data if e.get("negRisk")]
open_negrisk = [e for e in negrisk if not e.get("closed")]

print(f"Events on page 1: {total}")
print(f"NegRisk events (total): {len(negrisk)}")
print(f"Open NegRisk events: {len(open_negrisk)}")
print()

for e in open_negrisk[:20]:
    n_markets = len(e.get("markets", []))
    slug = e.get("slug", "?")[:60]
    vol = e.get("volume24hr", 0)
    print(f"  {slug:<60s} | mkts={n_markets:>3d} | vol24h=${vol:>12,.0f}")

# Also show some with prices
print("\n--- NegRisk price analysis (first 5 open events) ---")
for e in open_negrisk[:5]:
    slug = e.get("slug", "?")
    markets = e.get("markets", [])
    print(f"\nEvent: {slug}")
    print(f"  Markets: {len(markets)}")
    total_yes = 0.0
    for m in markets:
        q = m.get("question", "?")[:50]
        prices_raw = m.get("outcomePrices", "")
        outcomes_raw = m.get("outcomes", "")
        try:
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw
        except Exception:
            prices = []
            outcomes = []

        yes_price = 0.0
        for i, o in enumerate(outcomes):
            if str(o).lower() == "yes" and i < len(prices):
                yes_price = float(prices[i])

        total_yes += yes_price
        print(f"    {q:<50s} YES={yes_price:.3f}")

    complement = 1.0 - total_yes
    print(f"  Sum(YES)={total_yes:.3f}  Complement={complement:.3f}")
    if complement > 0:
        print(
            f"  >>> POTENTIAL ARB: buy all YES for ${total_yes:.3f}, guaranteed $1.00 payout"
        )

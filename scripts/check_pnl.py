"""Quick portfolio P&L check."""

import duckdb

conn = duckdb.connect("data/llm_quant.duckdb", read_only=True)

# Latest snapshot
snap = conn.execute(
    "SELECT date, nav, cash, daily_pnl "
    "FROM portfolio_snapshots ORDER BY snapshot_id DESC LIMIT 1"
).fetchone()

if not snap:
    print("No snapshots yet. Run 'pq run' first.")
    raise SystemExit(1)

nav = float(snap[1])
cash = float(snap[2])
pnl = nav - 100_000.0

print(f"Snapshot date: {snap[0]}")
print(f"NAV:  ${nav:,.2f}")
print(f"Cash: ${cash:,.2f}")
print(f"P&L:  ${pnl:+,.2f} ({pnl / 1000:.2f}%)")
print()

# Positions
positions = conn.execute(
    "SELECT p.symbol, p.shares, p.avg_cost, p.current_price, "
    "p.unrealized_pnl, p.weight "
    "FROM positions p "
    "WHERE p.snapshot_id = (SELECT MAX(snapshot_id) FROM portfolio_snapshots) "
    "ORDER BY p.unrealized_pnl DESC"
).fetchall()

print(f"Positions ({len(positions)}):")
header = f"{'Symbol':>6}  {'Shares':>6}  {'AvgCost':>9}"
header += f"  {'Current':>9}  {'P&L':>9}  {'Wt':>5}"
print(header)
print("-" * 55)

for r in positions:
    sym, shares, avg_cost, cur_price, upnl, wt = r
    print(
        f"{sym:>6}  {shares:>6.0f}  ${avg_cost:>8.2f}  ${cur_price:>8.2f}  "
        f"${float(upnl):>+8.2f}  {float(wt) * 100:>4.1f}%"
    )

# Check if prices have changed since we bought
print()
print("--- Latest market prices vs position prices ---")
stale = True
for r in positions:
    sym = r[0]
    pos_price = float(r[3])
    latest = conn.execute(
        "SELECT close, date FROM market_data_daily "
        "WHERE symbol = ? ORDER BY date DESC LIMIT 1",
        [sym],
    ).fetchone()
    if latest:
        mkt_price = float(latest[0])
        diff = mkt_price - pos_price
        if abs(diff) > 0.001:
            stale = False
            msg = f"  {sym:>6}: position ${pos_price:.2f}"
            msg += f" -> market ${mkt_price:.2f} (${diff:+.2f})"
            print(msg)
        else:
            print(f"  {sym:>6}: ${pos_price:.2f} (unchanged)")

if stale:
    print()
    print("ALL PRICES ARE UNCHANGED since purchase.")
    print("We bought everything on the same day and haven't fetched new prices since.")
    print("To see actual P&L, run: pq fetch  (to get today's prices)")
    print("Then run:                pq run    (to update portfolio and trade)")

conn.close()

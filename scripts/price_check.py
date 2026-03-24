"""Check latest market prices vs position cost basis."""

import duckdb

conn = duckdb.connect("data/llm_quant.duckdb", read_only=True)

symbols = ["SPY", "QQQ", "TLT", "GLD", "XLE", "XLV", "XLP", "SHY", "IEF", "EEM", "XLRE"]

print("Latest prices in market_data_daily:")
print(f"{'Symbol':>6}  {'Latest Date':>12}  {'Close':>9}  {'Our Cost':>9}  {'Diff':>8}")
print("-" * 55)

for sym in symbols:
    row = conn.execute(
        "SELECT date, close FROM market_data_daily "
        "WHERE symbol = ? ORDER BY date DESC LIMIT 1",
        [sym],
    ).fetchone()

    cost = conn.execute(
        "SELECT avg_cost FROM positions "
        "WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM portfolio_snapshots) "
        "AND symbol = ?",
        [sym],
    ).fetchone()

    if row and cost:
        diff = float(row[1]) - float(cost[0])
        dt = f"{row[0]!s:>12}"
        line = f"{sym:>6}  {dt}  ${float(row[1]):>8.2f}"
        line += f"  ${float(cost[0]):>8.2f}  ${diff:>+7.2f}"
        print(line)

# Also check what date range we have
minmax = conn.execute(
    "SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM market_data_daily"
).fetchone()
print(f"\nDB date range: {minmax[0]} to {minmax[1]} ({minmax[2]} trading days)")

conn.close()

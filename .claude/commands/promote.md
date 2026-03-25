---
description: "Run the strategy promotion checklist — verify all gates before deploying a strategy change"
---

# /promote — Strategy Promotion Checklist

You are the portfolio manager running a strategy change review. Walk through every gate in the Model Promotion Policy before approving any strategy modification.

**Reference docs:**
- `docs/governance/model-promotion-policy.md` — Full promotion pipeline
- `docs/governance/control-matrix.md` — Failure modes and kill switches

## Parse the user's argument: "$ARGUMENTS"

### No arguments → Run the full promotion checklist

Execute these steps in order. Do not skip any step. Report results as a markdown checklist.

---

### Step 1: Check strategy_changelog for recent entries

Look for a `strategy_changelog` table in DuckDB or a changelog file in the project. Display the most recent entries to establish context for what is being promoted.

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import duckdb
db = duckdb.connect('data/llm_quant.duckdb', read_only=True)
tables = [r[0] for r in db.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema='main'\").fetchall()]
if 'strategy_changelog' in tables:
    rows = db.execute('SELECT * FROM strategy_changelog ORDER BY timestamp DESC LIMIT 10').fetchall()
    for r in rows:
        print(r)
else:
    print('strategy_changelog table not found — no change history available')
db.close()
"
```

If no changelog exists, note this as a gap: "No strategy_changelog table found. Changes cannot be audited."

---

### Step 2: Run surveillance scan

Execute the surveillance scanner to check current portfolio and system health. This establishes baseline conditions before any promotion decision.

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/run_surveillance.py
```

If the surveillance script does not exist yet, run the manual health checks:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
from llm_quant.db.schema import init_schema
import duckdb

db = duckdb.connect('data/llm_quant.duckdb', read_only=True)

# Check hash chain integrity
print('=== Hash Chain Integrity ===')
try:
    snapshots = db.execute('SELECT COUNT(*) FROM portfolio_snapshots').fetchone()
    print(f'Portfolio snapshots: {snapshots[0]}')
except Exception as e:
    print(f'ERROR: {e}')

# Check data freshness
print('\n=== Data Freshness ===')
try:
    latest = db.execute('SELECT symbol, MAX(date) as latest FROM market_data GROUP BY symbol ORDER BY latest ASC LIMIT 5').fetchall()
    for row in latest:
        print(f'  {row[0]}: {row[1]}')
except Exception as e:
    print(f'ERROR: {e}')

# Check portfolio state
print('\n=== Portfolio State ===')
try:
    snap = db.execute('SELECT date, nav, cash, gross_exposure, net_exposure FROM portfolio_snapshots ORDER BY date DESC LIMIT 1').fetchone()
    if snap:
        print(f'  Date: {snap[0]}, NAV: \${snap[1]:,.2f}, Cash: \${snap[2]:,.2f}')
        print(f'  Gross exposure: {snap[3]:.1%}, Net exposure: {snap[4]:.1%}')
    else:
        print('  No snapshots found')
except Exception as e:
    print(f'ERROR: {e}')

db.close()
"
```

Report any warnings or issues from the scan.

---

### Step 3: Display the promotion checklist

Present the full checklist. For each item, indicate whether it can be verified now, requires manual input, or is not yet measurable.

```markdown
## Strategy Promotion Checklist

### Stage 1: Hard Vetoes (any failure = automatic rejection)
- [ ] Deflated Sharpe Ratio (DSR) >= 0.95
- [ ] Probability of Backtest Overfitting (PBO) <= 10%
- [ ] Stepwise SPA p-value <= 0.05
- [ ] Minimum Trail Length (MinTRL) >= 1 out-of-sample period

### Stage 2: Weighted Scorecard (composite >= 85 required)
- [ ] Risk-Adjusted Returns scored (25% weight)
- [ ] Drawdown Characteristics scored (20% weight)
- [ ] Trade Statistics scored (20% weight)
- [ ] Robustness scored (20% weight)
- [ ] Operational scored (15% weight)
- [ ] Composite score >= 85

### Stage 3: Paper Trading Minimums
- [ ] >= 50 trades executed
- [ ] >= 30 calendar days of paper trading
- [ ] Paper Sharpe >= 0.60
- [ ] All operational systems tested (data, risk, execution, reporting)

### Stage 4: Canary Gate
- [ ] 10% portfolio allocation assigned to canary
- [ ] >= 14 calendar days in canary
- [ ] Canary drawdown < 10%
- [ ] No kill switch triggers during canary
- [ ] No adverse interaction with remaining portfolio

### Stage 5: Deployment Readiness
- [ ] All 6 kill switches configured and active
- [ ] All failure mode detectors operational
- [ ] Changes documented in strategy_changelog
- [ ] Baseline metrics recorded for regime change detection
- [ ] Enhanced surveillance plan in place (30-day daily review)
```

---

### Step 4: Verify what can be checked automatically

For items that can be verified from current data, run the checks:

**Paper trading minimums:**
```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import duckdb
from datetime import date, timedelta

db = duckdb.connect('data/llm_quant.duckdb', read_only=True)

# Trade count
try:
    trades = db.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
    status = 'PASS' if trades >= 50 else 'FAIL'
    print(f'Trades executed: {trades} (need >= 50) — {status}')
except:
    print('Trades: Unable to query')

# Calendar days
try:
    dates = db.execute('SELECT MIN(date), MAX(date) FROM portfolio_snapshots').fetchone()
    if dates[0] and dates[1]:
        days = (dates[1] - dates[0]).days
        status = 'PASS' if days >= 30 else 'FAIL'
        print(f'Calendar days: {days} (need >= 30) — {status}')
    else:
        print('Calendar days: No snapshots found')
except:
    print('Calendar days: Unable to query')

# Kill switches
print('\nKill switch configuration:')
print('  1. NAV drawdown > 15%: Enforced by risk/manager.py')
print('  2. Single-day loss > 5%: Check surveillance')
print('  3. 5 consecutive losing trades: Check trade log')
print('  4. Correlation > 85%: Check surveillance')
print('  5. No fresh data > 72h: Check data freshness above')
print('  6. 3 consecutive risk failures: Check execution log')

db.close()
"
```

---

### Step 5: Log the promotion attempt

Record this promotion review in the strategy changelog:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import duckdb
from datetime import datetime

db = duckdb.connect('data/llm_quant.duckdb')

# Create strategy_changelog table if it does not exist
db.execute('''
    CREATE TABLE IF NOT EXISTS strategy_changelog (
        id INTEGER PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        change_type VARCHAR,
        description TEXT,
        outcome VARCHAR,
        details TEXT
    )
''')

# Log the promotion attempt
db.execute('''
    INSERT INTO strategy_changelog (change_type, description, outcome, details)
    VALUES (
        'promotion_review',
        'Strategy promotion checklist executed',
        'pending',
        'Review initiated on ' || CAST(CURRENT_TIMESTAMP AS VARCHAR)
    )
''')

print('Promotion review logged to strategy_changelog.')
db.close()
"
```

---

### Step 6: Summary and recommendation

After completing all steps, provide a summary:

1. **Items passed**: List all checklist items that are verified
2. **Items failed**: List all checklist items that did not meet thresholds
3. **Items unverifiable**: List items that require manual input or are not yet measurable
4. **Recommendation**: PROMOTE / CONDITIONAL / REJECT based on current evidence
5. **Next steps**: What must be done before the next review

Be direct. If the strategy is not ready, say so and explain why.

---

## Important

- Follow the Model Promotion Policy (`docs/governance/model-promotion-policy.md`) exactly
- Do not approve a promotion if any hard veto fails
- Do not skip stages -- the pipeline is sequential
- All promotion decisions must be logged to `strategy_changelog`
- Reference the Control Matrix (`docs/governance/control-matrix.md`) for kill switch verification

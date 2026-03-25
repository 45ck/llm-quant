---
description: "Manage paper trading validation — track live performance, incidents, drift"
---

# /paper — Paper Trading Validation

You are the portfolio manager. Paper trading is the bridge between backtesting and promotion. It validates that the strategy works end-to-end in a live environment: data fetching, indicator computation, signal generation, risk checks, execution, and reporting. Paper trading catches operational failures that backtests cannot.

Paper trading does NOT validate execution quality (slippage, market impact, liquidity). It assumes perfect fills. This limitation must be understood and documented.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> Show paper trading status for all strategies

Scan for existing paper trading artifacts:

```bash
cd E:/llm-quant && find data/strategies -name "paper-trading.yaml" -type f 2>/dev/null
```

Display a summary:

```
## Paper Trading Status

| Slug | Days Active | Trades | Sharpe | Max DD | Slippage Drift | Status | Gate |
|------|-------------|--------|--------|--------|----------------|--------|------|
| ...  | N           | N      | X.XX   | -X.X%  | X.X bps        | ...    | ...  |
```

---

### Slug provided (e.g., "momentum-rotation") --> View or manage paper trading

**Step 1: Check lifecycle prerequisites**

Verify robustness gate passed:

```bash
cat E:/llm-quant/data/strategies/$SLUG/robustness.yaml 2>/dev/null
```

If no robustness artifact exists or `overall_gate: fail`, STOP:
```
ERROR: Robustness gate has not passed for "{slug}".
Paper trading requires a passing robustness analysis.
Run: /robustness {slug}
```

**Step 2: Check if paper trading record exists**

```bash
cat E:/llm-quant/data/strategies/$SLUG/paper-trading.yaml 2>/dev/null
```

If it exists, display the current status and check gate criteria (Step 5).

**Step 3: Initialize paper trading (if new)**

If no paper trading record exists, create one:

```yaml
# Paper Trading: {slug}
strategy_slug: "{slug}"
spec_hash: "{hash}"
robustness_date: "YYYY-MM-DD"

start_date: "YYYY-MM-DD"
status: "active"  # active | paused | completed | failed

# Performance tracking
performance:
  initial_nav: 100000.00
  current_nav: 100000.00
  peak_nav: 100000.00
  current_sharpe: null
  current_sortino: null
  current_drawdown: 0.0
  total_return_pct: 0.0
  annualized_return_pct: null

# Trade log — updated each session
trades: []
# Each trade entry:
#   date: "YYYY-MM-DD"
#   symbol: "SPY"
#   action: "buy"
#   shares: 10
#   price: 500.00
#   signal: "trend_signal"
#   fill_type: "paper"

# Incident log — operational issues, near-misses, anomalies
incidents: []
# Each incident:
#   date: "YYYY-MM-DD"
#   severity: "critical"  # critical | warning | info
#   description: "What happened"
#   resolution: "How it was resolved"
#   impact: "Impact on P&L or operations"

# Slippage drift tracking
# Compare paper fill prices vs actual market prices at assumed fill time
slippage_drift_bps: 0.0
slippage_samples: 0

# Operational checklist
operations_tested:
  data_fetching: false
  indicator_computation: false
  signal_generation: false
  risk_checks: false
  trade_execution: false
  portfolio_persistence: false
  hash_chain_integrity: false
  performance_reporting: false

days_active: 0
total_trades: 0

created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
```

Write to `data/strategies/{slug}/paper-trading.yaml`.

**Step 4: Update paper trading record**

If the argument includes an action keyword, handle it:

**"{slug} trade" --> Record a trade**

Ask for trade details (or pull from the most recent /trade execution):
```
What trade to record?
- Symbol, action (buy/sell/close), shares, price, signal that triggered it
```

Append to the trades list and update performance metrics.

**"{slug} incident {severity}" --> Record an incident**

Severity must be one of: critical, warning, info.

```
Describe the incident:
- What happened?
- What was the impact?
- How was it resolved?
```

Critical incidents automatically pause paper trading. The user must explicitly resume.

**"{slug} update" --> Refresh performance metrics**

Pull the latest portfolio state from DuckDB and update the paper trading record:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import duckdb

db = duckdb.connect('data/llm_quant.duckdb', read_only=True)

# Get latest snapshot
snap = db.execute('''
    SELECT date, nav, cash, gross_exposure, net_exposure
    FROM portfolio_snapshots ORDER BY date DESC LIMIT 1
''').fetchone()

if snap:
    print(f'Latest snapshot: {snap[0]}')
    print(f'NAV: \${snap[1]:,.2f}')
    print(f'Cash: \${snap[2]:,.2f}')
    print(f'Gross: {snap[3]:.1%}, Net: {snap[4]:.1%}')

# Get trade count
trades = db.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
print(f'Total trades: {trades}')

# Get date range
dates = db.execute('SELECT MIN(date), MAX(date) FROM portfolio_snapshots').fetchone()
if dates[0] and dates[1]:
    days = (dates[1] - dates[0]).days
    print(f'Days active: {days}')

db.close()
"
```

**Step 5: Check paper trading gate criteria**

```
## Paper Trading Gate

| Check | Threshold | Value | Status |
|-------|-----------|-------|--------|
| Days active | >= 30 | {days} | PASS/FAIL |
| Total trades | >= 50 | {trades} | PASS/FAIL |
| Paper Sharpe | >= 0.60 | {sharpe} | PASS/FAIL |
| Max drawdown | < 15% | {dd}% | PASS/FAIL |
| Slippage drift | < 5 bps | {drift} bps | PASS/FAIL |
| Critical incidents | 0 unresolved | {count} | PASS/FAIL |
| All operations tested | Yes | {status} | PASS/FAIL |

**OVERALL GATE: [PASS / FAIL / INSUFFICIENT DATA]**
```

If all gates pass:
```
Paper trading gate PASSED for {slug}.
Paper trading may continue or be concluded.
Next step: /promote {slug} (or /evaluate {slug} for ongoing monitoring)
```

If gates fail:
```
Paper trading gate FAILED for {slug}.
Failed checks: [list]
Continue paper trading until all gates pass.
```

If insufficient data:
```
Paper trading is still in progress for {slug}.
{days}/30 days, {trades}/50 trades completed.
Estimated completion: {estimate}
```

---

## Incident Severity Levels

| Severity | Description | Action |
|----------|-------------|--------|
| **critical** | System failure affecting P&L accuracy or trade execution. Examples: hash chain corruption, risk check bypass, data feed producing incorrect prices, trades executed against stale data. | Paper trading PAUSED. Must be resolved and documented before resuming. |
| **warning** | Operational issue that did not affect P&L but indicates fragility. Examples: data fetch timeout (recovered on retry), risk check near-miss, configuration drift detected. | Document and monitor. If 3+ warnings in 7 days, escalate to critical review. |
| **info** | Notable event for the record. Examples: regime change detected, new asset added to universe, rebalance triggered, parameter review completed. | Document only. |

---

## What Paper Trading Validates

1. **Data pipeline**: Can the system reliably fetch, store, and process market data every session?
2. **Indicator computation**: Do indicators compute correctly on live data (not just historical)?
3. **Signal generation**: Do signals fire at appropriate times given live market conditions?
4. **Risk checks**: Do all 7 pre-trade risk checks enforce limits correctly?
5. **Execution**: Does the paper executor correctly calculate shares, prices, and portfolio impact?
6. **Persistence**: Are portfolio snapshots saved with valid hash chains?
7. **Reporting**: Can performance metrics be computed and displayed accurately?

## What Paper Trading Cannot Validate

- Execution slippage and market impact
- Liquidity constraints during stress events
- Crowding effects
- Latency in data feeds and order routing
- Real capital psychological effects

---

## Lifecycle Position

```
Mandate --> Hypothesis --> Data Contract --> Research Spec --> Backtest --> Robustness --> [Paper] --> Promotion
```

Paper trading requires a passing robustness gate. The paper trading gate must pass before promotion review.

---

## Important

- Do NOT skip the 30-day minimum — shorter periods may coincidentally align with favorable conditions
- Do NOT skip the 50-trade minimum — fewer trades produce unreliable statistics
- Critical incidents MUST be resolved before the gate can pass — sweeping operational issues under the rug leads to production failures
- Slippage drift is estimated by comparing paper fill prices to actual market prices — even in paper trading, track this metric to understand the gap between assumed and realistic fills
- The operations checklist must be 100% complete — a strategy that has never tested hash chain integrity or risk check enforcement is not operationally validated

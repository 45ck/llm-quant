---
description: "Evaluate live strategy performance — compare to backtest, check for decay, retirement triggers"
---

# /evaluate — Strategy Performance Evaluation

You are the portfolio manager. Evaluation is ongoing — it runs after deployment and continues throughout the strategy's life. This command compares live performance against backtest expectations, detects alpha decay and regime drift, and recommends whether to continue, reduce, or retire the strategy.

A strategy that was good at deployment can become bad. Markets adapt. Edges decay. Regimes shift. Evaluation catches these changes before they become drawdown events.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> Evaluate all active strategies

Scan for strategies with paper trading or live status:

```bash
cd E:/llm-quant && find data/strategies -name "paper-trading.yaml" -type f 2>/dev/null
```

For each active strategy, run a quick evaluation and display:

```
## Strategy Evaluations

| Slug | Days Live | Sharpe (Live) | Sharpe (BT) | Decay? | DD | Recommendation |
|------|-----------|---------------|-------------|--------|-----|----------------|
| ...  | N         | X.XX          | X.XX        | Yes/No | -X% | Continue/Reduce/Retire |
```

---

### Slug provided (e.g., "momentum-rotation") --> Full evaluation

**Step 1: Load all artifacts**

Read the complete artifact chain:

```bash
cat E:/llm-quant/data/strategies/$SLUG/mandate.yaml 2>/dev/null
cat E:/llm-quant/data/strategies/$SLUG/hypothesis.yaml 2>/dev/null
cat E:/llm-quant/data/strategies/$SLUG/research-spec.yaml 2>/dev/null
cat E:/llm-quant/data/strategies/$SLUG/robustness.yaml 2>/dev/null
cat E:/llm-quant/data/strategies/$SLUG/paper-trading.yaml 2>/dev/null
```

Also read the experiment registry for backtest baselines:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import json, os, sys

slug = sys.argv[1] if len(sys.argv) > 1 else ''
registry_path = f'data/strategies/{slug}/experiment-registry.jsonl'
if os.path.exists(registry_path):
    with open(registry_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('slug') == slug:
                print(json.dumps(entry, indent=2))
" "$SLUG"
```

**Step 2: Pull live performance metrics**

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import duckdb
from datetime import date, timedelta

db = duckdb.connect('data/llm_quant.duckdb', read_only=True)

# Portfolio performance
print('=== Portfolio Performance ===')
snapshots = db.execute('''
    SELECT date, nav, cash, gross_exposure, net_exposure
    FROM portfolio_snapshots
    ORDER BY date DESC
    LIMIT 30
''').fetchall()

if snapshots:
    latest = snapshots[0]
    oldest = snapshots[-1]
    print(f'Latest NAV: \${latest[1]:,.2f} ({latest[0]})')
    print(f'30-session NAV: \${oldest[1]:,.2f} ({oldest[0]})')

    # Peak NAV
    peak = db.execute('SELECT MAX(nav) FROM portfolio_snapshots').fetchone()[0]
    current_dd = (latest[1] - peak) / peak * 100
    print(f'Peak NAV: \${peak:,.2f}')
    print(f'Current drawdown: {current_dd:.1f}%')

# Trade statistics
print('\n=== Trade Statistics ===')
trades = db.execute('''
    SELECT COUNT(*) as total,
           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
           SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losers,
           AVG(pnl) as avg_pnl,
           SUM(pnl) as total_pnl
    FROM trades
    WHERE pnl IS NOT NULL
''').fetchone()

if trades and trades[0] > 0:
    win_rate = trades[1] / trades[0] * 100 if trades[0] > 0 else 0
    print(f'Total trades: {trades[0]}')
    print(f'Win rate: {win_rate:.1f}%')
    print(f'Avg P&L: \${trades[3]:,.2f}')
    print(f'Total P&L: \${trades[4]:,.2f}')

# Recent P&L trend
print('\n=== Recent 10-Day P&L Trend ===')
recent = db.execute('''
    SELECT date, nav
    FROM portfolio_snapshots
    ORDER BY date DESC
    LIMIT 10
''').fetchall()

for row in recent:
    print(f'  {row[0]}: \${row[1]:,.2f}')

db.close()
"
```

**Step 2b: Generate quantstats tearsheet**

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/generate_tearsheet.py "$SLUG"
```

This writes `data/strategies/$SLUG/evaluate-tearsheet.html` and prints a key metrics table
(Sharpe, Sortino, MaxDD, Calmar, CAGR) vs the appropriate benchmark:
- Track A strategies: 60/40 SPY/TLT blended benchmark
- Track B strategies: SPY benchmark

Use `--track b` to override the benchmark if the strategy is Track B but not auto-detected.
Use `--no-html` to print the metrics table only without generating the HTML file.

**Step 3: Compare live vs backtest performance**

```
## Performance Comparison: {slug}

### Live vs Backtest

| Metric | Backtest | Live | Delta | Status |
|--------|----------|------|-------|--------|
| Sharpe | X.XX | X.XX | -X.XX | OK/WARN/CRITICAL |
| Sortino | X.XX | X.XX | -X.XX | OK/WARN/CRITICAL |
| Max DD | -X.X% | -X.X% | +/-X.X% | OK/WARN/CRITICAL |
| Win Rate | X.X% | X.X% | -X.X% | OK/WARN/CRITICAL |
| Avg Trade P&L | $X.XX | $X.XX | -$X.XX | OK/WARN/CRITICAL |
| Annualized Return | X.X% | X.X% | -X.X% | OK/WARN/CRITICAL |

Status thresholds:
- OK: Live within 20% of backtest
- WARN: Live 20-50% below backtest
- CRITICAL: Live >50% below backtest
```

**Step 4: Alpha decay analysis**

Check for systematic deterioration of the strategy's edge over time.

```
### Alpha Decay Analysis

Rolling Sharpe Ratio (63-day window):

| Period | Rolling Sharpe | Full-History Sharpe | Ratio | Status |
|--------|---------------|--------------------:|-------|--------|
| Current | X.XX | X.XX | X.XX | OK/WARN/HALT |
| -1 month | X.XX | X.XX | X.XX | ... |
| -2 months | X.XX | X.XX | X.XX | ... |
| -3 months | X.XX | X.XX | X.XX | ... |

Decay detection (per control-matrix.md):
- Ratio > 0.60: OK — performance within normal range
- Ratio 0.40-0.60: WARNING — performance degrading, investigate
- Ratio < 0.40: HALT — alpha decay confirmed, reduce sizing 50%
```

Trend analysis: Is the rolling Sharpe declining systematically? Compute linear regression slope of the last 4 rolling Sharpe observations.

```
Rolling Sharpe trend: [positive / flat / negative]
Slope: X.XX per month
Projection (3 months): X.XX
```

**Step 5: Regime drift analysis**

Check whether the current market regime matches the regime under which the strategy was designed and tested.

```
### Regime Analysis

| Indicator | At Deployment | Current | Drift? |
|-----------|---------------|---------|--------|
| VIX | X.X | X.X | Yes/No |
| Yield Spread (10Y-2Y) | X.XX% | X.XX% | Yes/No |
| SPY Trend (vs SMA200) | Above/Below | Above/Below | Yes/No |
| Regime Classification | risk_on/off/transition | risk_on/off/transition | Yes/No |

Regime drift detected: [Yes/No]
```

If regime has changed, note the implications for the strategy.

**Step 6: Retirement trigger check**

```
### Retirement Triggers

| Trigger | Threshold | Current | Fired? |
|---------|-----------|---------|--------|
| Max drawdown breach | > 15% | {dd}% | Yes/No |
| Sustained alpha decay | Ratio < 0.40 for 21+ days | {days} | Yes/No |
| Kill switch activation | Any kill switch | {status} | Yes/No |
| Hypothesis falsified | Meets falsification criteria | {status} | Yes/No |
| Benchmark underperformance | < benchmark in 60%+ of rolling windows | {pct}% | Yes/No |

Retirement triggers fired: {count}
```

**Step 7: Generate recommendation**

Based on all analysis, recommend one of:

**CONTINUE** (all clear):
```
Recommendation: CONTINUE
- Live performance within acceptable range of backtest expectations
- No alpha decay detected
- No regime drift
- No retirement triggers fired
- Next review: {date + 30 days}
```

**REDUCE** (warning signs):
```
Recommendation: REDUCE exposure by 50%
Reasons:
- [List specific warning signs]
Actions:
- Reduce all position sizes to 50% of target weight
- Tighten trailing stops by 25%
- Increase evaluation frequency to weekly
- Re-evaluate in 14 days
```

**RETIRE** (critical failures):
```
Recommendation: RETIRE strategy
Reasons:
- [List specific retirement triggers]
Actions:
- Close all positions at next session
- Document the failure mode and lessons learned
- Record retirement in strategy_changelog
- Consider: was the hypothesis falsified, or was this an operational failure?
```

**Step 8: Write evaluation artifact**

Write the evaluation to `data/strategies/{slug}/evaluation-{date}.yaml`:

```yaml
# Evaluation: {slug} — {date}
strategy_slug: "{slug}"
evaluation_date: "YYYY-MM-DD"
evaluator: "automated"

live_vs_backtest:
  sharpe_delta_pct: -X.X
  sortino_delta_pct: -X.X
  drawdown_delta_pct: X.X

alpha_decay:
  current_ratio: X.XX
  trend_slope: X.XX
  status: "ok/warning/halt"

regime_drift:
  detected: true/false
  details: "..."

retirement_triggers_fired: 0
recommendation: "continue/reduce/retire"
next_review_date: "YYYY-MM-DD"
```

---

## Evaluation Schedule

| Context | Frequency | Scope |
|---------|-----------|-------|
| First 30 days post-deployment | Daily | Full evaluation |
| Steady state | Monthly | Full evaluation |
| After kill switch event | Immediately | Full evaluation + incident review |
| After regime change | Within 1 session | Regime-focused evaluation |
| Quarterly review | Quarterly | Full evaluation + scorecard re-scoring |
| Annual re-certification | Annually | Full promotion pipeline Stages 1-2 |

---

## Lifecycle Position

```
Mandate --> Hypothesis --> Data Contract --> Research Spec --> Backtest --> Robustness --> Paper --> Promotion --> [Evaluate] (ongoing)
```

Evaluation is the only lifecycle stage that runs continuously. It monitors deployed strategies for decay and triggers retirement when the edge is gone.

---

## Important

- Do NOT ignore alpha decay — it is slow and insidious, and by the time it shows up in cumulative P&L the damage is done
- Do NOT compare live performance to zero — compare to the benchmark and to backtest expectations
- Regime drift does not automatically mean the strategy is broken — but it means the strategy is operating outside its tested conditions
- A retired strategy is not a failure — it is the system working correctly. Edges decay. The failure is not retiring when you should.
- Always check the hypothesis falsification criteria — if the hypothesis is falsified, the strategy has no theoretical basis regardless of recent P&L
- Record every evaluation — the history of evaluations is itself informative (deteriorating trend across evaluations is a signal)

### Step 9: Generate daily report (if not already done today)

After evaluation, ensure today's daily report exists:
```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py daily
```

This ensures every evaluation session leaves an auditable trail.

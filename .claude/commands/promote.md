---
description: "Run the strategy promotion checklist — verify all gates before deploying a strategy change"
---

# /promote — Strategy Promotion Checklist

You are the portfolio manager running a strategy change review. Walk through every gate in the Model Promotion Policy before approving any strategy modification.

**Reference docs:**
- `docs/governance/model-promotion-policy.md` — Full promotion pipeline
- `docs/governance/control-matrix.md` — Failure modes and kill switches
- `docs/governance/quant-lifecycle.md` — Research lifecycle reference

## Parse the user's argument: "$ARGUMENTS"

If a slug is provided (e.g., `/promote sma_spy`), use it. Otherwise, ask which strategy to evaluate.

---

### Step 1: Machine-Enforced Gate Checks

Run the automated promotion gate. This checks all thresholds from the research lifecycle artifacts.

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import sys
from pathlib import Path
from llm_quant.backtest.artifacts import (
    strategy_dir, get_lifecycle_state, ExperimentRegistry,
    check_data_grade, LifecycleState,
)

slug = sys.argv[1].strip() if len(sys.argv) > 1 else 'NONE'
if slug == 'NONE':
    # List available strategies
    base = Path('data/strategies')
    if base.exists():
        slugs = [d.name for d in base.iterdir() if d.is_dir()]
        print('Available strategies:', slugs)
    else:
        print('No strategies found in data/strategies/')
    exit(0)

sdir = strategy_dir(Path('data'), slug)
state = get_lifecycle_state(sdir)
print(f'Strategy: {slug}')
print(f'Lifecycle state: {state.value}')
print()

# Check lifecycle progression
print('=== Lifecycle Gates ===')
required_states = [
    LifecycleState.MANDATE,
    LifecycleState.HYPOTHESIS,
    LifecycleState.DATA_CONTRACT,
    LifecycleState.RESEARCH_SPEC,
]
for rs in required_states:
    artifact_file = {
        LifecycleState.MANDATE: 'mandate.yaml',
        LifecycleState.HYPOTHESIS: 'hypothesis.yaml',
        LifecycleState.DATA_CONTRACT: 'data-contract.yaml',
        LifecycleState.RESEARCH_SPEC: 'research-spec.yaml',
    }[rs]
    exists = (sdir / artifact_file).exists()
    status = 'PASS' if exists else 'MISSING'
    print(f'  {rs.value}: {status}')

# Check frozen spec
spec_path = sdir / 'research-spec.yaml'
if spec_path.exists():
    import yaml
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    frozen = spec.get('frozen', False)
    print(f'  Spec frozen: {\"PASS\" if frozen else \"FAIL\"} (frozen={frozen})')
else:
    print('  Spec frozen: FAIL (no spec)')

print()

# Check experiment registry
print('=== Experiment Registry ===')
registry = ExperimentRegistry(sdir)
trial_count = registry.trial_count
print(f'  Total trials: {trial_count}')
if trial_count == 0:
    print('  WARNING: No experiments recorded — cannot compute DSR')
else:
    entries = registry.load_all()
    best = max(entries, key=lambda e: e.get('sharpe', 0))
    print(f'  Best Sharpe: {best.get(\"sharpe\", \"N/A\")}')
    print(f'  Best experiment: {best.get(\"experiment_id\", \"unknown\")}')
print()

# Data grade gate
print('=== Data Grade ===')
dc_path = sdir / 'data-contract.yaml'
if dc_path.exists():
    with open(dc_path) as f:
        dc = yaml.safe_load(f)
    grade = dc.get('quality_grade', 'unknown')
    passes = check_data_grade(grade, 'b')
    status = 'PASS' if passes else 'FAIL'
    print(f'  Grade: {grade} (minimum: b) — {status}')
else:
    print('  Grade: UNKNOWN (no data-contract.yaml)')
print()
" "$ARGUMENTS"
```

---

### Step 2: Check Research Gate Results

If a robustness gate has been run, check its results:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import sys
from pathlib import Path
import json

slug = sys.argv[1].strip() if len(sys.argv) > 1 else 'NONE'
if slug == 'NONE':
    exit(0)

sdir = Path('data/strategies') / slug

# Check for robustness results
rob_path = sdir / 'robustness-result.yaml'
if rob_path.exists():
    import yaml
    with open(rob_path) as f:
        rob = yaml.safe_load(f)
    print('=== Robustness Gate ===')
    print(f'  Overall: {\"PASS\" if rob.get(\"overall_passed\") else \"FAIL\"}')
    for gate, passed in rob.get('gate_details', {}).items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  {gate}: {status}')
else:
    print('=== Robustness Gate ===')
    print('  NOT RUN — execute /robustness first')

print()

# Check experiment metrics for DSR
registry_path = sdir / 'experiment-registry.jsonl'
if registry_path.exists():
    entries = []
    with open(registry_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print('=== DSR / Statistical Gates ===')
    if entries:
        best = max(entries, key=lambda e: e.get('sharpe', 0))
        dsr = best.get('dsr', 0)
        psr = best.get('psr', 0)
        sharpe = best.get('sharpe', 0)
        total_trials = len(entries)

        dsr_status = 'PASS' if dsr >= 0.95 else 'FAIL'
        print(f'  DSR: {dsr:.4f} (>= 0.95) — {dsr_status}')
        print(f'  PSR: {psr:.4f}')
        print(f'  Best Sharpe: {sharpe:.4f}')
        print(f'  Total trials (N): {total_trials}')
    else:
        print('  No experiments found')

    # Cost survival
    print()
    print('=== Cost Sensitivity ===')
    if entries:
        best_entry = max(entries, key=lambda e: e.get('sharpe', 0))
        metrics = best_entry.get('metrics', {})
        for mult_key in sorted(metrics.keys()):
            m = metrics[mult_key]
            sharpe = m.get('sharpe_ratio', 0) if isinstance(m, dict) else 0
            status = 'PASS' if sharpe > 0 else 'FAIL'
            print(f'  {mult_key}: Sharpe={sharpe:.3f} — {status}')

        cost_2x = metrics.get('2.0x', {})
        if isinstance(cost_2x, dict):
            survives = cost_2x.get('sharpe_ratio', 0) > 0
            print(f'  2x cost survival: {\"PASS\" if survives else \"FAIL\"}')
else:
    print('=== DSR / Statistical Gates ===')
    print('  No experiment registry found — run /backtest first')
" "$ARGUMENTS"
```

---

### Step 3: Check Paper Trading Record

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import sys
import duckdb
from datetime import date, timedelta

slug = sys.argv[1].strip() if len(sys.argv) > 1 else 'NONE'
if slug == 'NONE':
    exit(0)

print('=== Paper Trading Minimums ===')

db = duckdb.connect('data/llm_quant.duckdb', read_only=True)

# Trade count
try:
    trades = db.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
    status = 'PASS' if trades >= 50 else 'FAIL'
    print(f'  Trades executed: {trades} (need >= 50) — {status}')
except Exception:
    print('  Trades: Unable to query')

# Calendar days
try:
    dates = db.execute('SELECT MIN(date), MAX(date) FROM portfolio_snapshots').fetchone()
    if dates[0] and dates[1]:
        days = (dates[1] - dates[0]).days
        status = 'PASS' if days >= 30 else 'FAIL'
        print(f'  Calendar days: {days} (need >= 30) — {status}')
    else:
        print('  Calendar days: No snapshots found')
except Exception:
    print('  Calendar days: Unable to query')

# Paper Sharpe
try:
    navs = db.execute('SELECT nav FROM portfolio_snapshots ORDER BY date').fetchall()
    if len(navs) >= 2:
        import numpy as np
        nav_values = [r[0] for r in navs]
        returns = np.diff(nav_values) / nav_values[:-1]
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            status = 'PASS' if sharpe >= 0.60 else 'FAIL'
            print(f'  Paper Sharpe: {sharpe:.3f} (need >= 0.60) — {status}')
        else:
            print('  Paper Sharpe: Insufficient variance')
    else:
        print('  Paper Sharpe: Not enough snapshots')
except Exception:
    print('  Paper Sharpe: Unable to compute')

# Max drawdown
try:
    navs = db.execute('SELECT nav FROM portfolio_snapshots ORDER BY date').fetchall()
    if len(navs) >= 2:
        nav_values = [r[0] for r in navs]
        peak = nav_values[0]
        max_dd = 0
        for n in nav_values:
            peak = max(peak, n)
            dd = (peak - n) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        status = 'PASS' if max_dd < 0.15 else 'FAIL'
        print(f'  Max drawdown: {max_dd:.2%} (need < 15%) — {status}')
except Exception:
    print('  Max drawdown: Unable to compute')

# Incidents
print()
print('=== Incident Record ===')
try:
    tables = [r[0] for r in db.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema='main'\").fetchall()]
    if 'surveillance_scans' in tables:
        halts = db.execute(\"SELECT COUNT(*) FROM surveillance_scans WHERE overall_severity = 'halt'\").fetchone()[0]
        status = 'PASS' if halts == 0 else 'FAIL'
        print(f'  Critical incidents (halts): {halts} (need 0) — {status}')
    else:
        print('  No surveillance scan history found')
except Exception:
    print('  Unable to check incident record')

db.close()
" "$ARGUMENTS"
```

---

### Step 4: Promotion Checklist Summary

Present the complete promotion checklist with all gate results:

```markdown
## Strategy Promotion Checklist — {slug}

### Hard Vetoes (any failure = automatic rejection)
- [ ] DSR >= 0.95
- [ ] PBO <= 0.10
- [ ] 2x cost survival (Sharpe > 0 at 2x costs)
- [ ] Data grade >= B
- [ ] Research spec frozen before backtesting
- [ ] CPCV mean OOS Sharpe > 0
- [ ] Parameter stability > 50%

### Paper Trading Minimums
- [ ] >= 50 trades executed
- [ ] >= 30 calendar days of paper trading
- [ ] Paper Sharpe >= 0.60
- [ ] Max drawdown < 15%
- [ ] Slippage drift < 5 bps vs backtest assumptions
- [ ] Zero unresolved critical incidents

### Canary Gate
- [ ] 10% portfolio allocation assigned to canary
- [ ] >= 14 calendar days in canary
- [ ] Canary drawdown < 10%
- [ ] No kill switch triggers during canary

### Deployment Readiness
- [ ] All 7 kill switches configured and active
- [ ] All surveillance detectors operational
- [ ] Changes documented in strategy_changelog
- [ ] Baseline metrics recorded for regime change detection
- [ ] Enhanced surveillance plan in place (30-day daily review)
```

---

### Step 5: Log the promotion attempt

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import sys
import duckdb
from datetime import datetime

slug = sys.argv[1].strip() if len(sys.argv) > 1 else 'unknown'

db = duckdb.connect('data/llm_quant.duckdb')

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

db.execute('''
    INSERT INTO strategy_changelog (change_type, description, outcome, details)
    VALUES (
        'promotion_review',
        'Strategy promotion checklist executed for ' || ?,
        'pending',
        'Review initiated on ' || CAST(CURRENT_TIMESTAMP AS VARCHAR)
    )
''', [slug])

print('Promotion review logged to strategy_changelog.')
db.close()
" "$ARGUMENTS"
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

**Hard rules:**
- If ANY hard veto fails → REJECT (no exceptions)
- If DSR < 0.95 → REJECT (multiple testing penalty not satisfied)
- If PBO > 0.10 → REJECT (overfitting risk too high)
- If data grade < B → REJECT (data quality insufficient)
- If research spec was not frozen before backtesting → REJECT (methodology violation)
- If paper trading minimums not met → CONDITIONAL (needs more paper time)
- If canary gate not run → CONDITIONAL (deploy to canary first)

---

## Important

- Follow the Model Promotion Policy (`docs/governance/model-promotion-policy.md`) exactly
- Do not approve a promotion if any hard veto fails
- Do not skip stages — the pipeline is sequential
- All promotion decisions must be logged to `strategy_changelog`
- Reference the Control Matrix (`docs/governance/control-matrix.md`) for kill switch verification
- The experiment registry trial count N directly penalizes DSR — more trials = harder to pass

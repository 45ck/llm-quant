---
description: "Run a backtest against a frozen research spec — produces experiment artifact"
---

# /backtest — Run Backtest Experiment

You are the portfolio manager. A backtest is an experiment that tests a frozen research spec against historical data. It produces a durable experiment artifact recorded in an append-only registry. Backtests are never deleted or modified — they accumulate as evidence for or against the hypothesis.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> Show experiment registry

Read the experiment registry and display recent experiments:

```bash
cd E:/llm-quant && for d in data/strategies/*/; do cat "${d}experiment-registry.jsonl" 2>/dev/null; done | tail -20
```

If no registry exists:
```
No experiments found. Run a backtest with: /backtest <slug>
```

Display as a table:

```
## Experiment Registry

| # | Slug | Date | Spec Hash | Sharpe | Sortino | Max DD | DSR | Trial # | Status |
|---|------|------|-----------|--------|---------|--------|-----|---------|--------|
| 1 | ...  | ...  | ...       | ...    | ...     | ...    | ... | ...     | ...    |
```

---

### Slug provided (e.g., "momentum-rotation") --> Run backtest

**Step 1: Verify frozen research spec**

```bash
cat E:/llm-quant/data/strategies/$SLUG/research-spec.yaml 2>/dev/null
```

If the spec does not exist, STOP:
```
ERROR: No research spec found for strategy "{slug}".
Create one with: /research-spec {slug}
```

If the spec exists but `frozen: false`, STOP:
```
ERROR: Research spec for "{slug}" is NOT frozen.
You MUST freeze the spec before backtesting to prevent data snooping.
Run: /research-spec freeze {slug}
```

**Step 2: Read supporting artifacts**

Read the mandate, hypothesis, and data contract to understand the full context:

```bash
cat E:/llm-quant/data/strategies/$SLUG/mandate.yaml
cat E:/llm-quant/data/strategies/$SLUG/hypothesis.yaml
cat E:/llm-quant/data/strategies/$SLUG/data-contract.yaml
```

**Step 3: Determine trial number**

Count existing experiments for this slug to compute the trial number (used for DSR):

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import json, os, sys

slug = sys.argv[1] if len(sys.argv) > 1 else ''
registry_path = f'data/strategies/{slug}/experiment-registry.jsonl'
if not os.path.exists(registry_path):
    print('trial_number=1')
else:
    count = 0
    with open(registry_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('slug') == slug:
                count += 1
    print(f'trial_number={count + 1}')
" "$SLUG"
```

**Step 4: Run the backtest**

Parse optional arguments from `$ARGUMENTS` after the slug:
- `--symbols SYM1,SYM2,...` to override the universe (subset testing)
- `--years N` to limit the backtest period (default: full data contract range)
- `--no-costs` to run a zero-cost comparison (for diagnostic purposes only, not for evaluation)

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/run_backtest.py --slug $SLUG [--symbols SYMS] [--years N]
```

If `scripts/run_backtest.py` does not exist yet, inform the user:
```
Backtest script not yet implemented. To run the backtest manually:

1. Load data per the data contract
2. Compute indicators per the research spec
3. Generate signals per the signal definitions
4. Apply entry/exit/risk rules
5. Compute performance metrics
6. Record results to experiment registry

See the research spec for full methodology: data/strategies/{slug}/research-spec.yaml
```

**Step 5: Display results**

Present the backtest results in a structured format:

```
## Backtest Results: {slug} (Trial #{n})

**Spec hash:** {hash}
**Period:** {start} to {end}
**Universe:** {symbols}

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Annualized Return | X.X% | 8-15% | PASS/FAIL |
| Sharpe Ratio | X.XX | > 0.80 | PASS/FAIL |
| Sortino Ratio | X.XX | > 1.00 | PASS/FAIL |
| Calmar Ratio | X.XX | > 0.50 | PASS/FAIL |
| Max Drawdown | -X.X% | < 15% | PASS/FAIL |
| Win Rate | X.X% | - | - |
| Profit Factor | X.XX | - | - |
| Total Trades | N | - | - |

### Deflated Sharpe Ratio (DSR)

| Component | Value |
|-----------|-------|
| Observed Sharpe | X.XX |
| Trial Number | N |
| Skewness | X.XX |
| Kurtosis | X.XX |
| Track Length (days) | N |
| DSR | X.XX |
| DSR Gate (>= 0.95) | PASS/FAIL |

### Cost Sensitivity Analysis

| Cost Multiplier | Sharpe | Return | Max DD | Survives? |
|-----------------|--------|--------|--------|-----------|
| 1.0x (base) | X.XX | X.X% | -X.X% | Yes/No |
| 1.5x | X.XX | X.X% | -X.X% | Yes/No |
| 2.0x | X.XX | X.X% | -X.X% | Yes/No |
| 3.0x | X.XX | X.X% | -X.X% | Yes/No |

"Survives" = Sharpe > 0.40 and max drawdown < 15%

### Benchmark Comparison

| Metric | Strategy | Benchmark (60/40) | Difference |
|--------|----------|-------------------|------------|
| Return | X.X% | X.X% | +/-X.X% |
| Sharpe | X.XX | X.XX | +/-X.XX |
| Max DD | -X.X% | -X.X% | +/-X.X% |
```

**Step 6: Record to experiment registry**

Append the result to the experiment registry (JSONL format, one entry per line, append-only):

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import json, os, sys
from datetime import date

slug = sys.argv[1] if len(sys.argv) > 1 else ''

entry = {
    'experiment_id': '{trial_number}',
    'slug': slug,
    'spec_hash': '{hash}',
    'trial_number': {trial_number},
    'date': str(date.today()),
    'period_start': '{start}',
    'period_end': '{end}',
    'symbols': [{symbols}],
    'metrics': {
        'annualized_return': 0.0,
        'sharpe': 0.0,
        'sortino': 0.0,
        'calmar': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'total_trades': 0,
        'dsr': 0.0
    },
    'cost_sensitivity': {
        '1.0x': {'sharpe': 0.0, 'survives': True},
        '1.5x': {'sharpe': 0.0, 'survives': True},
        '2.0x': {'sharpe': 0.0, 'survives': True},
        '3.0x': {'sharpe': 0.0, 'survives': True}
    },
    'benchmark_comparison': {
        'strategy_sharpe': 0.0,
        'benchmark_sharpe': 0.0,
        'excess_sharpe': 0.0
    },
    'status': 'completed'
}

registry_path = f'data/strategies/{slug}/experiment-registry.jsonl'
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
with open(registry_path, 'a') as f:
    f.write(json.dumps(entry) + '\n')

print(f'Experiment recorded: trial #{entry[\"trial_number\"]} for {entry[\"slug\"]}')
" "$SLUG"
```

**Step 7: Next steps**

```
Experiment recorded to: data/strategies/{slug}/experiment-registry.jsonl

Next steps:
- Run additional experiments if needed (each increments the trial counter)
- When you have >= 2 experiments, run robustness analysis: /robustness {slug}
- Remember: more trials = higher DSR threshold needed (multiple testing penalty)
```

---

## Deflated Sharpe Ratio (DSR)

The DSR adjusts the observed Sharpe ratio for the number of trials conducted. More trials = more chances to find a lucky result = higher bar for statistical significance.

Formula (Bailey & Lopez de Prado, 2014):

```
DSR = Prob[ SR* > 0 ] where SR* is the Sharpe adjusted for:
  - Number of trials (N): more trials inflate expected max Sharpe
  - Skewness of returns: negative skew makes Sharpe unreliable
  - Kurtosis of returns: fat tails inflate Sharpe variance
  - Track length: longer track = more reliable estimate
```

Gate: DSR >= 0.95 (i.e., 95% probability that the observed Sharpe is not a multiple-testing artifact).

**Critical implication:** Every time you run a backtest with different parameters, you are conducting a trial. Even if you only keep the "best" result, the DSR penalizes you for all the trials you ran. This is why the research spec must be frozen before backtesting — it limits the temptation to run dozens of parameter sweeps.

---

## Experiment Registry Rules

1. **Append-only**: Never delete or modify existing entries. The registry is an audit trail.
2. **Every run is recorded**: Including failures, poor results, and diagnostic runs. Hiding bad results is data snooping.
3. **Spec hash links to spec**: Every experiment references the frozen spec hash, creating a verifiable chain from design to result.
4. **Trial counter is per-slug**: Different slugs have independent trial counters. This is why parameter changes should use a new slug, not modify the existing one.

---

## Lifecycle Position

```
Mandate --> Hypothesis --> Data Contract --> Research Spec (frozen) --> [Backtest] --> Robustness --> Paper --> Promotion
```

A backtest requires a frozen research spec. At least 2 backtest experiments are needed before robustness analysis.

---

## Important

- NEVER run a backtest on an unfrozen spec — this is the cardinal sin of quantitative research
- NEVER delete experiments from the registry — bad results are evidence too
- ALWAYS include costs — a zero-cost backtest is fiction (but can be run as a diagnostic with `--no-costs`)
- The DSR penalizes you for EVERY trial — be deliberate about what you test
- Compare against the benchmark from the mandate, not against zero return
- If the backtest script does not exist, guide the user through manual implementation but do NOT pretend results exist

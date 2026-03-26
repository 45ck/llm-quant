---
description: "Run robustness analysis — PBO, CPCV, perturbation suite, DSR gate"
---

# /robustness — Robustness Analysis

You are the portfolio manager. Robustness analysis is the gate between "this backtest looks good" and "this strategy is likely real." It subjects the strategy to every reasonable stress test: combinatorial overfitting analysis (PBO), cross-validated out-of-sample testing (CPCV), and systematic perturbation of parameters and assumptions.

A strategy that passes only one backtest is a hypothesis. A strategy that survives the robustness suite is a candidate.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> Show robustness status for all strategies

Scan for existing robustness artifacts:

```bash
cd E:/llm-quant && find data/strategies -name "robustness.yaml" -type f 2>/dev/null
```

Display a summary:

```
## Robustness Analysis Status

| Slug | Experiments | PBO | CPCV OOS Sharpe | Perturbation | DSR | Gate |
|------|------------|-----|-----------------|--------------|-----|------|
| ...  | N          | X%  | X.XX            | X/Y pass     | X.X | PASS/FAIL |
```

---

### Slug provided (e.g., "momentum-rotation") --> Run robustness analysis

**Step 0: Detect strategy track (Track A/B vs Track C structural arb)**

Track C strategies use different gates (paper persistence + fill rate, not DSR/CPCV).
Check if the slug is a Track C structural arb strategy:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import sys
slug = '$ARGUMENTS'.strip()
TRACK_C_PREFIXES = ('pm-arb-', 'cef-', 'funding-')
is_track_c = any(slug.startswith(p) for p in TRACK_C_PREFIXES)
print('TRACK_C' if is_track_c else 'TRACK_AB')
print(slug)
"
```

**If TRACK_C → run the Track C gate and stop (do not run DSR/CPCV/PBO steps below):**

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/run_track_c_robustness.py \
    --slug $ARGUMENTS --db data/arb.duckdb
```

This script:
- Routes `pm-arb-*` slugs to `PaperArbGate` (4 gates: persistence, fill_rate, capacity, days_elapsed)
- Routes `cef-*` slugs to the CEF backtest gate (Sharpe, MaxDD, beta, persistence)
- Routes `funding-*` slugs to the funding rate gate (annualized spread, fill rate, exchanges)
- Writes `data/strategies/{slug}/robustness-result.yaml` in the standard promote-compatible format
- Exits 0 (PROMOTE), 1 (REJECT), or 2 (CONTINUE_PAPER)

For Track C strategies, the robustness gate IS the 30-day paper track record.
DSR/CPCV/PBO do not apply — deductive arb has no in-sample overfitting risk.

**If TRACK_AB → continue with Step 1 below.**

---

**Step 1: Verify prerequisites**

Check that at least 2 backtest experiments exist:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import json, os, sys

slug = sys.argv[1] if len(sys.argv) > 1 else ''
registry_path = f'data/strategies/{slug}/experiment-registry.jsonl'
if not os.path.exists(registry_path):
    print('ERROR: No experiment registry found. Run at least 2 backtests first.')
    exit(1)

count = 0
with open(registry_path) as f:
    for line in f:
        entry = json.loads(line.strip())
        if entry.get('slug') == slug:
            count += 1

if count < 2:
    print(f'ERROR: Only {count} experiment(s) found for {slug}. Need >= 2.')
    print(f'Run more backtests with: /backtest {slug}')
    exit(1)
else:
    print(f'Found {count} experiments for {slug}. Proceeding with robustness analysis.')
" "$ARGUMENTS"
```

Also verify the frozen research spec exists:

```bash
cat E:/llm-quant/data/strategies/$SLUG/research-spec.yaml 2>/dev/null
```

**Step 2: Run PBO via Combinatorial Symmetric Cross-Validation (CSCV)**

PBO estimates the probability that the backtest's performance is an artifact of overfitting.

**Method:**
1. Split the full backtest period into S=16 equal sub-periods
2. For each of C(16,8) = 12,870 combinations:
   a. Use 8 sub-periods as in-sample (IS), 8 as out-of-sample (OOS)
   b. Select the strategy configuration that performs best IS
   c. Measure its OOS performance
3. PBO = fraction of combinations where OOS rank of the IS-best config is below median

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
# PBO via CSCV
# S=16 sub-periods, C(16,8) = 12870 combinations
from math import comb
S = 16
half = S // 2
n_combinations = comb(S, half)
print(f'CSCV Configuration:')
print(f'  Sub-periods (S): {S}')
print(f'  Combinations: C({S},{half}) = {n_combinations}')
print(f'  Each split: {half} IS + {half} OOS sub-periods')
print()
print('PBO = fraction of combinations where OOS performance of IS-best is below median')
print('Gate: PBO <= 0.10 (at most 10% of splits show overfitting)')
"
```

If the PBO computation script exists:
```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/run_pbo.py --slug $SLUG
```

Otherwise, document the computation that needs to be implemented and show what the output should look like.

**Step 3: Run CPCV (Combinatorially Purged Cross-Validation)**

Using the parameters from the research spec's validation section:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
from math import comb

N = 6   # groups
k = 2   # test groups per combination
n_combinations = comb(N, k)
purge_days = 5
embargo_pct = 0.01

print(f'CPCV Configuration:')
print(f'  Groups (N): {N}')
print(f'  Test groups (k): {k}')
print(f'  Combinations: C({N},{k}) = {n_combinations}')
print(f'  Each split: {N-k} train + {k} test groups')
print(f'  Purge gap: {purge_days} days between train/test')
print(f'  Embargo: {embargo_pct*100}% of test data at boundaries')
print()
print('For each of 15 combinations, train on 4 groups and test on 2 groups.')
print('This produces 15 independent OOS Sharpe estimates.')
"
```

If the CPCV computation script exists:
```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/run_cpcv.py --slug $SLUG
```

**Step 4: Run perturbation suite**

Systematically test the strategy's sensitivity to parameter changes, cost assumptions, execution delays, and rebalance frequency:

**4a. Parameter perturbation (+/-20%)**

For each free parameter in the research spec, test at 0.8x and 1.2x the specified value:

```
## Parameter Perturbation (+/-20%)

| Parameter | Base | -20% | Sharpe (-20%) | +20% | Sharpe (+20%) | Stable? |
|-----------|------|------|---------------|------|---------------|---------|
| sma_fast | 20 | 16 | X.XX | 24 | X.XX | Yes/No |
| sma_slow | 50 | 40 | X.XX | 60 | X.XX | Yes/No |
| ... | ... | ... | ... | ... | ... | ... |

"Stable" = perturbed Sharpe within 30% of base Sharpe
```

**4b. Cost multiplier sensitivity**

| Cost Multiplier | Sharpe | Max DD | Survives? |
|-----------------|--------|--------|-----------|
| 1.0x (base) | X.XX | -X.X% | Yes/No |
| 1.5x | X.XX | -X.X% | Yes/No |
| 2.0x | X.XX | -X.X% | Yes/No |
| 3.0x | X.XX | -X.X% | Yes/No |

**4c. Signal delay sensitivity**

| Fill Delay (bars) | Sharpe | Max DD | Survives? |
|-------------------|--------|--------|-----------|
| 0 (same-bar, unrealistic) | X.XX | -X.X% | N/A |
| 1 (next-bar, base) | X.XX | -X.X% | Yes/No |
| 2 (2-bar delay) | X.XX | -X.X% | Yes/No |

**4d. Rebalance frequency sensitivity**

| Rebalance (days) | Sharpe | Max DD | Turnover | Survives? |
|-------------------|--------|--------|----------|-----------|
| 5 (weekly) | X.XX | -X.X% | X.X% | Yes/No |
| 21 (monthly, base) | X.XX | -X.X% | X.X% | Yes/No |
| 63 (quarterly) | X.XX | -X.X% | X.X% | Yes/No |

**Step 5: Evaluate gate criteria**

ALL of the following must pass for the robustness gate to open:

```
## Robustness Gate

| Check | Threshold | Value | Status |
|-------|-----------|-------|--------|
| DSR | >= 0.95 | X.XX | PASS/FAIL |
| PBO | <= 0.10 | X.XX | PASS/FAIL |
| 2x costs survive | Sharpe > 0 | X.XX | PASS/FAIL |
| CPCV mean OOS Sharpe | > 0 | X.XX | PASS/FAIL |
| CPCV median OOS Sharpe | > 0 | X.XX | PASS/FAIL |
| Parameter stability | > 50% stable | X/Y | PASS/FAIL |

**OVERALL GATE: [PASS / FAIL]**
```

A single failure blocks the gate. There are no partial passes.

**Step 6: Write robustness artifact**

```bash
cd E:/llm-quant && mkdir -p data/strategies/$SLUG
```

Write the results to `data/strategies/{slug}/robustness.yaml`:

```yaml
# Robustness Analysis: {slug}
strategy_slug: "{slug}"
spec_hash: "{hash}"
date: "YYYY-MM-DD"
experiments_analyzed: N

pbo:
  method: "CSCV"
  sub_periods: 16
  combinations: 12870
  pbo_estimate: 0.XX
  gate_threshold: 0.10
  gate_status: "pass/fail"

cpcv:
  groups: 6
  test_groups: 2
  combinations: 15
  purge_days: 5
  embargo_pct: 0.01
  oos_sharpe_mean: X.XX
  oos_sharpe_median: X.XX
  oos_sharpe_std: X.XX
  oos_sharpe_min: X.XX
  oos_sharpe_max: X.XX
  gate_status: "pass/fail"

perturbation:
  parameters_tested: N
  parameters_stable: M
  stability_pct: X.X
  cost_2x_survives: true/false
  signal_delay_2bar_survives: true/false
  gate_status: "pass/fail"

dsr:
  observed_sharpe: X.XX
  trial_count: N
  dsr_value: X.XX
  gate_threshold: 0.95
  gate_status: "pass/fail"

overall_gate: "pass/fail"
```

**Step 7: Report and next steps**

If PASS:
```
Robustness gate PASSED for {slug}.
Next step: /paper {slug}
```

If FAIL:
```
Robustness gate FAILED for {slug}.
Failed checks: [list]

Options:
1. Revise the strategy design (new slug, new mandate)
2. Investigate specific failure modes
3. Accept the limitations and document why (NOT recommended for promotion)
```

---

## PBO Formula

The Probability of Backtest Overfitting (PBO) measures how often the best in-sample configuration fails out-of-sample.

Given S sub-periods, evaluate all C(S, S/2) ways to split them into in-sample (IS) and out-of-sample (OOS):

```
For each combination c:
  1. Rank all strategy variants by IS performance
  2. Select the IS-best variant
  3. Measure its OOS rank (relative to other variants)
  4. w_c = 1 if OOS rank is below median, 0 otherwise

PBO = (1/C) * sum(w_c for all c)
```

PBO = 0 means the IS-best always performs well OOS (no overfitting).
PBO = 1 means the IS-best always performs poorly OOS (complete overfitting).
Gate: PBO <= 0.10.

---

## Lifecycle Position

```
Mandate --> Hypothesis --> Data Contract --> Research Spec --> Backtest --> [Robustness] --> Paper --> Promotion
```

Robustness analysis requires at least 2 completed backtest experiments. All gate checks must pass before paper trading.

---

## Important

- ALL gates must pass — there is no partial credit
- PBO is the most computationally expensive check (12,870 splits) but the most informative
- Parameter stability > 50% means more than half of parameters are robust to +/-20% perturbation
- The 2x cost survival check is critical — if the strategy dies at 2x costs, the cost model is load-bearing and any cost estimation error kills the strategy
- Do NOT cherry-pick which perturbations to report — report all of them, including failures
- If computation is not yet implemented, document exactly what needs to be built and do NOT fabricate results

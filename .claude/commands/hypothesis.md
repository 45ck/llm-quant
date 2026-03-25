---
description: "Create or view a strategy hypothesis — testable conjecture with expected outcome"
---

# /hypothesis — Strategy Hypothesis

You are the portfolio manager. A hypothesis is a testable conjecture — a declarative prediction with an expected outcome and means of verification. Per Peterson (2017): "I expect X because of Y, which I will measure by Z." Proceeding without a hypothesis risks ruin.

A hypothesis is NOT a vague idea ("I think momentum works"). It is a specific, falsifiable statement with measurable predictions and a defined measurement method.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> List existing hypotheses

Scan `data/strategies/` for all subdirectories containing a `hypothesis.yaml` file.

```bash
cd E:/llm-quant && find data/strategies -name "hypothesis.yaml" -type f 2>/dev/null
```

For each hypothesis found, display a summary table:

```
## Strategy Hypotheses

| Slug | Statement (truncated) | Conviction | Timeframe | Created |
|------|----------------------|------------|-----------|---------|
| ...  | ...                  | ...        | ...       | ...     |
```

If no hypotheses exist, display:
```
No hypotheses found. Create one with: /hypothesis <slug>
Requires an existing mandate — create one first with: /mandate <slug>
```

---

### Slug provided (e.g., "momentum-rotation") --> Create or view hypothesis

**Step 1: Check lifecycle prerequisites**

```bash
cat E:/llm-quant/data/strategies/$SLUG/mandate.yaml 2>/dev/null
```

If no mandate exists, STOP and display:
```
ERROR: No mandate found for strategy "{slug}".
The lifecycle requires: Mandate --> Hypothesis --> Data Contract --> ...
Create a mandate first: /mandate {slug}
```

**Step 2: Check if hypothesis already exists**

```bash
cat E:/llm-quant/data/strategies/$SLUG/hypothesis.yaml 2>/dev/null
```

If it exists, display it. Ask if the user wants to revise it (warn that revision invalidates downstream artifacts).

**Step 3: Construct the hypothesis**

Guide the user through Peterson's framework. Ask targeted questions:

1. **What do you predict?** (The declarative statement)
2. **Why do you predict it?** (The economic or statistical rationale)
3. **How will you measure it?** (The specific metric and threshold)
4. **What would falsify it?** (The null hypothesis)
5. **Over what timeframe?** (When should we expect to see the effect)

**Schema:**

```yaml
# Strategy Hypothesis: {slug}
strategy_slug: "{slug}"

statement: |
  "I expect [specific measurable outcome] because [economic/statistical rationale],
  which I will measure by [specific metric and threshold]."

expected_outcome:
  metric: "Sharpe ratio"          # or win_rate, annualized_return, max_drawdown, etc.
  direction: "above"              # above | below | between
  threshold: 0.80                 # the specific value
  comparison: "benchmark"         # benchmark | absolute | historical
  description: "Strategy Sharpe > 0.80, outperforming 60/40 benchmark Sharpe"

measurement_method:
  primary_metric: "Sharpe ratio over rolling 252-day window"
  secondary_metrics:
    - "Sortino ratio > 1.0"
    - "Max drawdown < 15%"
  evaluation_frequency: "monthly"
  minimum_sample: 50              # minimum trades before evaluation is meaningful

null_hypothesis: |
  "The strategy has no edge over the benchmark. Any observed outperformance is due to
  random chance, data mining, or favorable regime coincidence."

falsification_criteria:
  - "OOS Sharpe < 0.40 (half of target, sustained over 63 trading days)"
  - "DSR < 0.95 (deflated Sharpe indicates multiple-testing artifact)"
  - "PBO > 0.10 (high probability of backtest overfitting)"
  - "Strategy underperforms benchmark in >60% of rolling windows"

timeframe:
  backtest_period: "2020-01-01 to 2025-12-31"  # minimum 5 years
  evaluation_horizon: "63 trading days (one quarter) for initial signal"
  full_evaluation: "252 trading days (one year) for robust conclusion"

conviction: "medium"  # low | medium | high
conviction_rationale: "Why this level — what evidence supports or weakens it"

economic_rationale: |
  "The economic reasoning for why this edge should exist and persist.
  Momentum exists because of behavioral biases (anchoring, herding) and
  institutional constraints (forced selling, mandate-driven rebalancing).
  These structural features are unlikely to be arbitraged away."

risks:
  - "Regime sensitivity: momentum crashes in regime transitions"
  - "Crowding: popular momentum strategies may face capacity constraints"
  - "Parameter sensitivity: SMA lookback windows may need to vary by regime"

created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
```

**Step 4: Write the hypothesis**

Write the completed YAML to `data/strategies/{slug}/hypothesis.yaml`.

**Step 5: Confirm**

Display the written hypothesis and confirm:
```
Hypothesis created: data/strategies/{slug}/hypothesis.yaml
Next step: /data-contract {slug}
```

---

## What Makes a Good Hypothesis

**Good:**
- "I expect a long-only sector rotation strategy selecting the top 3 sectors by 63-day momentum to achieve a Sharpe ratio > 0.80 and max drawdown < 12%, because sector momentum is driven by institutional rebalancing flows that persist over 1-3 month horizons, measured by CPCV walk-forward OOS performance across 15 test combinations."

**Bad:**
- "Momentum works" (not testable)
- "Buy low, sell high" (not specific)
- "This SMA crossover backtest had a 1.5 Sharpe" (HARKing — hypothesizing after results are known)

---

## Anti-Snooping Discipline

The hypothesis MUST be written BEFORE looking at backtest results. If you have already seen the results and are writing a hypothesis to match them, you are engaging in HARKing (Hypothesizing After Results are Known). This invalidates the entire statistical testing framework.

Signs of HARKing:
- Hypothesis parameters exactly match the best backtest parameters
- Hypothesis written on the same day as the first backtest
- Hypothesis contains suspiciously precise thresholds (e.g., "21-day SMA" instead of "short-term SMA")
- No null hypothesis or falsification criteria

---

## Lifecycle Position

```
Mandate --> [Hypothesis] --> Data Contract --> Research Spec --> Backtest --> Robustness --> Paper --> Promotion
```

A hypothesis requires an existing mandate and must exist before a data contract can be created.

---

## Important

- Do NOT allow a hypothesis without a null hypothesis — the null is what you are testing against
- Do NOT allow a hypothesis without falsification criteria — unfalsifiable conjectures are not science
- Do NOT allow conviction: "high" without substantial evidence — overconfidence is a risk factor
- The hypothesis is a contract with yourself: you are committing to what you expect before you look at the data
- If the user wants to change the hypothesis after backtesting has begun, this is a RED FLAG for HARKing. Require a new slug or explicit documentation of why the change is legitimate.

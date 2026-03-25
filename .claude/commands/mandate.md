---
description: "Create or view a strategy mandate — defines objective, benchmark, universe, constraints"
---

# /mandate — Strategy Mandate

You are the portfolio manager. A mandate is the foundational artifact for any strategy — it defines what you are trying to achieve, what you are measuring against, and what constraints bind you. No downstream work (hypothesis, data contract, research spec, backtest) can proceed without a mandate.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> List existing mandates

Scan `data/strategies/` for all subdirectories containing a `mandate.yaml` file.

```bash
cd E:/llm-quant && find data/strategies -name "mandate.yaml" -type f 2>/dev/null
```

For each mandate found, read the YAML and display a summary table:

```
## Strategy Mandates

| Slug | Name | Objective | Benchmark | Status | Created |
|------|------|-----------|-----------|--------|---------|
| ...  | ...  | ...       | ...       | ...    | ...     |
```

If no mandates exist, display:
```
No mandates found. Create one with: /mandate <slug>
Example: /mandate momentum-rotation
```

---

### Slug provided (e.g., "momentum-rotation") --> Create or update mandate

**Step 1: Ensure directory exists**

```bash
mkdir -p E:/llm-quant/data/strategies/$SLUG
```

**Step 2: Check if mandate already exists**

```bash
cat E:/llm-quant/data/strategies/$SLUG/mandate.yaml 2>/dev/null
```

If it exists, display the current mandate and ask: "Mandate already exists. Do you want to update it? (This will record a changelog entry.)"

**Step 3: Gather mandate information**

Walk the user through each section. Use sensible defaults from the project context (CLAUDE.md) but let the user override. Ask focused questions — do not dump a blank form.

**Required fields:**

```yaml
# Strategy Mandate: {slug}
name: "Human-readable strategy name"
slug: "{slug}"
status: "draft"  # draft | active | suspended | retired

objective: "What this strategy optimizes for (e.g., maximize risk-adjusted return via sector momentum rotation)"

benchmark:
  name: "60/40 SPY/TLT"
  symbols:
    SPY: 0.60
    TLT: 0.40
  rebalance_frequency: "monthly"
  return_type: "total_return"  # MUST be total_return — price return ignores dividends and coupons
  notes: "TLT has ~17yr effective duration; benchmark has significant rate sensitivity"

universe:
  symbols:
    - SPY
    - QQQ
    # ... subset of the 39-asset universe from config/universe.toml
  selection_rationale: "Why these symbols and not others"

constraints:
  max_drawdown: 0.15           # Hard limit from CLAUDE.md
  max_position_weight: 0.10    # Per-position cap (0.05 for crypto, 0.08 for forex)
  max_gross_exposure: 2.0      # 200% of NAV
  max_net_exposure: 1.0        # 100% of NAV
  max_sector_concentration: 0.30
  min_cash_reserve: 0.05
  max_trades_per_session: 5
  stop_loss_required: true

target_metrics:
  sharpe: 0.8
  sortino: 1.0
  calmar: 0.5
  annualized_return_min: 0.08
  annualized_return_max: 0.15

created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
```

**Step 4: Write the mandate**

Write the completed YAML to `data/strategies/{slug}/mandate.yaml`.

**Step 5: Confirm**

Display the written mandate and confirm:
```
Mandate created: data/strategies/{slug}/mandate.yaml
Next step: /hypothesis {slug}
```

---

## Benchmark Requirements

The benchmark is not decorative — it is the null hypothesis. The strategy must demonstrate value added over the benchmark, not just positive returns.

- `return_type` MUST be `total_return`. Price return ignores dividends (SPY ~1.3% yield) and coupon income (TLT ~3.5% yield). Comparing strategy total returns against benchmark price returns is a form of flattering the strategy.
- TLT has approximately 17-year effective duration. A 100bp rate move produces roughly a 17% price change. The benchmark has significant interest rate sensitivity — this must be understood and documented.
- The benchmark drives what "good performance" means. A strategy that underperforms 60/40 in every regime has no edge regardless of its absolute return.
- Benchmark choice comes from the mandate, not from CLAUDE.md defaults. 60/40 SPY/TLT is the default for this project, but a sector rotation strategy might benchmark against equal-weight sector ETFs, and a fixed income strategy against an aggregate bond index.

---

## Lifecycle Position

```
[Mandate] --> Hypothesis --> Data Contract --> Research Spec --> Backtest --> Robustness --> Paper --> Promotion
```

The mandate is the first artifact. It must exist before any other artifact can be created for this strategy slug.

---

## Important

- Do NOT skip the benchmark section — it is the most commonly omitted and most important part
- Do NOT allow a mandate without constraints — constraints are what prevent ruin
- The mandate defines the objective function; everything downstream serves this objective
- If the user wants to change a mandate after downstream artifacts exist, warn them that this invalidates all downstream artifacts (hypothesis, data contract, research spec, backtests)
- Record mandate creation/updates in `strategy_changelog` if the table exists

---
description: "Create or view a data contract — defines data requirements and quality grade"
---

# /data-contract — Data Contract

You are the portfolio manager. A data contract specifies exactly what data the strategy requires, its quality characteristics, and known limitations. This is not a formality — data quality determines the credibility of every downstream result. A backtest on grade-C data cannot support a promotion decision.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> List existing data contracts

Scan `data/strategies/` for all subdirectories containing a `data-contract.yaml` file.

```bash
cd E:/llm-quant && find data/strategies -name "data-contract.yaml" -type f 2>/dev/null
```

For each data contract found, display a summary table:

```
## Data Contracts

| Slug | Symbols | Date Range | Frequency | Quality Grade | Created |
|------|---------|------------|-----------|---------------|---------|
| ...  | ...     | ...        | ...       | ...           | ...     |
```

If no data contracts exist, display:
```
No data contracts found. Create one with: /data-contract <slug>
Requires an existing hypothesis — create one first with: /hypothesis <slug>
```

---

### Slug provided (e.g., "momentum-rotation") --> Create or view data contract

**Step 1: Check lifecycle prerequisites**

```bash
cat E:/llm-quant/data/strategies/$SLUG/hypothesis.yaml 2>/dev/null
```

If no hypothesis exists, STOP and display:
```
ERROR: No hypothesis found for strategy "{slug}".
The lifecycle requires: Mandate --> Hypothesis --> Data Contract --> ...
Create a hypothesis first: /hypothesis {slug}
```

Also verify the mandate exists:
```bash
cat E:/llm-quant/data/strategies/$SLUG/mandate.yaml 2>/dev/null
```

**Step 2: Check if data contract already exists**

```bash
cat E:/llm-quant/data/strategies/$SLUG/data-contract.yaml 2>/dev/null
```

If it exists, display it. Ask if the user wants to update it (warn that updates invalidate downstream artifacts if the research spec is already frozen).

**Step 3: Build the data contract from the mandate and hypothesis**

Read the mandate to get the universe symbols. Read the hypothesis to understand what data is needed. Then construct the contract:

**Schema:**

```yaml
# Data Contract: {slug}
strategy_slug: "{slug}"

# What symbols does the strategy trade?
symbols:
  tradeable:
    - SPY
    - QQQ
    # ... from mandate universe
  reference:
    - VIX      # Used for regime classification, not traded
    - "^TNX"   # 10Y yield, for yield curve slope

# Date range for historical data
date_range:
  start: "2018-01-01"     # At least 5 years for robust testing
  end: "2025-12-31"       # Or "latest" for live
  rationale: "5+ years covers multiple regimes including COVID crash, 2022 rate hiking, 2023-24 recovery"

# Data frequency
frequency: "daily"         # daily | weekly | monthly
after_hours: false         # Whether to include after-hours data

# Required fields per symbol per date
required_fields:
  - open
  - high
  - low
  - close
  - adj_close               # CRITICAL for total return calculations
  - volume

# Data quality assessment
quality_grade: "b"          # a | b | c | d (see grading rubric below)
quality_rationale: |
  "Yahoo Finance adjusted close prices. Corporate actions adjusted, but:
  - No bid/ask spreads (no microstructure analysis possible)
  - No intraday data (daily bars only)
  - ETF NAV may differ from market close price
  - Crypto trades 24/7 but daily bar alignment varies"

# Known data issues — be honest about these
known_issues:
  survivorship_bias:
    present: false
    notes: "ETF universe — no delisted securities. But ETFs that were closed/merged are excluded."
  look_ahead_bias:
    present: false
    notes: "All indicators use only causal (backward-looking) operations. Verified in data/indicators.py."
  coverage_gaps:
    present: true
    notes: "Some crypto symbols have shorter history (SOL-USD from ~2020). VGK/EWJ may have gaps on local holidays."
  corporate_actions:
    present: false
    notes: "Using adj_close from yfinance which adjusts for splits and dividends."
  data_source_changes:
    present: true
    notes: "Yahoo Finance occasionally changes data format or availability without notice."

# Data source
data_source:
  provider: "Yahoo Finance (via yfinance)"
  method: "yfinance Python library"
  update_frequency: "daily during market hours"
  reliability: "Generally reliable for daily bars. Occasional outages. No SLA."

# Freshness requirement for live trading
freshness_requirement:
  max_staleness_hours: 24     # For daily strategy
  weekend_exception: true     # Friday data is current on weekends
  alert_threshold_hours: 48   # Warn if older than this
  halt_threshold_hours: 72    # Kill switch: no fresh data > 72h

# Benchmark data (from mandate)
benchmark_data:
  symbols:
    - SPY
    - TLT
  return_type: "total_return"   # Uses adj_close for dividend/coupon reinvestment
  rebalance_frequency: "monthly"

created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
```

**Step 4: Validate data availability**

Run a quick check to see if the required data is available:

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import duckdb

db = duckdb.connect('data/llm_quant.duckdb', read_only=True)

# Check which symbols have data
symbols = db.execute('''
    SELECT symbol, MIN(date) as first_date, MAX(date) as last_date, COUNT(*) as rows
    FROM market_data
    GROUP BY symbol
    ORDER BY symbol
''').fetchall()

print('Symbol coverage in DuckDB:')
for sym, first, last, rows in symbols:
    print(f'  {sym}: {first} to {last} ({rows} rows)')

db.close()
"
```

Report any gaps between the contract requirements and actual data availability.

**Step 5: Write the data contract**

Write the completed YAML to `data/strategies/{slug}/data-contract.yaml`.

**Step 6: Confirm**

Display the written data contract and confirm:
```
Data contract created: data/strategies/{slug}/data-contract.yaml
Quality grade: {grade}
Next step: /research-spec {slug}
```

---

## Quality Grading Rubric

| Grade | Label | Description | Promotion Eligible |
|-------|-------|-------------|-------------------|
| **A** | Institutional | Tick-level or consolidated OHLCV from institutional data vendors (Bloomberg, Refinitiv). Bid/ask spreads available. Corporate actions verified. No gaps. | Yes |
| **B** | Adjusted Retail | Daily adjusted OHLCV from retail data sources (yfinance adj_close). Corporate actions handled by provider. Known gaps documented. | Yes |
| **C** | Unadjusted Retail | Raw OHLCV without adjustment for splits/dividends. Or adjusted data with undocumented gaps. | No — must upgrade to B |
| **D** | Incomplete | Missing symbols, large date gaps, unadjusted prices with known corporate actions, or data from unreliable sources. | No — must upgrade to B |

**Minimum for promotion: Grade B.** A strategy cannot be promoted to paper trading or beyond without at least grade-B data. The quality grade flows through to the backtest artifact and robustness analysis.

---

## Lifecycle Position

```
Mandate --> Hypothesis --> [Data Contract] --> Research Spec --> Backtest --> Robustness --> Paper --> Promotion
```

A data contract requires an existing hypothesis and must exist before a research spec can be created.

---

## Important

- Do NOT let a data contract pass without explicitly documenting known issues — silence about data problems is a form of data snooping
- `adj_close` is mandatory for any strategy using total return calculations — price return analysis with ETFs that pay dividends or coupons is systematically biased
- The data contract is a commitment to what data you have and what you know is wrong with it — honesty here prevents garbage-in-garbage-out downstream
- If the user is using crypto symbols, note that crypto trades 24/7 but yfinance daily bars use a specific cutoff time — document this
- If benchmark symbols are not in the data contract, add them — you cannot compute benchmark-relative performance without benchmark data

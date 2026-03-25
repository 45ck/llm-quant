---
description: "Create or freeze a research spec — frozen design document before backtesting"
---

# /research-spec — Research Specification

You are the portfolio manager. The research spec is the most critical methodological artifact in the lifecycle. It defines the complete strategy design — parameters, indicators, signals, rules, validation method, and cost model — and it MUST be frozen before any backtesting begins.

**Why freeze?** If the spec and the backtest are produced simultaneously, you will unconsciously adjust the design to fit the data. This is data snooping. A frozen spec creates a bright line between design (where you think) and evaluation (where you measure). Peterson calls this the separation of hypothesis from experiment.

## Parse the user's argument: "$ARGUMENTS"

---

### No arguments --> List existing research specs

Scan `data/strategies/` for all subdirectories containing a `research-spec.yaml` file.

```bash
cd E:/llm-quant && find data/strategies -name "research-spec.yaml" -type f 2>/dev/null
```

For each spec found, display a summary table:

```
## Research Specs

| Slug | Strategy Type | Parameters | Frozen | Frozen Hash | Created |
|------|--------------|------------|--------|-------------|---------|
| ...  | ...          | ...        | ...    | ...         | ...     |
```

---

### "freeze {slug}" --> Freeze a research spec

**Step 1: Check the spec exists and is not already frozen**

```bash
cat E:/llm-quant/data/strategies/$SLUG/research-spec.yaml 2>/dev/null
```

If already frozen (`frozen: true`), display:
```
ERROR: Research spec for "{slug}" is already frozen (hash: {hash}).
A frozen spec cannot be modified. To test a different design, create a new strategy slug.
```

**Step 2: Validate completeness**

Before freezing, verify ALL required sections are present and non-empty:
- strategy_type
- parameters (at least 1)
- indicators (at least 1)
- signals (at least 1)
- rules.entry (at least 1 rule)
- rules.exit (at least 1 rule)
- rules.risk (at least 1 rule)
- validation.primary_method
- cost_model (all fields populated)

If any section is missing or empty, list the gaps and refuse to freeze.

**Step 3: Compute the hash**

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
import hashlib, yaml, json

with open('data/strategies/$SLUG/research-spec.yaml', 'r') as f:
    spec = yaml.safe_load(f)

# Remove metadata fields before hashing
for key in ['frozen', 'frozen_at', 'frozen_hash', 'created_at', 'updated_at']:
    spec.pop(key, None)

# Canonical JSON for deterministic hashing
canonical = json.dumps(spec, sort_keys=True, default=str)
spec_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
print(f'SHA256 (truncated): {spec_hash}')
"
```

**Step 4: Update the spec with freeze metadata**

Set `frozen: true`, `frozen_at: YYYY-MM-DD`, `frozen_hash: <hash>` in the YAML file.

**Step 5: Confirm**

```
Research spec FROZEN: data/strategies/{slug}/research-spec.yaml
Hash: {hash}
Frozen at: {date}

This spec can no longer be modified. All backtests will reference this hash.
To test a different design, create a new strategy slug.

Next step: /backtest {slug}
```

---

### Slug provided (e.g., "momentum-rotation") --> Create or view research spec

**Step 1: Check lifecycle prerequisites**

```bash
cat E:/llm-quant/data/strategies/$SLUG/data-contract.yaml 2>/dev/null
```

If no data contract exists, STOP and display:
```
ERROR: No data contract found for strategy "{slug}".
The lifecycle requires: Mandate --> Hypothesis --> Data Contract --> Research Spec --> ...
Create a data contract first: /data-contract {slug}
```

**Step 2: Check if spec already exists**

```bash
cat E:/llm-quant/data/strategies/$SLUG/research-spec.yaml 2>/dev/null
```

If it exists and is frozen, display it as read-only. If it exists and is not frozen, display it and offer to edit or freeze.

**Step 3: Build the research spec**

Read the mandate, hypothesis, and data contract to inform the design. Guide the user through each section.

**Schema:**

```yaml
# Research Spec: {slug}
strategy_slug: "{slug}"

# Strategy classification
strategy_type: "momentum"   # momentum | mean_reversion | hybrid | trend_following | stat_arb | macro

# Free parameters — every tunable number must be listed here
# Anti-overfitting: fewer parameters = less overfitting risk. Target < 10 free params.
parameters:
  sma_fast: 20               # Fast SMA lookback (days)
  sma_slow: 50               # Slow SMA lookback (days)
  sma_trend: 200             # Trend filter lookback (days)
  rsi_period: 14             # RSI lookback (days)
  rsi_overbought: 70         # RSI overbought threshold
  rsi_oversold: 30           # RSI oversold threshold
  macd_fast: 12              # MACD fast EMA period
  macd_slow: 26              # MACD slow EMA period
  macd_signal: 9             # MACD signal line period
  atr_period: 14             # ATR lookback for vol sizing
  momentum_lookback: 63      # Momentum ranking window (days)
  top_n_sectors: 3           # Number of sectors to overweight
  rebalance_trigger_pct: 0.05  # Rebalance when drift exceeds this

# Indicators — derived from raw data, purely descriptive
# Each indicator must use only causal (backward-looking) operations
indicators:
  - name: "SMA_20"
    formula: "rolling_mean(close, 20)"
    lookback: 20
    causal: true
  - name: "SMA_50"
    formula: "rolling_mean(close, 50)"
    lookback: 50
    causal: true
  - name: "SMA_200"
    formula: "rolling_mean(close, 200)"
    lookback: 200
    causal: true
  - name: "RSI_14"
    formula: "standard RSI with 14-day lookback"
    lookback: 14
    causal: true
  - name: "MACD"
    formula: "EMA(12) - EMA(26), signal = EMA(9) of MACD"
    lookback: 35  # 26 + 9 effective warmup
    causal: true
  - name: "ATR_14"
    formula: "rolling_mean(true_range, 14)"
    lookback: 14
    causal: true

# Signals — interactions between indicators that produce directional predictions
signals:
  - name: "trend_signal"
    description: "SMA(20) crosses above SMA(50) while price > SMA(200)"
    type: "entry_long"
    indicators_used: ["SMA_20", "SMA_50", "SMA_200"]
  - name: "mean_reversion_signal"
    description: "RSI < 30 while price > SMA(200) (oversold in uptrend)"
    type: "entry_long"
    indicators_used: ["RSI_14", "SMA_200"]
  - name: "exit_signal"
    description: "SMA(20) crosses below SMA(50) or RSI > 70"
    type: "exit"
    indicators_used: ["SMA_20", "SMA_50", "RSI_14"]

# Rules — path-dependent decisions
rules:
  entry:
    - "Enter long when trend_signal fires AND regime is risk_on or transition"
    - "Enter long on mean_reversion_signal only when regime is risk_on"
    - "Position size = target_weight * (target_vol / realized_ATR_vol)"
  exit:
    - "Exit when exit_signal fires"
    - "Exit when stop_loss hit (ATR-based trailing stop)"
    - "Exit when regime changes to risk_off (reduce to 50%, full exit if sustained 5 days)"
  risk:
    - "Max 2% NAV per trade"
    - "Max 10% NAV per position (5% crypto, 8% forex)"
    - "Stop-loss required: initial stop at entry_price - 2*ATR"
    - "Trailing stop: moves up by 1*ATR when price advances by 2*ATR"
  rebalancing:
    - "Rebalance when any position drifts > 5% from target weight"
    - "Full re-ranking and rotation on rebalance_frequency_days schedule"
    - "Reduce positions proportionally when approaching exposure limits"

# Validation methodology — how the backtest will be structured
validation:
  primary_method: "cpcv"        # Combinatorially Purged Cross-Validation
  cpcv_groups: 6                # N groups to split the data into
  cpcv_test_groups: 2           # k groups held out for testing per combination
  # C(6,2) = 15 combinations, each using 4/6 for train, 2/6 for test
  purge_days: 5                 # Days purged between train/test to prevent leakage
  embargo_pct: 0.01             # Fraction of test data embargoed at boundaries
  holdout_reserved: true        # Reserve final 20% as true holdout (never touched during dev)
  walk_forward_secondary: true  # Also run walk-forward as secondary validation
  wfo_window_days: 252          # Walk-forward training window (1 year)
  wfo_step_days: 63             # Walk-forward step size (1 quarter)

# Cost model — assumptions for realistic P&L
cost_model:
  spread_bps: 5                 # Bid-ask spread cost in basis points (each way)
  slippage_volatility_factor: 0.1  # Slippage = factor * daily_vol * trade_size
  flat_slippage_bps: 2          # Minimum slippage floor in basis points
  commission_per_share: 0.005   # Commission per share (e.g., IBKR tiered)
  # Total cost per round-trip: ~14-20 bps for liquid ETFs

# Execution assumptions
fill_delay: 1                   # Bars of delay between signal and fill (1 = next bar)
warmup_days: 200                # Days of data needed before first signal (longest indicator lookback)

# Rebalance frequency
rebalance_frequency_days: 21    # Check for rebalancing every N trading days

# Freeze metadata — set by /research-spec freeze
frozen: false
frozen_at: null
frozen_hash: null

created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
```

**Step 4: Write the research spec**

Write the completed YAML to `data/strategies/{slug}/research-spec.yaml`.

**Step 5: Confirm**

```
Research spec created: data/strategies/{slug}/research-spec.yaml
Status: DRAFT (not frozen)

IMPORTANT: You must freeze this spec before backtesting.
Run: /research-spec freeze {slug}

Review the spec carefully before freezing — once frozen, it cannot be changed.
```

---

## Parameter Counting and Overfitting Risk

Count the free parameters in the spec. This is a critical anti-overfitting metric.

| Free Parameters | Risk Level | Guidance |
|----------------|------------|----------|
| 1-5 | Low | Simple strategies. Good parsimony. |
| 6-10 | Moderate | Acceptable if each parameter has economic justification. |
| 11-15 | High | Every parameter must be individually justified. Consider simplifying. |
| >15 | Very High | Almost certainly overfit. Reject unless extraordinary justification. |

Display the parameter count and risk level when creating or viewing a spec.

---

## Validation Configuration: CPCV

Combinatorially Purged Cross-Validation (CPCV) is the primary validation method because it:
1. Generates many train/test splits from limited data (C(N,k) combinations)
2. Purges data between train and test to prevent information leakage
3. Embargoes test boundaries to handle autocorrelation
4. Produces a distribution of OOS performance, not a single estimate

With N=6, k=2: C(6,2) = 15 combinations. Each uses 4 groups for training and 2 for testing. This provides 15 independent OOS Sharpe estimates. The distribution of these estimates is more informative than any single backtest.

---

## Cost Model Notes

The cost model must be specified BEFORE backtesting. Strategies that look profitable before costs but fail after costs are not real strategies.

- `spread_bps: 5` is conservative for liquid ETFs (SPY spread is ~1bp, less liquid ETFs can be 10-20bp)
- `slippage_volatility_factor: 0.1` models market impact as proportional to volatility
- `fill_delay: 1` means signals on bar T are filled on bar T+1. This prevents look-ahead bias in execution.
- The robustness suite will test cost sensitivity at 1x, 1.5x, 2x, and 3x multipliers

---

## Lifecycle Position

```
Mandate --> Hypothesis --> Data Contract --> [Research Spec] --> Backtest --> Robustness --> Paper --> Promotion
```

A research spec requires an existing data contract. It must be frozen before any backtest can run.

---

## Important

- NEVER allow backtesting on an unfrozen spec — this is the single most important methodological control
- The hash provides tamper evidence — if anyone modifies a frozen spec, the hash will not match
- If the user wants to test a different parameter set, they must create a new strategy slug or explicitly document the parameter change as a new trial (incrementing the trial counter for DSR purposes)
- Every parameter must have a rationale — "I picked 20 because the backtest was best at 20" is overfitting, not rationale
- The cost model is NOT optional — a zero-cost backtest is fiction

# Quantitative Strategy Lifecycle Reference

This document defines the complete lifecycle for developing, validating, deploying, and monitoring quantitative trading strategies in the llm-quant portfolio management system. Every strategy follows the same state machine. There are no shortcuts.

---

## Table of Contents

1. [Lifecycle State Machine](#lifecycle-state-machine)
2. [State Definitions and Required Artifacts](#state-definitions-and-required-artifacts)
3. [State Transitions](#state-transitions)
4. [Artifact Schemas](#artifact-schemas)
5. [Robustness Gate](#robustness-gate)
6. [Promotion Gate](#promotion-gate)
7. [Deflated Sharpe Ratio (DSR)](#deflated-sharpe-ratio-dsr)
8. [Probability of Backtest Overfitting (PBO)](#probability-of-backtest-overfitting-pbo)
9. [Combinatorially Purged Cross-Validation (CPCV)](#combinatorially-purged-cross-validation-cpcv)
10. [Cost Model](#cost-model)
11. [Benchmark Requirements](#benchmark-requirements)
12. [Experiment Registry](#experiment-registry)
13. [Commands Reference](#commands-reference)

---

## Lifecycle State Machine

```
                                    +---------------------------+
                                    |                           |
                                    v                           |
Idea --> Mandate --> Hypothesis --> Data Contract --> Research Spec (frozen)
                                                         |
                                                         v
                              Promotion <-- Paper <-- Robustness <-- Backtest
                                  |
                                  v
                             Evaluation (ongoing)
                                  |
                                  v
                             Retirement
```

The lifecycle is strictly sequential. A failure at any gate returns the strategy to the appropriate earlier stage. There is no way to skip a stage.

### State Summary

| # | State | Artifact | Command | Gate Required |
|---|-------|----------|---------|---------------|
| 1 | Idea | (none -- conversation only) | -- | -- |
| 2 | Mandate | `mandate.yaml` | `/mandate` | -- |
| 3 | Hypothesis | `hypothesis.yaml` | `/hypothesis` | Mandate exists |
| 4 | Data Contract | `data-contract.yaml` | `/data-contract` | Hypothesis exists |
| 5 | Research Spec | `research-spec.yaml` | `/research-spec` | Data contract exists |
| 6 | Backtest | `experiment-registry.jsonl` entries | `/backtest` | Spec frozen |
| 7 | Robustness | `robustness.yaml` | `/robustness` | >= 2 experiments |
| 8 | Paper Trading | `paper-trading.yaml` | `/paper` | Robustness gate passed |
| 9 | Promotion | `strategy_changelog` entry | `/promote` | Paper gate passed |

Post-promotion:

| Stage | Artifact | Command |
|-------|----------|---------|
| Evaluation | `evaluation-{date}.yaml` | `/evaluate` |
| Retirement | `strategy_changelog` entry | `/evaluate` (triggers retirement) |

---

## State Definitions and Required Artifacts

### 1. Idea

An idea is an informal observation about a potential market edge. It exists only in conversation. Ideas become mandates when they are formalized with an objective, benchmark, and constraints.

**No artifact produced.** Ideas that cannot be formalized into mandates are discarded.

### 2. Mandate

The mandate defines what the strategy is trying to achieve, what it measures against, and what constraints bind it. It is the foundational document -- all downstream artifacts serve the mandate's objective.

**Artifact:** `data/strategies/{slug}/mandate.yaml`

**Required fields:**
- `name`: Human-readable strategy name
- `slug`: URL-safe identifier
- `track`: `track_a` (Defensive Alpha) or `track_b` (Aggressive Alpha) — determines gate thresholds throughout lifecycle. See `docs/governance/research-tracks.md`.
- `objective`: What the strategy optimizes for
- `benchmark`: Name, symbol weights, rebalance frequency, return type (must be `total_return`)
- `universe`: List of tradeable symbols with selection rationale
- `constraints`: max_drawdown, max_position_weight, max_gross_exposure, max_net_exposure, max_sector_concentration, min_cash_reserve, max_trades_per_session, stop_loss_required
- `target_metrics`: Sharpe, Sortino, Calmar, return range
- `status`: draft | active | suspended | retired

### 3. Hypothesis

A testable conjecture following Peterson's framework: "I expect X because of Y, which I will measure by Z." The hypothesis must be written BEFORE looking at backtest results.

**Artifact:** `data/strategies/{slug}/hypothesis.yaml`

**Required fields:**
- `statement`: Declarative prediction
- `expected_outcome`: Metric, direction, threshold, comparison basis
- `measurement_method`: Primary metric, secondary metrics, minimum sample
- `null_hypothesis`: What you are testing against (typically "no edge over benchmark")
- `falsification_criteria`: Specific conditions that would reject the hypothesis
- `timeframe`: Backtest period, evaluation horizon
- `conviction`: low | medium | high with rationale
- `economic_rationale`: Why this edge should exist and persist
- `risks`: Known risk factors

### 4. Data Contract

Specifies exactly what data the strategy requires, its quality characteristics, and known limitations.

**Artifact:** `data/strategies/{slug}/data-contract.yaml`

**Required fields:**
- `symbols`: Tradeable and reference symbols
- `date_range`: Start, end, rationale
- `frequency`: daily | weekly | monthly
- `required_fields`: open, high, low, close, adj_close, volume
- `quality_grade`: a | b | c | d (minimum b for promotion)
- `known_issues`: Survivorship bias, look-ahead bias, coverage gaps, corporate actions, data source changes
- `data_source`: Provider, method, update frequency, reliability
- `freshness_requirement`: Max staleness, alert threshold, halt threshold
- `benchmark_data`: Symbols, return type, rebalance frequency

### 5. Research Spec

The complete strategy design document. Defines parameters, indicators, signals, rules, validation methodology, and cost model. MUST be frozen before backtesting.

**Artifact:** `data/strategies/{slug}/research-spec.yaml`

**Required fields:**
- `strategy_type`: momentum | mean_reversion | hybrid | trend_following | stat_arb | macro
- `parameters`: All free parameters with values (target < 10)
- `indicators`: Name, formula, lookback, causal flag
- `signals`: Name, description, type, indicators used
- `rules`: Entry, exit, risk, rebalancing
- `validation`: Primary method (cpcv), CPCV configuration, walk-forward configuration
- `cost_model`: Spread, slippage, commission
- `fill_delay`: Bars of delay between signal and fill (minimum 1)
- `warmup_days`: Days before first signal
- `rebalance_frequency_days`: How often to rebalance
- `frozen`: Boolean (must be true before backtesting)
- `frozen_hash`: SHA-256 hash of the spec content

### 6. Backtest

An experiment that tests a frozen spec against historical data. Produces metrics and records to the experiment registry.

**Artifact:** Entries in `data/strategies/experiment-registry.jsonl`

**Required fields per entry:**
- `experiment_id`: Unique identifier
- `slug`: Strategy slug
- `spec_hash`: Hash of the frozen research spec
- `trial_number`: Sequential trial counter (for DSR)
- `date`: When the experiment was run
- `period_start`, `period_end`: Backtest date range
- `metrics`: annualized_return, sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor, total_trades, dsr
- `cost_sensitivity`: Results at 1x, 1.5x, 2x, 3x cost multipliers
- `benchmark_comparison`: Strategy vs benchmark Sharpe and return

### 7. Robustness

Subjects the strategy to PBO, CPCV, and systematic perturbation. All gate checks must pass.

**Artifact:** `data/strategies/{slug}/robustness.yaml`

**Required fields:**
- `pbo`: Method, sub-periods, combinations, PBO estimate, gate status
- `cpcv`: Groups, test groups, combinations, OOS Sharpe statistics, gate status
- `perturbation`: Parameters tested/stable, cost survival, signal delay survival, gate status
- `dsr`: Observed Sharpe, trial count, DSR value, gate status
- `overall_gate`: pass | fail

### 8. Paper Trading

Live paper trading validation with trade logging, incident tracking, and operational verification.

**Artifact:** `data/strategies/{slug}/paper-trading.yaml`

**Required fields:**
- `start_date`, `status`, `performance`, `trades`, `incidents`
- `slippage_drift_bps`: Estimated paper vs real execution gap
- `operations_tested`: Checklist of all operational systems
- `days_active`, `total_trades`

### 9. Promotion Decision

Formal promotion review following the 5-stage Model Promotion Policy.

**Artifact:** Entry in `strategy_changelog` table (DuckDB) and optionally in the paper trading artifact.

**Stages:**
1. Hard vetoes (DSR, PBO, SPA, MinTRL)
2. Weighted scorecard (composite >= 85)
3. Paper trading minimums (30 days, 50 trades, Sharpe >= 0.60)
4. Canary gate (10% allocation, 14 days, drawdown < 10%)
5. Full deployment (kill switches active, baseline recorded)

---

## State Transitions

### Forward Transitions

| From | To | Condition |
|------|----|-----------|
| Idea | Mandate | Formalize objective, benchmark, constraints |
| Mandate | Hypothesis | Mandate exists and is not retired |
| Hypothesis | Data Contract | Hypothesis exists |
| Data Contract | Research Spec | Data contract exists |
| Research Spec | Backtest | Spec is frozen (`frozen: true`) |
| Backtest | Robustness | >= 2 experiments completed |
| Robustness | Paper Trading | All robustness gates pass |
| Paper Trading | Promotion | All paper trading gates pass |
| Promotion | Deployment | All promotion stages pass |
| Deployment | Evaluation | Ongoing -- runs continuously |

### Backward Transitions (Failures)

| Event | Returns To | Action |
|-------|-----------|--------|
| Robustness gate fails | Research Spec (new slug) | Redesign strategy |
| Paper trading critical incident | Robustness or Research Spec | Investigate root cause |
| Canary gate fails (strategy issue) | Scorecard (Stage 2) | Re-score |
| Canary gate fails (operational issue) | Paper Trading (Stage 3) | Re-test |
| Evaluation: RETIRE recommendation | Retirement | Close all positions |
| Mandate change | Invalidates ALL downstream | Must re-do from Hypothesis |

### Terminal States

| State | Meaning |
|-------|---------|
| Retired | Strategy edge is gone or hypothesis falsified. Positions closed. |
| Suspended | Temporary halt due to regime/operational issues. May resume. |

---

## Artifact Schemas

All artifacts are stored as YAML files in `data/strategies/{slug}/`. The experiment registry is a JSONL file (one JSON object per line) at `data/strategies/experiment-registry.jsonl`. Evaluation artifacts include the date in the filename: `evaluation-{YYYY-MM-DD}.yaml`.

### Storage Layout

```
data/strategies/
  experiment-registry.jsonl          # Global, append-only
  momentum-rotation/
    mandate.yaml
    hypothesis.yaml
    data-contract.yaml
    research-spec.yaml
    robustness.yaml
    paper-trading.yaml
    evaluation-2026-03-25.yaml
    evaluation-2026-04-25.yaml
  mean-reversion-bonds/
    mandate.yaml
    hypothesis.yaml
    ...
```

### Hash Chain

The research spec hash creates a verifiable chain:
1. Research spec is frozen with a SHA-256 hash of its content
2. Every backtest experiment records the spec hash
3. The robustness artifact records the spec hash
4. Any modification to the frozen spec will produce a different hash, breaking the chain

---

## Robustness Gate

All checks must pass. There is no partial credit. Gate thresholds differ by track — see `docs/governance/research-tracks.md`.

| Check | Track A Threshold | Track B Threshold | Description |
|-------|------------------|------------------|-------------|
| DSR | >= 0.95 | >= 0.95 | Integrity gate — same on both tracks |
| PBO | <= 0.10 | <= 0.10 | Integrity gate — same on both tracks |
| 2x costs survive | Sharpe > 0 | Sharpe > 0 | Same on both tracks |
| CPCV mean OOS Sharpe | > 0 | > 0 | Integrity gate — same on both tracks |
| CPCV median OOS Sharpe | > 0 | > 0 | Same on both tracks |
| Sharpe | >= 0.80 | >= 1.00 | Risk gate — higher bar for Track B |
| Max Drawdown | < 15% | < 30% | Risk gate — relaxed for Track B |
| Parameter stability | > 50% | > 50% | Same on both tracks |

---

## Promotion Gate

See `docs/governance/model-promotion-policy.md` for the full 5-stage pipeline. Summary of minimum requirements:

| Stage | Gate Criteria |
|-------|---------------|
| Hard Vetoes | DSR >= 0.95, PBO <= 10%, SPA p <= 0.05, MinTRL >= 1 OOS period |
| Scorecard | Composite score >= 85 (weighted across 5 dimensions) |
| Paper Trading | >= 30 days, >= 50 trades, Sharpe >= 0.60, all operations tested |
| Canary | 14+ days at 10% allocation, drawdown < 10%, no kill switches |
| Deployment | Kill switches active, baseline metrics recorded, changelog updated |

---

## Deflated Sharpe Ratio (DSR)

### Purpose

The DSR adjusts the observed Sharpe ratio for the number of strategies tested. When you test many strategies and select the best, the expected maximum Sharpe ratio increases purely by chance. The DSR quantifies the probability that the observed Sharpe is genuine rather than a multiple-testing artifact.

### Formula

Following Bailey and Lopez de Prado (2014):

```
DSR = P[ SR* > 0 ]

where SR* is the Sharpe ratio adjusted for:
  SR_0 = Expected maximum Sharpe from N independent trials
       = (1 - gamma) * Phi_inv(1 - 1/N) + gamma * Phi_inv(1 - 1/(N*e))
       where gamma is the Euler-Mascheroni constant (~0.5772)
       and Phi_inv is the inverse standard normal CDF

  The observed SR must exceed SR_0 with high probability, accounting for:
  - Skewness of the return distribution (negative skew inflates SR variance)
  - Excess kurtosis (fat tails make SR estimates unreliable)
  - Track length T (more data = more reliable estimate)
```

### Inputs

| Input | Source | Description |
|-------|--------|-------------|
| Observed SR | Backtest | The Sharpe ratio from the experiment |
| N (trials) | Experiment registry | Total number of experiments for this slug |
| Skewness | Return distribution | Third moment of strategy returns |
| Kurtosis | Return distribution | Fourth moment of strategy returns |
| T (track length) | Backtest period | Number of return observations |

### Thresholds

| DSR Value | Interpretation |
|-----------|---------------|
| >= 0.95 | PASS -- 95%+ probability the Sharpe is real |
| 0.80 - 0.95 | Marginal -- possible but inconclusive |
| < 0.80 | FAIL -- high probability of multiple-testing artifact |

### Critical Implication

Every backtest run increments the trial counter N, which raises the DSR bar. This is intentional -- it penalizes parameter sweeps and encourages deliberate, hypothesis-driven experimentation. Running 100 parameter combinations and keeping the best is data mining, not research.

---

## Probability of Backtest Overfitting (PBO)

### Purpose

PBO estimates the probability that the backtest's best-performing strategy configuration would underperform a randomly selected configuration out-of-sample. A high PBO indicates that in-sample optimization found noise, not signal.

### Method: Combinatorial Symmetric Cross-Validation (CSCV)

1. **Partition** the full backtest period into S = 16 equal-length sub-periods
2. **Enumerate** all C(S, S/2) = C(16, 8) = 12,870 ways to split the sub-periods into two equal halves
3. **For each combination c:**
   a. Designate one half as in-sample (IS), the other as out-of-sample (OOS)
   b. Compute the performance of each strategy variant on both IS and OOS
   c. Identify the IS-best variant (the one you would have selected)
   d. Measure its OOS rank relative to all variants
   e. Record w_c = 1 if OOS rank is below median (IS-best underperforms OOS), else 0
4. **Compute** PBO = (1/C) * sum(w_c)

### Formula

```
PBO = (1 / C(S, S/2)) * sum_{c=1}^{C(S,S/2)} I[ rank_OOS(best_IS) < S/2 ]

where:
  S = 16 (number of sub-periods)
  C(16, 8) = 12,870 (number of combinations)
  I[.] = indicator function
  rank_OOS(best_IS) = the OOS rank of the IS-best configuration
```

### Thresholds

| PBO Value | Interpretation |
|-----------|---------------|
| <= 0.10 | PASS -- low probability of overfitting |
| 0.10 - 0.25 | Marginal -- some overfitting risk |
| > 0.25 | FAIL -- significant overfitting detected |

### Interpretation

- PBO = 0.00: The IS-best configuration always performs well OOS. No overfitting detected.
- PBO = 0.50: The IS-best configuration is no better than random OOS. Complete overfitting.
- PBO = 1.00: The IS-best configuration always performs worst OOS. Extreme overfitting.

---

## Combinatorially Purged Cross-Validation (CPCV)

### Purpose

CPCV provides a robust estimate of out-of-sample strategy performance by generating many independent train/test splits while preventing information leakage through purging and embargo.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| N (groups) | 6 | Number of groups to split the data into |
| k (test groups) | 2 | Number of groups held out for testing per combination |
| Combinations | C(6,2) = 15 | Number of independent train/test splits |
| Purge gap | 5 days | Days removed between train and test sets to prevent leakage |
| Embargo | 1% | Fraction of test data embargoed at train/test boundaries |

### Method

1. **Split** the data into N = 6 chronologically ordered groups
2. **For each of C(N,k) = 15 combinations:**
   a. Hold out k = 2 groups for testing
   b. Use remaining N - k = 4 groups for training
   c. **Purge:** Remove `purge_days` observations at the end of each training group that borders a test group
   d. **Embargo:** Remove `embargo_pct` of test observations at the boundary with training data
   e. Train the strategy on the purged training set
   f. Evaluate on the embargoed test set
   g. Record the OOS Sharpe ratio
3. **Aggregate** the 15 OOS Sharpe ratios into a distribution

### Output Metrics

| Metric | Gate |
|--------|------|
| Mean OOS Sharpe | Must be > 0 |
| Median OOS Sharpe | Must be > 0 |
| Std of OOS Sharpe | Lower is better (more consistent) |
| Min OOS Sharpe | Informational (worst-case scenario) |
| Max OOS Sharpe | Informational (best-case scenario) |

### Why CPCV Over Simple Train/Test Split

A single train/test split gives one OOS estimate, which could be lucky or unlucky. CPCV provides 15 estimates, revealing the distribution of possible OOS outcomes. A strategy with a high mean but also a very negative minimum may be regime-dependent.

---

## Cost Model

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `spread_bps` | 5 | Bid-ask spread cost per side in basis points |
| `slippage_volatility_factor` | 0.1 | Slippage proportional to daily volatility * trade size |
| `flat_slippage_bps` | 2 | Minimum slippage floor in basis points |
| `commission_per_share` | 0.005 | Per-share commission (IBKR tiered pricing) |
| `fill_delay` | 1 bar | Delay between signal and execution |

### Total Cost Estimate

For a typical round-trip trade in a liquid ETF (e.g., SPY):
- Spread: ~2 * 5 = 10 bps
- Slippage: ~2 * 2 = 4 bps
- Commission: ~2 * 0.5 = 1 bp (at $500/share, 10 shares)
- **Total round-trip: ~15 bps**

For less liquid instruments or larger positions, costs scale up significantly.

### Cost Sensitivity Testing

The robustness suite tests at 4 cost levels:

| Multiplier | Description | Purpose |
|------------|-------------|---------|
| 1.0x | Base cost model | Normal operating assumption |
| 1.5x | Conservative | Moderate cost estimation error |
| 2.0x | Stress | Gate: strategy must survive at 2x costs |
| 3.0x | Extreme | Informational -- how sensitive is the strategy to costs? |

A strategy that dies at 1.5x costs has a thin edge that depends critically on cost assumptions being exactly right. This is fragile.

### Fill Delay

`fill_delay: 1` means signals generated on bar T are executed on bar T+1. This is mandatory for preventing look-ahead bias in execution. Setting `fill_delay: 0` (same-bar execution) is only valid for diagnostic purposes and must be explicitly documented as unrealistic.

---

## Benchmark Requirements

### Total Return vs Price Return

The benchmark MUST use total return (including dividends and coupons), not price return. This is specified via `return_type: "total_return"` in the mandate.

**Why this matters:**

| Component | SPY (S&P 500) | TLT (20+ Yr Treasury) |
|-----------|---------------|----------------------|
| Approximate annual yield | ~1.3% dividend | ~3.5% coupon |
| 5-year cumulative impact | ~6.5% | ~17.5% |

A strategy compared against price-return benchmarks starts with a 2-4% annual advantage that is not alpha -- it is just the benchmark not counting income. Over a 5-year backtest, this error compounds to 10-20%, which is enough to make a bad strategy look good.

### Default Benchmark: 60/40 SPY/TLT

| Component | Weight | Role |
|-----------|--------|------|
| SPY | 60% | Growth / equity exposure |
| TLT | 40% | Duration / rate sensitivity |

Rebalanced monthly. This is the passive multi-asset baseline that the strategy must beat on a risk-adjusted basis.

### TLT Duration Note

TLT has approximately 17-year effective duration. A 100-basis-point increase in long-term interest rates produces roughly a 17% price decline. The 60/40 benchmark therefore has significant interest rate sensitivity. Strategies that outperform during rate-cutting cycles may simply be expressing the same rate bet as TLT with leverage.

### Benchmark Override

The benchmark comes from the mandate, not from CLAUDE.md defaults. Strategies with different objectives may use different benchmarks:
- Sector rotation: Equal-weight sector ETFs
- Fixed income: AGG (aggregate bond index)
- Absolute return: Risk-free rate (SHY)
- Crypto: BTC-USD (if crypto-focused)

---

## Experiment Registry

### Format

The experiment registry is a JSONL file at `data/strategies/experiment-registry.jsonl`. Each line is a self-contained JSON object representing one backtest experiment.

### Rules

1. **Append-only**: Never delete or modify existing entries. The registry is an audit trail.
2. **Every run is recorded**: Including failures, poor results, and diagnostic runs.
3. **Spec hash links to spec**: Every entry references the frozen research spec hash.
4. **Trial counter is per-slug**: Different strategy slugs have independent trial counters.
5. **Never prune**: Bad results are evidence. Hiding them is data snooping.

### Schema

```json
{
  "experiment_id": "mom-rot-001",
  "slug": "momentum-rotation",
  "spec_hash": "a1b2c3d4e5f6g7h8",
  "trial_number": 1,
  "date": "2026-03-25",
  "period_start": "2020-01-01",
  "period_end": "2025-12-31",
  "symbols": ["SPY", "QQQ", "IWM"],
  "metrics": {
    "annualized_return": 0.12,
    "sharpe": 0.95,
    "sortino": 1.20,
    "calmar": 0.65,
    "max_drawdown": -0.11,
    "win_rate": 0.54,
    "profit_factor": 1.45,
    "total_trades": 187,
    "dsr": 0.97
  },
  "cost_sensitivity": {
    "1.0x": {"sharpe": 0.95, "survives": true},
    "1.5x": {"sharpe": 0.72, "survives": true},
    "2.0x": {"sharpe": 0.48, "survives": true},
    "3.0x": {"sharpe": 0.15, "survives": true}
  },
  "benchmark_comparison": {
    "strategy_sharpe": 0.95,
    "benchmark_sharpe": 0.62,
    "excess_sharpe": 0.33
  },
  "status": "completed"
}
```

---

## Commands Reference

| Command | Description | Lifecycle Stage |
|---------|-------------|-----------------|
| `/mandate {slug}` | Create or view a strategy mandate | Mandate |
| `/hypothesis {slug}` | Create or view a testable hypothesis | Hypothesis |
| `/data-contract {slug}` | Create or view a data contract | Data Contract |
| `/research-spec {slug}` | Create a research spec | Research Spec |
| `/research-spec freeze {slug}` | Freeze the research spec (immutable) | Research Spec |
| `/backtest {slug}` | Run a backtest experiment | Backtest |
| `/robustness {slug}` | Run robustness analysis suite | Robustness |
| `/paper {slug}` | Manage paper trading validation | Paper Trading |
| `/promote {slug}` | Run promotion checklist | Promotion |
| `/evaluate {slug}` | Evaluate live performance | Evaluation |
| `/governance` | Run surveillance scan | Any (operational) |
| `/trade` | Execute a trading session | Any (operational) |

---

## Anti-Overfitting Discipline

The lifecycle enforces several anti-overfitting mechanisms:

1. **Spec freeze before backtest**: Prevents simultaneous design and evaluation (the most common form of data snooping).

2. **DSR trial counting**: Every backtest increments the trial counter, raising the bar for statistical significance. This penalizes parameter sweeps.

3. **PBO via CSCV**: Directly estimates the probability that the best in-sample result is an overfitting artifact.

4. **CPCV with purge/embargo**: Prevents information leakage between train and test sets caused by autocorrelation in financial time series.

5. **Parameter perturbation**: Tests whether the strategy's performance is sensitive to small parameter changes. Fragile strategies break under perturbation.

6. **Cost sensitivity**: Tests whether the strategy survives realistic and stressed cost assumptions. Many "profitable" strategies die when costs are properly modeled.

7. **Paper trading minimum**: 30 days and 50 trades provide a minimum sample for live performance evaluation, catching operational and regime-specific issues that backtests miss.

8. **Hypothesis before results**: Peterson's framework requires the hypothesis to be written before seeing any results. The null hypothesis and falsification criteria create a pre-registered testing plan.

9. **Append-only experiment registry**: All results are recorded, preventing selective reporting of favorable outcomes.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-25 | Initial lifecycle reference. Covers all 9 states, artifact schemas, DSR/PBO/CPCV formulas, cost model, benchmark requirements, experiment registry. |

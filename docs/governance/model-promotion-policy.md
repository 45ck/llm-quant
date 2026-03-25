# Model Promotion Policy v1.0

A strategy change -- whether a new model, revised parameters, or a different signal combination -- must pass through a structured promotion pipeline before it can manage live capital. This document defines each stage, its acceptance criteria, and the required evidence.

The pipeline is sequential: a failure at any stage halts promotion. There are no shortcuts.

---

## Overview

```
Stage 1: Hard Vetoes ──> Stage 2: Scorecard ──> Stage 3: Paper Trading ──> Stage 4: Canary ──> Stage 5: Full Deployment
   (pass/fail)           (score >= 85)          (50 trades, 30 days)      (10% alloc, 14d)     (kill switches active)
```

---

## Stage 1: Hard Vetoes

Hard vetoes are binary. **Any single failure is an automatic rejection.** These checks guard against the most common backtest pathologies: overfitting, data snooping, and insufficient out-of-sample validation.

| Veto Check | Threshold | Pass Condition | What It Guards Against |
|-----------|-----------|----------------|----------------------|
| Deflated Sharpe Ratio (DSR) | >= 0.95 | DSR must be at or above 0.95 | Inflated Sharpe from multiple testing. DSR adjusts the Sharpe ratio for the number of strategies tested, penalizing data-mined results. |
| Probability of Backtest Overfitting (PBO) | <= 10% | PBO must be at or below 10% | Overfitting to in-sample data. PBO uses combinatorial cross-validation to estimate the probability that the backtest result is an artifact of overfitting. |
| Stepwise SPA p-value | <= 0.05 | SPA p-value must be at or below 0.05 | False discovery. The Stepwise Superior Predictive Ability test controls for the family-wise error rate when comparing multiple strategies against a benchmark. |
| Minimum Trail Length (MinTRL) | >= 1 OOS period | At least one full out-of-sample period must be available | Insufficient out-of-sample evidence. A strategy that has never been tested on unseen data has no credible performance claim. |

**Evidence required:** Statistical test results with full methodology documentation. The specific implementation of DSR, PBO, and SPA must be recorded so results are reproducible.

---

## Stage 2: Weighted Scorecard

The scorecard provides a holistic assessment across five dimensions. Each dimension is scored 0-100 and weighted to produce a composite score. **Minimum composite score of 85 required to proceed.**

### Scoring Dimensions

| Dimension | Weight | Components | Scoring Guidance |
|-----------|--------|------------|-----------------|
| **Risk-Adjusted Returns** | 25% | Sharpe ratio, Sortino ratio, Calmar ratio | 90-100: Sharpe > 1.2, Sortino > 1.5, Calmar > 0.8. 70-89: Sharpe 0.8-1.2, Sortino 1.0-1.5, Calmar 0.5-0.8. Below 70: Does not meet strategy targets. |
| **Drawdown Characteristics** | 20% | Maximum drawdown, longest drawdown duration, recovery speed | 90-100: Max DD < 10%, recovery < 30 days. 70-89: Max DD 10-15%, recovery 30-60 days. Below 70: Max DD > 15% (violates mandate). |
| **Trade Statistics** | 20% | Win rate, profit factor, average trade P&L, max consecutive losses | 90-100: Win rate > 55%, profit factor > 1.5, max consec losses < 4. 70-89: Win rate 45-55%, profit factor 1.2-1.5, max consec losses 4-5. Below 70: Win rate < 45% or profit factor < 1.2. |
| **Robustness** | 20% | Walk-forward consistency, parameter sensitivity, regime-conditional performance | 90-100: Consistent across all walk-forward windows and regimes, low parameter sensitivity. 70-89: Minor degradation in some windows/regimes, moderate sensitivity. Below 70: Significant inconsistency or high parameter sensitivity. |
| **Operational** | 15% | Complexity (parameter count), data dependency, execution feasibility | 90-100: Few parameters (< 5 free), single data source, simple execution. 70-89: Moderate parameters (5-10), multiple data sources, standard execution. Below 70: Many parameters (> 10), exotic data, complex execution requirements. |

### Composite Score Calculation

```
Composite = (Risk-Adjusted * 0.25) + (Drawdown * 0.20) + (Trade Stats * 0.20)
          + (Robustness * 0.20) + (Operational * 0.15)
```

### Score Interpretation

| Composite Score | Decision |
|----------------|----------|
| >= 85 | **Promote** to Stage 3 (paper trading). |
| 75 - 84 | **Conditional.** Identify specific weaknesses. May proceed if weaknesses are in Operational dimension only and have a remediation plan. |
| < 75 | **Reject.** Requires fundamental strategy revision before re-evaluation. |

**Evidence required:** Completed scorecard with per-dimension scores, component values, and brief justification for each score. Scorecard must be stored in `strategy_changelog`.

---

## Stage 3: Paper Trading Minimum

Before a strategy can manage real capital allocation (even in canary), it must demonstrate viability in the paper trading environment. This stage validates that the strategy works end-to-end: data feeds, indicator computation, signal generation, risk checks, execution, and reporting.

### Minimum Requirements

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Trades executed** | >= 50 | Sufficient sample size to evaluate trade statistics with basic statistical confidence. Fewer than 50 trades leaves win rate and profit factor estimates unreliable. |
| **Calendar days** | >= 30 | Captures at least one full market cycle of weekday trading sessions. Ensures the strategy has operated through varying market conditions, not just a single favorable window. |
| **Paper Sharpe** | >= 0.60 | Below the full strategy target (0.80) to account for paper trading limitations, but high enough to indicate a real edge exists. A strategy that cannot achieve 0.60 in paper trading is unlikely to achieve 0.80 live. |
| **Operational systems** | All tested | Every component of the pipeline must have been exercised: data fetching, indicator computation, signal generation, risk check enforcement, trade execution, portfolio snapshot persistence, hash chain integrity, and performance reporting. |

### What Paper Trading Cannot Validate

Paper trading assumes perfect fills, zero slippage, unlimited liquidity, and no market impact. The following risks are **not** assessed at this stage:

- Execution slippage and market impact
- Liquidity constraints during stress events
- Crowding effects from correlated positioning
- Latency in data feeds and order routing

These risks are addressed by the Crowding/Capacity and Execution Drift/TCA failure modes (currently deferred, to be implemented before live trading).

**Evidence required:** Paper trading performance report including trade log, equity curve, risk metrics, and confirmation that all operational systems were exercised.

---

## Stage 4: Canary Gate

The canary stage runs the new strategy alongside the existing strategy with a small capital allocation. It is the final gate before full deployment. The canary tests real portfolio interaction effects that paper trading cannot capture: impact on overall portfolio risk, correlation with existing positions, and behavior under actual capital constraints.

### Canary Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Capital allocation** | 10% of total portfolio | Small enough to limit damage if the strategy fails. Large enough to generate statistically meaningful trade signals. |
| **Minimum duration** | 14 calendar days | Two full weeks of trading, covering a range of market conditions. |
| **Drawdown limit** | 10% (tighter than full 15%) | The canary operates under stricter risk constraints. If it cannot stay within 10% drawdown at small scale, it will not survive 15% at full scale. |
| **Kill switch** | Immediate rollback | If the canary breaches its drawdown limit or triggers any kill switch, immediately revert to the previous strategy. No second chances at canary stage. |

### Canary Success Criteria

All of the following must be met:

- [ ] 14+ calendar days completed without kill switch trigger
- [ ] Canary drawdown remained below 10% throughout
- [ ] Canary Sharpe ratio >= 0.50 (relaxed from paper target -- real conditions are harder)
- [ ] No adverse interaction effects with the remaining 90% of the portfolio
- [ ] All operational systems performed reliably (no data gaps, no risk check failures, no hash chain issues)

### Canary Failure Protocol

If the canary fails:

1. Immediately revert the 10% allocation to the previous strategy
2. Document the failure mode: what triggered the rollback, market conditions at the time, whether the failure was strategy-related or operational
3. The failed strategy returns to Stage 2 for re-scoring if the failure was strategy-related, or to Stage 3 for re-testing if the failure was operational

### Ramp-Up After Canary Pass

If the canary passes all criteria:

1. **Week 1**: Increase allocation from 10% to 30%
2. **Week 2**: Increase allocation from 30% to 60%
3. **Week 3+**: Ramp to full target allocation
4. At each step, verify that portfolio-level risk metrics remain within limits
5. If any ramp step triggers a warning from the control matrix, pause the ramp and investigate

---

## Stage 5: Full Deployment

The strategy is now managing its full target allocation. Enhanced surveillance applies during the initial period.

### Day 1 Requirements

- All 6 kill switches active and verified (see [Control Matrix](control-matrix.md))
- All 8 failure mode detectors operational
- `strategy_changelog` entry documenting the promotion: date, strategy version, scorecard results, paper trading stats, canary results
- Baseline metrics recorded for regime change detection: Sharpe, win rate, realized vol at deployment

### Enhanced Surveillance (First 30 Days)

| Check | Frequency | Action on Failure |
|-------|-----------|-------------------|
| Performance vs scorecard projections | Daily | If trailing 5-day Sharpe < 0.30, flag for review |
| Drawdown trajectory | Daily | If drawdown pace suggests >15% within 10 days (linear extrapolation), tighten stops |
| Risk metric compliance | Every session | Any limit breach triggers immediate investigation |
| Control matrix scan | Every session | All 6 active failure modes checked |

### Steady-State Review Cycle

After the first 30 days, the strategy enters the standard quarterly review cycle:

| Review | Frequency | Scope |
|--------|-----------|-------|
| **Performance review** | Monthly | Compare actual vs projected returns, Sharpe, drawdown. Flag deviations > 1 standard deviation. |
| **Strategy review** | Quarterly | Full re-scoring against the Stage 2 scorecard. If composite score drops below 75, initiate strategy revision. |
| **Operational review** | Quarterly | Audit data feeds, risk checks, hash chain integrity, configuration drift. |
| **Annual re-certification** | Annually | Full promotion pipeline re-run (Stages 1-2 at minimum) to confirm the strategy still meets statistical standards. |

---

## Promotion Record Template

Every promotion must be recorded in `strategy_changelog` with the following information:

```
## Strategy Promotion: [Strategy Name] — [Date]

### Stage 1: Hard Vetoes
- DSR: [value] (threshold: >= 0.95) — [PASS/FAIL]
- PBO: [value] (threshold: <= 10%) — [PASS/FAIL]
- SPA p-value: [value] (threshold: <= 0.05) — [PASS/FAIL]
- MinTRL: [value] OOS periods (threshold: >= 1) — [PASS/FAIL]

### Stage 2: Scorecard
- Risk-Adjusted Returns: [score]/100 (weight: 25%)
- Drawdown Characteristics: [score]/100 (weight: 20%)
- Trade Statistics: [score]/100 (weight: 20%)
- Robustness: [score]/100 (weight: 20%)
- Operational: [score]/100 (weight: 15%)
- **Composite: [score]/100** (threshold: >= 85)

### Stage 3: Paper Trading
- Trades executed: [count] (threshold: >= 50)
- Calendar days: [count] (threshold: >= 30)
- Paper Sharpe: [value] (threshold: >= 0.60)
- Operational systems: [ALL TESTED / gaps noted]

### Stage 4: Canary
- Duration: [days] (threshold: >= 14)
- Max drawdown during canary: [value] (threshold: < 10%)
- Canary Sharpe: [value] (threshold: >= 0.50)
- Kill switch triggers: [none / details]

### Stage 5: Deployment
- Kill switches verified: [yes/no]
- Baseline metrics recorded: [yes/no]
- Enhanced surveillance start date: [date]

### Approved by: [name/role]
### Approval date: [date]
```

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-25 | Initial policy. Covers all 5 promotion stages with acceptance criteria. |

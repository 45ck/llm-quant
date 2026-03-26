# Implementation Gaps vs. Institutional Quant Standard

*Comparison of current llm-quant implementation against the standards in
`docs/research/institutional-quant-guide.md`. Updated: 2026-03-26.*

---

## Status Summary

| Area | Status | Priority |
|------|--------|----------|
| DSR >= 0.95 gate | IMPLEMENTED | — |
| PBO <= 0.10 gate | IMPLEMENTED | — |
| CPCV (15 OOS paths) | IMPLEMENTED | — |
| Experiment registry (strategy graveyard) | PARTIAL | P2 |
| t-stat threshold enforcement | MISSING | P2 |
| MinTRL computation | MISSING | P2 |
| Shuffled signal fraud detector | MISSING | P1 |
| Regime split validation (2 of 3) | INFORMAL | P2 |
| HRP portfolio construction | MISSING | P2 |
| Volatility targeting | MISSING | P2 |
| Alpha decay monitoring (generalization ratio) | MISSING | P2 |
| Capacity estimation | MISSING | P3 |
| Drawdown CUSUM regime-change detection | MISSING | P3 |
| Kelly / fractional Kelly position sizing | MISSING | P3 |
| Factor neutralization (beta/sector) | N/A (ETF-only) | — |

---

## Critical Gaps (P1)

### 1. Shuffled Signal Fraud Detector

**What's missing:** The most powerful fraud detector — randomize signal dates, run 1000
shuffles, confirm real Sharpe > 95th percentile of shuffled Sharpes. Controls for
time-in-market bias, volatility harvesting, bull market drift.

**Current state:** Not in robustness suite. CPCV and perturbation exist but do not
control for the "strategy is just long equity most of the time" problem.

**Impact:** A strategy that fires when equity is high and sits in cash when it's low
will show Sharpe > 0 simply by being in-the-market at the right times, not from timing.
The shuffled signal test is the only control for this.

**Implementation:** Add `shuffled_signal_test()` to `src/llm_quant/backtest/robustness.py`.
Run 1000 bootstrap samples with randomized signal dates. Return p-value of real Sharpe
vs. shuffled distribution. Gate: p-value < 0.05 (real Sharpe in top 5% of shuffled).

---

## Important Gaps (P2)

### 2. MinTRL Computation

**What's missing:** The Minimum Track Record Length formula tells us how long a backtest
(or live track record) must be before we can conclude the Sharpe is real at 95% confidence.
For SR=0.5 with normal returns, MinTRL ≈ 60+ months.

**Current state:** We use 5-year backtests (60 months) but don't compute or enforce
MinTRL explicitly. Some short-trade strategies may have insufficient observations.

**Formula:**
```
MinTRL = (1 - skew×SR + ((kurt-1)/4)×SR²) × (z_{1-α} / (SR - SR*))²
```

**Implementation:** Add `compute_mintrl()` to robustness.py. Report alongside DSR.
Flag strategies where backtest_length < MinTRL.

### 3. t-Statistic Threshold Enforcement

**What's missing:** Harvey, Liu, Zhu require t > 3.0 for new factor proposals, not the
traditional 2.0. Our DSR gate is related but not the same — DSR >= 0.95 implicitly
enforces a related threshold but doesn't report the raw t-statistic.

**Current state:** t-stat is computed internally in DSR but not reported or enforced
separately.

**Implementation:** Report raw t-stat in experiment registry and robustness.yaml.
Add soft gate: flag if t < 2.5 even if DSR passes (possible with very low trial count).

### 4. Strategy Graveyard Completeness

**What's missing:** Lopez de Prado's Third Law: "Every backtest must be reported with all
trials involved in its production." Our experiment-registry.jsonl captures all runs
through the backtest engine, but doesn't capture:
- Hypotheses that were killed before running (scan-level kills)
- Parameter variations considered informally
- Strategies abandoned based on economic reasoning alone

**Current state:** experiment-registry.jsonl is append-only — GOOD. But the scan-level
research session diary is in markdown docs, not in the registry.

**Implementation:** Add a `scan_registry.jsonl` for phase-1 kills. Each entry: hypothesis,
kill reason, kill date, who killed it, economic mechanism. This completes the graveyard.

### 5. HRP Portfolio Construction

**What's missing:** Current portfolio uses equal weights for all 11 strategies.
Lopez de Prado's Hierarchical Risk Parity:
1. Correlation distance matrix + hierarchical clustering
2. Quasi-diagonalization via dendrogram leaf order
3. Recursive bisection with inverse-variance weights

**Current state:** Equal weight. For uncorrelated strategies this is fine, but our 10
credit-equity strategies have avg rho=0.628 — HRP would correctly downweight the
credit-equity cluster.

**Impact:** Equal-weighting 10 correlated strategies is equivalent to holding 4.35
independent positions, not 10. HRP + inverse-vol would give appropriate weights.

**Implementation:** Add `scripts/compute_hrp_weights.py` using PyPortfolioOpt.
Report optimal weights vs equal weights. Expose as `pq weights` command.

### 6. Volatility Targeting

**What's missing:** Dynamic position sizing that scales each strategy's exposure by
σ_target / σ_realized. Moreira and Muir (2017): this is a "free lunch" that delivers
consistent risk contribution without material return sacrifice.

**Current state:** Fixed position weights (target_weight is static).

**Implementation:** Add vol-targeting layer to BacktestEngine. Each rebalancing period,
scale position size by σ_target / σ_hat where σ_hat is trailing 20-day realized vol.
Cap leverage at 1.5x. Test: does vol-targeted portfolio improve Sharpe vs fixed weight?

### 7. Alpha Decay / Generalization Ratio Monitoring

**What's missing:** Track the "generalization ratio" = live return / backtest return.
Use paired statistical tests to detect divergence. Typical decay: 15-40% from backtest
to live depending on strategy type.

**Current state:** Surveillance module monitors live performance (Sharpe, win rate,
drawdown) but doesn't compare to the specific backtest prediction. The evaluation.yaml
artifact is per-strategy but doesn't systematically track generalization ratio.

**Implementation:** Add `generalization_ratio` field to evaluation.yaml. Compute:
actual_paper_sharpe / backtest_sharpe. Alert if ratio < 0.60 (severe decay). Alert if
ratio < 0.80 (moderate decay). Track trend over time.

---

## Lower Priority Gaps (P3)

### 8. Capacity Estimation

**What's missing:** Daily strategy turnover × average daily volume × 1% participation
rate = approximate capacity. Strategies with capacity < $1M are academic, not investable.

**Current state:** Not computed. All our strategies use liquid ETFs (SPY, QQQ, TLT) with
$10B+ daily volume, so capacity is effectively unlimited at our scale, but worth
formalizing.

### 9. CUSUM Regime-Change Detection

**What's missing:** Formal CUSUM test for detecting strategy degradation over time.
Current kill switches are threshold-based, not sequential-test-based.

**Current state:** 7 kill switches exist but use fixed thresholds, not CUSUM.

### 10. Kelly / Fractional Kelly Position Sizing

**What's missing:** Theoretically optimal position sizes based on individual strategy
Sharpe ratios and correlation structure.

**Current state:** Fixed target_weight in research spec. Not Kelly-optimal.

**Note:** At our current scale (paper trading $100k), equal-weight with position limits
is functionally fine. Kelly sizing becomes important at $1M+ when position impact matters.

---

## Already Well-Implemented (vs. Institutional Standard)

These match or exceed institutional standard for a fund at our stage:

1. **DSR >= 0.95**: implemented, matches Bailey-Lopez de Prado exactly
2. **PBO <= 0.10**: implemented, matches the CSCV framework
3. **CPCV (15 OOS paths)**: implemented with purge + embargo
4. **Perturbation testing (3/5 parameters stable)**: implemented
5. **Spec freeze before backtest**: enforced via frozen_hash
6. **Append-only experiment registry**: implemented (no selective reporting)
7. **Anti-overfitting discipline (no HARKing)**: enforced by lifecycle state machine
8. **Cost modeling (spread + commission + slippage)**: implemented, sensitivity at 1x/2x/3x
9. **Kill switches (6 live monitoring conditions)**: implemented in surveillance module
10. **Paper trading minimum (30 days)**: enforced in promotion policy

---

## Recommended Implementation Order

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P1 | Shuffled signal test | 2-4 hours | High — catches time-in-market bias |
| P2 | MinTRL computation | 1-2 hours | Medium — validates track record length |
| P2 | t-stat reporting | 1 hour | Low — informational, DSR already covers this |
| P2 | HRP portfolio construction | 4-8 hours | High — correct weighting for correlated strategies |
| P2 | Volatility targeting | 4-8 hours | High — free Sharpe improvement |
| P2 | Generalization ratio | 2-4 hours | Medium — catches alpha decay early |
| P3 | Capacity estimation | 1 hour | Low at current scale |
| P3 | CUSUM detection | 4-8 hours | Medium — more sophisticated kill switch |

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-26 | Initial gap analysis vs. institutional-quant-guide.md |

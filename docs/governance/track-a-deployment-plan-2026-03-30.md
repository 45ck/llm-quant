# Track A Deployment Plan — 2026-03-30

**Author:** Portfolio Manager Agent
**Issue:** llm-quant-ivh5
**Scope:** 14 validated Track A strategies — paper trading audit, canary plan, capital allocation
**Status as of:** 2026-03-30

---

## 1. Executive Summary

Fourteen Track A strategies have passed all lifecycle gates (mandate → hypothesis → data-contract → research-spec → backtest → robustness). Paper trading started 2026-03-26 for the first two strategies; a third started 2026-03-30. The remaining 11 have zero paper trading artifacts on disk and need immediate starts.

The earliest any strategy can reach canary eligibility is **2026-04-25** (the two strategies that started 2026-03-26, assuming 50+ trades by that date). The canary gate then requires 14 additional days, meaning the earliest full deployment of the first batch is **2026-05-09**.

This plan organizes the 14 strategies into a phased deployment sequence designed to maximize Track A Sharpe while minimizing inter-strategy correlation in the live book.

---

## 2. Strategy Inventory and Paper Trading Status

### 2.1 Paper Trading Active (3 strategies)

| Slug | Mechanism | Backtest Sharpe | Backtest MaxDD | Paper Start | Days Active | Paper Trades | Earliest Promo Eligible |
|------|-----------|----------------|----------------|-------------|-------------|--------------|------------------------|
| lqd-spy-credit-lead | IG bond → US equity | 1.2502 | 12.4% | 2026-03-26 | 4 | 0 | 2026-04-25 |
| soxx-qqq-lead-lag | Semis → tech equity | 0.8610 | 14.4% | 2026-03-26 | 4 | 0 | 2026-04-25 |
| gld-slv-mean-reversion-v4 | Metals pairs ratio | 1.1967 | 9.6% | 2026-03-30 | 0 | 0 | 2026-04-29 |

**Note:** All three are correctly flat in the current risk-off regime (SPY -1.79% on day 1 of paper). Capital preservation is working as designed. Trade count at 0 because entry signals require credit recovery (lqd-spy) or SOXX positive momentum — both correctly suppressed in this environment. The 50-trade gate will require monitoring; these strategies may need the full 30 days to accumulate sufficient sample size in a trending environment.

### 2.2 Robustness Passed — Paper Trading Not Started (9 strategies)

These strategies have passed all research lifecycle gates per `docs/governance/research-tracks.md` (validated as of 2026-03-26). Their directories exist under `data/strategies/` but contain no lifecycle artifacts yet — they are governance debt from batch registration in `scripts/run_fraud_detectors.py` and `scripts/portfolio_optimizer.py`.

| Slug | Mechanism | Family | Reported Sharpe | Reported MaxDD | Action Required |
|------|-----------|--------|----------------|----------------|-----------------|
| agg-spy-credit-lead | Total bond → US equity | F1 | 1.145 | 8.4% | Start paper trading |
| agg-qqq-credit-lead | Total bond → tech equity | F1 | 1.080 | 11.2% | Start paper trading |
| vcit-qqq-credit-lead | Corp bond → tech equity | F1 | 1.037 | 14.5% | Start paper trading |
| lqd-qqq-credit-lead | IG bond → tech equity | F1 | 1.023 | 13.7% | Start paper trading |
| emb-spy-credit-lead | EM sovereign → US equity | F1 | 1.005 | 9.1% | Start paper trading |
| hyg-spy-5d-credit-lead | HY bond → US equity | F1 | 0.913 | 14.7% | Start paper trading |
| agg-efa-credit-lead | Total bond → intl equity | F1 | 0.860 | 10.3% | Start paper trading |
| hyg-qqq-credit-lead | HY bond → tech equity | F1 | 0.867 | 13.4% | Start paper trading |
| spy-overnight-momentum | Overnight gap momentum | F5 | 1.043 | 8.7% | Start paper trading |

**Important governance note:** These Sharpe/MaxDD figures come from the research-tracks.md summary table (written 2026-03-26), not from on-disk robustness.yaml artifacts. The lifecycle gap audit (2026-03-30) confirmed no experiment artifacts exist on disk for these 9 strategies. Before paper trading can formally start, each strategy needs: (a) `data/strategies/<slug>/` directory with lifecycle YAMLs, and (b) the `/backtest` and `/robustness` commands run to generate on-disk experiment artifacts. Creating a paper-trading.yaml without the underlying robustness artifacts is incomplete.

**Recommended action:** Create beads issues for each (see Section 6), then run `/backtest → /robustness` per strategy before marking paper trading as started.

### 2.3 No On-Disk Artifacts — Incomplete Lifecycle (3 strategies)

These appear in `scripts/run_fraud_detectors.py` but have no directories under `data/strategies/`. They must complete the full lifecycle before paper trading:

| Slug | Mechanism | Family | Status |
|------|-----------|--------|--------|
| tlt-spy-rate-momentum | TLT → SPY momentum | F6 | No artifacts — full lifecycle required |
| tlt-qqq-rate-tech | TLT → QQQ rate/tech | F6 | No artifacts — full lifecycle required |
| ief-qqq-rate-tech | IEF → QQQ rate/tech | F6 | No artifacts — full lifecycle required |

These are NOT counted in the "14 validated" figure. They are registered in the optimizer prematurely. Do not start paper trading until full lifecycle (mandate through robustness) is complete.

---

## 3. Promotion Requirements (from model-promotion-policy.md)

### Stage 3: Paper Trading Minimums (gate to canary)

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| Trades executed | >= 50 | Not >= 10 — the policy requires 50, despite lqd-spy paper file showing min_trades: 10 (which appears to be a typo from an earlier draft) |
| Calendar days | >= 30 | Minimum one full market cycle |
| Paper Sharpe | >= 0.60 | Relaxed from the 0.80 live target |
| All operational systems tested | Yes | Data fetch, indicators, signals, risk checks, execution, persistence, hash chain, reporting |

### Stage 4: Canary Gate

| Parameter | Value |
|-----------|-------|
| Capital allocation | 10% of total portfolio |
| Minimum duration | 14 calendar days |
| Drawdown limit | 10% (tighter than live 15%) |
| Kill switch | Immediate rollback on breach |

**Canary success criteria:** 14+ days clean, drawdown < 10%, canary Sharpe >= 0.50, no adverse portfolio interaction effects, no operational failures.

---

## 4. Timeline

### 4.1 Current State (2026-03-30)

```
Week 0 (now):
  soxx-qqq-lead-lag     ████░░░░░░░░░░░░░░░░░░░░░░░░░░  day 4/30
  lqd-spy-credit-lead   ████░░░░░░░░░░░░░░░░░░░░░░░░░░  day 4/30
  gld-slv-v4            █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  day 0/30
  All others            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  not started
```

### 4.2 Key Dates

| Milestone | Date | Strategies Involved |
|-----------|------|---------------------|
| Paper trading starts (Batch 2: 9 strategies) | 2026-03-31 | agg-spy, agg-qqq, vcit-qqq, lqd-qqq, emb-spy, hyg-spy, agg-efa, hyg-qqq, spy-overnight |
| 30-day clock: lqd-spy + soxx-qqq | 2026-04-25 | Earliest to reach Stage 3 completion |
| 30-day clock: gld-slv-v4 | 2026-04-29 | If paper started 2026-03-30 |
| 30-day clock: Batch 2 | 2026-04-30 | If paper started 2026-03-31 |
| Canary start: Batch 1 (3 strategies) | 2026-04-25 | Conditional on 50+ trades and Sharpe >= 0.60 |
| Canary completion: Batch 1 | 2026-05-09 | 14 days canary |
| Full deployment: Batch 1 | 2026-05-09 | Stage 5, begin 3-week ramp |
| Canary start: Batch 2 | 2026-04-30 | Subject to meeting all Stage 3 gates |
| Canary completion: Batch 2 | 2026-05-14 | 14 days canary |

**Critical dependency:** The 50-trade gate is the bottleneck, not the 30-day gate. In a low-signal risk-off environment, strategies that require credit recovery or positive momentum signals may not accumulate 50 trades in 30 days. Monitor trade counts weekly. If a strategy reaches 30 days with fewer than 50 trades, extend the paper period rather than waiving the trade count gate — the gate exists for statistical confidence in win rate and profit factor estimates.

---

## 5. Canary Deployment Plan

### 5.1 Batch 1 Canary: Three Strategies (Target Start: 2026-04-25)

These three are the highest priority for the first canary batch based on:
1. Earliest paper trading start (soxx-qqq, lqd-spy: 2026-03-26)
2. Highest backtest Sharpe (lqd-spy: 1.25, gld-slv: 1.20)
3. Lowest inter-strategy correlation (different mechanisms: credit lead-lag, semis lead-lag, metals pairs)

| Priority | Strategy | Backtest Sharpe | Mechanism | Correlation Group |
|----------|----------|----------------|-----------|-------------------|
| 1 | lqd-spy-credit-lead | 1.2502 | Credit info flow | F1 — IG bond signal |
| 2 | gld-slv-mean-reversion-v4 | 1.1967 | Metals mean reversion | F2 — independent of credit |
| 3 | soxx-qqq-lead-lag | 0.8610 | Semis lead-lag | F8 — tech sector, not credit |

**Why this combination:** These three span three distinct mechanism families (F1, F2, F8). The correlation between lqd-spy and soxx-qqq will be moderate (both equity-linked) but their entry signals are independent — one responds to IG bond momentum, the other to semiconductor price momentum. gld-slv is the most decorrelated: it is long/short metals pairs with no equity exposure during flat periods.

**What to watch during Batch 1 canary (14 days):**
- Portfolio-level correlation: if gld-slv and lqd-spy both fire simultaneously, monitor gross exposure vs 200% cap
- Drawdown pace on each canary leg — any 5% drawdown on a canary strategy triggers review
- Operational reliability: are all 8 pipeline components working without manual intervention?

### 5.2 Batch 2 Canary: Nine F1 Credit Variants + spy-overnight (Target Start: 2026-04-30)

If Batch 1 canary succeeds, Batch 2 introduces the remaining 9 strategies. However, these 9 are highly correlated: 8 are F1 credit lead-lag variants trading the same mechanism with different ETF pairs.

**Correlation risk:** Deploying all 9 simultaneously would create excessive concentration in the credit lead-lag signal. The combined position could exceed sector concentration limits if all 8 credit strategies fire together.

**Recommended staggering for Batch 2:**
1. First: `agg-spy-credit-lead` (highest Sharpe 1.145, lowest MaxDD 8.4%) — one canary slot
2. Second (1 week later): `spy-overnight-momentum` (independent F5 mechanism, Sharpe 1.043)
3. Third batch (2 weeks later): remaining 7 credit variants in 2-3 sub-batches

Do not deploy all 9 credit variants simultaneously into canary. The correlation kill list in `docs/research/extreme-sharpe-playbook.md` applies here — adding more of the same mechanism beyond 3 strategies provides diminishing Sharpe diversification benefit while adding concentration risk.

### 5.3 Correlation-Adjusted Deployment Caps

Per CLAUDE.md: "Current 11 strategies, avg ρ=0.584 → actual combined SR ≈ 1.35". The 9 F1 credit variants share the same signal. Deploying all 9 at full weight would not achieve 9x diversification benefit — they are effectively 1 strategy expressed in 9 pairs.

**Capital allocation guidance for F1 variants:**
- Treat all 8 credit-signal strategies as a single "credit lead-lag" sleeve
- Total sleeve allocation: 15-20% of the 70% Track A budget (i.e., 10.5-14% of total NAV)
- Individual strategy allocation within sleeve: 1.5-2.5% each
- This respects the 10% per-position limit while preventing the credit sleeve from dominating

---

## 6. Capital Allocation Plan

Track A budget: 70% of $100,000 NAV = $70,000

### 6.1 Target Allocation at Full Deployment (after all Stage 5 promotions)

| Strategy | Mechanism | Target Allocation | $ Amount | Rationale |
|----------|-----------|------------------|----------|-----------|
| lqd-spy-credit-lead | IG bond → SPY | 8.0% | $8,000 | Highest Sharpe (1.25), anchor position |
| gld-slv-mean-reversion-v4 | Metals pairs | 8.0% | $8,000 | Most decorrelated, Sharpe 1.20 |
| soxx-qqq-lead-lag | Semis → QQQ | 7.0% | $7,000 | Independent mechanism (F8) |
| spy-overnight-momentum | Overnight gap | 7.0% | $7,000 | Independent mechanism (F5) |
| agg-spy-credit-lead | Bond → SPY | 5.0% | $5,000 | Credit sleeve anchor |
| agg-qqq-credit-lead | Bond → QQQ | 4.0% | $4,000 | Credit sleeve |
| emb-spy-credit-lead | EM bond → SPY | 4.0% | $4,000 | Credit sleeve, some EM decorrelation |
| vcit-qqq-credit-lead | Corp bond → QQQ | 3.5% | $3,500 | Credit sleeve |
| lqd-qqq-credit-lead | IG bond → QQQ | 3.5% | $3,500 | Credit sleeve |
| hyg-spy-5d-credit-lead | HY bond → SPY | 3.0% | $3,000 | Credit sleeve, HY has different signal timing |
| agg-efa-credit-lead | Bond → EFA | 3.0% | $3,000 | International diversification |
| hyg-qqq-credit-lead | HY bond → QQQ | 3.0% | $3,000 | Credit sleeve |
| **Cash reserve** | — | **5.0%** | **$5,000** | Mandatory per Track A mandate |
| **Unallocated** | — | **35.0%** | **$35,000** | Track B (30%) + excess buffer (5%) |
| **Total Track A** | | **70%** | **$70,000** | |

**Key constraint:** The 8 credit-lead-lag strategies (F1) total 29% of NAV combined. Their signals are highly correlated. Under risk-on regimes, many will simultaneously hold equity positions, concentrating exposure. The per-position 10% cap and sector concentration 30% cap (from `risk/manager.py`) provide a hard floor — but active monitoring is required.

### 6.2 Canary Allocation (Stages 4)

During canary, each strategy receives 10% of its target allocation:
- Batch 1 canary (3 strategies): ~2.3% of total NAV deployed per strategy
- Total canary exposure: ~7% of NAV during Batch 1

This is well within risk limits and allows meaningful signal generation without concentrating early-stage risk.

---

## 7. Risk Notes

### 7.1 Batch Size Discipline

**Promote in batches of 3, observe for 14 days before expanding.** This is not just a policy preference — it is necessary because:

1. **Interaction effects are invisible in paper trading.** Paper trading assumes zero market impact and independent execution. The first time two strategies simultaneously enter SPY, the combined size may trigger concentration checks. The canary gate catches this; skipping it does not.

2. **Correlated drawdown amplification.** If 5 credit-lead strategies all exit simultaneously (because LQD drops), the portfolio takes a coordinated hit that is larger than any individual strategy's MaxDD. This correlation is documented (avg ρ=0.584 in CLAUDE.md) but its portfolio-level impact has not been stress-tested for the full 9-strategy credit sleeve.

3. **Operational load.** Each new canary adds monitoring burden. Three simultaneous canaries is manageable; nine is not.

### 7.2 Credit Sleeve Concentration

The 8 F1 credit lead-lag strategies are effectively variations on one alpha signal. Before deploying more than 3 of them, run the portfolio Sharpe math:

```
SR_P = SR_individual × √(N / (1 + (N-1) × ρ))
```

For N=8 strategies with Sharpe=1.0 and ρ=0.80 (a conservative estimate for same-mechanism strategies):
```
SR_P = 1.0 × √(8 / (1 + 7 × 0.80)) = 1.0 × √(8/6.6) = 1.0 × 1.10 = 1.10
```

Adding strategies 5-8 adds less than 10% to portfolio Sharpe while roughly doubling the correlated drawdown risk. The marginal value of adding the 5th+ credit strategy may not justify the complexity.

**Recommendation:** Cap the credit sleeve at 4 strategies (lqd-spy + agg-spy + emb-spy + spy-overnight as a parallel F5 strategy). The remaining 4 credit variants provide minimal diversification benefit at full deployment.

### 7.3 Trade Count Gate — Do Not Waive

The 50-trade minimum exists because win rate and profit factor estimates are unreliable below 50 observations. In the current risk-off environment, momentum and credit entry signals are suppressed. If the 30-day clock expires before 50 trades, the correct response is to extend paper trading — not to waive the gate. A strategy that cannot generate 50 paper trades in a reasonable timeframe is not operationally fit for live deployment in any regime.

### 7.4 Kill Switch Pre-Check Before Any Promotion

Per model-promotion-policy.md Stage 5: all 6 kill switches must be active and verified before full deployment. The current governance scan (`/governance`) confirms this. Run `/governance` immediately before each Stage 4 canary start and Stage 5 full deployment. Do not promote if any kill switch is flagged.

---

## 8. Summary Timeline

| Date | Action |
|------|--------|
| 2026-03-31 | Start paper trading for all 9 strategies in Section 2.2 (after creating lifecycle artifacts) |
| 2026-04-07 | First weekly trade count review for Batch 1 (soxx-qqq, lqd-spy, gld-slv) |
| 2026-04-14 | Second weekly review — verify operational systems checklist progress |
| 2026-04-21 | Final week of Batch 1 paper period |
| 2026-04-25 | **Batch 1 promotion decision:** soxx-qqq + lqd-spy (if 50+ trades and Sharpe >= 0.60) |
| 2026-04-25 | **Batch 1 canary start:** lqd-spy + gld-slv + soxx-qqq at 10% allocation each |
| 2026-04-29 | gld-slv reaches 30-day minimum (if Batch 1 canary already running, observe) |
| 2026-04-30 | Batch 2 paper period completes (30 days) — promotion decision for F1 variants + spy-overnight |
| 2026-05-09 | **Batch 1 canary completes** (14 days) — full deployment decision |
| 2026-05-09 | Week 1 ramp: increase Batch 1 from 10% → 30% allocation |
| 2026-05-16 | Week 2 ramp: increase Batch 1 from 30% → 60% allocation |
| 2026-05-14 | **Batch 2 canary completes** — full deployment decision for F1 + spy-overnight |
| 2026-05-23 | Week 3+: Batch 1 at full target allocation |

---

## 9. Appendix: Stage 3 → Stage 4 Checklist Template

Use this for each strategy before promoting to canary:

- [ ] Calendar days >= 30
- [ ] Trades executed >= 50
- [ ] Paper Sharpe >= 0.60
- [ ] All 8 operational systems tested (data_fetching, indicator_computation, signal_generation, risk_checks, trade_execution, portfolio_persistence, hash_chain_integrity, performance_reporting)
- [ ] No incidents with critical severity in the incident log
- [ ] Slippage drift <= 5 bps
- [ ] Max drawdown during paper period < 15%
- [ ] `/governance` scan clean (no halts, no warnings)
- [ ] `strategy_changelog` entry created

---

*Document generated: 2026-03-30. Review and update after each weekly paper trading snapshot.*

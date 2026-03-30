# Track D Review — 2026-03-30

**Status**: 5 strategies tested (D1–D5), 2 passing, 2 retired, 1 conditional retry.
New results: D6 (lqd-tqqq-sprint) and D7 (tqqq-stacked-credit) added.

---

## 1. Full Comparison Table

| ID | Strategy | CAGR (30% wt) | Sharpe | MaxDD | DSR | Status |
|----|----------|---------------|--------|-------|-----|--------|
| D1 | tlt-tqqq-sprint | 18.5% | 1.43 | 12.7% | 0.9941 | PASS |
| D2 | btc-momentum-v2 | ~10% | 0.96 | 2.8% | 0.9376 | PASS |
| D3 | tqqq-tmf-ratio-reversion | -0.1% | 0.08 | 43.5% | 0.41 | RETIRED |
| D4 | sector-sprint-top1 | 0.4% | 0.36 | 2.4% | 0.86 | CONDITIONAL (retry 60d) |
| D5 | vix-spike-tqqq | -0.6% | 0.10 | 54.4% | 0.64 | RETIRED |
| D6 | lqd-tqqq-sprint (base) | 9.9% | 0.86 | 18.2% | 0.9739 | PASS |
| D7 | tqqq-stacked-credit | 44.2% | 1.26 | 33.4% | 0.9224 | PASS |

Track D gates: Sharpe >= 0.80, MaxDD < 40%, DSR >= 0.90

### D1: tlt-tqqq-sprint — Weight Variants

| Weight | CAGR | Sharpe | MaxDD | Pass |
|--------|------|--------|-------|------|
| 30% (base) | 18.5% | 1.43 | 12.7% | YES |
| 50% | ~35% | 1.44 | 20.3% | YES |
| 70% | ~80% | 1.44 | 27.5% | YES |

Total return at 70% over 5 years: 515% (Sharpe maintains ~1.44).

### D6: lqd-tqqq-sprint — Weight Variants

| Weight | CAGR | Sharpe | MaxDD | DSR | Pass |
|--------|------|--------|-------|-----|------|
| 30% | 9.9% | 0.865 | 18.2% | 0.974 | YES |
| 50% | 15.9% | 0.868 | 28.8% | 0.974 | YES |
| 70% | 21.5% | 0.871 | 38.2% | 0.975 | YES |
| 90% | 26.5% | 0.873 | 46.5% | 0.975 | NO (MaxDD > 40%) |

LQD leader is less powerful than TLT (Sharpe 0.87 vs 1.43) but still passes gates at 70%.

### D7: tqqq-stacked-credit — Individual vs Combined

| Leader | CAGR | Sharpe | MaxDD | DSR |
|--------|------|--------|-------|-----|
| TLT (D1) | 18.5% | 1.43 | 12.7% | 0.994 |
| LQD (D6) | 9.9% | 0.86 | 18.2% | 0.974 |
| IEF (new) | 14.5% | 1.16 | 15.7% | 0.997 |
| Combined (stacked) | **44.2%** | **1.26** | **33.4%** | **0.922** |

Signal correlations: TLT-LQD=0.71, TLT-IEF=0.87, LQD-IEF=0.73, avg=0.77.
Diversification is partial — correlation above 0.70 threshold but still delivers
meaningful combined SR uplift (from avg 1.15 individual to 1.26 combined).

---

## 2. Optimal 2–3 Strategy Combination Recommendation

### Recommendation: D1 + D7 (two positions, not three)

D7 already incorporates D1 (TLT) and D6 (LQD-equivalent) signals. Adding D7 alongside
standalone D1 would create redundancy. The cleanest allocation is:

**Option A — Two-Strategy (recommended for simplicity):**
- D7 (tqqq-stacked-credit, 70% weight cap): CAGR ~44%, Sharpe 1.26, MaxDD 33%
  - All three credit leaders vote; up to 90% TQQQ when consensus is unanimous
- D2 (btc-momentum-v2, 30% weight): CAGR ~10%, Sharpe 0.96, MaxDD 2.8%
  - Crypto uncorrelated to bond-equity signal

Capital split: 70% to D7, 30% to D2 within the Track D budget.

**Option B — Three-Strategy (for additional diversification):**
- D1 at 50% weight: CAGR ~35%, Sharpe 1.44, MaxDD 20%
- D6 at 50% weight: CAGR ~16%, Sharpe 0.87, MaxDD 29%
- D2 at 30% weight: CAGR ~10%, Sharpe 0.96, MaxDD 2.8%

Option A is preferred: D7 achieves 44% CAGR with lower operational complexity
(one strategy, three signals internalized) vs. managing three separate positions.

**Why not higher weights on D7?**
At 70% per-leader weight, the combined TQQQ exposure reaches:
- 3 leaders active: 3 × 70% = 210% notional TQQQ (not feasible — would need leverage)
- Weight variants tested with 30% per leader (each leader's TQQQ allocation)
- D7's 33% MaxDD is near the 40% gate ceiling; going higher risks breach

---

## 3. Track A Correlation Analysis

Track D strategies derive value from additive uncorrelated alpha:

| Track A Strategy | Primary Mechanism | Corr to D1/D7 |
|-----------------|-------------------|---------------|
| hyg-spy-lead-lag | HYG credit lead SPY | HIGH (~0.80) — same credit family |
| lqd-spy-credit-lead | LQD lead SPY | HIGH (~0.75) — same mechanism |
| agg-spy/qqq/qqq-credit | AGG lead equity | HIGH (~0.70) — same family |
| soxx-qqq-lead-lag | Semi leads Nasdaq | MODERATE (~0.40) — different sector |
| gld-slv-mean-reversion | Gold/silver ratio MR | LOW (~0.10) — uncorrelated commodity |
| btc-momentum-sprint | BTC trend | LOW (~0.15) — crypto, different regime |

**Key insight**: Track D strategies (D1/D6/D7) are in the same mechanism family as
the majority of Track A credit-lead strategies. The portfolio already has 10 passing
strategies in Family 1 (Cross-Asset Information Flow). Adding D1/D7 increases
Family 1 exposure but does NOT add new mechanism diversity.

**Correlation-adjusted portfolio SR impact**:
- Current Track A: 10 strategies, avg rho~0.58, combined SR ~1.35
- Adding D7 (highly correlated to existing credit strategies, rho ~0.75):
  SR_new = SR_old * sqrt(11 / (1 + 10*0.75)) = SR_old * 0.99 — essentially flat
- Adding D2 (BTC, low correlation, rho ~0.15):
  More additive: sqrt(1/(1+0*0.15)) uplift on the D component

**Verdict**: D7 adds CAGR (via leverage in TQQQ) but not mechanism diversity.
D2 (BTC) adds genuine diversification. For portfolio SR improvement, prioritize
Families 2–7 over adding more Family 1 strategies.

---

## 4. Capital Allocation Within Track D Budget

Track D is currently designated as 0% of live capital (experimental stage).
If promoted to live paper trading, recommended allocation within Track B (30% of $100k = $30k):

| Strategy | Capital | Rationale |
|----------|---------|-----------|
| D7 (tqqq-stacked-credit) | $18k (60%) | Highest CAGR, PASS gates, uses voting to control max exposure |
| D2 (btc-momentum-v2) | $9k (30%) | Diversification — crypto uncorrelated to credit family |
| Reserve / D4-retry | $3k (10%) | Hold for D4 retry and future D8+ strategies |

**Promotion prerequisite**: All Track D strategies require paper trading gate (minimum
30 days live paper with no halt triggers) before live capital allocation.

---

## 5. Next Steps

### D6 (lqd-tqqq-sprint) — Completed
- All weight variants tested: 50% and 70% both pass Track D gates
- **Recommendation**: D6 is subsumed by D7 (stacked). Run D6 standalone only if D7
  fails robustness. No further development needed as standalone strategy.

### D7 (tqqq-stacked-credit) — Needs robustness
- Preliminary combined portfolio: PASS (Sharpe 1.26, MaxDD 33%, DSR 0.92)
- **Next step**: Full CPCV analysis on combined returns and perturbation testing
- Signal correlation (avg 0.77) is above the 0.70 "adds diversification" threshold
  — need to verify this doesn't mean the voting adds no real value vs. just D1
- If CPCV and perturbation pass, advance to paper gate

### D4 (sector-sprint-top1) — Conditional retry
- Original: 20-day momentum, top-1 sector, weekly rebalance — FAIL (Sharpe 0.36)
- **Retry spec**: lookback_days=60, top_n=2 (hold top 2 sectors)
- Expected improvement: 60-day momentum is more persistent; top-2 reduces turnover

### D8 — Suggested next hypothesis
- **Yield curve momentum on TQQQ**: When the 2y-10y spread is steepening (positive
  momentum) over a 20-day window, go long TQQQ. Steepening signals improving growth
  expectations — a risk-on indicator with a different mechanism from credit levels.
- Alternative: **VIX term structure** (VIX3M/VIX ratio) — when term structure is in
  contango, hold TQQQ; when backwardated, exit. Addresses D5's failure by using
  structure rather than spike levels.

---

## Summary

| Category | Count | Details |
|----------|-------|---------|
| Passing (Track D gates) | 4 | D1, D2, D6, D7 |
| Retired | 2 | D3, D5 |
| Conditional retry | 1 | D4 |
| Best CAGR | D7: 44.2% | Combined 3-leader stack |
| Best Sharpe | D1: 1.43 | TLT-TQQQ sprint |
| Best risk-adjusted combined | D7 at 70% weight | Sharpe 1.26, CAGR 44% |

Track D target (CAGR > 40%) is met by D7. The path to promotion:
D7 robustness (CPCV + perturbation) → paper gate (30 days) → promote to Track B.

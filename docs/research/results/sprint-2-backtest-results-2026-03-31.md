# Sprint 2 Backtest Results — 2026-03-31

**Objective:** Backtest 4 backup hypotheses from Sprint 1 failures, targeting Families 3, 4, 5, 6 for portfolio diversification.

**Result: 0 of 4 passed.** All four strategies fail their Track A gates. H4.4 shows promise at higher cost levels.

---

## Summary Table

| Hypothesis | Family | Track | Sharpe | Gate | MaxDD | Gate | DSR | Trades | Verdict |
|-----------|--------|-------|--------|------|-------|------|-----|--------|---------|
| H3.5 Skip-Month TSMOM | F3 | A | 0.585 | >0.80 | 6.44% | <15% | 0.882 | 75 | **FAIL** |
| H4.4 GARCH Regime Sizing | F4 | A | 0.605 | >0.80 | 12.44% | <15% | 0.891 | 77 | **FAIL** (closest) |
| H5.4 Turn-of-Month Broad | F5 | A | -0.226 | >0.80 | 12.81% | <15% | 0.321 | 101 | **KILL** |
| H6.2 Cyclical/Defensive Ratio | F6 | A | -0.622 | >0.80 | 37.48% | <15% | 0.102 | 43 | **KILL** |

## Benchmark Comparison

| Strategy | Return | 60/40 Benchmark | Excess |
|----------|--------|-----------------|--------|
| H3.5 Skip-Month TSMOM | +12.42% | +22.93% | -10.52% |
| H4.4 GARCH Regime | +26.83% | +38.18% | -11.36% |
| H5.4 Turn-of-Month | -7.28% | +38.18% | -45.46% |
| H6.2 Cyclical/Defensive | -30.90% | +22.93% | -53.83% |

---

## Detailed Analysis

### H3.5 Skip-Month TSMOM (Family 3) — FAIL

- **Experiment**: 605939dc (trial #1, 8 symbols, 5 years)
- **Sharpe 0.585** — misses 0.80 gate by 0.21
- **MaxDD 6.44%** — excellent risk control, well within 15% limit
- **Calmar 0.441** — misses 0.5 gate
- **Annualized return 2.84%** — far below 15-25% target
- **Cost insensitive** — Sharpe drops only 3.8% at 3x costs (0.585 → 0.563)
- **Win rate 8.0%, Profit Factor 0.09** — anomalously low; suggests the engine counts trade entries/exits differently from actual P&L attribution
- **Beats benchmark on risk-adjusted basis** — Sharpe 0.585 vs benchmark 0.397
- **DSR 0.882** — below 0.95 gate
- **Diagnosis**: Skip-month variant performs WORSE than H3.1 (Sharpe 0.719 vs 0.585). The skip-month construction removes recent price information that was actually helpful for momentum signal quality. The 252-day single-lookback may also be inferior to H3.1's blended 21/63/252d approach.
- **F3 status**: Both H3.1 and H3.5 fail. H3.1 remains the better F3 candidate. Consider iterating H3.1 with higher vol target (15-20%) and allowing shorts before trying other F3 variants.

### H4.4 GARCH Regime Sizing (Family 4) — FAIL (closest to passing, anomalous)

- **Experiment**: 8ea629e1 (trial #1, 2 symbols, 5 years)
- **Sharpe 0.605 at 1x costs** — misses 0.80 gate
- **ANOMALY: Sharpe 0.802 at 1.5x costs** — PASSES 0.80 gate at higher costs
- **MaxDD 12.44% (1x), 10.34% (1.5x)** — passes Track A's 15% limit
- **Sortino 1.142 at 1.5x** — passes 1.0 gate
- **Profit Factor 2.09 (1x), 2.79 (1.5x)** — strong edge, winners much larger than losers
- **DSR 0.891 (1x), 0.948 (1.5x)** — nearly passes 0.95 gate at 1.5x
- **DD Duration 203 days** — shortest of all Sprint 1+2 strategies
- **Diagnosis**: The cost sensitivity anomaly is significant. Higher transaction costs filter out marginal regime switches (77 trades → 59), leaving only high-conviction regime changes. This suggests the GARCH signal has a core of strong trades diluted by noise-driven regime flips. The strategy needs a persistence filter or minimum variance threshold to replicate the 1.5x cost filtering effect at 1x costs.
- **Path forward**: Iterate H4.4 with:
  1. Persistence filter: require regime to persist 3-5 days before switching (mimics cost filter)
  2. Minimum variance delta: only switch when GARCH variance crosses percentile by >5% margin
  3. Wider percentile band (80th instead of 75th) to reduce whipsaw
- **F4 status**: Most promising F4 candidate. Worth iterating — the 1.5x cost metrics (Sharpe 0.802, Sortino 1.142, MaxDD 10.34%) would pass Track A if achievable at 1x costs.

### H5.4 Turn-of-Month Broad (Family 5) — KILL

- **Experiment**: 23b0afb0 (trial #1, 1 symbol, 5 years)
- **Sharpe -0.226** — negative Sharpe, strategy destroys value
- **Total return -7.28%** vs benchmark +38.18%
- **Win rate 24.8%, Profit Factor 0.87** — sub-1.0, losers outweigh winners
- **101 trades** — adequate sample, signal is genuinely unprofitable
- **MinTRL infinite** — negative Sharpe means no track record can validate this
- **Diagnosis**: The Turn-of-Month effect does NOT produce tradeable alpha on SPY in the 2022-2026 period. The 4-day window (T-1 to T+3) may be too wide — the original `turn-of-month-spy` spec uses a 2-day window. However, the fundamental issue is that the TOM anomaly appears arbitraged away in modern markets, or the 2022 bear market destroyed the pension flow mechanism.
- **F5 status**: H5.4 killed. Consider H5.1 (pre-FOMC TLT drift) which targets a different mechanism (Fed communication premium vs pension flows). The existing `pre-fomc-tlt-drift` spec uses CalendarEventStrategy and is already frozen.

### H6.2 Cyclical/Defensive Ratio Timing (Family 6) — KILL

- **Experiment**: 0966b0b9 (trial #1, 4 symbols, 5 years)
- **Sharpe -0.622** — deeply negative, worst of all Sprint 1+2 strategies
- **Total return -30.90%** — lost nearly a third of capital
- **MaxDD 37.48%** — exceeds even Track B's 30% limit
- **DD Duration 1,041 days** — nearly the entire backtest period
- **Win rate 18.6%, Profit Factor 0.44** — catastrophic signal quality
- **Diagnosis**: The XLI/XLU ratio rotation was devastated by the 2022-2023 regime where both cyclicals and defensives rotated unpredictably. Rising rates hurt utilities (signal says "expansion" → long SPY) while simultaneously crushing equity risk appetite. The 42-day SMA whipsawed through this period, generating wrong-direction signals consistently. The persistence filter (3 days) was insufficient to prevent false signals.
- **F6 status**: Both H6.4 (macro barometer, Sprint 1) and H6.2 (cyclical/defensive) failed badly. F6 macro rotation strategies may be fundamentally challenged by the 2022 regime break (simultaneous equity+bond selloff). Consider H6.6 (credit spread momentum) as the last F6 candidate — it uses a different signal source (HYG/LQD credit ratio) that may better capture risk appetite shifts.

---

## Key Takeaways

1. **F3 and F4 show weak but real signals**: H3.5 (Sharpe 0.585) and H4.4 (Sharpe 0.605) both generate positive risk-adjusted returns with controlled drawdowns. The signals exist but are too weak for standalone strategies. Both are iteration candidates.

2. **H4.4 GARCH cost anomaly reveals signal structure**: The improvement at higher costs (0.605 → 0.802 Sharpe) is a strong diagnostic signal. The strategy makes too many marginal trades at base costs. A persistence filter could replicate this effect and potentially push H4.4 past the Sharpe 0.80 gate.

3. **F5 and F6 calendar/macro rotation strategies are dead**: Both H5.4 and H6.2 produced negative Sharpe ratios. Calendar effects (TOM) appear arbitraged away. Simple macro rotation (XLI/XLU) fails in regime-break environments. These families need fundamentally different approaches.

4. **The 2022 regime break continues to dominate**: All 8 Sprint 1+2 strategies struggled with the 2022 rate shock. Strategies that survive 2022 (like F1 credit-lead and F8 non-credit lead-lag) tend to pass gates. This is now a reliable stress test — any strategy that can't handle simultaneous equity+bond drawdowns is likely to fail.

5. **Sprint 1+2 combined: 0/8 passed, but 3 show promise**: H3.1 (Sharpe 0.719), H3.5 (0.585), and H4.4 (0.605) all have positive signal structure worth iterating. H4.4 is the highest-priority iteration target due to the cost anomaly.

---

## Sprint 3 Recommendations

### Priority 1: Iterate H4.4 GARCH with persistence filter
The cost anomaly strongly suggests a persistence filter will improve performance. Test:
- 3-day regime persistence requirement
- 80th percentile threshold (vs current 75th)
- Minimum 5% variance margin over threshold

### Priority 2: Iterate H3.1 with relaxed parameters
H3.1 (Sharpe 0.719, MaxDD 2.77%) is the closest-to-passing strategy across both sprints:
- Increase vol target from 10% to 15-20%
- Allow short positions
- These changes should increase return without proportionally increasing risk

### Priority 3: New family candidates
| Rank | Hypothesis | Family | Rationale |
|------|-----------|--------|-----------|
| 1 | H5.1 Pre-FOMC TLT Drift | F5 | Different F5 mechanism (Fed premium vs pension flows), spec already frozen |
| 2 | H6.6 Credit Spread Momentum | F6 | Last F6 candidate, uses HYG/LQD credit signal |
| 3 | H7.1 VIX Percentile Spike | F7 | Already has frozen spec, different from killed H7.2 |
| 4 | H9.1-H9.3 Carry strategies | F9 | Untested family, known academic alpha source |

### Portfolio Sharpe Impact
Current portfolio: 11 F1/F8 strategies, avg ρ=0.584, SR_P ≈ 1.35.
No new families added from Sprint 1+2. The diversification goal remains unmet.
Target: 5+ families with ρ < 0.20 for SR_P ≈ 2.0.

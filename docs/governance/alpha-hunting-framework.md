# Ruthless Alpha Hunting Framework

This document defines the research methodology for finding, validating, and combining
uncorrelated alpha sources. It applies to both Track A (Defensive Alpha) and Track B
(Aggressive Alpha) research programs.

---

## The Math: How Portfolio Sharpe Scales

**Truly uncorrelated strategies (ρ=0):**
```
SR_portfolio = √(Σ SR_i²)
```

**Correlated strategies (equal SR, pairwise correlation ρ, equal weight):**
```
SR_portfolio = SR_individual × √(N / (1 + (N-1)×ρ))
```

The correlation term is critical. At ρ=0.3, 25 strategies with SR=1.0 yield portfolio
SR=1.86 — not 5.0. The simplified "N_effective" approximation is optimistic.

**Realistic combined SR at different correlation levels:**

| N strategies (SR=1.0) | ρ=0.0 | ρ=0.1 | ρ=0.2 | ρ=0.3 | ρ=0.5 |
|---|---|---|---|---|---|
| 4 | 2.00 | 1.71 | 1.51 | 1.37 | 1.15 |
| 9 | 3.00 | 2.24 | 1.83 | 1.59 | 1.26 |
| 16 | 4.00 | 2.71 | 2.09 | 1.75 | 1.33 |
| 25 | 5.00 | 3.09 | 2.28 | 1.86 | 1.39 |

**Current state (2026-03-26):** 11 strategies, avg ρ=0.584 across portfolio. Combined
SR ≈ 1.0 × √(11 / (1 + 10×0.584)) ≈ 1.0 × √(1.82) ≈ **1.35** (corrected estimate —
the 2.3 estimate assumed zero correlation, which is wrong for our credit-heavy portfolio).

**Target state:** 8+ strategies across 5+ independent mechanism families with avg ρ < 0.20
→ Portfolio SR ~2.0–2.5. This requires adding genuinely orthogonal mechanism families.

**Crisis warning:** correlations spike during crises. Strategies with ρ=0.1 in normal
markets become ρ=0.5 in a crisis — cutting portfolio SR from 2.24 to 1.26. Trend
following is the primary crisis diversifier (historically negative correlation to equities
during drawdowns). See `docs/research/extreme-sharpe-playbook.md` for full treatment.

---

## The Kill Chain: 4 Phases

### Phase 1: HUNT — Cast Wide, Kill Fast

**Goal:** Screen 20+ mechanism families in 2 hours. Kill 80% in under 5 minutes each.

**Rules:**
- NEVER spend >30 minutes on a hypothesis that hasn't shown Sharpe > 0.6 in a raw scan
- NEVER run perturbation or CPCV on something that hasn't passed the smell test
- Run the CHEAPEST test first. If it fails, move on. No "maybe if I tweak the parameters..."
- Thoroughness is the enemy at this stage. It comes in Phase 3.

**Kill criteria (in order, stop at first failure):**

1. **Economic mechanism test (30 seconds, no code):** Can you explain WHY this should
   work in one sentence that doesn't use the word "correlation"? If the mechanism is
   "X goes up then Y goes up" with no causal channel → KILL.

2. **Existence test (2 minutes):** Run ONE parameter set (the theoretically motivated
   default). Sharpe < 0.6 → KILL.

3. **Not-obviously-broken test (2 minutes):** Does it trade enough? (>25 trades in
   5 years). MaxDD < 20%? Spend <80% time in cash? Any failure → KILL.

4. **Robustness sniff (5 minutes):** Run 3 nearby parameter sets. If the best is >2x
   the worst, the signal is fragile → KILL.

**Expected output:** ~3-5 survivors per session out of 20 screened.

---

### Phase 2: VALIDATE — Prove It's Not Fake

**Goal:** Determine if the signal is genuine or a statistical artifact. 30-60 minutes
per candidate.

**The 5 Fraud Detectors (run ALL, any failure = kill):**

#### Detector 1: Shuffled Returns Test
Randomize the signal dates (keep the return series intact, shuffle which days the signal
fires). Run the "strategy" on shuffled signals 1000 times. If your real Sharpe doesn't
exceed the 95th percentile of shuffled Sharpes → the signal has no predictive power.

This is the single most powerful fraud detector. It controls for: time-in-market bias,
volatility harvesting, bull market drift, and everything else that isn't genuine signal
timing.

#### Detector 2: Economic Regime Split
Split your backtest into 2-3 distinct economic regimes (e.g., 2021 recovery, 2022 bear,
2023-25 bull). The signal must work in at least 2 of 3 regimes. If it only works in one
regime, it's fitting to that regime, not capturing a real mechanism.

#### Detector 3: Out-of-Sample Period
Reserve the most recent 12 months as a TRUE holdout. Never look at it during development.
Only check it once, as final confirmation. If you've already peeked at recent data for any
hypothesis, that hypothesis is contaminated — you cannot un-see the results.

#### Detector 4: Mechanism Inversion
Flip the signal. If "buy when X rises" works, does "sell when X rises" lose money? If the
inverted signal is flat (not negative), the original signal isn't predictive — it's just
capturing market beta or time-in-market.

#### Detector 5: Alternative Instruments
If "AGG leads SPY" works, does "AGG leads IWM" also work? The real test: does a DIFFERENT
mechanism applied to the SAME instruments also work? If only one specific mechanism works
for a specific pair, it is more likely data-mined. Genuine market effects show up through
multiple lenses.

---

### Phase 3: STRESS — Prove It Survives the Real World

**Goal:** Your existing 5-gate framework (Sharpe, MaxDD, DSR, CPCV, Perturbation) goes
here. This is where thoroughness matters.

**Additional stress tests beyond the standard 5 gates:**

#### Stress 1: Transaction Cost Massacre
Run at 0 bps, 5 bps, 10 bps, 20 bps, 50 bps one-way. Plot Sharpe vs. cost. Find the
"break-even cost" — the level where Sharpe hits 0. If break-even < 20 bps, the strategy
won't survive live trading (ETF spreads + slippage + market impact easily eat 5-15 bps
per leg).

#### Stress 2: Capacity Estimation
Daily strategy turnover × average daily volume × 1% participation rate = approximate
capacity. If capacity < $1M, the strategy is interesting academically but not investable
at scale.

#### Stress 3: Drawdown Duration
MaxDD percentage matters but DURATION matters more. How many consecutive days/weeks is
the strategy underwater? A -10% drawdown lasting 3 months is manageable. A -10% drawdown
lasting 18 months will destroy conviction and lead to abandonment.

#### Stress 4: Tail Risk
Worst 1-day, 1-week, 1-month return. Is there a scenario (flash crash, circuit breaker,
overnight gap) where the strategy loses >20% in a day? For leveraged or short strategies,
this is existential risk.

---

### Phase 4: COMBINE — Build the Portfolio

**Goal:** Combine uncorrelated strategies into a portfolio achieving target Sharpe.

#### Rule 1: Correlation Matrix First
Before combining ANYTHING, compute the full correlation matrix of daily strategy returns.
Group strategies by correlation > 0.5. Each group counts as ONE effective strategy. Pick
the best representative from each group.

#### Rule 2: Inverse-Volatility Weighting (minimum viable)
Weight each strategy proportional to 1/σ where σ is its realized volatility. Simplest
approach that accounts for risk contribution.

#### Rule 3: Maximum Decorrelation
With >4 uncorrelated strategies, use mean-variance optimization (or Choueifaty's max
diversification portfolio) to find weights that maximize portfolio Sharpe. Constraints:
no single strategy > 30% weight, no single mechanism family > 50% weight.

#### Rule 4: Regime Overlay
Final portfolio layer: a regime detector that shifts allocation between strategies based
on current conditions. In high-VIX regimes, overweight mean-reversion and underweight
momentum. In low-VIX regimes, overweight trend-following. This is where cross-asset
regime detection (Mandate A) becomes the meta-strategy.

---

## Mechanism Diversification Checklist

Need strategies from at LEAST 5 of these 8 mechanism families for meaningful
diversification:

### Family 1: Cross-Asset Information Flow
**Status: COMPLETE — stop adding**
Credit → equity lead-lag. 10 strategies passing (LQD/AGG/HYG/VCIT/EMB → SPY/QQQ/EFA).
Pick best 2-3 representatives for the portfolio. Adding more credit-equity variants
increases concentration, not diversification.

### Family 2: Mean Reversion (Pairs/Ratios)
**Status: NEXT PRIORITY**
GLD/SLV ratio, sector rotation reversion, Brent/WTI spread.
GLD-SLV bb=60 std=2.0 scanned (Sharpe ~1.28 in prior session) — needs full lifecycle.
**Why uncorrelated:** Enters/exits based on ratio deviation, independent of credit conditions.

### Family 3: Momentum / Trend Following
**Status: FAILED — needs redesign**
Textbook SMA crossover is dead. Novel approach: risk-adjusted momentum (return/vol) with
Antonacci dual momentum filter (absolute + relative). Apply to asset classes.
SPY-TLT-GLD-BIL rotation failed perturbation (lookback sensitivity). Needs dual-momentum
redesign with absolute momentum gate.
**Why uncorrelated:** Trend signals fire on different timescales (months) vs. credit signals (days).

### Family 4: Volatility Regime Harvesting
**Status: UNTESTED — high priority**
VIX term structure (contango = sell vol, backwardation = buy vol). Use VIX ETPs or SPY
position sizing based on vol regime. One of the most well-documented persistent anomalies.
**Why uncorrelated:** Driven by options market dynamics, not credit or equity direction.

### Family 5: Calendar / Structural Flow Effects
**Status: PARTIALLY TESTED — falsified by 2022 rate hike cycle**
OPEX pinning, turn-of-month effect, CPI release day vol, earnings season vol premium.
F3 (month-end) and F4 (pre-FOMC TLT drift) falsified in 2022-2023. Re-test F1, F2, F5, F6.
**Why uncorrelated:** Driven by calendar dates and mechanical fund flows.

### Family 6: Macro Regime Rotation
**Status: UNTESTED — lower frequency**
Yield curve signals for factor rotation (value/growth), PMI acceleration for small/large
cap, real rate surprises. Monthly rebalancing, genuinely different mechanism.
**Why uncorrelated:** Driven by economic fundamentals, not market microstructure.

### Family 7: Sentiment Contrarian
**Status: UNTESTED — high theoretical appeal**
Crypto Fear & Greed extremes, put/call ratio extremes, VIX spikes as buy signals, Google
Trends attention decay. Fire rarely (5-10 times/year) — noisy individual Sharpe but
valuable in portfolio combination.
**Why uncorrelated:** Driven by behavioral extremes, orthogonal to trend and credit.

### Family 8: Non-Credit Cross-Market Lead-Lag
**Status: 1 PASSING (SOXX-QQQ) — expand**
SOXX→QQQ (passing). Nikkei→SPY, copper→industrial stocks, BTC weekend→Monday equity.
**Why partially correlated:** Some overlap with credit lead-lag (both information flow),
but different leader instruments create different entry timing.

---

## Prioritized Research Roadmap

| Priority | Family | Candidate | Expected Effort | Why Now |
|----------|--------|-----------|----------------|---------|
| 1 | 2 (Mean Reversion) | GLD-SLV full lifecycle | 1 session | Near-pass in prior scan, genuinely orthogonal |
| 2 | 4 (Vol Regime) | VIX term structure contango strategy | 1 session | Well-documented anomaly, zero data cost |
| 3 | 5 (Calendar) | F1 OPEX, F5 pre-earnings, F2 turn-of-quarter | 1 session | Quick to screen, calendar-driven = uncorrelated |
| 4 | 7 (Sentiment) | D8 Fear & Greed, VIX>30 buy signal | 1 session | Alt data, orthogonal mechanism |
| 5 | 3 (Momentum) | Dual-momentum SPY/TLT/GLD/BIL redesign | 1 session | Absol. momentum filter removes lookback sensitivity |
| 6 | 6 (Macro Regime) | N4 yield curve un-inversion on FRED 20yr data | 1-2 sessions | Needs FRED integration, strong macro signal |
| 7 | Portfolio | Combine all passing strategies from Families 2-8 | 1 session | After 3+ new families pass |

---

## Real Alpha vs. Fake Alpha

### Fake Alpha Signatures
- Works in only one time period (especially 2020-2021 or 2023-2024)
- Sharpe collapses when parameters change by 10-20%
- Strategy is in the market >80% of the time (capturing equity beta, not timing)
- Inverted signal is flat, not negative
- Shuffled signal test shows Sharpe within the noise distribution
- MaxDD sits right at the gate threshold (threshold-mining)
- "It works on SPY, QQQ, IWM, EFA..." — same mechanism on correlated assets is not diversification

### Real Alpha Signatures
- Works in at least 2 of 3 distinct market regimes
- Sharpe stable across ±20% parameter perturbation
- Clear economic mechanism another researcher would independently identify
- Inverted signal LOSES money (not just flat)
- Shuffled signal test shows Sharpe in the >99th percentile
- CPCV OOS/IS > 0.8
- Time-in-market < 60% (real timing decision, not just "be long equities")

---

## The Uncomfortable Truths

1. **The credit-equity signal is probably real.** CPCV OOS/IS > 1.0 across 9 variants
   is strong evidence. But it is ONE bet. When it stops working (and it will, temporarily),
   the whole portfolio goes dark.

2. **Daily frequency with free data has a Sharpe ceiling of ~1.5 per strategy.** Higher
   Sharpe requires higher frequency, better data, or both. This is an empirical ceiling,
   not a theoretical one.

3. **The path to portfolio Sharpe 2.0+ is diversification, not optimization.** No single
   strategy will show Sharpe 3.0 at daily frequency. But 4-6 uncorrelated strategies with
   Sharpe 0.7-1.2 combine to portfolio Sharpe 1.5-2.5.

4. **Medallion's Sharpe comes from 10,000+ trades per day across 4,000 instruments.**
   The daily-frequency equivalent is 50+ instruments with 20+ uncorrelated mechanisms.
   That is the direction to push toward — not squeezing more Sharpe from fewer mechanisms.

5. **The biggest risk is not finding alpha — it is sizing it correctly.** A strategy with
   Sharpe 1.0 sized at 2x leverage has the same expected return as Sharpe 2.0, but 4x
   the drawdown risk.

---

## One-Page Decision Framework

For every hypothesis, answer these 7 questions in order. Stop at the first "no."

1. **Is the mechanism different from credit-equity lead-lag?** No → SKIP (you have enough)
2. **Can you explain why it works without saying "correlation" or "historically"?** No → KILL
3. **Does a 2-minute scan show Sharpe > 0.6?** No → KILL
4. **Does the shuffled signal test show >95th percentile?** No → KILL (it's fake)
5. **Does it survive ±20% parameter perturbation?** No → KILL (it's fragile)
6. **Is correlation to the existing portfolio < 0.30?** No → SKIP (redundant, won't move portfolio SR)
7. **Does adding it increase portfolio SR by > 0.05?** No → SKIP (marginal contribution too small)

If all 7 yes: full lifecycle (5-gate framework + 5-stage promotion).

Questions 6 and 7 require running the correlation matrix and marginal contribution formula:
```
ΔSR_P ≈ (SR_k - ρ_{kP} × SR_P) / √(1 + 2×ρ_{kP}×SR_k/SR_P)
```
where ρ_{kP} = correlation of new strategy k with existing portfolio P.

See `docs/research/extreme-sharpe-playbook.md` for the full correlation mathematics.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-26 | Initial framework. Covers 4-phase kill chain, 8 mechanism families, fraud detectors, real vs. fake alpha signatures. |

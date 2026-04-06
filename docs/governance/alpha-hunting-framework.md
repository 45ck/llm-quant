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

**Current state (2026-04-06):** 34 strategies across 20 mechanism families (26 defined, 20 passing), avg ρ=0.186.
Empirical portfolio SR = **2.205** (18 cluster representatives, MaxDD=6.3%). Track D added
D3 TQQQ/TMF ratio MR (Sharpe=2.213, highest individual strategy ever). This was
achieved by expanding from a credit-heavy portfolio (11 strategies, ρ=0.584, SR≈1.35)
to a diversified multi-family portfolio. The key insight: fixed income ratio momentum
(F14/F15), commodity cycle (F11), sector rotation (F12), volatility regime (F13),
disinflation (F19), commodity carry (F18), and dollar-gold regime (F26) signals are
near-zero correlated with credit-equity signals.

Walk-forward HRP validation: 5-fold OOS/IS = **1.597** (no overfitting). Realized
portfolio vol = 3.8% (2.64x scale needed for 10% target).

**Target state:** ACHIEVED — 20 families with avg ρ < 0.20. Marginal value of next
uncorrelated strategy: +0.110 SR. Focus shifts to paper trading validation and promotion.
Commodity/macro/inflation mechanism space is SATURATED (F11-F22 cluster at rho=0.75).
New cluster reps require genuinely different mechanisms (dollar-gold F26 was the latest).

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

Strategies from 20 of 26 mechanism families now passing — well above the minimum 5
families needed for meaningful diversification.

### Family 1: Cross-Asset Information Flow
**Status: SATURATED — stop adding (10 strategies)**
Credit → equity lead-lag. LQD/AGG/HYG/VCIT/EMB → SPY/QQQ/EFA.
Best representatives for portfolio: LQD-SPY (1.250), AGG-SPY (1.145), EMB-SPY (1.005).

### Family 2: Mean Reversion (Pairs/Ratios)
**Status: PASSING (1 strategy: GLD-SLV v4 consensus windows)**
Sharpe=1.197, MaxDD=9.60%. Consensus windows [60,90,120] fixed the bb_window sensitivity
that caused v2/v3 perturbation failures. Near-zero correlation with credit-equity family.

### Family 3: Trend Following / TSMOM
**Status: PASSING (1 strategy: skip-month-tsmom v1)**
Sharpe=1.331, MaxDD=9.92%. Novy-Marx skip-month (t-252 to t-21) vol-scaled on SPY/TLT/GLD/EFA.
Most robust ever: perturbation 5/5 (100%). All TSMOM universe variants (EM, sector, DM) are
ρ=0.89-0.97 with v1 — vol-scaling normalization causes convergence, so don't add more.

### Family 4: Volatility Risk Premium
**Status: TESTED — REJECTED**
VIX contango timing is a LAGGING signal (exits at bottoms, re-enters after recovery).
GARCH-regime near-miss (Sharpe=0.863, DSR=0.89). Needs VXX-short redesign, not equity timing.

### Family 5: Calendar / Structural Microstructure
**Status: PASSING (1 strategy: SPY overnight momentum)**
Sharpe=1.043, MaxDD=8.68%. 10-day rolling avg of overnight returns. Pure equity
microstructure signal, mechanistically distinct from all cross-asset families.

### Family 6: Rate Momentum
**Status: PASSING (3 strategies: TLT-SPY, TLT-QQQ, IEF-QQQ)**
Rate cuts → equity up via discount rate channel. TLT-SPY (0.803), TLT-QQQ (0.935),
IEF-QQQ (0.979). Not credit-driven — pure interest rate mechanism. IEF-QQQ has best
CPCV OOS/IS in project (1.392).

### Family 7: Sentiment Contrarian
**Status: PASSING (1 strategy: behavioral/VIX-spike)**
Sharpe=0.700 — marginal. VIX spike contrarian has sound mechanism but event rarity
(4 events in 4 years) limits statistical confidence. Valuable in portfolio for crisis alpha.

### Family 8: Non-Credit Cross-Market Lead-Lag
**Status: PASSING (1 strategy: SOXX-QQQ)**
Sharpe=0.861, MaxDD=14.4%. First strategy to pass all gates. Semis → tech equity lead-lag.
BTC-SPY tested: REJECTED (perturbation 1/5 — BTC extreme vol makes threshold fragile).

### Family 9: Credit Spread Regime
**Status: PASSING (1 strategy: credit-spread-regime v1)**
Sharpe=0.990, MaxDD=10.75%. HYG/SHY ratio momentum + ratio vs SMA. Three-regime model
(risk_on/risk_off/neutral). NEAR-ZERO correlation with all existing strategies — unique mechanism.

### Family 10: Liquidity
**Status: FALSIFIED**
VolumeBreakout (3 trades/5yr on ETFs) and AmihudRegime (Sharpe=0.557, MaxDD=22.3%) both dead.
AP arbitrage prevents dislocations in liquid ETFs. Would need micro-cap universe.

### Family 11: Commodity Cycle
**Status: PASSING (1 strategy: DBA-commodity-cycle v1)**
Sharpe=1.010, MaxDD=13.50%. DBA 60-day absolute momentum. Inflation→GLD+SPY, Disinflation→SPY.
Most robust ever: perturbation 5/5 (100%).

### Family 12: Sector Rotation
**Status: PASSING (1 strategy: XLK-XLE-sector-rotation v1)**
Sharpe=1.525, MaxDD=11.48%. XLK/XLE ratio momentum + SMA. Growth→QQQ, Inflation→GLD+DBA.
HIGHEST individual Sharpe in portfolio. Pushed portfolio SR above 2.0.

### Family 13: Volatility Regime
**Status: PASSING (1 strategy: vol-regime v2)**
Sharpe=1.270, MaxDD=14.18%. SPY vs GLD 30-day realized vol comparison.
equity_stress→50% GLD, commodity_stress→80% SPY. Perturbation 5/5 (100%).

### Family 14: Curve Shape Momentum
**Status: PASSING (1 strategy: TLT/SHY curve momentum v1)**
Sharpe=1.044, MaxDD=14.28%. TLT/SHY price ratio 30-day momentum.
Flattening→80% SPY, Steepening→80% GLD. Near-zero corr with credit-equity. Perturbation 5/5 (100%).

### Family 15: Real Yield Proxy
**Status: PASSING (1 strategy: TIP/TLT real yield v1)**
Sharpe=1.313, MaxDD=13.34%. TIP/TLT price ratio 20-day momentum.
Loosening→75% SPY, Tightening→50% GLD + 20% SHY. Clusters with F14 (ρ=0.79).

### Family 16: Breakeven Inflation
**Status: PASSING (1 strategy: breakeven-inflation-v1)**
Sharpe=1.068, MaxDD=13.01%. TIP/IEF ratio 30-day momentum as breakeven inflation proxy.
Rising inflation→DBA+GLD+SPY, Falling→SPY+SHY. Different from F15 (TIP/TLT=real yield level vs TIP/IEF=breakeven rate).

### Family 17: Capital Flow (Global Yield)
**Status: PASSING (1 strategy: global-yield-flow-v2)**
Sharpe=0.900, MaxDD=10.90%. TLT/EFA ratio 30-day momentum.
US-preferred→80% SPY, International-preferred→40% EFA + 20% GLD. 100% perturbation stability.

### Family 18: Commodity Carry
**Status: PASSING (1 strategy: commodity-carry-v2)**
Sharpe=1.119, MaxDD=14.39%. USO/DBC 20-day ratio momentum as crude oil backwardation proxy.
Carry→USO+GLD+SPY, Contango→SPY+SHY. Genuinely orthogonal to credit-equity.

### Family 19: Disinflation Signal
**Status: PASSING (1 strategy: tlt-gld-disinflation-v1)**
Sharpe=1.313, MaxDD=8.48%. TLT/GLD ratio 30-day momentum.
Disinflation→80% SPY, Inflation→GLD+DBA+SPY. 100% perturbation stability (PERFECT).

### Family 20: Equity Sector Ratio
**Status: FAILED** — XLF/XLU near-miss (DSR=0.949), others too weak.
Equity sector ratio momentum has real signals (p=0.000) but effect size insufficient (Sharpe 0.63-0.89).

### Family 21: Commodity-Equity Regime
**Status: PASSING (1 strategy: dbc-spy-commodity-equity-v1)**
Sharpe=0.942, MaxDD=10.7%. DBC/SPY ratio 30-day momentum.
Commodity→GLD+DBA+SPY, Equity→SPY. Clusters with commodity/macro family.

### Family 22: Duration Rotation
**Status: PASSING (1 strategy: agg-tlt-duration-rotation-v2)**
Sharpe=0.915, MaxDD=13.4%. AGG/TLT ratio 20-day momentum.
Flattening→SPY+GLD, Steepening→SPY. Clusters with commodity/macro family.

### Family 23: VIX Fear Gauge
**Status: FAILED** — VIX spike contrarian has n=4 events in 5 years (too rare for systematic).

### Family 24: Treasury Auction Cycle
**Status: FAILED** — Not tested; data availability issues with auction schedules.

### Family 25: Sector Dispersion MR
**Status: FAILED** — Sector crowding signal Sharpe=0.509, p=1.0. Timing is random.

### Family 26: Dollar-Gold Regime
**Status: PASSING (1 strategy: dollar-gold-regime-v1)**
Sharpe=0.987, MaxDD=13.9%. UUP/GLD ratio 30-day momentum as purchasing power proxy.
Dollar-strong→60% SPY, Gold-strong→40% GLD + 15% DBA. lookback=30 critical.

---

**NOTE:** F11-F22 and F26 all cluster together (avg ρ=0.75-0.85). Adding more commodity/macro/inflation
ratio momentum strategies provides redundancy but ZERO new decorrelation. New cluster representatives
require genuinely different mechanisms — not more ratio momentum on different instrument pairs.

---

## Prioritized Research Roadmap

Research phase largely complete — 20 of 26 families tested. Focus shifts to paper trading
validation and portfolio optimization.

| Priority | Action | Expected Effort | Status |
|----------|--------|----------------|--------|
| 1 | Paper trading validation (34 strategies, 30-day minimum) | Ongoing | In progress (day ~11 of 30) |
| 2 | HRP optimizer for production weights | 1 session | Built (scripts/portfolio_optimizer.py) |
| 3 | Walk-forward engine for live monitoring | 1-2 sessions | Built (--walk-forward flag) |
| 4 | Volatility targeting overlay | 1 session | Not started |
| 5 | Track D paper trading (D1 TLT-TQQQ + D3 TQQQ/TMF) | 60 days | In progress (day 1-4) |
| 6 | Track C (Niche Arb) production readiness | 2-3 sessions | 4/17 gates done |
| 7 | Novel mechanism discovery (F27+) | Ongoing | Research phase |

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

1. **The credit-equity signal is real.** CPCV OOS/IS > 1.0 across 10 variants is strong
   evidence. But the portfolio is no longer dependent on it — with 20 mechanism families,
   credit-equity (F1) represents only ~15% of portfolio weight via clustering.

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
| 2.0 | 2026-04-01 | Updated to 15 mechanism families (13 passing), 23 strategies, portfolio SR=2.184. Roadmap shifted from research to paper trading validation. |
| 3.0 | 2026-04-06 | Updated to 26 mechanism families (20 passing), 34 strategies, portfolio SR=2.205. Added F16-F26 family descriptions. Track D review: D3 TQQQ/TMF (Sharpe=2.21) highest individual strategy. Commodity/macro saturation noted. |

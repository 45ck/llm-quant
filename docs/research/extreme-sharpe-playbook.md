# The Extreme Sharpe Ratio Playbook
## How Sharpe 3–10+ Actually Works — Mathematics, Mechanisms, and Brutal Constraints

*Reference document for the llm-quant research program.*

---

## The Statistical Ceiling on Arbitrage

Dacheng Xiu (Chicago Booth, 2025 keynote) proved that even with optimal ML-based
strategies, the achievable Sharpe ratio for equity stat arb is capped around **0.7** in
practice — while the theoretical "infeasible" Sharpe (with perfect knowledge) would be
~4.8. The gap is pure estimation error: learning alpha from noisy data introduces losses
that destroy most theoretical edge.

**Who actually achieves Sharpe 3+:**

| Entity | Estimated Sharpe | How | Capacity |
|--------|-----------------|-----|----------|
| Renaissance Medallion (post-2000) | 2.0–6.3 (peaked ~7.5 in 2004) | 150K–300K trades/day, 12.5–20× leverage, ~2-day holding period | ~$10–15B (closed since 1993) |
| Top HFT firms | 5–20+ | Sub-millisecond market making, co-located | Tiny per-strategy |
| Attention Factor Model (2025 paper) | 4.0 gross / 2.3 net | Deep learning on 500 largest US equities, 24-year OOS | Capacity-constrained |

**The iron law**: Sharpe × √Capacity ≈ constant. Extreme Sharpe ratios require extreme
capacity constraints or extreme infrastructure.

---

## The Three Paths to Extreme Sharpe

Every Sharpe 3+ strategy uses one or more of these.

### Path 1: The Breadth Bomb (Grinold's Law on Steroids)

```
IR ≈ IC × √BR
```

This is the entire secret of Renaissance. They don't predict better — they predict more
often. Robert Mercer: "We're right 50.75% of the time, but we're 100% right 50.75% of
the time."

**The math of breadth scaling** (IC = 0.005):

| Bets/year | IR |
|-----------|----|
| 1 | 0.005 |
| 100 | 0.05 |
| 10,000 | 0.5 |
| 100,000 | 1.6 |
| 50,000,000 | 35 (gross — transaction costs destroy most of this) |

At 150,000–300,000 trades/day × 252 days, that is 37–75 million bets/year. The game is
**maximize net IC × √BR after all costs**. RenTech's execution infrastructure is as
valuable as their models.

**What this requires:** intraday trading frequency, thousands of instruments, proprietary
execution algos, $10M+ annually in technology, 50+ PhDs on a single unified model.

### Path 2: The Uncorrelated Strategy Stack

For N uncorrelated strategies each with individual Sharpe SR_i:

```
SR_portfolio = √(Σ SR_i²)
```

**With equal individual Sharpe and pairwise correlation ρ:**

```
SR_portfolio = SR_individual × √(N / (1 + (N-1)×ρ))
```

**The correlation reality check:**

| N strategies (SR=1.0 each) | ρ=0.0 | ρ=0.1 | ρ=0.2 | ρ=0.3 | ρ=0.5 |
|---|---|---|---|---|---|
| 4 | 2.00 | 1.71 | 1.51 | 1.37 | 1.15 |
| 9 | 3.00 | 2.24 | 1.83 | 1.59 | 1.26 |
| 16 | 4.00 | 2.71 | 2.09 | 1.75 | 1.33 |
| 25 | 5.00 | 3.09 | 2.28 | 1.86 | 1.39 |

**At ρ=0.3 (realistic for most quant strategies), 25 strategies each with SR 1.0 yield
combined SR of only 1.86, not 5.0.** This is why Man AHL invests across 400–600 markets
and adds 25–50 new markets per year — they are hunting for genuinely uncorrelated return
streams. Their reported average cross-strategy correlation: ~10%.

**Crisis behavior:** correlations spike in crises. "9 uncorrelated strategies" at ρ=0.1
in normal markets become ρ=0.5 in a crisis, cutting portfolio SR from 2.24 to 1.26
overnight. Trend following has historically negative correlation to equities in crises
(convex "crisis alpha") — this is the most valuable diversifier.

**Strategy categories with genuinely low cross-correlation:**
1. Time-series momentum / trend following — correlated within, low to equity stat arb
2. Volatility risk premium harvesting — low correlation to momentum, negative to equity beta
3. Carry strategies (FX, commodity, fixed income) — moderate to equities in crises
4. Structural/event alpha (index rebalancing, fallen angel arb, tax-loss selling) — idiosyncratic
5. Alternative data signals (satellite, NLP) — potentially low if truly novel
6. Cross-asset relative value (credit vs. equity, cross-currency basis) — low to directional

### Path 3: Leverage on a High-Certainty Edge

Leverage does NOT change the Sharpe ratio. SR_levered = SR_unlevered. What leverage does:
it amplifies the absolute return from a high-certainty, low-volatility, high-frequency
strategy to a useful level.

Medallion's 66% gross annual return at ~32% vol = SR ~2.0. Without leverage: ~3–5% at
~2.5% vol = still SR ~2.0 but unusable absolute return.

**Preconditions for extreme leverage:**
1. Very low drawdowns — Medallion never had a negative year in 31 years
2. Short holding periods — limits overnight/weekend gap risk
3. Massive diversification — thousands of simultaneous positions, no single name > 0.1% NAV
4. Beta ~0 — market-neutral, leverage does not amplify market risk
5. Access to cheap financing — RenTech used basket options via banks (Barclays, Deutsche)
   for 20:1 leverage vs. ~7:1 for competitors

Zuckerman: "An overlooked key is ample and cheap leverage."

---

## The Practical Blueprint for a Systematic Fund

### Phase 1: Build the Strategy Zoo (Months 1–12)

Target 8–15 independently developed strategies across different alpha sources, asset
classes, and time horizons. Portfolio-level Sharpe benefits from diversity more than from
any individual strategy's brilliance.

**Strategy pipeline by category (current state 2026-04-01):**

| Category | Strategies | Status | Cross-category ρ |
|----------|-----------|--------|-------------------|
| F1: Credit-equity lead-lag | LQD/AGG/HYG/VCIT/EMB → SPY/QQQ/EFA | 10 passing, SATURATED | ~0.6 within family |
| F2: Mean reversion (pairs/ratios) | GLD-SLV v4 consensus | 1 passing | ~0.0 to credit |
| F3: Trend following (TSMOM) | Skip-month vol-scaled SPY/TLT/GLD/EFA | 1 passing | ~0.3 to credit |
| F5: Calendar/microstructure | SPY overnight momentum | 1 passing | ~0.4 to credit |
| F6: Rate momentum | TLT-SPY, TLT-QQQ, IEF-QQQ | 3 passing | ~0.1 to credit |
| F7: Sentiment contrarian | VIX spike behavioral | 1 passing (marginal) | ~-0.1 to credit |
| F8: Non-credit lead-lag | SOXX-QQQ | 1 passing | ~0.3 to credit |
| F9: Credit spread regime | HYG/SHY ratio regime | 1 passing | ~0.0 to credit |
| F11: Commodity cycle | DBA abs. momentum | 1 passing | ~0.0 to credit |
| F12: Sector rotation | XLK-XLE ratio regime | 1 passing | ~0.0 to credit |
| F13: Volatility regime | SPY vs GLD realized vol | 1 passing | ~0.1 to credit |
| F14: Curve shape momentum | TLT/SHY ratio momentum | 1 passing | ~0.0 to credit |
| F15: Real yield proxy | TIP/TLT ratio momentum | 1 passing | ~0.0 to credit |
| F4: Volatility risk premium | VIX contango/GARCH | REJECTED | — |
| F10: Liquidity | Volume breakout/Amihud | FALSIFIED | — |

### Phase 2: Maximize Decorrelation (Months 6–18)

1. Measure realized correlation matrix at **daily** frequency (monthly hides intra-month spikes)
2. Apply HRP hierarchical clustering on correlation distance matrix
3. For each cluster: keep highest Sharpe-to-correlation-ratio strategy OR build composite signal
4. Size across clusters: w_i* ∝ SR_i / σ_i (Sharpe-optimal for uncorrelated assets)
5. Monitor correlations in real-time via DCC-GARCH

**Crisis decorrelation enhancers:**
- Trend following: historically negative correlation to equities in crises
- Explicit tail hedges (OTM puts, CDS): negative correlation by construction
- Volatility targeting: automatically reduces exposure as correlations rise

### Phase 3: Capital Efficiency and Leverage (Months 12–24)

**Return stacking (portable alpha):**

On $100M NAV:
- $100M in S&P 500 futures (~$5M margin = 5% of NAV)
- $100M in market-neutral stat arb strategies
- $50M in CTA/trend-following (via futures, ~10% margin)
- Total notional: $250M on $100M NAV = 2.5× gross leverage

**Position-level leverage: fractional Kelly**

```
f_target = (1/4 to 1/3) × (SR_strategy / σ_strategy)
```

For SR=1.0, σ=10%: f_target = (1/4 to 1/3) × 10 = 2.5–3.3× leverage. Use 1/4 to 1/3
Kelly (not half-Kelly) due to estimation error, fat tails, and model uncertainty.

### Phase 4: Continuous Research Pipeline (Ongoing)

The assembly line matters more than any individual strategy.

| Stage | Scale | Gate |
|-------|-------|------|
| Research (ideas/year) | 100 | Economic rationale must exist |
| Initial backtest | 20 survive | t > 2.0 |
| Rigorous testing | 5–8 pass | DSR > 0.95, PBO < 0.10, capacity > $5M |
| Paper trading / incubation | 3–5 | 3–6 months at full intended size |
| Live deployment | 2–3/year | All gates passed |
| Retirement | 2–3 killed/year | Drawdown triggers, alpha decay, capacity exhaustion |

**Live portfolio monitoring thresholds:**
- Rolling 6-month Sharpe: must exceed 0.5 (annualized)
- Correlation to existing strategies: must remain below 0.3
- Generalization ratio (live/backtest): must exceed 0.6
- Max drawdown at 10% vol target: kill at 15%

---

## Realistic Sharpe by Tier

| Tier | Team | AUM | Portfolio SR | Strategies | Leverage | Net Return Target |
|------|------|-----|-------------|------------|----------|-------------------|
| 1: Solo/Small | 1–5 people | $1–50M | **0.8–1.5** | 5–10 | 2–4× | 15–30% |
| 2: Small Fund | 5–20 people | $50–500M | **1.0–2.0** | 15–30 | 3–8× | 10–25% |
| 3: Institutional | 50+ people | $1B+ | **1.0–2.5** | 50–200+ | 5–15× | 8–20% |
| 4: Medallion-class | 100+ PhDs | $10–15B cap | **2.0–7.5** | Hundreds | 12.5–20× | 66% gross |

**The gap between Tier 1 and Tier 4 is not intelligence — it is infrastructure, capital,
talent density, and decades of compounding advantages in all three.**

This project has exceeded Tier 1 range. Current: 23 strategies, 13 families, empirical
portfolio SR=2.184, avg ρ=0.187 — operating at bottom of Tier 2 without leverage.
Achieved via decorrelation (avg ρ from 0.584 → 0.187) rather than leverage.

---

## Strategy Correlation Kill List

**Reject any new strategy failing ANY of these:**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Individual SR (backtest) | > 1.0 | Won't contribute after real-world decay |
| DSR | > 0.95 | Multiple-testing corrected (non-negotiable) |
| PBO | < 0.10 | Overfit probability (non-negotiable) |
| Correlation to existing portfolio | < 0.30 | Redundant — doesn't improve portfolio SR |
| Marginal portfolio SR contribution | > 0.05 | Must meaningfully improve combined SR |
| Capacity | > $5M | Too small to justify infrastructure |
| Max drawdown in backtest | < 2× annual vol | Tail risk too high |
| Economic rationale | Must exist | Pure data mining |
| OOS performance (walk-forward) | > 60% of IS | Decays more than 40% — likely overfit |
| Transaction cost sensitivity | Profitable at 2× costs | Dies at double costs — fragile |

**Two new gates not in current 5-gate system:**
- Correlation to existing portfolio < 0.30 — prevents adding redundant credit-equity variants
- Marginal portfolio SR contribution > 0.05 — ensures each addition actually helps

---

## Key Formulas Reference

**Combined Sharpe (truly uncorrelated strategies):**
```
SR_P = √(Σ SR_i²)
```

**Combined Sharpe (correlated, equal SR, equal weight):**
```
SR_P = SR × √(N / (1 + (N-1)×ρ))
```

**Marginal SR contribution (adding strategy k to portfolio P):**
```
ΔSR_P ≈ (SR_k - ρ_{kP} × SR_P) / √(1 + 2×ρ_{kP}×SR_k/SR_P)
```
where ρ_{kP} = correlation of strategy k with existing portfolio P.

**Optimal weight for Sharpe maximization (uncorrelated):**
```
w_i* ∝ SR_i / σ_i = μ_i / σ_i²
```

**Fundamental Law of Active Management:**
```
IR = IC × √BR × TC
```

**Fractional Kelly leverage (use 1/4 to 1/3, never full):**
```
f* = (1/3) × Σ⁻¹μ
```

**Volatility targeting:**
```
w_t = (σ_target / σ_hat_t) × w_raw
```

**DSR:**
```
DSR = Φ((SR_hat - E[max SR_0]) × √(T-1) / √(1 - skew×SR + ((kurt-1)/4)×SR²))
```

---

## Essential Reading (Priority Order)

1. López de Prado — *Advances in Financial Machine Learning* (2018) — DSR, CPCV, PBO, triple barrier
2. Grinold & Kahn — *Active Portfolio Management* (2000) — Fundamental law, IC, breadth, TC
3. Ilmanen — *Expected Returns* (2011) — Comprehensive alpha taxonomy
4. Isichenko — *Quantitative Portfolio Management* (2021) — Modern systematic fund construction
5. Carver — *Systematic Trading* (2015) and *Advanced Futures Trading* — Practical CTA
6. Pedersen — *Efficiently Inefficient* (2015) — Factor investing, stat arb, macro
7. Harvey, Liu, Zhu — *"...and the Cross-Section of Expected Returns"* (2016) — t > 3.0, multiple testing
8. Bailey & López de Prado — *"The Deflated Sharpe Ratio"* (2014) — Selection bias correction
9. Da, Nagel, Xiu — *"The Statistical Limit of Arbitrage"* (2024) — Upper bounds on feasible SR
10. Man Group — *"Cooking Up Sharpe"* (2025) — Practical multi-strategy portfolio construction

---

## Bottom Line

Sharpe 3+ is not a backtest optimization problem — it is an engineering and diversification
problem:

> **(Many uncorrelated edges) × (Capital efficiency / leverage) × (Execution excellence) × (Continuous research pipeline)**

Each multiplier is necessary. None is sufficient alone. The strategies can each be modest
(SR 0.5–1.5) if you have enough of them and they are genuinely decorrelated. The portfolio
mathematics does the rest.

**Realistic path for a well-resourced prop desk:** 10–20 genuinely uncorrelated strategies
averaging SR ~1.0, average cross-correlation ~0.15, 3–6× leverage via capital-efficient
instruments, research pipeline replacing 20–30% of live strategies annually.

That gets to portfolio SR ~2.0–2.5. Beyond that requires HFT infrastructure, Medallion-
scale talent density, or severe capacity constraints.

Anything claiming Sharpe 5+ in a backtest without those preconditions is almost certainly
overfit. Apply the DSR. The math does not lie.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-26 | Initial playbook. Three paths (breadth, stacking, leverage), correlation math, practical blueprint, tier system, kill list, key formulas, essential reading. |
| 2.0 | 2026-04-01 | Updated strategy pipeline to 15 families (13 passing). Project now at Tier 2 (SR=2.184 without leverage). |

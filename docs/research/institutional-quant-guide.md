# Building a Systematic Quantitative Fund from Scratch

*Reference document for the llm-quant research program. Source: institutional practitioner
synthesis covering Renaissance Technologies, AQR, Man AHL, Two Sigma methodology.*

---

## Where Quantitative Hypotheses Originate

Alpha hypotheses emerge from five distinct sources, each with different signal-to-noise
profiles and decay characteristics. Understanding these origins determines whether a
strategy rests on durable economic foundations or ephemeral data patterns.

**Economic intuition and structural inefficiencies** provide the most durable alpha
sources. The rise of passive investing creates mispricings by making growing portions of
trading indifferent to fundamentals. Central bank policy generates tradeable dislocations
in fixed income and FX. Regulatory constraints force selling — fallen angel downgrades
compel investment-grade mandates to liquidate positions, index deletions create
predictable selling pressure, and capital requirements force banks to shed assets at
non-economic prices. These structural sources persist because they arise from institutional
mandates and regulatory frameworks, not from information asymmetry that arbitrage
eliminates.

**Academic literature mining** offers a vast but contaminated hunting ground. Cochrane's
2011 presidential address first flagged the "factor zoo" problem. Hou, Xue, and Zhang
(2017) compiled 447 anomaly variables; using their q-factor model, 93% of alphas were
insignificant at t < 3. Yet Jensen, Kelly, and Pedersen (2023) found a false discovery
rate of only ~1%, and Jacobs and Müller (2020) documented sustained post-publication
factor returns outside the US. The practical resolution: a parsimonious set of robust
factors survives scrutiny — value, momentum, quality, carry, and low volatility — but
hundreds of claimed anomalies are noise.

**Behavioral finance anomalies** offer perhaps the most theoretically grounded alpha
source because human cognitive biases are structural, not informational. Key mechanisms:
underreaction (conservatism bias, slow information diffusion, disposition effect) and
overreaction (herding, overconfidence, feedback trading). Moskowitz, Ooi, and Pedersen
(2012) showed return persistence for 1-12 months followed by partial reversal is
consistent with initial underreaction and delayed overreaction — providing the behavioral
foundation for both momentum and mean-reversion strategies.

**Market microstructure** exploits the mechanics of trading itself. Options market makers'
delta-hedging creates predictable mean-reversion at daily frequency. Order flow imbalances
generate short-lived repeatable patterns. Lopez de Prado's VPIN quantifies toxic flow in
real time. Note: microstructural alpha has decayed significantly as markets electronified.

**Alternative data** represents the fastest-growing source (52.1% CAGR through 2030).
Satellite imagery, NLP sentiment, credit card transactions, geolocation, web scraping, and
workforce analytics all feed modern alpha pipelines. J.P. Morgan found funds using
alternative data experienced 3% higher annual returns.

---

## The Taxonomy of Alpha

Alpha sources decompose along two axes: **cross-sectional** (relative rankings within an
asset class at a point in time) versus **time-series** (absolute predictions for individual
assets over time).

### Core Style Premia

**Carry** — expected return if prices remain unchanged:
- FX: domestic minus foreign interest rate (uncovered interest parity violation)
- Commodities: roll yield from backwardation
- Fixed income: yield + roll-down
- Ilmanen: carry strategies have "most consistently outperformed buy-and-hold" across
  multiple asset classes

**Momentum** — operates in two forms:
- Cross-sectional (Jegadeesh and Titman 1993): buy past winners, sell past losers,
  12-minus-1-month lookback
- Time-series (Moskowitz, Ooi, Pedersen 2012): long assets with positive trailing
  12-month returns, short those with negative. TSMOM across 58 instruments showed
  convex crisis-alpha — strongest during extreme market moves.

**Value** — exploits mean-reversion in relative pricing: long cheap assets (low P/B,
P/E), short expensive. Asness, Moskowitz, Pedersen (2013): value and momentum are
negatively correlated, enhancing diversification when combined.

**Volatility Risk Premium (VRP)** — persistent gap between implied and realized volatility.
S&P 500 implied vol averages 9 percentage points above realized vol. 2-month 10% OTM puts
cost roughly 3.5x their fair value. The risk profile: steady small gains punctuated by
infrequent large losses concentrated in worst times ("picking up pennies before a
steamroller").

**Structural and flow-based alpha** — predictable non-economic trading: index rebalancing,
forced selling from credit downgrades, tax-loss selling, pension fund rebalancing, options
hedging flows.

---

## How Top Quantitative Funds Generate Ideas

**Renaissance Technologies (Medallion Fund)**: ~66% gross annual returns since 1988, no
losing year. Edge: "Be right 50.75% of the time, 100% of the time." 150,000-300,000
trades/day, ~2-day average holding period, 12.5-20x leverage. ~275 employees, ~90 PhDs
(mathematicians, physicists, astronomers, computational linguists — none from Wall Street).
Deliberately capacity-constrained ($10-12B, closed since 1993). Single unified model, full
code transparency across the team.

**AQR Capital Management** ($226B AUM): Cliff Asness, Fama-French student. Four core
style premia: value, momentum, carry, defensive. "Craftsmanship alpha" from superior
implementation. Asness: "Our most potent weapon against data mining is the out-of-sample
test. Vying in importance is insisting on an economic rationale for why something works."
Machine learning now powers ~20% of trading signals.

**Man Group/AHL** ($168.6B AUM): CTA evolved from simple moving-average crossovers to
ML-augmented multi-asset systematic trading across 400-600 markets. Allocation: momentum
(25%), carry (25%), fundamental models (25%), alternative models including mean-reversion
and seasonality (25%). Key thesis: "the richest vein of research is new markets" — adds
25-50 new markets per year, finding low-correlation alpha in alternative instruments.

**Two Sigma** ($60B+ AUM): positioned as a "technology company" built on alternative data
mastery. **Citadel**: multi-strategy pod model with individual P&L accountability.
**DE Shaw**: pioneered quantitative approaches in the late 1980s.

### When Economic Rationale vs. Data Mining is Acceptable

For large, slow-moving factors (value, momentum, carry, quality): economic rationale
required. AQR has "walked away from good-looking factors we don't trust."

Data mining without prior rationale is acceptable ONLY when:
1. Properly controlled for multiple testing via DSR and PBO
2. Out-of-sample validation performed across independent datasets
3. Evidence survives across geographies and asset classes
4. Transaction costs are fully modeled
5. All trials are recorded (the strategy graveyard requirement)

Lopez de Prado's Three Laws:
- **First Law**: "Backtesting is not a research tool. Feature importance is."
- **Second Law**: "Backtesting while researching is like drink driving."
- **Third Law**: "Every backtest must be reported with all trials involved in its production."

---

## Grinold's Fundamental Law of Active Management

The central equation governing systematic strategy design:

```
IR ≈ IC × √BR

Extension: E(R_A) = TC × IC × √BR × σ_A
```

Where:
- **IR** = Information Ratio (excess return / tracking error)
- **IC** = Information Coefficient (correlation between forecasts and actual returns;
  typically 0.02-0.10 in practice)
- **BR** = Breadth (number of independent forecasts per year)
- **TC** = Transfer Coefficient (efficiency of translating insights into positions;
  TC=1 unconstrained, <1 with constraints)

**The implication:** A quant model with IC=0.05 across 3,000 stocks: IR=2.74. A single
analyst with IC=0.10 covering 1 stock: IR=0.10. Breadth dominates. "Doubling IC is worth
quadrupling breadth." This is why Renaissance trades 100,000+ times daily with tiny IC
per trade.

---

## Rigorous Backtesting and the Multiple Testing Crisis

### Multiple Hypothesis Testing Corrections

Traditional p < 0.05 (t > 2.0) is catastrophically inadequate for hundreds of strategies.

| Method | Formula | Conservative? |
|--------|---------|---------------|
| Bonferroni (FWER) | α* = α/m | Most conservative |
| Holm-Bonferroni (step-down) | Compare p(k) with α/(m-k+1) | Better power than Bonferroni |
| Benjamini-Hochberg (FDR) | Find largest k where p(k) ≤ (k/m) × q | Least conservative |

**Harvey, Liu, Zhu practical recommendation**: any newly proposed factor requires t > 3.0
(not 2.0) to be taken seriously. This threshold increases over time as more factors are
tested. Under all three corrections, roughly half of 316 published factors lose
significance.

### The Deflated Sharpe Ratio (DSR)

Bailey and Lopez de Prado (2014) correct for selection bias under multiple testing and
non-normal return distributions.

**Probabilistic Sharpe Ratio (PSR):**
```
PSR[SR*] = Φ((SR_hat - SR*) × √(T-1) / √(1 - skew×SR_hat + ((kurt-1)/4)×SR_hat²))
```

**Expected Maximum SR under null of zero skill:**
```
E[max{SR_n}] = √V[SR_hat] × [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N×e))]
```
where γ ≈ 0.5772 (Euler-Mascheroni constant), N = number of independent trials.

**DSR = PSR where benchmark = expected maximum SR.** If DSR < 0.95, the observed Sharpe
is likely a statistical artifact of multiple testing.

**Minimum Track Record Length (MinTRL):**
```
MinTRL = (1 - skew×SR_hat + ((kurt-1)/4)×SR_hat²) × (z_{1-α} / (SR_hat - SR*))²
```
For SR=0.5 annually with normal returns at 95% confidence: ~60+ months of data required.

**Critical note**: all SR values in DSR must be non-annualized (per-period).
Convert: SR_period = SR_annual / √(periods_per_year)

```python
import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015328606

def expected_max_sr(sr_var, num_trials):
    return np.sqrt(sr_var) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1.0/num_trials) +
        EULER_GAMMA * norm.ppf(1 - 1.0/(num_trials * np.e))
    )

def deflated_sharpe_ratio(sr_hat, sr_var, num_trials, T, skew, kurt):
    sr0 = expected_max_sr(sr_var, num_trials)
    num = (sr_hat - sr0) * np.sqrt(T - 1)
    den = np.sqrt(1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat**2)
    return norm.cdf(num / den)

def min_track_record_length(sr_hat, sr_benchmark, skew, kurt, alpha=0.05):
    z = norm.ppf(1 - alpha)
    return (1 - skew*sr_hat + ((kurt-1)/4)*sr_hat**2) * (z/(sr_hat - sr_benchmark))**2
```

### Realistic Sharpe Ratio Expectations

**A backtested Sharpe > 2.5 without HFT-level infrastructure is a strong signal of
overfitting.** The capacity-Sharpe tradeoff is fundamental.

| Strategy type | Typical Sharpe | Key characteristics |
|---|---|---|
| Passive equity (S&P 500) | 0.3-0.5 | Long-run historical benchmark |
| Equity factor (single factor) | 0.2-0.8 | Value, momentum, or quality alone |
| Diversified factor portfolio | 0.5-0.8 | Multi-factor combination |
| Trend following / CTA | 0.4-1.2 | Positive skewness, crisis alpha |
| Statistical arbitrage | 1.0-3.0 | Highly capacity-constrained |
| Volatility selling | 1.0-2.0 | Misleading due to negative skew |
| Systematic macro | 0.3-0.8 | Large capacity |
| Market making / HFT | 3.0-10+ | Extremely capacity-constrained |
| Multi-strategy diversified | 0.5-1.5 | Depends on correlation structure |

RenTech Medallion achieves ~Sharpe 2-3 but is limited to $10-15B, closed to outside
investors, and runs 12-20x leverage. Their scalable funds (RIEF, RIDA) achieve much lower.

### Cross-Validation for Financial Time Series

Standard k-fold cross-validation fails because random shuffling destroys temporal order
and creates information leakage.

**Lopez de Prado's Combinatorial Purged Cross-Validation (CPCV):**

1. **Purging**: remove training observations whose label horizons overlap with test set
2. **Embargoing**: add a buffer (~1% of total bars) after test set boundary
3. **Combinatorial paths**: partition T observations into N groups, test C(N,k) splits,
   generate φ[N,k] = k × C(N,k) / N independent backtest paths

For N=6, k=2: 15 splits producing 5 independent backtest paths.

CPCV produces a **distribution** of OOS performance metrics, not a single estimate.
This enables robust inference via DSR and PBO estimation.

---

## Kill Criteria and Strategy Evaluation

### Statistical Kill Criteria

| Criterion | Threshold | Basis |
|-----------|-----------|-------|
| t-statistic (floor) | > 2.0 | Absolute minimum |
| t-statistic (deployment) | > 2.5 | Live consideration |
| t-statistic (new factors) | > 3.0 | Harvey, Liu, Zhu |
| DSR | > 0.95 | Bailey and Lopez de Prado |
| PBO | < 0.10 | Bailey et al. |
| MinTRL | Satisfied | Track record ≥ MinTRL formula |

### Drawdown-Based Kill Switches

Man AHL's research (Harvey, Van Hemert et al. 2020): drawdown-based rules detect
degradation faster than total-return rules when strategy skill decays over time.

| Level | Trigger | Action |
|-------|---------|--------|
| Watch | Drawdown > 1σ (~10% at 10% vol) | Tighten monitoring |
| Warning | Drawdown > 1.5σ (~15%) | Reduce allocation 50% |
| Kill | Drawdown > 2σ (~20%) | Halt, full investigation |
| Hard kill | Drawdown > 3σ (~30%) | Immediate termination |

**CUSUM test** for regime-change detection:
```
S+_t = max(0, S+_{t-1} + (x_t - μ₀ - k))
```
where k = σ/2 and alarm fires when S+_t > h. Applied to rolling information ratios.

### Transaction Cost Modeling

**Almgren-Chriss model**: execution price = S_t + g(v_t) + h(v_t) where g = permanent
impact, h = temporary impact.

**Empirical square-root law**: Impact ∝ σ × √(Q/V), where Q = order size, V = daily
volume. Strategy capacity reached when marginal impact equals expected alpha/2.

### Economic and Practical Kill Criteria

- **Economic rationale**: explain why it works (risk premia, behavioral bias, or structural
  inefficiency). Strategies without rationale face ~50%+ OOS decay vs 10-15% for those
  with strong foundations.
- **Regime dependency**: test across at least 3 distinct regimes (bull/bear/crisis). Must
  work in at least 2 of 3.
- **Capacity**: if capacity < $50M, may not justify infrastructure costs.
- **Crowding risk**: CFM research shows OOS Sharpe decay of 33% for the largest 1,000
  stocks vs 20% for the largest 500.

---

## Portfolio Construction and Risk Management

### Kelly Criterion and Position Sizing

Multi-asset Kelly:
```
f* = Σ⁻¹μ
```
Equivalent to unconstrained mean-variance with risk aversion λ=1.

**Practitioners universally use fractional Kelly.** Half-Kelly delivers ~75% of expected
return with ~50% of volatility. Full Kelly: 40% probability of wealth dropping to 40% of
starting capital at some point. The reasons for fractional Kelly:
- Estimation error in μ and Σ makes over-betting likely
- Fat tails make ruin risk exceed Gaussian assumptions
- Kelly is asymptotically optimal but not optimal in finite time
- Parameter uncertainty shrinks optimal fraction toward zero

### Portfolio Construction Methods

**Risk Parity / Equal Risk Contribution (ERC):**
```
RC_i = w_i(Σw)_i / √(w'Σw)
```
Solves: min Σ_i Σ_j [w_i(Σw)_i - w_j(Σw)_j]² subject to 1'w=1, w≥0.
Spinu (2013) provides a strictly convex reformulation for large universes.

**Black-Litterman:**
- Implied equilibrium returns: Π = δΣw_mkt (δ ≈ 2.5-3.5)
- Posterior: E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q]

**Hierarchical Risk Parity (HRP)** (Lopez de Prado 2016):
Three steps: (1) correlation distance matrix + hierarchical clustering; (2) quasi-
diagonalization via dendrogram leaf order; (3) recursive bisection with inverse-variance
weights. Monte Carlo simulations show lower OOS variance than minimum-variance MVO, with
greater robustness to estimation error. Avoids matrix inversion entirely.

**Ledoit-Wolf shrinkage:**
```
Σ_shrunk = δF + (1-δ)S
```
F = structured target (constant-correlation or identity). For large N/T ratios, optimal
shrinkage is typically ~80% toward target. Nonlinear shrinkage (2017) uses Random Matrix
Theory to optimally transform all eigenvalues individually.

### Factor Exposure Management

Standard neutralization constraints for long-short equity:
```
w'β_mkt = 0            (beta neutral)
w'1_sector_s = 0       (sector neutral, each sector s)
w'B_k = 0              (style factor neutral, each factor k)
```
Use realized beta from daily returns over 1 year (not 5-year monthly beta). Research shows
long/short strategies benefit from sector neutralization ~78% of the time.

### Volatility Targeting and Dynamic Leverage

```
w_t^managed = (σ_target / σ_hat_t) × w_t^original
```
Cap leverage at 1.5-2.0x. Moreira and Muir (2017): volatility doesn't predict returns
but strongly predicts future volatility. A mean-variance investor should reduce risk
exposure by ~50% after a 1σ variance shock. Typical CTA targets: 10-15% annualized vol.

Volatility forecasting methods (increasing sophistication):
1. Max of trailing 10/20/30-day realized vol
2. EWMA: σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1} with λ=0.94
3. GARCH(1,1)
4. Realized volatility from intraday data

**DCC-GARCH** (Engle 2002) for time-varying correlations:
```
Q_t = (1-a-b)Q_bar + a(z_{t-1}z'_{t-1}) + b×Q_{t-1}
```
Enables real-time detection of rising correlations that signal regime shifts.

### Tail Risk Metrics

**CVaR (Expected Shortfall)** — coherent risk measure (Basel III/FRTB):
Under Gaussian: CVaR_α = μ_p - σ_p × φ(z_α)/α

**Maximum Drawdown and Calmar ratio**: capture path-dependent risk that VaR/CVaR miss.
For SR=0.5 at 10% vol over 10 years: ~62% probability of a 20% drawdown, ~25% for 30%.

---

## Overfitting Defenses

### Degrees of Freedom and the Strategy Graveyard

Bailey et al. (2015): with only 5 years of daily data, no more than 45 strategy variations
should be tested, or the best result achieves Sharpe ≥ 1.0 purely by chance even if the
true Sharpe is zero.

**Three types of overfitting** (Rob Carver):
- **Explicit**: automated parameter optimization (grid search, neural nets) — manageable
- **Implicit**: manual selection of rules/parameters based on backtest results
- **Tacit**: using knowledge from prior backtests to guide current design — hardest to
  control, the "time machine" problem

**The strategy graveyard**: track ALL tested strategies, not just winners. Every abandoned
idea, every parameter configuration must be recorded. Without this, the denominator of
the multiple testing correction is unknown. Lopez de Prado: this is non-negotiable.

### Researcher Degrees of Freedom

Simmons, Nelson, Simonsohn (2011): the "garden of forking paths" — at each decision point
(asset selection, indicator choice, parameter ranges, entry/exit rules, position sizing),
choices expand the effective number of trials. The solution: pre-registration of research
hypotheses, strict separation of research and backtest phases, honest accounting of all
forking paths.

CFM's approach: "conceive an idea before looking at data, build with standard statistical
toolkit with no freedom for parameter choice, accept or reject."

### Probability of Backtest Overfitting (PBO)

Bailey, Borwein, Lopez de Prado, Zhu — uses Combinatorially Symmetric Cross-Validation:
1. Collect N strategy configurations with T return observations
2. Partition into S equal subsets, form all C(S, S/2) combinations
3. For each: find IS-optimal configuration n*, measure its OOS rank
4. Compute logit: λ = ln(rank(n*, OOS) / (N - rank(n*, OOS)))
5. PBO = proportion of combinations where IS-optimal underperforms OOS median

PBO near 0: robust. PBO > 0.5: severe overfitting.

---

## Data Quality, Execution, and Live Monitoring

### Preventing Data Biases

**Survivorship bias**: overstates annual returns by 0.9-4% by excluding failed/delisted
companies. Use CRSP (gold standard for US equities), Compustat with point-in-time.

**Look-ahead bias** sources: using financial statements before reporting date, revised vs.
as-reported data, index reconstitution using current constituents. Prevention: strict
point-in-time data architecture — decisions on rebalancing day using only information
available at that timestamp.

### Expected Performance Degradation

Expected degradation from backtest to live: **15-40%** depending on strategy type.

| Strategy type | Expected OOS decay |
|---|---|
| Single-factor | 30-40% |
| Multi-factor | 20-30% |
| Purely statistical patterns | 50%+ |
| Strong theoretical foundations | 10-15% |

Track the "generalization ratio" (live return / backtest return) and use paired statistical
tests to detect significant divergence.

### Alpha Decay

Maven Securities estimates US market alpha decay costs 5.6% annually, European markets
9.9%, both accelerating.

Typical strategy lifespans:
- HFT strategies: days to weeks
- Momentum-based algos: 3-6 months
- Swing/position systems: 6-18 months
- Macro/fundamental strategies: 1-3 years

The fund that survives is not the one with the best single strategy but the one with the
best **research pipeline** — continuously generating, testing, incubating, deploying, and
killing strategies.

### Key Implementation Libraries

| Library | Purpose |
|---------|---------|
| mlfinlab (Hudson & Thames) | AFML algorithms: triple-barrier labeling, meta-labeling, CPCV, DSR, bet sizing, fractional differentiation |
| skfolio | Portfolio optimization with CombinatorialPurgedCV |
| PyPortfolioOpt | Black-Litterman, HRP, Ledoit-Wolf, efficient frontier |
| vectorbt | High-performance vectorized backtesting |
| statsmodels | Time series, GARCH, hypothesis testing |
| pypbo | Probability of backtest overfitting |
| cvxpy | Convex optimization for portfolio construction and risk parity |
| QuantLib | Derivatives pricing |

---

## Key Principles Summary

1. **t > 3.0 for new factors** (Harvey, Liu, Zhu) — not 2.0. This threshold increases as
   more factors are tested across the industry.

2. **DSR > 0.95 and PBO < 0.10** — the irreducible minimum before any strategy sees
   live capital. These correct for the multiple testing problem that makes most backtests
   misleading.

3. **CPCV produces distributions, not point estimates** — the correct output is a
   distribution of OOS Sharpe ratios across 15 independent paths, not a single number.

4. **The capacity-Sharpe tradeoff is the iron law.** Sharpe 2+ is achievable only at
   severely constrained capacity. At $50B+, 0.5-1.5 is realistic.

5. **Alpha decays.** The half-life of a quantitative strategy is months, not decades. The
   research pipeline matters more than any individual strategy.

6. **The strategy graveyard is mandatory.** Every tested variant must be recorded. Without
   it, the DSR denominator is unknown and all statistical claims are fraudulent.

7. **Breadth over IC.** IR ≈ IC × √BR means diversification across many uncorrelated
   strategies with modest individual Sharpe is the path to high portfolio Sharpe.

8. **HRP over equal-weight** for portfolio construction. Avoid matrix inversion;
   hierarchical clustering provides robust weights.

9. **Volatility targeting is a free lunch.** Scaling positions by 1/σ̂ delivers consistent
   risk contribution without material return sacrifice.

10. **Economic rationale is required for sustainable alpha.** Strategies that work by
    luck look identical to strategies that work by mechanism — until OOS.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-26 | Initial document covering hypothesis origins, alpha taxonomy, top fund approaches, DSR/PBO/CPCV mathematics, kill criteria, portfolio construction, overfitting defenses, live monitoring. |

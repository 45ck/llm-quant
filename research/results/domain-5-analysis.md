# Domain 5 Analysis: Regime Detection, Momentum & Inflation Allocation

**Date**: 2026-03-24
**Briefs reviewed**: 5.1 (4-Factor HMM), 5.2 (Cross-Asset TSMOM), 5.3 (Inflation 2x2 Matrix)
**Current system**: VIX level + TLT/SHY yield spread proxy + SPY SMA trend heuristic

---

## Key Findings

### 1. 4-state HMM fits better statistically but 2-state often wins in portfolio performance

Guidolin & Timmermann (2007) found four states (crash, slow growth, bull, recovery) necessary to capture the joint stock-bond return distribution. BIC analysis by Nguyen & Nguyen (2021) confirmed 4-state optimality across multiple information criteria. However, Fons et al. (2021) tested 200 random 5-asset portfolios and found **portfolio performance was significantly better with 2 states** despite worse statistical fit. Practitioners overwhelmingly agree: Kritzman, Page & Turkington (2012), First Sentier, Nystrup et al. (2015-2019), and Shu & Mulvey (2024) all use 2-state models. The tension is real -- statistical fit and investment performance are different objectives. For our $100k portfolio with a 15% max drawdown constraint, the parsimony argument dominates.

### 2. Credit spreads DO lead equity vol but the relationship is unstable

Gilchrist & Zakrajsek (2012) showed the excess bond premium (EBP) predicts recessions 12 months ahead. Wellington Management (2022) documented credit widening leading VIX during 2022's drawdown. The Merton framework provides theoretical grounding: credit markets sit closer to refinancing risk and reprice before equity vol. **However, this relationship broke down for ~2 years post-COVID** when Fed intervention normalized spreads while VIX stayed elevated. The lead-lag is regime-dependent, not constant. For our system, credit spreads are valuable as an *additional* stress signal but not a standalone timing tool.

### 3. TSMOM is robust but crash prediction at 2-4 weeks is unproven

Moskowitz, Ooi & Pedersen (2012) showed positive TSMOM across all 58 tested instruments. Multi-lookback blended portfolios (1, 3, 12 month) achieve gross Sharpe of 1.3-1.8 due to low cross-correlations between horizons (Hurst et al. 2013, Baltas & Kosowski 2013). Volatility scaling nearly doubles Sharpe (Barroso & Santa-Clara 2015). Daniel & Moskowitz (2016) showed crashes cluster at bear-to-bull transitions. **But no academic evidence supports crowding indicators predicting crashes 2-4 weeks ahead** -- Lou & Polk (2021) and Baltas (2019) find crowding signals operate at 6-24 month horizons. The strongest signal is blended multi-lookback TSMOM with vol scaling, not crash prediction.

### 4. 2x2 inflation matrix has strong foundations but stagflation detection is hard

The Bridgewater All-Weather framework (launched 1996) and FTSE Russell's Balanced Macro (2025) both validate four economically distinct quadrants. Balanced Macro achieved Sharpe 0.58 vs 0.51 for 60/40, with max drawdown of -16.7% vs -29.8%. **Stagflation is the hardest quadrant and the most valuable for capital preservation** -- Cliffwater (2022) calls predicting its onset "nearly impossible." Gold hedges inflation only above ~6.6% annualized (threshold study, ScienceDirect 2022). The incremental Sharpe from adding the inflation dimension is real but modest: ~0.07-0.23 depending on methodology.

### 5. DXY as regime factor is genuinely novel

No published paper jointly models regime detection using a multivariate HMM with credit spreads, VIX, yield curve, AND DXY momentum. BIS (2024) shows the dollar is a "dominant driver" of EME capital flows with distinct information content (correlation with FCI = 0.45 vs VIX-FCI = 0.62). Engel & Hamilton (1990) documented Markov-switching in exchange rates. This represents an opportunity for original contribution, but also means no validation exists -- proceed with caution and rigorous out-of-sample testing.

---

## Recommendations (Priority-Ordered)

### R1. Upgrade regime detection from heuristic to 2-state HMM

**What to implement**: Replace the current VIX + TLT/SHY + SPY trend heuristic with a 2-state (risk_on / risk_off) Hidden Markov Model. Use weekly observations with ~130-day lookback (Nystrup et al. 2019). Observation variables: VIX level, yield curve slope, SPY momentum. Keep Gaussian emissions (the regime-switching structure itself generates fat tails). Retain the existing heuristic as a fallback/comparison.

**Expected impact**: Better state persistence (fewer regime flips), probabilistic confidence scores instead of binary thresholds, more disciplined position sizing during transitions. Literature suggests Sharpe improvements of 0.2-0.5 vs static allocation, though marginal gain over a well-designed heuristic is modest.

**Complexity**: Medium. Requires `hmmlearn` or custom implementation. The 2-state model is well-understood and there are reference implementations.

**Files that change**:
- `src/llm_quant/brain/context.py` -- add `_classify_regime_hmm()` alongside existing heuristic functions
- `src/llm_quant/brain/models.py` -- add `regime_confidence: float` to `MarketContext`, keep `MarketRegime` enum (risk_on/risk_off/transition maps to HMM states + low-confidence threshold)
- `src/llm_quant/data/indicators.py` -- ensure weekly aggregation is available for HMM fitting
- `config/settings.toml` -- add `[regime]` section with HMM parameters
- New: `src/llm_quant/regime/hmm.py` -- HMM fitting, state inference, confidence scoring

### R2. Add credit spread (BAMLC0A0CM) as a leading stress indicator

**What to implement**: Fetch the ICE BofA US Corporate Index OAS from FRED (series BAMLC0A0CM). Compute 20-day z-score. Surface it in the MarketContext as a stress leading indicator. Do NOT use it as a standalone regime switch -- the relationship is unstable (post-COVID breakdown). Use it as supplementary context: when credit z-score > 1.5 AND VIX is still low, flag "silent stress" in the decision prompt.

**Expected impact**: Early warning for stress episodes where credit leads equity vol. Wellington (2022) documented this exact pattern during the 2022 drawdown. Most valuable for our 15% drawdown constraint -- catching stress 2-4 weeks early could prevent 3-5% of drawdown.

**Complexity**: Low. FRED API is well-documented. Main work is data pipeline integration.

**Files that change**:
- `src/llm_quant/data/fetcher.py` -- add FRED data source for BAMLC0A0CM
- `src/llm_quant/brain/context.py` -- add `_get_credit_spread()` function, include in MarketContext
- `src/llm_quant/brain/models.py` -- add `credit_spread_zscore: float` field to MarketContext
- `config/settings.toml` -- add FRED API key config, credit spread thresholds
- `config/templates/decision.md.j2` -- surface credit spread in prompt

### R3. Implement multi-lookback blended TSMOM signals

**What to implement**: For each asset in the universe, compute TSMOM signals at 1-month (21 days), 3-month (63 days), and 12-month (252 days) lookbacks. Blend with equal weight (literature shows low cross-correlation makes this near-optimal). Apply volatility scaling using 126-day realized variance targeting 12% annualized vol per position. This replaces or supplements the current SMA crossover + RSI signal logic.

**Expected impact**: Strongest evidence base of any signal in the research (140+ years, 67 markets). Multi-lookback blending captures distinct return continuation phenomena. Vol scaling nearly doubles Sharpe (Barroso & Santa-Clara 2015). Individual instrument Sharpe ~0.3-0.4, but portfolio-level Sharpe >1.0 through diversification.

**Complexity**: Medium. The math is straightforward (return sign * vol-scaled position). Integration with existing signal framework requires care to avoid signal conflicts.

**Files that change**:
- New: `src/llm_quant/signals/tsmom.py` -- multi-lookback TSMOM computation, vol scaling
- `src/llm_quant/data/indicators.py` -- add trailing return computations at 21/63/252 day windows
- `src/llm_quant/brain/context.py` -- include TSMOM signals in MarketContext
- `src/llm_quant/brain/models.py` -- add TSMOM signal fields or a new signal model
- `config/settings.toml` -- TSMOM parameters (lookbacks, vol target, blending weights)

### R4. Add inflation regime overlay (2x2 matrix) -- after R1 is stable

**What to implement**: Classify the inflation environment using TIPS breakevens (5y and 5y5y forward) and CPI surprise direction. Create a 2x2 matrix overlay: {rising/falling growth} x {rising/falling inflation}. Use this to tilt sector allocations -- overweight commodities/energy in reflationary boom, overweight bonds in deflationary bust, overweight gold/TIPS in stagflation. **Depends on having the risk regime (R1) right first** -- the inflation overlay modifies allocations within a regime, not the regime itself.

**Expected impact**: Incremental Sharpe improvement of ~0.07-0.20. Primary value is drawdown reduction during inflationary regimes (2022 validation: stock-bond positive correlation destroyed 60/40). Most valuable when inflation is elevated (>3% annualized).

**Complexity**: Medium-High. Requires additional data sources (TIPS breakevens from FRED, CPI release calendar). Stagflation detection is acknowledged as the hardest quadrant. Detection lag of 2-4 months during regime shifts means this is a strategic tilt, not a tactical signal.

**Files that change**:
- New: `src/llm_quant/regime/inflation.py` -- inflation regime classification
- `src/llm_quant/data/fetcher.py` -- add FRED sources for TIPS breakevens (T5YIE, T5YIFR)
- `src/llm_quant/brain/context.py` -- include inflation regime in MarketContext
- `src/llm_quant/brain/models.py` -- add `InflationRegime` enum and field
- `config/templates/decision.md.j2` -- surface inflation quadrant in prompt

---

## Key Debates

### 2-state vs 4-state HMM: what does the evidence actually say?

The evidence creates a genuine paradox. **Statistical fit**: 4-state wins on BIC/AIC/HQC across multiple studies (Nguyen & Nguyen 2021, Guidolin & Timmermann 2007). **Portfolio performance**: 2-state wins significantly (Fons et al. 2021, 200 portfolio combinations). The resolution: BIC penalizes parameters but rewards likelihood, so it favors models that describe the data well -- but describing data well and making money are different objectives. More states means more parameters, more estimation error, more regime flipping, and more turnover costs. For our system: **start with 2-state, measure out-of-sample performance for 3+ months, then test whether 3 or 4 states improve risk-adjusted returns net of turnover costs.** The "transition" state in our current MarketRegime enum could be implemented as a low-confidence zone between the two HMM states rather than a third state.

### Should we add credit spreads as a LEADING indicator or just additional context?

The research supports "additional context with a leading bias" rather than a standalone leading indicator. The credit-to-equity-vol lead is real during stress (Wellington 2022, Gilchrist & Zakrajsek 2012) but broke down for 2 years post-COVID. **Recommendation**: surface the credit spread z-score in the decision prompt and flag divergences (credit widening while VIX calm = "silent stress"), but do NOT use it as a hard regime switch trigger. Let the LLM weigh it alongside other signals. This respects the unstable nature of the relationship while still capturing its information content. If we later move to a formal HMM, credit spreads could become an observation variable -- but even then, the instability means the HMM's state inference would naturally downweight it when the relationship breaks down.

### TSMOM: use multi-lookback blended (strongest evidence) or single 12M?

**Multi-lookback blended is the clear winner.** The 12-month lookback is the strongest single signal (Sharpe ~0.38 vs 0.29 for 1-month, per Hurst et al. 2013), but the three horizons have **low cross-correlation**, meaning they capture different phenomena -- short-term flow effects (1M), intermediate positioning (3M), and fundamental trend (12M). Blending produces Sharpe above 1.0 at the portfolio level (Baltas & Kosowski 2013). Furthermore, Levine & Pedersen (2016) proved TSMOM and moving-average crossovers are mathematically equivalent as linear filters, so our existing SMA crossover signals are already a form of TSMOM -- the upgrade is adding the multi-frequency dimension and vol scaling. **Implementation**: equal-weight blend of 1M/3M/12M signals, each vol-scaled to 12% annualized using 126-day realized variance.

---

## Cross-Cutting Insights for System Design

1. **Volatility scaling is the single most impactful technique** across all three research domains. It nearly doubles momentum Sharpe (Barroso & Santa-Clara), is foundational to risk parity, and implicitly present in the 2x2 framework. Our system should treat it as infrastructure, not an optional enhancement.

2. **Regime detection adds value primarily through avoiding catastrophic drawdowns**, not capturing upside. FTSE Russell Balanced Macro cut max drawdown from -51.1% to -16.7% while accepting lower absolute returns. This aligns perfectly with our capital-preservation-first mandate and 15% max drawdown constraint.

3. **Simplicity outperforms complexity in portfolio performance.** This is the strongest recurring theme. 2-state beats 4-state HMMs. Static risk parity captures most of the dynamic benefit. Vol-scaled momentum with simple weighting nearly doubles Sharpe without crash prediction. Guard against model complexity that improves in-sample fit but degrades out-of-sample.

4. **Post-publication alpha decay is real.** McLean & Pontiff (2016): 26% lower out-of-sample, 58% lower post-publication. TSMOM profitability has declined since 2008 (Huang et al. 2020). Our return targets (Sharpe >0.8) should account for this -- the literature Sharpe ratios above 1.0 are gross, pre-cost, and often in-sample.

---

## Beads Issues to Create

1. **Implement 2-state HMM regime detection** (Priority 2, Task) -- Replace VIX/TLT/SHY heuristic with 2-state HMM using weekly observations, ~130-day lookback, Gaussian emissions. Keep heuristic as fallback. Files: `src/llm_quant/regime/hmm.py` (new), `brain/context.py`, `brain/models.py`, `config/settings.toml`.

2. **Add credit spread (BAMLC0A0CM) as stress indicator** (Priority 2, Task) -- Fetch from FRED, compute 20-day z-score, surface in MarketContext. Flag "silent stress" when credit z >1.5 and VIX calm. Files: `data/fetcher.py`, `brain/context.py`, `brain/models.py`, `config/templates/decision.md.j2`.

3. **Implement multi-lookback blended TSMOM signals** (Priority 2, Task) -- Compute TSMOM at 21/63/252 day lookbacks, equal-weight blend, vol-scale to 12% using 126-day realized variance. Files: `signals/tsmom.py` (new), `data/indicators.py`, `brain/context.py`, `config/settings.toml`.

4. **Add volatility scaling infrastructure** (Priority 2, Task) -- Compute 126-day realized variance for all universe assets. Apply inverse-variance position sizing. This is foundational for TSMOM and risk parity. Files: `data/indicators.py`, `trading/executor.py`, `risk/manager.py`.

5. **Design inflation regime overlay** (Priority 3, Task) -- Research spike: fetch TIPS breakevens (T5YIE, T5YIFR) from FRED, prototype 2x2 classification. Blocked on R1 (risk regime must be stable first). Files: `regime/inflation.py` (new), `data/fetcher.py`.

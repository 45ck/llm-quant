# Research Synthesis & Implementation Roadmap

## Executive Summary

Across 10 completed research briefs spanning market microstructure, alternative data, risk management, and strategy design, the evidence converges on a clear message: **our system's core architecture is directionally sound, but several key assumptions were wrong in ways that matter.** The biggest surprise is that simplicity dominates complexity at nearly every decision point -- 2-state regime models beat 4-state HMMs in portfolio performance, conditional (not continuous) volatility targeting outperforms, and ATR(14) with Wilder smoothing is measurably inferior to EMA-based alternatives with a VIX overlay. Execution costs at $100k are negligible for everything except international ETFs and small forex positions, meaning our engineering effort should go to signal quality and risk management, not execution optimization. The strongest evidence-backed upgrades are: (1) replacing Wilder ATR(14) with EMA-ATR plus conditional VIX sizing, (2) implementing multi-lookback TSMOM as the primary momentum signal, (3) adding credit spreads as a leading stress indicator for regime detection, and (4) adopting the Bridgewater-style 2x2 growth/inflation allocation matrix for strategic tilts. COT positioning data is real but decaying -- useful as a secondary overlay for gold and crude oil, not a primary signal. Crash prediction at 2-4 week horizons is not supported by the literature; crowding signals operate on 6-24 month horizons. The biggest remaining gap is the untested ML/DL research (briefs 4.1-4.3), which the evidence suggests we should deprioritize relative to the proven techniques above.

---

## Top 10 Actionable Findings (Ranked by Impact)

### 1. Replace Wilder ATR(14) with EMA-ATR(10-14) + Conditional VIX Overlay
**Source**: Briefs 1.2, 3.2 | **Impact**: High | **Confidence**: Very High

Wilder's smoothing has a 9.55-day half-life that creates dangerous blind spots during regime transitions (COVID: ATR could not reflect >50% of new volatility until 10+ days in). EMA-based ATR is ~2x faster (5-day half-life) -- a free improvement. Combine with conditional VIX sizing using 126-day percentile rank: full size in 20th-80th percentile, reduce 25-30% above 80th, increase 10-15% below 20th. Bongaerts et al. (2020, FAJ) showed conditional targeting consistently beat continuous scaling.

### 2. Implement Multi-Lookback TSMOM as Primary Momentum Framework
**Source**: Brief 5.2 | **Impact**: High | **Confidence**: High

Time-series momentum is the single most robustly validated alpha source in the systematic trading literature (140+ years, 67+ markets). Diversified TSMOM achieves gross Sharpe of 1.3-1.8. The critical insight: individual instrument Sharpe is only 0.29-0.38, but combining 1-month, 3-month, and 12-month lookbacks with volatility scaling pushes portfolio Sharpe above 1.0. Volatility scaling (126-day realized variance, targeting 12% annualized) is essential -- Kim et al. (2016) argue it IS the source of TSMOM profits.

### 3. Add Credit Spread as Leading Regime Indicator
**Source**: Brief 5.1 | **Impact**: High | **Confidence**: High

The Gilchrist-Zakrajsek excess bond premium predicts recessions 12 months ahead and leads VIX by 2-4 weeks during stress episodes (Wellington 2022). IG credit spreads widening while VIX stays calm is the classic early warning. This is immediately implementable with our existing data pipeline -- use HYG-TLT spread or Baa-Aaa spread as an additional regime input alongside VIX and yield curve.

### 4. Adopt 2x2 Growth/Inflation Allocation Matrix for Strategic Tilts
**Source**: Brief 5.3 | **Impact**: Medium-High | **Confidence**: High

FTSE Russell's Balanced Macro framework achieved Sharpe 0.58 vs. 0.51 for 60/40 with max drawdown of -16.7% vs. -29.8%. The key: risk parity WITHIN quadrants (where correlations are stable), NO correlation assumptions ACROSS quadrants. For our system, this means classifying the current environment into one of four quadrants (reflationary boom, disinflationary boom, stagflation, deflation) and tilting allocations accordingly. Energy (correlation 0.6 to inflation) and gold (inflation beta 3.37) are the primary stagflation hedges.

### 5. ATR-Based Risk Budgeting with Buffer Rules
**Source**: Brief 3.2 | **Impact**: Medium-High | **Confidence**: Very High

This is the single most robustly supported risk management technique. Harvey et al. (2018, Man AHL) showed vol targeting reduces vol-of-vol by 61% and max drawdown by 35-50%. Implement with: 2x ATR stop-loss (the Turtle standard), rebalance only when position deviation exceeds 20-25% of target, shorter ATR periods for crypto (7-10 days). The current system's inverse-vol weighting is naive risk parity -- consider two-layer sizing (ATR within asset classes, correlation-aware across classes).

### 6. Use MOC for US ETFs, Overlap Windows for International/FX
**Source**: Brief 1.1 | **Impact**: Medium | **Confidence**: Very High

At $100k, market impact is literally zero. MOC orders are the correct default for US equity and fixed income ETFs. The ONE exception with real money at stake: trade international ETFs during European-US overlap (8:00-11:30 AM ET) -- stale NAV costs are 20-50 bps per round-trip otherwise. Forex execution during London-NY overlap (8:00 AM-12:00 PM ET), avoid WM/Reuters fix window. Bitcoin ETFs (IBIT at 3-9 bps round-trip) dominate direct exchange access.

### 7. CVaR Constraints as Binding Risk Limit (Not Primary Sizing Tool)
**Source**: Brief 3.3 | **Impact**: Medium | **Confidence**: Medium-High

CVaR is coherent, can be minimized via LP, and improves Sharpe by ~0.2 when used as a constraint. But 252-day historical simulation uses only ~13 tail observations -- dangerously few. Must augment with Filtered Historical Simulation (GARCH-filtered residuals), extend lookback to 500+ days, and include named stress scenarios (COVID, 2022 rates, crypto winter). Use proportional scaling (CPPI-style), not binary blocking, to avoid whipsaw.

### 8. COT Data as Secondary Overlay for Gold, Crude, and FX Only
**Source**: Brief 2.3 | **Impact**: Medium-Low | **Confidence**: Medium

COT positioning is statistically real but economically modest and decaying. Portfolio-level COT strategies fail (Dreesmann et al. 2023). Best use: contrarian on commercial positioning at extremes (20/80 thresholds, 156-week lookback) for gold (COMEX 088691), crude oil, and EUR/JPY. Signal horizon 4-6 weeks. Apply Friday-published data starting Monday open. Do NOT apply COT signals to USO (broken post-April 2020). Fix CFTC commodity codes (gold=088691, not 084691).

### 9. Downgrade 4-State HMM to 2-State Risk-On/Risk-Off
**Source**: Brief 5.1 | **Impact**: Medium | **Confidence**: Medium

Four-state HMMs fit better statistically (BIC) but 2-state models deliver significantly better portfolio performance (Fons et al. 2021). Statistical Jump Models outperform HMMs due to better state persistence. The marginal advantage of HMM over well-designed heuristic rules is modest. Start with a 2-state model using credit spreads + VIX + yield curve at weekly frequency with ~130-day lookback. The DXY momentum factor as a distinct regime channel is untested but theoretically motivated -- worth exploring but not in Phase 1.

### 10. Embed Asymmetric Weekend Crypto Risk Awareness
**Source**: Brief 1.3 | **Impact**: Low-Medium | **Confidence**: Medium

Negative weekend crypto returns significantly predict Monday equity declines, while positive weekend returns have no portfolio-level benefit. This asymmetric spillover strengthened after LUNA (May 2022). Practical implication: reduce crypto position sizes going into weekends, or at minimum, account for the asymmetric tail in risk calculations. Bitcoin momentum signals persist 1-6 days. The Monday Asia Open Effect (Sunday 7 PM NY) is a real pattern worth monitoring.

---

## Revised Priority Matrix

| Rank | Item | Original Priority | Revised Priority | Rationale |
|------|------|-------------------|------------------|-----------|
| 1 | 3.2 ATR Risk Budgeting | High | **Critical** | Most robustly validated technique across all research. Foundation for everything else. |
| 2 | 1.2 VIX-Adjusted ATR Sizing | High | **Critical** | EMA-ATR + conditional VIX overlay is a strict improvement over current Wilder ATR(14). Free upgrade. |
| 3 | 5.2 TSMOM Signals | High | **Critical** | 140+ years of evidence, gross Sharpe 1.3-1.8 when diversified across lookbacks. Primary alpha source. |
| 4 | 5.1 Regime Detection (Credit Spreads) | High | **High** | Credit spread leading indicator is well-validated. But downgrade from 4-state HMM to 2-state model. |
| 5 | 5.3 Inflation-Risk Matrix | Medium | **High** | Bridgewater 2x2 framework has strong recent validation. Stagflation protection is the key value-add. |
| 6 | 1.1 Execution Timing Rules | Medium | **Medium** | International ETF overlap windows matter (20-50 bps savings). US ETFs: MOC is correct, no optimization needed. |
| 7 | 3.3 CVaR Constraints | High | **Medium** | Theoretically sound but estimation is fragile with 252 days. Needs FHS augmentation. Lower priority than vol sizing. |
| 8 | 3.1 Dynamic Correlation Limits | High | **Medium-Low** | Rolling window correlations are the "worst performer" (Engle 2002). Trade-blocking mechanism is untested. DCC-GARCH is feasible but complex. |
| 9 | 2.3 COT Positioning | Medium | **Low** | Signal has decayed substantially. Portfolio-level strategies fail. Useful only as secondary overlay for gold/crude/FX. |
| 10 | 1.3 Crypto/FX Session Patterns | Medium | **Low** | Interesting alpha concentration patterns but narrow application. Weekend asymmetry is a risk management input, not a signal. |

### Items NOT Researched (Original Briefs Without Results)

| Item | Original Priority | Should We Pursue? | Rationale |
|------|-------------------|-------------------|-----------|
| 2.1 FOMC Calendar Effects | Medium | **Defer** | Our TSMOM framework already captures macro regime shifts. Calendar effects are well-documented but small edge. |
| 2.2 Reddit/Social Sentiment | Low | **Drop** | Evidence from Brief 5.2 shows crowding signals operate on 6-24 month horizons. Social sentiment is noise at our daily frequency. |
| 4.1 ML Feature Selection | High | **Defer** | Brief 5.1 showed Statistical Jump Models outperform HMMs. Worth exploring later, but simpler proven methods first. |
| 4.2 DL Price Prediction | Medium | **Drop** | Literature consistently shows simplicity beats complexity in portfolio performance. Our $100k scale does not justify this engineering. |
| 4.3 Reinforcement Learning | Low | **Drop** | Overfitting risk is extreme with RL on financial data. Bailey et al. (2016): with 5 years of daily data, no more than 45 variations before Sharpe>=1.0 appears by chance. |

---

## Implementation Roadmap (Phases)

### Phase 1: Quick Wins (1-2 weeks)

These require minimal code changes, high confidence, and immediate risk-reduction benefit.

**1.1 Switch ATR from Wilder to EMA smoothing**
- Change: In indicator calculation, replace `ATR_t = (13/14) * ATR_{t-1} + (1/14) * TR_t` with `ATR_t = (12/14) * ATR_{t-1} + (2/14) * TR_t` (EMA alpha = 2/(n+1))
- Benefit: Half-life drops from 9.55 days to ~5 days. Faster regime response.
- Risk: None. Strict improvement.
- Files: `src/llm_quant/data/` indicator pipeline

**1.2 Add conditional VIX sizing overlay**
- Change: Before position sizing, check VIX 126-day percentile rank. Scale position size: 100% if VIX in 20th-80th pctile, 70-75% if >80th pctile, 110-115% if <20th pctile.
- Benefit: Bongaerts et al. show this consistently improves Sharpe with low turnover.
- Risk: Low. Discrete adjustments, not continuous scaling.
- Files: `src/llm_quant/risk/manager.py`, position sizing logic

**1.3 Implement execution timing rules**
- Change: Add asset-class-aware execution windows to decision prompts: MOC for US ETFs, overlap window (8:00-11:30 AM ET) for international ETFs, London-NY overlap for forex.
- Benefit: 20-50 bps round-trip savings on international ETF trades.
- Risk: None. Just timing guidance in prompts.
- Files: `config/` prompt templates, `scripts/execute_decision.py`

**1.4 Fix CFTC commodity codes**
- Change: Gold COMEX = 088691 (not 084691), Silver COMEX = 084691 (not 084651). Use TFF report for FX, not Disaggregated.
- Benefit: Correct data if COT integration is implemented later.
- Risk: None.
- Files: Config/data pipeline where COT codes are defined

**1.5 Add 20-25% rebalancing buffer rule**
- Change: Only rebalance when position weight deviates >20% from target. Prevents unnecessary turnover.
- Benefit: Harvey et al. (2018) showed vol-targeted portfolios have 2.4-4.4x higher turnover; buffers control this.
- Risk: None. Standard practice.
- Files: `src/llm_quant/trading/executor.py`

### Phase 2: Core Upgrades (2-4 weeks)

Substantive improvements with strong evidence requiring moderate engineering effort.

**2.1 Multi-lookback TSMOM signal generation**
- Change: Compute TSMOM signals at 1-month, 3-month, and 12-month lookbacks for all 39 assets. Combine with equal weight. Scale each position by inverse 126-day realized variance targeting 12% annualized vol.
- Benefit: This is the primary alpha source. Diversified TSMOM Sharpe 1.3-1.8 gross.
- Dependency: Phase 1.1 (EMA-ATR) for vol estimation.
- Files: `src/llm_quant/data/` indicators, `src/llm_quant/brain/` signal logic

**2.2 Credit spread regime indicator**
- Change: Add HYG-TLT spread (or Baa-Aaa if available) as a regime input alongside VIX and yield curve slope. Compute 126-day percentile rank. Flag "stress approaching" when credit spread percentile exceeds 80th while VIX is still below 25.
- Benefit: 2-4 week lead time on equity stress. Gilchrist-Zakrajsek EBP predicts recessions 12 months ahead.
- Dependency: None (data already in universe).
- Files: `src/llm_quant/data/`, regime classification logic

**2.3 2-State regime model (replace heuristic thresholds)**
- Change: Implement 2-state risk-on/risk-off model using VIX + credit spread + yield curve at weekly frequency. Use ~130-day lookback. Consider adaptive VIX thresholds based on percentiles (~17.8, 23.1) instead of fixed 20/25.
- Benefit: Fons et al. (2021) showed 2-state HMM outperforms 4-state in portfolio returns. RegimeFolio study showed adaptive thresholds improve Sharpe by 24% vs fixed.
- Dependency: Phase 2.2 (credit spread data).
- Files: `src/llm_quant/brain/` regime logic

**2.4 Inflation quadrant classification**
- Change: Classify current environment into one of four quadrants using: TIPS breakevens (5y5y forward preferred), CPI surprise direction, commodity momentum (GSCI or energy ETF trend). Tilt allocations: overweight commodities/energy/gold in stagflation, overweight nominal bonds in deflation, overweight equities in disinflationary boom.
- Benefit: Max drawdown reduction from -29.8% to -16.7% (FTSE Russell). Stagflation protection.
- Dependency: Phase 2.3 (regime model as base layer).
- Files: `src/llm_quant/brain/` allocation logic, prompt templates

**2.5 Backtest realistic execution cost assumptions**
- Change: Embed cost model in all backtests: 5 bps US ETFs, 15-30 bps international/EM ETFs, 8-15 bps fixed income, 8-10 bps crypto ETF, 15-20 bps small forex. Apply 1.5-2x backtest-to-live multiplier.
- Benefit: Quantopian showed backtest Sharpe has R^2 < 0.025 with live performance. Conservative costs prevent overfitting.
- Dependency: None.
- Files: Backtesting infrastructure

### Phase 3: Advanced Features (4-8 weeks)

More complex additions that depend on Phase 1-2 foundation.

**3.1 DCC-GARCH correlation monitoring**
- Change: Replace rolling window correlations with DCC-GARCH (2-step: 39 univariate GARCH + 2 scalar DCC parameters). Runs in 30-60 seconds. Use for diversification-weighted sizing across asset classes, NOT for trade blocking.
- Benefit: Engle (2002) showed rolling windows are the worst correlation estimator. DCC corrects for Forbes-Rigobon heteroskedasticity bias.
- Dependency: Phase 2.1 (TSMOM signals need correlation-aware weighting).
- Risk: Medium. Requires careful implementation. Consider EWMA (lambda 0.94-0.97) as simpler alternative.

**3.2 CVaR constraints with Filtered Historical Simulation**
- Change: Implement CVaR at 95% confidence using FHS (GARCH-filtered residuals rescaled to current vol). Extend lookback to 500+ days. Include named stress scenarios (COVID, 2022 rates, crypto winter, SVB). Proportional scaling enforcement (CPPI-style).
- Benefit: Catches tail scenarios that vol sizing misses. Krokhmal et al. (2002) showed combining multiple risk constraint types improves OOS performance.
- Dependency: Phase 3.1 (covariance estimates for portfolio CVaR).
- Risk: Medium-high. Estimation error is significant with limited tail observations.

**3.3 COT overlay for gold, crude, FX**
- Change: Implement COT Index (min-max normalization, 156-week lookback, 20/80 thresholds) for COMEX gold (088691), WTI crude (067651), EUR (099741), JPY (097741). Contrarian on commercial positioning at extremes. Signal horizon 4-6 weeks. Apply Friday data starting Monday open.
- Benefit: Modest secondary signal. Tornell & Yuan (2012) show EUR speculator peak/trough methodology predicts 0.48%/week.
- Dependency: None, but low priority.
- Risk: Low alpha, signal decay documented. Exclude USO.

**3.4 Momentum crash protection via regime awareness**
- Change: During bear-to-bull transitions (detected by Phase 2.3 regime model), reduce momentum exposure or switch to shorter lookback windows. Daniel & Moskowitz (2016): 14 of 15 worst momentum returns occurred when past 2-year market return was negative and contemporaneous return was positive.
- Benefit: Avoids the -73% to -92% momentum crash tail.
- Dependency: Phase 2.1 (TSMOM) + Phase 2.3 (regime model).

### Deferred / Needs More Research

**4-state HMM with DXY factor**: The DXY momentum channel is theoretically motivated (BIS 2024 shows dollar is dominant EME capital flow driver, correlation to VIX only 0.45) but untested in published literature. Defer until 2-state model is operational and validated.

**Short-term crash prediction (2-4 week horizon)**: No robust academic evidence supports this timeline. Crowding signals (Lou & Polk 2021, Baltas 2019) operate on 6-24 month horizons. Reallocate engineering effort to longer-horizon crowding indicators or drop entirely.

**Reddit/social sentiment (Brief 2.2)**: Given that crowding at weekly-monthly horizons is already questionable, social media sentiment at daily frequency is almost certainly noise. Drop.

**ML/DL approaches (Briefs 4.1-4.3)**: The consistent finding across all research is that simplicity beats complexity in OOS portfolio performance. With only 3-5 major crises in 20 years, we have ~5 data points for risk parameter fitting. ML/DL amplifies overfitting risk. Defer indefinitely unless a specific, narrowly scoped application emerges (e.g., Statistical Jump Models as a regime detector per Brief 5.1).

**FOMC calendar effects (Brief 2.1)**: Well-documented in the literature but marginal edge for a daily-frequency macro strategy. Our TSMOM framework already captures post-FOMC momentum shifts. Low priority.

---

## Cross-Domain Dependencies

```
Phase 1 (Foundation)
  |
  +-- 1.1 EMA-ATR -----> 2.1 TSMOM (needs vol estimates)
  |                         |
  +-- 1.2 VIX Overlay ---> 2.3 2-State Regime Model
  |                         |     |
  +-- 1.3 Exec Timing      |     +--> 3.4 Crash Protection
  |                         |     |
  +-- 1.5 Buffer Rules      |     +--> 2.4 Inflation Quadrant
  |                         |
  +-- 2.2 Credit Spread ----+---> 2.3 2-State Regime
                            |
                            +--> 3.1 DCC-GARCH (correlation-aware weighting)
                            |         |
                            |         +--> 3.2 CVaR (needs covariance)
                            |
                            +--> 3.3 COT Overlay (independent, low priority)
```

**Critical path**: EMA-ATR (1.1) --> TSMOM signals (2.1) --> Correlation-aware weighting (3.1) --> CVaR constraints (3.2)

**Parallel path**: Credit spread indicator (2.2) --> 2-State regime model (2.3) --> Inflation quadrant (2.4) --> Crash protection (3.4)

**Independent**: Execution timing rules (1.3), buffer rules (1.5), COT overlay (3.3)

---

## What We Got Wrong

### 1. ATR(14) is adequate for position sizing
**Reality**: Wilder's smoothing has a mathematically provable 9.55-day half-life that makes it unacceptably slow for regime transitions. During COVID, it could not have reflected >50% of the new volatility regime until 10+ days in. EMA-based ATR is strictly superior -- same concept, ~2x faster adaptation. We were using a 1978-era smoothing formula when a free improvement was available.

### 2. Execution timing matters at this scale
**Reality**: At $2,500 per position in SPY (daily volume >$30B), our order is <0.00001% of daily volume. Market impact is literally zero. MOC is the correct default and no algorithmic execution can be justified. The ONE exception: international ETFs during the overlap window, where stale NAV costs are real (20-50 bps round-trip). We overestimated execution complexity.

### 3. 4-state regime models are better than 2-state
**Reality**: Statistical fit (BIC) favors 4 states. Portfolio performance favors 2 states (Fons et al. 2021). Statistical Jump Models outperform HMMs entirely (Shu & Mulvey 2024). The marginal HMM advantage over well-designed heuristic rules is modest and context-dependent. We were optimizing for in-sample fit, not out-of-sample returns.

### 4. COT positioning is a meaningful alpha source
**Reality**: The signal is real but decaying. Portfolio-level COT strategies fail (Dreesmann et al. 2023). The most comprehensive tests show the signal has been substantially arbitraged. Best use is as a secondary overlay for gold, crude, and FX at extreme readings -- not a standalone alpha generator. We overestimated its contribution.

### 5. Crash prediction works at 2-4 week horizons
**Reality**: No robust academic evidence supports this timeline. The most rigorous crowding research (Lou & Polk 2021, Baltas 2019) operates on 6-24 month horizons. The composite crowding index we hypothesized would require significant adaptation beyond published research. We were conflating "regime detection" (which works) with "crash timing" (which does not at short horizons).

### 6. Correlation-based trade blocking reduces drawdowns 20-30%
**Reality**: The trade-blocking mechanism is novel and has NO direct empirical validation in the literature. Rolling window correlations are the "worst performer" among dynamic correlation methods (Engle 2002). The claimed drawdown reduction numbers are untested. Better to use correlations implicitly through diversification-weighted sizing than through a blocking mechanism.

### 7. Continuous volatility targeting is optimal
**Reality**: Bongaerts et al. (2020, FAJ) showed continuous vol targeting FAILS to consistently improve performance and can INCREASE drawdowns (realized-to-target vol ratio of 1.16). Conditional targeting -- adjusting only at regime extremes -- consistently enhances Sharpe with low turnover. We assumed more granular = better; the evidence says discrete beats continuous.

---

## Remaining Gaps

### Briefs Without Results

| Brief | Topic | Still Worth Pursuing? | Priority | Rationale |
|-------|-------|----------------------|----------|-----------|
| 2.1 | FOMC Calendar Effects | Maybe later | Low | TSMOM already captures post-FOMC momentum. Calendar effects are well-documented but marginal. Could add as a simple binary filter (avoid new positions on FOMC days) with minimal effort. |
| 2.2 | Reddit/Social Sentiment | No | Drop | Crowding signals operate on 6-24 month horizons. Social media sentiment at daily frequency is noise. The NLP engineering required does not justify the expected information content. |
| 4.1 | ML Feature Selection | Maybe later | Low | Could be useful for narrowly scoped applications like regime detection (Statistical Jump Models). But the literature consistently shows simplicity beats complexity in OOS performance. Not worth the overfitting risk at this stage. |
| 4.2 | DL Price Prediction | No | Drop | Every piece of evidence from the 10 completed briefs points away from complex models. Quantopian's study of 888 strategies: backtest Sharpe R^2 < 0.025 with live performance. More complexity = more overfitting = worse live results. |
| 4.3 | Reinforcement Learning | No | Drop | With 3-5 major crises in 20 years, we have ~5 data points for risk parameter fitting. RL would memorize these. Bailey et al. (2016): with 5 years of daily data, no more than 45 strategy variations should be tested before Sharpe >= 1.0 appears by chance. |

### Genuine Knowledge Gaps the Research Revealed

1. **DXY momentum as a distinct regime factor**: No published paper jointly models VIX + credit spreads + yield curve + DXY momentum in a multivariate regime model. BIS data suggests DXY carries distinct information (correlation to FCI is 0.45 vs VIX at 0.62). This is an opportunity for original contribution but not a Phase 1 priority.

2. **Optimal vol scaling window for crypto**: Literature uses 126-day realized variance calibrated to equities. Crypto vol structure is different (annualized vol 60-80% vs 15-20% for equities). The right lookback window for crypto vol estimation is undetermined.

3. **Stagflation detection lead time**: The literature agrees stagflation is the hardest quadrant and "predicting its onset is nearly impossible" (Cliffwater 2022). TIPS breakevens have 2-4 month detection lag. We need to determine whether commodity momentum or energy sector relative strength can provide faster signals.

4. **Backtest-to-live degradation for our specific strategy**: Quantopian found 30-50% theoretical return erosion and 1.5-2x drawdown multiplier. We need to run our system through this filter once TSMOM and regime detection are implemented, using CPCV (not simple walk-forward) to estimate overfitting probability.

5. **Transaction cost sensitivity for small forex positions**: IBKR's $2 minimum commission makes $2,500 forex trades cost 16 bps round-trip -- disproportionately expensive. We need to determine whether forex positions should be sized larger (reducing diversification) or dropped from the active trading universe.

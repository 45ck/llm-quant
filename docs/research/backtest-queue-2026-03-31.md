# Priority Backtest Queue
**Date:** 2026-03-31
**Analyst:** Macro Strategist Agent
**Purpose:** Rank top hypotheses by expected portfolio SR contribution (diversification value)

---

## Calibration Note

**Backtested Sharpes consistently halve in live trading** (TSMOM: 1.0 -> 0.6, macro momentum:
1.2 -> 0.7). All expected Sharpe values below include a column for the 50% live haircut.
A hypothesis needs expected backtest SR >= 1.0 to realistically deliver SR >= 0.5 live.

**Current portfolio:** 11 strategies, avg rho = 0.584, portfolio SR ~1.35.
Adding ONE strategy with SR 1.0 and rho 0.05 lifts portfolio SR from 1.35 to ~1.97.
The marginal value of low-rho strategies is enormous.

---

## Top 15 Priority Queue

Ranked by estimated marginal portfolio SR contribution (delta_SR_P), which accounts for
both standalone SR and correlation to the existing F1-heavy portfolio.

| Rank | Hypothesis | Family | Expected Backtest SR | Live SR (50% haircut) | Est. rho to Portfolio | delta_SR_P | Data Avail | Lifecycle Effort |
|------|-----------|--------|---------------------|-----------------------|-----------------------|------------|------------|-----------------|
| **1** | **H3.1 Vol-Scaled TSMOM** | F3 | 0.95 | 0.48 | **0.05** | **+0.62** | YES (yfinance) | 1 session |
| **2** | **H3.5 Skip-Month TSMOM** | F3 | 1.00 | 0.50 | **0.05** | **+0.62** | YES | 1 session |
| **3** | **H2.5 Sector ETF Kalman Pairs** | F2 | 0.95 | 0.48 | **0.05** | **+0.60** | YES | 1 session |
| **4** | **F9-H1 Commodity Carry Contango** | F9 | 0.80 | 0.40 | **0.05** | **+0.50** | Partial (ETF proxy) | 1-2 sessions |
| **5** | **H7.2 RSI-2 Multi-Asset Contrarian** | F7 | 1.05 | 0.53 | **0.15** | **+0.49** | YES | 1 session |
| **6** | **H5.8 Overnight Gap Mean-Reversion** | F5 | 0.65 | 0.33 | **0.00** | **+0.46** | YES | 1 session |
| **7** | **H7.1 VIX Percentile Spike Reversion** | F7 | 0.90 | 0.45 | **0.10** | **+0.45** | YES | 1 session |
| **8** | **H4.2 VRP-SPY Timing** | F4 | 0.95 | 0.48 | **0.15** | **+0.44** | YES (VIX + SPY) | 1 session |
| **9** | **H6.4 Macro Momentum Barometer** | F6 | 0.95 | 0.48 | **0.15** | **+0.44** | YES | 1 session |
| **10** | **H4.7 SKEW Tail Risk Timing** | F4 | 0.83 | 0.41 | **0.05** | **+0.43** | YES (^SKEW) | 1 session |
| 11 | H3.3 Multi-TF TSMOM Ensemble | F3 | 0.93 | 0.46 | 0.05 | +0.58 | YES | 1 session |
| 12 | H4.4 GARCH Regime Equity Sizing | F4 | 0.93 | 0.46 | 0.15 | +0.42 | YES | 1 session |
| 13 | F9-H2 Bond Rolldown Carry | F9 | 0.70 | 0.35 | 0.25 | +0.27 | Partial (FRED) | 1-2 sessions |
| 14 | H2.4 EFA-EEM Developed/Emerging Pairs | F2 | 0.75 | 0.38 | 0.10 | +0.38 | YES | 1 session |
| 15 | H7.6 Sector Dispersion Regime | F7 | 0.90 | 0.45 | 0.10 | +0.43 | YES | 1 session |

---

## Why This Ranking (Not the Research-Recommended Order)

The task description included a research-recommended order starting with H9.3 (cross-asset
carry rotation), H10.2 (volume-liquidity breakout), etc. I've reranked based on the
portfolio correlation analysis findings:

**Key differences:**

1. **F3 (TSMOM) hypotheses rank #1-2**, not #4. F3 has the lowest expected correlation
   to the existing F1-heavy portfolio (rho ~0.05) AND is the only family with documented
   negative crisis correlation. This makes F3 the single most valuable family for
   portfolio construction. The research ranking placed H3.6 at #4 -- I've promoted
   H3.1 and H3.5 above it because they have stronger academic backing and lower
   intra-F3 correlation to each other.

2. **H2.5 Sector Kalman Pairs ranks #3.** Among F2 hypotheses, H2.5 has the lowest
   expected rho to F1 (0.05 vs 0.15-0.35 for others) because it uses pure sector
   relationships with no credit instruments. The multi-pair portfolio design also
   provides within-strategy diversification.

3. **F9-H1 Commodity Carry ranks #4**, consistent with research recommendation that
   carry is a high-priority new family. But data availability is a concern (need
   futures curve proxy from ETF returns).

4. **H10.2 Volume-Liquidity Breakout deferred.** The research ranked it #2, but
   hypothesis files for F10 haven't been written yet (task #12 pending). Cannot
   rank what doesn't exist. Will integrate when available.

---

## Detailed Assessments (Top 10)

### 1. H3.1 Vol-Scaled TSMOM (Family 3)

**Why #1:** Multi-asset TSMOM is the gold standard for portfolio diversification.
Historically negative correlation to equities during drawdowns. Hurst et al. (2017)
shows >100 years of evidence. Vol-scaling (Barroso-Santa-Clara 2015) improves Sharpe
by ~50% and dramatically reduces crash risk.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.85-1.10 |
| Expected rho to F1 | 0.05 |
| 2022 survival | **HIGH** (CTAs: SG Trend Index +27.3% in 2022) |
| Data availability | YES -- all 14 instruments in yfinance |
| Implementation complexity | MODERATE (vol-scaling, multi-asset) |
| Lifecycle timeline | mandate -> hypothesis (done) -> data-contract -> spec -> backtest -> robustness = 1 session |
| Blockers | None |
| Key risk | Daily ETF frequency may underperform futures-based TSMOM (basis, roll costs) |

### 2. H3.5 Skip-Month TSMOM (Family 3)

**Why #2:** Novy-Marx (2012 JFE) skip-month refinement isolates pure trend component by
removing short-term mean-reversion contamination. Combined with vol-scaling, addresses
two known TSMOM failure modes simultaneously.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.85-1.15 |
| Expected rho to F1 | 0.05 |
| Expected rho to H3.1 | 0.55-0.70 |
| 2022 survival | HIGH |
| Data availability | YES |
| Implementation complexity | LOW (simpler than H3.1 -- no multi-horizon blending) |
| Lifecycle timeline | 1 session |
| Blockers | None |
| Key risk | Monthly rebalance may miss fast reversals |

**Note:** H3.1 and H3.5 are partially correlated (~0.60). Run BOTH but expect
diminishing marginal SR from the second. After the first F3 passes, the second
adds ~+0.25 SR vs the first's ~+0.62.

### 3. H2.5 Sector ETF Kalman Pairs (Family 2)

**Why #3:** Portfolio of 3 cointegrated sector pairs (XLE-XLI, XLK-XLC, XLF-XLRE)
provides within-strategy diversification. Kalman filter hedge ratio adapts to
time-varying relationships. No credit instruments = lowest F1 correlation in F2.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.70-1.20 |
| Expected rho to F1 | 0.05 |
| 2022 survival | MODERATE (sector relationships may stress) |
| Data availability | YES |
| Implementation complexity | HIGH (Kalman filter, 3 simultaneous pairs) |
| Lifecycle timeline | 1 session (may need pykalman dependency) |
| Blockers | None |
| Key risk | Cointegration breakdown during regime shifts (2022 energy decoupling) |

### 4. F9-H1 Commodity Carry Contango (Family 9 -- NEW)

**Why #4:** Carry is a well-documented risk premium (Koijen et al. JFE 2018, Sharpe ~0.7).
Commodity carry via backwardation/contango is genuinely uncorrelated with credit signals.
New family = maximum diversification impact.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.60-1.00 |
| Expected rho to F1 | 0.05 |
| 2022 survival | MODERATE (commodity vol spiked, but backwardation signals were correct for oil) |
| Data availability | **PARTIAL** -- need futures curve proxy from ETF returns; yfinance has ETF prices but not futures term structure |
| Implementation complexity | MODERATE (roll yield proxy estimation from ETF returns) |
| Lifecycle timeline | 1-2 sessions (data pipeline + backtest) |
| Blockers | ETF roll yield proxy needs design validation |
| Key risk | UNG contango decay was -70% from 2010-2020 -- ETF vehicle risk |

### 5. H7.2 RSI-2 Multi-Asset Contrarian (Family 7)

**Why #5:** Connors RSI-2 is one of the simplest and most replicated short-term
contrarian signals. Golden cross filter (50-EMA > 200-EMA) addresses the key failure
mode (buying dips in bear markets). Multi-asset application provides breadth.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.90-1.20 |
| Expected rho to F1 | 0.15 |
| 2022 survival | MODERATE (golden cross filter should exit before deep drawdowns) |
| Data availability | YES |
| Implementation complexity | LOW (RSI, EMA, simple rules) |
| Lifecycle timeline | 1 session |
| Blockers | None |
| Key risk | RSI-2 may be crowded (widely published) |

### 6. H5.8 Overnight Gap Mean-Reversion (Family 5)

**Why #6:** Zero expected correlation to F1 (contrarian intraday signal vs. multi-day
credit lead-lag). Very low time-in-market (~15-25 trades/year). Negative correlation
to SPY overnight momentum (existing strategy) -- genuine decorrelator.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.50-0.80 |
| Expected rho to F1 | 0.00 |
| 2022 survival | HIGH (intraday hold = no overnight risk) |
| Data availability | YES (need open prices) |
| Implementation complexity | LOW |
| Lifecycle timeline | 1 session |
| Blockers | May need intraday data for precise execution |
| Key risk | Bid-ask spread at open may eat the signal |

### 7. H7.1 VIX Percentile Spike Reversion (Family 7)

**Why #7:** Multi-condition filter (VIX percentile + spike + regime + RSI) isolates
panic exhaustion from sustained crisis. Rare but high-conviction entries.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.80-1.00 |
| Expected rho to F1 | 0.10 |
| 2022 survival | MODERATE (filter should avoid 2022 because VIX didn't spike >15% in single sessions repeatedly) |
| Data availability | YES |
| Implementation complexity | LOW |
| Lifecycle timeline | 1 session |
| Blockers | None (already backtest_ready) |
| Key risk | Low trade count (<30 in 5 years) makes DSR hard to pass |

### 8. H4.2 VRP-SPY Timing (Family 4)

**Why #8:** Variance Risk Premium is one of the most well-documented return predictors
(Bollerslev et al. 2009, t-stat 5.11). Harvests a genuine risk premium, not timing.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.80-1.10 |
| Expected rho to F1 | 0.15 |
| 2022 survival | MODERATE (VRP collapsed during 2022 -- realized vol caught up to implied) |
| Data availability | YES (VIX + SPY realized vol) |
| Implementation complexity | LOW |
| Lifecycle timeline | 1 session |
| Blockers | None |
| Key risk | VRP may be compressed by 0DTE options growth post-2020 |

### 9. H6.4 Macro Momentum Barometer (Family 6)

**Why #9:** AQR-backed (Brooks 2017), 4 independent macro themes with documented
low cross-theme correlation. Price-based proxies enable daily-frequency regime detection.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.80-1.10 |
| Expected rho to F1 | 0.15 |
| 2022 survival | HIGH (macro momentum should have gone risk-off in 2022) |
| Data availability | YES |
| Implementation complexity | MODERATE (z-score normalization, regime classification) |
| Lifecycle timeline | 1 session |
| Blockers | None |
| Key risk | Monthly rebalance may lag fast regime transitions |

### 10. H4.7 SKEW Tail Risk Timing (Family 4)

**Why #10:** Uses SKEW (not VIX) -- a fundamentally different information source
capturing institutional tail hedging behavior. Near-zero correlation to VIX-based
strategies and F1. Preceded 5/7 major drawdowns since 2010.

| Property | Value |
|----------|-------|
| Expected backtest SR | 0.70-0.95 |
| Expected rho to F1 | 0.05 |
| 2022 survival | UNCERTAIN (SKEW behavior during 2022 rate hike needs investigation) |
| Data availability | YES (^SKEW from yfinance, since 2010) |
| Implementation complexity | LOW |
| Lifecycle timeline | 1 session |
| Blockers | None |
| Key risk | Low signal frequency (15-30 episodes in 5yr), DSR may fail |

---

## Execution Roadmap

### Sprint 1 (Immediate -- no blockers)

Run these 4 in parallel, 1 session each. All have full data availability and low complexity:

| Slot | Hypothesis | Family | Rationale |
|------|-----------|--------|-----------|
| A | H3.1 Vol-Scaled TSMOM | F3 | Highest diversification value, crisis alpha |
| B | H7.2 RSI-2 Multi-Asset Contrarian | F7 | Lowest implementation complexity, high expected SR |
| C | H4.2 VRP-SPY Timing | F4 | Well-documented, simple signal |
| D | H6.4 Macro Momentum Barometer | F6 | Fills F6 gap, strong academic backing |

**Expected outcome:** 2-3 pass gates (50-75% success rate based on historical pipeline).
Even 1 passing from different family lifts portfolio SR significantly.

### Sprint 2 (After Sprint 1 results)

| Slot | Hypothesis | Family | Rationale |
|------|-----------|--------|-----------|
| E | H3.5 Skip-Month TSMOM | F3 | Second F3 pick if H3.1 fails; redundant if H3.1 passes |
| F | H2.5 Sector Kalman Pairs | F2 | Fills F2 gap (GLD-SLV MR v4 already passing) |
| G | H7.1 VIX Percentile Spike Reversion | F7 | Second F7 pick |
| H | H4.7 SKEW Tail Risk Timing | F4 | Orthogonal to H4.2 |

### Sprint 3 (New families, partial data needs)

| Slot | Hypothesis | Family | Rationale |
|------|-----------|--------|-----------|
| I | F9-H1 Commodity Carry | F9 | New family, needs roll yield proxy |
| J | H5.8 Overnight Gap MR | F5 | Zero-correlation calendar strategy |

### Deploy Now (Already Passing)

| Strategy | Action |
|----------|--------|
| GLD-SLV MR v4 | Start paper trading immediately -- all 5 gates pass |
| BTC Momentum v2 | Start Track D paper trading -- all 5 gates pass |
| TLT-TQQQ Sprint | Start Track D paper trading -- all 5 gates pass |

---

## Family Coverage After Full Queue Execution

| Family | Current | After Sprint 1 | After Sprint 2 | After Sprint 3 | Target |
|--------|---------|-----------------|-----------------|-----------------|--------|
| F1 (Credit) | 10 (prune to 4) | 4 | 4 | 4 | 3-4 |
| F2 (Mean Rev) | 1 (GLD-SLV v4) | 1 | 2 | 2 | 2-3 |
| F3 (TSMOM) | 1 (BTC Mom v2, Track D) | 2 | 2-3 | 2-3 | 2-3 |
| F4 (Vol Regime) | 0 | 1 | 2 | 2 | 1-2 |
| F5 (Calendar) | 0 | 0 | 0 | 1 | 1 |
| F6 (Macro) | 0 | 1 | 1 | 1 | 1-2 |
| F7 (Sentiment) | 0 | 1 | 2 | 2 | 1-2 |
| F8 (Non-Credit) | 1 (SOXX-QQQ) | 1 | 1 | 1 | 1 |
| F9 (Carry) | 0 | 0 | 0 | 1 | 1-2 |
| **Total** | **12** | **11** | **13-14** | **14-15** | **15-20** |
| **Families covered** | **3** | **6** | **7** | **8** | **6+** |
| **Est. portfolio SR** | **1.35** | **1.7-2.0** | **1.9-2.3** | **2.0-2.5** | **1.5-2.0 (realistic)** |

---

## Hypotheses Excluded from Queue (and Why)

| Hypothesis | Reason |
|------------|--------|
| H3.4 Donchian Breakout | Redundant with H3.1 (same family, similar mechanism, higher rho) |
| H3.3 Multi-TF Ensemble | Similar to H3.1 but more complex; run H3.1 first |
| H3.6 Factor Momentum | rho 0.20 to F1 (equity sector overlap), lower priority |
| H3.7 Macro Trend Economic | rho 0.25 to F1 (uses credit spread), overlaps with H6.4 |
| H5.4 Turn-of-Month | Prior TOM test failed (SPY Sharpe = -0.107); needs redesign |
| H5.5 Pre-Holiday Drift | 10 trades/year = no statistical significance achievable |
| H5.6 January Small-Cap | 1 trade/year = academic exercise, not tradeable |
| H5.7 BTC Tuesday Effect | Low expected SR (0.40-0.70), crypto noise |
| H2.2 IWM-SPY Pairs | Crowded academic pair, edge likely arbitraged |
| H2.3 HYG-LQD Pairs | rho ~0.25-0.35 to F1 (same credit instruments) |
| H4.4 GARCH Regime | Redundant with H4.2 (both size by vol, correlated ~0.35) |
| H4.5 Sector Vol Dispersion | Low trade count (~30-60), similar to H7.6 |
| H4.6 VIX Percentile Adaptive | Redundant with H4.7 (SKEW is more differentiated) |
| F9-H2 Bond Rolldown | rho 0.25 to F1 (duration overlap), needs FRED data |
| F9-H3 FX Carry | rho 0.30 to F1 (both risk-on premia), crash risk |
| H7.3 Leveraged ETF MR | Track D only, high MaxDD tolerance required |
| H7.4 AAII Sentiment | Needs external data pipeline |
| H7.5 Put/Call Ratio | Needs external data, 0DTE contamination risk |

---

*Version 1.0 | 2026-03-31 | Priority backtest queue created by Macro Strategist*

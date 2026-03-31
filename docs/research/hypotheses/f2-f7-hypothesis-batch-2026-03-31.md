# Hypothesis Batch: F2 Mean-Reversion Pairs + F7 Sentiment Contrarian

**Date:** 2026-03-31
**Author:** Alpha Researcher Agent
**Families:** F2 (Mean Reversion), F7 (Sentiment Contrarian)

---

## Summary

8 new hypotheses synthesized from research findings. 4 for Family 2 (cointegration-based pairs trading) and 4 for Family 7 (sentiment/contrarian signals). These target the two highest-priority untested families for portfolio diversification.

### External Data Requirements

| Hypothesis | External Data Needed | Source | Testable with Existing Data? |
|-----------|---------------------|--------|------------------------------|
| H2.2 IWM-SPY | No | Yahoo Finance | YES |
| H2.3 HYG-LQD | No | Yahoo Finance | YES |
| H2.4 EFA-EEM | No | Yahoo Finance | YES |
| H2.5 Sector Pairs | No | Yahoo Finance | YES |
| H7.4 AAII Sentiment | **YES** | AAII website (weekly CSV) | NO — needs data pipeline |
| H7.5 Put/Call Ratio | **YES** | CBOE DataShop / Quandl | NO — needs data pipeline |
| H7.6 Sector Dispersion | No | Yahoo Finance | YES |
| H7.7 Correlation Premium | Partial (v2 only) | Yahoo Finance (v1) | YES (v1 proxy) |

**Recommendation:** Prioritize H2.2-H2.5, H7.6, H7.7 for immediate backtesting (no external data needed). H7.4 and H7.5 require data pipeline work before testing.

---

## Family 2 — Mean Reversion (Pairs/Ratios)

### Design Principles

All F2 hypotheses share a common framework:
- **Kalman filter dynamic hedge ratio** (vs static OLS) — adapts to time-varying relationships
- **Z-score entry/exit**: enter at |z| > 2.0, exit at z = 0
- **60-day rolling window** for z-score calculation
- **252-day rolling ADF test** to verify ongoing cointegration
- **Falsification**: cointegration breakdown, inverted signal test, shuffled signal test

### H2.2 — IWM-SPY Cointegration Pairs

| Property | Value |
|----------|-------|
| **Slug** | `iwm-spy-cointegration-pairs` |
| **Pair** | IWM (Russell 2000) vs SPY (S&P 500) |
| **Mechanism** | Size factor mean-reversion: small-cap vs large-cap spread dislocations from risk-on/off flows, index rebalancing |
| **Expected Sharpe** | 0.60 - 1.00 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.10 - 0.25 |
| **Key Risk** | Crowded academic pair — edge may be arbitraged away |
| **External Data** | None |

### H2.3 — HYG-LQD Credit Quality Pairs

| Property | Value |
|----------|-------|
| **Slug** | `hyg-lqd-credit-quality-pairs` |
| **Pair** | HYG (high-yield) vs LQD (investment-grade) |
| **Mechanism** | Credit risk premium mean-reversion: credit quality rotation driven by fallen angel/rising star flows |
| **Expected Sharpe** | 0.70 - 1.10 |
| **Expected MaxDD** | < 12% |
| **F1 Correlation** | 0.15 - 0.35 (CAUTION: same instruments as F1, different signal) |
| **Key Risk** | Correlation to F1 may exceed 0.30 — must verify before lifecycle |
| **External Data** | None |

### H2.4 — EFA-EEM Developed vs Emerging Pairs

| Property | Value |
|----------|-------|
| **Slug** | `efa-eem-developed-emerging-pairs` |
| **Pair** | EFA (EAFE developed) vs EEM (emerging markets) |
| **Mechanism** | Global capital allocation cycles: DM/EM flows, USD strength cycles, valuation convergence |
| **Expected Sharpe** | 0.60 - 0.90 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.05 - 0.20 (different geography, different risk factors) |
| **Key Risk** | USD-driven regime dependency — may only work in weak-USD periods |
| **External Data** | None |

### H2.5 — Sector ETF Kalman Pairs (Portfolio of 3 Pairs)

| Property | Value |
|----------|-------|
| **Slug** | `sector-etf-kalman-pairs` |
| **Pairs** | XLE-XLI, XLK-XLC, XLF-XLRE |
| **Mechanism** | Structural economic linkages create cointegrated sector relationships; portfolio of 3 pairs diversifies single-pair risk |
| **Expected Sharpe** | 0.70 - 1.20 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.05 - 0.15 (lowest expected — pure sector-level, no credit) |
| **Key Risk** | Sector pairs may not be cointegrated (structural breaks during regime shifts) |
| **External Data** | None |

---

## Family 7 — Sentiment Contrarian

### Existing F7 Hypotheses (for reference)

| ID | Strategy | Signal Type | Status |
|----|----------|-------------|--------|
| H7.1 | VIX percentile spike reversion | VIX percentile + spike + RSI | backtest_ready |
| H7.2 | RSI-2 multi-asset contrarian | RSI(2) extreme + golden cross | hypothesis_written |
| H7.3 | Leveraged ETF mean-reversion | 3x ETF consecutive down days + VIX | backtest_ready |

### New F7 Hypotheses

### H7.4 — AAII Sentiment Contrarian

| Property | Value |
|----------|-------|
| **Slug** | `aaii-sentiment-contrarian` |
| **Signal** | AAII weekly bullish/bearish sentiment > 50% |
| **Mechanism** | Retail investor herding at extremes: systematic late positioning creates contrarian edge |
| **Expected Sharpe** | 0.60 - 0.90 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.05 - 0.15 (behavioral signal, independent of price) |
| **Key Risk** | Signal fires rarely (5-10x/year), long holding period (4-8 weeks) |
| **External Data** | **YES — AAII weekly sentiment survey CSV** |

### H7.5 — Put/Call Ratio Contrarian

| Property | Value |
|----------|-------|
| **Slug** | `put-call-ratio-contrarian` |
| **Signal** | Equity put/call ratio > 90th percentile + RSI(14) < 35 |
| **Mechanism** | Options hedging demand extremes: institutional put buying creates mechanical selling that reverts |
| **Expected Sharpe** | 0.70 - 1.10 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.05 - 0.15 |
| **Correlation to H7.1** | 0.25 - 0.40 (partial overlap — both fire during fear events) |
| **Key Risk** | 0DTE options explosion post-2020 may have altered put/call ratio dynamics |
| **External Data** | **YES — CBOE equity put/call ratio (daily)** |

### H7.6 — Sector Dispersion Regime Switch

| Property | Value |
|----------|-------|
| **Slug** | `sector-dispersion-regime` |
| **Signal** | Cross-sector return dispersion percentile as momentum/MR regime classifier |
| **Mechanism** | High dispersion = differentiation (momentum works); low dispersion = herding (mean-reversion works) |
| **Expected Sharpe** | 0.70 - 1.10 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.05 - 0.20 |
| **Key Risk** | Regime switches may be too frequent (transaction cost drag) |
| **External Data** | None — uses sector ETF prices only |

### H7.7 — Correlation Risk Premium (Dispersion Proxy)

| Property | Value |
|----------|-------|
| **Slug** | `correlation-risk-premium` |
| **Signal** | Realized sector dispersion > 80th percentile (proxy for correlation premium) |
| **Mechanism** | Index options premium creates correlation risk premium; harvestable via long sectors / short index |
| **Expected Sharpe** | 0.60 - 1.00 |
| **Expected MaxDD** | < 15% |
| **F1 Correlation** | 0.05 - 0.15 |
| **Key Risk** | Proxy may not track true implied correlation premium closely enough |
| **External Data** | None for v1 (realized dispersion proxy). v2 needs CBOE DSPX + options IVs |

---

## Prioritized Testing Order

Based on data availability, expected Sharpe, and correlation properties:

| Priority | Hypothesis | Rationale |
|----------|-----------|-----------|
| 1 | H2.5 Sector Pairs | Highest expected Sharpe, lowest F1 correlation, no external data, multi-pair diversification |
| 2 | H7.6 Sector Dispersion | No external data, genuinely different mechanism, high time-in-market |
| 3 | H2.3 HYG-LQD | Strong academic backing, but MUST check F1 correlation first |
| 4 | H2.4 EFA-EEM | Lowest expected F1 correlation among F2 hypotheses, different geography |
| 5 | H7.7 Correlation Premium | v1 proxy testable immediately, strong academic foundation |
| 6 | H2.2 IWM-SPY | Crowding risk lowers priority despite strong academic support |
| 7 | H7.4 AAII Sentiment | Needs external data pipeline — defer until data ingestion built |
| 8 | H7.5 Put/Call Ratio | Needs external data + 0DTE contamination risk — lowest priority |

---

## Correlation Matrix (Expected)

Estimated pairwise correlations between new hypotheses and existing portfolio:

```
         F1-avg  H7.1  H7.2  H7.3  H2.2  H2.3  H2.4  H2.5  H7.4  H7.5  H7.6  H7.7
F1-avg   1.00
H7.1     0.15   1.00
H7.2     0.10   0.20  1.00
H7.3     0.15   0.35  0.25  1.00
H2.2     0.15   0.10  0.15  0.10  1.00
H2.3     0.25   0.15  0.10  0.10  0.20  1.00
H2.4     0.10   0.05  0.10  0.05  0.15  0.10  1.00
H2.5     0.10   0.05  0.10  0.05  0.10  0.10  0.10  1.00
H7.4     0.10   0.15  0.20  0.10  0.10  0.10  0.10  0.05  1.00
H7.5     0.10   0.35  0.15  0.25  0.10  0.10  0.05  0.05  0.20  1.00
H7.6     0.15   0.15  0.10  0.10  0.10  0.10  0.10  0.30  0.10  0.10  1.00
H7.7     0.10   0.10  0.05  0.05  0.05  0.05  0.10  0.15  0.05  0.05  0.40  1.00
```

**Notes:**
- H7.6 and H7.7 have expected correlation ~0.40 (both use sector dispersion). May need to pick one.
- H2.3 (HYG-LQD) shows highest expected F1 correlation (~0.25) — monitor carefully.
- H2.4 (EFA-EEM) and H2.5 (Sector Pairs) are the most orthogonal to existing portfolio.

---

## Excluded Hypotheses (and why)

1. **UNG-USO energy spread**: "The Weak Tie" MIT paper documents weak cointegration. UNG has severe contango decay (futures roll costs). Not viable for daily-frequency pairs trading. Would need futures-based implementation.

2. **AAII + RSI-2 combo**: Too close to H7.2 (RSI-2 multi-asset contrarian). Adding AAII as a filter to RSI-2 would be a parameter variant, not a new hypothesis.

3. **Google Trends attention decay**: Requires Google Trends API with unreliable rate limits and unstable data revisions. Signal-to-noise ratio at daily frequency is low. Better suited for weekly/monthly macro regime classification (Family 6).

4. **VIX > 30 buy signal**: Already subsumed by H7.1 (VIX percentile spike reversion) which uses a more sophisticated multi-condition filter.

# New Mechanism Family Assessment

**Analyst:** Robustness Analyst
**Date:** 2026-03-31

---

## Summary

5 candidate families evaluated for inclusion beyond the existing F1-F8 framework.
Assessment: 1 strong new family (CARRY = F9), 2 conditional (MICROSTRUCTURE/OVERNIGHT
folds into F5; CRYPTO PAIRS too risky), 2 rejected for now (LIQUIDITY folds into F6;
LOTTERY/SKEWNESS not implementable with ETFs).

| Candidate | Verdict | Designation | Hypotheses Written |
|-----------|---------|-------------|-------------------|
| CARRY | NEW FAMILY | F9 | 3 hypotheses |
| LIQUIDITY | Fold into F6 | — | 0 (not distinct enough) |
| MICROSTRUCTURE/OVERNIGHT | Fold into F5 | — | 1 hypothesis under F5 |
| LOTTERY/SKEWNESS | REJECT | — | 0 |
| CRYPTO PAIRS | CAUTION — conditional | Tentative F10 | 1 hypothesis (with regime gate) |

---

## Family 9: CARRY (Cross-Asset) — APPROVED AS NEW FAMILY

### Why this warrants its own family

Carry is a distinct return-generating mechanism from all 8 existing families:

- **F1 (Credit lead-lag):** Exploits *information flow* from bonds to equities at daily frequency.
  Carry exploits *yield differentials* across asset classes at weekly-monthly frequency.
- **F2 (Mean reversion):** Enters on deviation from equilibrium. Carry enters on the *level*
  of yield spread, not deviation from it.
- **F3 (Momentum):** Trend-following. Carry is fundamentally anti-trend — it tends to work
  best when trends reverse (carry unwind events are trend reversals).
- **F6 (Macro regime):** Uses macro indicators to rotate. Carry uses *yield* as the signal,
  not macro PMI/yield curve slope.

**Academic support is strong:**
- Koijen, Moskowitz, Pedersen, Vrugt (JFE 2018): Carry predicts returns across equities,
  bonds, FX, and commodities. Portfolio Sharpe ~0.7. This is one of the most replicated
  results in cross-asset factor investing.
- The carry mechanism is economically intuitive: you are compensated for bearing
  identifiable risks (duration risk, currency risk, storage cost, convenience yield).
- Carry correlations with existing families: expected rho to F1 (credit lead-lag) ~0.15-0.25
  (different signal frequency and mechanism), rho to F3 (momentum) ~-0.10 to 0.10
  (carry and momentum are historically near-orthogonal — Asness et al. 2013).

**Expected contribution to portfolio SR:**
Using the marginal contribution formula with current portfolio SR ~1.35 and rho ~0.20:
`delta_SR ~= (0.7 - 0.20 * 1.35) / sqrt(1 + 2*0.20*0.7/1.35) = 0.43 / 1.09 = 0.39`
Adding a Sharpe 0.7 carry strategy with rho=0.20 to the portfolio would increase
portfolio SR by ~0.39. This is a meaningful contribution.

### Implementable sub-strategies with ETF-only universe

1. **Commodity carry (contango/backwardation):** Long USO when in backwardation (supply
   shortage = positive carry), short/avoid when in deep contango. Measurable via
   futures curve or ETF roll yield. Available: USO, UNG, DBC.

2. **Bond roll-down carry:** Long TLT when yield curve is steep (high roll-down return),
   reduce when flat/inverted. The steeper the curve, the more an investor earns just by
   holding and aging down the curve. Available: TLT, IEF, SHY (all yfinance).

3. **EM carry trade (EM vs DM yield spread):** Already partially captured by
   `em-carry-trade` hypothesis. Long EEM when EM-DM yield spread is wide. This is the
   classic carry trade applied to equity indices.

### Hypotheses written

- `F9-H1`: Commodity contango/backwardation carry (USO)
- `F9-H2`: Bond yield curve roll-down carry (TLT/IEF)
- `F9-H3`: FX carry via currency ETFs (FXE/FXY)

---

## LIQUIDITY — Fold into F6 (Macro Regime Rotation)

### Why it does NOT warrant its own family

- **ETF universe limitation:** The Amihud illiquidity ratio and Pastor-Stambaugh
  liquidity factor are designed for individual stock cross-sections. With our ETF-only
  universe of 39 assets, the cross-sectional spread in liquidity is too narrow to
  generate meaningful signals. SPY, QQQ, TLT, GLD all have deep liquidity —
  there is no illiquidity premium to harvest.

- **IWM-SPY spread as proxy:** The one implementable variant — overweight IWM when
  liquidity is abundant (small-cap premium) — is effectively a macro regime rotation
  signal (risk-on = long small caps). This fits cleanly into F6.

- **Overlap with F6:** Liquidity conditions are a *symptom* of macro regimes, not an
  independent mechanism. Tight liquidity = risk-off, abundant liquidity = risk-on.
  The signal information content is largely redundant with yield curve, VIX, and
  other F6 indicators.

### Recommendation

Do not create a separate family. If a liquidity timing strategy is developed,
classify it under F6 (macro regime rotation) with a liquidity sub-indicator.

---

## MICROSTRUCTURE / OVERNIGHT — Fold into F5 (Calendar/Structural Flow)

### Assessment

The overnight return anomaly is real and well-documented:
- Majority of equity returns accrue overnight (Cliff Smith, Kelly 2019)
- Institutional vs retail flow imbalance creates the pattern
- The mechanism is *structural* (market close/open dynamics) not informational

This is a calendar/structural effect — it fires on a fixed schedule (every market
close), driven by mechanical flow patterns. This places it squarely in F5, not as
a new family.

**However:** `spy-overnight-momentum` already exists as an F5 hypothesis. The
existing hypothesis captures this mechanism. If the overnight effect is orthogonal
to existing F5 strategies (OPEX, turn-of-month), that diversification benefit
accrues within F5, not as a new family.

### Cross-asset lead-lag at daily frequency

The suggestion that "Treasury leads equity at daily frequency" is an extension of
F1 (credit lead-lag) or F8 (non-credit lead-lag), not a microstructure effect.
Treasury price movements reflecting information before equities is *information flow*,
not *microstructure*.

### Hypothesis written

- 1 hypothesis under F5: `F5-H7` overnight session momentum (extends existing
  `spy-overnight-momentum` to multi-asset — QQQ, IWM, EFA)

---

## LOTTERY / SKEWNESS — REJECT

### Why this does not work with our universe

- **Cross-sectional effect:** The MAX effect (Bali-Cakici-Whitelaw 2011) operates in
  the *cross-section of individual stocks* — stocks with extreme recent daily returns
  underperform. This requires a universe of 100+ stocks to sort into decile portfolios.

- **ETF inapplicability:** With 39 ETFs, there is no meaningful cross-sectional spread
  in recent max returns. ETFs diversify away the lottery characteristics that individual
  stocks exhibit.

- **No implementable strategy:** The skewness premium requires shorting high-skewness
  assets and going long low-skewness assets. ETFs have similar skewness profiles
  because they are diversified. The long-short spread would be negligibly small.

### Recommendation

Reject entirely for this project's ETF-only universe. If the universe expands to
individual stocks in the future, revisit.

---

## CRYPTO PAIRS (BTC-ETH Cointegration) — CAUTION, Conditional

### Assessment

**The cointegration thesis is scientifically suspect post-2022:**

- BTC-ETH cointegration was documented in multiple papers using 2015-2021 data
- The Ethereum Merge (Sep 2022) fundamentally changed ETH's economic model:
  - Pre-Merge: ETH was a PoW mining token correlated with BTC via shared energy costs
  - Post-Merge: ETH is a PoS yield-bearing asset with burn mechanism (EIP-1559)
  - The shared cost basis that drove cointegration no longer exists
- ETH/BTC ratio hit all-time lows in 2024-2025, suggesting structural decorrelation
- Tadi-Kortchemski (2021) Sharpe ~1.5 was in-sample on pre-Merge data — contaminated

**Regime-conditional approach is the only honest path:**

If BTC-ETH cointegration still operates post-Merge (testable hypothesis), the
strategy must include a regime gate: run the cointegration test on a rolling
window and only trade when the null hypothesis of no-cointegration is rejected
at p < 0.05. When cointegration breaks down, the strategy goes flat.

**Correlation risk:** Both BTC and ETH correlate with risk-on equity positioning
(rho ~0.40-0.60 with SPY during risk-off events). A crypto pairs strategy adds
tail risk correlation with F1 (credit lead-lag) during the exact periods when
diversification matters most.

### Verdict

Tentatively designate as F10 (Crypto Structural) but with strict conditions:
- Must include rolling cointegration test as a regime gate
- Must demonstrate post-Merge (post Sep 2022) performance separately
- Must show correlation to existing portfolio < 0.30 in stress periods
- If post-Merge cointegration is broken, KILL — do not force it

### Hypothesis written

- `F10-H1`: BTC-ETH regime-conditional mean reversion (with cointegration gate)

---

## Updated Family Registry

| Family | Name | Status | Strategies |
|--------|------|--------|------------|
| F1 | Cross-Asset Information Flow | COMPLETE | 10 passing |
| F2 | Mean Reversion (Pairs/Ratios) | NEXT PRIORITY | In lifecycle |
| F3 | Momentum / Trend Following | FAILED — needs redesign | — |
| F4 | Volatility Regime Harvesting | IN PROGRESS | H4.1, H4.3 (FAILED robustness) |
| F5 | Calendar / Structural Flow | PARTIALLY TESTED | Add overnight (F5-H7) |
| F6 | Macro Regime Rotation | UNTESTED | Absorbs liquidity sub-signals |
| F7 | Sentiment Contrarian | UNTESTED | — |
| F8 | Non-Credit Cross-Market Lead-Lag | 1 PASSING (SOXX-QQQ) | — |
| **F9** | **Carry (Cross-Asset)** | **NEW — HIGH PRIORITY** | **3 hypotheses written** |
| **F10** | **Crypto Structural** | **NEW — CONDITIONAL** | **1 hypothesis (regime-gated)** |

### Expected correlation matrix (new families vs existing)

| | F1 | F2 | F3 | F4 | F5 | F6 | F9 | F10 |
|---|---|---|---|---|---|---|---|---|
| F9 (Carry) | 0.20 | 0.10 | -0.05 | 0.15 | 0.05 | 0.30 | 1.00 | 0.20 |
| F10 (Crypto) | 0.40 | 0.15 | 0.25 | 0.30 | 0.10 | 0.35 | 0.20 | 1.00 |

F9 (Carry) has the best diversification profile — low expected correlation to
most existing families, especially F1-F3. This is the highest-priority new family
for portfolio SR improvement.

F10 (Crypto) has concerning rho with F1 (~0.40 during stress) and F6 (~0.35).
Its marginal SR contribution is questionable given the correlation structure.

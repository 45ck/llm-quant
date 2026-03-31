# Hypothesis Registry — 2026-03-31

**Purpose:** Single source of truth for all hypotheses written during the 2026-03-31 research sprint.
**Analyst:** Robustness Analyst (reconciliation + quality review)
**Total hypotheses inventoried:** 58 (after collision resolution — 6 collisions resolved, 4 drafts deleted, 2 renumbered)
**Backtested this session:** 22 strategies across Sprints 1-6 — 5 passed (22.7% pass rate)

---

## 1. Complete Inventory

### Family 2: Mean Reversion (Pairs/Ratios)

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H2.2 | iwm-spy-cointegration-pairs | `data/strategies/` | alpha-researcher-1 | 0.60-1.00 | 0.10-0.25 | yfinance | P1 | **CONFLICT with H2.6** — same pair, different methodology. See Section 3. |
| H2.3 | hyg-lqd-credit-quality-pairs | `data/strategies/` | alpha-researcher-1 | 0.80+ | 0.20-0.35 | yfinance | P1 | **CONFLICT with H2.7** — same pair, different methodology. **F1 OVERLAP WARNING**: HYG/LQD are inputs to F1 strategies. Must verify rho < 0.30. |
| H2.4 | efa-eem-developed-emerging-pairs | `data/strategies/` | alpha-researcher-1 | 0.70+ | 0.10-0.20 | yfinance | P2 | Clean. Low F1 correlation expected. |
| H2.5 | sector-etf-kalman-pairs | `data/strategies/` | alpha-researcher-1 | 0.80+ | 0.10-0.20 | yfinance | P2 | 3 sub-pairs — increases parameter count. Monitor for overfitting. |
| H2.6 | iwm-spy-small-large-cap-spread | `docs/research/hypotheses/` | alpha-researcher-2 | 0.85+ | 0.10-0.20 | yfinance | P1 | **CONFLICT with H2.2** — see Section 3. |
| H2.7 | hyg-lqd-credit-quality-pair | `docs/research/hypotheses/` | alpha-researcher-2 | 0.90+ | 0.15-0.25 | yfinance | P2 | **CONFLICT with H2.3** — see Section 3. Track B (higher DD tolerance). |
| H2.8 | xle-xli-energy-industrial-pair | `docs/research/hypotheses/` | alpha-researcher-2 | 0.70-0.90 | 0.05-0.15 | yfinance | P2 | Clean. Energy/industrial cost linkage. |
| H2.9 | xle-xli-energy-industrial-pairs | `docs/research/hypotheses/` | alpha-researcher-2 | 0.60-1.00 | 0.05-0.15 | yfinance | P2 | **RENUMBERED from H2.6** (collision with IWM/SPY). Kalman filter variant of H2.8. |

### Family 3: Trend Following / TSMOM

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H3.3 | multi-timeframe-tsmom-ensemble | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80-1.05 | 0.05-0.15 | yfinance + FX | P1 | Clean. Majority vote is a good de-noising approach. FX data needs `EURUSD=X` format. |
| H3.4 | donchian-breakout-multi-asset | `docs/research/hypotheses/` | alpha-researcher-2 | 0.70-0.95 | 0.05-0.15 | yfinance + FX | P2 | Clean. Fewest parameters (2) = lowest overfitting risk. Good Occam baseline. |
| H3.5 | skip-month-cross-asset-momentum | `docs/research/hypotheses/` | alpha-researcher-2 | 0.85-1.15 | 0.05-0.15 | yfinance + FX | P1 | **BACKTESTED Sprint 2: FAIL** — Sharpe 0.585, MaxDD 6.44%. Worse than H3.1. Skip-month removes helpful recent info. |
| H3.6 | factor-momentum-sector-rotation | `docs/research/hypotheses/` | alpha-researcher-2 | 0.75-1.00 | 0.15-0.25 | yfinance | P2 | Factor beta estimation from ETFs is noisy. More complex = more fragile. |
| H3.7 | macro-trend-economic-momentum | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80-1.10 | 0.20-0.30 | yfinance | P2 | **F1 OVERLAP WARNING**: credit trend signal (LQD/IEF) overlaps with F1 credit lead-lag. Expected rho 0.20-0.30 may underestimate true overlap. **F6 OVERLAP**: macro regime rotation uses similar inputs — check rho to H6.1-H6.3. |
| H3.8 | skip-month-intermediate-momentum | `docs/research/hypotheses/` | alpha-researcher-3 | 0.75-1.00 | 0.05-0.15 | yfinance | P2 | Intermediate timeframe variant of H3.5. |
| H3.9 | donchian-channel-sma-filtered | `docs/research/hypotheses/` | alpha-researcher-3 | 0.70-0.95 | 0.05-0.15 | yfinance | P2 | SMA filter variant of H3.4. |
| H3.10 | factor-momentum-long-short-sectors | `docs/research/hypotheses/` | alpha-researcher-3 | 0.75-1.05 | 0.15-0.25 | yfinance | P2 | Long-short sector factor momentum. |

### Family 4: Volatility Regime Harvesting

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H4.3 | vix-spike-mean-reversion | `data/strategies/` | quant-researcher | 1.08 (backtest) | 0.05-0.15 | yfinance | — | **FAILED ROBUSTNESS** — 8 trades, DSR misleading, CPCV cannot run. See `docs/research/results/h4.3-vix-spike-mr-robustness.md`. Needs reformulation as v2. |
| H4.4 | garch-regime-equity-sizing | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80-1.05 | 0.10-0.20 | yfinance | P1 | **BACKTESTED Sprint 2+3: FAIL** — v1 Sharpe 0.605, v2 (persistence filter) 0.723. Improved but still misses 0.80 gate. Best F4 candidate. |
| H4.5 | sector-vol-dispersion-mr | `docs/research/hypotheses/` | alpha-researcher-2 | 0.70-1.00 | 0.05-0.15 | yfinance | P2 | Clean. Interesting second-moment signal. **LOW TRADE COUNT RISK**: 30-60 trades in 5yr is marginal for CPCV. |
| H4.6 | vix-percentile-adaptive-allocation | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80-1.10 | 0.10-0.20 | yfinance | P1 | **2022 RISK**: TLT lost 31% while VIX was elevated. Stock-bond correlation flipped positive. This breaks the flight-to-quality assumption. Falsification criterion #3 specifically addresses this — good. |
| H4.7 | skew-tail-risk-timing | `docs/research/hypotheses/` | alpha-researcher-2 | 0.70-0.95 | 0.00-0.10 | yfinance (^SKEW) | P2 | **LOW TRADE COUNT RISK**: 15-30 divergence episodes in 5yr. Same problem as H4.3 — too rare for CPCV. **DATA RISK**: SKEW only available since 2010 (~16yr). |
| H4.8 | sector-iv-rv-rotation | `docs/research/hypotheses/` + `data/strategies/` | alpha-researcher-3 | 0.75-1.00 | 0.10-0.20 | yfinance | P2 | Sector rotation via implied-realized vol spread. |
| H4.3-v2 | vix-spike-mr-enhanced | `docs/research/hypotheses/` | robustness-analyst | 0.80-1.10 | 0.05-0.15 | yfinance | P1 | Reformulated from failed H4.3 — lower threshold to generate 40-80 trades. |

### Family 5: Calendar / Structural Flow Effects

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H5.4 | turn-of-month-broad-equities | `docs/research/hypotheses/` | alpha-researcher-2 | 0.65-0.90 | 0.10-0.20 | yfinance | P1 | **BACKTESTED Sprint 2: KILL** — Sharpe -0.226, return -7.28%. TOM anomaly arbitraged away in 2022-2026 period. |
| H5.5 | pre-holiday-equity-drift | `docs/research/hypotheses/` | alpha-researcher-2 | 0.40-0.70 | 0.05-0.10 | yfinance | P3 | **LOW TRADE COUNT**: ~50 trades in 5yr. **LOW SR**: expected 0.40-0.70 may not pass Sharpe >= 0.80 gate. Best as portfolio diversifier, not standalone. |
| H5.6 | january-small-cap-premium | `docs/research/hypotheses/` | alpha-researcher-2 | 0.35-0.60 | 0.05-0.10 | yfinance | P3 | **CRITICALLY LOW TRADE COUNT**: 5 trades in 5yr (1/year). **CANNOT PASS any robustness gate.** Needs 20+ years of data minimum. Effect weakened since publication. |
| H5.7 | btc-tuesday-effect | `docs/research/hypotheses/` | alpha-researcher-2 | 0.40-0.70 | 0.00-0.10 | yfinance (crypto) | P3 | **WEAK MECHANISM**: Day-of-week effects in crypto are unstable. BTC market structure changed dramatically (ETF approvals, institutional adoption). Likely data-mined in short sample. |
| H5.8 | overnight-gap-mean-reversion | `docs/research/hypotheses/` | alpha-researcher-2 | 0.50-0.80 | 0.00-0.10 | yfinance | P2 | **EXECUTION RISK**: Requires open-price execution (slippage at market open is higher than close). Cost sensitivity critical. |
| F5-H7 | overnight-multi-asset-momentum | `data/strategies/` | robustness-analyst | 0.60+ | 0.20 | yfinance | P2 | Extension of spy-overnight-momentum. Needs open/close data quality check. |

### Family 6: Macro Regime Rotation

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H6.1 | real-yield-threshold-rotation | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80+ | 0.15-0.25 | yfinance (TIP/IEF proxy) | P1 | Clean. Real yield is a strong macro signal. TIP/IEF proxy avoids FRED dependency. |
| H6.2 | cyclical-defensive-ratio-timing | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80+ | 0.10-0.20 | yfinance | P1 | **BACKTESTED Sprint 2: KILL** — Sharpe -0.622, MaxDD 37.48%, return -30.90%. Devastated by 2022 regime break. |
| H6.3 | credit-momentum-overlay | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80+ | 0.30-0.45 | yfinance | P2 | **F1 OVERLAP WARNING**: HYG/IEF signal is conceptually similar to F1 credit lead-lag. Expected rho 0.30-0.45 is DANGEROUSLY HIGH. Must verify actual rho < 0.30 in backtest. If rho > 0.30, demote to F1 variant, not independent F6. |
| H6.5 | multi-asset-carry-rotation | `docs/research/hypotheses/` | alpha-researcher-2 | 0.80+ | 0.15-0.25 | yfinance | P2 | **F9 OVERLAP**: This is carry applied to multi-asset rotation. Conceptually belongs in F9 (Carry), not F6 (Macro). **TAXONOMY CONFLICT** — see Section 3. |
| H6.6 | credit-spread-momentum-rotation | `docs/research/hypotheses/` | alpha-researcher-3 | 0.70-1.10 | 0.20-0.35 | yfinance | P2 | **RENUMBERED from H6.3** (collision). HYG/LQD 21d ROC macro rotation. Similar to H6.3 but uses different pair. |

### Family 7: Sentiment Contrarian

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H7.1 | vix-percentile-spike-reversion | `data/strategies/` | quant-researcher | 0.80+ | 0.05-0.15 | yfinance | P1 | Good multi-filter approach. **Overlap with H4.3 v2** — if H4.3 is reformulated with VIX percentile, these may merge. |
| H7.4 | aaii-sentiment-contrarian | `data/strategies/` | alpha-researcher-1 | 0.70+ | 0.05 (genuine orthogonality) | **EXTERNAL: AAII CSV** | P2 | **EXTERNAL DATA REQUIRED**: AAII weekly sentiment CSV not in yfinance. Needs data pipeline. Genuinely orthogonal mechanism — high diversification value if data is obtained. |
| H7.5 | put-call-ratio-contrarian | `data/strategies/` | alpha-researcher-1 | 0.80+ | 0.05-0.15 | **EXTERNAL: CBOE** | P2 | **EXTERNAL DATA REQUIRED**: CBOE equity put/call ratio. Not in yfinance. Same data pipeline challenge as H7.4. |
| H7.6 | sector-dispersion-regime | `data/strategies/` | alpha-researcher-1 | 0.80+ | 0.10-0.20 | yfinance | P1 | Clean. Dispersion as regime signal is well-motivated. **FAMILY CLASSIFICATION QUESTION**: Could fit F4 (vol regime) or F6 (macro regime) equally well. Placed in F7 by author but mechanism is not purely contrarian. |
| H7.7 | correlation-risk-premium | `data/strategies/` | alpha-researcher-1 | 0.80+ | 0.10-0.20 | yfinance (v1) | P2 | V1 uses ETF-proxy for implied correlation — may be too noisy. Full version needs options data. |
| H7.8 | implied-correlation-regime-signal | `docs/research/hypotheses/` | alpha-researcher-3 | 0.70-0.95 | 0.10-0.20 | yfinance | P2 | Regime signal from implied correlation. Related to H7.7. |

### Family 8: Non-Credit Cross-Market Lead-Lag

No new hypotheses written today. Existing: SOXX-QQQ (passing).

### Family 9: Carry (Cross-Asset) — NEW

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| F9-H1 | commodity-carry-contango | `data/strategies/` | robustness-analyst | 0.60+ | 0.15 | yfinance (USO/DBC) | P1 | **DATA CHALLENGE**: Futures curve not in yfinance. Must proxy via ETF roll yield (tricky). |
| F9-H2 | bond-rolldown-carry | `data/strategies/` | robustness-analyst | 0.60+ | 0.25 | yfinance (TLT/IEF) | P1 | **NEEDS YIELD DATA**: 10y-2y spread from FRED or TLT/IEF return differential proxy. |
| F9-H3 | fx-carry-basket | `data/strategies/` | robustness-analyst | 0.50+ | 0.30 | yfinance (FXA/FXY) | P2 | **CRASH RISK**: FX carry has severe negative skewness. Track B only. VIX filter essential. |
| H9.1 | commodity-term-structure-carry | `docs/research/hypotheses/` | paper-monitor | 0.80+ | low | yfinance (USO) | P1 | **DUPLICATE of F9-H1** — same mechanism, different file. See Section 3. |
| H9.2 | bond-curve-carry-regime | `docs/research/hypotheses/` | paper-monitor | 0.70+ | low | yfinance (TLT/SHY) | P2 | **DUPLICATE of F9-H2** — same mechanism, different file. See Section 3. |
| H9.3 | cross-asset-carry-rotation | `docs/research/hypotheses/` | paper-monitor | 0.60+ | 0.20-0.30 | yfinance | P2 | Multi-asset carry rotation across bonds, commodities, FX. |

### Family 10: Crypto Structural — NEW (Conditional)

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| F10-H1 | btc-eth-regime-cointegration | `data/strategies/` | robustness-analyst | 0.80+ (if cointegration holds) | 0.40 | yfinance (BTC/ETH) | P3 | **HIGH RISK**: Post-Merge cointegration may have broken permanently. Mandatory regime gate. rho 0.40 to F1 during stress is concerning. |
| H10.1 | amihud-illiquidity-regime-timer | `docs/research/hypotheses/` | alpha-researcher-3 | 0.80-1.10 | 0.10-0.20 | yfinance | P1 | Amihud ratio as regime timer. Clean yfinance implementation. |
| H10.2 | volume-liquidity-breakout-signal | `docs/research/hypotheses/` | alpha-researcher-3 | 0.90-1.20 | 0.10 | yfinance | P1 | Triple-trigger forced-selling detection across 15 ETFs. **Canonical version** (shorter draft deleted). |
| H10.3 | cross-etf-liquidity-spread-premium | `docs/research/hypotheses/` | alpha-researcher-3 | 0.80-1.10 | 0.25 | yfinance | P2 | Dollar-neutral liquidity spreads (IWM/SPY, EEM/EFA, HYG/LQD). Track B. **Canonical version** (shorter draft deleted). |

### Family 11: Microstructure — NEW

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H11.1 | treasury-equity-lead-signal | `docs/research/hypotheses/` | alpha-researcher-3 | 0.70-0.95 | 0.15-0.25 | yfinance | P2 | Treasury market microstructure leading equity. Related to F1 but uses different timescale. |
| H11.2 | overnight-return-decomposition | `docs/research/hypotheses/` | alpha-researcher-3 | 0.65-0.90 | 0.10-0.20 | yfinance | P2 | Overnight vs intraday return decomposition signal. Execution risk (requires open-price). |

### Family 12: Dispersion — NEW

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H12.1 | sector-dispersion-regime-indicator | `docs/research/hypotheses/` | alpha-researcher-3 | 0.70-1.00 | 0.10-0.20 | yfinance | P2 | Sector return dispersion as regime indicator. Related to H7.6. |
| H12.2 | correlation-spike-mean-reversion | `docs/research/hypotheses/` | alpha-researcher-3 | 0.65-0.90 | 0.15-0.25 | yfinance | P2 | Cross-sector correlation spike followed by reversion. |

### Family 13: Skewness — NEW (Lowest Priority)

| ID | Slug | Location | Author | Expected SR | rho to F1 | Data | Priority | Robustness Notes |
|----|------|----------|--------|-------------|-----------|------|----------|------------------|
| H13.1 | etf-max-effect | `docs/research/hypotheses/` | alpha-researcher-3 | 0.50-0.80 | 0.05-0.10 | yfinance | P3 | Lottery effect in ETFs — academic evidence primarily for individual stocks. **WEAK**: ETF aggregation dampens effect. |
| H13.2 | skew-weighted-tail-risk-premium | `docs/research/hypotheses/` | alpha-researcher-3 | 0.60-0.90 | 0.05-0.10 | yfinance | P3 | Tail risk premium via return skewness. Requires long history for stability. |

---

## 2. External Data Requirements Summary

| Data Source | Hypotheses Affected | Implementation Effort | Priority |
|-------------|--------------------|-----------------------|----------|
| yfinance (price/volume) | 35+ hypotheses | None (existing) | — |
| yfinance ^SKEW | H4.7 | Minimal (add symbol) | Low |
| yfinance FX (EURUSD=X etc.) | H3.3, H3.4, H3.5 | Minimal (add symbols) | Medium |
| yfinance FXA/FXY | F9-H3 | Minimal (add symbols) | Medium |
| FRED (10y-2y spread) | F9-H2, H6.1 | Moderate (new API) | Medium |
| AAII Sentiment CSV | H7.4 | Moderate (scraper/pipeline) | High (unique signal) |
| CBOE Put/Call Ratio | H7.5 | Moderate (Quandl or CBOE DataShop) | High (unique signal) |
| `arch` Python package | H4.4 | Minimal (pip install) | Low |
| Futures curve data | F9-H1, H9.1 | Significant (new data source) | Medium |

---

## 3. Conflicts, Duplicates, and Taxonomy Issues

### CONFLICT 1: H2.2 vs H2.6 (IWM-SPY Pairs)

Both target the IWM/SPY spread with mean-reversion logic. Differences:
- **H2.2** (alpha-researcher-1): Kalman filter hedge ratio, z-score entry/exit at 2.0/0.0
- **H2.6** (alpha-researcher-2): OLS hedge ratio, z-score entry/exit at 2.0/0.5, 25-day time stop, |z|>3.5 stop-loss

**Recommendation:** Merge into a single hypothesis. Use H2.6's risk management (time stop + stop-loss) with H2.2's Kalman filter. Canonical ID: **H2.2** (first-written takes priority). Delete H2.6 or mark as "variant".

### CONFLICT 2: H2.3 vs H2.7 (HYG-LQD Pairs)

Both target HYG/LQD credit quality spread. Differences:
- **H2.3** (alpha-researcher-1): Kalman filter, z-score 2.0/0.0, Track A
- **H2.7** (alpha-researcher-2): OLS + IEF duration neutralization, z-score 1.8/0.3, Track B

**Recommendation:** H2.7 is actually a better formulation — the IEF duration neutralization is a meaningful improvement. Merge: adopt H2.7's duration-neutral approach with H2.3's Kalman filter. Canonical ID: **H2.3** (but with H2.7's IEF hedge). Both files should be updated.

**F1 OVERLAP CONCERN:** Both HYG and LQD are used in F1 credit lead-lag strategies. The pairs spread signal may correlate with F1 entry timing. This must be tested empirically before promoting either.

### CONFLICT 3: F9-H1 vs H9.1, F9-H2 vs H9.2 (Carry Duplicates)

Two agents independently wrote carry hypotheses:
- **F9-H1** (robustness-analyst) in `data/strategies/commodity-carry-contango/`
- **H9.1** (paper-monitor) in `docs/research/hypotheses/H9.1-commodity-term-structure-carry.md`
- **F9-H2** (robustness-analyst) in `data/strategies/bond-rolldown-carry/`
- **H9.2** (paper-monitor) in `docs/research/hypotheses/H9.2-bond-curve-carry-regime.md`

**Recommendation:** Use the `data/strategies/` files as canonical (they follow the standard artifact location). The `docs/research/hypotheses/` files are supplementary. Canonical IDs: **F9-H1**, **F9-H2**. Mark H9.1/H9.2 as "superseded by F9-H1/F9-H2" or consolidate into the data/strategies files.

### RESOLVED: 6 ID Collisions (2026-03-31 Session 2)

| Original ID | Collision | Resolution |
|-------------|-----------|------------|
| H2.6 | IWM/SPY vs XLE/XLI — two different hypotheses | XLE/XLI renumbered to **H2.9** |
| H6.3 | Credit overlay vs credit momentum rotation | Credit momentum rotation renumbered to **H6.6** |
| H7.6 | sector-dispersion-regime (data/) vs put-call-ratio-contrarian (docs/) | Deleted docs/ version — duplicate of H7.5 |
| H7.7 | correlation-risk-premium (data/) vs aaii-bull-bear-contrarian (docs/) | Deleted docs/ version — duplicate of H7.4 |
| H10.2 | Comprehensive (15-inst) vs draft (3-inst) | Deleted shorter draft |
| H10.3 | Detailed (4-component z-score) vs simpler (Amihud+VIX) | Deleted shorter draft |

### TAXONOMY ISSUE 1: H6.5 (Multi-Asset Carry Rotation)

H6.5 is filed under F6 (Macro Regime Rotation) but its mechanism is carry — ranking assets by trailing risk-adjusted return, which is a carry proxy. This belongs in F9 (Carry), not F6.

**Recommendation:** Reclassify H6.5 as **F9-H4** (multi-asset carry rotation). This avoids double-counting carry exposure under two family labels.

### TAXONOMY ISSUE 2: H7.6 (Sector Dispersion Regime)

Filed under F7 (Sentiment Contrarian) but mechanism is regime-based switching between momentum and mean-reversion based on dispersion. This is closer to F4 (Vol Regime Harvesting) or F6 (Macro Regime) than F7 (Sentiment).

**Recommendation:** Keep in F7 for now (it does use a contrarian signal in low-dispersion regimes), but flag that its correlation with F4/F6 strategies should be checked. If rho > 0.30 to any F4 or F6 strategy, reclassify.

### TAXONOMY ISSUE 3: H6.3 (Credit Momentum Overlay)

H6.3 uses HYG/IEF as a credit momentum signal. This is dangerously close to F1 (Credit Lead-Lag). The difference is frequency (H6.3 uses 21-day momentum vs F1's 5-day lead-lag), but the underlying information source is the same: credit spreads leading equity.

**Recommendation:** Must demonstrate rho < 0.30 to F1 credit strategies in backtest. If rho > 0.30, reclassify as F1 variant and do not count toward family diversification.

---

## 4. Robustness Flags by Severity

### RED FLAGS (likely to fail robustness)

| Hypothesis | Issue | Recommendation |
|------------|-------|----------------|
| H4.3 | 8 trades — FAILED robustness | Reformulate as v2 |
| H5.6 | 5 trades/5yr (1/year) — CANNOT pass any gate | Extend to 20+ years or kill |
| F10-H1 | Post-Merge cointegration breakdown risk | Regime gate mandatory; kill if post-2022 cointegration rejected |
| H5.7 | Day-of-week effects in crypto are notoriously unstable | Low priority; likely data artifact |

### YELLOW FLAGS (marginal, needs attention)

| Hypothesis | Issue | Recommendation |
|------------|-------|----------------|
| H4.5 | 30-60 trades in 5yr — marginal for CPCV | Consider extending backtest window |
| H4.7 | 15-30 SKEW divergence episodes — marginal | Same trade count concern as H4.3 |
| H5.5 | ~50 trades/5yr, expected SR 0.40-0.70 — may not pass Sharpe gate | Useful as diversifier only, not standalone |
| H2.3/H2.7 | F1 correlation risk with HYG/LQD | Must verify rho < 0.30 empirically |
| H6.3 | F1 overlap via credit momentum | Must verify rho < 0.30 empirically |
| H3.7 | F1 overlap via credit trend + F6 overlap via macro regime | Multiple family overlaps |
| F9-H1/H9.1 | Futures curve proxy from yfinance is noisy | Data quality may limit signal fidelity |
| F9-H3 | FX carry crash risk — negative skewness | Track B only; VIX filter essential |

### GREEN (clean, ready for backtest)

| Hypothesis | Notes |
|------------|-------|
| H2.2/H2.6 | IWM-SPY pairs — well-documented, adequate trade count |
| H2.4 | EFA-EEM pairs — clean, low F1 correlation |
| H2.5 | Sector pairs portfolio — diversified across 3 sub-pairs |
| H3.3 | Multi-timeframe TSMOM — well-motivated majority vote |
| H3.4 | Donchian breakout — simplest F3 model, good Occam baseline |
| H4.6 | VIX percentile adaptive — continuous allocation avoids whipsaw |
| H6.1 | Real yield rotation — strong macro signal |
| H7.6 | Sector dispersion regime — clean, yfinance data |

### BACKTESTED — Sprint 1-6 Results (2026-03-31)

| Hypothesis | Family | Sharpe | MaxDD | DSR | Verdict | Sprint |
|------------|--------|--------|-------|-----|---------|--------|
| SPY Overnight Momentum | F11 | 1.044 | 8.68% | 0.982 | **PASS A+B** | 5 |
| AGG-SPY Credit Lead | F1 | 1.012 | 8.34% | 0.981 | **PASS A+B** | 4 |
| LQD-TQQQ Sprint | F1/D | 0.963 | 11.88% | 0.977 | **PASS A** | 6 |
| AGG-QQQ Credit Lead | F1 | 0.888 | 12.30% | 0.965 | **PASS A** | 3 |
| EMB-SPY Credit Lead | F1 | 0.829 | 12.85% | 0.955 | **PASS A** | 4 |
| VCIT-QQQ Credit Lead | F1 | 0.783 | 16.00% | 0.944 | FAIL | 5 |
| HYG-SPY-5D Credit Lead | F1 | 0.752 | 14.02% | 0.937 | FAIL | 4 |
| AGG-EFA Credit Lead | F1 | 0.724 | 12.16% | 0.931 | FAIL | 3 |
| H4.4-v2 GARCH Persist. | F4 | 0.723 | 10.23% | 0.931 | FAIL (best F4) | 3 |
| H3.1 Vol-Scaled TSMOM | F3 | 0.719 | 2.77% | 0.698 | FAIL (best F3) | 1 |
| LQD-QQQ Credit Lead | F1 | 0.719 | 16.00% | 0.928 | FAIL | 5 |
| HYG-QQQ Credit Lead | F1 | 0.609 | 17.44% | 0.893 | FAIL | 5 |
| H4.4 GARCH Regime | F4 | 0.605 | 12.44% | 0.891 | FAIL | 2 |
| H3.1-v2 Relaxed TSMOM | F3 | 0.593 | 3.82% | 0.875 | FAIL (regressed) | 3 |
| H3.5 Skip-Month TSMOM | F3 | 0.585 | 6.44% | 0.882 | FAIL | 2 |
| H4.2 VRP Timing | F4 | 0.539 | 10.69% | 0.604 | FAIL | 1 |
| H6.4 Macro Barometer | F6 | 0.446 | 20.79% | 0.649 | FAIL | 1 |
| VIX Backwardation | F4 | 0.273 | 19.33% | 0.383 | FAIL | 5 |
| H7.2 RSI-2 Contrarian | F7 | -0.023 | 4.20% | 0.184 | **KILL** | 1 |
| H5.4 Turn-of-Month | F5 | -0.226 | 12.81% | 0.321 | **KILL** | 2 |
| H6.2 Cyclical/Defensive | F6 | -0.622 | 37.48% | 0.102 | **KILL** | 2 |

---

## 5. Recommended Canonical Numbering

After resolving conflicts and duplicates, the canonical hypothesis set is:

| Canonical ID | Slug | Family | Files to Keep | Files to Deprecate |
|-------------|------|--------|---------------|-------------------|
| H2.2 | iwm-spy-cointegration-pairs | F2 | `data/strategies/` | Merge H2.6 improvements into H2.2 |
| H2.3 | hyg-lqd-credit-quality-pairs | F2 | `data/strategies/` | Merge H2.7 duration-neutral approach |
| H2.4 | efa-eem-developed-emerging-pairs | F2 | `data/strategies/` | — |
| H2.5 | sector-etf-kalman-pairs | F2 | `data/strategies/` | — |
| H3.3 | multi-timeframe-tsmom-ensemble | F3 | `docs/research/hypotheses/` | — |
| H3.4 | donchian-breakout-multi-asset | F3 | `docs/research/hypotheses/` | — |
| H3.5 | skip-month-cross-asset-momentum | F3 | `docs/research/hypotheses/` | — |
| H3.6 | factor-momentum-sector-rotation | F3 | `docs/research/hypotheses/` | — |
| H3.7 | macro-trend-economic-momentum | F3 | `docs/research/hypotheses/` | — |
| H4.4 | garch-regime-equity-sizing | F4 | `docs/research/hypotheses/` | — |
| H4.5 | sector-vol-dispersion-mr | F4 | `docs/research/hypotheses/` | — |
| H4.6 | vix-percentile-adaptive-allocation | F4 | `docs/research/hypotheses/` | — |
| H4.7 | skew-tail-risk-timing | F4 | `docs/research/hypotheses/` | — |
| H5.4 | turn-of-month-broad-equities | F5 | `docs/research/hypotheses/` | — |
| H5.5 | pre-holiday-equity-drift | F5 | `docs/research/hypotheses/` | — |
| H5.6 | january-small-cap-premium | F5 | `docs/research/hypotheses/` | RED FLAG — extend data or kill |
| H5.7 | btc-tuesday-effect | F5 | `docs/research/hypotheses/` | RED FLAG — likely data artifact |
| H5.8 | overnight-gap-mean-reversion | F5 | `docs/research/hypotheses/` | — |
| F5-H7 | overnight-multi-asset-momentum | F5 | `data/strategies/` | — |
| H6.1 | real-yield-threshold-rotation | F6 | `docs/research/hypotheses/` | — |
| H6.2 | cyclical-defensive-ratio-timing | F6 | `docs/research/hypotheses/` | — |
| H6.3 | credit-momentum-overlay | F6 | `docs/research/hypotheses/` | **VERIFY rho < 0.30 to F1** |
| H7.1 | vix-percentile-spike-reversion | F7 | `data/strategies/` | May merge with H4.3 v2 |
| H7.4 | aaii-sentiment-contrarian | F7 | `data/strategies/` | Needs external data pipeline |
| H7.5 | put-call-ratio-contrarian | F7 | `data/strategies/` | Needs external data pipeline |
| H7.6 | sector-dispersion-regime | F7 | `data/strategies/` | Check F4/F6 correlation |
| H7.7 | correlation-risk-premium | F7 | `data/strategies/` | V1 ETF proxy; full needs options data |
| F9-H1 | commodity-carry-contango | F9 | `data/strategies/` | Deprecate `docs/...H9.1` |
| F9-H2 | bond-rolldown-carry | F9 | `data/strategies/` | Deprecate `docs/...H9.2` |
| F9-H3 | fx-carry-basket | F9 | `data/strategies/` | — |
| F9-H4 | multi-asset-carry-rotation | F9 | Reclassify from H6.5 | Remove from F6 |
| F10-H1 | btc-eth-regime-cointegration | F10 | `data/strategies/` | Conditional — may be killed |

**Total canonical hypotheses: 48** (after collision resolution: 2 renumbered, 4 drafts deleted, merging recommendations pending)

---

## 6. Backtest Priority Queue (Robustness-Informed)

Based on: (a) expected diversification value (low rho to F1), (b) data readiness (yfinance only), (c) expected trade count (>25 in 5yr), (d) mechanism quality.

| Rank | ID | Family | Rationale |
|------|-----|--------|-----------|
| 1 | H3.5 | F3 | Highest expected SR (0.85-1.15), skip-month well-validated, rho to F1 ~0.05. Orthogonal family. |
| 2 | H2.2 | F2 | IWM-SPY pairs — clean, liquid, rho to F1 ~0.15. New family (F2 untested). |
| 3 | H6.2 | F6 | Cyclical/defensive ratio — parsimonious, rho to F1 ~0.15. New family (F6 untested). |
| 4 | H4.4 | F4 | GARCH regime — well-established, rho to F1 ~0.15. New family (F4 partially tested). |
| 5 | H3.3 | F3 | Multi-timeframe TSMOM — majority vote de-noises. Backup if H3.5 fails. |
| 6 | H5.4 | F5 | Turn-of-month — 120 trades, calendar-driven, rho to F1 ~0.15. |
| 7 | H4.6 | F4 | VIX percentile adaptive — continuous, avoids whipsaw. Backup F4 if H4.4 fails. |
| 8 | H6.1 | F6 | Real yield rotation — strong macro signal. Backup F6 if H6.2 fails. |
| 9 | H2.4 | F2 | EFA-EEM pairs — international diversification. Backup F2 if H2.2 fails. |
| 10 | H7.6 | F7 | Sector dispersion regime — yfinance data, ~120 trades/5yr. |

**Rationale:** The top 10 spans 5 families (F2, F3, F4, F5, F6) plus F7. Each family gets a primary and backup candidate. F9 (Carry) is excluded from the immediate queue due to data pipeline needs (futures curve proxies). F10 (Crypto) is excluded due to high risk of mechanism breakdown.

---

## 7. File Location Inconsistency

Hypothesis files are split across two directories:
- `data/strategies/{slug}/hypothesis.yaml` — standard lifecycle location
- `docs/research/hypotheses/{id}.md` — research working directory

**Recommendation:** All hypotheses that advance past the HUNT phase should have their canonical file in `data/strategies/{slug}/hypothesis.yaml` per the lifecycle spec. The `docs/research/hypotheses/` files should be treated as working drafts that get formalized into `data/strategies/` when they enter the mandate stage.

No immediate action needed — this is a process note for future sessions.

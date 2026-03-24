# Domain 2 Analysis: COT Positioning for Commodities & Forex

**Date**: 2026-03-24
**Research input**: `research/results/2.3/output.md`
**Verdict**: COT is a valid *overlay* signal for regime confirmation and crowding detection, not a standalone alpha source. Downgrade from P1 to P2.

---

## Key Findings

1. **COT fails at the portfolio level.** Dreesmann et al. (2023) backtested Williams Commercial Index strategies across all US futures 1986-2020 against 100,000 Monte Carlo portfolios. COT long-only strategies are significant in only 6 individual markets and *underperform* as a diversified portfolio. The signal has been substantially arbitraged away. This is the single most important finding for our system.

2. **Strongest signal in gold, crude oil, and FX -- weak elsewhere.** Wang (2003) found significant monthly abnormal returns for crude oil (+2.64%, t=2.29) and Japanese yen (+2.88%, t=2.75). Sanders et al. (2009) found essentially zero predictive power across 10 agricultural futures. For our 39-asset universe, COT applies meaningfully to at most 8 instruments: GLD, SLV, USO (with caveats), EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF.

3. **Critical data corrections required.** Two CFTC commodity codes were wrong in prior specifications:
   - Gold COMEX: **088691** (not 084691)
   - Silver COMEX: **084691** (not 084651)
   - FX instruments require the **TFF report** (Traders in Financial Futures), not the Disaggregated report

4. **USO is structurally broken for COT signals.** Post-April 2020 restructuring (shifted from front-month to multi-contract-month holdings, 8:1 reverse split), COT data for front-month WTI no longer maps to USO's actual exposures. Contango roll costs alone can exceed 75% annualized in severe contango periods. COT signals should not be applied to USO without replacement or adjustment.

5. **Confirmed parameters:** COT Index = min-max normalization over 156-week (3-year) lookback, 20/80 thresholds, 4-6 week signal horizon, apply Friday-published data on Monday open (not Tuesday -- avoids 3-day look-ahead bias).

---

## Recommendations (Priority-Ordered)

### R1: Add COT data as a regime overlay in `build_context.py`

**What**: Fetch weekly CFTC COT data for 8 target instruments via the COT PRE API (`publicreporting.cftc.gov`). Compute a COT Index (min-max over 156 weeks) per instrument. Surface extreme readings (>80 or <20) as a `cot_crowding` field in `MarketContext`, alongside existing VIX/yield_spread/spy_trend macro inputs. Do NOT use as a standalone trade signal.

**Expected impact**: Adds a crowding/confirmation dimension to regime classification. When COT extremes align with existing momentum signals, conviction increases. When COT shows extreme crowding against a trade, it serves as a caution flag. Estimated improvement: +0.05-0.10 Sharpe from avoided crowded trades, based on Tornell & Yuan (2012) showing 0.48%/week returns from extreme positioning reversals in FX.

**Complexity**: Medium. Requires new data fetcher, new DB table, new indicator computation, and context integration.

**Files changed**:
- `src/llm_quant/db/schema.py` -- Add `cot_weekly` table (symbol, report_date, commercial_net, noncommercial_net, open_interest, cot_index)
- `src/llm_quant/data/fetcher.py` or new `src/llm_quant/data/cot_fetcher.py` -- CFTC PRE API client
- `src/llm_quant/data/indicators.py` -- Add `compute_cot_index()` function
- `src/llm_quant/brain/models.py` -- Add `cot_crowding` field to `MarketContext`
- `src/llm_quant/brain/context.py` -- Integrate COT data into context builder
- `scripts/build_context.py` -- Wire COT fetch into the staleness check
- `config/universe.toml` -- Add `cftc_code` field to applicable assets

### R2: Flag USO for COT exclusion in universe config

**What**: Add a `cot_eligible = false` field to USO in `universe.toml`. This prevents the COT pipeline from generating signals for an instrument where the underlying data mapping is broken. Consider adding a universe-level note documenting the structural break.

**Expected impact**: Prevents false signals from a structurally misaligned instrument. The research is unambiguous that post-2020 USO does not track front-month WTI COT positioning. Small but important -- one bad signal on a 10% position is a 1-2% NAV hit.

**Complexity**: Low. Config change only.

**Files changed**:
- `config/universe.toml` -- Add `cot_eligible` field to commodity assets
- `src/llm_quant/data/universe.py` -- Respect `cot_eligible` flag when selecting COT fetch universe

### R3: Add COT data availability fallback mechanism

**What**: Implement a graceful degradation path for when COT data is unavailable (government shutdowns, CFTC cyber incidents, publication delays). The 2025 shutdown suspended COT reporting for 43 days. The system should: (a) detect stale COT data (>10 days old), (b) log a warning, (c) exclude COT from the signal composite, (d) continue operating on price-based indicators alone.

**Expected impact**: Prevents the system from trading on stale positioning data or crashing during data outages. The 2025 shutdown and 2023 ION cyber incident prove this is not hypothetical.

**Complexity**: Low. Staleness check + conditional logic in the context builder.

**Files changed**:
- `scripts/build_context.py` -- Add COT staleness check parallel to market data staleness
- `src/llm_quant/brain/context.py` -- Conditional inclusion of COT data

---

## Risk Warnings

### COT is "fragile" -- what this means for implementation

The research uses "fragile" in three specific senses:

1. **Portfolio-level failure**: Individual market signals are statistically significant, but combining them across a diversified portfolio does not produce excess returns (Dreesmann et al. 2023). This means we cannot treat COT as an allocation signal the way we treat momentum or regime -- it works as a *filter* or *confirmation*, not a *driver*.

2. **Signal decay over time**: Post-2008 financialization, trader reclassification (hedge funds classified as "commercial"), and broader awareness of COT patterns have eroded the signal. No major quant firm (AQR, Man Group) publishes COT as a primary signal. The edge is shrinking.

3. **Regime dependence**: Chen & Yang (2023) show gold futures positioning flips behavior around a price threshold (~$1,366 in their sample). The same COT reading can be bullish or bearish depending on the regime. Static rules will overfit.

**Implementation guardrail**: COT signals should ONLY be used to modify conviction on trades already identified by the primary momentum/mean-reversion signals. They should never generate trades independently. In the decision prompt, COT should appear as a confirmation/warning flag, not in the signal list.

### Overfitting risks specific to COT

- **Threshold optimization is a trap**: The 20/80 standard thresholds have no rigorous comparative study backing them. Optimizing thresholds on historical data will produce a curve-fit that fails out-of-sample. Use the 20/80 convention without optimization.
- **Lookback sensitivity**: 156-week vs 104-week vs 52-week lookbacks change signal timing materially. Pick 156 weeks (the standard) and do not optimize.
- **Structural breaks invalidate backtests**: Pre-2006 Legacy report data should not calibrate post-2006 signals. Pre-2020 USO data should not calibrate post-2020 USO signals. Five structural breaks are documented in the research. Any backtest spanning these breaks without regime adjustment is misleading.
- **Look-ahead bias from publication delay**: COT measures Tuesday positions but publishes Friday 3:30 PM ET. Using Tuesday as application date inflates backtests by 3 days. Our implementation MUST apply on Monday open following Friday release.

---

## Beads Issues to Create

1. **"Add COT data pipeline as regime overlay"** (P2, task) -- Implement R1: CFTC PRE API fetcher, cot_weekly DB table, COT Index computation, MarketContext integration. Use 156-week lookback, 20/80 thresholds, Monday-open application. Target instruments: GLD, SLV, 5 FX pairs, crude oil (only if USO replaced). Estimated 3-4 sessions of work.

2. **"Flag USO as COT-ineligible and document structural break"** (P2, task) -- Implement R2: Add cot_eligible field to universe.toml, update universe.py to respect it. Document the April 2020 structural break in config comments. Estimated 1 session.

3. **"Add COT data staleness detection and graceful fallback"** (P2, task) -- Implement R3: Detect stale COT data (>10 days), log warning, exclude from context, continue on price-based signals. Account for government shutdown and cyber incident scenarios. Estimated 1 session.

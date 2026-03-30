# Kalshi PM Arb Scan — 2026-03-30

## Overview

Live scan of Kalshi prediction markets for NegRisk arbitrage opportunities.
Scan date: 2026-03-27. Universe: 2,334 ME event series.
Scanner: `scripts/run_pm_scanner.py --source kalshi` (bulk + per-event fallback).
Storage: `data/arb.duckdb`

## All 7 Opportunities Found

| # | Event | Net Spread | Conditions | Vol/24h | Status |
|---|-------|-----------|------------|---------|--------|
| 1 | KXSTATE51-29 (51st state) | 79% | 7 | 338-612 | STALE/POLITICAL — not actionable |
| 2 | KXNBERRECESSQ (recession timing) | 66.4% | multiple | thin | STALE past quarters — not actionable |
| 3 | KXRANKLISTGOOGLESEARCH-26DEC (Google #1 search) | 14% | 11 | thin | Potentially actionable, thin liquidity |
| 4 | KXSENATEVAR-26 (VA Senate Republican primary) | 5% | 7 | 10-97 | MOST ACTIONABLE — active market |
| 5 | KXNY4R-26 (NY-04 Republican primary) | 1% | 3 | 12-42 | Marginal after 2% Kalshi fee |
| 6-7 | (two additional sub-1% spreads) | <1% | — | — | Below fee threshold |

## Key Finding

The Kalshi arb scanner is working correctly. NegRisk arbitrage opportunities are real
but practical yield for liquid markets is 1-5% net spread after Kalshi's ~2% fee.
The high apparent spreads (79%, 66.4%) are stale or illiquid — not tradeable.

For active election markets with vol_24h > 50 per condition, the realistic net spread
range is 1-5%. This is consistent with the efficient-market hypothesis on liquid events
and with documented Kalshi arb research.

## 3 Actionable Trades with Kelly Sizing

Kelly formula for NegRisk arb: `f* = net_spread / (1 + net_spread)`
Portfolio cap: max 5% NAV per event (risk constraint given fill uncertainty).

### 1. KXSENATEVAR-26 — VA Senate Republican Primary (BEST)

- Net spread: 5%
- Conditions: 7 candidates
- Vol 24h: 10-97 per condition (moderate, primary season active)
- Kelly f*: 0.05 / 1.05 = **4.76%** -> capped at **5% NAV** = ~$5,000
- Mechanics: Buy all 7 conditions at prevailing asks; total outlay < $1 per condition basket;
  collect guaranteed $1 regardless of winner. Profit = net spread after fees.
- Risk: Fill mechanics on thin books — not all conditions may fill simultaneously.
  Partial fills leave unhedged directional exposure.
- Verdict: **Paper trade candidate** — test fill mechanics before allocating real capital.

### 2. KXRANKLISTGOOGLESEARCH-26DEC — Google #1 Search (CONDITIONAL)

- Net spread: 14%
- Conditions: 11
- Vol 24h: thin
- Kelly f*: 0.14 / 1.14 = **12.3%** -> capped at **5% NAV** = ~$5,000
- Risk: Thin liquidity likely prevents full execution without moving the market.
  The 14% spread may close to 0% after market impact on 11 conditions.
- Verdict: Monitor only. Do not trade until vol_24h > 50 per condition.

### 3. KXNY4R-26 — NY-04 Republican Primary (MARGINAL)

- Net spread: 1%
- Conditions: 3
- Vol 24h: 12-42 per condition
- Kelly f*: 0.01 / 1.01 = **0.99%** -> effective size: **~1% NAV** = ~$1,000
- Risk: 2% Kalshi fee wipes the spread entirely. Only viable if fee waiver applies.
- Verdict: **Do not trade** at standard fee rate.

## Summary: Only 1 Actionable Trade

After fees and liquidity filtering, only KXSENATEVAR-26 clears the bar for a paper trade.
The Google search ranking market is worth monitoring as liquidity builds.

## Recommended Monitoring Schedule

Primary season runs through June 2026 (major state primaries). During this period:

- **Weekly re-scan**: Re-run `scripts/run_pm_scanner.py --source kalshi` every Monday
- **Target filter**: Events with 5-15 conditions AND vol_24h > 50 per condition
- **Spread threshold**: Net spread > 3% after 2% Kalshi fee
- **Markets to watch**:
  - Senate/House primary markets (June 2026 primary calendar)
  - KXSENATEVAR-26 — re-check weekly until filled or expired
  - KXRANKLISTGOOGLESEARCH-26DEC — watch for liquidity growth

Re-scan command:
```bash
cd /c/Projects/llm-quant && PYTHONPATH=src python scripts/run_pm_scanner.py --source kalshi
```

## Decision: Paper Trade KXSENATEVAR-26?

**Recommendation: YES — as a fill mechanics test only.**

Rationale:
- 5% net spread clears the fee threshold (barely — 3% net after 2% fee)
- Active primary market with moderate liquidity (vol 10-97/condition)
- Kelly sizing suggests ~5% NAV maximum; paper capital allocation is low-risk
- Primary purpose: validate that all 7 conditions can be filled simultaneously
  without partial-fill directional exposure
- If paper fills succeed cleanly: proceed to live execution in next primary cycle
- If partial fills occur: document mechanics, build fill-order optimization before live

Paper trade entry criteria:
- All 7 conditions available simultaneously at prevailing ask
- Total basket cost < $0.97 (leaving $0.03 net per $1 basket after fees)
- Confirmed net spread > 3% at time of execution

## Infrastructure Notes

- Scanner code: `scripts/run_pm_scanner.py --source kalshi`
- Database: `data/arb.duckdb`
- Bulk scan + per-event fallback handles API rate limits correctly
- Next enhancement: build simultaneous-fill order manager to reduce partial-fill risk

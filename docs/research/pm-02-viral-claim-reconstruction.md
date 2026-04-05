# PM-02: Viral Claim Reconstruction

## Research Brief

**Program**: Track C — Structural Arbitrage
**Status**: Research-only (no live trading)
**Created**: 2026-04-06

---

## Core Question

**Can extreme prediction-market compounding claims be structurally plausible under realistic execution assumptions?**

This is a reconstruction exercise, not a validation exercise. The goal is to determine *what would have had to be true* for the viral outcome to be achievable — not to prove the story true.

### Subquestions

1. What minimum edge per trade is required to compound from $X to $Y over the claimed time period?
2. What fill rates, trade sizes, and market depths are needed to sustain that compounding path?
3. How sensitive is the compounding path to realistic fee structures, slippage, and execution latency?
4. Do the claimed strategies (NegRisk compounding, YES+NO rebalancing, combinatorial arb, latency capture) produce the required edge under simulation?

---

## Hypothesis Families

### H1: NegRisk Compounding
**Mechanism**: Multi-outcome NegRisk events where sum(YES prices) < 1.00 - fees create guaranteed profit per round-trip. Compounding reinvests profits across events.
**Testable prediction**: NegRisk mispricing events with spread > 2% (net of fees) occur at frequency F and average depth D sufficient to sustain X% daily compounding.
**Falsification**: If historical NegRisk spreads (net of fees) are < 1% on 90%+ of observations, OR average depth at exploitable prices is < $100, the compounding path is implausible.

### H2: YES+NO Rebalancing
**Mechanism**: Buying both YES and NO at prices summing to < $1.00, locking in per-unit profit. Rebalancing across correlated markets as prices shift.
**Testable prediction**: Complement mispricing (YES + NO < $0.98) occurs in at least N markets per day with sufficient depth to deploy $K per occurrence.
**Falsification**: If complement mispricings > 2% occur in < 5 markets/day with depth > $500, the rebalancing path cannot sustain claimed returns.

### H3: Combinatorial Arbitrage
**Mechanism**: Cross-event logical dependencies create mispricings. If Event A implies Event B, but P(A) > P(B), arbitrage exists.
**Testable prediction**: Logically dependent event pairs with price inconsistency > 3% exist at frequency sufficient for the claimed compounding.
**Falsification**: If Claude-detected logical dependencies produce < 10 actionable opportunities per week with spreads > 3%, the edge is too sparse.

### H4: Latency / Microstructure
**Mechanism**: Polymarket orderbooks lag external price feeds (Binance BTC) by 2-10 seconds during sharp moves, creating directional edge in 5-minute markets.
**Testable prediction**: BTC 5-min direction is deterministic >85% of the time 10 seconds before close, and Polymarket prices lag sufficiently to capture this.
**Falsification**: If post-fee, post-slippage expected value per trade is < 0.5%, latency capture cannot sustain compounding.

---

## Evidence Policy

| Tier | Definition | Decision weight |
|------|-----------|----------------|
| 0 | Structural proof (math, invariant, smart contract logic) | Full — can drive PURSUE |
| 1 | Live data confirmation (real-time API observation, on-chain verification) | High — can drive PURSUE with caveats |
| 2 | Historical backtest (replay of archived data with realistic execution model) | Moderate — supports PURSUE only with Tier 0/1 corroboration |
| 3 | Analogical reasoning (academic papers, similar platform studies) | Low — can support DEFER, never PURSUE alone |
| 4 | Anecdotal (social media claims, unverified screenshots, influencer narratives) | Zero — can motivate research question only, never drive decisions |

---

## Why PM-02 Is Research-Only

1. **AU legal constraint**: Australian residents blocked from Polymarket trading (ACMA, Aug 2025). Read-only data access permissible.
2. **Reconstruction framing**: We are testing plausibility, not building a production system. The output is a verdict document, not a trading bot.
3. **Viral claims require skepticism**: Extraordinary claims require extraordinary evidence. Default assumption: the claim is false until structurally proven plausible.
4. **No capital at risk**: All simulations use synthetic bankroll paths. No USDC, no wallets, no on-chain transactions.

---

## Required Simulator Realism Assumptions

Every simulation MUST model:

1. **Fees**: Category-specific taker fees (0-7.2%), maker rebates (0-50%). Fee formula: `fee = C × p × feeRate × (p × (1-p))^exponent`.
2. **Slippage**: Orderbook-depth-aware. If trade size > 10% of visible depth at target price, model partial fills and price impact.
3. **Fill probability**: Not all limit orders fill. Model fill rate as function of distance from midpoint, time-in-force, and market activity.
4. **Latency**: Minimum 66ms end-to-end for optimized pipeline (10ms WS receipt + 1ms processing + 5ms signing + 50ms network). Model cancel/replace races.
5. **Market depth**: Use historical orderbook snapshots (pmxt Archive or Telonex) for depth profiles. Do not assume infinite liquidity.
6. **Resolution risk**: Model probability of disputed resolution (1.3% base rate). For cross-event arb, model semantic non-fungibility risk.
7. **Compounding friction**: Reinvestment delay (settlement ~2s on Polygon, but capital redeployment requires new order cycle). Model idle capital time.

---

## Required Outputs Per Hypothesis Family

### H1 (NegRisk Compounding)
- Distribution of NegRisk spreads (net of fees) across all multi-outcome events
- Depth profile at exploitable price levels
- Simulated compounding path with realistic fills
- Minimum required frequency × edge × depth to match viral claim
- Verdict: PLAUSIBLE / IMPLAUSIBLE / INDETERMINATE

### H2 (YES+NO Rebalancing)
- Frequency of complement mispricing > 2% across all active markets
- Depth at mispriced levels
- Simulated round-trip P&L per rebalance
- Capital efficiency (idle time, settlement delay)
- Verdict: PLAUSIBLE / IMPLAUSIBLE / INDETERMINATE

### H3 (Combinatorial Arbitrage)
- Catalog of logically dependent event pairs with price inconsistency
- Estimated actionable opportunities per week
- Average spread, depth, and duration of opportunity
- Claude detection accuracy on synthetic test cases
- Verdict: PLAUSIBLE / IMPLAUSIBLE / INDETERMINATE

### H4 (Latency / Microstructure)
- BTC 5-min direction predictability from external feeds
- Polymarket lag measurement (seconds behind Binance/CME)
- Post-fee, post-slippage EV per trade
- Required win rate and frequency to sustain compounding
- Verdict: PLAUSIBLE / IMPLAUSIBLE / INDETERMINATE

---

## Final Synthesis Output

A single document: `docs/research/results/pm-02-synthesis.md`

Contents:
1. Per-hypothesis verdict with evidence tier citations
2. Combined plausibility assessment: "Could a systematic trader with optimal infrastructure have achieved the claimed outcome?"
3. Sensitivity analysis: which assumptions, if wrong, would flip the verdict?
4. Lessons for Track C strategy design (what edges are real vs. fake)

---

## Relationship to Track C Mandate

PM-02 is a research exercise within Track C (Structural Arbitrage). It does NOT aim to build a deployable strategy. Findings may inform future Track C strategy development (e.g., if NegRisk compounding is plausible, it motivates a full lifecycle for a NegRisk strategy). But PM-02 itself produces verdicts, not trading systems.

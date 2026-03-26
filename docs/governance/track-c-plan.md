# Track C: Niche Arbitrage — Debate Synthesis & Implementation Plan

## Date: 2026-03-27
## Status: APPROVED (debate consensus)

## Debate Summary

Four-agent team (skeptic, pragmatist, risk-officer, builder) evaluated 5 proposed
strategies plus 1 emergent strategy (prediction market arbitrage). Unanimous consensus
on the final prioritized shortlist below.

---

## Strategy Verdicts

### 1. Prediction Market Arbitrage (Polymarket) — PROCEED (Priority 1)
**Verdict: UNANIMOUS PROCEED**

The pragmatist agent independently identified and built this as the highest-value
opportunity. Already implemented as `src/llm_quant/arb/` module.

**Why it wins:**
- Perfect stack fit: DuckDB persistence, Python, free public API (Gamma)
- Academic backing: Saguillo et al. 2025 documented $17.3M in NegRisk arb
- Two arb types: NegRisk (buy all YES when sum < 1.0) and single rebalance (YES+NO < 1)
- Claude combinatorial detector adds a unique LLM-powered edge (logical dependency detection)
- Zero market beta (prediction markets uncorrelated with equities)
- No special broker needed (Polymarket direct, no IB required)

**Skeptic concerns (mitigated):**
- Liquidity: filtered by min $1K 24h volume per condition
- Regulatory: Polymarket's CFTC status uncertain — monitor but proceed
- Competition: NegRisk arb window shrinking but combinatorial arb (LLM-detected) is novel

**Risk framework:**
- Max allocation: 15% of total portfolio ($15K)
- Max per market: $2K
- Max exchange concentration: 100% (Polymarket only for now)
- Kill switch: halt if spreads consistently < 3% net (edge gone)

### 2. CEF Discount Mean-Reversion — PROCEED (Priority 2)
**Verdict: CONDITIONAL PROCEED**

**Why it works:**
- Strong academic backing (CUNY: Sharpe 1.519, 14.9% annualized)
- Wide current discounts (-6.89% avg Q4 2025) = attractive entry timing
- Uses existing yfinance + IB infrastructure
- Genuinely uncorrelated with credit-equity lead-lag portfolio
- Monthly rebalancing = low execution complexity

**Skeptic concerns (accepted):**
- No convergence mechanism (unlike ETFs, CEF discounts can persist indefinitely)
- Need $50K+ for diversification across 10-20 CEFs (competes with Track A capital)
- Tax treatment of distributions complex (ROC, qualified vs ordinary)

**Risk framework:**
- Max allocation: 15% of portfolio ($15K initially, grow to $25K)
- Max per CEF: $3K
- Rebalance: monthly
- Kill switch: halt if avg portfolio discount widens >3 standard deviations from entry

**Implementation:** Can reuse existing backtest engine with minor modifications.
CEF prices are in yfinance; NAV data needs scraping from CEFConnect.

### 3. Crypto Funding Rate Arbitrage — CONDITIONAL (Priority 3)
**Verdict: CONDITIONAL — proceed only after Polymarket and CEF are stable**

**Why conditional, not immediate:**
- Requires crypto exchange accounts (Binance/OKX/Bybit) + KYC
- Counterparty risk is real (FTX precedent) — need 3+ exchange diversification
- VIP fee tiers ($50K+ per exchange) needed for profitability
- Our $100K capital spread across 3 exchanges = $33K each = still retail fees
- Operational complexity: 8-hourly monitoring, WebSocket feeds, liquidation management

**Skeptic fatal flaw:**
- At retail fee levels (0.1% maker), a 10% annualized gross return becomes ~4% net
- Need VIP status (0.02% maker) which requires high volume — chicken-and-egg problem

**Risk framework (if proceed):**
- Max 25% per exchange, min 3 exchanges
- Max leverage: 2x
- Stop-loss: 5% unrealized loss on any position
- Exchange health monitoring: withdrawal delays > 24h = reduce position

### 4. Merger Arbitrage — CONDITIONAL (Priority 4)
**Verdict: CONDITIONAL — valuable but complex event-driven pipeline**

**Why conditional:**
- Proven Sharpe (1.46 Eurekahedge) but deal pipeline needs constant monitoring
- Asymmetric risk: 3-5% gains per deal vs 20-40% losses on deal breaks
- Needs 10+ concurrent deals for diversification = $5K per deal = manageable
- SEC EDGAR scraping + deal scoring = significant pipeline build

**Implementation note:** This is a research project in itself. Build the deal
monitoring pipeline first, paper trade for 6 months before allocating capital.

### 5. VIX Contango Harvesting — REJECT
**Verdict: UNANIMOUS REJECT**

**Fatal flaw (skeptic + risk officer agree):**
- XIV lost 96% in ONE DAY (Feb 5, 2018). SVXY lost 80-91%.
- Even at 5% allocation ($5K), a vol spike can wipe the position to zero
- The 0.4-0.6 Sharpe doesn't compensate for existential tail risk
- Contradicts Track C mandate (MaxDD < 10%, near-zero beta)
- Not arbitrage — it's a directional vol bet with asymmetric downside

**Decision:** VIX contango violates the fundamental Track C principle of near-riskless
mispricings. Rejected permanently. If desired, could explore under Track B (aggressive).

### 6. Crypto Cash-and-Carry Basis — DEFER
**Verdict: DEFER to after funding rate arb is operational**

**Why defer:**
- Basis declining (25% → 5% from 2024 to 2025) — edge compressing
- Requires same exchange infrastructure as funding rate arb
- Lower priority than funding rate (which has continuous cash flow vs locked capital)
- Build this AFTER exchange accounts are set up for Strategy 3

---

## Final Priority Order

| # | Strategy | Verdict | Effort | Expected Sharpe | Portfolio ρ |
|---|----------|---------|--------|-----------------|-------------|
| 1 | Polymarket NegRisk + Combinatorial | PROCEED | Done (v1 built) | 2.0-4.0+ | ~0.00 |
| 2 | CEF Discount Mean-Reversion | PROCEED | 3-5 days | 1.5-1.9 | ~0.05 |
| 3 | Crypto Funding Rate | CONDITIONAL | 5-7 days | 1.5-3.0 | ~0.05 |
| 4 | Merger Arbitrage | CONDITIONAL | 7-10 days | 0.9-1.5 | ~0.10 |
| 5 | VIX Contango | REJECT | — | — | — |
| 6 | Crypto Basis | DEFER | 2-3 days* | 1.0-2.0 | ~0.05 |

*After exchange infrastructure from Strategy 3 is built.

---

## Implementation Timeline

### Week 1 (now): Polymarket Scanner v1
- [x] Build Gamma API client (gamma_client.py)
- [x] Build NegRisk scanner (scanner.py)
- [x] Build Claude combinatorial detector (detector.py)
- [x] Build CLI runner (run_pm_scanner.py)
- [x] DuckDB schema (schema.py)
- [ ] Run first live scan
- [ ] Validate against known historical arbs
- [ ] Paper trade first opportunities

### Week 2-3: CEF Discount Pipeline
- [ ] Build CEF data scraper (NAV from CEFConnect, prices from yfinance)
- [ ] Create CEFDiscountStrategy class in STRATEGY_REGISTRY
- [ ] Backtest quintile strategy on 5-year history
- [ ] Run through full lifecycle (hypothesis → robustness)

### Week 4-6: Crypto Funding Rate (conditional)
- [ ] Set up exchange accounts (Binance, OKX, Bybit)
- [ ] Build funding rate data pipeline via CCXT
- [ ] Create FundingRateArbStrategy class
- [ ] Backtest on historical funding rate data
- [ ] Paper trade with minimum capital

### Month 2-3: Merger Arb Pipeline
- [ ] Build SEC EDGAR scraper for merger announcements
- [ ] Create deal scoring model
- [ ] Build MergerArbStrategy class
- [ ] Paper trade through 2-3 deal cycles

---

## Capital Allocation Plan

**Phase 1 (now):** $100K allocated as current (Track A 70%, Track B 30%)
**Phase 2 (after CEF validated):** Reallocate to Track A 60%, Track B 20%, Track C 20%
**Phase 3 (after 3+ Track C strategies stable):** Track A 50%, Track B 20%, Track C 30%

Track C allocation breakdown:
- Polymarket: 40% of Track C
- CEF: 35% of Track C
- Funding rate: 25% of Track C (when active)

---

## Track C Risk Framework (Risk Officer Design)

### Kill Switches (Track C specific)
1. **Exchange outage** — halt all positions on affected exchange
2. **Spread collapse** — halt if target spread < breakeven for 3 consecutive checks
3. **Counterparty stress** — reduce position if exchange withdrawal delays > 24h
4. **Beta breach** — halt if realized beta to SPY > 0.15 over rolling 30 days
5. **Correlation breach** — halt if correlation to Track A > 0.30

### Monitoring Cadence
- Polymarket: scan every 4 hours during active markets
- CEF: daily NAV comparison at market close
- Crypto funding: every 8 hours (funding payment cycle)
- Merger arb: daily deal spread monitoring

### Position Limits
- Max 20% of Track C capital per strategy
- Max $2K per individual prediction market
- Max $3K per individual CEF
- Max 25% on any single crypto exchange

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-27 | Initial plan from 4-agent debate. Polymarket (PROCEED), CEF (PROCEED), Funding Rate (CONDITIONAL), Merger (CONDITIONAL), VIX (REJECT), Basis (DEFER). |

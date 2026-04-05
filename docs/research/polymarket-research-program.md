# Polymarket Research Program -- Track C Structural Arbitrage

**Date:** 2026-04-06
**Track:** C (Structural Arbitrage)
**Status:** Research-only -- no live trading, no capital deployment
**Mandate:** Market-neutral returns from structural pricing inefficiencies
**Benchmark:** Risk-free rate (3-month T-bill)

---

## 1. Why Prediction Markets

This section establishes what makes prediction markets structurally different from the ETF and crypto instruments in Tracks A/B/D. These differences are not incidental -- they determine which edges can exist, which analytical tools apply, and which assumptions from traditional quant finance must be abandoned.

### Binary outcome vs. continuous price

The defining structural difference: prediction market contracts pay exactly $1.00 or $0.00 at resolution. There is no continuous return distribution. The bounded [0,1] price range means:

- **Sharpe ratio interpretation changes.** Returns are capped. A contract bought at $0.95 that resolves YES pays 5.26% gross. The same contract resolving NO loses 100% of invested capital. The asymmetry at price extremes is severe and has no parallel in ETF trading.
- **Position sizing requires different math.** The Kelly criterion for binary outcomes (`f* = (bp - q) / b` where `b = (1 - market_price) / market_price`) produces position sizes that are hypersensitive to probability estimation error near 50%. Fractional Kelly (0.25x) is not merely advisable -- it is structurally necessary because overconfidence in a binary world destroys capital exponentially.
- **Terminal value is deterministic.** Every position resolves to a known value. There is no "sell at a loss" in the equity sense -- there is only "wait for resolution or exit to another participant." This creates a fundamentally different exit decision framework.

### Resolution risk as a unique risk factor

Resolution risk has no analogue in ETF trading. When a position in SPY goes against you, the underlying asset still exists and can recover. When a Polymarket resolution goes against you -- including through oracle manipulation -- the loss is total and irreversible.

Concrete instances:

- **Ukraine mineral deal (March 2025):** A UMA whale holding ~5M tokens (~25% of voting power) forced an incorrect resolution on a $7M market. Polymarket refused refunds. Traders who were factually correct lost everything.
- **Zelenskyy suit market ($242M volume):** A dispute over whether a blazer constitutes a "suit" caused YES prices to crash from $0.19 to $0.04 during the resolution process. The ambiguity was in the question design, not in the facts.
- **Semantic non-fungibility across platforms:** During the 2024 government shutdown debate, Polymarket resolved "Yes" and Kalshi resolved "No" on the same event due to different resolution criteria. "Identical" markets are not identical.

The UMA Optimistic Oracle's economic security model is structurally insufficient: with a market cap of ~$85M, only ~5M tokens are needed to control a vote, and the attack cost can be lower than profits from manipulated positions. The August 2025 MOOV2 upgrade restricted proposal submissions to a 37-address whitelist, but DVM vote manipulation remains feasible.

**Implication for research:** Any strategy evaluation that treats resolution as deterministic is wrong. Resolution risk must be modeled as a non-negligible tail event (base rate ~1.3% for disputes, higher for subjective or politically contentious markets).

### Orderbook dynamics: the CLOB on Polygon

Polymarket operates a hybrid-decentralized Central Limit Order Book -- orders match off-chain for speed, settle on-chain via Polygon for auditability. This architecture differs from traditional CLOBs in one critical way: **supply is dynamically created.** When opposing limit orders match (YES at $0.65 + NO at $0.35 = $1.00), a new token pair is minted atomically from USDC.e collateral. Supply is theoretically unlimited and entirely demand-driven.

Key microstructure characteristics:

- **Extreme liquidity concentration.** Of 295,000+ historical markets, just 505 contracts with >$10M volume captured 47% of total volume. 63% of active short-term markets have zero volume in the past 24 hours. The bimodal distribution means strategies must be designed for either the hyper-liquid top or the illiquid long tail, not both.
- **Kyle's lambda declined 50x** in the presidential election market (from 0.518 in early 2024 to 0.01 by October 2024), reflecting maturation. Liquid markets now require 3.5x more notional volume to move price than Kalshi.
- **No designated market makers.** Polymarket incentivizes liquidity through a rewards program ($5M+/month for sports and esports) and maker rebates. Reported 3-4 serious LPs per market. 8 of the top 10 profitable wallets are bots.
- **Order mirroring:** A buy of 100 YES at $0.40 automatically displays as a sell of 100 NO at $0.60. This maintains the $1.00 invariant but creates mechanically linked visible depth.

### Fee structure

Polymarket's fee model, overhauled March 30, 2026, is category-specific and nonlinear:

| Category | Fee rate | Exponent | Maker rebate | Peak effective rate |
|----------|----------|----------|-------------|-------------------|
| Geopolitics | 0 (fee-free) | -- | -- | 0% |
| Sports | 0.03 | 1.0 | 25% | 0.75% |
| Economics | 0.03 | 0.5 | 25% | 1.50% |
| Finance | 0.04 | 1.0 | 50% | 1.00% |
| Politics | 0.04 | 1.0 | 25% | 1.00% |
| Culture | 0.05 | 1.0 | 25% | 1.25% |
| Crypto | 0.072 | 1.0 | 20% | 1.80% |

Fee formula: `fee = C * p * feeRate * (p * (1-p))^exponent`. Fees peak at 50% probability and approach zero at extremes. **Only takers pay fees; makers never pay and receive daily rebates.** The `feeRateBps` must be fetched dynamically per market -- hardcoding is a confirmed source of error (the existing scanner hardcodes 2%, which is incorrect for most categories).

The fee model has changed 4+ times in 18 months. Any strategy whose edge is smaller than the peak fee for its category is structurally fragile to fee changes.

### Regulatory landscape

The regulatory environment is actively contested with no stable equilibrium:

- **US:** The Kalshi v. CFTC ruling (September 2024) declared election contracts are not "gaming." Polymarket acquired QCEX for $112M and launched Polymarket US as a CFTC-regulated DCM (December 2025, invite-only). But 10+ Congressional bills have been introduced in 2026 to restrict prediction markets, and state AGs in Arizona, Nevada, New Jersey, and Maryland are actively challenging federal preemption. The CFTC has sued three states to assert exclusive jurisdiction (April 2026).
- **Australia (binding constraint for this project):** ACMA added Polymarket to its national blocked gambling website list on August 13, 2025, directing ISPs to block access. ASIC's binary options ban (extended until 2031) adds a second legal layer. Read-only data access and research remain permissible; live order placement is prohibited.
- **Other jurisdictions:** France, Belgium, Germany, Italy, Singapore, Switzerland, Poland, Portugal, Hungary, and others have restricted or banned access. OFAC-sanctioned countries (Cuba, Iran, North Korea, Syria, Russia, Belarus) are fully blocked.
- **Insider trading risk:** Federal prosecutors (SDNY) are actively examining whether PM trades constitute insider trading. Documented cases include Israeli military personnel using classified information, a Google insider profiting >$1M on Year in Search rankings, and traders earning $553K betting on Iran strikes 71 minutes before news broke.

**Implication:** Regulatory disruption is an existential risk for any PM-dependent strategy. It cannot be backtested. It must be monitored continuously.

---

## 2. Research Scope and Constraints

### Research-only mandate

This is a hard constraint, not a preference. Australian residents are geoblocked from Polymarket trading since August 2025 (ACMA enforcement). The legal boundary is unambiguous:

**Permissible from Australia:**
- Accessing all read-only API endpoints (Gamma, Data, public CLOB data)
- Storing and analyzing historical market data
- Building paper trading simulations
- Developing and backtesting algorithmic strategies
- Using open-source SDKs (MIT-licensed)
- Querying on-chain Polygon data
- Publishing research

**Prohibited from Australia:**
- Placing trades (orders from AU IPs are automatically rejected)
- Using VPNs to bypass geoblocking (violates ToS Section 2.1.4)
- Hosting trading bots that place live orders
- Promoting Polymarket to Australians

### Simulation-first approach

All strategy evaluation uses historical replay and paper trading. No live execution. This introduces a structural limitation: paper trading systematically overestimates real performance because it cannot model:

- Adverse selection (informed traders picking off stale quotes during news events)
- Liquidity withdrawal during high-volatility periods
- Oracle manipulation dynamics
- Fill quality degradation as strategy capital scales

These limitations must be acknowledged in every paper-trading result. A passing paper trade is necessary but not sufficient evidence.

### Evidence tiers

Evidence quality determines what decisions it can support. This framework is borrowed from PM-02 (Viral Claim Reconstruction) and applies to all Track C PM research:

| Tier | Definition | Decision weight |
|------|-----------|----------------|
| 0 | Structural proof (math, invariant, smart contract logic) | Full -- can drive PURSUE |
| 1 | Live data confirmation (real-time API observation, on-chain verification) | High -- can drive PURSUE with caveats |
| 2 | Historical backtest (replay of archived data with realistic execution model) | Moderate -- supports PURSUE only with Tier 0/1 corroboration |
| 3 | Analogical reasoning (academic papers, similar platform studies) | Low -- can support DEFER, never PURSUE alone |
| 4 | Anecdotal (social media claims, unverified screenshots, influencer narratives) | Zero -- can motivate research question only, never drive decisions |

**Rule:** No strategy advances past the hypothesis stage on Tier 3-4 evidence alone. Academic papers documenting the favorite-longshot bias (Tier 3) motivate a hypothesis; they do not validate a strategy.

### Anti-overfitting controls

The same discipline that governs Track A/B applies here without relaxation:

- **DSR >= 0.95** (Track C gate) with family trial counting. Every tested PM hypothesis counts against the family's trial budget. The more hypotheses tested within a family, the higher the Sharpe bar for the next one to pass DSR.
- **CPCV OOS/IS > 0** (mean and median). Combinatorial purged cross-validation is the primary guard against in-sample overfitting. PM strategies must pass this gate before advancing to paper trading.
- **No post-hoc parameter adjustment.** Research spec freeze before backtest is mandatory. If a backtest fails, the next attempt is a NEW hypothesis with a NEW spec, not a parameter tweak on the failed one.
- **Shuffled signal test.** Randomize signal dates, keep return series intact, run 1000 shuffled backtests. Real Sharpe must exceed the 95th percentile of shuffled Sharpes. This is the single most powerful fraud detector and it applies to PM strategies without modification.

### Paper-trading gates specific to Track C

Track C paper trading adds gates beyond the standard robustness pipeline:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Sharpe | >= 1.50 | Structural arb should deliver higher risk-adjusted returns than directional trading |
| Max drawdown | < 10% | Market-neutral strategies should have tight drawdown |
| Beta to SPY | < 0.15 | Must be genuinely market-neutral |
| Min trades | >= 50 | Statistical significance for binary outcome strategies |
| Min days | >= 30 | Minimum observation window for paper trading |

---

## 3. Candidate Edge Families

Six candidate edge families have been identified from the research literature and platform microstructure analysis. For each family, the default assumption is that the claimed edge is fake until proven otherwise. The priority assessment reflects what can actually be tested given AU legal constraints and solo developer bandwidth.

### F-PM1: Latency / Stale Quote Capture

**Mechanism:** Polymarket orderbooks lag external price feeds (Binance, CME) by 2-10 seconds during sharp moves. BTC 5-minute direction markets are the primary venue. The removal of Polymarket's 500ms taker quote delay in early 2026 shifted the competitive landscape, but the lag between external price feeds and PM orderbook updates remains exploitable in theory.

BTC 5-minute direction is reportedly deterministic >85% of the time ~10 seconds before close, and Polymarket prices lag sufficiently to capture this. Documented arbitrage windows that paid 3-5% in 2024 now yield 1-2%.

**Data requirements:**
- Real-time WebSocket feed from Polymarket CLOB (`wss://ws-subscriptions-clob.polymarket.com`)
- Real-time price feed from Binance/CME for reference pricing
- Sub-second timestamp precision for lag measurement
- Historical orderbook snapshots (pmxt Archive or Telonex) for backtest

**Execution assumptions:**
- Latency budget: ~66ms minimum end-to-end (10ms WS receipt + 1ms processing + 5ms signing + 50ms network)
- Taker fees: 1.80% peak (crypto category)
- Fill probability: depends on aggressive limit order placement and whether stale quotes are still resting
- Requires London-proximate infrastructure for sub-100ms execution (Polymarket servers on AWS eu-west-2)

**Failure modes:**
- Competitive compression: professional HFT firms (Susquehanna, DRW have built PM desks) crowd out retail latency arb
- Fee erosion: crypto category carries the highest fee (1.80% peak), eating into already-thin edges
- Polymarket may further reduce latency asymmetry (as they did by removing the 500ms taker delay)
- Chainlink Data Streams now used for crypto price resolution, reducing classic oracle arb

**Simulation-friendliness:** POOR. Latency arb is intrinsically about real-time execution. Historical replay cannot capture the competitive dynamics, queue priority, or fill races that determine profitability. Simulations will systematically overestimate edge.

**Priority: MAYBE NEVER**

Rationale: Requires sub-100ms infrastructure colocated near London, which is architecturally incompatible with a solo AU-based research operation. The competitive landscape includes institutional desks with dedicated infrastructure. The edge is compressing (3-5% in 2024 to 1-2% in 2026). Even if the edge exists, we cannot test it with adequate fidelity in simulation, and we cannot execute it live from Australia. This family is permanently deferred unless the operational constraint changes.

---

### F-PM2: Correlated Market Inconsistency (NegRisk Arbitrage)

**Mechanism:** NegRisk multi-outcome events (e.g., "Who will win the presidential election?" with 17+ candidates) enforce mutual exclusivity -- exactly one outcome must resolve YES. When the sum of all YES prices deviates from 1.00, an arbitrage exists:

- `sum(YES) < 1.00 - fees`: Buy all YES outcomes, one must pay $1.00. Guaranteed profit of `1.00 - sum(YES) - total_fees`.
- `sum(YES) > 1.00 + fees`: Buy all NO outcomes. The one outcome that resolves NO (all but the winner) pays $1.00 per NO contract. Net profit from the complete set.

This is a Tier 0 edge -- the math is structural and invariant-based. The question is not whether the edge exists but whether it persists at sufficient magnitude and depth to be worth capturing.

**Data requirements:**
- Gamma API market metadata (to identify NegRisk events via `negRisk: true` and `negRiskMarketId`)
- CLOB API orderbook depth for each outcome token (to assess executable depth at exploitable prices)
- Historical NegRisk spread data for frequency analysis
- Existing infrastructure: `scanner.py` already implements NegRisk scanning, `gamma_client.py` already fetches market data

**Execution assumptions:**
- Fees: category-dependent. Political NegRisk events are 1.00% peak; geopolitical NegRisk events are fee-free.
- Slippage: must walk the book across ALL outcomes simultaneously. If any one outcome lacks depth, the arb is incomplete.
- Fill probability: batch orders (up to 15 per request) enable simultaneous execution, but partial fills on one leg create unhedged exposure.
- Capital lockup: positions may be locked for weeks or months until resolution. Opportunity cost is real.

**Failure modes:**
- Spread too thin: if NegRisk spreads (net of category-specific fees) are consistently < 1%, the edge does not cover operational overhead
- Depth insufficient: if exploitable price levels show < $100 in executable depth, compounding is impossible
- Leg risk: partial fills create directional exposure in what should be a market-neutral trade
- Resolution risk: if the NegRisk event resolves incorrectly (UMA manipulation), the hedged position is destroyed
- Capital lockup: long-dated events tie up capital for months with no interim return

**Simulation-friendliness:** GOOD. NegRisk spreads are observable from historical API data and on-chain records. The spread calculation is deterministic. Depth profiles can be estimated from orderbook snapshots. The primary simulation gap is fill probability for simultaneous multi-leg execution.

**Priority: NOW**

Rationale: Existing infrastructure already supports NegRisk scanning (`scanner.py`). The edge is Tier 0 (structural/mathematical). The key research question is empirical: how often do exploitable spreads (net of category-specific fees) occur, at what depth, and for how long? This can be answered with data we can already collect. PM-02 Hypothesis H1 (NegRisk Compounding) directly targets this question.

---

### F-PM3: Cross-Venue Inconsistency (Polymarket vs. Kalshi)

**Mechanism:** Price discrepancies between prediction market platforms persist because:

1. Different user bases with different information sets and biases
2. Capital fragmentation (funds locked on one platform cannot arbitrage the other)
3. Different fee structures affecting equilibrium prices
4. Different resolution criteria creating "semantic non-fungibility"

Documented evidence: prices for identical contracts diverged by 3-8 cents during the 2024 presidential election. YES + NO sums deviated from $1.00 on at least one platform on 62 of 65 days before the election. An estimated $40M in arbitrage profits were extracted in a single recent year, with $39.6M from single-market rebalancing.

Polymarket leads Kalshi in price discovery, particularly during high-liquidity periods. This lead-lag relationship creates directional signals even when true arbitrage (simultaneous opposing positions) is not possible.

**Data requirements:**
- Polymarket pricing via Gamma/CLOB API (existing `gamma_client.py`)
- Kalshi pricing via REST API (existing `kalshi_client.py`)
- Market matching: mapping "identical" events across platforms (existing `kalshi_detector.py` handles this for combinatorial arb)
- Historical price series from both platforms for lead-lag analysis

**Execution assumptions:**
- Fees: Polymarket category-specific (0-1.80%) + Kalshi 30bps taker / 20bps maker rebate
- Settlement risk: resolution criteria differ across platforms. The government shutdown precedent (Polymarket YES, Kalshi NO on the same event) is not an edge case -- it is a structural feature.
- Capital lockup: positions on both platforms lock capital until resolution. True hedged arb requires 2x capital.
- Latency: cross-venue arb windows are sub-second for competitive opportunities. Lead-lag signals persist longer (minutes to hours).

**Failure modes:**
- Semantic non-fungibility: "identical" markets resolve differently. This is not risk-free arbitrage -- it is a correlated bet with resolution basis risk.
- Competitive compression: automated bots and commercial services (ArbBets claims "100+ opportunities daily") are already active. 78% of arb opportunities in low-volume markets fail due to execution inefficiencies.
- Regulatory divergence: Kalshi operates under CFTC regulation with different market creation and resolution processes. Platform-specific regulatory changes can invalidate the arb thesis.
- Capital efficiency: locking capital on two platforms for weeks/months reduces return on capital below the threshold where the operational complexity is justified.

**Simulation-friendliness:** MODERATE. Historical price series from both platforms can be compared. Lead-lag relationships are measurable. The simulation gap is resolution divergence -- the probability that ostensibly identical markets resolve differently cannot be estimated from price data alone. Requires a resolution-criteria audit of historical market pairs.

**Priority: LATER (after PM-02 complete)**

Rationale: Existing infrastructure partially supports this (both `gamma_client.py` and `kalshi_client.py` are built and tested). The edge is documented in peer-reviewed research (Tier 2-3). But the semantic non-fungibility problem means this is NOT risk-free arbitrage, and the simulation will systematically underestimate resolution basis risk. Best sequenced after PM-02 provides empirical data on NegRisk spread frequency, which will calibrate execution realism assumptions for cross-venue work.

---

### F-PM4: Orderbook Imbalance / Microstructure

**Mechanism:** Multiple microstructure patterns create structural edges for participants who understand orderbook dynamics:

1. **Maker-taker wealth transfer:** Makers earn +1.12% average excess return vs. takers at -1.12% (Becker 2026). The gap is widest in Entertainment (4.79pp) and World Events (7.32pp), narrowest in Finance (0.17pp). The Polymarket fee structure (zero maker fees + rebates) explicitly subsidizes this edge.
2. **Buy-sell depth asymmetry:** Documented 11x buy-side vs. sell-side depth in NBA markets. Asymmetric depth creates predictable price impact patterns.
3. **Price-time priority exploitation:** The matching engine uses price-time priority. Understanding queue position dynamics enables strategic order placement.
4. **Heartbeat mechanism:** All open orders auto-cancel if heartbeat is not sent every 10 seconds. Disconnected bots leave stale quotes that can be picked off.

**Data requirements:**
- Real-time orderbook stream (WebSocket or RTDS feed)
- Historical orderbook snapshots (pmxt Archive: hourly Parquet files)
- Trade-level data (who is maker vs. taker, not available from public API -- must be inferred)
- Continuous connection for heartbeat-based strategies

**Execution assumptions:**
- Maker strategies require persistent order management with 5-second heartbeat cadence
- Cancel/replace cycles must complete in < 200ms to avoid adverse selection
- Post-only orders (reject if they would cross the spread) are the primary tool for ensuring maker status
- Minimum capital: $10K+ deployed across multiple markets to achieve meaningful spread income

**Failure modes:**
- Adverse selection: informed traders pick off stale quotes during news events. A single adverse selection event (40-50 point price jump) can erase days of spread income. Binary settlement makes inventory errors catastrophic.
- Competition: only 3-4 serious LPs per market, but they are sophisticated. Reports of $700-800/day at peak declining as competition increases.
- WebSocket instability: documented bug where data stops flowing after ~20 minutes despite healthy connection. Requires proactive reconnection every 10-15 minutes.
- Infrastructure requirements: persistent server, London-proximate for latency, 24/7 monitoring

**Simulation-friendliness:** POOR. Orderbook microstructure strategies depend on real-time dynamics that historical snapshots (hourly granularity at best) cannot capture. Fill probability, queue position, and adverse selection are all functions of the live competitive environment. Hourly snapshots from pmxt miss the intra-hour dynamics where microstructure edge is concentrated.

**Priority: MAYBE NEVER**

Rationale: The maker-taker edge is real and documented, but capturing it requires persistent infrastructure (24/7 server near London), real-time orderbook management (heartbeat every 5-10 seconds), and live execution capability. All three are incompatible with the AU legal constraint and solo developer operational model. The historical data granularity available (hourly snapshots) is insufficient to simulate microstructure strategies with adequate fidelity. Even if simulated results look promising, the gap between simulation and reality is too large to bridge without live trading infrastructure.

---

### F-PM5: Event Resolution / Settlement Workflow

**Mechanism:** The UMA Optimistic Oracle resolution process creates a tradeable window:

1. **Resolution proposal:** A whitelisted proposer stakes $750 USDC and asserts an outcome. A 2-hour challenge window begins.
2. **During the challenge window:** Markets remain tradeable. If the proposed resolution is correct, YES (or NO) tokens at the proposed outcome are worth $1.00 but may still trade below par. The 2-hour window creates a known-expiry trade with bounded risk.
3. **Dispute dynamics:** ~1.3% dispute rate. If disputed, the adapter auto-resets. A second dispute escalates to DVM voting (48-72 hours). During DVM escalation, markets trade as a "bet on how UMA will decide," with prices reflecting dispute outcome probabilities rather than event fundamentals.
4. **Resolution timing patterns:** Undisputed resolution takes ~2 hours. DVM escalation takes 48-96 hours. Capital locked during resolution cannot be redeployed.

The edge hypothesis: systematic buying of tokens at the proposed resolution side during the challenge window, when prices have not yet fully converged to $1.00, captures the "resolution discount." The 98.5% undisputed resolution rate means this is a high-frequency, high-win-rate strategy with rare but severe losses (when the proposed resolution is incorrect or overturned).

**Data requirements:**
- On-chain monitoring of UMA `ProposePrice` and `DisputePrice` events (via Polygon RPC or The Graph subgraph)
- Resolution timing data: time from proposal to finalization for historical markets
- Price behavior during challenge windows (requires tick-level or minute-level data during the 2-hour window)
- Dispute outcome data: how often do disputes overturn proposals, and what happens to prices during disputes

**Execution assumptions:**
- Timing-sensitive: the 2-hour challenge window is the trade window. Must detect proposals quickly and act within minutes.
- Fees: category-dependent. Near-resolution prices are near $0.00 or $1.00, where fees approach zero (good for this strategy).
- Position sizing: the $750 bond creates a known cost to dispute. Strategy capital should not exceed the level where it makes disputing profitable for an adversary.

**Failure modes:**
- Oracle manipulation: the entire edge is predicated on proposals being correct 98.5% of the time. If the dispute rate increases (due to strategic manipulation or contentious markets), the win rate drops and catastrophic losses mount.
- Slim margin: if prices during the challenge window are already at $0.99+ for the proposed side, the gross edge is < 1% and does not survive fees.
- Selection bias in dispute data: the 1.3% dispute rate is an average. Subjective markets, politically contentious markets, and high-value markets likely have much higher dispute rates.
- Adversarial dynamics: if this strategy becomes known, adversaries could propose incorrect resolutions to bait trades, then dispute.

**Simulation-friendliness:** MODERATE. Historical resolution data is available on-chain. Price behavior during challenge windows can be reconstructed from tick-level data if available. The primary simulation gap is the adversarial response: will dispute rates change if this strategy deploys capital?

**Priority: LATER**

Rationale: This family requires deep UMA oracle expertise and on-chain event monitoring infrastructure that does not yet exist in the codebase. The edge is conceptually sound (Tier 0 math on the 98.5% undisputed rate), but the failure mode (oracle manipulation) is precisely the existential risk documented in Section 1. Best sequenced after the NegRisk research validates the fundamental data pipeline and fee modeling infrastructure.

---

### F-PM6: Retail Overreaction / Narrative Overshoot

**Mechanism:** Three documented behavioral biases create systematic mispricing:

1. **Favorite-longshot bias (FLB):** Contracts below 20 cents underperform their implied odds; contracts above 80 cents outperform (Becker 2026, 72.1M trades on Kalshi). At 1-cent contracts, takers win only 0.43% of the time vs. 1% implied -- a -57% mispricing. At 5-cent contracts, 4.18% win rate vs. 5% implied -- a -16.36% mispricing. Systematic selling of longshots and buying near-certainties generates positive expected value in theory.

2. **YES optimism tax:** Takers disproportionately buy YES contracts. At equivalent prices, YES underperforms NO at 69 of 99 price levels. Dollar-weighted excess returns: -1.02% for YES buyers vs. +0.83% for NO buyers -- a 1.85pp gap. NO contracts at 1 cent deliver +23% expected value while YES at 1 cent delivers -41% EV.

3. **Overreaction / negative serial correlation:** 58% of Polymarket national presidential markets show negative serial correlation (Clinton & Huang 2025). Large single-day price moves tend to partially reverse within 72 hours. Price spikes from the Biden-Trump debate largely reversed (noise); the Trump assassination attempt repricing persisted (information). Distinguishing noise from information is the critical modeling challenge.

**Data requirements:**
- Historical price series for resolution outcome analysis (CLOB `/prices-history`)
- Resolution outcomes (on-chain via The Graph or Gamma API `closed` status with `outcomePrices`)
- Cross-sectional price data across many markets for FLB analysis
- Daily price change series for serial correlation measurement
- Volume and days-to-resolution for filtering

**Execution assumptions:**
- Fees: FLB strategies trade at price extremes (<$0.20 or >$0.80) where fees approach zero -- structurally favorable.
- Holding period: FLB positions are hold-to-resolution. Capital is locked until the event resolves.
- Position sizing: fractional Kelly at 0.25x. At extreme prices (buying YES at $0.95), Kelly allocation is large but absolute edge is small. At low prices (selling YES at $0.05), Kelly allocation is small but absolute edge is larger.
- Volume filter: exclude markets with < $50K 24h volume. Below this threshold, mispricing is extreme but depth is zero.

**Failure modes:**
- After-fee insufficiency: the FLB is well-documented but after transaction costs (now 1-1.8% on Polymarket), net profitability is uncertain. Only ~16.8% of Polymarket wallets show a net gain -- this includes all strategy types, not just FLB exploitation.
- Publication decay: McLean & Pontiff (2016) found anomaly returns 26% lower OOS and 58% lower post-publication in equity markets. The FLB is now widely documented. The magnitude may have compressed since the studies were published.
- Selection bias in academic evidence: Becker's data is from Kalshi, not Polymarket. Reichenbach & Walther (2025) found NO general favorite-longshot bias on Polymarket specifically, though extreme longshots still appeared to perform well. The evidence is contradictory across studies.
- Overreaction strategy timing: the 72-hour reversal window requires knowing whether a price move is noise (will reverse) or information (will persist). The Biden debate (noise) vs. assassination attempt (information) distinction is obvious ex post but not ex ante.
- Concentration risk: FLB strategies involve holding many low-probability positions. Correlation between outcomes (e.g., all political markets moving together during a regime shift) creates hidden portfolio risk.

**Simulation-friendliness:** GOOD. FLB analysis requires only historical prices and resolution outcomes -- both available. Overreaction analysis requires daily price series, also available. The primary simulation gap is position-level execution: can you actually buy at $0.95 or sell at $0.05 with sufficient depth? Orderbook data at price extremes is needed for execution realism.

**Priority: LATER (after PM-02 complete)**

Rationale: This is the richest family in terms of academic evidence, but the evidence is contradictory (Becker vs. Reichenbach & Walther on FLB magnitude). The right sequencing is: (1) build the historical data pipeline during NegRisk research (NOW phase), (2) use that pipeline to run FLB and overreaction analysis on Polymarket-specific data, (3) evaluate whether the documented biases survive on this specific platform with this specific fee structure. The data infrastructure built for F-PM2 directly enables F-PM6 analysis.

---

## 4. Data Infrastructure

### Gamma API: market discovery and metadata

**Existing implementation:** `src/llm_quant/arb/gamma_client.py` (~420 lines, tested). Supports paginated market listing, single market fetch, parsing into `Market` and `ConditionPrice` dataclasses. Includes US API fallback for geo-blocked access.

**Known gaps:**
- Does not store `clobTokenIds` (required for CLOB API calls)
- Does not store `questionId` (required for UMA resolution tracking)
- Only fetches active markets (no `fetch_resolved_markets()` for historical research)

**Rate limit:** ~4,000 requests/10 seconds. Sufficient for research scanning.

### CLOB API: orderbook, pricing, trade execution

**Status:** No dedicated client in the codebase. The Gamma client handles market discovery; CLOB endpoints are needed for:

- `GET /book?token_id={id}` -- Full L2 orderbook snapshot (50/10s rate limit)
- `GET /price?token_id={id}&side=BUY` -- Best price (100/10s)
- `GET /midpoint?token_id={id}` -- Midpoint between best bid/ask
- `GET /spread?token_id={id}` -- Current spread
- `GET /prices-history` -- Historical price time series (configurable 1min to weekly intervals)

**Known limitation:** `/prices-history` for resolved markets may only return 12+ hour granularity regardless of fidelity setting. This constrains backtest resolution for historical markets.

**Authentication:** Read-only endpoints require no auth. Trading endpoints require L1 (EIP-712 wallet signature) deriving L2 credentials (HMAC-SHA256). Not needed for research-only operation.

### US API: CFTC-regulated fallback

**Existing implementation:** Fallback path in `gamma_client.py`. Returns at most ~20 markets without API key. Full access requires Ed25519 authentication.

**Use case:** Fallback when Gamma API is geo-blocked (e.g., from US-based CI/CD or cloud infrastructure). Not a primary research data source.

### Kalshi API: cross-venue data

**Existing implementation:** `src/llm_quant/arb/kalshi_client.py` (~360 lines, tested). Public REST API client for market listing and pricing.

**Use case:** Cross-venue price comparison for F-PM3 research. Lead-lag analysis between Polymarket and Kalshi.

### External reference feeds

For latency and cross-asset research:

- **Binance:** BTC/ETH real-time price feeds for F-PM1 latency measurement. Free REST API, WebSocket available.
- **CME FedWatch:** Fed rate probabilities for comparison with Polymarket fed markets.
- **FRED API:** 800K+ economic time series for macro-event market calibration.
- **Metaculus API:** 12+ years, 23,400+ prediction questions with community prediction timeseries and resolution outcomes. Primary calibration benchmark for forecasting model development.
- **GDELT:** Global event data with ~15-minute delay. Free. Potential signal source for news-driven event markets.

### Historical data sources for replay/backtest

| Source | Coverage | Format | Granularity | Cost |
|--------|----------|--------|-------------|------|
| CLOB `/prices-history` | Per-token, active + resolved | JSON | 1min-weekly (12h+ for resolved) | Free (rate-limited) |
| pmxt Data Archive | Hourly orderbook + trade snapshots | Parquet | Hourly | Free |
| PolyBackTest | Sub-second to 1-minute orderbook | API + bulk | Sub-second | Freemium |
| Polygon on-chain (The Graph) | Settlement events, token transfers | GraphQL | Block-level (~2s) | 100K queries/month free |
| Dune Analytics | Transaction-level on-chain data | SQL | Weekly refresh | Free (community dashboards) |
| Telonex | Tick-level trades + orderbook | Parquet | Tick-level | Paid |
| Kaggle "Polymarket Prediction Markets" | Historical dataset | CSV | Varies | Free |

### Data recording requirements

For research reproducibility, the following data must be recorded and versioned:

1. **Market metadata snapshots:** Daily export of all active market metadata from Gamma API (condition IDs, questions, categories, NegRisk status, end dates, CLOB token IDs)
2. **Price snapshots:** Hourly midpoint prices for all markets in the research universe (via CLOB API or pmxt Archive)
3. **Resolution outcomes:** Event resolution data (which outcome resolved YES, resolution timestamp, any dispute history)
4. **Fee schedule changelog:** Record any fee schedule changes with effective dates

Storage: DuckDB (`data/quant.db`), consistent with the existing Track A/B/C data infrastructure. Schema extensions proposed in the ADR (`docs/architecture/polymarket-track.md`).

---

## 5. Anti-Overfitting Controls

Prediction market strategies face the same overfitting risks as traditional quant strategies, plus additional risks specific to the binary outcome domain. The existing anti-overfitting framework applies without relaxation, with PM-specific additions.

### Pre-registered hypotheses (mandatory)

Every PM research hypothesis must be registered BEFORE any data analysis:

```yaml
hypothesis_id: PM-XX
edge_family: F-PM[1-6]
mechanism: [structural rationale -- WHY does this edge exist?]
testable_prediction: [specific, measurable, falsifiable]
falsification_criteria: [what result would DISPROVE this hypothesis?]
data_requirements: [sources, granularity, minimum history]
execution_assumptions: [fees, slippage, fill rate, latency]
evidence_tier: [0-4]
resolution_risk: [specific to this market type]
```

This format is enforced by the Polymarket Researcher agent (`.claude/agents/polymarket-researcher.md`). No backtest runs without a frozen research spec.

### Falsification criteria before testing

For each hypothesis, state what would disprove it BEFORE running any backtest. Examples:

- **F-PM2 (NegRisk):** "If NegRisk spreads net of fees are < 1% on 90%+ of observations, OR average depth at exploitable prices is < $100, the compounding hypothesis is falsified."
- **F-PM6 (FLB):** "If contracts priced below $0.20 win at rates >= their implied probability (after adjusting for the category-specific fee), the favorite-longshot bias does not exist on Polymarket."
- **F-PM6 (Overreaction):** "If serial correlation of daily price changes is >= 0 in a majority of markets with >$100K volume, mean reversion is not present."

If you cannot state falsification criteria, the hypothesis is not testable. Do not proceed.

### Family trial counting

Every tested PM hypothesis counts against its edge family's trial budget. The Deflated Sharpe Ratio (DSR) adjusts for the number of trials attempted:

- Family F-PM2: 4 sub-hypotheses defined in PM-02 (H1-H4). Each backtest counts as a trial.
- Family F-PM6: FLB, YES bias, and overreaction are three distinct hypotheses within one family. Each counts independently.

The practical consequence: to pass DSR >= 0.95 after 5 trials, a strategy needs Sharpe >= ~0.85. After 10 trials, ~0.95. After 20 trials, ~1.05. Trial inflation is the primary mechanism by which data snooping is penalized.

### No post-hoc parameter adjustment

The research spec freeze is a hard gate:

1. Write the hypothesis, including all parameters (threshold values, lookback windows, position sizing rules).
2. Freeze the spec. Record the hash.
3. Run the backtest with the frozen parameters.
4. If it fails, do NOT adjust parameters and re-run. Instead: write a NEW hypothesis with NEW parameters, counting it as a new trial against the family budget.

This is the single most important discipline for prediction market research, because the small number of historical markets and the binary outcome structure make it trivially easy to overfit. With only ~100 resolved markets in a given category, a handful of parameter tweaks can make any strategy look profitable in-sample.

### Evidence tier constraints on decision-making

The evidence tier system (Section 2) constrains what actions each tier of evidence can support:

| Tier | Can it advance a hypothesis? | Can it drive capital allocation? |
|------|------------------------------|--------------------------------|
| 0 (structural proof) | Yes | Yes, with paper trading validation |
| 1 (live data confirmation) | Yes, with caveats | Yes, with paper trading validation |
| 2 (historical backtest) | Only with Tier 0/1 corroboration | Not alone |
| 3 (academic papers) | Only to DEFER, never PURSUE | Never |
| 4 (anecdotal) | Only to motivate research question | Never |

The practical rule: a promising backtest (Tier 2) of the FLB strategy (motivated by Tier 3 academic evidence) cannot advance to paper trading unless corroborated by live data confirmation (Tier 1) showing the bias persists in current Polymarket pricing.

---

## 6. Paper-Trading Gates (Track C Specific)

Track C paper trading gates are stricter than Track A/B because structural arbitrage should deliver higher risk-adjusted returns with lower drawdowns than directional trading. These gates apply to all Track C PM strategies.

### Gate definitions

| Gate | Threshold | Measurement | Rationale |
|------|-----------|-------------|-----------|
| Persistence | >= 0.50 | Fraction of paper-trading days where the strategy had a position or signal | Edge must persist across time, not cluster in a few days |
| Fill rate | >= 0.80 | Fraction of signals that would have been executable at the simulated price | Execution must be achievable -- a strategy that generates signals but cannot fill them is worthless |
| Capacity | <= 10% | Strategy's daily volume as a fraction of market's daily volume | Strategy must not consume a dominant fraction of market liquidity -- if it does, the edge will self-destruct |
| Days elapsed | >= 30 | Calendar days of paper trading | Minimum observation period for statistical significance with binary outcomes |
| Sharpe | >= 1.50 | Annualized Sharpe ratio of daily mark-to-market P&L | Structural arb should produce higher SR than directional trading |

### Existing infrastructure

The `paper_gate.py` module (`src/llm_quant/arb/paper_gate.py`, ~410 lines) implements 4 of these 5 gates (persistence, fill rate, capacity, days elapsed) for the existing Kalshi arbitrage paper trader. The Sharpe gate needs to be added, and the gate framework needs to be extended to handle Polymarket-specific paper trades.

### What paper trading cannot test

Paper trading systematically overestimates the following:

1. **Fill quality:** Simulated fills assume the orderbook state at signal time persists through execution. In reality, aggressive participants move the book within milliseconds of the signal.
2. **Adverse selection:** Paper trading cannot model the informational content of the counterparty. In live trading, the fact that your order filled may itself be negative information (the counterparty knows something you do not).
3. **Resolution risk:** A paper trade that "resolves correctly" in simulation does not capture the tail risk of oracle manipulation or ambiguous resolution.
4. **Market impact:** Strategy capital at scale moves prices. A strategy that works at $100/trade may fail at $1,000/trade due to its own price impact.

These limitations do not invalidate paper trading -- they constrain the confidence level of paper trading results. A paper-trading Sharpe of 1.50 should be discounted by at least 30-50% for live execution reality.

---

## 7. Rejection Criteria

When to kill a PM research hypothesis. Any single criterion triggers REJECT -- no second chances, no parameter tweaking.

### Statistical rejection

- **Shuffled signal test fails (p > 0.05):** If the strategy's real Sharpe does not exceed the 95th percentile of 1000 shuffled-signal backtests, the signal has no predictive power. This test controls for time-in-market bias, volatility harvesting, and directional drift.
- **DSR < 0.95:** After accounting for all trials in the family, the deflated Sharpe is insufficient. This gates against data snooping across multiple hypotheses.
- **CPCV OOS/IS <= 0:** Out-of-sample performance is zero or negative relative to in-sample. The strategy is overfit.

### Economic rejection

- **Edge < 1% net of ALL costs:** After category-specific fees, estimated slippage (walking the book), resolution risk discount, and capital lockup opportunity cost, the net edge must exceed 1%. Below this threshold, fee model changes or spread compression will erase the edge.
- **Fill rate < 50% in simulation:** If fewer than half of generated signals could have been executed at the simulated price, the strategy is execution-infeasible. Note this is a REJECTION threshold, not the paper-trading gate (which requires 80%). Strategies between 50-80% fill rate may be DEFERRED for further analysis rather than immediately rejected.
- **Capacity < $100/day:** If the strategy's total daily deployable capital (given market depth constraints) is less than $100, the edge is real but economically irrelevant. The operational overhead of monitoring, data collection, and execution infrastructure is not justified.

### Risk rejection

- **Resolution risk undocumented or assumed "negligible":** Any hypothesis that does not explicitly model resolution risk -- including the probability of incorrect resolution, the cost of capital lockup during disputes, and the tail risk of oracle manipulation -- is rejected on process grounds. Resolution risk is the defining risk factor of prediction markets. Assuming it away is not skepticism-compatible.
- **Beta to SPY > 0.15:** Track C strategies must be market-neutral. If a PM strategy's returns correlate meaningfully with SPY, it is a disguised directional bet and belongs in Track A/B, not Track C.
- **Drawdown > 10%:** Track C mandate. Structural arbitrage strategies should not experience equity-like drawdowns.

### Process rejection

- **No pre-registered hypothesis:** Strategy was developed by exploring data first, then constructing a post-hoc rationale. This is HARKing (Hypothesizing After Results are Known) and invalidates the research.
- **Parameter adjustment after seeing results:** Research spec was modified after backtest results were observed. This invalidates the trial and counts as a trial against the family budget with no passing result.

---

## 8. Prioritized Roadmap

### NOW (active)

These are the current workstreams. They build foundational infrastructure that enables all later research.

**PM-02: Viral Claim Reconstruction (research exercise)**
- Status: Research brief complete (`docs/research/pm-02-viral-claim-reconstruction.md`)
- Scope: Test structural plausibility of extreme compounding claims via 4 hypothesis families (NegRisk compounding, YES+NO rebalancing, combinatorial arb, latency capture)
- Output: Per-hypothesis verdict (PLAUSIBLE / IMPLAUSIBLE / INDETERMINATE) with evidence tier citations
- Value: Calibrates execution realism assumptions for all future PM research. Even if the viral claim is implausible, the exercise produces empirical data on NegRisk spread frequency, complement mispricing depth, and fee impact.

**NegRisk scanner validation (existing code in arb/)**
- Status: `scanner.py` implements NegRisk and single-market rebalance arb detection
- Gap: Uses hardcoded 2% fee constant instead of category-aware fee model
- Action: Validate scanner output against live Gamma API data. Record NegRisk spread time series. Feed into PM-02 H1 analysis.
- Dependency: `pm_fees.py` (proposed in ADR) for category-aware fee correction

**Gamma API data recording**
- Status: `gamma_client.py` fetches live market snapshots
- Gap: No persistent recording of historical snapshots for research replay
- Action: Build daily snapshot recorder that persists market metadata, prices, and NegRisk event structures to DuckDB
- Value: Creates the historical dataset needed for F-PM2 and F-PM6 research

### LATER (after PM-02 complete)

These workstreams depend on infrastructure and empirical findings from the NOW phase.

**F-PM3: Cross-venue arbitrage (Polymarket vs. Kalshi)**
- Prerequisites: Validated fee model, historical price data pipeline, resolution-criteria audit methodology
- Scope: Measure price divergence frequency, lead-lag relationships, and resolution basis risk across platforms
- Key question: Are cross-venue price discrepancies large enough (net of fees and resolution basis risk) to survive as a systematic edge?
- Sequencing rationale: The NegRisk research validates data pipeline and fee modeling. Cross-venue work adds the second platform dimension.

**F-PM6: Narrative overshoot (event category analysis)**
- Prerequisites: Historical price series with resolution outcomes, daily price change series for serial correlation analysis
- Scope: Test FLB magnitude on Polymarket-specific data (not Kalshi), measure serial correlation by category, build overreaction detection signal
- Key question: Do the biases documented in academic literature (Becker on Kalshi, Clinton & Huang on elections) persist on Polymarket specifically, and do they survive category-specific fees?
- Sequencing rationale: Requires the historical data pipeline built during the NOW phase. Benefits from PM-02 execution realism findings.

**F-PM5: Event resolution workflow**
- Prerequisites: On-chain monitoring infrastructure (UMA event listener), resolution timing dataset
- Scope: Measure resolution discount during challenge windows, dispute frequency by market type
- Key question: Is the 98.5% undisputed resolution rate sufficient to sustain a systematic strategy, and how does the strategy's presence alter dispute incentives?
- Sequencing rationale: Requires on-chain data infrastructure not currently in the codebase. Lower priority than F-PM2/F-PM3/F-PM6 because the edge is narrower and the infrastructure cost is higher.

### MAYBE NEVER

These workstreams are permanently deferred unless operational constraints change.

**F-PM1: Latency capture**
- Blocker: Requires sub-100ms infrastructure colocated near London (Polymarket servers on AWS eu-west-2). Incompatible with AU-based solo research operation.
- Competitive landscape: Institutional desks (Susquehanna, DRW) with dedicated infrastructure already operate in this space. Edge is compressing from 3-5% (2024) to 1-2% (2026).
- Simulation fidelity: Cannot be tested with adequate realism without live execution capability.
- Condition to revisit: Only if operational base moves to a non-geoblocked jurisdiction with London-proximate infrastructure.

**F-PM4: Orderbook microstructure (market making)**
- Blocker: Requires persistent live orderbook stream (WebSocket with 5-second heartbeat), 24/7 server infrastructure, and live execution capability.
- Known bug: WebSocket data stops after ~20 minutes; requires proactive reconnection.
- Simulation gap: Hourly orderbook snapshots (best available from pmxt) miss the sub-minute dynamics where microstructure edge lives.
- Condition to revisit: Only if the project acquires live trading capability and persistent infrastructure. The maker-taker edge (+1.12% documented) is real but requires infrastructure that does not exist in this project.

---

## 9. Relationship to Existing Tracks

### Track C is market-neutral

Track C strategies must demonstrate near-zero correlation with Track A/B:

- **Beta to SPY < 0.15** (Track C gate). Prediction market strategies that are directionally correlated with equity markets belong in Track A/B, not Track C.
- **Cross-strategy correlation < 0.30** (kill switch). If Track C PM strategies begin correlating with Track A strategies during stress periods, the diversification benefit is illusory. This kill switch is defined in the Track C mandate but not yet implemented (part of the 11/17 governance gap).

### Benchmark: risk-free rate

Track C strategies are benchmarked against T-bills, not 60/40 or SPY. Any positive alpha above the risk-free rate justifies the operational complexity. This is a lower return bar but a higher Sharpe bar (SR >= 1.50) than Track A (SR >= 0.80) or Track B (SR >= 1.00).

### Separate pod, separate capital allocation

PM research strategies operate in their own pod (`pod_id: track_c_pm`) within the existing multi-strategy pod framework. Capital allocation:

- **Current:** 0% of portfolio capital. Track C is in research phase.
- **Target (post-validation):** 10-20% of portfolio capital, per the research tracks specification.
- **Gate:** Cannot promote to production until all 17 Track C mandate gates pass. Currently 4 of 17 are implemented.

### Cannot promote to production until Track C governance gaps close

The March 2026 mandate audit found 11 of 17 Track C gates unimplemented:

- Missing: Sharpe/MaxDD/Beta gates in the robustness pipeline for Track C
- Missing: Kill switches (exchange outage, funding rate reversal, spread collapse, beta breach, correlation spike)
- Missing: Central risk manager integration for Track C strategies

The PM research module adds research infrastructure -- it does not add execution risk. Research can proceed while governance gaps are addressed in parallel. But no PM strategy can be promoted to production capital allocation until the governance gaps are closed. This is tracked in beads issues.

### Integration with existing robustness pipeline

The proposed architecture (from `docs/architecture/polymarket-track.md`) routes PM strategy results through the same robustness pipeline as Track A/B:

- PM behavioral strategies (FLB, overreaction, YES bias) use equity-style robustness gates (Sharpe, MaxDD, DSR, CPCV) applied to daily mark-to-market returns
- PM structural arb strategies (NegRisk, rebalancing) use the existing `PaperArbGate` framework (persistence, fill rate, capacity, days elapsed)
- Both output the same `BacktestMetrics` dataclass, enabling uniform robustness analysis

This design means the existing walk-forward HRP validation, the shuffled signal test, and the CPCV implementation apply to PM strategies without modification. The binary outcome structure affects the return distribution (fat tails, bounded payoff) but not the statistical tests themselves.

---

## Appendix: Key Reference Documents

| Document | Path | Purpose |
|----------|------|---------|
| Research landscape | `docs/research/polymarket/polymarket-research-landscape.md` | Platform overview, academic evidence, risk analysis |
| Trading infrastructure | `docs/research/polymarket/polymarket-trading-infrastructure.md` | CLOB architecture, microstructure, fee structure |
| Module design (API reference) | `docs/research/polymarket/polymarket-research-module-design.md` | API endpoints, SDK options, engineering gotchas |
| Integration ADR | `docs/architecture/polymarket-track.md` | Architecture decision: extend arb/ module |
| PM-02 research brief | `docs/research/pm-02-viral-claim-reconstruction.md` | Viral claim reconstruction exercise |
| Researcher agent | `.claude/agents/polymarket-researcher.md` | Agent configuration with edge families and principles |
| Gamma client | `src/llm_quant/arb/gamma_client.py` | Existing Polymarket data client |
| Arb scanner | `src/llm_quant/arb/scanner.py` | NegRisk and rebalance arb detection |
| Research tracks | `docs/governance/research-tracks.md` | Track A/B/C/D mandates and gates |
| Alpha hunting framework | `docs/governance/alpha-hunting-framework.md` | Kill chain, anti-overfitting discipline |

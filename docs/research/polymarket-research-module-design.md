# Building a systematic quantitative trading research module for Polymarket

Polymarket offers a surprisingly mature API surface for quantitative research — three REST APIs, WebSocket feeds, and on-chain subgraphs — but an Australian-based researcher faces a hard legal constraint: **Australia has been explicitly blocked since August 2025** under ACMA enforcement, making live trading impossible while read-only data access remains permissible. This report provides a complete technical reference across API infrastructure, market microstructure, academic alpha sources, engineering architecture, and regulatory boundaries, organized for a solo developer building research infrastructure from scratch.

---

## The three-layer API architecture and how to navigate it

Polymarket's data infrastructure spans three distinct REST APIs, a WebSocket layer, and on-chain subgraphs. Understanding which surface to query for what — and the identifier hierarchy that links them — is the single most important prerequisite.

### Gamma API — market discovery and metadata

**Base URL:** `https://gamma-api.polymarket.com`
**Authentication:** None (fully public, read-only)
**Rate limit:** ~4,000 requests/10 seconds

The Gamma API is the entry point for all market discovery. Markets are organized hierarchically: **Series → Events → Markets**. A Series groups related events (e.g., "2025 Fed Rate Decisions"), an Event is a specific question containing one or more Markets, and each Market represents a single tradeable binary outcome.

Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /events` | List events with filtering (`limit`, `offset`, `closed`, `active`) |
| `GET /events/{id}` | Single event by ID |
| `GET /markets` | List markets with filtering |
| `GET /markets/{condition_id}` | Single market by condition ID |
| `GET /search` | Full-text search across markets, events, profiles |
| `GET /tags` | Categories/tags for market classification |
| `GET /series` | Series groupings |

A market object contains the critical identifiers that bridge to the CLOB:

```json
{
  "id": "unique-market-id",
  "conditionId": "0x...",
  "questionId": "0x...",
  "clobTokenIds": "[\"71321045...\", \"71321046...\"]",
  "outcomes": "[\"Yes\", \"No\"]",
  "outcomePrices": "[\"0.65\", \"0.35\"]",
  "volume": "1500000",
  "minimumOrderSize": 5,
  "minimumTickSize": 0.01,
  "negRisk": false,
  "active": true,
  "closed": false
}
```

The **identifier hierarchy** is the #1 source of developer confusion. The `conditionId` (hex string) identifies the CTF condition on-chain. The `clobTokenIds` (large integer strings) identify each outcome token on the CLOB — **these are what you pass to CLOB API calls**, not the conditionId. The `questionId` is used in UMA oracle requests. Multi-outcome markets use the **neg-risk framework** where `negRisk: true` and each outcome gets its own binary YES/NO pair linked via `negRiskMarketId`.

### CLOB API — orderbook, pricing, and trading

**Base URL:** `https://clob.polymarket.com`
**Rate limit:** 9,000 requests/10 seconds (general); endpoint-specific limits apply
**Authentication:** Two-level system

Public endpoints (no auth):

| Endpoint | Rate limit | Description |
|----------|-----------|-------------|
| `GET /book?token_id={id}` | 50/10s (API) | Full order book snapshot |
| `POST /books` | 50/10s (API) | Batch order books |
| `GET /price?token_id={id}&side=BUY` | 100/10s | Best price |
| `GET /midpoint?token_id={id}` | — | Midpoint between best bid/ask |
| `GET /spread?token_id={id}` | — | Current spread |
| `GET /last-trade-price?token_id={id}` | — | Last execution price |
| `GET /tick-size?token_id={id}` | — | Current tick size for market |
| `GET /prices-history` | — | Historical price time series |

The `/prices-history` endpoint accepts `market` (CLOB token ID), `startTs`/`endTs` (Unix timestamps), `interval` (`1h`, `6h`, `1d`, `1w`, `1m`, `max`), and `fidelity` (resolution in minutes). Response format: `{"history": [{"t": 1697875200, "p": 0.65}, ...]}`. **Known issue**: resolved markets may only return 12+ hour granularity regardless of fidelity setting.

Order book response format:
```json
{
  "market": "0x1b6f76e5...",
  "asset_id": "TOKEN_ID",
  "bids": [{"price": "0.64", "size": "500"}, {"price": "0.63", "size": "1200"}],
  "asks": [{"price": "0.66", "size": "300"}, {"price": "0.67", "size": "800"}],
  "min_order_size": "5",
  "tick_size": "0.001"
}
```

**Authentication** uses a two-level system. **L1 auth** signs an EIP-712 `ClobAuth` struct with your Ethereum private key, producing headers `POLY_ADDRESS`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`, `POLY_NONCE`. This derives **L2 credentials** — an `apiKey`, `secret`, and `passphrase` — used for HMAC-SHA256 signing of subsequent trading requests. Trading endpoints require L2 auth with headers `POLY_API_KEY`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`, `POLY_PASSPHRASE`.

Authenticated trading endpoints:

| Endpoint | Rate limit | Description |
|----------|-----------|-------------|
| `POST /order` | 3,500/10s burst; 36,000/10min sustained | Place single order |
| `POST /orders` | 1,000/10s burst | Batch orders (up to 15) |
| `DELETE /order/{id}` | 3,000/10s burst | Cancel single order |
| `DELETE /cancel-all` | — | Cancel all orders |
| `POST /heartbeat` | — | Keep-alive signal (mandatory for market makers) |

**Order types**: GTC (Good Till Cancelled), GTD (Good Till Date, with expiration), FOK (Fill or Kill — all or nothing), FAK (Fill and Kill — partial fill then cancel remainder), and **Post Only** (rejects if it would cross the spread). All orders are technically limit orders. For market-like execution, submit a FOK/FAK at a price that crosses the spread.

**Tick sizes are dynamic**: `0.1`, `0.01`, `0.001`, or `0.0001`, changing when prices exceed **0.96** or fall below **0.04** (tighter ticks near extremes). Always fetch via `GET /tick-size` — do not hardcode. The minimum order size is typically **5 USDC**, available in the market object's `minimum_order_size` field.

### Data API — positions, trades, and analytics

**Base URL:** `https://data-api.polymarket.com`
**Rate limit:** ~1,000 requests/10 seconds

| Endpoint | Description |
|----------|-------------|
| `GET /positions?user={address}` | Current positions (up to 500 per page) |
| `GET /trades?user={address}` | Trade history with filtering |
| `GET /activity?user={address}` | On-chain activity (TRADE, SPLIT, MERGE, REDEEM) |
| `GET /holders?market={id}` | Top holders for a market |
| `GET /open-interest?market={id}` | Open interest |
| `GET /leaderboard` | Trader rankings |
| `GET /accounting-snapshot` | Bulk CSV export (ZIP) |

### WebSocket feeds

**Market channel:** `wss://ws-subscriptions-clob.polymarket.com/ws/market`

Subscribe by sending:
```json
{
  "type": "subscribe",
  "markets": ["TOKEN_ID_1", "TOKEN_ID_2"],
  "assets_id": ["TOKEN_ID_1"]
}
```

Events received include order book summaries, price changes, last trade updates, and market resolution events. A separate authenticated **user channel** at `/ws/user` streams order fills and position changes. **Critical known bug**: data stops flowing after ~20 minutes even though the connection remains healthy. Implement proactive reconnection every 10–15 minutes.

### Subgraph (The Graph)

Endpoint: `https://gateway.thegraph.com/api/{api-key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp`

The official `Polymarket/polymarket-subgraph` repository contains sub-subgraphs for orderbook events, activity, PnL, open interest, fees, and wallet data. Queryable entities include markets (conditions, volumes, trade counts), positions (balances by condition and outcome), trades, and user accounts. On-chain data covers settlement, token balances, and resolution events; off-chain data (order matching, book state, metadata) is not available through the subgraph. Free tier provides 100K queries/month.

### Resolution mechanics via UMA

Markets resolve through the **UMA Optimistic Oracle**. The flow: (1) `UmaCtfAdapter.initialize()` creates an on-chain price request with the question's ancillary data; (2) anyone proposes a resolution (Yes=1, No=0) by posting a **$750 USDC bond**; (3) a **2-hour liveness period** begins; (4) if undisputed, resolution finalizes and the proposer recovers their bond; (5) if disputed, the adapter auto-resets with a new request; (6) a second dispute escalates to UMA's **Data Verification Mechanism** where UMA token holders vote over **48–96 hours**. Resolved markets show `closed: true`, `active: false`, and `outcomePrices` snap to `["1","0"]` or `["0","1"]`. Capital remains locked throughout the resolution period.

---

## Market microstructure reveals a thin but functional order book

Polymarket's hybrid-decentralized CLOB matches orders off-chain for gas-free order management, then settles on-chain via Polygon smart contracts. This architecture produces microstructure characteristics that sit between a traditional exchange and a DeFi AMM.

### Spreads, depth, and the power law

Popular markets exhibit spreads as tight as **0.3 cents** — during the 2024 presidential election, six- and seven-figure trades moved prices by fractions of a cent. Typical spreads on popular markets run **2–5 cents** versus Kalshi's 3–8 cents. Niche markets with under $10K volume see spreads widen to **5%+ with extremely thin depth**.

Order book depth follows a severe power law. The October 2024 Trump presidential market showed **$33.5M total depth** but with $28M of that clustered above $0.99 in asks — only ~$4M in sell-side depth sat between the current price and $0.99. Polymarket requires approximately **3.5× more notional volume to move price** compared to Kalshi over a 60-second horizon. Kyle's λ (price impact coefficient) declined more than **10× over the market's lifetime**, from ~0.518 in early months to ~0.01 by October 2024, reflecting the dramatic liquidity growth as resolution approached.

### Fee structure as of early 2026

**Only takers pay fees; makers never pay and receive rebates.** The fee formula is `fee = C × p × feeRate × (p × (1-p))^exponent`, where fees peak at 50% probability and approach zero at extremes. Current rates by category:

| Category | Fee rate | Peak effective rate | Maker rebate |
|----------|----------|-------------------|-------------|
| Crypto | 0.072 | 1.80% | 20% |
| Sports | 0.03 | 0.75% | 25% |
| Finance | 0.04 | 1.00% | 50% |
| Politics | 0.04 | 1.00% | 25% |
| Geopolitics | **Free** | 0% | — |

This makes Polymarket significantly cheaper than Kalshi ($35 for a $1K trade at 50/50 on Kalshi vs ~$1–$16 on Polymarket depending on category). The fee schedule has evolved rapidly — from zero fees pre-2025, to crypto-only fees in January 2025, to category-wide fees by March 2025, with updates ongoing.

### Settlement and the proxy wallet architecture

All settlement occurs in **USDC.e on Polygon** (6 decimals, contract `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`). Polymarket **sponsors gas** through a relayer, so users never need POL/MATIC. When a user first trades, a **1-of-1 Gnosis Safe multisig** deploys as their proxy wallet via CREATE2. Positions are ERC-1155 tokens under the Gnosis Conditional Token Framework: splitting 1 USDC creates 1 YES + 1 NO token; merging reverses this; winning tokens redeem at $1.00 after resolution.

Key contracts on Polygon:

| Contract | Address |
|----------|---------|
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8b8982e` |
| NegRisk CTF Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |
| Conditional Tokens (ERC-1155) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` |
| USDC.e | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |

### Market maker mechanics and the heartbeat requirement

There are **no designated market makers**. Instead, Polymarket incentivizes liquidity through a **Liquidity Rewards Program** (daily USDC payouts scored every minute using a quadratic spread function, inspired by dYdX) and **Maker Rebates** (portion of taker fees redistributed daily). In July 2024, $433K in rewards were paid on $342M volume. Early LPs reported $200–$800/day at peak, declining as competition increased.

The **heartbeat mechanism** is critical for active order management: a heartbeat must be sent every **10 seconds** (5-second recommended cadence), or **all open orders are automatically cancelled**. Each heartbeat request includes the most recent `heartbeat_id`. This protects against disconnected bots leaving stale quotes.

Algorithmic activity is extensive. Bots exploit **latency arbitrage** between Polymarket and spot exchanges on 15-minute crypto markets. After major news events, a **30–60 second repricing window** exists before automated systems restore balance. Only **0.5% of Polymarket users** have earned more than $1K in profit.

---

## Academic literature points to specific exploitable patterns

The prediction market research literature has expanded significantly since 2024, with several Polymarket-specific empirical studies now available. The findings paint a nuanced picture: markets are approximately efficient but exhibit systematic biases that differ meaningfully from traditional betting markets.

### Prices are well-calibrated but show event-specific biases

Wolfers and Zitzewitz's foundational work (2004, 2006) established that prediction market prices approximate mean beliefs under broad conditions, with average absolute errors of ~1.5 percentage points for election markets versus 1.9 for Gallup polls. The most comprehensive Polymarket-specific study — Reichenbach and Walther (2025), analyzing **124 million trades and $48 billion in volume** — finds that market prices closely track realized probabilities and slightly outperform bookmaker odds.

Critically, this study finds **no general favorite-longshot bias** on Polymarket, unlike traditional sports betting where longshots are systematically overpriced. However, extreme longshots (very small probabilities) appear to perform consistently well, suggesting systematic underestimation of tail events. There is also a systematic **"Yes"/default bias**: traders overtrade the affirmative outcome by **6+ percentage points** on average. Top-decile traders by profit are measurably less susceptible to this bias.

### Mean reversion dominates momentum in prediction markets

Clinton and Huang (2025), analyzing $2.4 billion across four platforms during the 2024 election, found daily price changes show **negative autocorrelation** — consistent with short-term mean reversion. National presidential markets showed "short-term reversals rather than smooth convergence," with overreaction followed by correction. Sung et al. (2019), studying 6,058 betting exchange markets with 8.4 million price points, confirmed systematic overreaction to price movements and developed a correction methodology. **This is the opposite of traditional equity markets**, where short-run momentum (3–12 months) is the dominant anomaly.

### Cross-platform arbitrage is real and persistent

Clinton and Huang documented substantial cross-platform inefficiency: prices for identical contracts diverged across Polymarket, Kalshi, PredictIt, and Robinhood, with YES + NO sums deviating from $1.00 on **62 of 65 days** before the election on at least one platform. Arbitrage opportunities **peaked in the final two weeks** before resolution. Saguillo et al. (2025) quantified approximately **$40M in realized arbitrage profits** across 86 million trades — $39.6M from single-market rebalancing and only $95K from combinatorial strategies.

Within Polymarket, the most robust alpha sources documented in the literature are:

- **Yes/default bias**: Systematic overweighting of affirmative outcomes — a consistent edge for contrarian No positions
- **Late-stage inefficiency**: As sophisticated traders exit near resolution, less-skilled traders drive prices, creating exploitable patterns
- **Overreaction to price movements**: Contrarian strategies exploiting 72-hour overcorrection windows after major developments
- **Extreme longshot underpricing**: Events below 10% implied probability occurring ~14% of the time
- **Cross-platform arbitrage**: Persistent price divergences across venues, constrained by capital requirements and regulatory fragmentation

### Position sizing requires fractional Kelly

The bounded [0,1] payoff structure alters traditional position sizing. The Kelly formula for binary prediction markets: `f* = (bp - q) / b`, where p = estimated true probability, q = 1-p, b = (1 - market_price) / market_price. **Fractional Kelly at 0.25x–0.5x** is essential because overconfidence in probability estimation destroys capital exponentially in binary contracts. Correlated positions across related markets require portfolio-level Kelly extensions.

---

## Engineering architecture for a solo research developer

### Python libraries ranked by utility

**`polymarket-apis`** (PyPI, v0.5.7, actively maintained) is the strongest choice for research infrastructure. It wraps all four APIs (CLOB, Gamma, Data, WebSocket) with Pydantic-validated models, handles pagination, and includes WebSocket support that the official client lacks. Requires Python ≥3.12.

**`py-clob-client`** (official, v0.34.6) covers CLOB operations with L1/L2 authentication. It has 987 GitHub stars and 64 open issues. The critical limitation: **no WebSocket support and no async** — it is synchronous-only. Known bugs include tick size cache invalidation (fixed in v0.34.6) and HMAC signature edge cases.

**NautilusTrader** (`pip install nautilus_trader[polymarket]`) provides the most production-grade integration, with a Rust-based WebSocket client, instrument provider, data loader for historical replay, and execution reconciliation. It treats Polymarket markets as `BinaryOption` instruments and includes `PolymarketDataLoader` for backtesting.

### Historical data sources

| Source | Coverage | Format | Cost |
|--------|----------|--------|------|
| pmxt Data Archive (archive.pmxt.dev) | Hourly orderbook + trade snapshots | Parquet | Free |
| PolyBackTest (polybacktest.com) | Sub-second to 1-minute orderbook | API + bulk | Freemium |
| CLOB API `/prices-history` | Price time series per token | JSON | Free (rate-limited) |
| Polygon on-chain via The Graph | All settlement events | GraphQL | 100K queries/month free |
| Kaggle "Polymarket Prediction Markets" | Historical dataset | CSV | Free |

### Recommended storage architecture

For a solo developer building a research module, **DuckDB + Parquet** is the clear starting point: zero configuration, in-process, native Parquet/CSV support, excellent pandas integration, handles gigabytes easily. Polymarket itself uses **ClickHouse** for analytics (confirmed by their senior data engineer in a January 2026 interview), with PostgreSQL for transactional data and Goldsky indexing on-chain events into ClickHouse.

A practical schema for prediction market data:

```sql
-- Markets (slowly changing dimension)
CREATE TABLE markets (
    condition_id TEXT PRIMARY KEY,
    question TEXT, slug TEXT, category TEXT,
    end_date TIMESTAMP, outcomes JSONB,
    clob_token_ids JSONB, neg_risk BOOLEAN,
    active BOOLEAN, closed BOOLEAN
);

-- Order book snapshots (high-volume time series)
CREATE TABLE orderbook_snapshots (
    timestamp TIMESTAMP, token_id TEXT,
    side TEXT, price DECIMAL(10,4), size DECIMAL(18,6)
);

-- Trades (append-only, event-sourced)
CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY, timestamp TIMESTAMP,
    condition_id TEXT, token_id TEXT,
    side TEXT, price DECIMAL(10,4), size DECIMAL(18,6)
);
```

### Notable open-source projects worth studying

**`Polymarket/agents`** — official AI agent trading framework with LLM integration and ChromaDB vectorization. **`ent0n29/polybot`** — Java microservices architecture with ClickHouse + Redpanda event pipeline, Grafana monitoring, and Python research scripts; the most architecturally sophisticated community project. **`evan-kolberg/prediction-market-backtesting`** — NautilusTrader fork with custom Polymarket and Kalshi adapters specifically for backtesting. **`warproxxx/poly-maker`** — automated market making bot with band-based strategy.

### Critical engineering gotchas

**WebSocket data stops after ~20 minutes** (documented bug) — implement proactive reconnection every 10–15 minutes with sequence number tracking. **Token IDs from Gamma API `clobTokenIds`** must be used for CLOB calls — using `conditionId` is the #1 developer mistake. **EIP-712 timestamps** are rejected if clock drift exceeds 60 seconds. **FOK market orders** are limited to 2 decimal places for maker amount. **Tick sizes change dynamically** — cache invalidation is mandatory. All rate limits use **Cloudflare throttling** (requests queued/delayed before hard 429 rejection), so monitor `X-RateLimit-Remaining` headers.

---

## Australia is blocked, but read-only research remains permissible

The regulatory situation is unambiguous and constraining. **Australia has been explicitly geoblocked since August 13, 2025**, when ACMA formally added Polymarket to its national list of blocked gambling websites under the Interactive Gambling Act 2001, directing ISPs to block access. This followed an ACMA investigation triggered by Polymarket paying Australian influencers to promote federal election betting markets.

### The two Polymarkets

Since late 2025, two separate platforms exist. **Polymarket International** (polymarket.com) remains offshore, crypto-native (USDC on Polygon), with minimal KYC — but blocks 33 countries including Australia, the US, the UK, France, Germany, and others. **Polymarket US** (polymarket.us) launched in December 2025 as a CFTC-regulated Designated Contract Market through the $112M acquisition of QCEX, using Ed25519 authentication and requiring full KYC through Futures Commission Merchants. The international platform documented throughout this report is the primary developer surface with the richer API.

### What an Australian researcher can and cannot do

**Legally permissible from Australia**: accessing all read-only API endpoints (Gamma, Data, public CLOB data), storing and analyzing historical market data, building paper trading simulations, developing and backtesting algorithmic strategies, using open-source SDKs (MIT-licensed), querying on-chain Polygon data, and publishing academic research. Polymarket's own documentation confirms "data and information is viewable globally" — the geoblock check applies to order placement, not data consumption.

**Prohibited from Australia**: placing trades (orders from AU IPs are automatically rejected), using VPNs to bypass geoblocking (violates ToS Section 2.1.4 — consequences include account closure and fund freezing), hosting trading bots that place live orders, and promoting Polymarket to Australians. ASIC's binary options ban (extended until 2031) adds a second legal layer, and if the ATO classifies systematic trading activity as carrying on a business, profits would be assessable as ordinary income rather than tax-exempt gambling winnings.

The practical architecture for an Australian researcher is clear: use public APIs and third-party data providers (The Graph, pmxt archive, PolyBackTest) for data collection, build paper trading infrastructure, and develop/backtest strategies without live execution. If live trading becomes a goal, it would require physically operating from an unrestricted jurisdiction.

---

## Conclusion: a viable but constrained research platform

Polymarket's API infrastructure is surprisingly well-documented and programmatically accessible for a prediction market — **three REST APIs with generous rate limits, WebSocket feeds, on-chain data via subgraphs, and official SDKs in Python, TypeScript, and Rust**. The microstructure is thin compared to traditional financial markets but functional, with spreads as tight as 0.3 cents on popular markets and fee structures that meaningfully reward makers.

The academic literature identifies concrete alpha patterns — **Yes/default bias, mean reversion after overreaction, late-stage inefficiency as sophisticated traders exit, and persistent cross-platform arbitrage** — that differ qualitatively from traditional equity anomalies. The mean-reversion dominance over momentum is the single most important structural difference for strategy design.

For engineering, the **`polymarket-apis` package + DuckDB/Parquet + NautilusTrader** stack provides the most practical foundation for a solo developer. The pmxt Data Archive and PolyBackTest provide historical data without needing to build collection infrastructure from scratch. The key technical hazards are WebSocket disconnection after 20 minutes, dynamic tick sizes, and token ID confusion between Gamma and CLOB identifiers.

The binding constraint for an Australian researcher is regulatory, not technical. Read-only data collection and paper trading are legally defensible; live trading is not. The platform's rapid regulatory evolution — from CFTC settlement in 2022 to full DCM designation in 2025 — suggests this landscape may continue shifting, but Australian ACMA and ASIC restrictions show no signs of relaxing in the near term.
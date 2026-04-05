# Polymarket Data Surface Map

## Overview

This document enumerates every data source available for Polymarket research within
the llm-quant project. Each surface is described by its capabilities, authentication
requirements, identifier conventions, rate limits, failure modes, and integration
status. The goal is to provide a single reference for answering the question: "Where
do I get X data, and what are the constraints?"

The surfaces fall into three tiers:

1. **First-party APIs** -- operated by Polymarket (Gamma, CLOB, Data, US API, WebSocket)
2. **On-chain data** -- Polygon blockchain via The Graph subgraph or direct RPC
3. **Third-party archives** -- historical data providers (pmxt, PolyBackTest, Dune, Kaggle)

A cross-platform comparison surface (Kalshi) is documented separately. Kalshi is
already integrated via `src/llm_quant/arb/kalshi_client.py`.

---

## 1. Gamma API

**Base URL:** `https://gamma-api.polymarket.com`
**Purpose:** Market discovery, metadata, event listing, series grouping, search
**Authentication:** None (fully public, read-only)
**Rate limit:** ~4,000 requests / 10 seconds
**Geo-block:** US IPs blocked (Azure Front Door returns 404 with HTML body)

### Endpoints

| Endpoint | Method | Description | Key Parameters |
|----------|--------|-------------|----------------|
| `/markets` | GET | Paginated market listing | `limit`, `offset`, `active`, `closed` |
| `/markets/{condition_id}` | GET | Single market by condition ID | -- |
| `/events` | GET | Paginated event listing | `limit`, `offset`, `active`, `closed` |
| `/events/{id}` | GET | Single event by ID | -- |
| `/search` | GET | Full-text search across markets, events, profiles | `query` |
| `/tags` | GET | Category/tag listing for market classification | -- |
| `/series` | GET | Series groupings (e.g., "2025 Fed Rate Decisions") | -- |

### Identifier Hierarchy

The Gamma API exposes three critical identifiers per market:

| Identifier | Format | Usage |
|------------|--------|-------|
| `id` | Opaque string (numeric or UUID) | Gamma-internal market ID |
| `conditionId` | Hex string (`0x...`) | CTF condition on Polygon; links to on-chain settlement |
| `clobTokenIds` | JSON-encoded array of large integer strings | **CLOB API calls require these**, not `conditionId` |
| `questionId` | Hex string (`0x...`) | UMA oracle price request identifier |
| `slug` | URL-safe string | Human-readable market identifier |
| `negRiskMarketId` | Hex string | Parent market for NegRisk multi-outcome markets |

**Developer hazard:** The `conditionId` vs `clobTokenIds` confusion is the number one
source of integration errors. All CLOB API pricing and orderbook calls use `clobTokenIds`
(the `token_id` parameter). Passing `conditionId` to the CLOB API returns empty or
incorrect results.

### Response Shape (Market Object)

```json
{
  "id": "123456",
  "conditionId": "0xabc123...",
  "questionId": "0xdef456...",
  "clobTokenIds": "[\"71321045...\", \"71321046...\"]",
  "outcomes": "[\"Yes\", \"No\"]",
  "outcomePrices": "[\"0.65\", \"0.35\"]",
  "volume": "1500000",
  "volumeNum24hr": "85000",
  "openInterest": "230000",
  "minimumOrderSize": 5,
  "minimumTickSize": 0.01,
  "negRisk": false,
  "active": true,
  "closed": false,
  "endDate": "2026-06-15T00:00:00Z",
  "slug": "will-fed-cut-rates-june-2026",
  "category": "finance",
  "tokens": [
    {"token_id": "71321045", "outcome": "Yes", "price": 0.65, "winner": false},
    {"token_id": "71321046", "outcome": "No", "price": 0.35, "winner": false}
  ]
}
```

**Note:** `outcomes` and `outcomePrices` are JSON-encoded strings, not native arrays.
The `tokens` array (when present) provides more accurate pricing and includes the
`token_id` needed for CLOB calls.

### Pagination

Offset-based: `offset=0&limit=100`. Default page size is 100. The response is a
bare JSON array (no wrapper object with cursor or total count). An empty array signals
the final page.

### Caching Strategy

- Market metadata (category, slug, endDate): 15-minute cache is appropriate; these
  fields change rarely.
- Prices (outcomePrices, tokens.price): Near real-time; do not cache if price
  accuracy matters. For live scanning, re-fetch on each pass.
- Volume/OI fields: 5-minute cache is acceptable.

### Failure Modes

1. **Geo-block (US IPs):** HTTP 404 with `text/html` content type from Azure Front Door.
   The response body is an Azure error page, not JSON. Detected by checking
   `Content-Type` header and `x-azure-ref` header.
2. **Rate limiting:** Returns HTTP 429 with `Retry-After` header. Cloudflare-based
   throttling may queue requests before hard rejection.
3. **SSL certificate errors (Windows):** The bundled certifi store may fail on Windows.
   The existing `GammaClient` defaults to `ssl_verify=False` on Windows.
4. **Schema drift:** Field names and response shapes have changed historically (e.g.,
   `volume24hr` vs `volumeNum24hr`). Defensive parsing is required.

### Integration Status

**WORKING.** Implemented in `src/llm_quant/arb/gamma_client.py`. Fixed in commit
21f820a. Automatic failover to US API on geo-block.

**Known gaps in current implementation:**
- `clobTokenIds` are parsed from the response but NOT stored in the `Market` dataclass.
  This blocks CLOB API `/prices-history` calls.
- `questionId` is not stored. This blocks UMA resolution tracking.
- No `fetch_resolved_markets()` method. Only active markets are fetched.
- No `/search` or `/tags` endpoint usage.
- No `/series` endpoint usage.

---

## 2. CLOB API

**Base URL:** `https://clob.polymarket.com`
**Purpose:** Orderbook depth, pricing, tick sizes, price history, trade execution
**Authentication:** Public for read-only endpoints; L1/L2 key pair for trading
**Rate limit:** 9,000 requests / 10 seconds (general); endpoint-specific limits below
**Geo-block:** US IPs blocked

### Public Endpoints (No Authentication)

| Endpoint | Method | Rate Limit (per 10s) | Description |
|----------|--------|---------------------|-------------|
| `/book` | GET | 50 | Full L2 orderbook snapshot for a single token |
| `/books` | POST | 50 | Batch orderbook snapshots (multiple tokens) |
| `/price` | GET | 100 | Best bid or ask for a token |
| `/midpoint` | GET | -- | Midpoint between best bid and ask |
| `/spread` | GET | -- | Current spread |
| `/last-trade-price` | GET | -- | Last execution price |
| `/tick-size` | GET | -- | Current tick size for a token |
| `/prices-history` | GET | -- | Historical price time series |

All read-only endpoints use `token_id` as the primary parameter. This is the CLOB
token ID from `clobTokenIds` in the Gamma API response -- NOT the `conditionId`.

### Price History Endpoint

`GET /prices-history` is the primary source for historical price data per market.

| Parameter | Type | Description |
|-----------|------|-------------|
| `market` | string | CLOB token ID (required) |
| `startTs` | integer | Start Unix timestamp (seconds) |
| `endTs` | integer | End Unix timestamp (seconds) |
| `interval` | string | `1m`, `1h`, `6h`, `1d`, `1w`, `1m`, `max` |
| `fidelity` | integer | Resolution in minutes (e.g., 60 for hourly) |

**Response:**
```json
{
  "history": [
    {"t": 1697875200, "p": 0.65},
    {"t": 1697878800, "p": 0.66}
  ]
}
```

**Known limitation:** Resolved markets may only return 12+ hour granularity regardless
of fidelity setting. This degrades backtesting precision for resolved markets.

### Orderbook Response Shape

```json
{
  "market": "0x1b6f76e5...",
  "asset_id": "71321045",
  "hash": "0xabc...",
  "bids": [
    {"price": "0.64", "size": "500"},
    {"price": "0.63", "size": "1200"}
  ],
  "asks": [
    {"price": "0.66", "size": "300"},
    {"price": "0.67", "size": "800"}
  ],
  "min_order_size": "5",
  "tick_size": "0.001"
}
```

All prices and sizes are strings, not floats. Parsing must cast explicitly.

### Authenticated Endpoints (Trading -- L2 Key Required)

| Endpoint | Method | Rate Limit (per 10s) | Description |
|----------|--------|---------------------|-------------|
| `/order` | POST | 3,500 burst; 36,000/10min sustained | Place single order |
| `/orders` | POST | 1,000 burst | Batch orders (up to 15) |
| `/order/{id}` | DELETE | 3,000 burst | Cancel single order |
| `/cancel-all` | DELETE | -- | Cancel all open orders |
| `/heartbeat` | POST | -- | Keep-alive (mandatory for market makers) |

**Authentication model:** Two-level system.
- **L1:** EIP-712 `ClobAuth` struct signed with Ethereum private key. Produces
  headers: `POLY_ADDRESS`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`, `POLY_NONCE`.
- **L2:** Derived API credentials (`apiKey`, `secret`, `passphrase`) used for
  HMAC-SHA256 signing. Headers: `POLY_API_KEY`, `POLY_SIGNATURE`, `POLY_TIMESTAMP`,
  `POLY_PASSPHRASE`.

Trading authentication is OUT OF SCOPE for the llm-quant research module (AU
regulatory constraint).

### Tick Size Dynamics

Tick sizes are not static. They change when prices exceed 0.96 or fall below 0.04:

| Price Range | Tick Size |
|-------------|-----------|
| 0.04 -- 0.96 | 0.01 (most common) |
| < 0.04 or > 0.96 | 0.001 or 0.0001 |

Always fetch via `GET /tick-size` per token. Do not hardcode.

### Failure Modes

1. **Geo-block (US IPs):** Same as Gamma API -- Azure Front Door 404.
2. **Rate limiting:** Per-endpoint limits. `/book` is the tightest (50/10s per API).
3. **Token ID confusion:** Using `conditionId` instead of `clobTokenIds` returns empty
   results or errors.
4. **Empty book:** Low-liquidity markets may return empty `bids`/`asks` arrays.
5. **Stale data:** No documented SLA on data freshness.

### Integration Status

**NOT INTEGRATED.** The existing `gamma_client.py` connects to the Gamma API, not
the CLOB API, despite the `CLOB_BASE` constant being defined. No code in the project
currently calls CLOB API endpoints.

**Required for:**
- Orderbook depth analysis
- `/prices-history` for backtesting
- `/tick-size` for accurate fee computation
- Real-time price monitoring

---

## 3. Data API

**Base URL:** `https://data-api.polymarket.com`
**Purpose:** Position tracking, trade history, user activity, leaderboard, open interest
**Authentication:** Required for user-specific data; public for aggregate data
**Rate limit:** ~1,000 requests / 10 seconds

### Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/positions` | GET | Current positions for a wallet address | No (public by address) |
| `/trades` | GET | Trade history for a wallet address | No (public by address) |
| `/activity` | GET | On-chain activity (TRADE, SPLIT, MERGE, REDEEM) | No |
| `/holders` | GET | Top holders for a market | No |
| `/open-interest` | GET | Open interest for a market | No |
| `/leaderboard` | GET | Trader rankings | No |
| `/accounting-snapshot` | GET | Bulk CSV export (ZIP) | Unknown |

### Key Parameters

- `user` -- Ethereum wallet address (for positions, trades, activity)
- `market` -- Market identifier (for holders, open-interest)
- Pagination: up to 500 results per page for positions

### Use Cases for Research

The Data API is uniquely valuable for:

1. **Whale tracking:** Monitor top holder positions in specific markets via `/holders`.
2. **Smart money signals:** Track positions of historically profitable wallets via
   `/positions?user={address}`, cross-referenced with `/leaderboard`.
3. **Flow analysis:** `/trades` provides trade-by-trade history for any wallet,
   enabling order flow analysis.
4. **Open interest tracking:** `/open-interest` provides a proxy for conviction --
   rising OI with price confirms trend, rising OI against price signals distribution.

### Failure Modes

1. **Rate limiting:** 1,000/10s is the tightest of all Polymarket APIs. Bulk data
   collection requires careful throttling.
2. **Privacy concerns:** All wallet addresses and positions are public. This is by
   design (on-chain transparency) but may change under regulatory pressure.
3. **Data completeness:** Some fields may be empty for inactive or historical wallets.

### Integration Status

**NOT INTEGRATED.** No code in the project references `data-api.polymarket.com`.

**Priority:** Medium. The Data API enables smart-money and flow analysis strategies
that are orthogonal to the structural arb currently implemented. Lower priority than
CLOB `/prices-history` for backtest infrastructure.

---

## 4. Polymarket US API

**Base URL:** `https://api.polymarket.us`
**Purpose:** CFTC-regulated US access, market listing, alternative data source
**Authentication:** Ed25519 key pair for full access; limited public access without key
**Geo-block:** None (US-accessible by design)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/markets` | GET | Market listing |
| `/v1/markets/{id}` | GET | Single market by ID |

### Limitations Without API Key

- Maximum ~20 markets returned per request
- No pagination, filtering, or sorting controls documented
- No orderbook, pricing, or trade data -- metadata only

### Limitations With API Key

- Full market access with pagination
- Authentication via Ed25519 key (not Ethereum wallet-based)
- As of April 2026, still invite-only / waitlist for full API access

### Fee Structure (Distinct from International)

The US platform uses a simpler fee model:
- 30 bps taker fee on Total Contract Premium
- 20 bps maker rebate
- No category-based differentiation

### Response Format Differences

The US API response shape differs from Gamma API in several fields:

| Field | Gamma API | US API |
|-------|-----------|--------|
| Market identifier | `id`, `conditionId` | `id` |
| Price data | `outcomePrices`, `tokens` | `outcomePrices`, `marketSides` |
| Category | Inferred from text | Explicit `category` field |
| Volume | `volumeNum24hr` | Not consistently available |

The existing `parse_market()` in `gamma_client.py` handles both formats with
defensive field access.

### Integration Status

**WORKING** as fallback in `gamma_client.py`. When the Gamma API returns a geo-block
error (HTTP 404 with HTML), the client automatically falls back to
`api.polymarket.us/v1/markets`.

**Known gap:** Returns only ~20 markets without an API key. For comprehensive
market coverage from US IPs, an Ed25519 API key is required.

---

## 5. WebSocket Feeds

### Market Channel

**URL:** `wss://ws-subscriptions-clob.polymarket.com/ws/market`
**Authentication:** None (public)
**Purpose:** Real-time orderbook updates, price changes, trade notifications

### Subscribe Message

```json
{
  "type": "subscribe",
  "markets": ["TOKEN_ID_1", "TOKEN_ID_2"],
  "assets_id": ["TOKEN_ID_1"]
}
```

### Event Types

| Event | Description |
|-------|-------------|
| `book` | Order book summary update |
| `price_change` | Best bid/ask price change |
| `last_trade_price` | Last trade execution |
| `tick_size_change` | Dynamic tick size adjustment |

### User Channel (Authenticated)

**URL:** `wss://ws-subscriptions-clob.polymarket.com/ws/user`
**Authentication:** L2 API key required
**Purpose:** Order fills, position changes, balance updates

### RTDS (Real-Time Data Socket)

**URL:** `wss://ws-live-data.polymarket.com`
**Purpose:** Low-latency feed optimized for market makers
**Details:** Underdocumented. Referenced in trading infrastructure research. May
require special access.

### Critical Known Bug

**Data stops flowing after approximately 20 minutes** even though the WebSocket
connection remains healthy (no close frame, heartbeat succeeds). This is a documented
bug across multiple community reports. Mitigation: implement proactive reconnection
every 10--15 minutes with sequence number tracking to detect gaps.

### Failure Modes

1. **Silent data cessation:** The 20-minute bug described above.
2. **Subscription limits:** Community reports indicate instability when subscribing
   to 250+ tokens simultaneously.
3. **Geo-block:** Likely blocked from US IPs (same infrastructure as CLOB API).
   Not independently verified.
4. **No replay:** WebSocket is real-time only. Historical data requires the REST
   `/prices-history` endpoint or third-party archives.

### Integration Status

**NOT INTEGRATED.** No WebSocket code exists in the project. The ADR
(`docs/architecture/polymarket-track.md`) defers WebSocket integration as lower
priority than REST-based historical data.

---

## 6. The Graph Subgraph (On-Chain Data)

**Endpoint:** `https://gateway.thegraph.com/api/{api-key}/subgraphs/id/Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp`
**Purpose:** Historical on-chain data -- trades, positions, settlements, token balances
**Authentication:** The Graph API key (free tier: 100K queries/month)
**Refresh cadence:** Block-level (~2 second Polygon blocks)

### Queryable Entities

| Entity | Description |
|--------|-------------|
| `Market` / `Condition` | Market metadata, condition IDs, volumes, trade counts |
| `Trade` | Individual on-chain trade executions |
| `Position` | Token balances by condition, outcome, and user |
| `User` / `Account` | Wallet-level aggregates |
| `FixedProductMarketMaker` | Legacy AMM data (pre-CLOB era) |

### What Is Available vs. What Is Not

**Available on-chain:**
- Settlement events (token mints, merges, redemptions)
- Token transfers and balances
- Resolution outcomes
- Trade execution (after matching)
- Historical volume and open interest

**NOT available on-chain (off-chain only):**
- Live orderbook state (matching is off-chain)
- Order placement and cancellation events
- Market metadata (question text, categories, end dates)
- Price history at sub-block resolution

### GraphQL Query Example

```graphql
{
  conditions(
    first: 100
    where: { resolved: true }
    orderBy: volume
    orderDirection: desc
  ) {
    id
    questionId
    outcomeSlotCount
    resolved
    payouts
    volume
    trades {
      id
      amount
      price
      timestamp
      trader { id }
    }
  }
}
```

### Sub-Subgraphs

The official `Polymarket/polymarket-subgraph` repository contains specialized
sub-subgraphs:

- Orderbook events
- Activity tracking
- PnL computation
- Open interest
- Fee collection
- Wallet analytics

### Failure Modes

1. **Query complexity limits:** The Graph enforces query cost limits. Deeply nested
   queries (e.g., fetching all trades for all conditions) will be rejected.
2. **Free tier exhaustion:** 100K queries/month can be consumed quickly during bulk
   data collection. Each paginated query counts separately.
3. **Indexing lag:** The subgraph may lag behind the chain tip during high-volume
   periods. Lag is typically under 30 seconds but can exceed minutes during Polygon
   congestion.
4. **Deprecation risk:** The Graph ecosystem has undergone multiple migration cycles.
   The specific subgraph ID may change.

### Integration Status

**NOT INTEGRATED.** No code in the project queries The Graph.

**Use case:** Best for resolution outcome data (required for backtest ground truth),
historical trade flow analysis, and on-chain settlement verification.

---

## 7. Dune Analytics

**URL:** `https://dune.com`
**Purpose:** SQL-queryable on-chain transaction data, community dashboards
**Authentication:** Dune API key (free tier available)
**Refresh cadence:** Weekly for most dashboards; real-time for custom queries via API

### Key Community Dashboards

| Dashboard | Creator | Coverage |
|-----------|---------|----------|
| `dune.com/rchen8/polymarket` | rchen8 | Market-level volume, OI, trader counts |
| `dune.com/alexmccullough/how-accurate-is-polymarket` | alexmccullough | Calibration analysis |

### API Access

The Dune API (v2) allows programmatic execution of SQL queries against indexed
Polygon blockchain data:

```
POST https://api.dune.com/api/v1/query/{query_id}/execute
GET  https://api.dune.com/api/v1/execution/{execution_id}/results
```

Free tier: 2,500 credits/month, 10 queries/minute.

### What Dune Provides That Other Sources Do Not

- Arbitrary SQL aggregations across all on-chain Polymarket data
- Cross-protocol analysis (e.g., correlate Polymarket activity with Uniswap volume)
- Community-maintained analytical queries

### Failure Modes

1. **Query execution time:** Complex queries may timeout (10 minute limit on free tier).
2. **Data freshness:** Pre-built dashboards update weekly. Custom API queries execute
   against fresher data but may hit indexing lag.
3. **Schema changes:** Dune's decoded tables depend on ABI uploads. Contract upgrades
   may temporarily break queries.

### Integration Status

**NOT INTEGRATED.** No Dune API code exists in the project.

**Priority:** Low for initial research. Useful for one-off analytical queries but
not suitable as a primary data pipeline due to latency and rate limits.

---

## 8. Third-Party Historical Data Archives

### 8.1 pmxt Data Archive

**URL:** `https://archive.pmxt.dev`
**Format:** Parquet files
**Cost:** Free
**Coverage:** Hourly orderbook + trade snapshots
**Size:** 150--460 MB per hour of data
**Date range:** Unknown -- documentation does not specify start date

**Use case:** The most efficient source for bulk historical orderbook analysis.
Download Parquet files, load with Polars/DuckDB, analyze locally.

**Limitation:** No SLA, no guarantee of continued availability. Must download and
store locally for reproducibility.

### 8.2 PolyBackTest

**URL:** `https://polybacktest.com`
**Format:** API + bulk download
**Cost:** Freemium (free tier with limits, paid for full access)
**Coverage:** Sub-second to 1-minute L2 orderbook snapshots from market creation
**Source:** PolymarketData.co

**Use case:** Highest-fidelity historical data for microstructure research and limit
order simulation. The only source that provides sub-minute orderbook snapshots.

### 8.3 Telonex

**URL:** `https://telonex.io`
**Format:** Parquet
**Cost:** Paid
**Coverage:** Tick-level trades, orderbook data, on-chain fills across 500K+ markets
  and 20B+ data points

**Use case:** The most comprehensive commercial historical data source. Necessary for
academic-grade microstructure research.

### 8.4 Kaggle Datasets

**Dataset:** "Polymarket Prediction Markets"
**Format:** CSV
**Cost:** Free
**Coverage:** Historical market metadata and pricing

**Use case:** Quick-start for exploratory analysis. Quality and freshness unverified.

### Integration Status

**NONE INTEGRATED.** The ADR proposes a `PmxtArchiveLoader` class in `pm_history.py`
for loading pmxt Parquet files. Not yet implemented.

---

## 9. Kalshi API (Cross-Platform Reference)

**Base URL:** `https://api.elections.kalshi.com/trade-api/v2`
**Purpose:** US-regulated prediction market; cross-platform arb reference
**Authentication:** API key (for trading); public for market data reads
**Geo-block:** US-only (inverse of Polymarket International)
**Fee:** 3% on winning positions (vs. Polymarket's category-dependent fees)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events` | GET | Event listing with cursor pagination |
| `/markets` | GET | Market listing within events |

### Key Differences from Polymarket

| Dimension | Polymarket | Kalshi |
|-----------|------------|--------|
| Regulation | CFTC (US only via polymarket.us) | CFTC (full DCM since 2020) |
| Settlement | USDC on Polygon | USD in custodial account |
| Fee | Category-dependent, nonlinear | Flat 3% on wins |
| Market creation | Polymarket Markets Team | Kalshi team |
| Oracle | UMA Optimistic Oracle | Kalshi-operated |
| Pagination | Offset-based | Cursor-based |
| Identifiers | `conditionId`, `clobTokenIds` | `ticker`, `event_ticker` |
| Mutually exclusive | NegRisk framework (`negRisk: true`) | `mutually_exclusive` flag |

### Integration Status

**FULLY INTEGRATED.** Implemented in `src/llm_quant/arb/kalshi_client.py` with
bulk and per-event market fetching, condition parsing, and NegRisk event detection.

---

## 10. External Reference Feeds

These are non-Polymarket data sources used to calibrate or cross-reference prediction
market signals.

### 10.1 Binance / Crypto Exchanges (via ccxt)

**Purpose:** Real-time BTC/ETH prices for crypto-linked PM markets
**Library:** `ccxt` (unified API across 100+ exchanges)
**Integration:** Available via `src/llm_quant/arb/funding_rates.py` (CCXT already
used for funding rate collection from Binance, OKX, Bybit)

### 10.2 CME FedWatch

**Purpose:** Fed funds futures probabilities for rate-linked PM markets
**Access:** Public web data; no official API for free tier
**Use case:** Cross-reference PM "Fed rate cut" market prices against CME implied
probabilities

### 10.3 Metaculus API

**Purpose:** 12+ years of prediction data across 23,400+ questions with community
prediction time series and resolution outcomes
**Use case:** Calibration baseline for LLM forecasting strategies; base rate estimation
**Library:** `Metaculus/forecasting-tools` (official Python package)

### 10.4 FRED API

**Purpose:** 800K+ economic time series for macro-event market calibration
**Library:** `fredapi` Python wrapper
**Use case:** CPI, GDP, employment data for finance/economics PM market research

### 10.5 GDELT

**Purpose:** Global event monitoring for news-driven PM strategies
**Access:** Free, ~15-minute delay, Python client available
**Use case:** Event detection for overreaction mean-reversion strategies

### Integration Status

Only Binance/CCXT is integrated (via funding rates module). All others are unintegrated.

---

## 11. Data Surface Matrix

Summary of what each source provides:

| Source | Markets | Prices | Orderbook | History | Trades | Positions | Resolution | Auth | Geo |
|--------|---------|--------|-----------|---------|--------|-----------|------------|------|-----|
| Gamma API | Yes | Yes (snapshot) | No | No | No | No | No | None | US blocked |
| CLOB API | No | Yes (real-time) | Yes (L2) | Yes (price series) | No | No | No | R/O: None; Trade: L2 | US blocked |
| Data API | No | No | No | No | Yes (per wallet) | Yes | No | Partial | Unknown |
| US API | Yes | Yes (snapshot) | No | No | No | No | No | Ed25519 (optional) | US only |
| WebSocket | No | Yes (streaming) | Yes (streaming) | No | Yes (streaming) | No | No | R/O: None; User: L2 | Likely US blocked |
| The Graph | Yes | No | No | Yes (block-level) | Yes (on-chain) | Yes | Yes | API key | None |
| Dune | Yes (via SQL) | No | No | Yes (via SQL) | Yes (via SQL) | Yes (via SQL) | Yes (via SQL) | API key | None |
| pmxt Archive | No | Yes | Yes | Yes (hourly) | Yes | No | No | None | None |
| PolyBackTest | No | Yes | Yes (L2, sub-min) | Yes | No | No | No | Freemium | None |
| Kalshi | Yes | Yes | Yes | No | Yes | No | Yes | API key | US only |

### Canonical Data Paths

For common research tasks, the recommended data source:

| Research Task | Primary Source | Fallback |
|---------------|---------------|----------|
| Market discovery / metadata | Gamma API | US API |
| Real-time best price | CLOB `/price` | Gamma API `outcomePrices` |
| Orderbook depth | CLOB `/book` | pmxt Archive (historical) |
| Price history (per market) | CLOB `/prices-history` | pmxt Archive |
| Bulk historical prices | pmxt Archive | PolyBackTest |
| Resolution outcomes | The Graph subgraph | Gamma API (`closed` + `outcomePrices`) |
| Trade flow analysis | Data API `/trades` | The Graph |
| Whale/smart money tracking | Data API `/positions` + `/leaderboard` | -- |
| Cross-platform arb signals | Kalshi API | -- |
| Macro calibration | FRED, CME FedWatch | Metaculus |

---

## 12. Identifier Cross-Reference

Navigating between data surfaces requires mapping between identifier systems:

```
Gamma API
  id (market_id) ──────────────────────────┐
  conditionId ─────────────────────────────┼── The Graph (conditions.id)
  questionId ──────────────────────────────┼── UMA Oracle (price request)
  clobTokenIds[0] (YES token) ─────────────┼── CLOB API (token_id parameter)
  clobTokenIds[1] (NO token) ──────────────┤   WebSocket (subscribe.markets)
  slug ────────────────────────────────────┤   Data API (market parameter)
  negRiskMarketId ─────────────────────────┘   pmxt Archive (market identifier)

Kalshi
  ticker ──────────────────── Market identifier
  event_ticker ────────────── Event grouping
  series_ticker ───────────── Series grouping

Cross-platform mapping:
  No canonical mapping exists between Polymarket conditionId and Kalshi ticker.
  Matching requires fuzzy text matching on question/title text or manual curation.
```

### Join Path: Gamma -> CLOB -> Data API -> The Graph

1. Gamma API `/markets` returns `conditionId` and `clobTokenIds`.
2. CLOB API uses `clobTokenIds` for `/book`, `/price`, `/prices-history`.
3. Data API uses wallet `address` for `/positions`, `/trades`.
4. The Graph uses `conditionId` as `conditions.id` for on-chain data.

The `conditionId` is the canonical cross-surface identifier for Polymarket data.
The `clobTokenIds` are required only for CLOB API and WebSocket calls. Both must
be stored in any normalized schema.

---

## 13. Rate Limit Summary

| API | General Limit | Tightest Endpoint | Notes |
|-----|---------------|-------------------|-------|
| Gamma API | 4,000 / 10s | -- | Generous for research |
| CLOB API | 9,000 / 10s | `/book`: 50/10s | Book queries are the bottleneck |
| Data API | 1,000 / 10s | -- | Tightest overall API |
| Kalshi API | ~5 req/s observed | -- | No official documentation |
| The Graph | 100K queries/month (free) | -- | Monthly, not per-second |
| Dune API | 10 queries/min (free) | -- | Credit-based |

### Practical Throughput for Bulk Operations

| Operation | Estimated Time |
|-----------|---------------|
| Fetch all active PM markets (5,000) | ~13 seconds (50 pages x 0.25s rate limit) |
| Fetch orderbook for 100 markets | ~20 seconds (50/10s limit on `/book`) |
| Fetch price history for 1,000 markets | ~17 minutes (1 req/s conservative) |
| Fetch all Kalshi open events + markets | ~30 seconds (bulk mode) |

---

*Document generated 2026-04-06. Data surface availability and API specifications
are subject to change. All rate limits and endpoints should be verified against
current Polymarket documentation before implementation.*

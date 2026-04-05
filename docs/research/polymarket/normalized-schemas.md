# Polymarket Normalized Schemas

## Overview

This document defines normalized data schemas for cross-source prediction market
analysis in the llm-quant project. Each schema is designed for DuckDB storage with
Pydantic model equivalents in Python. The schemas normalize data from multiple
Polymarket APIs (Gamma, CLOB, Data, US API, The Graph) and Kalshi into a unified
representation suitable for research, backtesting, and paper trading.

### Design Principles

1. **Source-agnostic.** Each schema represents a logical concept (Event, Market,
   Trade) independent of which API provided the data. Source tracking is a metadata
   field, not a structural concern.
2. **Canonical IDs.** Every entity has a single canonical identifier used for joins
   across schemas. Where APIs use different identifier systems, the schema defines
   the mapping.
3. **Temporal versioning.** Price and orderbook data are time-stamped snapshots.
   Market metadata tracks both creation time and last-observed time.
4. **Nullable for optional.** Fields available from some sources but not others are
   nullable, not omitted. This makes cross-source joins possible without schema
   changes.
5. **Compatible with existing `arb/schema.py`.** New schemas extend the existing
   DuckDB DDL in `src/llm_quant/arb/schema.py`. Existing tables (`pm_markets`,
   `pm_conditions`, etc.) remain as-is for backward compatibility. The normalized
   schemas described here are a superset.

### Relationship to Existing Schemas

The existing `arb/schema.py` defines 8 tables optimized for arb scanning:

| Existing Table | Normalized Equivalent | Relationship |
|---------------|----------------------|--------------|
| `pm_markets` | `Event` + `Market` | Split: events and markets are distinct entities |
| `pm_conditions` | `OutcomeToken` | 1:1 mapping with added CLOB token identifiers |
| `pm_negrisk_groups` | `Market` (where `is_negrisk = true`) | Absorbed into Market metadata |
| `pm_arb_opportunities` | No direct equivalent | Arb detection is downstream of normalized data |
| `pm_combinatorial_pairs` | No direct equivalent | Arb detection is downstream |
| `pm_scan_log` | No direct equivalent | Operational logging, not research data |
| `pm_executions` | `TradePrint` (for paper trades) | Paper trade records |

The normalized schemas are designed to coexist with the existing tables. Migration
from existing tables to normalized schemas is optional -- both can be populated in
parallel during a transition period.

---

## Schema 1: Event

An Event is a top-level container representing a real-world occurrence that one or
more markets are created to track. Examples: "2026 US Midterm Elections", "June 2026
FOMC Meeting", "NBA Finals 2026".

### Pydantic Model

```python
from datetime import datetime
from pydantic import BaseModel, Field


class Event(BaseModel):
    """Top-level event container.

    An event groups related markets. In Polymarket, events map to the
    /events endpoint. In Kalshi, events map to event_ticker groupings.
    An event may contain 1 market (simple binary) or 50+ markets
    (multi-outcome NegRisk, e.g., presidential election with 17 candidates).
    """

    event_id: str = Field(
        description="Canonical event identifier. "
        "Polymarket: Gamma API event ID. "
        "Kalshi: event_ticker."
    )
    source: str = Field(
        description="Data source platform. "
        "Values: 'polymarket', 'kalshi'."
    )
    title: str = Field(
        description="Human-readable event title."
    )
    slug: str | None = Field(
        default=None,
        description="URL-safe slug. Polymarket only."
    )
    category: str = Field(
        default="other",
        description="Event category. "
        "Polymarket: 'politics', 'sports', 'crypto', 'finance', "
        "'geopolitics', 'culture', 'other'. "
        "Kalshi: uses its own category taxonomy."
    )
    series_id: str | None = Field(
        default=None,
        description="Series grouping ID. "
        "Polymarket: from /series endpoint. "
        "Kalshi: series_ticker."
    )
    is_negrisk: bool = Field(
        default=False,
        description="True if the event uses NegRisk framework "
        "(mutually exclusive outcomes). "
        "Polymarket: negRisk field. "
        "Kalshi: mutually_exclusive field."
    )
    market_count: int = Field(
        default=1,
        description="Number of markets within this event."
    )
    start_date: datetime | None = Field(
        default=None,
        description="Event start date, if applicable."
    )
    end_date: datetime | None = Field(
        default=None,
        description="Event end/resolution date. "
        "Polymarket: endDate field. "
        "Kalshi: close_time or expiration_time."
    )
    resolution_source: str | None = Field(
        default=None,
        description="Oracle or resolution mechanism. "
        "Values: 'uma_optimistic', 'chainlink', 'markets_team', "
        "'kalshi_internal'."
    )
    active: bool = Field(
        default=True,
        description="Whether the event is currently active/tradeable."
    )
    created_at: datetime | None = Field(
        default=None,
        description="When the event was first observed in our system."
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last time event metadata was refreshed."
    )
    raw_json: dict | None = Field(
        default=None,
        description="Full raw API response for debugging and "
        "forward-compatibility."
    )
```

### DuckDB DDL

```sql
CREATE TABLE IF NOT EXISTS n_events (
    event_id          VARCHAR PRIMARY KEY,
    source            VARCHAR NOT NULL,
    title             VARCHAR NOT NULL,
    slug              VARCHAR,
    category          VARCHAR DEFAULT 'other',
    series_id         VARCHAR,
    is_negrisk        BOOLEAN NOT NULL DEFAULT FALSE,
    market_count      INTEGER DEFAULT 1,
    start_date        TIMESTAMPTZ,
    end_date          TIMESTAMPTZ,
    resolution_source VARCHAR,
    active            BOOLEAN NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    raw_json          JSON
);
```

### Source Mapping

| Field | Gamma API | Kalshi API | US API | The Graph |
|-------|-----------|------------|--------|-----------|
| `event_id` | `events[].id` | `event_ticker` | -- | -- |
| `title` | `events[].title` | `title` | -- | -- |
| `slug` | `events[].slug` | -- | -- | -- |
| `category` | Inferred from text | `category` | -- | -- |
| `series_id` | `events[].series_id` | `series_ticker` | -- | -- |
| `is_negrisk` | `events[].negRisk` | `mutually_exclusive` | -- | -- |
| `end_date` | `events[].endDate` | `close_time` | -- | -- |
| `resolution_source` | Inferred from market type | Always 'kalshi_internal' | -- | -- |

**Ambiguity:** Polymarket does not always expose a clean event-level endpoint for
every market. Some markets exist without a parent event. In those cases, a synthetic
event is created with `event_id = "pm_synthetic_{market_id}"` and `market_count = 1`.

---

## Schema 2: Market

A Market is a single tradeable binary contract within an Event. Each Market has
exactly two outcome tokens (YES and NO). Multi-outcome events contain multiple
Markets, one per possible outcome.

### Pydantic Model

```python
class Market(BaseModel):
    """Single binary market within an event.

    In Polymarket, a market maps to the /markets endpoint. Each market
    has exactly two outcomes (YES/NO) with corresponding CLOB token IDs.
    In NegRisk events, each outcome gets its own market.

    In Kalshi, a market maps to a single ticker within an event_ticker
    group.
    """

    market_id: str = Field(
        description="Canonical market identifier. "
        "Polymarket: Gamma API id or conditionId. "
        "Kalshi: ticker."
    )
    event_id: str = Field(
        description="Parent event ID. FK to Event.event_id."
    )
    source: str = Field(
        description="Data source: 'polymarket' | 'kalshi'."
    )
    condition_id: str | None = Field(
        default=None,
        description="Polymarket CTF condition ID (hex). "
        "The canonical on-chain identifier. "
        "Null for Kalshi markets."
    )
    question_id: str | None = Field(
        default=None,
        description="Polymarket UMA oracle question ID (hex). "
        "Used for resolution tracking. "
        "Null for Kalshi markets."
    )
    question: str = Field(
        description="The binary question text. "
        "Polymarket: question field. "
        "Kalshi: title or yes_sub_title."
    )
    slug: str | None = Field(
        default=None,
        description="URL-safe slug for this market."
    )
    clob_token_id_yes: str | None = Field(
        default=None,
        description="CLOB token ID for the YES outcome. "
        "Required for CLOB API calls (orderbook, price, history). "
        "Polymarket only. Extracted from clobTokenIds JSON array."
    )
    clob_token_id_no: str | None = Field(
        default=None,
        description="CLOB token ID for the NO outcome. "
        "Polymarket only."
    )
    neg_risk_market_id: str | None = Field(
        default=None,
        description="Parent NegRisk market ID. "
        "Non-null only for NegRisk multi-outcome markets. "
        "Polymarket only."
    )
    is_negrisk: bool = Field(
        default=False,
        description="Whether this market is part of a NegRisk group."
    )
    active: bool = Field(
        default=True,
        description="Whether the market is currently tradeable."
    )
    closed: bool = Field(
        default=False,
        description="Whether the market has been closed/resolved."
    )
    resolved_outcome: str | None = Field(
        default=None,
        description="Resolution result: 'yes', 'no', or null if unresolved. "
        "Populated after market closes."
    )
    resolution_timestamp: datetime | None = Field(
        default=None,
        description="When the market was resolved."
    )
    end_date: datetime | None = Field(
        default=None,
        description="Expected resolution date."
    )
    min_order_size: float = Field(
        default=5.0,
        description="Minimum order size in USDC."
    )
    min_tick_size: float = Field(
        default=0.01,
        description="Minimum price increment. "
        "Dynamic on Polymarket (changes near 0/1 extremes). "
        "Fetch via CLOB /tick-size for accuracy."
    )
    category: str = Field(
        default="other",
        description="Category for fee computation. "
        "Polymarket: determines fee rate. "
        "Kalshi: informational only."
    )
    fee_rate: float | None = Field(
        default=None,
        description="Category-specific fee rate (feeRateBps). "
        "Polymarket: 0.0 (geopolitics) to 0.072 (crypto). "
        "Kalshi: 0.03 (flat)."
    )
    created_at: datetime | None = Field(
        default=None,
        description="When the market was first observed."
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last metadata refresh timestamp."
    )
    raw_json: dict | None = Field(
        default=None,
        description="Full raw API response."
    )
```

### DuckDB DDL

```sql
CREATE TABLE IF NOT EXISTS n_markets (
    market_id           VARCHAR PRIMARY KEY,
    event_id            VARCHAR NOT NULL REFERENCES n_events(event_id),
    source              VARCHAR NOT NULL,
    condition_id        VARCHAR,
    question_id         VARCHAR,
    question            VARCHAR NOT NULL,
    slug                VARCHAR,
    clob_token_id_yes   VARCHAR,
    clob_token_id_no    VARCHAR,
    neg_risk_market_id  VARCHAR,
    is_negrisk          BOOLEAN NOT NULL DEFAULT FALSE,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    closed              BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_outcome    VARCHAR,
    resolution_timestamp TIMESTAMPTZ,
    end_date            TIMESTAMPTZ,
    min_order_size      DOUBLE DEFAULT 5.0,
    min_tick_size       DOUBLE DEFAULT 0.01,
    category            VARCHAR DEFAULT 'other',
    fee_rate            DOUBLE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    raw_json            JSON
);

CREATE INDEX IF NOT EXISTS idx_n_markets_event
    ON n_markets(event_id);
CREATE INDEX IF NOT EXISTS idx_n_markets_condition
    ON n_markets(condition_id);
CREATE INDEX IF NOT EXISTS idx_n_markets_source
    ON n_markets(source);
```

### Source Mapping

| Field | Gamma API | CLOB API | Kalshi API | US API | The Graph |
|-------|-----------|----------|------------|--------|-----------|
| `market_id` | `id` | -- | `ticker` | `id` | -- |
| `condition_id` | `conditionId` | -- | -- | -- | `conditions.id` |
| `question_id` | `questionId` | -- | -- | -- | `conditions.questionId` |
| `question` | `question` | -- | `title` | `question` | -- |
| `clob_token_id_yes` | `clobTokenIds[0]` | `asset_id` (from `/book`) | -- | -- | -- |
| `clob_token_id_no` | `clobTokenIds[1]` | -- | -- | -- | -- |
| `is_negrisk` | `negRisk` | -- | -- | -- | -- |
| `min_tick_size` | `minimumTickSize` | `tick_size` (from `/book`, `/tick-size`) | -- | -- | -- |
| `fee_rate` | -- | `feeRateBps` (from order context) | -- | -- | -- |
| `resolved_outcome` | Inferred from `outcomePrices` at close | -- | From `status` + `result` | -- | `conditions.payouts` |

### Canonical ID Strategy

For Polymarket markets, the canonical `market_id` is the Gamma API `id` field
(which is typically the `conditionId` or a numeric ID). The `condition_id` field
stores the hex `conditionId` separately for on-chain lookups.

For Kalshi markets, the canonical `market_id` is the `ticker` field.

Cross-platform market matching (same real-world question on both platforms) has no
canonical mapping. Matching requires fuzzy text similarity on `question` fields or
manual curation. A future `market_pairs` table could formalize these mappings.

---

## Schema 3: OutcomeToken

An OutcomeToken represents one side (YES or NO) of a binary market, with current
pricing and volume data. Each Market has exactly two OutcomeTokens.

### Pydantic Model

```python
class OutcomeToken(BaseModel):
    """YES or NO outcome token with pricing snapshot.

    Each binary market has exactly two OutcomeTokens. Their prices
    should sum to approximately 1.0 (deviations indicate arb opportunities
    or fee effects).

    On Polymarket, each outcome token has a unique CLOB token ID
    (large integer string) used for orderbook and price queries.
    """

    token_id: str = Field(
        description="Canonical token identifier. "
        "Polymarket: CLOB token ID (from clobTokenIds). "
        "Kalshi: '{ticker}_YES' or '{ticker}_NO' (synthetic)."
    )
    market_id: str = Field(
        description="Parent market ID. FK to Market.market_id."
    )
    outcome: str = Field(
        description="Outcome side: 'yes' or 'no'."
    )
    price: float = Field(
        default=0.0,
        description="Current price (0.0 to 1.0). "
        "Source priority: CLOB tokens > Gamma outcomePrices > US API."
    )
    price_source: str = Field(
        default="unknown",
        description="Which API provided this price. "
        "Values: 'clob_book', 'clob_price', 'gamma_tokens', "
        "'gamma_outcomePrices', 'us_api', 'kalshi_api'."
    )
    volume_24h: float = Field(
        default=0.0,
        description="24-hour trading volume in USD."
    )
    open_interest: float = Field(
        default=0.0,
        description="Current open interest in USD. "
        "Only available from Gamma API and Data API."
    )
    best_bid: float | None = Field(
        default=None,
        description="Best bid price. "
        "Requires CLOB /book or Kalshi yes_bid_dollars."
    )
    best_ask: float | None = Field(
        default=None,
        description="Best ask price. "
        "Requires CLOB /book or Kalshi yes_ask_dollars."
    )
    bid_depth_usd: float | None = Field(
        default=None,
        description="Total bid-side depth in USD. "
        "Requires CLOB /book with full aggregation."
    )
    ask_depth_usd: float | None = Field(
        default=None,
        description="Total ask-side depth in USD."
    )
    last_trade_price: float | None = Field(
        default=None,
        description="Last trade execution price. "
        "From CLOB /last-trade-price."
    )
    winner: bool | None = Field(
        default=None,
        description="Whether this outcome won (after resolution). "
        "True = pays $1.00, False = pays $0.00, None = unresolved."
    )
    observed_at: datetime = Field(
        description="Timestamp when this snapshot was taken."
    )
```

### DuckDB DDL

```sql
CREATE TABLE IF NOT EXISTS n_outcome_tokens (
    token_id          VARCHAR NOT NULL,
    market_id         VARCHAR NOT NULL REFERENCES n_markets(market_id),
    outcome           VARCHAR NOT NULL,
    price             DOUBLE DEFAULT 0.0,
    price_source      VARCHAR DEFAULT 'unknown',
    volume_24h        DOUBLE DEFAULT 0.0,
    open_interest     DOUBLE DEFAULT 0.0,
    best_bid          DOUBLE,
    best_ask          DOUBLE,
    bid_depth_usd     DOUBLE,
    ask_depth_usd     DOUBLE,
    last_trade_price  DOUBLE,
    winner            BOOLEAN,
    observed_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (token_id, observed_at)
);

CREATE INDEX IF NOT EXISTS idx_n_outcome_tokens_market
    ON n_outcome_tokens(market_id);
```

**Note:** The primary key is `(token_id, observed_at)` because this is a
time-series table. Each row is a snapshot of the token's state at a specific time.
Querying the latest snapshot uses `WHERE token_id = ? ORDER BY observed_at DESC LIMIT 1`.

### Source Mapping

| Field | Gamma API | CLOB API | Kalshi API | Data API | The Graph |
|-------|-----------|----------|------------|----------|-----------|
| `token_id` | `tokens[].token_id` | `asset_id` | Synthetic | -- | -- |
| `price` | `tokens[].price` or `outcomePrices[]` | `/price` or `/midpoint` | `yes_ask_dollars`, `no_ask_dollars` | -- | -- |
| `volume_24h` | `volumeNum24hr` (market-level) | -- | `volume_24h_fp` | -- | -- |
| `open_interest` | `openInterest` (market-level) | -- | -- | `/open-interest` | `conditions.volume` |
| `best_bid` | -- | `/book` bids[0].price | `yes_bid_dollars` | -- | -- |
| `best_ask` | -- | `/book` asks[0].price | `yes_ask_dollars` | -- | -- |
| `winner` | `tokens[].winner` | -- | Inferred from `result` | -- | `conditions.payouts` |

**Ambiguity:** Gamma API volume and open interest are market-level, not per-token.
The same value is assigned to both YES and NO tokens. Per-token volume requires
the Data API or on-chain data.

---

## Schema 4: OrderbookSnapshot

A point-in-time capture of the full L2 orderbook for a single outcome token.
This schema supports both live CLOB snapshots and historical data from pmxt/PolyBackTest.

### Pydantic Model

```python
class OrderbookLevel(BaseModel):
    """A single price level in the orderbook."""

    price: float = Field(
        description="Price level (0.0 to 1.0)."
    )
    size: float = Field(
        description="Total size at this price level in contracts/USDC."
    )


class OrderbookSnapshot(BaseModel):
    """Point-in-time L2 orderbook snapshot.

    Captures full bid and ask depth at a specific timestamp.
    The CLOB API returns orderbooks per token_id (one side of a market).
    Due to order mirroring, a buy of YES at $0.40 appears as a sell of
    NO at $0.60 on the opposite token's book.
    """

    snapshot_id: str = Field(
        description="Unique identifier for this snapshot. "
        "Generated as '{token_id}_{timestamp_ms}'."
    )
    token_id: str = Field(
        description="CLOB token ID. FK to OutcomeToken.token_id."
    )
    market_id: str = Field(
        description="Parent market ID for convenience. "
        "FK to Market.market_id."
    )
    source: str = Field(
        description="Data source: 'clob_api', 'pmxt_archive', "
        "'polybacktest', 'websocket'."
    )
    timestamp: datetime = Field(
        description="When the snapshot was captured."
    )
    best_bid: float | None = Field(
        default=None,
        description="Best (highest) bid price."
    )
    best_ask: float | None = Field(
        default=None,
        description="Best (lowest) ask price."
    )
    spread: float | None = Field(
        default=None,
        description="best_ask - best_bid. "
        "Null if either side is empty."
    )
    midpoint: float | None = Field(
        default=None,
        description="(best_bid + best_ask) / 2."
    )
    bid_depth_total: float = Field(
        default=0.0,
        description="Total bid-side depth in USDC."
    )
    ask_depth_total: float = Field(
        default=0.0,
        description="Total ask-side depth in USDC."
    )
    bid_levels: int = Field(
        default=0,
        description="Number of distinct bid price levels."
    )
    ask_levels: int = Field(
        default=0,
        description="Number of distinct ask price levels."
    )
    bids: list[OrderbookLevel] = Field(
        default_factory=list,
        description="Full bid side, sorted descending by price."
    )
    asks: list[OrderbookLevel] = Field(
        default_factory=list,
        description="Full ask side, sorted ascending by price."
    )
    tick_size: float = Field(
        default=0.01,
        description="Tick size at time of snapshot."
    )
    min_order_size: float = Field(
        default=5.0,
        description="Minimum order size in USDC."
    )
```

### DuckDB DDL

The orderbook is stored in two tables: a summary table for efficient querying and
a levels table for full depth reconstruction.

```sql
-- Summary table: one row per snapshot
CREATE TABLE IF NOT EXISTS n_orderbook_snapshots (
    snapshot_id       VARCHAR PRIMARY KEY,
    token_id          VARCHAR NOT NULL,
    market_id         VARCHAR NOT NULL,
    source            VARCHAR NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL,
    best_bid          DOUBLE,
    best_ask          DOUBLE,
    spread            DOUBLE,
    midpoint          DOUBLE,
    bid_depth_total   DOUBLE DEFAULT 0.0,
    ask_depth_total   DOUBLE DEFAULT 0.0,
    bid_levels        INTEGER DEFAULT 0,
    ask_levels        INTEGER DEFAULT 0,
    tick_size         DOUBLE DEFAULT 0.01,
    min_order_size    DOUBLE DEFAULT 5.0
);

CREATE INDEX IF NOT EXISTS idx_n_ob_snapshots_token_ts
    ON n_orderbook_snapshots(token_id, timestamp);

-- Levels table: individual price levels per snapshot
CREATE TABLE IF NOT EXISTS n_orderbook_levels (
    snapshot_id       VARCHAR NOT NULL REFERENCES n_orderbook_snapshots(snapshot_id),
    side              VARCHAR NOT NULL,       -- 'bid' | 'ask'
    price             DOUBLE NOT NULL,
    size              DOUBLE NOT NULL,
    level_rank        INTEGER NOT NULL,       -- 1 = best, 2 = second best, ...
    PRIMARY KEY (snapshot_id, side, level_rank)
);
```

### Source Mapping

| Field | CLOB `/book` | pmxt Archive | PolyBackTest | WebSocket |
|-------|-------------|-------------|--------------|-----------|
| `token_id` | `asset_id` | Market identifier | Market identifier | Subscribe token |
| `bids` | `bids[]` (price, size as strings) | Parquet columns | API response | `book` event |
| `asks` | `asks[]` (price, size as strings) | Parquet columns | API response | `book` event |
| `tick_size` | `tick_size` | -- | -- | `tick_size_change` event |
| `min_order_size` | `min_order_size` | -- | -- | -- |

---

## Schema 5: TradePrint

An individual trade execution -- the record that a transaction occurred at a
specific price and size. Sources include the Data API (per-wallet trades), The Graph
(on-chain settlement), and pmxt archive (historical trades).

### Pydantic Model

```python
class TradePrint(BaseModel):
    """Individual trade execution record.

    A TradePrint records that a specific quantity of a specific outcome
    token changed hands at a specific price. This can represent:
    - A live trade observed via WebSocket or Data API
    - A historical trade from The Graph or pmxt archive
    - A paper trade simulated by the research module

    Polymarket trades are technically limit order matches. There are no
    true market orders -- a FOK order at an aggressive price acts as a
    market order but is still a limit order fill.
    """

    trade_id: str = Field(
        description="Unique trade identifier. "
        "On-chain: transaction hash + log index. "
        "Data API: trade ID from response. "
        "Paper: UUID generated locally."
    )
    market_id: str = Field(
        description="Market ID. FK to Market.market_id."
    )
    token_id: str = Field(
        description="CLOB token ID of the traded outcome. "
        "FK to OutcomeToken.token_id."
    )
    source: str = Field(
        description="Data source: 'data_api', 'the_graph', "
        "'pmxt_archive', 'websocket', 'paper'."
    )
    timestamp: datetime = Field(
        description="Trade execution timestamp."
    )
    side: str = Field(
        description="Trade side: 'buy' or 'sell'. "
        "Relative to the outcome token (buy YES = bullish on outcome)."
    )
    price: float = Field(
        description="Execution price (0.0 to 1.0)."
    )
    size: float = Field(
        description="Trade size in contracts. "
        "1 contract = $1 at resolution if winning."
    )
    size_usd: float = Field(
        description="Trade notional in USD = price * size."
    )
    fee: float = Field(
        default=0.0,
        description="Fee paid on this trade in USD. "
        "Polymarket: taker fee (nonlinear, category-dependent). "
        "Kalshi: 3% on winning outcome."
    )
    is_taker: bool | None = Field(
        default=None,
        description="Whether this trade was taker-initiated. "
        "True = crossed the spread; False = resting order filled. "
        "Null if unknown."
    )
    maker_address: str | None = Field(
        default=None,
        description="Ethereum address of the maker (resting order). "
        "Available from The Graph and Data API."
    )
    taker_address: str | None = Field(
        default=None,
        description="Ethereum address of the taker."
    )
    block_number: int | None = Field(
        default=None,
        description="Polygon block number for on-chain settlement. "
        "Available from The Graph only."
    )
    tx_hash: str | None = Field(
        default=None,
        description="Polygon transaction hash. "
        "Available from The Graph only."
    )
```

### DuckDB DDL

```sql
CREATE TABLE IF NOT EXISTS n_trade_prints (
    trade_id          VARCHAR PRIMARY KEY,
    market_id         VARCHAR NOT NULL,
    token_id          VARCHAR NOT NULL,
    source            VARCHAR NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL,
    side              VARCHAR NOT NULL,
    price             DOUBLE NOT NULL,
    size              DOUBLE NOT NULL,
    size_usd          DOUBLE NOT NULL,
    fee               DOUBLE DEFAULT 0.0,
    is_taker          BOOLEAN,
    maker_address     VARCHAR,
    taker_address     VARCHAR,
    block_number      BIGINT,
    tx_hash           VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_n_trade_prints_market_ts
    ON n_trade_prints(market_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_n_trade_prints_token_ts
    ON n_trade_prints(token_id, timestamp);
```

### Source Mapping

| Field | Data API `/trades` | The Graph | pmxt Archive | WebSocket |
|-------|-------------------|-----------|-------------|-----------|
| `trade_id` | Response ID | `trades.id` (hash+index) | Archive row ID | -- |
| `side` | `side` | Inferred from token flow | Column | -- |
| `price` | `price` | `trades.price` | Column | `last_trade_price` event |
| `size` | `size` | `trades.amount` | Column | -- |
| `maker_address` | -- | `trades.maker.id` | -- | -- |
| `taker_address` | `user` | `trades.taker.id` | -- | -- |
| `block_number` | -- | Block context | -- | -- |
| `tx_hash` | -- | Transaction context | -- | -- |

---

## Schema 6: PriceSeries

A time series of market prices. This is the primary data structure for backtesting.
Sourced from CLOB `/prices-history`, pmxt archive, or aggregated from TradePrints.

### Pydantic Model

```python
class PricePoint(BaseModel):
    """A single price observation in a time series."""

    timestamp: datetime
    price: float = Field(description="Price (0.0 to 1.0).")


class PriceSeries(BaseModel):
    """Time series of prices for a single outcome token.

    Each PriceSeries covers one side (YES or NO) of one market.
    For a complete market view, join the YES and NO series by timestamp.

    The canonical source is CLOB /prices-history, but this endpoint
    has known limitations for resolved markets (reduced granularity).
    Historical data from pmxt or PolyBackTest may provide better coverage.
    """

    series_id: str = Field(
        description="Unique identifier: '{token_id}_{interval}_{start}_{end}'."
    )
    token_id: str = Field(
        description="CLOB token ID. FK to OutcomeToken.token_id."
    )
    market_id: str = Field(
        description="Parent market ID. FK to Market.market_id."
    )
    outcome: str = Field(
        description="Outcome side: 'yes' or 'no'."
    )
    source: str = Field(
        description="Data source: 'clob_prices_history', 'pmxt_archive', "
        "'polybacktest', 'aggregated_trades'."
    )
    interval: str = Field(
        description="Sampling interval: '1m', '5m', '1h', '6h', '1d', '1w'."
    )
    start_ts: datetime = Field(
        description="First observation timestamp."
    )
    end_ts: datetime = Field(
        description="Last observation timestamp."
    )
    point_count: int = Field(
        description="Number of price observations."
    )
    data_coverage: float = Field(
        default=1.0,
        description="Fraction of expected intervals with data (0.0 to 1.0). "
        "Calculated as actual_points / expected_points. "
        "Coverage < 0.80 should trigger a warning in backtests."
    )
    points: list[PricePoint] = Field(
        default_factory=list,
        description="The price observations, sorted ascending by timestamp."
    )
```

### DuckDB DDL

The series metadata and individual price points are stored separately for
efficient querying.

```sql
-- Series metadata
CREATE TABLE IF NOT EXISTS n_price_series (
    series_id         VARCHAR PRIMARY KEY,
    token_id          VARCHAR NOT NULL,
    market_id         VARCHAR NOT NULL,
    outcome           VARCHAR NOT NULL,
    source            VARCHAR NOT NULL,
    interval          VARCHAR NOT NULL,
    start_ts          TIMESTAMPTZ NOT NULL,
    end_ts            TIMESTAMPTZ NOT NULL,
    point_count       INTEGER NOT NULL,
    data_coverage     DOUBLE DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_n_price_series_token
    ON n_price_series(token_id);
CREATE INDEX IF NOT EXISTS idx_n_price_series_market
    ON n_price_series(market_id);

-- Individual price points
CREATE TABLE IF NOT EXISTS n_price_points (
    token_id          VARCHAR NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL,
    price             DOUBLE NOT NULL,
    source            VARCHAR NOT NULL,
    PRIMARY KEY (token_id, timestamp, source)
);

CREATE INDEX IF NOT EXISTS idx_n_price_points_token_ts
    ON n_price_points(token_id, timestamp);
```

### Source Mapping

| Field | CLOB `/prices-history` | pmxt Archive | PolyBackTest |
|-------|----------------------|-------------|--------------|
| `token_id` | `market` param (CLOB token ID) | Market identifier column | Market identifier |
| `timestamp` | `history[].t` (Unix seconds) | Parquet timestamp column | API timestamp |
| `price` | `history[].p` | Parquet price column | API price |
| `interval` | `interval` param | Hourly (fixed) | Varies (sub-minute to daily) |

**Known limitation:** CLOB `/prices-history` for resolved markets may collapse to
12+ hour granularity regardless of the `fidelity` parameter. The `data_coverage`
field flags series where this degradation occurred.

---

## Schema 7: MarketMetadata

Extended metadata about a market that does not change frequently. This schema
captures resolution criteria, tags, fee parameters, and other slowly changing
attributes that inform strategy selection and fee computation.

### Pydantic Model

```python
class MarketMetadata(BaseModel):
    """Extended market metadata for research and strategy selection.

    This schema captures attributes that change rarely or never:
    resolution criteria, tags, fee schedule, liquidity characteristics.
    Separated from the Market schema to avoid bloating the core table
    with infrequently queried fields.
    """

    market_id: str = Field(
        description="FK to Market.market_id."
    )
    # Resolution details
    resolution_criteria: str | None = Field(
        default=None,
        description="Human-readable resolution criteria text. "
        "Often stored on IPFS (accessed via questionId)."
    )
    oracle_type: str | None = Field(
        default=None,
        description="Resolution oracle: 'uma_optimistic', 'chainlink', "
        "'markets_team', 'kalshi_internal'."
    )
    oracle_address: str | None = Field(
        default=None,
        description="On-chain oracle contract address (Polygon)."
    )
    # Classification
    tags: list[str] = Field(
        default_factory=list,
        description="Market tags from Gamma /tags endpoint. "
        "Examples: 'elections', 'nba', 'fed-rates', 'btc-price'."
    )
    subcategory: str | None = Field(
        default=None,
        description="Finer-grained category. "
        "Example: category='sports', subcategory='nba'."
    )
    # Fee parameters (fetched from CLOB API per market)
    fee_rate_bps: float | None = Field(
        default=None,
        description="Taker fee rate from CLOB API. "
        "Used in fee formula: fee = C * p * feeRate * (p*(1-p))^exp."
    )
    fee_exponent: float | None = Field(
        default=None,
        description="Fee exponent. 1.0 for most categories, "
        "0.5 for economics."
    )
    maker_rebate_pct: float | None = Field(
        default=None,
        description="Maker rebate as fraction of taker fees. "
        "0.20 (crypto) to 0.50 (finance)."
    )
    # Liquidity characteristics (computed from observations)
    avg_spread_7d: float | None = Field(
        default=None,
        description="Average bid-ask spread over trailing 7 days. "
        "Computed from OrderbookSnapshot data."
    )
    avg_volume_7d: float | None = Field(
        default=None,
        description="Average daily volume over trailing 7 days."
    )
    avg_depth_7d: float | None = Field(
        default=None,
        description="Average total book depth (bid+ask) over 7 days."
    )
    # Lifecycle
    first_trade_date: datetime | None = Field(
        default=None,
        description="Date of first trade in this market."
    )
    total_volume: float | None = Field(
        default=None,
        description="Lifetime cumulative volume in USD."
    )
    total_trades: int | None = Field(
        default=None,
        description="Lifetime trade count."
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last metadata refresh."
    )
```

### DuckDB DDL

```sql
CREATE TABLE IF NOT EXISTS n_market_metadata (
    market_id           VARCHAR PRIMARY KEY REFERENCES n_markets(market_id),
    resolution_criteria VARCHAR,
    oracle_type         VARCHAR,
    oracle_address      VARCHAR,
    tags                VARCHAR[],
    subcategory         VARCHAR,
    fee_rate_bps        DOUBLE,
    fee_exponent        DOUBLE,
    maker_rebate_pct    DOUBLE,
    avg_spread_7d       DOUBLE,
    avg_volume_7d       DOUBLE,
    avg_depth_7d        DOUBLE,
    first_trade_date    TIMESTAMPTZ,
    total_volume        DOUBLE,
    total_trades        INTEGER,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
```

### Source Mapping

| Field | Gamma API | CLOB API | Data API | The Graph |
|-------|-----------|----------|----------|-----------|
| `resolution_criteria` | -- | -- | -- | `conditions.questionId` -> IPFS |
| `oracle_type` | Inferred from market config | -- | -- | Oracle address |
| `tags` | `/tags` endpoint | -- | -- | -- |
| `fee_rate_bps` | -- | From order context / `/tick-size` response | -- | -- |
| `total_volume` | `volume` | -- | -- | `conditions.volume` |
| `total_trades` | -- | -- | -- | `conditions.tradeCount` |

**Ambiguity:** Fee parameters (`fee_rate_bps`, `fee_exponent`, `maker_rebate_pct`)
must be fetched from the CLOB API at order-signing time. There is no dedicated
"fee schedule" endpoint -- the values are embedded in the order context. The
hardcoded fallback values in `pm_fees.py` (proposed in the ADR) provide defaults
when the CLOB API is unavailable.

---

## Schema 8: ExperimentSnapshot

A point-in-time capture of all relevant market data for research replay. This
schema bundles market state, pricing, and orderbook data into a single atomic
snapshot that can be replayed deterministically for backtesting.

### Pydantic Model

```python
class ExperimentSnapshot(BaseModel):
    """Point-in-time capture of market state for research replay.

    An ExperimentSnapshot records everything needed to replay a single
    decision point in a backtest:
    - Which markets were active
    - What prices were available
    - What orderbook depth existed
    - What the fee schedule was

    Snapshots are immutable once created. They serve as the input to
    backtest engine steps and paper trading decisions. A sequence of
    snapshots constitutes the backtest's "tape" of market history.
    """

    snapshot_id: str = Field(
        description="Unique snapshot identifier. "
        "Format: '{source}_{timestamp_iso}_{sequence}'."
    )
    timestamp: datetime = Field(
        description="Point-in-time for this snapshot."
    )
    source: str = Field(
        description="How the snapshot was created: "
        "'live_scan' (from API), 'historical_replay' (from archive), "
        "'synthetic' (generated for testing)."
    )
    # Market universe at this point in time
    active_market_count: int = Field(
        default=0,
        description="Number of active/tradeable markets."
    )
    total_volume_24h: float = Field(
        default=0.0,
        description="Aggregate 24h volume across all markets in USD."
    )
    # Per-market state
    markets: list[MarketState] = Field(
        default_factory=list,
        description="State of each observed market at snapshot time."
    )
    # Scan metadata
    scan_duration_ms: int | None = Field(
        default=None,
        description="How long the data collection took in milliseconds."
    )
    data_quality: str = Field(
        default="unknown",
        description="Data quality assessment: 'complete', 'partial', "
        "'degraded'. Based on API response completeness."
    )
    notes: str | None = Field(
        default=None,
        description="Free-text notes for this snapshot."
    )


class MarketState(BaseModel):
    """Per-market state within an ExperimentSnapshot."""

    market_id: str
    condition_id: str | None = None
    question: str
    category: str
    active: bool
    is_negrisk: bool = False
    yes_price: float
    no_price: float
    spread: float = Field(
        description="yes_price + no_price - 1.0. "
        "Negative = potential rebalancing arb."
    )
    volume_24h: float = 0.0
    open_interest: float = 0.0
    best_bid_yes: float | None = None
    best_ask_yes: float | None = None
    best_bid_no: float | None = None
    best_ask_no: float | None = None
    days_to_resolution: float | None = None
    fee_rate: float | None = None
```

### DuckDB DDL

```sql
-- Snapshot header
CREATE TABLE IF NOT EXISTS n_experiment_snapshots (
    snapshot_id         VARCHAR PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    source              VARCHAR NOT NULL,
    active_market_count INTEGER DEFAULT 0,
    total_volume_24h    DOUBLE DEFAULT 0.0,
    scan_duration_ms    INTEGER,
    data_quality        VARCHAR DEFAULT 'unknown',
    notes               VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_n_exp_snapshots_ts
    ON n_experiment_snapshots(timestamp);

-- Per-market state within a snapshot
CREATE TABLE IF NOT EXISTS n_experiment_market_states (
    snapshot_id         VARCHAR NOT NULL
                        REFERENCES n_experiment_snapshots(snapshot_id),
    market_id           VARCHAR NOT NULL,
    condition_id        VARCHAR,
    question            VARCHAR NOT NULL,
    category            VARCHAR,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    is_negrisk          BOOLEAN NOT NULL DEFAULT FALSE,
    yes_price           DOUBLE NOT NULL,
    no_price            DOUBLE NOT NULL,
    spread              DOUBLE NOT NULL,
    volume_24h          DOUBLE DEFAULT 0.0,
    open_interest       DOUBLE DEFAULT 0.0,
    best_bid_yes        DOUBLE,
    best_ask_yes        DOUBLE,
    best_bid_no         DOUBLE,
    best_ask_no         DOUBLE,
    days_to_resolution  DOUBLE,
    fee_rate            DOUBLE,
    PRIMARY KEY (snapshot_id, market_id)
);
```

---

## Join Paths

### Entity Relationships

```
Event (1) ──────< Market (N)
                    │
                    ├──< OutcomeToken (2 per market: YES + NO)
                    │       │
                    │       ├──< OrderbookSnapshot (time series)
                    │       │
                    │       ├──< PriceSeries (per interval)
                    │       │       │
                    │       │       └──< PricePoint (time series)
                    │       │
                    │       └──< TradePrint (time series)
                    │
                    ├──  MarketMetadata (1:1)
                    │
                    └──< ExperimentSnapshot.MarketState (via snapshot)

ExperimentSnapshot (1) ──────< MarketState (N)
```

### Common Join Queries

**Get all markets in an event with latest prices:**

```sql
SELECT
    e.title AS event_title,
    m.question,
    ot_yes.price AS yes_price,
    ot_no.price AS no_price,
    ot_yes.volume_24h
FROM n_events e
JOIN n_markets m ON m.event_id = e.event_id
LEFT JOIN LATERAL (
    SELECT price, volume_24h
    FROM n_outcome_tokens
    WHERE market_id = m.market_id AND outcome = 'yes'
    ORDER BY observed_at DESC LIMIT 1
) ot_yes ON true
LEFT JOIN LATERAL (
    SELECT price
    FROM n_outcome_tokens
    WHERE market_id = m.market_id AND outcome = 'no'
    ORDER BY observed_at DESC LIMIT 1
) ot_no ON true
WHERE e.event_id = ?;
```

**Get price history for a market (both sides):**

```sql
SELECT
    pp_yes.timestamp,
    pp_yes.price AS yes_price,
    pp_no.price AS no_price
FROM n_price_points pp_yes
JOIN n_markets m ON m.clob_token_id_yes = pp_yes.token_id
JOIN n_price_points pp_no
    ON pp_no.token_id = m.clob_token_id_no
    AND pp_no.timestamp = pp_yes.timestamp
    AND pp_no.source = pp_yes.source
WHERE m.market_id = ?
ORDER BY pp_yes.timestamp;
```

**Get NegRisk arb opportunities from latest snapshot:**

```sql
SELECT
    es.snapshot_id,
    es.timestamp,
    ems.market_id,
    ems.question,
    ems.spread,
    ems.volume_24h
FROM n_experiment_snapshots es
JOIN n_experiment_market_states ems
    ON ems.snapshot_id = es.snapshot_id
WHERE ems.is_negrisk = TRUE
  AND ems.spread < -0.03
ORDER BY ems.spread ASC;
```

**Cross-platform price comparison (Polymarket vs Kalshi):**

```sql
-- Requires manual market_id pairing or fuzzy question matching
SELECT
    pm.market_id AS pm_market_id,
    pm.question AS pm_question,
    pm_tok.price AS pm_yes_price,
    k.market_id AS kalshi_ticker,
    k.question AS kalshi_question,
    k_tok.price AS kalshi_yes_price,
    ABS(pm_tok.price - k_tok.price) AS price_divergence
FROM n_markets pm
JOIN n_markets k ON k.source = 'kalshi'
    -- Fuzzy match: requires curation or similarity function
    AND levenshtein(LOWER(pm.question), LOWER(k.question)) < 20
JOIN LATERAL (
    SELECT price FROM n_outcome_tokens
    WHERE market_id = pm.market_id AND outcome = 'yes'
    ORDER BY observed_at DESC LIMIT 1
) pm_tok ON true
JOIN LATERAL (
    SELECT price FROM n_outcome_tokens
    WHERE market_id = k.market_id AND outcome = 'yes'
    ORDER BY observed_at DESC LIMIT 1
) k_tok ON true
WHERE pm.source = 'polymarket'
  AND ABS(pm_tok.price - k_tok.price) > 0.03
ORDER BY price_divergence DESC;
```

---

## Canonical ID Strategy Summary

| Entity | Canonical ID | Format | Source of Truth |
|--------|-------------|--------|-----------------|
| Event | `event_id` | Gamma event ID (PM) / `event_ticker` (Kalshi) | Gamma API / Kalshi API |
| Market | `market_id` | Gamma market ID (PM) / `ticker` (Kalshi) | Gamma API / Kalshi API |
| OutcomeToken | `token_id` | CLOB token ID (PM) / `{ticker}_{YES\|NO}` (Kalshi) | Gamma `clobTokenIds` / Synthetic |
| OrderbookSnapshot | `snapshot_id` | `{token_id}_{timestamp_ms}` | Generated |
| TradePrint | `trade_id` | API trade ID / tx hash+index / UUID | Source-dependent |
| PriceSeries | `series_id` | `{token_id}_{interval}_{start}_{end}` | Generated |
| MarketMetadata | `market_id` | Same as Market.market_id | FK to Market |
| ExperimentSnapshot | `snapshot_id` | `{source}_{timestamp_iso}_{sequence}` | Generated |

### Cross-Surface Identifier Resolution

To look up data for a market across all surfaces:

1. Start with `market_id` from the `n_markets` table.
2. Use `condition_id` for The Graph queries and on-chain lookups.
3. Use `clob_token_id_yes` / `clob_token_id_no` for CLOB API calls.
4. Use `question_id` for UMA oracle resolution tracking.
5. For Kalshi, the `market_id` (= `ticker`) is used directly.

```python
# Pseudocode: resolve all identifiers for a market
market = db.query("SELECT * FROM n_markets WHERE market_id = ?", [mid])

# Gamma API lookups
gamma_market = gamma_client.fetch_market(market.market_id)

# CLOB API lookups (price, orderbook, history)
clob_price = clob_client.get_price(token_id=market.clob_token_id_yes)
clob_book = clob_client.get_book(token_id=market.clob_token_id_yes)
clob_history = clob_client.get_prices_history(market=market.clob_token_id_yes)

# The Graph lookups
graph_condition = subgraph.query(
    "{ conditions(where: {id: $cid}) { ... } }",
    variables={"cid": market.condition_id}
)

# Data API lookups (by wallet, not by market)
# Use market_id for open-interest queries
oi = data_api.get_open_interest(market=market.market_id)
```

---

## Schema Versioning

All table names use the `n_` prefix ("normalized") to distinguish from existing
`pm_` tables in `arb/schema.py`. This allows both schema generations to coexist
during migration.

| Version | Tables | Purpose |
|---------|--------|---------|
| v1 (`pm_*`) | `pm_markets`, `pm_conditions`, `pm_negrisk_groups`, `pm_arb_opportunities`, `pm_combinatorial_pairs`, `pm_scan_log`, `pm_executions`, `kalshi_combinatorial_pairs` | Arb scanning (existing) |
| v2 (`n_*`) | `n_events`, `n_markets`, `n_outcome_tokens`, `n_orderbook_snapshots`, `n_orderbook_levels`, `n_trade_prints`, `n_price_series`, `n_price_points`, `n_market_metadata`, `n_experiment_snapshots`, `n_experiment_market_states` | Normalized research (this document) |

Migration from v1 to v2 is not required. The v1 tables continue to serve the arb
scanner. New research modules should write to v2 tables. If a unified schema is
desired later, v1 data can be ETL'd into v2 tables using `market_id` and
`condition_id` as join keys.

---

## Open Questions and Ambiguities

1. **Gamma API `id` stability.** It is unclear whether Gamma API market `id` values
   are stable across API versions or whether they can change. If they change, the
   `conditionId` (hex, on-chain) should become the canonical `market_id` for
   Polymarket. Currently, both are stored.

2. **CLOB token ID assignment.** The mapping of `clobTokenIds[0]` to YES and
   `clobTokenIds[1]` to NO is inferred from the `outcomes` array ordering and
   the `tokens` array `outcome` field. There is no explicit documentation confirming
   this mapping is guaranteed. The existing `gamma_client.py` parsing logic uses
   the `tokens[].outcome` field when available, falling back to positional mapping.

3. **Fee parameter availability.** The `feeRateBps` and exponent values are not
   available from a dedicated endpoint. They must be fetched from the CLOB API at
   order-signing time. For read-only research, hardcoded fallback values (as of
   March 2026) are used. These may become stale.

4. **Historical data date ranges.** The pmxt archive and PolyBackTest do not
   publish guaranteed date ranges. Coverage must be empirically verified before
   relying on these sources for multi-year backtests.

5. **Cross-platform market matching.** No canonical mapping exists between
   Polymarket `conditionId` and Kalshi `ticker` for markets covering the same
   real-world event. Matching requires fuzzy text similarity, manual curation,
   or a dedicated mapping table. This is a prerequisite for cross-platform arb
   research.

6. **The Graph subgraph stability.** The specific subgraph ID
   (`Bx1W4S7kDVxs9gC3s2G6DS8kdNBJNVhMviCtin2DiBp`) may change during The Graph
   network migrations. The `n_price_points.source` field tracks which source
   provided each data point, allowing data to be refreshed if a source becomes
   unavailable.

7. **Volume attribution.** Gamma API provides volume at the market level, not
   per-outcome-token. The `OutcomeToken.volume_24h` field stores this market-level
   value for both YES and NO tokens. Per-token volume disaggregation requires
   trade-level data from the Data API or The Graph.

---

*Document generated 2026-04-06. Schema definitions should be updated when API
specifications change or when new data surfaces become available.*

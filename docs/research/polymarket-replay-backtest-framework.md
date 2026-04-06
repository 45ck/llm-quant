# Polymarket replay/backtest framework for llm-quant

**A deterministic, event-driven backtesting system for offline evaluation of Polymarket prediction market trading hypotheses can be built on top of Polymarket's hybrid CLOB architecture, combining on-chain trade reconstruction with real-time orderbook capture.** The framework centers on three pillars: a Parquet-based replay data store ingested from Polymarket's public APIs and on-chain settlement events, a CLOB simulator that models Polymarket-specific execution (dynamic fees, tick-size enforcement, level-by-level book walking, partial fills), and a metrics engine purpose-built for binary-outcome payoffs. This design integrates as a `polymarket/` subpackage within the existing `llm-quant` repo, sharing its event bus and configuration patterns.

---

## How Polymarket's hybrid CLOB actually works

Polymarket operates a **hybrid-decentralized Central Limit Order Book** on Polygon. Orders are EIP-712 signed off-chain, matched by a centralized operator using price-time priority, then settled atomically on-chain via the CTF Exchange smart contract. This means off-chain order placement is sub-second and gasless, while settlement finality arrives in **~2 seconds** (Polygon block time). All outcomes are tokenized as ERC-1155 conditional tokens via the **Gnosis Conditional Token Framework (CTF)**: each binary market mints YES and NO tokens where **1 YES + 1 NO ≡ $1.00 USDC** — a hard invariant the simulator must enforce.

The CLOB supports five order types: **GTC** (rests on book), **GTD** (expires at timestamp), **FOK** (fill-or-kill), **FAK** (fill-and-kill / IOC equivalent), and **Post-Only** (rejected if it would cross the spread). Maker fees are **zero across all categories**. Taker fees follow a dynamic formula `fee = C × p × feeRate × (p × (1−p))^exponent`, where rates vary by market category — from **0% for geopolitics** to **1.80% peak for crypto** at p=0.50. This fee curve peaks at mid-probability and decays toward the extremes, meaning execution cost modeling must be price-dependent, not flat.

Markets resolve via three oracle paths: **UMA Optimistic Oracle** (primary, ~2-hour dispute window), **Chainlink Data Streams** (crypto price markets, 15-minute contracts), or the Polymarket Markets Team. The standard UMA flow involves a $750 USDC bond, a 2-hour challenge period, and potential escalation to DVM token-holder vote (4–6 days). The simulator must track resolution state to compute terminal PnL.

---

## Data infrastructure and ingestion pipeline

### What Polymarket exposes

Polymarket provides four API surfaces, all with public read access (no authentication required for market data):

| API | Base URL | Key data |
|-----|----------|----------|
| CLOB | `clob.polymarket.com` | Live orderbook (`GET /book`), prices, spread, midpoint, price history |
| Gamma | `gamma-api.polymarket.com` | Market metadata, event details, condition IDs, token IDs, resolution status |
| Data | `data-api.polymarket.com` | Positions, activity, leaderboards |
| WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | Real-time orderbook deltas, price updates, trade events |

The **`GET /prices-history`** endpoint returns time-series price data at configurable fidelity (down to 1-minute resolution), which supports basic signal research. However, **no historical orderbook snapshots exist via the API** — this is the critical gap. For orderbook replay, you must either capture snapshots forward-looking via polling or WebSocket, or reconstruct trade flow from on-chain events.

### Historical data sources for the replay store

The ingestion pipeline should combine three tiers of data, prioritized by fidelity:

**Tier 1 — On-chain trade reconstruction.** Every matched trade settles on Polygon, emitting `OrderFilled` and `OrdersMatched` events from the CTF Exchange contract (`0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`). The open-source `poly_data` pipeline (github.com/warproxxx/poly_data) scrapes these via Goldsky subgraph and outputs structured Parquet with fields: `timestamp`, `maker`, `makerAssetId`, `makerAmountFilled`, `taker`, `takerAssetId`, `takerAmountFilled`, `transactionHash`. The `prediction-market-analysis` repo (1.9k stars) provides the largest public Parquet dataset of Polymarket trades and market metadata.

**Tier 2 — Orderbook snapshots.** The **pmxt archive** (`archive.pmxt.dev/Polymarket`) provides free hourly orderbook snapshots in Parquet (130–180 MB each). For higher fidelity, **PolyBackTest** (`polybacktest.com`) offers sub-second L2 orderbook depth across 90M+ snapshots. For self-hosted capture, poll `GET /book` at the CLOB's 300 req/10s rate limit or subscribe to the WebSocket market channel for real-time deltas.

**Tier 3 — Price time-series and metadata.** The CLOB `/prices-history` endpoint and Gamma API `/markets` endpoint provide price OHLC and full market metadata (question text, category, resolution status, token IDs, outcomes, volume, liquidity). These are essential for market selection, universe construction, and resolution labeling.

### Replay data schemas

All data stored as **Parquet** (columnar, compressed, schema-aware). Four core tables:

```
MarketMeta (partition by: category, year)
──────────────────────────────────────────
condition_id     : string        # primary key
question         : string
category         : string        # crypto, politics, sports, etc.
tokens           : list<struct{token_id: string, outcome: string}>
neg_risk         : bool
tick_size        : float64       # 0.01, 0.001, etc.
min_order_size   : float64       # typically 1.0
fee_rate_bps     : float64       # category-specific
fee_exponent     : float64
maker_rebate_pct : float64
created_at       : timestamp[ns]
end_date         : timestamp[ns]
resolved_at      : timestamp[ns, nullable]
resolution       : string        # YES / NO / null
volume_usdc      : float64
```

```
Trade (partition by: condition_id, date)
────────────────────────────────────────
timestamp        : timestamp[ns]
condition_id     : string
token_id         : string
side             : enum{BUY, SELL}
price            : float64
size             : float64
maker_address    : string
taker_address    : string
tx_hash          : string
fee_usdc         : float64
```

```
OrderbookSnapshot (partition by: token_id, date)
─────────────────────────────────────────────────
timestamp        : timestamp[ns]
token_id         : string
bids             : list<struct{price: f64, size: f64}>
asks             : list<struct{price: f64, size: f64}>
best_bid         : float64
best_ask         : float64
mid_price        : float64
spread           : float64
book_depth_bid   : float64   # total bid size in USDC
book_depth_ask   : float64
```

```
PriceBar (partition by: token_id, interval)
────────────────────────────────────────────
timestamp        : timestamp[ns]
token_id         : string
open             : float64
high             : float64
low              : float64
close            : float64
volume           : float64
num_trades       : int64
```

---

## Minimum viable simulator design

The simulator is event-driven, deterministic, and Polymarket-specific. Every component communicates through a single-threaded event queue — no concurrency, no non-determinism.

### Event loop architecture

```
┌──────────────────────────────────────────────────────────┐
│                    REPLAY ENGINE                          │
│                                                          │
│  DataReplay ──▶ MarketEvent ──▶ Strategy ──▶ SignalEvent │
│       │                                         │        │
│       │         ┌──────────┐                    ▼        │
│       │         │  EVENT   │◀──────────── OrderEvent     │
│       └────────▶│  QUEUE   │                    │        │
│                 └──────────┘                    ▼        │
│                      │              SimulatedExchange     │
│                      ▼                    │               │
│                 Portfolio ◀────────── FillEvent           │
│                      │                                    │
│                      ▼                                    │
│                MetricsEngine                              │
└──────────────────────────────────────────────────────────┘
```

### Core components

**DataReplay** reads Parquet files in timestamp order and emits `MarketEvent`s (orderbook snapshots or trade prints). It enforces strict chronological ordering across multiple token feeds. A `clock` attribute tracks simulated time — no `datetime.now()` calls anywhere in the simulator.

**SimulatedExchange** is the heart of the framework. It maintains a synthetic orderbook per token, updated from replay data, and processes incoming `OrderEvent`s against it. This component encodes all Polymarket-specific logic:

```python
@dataclass(frozen=True)
class FillModel:
    """Polymarket CLOB fill simulation."""

    def execute_against_book(
        self,
        order: Order,
        book: OrderBook,
        sim_time: int,
    ) -> list[Fill]:
        """Walk the book level-by-level, respecting tick size and min order size."""
        fills = []
        remaining = order.size

        levels = book.asks if order.side == Side.BUY else book.bids
        for level in levels:
            if remaining <= 0:
                break
            if not self._price_crosses(order, level):
                break

            fill_size = min(remaining, level.size * self.fill_rate)
            fill_price = level.price  # taker gets maker's price (price improvement)

            fee = self._compute_fee(fill_size, fill_price, order.market_meta)
            fills.append(Fill(
                price=fill_price,
                size=fill_size,
                fee=fee,
                timestamp=sim_time,
                is_maker=False,
            ))
            remaining -= fill_size

        return fills
```

**Five sub-models within the exchange:**

- **Fill model.** Level-by-level book walking with configurable fill rate (0.0–1.0) to model partial availability. FOK orders require full fill or reject; FAK fills what's available; GTC/GTD rest unfilled remainder on a synthetic maker book. Default fill rate of **0.8** for conservative simulation (assumes 20% of resting liquidity is stale or would cancel before your order arrives).

- **Fee model.** Implements the exact Polymarket dynamic fee formula: `fee = shares × price × fee_rate × (price × (1 − price))^exponent`. Fee parameters loaded per market from `MarketMeta`. Maker rebates applied when simulating maker fills. Minimum fee granularity: **0.0001 USDC**.

- **Spread model.** For periods without orderbook snapshots, estimates spread from historical trade data using a configurable model. Default: `spread = max(tick_size, base_spread × (1 / sqrt(daily_volume)))`. The spread widens inversely with the square root of volume — consistent with empirical findings that Kyle's λ on Polymarket ranged from **0.518** (thin early markets) to **0.01** (mature election markets).

- **Slippage model.** Beyond the mechanical book-walking slippage, applies an additional **market impact penalty**: `impact = κ × sqrt(order_size / avg_daily_volume)` where κ is calibrated per liquidity regime. For liquid markets (>$1M volume): κ ≈ 0.005. For illiquid markets (<$50K volume): κ ≈ 0.05. This captures the empirical observation that even moderate orders in thin prediction market books produce step-function price impact.

- **Stale-book handler.** Detects stale conditions: if the last orderbook snapshot is older than a configurable threshold (default: 300 seconds), the exchange flags the book as stale and applies a **spread multiplier** (default: 2.0×). If stale beyond a hard cutoff (default: 3600 seconds), all non-FOK orders are rejected. Near resolution (within configurable hours of `end_date`), the stale threshold tightens and spread multiplier increases — modeling the empirical pattern of market makers pulling liquidity pre-resolution.

**Tick-size enforcement.** Every order price is validated against the market's tick size before acceptance. Prices not aligned to the tick grid are **rejected**, not rounded — matching Polymarket's actual behavior. Size is validated to 2 decimal places.

**Portfolio** tracks positions per token, cash balance, unrealized PnL (mark-to-market at current mid), and realized PnL (from fills and resolutions). At resolution, winning positions pay $1.00 per share and losing positions pay $0.00. The portfolio enforces a **no-leverage constraint**: total position cost cannot exceed available cash (fully collateralized, matching Polymarket's model).

### Determinism guarantees

The simulator achieves full determinism through: (1) no wall-clock time references — all timestamps from replay data, (2) single-threaded event queue with stable sort on `(timestamp, event_type_priority)`, (3) fixed random seed for any stochastic components (e.g., fill rate jitter), (4) all floating-point operations use `Decimal` for fee calculations to avoid rounding drift. Every run produces identical output given identical inputs and configuration — essential for auditability.

---

## Experiment output metrics

The metrics engine computes a comprehensive set of statistics purpose-built for binary-outcome prediction market strategies. Metrics are computed both per-market and in aggregate, exported as JSON and Parquet.

### Core PnL metrics

**Realized PnL** decomposes into four components: (1) **Edge PnL** = Σ(resolution_price − entry_price) × size — pure forecasting alpha; (2) **Execution PnL** = Σ(theoretical_mid − actual_fill) × size — execution quality; (3) **Fee PnL** = −Σ fees_paid — cost of access; (4) **Slippage PnL** = Σ(expected_fill − actual_fill) × size — model vs. reality gap. This decomposition reveals whether a strategy's edge is real or consumed by execution costs.

**Hit rate** is the fraction of positions that resolved profitably, but reported alongside **calibration-adjusted hit rate** — the excess of actual wins over the expected win rate implied by entry prices. A strategy buying YES at $0.80 should win 80% of the time; value comes only from exceeding that baseline.

### Prediction-market-specific metrics

**Average edge captured** = mean(resolution_value − entry_price) across all positions, where resolution_value ∈ {0, 1}. This directly measures the strategy's probability estimation advantage in dollar terms.

**Closing Line Value (CLV)** adapts the gold-standard sports betting metric: `CLV = (closing_mid − entry_price) / entry_price`, where closing_mid is the midpoint price at a defined cutoff before resolution (default: 1 hour). Consistently positive CLV predicts long-term profitability with far fewer observations (~100 trades) than PnL significance requires.

**Brier Score** = (1/N) × Σ(implied_probability − actual_outcome)², computed on the strategy's entry prices as probability forecasts. Lower is better; **<0.125 is good, <0.10 is excellent**. Polymarket's own 12-hour-ahead Brier Score is 0.058 across ~90K predictions.

**Kelly fraction utilization** = actual position size / Kelly-optimal size, tracking whether the strategy appropriately sizes bets. The Kelly optimal for binary outcomes is `f* = (bp − q) / b` where `b = (1 − market_price) / market_price`.

### Risk and robustness metrics

**Deflated Sharpe Ratio (DSR)** corrects for multiple testing, non-normal returns (binary payoffs produce extreme skewness/kurtosis), and short sample lengths — critical since prediction market backtests are inherently sample-limited. Implemented via the Probabilistic Sharpe Ratio framework: `PSR = Φ((SR − SR*) × √(T−1) / √(1 − γ₃×SR + (γ₄−1)/4 × SR²))`.

**Maximum drawdown** and **drawdown duration** are tracked on the equity curve. Given binary payoffs can produce −100% on individual positions, drawdown analysis is essential for position-sizing validation.

**Realized vs. expected slippage** ratio: `slip_ratio = mean(realized_slippage) / mean(model_predicted_slippage)`. Values consistently >1.0 indicate the slippage model is too optimistic and needs recalibration.

**Latency-adjusted performance** applies a configurable delay (default: 2 seconds for Polygon settlement, 100ms for WebSocket feed lag) to all signal timestamps before execution. The delta between zero-latency and latency-adjusted PnL quantifies time-sensitivity of the strategy.

### Output format

```python
@dataclass
class ExperimentResult:
    # Identification
    experiment_id: str
    strategy_name: str
    run_timestamp: str          # ISO 8601
    config_hash: str            # SHA-256 of full config for reproducibility

    # PnL decomposition
    total_pnl: Decimal
    edge_pnl: Decimal
    execution_pnl: Decimal
    fee_pnl: Decimal
    slippage_pnl: Decimal

    # Rates
    hit_rate: float
    calibration_adjusted_hit_rate: float
    avg_edge_captured: float
    avg_entry_price: float

    # Prediction quality
    brier_score: float
    log_loss: float
    clv_mean: float
    clv_median: float

    # Risk
    sharpe_ratio: float
    deflated_sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration_hours: float
    kelly_utilization_mean: float

    # Execution quality
    realized_vs_expected_slippage: float
    avg_fill_rate: float
    partial_fill_pct: float
    rejected_order_pct: float
    latency_adjusted_pnl: Decimal

    # Volume
    num_trades: int
    num_markets: int
    total_volume_usdc: Decimal

    # Per-market breakdown
    market_results: list[MarketResult]

    # Full equity curve (timestamp, equity) for plotting
    equity_curve: list[tuple[int, Decimal]]
```

---

## Module structure for integration into llm-quant

The framework integrates as a `polymarket/` subpackage within the existing `llm-quant` repository structure, following the repo's conventions while keeping prediction-market logic cleanly isolated.

```
llm-quant/
├── ...                              # existing modules
├── polymarket/                      # new subpackage
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                 # Market, Token, Order, Fill, Position dataclasses
│   │   ├── events.py                # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   │   ├── enums.py                 # Side, OrderType, Resolution, MarketCategory
│   │   └── config.py               # PolymarketConfig (fees, tick sizes, latency params)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py               # Parquet schema definitions (pa.schema objects)
│   │   ├── gamma_client.py          # Market metadata fetcher (Gamma API)
│   │   ├── clob_client.py           # Thin wrapper around py-clob-client
│   │   ├── trade_fetcher.py         # On-chain trade reconstruction (Goldsky/Dune)
│   │   ├── book_collector.py        # Orderbook snapshot collector (poll / WS)
│   │   ├── pmxt_loader.py           # Loader for pmxt archive Parquet files
│   │   └── store.py                 # Unified read/write to local Parquet store
│   │
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── engine.py                # Event loop: DataReplay → Strategy → Exchange → Portfolio
│   │   ├── exchange.py              # SimulatedExchange: book state, order processing
│   │   ├── fill_model.py            # Level-by-level book walking, fill rate, partial fills
│   │   ├── fee_model.py             # Dynamic Polymarket fee formula
│   │   ├── spread_model.py          # Spread estimation for missing book data
│   │   ├── slippage_model.py        # Market impact: sqrt model + stale-book penalty
│   │   ├── orderbook.py             # In-memory orderbook with tick-size enforcement
│   │   └── portfolio.py             # Position tracker, cash, PnL, resolution settlement
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                  # ABC: on_market_event() → list[SignalEvent]
│   │   ├── example_edge.py          # Reference: buy when model_prob > market_price + threshold
│   │   └── example_mm.py            # Reference: two-sided quoting with inventory skew
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── pnl.py                   # PnL decomposition (edge, execution, fee, slippage)
│   │   ├── prediction.py            # Brier score, log-loss, CLV, calibration curve
│   │   ├── risk.py                  # Sharpe, DSR, drawdown, Kelly utilization
│   │   ├── execution.py             # Realized vs expected slippage, fill rate stats
│   │   └── report.py                # ExperimentResult assembly, JSON/Parquet export
│   │
│   └── scripts/
│       ├── ingest.py                # CLI: fetch and store historical data
│       ├── run_backtest.py          # CLI: run experiment from config YAML
│       └── analyze.py               # CLI: generate tearsheet from ExperimentResult
│
├── tests/
│   ├── ...                          # existing tests
│   └── polymarket/
│       ├── test_orderbook.py        # Tick enforcement, level insertion/removal
│       ├── test_fill_model.py       # Book walking, partial fills, FOK/FAK semantics
│       ├── test_fee_model.py        # Fee formula vs known examples
│       ├── test_engine.py           # End-to-end determinism: same input → same output
│       ├── test_portfolio.py        # Resolution settlement, PnL accounting
│       └── test_metrics.py          # Brier score, CLV, DSR calculations
```

---

## Scaffold code

### Core types (`polymarket/core/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto

class Side(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    GTC = auto()
    GTD = auto()
    FOK = auto()
    FAK = auto()

class Resolution(Enum):
    YES = auto()
    NO = auto()
    UNRESOLVED = auto()

@dataclass(frozen=True)
class MarketMeta:
    condition_id: str
    question: str
    category: str
    yes_token_id: str
    no_token_id: str
    tick_size: Decimal
    min_order_size: Decimal
    fee_rate: Decimal
    fee_exponent: Decimal
    maker_rebate_pct: Decimal
    end_date_ts: int          # unix seconds
    resolution: Resolution

@dataclass(frozen=True)
class PriceLevel:
    price: Decimal
    size: Decimal

@dataclass(frozen=True)
class BookSnapshot:
    token_id: str
    timestamp_ns: int
    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

@dataclass(frozen=True)
class Order:
    order_id: str
    token_id: str
    side: Side
    order_type: OrderType
    price: Decimal            # limit price
    size: Decimal             # in shares
    timestamp_ns: int
    expiry_ts: int | None = None  # for GTD

@dataclass(frozen=True)
class Fill:
    order_id: str
    token_id: str
    side: Side
    price: Decimal
    size: Decimal
    fee: Decimal
    is_maker: bool
    timestamp_ns: int

@dataclass
class Position:
    token_id: str
    size: Decimal = Decimal("0")
    avg_entry: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
```

### Fee model (`polymarket/sim/fee_model.py`)

```python
from decimal import Decimal, ROUND_HALF_UP
from polymarket.core.types import MarketMeta

MINIMUM_FEE = Decimal("0.0001")

def compute_taker_fee(
    shares: Decimal,
    price: Decimal,
    meta: MarketMeta,
) -> Decimal:
    """Exact Polymarket dynamic taker fee.

    fee = C × p × feeRate × (p × (1 - p))^exponent
    """
    if meta.fee_rate == 0:
        return Decimal("0")

    p = price
    q = Decimal("1") - price
    base = (p * q) ** meta.fee_exponent
    fee = shares * p * meta.fee_rate * base

    # Round to fee precision (4 decimal places for most categories)
    fee = fee.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return max(fee, MINIMUM_FEE) if fee > 0 else Decimal("0")

def compute_maker_rebate(
    taker_fee: Decimal,
    meta: MarketMeta,
) -> Decimal:
    """Maker rebate as percentage of collected taker fees."""
    return (taker_fee * meta.maker_rebate_pct).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
```

### Orderbook with tick enforcement (`polymarket/sim/orderbook.py`)

```python
from decimal import Decimal
from bisect import insort
from dataclasses import dataclass, field
from polymarket.core.types import PriceLevel, BookSnapshot, Side

@dataclass
class OrderBook:
    token_id: str
    tick_size: Decimal
    bids: list[PriceLevel] = field(default_factory=list)  # descending by price
    asks: list[PriceLevel] = field(default_factory=list)  # ascending by price
    last_update_ns: int = 0

    def validate_price(self, price: Decimal) -> bool:
        """Reject prices not aligned to tick grid."""
        if self.tick_size == 0:
            return True
        remainder = price % self.tick_size
        return remainder == 0 and Decimal("0") < price < Decimal("1")

    def update_from_snapshot(self, snap: BookSnapshot) -> None:
        self.bids = sorted(snap.bids, key=lambda l: l.price, reverse=True)
        self.asks = sorted(snap.asks, key=lambda l: l.price)
        self.last_update_ns = snap.timestamp_ns

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def is_stale(self, current_ns: int, threshold_ns: int) -> bool:
        return (current_ns - self.last_update_ns) > threshold_ns
```

### Fill model (`polymarket/sim/fill_model.py`)

```python
from decimal import Decimal
from polymarket.core.types import Order, Fill, Side, OrderType, MarketMeta
from polymarket.sim.orderbook import OrderBook
from polymarket.sim.fee_model import compute_taker_fee

@dataclass
class FillModelConfig:
    fill_rate: Decimal = Decimal("0.80")      # fraction of resting liquidity available
    stale_threshold_ns: int = 300_000_000_000  # 300 seconds
    stale_spread_mult: Decimal = Decimal("2")
    hard_stale_ns: int = 3_600_000_000_000     # 1 hour — reject non-FOK
    impact_kappa_liquid: Decimal = Decimal("0.005")
    impact_kappa_illiquid: Decimal = Decimal("0.05")
    volume_threshold_liquid: Decimal = Decimal("1000000")

class FillModel:
    def __init__(self, config: FillModelConfig | None = None):
        self.cfg = config or FillModelConfig()

    def try_fill(
        self,
        order: Order,
        book: OrderBook,
        meta: MarketMeta,
        sim_time_ns: int,
        daily_volume: Decimal = Decimal("100000"),
    ) -> list[Fill]:
        # Tick-size validation
        if not book.validate_price(order.price):
            return []  # rejected

        # Min order size validation
        if order.size < meta.min_order_size:
            return []

        # Stale-book check
        if book.is_stale(sim_time_ns, self.cfg.hard_stale_ns):
            if order.order_type not in (OrderType.FOK, OrderType.FAK):
                return []  # reject non-immediate orders on hard-stale books

        fills: list[Fill] = []
        remaining = order.size
        levels = book.asks if order.side == Side.BUY else book.bids

        for level in levels:
            if remaining <= 0:
                break
            if not self._price_crosses(order, level):
                break

            available = level.size * self.cfg.fill_rate
            if book.is_stale(sim_time_ns, self.cfg.stale_threshold_ns):
                available *= (Decimal("1") / self.cfg.stale_spread_mult)

            fill_size = min(remaining, available)
            if fill_size <= 0:
                continue

            fee = compute_taker_fee(fill_size, level.price, meta)
            fills.append(Fill(
                order_id=order.order_id,
                token_id=order.token_id,
                side=order.side,
                price=level.price,
                size=fill_size,
                fee=fee,
                is_maker=False,
                timestamp_ns=sim_time_ns,
            ))
            remaining -= fill_size

        # FOK: all or nothing
        if order.order_type == OrderType.FOK:
            total_filled = sum(f.size for f in fills)
            if total_filled < order.size:
                return []

        return fills

    @staticmethod
    def _price_crosses(order: Order, level: PriceLevel) -> bool:
        if order.side == Side.BUY:
            return order.price >= level.price
        else:
            return order.price <= level.price
```

### Engine (`polymarket/sim/engine.py`)

```python
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import Protocol
from polymarket.core.types import BookSnapshot, Fill, Order, MarketMeta
from polymarket.sim.exchange import SimulatedExchange
from polymarket.sim.portfolio import Portfolio

class EventPriority(IntEnum):
    MARKET = 0
    SIGNAL = 1
    ORDER = 2
    FILL = 3
    RESOLUTION = 4

@dataclass(order=True)
class Event:
    timestamp_ns: int
    priority: EventPriority
    payload: object = field(compare=False)

class Strategy(Protocol):
    def on_market(self, book: BookSnapshot, meta: MarketMeta, sim_time_ns: int) -> list[Order]:
        ...

@dataclass
class BacktestEngine:
    exchange: SimulatedExchange
    portfolio: Portfolio
    strategy: Strategy
    latency_ns: int = 2_000_000_000  # 2 seconds default

    def run(
        self,
        snapshots: list[BookSnapshot],
        market_metas: dict[str, MarketMeta],
    ) -> list[dict]:
        """Deterministic replay of market data through strategy and exchange."""
        equity_curve = []

        # Sort all snapshots chronologically
        snapshots_sorted = sorted(snapshots, key=lambda s: s.timestamp_ns)

        for snap in snapshots_sorted:
            sim_time = snap.timestamp_ns
            meta = market_metas.get(self._token_to_condition(snap.token_id))
            if meta is None:
                continue

            # Update exchange book state
            self.exchange.update_book(snap)

            # Strategy generates orders
            orders = self.strategy.on_market(snap, meta, sim_time)

            # Apply latency: orders execute at sim_time + latency
            exec_time = sim_time + self.latency_ns

            for order in orders:
                fills = self.exchange.process_order(order, meta, exec_time)
                for fill in fills:
                    self.portfolio.on_fill(fill, meta)

            # Record equity
            equity_curve.append({
                "timestamp_ns": sim_time,
                "equity": float(self.portfolio.total_equity(
                    self.exchange.get_mids()
                )),
            })

        # Settle all resolved markets
        for cid, meta in market_metas.items():
            if meta.resolution != Resolution.UNRESOLVED:
                self.portfolio.settle(meta)

        return equity_curve

    def _token_to_condition(self, token_id: str) -> str:
        return self.exchange.token_to_condition.get(token_id, "")
```

### Key test scaffolds (`tests/polymarket/test_fee_model.py`)

```python
from decimal import Decimal
import pytest
from polymarket.sim.fee_model import compute_taker_fee
from polymarket.core.types import MarketMeta, Resolution

def _make_meta(fee_rate: str, exponent: str, rebate: str = "0.25") -> MarketMeta:
    return MarketMeta(
        condition_id="test", question="test", category="crypto",
        yes_token_id="y", no_token_id="n",
        tick_size=Decimal("0.01"), min_order_size=Decimal("1"),
        fee_rate=Decimal(fee_rate), fee_exponent=Decimal(exponent),
        maker_rebate_pct=Decimal(rebate),
        end_date_ts=9999999999, resolution=Resolution.UNRESOLVED,
    )

class TestTakerFee:
    def test_crypto_at_midpoint(self):
        """100 shares of Crypto at $0.50 → $0.90 fee (1.80% effective)."""
        meta = _make_meta("0.072", "1")
        fee = compute_taker_fee(Decimal("100"), Decimal("0.50"), meta)
        assert fee == Decimal("0.9000")

    def test_crypto_at_extreme(self):
        """100 shares of Crypto at $0.10 → lower fee due to (p*(1-p))^1."""
        meta = _make_meta("0.072", "1")
        fee = compute_taker_fee(Decimal("100"), Decimal("0.10"), meta)
        assert fee == Decimal("0.0648")

    def test_geopolitics_zero_fee(self):
        """Geopolitics markets are fee-free."""
        meta = _make_meta("0", "1")
        fee = compute_taker_fee(Decimal("1000"), Decimal("0.50"), meta)
        assert fee == Decimal("0")

    def test_sports_fee(self):
        """Sports: fee_rate=0.03, exponent=1, peak 0.75%."""
        meta = _make_meta("0.03", "1", "0.25")
        fee = compute_taker_fee(Decimal("100"), Decimal("0.50"), meta)
        assert fee == Decimal("0.3750")

class TestTickEnforcement:
    def test_valid_tick(self):
        from polymarket.sim.orderbook import OrderBook
        book = OrderBook(token_id="t", tick_size=Decimal("0.01"))
        assert book.validate_price(Decimal("0.55")) is True
        assert book.validate_price(Decimal("0.555")) is False
        assert book.validate_price(Decimal("0")) is False
        assert book.validate_price(Decimal("1")) is False

class TestDeterminism:
    def test_identical_runs(self):
        """Two runs with identical inputs produce identical outputs."""
        # Setup identical engine, data, strategy
        # Run twice, assert equity curves are byte-identical
        ...
```

---

## Critical design decisions and their rationale

### Why not use NautilusTrader directly?

NautilusTrader has a Polymarket adapter and is production-grade, but its Rust core creates a steep integration barrier for a Python-first quant repo. The `hftbacktest` library offers superior L2/L3 replay fidelity but is similarly Rust-heavy. **Building a lightweight Python-native simulator** keeps the codebase uniform with `llm-quant`, allows direct manipulation of prediction-market-specific logic (binary resolution, CTF invariants, category-dependent fees), and produces a system where every line is auditable. For strategies that graduate to production, the path is: validate in this simulator → deploy via `py-clob-client` using the same `Strategy` interface.

### The historical orderbook gap and how to close it

Polymarket's API provides no historical orderbook snapshots — only current state. This is the single largest constraint on backtest fidelity. Three mitigation strategies, in order of preference: (1) **pmxt hourly archive** — free, available now, sufficient for strategies operating on hourly+ timeframes; (2) **forward-looking capture** — deploy a collector polling `GET /book` every 10 seconds (within the 300 req/10s rate limit) for target markets, storing to Parquet; (3) **synthetic book reconstruction** — use on-chain trade data to infer approximate book depth via the spread model. The framework should support all three via the `DataReplay` interface, with an explicit `data_quality` tag on each `BookSnapshot` indicating its provenance.

### The YES/NO invariant as a free consistency check

Since **P(YES) + P(NO) = $1.00** is enforced by the CTF, the simulator can cross-validate orderbook data by checking that the YES best-ask and NO best-bid sum to ≤ $1.00 (violations indicate arbitrage or stale data). This invariant check should run on every `BookSnapshot` ingestion, logging warnings for violations exceeding the tick size. The strategy layer can exploit persistent violations as a signal (complementary token arbitrage).

### Fee model must be dynamic, not static

Polymarket's fee schedule changed in March 2026 and varies by category with different rate/exponent pairs. The simulator must load fee parameters per market from `MarketMeta` — never hardcode. The `fee_rate_bps` field from the CLOB API response provides the ground truth. For historical backtests spanning fee regime changes, the `MarketMeta` should carry a `fee_schedule: list[FeeRegime]` with effective dates.

---

## Conclusion

This framework makes three bets that differ from generic backtesting approaches. First, it treats the **orderbook as a first-class replay artifact** rather than relying on OHLCV bars — because in thin prediction markets, the difference between midpoint-based simulation and level-by-level book walking can exceed the strategy's entire edge. Second, it decomposes PnL into **edge vs. execution vs. fees vs. slippage** — because prediction market edges are typically small (2–5 cents) and easily consumed by execution costs that conventional backtests ignore. Third, it uses **prediction-market-native metrics** (Brier score, CLV, calibration-adjusted hit rate) alongside financial metrics (Sharpe, DSR) — because a strategy that is well-calibrated but poorly-timed will show profit in a naive backtest but bleed in production.

The minimum viable implementation requires roughly 2,000 lines of Python across the `sim/` and `metrics/` modules, plus ~500 lines for data ingestion. The `poly_data` pipeline and pmxt archive provide immediate access to historical trade and orderbook data without building collection infrastructure from scratch. The entire system is deterministic by construction, auditable line-by-line, and designed so that the `Strategy` interface is identical in backtest and live execution via `py-clob-client`.
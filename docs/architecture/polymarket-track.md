# ADR: Polymarket Research Module Integration

**Date:** 2026-04-06
**Status:** PROPOSED
**Authors:** Architecture Agent
**Tracking:** llm-quant-ob4b

---

## 1. Summary

This ADR proposes the architecture for integrating a Polymarket-specific quantitative
research module into the existing llm-quant codebase. The module targets read-only
data collection, hypothesis-driven research, historical backtesting, and paper trading
simulation for prediction market strategies. Live trading is explicitly out of scope
due to Australian regulatory constraints (ACMA geoblock, ASIC binary options ban).

The core decision: **extend the existing `src/llm_quant/arb/` module rather than
creating a parallel `src/llm_quant/polymarket/` package**, while adding a dedicated
research and backtest layer that the current arb module lacks.

---

## 2. Context and Drivers

### 2.1 Current State of Track C / arb Module

The existing `src/llm_quant/arb/` module was built during the Track C (Structural
Arbitrage) sprint in late March 2026. It contains 12 Python files across three
strategy families:

**Prediction market arb (Polymarket + Kalshi):**

| File | Purpose | Lines |
|------|---------|-------|
| `gamma_client.py` | Polymarket CLOB/Gamma API client with US API fallback | ~420 |
| `kalshi_client.py` | Kalshi public REST API client | ~360 |
| `scanner.py` | NegRisk + single-rebalance arb detection for Polymarket | ~700 |
| `detector.py` | Claude-based combinatorial arb detector (Appendix B prompt) | ~400 |
| `kalshi_detector.py` | Cross-event dependency detector for Kalshi | ~490 |
| `execution.py` | Kalshi paper execution engine with Kelly sizing | ~440 |
| `paper_gate.py` | 30-day paper validation (4 gates: persistence, fill, capacity, days) | ~410 |
| `schema.py` | DuckDB DDL for 8 tables (markets, conditions, opportunities, etc.) | ~180 |

**CEF discount strategy:**

| File | Purpose |
|------|---------|
| `cef_data.py` | CEF price fetching + NAV estimation from benchmark ETFs |
| `cef_strategy.py` | Discount mean-reversion strategy (standalone + STRATEGY_REGISTRY) |

**Crypto funding rates:**

| File | Purpose |
|------|---------|
| `funding_rates.py` | CCXT-based funding rate collector (Binance, OKX, Bybit) |
| `funding_scanner.py` | High-rate + cross-exchange differential scanner |

**Test coverage:** 6 test files exist (`test_pm_arb.py`, `test_arb_execution.py`,
`test_arb_paper_gate.py`, `test_cef_strategy.py`, `test_funding_rates.py`,
`test_track_c_robustness.py`), all at the repo root `tests/` directory.

**Governance status:** The 2026-03-30 mandate audit found 4 of 17 Track C gates
fully implemented, with 11 not implemented. Key gaps: no Sharpe/MaxDD/beta gates,
no kill switches, no central risk manager integration. The mandate YAML file itself
was never committed to git.

### 2.2 What the Polymarket Research Reports Reveal

Three research reports (`docs/research/polymarket/`) provide detailed technical
reference. Key findings relevant to architecture:

1. **Three-layer API architecture:** Gamma API (market discovery, public, 4K req/10s),
   CLOB API (orderbook + trading, public reads at 9K req/10s), Data API (positions,
   leaderboards, 1K req/10s). Plus WebSocket feeds (known 20-min disconnect bug).

2. **Identifier hierarchy is the primary developer hazard:** `conditionId` (hex) vs
   `clobTokenIds` (large integer) vs `questionId`. The existing `gamma_client.py`
   partially handles this but does not store `clobTokenIds` or `questionId` -- only
   `conditionId` and `market_id`.

3. **Historical data sources exist beyond the live API:** pmxt Data Archive (hourly
   orderbook Parquet), PolyBackTest (sub-second orderbook), CLOB `/prices-history`
   endpoint, Polygon on-chain subgraph. The current `gamma_client.py` only uses the
   live CLOB/Gamma endpoints -- no historical data pipeline exists.

4. **Academic alpha patterns documented:** Yes/default bias (6+ pp overpricing of YES),
   mean reversion after overreaction (58% negative serial correlation), late-stage
   inefficiency, favorite-longshot bias (FLB), cross-platform arb. The current
   scanner only targets NegRisk + rebalancing arb, not these behavioral patterns.

5. **Fee structure is category-dependent and dynamic:** Ranges from 0% (geopolitics)
   to 1.8% peak (crypto). The current scanner hardcodes `POLYMARKET_WIN_FEE = 0.02`
   -- this is incorrect for most market categories.

6. **AU legal constraint is binding:** Read-only data collection and paper trading
   are permissible. Live order placement is prohibited. This constraint is
   architecture-defining: the entire module operates in research/simulation mode.

### 2.3 Architectural Drivers

| Driver | Weight | Implication |
|--------|--------|-------------|
| Read-only constraint (AU legal) | Hard | No execution module needed. Paper trading only. |
| Existing arb module overlap | High | Polymarket client, scanner, schema already exist in `arb/`. |
| Research lifecycle compliance | High | Must produce mandate, hypothesis, backtest, robustness artifacts. |
| DuckDB + Polars stack | High | All data storage and analysis through existing infrastructure. |
| Solo developer | High | Minimize new module surface area. Prefer extension over parallel structure. |
| Track C governance gaps | Medium | 11 of 17 gates missing. New code should not add more ungated surface. |
| Historical data needs | Medium | Current code is live-scan only. Research requires historical replay. |

---

## 3. Decision: Extend arb/ With a Polymarket Research Sub-Package

### 3.1 Options Considered

**Option A: Standalone `src/llm_quant/polymarket/` package**

A new top-level package parallel to `arb/`, containing its own data adapters,
research module, backtest engine, and paper trading infrastructure.

- Pro: Clean separation, no risk of breaking existing arb code.
- Con: Duplicates data models (`Market`, `ConditionPrice`), client code
  (`GammaClient`), schema DDL, and DuckDB connection management. The existing
  `gamma_client.py` IS the Polymarket data adapter -- a new package would either
  import from `arb.gamma_client` (creating a cross-package dependency that defeats
  the purpose of separation) or duplicate 400+ lines of working, tested code.
- Con: Creates a second arb-like module that the solo developer must maintain in
  parallel. The CEF and funding rate code would remain orphaned in `arb/` while
  Polymarket logic splits across two locations.

**Option B: Extend `arb/` with Polymarket-specific sub-modules**

Add new files to `src/llm_quant/arb/` for research, historical data, and paper
simulation. Refactor the existing `gamma_client.py` to support both live scanning
and historical data access.

- Pro: Reuses existing data models, client, schema, scanner, detector. No duplication.
- Pro: Keeps all Track C code in one place. CEF, funding rate, and PM arb share
  the same governance gates, risk framework, and DuckDB database.
- Con: `arb/` grows from 12 files to ~18 files. May benefit from an internal
  sub-package (`arb/polymarket/`) if the file count becomes unwieldy.
- Con: Risk of coupling Polymarket-specific research logic to the generic arb
  scanner. Mitigated by clean file boundaries and no circular imports.

**Option C: Sub-package within arb (`arb/polymarket/`)**

Create `src/llm_quant/arb/polymarket/` as a Python sub-package within `arb/`,
containing research, historical data, and backtest modules. Generic arb
infrastructure (scanner, execution, paper_gate) stays at the `arb/` level.

- Pro: Logical grouping without duplication. Polymarket-specific files are namespaced.
- Pro: Common infrastructure (schema, scanner, execution) remains shared.
- Con: Adds a package nesting level. Imports become `from llm_quant.arb.polymarket.research import ...`.
- Con: The existing `gamma_client.py` and `detector.py` are already Polymarket-specific
  but live at the `arb/` level. Moving them into the sub-package would break all
  existing imports and tests. Leaving them creates an inconsistent split.

### 3.2 Decision

**Option B: Extend `arb/` with new files, no sub-package.**

Rationale:

1. The existing `gamma_client.py` (420 lines, tested) already IS the Polymarket data
   adapter. Creating a second adapter in a separate package serves no purpose.

2. The scanner, detector, schema, and execution modules already handle Polymarket data
   end-to-end. What is missing is historical data access, research tooling, and a
   structured backtest framework for behavioral strategies. These are additive --
   they do not require restructuring existing code.

3. The solo developer constraint is decisive. Maintaining two parallel packages with
   overlapping data models is a structural tax that produces no benefit when there is
   one maintainer. The cognitive overhead of "which package owns Market?" is waste.

4. The file count concern (growing from 12 to ~18) is manageable. If the module later
   exceeds ~25 files, refactoring into sub-packages becomes justified by the weight
   of evidence rather than speculative separation.

### 3.3 When This Decision Should Be Revisited

- If a second developer joins and needs ownership boundaries between PM arb and
  other Track C strategies.
- If the arb module exceeds 25 files or 8,000 lines of production code.
- If Polymarket research requires a fundamentally different data model that cannot
  extend the existing `Market` / `ConditionPrice` dataclasses.

---

## 4. Proposed Directory Structure

### 4.1 New Source Files

```
src/llm_quant/arb/
    __init__.py                    # existing
    gamma_client.py                # existing — extend with historical data methods
    kalshi_client.py               # existing — no changes
    scanner.py                     # existing — refactor fee to be category-aware
    detector.py                    # existing — no changes
    kalshi_detector.py             # existing — no changes
    execution.py                   # existing — no changes
    paper_gate.py                  # existing — no changes
    schema.py                      # existing — extend with research tables
    cef_data.py                    # existing — no changes
    cef_strategy.py                # existing — no changes
    funding_rates.py               # existing — no changes
    funding_scanner.py             # existing — no changes
    pm_history.py                  # NEW: historical price data pipeline
    pm_research.py                 # NEW: hypothesis testing + signal generation
    pm_backtest.py                 # NEW: backtest engine for PM strategies
    pm_paper.py                    # NEW: paper trading simulator
    pm_fees.py                     # NEW: category-aware fee model
```

### 4.2 New Test Files

```
tests/
    test_pm_arb.py                 # existing
    test_arb_execution.py          # existing
    test_arb_paper_gate.py         # existing
    test_pm_history.py             # NEW: historical data pipeline tests
    test_pm_research.py            # NEW: hypothesis testing tests
    test_pm_backtest.py            # NEW: backtest engine tests
    test_pm_paper.py               # NEW: paper trading simulator tests
    test_pm_fees.py                # NEW: fee model tests
```

### 4.3 New Documentation

```
docs/
    architecture/
        polymarket-track.md        # THIS FILE — ADR
    research/
        polymarket/
            polymarket-research-landscape.md        # existing
            polymarket-trading-infrastructure.md    # existing
            polymarket-research-module-design.md    # existing
```

### 4.4 New Configuration

```
config/
    strategies/
        pm-yes-bias-contrarian.yaml    # research spec for Yes/default bias strategy
        pm-mean-reversion.yaml         # research spec for overreaction MR strategy
        pm-flb-longshot.yaml           # research spec for favorite-longshot bias
        pm-cross-platform-arb.yaml     # research spec for PM-Kalshi cross-platform
```

### 4.5 Strategy Artifacts (Research Lifecycle)

```
data/strategies/
    pm-yes-bias-contrarian/
        mandate.yaml
        hypothesis.yaml
        data-contract.yaml
        research-spec.yaml
        backtest-result.yaml
        robustness-result.yaml
    pm-mean-reversion/
        ...
```

---

## 5. Module Design

### 5.1 pm_fees.py -- Category-Aware Fee Model

**Problem:** The current scanner hardcodes `POLYMARKET_WIN_FEE = 0.02`. The actual
fee structure is:
- Category-dependent (`feeRateBps` varies: 0.03 for sports, 0.04 for politics/finance,
  0.072 for crypto, 0 for geopolitics)
- Nonlinear: `fee = C * p * feeRate * (p * (1-p))^exponent`
- Dynamic: must be fetched per market from the CLOB API

**Design:**

```python
@dataclass
class PolymarketFeeSchedule:
    """Category-specific fee parameters as of 2026-03-30."""
    category: str
    fee_rate: float       # feeRateBps (e.g., 0.03 for sports)
    exponent: float       # 1.0 for most, 0.5 for economics
    maker_rebate: float   # fraction of taker fees returned to makers

def compute_taker_fee(price: float, size: float, schedule: PolymarketFeeSchedule) -> float:
    """Compute taker fee for a given price and size."""

def compute_net_spread(gross_spread: float, category: str) -> float:
    """Compute net arb spread after category-specific fees."""

# Hardcoded schedules (updated from research reports, fetched dynamically when possible)
FEE_SCHEDULES: dict[str, PolymarketFeeSchedule] = {
    "geopolitics": PolymarketFeeSchedule("geopolitics", 0.0, 1.0, 0.0),
    "sports": PolymarketFeeSchedule("sports", 0.03, 1.0, 0.25),
    "finance": PolymarketFeeSchedule("finance", 0.04, 1.0, 0.50),
    "politics": PolymarketFeeSchedule("politics", 0.04, 1.0, 0.25),
    "crypto": PolymarketFeeSchedule("crypto", 0.072, 1.0, 0.20),
}
```

**Impact on existing code:** `scanner.py` should import `compute_net_spread` from
`pm_fees.py` instead of using the hardcoded constant. This is a backward-compatible
change -- the scanner's public API does not change.

### 5.2 pm_history.py -- Historical Data Pipeline

**Problem:** The current `gamma_client.py` only fetches live market snapshots. Research
and backtesting require historical price time series.

**Data sources (ranked by utility for research):**

1. **CLOB API `/prices-history`** -- per-token price time series. Public, rate-limited.
   Best for individual market analysis. Known limitation: resolved markets may only
   return 12h+ granularity.
2. **pmxt Data Archive (archive.pmxt.dev)** -- hourly orderbook + trade snapshots in
   Parquet. Free, comprehensive. Best for bulk historical analysis.
3. **PolyBackTest (polybacktest.com)** -- sub-second to 1-minute orderbook. Freemium.
   Best for microstructure research.
4. **The Graph subgraph** -- on-chain settlement events. 100K queries/month free.
   Best for resolution timing and on-chain flow analysis.

**Design:**

```python
class PolymarketHistoryClient:
    """Fetches historical price data from CLOB API /prices-history endpoint."""

    def fetch_price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        interval: str = "1h",
        fidelity: int = 60,
    ) -> pl.DataFrame:
        """Returns DataFrame with columns: timestamp, price."""

    def fetch_market_history(
        self,
        condition_id: str,
        start_ts: int,
        end_ts: int,
    ) -> pl.DataFrame:
        """Fetch YES and NO price history for a market, joined by timestamp."""


class PmxtArchiveLoader:
    """Loads historical data from pmxt Parquet archive files."""

    def load_orderbook_snapshots(
        self, parquet_path: Path
    ) -> pl.DataFrame:
        """Load orderbook snapshots from local Parquet files."""

    def load_trades(
        self, parquet_path: Path
    ) -> pl.DataFrame:
        """Load historical trades from local Parquet files."""
```

**DuckDB schema extension (in `schema.py`):**

```sql
CREATE TABLE IF NOT EXISTS pm_price_history (
    token_id      VARCHAR NOT NULL,
    condition_id  VARCHAR NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL,
    price         DOUBLE NOT NULL,
    side          VARCHAR NOT NULL,    -- 'yes' | 'no'
    PRIMARY KEY (token_id, timestamp, side)
);

CREATE TABLE IF NOT EXISTS pm_trades_history (
    trade_id      VARCHAR PRIMARY KEY,
    condition_id  VARCHAR NOT NULL,
    token_id      VARCHAR NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL,
    side          VARCHAR NOT NULL,    -- 'buy' | 'sell'
    price         DOUBLE NOT NULL,
    size          DOUBLE NOT NULL
);
```

### 5.3 pm_research.py -- Hypothesis Testing and Signal Generation

**Problem:** The current arb module only detects structural arb (NegRisk complement,
rebalancing, combinatorial logic). The research reports identify behavioral alpha
patterns that require a different signal generation framework: time-series mean
reversion, cross-sectional bias exploitation, and event-driven overreaction capture.

**Design principles:**

1. Each strategy is a testable hypothesis with a declarative prediction, expected
   outcome, and measurement method -- matching the project's research lifecycle.
2. Signals are pure functions of observable data (prices, volume, resolution dates).
   No look-ahead.
3. Each strategy class produces a `PmResearchSignal` that feeds into the backtest
   engine.

**Proposed strategy classes:**

```python
@dataclass
class PmResearchSignal:
    """A single prediction market research signal."""
    condition_id: str
    question: str
    category: str
    signal_type: str          # 'yes_bias' | 'mean_reversion' | 'flb' | 'cross_platform'
    direction: str            # 'buy_yes' | 'buy_no' | 'sell_yes' | 'sell_no'
    price_at_signal: float
    estimated_edge: float     # expected profit after fees
    confidence: float         # 0-1
    reasoning: str
    timestamp: datetime

class YesBiasContrarian:
    """Exploits systematic overpricing of YES outcomes.

    Hypothesis: YES outcomes are overpriced by ~6pp on average (Reichenbach &
    Walther 2025). Contrarian NO positions on markets where YES price exceeds
    a model-estimated fair probability should generate positive expected value.

    Signal: Buy NO when YES price > fair_probability + bias_threshold.
    Fair probability estimated from: category base rates, resolution history,
    cross-platform consensus.
    """

class OverreactionMeanReversion:
    """Captures short-term overreaction to price movements.

    Hypothesis: 58% of Polymarket national markets show negative serial
    correlation (Clinton & Huang 2025). Large single-day price moves
    (> 2 sigma) tend to partially reverse within 72 hours.

    Signal: Contrarian position after price move > threshold, hold for
    fixed window (1-3 days), exit at target or timeout.
    """

class FavoriteLongshotBias:
    """Exploits mispricing at probability extremes.

    Hypothesis: Contracts below 20 cents underperform implied odds; contracts
    above 80 cents outperform (Becker 2026). Systematic selling of longshots
    (YES at < 0.20) and buying near-certainties (YES at > 0.80) generates
    positive expected value.

    Signal: Sell YES on markets where price < 0.20; buy YES where price > 0.80.
    Filter by volume and days-to-resolution.
    """

class CrossPlatformArb:
    """Captures price divergences between Polymarket and Kalshi.

    Hypothesis: Prices for identical contracts diverge across platforms by
    3-8 cents during high-activity periods (Clinton & Huang 2025). The lead-lag
    relationship (Polymarket leads in price discovery) creates directional signals.

    Signal: When Polymarket price leads Kalshi by > threshold, take the
    Polymarket-indicated direction on Kalshi (or vice versa for convergence).
    """
```

### 5.4 pm_backtest.py -- Prediction Market Backtest Engine

**Problem:** The existing backtest engine (`src/llm_quant/backtest/engine.py`) is
designed for equity-like instruments with continuous prices. Prediction markets have
fundamentally different properties:

- Bounded payoff: prices in [0, 1], payout is binary ($0 or $1).
- Finite horizon: every market has a resolution date.
- Nonlinear fees: category-dependent, probability-dependent.
- Position resolution: positions do not need to be sold -- they settle automatically.

The existing `Strategy` ABC signature (`generate_signals(as_of_date, indicators_df,
portfolio, prices)`) assumes a universe of continuously tradeable assets with OHLCV
data. Prediction markets have a rotating universe where markets are created and
resolved daily.

**Design:**

The PM backtest engine should be a separate class that does NOT inherit from the
equity backtest engine. The instruments, payoff structure, and position lifecycle
are too different to share an engine. However, the output format (daily returns
series, performance metrics) should be compatible with the existing `BacktestMetrics`
dataclass so that robustness analysis (DSR, CPCV) can be applied uniformly.

```python
@dataclass
class PmBacktestConfig:
    """Configuration for prediction market backtest."""
    initial_capital: float = 100_000.0
    max_position_usd: float = 2_000.0
    max_concurrent_positions: int = 20
    fee_model: str = "category_aware"     # 'flat_2pct' | 'category_aware'
    kelly_fraction: float = 0.25          # fractional Kelly
    min_volume_24h: float = 1_000.0
    min_days_to_resolution: int = 2

@dataclass
class PmBacktestResult:
    """Result of a prediction market backtest."""
    daily_returns: pl.DataFrame           # date, return columns
    trades: list[PmPaperTrade]
    metrics: BacktestMetrics              # reuse existing metrics dataclass
    config: PmBacktestConfig

class PmBacktestEngine:
    """Backtest engine for prediction market strategies.

    Unlike the equity engine, this handles:
    - Binary payoff at resolution (not continuous P&L)
    - Rotating universe (markets created and resolved over time)
    - Fractional Kelly sizing for binary contracts
    - Category-aware fee deduction
    """

    def run(
        self,
        strategy: PmResearchStrategy,
        price_history: pl.DataFrame,
        resolution_data: pl.DataFrame,
        config: PmBacktestConfig,
    ) -> PmBacktestResult:
        """Run backtest over historical prediction market data."""
```

**Key design choice:** The PM backtest engine outputs `BacktestMetrics` (the same
dataclass used by the equity backtest engine). This means the existing robustness
pipeline (`src/llm_quant/backtest/robustness.py`) can compute DSR, CPCV, and
walk-forward metrics on PM strategy returns without modification. The existing
`run_track_c_robustness.py` script would route PM research strategies through
the same gate framework.

### 5.5 pm_paper.py -- Paper Trading Simulator

**Problem:** The existing `execution.py` is Kalshi-specific (uses `KalshiEvent`
dataclass, hardcodes Kalshi 3% fee). A Polymarket paper trading simulator needs to:

1. Consume signals from `pm_research.py` strategy classes.
2. Apply category-aware fees from `pm_fees.py`.
3. Track open positions through to market resolution.
4. Compute mark-to-market P&L daily (using live or historical prices).
5. Produce a track record compatible with the existing paper gate framework.

**Design:**

```python
@dataclass
class PmPaperTrade:
    """A single paper trade on Polymarket."""
    trade_id: str
    condition_id: str
    question: str
    category: str
    direction: str            # 'buy_yes' | 'buy_no'
    entry_price: float
    entry_size_usd: float
    entry_fee: float
    entry_timestamp: datetime
    exit_price: float | None
    exit_timestamp: datetime | None
    resolution_outcome: str | None   # 'yes' | 'no' | None (still open)
    pnl: float | None
    status: str               # 'open' | 'resolved' | 'exited'

class PmPaperTrader:
    """Paper trading engine for Polymarket research strategies.

    Consumes PmResearchSignal objects, sizes positions via fractional Kelly,
    applies category-aware fees, and tracks positions to resolution.
    """

    def __init__(self, db_path: Path, nav_usd: float = 100_000.0): ...

    def execute_signal(self, signal: PmResearchSignal) -> PmPaperTrade | None:
        """Evaluate a signal, apply pre-trade checks, record paper trade."""

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Update open positions with current prices. Returns total P&L."""

    def resolve_market(self, condition_id: str, outcome: str) -> float:
        """Resolve an open position. Returns realized P&L."""

    def get_track_record(self) -> pl.DataFrame:
        """Export trade history for paper gate evaluation."""
```

**DuckDB schema extension:**

```sql
CREATE TABLE IF NOT EXISTS pm_paper_trades (
    trade_id        VARCHAR PRIMARY KEY,
    condition_id    VARCHAR NOT NULL,
    question        VARCHAR,
    category        VARCHAR,
    direction       VARCHAR NOT NULL,
    entry_price     DOUBLE NOT NULL,
    entry_size_usd  DOUBLE NOT NULL,
    entry_fee       DOUBLE NOT NULL,
    entry_timestamp TIMESTAMPTZ NOT NULL,
    exit_price      DOUBLE,
    exit_timestamp  TIMESTAMPTZ,
    resolution_outcome VARCHAR,
    pnl             DOUBLE,
    status          VARCHAR NOT NULL DEFAULT 'open'
);
```

### 5.6 Existing Module Changes

**`gamma_client.py` -- extend, do not replace:**

1. Store `clobTokenIds` in the `Market` dataclass (currently missing). This is
   required for CLOB API `/prices-history` calls.
2. Add a `fetch_resolved_markets()` method for historical research (currently only
   fetches active markets).
3. Add `questionId` storage for UMA resolution tracking.

**`scanner.py` -- replace hardcoded fee with `pm_fees` import:**

Replace `POLYMARKET_WIN_FEE = 0.02` with a call to `pm_fees.compute_taker_fee()`.
The scanner's public API (`run_scan()`, `run_negrisk_scan()`) does not change.

**`schema.py` -- add new table DDL:**

Add `pm_price_history`, `pm_trades_history`, and `pm_paper_trades` table definitions
to the existing `_ALL_DDL` list.

---

## 6. Integration Points With Existing Infrastructure

### 6.1 Research Lifecycle

PM research strategies follow the same lifecycle as Track A/B strategies:

```
/mandate -> /hypothesis -> /data-contract -> /research-spec ->
/research-spec freeze -> /backtest -> /robustness -> /paper -> /promote
```

The difference is in the backtest and robustness steps: PM strategies use
`PmBacktestEngine` instead of the equity engine, but output the same
`BacktestMetrics` object for robustness analysis.

### 6.2 Robustness Pipeline

The existing `scripts/run_track_c_robustness.py` already routes Track C strategies
to a PM-arb-specific gate path. PM research strategies should follow the same
routing pattern:

- Structural arb (NegRisk, rebalancing): Use existing `PaperArbGate` (persistence,
  fill rate, capacity, days elapsed).
- Behavioral strategies (Yes bias, MR, FLB): Use equity-style robustness gates
  (Sharpe, MaxDD, DSR, CPCV) applied to `PmBacktestResult.metrics`.

This is architecturally clean: structural arb is deterministic (no Sharpe needed --
the profit is deductive), while behavioral strategies are statistical (Sharpe/DSR
gates are appropriate).

### 6.3 Surveillance and Risk Manager

Track C governance gaps (11 of 17 gates missing) are a known debt tracked in beads
issues. The Polymarket research module should NOT attempt to close these gaps -- it
is a research module, not a production trading system. The existing kill switch
gaps apply to all Track C strategies equally and should be addressed in dedicated
implementation issues.

The paper trading simulator (`pm_paper.py`) should self-enforce:
- Max $2,000 per market position (matching existing `execution.py` cap).
- Max 20 concurrent positions.
- Position-level stop loss (exit if price moves against by > X%).

These are local to the paper trader and do not require central risk manager
integration (which is itself unbuilt for Track C).

### 6.4 DuckDB Database

All new tables live in the same DuckDB database (`data/quant.db`) as the existing
arb tables. The `schema.py` module manages all DDL. No separate database.

---

## 7. Data Adapter Design

### 7.1 Live Data (CLOB API + Gamma API)

The existing `GammaClient` class handles live market discovery and price fetching.
Proposed changes:

| Change | Rationale |
|--------|-----------|
| Add `clob_token_ids: list[str]` to `Market` dataclass | Required for CLOB API `/prices-history` and WebSocket subscriptions |
| Add `question_id: str` to `Market` dataclass | Required for resolution tracking via UMA |
| Add `fetch_resolved_markets()` method | Historical research needs resolved market data |
| Add `fetch_market_by_slug(slug: str)` method | More ergonomic market lookup for research |

### 7.2 Historical Data (CLOB /prices-history)

New `PolymarketHistoryClient` class wraps the CLOB API `/prices-history` endpoint.
This is the primary historical data source for individual market analysis.

Rate limit: The `/prices-history` endpoint has no documented per-endpoint limit
beyond the general 9K/10s CLOB limit, but practical throughput for historical
backfill is constrained by the one-market-at-a-time query model.

### 7.3 Bulk Historical Data (pmxt Archive)

The pmxt Data Archive provides hourly orderbook and trade snapshots in Parquet
format. This is the most efficient source for bulk historical analysis but requires
downloading archive files and loading them locally.

`PmxtArchiveLoader` is a stateless reader class. It does not manage downloads --
the researcher downloads Parquet files manually and points the loader at the
local directory.

### 7.4 WebSocket (Deferred)

WebSocket integration is deferred. The known 20-minute disconnect bug, the
reconnection complexity, and the read-only constraint make WebSocket lower priority
than REST-based historical data. If real-time price monitoring becomes needed for
paper trading, the WebSocket client should be added as a separate module
(`pm_websocket.py`).

---

## 8. Testing Strategy

### 8.1 Unit Tests (All New Modules)

Each new module gets a corresponding test file in `tests/`:

| Module | Test File | Key Test Cases |
|--------|-----------|----------------|
| `pm_fees.py` | `test_pm_fees.py` | Fee computation at 50% price (peak), at extremes (near 0/1), per category, edge cases (geopolitics = free) |
| `pm_history.py` | `test_pm_history.py` | Parse `/prices-history` response, handle empty/malformed data, DuckDB persistence round-trip |
| `pm_research.py` | `test_pm_research.py` | Each strategy class produces valid signals, no look-ahead in signal generation, edge cases (no data, single market) |
| `pm_backtest.py` | `test_pm_backtest.py` | Binary payoff correctness, fee deduction, Kelly sizing, position resolution, metrics output compatibility |
| `pm_paper.py` | `test_pm_paper.py` | Trade lifecycle (open -> resolve), mark-to-market, DuckDB persistence, position limits enforced |

### 8.2 Test Data Strategy

All tests use synthetic/fixture data. No live API calls in tests. The existing
test pattern (`test_pm_arb.py` uses `@pytest.fixture` with hand-constructed
`Market` and `ConditionPrice` objects) should be followed for consistency.

For backtest tests, construct a minimal historical dataset (10 markets, 90 days
of daily prices, known resolution outcomes) that exercises the full engine path.

### 8.3 Integration Tests

One integration test script (`scripts/test_pm_research_pipeline.py`) should run the
full pipeline: load fixture data -> generate signals -> run backtest -> compute
metrics -> verify output format matches equity metrics dataclass. This validates
that the PM backtest output is compatible with the robustness pipeline.

### 8.4 No Network Tests

All tests must pass without network access. Mock all HTTP calls. This is consistent
with the existing test approach (no test in the repo makes live API calls).

---

## 9. Key Constraints

### 9.1 Australian Legal Constraint (Architecture-Defining)

- **Permissible:** Accessing all read-only API endpoints, storing and analyzing
  historical data, building paper trading simulations, backtesting strategies,
  publishing research.
- **Prohibited:** Placing orders, using VPNs to bypass geoblocking, hosting
  trading bots that place live orders.
- **Architectural consequence:** No `OrderClient`, no `LiveExecution` module, no
  authenticated CLOB endpoints. The entire module operates in simulation mode.
  The `pm_paper.py` module tracks hypothetical positions -- it never touches the
  CLOB order placement API.

### 9.2 Fee Model Staleness

The fee schedule changes frequently (zero fees pre-2025, crypto-only Jan 2025,
category-wide March 2025, updated March 2026). The `pm_fees.py` module hardcodes
current fee schedules as a fallback but should attempt to fetch `feeRateBps` from
the CLOB API per market when available. Historical backtests must use the fee
schedule in effect at the time of the simulated trade -- this requires either a
fee schedule changelog or conservative assumptions (use highest observed fee rate).

### 9.3 Historical Data Completeness

The CLOB `/prices-history` endpoint has known limitations for resolved markets
(reduced granularity). The pmxt archive provides better coverage but requires
manual download. The backtest engine should handle missing data gracefully (skip
markets with insufficient history, log gaps).

### 9.4 Identifier Mapping

The `conditionId` (hex) vs `clobTokenIds` (large integer) confusion is the #1
developer hazard documented in the research reports. The `Market` dataclass must
store both, and all CLOB API calls must use `clobTokenIds`. The `gamma_client.py`
already parses `clobTokenIds` from the Gamma API response but does not store them
in the `Market` dataclass -- this must be fixed.

---

## 10. Risks and Tradeoffs

### 10.1 Risk: arb/ Module Size Growth

**Likelihood:** High
**Impact:** Medium (maintainability)
**Mitigation:** Monitor file count. If the module exceeds 25 files, extract
Polymarket-specific files into `arb/polymarket/` sub-package. The current decision
to keep files flat is reversible.

### 10.2 Risk: Historical Data Gaps Invalidate Backtests

**Likelihood:** Medium
**Impact:** High (false confidence in strategy performance)
**Mitigation:** Require minimum data coverage thresholds in backtest config
(e.g., 80% of trading days must have price data). Report data coverage in
backtest artifacts. Use conservative fee assumptions for periods where fee
schedules are uncertain.

### 10.3 Risk: Behavioral Strategies Are Overfit to Known Patterns

**Likelihood:** Medium
**Impact:** High (strategies fail in production)
**Mitigation:** Apply the same anti-overfitting gates as Track A/B: DSR >= 0.95,
CPCV OOS/IS > 0, regime-split testing. The Yes/default bias and FLB are documented
across multiple independent studies -- but the magnitude of the edge may be
overstated in academic papers (McLean & Pontiff 2016: anomaly returns 26% lower
OOS, 58% lower post-publication).

### 10.4 Risk: Fee Model Changes Break Strategy Viability

**Likelihood:** High (Polymarket has changed fees 4+ times in 18 months)
**Impact:** Medium (strategies may become unprofitable)
**Mitigation:** Cost stress test (2x fees) as a robustness gate. Fee changes
should trigger re-evaluation of all PM strategies through the robustness pipeline.

### 10.5 Risk: Track C Governance Gaps Remain Open

**Likelihood:** Certain (11 of 17 gates are unimplemented)
**Impact:** Low for research module (it does not deploy capital)
**Mitigation:** The research module should not be promoted to production until
Track C governance gaps are closed. This is already tracked in beads issues
(llm-quant-y7kg, llm-quant-bfvd, llm-quant-tcdt). The research module adds
research infrastructure, not execution risk.

### 10.6 Tradeoff: Flat File Structure vs Sub-Package

**Accepted consequence:** The `arb/` directory will contain ~18 files mixing
Polymarket-specific, Kalshi-specific, CEF-specific, and funding-rate-specific code.
A developer looking for "all Polymarket code" must scan file names rather than
navigating to a sub-package. This is acceptable for a solo developer who already
navigates the codebase by search rather than directory browsing. The alternative
(sub-packages) creates import complexity and partially breaks the existing `arb/`
surface for marginal organizational benefit.

---

## 11. Evidence vs Assumptions

### Evidence (high confidence)

- The Gamma API, CLOB API, and Data API endpoints, rate limits, and response formats
  are documented in the research reports and cross-referenced with official Polymarket
  documentation.
- The fee structure as of March 2026 is documented in two independent research reports
  with consistent figures.
- The Australian legal constraint is documented via ACMA enforcement action (August
  2025) and confirmed in the research reports.
- The existing arb module code has been read in full. All file counts, function
  signatures, and schema definitions are verified from source.
- The Track C governance gap count (4/17 implemented) is from the committed audit
  document dated 2026-03-30.

### Assumptions (medium confidence)

- The pmxt Data Archive will remain available and free. No SLA exists.
- The CLOB `/prices-history` endpoint will continue to provide hourly granularity
  for active markets. This is undocumented and may change.
- The fee schedule in `pm_fees.py` will need updating within 6 months based on
  Polymarket's track record of frequent fee changes.
- The Yes/default bias (~6pp) and FLB are persistent market features, not artifacts
  of the specific time periods studied. This is plausible but unverified for post-2024
  data.

### Unknowns (low confidence)

- Whether the pmxt archive has sufficient coverage for multi-year backtests. The
  archive description says "hourly" but does not specify date range.
- Whether the `PolymarketHistoryClient` rate limit is sufficient for bulk historical
  backfill (fetching price history for 1000+ resolved markets).
- Whether prediction market behavioral strategies survive the Track A/B robustness
  gates (DSR, CPCV). No PM-specific backtest has been run yet. The entire research
  module is speculative until backtests produce results.

---

## 12. Implementation Sequence

### Phase 1: Foundation (Estimated 2-3 days)

1. `pm_fees.py` + `test_pm_fees.py` -- Category-aware fee model.
2. Extend `gamma_client.py` with `clob_token_ids` and `question_id` in `Market`.
3. Extend `schema.py` with `pm_price_history` and `pm_paper_trades` DDL.
4. Update `scanner.py` to use `pm_fees.compute_taker_fee()`.

### Phase 2: Historical Data (Estimated 2-3 days)

5. `pm_history.py` + `test_pm_history.py` -- CLOB /prices-history client.
6. `PmxtArchiveLoader` for bulk Parquet loading.
7. Historical data backfill script for target markets.

### Phase 3: Research Strategies (Estimated 3-5 days)

8. `pm_research.py` + `test_pm_research.py` -- Strategy classes.
9. Research lifecycle artifacts for first strategy (mandate, hypothesis, data
   contract, research spec).

### Phase 4: Backtest + Paper Trading (Estimated 3-5 days)

10. `pm_backtest.py` + `test_pm_backtest.py` -- PM backtest engine.
11. `pm_paper.py` + `test_pm_paper.py` -- Paper trading simulator.
12. Integration with `run_track_c_robustness.py` for PM behavioral strategies.

### Total estimated implementation: 10-16 days for a solo developer.

---

## 13. Open Questions

1. **Should the PM backtest engine support limit order simulation (with fill
   probability modeling) or only market order simulation (assume immediate fill at
   midpoint)?** Limit order simulation is more realistic but requires orderbook
   depth data that may not be available historically. Recommendation: start with
   market order simulation, add limit order modeling in a later phase if orderbook
   data from pmxt proves sufficient.

2. **Should the PM paper trader track positions as contracts (YES/NO tokens) or
   as USD notional?** Contract-level tracking is more faithful to the actual
   instrument but adds complexity (resolution converts contracts to USD). USD
   notional tracking is simpler but loses the contract-level detail. Recommendation:
   track at contract level internally, report in USD externally.

3. **Should cross-platform arb (Polymarket vs Kalshi) be modeled as a single
   strategy or two independent positions?** A single-strategy model captures the
   hedged nature of the trade (long on one platform, short on another). Two
   independent positions would be tracked and risk-managed separately.
   Recommendation: model as a single strategy with two legs, but this requires
   the paper trader to understand paired positions.

---

## 14. Recommended Next Steps

1. Claim beads issue llm-quant-ob4b and mark this ADR as the deliverable.
2. Create implementation issues for Phase 1-4 with dependency ordering.
3. Begin with Phase 1 (pm_fees.py) as the lowest-risk, highest-value change --
   it fixes the hardcoded fee constant that makes the existing scanner inaccurate
   for non-crypto markets.
4. After Phase 2, evaluate pmxt archive data quality before committing to Phase 3
   strategy design. If historical data is insufficient, the behavioral strategy
   hypotheses cannot be tested and should be deferred.

---

*End of ADR.*

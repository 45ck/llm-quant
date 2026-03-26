# llm-quant

LLM-powered paper trading system where **Claude acts as the portfolio manager** — receiving market data, making trade decisions with reasoning, and tracking performance.

The unique angle: the LLM *is* the strategy, not hard-coded rules.

## How It Works

1. **Fetch** daily OHLCV data for 30 liquid US ETFs via Yahoo Finance
2. **Compute** technical indicators (SMA, RSI, MACD, ATR) using Polars
3. **Send** market context + portfolio state to Claude as a structured prompt
4. **Receive** JSON trade decisions with regime analysis and per-signal reasoning
5. **Execute** paper trades after pre-trade risk checks
6. **Track** everything in DuckDB — trades, decisions, portfolio snapshots

## Research Lab Results

This system runs a **133-hypothesis quantitative research lab** — every strategy passes through a strict 5-gate robustness filter before any capital is committed.

### The Funnel

```
133  hypotheses in scope (across 16 mandate categories)
 68  strategy variants backtested (5-year window, 2022-2026)
 11  passed all 5 robustness gates                           (16% pass rate)
 11  currently in paper trading
  0  promoted to live capital
```

### 5-Gate Robustness Filter

Every strategy must clear **all five** before paper trading:

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Sharpe Ratio | > 0.80 | Alpha exists and is meaningful |
| Max Drawdown | < 15% | Portfolio-safe risk profile |
| DSR (Deflated Sharpe) | >= 0.95 | Adjusts for multiple testing; p-value for alpha |
| CPCV OOS/IS | >= 0.50 | Out-of-sample generalization (not curve-fit) |
| Perturbation stability | >= 3/5 | Parameters are economically robust, not arbitrary |

### Passing Strategies (11 of 68 tested)

All 11 are in paper trading as of 2026-03-26. Promotion requires 30+ days of paper track record.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Mechanism |
|---------|--------|-------|-----|-------------|-----------|
| LQD-SPY credit lead | 1.250 | 12.4% | 0.9950 | 1.023 | IG bond → US equity |
| AGG-SPY credit lead | 1.145 | 8.4% | 0.9938 | 1.039 | Total bond → US equity |
| SPY overnight momentum | 1.043 | 8.7% | 0.9506 | 1.011 | Overnight gap microstructure |
| AGG-QQQ credit lead | 1.080 | 11.2% | 0.9894 | 1.031 | Total bond → tech equity |
| VCIT-QQQ credit lead | 1.037 | 14.5% | 0.9820 | 1.010 | Corp bond → tech equity |
| LQD-QQQ credit lead | 1.023 | 13.7% | 0.9824 | 1.031 | IG bond → tech equity |
| EMB-SPY credit lead | 1.005 | 9.1% | 0.9802 | 0.980 | EM sovereign → US equity |
| HYG-SPY credit lead | 0.913 | 14.7% | 0.9650 | 1.111 | HY bond → US equity |
| AGG-EFA credit lead | 0.860 | 10.3% | 0.9656 | 1.134 | Total bond → intl equity |
| HYG-QQQ credit lead | 0.867 | 13.4% | 0.9606 | 1.050 | HY bond → tech equity |
| SOXX-QQQ lead-lag | 0.861 | 14.4% | 0.9603 | 0.819 | Semis → tech equity |

### Portfolio Correlation Reality

10 of 11 passing strategies share the same underlying mechanism (credit-equity lead-lag).
Running the equal-weight portfolio as 11 separate strategies overstates diversification:

| Metric | Credit-only (10) | Full portfolio (11) |
|--------|-----------------|---------------------|
| Average pairwise correlation | 0.628 | 0.584 |
| Effective independent N | 4.35 | 5.16 |
| Estimated equal-weight Sharpe | ~2.0 | ~2.3 |

The SPY overnight momentum strategy (C7) is the only mechanistically distinct passer —
average correlation 0.386 with the credit-equity family.

### What Gets Rejected and Why

| Failure mode | Count | Examples |
|-------------|-------|---------|
| DSR < 0.95 (insufficient alpha after trial penalty) | ~18 | Correlation regime, VoV, XLU inverse |
| MaxDD > 15% (2022 bear market too harsh) | ~12 | Factor rotation, asset rotation, pairs |
| Sharpe < 0.80 (weak signal) | ~8 | Calendar effects, size rotation |
| Perturbation unstable (over-fit parameters) | ~6 | SPY-TLT-GLD-BIL, L-series OHLCV |
| Falsified (signal in wrong direction) | ~4 | Pre-FOMC TLT drift, turn-of-month |

## Live Portfolio Performance

| Metric | Value | Target |
|--------|-------|--------|
| NAV | $100,000 | — |
| Total Return | 0.00% | 8-15% ann. |
| Sharpe Ratio | — | > 0.80 |
| Sortino Ratio | — | > 1.00 |
| Max Drawdown | 0.00% | < 15% |
| Benchmark (60/40) | — | — |

> Updated daily via [automated reports](reports/).
> Research lab results updated 2026-03-26.

## Reports

Performance reports are generated automatically and committed to git as an immutable public record.

- [Daily Reports](reports/daily/) — Portfolio snapshot, trades, metrics
- [Weekly Reports](reports/weekly/) — Weekly performance summary
- [Monthly Reports](reports/monthly/) — Full metrics dashboard and trade analysis

Reports are generated from the live DuckDB database by `scripts/generate_report.py` and auto-committed via GitHub Actions.

## Transparency

This is a live paper trading system. Every trade decision is:

1. **Logged with reasoning** — Each trade includes the LLM's hypothesis and conviction level
2. **Hash-chain verified** — Trade ledger uses SHA-256 hash chain for tamper evidence
3. **Git-tracked** — All reports committed automatically, creating an immutable public record
4. **Auditable** — Run `pq verify` to validate the entire trade history

The system benchmarks against a passive 60/40 SPY/TLT portfolio. All performance metrics are computed from raw trade data, not self-reported.

## Quick Start

### Prerequisites
- Python 3.12+
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Install

```bash
git clone https://github.com/45ck/llm-quant.git
cd llm-quant
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run

```bash
# Initialize database and universe
pq init

# Fetch market data
pq fetch

# Run trading cycle (dry run first)
pq run --dry-run

# Execute live paper trades
pq run

# Check portfolio status
pq status

# View trade history with reasoning
pq trades
```

## Architecture

```
src/llm_quant/
├── cli.py          # Typer CLI (pq command)
├── config.py       # Pydantic config from TOML
├── data/           # Market data pipeline
│   ├── fetcher.py  # Yahoo Finance downloader
│   ├── store.py    # DuckDB read/write layer
│   ├── indicators.py # SMA, RSI, MACD, ATR
│   └── universe.py # ETF universe management
├── brain/          # LLM integration
│   ├── engine.py   # Claude API signal engine
│   ├── prompts.py  # Jinja2 prompt templates
│   ├── parser.py   # JSON response parser
│   ├── context.py  # Market context builder
│   └── models.py   # Domain dataclasses
├── trading/        # Paper trading
│   ├── portfolio.py # Portfolio state
│   ├── executor.py # Trade execution
│   ├── ledger.py   # Trade logging
│   └── performance.py # Metrics (Sharpe, drawdown)
├── risk/           # Pre-trade risk
│   ├── manager.py  # Risk check orchestrator
│   └── limits.py   # Individual limit checks
└── db/
    └── schema.py   # DuckDB schema
```

## Configuration

All config lives in `config/`:

- **`default.toml`** — General settings (model, capital, lookback)
- **`universe.toml`** — ETF universe (30 symbols across equities, bonds, commodities)
- **`risk.toml`** — Risk limits (2% max trade, 10% max position, 200% gross cap)
- **`prompts/`** — Jinja2 templates for the Claude PM persona

## Risk Constraints

Every trade passes through pre-trade risk checks:

| Limit | Value |
|-------|-------|
| Max single trade | 2% of NAV |
| Max position weight | 10% of NAV |
| Max gross exposure | 200% of NAV |
| Max net exposure | 100% of NAV |
| Max sector concentration | 30% |
| Min cash reserve | 5% of NAV |
| Stop-loss required | Yes |
| Max trades per session | 5 |

## Cost

Claude Sonnet at ~$0.01 per daily signal call. Running daily for a year costs roughly $2.50.

## Tech Stack

- **Polars** — Fast DataFrames (no pandas)
- **DuckDB** — Embedded analytics database
- **yfinance** — Market data
- **anthropic** — Claude API
- **Typer + Rich** — Beautiful CLI
- **Pydantic** — Config validation

## Testing

```bash
pytest
pytest -v --tb=short  # verbose
```

## License

MIT

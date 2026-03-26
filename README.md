# llm-quant

LLM-powered systematic trading research program running **two parallel alpha tracks** — a conservative base (Defensive Alpha) and a high-return research track (Aggressive Alpha). Claude acts as portfolio manager, researcher, and quant analyst.

## Dual-Track Research Program

```
Track A — Defensive Alpha      Track B — Aggressive Alpha
─────────────────────────      ────────────────────────────
Target: 15-25% CAGR            Target: 40-80% CAGR
MaxDD gate: < 15%              MaxDD gate: < 30%
Sharpe gate: > 0.80            Sharpe gate: > 1.0
Portfolio weight: 70%          Portfolio weight: 30%
Status: 11 strategies active   Status: research phase
```

**Integrity gates are the same on both tracks** — DSR >= 0.95, CPCV OOS/IS > 0. These are
anti-overfitting controls, not risk controls, and are non-negotiable.

See [research-tracks.md](docs/governance/research-tracks.md),
[alpha-hunting-framework.md](docs/governance/alpha-hunting-framework.md), and
[institutional-quant-guide.md](docs/research/institutional-quant-guide.md) for full specifications.

## How It Works

1. **Fetch** daily OHLCV data for 39 liquid US ETFs + crypto via Yahoo Finance
2. **Compute** technical indicators (SMA, RSI, MACD, ATR, rolling correlation) using Polars
3. **Send** market context + portfolio state to Claude as a structured prompt
4. **Receive** JSON trade decisions with regime analysis and per-signal reasoning
5. **Execute** paper trades after pre-trade risk checks (7 automated limits)
6. **Track** everything in DuckDB — trades, decisions, portfolio snapshots, hash chain

## Research Lab Results

This system runs a **133-hypothesis quantitative research lab** — every strategy passes through a 5-gate robustness filter before any capital is committed.

### The Funnel (Track A)

```
133  hypotheses in scope (across 16 mandate categories)
 68  strategy variants backtested (5-year window, 2022-2026)
 11  passed all 5 robustness gates                           (16% pass rate)
 11  currently in paper trading
  0  promoted to live capital
```

### Gate Comparison by Track

| Gate | Track A (Defensive) | Track B (Aggressive) | Purpose |
|------|---------------------|---------------------|---------|
| Sharpe Ratio | > 0.80 | > 1.00 | Alpha exists and is meaningful |
| Max Drawdown | < 15% | < 30% | Portfolio-safe risk profile |
| DSR (Deflated Sharpe) | >= 0.95 | >= 0.95 | Anti-overfitting — same on both tracks |
| CPCV OOS/IS | > 0 | > 0 | Out-of-sample generalization — same on both tracks |
| Perturbation stability | >= 3/5 | >= 3/5 | Parameter robustness — same on both tracks |

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

| Metric | Value | Track A Target | Track B Target |
|--------|-------|---------------|---------------|
| NAV | $100,000 | — | — |
| Total Return | 0.00% | 15-25% ann. | 40-80% ann. |
| Sharpe Ratio | — | > 0.80 | > 1.00 |
| Sortino Ratio | — | > 1.00 | > 1.50 |
| Max Drawdown | 0.00% | < 15% | < 30% |
| Benchmark | — | 60/40 SPY/TLT | 100% SPY |

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

## Research Methodology

Statistical rigor follows institutional standards documented in
[docs/research/institutional-quant-guide.md](docs/research/institutional-quant-guide.md):

- **DSR >= 0.95** — Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) corrects for
  multiple testing across all strategy variants tested
- **PBO <= 10%** — Probability of Backtest Overfitting via Combinatorial Symmetric CV
- **CPCV (15 OOS paths)** — Combinatorially Purged Cross-Validation with purge + embargo
- **t-stat > 3.0** — Harvey, Liu & Zhu (2016) threshold for new factor proposals
- **Spec freeze before backtest** — Hypothesis pre-registered, no HARKing
- **Append-only experiment registry** — Every trial recorded, no selective reporting

Current implementation gaps tracked in
[docs/research/implementation-gaps.md](docs/research/implementation-gaps.md):
shuffled signal fraud detector (P1), HRP portfolio weights (P2), volatility targeting (P2).

## Configuration

All config lives in `config/`:

- **`default.toml`** — General settings (model, capital, lookback)
- **`universe.toml`** — ETF universe (39 symbols across equities, bonds, commodities, crypto)
- **`risk.toml`** — Risk limits — Track A (default) and `[track_b]` section
- **`prompts/`** — Jinja2 templates for the Claude PM persona

## Risk Constraints

Every trade passes through pre-trade risk checks. Limits differ by track:

| Limit | Track A | Track B |
|-------|---------|---------|
| Max single trade | 2% of NAV | 3% of NAV |
| Max position weight | 10% of NAV | 15% of NAV |
| Max gross exposure | 200% of NAV | 200% of NAV |
| Max net exposure | 100% of NAV | 100% of NAV |
| Max sector concentration | 30% | 30% |
| Min cash reserve | 5% of NAV | 3% of NAV |
| Max drawdown circuit breaker | 15% | 30% |
| Stop-loss required | Yes | Yes |
| Max trades per session | 5 | 5 |

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

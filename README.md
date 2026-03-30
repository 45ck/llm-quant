# llm-quant

LLM-powered systematic trading research program running **four parallel alpha tracks** across US equities, fixed income, commodities, crypto, and forex. Claude acts as portfolio manager, researcher, and quant analyst over a $100k paper trading portfolio spanning 39 tradeable assets.

## Four-Track Research Program

```
Track A — Defensive Alpha      Track B — Aggressive Alpha
─────────────────────────      ────────────────────────────
Target: 15-25% CAGR            Target: 40-80% CAGR
MaxDD gate: < 15%              MaxDD gate: < 30%
Sharpe gate: > 0.80            Sharpe gate: > 1.0
Portfolio weight: 70%          Portfolio weight: 30%
Status: 11 strategies paper    Status: research phase

Track C — Structural Arb       Track D — Sprint Alpha
─────────────────────────      ────────────────────────────
Target: risk-free + alpha      Target: 60-120% CAGR
MaxDD gate: < 10%              MaxDD gate: < 40%
Sharpe gate: > 1.50            Sharpe gate: > 0.80
Benchmark: T-bills             Benchmark: 100% TQQQ
Status: research (4/17 gates)  Status: experimental (backtest)
```

**Integrity gates are the same on all tracks** — DSR >= 0.95, CPCV OOS/IS > 0. These are
anti-overfitting controls, not risk controls, and are non-negotiable.

See [research-tracks.md](docs/governance/research-tracks.md),
[alpha-hunting-framework.md](docs/governance/alpha-hunting-framework.md), and
[institutional-quant-guide.md](docs/research/institutional-quant-guide.md) for full specifications.

## How It Works

1. **Fetch** daily OHLCV + macro data (FRED, COT, FOMC) for 39 liquid assets via Yahoo Finance
2. **Compute** indicators (SMA, RSI, MACD, ATR, TSMOM) + regime detection (HMM, inflation 2x2) using Polars
3. **Send** market context + portfolio state + regime signals to Claude as a structured prompt
4. **Receive** JSON trade decisions with regime analysis and per-signal reasoning
5. **Execute** paper trades after pre-trade risk checks (14 automated limits + CVaR constraints)
6. **Track** everything in DuckDB — trades, decisions, portfolio snapshots, hash chain

## Research Lab Results

This system runs a **133-hypothesis quantitative research lab** across 8 mechanism families — every strategy passes through a 5-gate robustness filter before any capital is committed.

### The Funnel (Track A)

```
133  hypotheses in scope (across 8 mechanism families)
 94  strategy directories with lifecycle artifacts
 68  strategy variants backtested (5-year window, 2022-2026)
 11  passed all 5 robustness gates                           (16% pass rate)
 11  currently in paper trading
  0  promoted to live capital
```

### Gate Comparison by Track

| Gate | Track A | Track B | Track C | Track D | Purpose |
|------|---------|---------|---------|---------|---------|
| Sharpe Ratio | > 0.80 | > 1.00 | > 1.50 | > 0.80 | Alpha meaningful |
| Max Drawdown | < 15% | < 30% | < 10% | < 40% | Risk profile |
| DSR | >= 0.95 | >= 0.95 | >= 0.95 | >= 0.90 | Anti-overfitting |
| CPCV OOS/IS | > 0 | > 0 | > 0 | > 0 | OOS generalization |
| Perturbation | >= 3/5 | >= 3/5 | >= 3/5 | >= 3/5 | Parameter robustness |

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
average correlation 0.386 with the credit-equity family. GLD-SLV v4 has the lowest correlation
(avg rho = 0.051) but **fails 2/3 fraud detectors** — mechanism inversion shows it captures
momentum, not mean-reversion. Do not promote until signal logic is reworked.

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
> Research lab results updated 2026-03-31.

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
├── cli.py              # Typer CLI (pq command)
├── config.py           # Pydantic config from TOML
├── data/               # Market data pipeline
│   ├── fetcher.py      # Yahoo Finance downloader
│   ├── store.py        # DuckDB read/write layer
│   ├── indicators.py   # SMA, RSI, MACD, ATR + realized variance + vol scalar
│   ├── universe.py     # ETF universe management
│   ├── fred_fetcher.py # FRED macro data (T10Y2Y, UNRATE, CPI, breakevens)
│   └── cot_fetcher.py  # CFTC Commitments of Traders (156-week lookback)
├── brain/              # LLM integration
│   ├── engine.py       # Claude API signal engine
│   ├── prompts.py      # Jinja2 prompt templates
│   ├── parser.py       # JSON response parser
│   ├── context.py      # Market context builder (credit, VIX, COT, TSMOM, regime)
│   └── models.py       # Domain dataclasses
├── regime/             # Market regime detection
│   ├── hmm.py          # 2-state HMM (risk_on/risk_off) — Nystrup et al.
│   └── inflation.py    # 2x2 growth/inflation matrix (Bridgewater framework)
├── signals/            # Signal generation
│   └── tsmom.py        # Multi-lookback TSMOM with vol scaling (21/63/252d)
├── analysis/           # Signal analysis
│   └── ic_analysis.py  # alphalens IC/ICIR/decay analysis
├── backtest/           # Strategy backtesting
│   ├── engine.py       # Core backtest engine with meta-filters + vol targeting
│   ├── robustness.py   # 8-gate robustness: DSR, CPCV, perturbation, MinTRL
│   └── walk_forward.py # Anchored + rolling walk-forward validation (WF-OOS/IS gate)
├── trading/            # Paper trading
│   ├── portfolio.py    # Portfolio state
│   ├── executor.py     # Trade execution with vol scaling
│   ├── ledger.py       # Trade logging
│   └── performance.py  # Metrics via empyrical (Sharpe, Sortino, drawdown, Calmar)
├── risk/               # Pre-trade risk (14 checks)
│   ├── manager.py      # Risk check orchestrator (Track A/B/C/D aware)
│   ├── limits.py       # Individual limit checks + ATR sizing + CVaR
│   ├── correlation.py  # DCC-GARCH dynamic correlation (EWMA fallback)
│   └── cvar.py         # Filtered Historical Simulation CVaR + stress scenarios
├── surveillance/       # Post-trade governance (7 detectors + kill switches)
│   ├── scanner.py      # Full surveillance scan
│   ├── track_c_detectors.py  # Track C kill switches (5 types)
│   └── track_d_monitor.py    # Track D hold period + VIX + beta decay monitors
├── arb/                # Structural arbitrage (Track C)
│   ├── detector.py     # Polymarket/Kalshi arb detection
│   ├── scanner.py      # Multi-venue arb scanner
│   ├── funding_rates.py # Crypto perpetual funding rate capture
│   └── gamma_client.py # Polymarket CLOB API client
├── nlp/                # Text/NLP signals (Family 6)
│   ├── text_classifier.py  # Claude-powered sentence classification
│   ├── edgar_fetcher.py    # SEC EDGAR 10-K/MD&A extraction
│   └── fomc_fetcher.py     # FOMC minutes hedging language scorer
└── db/
    └── schema.py       # DuckDB schema + hash chain integrity
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

Previously tracked implementation gaps are now mostly resolved:
shuffled signal fraud detector (**implemented**), HRP portfolio weights (**implemented** via Riskfolio-Lib),
volatility targeting (**implemented** via 126d realized variance), portfolio correlation gate (**implemented**
via DCC-GARCH), marginal SR contribution gate (**implemented**), walk-forward validation (**implemented**),
CVaR constraints (**implemented** via Filtered Historical Simulation).

Portfolio construction mathematics and the path to extreme Sharpe documented in
[docs/research/extreme-sharpe-playbook.md](docs/research/extreme-sharpe-playbook.md):
three paths (breadth, uncorrelated stack, leverage), correlation reality table, tier benchmarks.

## Configuration

All config lives in `config/`:

- **`default.toml`** — General settings, regime config, TSMOM, market context
- **`universe.toml`** — ETF universe (39 symbols, CFTC codes, COT eligibility flags)
- **`risk.toml`** — Risk limits (Track A/B/C/D), ATR params, execution costs, vol scaling, CVaR
- **`governance.toml`** — Surveillance thresholds, kill switch parameters
- **`macro-briefing.md`** — Structured macro context for regime assessment
- **`prompts/`** — Jinja2 templates for the Claude PM persona

## Risk Constraints

Every trade passes through 14 pre-trade risk checks. Limits differ by track:

| Limit | Track A | Track B | Track C | Track D |
|-------|---------|---------|---------|---------|
| Max single trade | 2% NAV | 3% NAV | $2,000 | 5% NAV |
| Max position weight | 10% NAV | 15% NAV | — | 30-50% NAV |
| Max gross exposure | 200% NAV | 200% NAV | — | — |
| Max sector concentration | 30% | 30% | — | — |
| Min cash reserve | 5% NAV | 3% NAV | — | — |
| Max drawdown circuit breaker | 15% | 30% | 10% | 40% |
| Stop-loss required | Yes | Yes | — | Yes |
| Max hold period | — | — | — | 5 days |
| ATR-based sizing | Yes | Yes | — | Yes |
| CVaR constraint (95%) | 5% | 5% | — | — |
| DCC-GARCH correlation | Advisory | Advisory | — | — |
| Vol scaling (12% target) | Yes | Yes | — | — |

## Cost

Claude Sonnet at ~$0.01 per daily signal call. Running daily for a year costs roughly $2.50.

## Tech Stack

- **Polars** — Fast DataFrames (no pandas in core — pandas only at library boundaries)
- **DuckDB** — Embedded analytics database
- **yfinance** — Market data
- **anthropic** — Claude API (Sonnet for signals, Opus for research)
- **Typer + Rich** — Beautiful CLI
- **Pydantic** — Config validation
- **empyrical-reloaded** — Performance metrics (Sharpe, Sortino, Calmar, drawdown)
- **hmmlearn** — Hidden Markov Model regime detection
- **arch** — GARCH volatility modeling + DCC correlation
- **Riskfolio-Lib** — Hierarchical Risk Parity portfolio construction
- **alphalens-reloaded** — Signal IC/ICIR analysis
- **quantstats** — Tearsheet generation
- **scipy** — Ward linkage (HRP), bootstrap CI (CVaR)

## Testing

```bash
pytest
pytest -v --tb=short  # verbose
```

## License

MIT

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
Status: 19 strategies paper    Status: 3 strategies paper
(research complete, backlog)

Track C — Structural Arb       Track D — Sprint Alpha
─────────────────────────      ────────────────────────────
Target: risk-free + alpha      Target: 50-120% CAGR
MaxDD gate: < 10%              MaxDD gate: < 40%
Sharpe gate: > 1.50            Sharpe gate: > 0.80
Benchmark: T-bills             Benchmark: 100% TQQQ
Status: infra built, scanning  Status: 14 strategies passing
                               (PRIMARY RESEARCH FOCUS)
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

This system runs a **200+ hypothesis quantitative research lab** across 15 mechanism families — every strategy passes through a 6-gate robustness filter before any capital is committed.

### The Funnel (All Tracks)

```
250+ hypotheses tested (across 48+ candidate mechanism families)
 22  mechanism families yielded at least one passing strategy
 36  strategies passed all 6 robustness gates — now in paper trading
  0  promoted to live capital (requires 30+ day track record)
```

Falsification is as productive as discovery: 18+ mechanism families were tested and
rejected outright (F4, F10, F20, F23-F25, F28-F29, F31-F32, F34-F41). Recent
falsifications include F37 volume-price divergence (Sharpe=-0.93), F38 vol-transmission,
F41 FX-carry risk appetite, and the retirement of D3 TQQQ/TMF ratio MR after independent
replication showed Sharpe=-1.08 (the original 2.21 was a signal bug that passed initial
gates). This kind of rigorous replication discipline is what the integrity gates are for.

### Gate Comparison by Track

| Gate | Track A | Track B | Track C | Track D | Purpose |
|------|---------|---------|---------|---------|---------|
| Sharpe Ratio | > 0.80 | > 1.00 | > 1.50 | > 0.80 | Alpha meaningful |
| Max Drawdown | < 15% | < 30% | < 10% | < 40% | Risk profile |
| DSR | >= 0.95 | >= 0.95 | >= 0.95 | >= 0.90 | Anti-overfitting |
| CPCV OOS/IS | > 0 | > 0 | > 0 | > 0 | OOS generalization |
| Perturbation | >= 3/5 | >= 3/5 | >= 3/5 | >= 3/5 | Parameter robustness |

### Passing Strategies (32 in paper trading)

All 32 are in paper trading as of 2026-04-01. Promotion requires 30+ days of paper track record.
Daily signals generated via `scripts/run_paper_batch.py` (batch runner for all strategies).

**Family 1 — Credit-Equity Lead-Lag** (9 strategies)

Bond markets price risk before equity markets. When credit spreads tighten (bond prices rise), equities follow 3-5 days later. Signal: 5-day bond return exceeds threshold → enter equity follower.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Leader → Follower |
|----------|--------|-------|-----|-------------|-------------------|
| LQD-SPY | 1.250 | 12.4% | 0.9950 | 1.023 | IG bond → S&P 500 |
| AGG-SPY | 1.145 | 8.4% | 0.9938 | 1.039 | Total bond → S&P 500 |
| AGG-QQQ | 1.080 | 11.2% | 0.9894 | 1.031 | Total bond → Nasdaq |
| VCIT-QQQ | 1.037 | 14.5% | 0.9820 | 1.010 | Corp bond → Nasdaq |
| LQD-QQQ | 1.023 | 13.7% | 0.9824 | 1.031 | IG bond → Nasdaq |
| EMB-SPY | 1.005 | 9.1% | 0.9802 | 0.980 | EM sovereign → S&P 500 |
| HYG-SPY | 0.913 | 14.7% | 0.9650 | 1.111 | HY bond → S&P 500 |
| HYG-QQQ | 0.867 | 13.4% | 0.9606 | 1.050 | HY bond → Nasdaq |
| AGG-EFA | 0.860 | 10.3% | 0.9656 | 1.134 | Total bond → Intl DM |

**Family 2 — Mean Reversion** (1 strategy)

Precious metals ratio (GLD/SLV) mean-reverts on quarterly timescales. Consensus voting across 60/90/120-day Bollinger Bands eliminates single-lookback sensitivity.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| GLD-SLV v4 consensus | 1.197 | 9.6% | 0.9910 | 1.244 | 3-window Bollinger Band vote |

**Family 3 — Trend Following / TSMOM** (1 strategy)

Novy-Marx skip-month time-series momentum: use returns from t-252 to t-21 (skip most recent month) with vol scaling across SPY/TLT/GLD/EFA.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Skip-month TSMOM v1 | 1.331 | 9.9% | 0.971 | 1.081 | Vol-scaled skip-month momentum |

**Family 5 — Overnight Momentum** (1 strategy)

SPY overnight returns (open vs prior close) exhibit serial momentum. A 10-day rolling average of overnight gaps predicts next-day direction.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| SPY overnight | 1.043 | 8.7% | 0.9506 | 1.011 | 10d avg overnight gap > 0.20% |

**Family 6 — Rate Momentum** (3 strategies)

Treasury price changes lead equity returns via the discount rate channel. When bonds rally (rates falling), equities follow with a 5-day lag.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Leader → Follower |
|----------|--------|-------|-----|-------------|-------------------|
| IEF-QQQ | 0.979 | 14.5% | 0.9908 | 1.392 | 7-10yr Treasury → Nasdaq |
| TLT-QQQ | 0.935 | 11.8% | 0.9769 | 1.182 | 20yr+ Treasury → Nasdaq |
| TLT-SPY | 0.803 | 10.8% | 0.9506 | 1.184 | 20yr+ Treasury → S&P 500 |

**Family 8 — Non-Credit Lead-Lag** (1 strategy)

Semiconductor sector leads broader tech by ~5 trading days due to supply chain information flow.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Leader → Follower |
|----------|--------|-------|-----|-------------|-------------------|
| SOXX-QQQ | 0.861 | 14.4% | 0.9603 | 0.819 | Semiconductors → Nasdaq |

**Family 9 — Credit Spread Regime** (1 strategy)

HYG/SHY ratio momentum + ratio vs SMA determines three regimes. Near-zero correlation with all other families.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Credit-spread-regime v1 | 0.990 | 10.8% | 0.9612 | 0.889 | HYG/SHY ratio regime |

**Family 11 — Commodity Cycle** (1 strategy)

DBA 60-day absolute momentum distinguishes inflation (→ GLD+SPY) vs disinflation (→ SPY) regimes.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| DBA commodity cycle v1 | 1.010 | 13.5% | 0.9632 | 0.918 | DBA absolute momentum |

**Family 12 — Sector Rotation** (1 strategy)

XLK/XLE ratio momentum + SMA classifies growth vs inflation vs neutral regimes. Highest individual Sharpe in portfolio.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| XLK-XLE sector rotation v1 | 1.525 | 11.5% | 0.9880 | 0.888 | Tech/energy ratio regime |

**Family 13 — Volatility Regime** (1 strategy)

SPY vs GLD 30-day realized vol comparison identifies equity stress vs commodity stress regimes.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Vol-regime v2 | 1.270 | 14.2% | 0.980 | 0.970 | SPY vs GLD realized vol |

**Family 14 — Curve Shape Momentum** (1 strategy)

TLT/SHY price ratio 30-day momentum captures yield curve shape changes. Near-zero corr with credit-equity.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| TLT/SHY curve momentum v1 | 1.044 | 14.3% | 0.966 | 0.906 | TLT/SHY ratio momentum |

**Family 15 — Real Yield Proxy** (1 strategy)

TIP/TLT price ratio 20-day momentum proxies real yield changes. Loosening → SPY, tightening → GLD+SHY.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| TIP/TLT real yield v1 | 1.313 | 13.3% | 0.982 | 0.861 | TIP/TLT ratio momentum |

**Family 16 — Breakeven Inflation** (1 strategy)

TIP/IEF ratio 30-day momentum proxies breakeven inflation rate. Rising inflation tilts to commodities+gold, falling to equities.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Breakeven inflation v1 | 1.068 | 13.0% | 0.968 | 1.053 | TIP/IEF ratio momentum |

**Family 17 — Capital Flow** (1 strategy)

TLT/EFA ratio momentum distinguishes US-preferred vs international-preferred capital flows.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Global yield flow v2 | 0.900 | 10.9% | 0.951 | 1.113 | TLT/EFA ratio momentum |

**Family 18 — Commodity Carry** (1 strategy)

USO/DBC ratio momentum as crude oil backwardation proxy. Carry regime tilts to energy+gold.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Commodity carry v2 | 1.119 | 14.4% | 0.972 | 1.026 | USO/DBC ratio momentum |

**Family 19 — Disinflation Signal** (1 strategy)

TLT/GLD ratio momentum as pure disinflation proxy. Disinflation → equities, inflation → gold+commodities.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| TLT/GLD disinflation v1 | 1.313 | 8.5% | 0.982 | 1.021 | TLT/GLD ratio momentum |

**Family 21 — Commodity-Equity Rotation** (1 strategy)

DBC/SPY ratio momentum rotates between commodity and equity regimes.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| DBC/SPY commodity-equity v1 | 0.942 | 10.7% | 0.956 | 0.928 | DBC/SPY ratio momentum |

**Family 22 — Duration Rotation** (1 strategy)

AGG/TLT ratio momentum rotates between short-duration and long-duration bond exposure.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| AGG/TLT duration rotation v2 | 0.915 | 13.4% | 0.953 | 0.933 | AGG/TLT ratio momentum |

**Family 26 — Dollar-Gold Regime** (1 strategy, NEW 2026-04-01)

UUP/GLD ratio 30-day momentum as purchasing power proxy. Dollar strength → equities, gold strength → commodities. Different from F19 (which uses TLT/GLD as a yield signal).

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Dollar-gold regime v1 | 0.987 | 13.9% | 0.961 | 1.144 | UUP/GLD ratio momentum |

**Conditional Pass — Behavioral/Structural** (1 strategy)

Low-volatility sector rotation with correlation-based regime detection. Sharpe (0.70) below 0.80 threshold. Included with documented exception due to exceptional parameter stability and unique mechanism.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS | Perturb | Note |
|----------|--------|-------|-----|----------|---------|------|
| Behavioral-structural | 0.699 | 3.6% | 0.9632 | 0.047 | 18/18 | Sharpe below gate |

**Track B — Aggressive Alpha** (3 strategies)

Higher drawdown tolerance (30%) for stronger signals. Zero equity exposure in pair trades.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| SOXX-QQQ lead-lag | 0.861 | 14.4% | 0.960 | 0.819 | Semis → Nasdaq lead-lag |
| USO/XLE energy MR v2 | 1.012 | 22.5% | 0.962 | 1.038 | Energy pair z-score |
| GDX/GLD miners MR v1 | 1.025 | 26.4% | 0.963 | 0.918 | Gold miners pair z-score |

**Family 30 — ERP Valuation Regime** (1 strategy, NEW 2026-04)

Equity Risk Premium (SPY 1-year return minus ^TNX yield, 252-day z-score) drives rotation between stocks and bonds/gold. Institutional capital allocation channel — different from rate momentum (F6) or curve shape (F14).

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| ERP regime v1 | **1.566** | **8.8%** | **0.9963** | 0.954 | SPY-TNX z-score regime |

**Family 33 — REIT Divergence** (1 strategy, NEW 2026-04)

XLRE/SPY ratio z-score + momentum as financial conditions canary. REITs are rate-sensitive so their relative performance leads tightening/easing regimes.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| REIT divergence v2 | 1.075 | 13.3% | 0.972 | 1.291 | XLRE/SPY z-score + mom |

**Family 42 — Dividend Yield Regime** (1 strategy, NEW 2026-04)

SPYD/SPY ratio momentum captures bond-proxy equity rotation (utilities, REITs, staples) vs growth. **100% perturbation stability — most robust strategy ever tested.**

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| Dividend yield regime v1 | 1.159 | 13.8% | 0.9943 | 1.052 | SPYD/SPY ratio regime |

**Track B — Aggressive Alpha** (3 strategies)

Higher drawdown tolerance (30%) for stronger signals. Zero equity exposure in pair trades.

| Strategy | Sharpe | MaxDD | DSR | CPCV OOS/IS | Signal |
|----------|--------|-------|-----|-------------|--------|
| SOXX-QQQ lead-lag | 0.861 | 14.4% | 0.960 | 0.819 | Semis → Nasdaq lead-lag |
| USO/XLE energy MR v2 | 1.012 | 22.5% | 0.962 | 1.038 | Energy pair z-score |
| GDX/GLD miners MR v1 | 1.025 | 26.4% | 0.963 | 0.918 | Gold miners pair z-score |

**Track D — Sprint Alpha** (14 strategies, PRIMARY RESEARCH FOCUS)

Leveraged re-expression of proven signals via 3x ETFs (TQQQ, UPRO, SOXL). Every
strategy must include a VIX>30 crash filter and maximum 5-day holding period to manage
beta decay and volatility drag. Key finding: **rate momentum (TLT) survives all 3x
vehicles; credit lead-lag survives TQQQ but NOT SOXL; commodity carry has no causal
link to semis (FALSIFIED as D16).**

| Strategy | Sharpe | CAGR | MaxDD | DSR | Follower | Note |
|----------|--------|------|-------|-----|----------|------|
| **D13 TSMOM-UPRO** | **1.345** | 20.1% | 15.9% | 0.971 | UPRO | Highest Track D Sharpe, 100% perturb |
| **D10 XLK-XLE-SOXL** | 1.171 | **27.7%** | 22.6% | 0.975 | SOXL | Highest Track D CAGR, sector rotation |
| AGG-TQQQ | 1.079 | 10.9% | 10.6% | 0.969 | TQQQ | Credit lead → 3x tech |
| TLT-TQQQ | 1.030 | 12.4% | 10.2% | 0.965 | TQQQ | Rate momentum → 3x tech |
| D15 Vol-Regime-TQQQ | 0.978 | 15.3% | 19.9% | 0.964 | TQQQ | SPY/GLD vol regime |
| TLT-SOXL | 0.936 | 16.6% | 17.1% | 0.955 | SOXL | Rate momentum → 3x semis |
| TLT-UPRO | 0.903 | 7.4% | 12.9% | 0.951 | UPRO | Rate momentum → 3x S&P |
| **D12 TIP-UPRO** | 0.895 | 9.7% | 16.7% | **0.9998** | UPRO | Real yield → UPRO, **highest DSR ever** |
| D14 Disinflation-TQQQ | 0.883 | 12.6% | 19.5% | 0.953 | TQQQ | TLT/GLD ratio → TQQQ |
| VCIT-TQQQ | 0.880 | 11.5% | 18.6% | 0.948 | TQQQ | Corp credit → TQQQ |
| IEF-TQQQ | 0.852 | 11.1% | 11.0% | 0.944 | TQQQ | 10yr rate → TQQQ |
| **AGG-UPRO** | 0.841 | 6.1% | **7.4%** | 0.942 | UPRO | **Lowest Track D MaxDD** |
| D11 SOXX-SOXL | 0.818 | 25.1% | 37.2% | 0.966 | SOXL | Semi lead-lag → SOXL (marginal, p=0.048) |
| LQD-TQQQ | 0.803 | 8.8% | 12.6% | 0.936 | TQQQ | IG credit → TQQQ |

**Track D optimal 2-strategy portfolio** (walk-forward validated):
XLK-SOXL @ 60% weight + TLT-TQQQ @ 70% weight = **CAGR 42.8%, MaxDD 22.8%, Sharpe 1.533**.
5.6x the CAGR of TQQQ buy-and-hold at 0.28x the drawdown.

**Track D retirements / falsifications**:
- **D3 TQQQ/TMF ratio MR — RETIRED**. Original backtest showed Sharpe=2.21. Independent replication showed Sharpe=-1.08. Signal direction bug. Do not trust marketed numbers without independent replication.
- **D16 commodity-carry-SOXL — FALSIFIED**. Sharpe=-0.358, MaxDD=61.5%. Energy backwardation has no causal link to semiconductor returns. Signal and vehicle must share a causal mechanism.
- **D2 BTC-momentum, D4-D6 credit-SOXL variants — all REJECTED**. Credit lead-lag does NOT survive 3x SOXL leverage.

**Track C — Structural Arbitrage** (infrastructure built, live scanning)

- Polymarket NegRisk complement arb scanner (7 opportunities found in last live scan)
- Kalshi client with bulk-fetch + per-event fallback
- Closed-end fund (CEF) discount mean-reversion with TLT rate hedge
- Crypto perpetual funding rate pipeline (Binance/OKX/Bybit)
- 351/351 arb tests passing. First live paper trade pending execution bridge validation.

### Portfolio Construction (Track A/B research portfolio)

36 strategies cluster into 18 groups (complete linkage, threshold=0.70). The optimized 18-representative portfolio:

| Metric | All 36 equal-weight | Optimized 18-rep |
|--------|---------------------|-----------------|
| Empirical portfolio Sharpe | ~2.1 | **2.205** |
| Max drawdown | ~6% | **6.3%** |
| Average pairwise correlation | ~0.25 | **0.186** |
| Mechanism families represented | 22 | 18 |

Walk-forward HRP validation (5-fold anchored expanding window): OOS/IS ratio = **1.597** (no overfitting detected — OOS Sharpe exceeds IS Sharpe). Volatility targeting: realized vol 3.8%, scale factor 2.64x to reach 10% target.

The portfolio Sharpe formula with correlation: **SR_P = SR_i x sqrt(N / (1 + (N-1) x rho))**

At avg rho=0.186 with 18 representatives at avg SR=1.087, the formula yields SR≈2.26 — matching the empirical 2.205. The key to reaching SR 2.0+ was reducing avg rho from 0.584 (credit-heavy) to 0.186 by adding genuinely orthogonal mechanism families (F9, F11-F22, F26, F30, F33, F42). Track A research is now considered **complete** — the commodity/macro mechanism space is saturated; further research effort goes to Track D and Track C.

### What Gets Rejected and Why

| Failure mode | Count | Examples |
|-------------|-------|---------|
| DSR < 0.95 (insufficient alpha after trial penalty) | ~20 | Correlation regime, VoV, XLU inverse |
| MaxDD > 15% (2022 bear market too harsh) | ~14 | Factor rotation, asset rotation, pairs, DXY-commodity |
| Sharpe < 0.80 (weak signal) | ~12 | Calendar effects, size rotation, FX-equity, VIX fear gauge |
| Shuffled signal p > 0.05 (no real edge) | ~10 | LQD-TQQQ, treasury auction cycle, sector dispersion |
| Perturbation unstable (over-fit parameters) | ~8 | SPY-TLT-GLD-BIL, BTC-SPY, EURUSD-VGK |
| Falsified (signal in wrong direction) | ~8 | Pre-FOMC TLT drift, turn-of-month, USDJPY-SPY, VIX-MR-TQQQ |
| Leverage transfer failure (3x noise > signal) | 4 | LQD-TQQQ, LQD-UPRO, AGG-SOXL (Track D) |

## Live Portfolio Performance

Data through 2026-04-16 (23 calendar days since inception 2026-03-24).

| Metric | Value | Note |
|--------|-------|------|
| NAV | **$101,268.55** | +1.27% since inception |
| SPY total return (same period) | +7.42% | — |
| 60/40 SPY/TLT (same period) | +4.58% | — |
| Alpha vs SPY | **-6.15%** | Materially behind |
| Alpha vs 60/40 | **-3.31%** | Behind |
| Max Drawdown (realized) | -0.04% | Effectively none |
| Cash | 61% | Deployed only 39% of capital |
| Positions | 14 | All green (no stops triggered) |
| Total trades | 54 | 34 buys, 5 sells, 15 closes |

**Honest assessment**: the research portfolio has an empirical Sharpe of 2.205, but the
live paper book is underperforming benchmarks by ~6% because it stayed 33-69% in cash
through a risk-on rally. A mass-close event on 2026-04-01 (QQQ, XLF, XLRE, XLC, XLI,
SOL-USD) locked in losses right before a rebound — a visible whipsaw the rules engine
needs to resolve. Max drawdown is effectively zero, but that is a symptom of
under-deployment rather than strategy strength. **The gap between research quality and
live execution is the most important item on the current work backlog.**

**Paper trading is early**: 37 of 45 research strategies are logging paper NAV, but
most have only 2-5 days of data — reported Sharpe values are annualizations of tiny
samples and are statistically meaningless until 30+ days accumulate. No strategy has
reached the 30-day gate for promotion yet. SOXX-QQQ lead-lag is the current leader at
5 days.

> Updated daily via [automated reports](reports/).
> Research lab results updated 2026-04-17.

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

## Future Direction: Community Quant Research Lab

The long-term goal is to evolve llm-quant from a solo research program into an **open community-driven quant research lab** — where independent researchers contribute hypotheses, share reproducible evaluations, and build on each other's work.

**What this looks like:**

- **Open research contributions** — Anyone can submit a strategy hypothesis through a standardized template. Contributions go through the same 6-gate robustness pipeline that internal strategies use.
- **Shared evaluation infrastructure** — Reproducible backtests, robustness checks, and walk-forward validation available to all contributors.
- **Paper-trading leaderboards** — Public rankings of strategies in paper trading, scored by risk-adjusted performance over time.
- **Contribution scoring** — Merit-based reputation system that tracks research quality (pass rate, signal novelty, peer review quality), not capital.
- **Agent-assisted review** — Claude-powered review workflows that check submissions for look-ahead bias, overfitting signatures, and methodology gaps before human review.

**What this does NOT mean:**

- No pooled capital. Paper trading only until governance, legal, and audit infrastructure is mature.
- No "hedge fund" structure. This is a research lab — contributors share knowledge, not P&L.
- No timeline promises. Phase 4 (any form of structured deployment) has explicit prerequisites, not a date.

The existing infrastructure — formal lifecycle gates, integrity checks (DSR, CPCV), spec-freeze-before-backtest discipline, append-only experiment registries, and hash-chain auditability — provides the foundation that makes community research trustworthy by default.

See [Community Lab Roadmap](docs/governance/community-lab-roadmap.md) for the full four-phase plan.

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
volatility targeting (**implemented** via 126d realized variance + walk-forward vol targeting to 10%),
portfolio correlation gate (**implemented** via DCC-GARCH), marginal SR contribution gate (**implemented**),
walk-forward validation (**implemented** — 5-fold HRP OOS/IS=1.60, no overfitting detected),
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

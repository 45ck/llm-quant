# Research Tracks: Defensive Alpha vs. Aggressive Alpha

This document formally defines the two parallel research tracks. Both tracks follow the
same lifecycle (mandate → hypothesis → data-contract → research-spec → backtest →
robustness → paper → promote), but with different mandates, gate thresholds, and
position sizing.

---

## Why Two Tracks?

A single conservative filter (MaxDD < 15%) rejects strategies that are statistically
genuine but inherently volatile. A second track with relaxed risk gates — but the same
anti-overfitting gates — captures these strategies without compromising research integrity.

The integrity gates (DSR >= 0.95, CPCV OOS/IS > 0) are non-negotiable on both tracks.
They guard against data snooping. The risk gates (MaxDD, position sizing) are different
because the two tracks serve different portfolio functions.

---

## Track A — Defensive Alpha

**Purpose:** Stable, uncorrelated base returns. Sleep-at-night portfolio.

| Gate | Threshold |
|------|-----------|
| Sharpe | >= 0.80 |
| Max Drawdown | < 15% |
| DSR | >= 0.95 |
| CPCV OOS/IS | > 0 (mean and median) |
| Perturbation stability | >= 3/5 |

**Position sizing:**
- Max position: 10% of NAV (5% crypto, 8% forex)
- Max trade: 2% of NAV
- Cash reserve: >= 5%

**Benchmark:** 60/40 SPY/TLT (monthly rebalanced, total return)

**Return target:** 15-25% annualized, Sharpe > 0.80

**Portfolio allocation:** 70% of total capital

**Current strategies (23 passing as of 2026-04-01, 13 mechanism families):**

| # | Strategy | Sharpe | MaxDD | Family | Mechanism |
|---|---------|--------|-------|--------|-----------|
| 1 | XLK-XLE sector rotation v1 | 1.525 | 11.5% | F12 | Tech/energy ratio momentum |
| 2 | Skip-month TSMOM v1 | 1.331 | 9.9% | F3 | Novy-Marx skip-month vol-scaled |
| 3 | TIP/TLT real yield v1 | 1.313 | 13.3% | F15 | Real yield proxy momentum |
| 4 | Vol-regime v2 | 1.270 | 14.2% | F13 | SPY vs GLD realized vol |
| 5 | LQD-SPY credit lead | 1.250 | 12.4% | F1 | IG bond → US equity |
| 6 | GLD-SLV v4 consensus | 1.197 | 9.6% | F2 | Metals ratio mean reversion |
| 7 | AGG-SPY credit lead | 1.145 | 8.4% | F1 | Total bond → US equity |
| 8 | AGG-QQQ credit lead | 1.080 | 11.2% | F1 | Total bond → tech equity |
| 9 | TLT/SHY curve momentum v1 | 1.044 | 14.3% | F14 | Curve shape ratio momentum |
| 10 | SPY overnight momentum | 1.043 | 8.7% | F5 | Overnight gap microstructure |
| 11 | VCIT-QQQ credit lead | 1.037 | 14.5% | F1 | Corp bond → tech equity |
| 12 | LQD-QQQ credit lead | 1.023 | 13.7% | F1 | IG bond → tech equity |
| 13 | DBA commodity cycle v1 | 1.010 | 13.5% | F11 | Commodity abs. momentum |
| 14 | EMB-SPY credit lead | 1.005 | 9.1% | F1 | EM sovereign → US equity |
| 15 | Credit-spread-regime v1 | 0.990 | 10.8% | F9 | HYG/SHY spread regime |
| 16 | IEF-QQQ rate-tech | 0.979 | 14.5% | F6 | Intermediate rate → tech |
| 17 | TLT-QQQ rate-tech | 0.935 | 11.8% | F6 | Long rate → tech |
| 18 | HYG-SPY credit lead | 0.913 | 14.7% | F1 | HY bond → US equity |
| 19 | HYG-QQQ credit lead | 0.867 | 13.4% | F1 | HY bond → tech equity |
| 20 | SOXX-QQQ lead-lag | 0.861 | 14.4% | F8 | Semis → tech equity |
| 21 | AGG-EFA credit lead | 0.860 | 10.3% | F1 | Total bond → intl equity |
| 22 | TLT-SPY rate momentum | 0.803 | 10.8% | F6 | Long rate → US equity |
| 23 | Behavioral (VIX spike) | 0.700 | ~5% | F7 | Contrarian crisis alpha |

**Portfolio (optimized 15 reps):** Empirical SR=2.184, MaxDD=5.1%, avg ρ=0.187.

---

## Track B — Aggressive Alpha

**Purpose:** Maximum CAGR with higher drawdown tolerance. High-variance upside.

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Sharpe | >= 1.0 | Higher bar to compensate for relaxed drawdown |
| Max Drawdown | < 30% | Accepts larger drawdowns for higher CAGR |
| DSR | >= 0.95 | Unchanged — integrity gate, not risk gate |
| CPCV OOS/IS | > 0 (mean and median) | Unchanged — integrity gate |
| Perturbation stability | >= 3/5 | Unchanged |

**Position sizing:**
- Max position: 15% of NAV (8% crypto, 10% leveraged ETFs)
- Max trade: 3% of NAV
- Cash reserve: >= 3%

**Benchmark:** 100% SPY (growth-oriented, not risk-adjusted)

**Return target:** 40-80% annualized CAGR

**Portfolio allocation:** 30% of total capital

**Universe expansion (beyond Track A):**
- Leveraged equity ETFs: TQQQ, UPRO, SPXL
- Leveraged bond ETFs: TMF, TYD
- Crypto: BTC-USD, ETH-USD
- Concentrated sector bets: single-sector ETFs at full weight

**Near-pass candidates from Track A pipeline:**

| Strategy | Sharpe | MaxDD | Why Failed Track A | Track B verdict |
|----------|--------|-------|--------------------|-----------------|
| K1 QUAL factor rotation | 0.594 | 21.0% | MaxDD gate | Re-examine w/ regime filter |
| O3 commodity rotation | 0.713 | 27.6% | MaxDD gate | V2 with lookback=90 + VIX overlay |
| C7 window=7 | 1.242 | ~9% | Not pre-specified | Re-specify as v2 |

---

## Portfolio Combination

At full deployment, the target combined portfolio:

| Track | Allocation | Expected CAGR | Expected Sharpe |
|-------|-----------|---------------|-----------------|
| Track A (23 strategies, 13 families) | 70% | ~25-35% | ~2.18 |
| Track B (target: 3-5 strategies) | 20% | ~50-80% | ~1.5 |
| Track D (1 strategy: TLT-TQQQ) | 0-10% | ~60-120% | ~1.0 |
| **Combined** | **100%** | **~35-50%** | **~2.0-2.5** |

The combined portfolio achieves asymmetric returns: Track A provides diversified base with
empirical SR=2.184, Track B/D add leveraged upside.

---

## Track Designation in Artifacts

Every mandate.yaml must include a `track` field:

```yaml
track: track_a   # or track_b or track_c or track_d
```

This determines which gate thresholds apply throughout the lifecycle.

---

## Lifecycle Gates by Track

| Stage | Track A | Track B | Track C | Track D |
|-------|---------|---------|---------|---------|
| Mandate | max_drawdown: 0.15 | max_drawdown: 0.30 | max_drawdown: 0.10 | max_drawdown: 0.40 |
| Robustness: Sharpe gate | >= 0.80 | >= 1.00 | >= 1.50 | >= 0.80 |
| Robustness: MaxDD gate | < 15% | < 30% | < 10% | < 40% |
| Robustness: DSR gate | >= 0.95 | >= 0.95 | >= 0.95 | >= 0.90 |
| Robustness: CPCV gate | OOS/IS > 0 | OOS/IS > 0 | OOS/IS > 0 | OOS/IS > 0 |
| Promotion: Paper Sharpe | >= 0.60 | >= 0.80 | >= 1.20 | >= 0.60 |
| Promotion: Canary MaxDD | < 10% | < 20% | < 8% | < 30% |
| Promotion: MAR gate | n/a | n/a | n/a | >= 1.0 after 90d |
| Max holding period | n/a | n/a | n/a | 5 calendar days |

---

## Track C -- Niche Arbitrage

**Purpose:** Capture structural mispricings with near-zero market beta.

| Gate | Threshold |
|------|-----------|
| Sharpe | >= 1.5 |
| Max Drawdown | < 10% |
| DSR | >= 0.95 |
| CPCV OOS/IS | > 0 |
| Perturbation stability | >= 3/5 |
| Beta to SPY | < 0.15 |
| Min trades | >= 50 |

**Benchmark:** Risk-free rate (3-month T-bill)

**Return target:** Risk-free + 5-15% (absolute return)

**Portfolio allocation:** 10-20% of total capital

**Mandate:** `data/strategies/niche-arbitrage/mandate.yaml`

**Full plan:** `docs/governance/track-c-plan.md`

---

## Track D — Sprint Alpha

**Purpose:** Maximize CAGR by re-expressing validated Family 1 and Family 8 signals through 3x leveraged ETFs. Not a new signal search — a re-expression engine. Only signals that have already passed Track A or Track B gates are candidates.

**Mandate:** `data/strategies/sprint-alpha/mandate.yaml`

**Rebalancing reference:** `docs/governance/track-d-rebalancing.md`

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Sharpe | >= 0.80 | Lower bar vs. Track B — leverage amplifies both signal and noise |
| Max Drawdown | < 40% | Accepts aggressive drawdowns; leverage makes 30%+ drawdowns structurally inevitable |
| DSR | >= 0.90 | Slightly relaxed vs. 0.95 — shorter effective history from 3x ETF launch dates |
| CPCV OOS/IS | > 0 (mean and median) | Unchanged — integrity gate, non-negotiable |
| Perturbation stability | >= 3/5 | Unchanged |
| MAR (CAGR/MaxDD) | >= 1.0 after 90 days paper | Kill condition — if leverage isn't producing asymmetric returns, retire immediately |
| Max holding period | <= 5 calendar days | Hard rule — beta decay and volatility drag make multi-week holds structurally losing |

**Position sizing:**
- Max position: 30-50% of NAV (concentrated by design — diversification destroys leveraged returns)
- Max trade: 5% of NAV
- Cash reserve: >= 10% (higher buffer — leveraged drawdowns are fast and deep)
- No more than 2 leveraged positions simultaneously

**Benchmark:** 100% TQQQ buy-and-hold (the monster baseline — if you can't beat passive TQQQ, the strategy adds no value)

**Return target:** 60-120% annualized CAGR (gross)

**Portfolio allocation:** 0% until paper trading gate passes; target 10-15% of total capital at full deployment

**Universe:**
- Leveraged equity: TQQQ (3x QQQ), UPRO (3x SPY), SOXL (3x semis)
- Leveraged bond: TMF (3x TLT), TLTW (3x TLT + covered calls)
- No inverse ETFs — short signals expressed via exit/cash, not SQQQ/SPXS

**Signal sources (re-expression only):**
- Family 1 (Cross-Asset Information Flow): LQD/AGG/HYG lead signals re-expressed as TQQQ/UPRO entries
- Family 8 (Non-Credit Lead-Lag): SOXX-QQQ lead re-expressed as SOXL entries
- No new signal research in Track D — only re-expression of promoted Track A/B strategies

**Kill conditions (any one triggers immediate retirement):**
1. MAR < 1.0 after 90 days of paper trading
2. Three consecutive weekly losses exceeding 15% each
3. Single-session loss > 20%
4. Beta decay drag > 5% annualized vs. 3x theoretical return (measured monthly)

**Status:** 1 strategy passing all 6 Track D gates (TLT-TQQQ: Sharpe=1.030, MaxDD=10.16%, DSR=0.965). Advancing to paper trading. LQD-SPY signal → TQQQ in backtest phase.

---

## Portfolio Combination (Four Tracks)

| Track | Allocation | Expected Sharpe | Beta | Status |
|-------|-----------|-----------------|------|--------|
| A (23 strategies, 13 families) | 60% | ~2.18 | ~0.4 | Paper trading |
| B (3-5 target) | 20% | ~1.5 | ~0.8 | Research |
| C (2-3 target) | 10-20% | ~2.0 | ~0.0 | Planning |
| D (1 strategy: TLT-TQQQ) | 0-15% | ~1.0 | ~2.0 | Paper trading |
| **Combined** | **100%** | **~2.0-2.5** | ~0.4 | |

Track D allocation is gated: 0% until paper trading gate passes, then grows from 5% → 15% over 6 months contingent on MAR >= 1.0.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-26 | Initial dual-track framework. |
| 2.0 | 2026-03-27 | Added Track C (Niche Arbitrage). |
| 3.0 | 2026-03-30 | Added Track D (Sprint Alpha — leveraged re-expression). |
| 4.0 | 2026-04-01 | Updated Track A to 23 strategies across 13 families (SR=2.184). Track D first pass (TLT-TQQQ). |

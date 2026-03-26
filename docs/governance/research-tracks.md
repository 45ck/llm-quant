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

**Current strategies (11 passing as of 2026-03-26):**

| Strategy | Sharpe | MaxDD | Mechanism |
|---------|--------|-------|-----------|
| LQD-SPY credit lead | 1.250 | 12.4% | IG bond → US equity |
| AGG-SPY credit lead | 1.145 | 8.4% | Total bond → US equity |
| SPY overnight momentum | 1.043 | 8.7% | Overnight gap microstructure |
| AGG-QQQ credit lead | 1.080 | 11.2% | Total bond → tech equity |
| VCIT-QQQ credit lead | 1.037 | 14.5% | Corp bond → tech equity |
| LQD-QQQ credit lead | 1.023 | 13.7% | IG bond → tech equity |
| EMB-SPY credit lead | 1.005 | 9.1% | EM sovereign → US equity |
| HYG-SPY credit lead | 0.913 | 14.7% | HY bond → US equity |
| AGG-EFA credit lead | 0.860 | 10.3% | Total bond → intl equity |
| HYG-QQQ credit lead | 0.867 | 13.4% | HY bond → tech equity |
| SOXX-QQQ lead-lag | 0.861 | 14.4% | Semis → tech equity |

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
| Track A (11 strategies) | 70% | ~20% | ~2.3 |
| Track B (target: 3-5 strategies) | 30% | ~50-80% | ~1.5 |
| **Combined** | **100%** | **~30-40%** | **~2.0** |

The combined portfolio achieves asymmetric returns: Track A limits downside, Track B
provides leveraged upside.

---

## Track Designation in Artifacts

Every mandate.yaml must include a `track` field:

```yaml
track: track_a   # or track_b
```

This determines which gate thresholds apply throughout the lifecycle.

---

## Lifecycle Gates by Track

| Stage | Track A | Track B |
|-------|---------|---------|
| Mandate | max_drawdown: 0.15 | max_drawdown: 0.30 |
| Robustness: Sharpe gate | >= 0.80 | >= 1.00 |
| Robustness: MaxDD gate | < 15% | < 30% |
| Robustness: DSR gate | >= 0.95 | >= 0.95 |
| Robustness: CPCV gate | OOS/IS > 0 | OOS/IS > 0 |
| Promotion: Paper Sharpe | >= 0.60 | >= 0.80 |
| Promotion: Canary MaxDD | < 10% | < 20% |

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-26 | Initial dual-track framework. Track A = current 11 strategies. Track B = new aggressive alpha track with relaxed risk gates. |

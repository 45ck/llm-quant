# D10 XLK/XLE Sector Rotation — Paper Monitor Snapshot

**Date:** 2026-04-17
**Strategy slug:** `xlk-xle-sector-rotation-v1`
**Paper start:** 2026-04-04
**Bead:** llm-quant-whzy (P2)

> **Note on slug mapping:** The user task references "D10 xlk-xle-sector-rotation" with the
> SOXL-leveraged backtest profile (Sharpe=1.171, CAGR=27.7%, MaxDD=22.6%). The *active*
> paper-trading track on disk is `xlk-xle-sector-rotation-v1` (non-leveraged rotation into
> GLD/DBA/SPY/cash — robustness Sharpe=1.525, MaxDD=11.5%). The leveraged twin
> `xlk-xle-soxl-rotation-v1` has no `paper-trading.yaml` yet. This report evaluates against
> BOTH backtest references so the call is invariant to which version we're tracking.

---

## 1. Current Paper Metrics

| Metric | Value |
|---|---|
| Initial NAV | $100,000.00 |
| Current NAV | $99,414.85 |
| Peak NAV | $100,000.00 |
| Cumulative return | **-0.585%** |
| Current drawdown from peak | 0.585% |
| Calendar days elapsed | 13 |
| Trading/signal days logged | 3 (Apr 2, Apr 7, Apr 14) |
| Trades executed | 1 (rotation event) |
| Running Sharpe (annualized, daily convention) | -4.36 |
| Hit rate | 1 / 3 = 33.3% |

### Daily log detail

| Date | Day | Regime | NAV | Daily PnL | Daily Ret | Allocation |
|---|---|---|---|---|---|---|
| 2026-04-02 | 1 | inflation | 99,105.98 | -894.02 | -0.89% | GLD 50% / DBA 30% |
| 2026-04-07 | 2 | inflation | 98,902.22 | -203.76 | -0.21% | GLD 50% / DBA 30% |
| 2026-04-14 | 3 | neutral | 99,414.85 | +512.63 | +0.52% | SPY 45% |

The Sharpe figure is mechanically meaningless at n=3 — reported only because paper-trading.yaml persists it. Do not read regime signal from it.

---

## 2. Proportional Comparison to Backtest

Expected return over elapsed window = backtest CAGR × (days / 252).
Expected 1σ band = backtest vol × √(days / 252), where vol = CAGR / Sharpe.

Using **9 approximate trading days** (Apr 4 → Apr 17 spans two full weeks; 3 signal points captured).

### Reference A — SOXL-leveraged version (user's stated D10 backtest)

| Backtest metric | Value |
|---|---|
| CAGR | 27.7% |
| Sharpe | 1.171 |
| MaxDD | 22.6% |
| Implied annualized vol | ~23.6% |

| Projection | Value |
|---|---|
| Expected return (9 td) | +0.99% |
| Expected 1σ band | ±4.47% |
| **Actual return** | **-0.585%** |
| **Z-score** | **-0.35 σ** |

### Reference B — v1 non-leveraged (what's actually on the paper track)

| Backtest metric | Value |
|---|---|
| Implied CAGR (10y, total_return=142%) | 9.24% |
| Sharpe | 1.525 |
| MaxDD | 11.5% |
| Implied annualized vol | ~6.0% |

| Projection | Value |
|---|---|
| Expected return (9 td) | +0.33% |
| Expected 1σ band | ±1.15% |
| **Actual return** | **-0.585%** |
| **Z-score** | **-0.80 σ** |

Both references give z-scores comfortably inside 1σ. The bigger deviation (Reference B, non-leveraged) still shows the actual outcome is only 0.8σ below expectation — a completely ordinary coin-flip over a ~2-week window.

---

## 3. Statistical Assessment

**Verdict: WITHIN NORMAL NOISE.**

- Z-score vs SOXL reference: -0.35σ (well inside ±1σ)
- Z-score vs v1 reference: -0.80σ (inside ±1σ)
- Drawdown of 0.585% is <1/20th of backtest MaxDD on both references — not structurally concerning
- Sample size (n=3 signal days) is far too small to make any inference about Sharpe decay, hit-rate drift, or regime breakdown. PBO-style arguments require at minimum ~30 observations.
- Regime flow (inflation → inflation → neutral) is behaving as designed; the strategy is actively responding to XLK/XLE momentum dynamics rather than stuck in one sleeve.

---

## 4. Signal Diagnostics

XLK/XLE momentum has rotated from **-0.168 → -0.118 → -0.011** across the three signal dates — mean-reverting toward zero. `vs_sma(20d)` flipped positive on Apr 14 (+0.105), triggering the regime exit from inflation into neutral. This is mechanism-consistent behavior: the rotation signal is doing what the spec predicted.

No divergence between signal and allocation. No stale data. No skipped rebalances.

---

## 5. Recommendation

**CONTINUE. No tightening, no retirement trigger.**

- The -0.59% draw is indistinguishable from noise at n=3.
- Continue accumulating paper observations at the current rebalance cadence.
- **Next review trigger:** 2026-05-04 (30 calendar days of paper), OR any of the following early triggers:
  - Cumulative return < -3% (would approach -2σ on either reference)
  - Drawdown > 5%
  - 5 consecutive negative signal days
  - Signal-vs-regime divergence (allocated sleeve not matching regime classification)
- Do **not** re-tune parameters, shift the universe, or second-guess allocation weights based on a 3-point sample. That's the data-snooping path the lifecycle is designed to prevent.
- Flag for the team: clarify whether the SOXL-leveraged version is also being paper-traded anywhere, or whether D10-SOXL is still pre-paper. If the team's internal "D10" refers to the SOXL vehicle, we need a separate paper track for that slug.

---

## 6. Next Steps

- Keep bead **llm-quant-whzy** OPEN (ongoing monitor).
- Append future weekly snapshots to this file or create rolling `paper-monitor-YYYY-MM-DD.md` artifacts.
- Escalate to CONCERN only if actual falls more than 2σ below expectation on the Reference B (tighter) band — i.e., cumulative return < ~-2.0% in the near term, widening as the window grows.

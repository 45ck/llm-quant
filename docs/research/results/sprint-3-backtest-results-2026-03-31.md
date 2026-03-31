# Sprint 3 Backtest Results — 2026-03-31

**Objective:** Iterate promising Sprint 1+2 failures (H3.1-v2, H4.4-v2) and backtest unbacktested F1 credit lead-lag variants (AGG-QQQ, AGG-EFA).

**Result: 1 of 4 passed.** AGG-QQQ credit lead-lag passes Track A gates.

---

## Summary Table

| Hypothesis | Family | Track | Sharpe | Gate | MaxDD | Gate | DSR | Trades | Verdict |
|-----------|--------|-------|--------|------|-------|------|-----|--------|---------|
| AGG-QQQ Credit Lead | F1 | A | 0.888 | >0.80 | 12.30% | <15% | 0.965 | 76 | **PASS** |
| AGG-EFA Credit Lead | F1 | A | 0.724 | >0.80 | 12.16% | <15% | 0.931 | 72 | **FAIL** |
| H4.4-v2 GARCH Persistence | F4 | A | 0.723 | >0.80 | 10.23% | <15% | 0.931 | 42 | **FAIL** (improved) |
| H3.1-v2 Relaxed TSMOM | F3 | A | 0.593 | >0.80 | 3.82% | <15% | 0.875 | 432 | **FAIL** (regressed) |

## Benchmark Comparison

| Strategy | Return | Benchmark | Excess |
|----------|--------|-----------|--------|
| AGG-QQQ Credit Lead | +52.55% | 0% (cash) | +52.55% |
| AGG-EFA Credit Lead | +28.82% | 0% (cash) | +28.82% |
| H4.4-v2 GARCH | +34.38% | +38.18% (60/40) | -3.80% |
| H3.1-v2 TSMOM | +7.00% | +22.93% (60/40) | -15.94% |

---

## Detailed Analysis

### AGG-QQQ Credit Lead-Lag (Family 1) — PASS Track A

- **Experiment**: 6d05646b (trial #1, AGG+QQQ, 5 years)
- **Sharpe 0.888** — passes 0.80 gate with margin
- **MaxDD 12.30%** — within 15% Track A limit
- **Sortino 1.309** — passes 1.0 gate
- **Calmar 0.865** — passes 0.5 gate
- **DSR 0.965** — passes 0.95 anti-overfitting gate
- **Profit Factor 2.15** — winners 2x losers
- **76 trades** — adequate statistical significance
- **Cost sensitivity**: Sharpe degrades 0.888 → 0.765 at 3x costs (13.9% drop)
- **Note**: Benchmark shows 0% (likely no benchmark weights in spec). Actual benchmark comparison needed.
- **Assessment**: AGG leading QQQ is the strongest AGG-based credit lead-lag pair. QQQ's rate sensitivity amplifies the AGG signal — when broad bonds rally (AGG up), QQQ benefits from both risk-on and rate-decline tailwinds.
- **Next step**: Advance to robustness testing (CPCV, regime splits, shuffled returns).

### AGG-EFA Credit Lead-Lag (Family 1) — FAIL

- **Experiment**: f33af5e4 (trial #1, AGG+EFA, 5 years)
- **Sharpe 0.724** — misses 0.80 gate by 0.08
- **DSR 0.931** — misses 0.95 gate
- **Sortino 1.055** — passes 1.0 gate
- **MaxDD 12.16%** — within limits
- **Diagnosis**: AGG's US credit signal has weaker forward predictive power for international equities (EFA) than for US equities (QQQ/SPY). Cross-border information flow attenuates the signal.
- **Path forward**: Consider as part of a multi-pair credit lead-lag portfolio rather than standalone.

### H4.4-v2 GARCH Regime with Persistence Filter (Family 4) — FAIL (improved)

- **Experiment**: 0b92aa69 (trial #1, SPY+SHY, 5 years)
- **Sharpe 0.723** — up from 0.605 (v1), but still misses 0.80
- **Trades 42** — down from 77 (v1), persistence filter working as designed
- **MaxDD 10.23%** — excellent
- **Profit Factor 2.69** — very strong
- **Sortino 1.057** — passes 1.0 gate
- **Calmar 0.716** — passes 0.5 gate
- **DSR 0.931** — improved from 0.891, but still misses 0.95
- **Improvement summary**: Persistence filter + 80th pctile improved Sharpe by 19.5% (0.605→0.723) and reduced trades by 45% (77→42). The v1 cost anomaly (0.802 at 1.5x) is partially replicated at base costs.
- **Path forward**: Further iteration possible — try 85th percentile, 5-day persistence, or combine with VIX term structure signal (H4.1).

### H3.1-v2 Relaxed TSMOM (Family 3) — FAIL (regressed)

- **Experiment**: b81cf287 (trial #1, 14 symbols, 5 years)
- **Sharpe 0.593** — WORSE than H3.1's 0.719! Relaxed params degraded performance.
- **MaxDD 3.82%** — still excellent risk control
- **432 trades** — 2x more than H3.1 (227), allow_short doubled trade count
- **Profit Factor 1.43** — weaker than H3.1's implied >1.5
- **Diagnosis**: Allowing short positions was counterproductive. In a predominantly uptrending asset universe (2022-2026 includes strong 2023-2025 rally), shorts generated losses. Higher vol target (15% vs 10%) didn't compensate. The original H3.1 long-only constraint was actually protective, not just conservative.
- **F3 status**: H3.1 (Sharpe 0.719) remains the best F3 candidate. Further iteration should focus on signal quality (better lookback blending, adaptive thresholds) rather than parameter relaxation.

---

## Key Takeaways

1. **AGG-QQQ passes Track A** — first new passing strategy from Sprint 3. However, this is still Family 1 (credit lead-lag), NOT a new mechanism family. It adds another F1 strategy but doesn't help with portfolio diversification away from the credit-heavy portfolio.

2. **Persistence filter validates the cost anomaly hypothesis**: H4.4-v2's improvement (Sharpe +19.5%, trades -45%) confirms that the v1 cost anomaly was caused by marginal trade filtering. The persistence filter partially replicates this but doesn't fully close the gap to the 0.80 gate.

3. **Relaxing TSMOM parameters hurts**: Allowing shorts in a predominantly uptrending universe degraded H3.1's performance. This is a useful lesson — parameter relaxation isn't always improvement.

4. **F1 continues to dominate**: The passing strategies in this program are overwhelmingly F1 credit lead-lag. Families 2-7 continue to fail. The diversification problem persists.

---

## Cumulative Sprint Results (Sprints 1-3)

| Sprint | Tested | Passed | Best Result |
|--------|--------|--------|-------------|
| Sprint 1 | 4 | 0 | H3.1 (Sharpe 0.719) |
| Sprint 2 | 4 | 0 | H4.4 (Sharpe 0.605, 0.802@1.5x) |
| Sprint 3 | 4 | 1 | AGG-QQQ (Sharpe 0.888) ← PASS |
| **Total** | **12** | **1** | |

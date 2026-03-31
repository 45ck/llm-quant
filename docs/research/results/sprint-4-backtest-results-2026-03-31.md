# Sprint 4 Backtest Results — 2026-03-31

**Objective:** Backtest 4 unbacktested F1 credit lead-lag variants (AGG-SPY, EMB-SPY, HYG-QQQ, HYG-SPY-5D).

**Result: 2 of 4 passed.** AGG-SPY and EMB-SPY pass Track A gates.

---

## Summary Table

| Hypothesis | Family | Track | Sharpe | Gate | MaxDD | Gate | DSR | Trades | Verdict |
|-----------|--------|-------|--------|------|-------|------|-----|--------|---------|
| AGG-SPY Credit Lead | F1 | A | 1.012 | >0.80 | 8.34% | <15% | 0.981 | 74 | **PASS** |
| EMB-SPY Credit Lead | F1 | A | 0.829 | >0.80 | 12.85% | <15% | 0.955 | 70 | **PASS** |
| HYG-SPY-5D Credit Lead | F1 | A | 0.752 | >0.80 | 14.02% | <15% | 0.937 | 58 | **FAIL** |
| HYG-QQQ Credit Lead | F1 | A | 0.609 | >0.80 | 17.44% | <15% | 0.893 | 58 | **FAIL** |

## Benchmark Comparison

| Strategy | Return | Benchmark | Excess |
|----------|--------|-----------|--------|
| AGG-SPY | +45.78% | +38.18% (60/40) | +7.59% |
| EMB-SPY | +34.60% | +38.18% (60/40) | -3.59% |
| HYG-SPY-5D | +32.24% | +38.18% (60/40) | -5.95% |
| HYG-QQQ | +31.09% | 0% (cash) | +31.09% |

---

## Detailed Analysis

### AGG-SPY Credit Lead-Lag — PASS Track A (strongest Sprint 4 result)

- **Experiment**: 7e7e16f4 (trial #1)
- **Sharpe 1.012** — passes Track A (0.80) AND Track B (1.0) gates
- **Sortino 1.529, Calmar 1.132** — excellent risk-adjusted metrics
- **MaxDD 8.34%** — well within limits, shortest DD duration (103 days)
- **Profit Factor 2.74** — highest of all Sprint 4 strategies
- **DSR 0.981** — passes 0.95 gate with margin, robust at all cost levels
- **Excess return +7.59%** — beats 60/40 benchmark on both absolute AND risk-adjusted basis
- **Assessment**: AGG→SPY is a clean, strong credit lead-lag signal. Broad bond health (AGG) reliably leads SPY. This is the strongest F1 variant discovered in Sprint 3+4.

### EMB-SPY Credit Lead-Lag — PASS Track A (marginal)

- **Experiment**: 6e14fdd0 (trial #1)
- **Sharpe 0.829** — passes 0.80 gate, but marginal
- **DSR 0.955** — passes 0.95 gate at 1.0x only (fails at 2.0x+)
- **Sortino 1.221, Calmar 0.574** — pass respective gates
- **MaxDD 12.85%** — within 15% limit but elevated
- **DD Duration 310 days** — nearly a year underwater
- **Assessment**: EMB (emerging market bonds) leads SPY through a global risk appetite channel. Weaker than AGG-SPY but passes gates. Cross-border information flow works but with more noise.

### HYG-SPY-5D Credit Lead-Lag — FAIL

- **Experiment**: 3c22a605 (trial #1)
- **Sharpe 0.752** — misses 0.80 by 0.05
- **DSR 0.937** — misses 0.95 gate
- **MaxDD 14.02%** — within limits but tight
- **Profit Factor 2.11** — solid edge exists
- **Diagnosis**: HYG→SPY has a real signal (Profit Factor 2.11) but it's noisier than AGG→SPY or LQD→SPY. HYG's higher volatility introduces false signals that degrade Sharpe.

### HYG-QQQ Credit Lead-Lag — FAIL

- **Experiment**: e480b8d2 (trial #1)
- **Sharpe 0.609** — well below 0.80 gate
- **MaxDD 17.44%** — exceeds Track A's 15% limit
- **DSR 0.893** — fails 0.95 gate
- **Diagnosis**: HYG→QQQ is the noisiest credit lead-lag pairing. Both HYG and QQQ are high-beta assets — their combined volatility amplifies drawdowns beyond Track A tolerance. Would need Track B evaluation but Sharpe is too low even for Track B (requires 1.0).

---

## F1 Credit Lead-Lag Ranking (all tested pairs)

| Rank | Pair | Sharpe | MaxDD | DSR | Status |
|------|------|--------|-------|-----|--------|
| 1 | AGG-SPY | 1.012 | 8.34% | 0.981 | **PASS** |
| 2 | AGG-QQQ | 0.888 | 12.30% | 0.965 | **PASS** |
| 3 | EMB-SPY | 0.829 | 12.85% | 0.955 | **PASS** |
| 4 | HYG-SPY-5D | 0.752 | 14.02% | 0.937 | FAIL |
| 5 | AGG-EFA | 0.724 | 12.16% | 0.931 | FAIL |
| 6 | HYG-QQQ | 0.609 | 17.44% | 0.893 | FAIL |

**Pattern**: AGG-based pairs outperform HYG-based pairs. Investment-grade/aggregate credit provides a cleaner leading signal than high-yield credit. SPY followers outperform QQQ/EFA followers (lower volatility = tighter risk metrics).

---

## Cumulative Sprint Results (Sprints 1-4)

| Sprint | Tested | Passed | Best Result |
|--------|--------|--------|-------------|
| Sprint 1 | 4 | 0 | H3.1 (Sharpe 0.719) |
| Sprint 2 | 4 | 0 | H4.4 (Sharpe 0.605) |
| Sprint 3 | 4 | 1 | AGG-QQQ (Sharpe 0.888) |
| Sprint 4 | 4 | 2 | AGG-SPY (Sharpe 1.012) |
| **Total** | **16** | **3** | |

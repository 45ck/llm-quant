# Sprint 1 Backtest Results — 2026-03-31

**Objective:** Backtest 4 hypotheses from 4 different families to begin portfolio diversification away from F1-heavy portfolio.

**Result: 0 of 4 passed.** All four strategies fail their respective track gates.

---

## Summary Table

| Hypothesis | Family | Track | Sharpe | Gate | MaxDD | Gate | DSR | Trades | Verdict |
|-----------|--------|-------|--------|------|-------|------|-----|--------|---------|
| H3.1 Vol-Scaled TSMOM | F3 | A | 0.719 | >0.80 | 2.77% | <15% | 0.698 | 227 | **FAIL** (closest) |
| H7.2 RSI-2 Contrarian | F7 | A | -0.023 | >0.80 | 4.20% | <15% | 0.184 | 636 | **KILL** |
| H4.2 VRP-SPY Timing | F4 | B | 0.539 | >1.00 | 10.69% | <30% | 0.604 | 32 | **FAIL** |
| H6.4 Macro Barometer | F6 | A | 0.446 | >0.80 | 20.79% | <15% | 0.649 | 283 | **FAIL** |

## Benchmark Comparison

| Strategy | Return | 60/40 Benchmark | Excess |
|----------|--------|-----------------|--------|
| H3.1 TSMOM | +7.46% | +22.93% | -15.48% |
| H7.2 RSI-2 | -0.35% | +22.93% | -23.29% |
| H4.2 VRP | +16.08% | +66.51% (SPY) | -50.44% |
| H6.4 Macro | +19.11% | +22.93% | -3.82% |

---

## Detailed Analysis

### H3.1 Vol-Scaled TSMOM (Family 3) — FAIL (closest to passing)

- **Experiment**: 0a96e85e (trial #3, 14 symbols, 5 years)
- **Sharpe 0.719** — misses 0.80 gate by 0.08
- **MaxDD 2.77%** — excellent risk control, well within 15% limit
- **Calmar 0.627** — passes 0.5 gate
- **Annualized return 1.74%** — far below 15-25% target
- **Cost insensitive** — Sharpe drops only 7.8% at 3x costs (0.719 → 0.663)
- **Diagnosis**: Strategy is too conservative. Long-only constraint + 10% vol target + 0.2 flat threshold keeps it mostly in cash. Good risk properties but insufficient return.
- **Path forward**: Consider higher vol target (15-20%), allow short positions, or try H3.5 (skip-month TSMOM) as the F3 backup.

### H7.2 RSI-2 Contrarian (Family 7) — KILL

- **Experiment**: ca0f5f83 (trial #3, 13 symbols, 5 years)
- **Sharpe -0.023** — negative Sharpe, strategy destroys value
- **636 trades** — adequate sample, signal is genuinely worthless
- **Win rate 27.8%** — very low, mean-reversion not occurring
- **Diagnosis**: RSI(2) extreme oversold in uptrending ETFs does not produce reliable bounce signals in this universe over this period. The golden cross filter may be too restrictive, or the mechanism is simply crowded/arbitraged away in liquid ETFs.
- **Path forward**: Kill H7.2. Try H7.1 (VIX percentile spike reversion) or H7.6 (sector dispersion regime) as F7 alternatives.

### H4.2 VRP-SPY Timing (Family 4) — FAIL

- **Experiment**: a2af94ce (trial #3, 6 symbols, 5 years)
- **Sharpe 0.539** — far below Track B's 1.0 gate
- **Only 32 trades** — low trade count, similar to H4.3's problem
- **Win rate 31.2%** — low, but profit factor 1.92 suggests winners are larger
- **MaxDD 10.69%** — acceptable for Track B
- **Diagnosis**: VRP signal is real (profit factor 1.92) but too infrequent and too weak for a standalone strategy. The 2022 regime (rising rates, VRP compression) hurt significantly. 643-day drawdown duration.
- **Path forward**: Consider combining VRP with other F4 signals (H4.4 GARCH, H4.6 VIX percentile) into a composite vol regime strategy rather than running standalone.

### H6.4 Macro Momentum Barometer (Family 6) — FAIL

- **Experiment**: 6fabc353 (trial #2, 5 symbols loaded of 9)
- **Sharpe 0.446** — well below 0.80 gate
- **MaxDD 20.79%** — exceeds 15% Track A limit
- **Note**: Only 5 of 9 symbols loaded in trial #2 (DBC, EFA, AGG, HYG missing). Trial #1 (da444e84) with full symbols showed similar Sharpe (0.415).
- **Diagnosis**: The 4-barometer regime classification doesn't generate useful timing signals. The strategy underperforms even passive 60/40. The macro momentum concept as a standalone rotation signal lacks alpha.
- **Path forward**: Try H6.2 (cyclical/defensive ratio timing — more parsimonious) or H6.1 (real yield threshold rotation) as F6 alternatives.

---

## Key Takeaways

1. **Live Sharpe haircut is real**: Expected backtested Sharpes (0.80-1.20) collapsed to 0.45-0.72 in these 5-year backtests. The portfolio-correlation-analysis estimate of "50% haircut from backtest to live" is confirmed — even in backtest, these strategies underperform expectations.

2. **The 2022 regime was devastating**: All four strategies struggled through 2022 (rising rates, equity drawdown, VRP compression, trend reversal). This is a legitimate stress test — strategies must survive adverse regimes.

3. **Trade count is a recurring bottleneck**: H4.2 (32 trades) and H6.4 (68-283 trades) both suffer from insufficient or low-quality signal generation. F4 strategies in particular generate too few trades for statistical significance.

4. **H3.1 is the most promising failure**: Sharpe 0.72 with 2.77% max drawdown suggests the TSMOM mechanism works but the implementation is too conservative. This is the most likely candidate for iteration.

5. **Cost sensitivity is NOT the problem**: All four strategies show minimal cost degradation. The issue is signal quality, not execution costs.

---

## Sprint 2 Recommendations

Based on Sprint 1 failures, prioritize backup hypotheses from each family:

| Rank | Hypothesis | Family | Rationale |
|------|-----------|--------|-----------|
| 1 | H3.5 Skip-month TSMOM | F3 | Alternative TSMOM construction, highest expected SR in F3 |
| 2 | H6.2 Cyclical/defensive ratio | F6 | More parsimonious than H6.4 (2 inputs vs 4) |
| 3 | H4.4 GARCH regime sizing | F4 | Different F4 mechanism, well-established model |
| 4 | H5.4 Turn-of-month | F5 | New family (F5), 120 trades expected, well-documented |

Also consider:
- Iterating H3.1 with less conservative parameters (higher vol target, allow shorts)
- Composite F4 strategy combining VRP + GARCH + VIX percentile signals

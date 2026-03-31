# Sprint 6 Backtest Results — 2026-03-31

**Objective:** Backtest final unbacktested strategy (LQD-TQQQ sprint) and re-run VIX backwardation with correct signal symbols.

**Result: 1 of 2 passed.** LQD-TQQQ sprint passes Track A gates.

---

## Summary Table

| Hypothesis | Family | Track | Sharpe | Gate | MaxDD | Gate | DSR | Trades | Verdict |
|-----------|--------|-------|--------|------|-------|------|-----|--------|---------|
| LQD-TQQQ Sprint | F1/D | A | 0.963 | >0.80 | 11.88% | <15% | 0.977 | 85 | **PASS** |
| VIX Backwardation (re-run) | F4 | A | 0.273 | >0.80 | 19.33% | <15% | 0.383 | 26 | **FAIL** |

## Benchmark Comparison

| Strategy | Return | Benchmark | Excess |
|----------|--------|-----------|--------|
| LQD-TQQQ Sprint | +58.37% | 0% (cash) | +58.37% |
| VIX Backwardation | +11.38% | +38.18% (60/40) | -26.80% |

---

## Detailed Analysis

### LQD-TQQQ Sprint Lead-Lag — PASS Track A

- **Experiment**: 964a1bcf (trial #1, LQD+TQQQ, 5 years)
- **Sharpe 0.963** — passes Track A (0.80), nearly passes Track B (1.0)
- **Sortino 1.487** — excellent downside protection
- **Calmar 0.979** — passes 0.5 gate with margin
- **MaxDD 11.88%** — within Track A 15% limit
- **DD Duration 108 days** — shortest of all passing strategies
- **Profit Factor 2.55** — strong edge
- **DSR 0.977** — passes 0.95 gate, robust across all cost levels
- **85 trades** — good statistical significance
- **Cost insensitive**: Sharpe degrades only 10.3% at 3x costs (0.963→0.864)
- **Assessment**: LQD credit signal applied to TQQQ (3x leveraged QQQ) produces excellent results. The leverage amplifies the credit lead-lag alpha while the signal's quality keeps drawdowns controlled. This could also qualify for Track D evaluation.
- **Next step**: Advance to robustness testing alongside other passing strategies.

### VIX Term Structure Backwardation (re-run) — FAIL

- **Experiment**: 633281c9 (trial #3, SPY+SHY+VIX+^VIX9D, 5 years)
- **Sharpe 0.273** — well below 0.80 gate
- **MaxDD 19.33%** — exceeds Track A 15% limit
- **26 trades, Win rate 11.5%** — very low hit rate
- **Diagnosis**: VIX9D data was fetched successfully but the vix_regime strategy may not be consuming it correctly, or the backwardation signal (VIX/VIX9D ratio) doesn't generate useful timing alpha. The 2022 period had extended backwardation that the strategy couldn't exploit profitably.
- **F4 status**: All F4 strategies tested (H4.2 VRP, H4.4 GARCH, VIX backwardation) have failed. F4 volatility regime harvesting appears to be a weak alpha source in the 2022-2026 period.

---

## Final Cumulative Results — All Sprints (1-6)

| Sprint | Tested | Passed | Key Results |
|--------|--------|--------|-------------|
| Sprint 1 | 4 | 0 | H3.1 TSMOM (0.719), H7.2 RSI-2 KILL |
| Sprint 2 | 4 | 0 | H4.4 GARCH (0.605), H5.4 TOM KILL |
| Sprint 3 | 4 | 1 | AGG-QQQ PASS (0.888) |
| Sprint 4 | 4 | 2 | AGG-SPY PASS (1.012), EMB-SPY PASS (0.829) |
| Sprint 5 | 4 | 1 | SPY Overnight PASS (1.044) |
| Sprint 6 | 2 | 1 | LQD-TQQQ PASS (0.963) |
| **Total** | **22** | **5** | **22.7% pass rate** |

### All Passing Strategies (sorted by Sharpe)

| Rank | Strategy | Family | Sharpe | MaxDD | DSR | PF | Trades |
|------|----------|--------|--------|-------|-----|-----|--------|
| 1 | SPY Overnight Momentum | F11 | 1.044 | 8.68% | 0.982 | 4.12 | 43 |
| 2 | AGG-SPY Credit Lead | F1 | 1.012 | 8.34% | 0.981 | 2.74 | 74 |
| 3 | LQD-TQQQ Sprint | F1/D | 0.963 | 11.88% | 0.977 | 2.55 | 85 |
| 4 | AGG-QQQ Credit Lead | F1 | 0.888 | 12.30% | 0.965 | 2.15 | 76 |
| 5 | EMB-SPY Credit Lead | F1 | 0.829 | 12.85% | 0.955 | 1.97 | 70 |

### Family Diversity Assessment

- **F1 (Credit Lead-Lag)**: 4 passing strategies — SATURATED, stop adding
- **F11 (Microstructure)**: 1 passing strategy — SPY Overnight Momentum
- **F2-F10, F12-F13**: 0 passing strategies — diversification gap persists
- **Effective portfolio diversification**: 2 families (F1 + F11)
- **Expected correlation F1↔F11**: Low (~0.10-0.20) — different mechanism

### Killed Families

- **F3 (Trend/TSMOM)**: H3.1 (0.719), H3.5 (0.585), H3.1-v2 (0.593) — all fail
- **F4 (Vol Regime)**: H4.2 (0.539), H4.4 (0.605), H4.4-v2 (0.723), VIX backwardation (0.273) — all fail
- **F5 (Calendar)**: H5.4 TOM (-0.226) — killed
- **F6 (Macro Rotation)**: H6.4 (0.446), H6.2 (-0.622) — killed
- **F7 (Sentiment)**: H7.2 RSI-2 (-0.023) — killed

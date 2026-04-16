# Track D Batch Paper-Performance Snapshot — 2026-04-17

Monitors 14 Track D strategies currently in paper trading (beads llm-quant-twtl). Compares actual paper performance against proportional backtest expectations.

## Status Summary

- **OK**: 0
- **WATCH**: 0
- **CONCERN**: 0
- **DORMANT**: 0
- **NO_TRADES**: 1
- **MISSING**: 13

## Per-Strategy Snapshot

| # | Slug | Status | Days | Entries | Actual CumRet | Expected CumRet | Z | Paper Sharpe | Current DD | Hit Rate | Last Signal |
|---|------|--------|------|---------|---------------|-----------------|---|--------------|------------|----------|-------------|
| 1 | `tsmom-upro-trend-v1` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 2 | `xlk-xle-soxl-rotation-v1` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 3 | `agg-tqqq-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 4 | `tlt-tqqq-sprint` | NO_TRADES | 17 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 5 | `d15-vol-regime-tqqq` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 6 | `tlt-soxl-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 7 | `tlt-upro-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 8 | `tip-tlt-upro-real-yield-v1` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 9 | `d14-disinflation-tqqq` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 10 | `vcit-tqqq-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 11 | `ief-tqqq-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 12 | `agg-upro-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 13 | `soxx-soxl-lead-lag-v1` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |
| 14 | `lqd-tqqq-sprint` | MISSING | 0 | 0 | +0.00% | +0.00% | n/a | n/a | +0.00% | n/a | - |

## Backtest Reference

| # | Slug | Alias | BT Sharpe | BT CAGR | BT MaxDD |
|---|------|-------|-----------|---------|----------|
| 1 | `tsmom-upro-trend-v1` | D13 TSMOM-UPRO | 1.34 | +20.1% | 15.9% |
| 2 | `xlk-xle-soxl-rotation-v1` | D10 XLK-XLE-SOXL | 1.17 | +27.7% | 22.6% |
| 3 | `agg-tqqq-sprint` | AGG-TQQQ Sprint | 1.08 | +10.9% | 10.6% |
| 4 | `tlt-tqqq-sprint` | TLT-TQQQ Sprint (D1) | 1.03 | +12.4% | 10.2% |
| 5 | `d15-vol-regime-tqqq` | D15 Vol-Regime-TQQQ | 0.98 | +15.3% | 19.9% |
| 6 | `tlt-soxl-sprint` | TLT-SOXL Sprint | 0.94 | +16.6% | 17.1% |
| 7 | `tlt-upro-sprint` | TLT-UPRO Sprint | 0.90 | +7.4% | 12.9% |
| 8 | `tip-tlt-upro-real-yield-v1` | D12 TIP-TLT-UPRO | 0.90 | +9.7% | 16.7% |
| 9 | `d14-disinflation-tqqq` | D14 Disinflation-TQQQ | 0.88 | +12.6% | 19.5% |
| 10 | `vcit-tqqq-sprint` | VCIT-TQQQ Sprint | 0.88 | +11.5% | 18.6% |
| 11 | `ief-tqqq-sprint` | IEF-TQQQ Sprint | 0.85 | +11.1% | 11.0% |
| 12 | `agg-upro-sprint` | AGG-UPRO Sprint | 0.84 | +6.1% | 7.4% |
| 13 | `soxx-soxl-lead-lag-v1` | D11 SOXX-SOXL | 0.82 | +25.1% | 37.2% |
| 14 | `lqd-tqqq-sprint` | LQD-TQQQ Sprint | 0.80 | +8.8% | 12.6% |

## Flagged Strategies

### [NO_TRADES] TLT-TQQQ Sprint (D1) (`tlt-tqqq-sprint`)

- Days elapsed: 17
- Entries logged: 0
- Actual cum return: +0.00%
- Expected cum return (proportional): +0.00%
- Z-score: n/a
- Current DD: +0.00% (BT MaxDD 10.2%)
- Note: paper-trading.yaml exists but daily_log is empty

## Missing / Not Initialized for Paper Trading

| # | Slug | Alias |
|---|------|-------|
| 1 | `tsmom-upro-trend-v1` | D13 TSMOM-UPRO |
| 2 | `xlk-xle-soxl-rotation-v1` | D10 XLK-XLE-SOXL |
| 3 | `agg-tqqq-sprint` | AGG-TQQQ Sprint |
| 5 | `d15-vol-regime-tqqq` | D15 Vol-Regime-TQQQ |
| 6 | `tlt-soxl-sprint` | TLT-SOXL Sprint |
| 7 | `tlt-upro-sprint` | TLT-UPRO Sprint |
| 8 | `tip-tlt-upro-real-yield-v1` | D12 TIP-TLT-UPRO |
| 9 | `d14-disinflation-tqqq` | D14 Disinflation-TQQQ |
| 10 | `vcit-tqqq-sprint` | VCIT-TQQQ Sprint |
| 11 | `ief-tqqq-sprint` | IEF-TQQQ Sprint |
| 12 | `agg-upro-sprint` | AGG-UPRO Sprint |
| 13 | `soxx-soxl-lead-lag-v1` | D11 SOXX-SOXL |
| 14 | `lqd-tqqq-sprint` | LQD-TQQQ Sprint |

_These Track D strategies have passed all 6 robustness gates but do not yet have `paper-trading.yaml` files. They need to be wired into `scripts/run_paper_batch.py` dispatch tables and initialized before their paper track records can be validated._

## Top / Bottom Performers

_No Track D strategies currently have live paper-trading entries. `tlt-tqqq-sprint` has a paper-trading.yaml with empty daily_log; the remaining 13 have not been wired into the batch runner yet._

## Methodology

- **days_elapsed**: calendar days between `start_date` in paper-trading.yaml and as-of date (2026-04-17)
- **Actual CumRet**: `(last_nav / initial_nav) - 1` from `daily_log` NAV
- **Expected CumRet**: `(1 + bt_cagr)^(days_elapsed/365.25) - 1` — proportional scaling of backtest CAGR over elapsed window
- **Z-score**: `(actual - expected) / (annual_vol * sqrt(years))`, annual_vol = `bt_cagr / bt_sharpe`
- **Paper Sharpe**: annualized Sharpe from daily_return series (requires >=5 days with non-zero variance)
- **Flag rules**:
  - CONCERN: z-score < -2.0 (actual > 2 sigma below proportional expected)
  - WATCH: current drawdown > 30% of backtest MaxDD (and > 2% absolute)
  - DORMANT: last signal date > 5 calendar days before as-of date
  - NO_TRADES: paper-trading.yaml exists but daily_log is empty
  - MISSING: no paper-trading.yaml file


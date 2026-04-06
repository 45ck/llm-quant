# Track D Capital Scenarios — Realistic Return Expectations

## Core Numbers (from portfolio optimizer, 5yr backtest 2021-2026)

| Portfolio | CAGR | MaxDD | Sharpe | Monthly Equiv |
|-----------|------|-------|--------|---------------|
| XLK-SOXL@60% + TLT-TQQQ@70% | 42.8% | 22.8% | 1.533 | ~3.0% |
| XLK-SOXL@50% + TLT-TQQQ@70% | 38.4% | 20.2% | 1.528 | ~2.7% |
| 3-strat (+ TSMOM@50%) | 34.1% | 17.3% | 1.629 | ~2.5% |
| Top-6 EW (conservative) | 14.7% | 10.0% | 1.364 | ~1.1% |

## Capital Growth Projections

### At 3% monthly (42.8% CAGR — aggressive 2-strategy)

| Starting | 6mo | 12mo | 18mo | 24mo |
|----------|-----|------|------|------|
| $500 | $597 | $713 | $852 | $1,018 |
| $1,000 | $1,194 | $1,426 | $1,703 | $2,035 |
| $2,000 | $2,389 | $2,852 | $3,407 | $4,070 |
| $5,000 | $5,972 | $7,130 | $8,517 | $10,175 |
| $10,000 | $11,943 | $14,260 | $17,034 | $20,350 |

**Time to double: ~23.4 months**

### At 2.5% monthly (34% CAGR — balanced 3-strategy)

| Starting | 6mo | 12mo | 18mo | 24mo |
|----------|-----|------|------|------|
| $500 | $580 | $672 | $779 | $903 |
| $1,000 | $1,160 | $1,345 | $1,558 | $1,806 |
| $2,000 | $2,320 | $2,689 | $3,117 | $3,613 |
| $5,000 | $5,800 | $6,723 | $7,792 | $9,032 |
| $10,000 | $11,599 | $13,446 | $15,585 | $18,063 |

**Time to double: ~28.1 months**

### At 1.1% monthly (14.7% CAGR — conservative top-6)

| Starting | 6mo | 12mo | 18mo | 24mo |
|----------|-----|------|------|------|
| $500 | $534 | $570 | $609 | $650 |
| $1,000 | $1,068 | $1,140 | $1,217 | $1,300 |
| $5,000 | $5,339 | $5,701 | $6,087 | $6,500 |
| $10,000 | $10,678 | $11,402 | $12,175 | $13,000 |

**Time to double: ~63.4 months**

## Position Sizing Reality Check

With Track D position limits (30-50% per position):

- **$500 account, 50% max position**: $250 in SOXL/TQQQ
- **$1,000 account, 50% max position**: $500 in SOXL/TQQQ
- **$5,000 account, 50% max position**: $2,500 in SOXL/TQQQ

Minimum viable account for Track D: ~$1,000 (to have meaningful position sizes above commission thresholds).

## MaxDD in Dollar Terms (worst-case drawdown)

| Starting | 17.3% DD (3-strat) | 22.8% DD (2-strat) | 33.1% DD (XLK-SOXL solo) |
|----------|--------------------|--------------------|--------------------------|
| $500 | -$87 | -$114 | -$166 |
| $1,000 | -$173 | -$228 | -$331 |
| $5,000 | -$865 | -$1,140 | -$1,655 |
| $10,000 | -$1,730 | -$2,280 | -$3,310 |

## Caveats

1. **These are backtested, not live results.** Backtest Sharpe often degrades 20-50% live.
2. **TQQQ/SOXL have beta decay.** Multi-day holds in volatile markets underperform theoretical 3x.
3. **The 2021-2026 window was favorable.** XLK/XLE rotation benefited from tech/energy dispersion.
4. **Path dependency matters.** A 22.8% drawdown early destroys compounding more than late.
5. **D3 TQQQ/TMF (claimed Sharpe=2.21) is broken.** Independent replication shows negative returns.
6. **Paper trading (30+ days) is mandatory** before any capital deployment.

## Recommended Approach for Small Capital

1. **Paper trade all 3 strategies for 30 days** (XLK-SOXL, TLT-TQQQ, TSMOM-UPRO)
2. **Start with $500-1000** in the strategy that shows best paper results
3. **Target 2-3% monthly** — sustainable, not heroic
4. **Scale up only after 90 days of live results** matching backtest Sharpe within 50%
5. **Never expect 2x in 1 month** — that's ~100% monthly = 4096x annual, which is fantasy

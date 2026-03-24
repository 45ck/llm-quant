# Reports

Auto-generated Markdown reports from the llm-quant DuckDB portfolio database.

## Directory Structure

```
reports/
  daily/YYYY-MM-DD.md     # Daily portfolio snapshot, trades, and metrics
  weekly/YYYY-WNN.md      # Weekly aggregate with daily breakdown
  monthly/YYYY-MM.md      # Full monthly dashboard with YTD and benchmarks
```

## Generation

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py [daily|weekly|monthly] [--date YYYY-MM-DD]
```

If `--date` is omitted, today's date is used.

## Report Contents

### Daily
- Market regime (from LLM decisions)
- Portfolio summary: NAV, cash, exposure, P&L
- Current positions with weights and stop losses
- Trades executed that day with conviction and reasoning
- Performance metrics: Sharpe, Sortino, Calmar, max drawdown, win rate
- Benchmark comparison vs 60/40 SPY/TLT

### Weekly
- Start/end NAV with weekly return
- Daily NAV breakdown
- All trades for the week
- Position weight changes (start vs end of week)
- Regime history

### Monthly
- Monthly and YTD returns
- Full performance metrics dashboard
- Trade statistics grouped by conviction level
- Top/bottom performers by P&L %
- Regime breakdown (days per regime)
- Benchmark comparison

## Integrity

Every report footer includes the hash-chain verification status (`PASS` or `FAIL`) confirming the trade ledger has not been tampered with.

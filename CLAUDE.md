# llm-quant

## Project Overview
LLM paper trading system where Claude Code acts as portfolio manager. Receives market data, makes trade decisions with reasoning, tracks performance. No external API key needed - Claude Code IS the portfolio manager.

## Commands
- `pq init` - Create DuckDB schema + default configs
- `pq fetch` - Fetch/update market data from Yahoo Finance
- `pq run [--dry-run]` - Full cycle: fetch -> indicators -> Claude -> trade -> log
- `pq status` - Show NAV, positions, metrics
- `pq trades` - Recent trades with LLM reasoning
- `pytest` - Run tests

## /trade Skill (Primary Workflow)
The `/trade` skill runs the full autonomous trading cycle:
1. **Build context**: `PYTHONPATH=src python scripts/build_context.py` fetches data if stale, computes indicators, builds market context as JSON
2. **Analyze & decide**: Claude Code reads portfolio state + market data + macro indicators, then acts as the quant PM to produce a JSON trading decision
3. **Execute**: `PYTHONPATH=src python scripts/execute_decision.py` parses JSON, runs 7 risk checks, executes paper trades, saves snapshot
4. **Report**: Displays regime, trades, risk rejections, updated portfolio

### Helper Scripts
- `scripts/build_context.py` - Fetches data, computes indicators, outputs JSON context to stdout
- `scripts/execute_decision.py` - Reads JSON from stdin, risk-checks, executes, saves, outputs summary

### Trading Constraints
- Max 2% of NAV per trade, 10% per position
- Gross exposure < 200%, Net exposure < 100%
- Sector concentration < 30%, Cash reserve >= 5%
- Stop-loss required on every new position
- Max 5 trades per session

## Architecture
- `src/llm_quant/data/` - Market data pipeline (yfinance -> Polars -> DuckDB)
- `src/llm_quant/brain/` - LLM integration (prompts, context, response parsing)
- `src/llm_quant/trading/` - Paper trading (portfolio, executor, ledger, performance)
- `src/llm_quant/risk/` - Pre-trade risk checks (7 limits)
- `src/llm_quant/db/` - DuckDB schema
- `src/llm_quant/cli.py` - Typer CLI entry point
- `src/llm_quant/config.py` - Pydantic config from TOML
- `config/` - TOML configs + Jinja2 prompt templates
- `scripts/` - Helper scripts for Claude Code integration
- `.claude/agents/portfolio-manager.md` - PM agent for team workflows

## Conventions
- Python 3.12, type hints everywhere
- Polars for DataFrames (not pandas)
- DuckDB for persistent storage
- Pydantic for config validation, dataclasses for domain models
- All monetary values in USD floats (paper trading, precision not critical)
- Dates as `datetime.date` objects
- Logging via `logging` stdlib
- Always run Python from project root: `cd E:/llm-quant && PYTHONPATH=src python ...`

# /loop — Gated Trading Cycle

Run the full trading cycle with mechanical verification via prompt-language flow. Retries on failure; verifies completion by checking that a portfolio snapshot was saved to the database for today's date.

Use `/loop` instead of `/trade` when you want autonomous execution with guaranteed completion verification.

## Flow

```
Goal: Execute the llm-quant trading cycle with verified completion

flow:
  retry max 3
    run: cd E:/llm-quant && PYTHONPATH=src python scripts/build_context.py
    if command_failed
      prompt: Context builder failed. Troubleshoot — check DB exists, packages installed, Yahoo Finance accessible. Fix the issue.
    end
    if command_succeeded
      prompt: Act as the portfolio manager from CLAUDE.md. Parse the JSON market context above. Analyze using Filters → Indicators → Signals → Rules. Produce a JSON trading decision and pipe it to the executor: echo '<your_json>' | cd E:/llm-quant && PYTHONPATH=src python scripts/execute_decision.py. Then display results as markdown tables.
    end
  end

done when:
  gate snapshot_today: cd E:/llm-quant && PYTHONPATH=src python -c "import duckdb,sys;db=duckdb.connect('E:/llm-quant/data/llm_quant.duckdb');r=db.execute('SELECT COUNT(*) FROM portfolio_snapshots WHERE date=CURRENT_DATE').fetchone();sys.exit(0 if r[0]>0 else 1)"
  gate governance_clear: cd E:/llm-quant && PYTHONPATH=src python -c "import duckdb,sys;db=duckdb.connect('E:/llm-quant/data/llm_quant.duckdb');r=db.execute(\"SELECT overall_severity FROM surveillance_scans ORDER BY scan_id DESC LIMIT 1\").fetchone();sys.exit(0 if r and r[0]!='halt' else 1)"
```

## What the Gates Enforce

The `snapshot_today` gate runs a DuckDB query checking that a portfolio snapshot exists for today. This only happens after `execute_decision.py` completes successfully — meaning data was fetched, indicators computed, a decision made, risk checks passed, and the portfolio state was persisted. Claude cannot declare the cycle complete without this proof.

The `governance_clear` gate checks that the most recent surveillance scan is not in `halt` status. If governance detects a kill switch trigger, the trading cycle cannot be marked complete until the halt condition is resolved or acknowledged.

## When This Retries

- **Context build fails**: Yahoo Finance down, DB locked, missing packages — flow retries after Claude troubleshoots
- **Execution fails**: JSON parse error, risk rejection on all trades, DB write failure — flow retries with fresh analysis
- **Max 3 attempts** before stopping and reporting what went wrong

## Rules

- Follow CLAUDE.md: PM identity, Filters → Indicators → Signals → Rules, hard constraints
- Do not trade on stale data (>2 trading days old)
- If all 3 attempts fail, explain what blocked the cycle

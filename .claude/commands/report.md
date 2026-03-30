---
description: "Generate daily/weekly/monthly portfolio reports from DuckDB + git history"
---

# /report — Portfolio Report Generation

You are the portfolio manager. Reports are the auditable record of everything this portfolio does. They capture market conditions, decisions made, performance metrics, and portfolio state changes.

## Parse the user's argument: "$ARGUMENTS"

### No arguments --> Generate today's daily report

1. Run the report generator:
   ```bash
   cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py daily --date $(date +%Y-%m-%d)
   ```

2. Read the generated report:
   ```bash
   cat reports/daily/$(date +%Y-%m-%d).md
   ```

3. Display to the user. If the report is empty or has no trades, generate a git-based activity report instead (see Step 5).

### "daily YYYY-MM-DD" --> Generate report for specific date

1. Run: `cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py daily --date $DATE`
2. Read and display the report.

### "weekly" or "weekly YYYY-WNN" --> Generate weekly report

1. Run: `cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py weekly [--date $DATE]`
2. Read and display.

### "monthly" or "monthly YYYY-MM" --> Generate monthly report

1. Run: `cd E:/llm-quant && PYTHONPATH=src python scripts/generate_report.py monthly [--date $DATE]`
2. Read and display.

### "backfill" --> Generate all missing reports from git history

1. Get all dates with git activity:
   ```bash
   cd E:/llm-quant && git log --format="%ad" --date=short | sort -u
   ```

2. Check which daily reports already exist:
   ```bash
   ls reports/daily/*.md 2>/dev/null
   ```

3. For each missing date, generate the daily report (see "all YYYY-MM-DD" below).

4. Generate weekly summaries for each ISO week that has activity.

5. Generate monthly summary.

6. Commit all new reports:
   ```bash
   git add reports/ && git commit -m "docs: backfill daily/weekly/monthly reports"
   ```

### "all YYYY-MM-DD" --> Generate git-based activity report for a date

When DuckDB has no trade data for a date (e.g., a research-only day), generate a report from git history:

1. Get commits for the date:
   ```bash
   git log --after="$DATE 00:00:00" --before="$DATE 23:59:59" --format="%H|%s" --date=local
   ```

2. Get files changed:
   ```bash
   git log --after="$DATE 00:00:00" --before="$DATE 23:59:59" --stat --date=local
   ```

3. Get beads issues closed that day:
   ```bash
   bd list --status closed 2>/dev/null | head -50
   ```

4. Build a report in this format:

```markdown
# Daily Report — YYYY-MM-DD

## Session Summary
- Commits: N
- Files changed: N
- Issues closed: N

## Activity

### Research & Development
[List code changes grouped by category: infrastructure, strategies, governance, risk]

### Strategy Pipeline
[Any lifecycle advances — new hypotheses, backtests, promotions]

### Issues Resolved
[Key issues from beads]

## Portfolio State
[Run: PYTHONPATH=src python scripts/generate_report.py daily --date YYYY-MM-DD]
[Include NAV, positions, metrics if available]

## Next Session Priorities
[Outstanding items from beads: bd ready | head -5]
```

5. Save to `reports/daily/YYYY-MM-DD.md`

---

## After generating any report

Always:
1. Display the report to the user
2. If new files were created, stage them: `git add reports/`
3. Note if any anomalies detected (drawdown spikes, missing data, hash chain failures)

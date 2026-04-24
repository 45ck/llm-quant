# Forensic Finding F-01 — DuckDB Empty State Origin

**Question:** The DuckDB at `data/llm_quant.duckdb` has 0 trades, 0 portfolio_snapshots, 0 llm_decisions. The 2026-04-17 capital-deployment-diagnosis referenced live trades. What happened?

## Evidence

### Current DB state (2026-04-24)
```
config_hashes                            | rows=5
cot_weekly                               | rows=0
llm_decisions                            | rows=0
market_data_daily                        | rows=0
pods                                     | rows=1     ← single 'default' pod, created 2026-03-31 11:11:53
portfolio_snapshots                      | rows=0
positions                                | rows=0
schema_meta                              | rows=1     ← version='5'
strategy_changelog                       | rows=0
surveillance_scans                       | rows=1     ← single scan from 2026-04-24 14:26 (today)
trades                                   | rows=0
universe                                 | rows=42
```

The single `pods` row:
```
('default', 'Default Pod', 'regime_momentum', 100000.0, 'active',
 datetime(2026, 3, 31, 11, 11, 53, 376810), None, None)
```

### Evidence the DB previously held real data
From `docs/investigations/capital-deployment-diagnosis.md` (2026-04-17):
> Extracted from `portfolio_snapshots` table (pod_id=default). Duplicate rows per day are intra-day re-runs; latest per day shown:
> ...
> 34 BUY trades totalled $72,793 — average $2,141 per trade
> Portfolio NAV: $100,000 → $101,268.55 (+1.27%) over 30 days
> 60-69% cash for 3 weeks while SPY rallied +7.42%

So as of 2026-04-17, the DB had:
- ≥30 days of `portfolio_snapshots` rows
- ≥34 `trades` rows
- 0 `llm_decisions` rows (this was already known to be a stub bug)

### Evidence of the wipe path
- `data/llm_quant.duckdb` is gitignored (per `.gitignore: /data/`).
- File mtime: 2026-04-24 14:26 — exactly when today's `/governance` scan ran. The scan inserts to `surveillance_scans`; that's why mtime updated.
- File size: 3.68 MB — consistent with schema + universe + small metadata, no trade history.
- No commit in the Apr 17–24 window mentions DB reset, init, or migration. Search of `git log` for `wipe|reset|truncate|drop|migrate|schema|init` in source returns zero hits.
- Search of source code for explicit `DELETE FROM trades` / `TRUNCATE` / `DROP TABLE`: zero hits in `src/` and `scripts/`.
- The DB's `pods.created_at = 2026-03-31 11:11:53` matches the original `pq init` timestamp from initial commit `dc5fba3`.

### Apr 17 → Apr 24 gap
22 commits on 2026-04-17, then **5-day gap**, then 3 commits on 2026-04-22, then today's session.

The 04-22 commits were:
- `90f5d1f chore(paper): batch run 2026-04-22 daily_log appends for 2026-04-21`
- `f6eb942 feat(paper): backfill 2026-04-17 and 2026-04-20 via --date flag`
- `926dad3 docs(paper): 2026-04-22 status report`

None touched DB schema or data.

## Diagnosis

**Most likely scenario:** The DB file was deleted or recreated locally, off-commit, sometime between 2026-04-17 and 2026-04-24. Because `data/` is gitignored, this leaves no audit trail.

**Three plausible mechanisms:**
1. **Manual `pq init` re-run** — wipes everything and rebuilds schema. Would explain why pods.created_at is fresh-looking but only one pod exists. *Most likely.*
2. **DB file deletion + auto-regeneration** — running any script that opens the DB with `read_only=False` would create a fresh schema-only file if the original was deleted.
3. **Selective `DELETE FROM trades; DELETE FROM portfolio_snapshots; DELETE FROM positions` via ad-hoc psql/duckdb session** — possible but no evidence.

The pods table having exactly 1 row (the original `default` pod with original Mar 31 timestamp) argues against full deletion + recreation, since a fresh `pq init` would set a new timestamp. So the most likely vector is **selective table-level DELETE** done outside source control, OR the pods table preserves `created_at` even on `INSERT OR REPLACE` style upserts.

## Implications

1. **The "operations" path that wrote to DuckDB has been silently abandoned.** No trades have been executed via `/trade` since at least 2026-04-17, and likely the data was deliberately cleared on or shortly after that date.
2. **The 2026-04-17 README commit message ("honest paper trading reality")** suggests an intentional pivot — the team may have decided to formally retire the DuckDB execution path in favor of YAML-only.
3. **`/governance`, `/trade`, `/evaluate` are running against an empty DB.** They produce "warning: no portfolio snapshots" outputs (visible in today's surveillance scan) but no one has fixed this in the 7+ days since.
4. **The architectural decision (Option A vs Option B in the strategic review) has effectively already been made by inaction:** the YAML system kept running, the DB system did not. **Today's reality is Option A by default — but the codebase still pretends Option B exists.**

## Recommended Action

- Confirm with PM whether the DB wipe was intentional (Option A pivot) or accidental (data loss).
- If intentional: formalize Option A in ADR-001, delete the unused DB-execution code paths, drop `/trade` and `/governance` slash commands.
- If accidental: investigate restore from any local backup; document the mechanism so it can't happen again silently.

## Confidence

**High** that the data was deleted between Apr 17 and Apr 24. **Medium** on the specific mechanism (DELETE statement most likely, but `pq init` re-run also possible if it preserves pods.created_at).

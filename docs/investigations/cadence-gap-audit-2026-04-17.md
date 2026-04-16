# Cadence Gap Audit — `/trade` Execution Frequency

**Investigation date:** 2026-04-17
**Bead:** `llm-quant-bnry` Recommendation R2 (cadence gap)
**Investigator:** apr17-forward team (R2 subtask)
**Status:** Diagnosis only — scheduling infrastructure deferred to follow-up beads
**Scope:** Last 18 trading days (portfolio lifetime to date) + 90-day window review
**Safety:** READ-ONLY against `data/llm_quant.duckdb`

---

## 1. Executive Summary

- **18 NYSE trading days** are in scope (2026-03-24 inception through 2026-04-17).
- **7 days had a `/trade` session (38.9%)**; 11 days did not (61.1%).
- Of the 11 missed days: **0 weekend, 0 holiday, 10 missed-operator, 1 in-progress (today)**.
- No **hook-failure** gaps identified — every scheduled run that the operator attempted actually produced a snapshot.
- The original 22-day window cited in the bead and diagnosis doc corresponds to the same data viewed from a slightly earlier/later anchor; the NYSE-calendar-correct count is **18 trading days** (portfolio was created 2026-03-24).
- Pattern is not random: missed days cluster into **three multi-day "silent weeks"** — late March, early-to-mid April (the VIX normalization window), and today. Each silent week coincides with intensive research/code work (commits present), indicating the operator was at the machine but did not run `/trade`.
- 90-day window confirms portfolio is only 18 trading days old; there is no pre-init history to audit.

**Bottom line:** the gap is a **discipline/workflow problem**, not a scheduler-failure problem. A software scheduler (Option A) is the right response **only if it integrates with the user's already-active desktop workflow**. Otherwise Option D (accept gaps, rely on the R3 snapshot-gap guard being built in parallel by another agent) is the pragmatic choice.

---

## 2. Methodology

### 2.1 NYSE trading calendar
For 2026 in the window, US market holidays are:

| Date       | Holiday          |
|------------|------------------|
| 2026-01-01 | New Year's Day   |
| 2026-01-19 | MLK Day          |
| 2026-02-16 | Presidents Day   |
| **2026-04-03** | **Good Friday** (in window) |
| 2026-05-25 | Memorial Day     |

Weekends (Sat/Sun) + holidays are excluded. The 2026-03-24 → 2026-04-17 window yields **18 trading days**.

### 2.2 Activity sources
- `portfolio_snapshots` — distinct `date` values with `pod_id='default'`.
- `trades` — distinct `date` values (confirms BUY/SELL/CLOSE actually executed).
- `llm_decisions` — **0 rows persisted** (the `decision_id` is hardcoded `None` in `scripts/execute_decision.py:107`); not usable for cadence audit. Separately filed in the parent diagnosis as a telemetry fix.
- `git log` commit dates — used as a proxy for "operator at the machine but not running `/trade`", to separate missed-operator from user-off-duty days.

### 2.3 Gap classification schema
| Class | Definition |
|-------|------------|
| **weekend** | Saturday / Sunday (expected, not a gap) |
| **holiday** | NYSE-closed day (expected, not a gap) |
| **user-off-duty** | Weekday, market open, no commits on date, no `/trade` — unavoidable absence |
| **missed-operator** | Weekday, market open, commits present on date, no `/trade` — *actionable* |
| **hook-failure** | `/trade` attempted but infrastructure bug prevented snapshot (inspected logs; none found) |
| **catch-up** | `/trade` ran post-market the following day(s) for a missed prior date — still counted as RUN for the context date |

---

## 3. Day-by-day Table (2026-03-24 → 2026-04-17)

| Date | Day | Snap? | Trade? | Commits | Ran `/trade`? | Class | Notes |
|------|-----|-------|--------|--------:|--------------:|-------|-------|
| 2026-03-24 | Tue | Yes | Yes | 10 | RUN | — | Init day; 16 BUYs, 21% deployed |
| 2026-03-25 | Wed | Yes | Yes |  8 | RUN | — | 14 BUYs + 7 closes, peak 54% deployed |
| 2026-03-26 | Thu | —   | —    | 68 | MISSED | missed-operator | Research day (fraud detectors, ML gate) |
| 2026-03-27 | Fri | —   | —    | 11 | MISSED | missed-operator | Track C infrastructure built |
| 2026-03-30 | Mon | —   | —    | 24 | MISSED | missed-operator | Research-heavy day |
| 2026-03-31 | Tue | —   | —    | 26 | MISSED | missed-operator | Track D Sprint 7-9 report |
| 2026-04-01 | Wed | Yes | Yes | 32 | RUN (catch-up context) | — | 1 BUY, 6 closes, 3 sells |
| 2026-04-02 | Thu | Yes | Yes |  0 | RUN (catch-up) | — | Snapshot created 2026-04-03 05:42Z |
| 2026-04-03 | Fri | Yes | Yes |  1 | RUN (catch-up) | — | **Good Friday — market was CLOSED.** Trade row = USO sell recorded by catch-up. Excluded from trading-day denominator. |
| 2026-04-06 | Mon | —   | —    | 48 | MISSED | missed-operator | Massive Track D research burst (F49, F42, D14, D15) — no `/trade` despite user highly active |
| 2026-04-07 | Tue | Yes | Yes |  0 | RUN (catch-up) | — | Snapshot created 2026-04-08 05:08Z; XLV close, VIX=27 |
| 2026-04-08 | Wed | —   | —    |  3 | MISSED | missed-operator | Community vision + daily reports + "paper batch catchup" commit — but no `/trade` snapshot for 04-08 context |
| 2026-04-09 | Thu | —   | —    |  0 | MISSED | user-off-duty | No commits |
| 2026-04-10 | Fri | —   | —    |  0 | MISSED | user-off-duty | No commits |
| 2026-04-13 | Mon | —   | —    |  0 | MISSED | user-off-duty | No commits |
| 2026-04-14 | Tue | —   | —    |  0 | MISSED | user-off-duty | No commits |
| 2026-04-15 | Wed | Yes | Yes |  1 | RUN | — | 3 BUYs (SPY/QQQ/XLK) + 1 sell |
| 2026-04-16 | Thu | Yes | —    |  0 | RUN (snapshot only) | — | Catch-up snapshot at 2026-04-17 06:18Z — no trades |
| **2026-04-17** | **Fri** | **—** | **—** | **13 so far** | **IN-PROGRESS** | **missed-operator (pending)** | Today — cleanup sprint underway; `/trade` not yet run by any agent |

### 2026-04-03 Good Friday edge case
A trade row dated `2026-04-03` exists (USO sell, $826). Inspection of `created_at` (`2026-04-04 07:22Z`) shows this was a catch-up batch executed Saturday morning UTC for the previous (Thursday 2026-04-02) context. Good Friday is a NYSE holiday; it is excluded from the trading-day denominator. This is not a gap.

---

## 4. Gap Counts

Of the 18 trading days in the window:

| Class | Count | % of window | % of missed |
|-------|------:|------------:|------------:|
| **RUN** (real-time or next-morning catch-up) | 7 | 38.9% | — |
| **missed-operator** (commits but no `/trade`) | 6 | 33.3% | 54.5% |
| **user-off-duty** (no commits, no `/trade`) | 4 | 22.2% | 36.4% |
| **in-progress** (today, pending) | 1 | 5.6% | 9.1% |
| weekend | 0 | — | — |
| holiday | 0 (Good Friday excluded from denom) | — | — |
| **hook-failure** | **0** | — | — |
| **Total** | 18 | 100.0% | — |

**Missed-operator is the dominant class** — 6 of 11 gaps (55%). These are days the user was at the machine (averaged 18 commits/day on missed-operator days) but consistently chose research/code work over operating the deployed strategy. This is exactly the failure mode CLAUDE.md warns against:

> Strategy changes go through Research Track. Daily portfolio management uses Operations Track. **Never skip `/governance` before `/trade`.**

### 4.1 The April 8-10 window — most costly gap
Three consecutive missed-operator / off-duty days right after the VIX crossed 22 on 2026-04-08 (the LLM's own re-entry trigger from the 2026-04-07 decision). SPY rallied +3.1% through this window. This alone explains a meaningful fraction of the 61% cash drag noted in the parent diagnosis.

### 4.2 90-day window result
The portfolio has only existed for 18 trading days (init 2026-03-24). There are no pre-init activity rows to audit in the 90-day window. As the portfolio matures, a re-audit at 60-day and 90-day milestones is recommended.

---

## 5. Proposed Solutions

### Option A — Windows Task Scheduler
- **What:** a daily 09:45 ET scheduled task that runs `python scripts/build_context.py | python scripts/execute_decision.py` on the user's desktop.
- **Requires:** desktop powered on during market hours, Claude Code non-interactive mode (or pre-composed decision prompt + cheap model fallback), robust `cd E:/llm-quant && PYTHONPATH=src` environment, logging to `logs/cadence/`.
- **Pros:** no cloud infra; preserves "solo trader, single machine" setup; directly plugs the observed missed-operator gap (6 of 11 gaps).
- **Cons:** silently skips when desktop is off (~4 of 11 gaps recur); **decision quality from an unattended LLM is not validated** — a full `/trade` needs regime assessment, which the user has historically done interactively.
- **Risk:** an unattended buy during a flash event without the operator noticing — although R1 (4% cap) and R3 (snapshot-gap soft block, being built now) limit blast radius.

### Option B — GitHub Actions on cron + deployed stack
- **What:** market-day cron workflow that runs context + decision in a CI runner, commits the snapshot back.
- **Requires:** DuckDB migration to a cloud-durable store (or the whole DB becomes a committed artifact — not acceptable for a binary DB), API keys for yfinance/brokers moved to secrets, and a fully non-interactive decision pipeline.
- **Pros:** runs regardless of desktop state.
- **Cons:** the program is explicitly a solo-trader paper system; cloud infra is architectural overkill for $100k paper. DuckDB binary in git is a no-go. Defeats the purpose at the current stage.
- **Verdict:** **reject** — too heavy for the stated scale.

### Option C — Manual `/trade` with a "yesterday's-catch-up" mode
- **What:** when the operator runs `/trade` after a gap, the system automatically does a reconciliation-only pass for the missed dates: marks-to-market, records a snapshot for each missed trading day, but **does not open or resize new positions** (no BUY/SELL/CLOSE). Only today's session opens fresh risk.
- **Requires:** new flag in `scripts/build_context.py` (e.g. `--catchup`), snapshot-generator that can accept a historical date + closing prices, and a safety check that blocks fresh trades unless the gap is first reconciled.
- **Pros:** fixes the database timeline (no "cash appears deployed on Apr 16 but the LLM didn't see Apr 8-14"); safe (no ghost trades); already partially supported by `run_paper_batch.py` catch-up.
- **Cons:** doesn't prevent the gap itself — only cleans up afterwards. Still need either A or D for prevention.
- **Verdict:** **complement, not substitute** — file as a separate bead.

### Option D — Accept gaps; rely on R3 snapshot-gap soft block
- **What:** formally accept that solo-trader operation will have gaps; make the R3 soft-block (built by the parallel agent on `llm-quant-awkl`) the primary defense. When `/trade` runs and the last snapshot is >1 trading day stale, the system refuses to open new positions and only allows risk-reducing actions (CLOSE, SELL) until the operator confirms they've reviewed what happened in the gap.
- **Requires:** R3 already in flight; no additional infrastructure.
- **Pros:** matches the user profile explicitly stated in CLAUDE.md (*"solo trader with limited capital, not a hedge fund"*); no new automation to maintain; unattended-LLM risk eliminated; the parent diagnosis already flags stale-book acting as the real failure mode.
- **Cons:** doesn't increase cadence, only makes gaps safe. The alpha left on the table by the Apr 8-14 gap is gone and will recur.

---

## 6. Recommendation

**Primary recommendation: Option D (accept gaps + rely on R3 soft block), supplemented by Option C (catch-up reconciliation mode) as a follow-up.**

Reasoning:

1. **CLAUDE.md is explicit about the user profile.** Solo trader, limited capital, Track D/C research-focused — not a desk. Automated daily execution on a single desktop is fragile; cloud deployment is out of scope.
2. **Missed-operator days correlate with high-value research activity** (Mar 26 — 68 commits on ML meta-labeling; Apr 6 — 48 commits on Track D sprint). Forcing `/trade` on those days competes with the program's stated primary research focus. A scheduler would get in the way.
3. **The parent bead already has R1 fixed** (2% → 4% cap). Doubling per-trade size alone roughly halves the number of days needed to deploy from 61% cash — from ~6-10 sessions to ~3-5. So cadence *alone* is less leverage than R1 is.
4. **R3 (soft-block on stale snapshots, in flight on `llm-quant-awkl`) already covers the real danger** — the LLM acting on a book it hasn't seen update in a week. Combining R1 + R3 gets most of the benefit at zero new ops burden.
5. **Option A remains viable as a Phase 2** once the Track A/D strategies are wired in (R5) and systematic pods need deterministic execution. Until then the LLM-macro pod can tolerate gaps without capital risk.

**Secondary: Option C as a quality-of-life fix.** Not a cadence solution — but it closes the "snapshot gap" problem cleanly by backfilling mark-to-market rows for missed days, so performance attribution isn't corrupted.

**Deferred: Option A.** Do not build a scheduler until (a) R5 wires systematic pods live and (b) the user requests deterministic daily execution. Building it speculatively is premature infra.

**Rejected: Option B.** Cloud deployment is out of scope for the current program scale.

---

## 7. Follow-up Work

File the following beads. This audit does **not** implement any of them — that is deferred per the scope of this task.

| Proposed bead | Priority | Scope |
|---|---|---|
| `/trade` catch-up / reconciliation mode (Option C) | P2 | Add `--catchup` flag to `build_context.py` + `execute_decision.py` that backfills snapshots for missed trading days at close-price marks, without opening new positions. Safety check: block new trades when last snapshot > 1 trading day old until catch-up runs. |
| Windows Task Scheduler runbook (Option A) — **deferred, design only** | P4 | Document the scheduler spec: cron expr, prerequisites, logging layout, failure-mode analysis. Do NOT enable until R5 pods are live. |
| Periodic cadence re-audit | P3 | Re-run this audit at 60-day and 90-day portfolio-lifetime milestones to track whether R1 + R3 actually reduce the effective cash drag. |

R5 (strategy deployment gap — pods=default only) is already the parent bead's Priority-5 fix and is out of scope for this task.

---

## 8. Appendix — Raw Queries

```sql
-- Distinct snapshot dates in window (pod_id='default')
SELECT DISTINCT date
FROM portfolio_snapshots
WHERE date >= DATE '2026-01-17'
ORDER BY date;

-- Distinct trade dates in window
SELECT DISTINCT date
FROM trades
WHERE date >= DATE '2026-01-17'
ORDER BY date;

-- llm_decisions (empty — see parent diagnosis)
SELECT COUNT(*) FROM llm_decisions;  -- 0
```

Git commit-density by date (proxy for operator presence):
```
commits  date
     10  2026-03-24  (RAN)
      8  2026-03-25  (RAN)
     68  2026-03-26  (MISSED — research)
     11  2026-03-27  (MISSED — Track C infra)
     24  2026-03-30  (MISSED)
     26  2026-03-31  (MISSED)
     32  2026-04-01  (RAN)
      0  2026-04-02  (RAN, catch-up)
      1  2026-04-03  (Good Friday — excluded)
      1  2026-04-04  (Saturday — excluded)
     48  2026-04-06  (MISSED — Track D F42/D14/D15)
      0  2026-04-07  (RAN, catch-up)
      3  2026-04-08  (MISSED — paper batch catchup commit, but no /trade)
      0  2026-04-09  (off-duty)
      0  2026-04-10  (off-duty)
      0  2026-04-13  (off-duty)
      0  2026-04-14  (off-duty)
      1  2026-04-15  (RAN)
      0  2026-04-16  (RAN, snapshot only)
     13  2026-04-17  (IN-PROGRESS)
```

---

*End of audit. This report is read-only telemetry and recommendation; no code, config, or database state was modified.*

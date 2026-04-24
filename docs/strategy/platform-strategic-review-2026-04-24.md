# Platform Strategic Review — 2026-04-24

**Status:** Decision-grade. Authored from a 4-agent parallel deep dive. All 4 agents reached convergent diagnoses independently. This document is meant to be acted on, not filed.

---

## Executive Summary

**This platform has reached an architectural inflection point.** It has accumulated 173 strategy folders, 53 paper-trading YAMLs, 264 scripts, 4 nominal tracks, and a sophisticated lifecycle/governance vocabulary — while the production database holds **zero trades, zero portfolio snapshots, and zero LLM decisions**. The research throughput is real (~140 commits in 3 weeks). The execution loop does not exist as a working system.

The platform claims to be a $100k portfolio manager. It is in fact a **backtest publication mill that uses the language of trading**. This is a fixable problem, but only if the next sprint is spent on consolidation rather than yet more strategy generation.

The recommended path forward has three pillars:
1. **Stop adding strategies.** Freeze the slug list. Retire ~116 zombies that have no execution artifact.
2. **Pick one paper system and kill the other.** YAML-only or DuckDB-pod-based — both is untenable.
3. **Make at least one strategy place a real (paper) trade end-to-end.** This is the binding constraint, not Sharpe.

---

## The Diagnosis (Synthesized from 4 Independent Audits)

### Agent 1 — Strategic Identity
> "Research factory cosplaying as a portfolio manager."

The CLAUDE.md, mandates, lifecycle docs, and surveillance kill switches all describe a working PM with a daily `/trade` cycle. On disk: 173 strategy folders, 53 paper YAMLs, **DuckDB has 0 trades / 0 portfolio_snapshots / 0 llm_decisions / 0 market_data_daily rows.** The `tlt-tqqq-sprint` paper-trading YAML carries `current_nav: 100000.0`, `total_trades: 0`, and every `operations_tested` flag is `false` six weeks after going "active".

The four tracks are **sediment, not architecture**:
- Track A is "complete on backlog" — but its allocation just dropped from 70% → 40% in the same edit that says it's done. Completeness language is being used as a graceful exit so attention can move to Track D.
- Track B is dead but still on the org chart.
- Track C is honest research.
- Track D is where the actual work is happening.

The platform is **optimizing for backtest Sharpe** when the binding constraint is **the live-execution gap that has never been measured**. `implementation-gaps.md` ranks "generalization ratio (live/backtest Sharpe)" as P2 — it should be P0.

### Agent 2 — Track C Verdict: KILL
- **Built**: 7,800 LOC across `src/llm_quant/arb/` (Polymarket clients, Kalshi client, CEF stubs, funding rate stubs, paper gate)
- **Mandate gates**: 4 of 17 fully implemented; mandate file itself doesn't exist (`data/strategies/niche-arbitrage/mandate.yaml` is missing)
- **Real edge**: live Kalshi scan returned exactly 1 actionable trade at 5% spread on a thin VA Senate primary (~$5k Kelly cap). Funding rate arb requires VIP fee tiers ($50k+/exchange) — economically infeasible at $100k retail.
- **Already de facto dead**: CLAUDE.md explicitly says "0% of capital until all 17 gates pass."
- **Recommendation**: Hibernate or delete. Maintenance liability with no payoff path.

### Agent 3 — Track D Infrastructure Debt
- **`overnight-tqqq-sprint` failure**: 30-minute fix. Bug is a **spec-param-name mismatch** (`OvernightMomentumStrategy` reads `symbol`, frozen spec writes `ticker`). The strategy class already supports any ticker. Spec falls back to default `SPY`, runs zero trades on TQQQ data that was never fetched.
- **`vcit-tqqq` / `ief-tqqq` robustness scripts**: 100% template-able. Each is ~500 LOC of copy-paste from `robustness_agg_tqqq_sprint.py`. All required values already in `LEAD_LAG_PARAMS` at `scripts/run_paper_batch.py:43-73`.
- **Script proliferation**: 264 files in `scripts/`. **202 are strategy-specific** (37 backtest, 43 robustness, 122 run_*_robustness). Bug fixes need to be applied in 165+ places. Each new strategy adds ~500 LOC of duplication.
- **Zombies**: Of 170 strategy directories, **117 have no `paper-trading.yaml` and no matching script.** Many are hypothesis-only stubs that will never execute.
- **Generic runner**: 2-3 days to build `scripts/run_robustness.py --slug <slug>` that reads spec + dispatches to strategy class. Deletes ~80k LOC of duplication.

### Agent 4 — Paper→Live Execution Gap
- **Two parallel paper systems**:
  - **(a) YAML logger** (`run_paper_batch.py`): writes `data/strategies/<slug>/paper-trading.yaml` daily. 53 strategies. Hypothetical NAV per-strategy. **Never touches DB tables. Never moves capital, even on paper.**
  - **(b) DuckDB pods**: expected by `/trade`, `/governance`. Currently empty — recently truncated/wiped (the 2026-04-17 capital-deployment-diagnosis referenced live rows that no longer exist).
- **`/trade` actually works** — for the single discretionary `default` pod. But it has **zero integration with the 53 systematic strategies in run_paper_batch.** The `signal_aggregator.py`, `track_router.py`, `execution_bridge.py` modules exist in `src/llm_quant/trading/` but are not consumed by `/trade`.
- **The 50-trade promotion gate is structurally unreachable.** Survey of 53 paper YAMLs: max trades = 4, mean = 0.9, **zero strategies have ≥10 trades**. At current 0.5 trades/wk pace, hitting the 50-trade gate takes **2 years**. The gate has never been tested by a single strategy.
- **Cumulative metrics silently dropped**: `compute_cumulative_metrics()` is correctly implemented and called, but its result is assigned to a local variable and **never written back to the YAML.** Naming/contract drift between runner and survey.

---

## The Inflection Point

### What this platform IS today
A research workflow with strong governance vocabulary, a 13-family alpha-hunting framework, a 9-stage lifecycle state machine, robust per-strategy backtest and robustness gates, and a high cadence of hypothesis generation (Sprints 1-9 in 3 weeks, ~22% pass rate).

### What this platform is NOT
- **Not a portfolio manager**: no execution loop has ever moved capital, even on paper.
- **Not a deployable trading system**: the systematic strategies are siloed from the only execution path that works.
- **Not maintainable at current trajectory**: 202 bespoke scripts + 117 zombie strategies + two disconnected paper systems = exponential maintenance burden.

### What this platform SHOULD be
**One of two things — pick exactly one:**

**Option A — Research lab with disciplined retirement** (no real money, no operational claims)
- YAML logger is canonical. DuckDB execution stack is deleted.
- `/trade` is removed; `/lifecycle`, `/research-spec`, `/backtest`, `/robustness`, `/paper` (yaml-only) remain.
- All "kill switches" / "surveillance" framing dropped. Replaced with "research integrity gates."
- Output: published sprint reports with mechanism insights. The platform's value is *knowledge*, not PnL.
- **This is what the platform actually IS. Embracing it would clean up 80% of the contradictions.**

**Option B — Real systematic PM** (genuinely deployable)
- DuckDB pods are canonical. Each strategy = its own pod, executed by `run_paper_batch` via a real execution path that writes to `trades` and `portfolio_snapshots`.
- The YAML files become render artifacts of DuckDB queries, not source of truth.
- `/trade` is rewritten as a multi-pod orchestrator, not a single-LLM-discretionary loop.
- 50-trade gate replaced with regime-aware variant (15 trades + 95% Sharpe CI for slow signals).
- **This is a 2-3 month engineering project, not a sprint.**

The tension between A and B is the unspoken architectural debate this project has been having with itself for weeks. Choosing one is the most important decision in front of the PM.

---

## Strategic Decisions Required (Before Any Further Sprints)

| # | Decision | Default Recommendation | Owner |
|---|---|---|---|
| 1 | Track C: keep / hibernate / kill | **Kill** — delete `src/llm_quant/arb/`, retire 4 strategy dirs, archive docs to `docs/archive/track-c/` | PM |
| 2 | Paper system canonical | **YAML if Option A, DuckDB if Option B** — current state (both) is untenable | PM |
| 3 | Slug freeze | **Freeze at 173. No new mandates this month.** | PM |
| 4 | Zombie retirement | **Auto-archive** ~116 dirs with no paper YAML AND no script. Move to `data/strategies/_archive/` | Automatable |
| 5 | Track A "completeness" | **Either truly close (delete /mandate, /hypothesis ability for Track A slugs) or admit it's still active** | PM |
| 6 | 50-trade gate | **Lower to 15 trades + Sharpe CI for slow signals** OR commit to 6-month paper windows | PM |
| 7 | Generic robustness runner | **Build it.** 2-3 days for ~80k LOC of duplication elimination | Engineering |
| 8 | overnight-tqqq fix | **Apply param aliasing.** 30 minutes | Engineering |

---

## 2-Week Roadmap (Specific, Bound)

**The premise:** STOP creating new strategies. Spend two weeks paying down the platform's compounding debt.

### Week 1 — Triage and Decide
**Mon-Tue (decisions):**
- [ ] PM picks Option A (research lab) or Option B (deployable PM). Document as ADR `docs/decisions/ADR-001-platform-identity.md`.
- [ ] Decide Track C fate. If kill: open PR deleting `src/llm_quant/arb/`, archiving docs, retiring strategy dirs.
- [ ] Lower or replace the 50-trade gate. Edit `docs/governance/quant-lifecycle.md` and `docs/governance/model-promotion-policy.md`.

**Wed-Fri (housecleaning):**
- [ ] Write `scripts/audit_zombie_strategies.py` — finds dirs with no paper YAML AND no script reference. Move them to `data/strategies/_archive/`.
- [ ] Apply 30-min `overnight-tqqq` fix. Re-run, verify trades fire.
- [ ] Resolve DuckDB-empty mystery. Check git for migration scripts between 04-17 and 04-24. Either restore or document the reset.

### Week 2 — Build the Generic Runner OR Commit to YAML-Only
**If Option A (YAML-only):**
- [ ] Delete `scripts/build_context.py`, `scripts/execute_decision.py`, `src/llm_quant/trading/` (or quarantine in `_legacy/`).
- [ ] Drop `/trade` and `/governance` slash commands. Keep `/research-spec`, `/backtest`, `/robustness`, `/paper`, `/lifecycle`, `/promote`.
- [ ] Fix `compute_cumulative_metrics` write-back bug in `run_paper_batch.py:1976` so YAMLs carry computed Sharpe/MaxDD/return.
- [ ] Rewrite Identity section of CLAUDE.md to match reality.

**If Option B (deployable PM):**
- [ ] Spec out `scripts/run_paper_batch.py` → DuckDB integration. Each strategy gets a pod row in `pods` table; signals execute as `Trade` rows; daily snapshots write to `portfolio_snapshots`.
- [ ] Pick 3 strategies (1 Track A, 1 Track D, 1 microstructure) and drive them end-to-end through DuckDB. Verify `/governance` and `/evaluate` work on real data.
- [ ] Reconcile `pods` table — currently has only 1 row (`default` discretionary pod). Decide if discretionary pod stays, dies, or becomes one pod among many.

### Either Way
- [ ] Build `scripts/run_robustness.py --slug <slug>` generic dispatcher. Migrate 1 Track D sprint as proof. Document in `docs/engineering/generic-robustness-runner.md`.
- [ ] Author `data/strategies/<slug>/STATUS.md` template — single file per strategy with current state, last verdict, kill-or-keep decision date. Replace the 7-file lifecycle artifact sprawl with a single canonical readme.

---

## 30-Day Outlook

If the 2-week roadmap completes:
- Strategy directory: 173 → ~57 (after zombie retirement)
- Script directory: 264 → ~80 (after generic runner consolidation)
- Active paper-trading book: 53 → 8-15 (forced retirement of strategies with 0 trades after 14 days)
- One canonical paper system (YAML or DuckDB, not both)
- One identity (research lab or PM, not both)
- One generic robustness/backtest path; deletion of 165 bespoke scripts queued
- **The platform can answer the question "is this strategy good enough to trade real money?" — currently it cannot.**

---

## Success Metrics for This Strategic Pivot

| Metric | Today | 30-day target |
|---|---|---|
| Strategy slugs | 173 | ≤60 (active) + archive |
| Bespoke scripts | 202 | ≤40 |
| Paper systems | 2 | 1 |
| Strategies with ≥10 paper trades | 0 | ≥3 |
| Documented platform identity (Option A or B) | implicit/conflicting | one ADR |
| Generic robustness runner | 0 | 1 working |
| Track count (active) | 4 | 2 (one focused + one backlog) |
| `cumulative_metrics` populated in paper YAMLs | 0/53 | 53/53 (if Option A) |
| Trades in DuckDB | 0 | ≥50 (if Option B) |

---

## What This Platform Should NOT Try to Be

The strategic temptation will be to add another track, another sprint, another mechanism family. Resist. The platform already has more passing strategies than it can validate, more backtest infrastructure than it can maintain, and more governance vocabulary than it has executed against. The next interesting work is **subtractive**, not additive.

Specifically — do NOT:
- Spin up Track E or any new track until Track A and B are formally closed
- Run more sprint backtests until generic robustness runner exists (saves authoring 4-8 new scripts per sprint)
- Add new mechanism families to the F1-F50 alphabet until existing F-series strategies have produced ≥1 paper trade each
- Promote any strategy to live capital without a working trades/snapshots loop

---

## Open Questions / Items the Audits Could Not Resolve

1. **Why is the DuckDB empty?** The 2026-04-17 capital deployment diagnosis referenced live trades/snapshots. Was there a reset, a path move, a failed migration? Investigate before week 1.
2. **Is `/trade` still being run?** If nobody has run it in 2 weeks, the operations track has been fully crowded out by the research track. Confirm via session logs.
3. **What is the actual capital intent?** $100k paper for research output, or $100k as a real future deployment? The two require different platforms.
4. **How much of the existing infrastructure is the user's, and how much is generated by previous Claude sessions?** This affects retire-vs-rewrite decisions.

---

**Authors:** 4 parallel investigative agents (strategic identity, Track C, Track D infrastructure, paper→live readiness).
**Date:** 2026-04-24.
**Decision required by:** PM, before next research sprint is launched.
**Recommended next step:** PM reads this, decides Option A vs Option B, opens ADR-001, and freezes the strategy slug list.

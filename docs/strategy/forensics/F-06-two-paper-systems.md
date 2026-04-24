# Forensic Finding F-06 — The Two Paper Systems Have No Bridge

**Question:** Agent 4 claimed two parallel paper-trading systems with no integration. Verify mechanically.

## The Two Systems

### System A — YAML logger (`scripts/run_paper_batch.py`)
- **Output:** writes `data/strategies/<slug>/paper-trading.yaml` per strategy
- **Capital model:** each strategy assumes its own fresh $100k initial NAV
- **Total simulated capital today:** **$5,300,000** across 53 strategies (53× the platform's actual $100k mandate)
- **Touches DuckDB?** No. Verified: `grep -n "log_trades\|save_portfolio_snapshot" scripts/run_paper_batch.py` returns zero hits.
- **Status today:** actively run. Produced today's daily logs (entries through 2026-04-23) for 52 strategies.

### System B — DuckDB pod execution (`scripts/build_context.py` + `scripts/execute_decision.py`)
- **Output:** writes to `trades`, `portfolio_snapshots`, `positions`, `llm_decisions` tables
- **Capital model:** single shared `$100,000` portfolio under one pod (`pods.default`)
- **Total capital:** $100k (the actual mandate)
- **Touches YAML files?** No. Verified: `execute_decision.py:32` imports `log_trades, save_portfolio_snapshot` from `llm_quant.trading.ledger` — these write to DB, not yaml.
- **Status today:** dormant. The DB has 0 trades, 0 snapshots (see F-01). The slash command `/trade` exists but its writes have produced nothing visible since at least 2026-04-17.

## The Disconnect, Quantified

| Dimension | System A (YAML) | System B (DuckDB) |
|---|---|---|
| Strategies | 53 | 0 (one configured pod, no signals registered) |
| Hypothetical capital | $5.3M | $100k |
| Trade records | 46 (in YAML logs) | 0 (in DB) |
| LLM-mediated? | No (deterministic dispatch) | Yes (`build_context.py` → LLM → `execute_decision.py`) |
| Risk manager checks? | No | Yes (`risk/manager.py`, 7 hard checks) |
| Hash-chain ledger integrity? | No | Yes (`db.hash_chain`) |
| Surveillance kill switches? | Reports against (empty) DB | Active monitoring of DB state |
| Mandated by CLAUDE.md? | Implicitly (via `/loop` and the lifecycle) | Explicitly (the `/trade` Command section) |

## What This Reveals

**The platform has two paper-trading systems with mutually exclusive properties:**

- **System A** is the *systematic* path — every strategy in the slug list runs every day, signals are deterministic, no LLM in the loop. But it doesn't enforce risk limits, capital is fictional, and there's no integrity ledger.
- **System B** is the *discretionary-LLM* path — a single LLM agent reads context daily, decides on trades, executes through the risk manager. Real capital semantics, real audit trail, but only one strategy at a time and currently producing nothing.

These are not two implementations of the same idea. **They are two different products** living in the same repo:

- System A is a *backtest-on-rails* — every strategy tested as if it had its own $100k for the rest of time.
- System B is a *single-pod LLM-discretionary trader* — one portfolio, one LLM, one decision per day.

The CLAUDE.md describes System B in detail (risk manager, surveillance, kill switches, hash chain, daily `/trade` command) and barely mentions System A. The actual usage pattern is the inverse: System A runs every day; System B has been dormant for a week.

## Implications for the Strategic Decision

The Option A / Option B framing in the strategic review maps directly:
- **Option A (research lab)** = canonicalize System A. Drop System B entirely (delete `build_context.py`, `execute_decision.py`, the `/trade` and `/governance` slash commands, surveillance, hash chain). Reframe the platform as a research workflow.
- **Option B (deployable PM)** = canonicalize System B. Either (a) give every strategy its own pod and write a multi-pod orchestrator for `run_paper_batch.py` that calls `log_trades`/`save_portfolio_snapshot`, or (b) give up the multi-strategy book and run only the LLM-discretionary single pod.

Note that **Option B(a) — multi-pod orchestrator — is a 2-3 month engineering project.** The current trading layer (`src/llm_quant/trading/portfolio.py`, `executor.py`, `ledger.py`) was designed for one pod. Migrating 53 strategies to per-pod execution requires:
- A `pods` table with one row per strategy
- A risk manager that enforces cross-pod aggregate limits (currently it only sees one portfolio)
- A signal aggregator that resolves cross-strategy position conflicts (e.g., AGG-SPY says BUY SPY 0.8 weight, TLT-SPY says SELL SPY 0.0 weight — what wins?)
- A pod-level surveillance ruleset that doesn't trigger on a single strategy's normal drawdown

**Option B(b) — single LLM pod only — is what System B was designed for** but it doesn't engage with the 53-strategy research book at all. The paper YAMLs become advisory information for the LLM, not execution instructions.

## Why the Confusion Has Persisted

CLAUDE.md was authored before the systematic strategy book existed in its current form. As the research workflow grew, `run_paper_batch.py` became the de facto operations command without ever being recognized as such. The `/trade` slash command continues to be documented as the operations entrypoint because nobody updated CLAUDE.md after the systematic book displaced the LLM-discretionary path.

## Confidence

**Maximum.** Both files were read directly. The grep shows the integration boundary doesn't exist. The DuckDB inspection confirms System B has produced no output. The YAML survey confirms System A has 53 active records.

# Lifecycle Gap Audit — 2026-03-30

**Auditor:** Governance Auditor Agent
**Issue:** llm-quant-5nsn
**Scope:** 14 strategies in `scripts/portfolio_optimizer.py` that lack lifecycle artifact directories

---

## 1. Executive Summary

`scripts/portfolio_optimizer.py` registers 16 strategies with experiment IDs in `STRATEGY_EXPERIMENTS`. One strategy (`soxx-qqq-lead-lag`) has full lifecycle artifacts. One strategy (`lqd-spy-credit-lead`) has a strategy directory with artifacts but no experiments subfolder with the referenced experiment ID. The remaining **14 strategies have zero artifacts on disk** — no strategy directory, no experiment YAML, no mandate or hypothesis.

The optimizer silently skips missing artifacts at runtime (logs a warning and continues), so it would return results using only the 1-2 strategies with data. This is a silent data gap — no error is raised to the caller.

---

## 2. Artifact Inventory

### 2.1 Strategies with Full Artifacts

| Slug | Dir Exists | Experiment YAML | Status |
|------|-----------|-----------------|--------|
| soxx-qqq-lead-lag | YES | `57fba00d.yaml` — confirmed | COMPLETE |

### 2.2 Strategy with Dir but Missing Experiment

| Slug | Dir Exists | Experiment ID | Artifact Found |
|------|-----------|---------------|----------------|
| lqd-spy-credit-lead | YES (full lifecycle artifacts) | `b0588e6d` | NOT FOUND in `experiments/` |

`data/strategies/lqd-spy-credit-lead/` contains: `data-contract.yaml`, `experiment-registry.jsonl`, `hypothesis.yaml`, `mandate.yaml`, `paper-trading.yaml`, `research-spec.yaml`, `robustness.yaml`. The `experiments/` subdirectory does not exist. The referenced experiment ID `b0588e6d` is not on disk.

### 2.3 Strategies with No Artifacts At All (14 strategies)

None of the following have a directory under `data/strategies/`:

| Slug | Experiment ID | Mechanism Family |
|------|--------------|-----------------|
| agg-spy-credit-lead | `66bec9a0` | F1: Credit Lead-Lag |
| hyg-spy-5d-credit-lead | `1736ac56` | F1: Credit Lead-Lag |
| agg-qqq-credit-lead | `eaf37299` | F1: Credit Lead-Lag |
| lqd-qqq-credit-lead | `ec8745f9` | F1: Credit Lead-Lag |
| vcit-qqq-credit-lead | `b99dac63` | F1: Credit Lead-Lag |
| hyg-qqq-credit-lead | `ba0c05a2` | F1: Credit Lead-Lag |
| emb-spy-credit-lead | `90e531d1` | F1: Credit Lead-Lag |
| agg-efa-credit-lead | `bef23aa4` | F1: Credit Lead-Lag |
| spy-overnight-momentum | `22cddf8c` | F5: Overnight Momentum |
| tlt-spy-rate-momentum | `9e14ce90` | F6: Rate Momentum |
| tlt-qqq-rate-tech | `2338b9e5` | F6: Rate Momentum |
| ief-qqq-rate-tech | `594c4f53` | F6: Rate Momentum |
| behavioral-structural | `7cb2cace` | F7: Behavioral/Structural |
| gld-slv-mean-reversion-v4 | `14cdfaaf` | F2: Mean Reversion |

---

## 3. Root Cause Analysis

The strategies were added to `STRATEGY_EXPERIMENTS` in `scripts/portfolio_optimizer.py` as part of the multi-family credit lead-lag expansion (F1), rate momentum work (F6), and the addition of gld-slv-mean-reversion-v4 (referenced in recent commits). However, the experiment YAML artifacts — which contain `daily_returns` and metrics needed by the optimizer — were either:

1. **Never generated** (strategy research was tracked conceptually but backtests were never run and saved to disk), or
2. **Generated in a prior session and never committed** (artifacts exist only in a local environment that was not pushed to git), or
3. **Registered prematurely** (the optimizer was updated to reference strategies before their lifecycle was complete).

The `load_artifact()` call in the optimizer expects: `data/strategies/<slug>/experiments/<exp_id>.yaml`.

The optimizer's fallback behavior at line 101 (`logger.warning("Artifact not found: %s — skipping", artifact_path)`) means the portfolio optimization runs silently on a skeleton of 1 strategy instead of 16, producing meaningless output. No error propagates to the caller.

---

## 4. Governance Status Assessment

### 4.1 soxx-qqq-lead-lag
- Status: **COMPLIANT** — full artifacts, experiment YAML confirmed.

### 4.2 lqd-spy-credit-lead
- Status: **PARTIAL** — lifecycle artifacts present (mandate through robustness), experiment YAML missing from disk.
- Likely cause: experiment YAML exists locally but was never committed, or experiments/ directory not created.
- Risk: optimizer will skip this strategy at runtime.

### 4.3 8 F1 Credit Lead-Lag variants (agg-spy, hyg-spy-5d, agg-qqq, lqd-qqq, vcit-qqq, hyg-qqq, emb-spy, agg-efa)
- Status: **GOVERNANCE DEBT** — no lifecycle artifacts of any kind.
- These appear to be systematically derived variants of lqd-spy-credit-lead applied to different credit ETF pairs.
- The alpha hunting framework note in CLAUDE.md states "Family 1 (Cross-Asset Information Flow): 10 strategies passing — STRONG, stop adding." This implies these were considered passing in some context, but no on-disk evidence exists.

### 4.4 spy-overnight-momentum (F5), tlt-spy-rate-momentum, tlt-qqq-rate-tech, ief-qqq-rate-tech (F6)
- Status: **GOVERNANCE DEBT** — no lifecycle artifacts. F5 and F6 are listed as "UNTESTED" in CLAUDE.md despite being registered in the optimizer.

### 4.5 behavioral-structural (F7)
- Status: **GOVERNANCE DEBT** — no lifecycle artifacts. F7 listed as "UNTESTED" in CLAUDE.md.

### 4.6 gld-slv-mean-reversion-v4 (F2)
- Status: **GOVERNANCE DEBT** — no lifecycle artifacts despite being the most recently added strategy (added per git log). `data/strategies/gold-silver-ratio-mr/` exists for a different slug (`gold-silver-ratio-mr` vs `gld-slv-mean-reversion-v4`).

---

## 5. Decision: Option B — Acknowledge Debt + Governance Note

**Rationale for Option B over Option A (reconstruct):**

Reconstructing 14 experiment artifacts without re-running the actual backtests would produce fabricated data. The artifacts must contain actual `daily_returns` arrays derived from real backtests — these cannot be reconstructed from metadata alone. Option A is only viable if the backtests are re-run through the proper lifecycle.

The recommended path is:
1. Acknowledge the gap with this audit document.
2. Create a tracking issue for each missing strategy cluster to run through `/backtest` → `/robustness` properly.
3. Either remove the 14 missing entries from `STRATEGY_EXPERIMENTS` (so the optimizer doesn't silently degrade), or add a hard failure mode when fewer than N strategies are loadable.

---

## 6. Operational Risk

**Current impact:** `scripts/portfolio_optimizer.py` will run but produce results for 1 strategy (soxx-qqq-lead-lag only) while silently skipping 15 entries. Any portfolio allocation decisions based on its output are invalid.

**Severity:** HIGH — the optimizer is producing misleading results (Section 4 report claims "15 strategies" but only 1 loads).

---

## 7. Recommended Remediation

### Immediate (this session)
- [ ] Add a hard failure or warning to `portfolio_optimizer.py` when fewer than a minimum number of strategies (e.g., 5) load successfully.
- [ ] Remove or comment out the 14 missing entries from `STRATEGY_EXPERIMENTS` until their lifecycle is complete.

### Short-term (1-2 weeks)
- [ ] For lqd-spy-credit-lead: create the `experiments/` directory and run `/backtest` to generate the `b0588e6d` artifact.
- [ ] For 8 F1 credit-lead variants: run systematically through `/hypothesis` → `/backtest` → `/robustness` using lqd-spy-credit-lead as the template. These share the same mechanism; only the asset pair changes.
- [ ] For gld-slv-mean-reversion-v4: confirm whether `data/strategies/gold-silver-ratio-mr/` is the same strategy under a different slug. If so, rename or alias. If different, run full lifecycle.

### Medium-term (1 month)
- [ ] F5 (spy-overnight-momentum), F6 (tlt-spy-rate-momentum, tlt-qqq-rate-tech, ief-qqq-rate-tech), F7 (behavioral-structural): UNTESTED families per CLAUDE.md. Run full lifecycle for each before adding to the optimizer.
- [ ] Enforce a CI check: strategies in `STRATEGY_EXPERIMENTS` must have a corresponding directory and experiment YAML.

---

## 8. Artifacts Checked

- `scripts/portfolio_optimizer.py` — strategy registry and artifact loading logic
- `data/strategies/` — enumerated all 30 existing strategy directories
- `data/strategies/soxx-qqq-lead-lag/experiments/` — confirmed `57fba00d.yaml` present
- `data/strategies/lqd-spy-credit-lead/` — confirmed lifecycle artifacts, no experiments/ subdir

*Audit date: 2026-03-30*

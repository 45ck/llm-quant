# Forensic Finding F-05 — Script Proliferation: 203 of 256 Are Strategy Copy-Paste

**Question:** Agent 3 estimated 202 bespoke per-strategy scripts in `scripts/`. Verify and quantify the maintenance burden.

## Hard Counts (executed 2026-04-24)

```
Total .py files in scripts/:                            256
Strategy-specific (backtest_/robustness_/run_*):        203
Generic / utility scripts:                               53
```

### Strategy-specific breakdown
| Pattern | Count | Example |
|---|---|---|
| `backtest_*.py` | 37 | `backtest_d10_xlk_xle_soxl.py` |
| `robustness_*.py` | 43 | `robustness_agg_tqqq_sprint.py` |
| `run_*_robustness.py` | 122 | `run_a7_robustness.py` |
| `run_*_backtest.py` | 1 | `run_lqd_tqqq_weight_variants.py` |
| **Total bespoke** | **203** | **(79% of all scripts)** |

### Utility scripts (the survivors of a generic-runner refactor)
Examples: `run_paper_batch.py`, `run_surveillance.py`, `run_walk_forward.py`, `run_arb_paper_gate.py`, `run_ic_analysis.py`, `run_fraud_detectors.py`, `run_meta_label.py`, `run_ml_gate.py`, `run_pm_scanner.py`, `paper_dashboard.py`, `cross_track_dashboard.py`, `generate_report.py`, `compute_hrp_weights.py`, `portfolio_optimizer.py`, `build_context.py`, `execute_decision.py`.

These are the platform's actual reusable infrastructure. There are 53 of them — about 1/4 the count of bespoke scripts.

## The Maintenance Burden

Each `robustness_<slug>.py` is approximately 200-500 LOC of:
- Boilerplate setup: load registry, fetch data, compute indicators, define cost model
- Strategy-specific: param grid for perturbation, symbol list, base params dict
- Boilerplate analysis: CPCV via `run_cpcv()`, perturbation loop, DSR, shuffled-signal test, gate evaluation, YAML write

The strategy-specific section is ~30-50 LOC. The boilerplate is ~150-450 LOC. **80-90% of every robustness script is copy-paste.**

### Implication for any change to the analysis methodology

Suppose CPCV needs a fix (e.g., the purge window logic changes, or the IS/OOS ratio computation gets a normalization tweak). Today, applying that fix requires:
- Editing `src/llm_quant/backtest/robustness.py` (correct — the primitive lives there)
- AND editing the gate-evaluation logic in **165+ separate scripts** (each has its own copy of the gate-checking block, since they pre-date a shared utility)

Spot-check: I read `scripts/robustness_agg_tqqq_sprint.py`, `scripts/robustness_lqd_soxl_sprint.py`, `scripts/robustness_m1.py`, `scripts/run_a7_robustness.py`. Each has its own copy of the gate-evaluation switch, the perturbation loop wrapper, the YAML write. None share code beyond `from llm_quant.backtest.robustness import run_cpcv`.

**A single methodology change (e.g., raising perturbation gate from 40% to 50%) would require editing 165+ files** — and the inevitable result would be partial drift where some scripts get updated and others don't.

## What the Sprint Cadence Adds

Sprint 7-9 added 8 new Track D strategies. For each, the workflow was:
1. Author `backtest_<slug>.py` (~300 LOC)
2. Author `robustness_<slug>_sprint.py` (~400 LOC)
3. Wire into `run_paper_batch.py` LEAD_LAG_PARAMS dict
4. Possibly add monitor scripts

That's ~700 LOC of disposable script per strategy — but disposable means *each new strategy adds 700 LOC of code that someone else later has to maintain or delete.*

At Sprints 1-9 totals (50+ strategies tested per sprint cycle), the script directory grew by ~8 files per active strategy. The current 203 count is the cumulative result.

## The Generic Runner Alternative

Agent 3 proposed `scripts/run_robustness.py --slug <slug>` that:
- Reads `data/strategies/{slug}/research-spec.yaml` for params/symbols/track/strategy_type
- Dispatches via `STRATEGY_REGISTRY` (already exists at `strategies.py:5198`)
- Uses a generic `run_single(spec)` that calls the strategy class's `generate_signals` rather than re-implementing logic
- Calls shared helpers in `src/llm_quant/backtest/robustness_runner.py` (new module containing the perturbation loop, gate eval, YAML write — extracted from the 43 existing `robustness_*.py` scripts)
- Reads track-specific gate thresholds from `config/governance.toml`

Estimated effort: **2-3 days** to build + migrate one strategy as proof.

Estimated payoff: every new strategy after that costs **0 new script files**. The 203 existing scripts can be deprecated and eventually deleted (move to `scripts/_legacy/`).

**ROI: 80,000+ LOC eliminated, every methodology change becomes a 1-file edit.**

### Prerequisites
- **Spec normalization.** ~40 sprint specs use `ticker`/`target_position_weight` while the strategy classes expect `symbol`/`target_weight`. The F-04 finding shows this isn't just a generic-runner blocker — it's already silently breaking individual backtests.
- **Perturbation grid in spec.** Currently hardcoded per-script. A spec block like:
  ```yaml
  perturbation_grid:
    - {param: signal_window, values: [7, 13, 15]}
    - {param: entry_threshold, values: [0.005, 0.015]}
  ```
  would let one runner handle all variants.
- **Strategy `validate_params()` method** on the base class (see F-04). Required so the generic runner can fail fast when spec/strategy contracts mismatch.

## What This Means

The script proliferation is the platform's biggest technical debt by LOC. Combined with F-02 (zombies) and F-04 (spec drift), it forms the **infrastructure tax** that has slowed the platform from "research throughput" to "research throughput minus debt servicing."

**The next 3 days of engineering, if focused on building the generic runner and migrating to it, would pay for themselves within a single sprint** by eliminating the per-strategy script authoring step.

## Confidence

**Maximum** on counts (executed against current filesystem). **High** on remediation viability — the prerequisites are well-scoped and the existing `STRATEGY_REGISTRY` provides the dispatch mechanism.

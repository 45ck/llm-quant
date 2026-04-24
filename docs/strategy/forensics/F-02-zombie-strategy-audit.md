# Forensic Finding F-02 — Zombie Strategy Audit

**Question:** The strategic review claimed ~117 zombie strategies (no paper-trading entry, no script). What's the actual count and breakdown?

## Hard Counts (executed 2026-04-24)

**Total strategy directories:** 170

| Lifecycle Bucket | Count | Description |
|---|---|---|
| `paper_active` | **52** | Has `paper-trading.yaml` with daily log entries |
| `paper_empty` | 1 | Has `paper-trading.yaml` but log is empty |
| `robustness_only` | 29 | Has robustness yaml; never reached paper trading |
| `backtest_only` | 28 | Has experiment registry; never reached robustness |
| `spec_only` | 1 | Has frozen research-spec; never backtested |
| `mandate_only` | 11 | Has mandate; never wrote a hypothesis or spec |
| `idea_only` | **48** | Empty directory or stub only |

**Zombies = no paper trading file AND no executable script:** **112 directories** (close to the agent's 117 estimate; small drift due to script-name regex).

## Zombie Breakdown by Severity

### Tier 1 — Pure Debris (66 directories)
These have nothing of research value: just a directory and maybe a stub yaml.
- 55 are empty or stub-only (no mandate file)
- 11 have a mandate file but never advanced past it

**Examples (idea-only):**
```
a1-cu-au-ratio-equity-lead, a3-dxy-commodity-underperformance,
a6-eem-spy-momentum-divergence, a9-global-breadth-composite,
aaii-sentiment-contrarian, bond-rolldown-carry, btc-eth-regime-cointegration,
btc-momentum-sprint, c7-vix-regime-momentum-failure, cef-discount-mean-reversion,
commodity-carry-contango, correlation-risk-premium, credit-spread-momentum-rotation,
crypto-funding-rate-arb, defensive-factor-vix-overlay, efa-eem-developed-emerging-pairs,
em-carry-trade, erp-valuation-timing, f6-cpi-tips-drift, fx-carry-basket
```
(Full list: 55 idea-only + 11 mandate-only)

**Examples (mandate-only):**
```
ceo-narcissism-short, dual-momentum-antonacci, fomc-hedging-tlt-vol,
fomc-pre-drift-spy, mda-forward-density-earnings, opex-calendar-pinning,
tenk-boilerplate-short, tenk-readability-momentum, tenk-riskfactor-bloat,
wsb-momentum-reversal, yield-curve-momentum
```

These look like brainstorm dumps. NLP-themed ones (tenk-*, wsb-, ceo-narcissism) suggest a shelved alt-data initiative.

**Recommendation:** Move all 66 to `data/strategies/_archive/idea-graveyard/`. Lossless — preserves the brainstorm record without polluting the lifecycle dashboard.

### Tier 2 — Research Output Without Promotion Path (57 directories)
These ran experiments but never made it to paper trading. They're not pure debris — they have actual evidence — but they've been stuck for weeks.

| Sub-bucket | Count | What's there |
|---|---|---|
| robustness_only | 29 | Robustness yaml exists; some have PASS verdicts but no paper-trading.yaml ever created |
| backtest_only | 28 | Have experiment registry; never had robustness run |

**Critical sub-finding:** the 29 robustness-only directories include strategies that **passed their robustness gate but were never added to `LEAD_LAG_PARAMS`** in `run_paper_batch.py`. This is the same gap that left `lqd-soxl-sprint` stuck until I added it earlier today (commit bb35be5).

Selected examples (from earlier lifecycle dashboard, "Robustness" state with results):
- ashr-eem-lead-lag (2 trials, ROB done)
- copper-gold-ratio-mr (1 trial, ROB done)
- correlation-surprise-regime (1 trial, ROB done)
- gold-silver-ratio-mr (1 trial, ROB done)
- spy-tlt-corr-sign-change (no trials but ROB done)
- xlf-spy-lead-lag (1 trial, ROB done)
- xlu-spy-inverse-lead-lag (1 trial, ROB done)

**Recommendation:** Triage in two passes.
1. For each of the 29 robustness_only: read the result, decide PASS→register-in-paper-batch OR FAIL→archive.
2. For each of the 28 backtest_only: decide whether to run robustness OR archive.

This is ~1 hour of triage work, not days.

## Why This Happened

1. **No retirement enforcement.** The lifecycle has 9 forward states but no automated state-transition for "abandoned." A directory created during ideation persists forever.
2. **Script-per-strategy pattern (F-03 to come)** means even running robustness requires authoring 500 LOC. So strategies that pass backtest get stuck waiting for the human to write their robustness script.
3. **Sprint cadence rewards new strategy creation** more than completing the lifecycle on existing ones. Sprints 1-9 added ~30 new strategies; the rate of advancement past robustness is ~10/month.
4. **No cost to keeping a directory alive** — the lifecycle dashboard counts artifact existence, not activity.

## Recommended Cleanup Script

```python
# scripts/audit_zombie_strategies.py
# Move directories matching {idea_only, mandate_only} bucket to _archive/idea-graveyard/
# Move directories in {robustness_only, backtest_only} bucket older than 14 days
#   without progress to _archive/stuck/
# Print summary table.
```

**Effort:** 1 hour to write, 30 min to triage outputs.
**Payoff:** Strategy directory drops from 170 → ~57. `/lifecycle` becomes meaningful. New mandates have a clean namespace to work in.

## Confidence

**Very high** on counts (executed against current filesystem state). **High** on remediation viability — the existing lifecycle artifact format makes archive-vs-active a one-line YAML field add (`status: archived`).

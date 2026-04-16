# Stash Cleanup Follow-up: Residual stashes from apr17-cleanup (2026-04-17)

**Beads:** llm-quant-ejal (P3)
**Team:** apr17-fixes
**Reviewer:** stash-cleaner
**Prior review:** `docs/investigations/stash-review-apr17.md` (llm-quant-lb2a, closed)

## Starting State

Two stashes present on master after the previous stash-review pass:

```
stash@{0}: On master: d10-monitor: teammate in-flight files
stash@{1}: On master: teammate-work-apr17-cleanup
```

Both sit on top of master HEAD `d624ef5` (fix(prompts): align trade cap with risk.toml).

## Stash@{1} — DROPPED (fully redundant)

**Files:**
- `data/strategies/xlk-xle-sector-rotation-v1/paper-monitor-2026-04-17.md` (+127)
- `docs/investigations/dormant-paper-strategies.md` (+85)
- `scripts/run_paper_batch.py` (+3)

**Verification** (per-file byte diff `git show stash@{1}:<path>` vs `git show HEAD:<path>`):

| File | Diff vs HEAD | Committed in |
|---|---|---|
| `paper-monitor-2026-04-17.md` | Identical | `ff8744b` |
| `run_paper_batch.py` | Identical | `1e9dc57` / `2ca9f0f` |
| `dormant-paper-strategies.md` | Minor early-draft wording ("EURJPY/JPY symbol not in ALL_SYMBOLS" vs HEAD's cleaner "JPY feed not in ALL_SYMBOLS"; dropped trailing dry-run stub lines) | `1e9dc57` (polished text) |

The dormant-paper-strategies.md differences are strictly cosmetic prose edits that were improved in the committed version. No signal, parameter, or data lost.

**Action:** `git stash drop stash@{1}` executed (dropped `64abb8b2`).

## Stash@{0} — RETAINED, flagged for human review

**Files:**
- `data/strategies/fraud-detector-results.yaml` (6 lines changed)
- `scripts/run_paper_batch.py` (+3 lines)

**Verification:**

| File | Diff vs HEAD | Notes |
|---|---|---|
| `run_paper_batch.py` | Identical | Same tlt-tqqq-sprint registration already committed in `1e9dc57` / `2ca9f0f` |
| `fraud-detector-results.yaml` | **Different** — new shuffle/inversion numbers for `gld-slv-mean-reversion-v4` |  |

### Detail of the unique hunk

```
- slug: gld-slv-mean-reversion-v4
HEAD:                              STASH:
  shuffle_real_sharpe: 0.9478        shuffle_real_sharpe: 1.2346
  shuffle_95th: 1.429                shuffle_95th: 1.5534
  shuffle_p_value: 0.274             shuffle_p_value: 0.169
  shuffle_passed: false              shuffle_passed: false         <-- same verdict
  time_in_market_pct: 0.3191         time_in_market_pct: 0.3848
  time_in_market_passed: true        time_in_market_passed: true   <-- same verdict
  inversion_original_sharpe: 0.9478  inversion_original_sharpe: 1.2346
  inversion_inverted_sharpe: 1.0753  inversion_inverted_sharpe: 1.0521
  inversion_differential: -0.1274    inversion_differential: 0.1825
  inversion_passed: false            inversion_passed: false       <-- same verdict
```

### Interpretation

- The stash represents a **re-run** of the non-deterministic shuffle / inversion fraud detectors on the same strategy, using a different random seed (or an updated return series).
- All three boolean verdicts (`shuffle_passed`, `time_in_market_passed`, `inversion_passed`) are identical between HEAD and stash. The strategy remains REJECTED under both runs.
- Text search across the entire repo (`1.2346`, `1.5534`, `0.1825`) returns zero matches — these numbers live only in the stash.

### Why this was NOT auto-dropped

Per the stash-cleaner task rules: *"If a stash contains UNIQUE changes not in any commit, STOP and flag via `bd human`."*

Although the practical impact is low (same FAIL verdict; fraud-detector script is reproducible in-repo so a future run can always regenerate numbers), the content is not byte-identical to any commit. Conservative interpretation: retain and flag.

**Recommendation to human:** Likely safe to drop. The numeric diff is shuffle-noise on a rejected strategy, not meaningful lost research. If keeping the newer numbers is desired, the correct action is to re-run the fraud detector end-to-end and commit cleanly rather than restoring a stale stash. But the decision is yours.

## Action Summary

| Stash | Decision | Reason |
|---|---|---|
| stash@{1} | Dropped | Fully redundant with `1e9dc57`, `2ca9f0f`, `ff8744b` |
| stash@{0} | Retained, flagged | Unique fraud-detector numeric re-run on rejected strategy; defer to human |

Bead `llm-quant-ejal` left OPEN pending human call on stash@{0}. Label `human` added.

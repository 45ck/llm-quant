# Stash Review: stash@{0} (2026-04-17)

**Beads:** llm-quant-lb2a (P3)
**Team:** apr17-cleanup
**Reviewer:** stash-reviewer

## Stash Identification

- **Stash ref:** `stash@{0}`
- **Label:** `WIP on master: c9ac29e fix: data pipeline persistence — bulk upsert + Windows encoding`
- **Base commit:** `c9ac29e`
- **Master head at review time:** `6f05f42` (approximately 100 commits ahead of stash base)

## Stash Contents Summary

- **Files changed:** 39
- **LOC delta:** +660 / -449
- **Character of changes:** Automated ruff / code-quality cleanup — import pruning, type annotation touch-ups, f-string reformatting, `# noqa` removals, blank-line normalization.
- **Touched areas (high level):**
  - `scripts/` (8 files) — tearsheet, HRP weights, portfolio optimizer, robustness runners
  - `src/llm_quant/backtest/` (6 files)
  - `src/llm_quant/brain/`, `data/`, `nlp/`, `regime/`, `risk/`, `surveillance/`, `trading/`, `signals/`, `analysis/`, `config.py`, `arb/detector.py`
  - `tests/` (2 files)

## Relevance Analysis

### Commit 0314f13 is the twin of this stash

Commit `0314f13 chore: code quality improvements across 39 files (ruff fixes + type hints)` (Apr 1 2026, 134 ruff fixes, 338 -> 204 errors) modifies the exact same 39 file set with near-identical LOC accounting (+659 / -448 vs stash's +660 / -449).

Unified-diff comparison between `git stash show -p stash@{0}` and `git show 0314f13` yields only cosmetic differences plus **one material hunk**:

- File: `scripts/run_tlt_tqqq_sprint_robustness.py`
- Stash changed `fetch_ohlcv(SYMBOLS, lookback_days=5 * 365 + 30)` -> `lookback_days=5 * 365`
- Committed version of `0314f13` kept `5 * 365 + 30`

### That one difference has since been superseded

The script was subsequently rewritten in commit `009566b feat: vol-regime-v2 (F13), TLT-TQQQ Track D, scan v6 new mechanisms` (Apr 1 2026). The file now reads:

```python
LOOKBACK_DAYS = 5 * 365
...
prices = fetch_ohlcv(SYMBOLS, lookback_days=LOOKBACK_DAYS)
```

So the current master file already uses `5 * 365` — exactly what the stash intended — via a new named constant. Applying the stash today would only produce merge conflicts against the rewritten file; there is no unique signal or parameter change that would be lost.

### No unique uncommitted work

Everything else in the stash (the ruff cleanup across 38 files) was committed verbatim in `0314f13`. The full stash is therefore a pre-commit WIP save of work that was captured in the repository two commits later, plus one tiny tweak that a subsequent refactor already folded in.

## Recommendation: **DROP**

Justification: all changes superseded by committed work.

- 38 of 39 files: byte-identical to commit `0314f13` (committed Apr 1 2026)
- 1 remaining hunk (`run_tlt_tqqq_sprint_robustness.py` lookback): semantically preserved by commit `009566b`, which refactored that script and set the exact `5 * 365` value via the new `LOOKBACK_DAYS` constant

Keeping the stash offers zero recoverable value and adds ambient clutter for future sessions. No risk of lost work on drop.

## Action Taken

`git stash drop stash@{0}` executed after this review was written.

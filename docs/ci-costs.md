# CI cost and delivery

Keep necessary verification and ship useful changes. Reduce redundant candidates
and repeated work; a smaller bill is not evidence of adequate QA.

## Repository workflow

Read `RESEARCH_POLICY.md` and target `master`. Preserve the existing fast-before-commit and slow-before-PR requirements; focused research checks complement them.

## Choose checks by the changed risk

| Change | First check | Required follow-through |
| --- | --- | --- |
| Commit gate | `noslop check --tier=fast`, or `black --check .` and `ruff check .` | Use the existing supported alternative; do not bypass installed hooks. |
| PR gate | `noslop check --tier=slow`, or `pytest`; run `mypy .` for the current quality workflow | Use the full configured environment and report failures or unavailable prerequisites. |
| Decision Lab/research | `PYTHONPATH=src python -m pytest -q tests/test_decision_lab*.py`; existing research docs/standards/catalog checks | Retain offline accounting, leakage, source identity and evidence consistency checks. |

Focused checks shorten diagnosis. They do not replace existing merge, release,
security, compatibility, coverage or acceptance requirements. Shared runtime,
dependency, test-runner or workflow changes need the broader applicable gates.

## Publish a coherent candidate

1. Inspect current branches and open PRs. Assign one integrating writer for shared
   CI, configuration and lockfiles; preserve other agents' work.
2. Finish one reviewable increment, run its focused checks and required local gate,
   then publish together. Keep ownership claims timely where the repository requires them.
3. Read the actual remote result for that commit. On failure, capture the first
   relevant error, diagnose locally where possible, and retry the smallest failed
   job set only when a fix or transient-failure explanation warrants it.
4. Request a preview or manual heavy run when someone needs that result. Reuse
   valid evidence for an unchanged candidate; do not poll by dispatching workflows.
5. Record commit/PR, check scope, commands, outcomes, run links and retry reason in
   the existing handoff or change record. Keep blocked and unrun checks explicit.

## CI/CD warnings

> [!WARNING]
> Quality and Decision Lab checks have separate obligations. Optional native research integrations are manual and selected by backend; do not dispatch every backend for a documentation change. Existing report schedules are not a reason to create a new polling workflow.

GitHub can repeat dependent jobs when a failed job is rerun. A cancelled or
previously green run is not proof about a changed candidate. Do not add `[skip ci]`,
disable required jobs, lower thresholds or use forced green results as a cost control.
Path-filter changes require coverage proof and compatible required-check behaviour.

Research only: no live trading, paid feeds, brokerage calls or strategy promotion. Never initialize an empty evidence database to make a report workflow pass, or relabel billing-blocked or partial tests as successful.

## Measure the outcome

Use the existing run/deployment records: accepted change, commit, run attempt, job
duration, outcome and retry reason. Compare runner minutes, elapsed delivery time
and failed/repeated work per accepted change. Keep provider CPU usage, credits and
actual invoices separate; job counts and elapsed wall time alone are not billed cost.
This guidance adds no schedule, workflow, dependency or paid service.

References: [GitHub job reruns](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)
and [skipped workflow checks](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs).

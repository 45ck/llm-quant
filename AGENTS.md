# Agent guidelines

Read **RESEARCH_POLICY.md** first. It is the current research-status authority;
older track documents and role prompts are not evidence of validated performance.
Preserve risk controls and source history. Research only; no live trading approval.

## Before every commit

```sh
noslop check --tier=fast
# or: black --check . && ruff check .
```

## Before opening a PR

```sh
noslop check --tier=slow
# or: pytest
```

Also run the affected research checks:

```sh
PYTHONPATH=src python -m pytest -q tests/test_decision_lab*.py
PYTHONPATH=src python -m llm_quant.decision_lab.catalog --check
python scripts/check_research_standards.py
python scripts/check_research_docs.py
```

Never use `git commit --no-verify`, force-push, weaken protections or hide a failed
check. Fix lint errors; justify any targeted suppression. If a required tool or
source is unavailable, say exactly which check was not performed; do not turn
that into a claimed pass. Any owner-authorized research publication with disclosed
limitations must retain those limitations in the PR and evidence receipt.

Use the resolved research environment for reproducible numerical work. No network,
brokerage calls or credentials in offline tests. Source/schema corruption, future
information, incorrect fees, missing outcomes and status drift need negative tests.

`noslop doctor` checks the optional local gate setup. Maintainers use Beads when
available; public contributions must not require private tracker access.

## Repo-local quantitative skills

Read [the skills index](docs/skills/README.md) and select the matching procedure in
`.claude/skills/quant-*/SKILL.md` for integration, data, backtesting, uncertainty,
experiment or QA work. The current RESEARCH_POLICY.md still takes precedence.
Optional libraries must not be imported or installed merely to discover a skill.

## CI cost and delivery

- Batch a coherent implementation candidate after focused local checks; avoid repeated cosmetic pushes or unchanged full reruns.
- Use one integration owner for shared CI, dependency and lockfile changes; reconcile concurrent work before publishing.
- Read failed logs, reproduce the smallest failure, then retry only justified failed work. Code fixes need a new candidate; a rerun repeats the original revision.
- Keep required QA, security, coverage and release gates. Never bypass hooks, use skip markers or weaken checks to reduce cost.
- Request previews or manual heavy checks only for a concrete review or acceptance need under the existing delivery policy.
- Record the commit, actual commands/results, run links and retry reason; mark unavailable checks NOT_RUN. Runner minutes and deployment counts are not invoices.
- Retain fast/slow gates, offline research checks and `RESEARCH_POLICY.md`. No live-trading or paid-data authority is added.
- Use the [CI cost guide](docs/ci-costs.md) for this repository's commands and warnings.

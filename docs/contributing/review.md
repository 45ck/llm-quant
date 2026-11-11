# Research and code review

[Contributing](../../CONTRIBUTING.md) · [Research guide](research.md)

## Two independent decisions

**Engineering:** is the change understandable, tested and safe to integrate?
**Research:** what conclusion does the evidence actually justify?
A software pass does not imply research success. A well-documented failed hypothesis
can be a useful accepted contribution.

## Review checklist

- [ ] Scope, baseline, units, horizon and evidence type are explicit.
- [ ] Data rights, source identities, vintage and availability times are recorded.
- [ ] Frozen specification precedes confirmatory evaluation, or exploration is declared.
- [ ] Tests cover leakage, missing/stale values and the relevant failure cases.
- [ ] Trial count, costs, parameter choices and exclusions are disclosed.
- [ ] Metrics and figures agree with public-safe machine-readable evidence.
- [ ] Benchmark comparisons use compatible samples and do not pool dependent outcomes.
- [ ] Commands were executed as stated; blocked checks are not called passing.
- [ ] Private data and credentials are absent from code, logs, fixtures and images.
- [ ] A maintainer's disposition is recorded; no automatic trading/promotion is introduced.

Review the complete diff, including workflows. Read-only PR jobs should not request
secrets or run paid network experiments. Do not use `pull_request_target` to execute
untrusted contributor code. Do not hide failed commands with `|| true`.

## What to record on the PR

State the code commit, scope reviewed, checks actually run, material unresolved
issues, and whether results were independently replicated or only inspected.
Separate infrastructure blockers from a numerical failure and from insufficient
evidence. Never describe an unavailable hosted runner as a passing check.

Normal GitHub merge rules still apply. Required checks and protections must not be
bypassed. Publishing explicitly labelled research code with a disclosed limitation
is not the same as approving that model for forecasts or capital use.

## Current evidence boundaries

The [dated audit](../research/model-audit-2026-11-11.md) contains recorded strategy
failures, a reproduced but weak housing diagnostic and missing paper-performance
evidence. Legacy return targets and "proven" wording do not override that audit.
The existing strategy mandates remain in force until explicitly reconciled by a
maintainer. This guide does not silently change them.

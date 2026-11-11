# Recorded-model and reporting audit — 11 November 2026

**Conclusion: no validated investment edge established.** Software checks, historical
research records and prospective trading results have different evidence standards.
This review updates the README without turning old backtests into current forecasts.

## Scope and source freeze

Repository source commit: `a08cac5ee98c96e526b824ac649788d6e1b51bd7`.
The public evidence JSON pins the complete Track D report and latest daily report by
Git blob identity and SHA-256. It includes aggregate results, not provider price histories.

| Source | Role | Git blob SHA-1 |
| --- | --- | --- |
| `docs/research/results/track-d-robustness-results-2026-04-24.md` | Complete eight-strategy historical cohort | `4b8596c512f5a99af0d589ab7a947317988d870f` |
| `reports/daily/2026-11-06.md` | Latest committed daily report at the audited commit | `3967217fa5d6d54f46b1b6b521d60efa20f00488` |
| `data/strategies/agg-soxl-sprint/robustness_results.json` | Corroborates Sharpe and shuffled-signal failure | `b6a00d1d694a1b22ed7922471ce8459f2949055a` |
| `data/strategies/agg-tqqq-sprint/robustness_results.json` | Inspected metric fields expose OOS-ratio label error | `cb69378673da199fff8f539a1f1ddb34b74e04a2` |
| `data/strategies/lqd-upro-sprint/robustness_results.json` | Inspected metric fields expose OOS-ratio label error | `9b8968e37c1f676bc41642cc0b01e3f96e1d2796` |
| `.github/workflows/reports.yml` | Reviewed missing-database initialization and swallowed errors | `58482361adfa02cbf8f48938bc162b201b34b9be` |
| Original `README.md` | Archived claims, not proof of investment returns | `6fb5d13da687a2a6945e068d76f2fa787571dd19` |

Additional corroborating records: `docs/governance/research-tracks.md`, the 22 April
paper-status report (D3 retirement warning), and the 29 April paper-status report
(short samples, differing batch/registry counts, backfilled dates). These records
were read, not independently reconstructed from a trade ledger.

## Findings and remedies

### 1. Missing portfolio evidence was displayed as numerical performance — high impact

The 6 November daily report explicitly lacks snapshots but displays 0.00 ratios,
0.0% return and a hash-chain PASS footer. These are not verified performance results.
The scheduled report workflow initializes an empty database when none is available
and hides generation failures with `|| true`.

**Change:** require an existing nonempty database before installation/generation,
propagate generation failures, and check changed report files before publication.
Missing-snapshot markers or absent/invalid NAV rows block publication. The historical
report is preserved as evidence; the checker labels its return unavailable, not zero.

**Boundary:** the guard is not a financial audit. Positive NAV text alone cannot prove
price freshness, correct period selection, execution, benchmark comparability, or
ledger integrity. Legacy metric calculations and database restoration still require
full integration review. No hosted behavior is claimed tested while Actions is blocked.

### 2. Recorded research advances are not evidence of trading success — high impact

For the specified eight-strategy Track D cohort, five unambiguous thresholds reconcile
to six reported advances and two reported failures. AGG→SOXL fails Sharpe and the
shuffled-signal test; LQD→UPRO fails the Sharpe threshold. These are historical report
results, not newly run backtests. The D3 TQQQ/TMF strategy is separately recorded as
retired after a signal-direction defect changed its reported replication Sharpe.

**Change:** surface failures and evidence type in the README. Remove target returns
and unreproduced portfolio figures from the current headline. Preserve a link to the
old README so the historical record remains inspectable. Do not change strategy
mandates, relax thresholds, or promote a strategy.

### 3. Two CPCV results are labelled as the wrong metric — material interpretation risk

| Strategy | Narrative calls a ratio | JSON mean OOS Sharpe | JSON OOS/IS ratio |
| --- | ---: | ---: | ---: |
| AGG→TQQQ | 1.091 | 1.0912 | 1.0123 |
| LQD→UPRO | 0.664 | 0.6637 | 0.8536 |

**Change:** retain the narrative field verbatim as ambiguous; do not use it to claim
validation of all six gates. The current README states the separate primary-record
fields. Old evidence documents are not silently rewritten. Rounded p=0.0000 values
must not be interpreted as an exact zero probability.

### 4. Housing advantage is not stable across annual start phases — high impact for advice

The initial 36-window calculation reproduces, including model Brier 0.02769988 versus
baseline 0.02780638. All four possible quarterly starting phases were then evaluated
with otherwise fixed settings. Two slightly favour the model; two favour the baseline.
Each phase contains only zero or one terminal loss of at least 10%.

**Change:** show all four phases, not a selected best phase. The chart uses absolute
Brier-score differences rather than an impressive relative percentage on a near-zero
baseline. Within-phase annual outcomes do not overlap, but outcomes across phases do:
they cannot be pooled as independent observations. This follow-up diagnostic was
chosen after the initial result and is not a pristine holdout test.

### 5. Current forecasting is not supported by the inspected data — high impact

The observed input is a rehydration of the earlier browser table export of BIS/FRED
`QAUR628BIS`: 225 quarterly observations from 1970 Q1 to 2026 Q1, real residential
prices across eight large Australian cities. It is not a current Sydney index or an
investable total-return series. Its raw SHA-256 matches the prior export:

`8c030e25325859b66db4619893e6d891885e0d4dbd84e712fcabc10e17713b9a`

The current metadata records rehydration rather than falsely claiming a fresh download.
Native release-aware vintages and a current licensed Sydney history remain unavailable.
The existing 200-day current-data limit rejects the 314-day-old observation label.
Historical reproduction does not override that rejection.

## What was executed

- All 131 focused Decision Lab/evidence regression cases, without network calls.
- Compilation and the offline demo/project commands.
- Housing audit twice with identical input and environment; JSON outputs compared.
- Independent arithmetic recomputation of Brier scores from each fold's probability
  and realized outcome; this does not independently reproduce the bootstrap algorithm.
- Source-byte hash checks and negative publication/freshness controls.
- Two README figures generated from the aggregate JSON and visually inspected.

The test receipt records commands, environment, scope and remaining blockers.
No source dataset, credentials, real client fees or internal financial records are
included in this public increment.

## What was not executed or established

No full legacy test-suite run, native live-provider verification, original trading
backtest replication, independent trade-ledger reconciliation, production workflow
execution, untouched holdout, or prospective forecast track record. Black/Ruff/mypy
could not be installed in the network-restricted runtime. GitHub Actions was already
failing before runner assignment; the earlier public annotation identifies an account
billing lock. No billing setting or required check was changed or bypassed.

The research PR stays a draft until the repository checks can execute. These limitations
are separate from whether the focused test suite passes. There is no live-trading or
model-promotion authorization.

## Reproduction

See the root README for commands. Use the retained source capture and matching snapshot
metadata to reproduce the published run; a different revision requires a new dated run.
The renderer consumes only `docs/evidence/model-audit-2026-11-11.json`. Public readers can
regenerate figures without obtaining the raw provider history.

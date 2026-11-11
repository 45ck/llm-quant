# Decision Lab: verified-history follow-up

Research increment on 11 November 2026. No strategy promotion, live trading,
billing changes, scheduled downloads or private financial-data publication.

## What changed

- The FRED graph CSV URL returned HTTP 404 in a public browser check. The direct
  adapter now targets the public data table and checks series ID, units, frequency,
  source, update metadata and the declared observation range. It fails on schema
  changes rather than inventing or filling values.
- A separate, offline browser-export importer verifies a complete-table receipt:
  source URL, extraction method, timestamp, row count, endpoints and byte hash.
  It retains the exported source and receipt. A receipt is provenance, not a
  provider signature; table exports are never described as original CSV downloads.
- Historical studies require the retained source payload to match the recorded
  hash. Stale historical observations can be evaluated retrospectively without
  loosening the existing current-data gate. Future observations/captures fail.
- A predeclared history-window/block-length grid exposes sensitivity without
  choosing a winner. Rolling-origin checks now include reliability bins, central
  interval coverage and interval scores, alongside the training-only Brier baseline.
- The model rejects empty wealth matrices and uses an absolute-only weight-sum
  tolerance. Yahoo calls explicitly retain missing rows, disable repairs and raise
  errors; their live execution still requires a network-enabled environment.
- The focused workflow runs both test modules and compilation on ubuntu-24.04,
  with its existing 3-minute timeout, read-only permissions and no cron.

## Real-data evidence

Public source: https://fred.stlouisfed.org/data/QAUR628BIS

Source credit: National sources, BIS Residential Property Price database,
https://www.bis.org/statistics/pp.htm, via FRED.

| Item | Observed or executed result |
| --- | --- |
| Series | QAUR628BIS; real residential property prices; Index 2010=100 |
| Geography | Eight large Australian cities; not Sydney alone |
| History | 225 quarterly observations, 1970 Q1 through 2026 Q1 |
| Source last updated | 27 August 2026, 11:03 AM CDT, as displayed |
| Acquisition | Browser table-cell export, captured 11 November 2026 |
| Export SHA-256 | `8c030e25325859b66db4619893e6d891885e0d4dbd84e712fcabc10e17713b9a` |
| Snapshot SHA-256 | `a7f16eb089d8924bb4b2f44b90b5226a10fe380940096a98e9729e7ae05494f0` |
| Sensitivity grid | 80/120/224 return periods x block lengths 1/4/8 |
| Simulation | 2,000 paths per setting, 4-quarter horizon, seed 20261111 |
| Retrospective diagnostic | 80-quarter initial training, block 4; 36 non-overlapping annual outcome windows |
| Severe-loss labels | 1 realized decline of at least 10% in these 36 windows |
| Model Brier / baseline Brier | 0.02769988 / 0.02780638; lower is better |
| 95% interval coverage | 33/36 = 91.7%; descriptive, not a confidence guarantee |
| Repeat run | Identical JSON outputs for identical inputs/environment |
| Current-data gate | Rejected: 314 days from the observation label exceeds the specified 200-day limit |

The Brier difference is tiny and only one severe-loss label is observed in these
particular evaluation windows. That is not credible evidence of a calibrated
crash predictor, investment edge or general historical crash frequency. No
probability from this study is promoted to a current market forecast. The dataset
is latest-vintage, not a reconstruction of each historical release. The quarterly
label is the first day of the quarter; freshness here is measured conservatively
from that label, not the quarter end or publication date.

Source histories, private inputs and full run outputs remain in gitignored local
storage. This document publishes an audit summary, not the vendor history.

## Commands

In a network-enabled environment the existing fetch command now requests HTML:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab fetch fred QAUR628BIS --online
```

For an explicitly declared browser table export and its receipt:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.fred_table \
  EXPORT.csv RECEIPT.json
```

The receipt schema is exercised in `tests/test_decision_lab_study.py` by
`make_export`. Never relabel a browser export as a direct HTTP source capture.

For the 225-observation historical dataset (use the printed snapshot path):

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.study SNAPSHOT.json \
  --as-of 2026-11-11 --max-age-days 200 --horizon 4 \
  --blocks 1 4 8 --windows 80 120 224 --validation-block 4 \
  --min-train 80 --paths 2000 --seed 20261111

PYTHONPATH=src python -m pytest -q tests/test_decision_lab*.py
```

`study` is explicitly historical. It does not bypass the current `risk` command's
freshness rejection or authorize portfolio recommendations.

## Verification and remaining limits

**Local:** 96 focused tests passed on Python 3.13.5, NumPy 2.3.5 and pytest 9.0.2.
Compilation and the synthetic-demo/project CLIs passed. The live-source browser
export was imported and evaluated successfully. The two repeated historical
studies produced byte-identical JSON; the stale current-data command exited 2
without creating an output file. Focused combined statement/branch coverage is
88%; this is not whole-repository coverage or proof of statistical validity.

**Hosted CI:** the public page for run 42921211297 states:

> The job was not started because your account is locked due to a billing issue.

Source: https://github.com/45ck/llm-quant/actions/runs/42921211297

This account-level blocker cannot be repaired by changing model code. No billing
or account permission was changed, no checks bypassed and no repeated reruns
requested. The PR remains a draft until appropriate checks can execute.

**Not completed:** a native Python HTTP download in this restricted runtime;
live Yahoo ingestion; full-repository tests; Black/Ruff/mypy checks; exchange
calendars and independently verified corporate actions; historical-release-aware
vintages; licensed Sydney data; an untouched holdout; independent replication;
and a forward forecast record. The runtime cannot resolve external hosts and
formatting/type-checking packages could not be installed. The public table and
its browser export were inspected independently; the direct HTML adapter is
covered by fixture tests, not falsely described as live HTTP-tested.

# Historical reports

[Repository home](../README.md) · [Evidence review](../docs/research/model-audit-2026-11-11.md)

| Directory | Contents |
| --- | --- |
| `daily/` | Dated portfolio snapshots, trades and metrics when observations exist. |
| `weekly/` | Period aggregates and daily breakdowns. |
| `monthly/` | Period metrics, benchmarks and trade analysis. |
| `track_c/` | Existing structural-arbitrage scan reports. |

Read a report's actual observation dates and evidence, not just its filename.
A generated page with no portfolio snapshots is **not** a verified zero return.
A hash-chain check tests the recorded chain; it does not prove that a complete
ledger, positions or economic results exist. Earlier reports are preserved rather
than silently rewritten to make the record look better.

## Generation and publication

From a configured full development environment at the repository root, with
`PYTHONPATH=src`, use the existing report generator:

```sh
python scripts/generate_report.py daily --date YYYY-MM-DD
```

Replace the date with the intended period. Do not run on an empty substitute
database and present the result as observed performance. The workflow refuses a
missing database, propagates generation errors and checks changed reports for
portfolio evidence before publication. That publication guard is not an independent
financial audit. Review logs for private fields before sharing any output.

See [tests](../tests/README.md) for the distinction between focused checks and full
repository verification. Publication does not authorize live execution or promotion.

# Repository map

[Documentation](README.md) · [Contribution guide](../CONTRIBUTING.md)

## Stable responsibilities

| Location | Responsibility | Do not put here |
| --- | --- | --- |
| `src/llm_quant/` | Reusable implementation; existing package/import paths remain stable. | Generated results or private credentials |
| `src/llm_quant/decision_lab/` | Research-only source checks, scenarios and diagnostics. | Brokerage execution or automatic promotion |
| `config/` | Explicit configuration and labelled synthetic examples. | Personal account settings or secrets |
| `scripts/` | Thin entry points, batch helpers and figure reproduction. | A parallel implementation hidden from tests |
| `tests/` | Deterministic tests, synthetic fixtures and regression cases. | Tests that silently spend money or trade |
| `data/strategies/` | Existing strategy lifecycle records and experiment registries. | Unlicensed raw price histories |
| `docs/research/` | Hypotheses, interpretations, replications and dated results. | Unlabelled targets presented as returns |
| `docs/evidence/` | Public-safe aggregates, source identities and verification receipts. | Raw private or restricted vendor data |
| `docs/assets/` | Figures generated from identified evidence. | Hand-edited numbers or fabricated charts |
| `reports/` | Historical paper and scan reports. | Zero-filled substitutes for missing observations |
| `research/` | Existing experiment manifest, prompts and result artifacts. | A competing strategy lifecycle registry |
| `research-local/` | Gitignored downloads, private inputs and scratch run output. | Files that must be accessible to public reviewers |
| `.github/` | Public issue/PR templates and workflows. | Secrets, payment changes or hidden research policy |
| `.beads/` | Existing maintainer work tracking. | External contributors' fabricated status updates |

## Naming and provenance

Use `YYYY-MM-DD` in new dated report names and descriptive kebab-case for documents.
Preserve existing slugs and paths: scripts, source hashes and historical links may
refer to them. Add an index instead of moving a whole legacy tree for appearance.

A research result must name its source commit, dataset identity/vintage, commands,
parameters, seed, environment and limitations. The [result template](templates/result.md)
provides the fields. Use `docs/evidence/<study>-<date>.json` for public-safe
machine-readable aggregates and reference them from the dated Markdown report.

For a new strategy, reuse `data/strategies/<stable-slug>/` and the existing lifecycle
schema. For a document-only replication, use `docs/research/` plus a linked issue.
Do not invent a paid-data export or a new success label just to complete a template.

## Navigation maintenance

The lightweight `scripts/check_research_docs.py` checker validates local inline links
in the maintained entry-point documents. It is deliberately not a whole-repository,
external-URL or Markdown-fragment validator. Add new entry points to its `DOCUMENTS`
list. A missing target fails the check; it is not silently ignored.

Existing source paths, strategy IDs, historical evidence and the research manifest
are retained. This organisation pass changes discovery and contribution paths,
not model mandates or the underlying evidence.

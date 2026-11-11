# Contributing

[Repository home](README.md) · [Research guide](docs/contributing/research.md) ·
[Reviewer checklist](docs/contributing/review.md) · [Repository map](docs/repository-map.md)

## Choose a contribution

| Contribution | What a useful first PR looks like |
| --- | --- |
| Replication or negative result | Reproduce one named claim; show agreements, discrepancies and missing inputs. |
| Data quality | One documented source contract, a permitted fixture, and failure-case tests. |
| Model research | A bounded hypothesis and baseline with a frozen evaluation plan. |
| Engineering | A regression test and the smallest justified fix. |
| Documentation | Correct a misleading claim, improve an example or explain a limitation. |

The current priority is reproducible evidence, not a target return. D3 TQQQ/TMF
is **recorded as retired**, not a proven candidate. See the [current evidence
review](docs/research/model-audit-2026-11-11.md). Do not reinstate failed claims
from old README text or stale paper logs.

## Public entry point

Open an issue using the repository's research, replication or data/engineering
template. Plain issues remain available. For a small documentation correction,
a PR is sufficient. No private account, Beads installation or LLM key is required.
Maintainers use Beads for internal work tracking and link accepted public issues;
contributors should not edit `.beads/` to manufacture workflow state.

Templates become selectable in GitHub once merged to the default branch. On a
feature branch, their Markdown files can be copied into a normal issue.

## Local setup

Fork the repository, clone your fork, and create a focused branch from `master`.
Use Python 3.12+ and a virtual environment:

```sh
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install numpy==2.3.5 pytest==9.0.2
```

For focused research, set `PYTHONPATH=src` in your shell. PowerShell:
`$env:PYTHONPATH = "src"`. From the repository root:

```sh
python -m llm_quant.decision_lab demo
python -m pytest -q tests/test_decision_lab.py tests/test_decision_lab_study.py tests/test_decision_lab_evidence.py
python -m unittest discover -s tests -p test_research_docs.py
python scripts/check_research_docs.py
```

For the **full existing lab**, install `python -m pip install -e ".[dev]"`.
Install Black separately when running the repository's Black gate. Full setup has
more dependencies; do not claim a focused run proves the full suite passes.
The legacy LLM path needs separately configured credentials; offline research does not.

## PR workflow

Use one concern per PR and target `master`. Explain the hypothesis or defect,
changed files, commands actually executed, and remaining limitations. Attach
machine-readable aggregates and permitted synthetic fixtures where appropriate.
Report failures and unexecuted checks explicitly.

Before committing, follow [AGENTS.md](AGENTS.md): `noslop check --tier=fast`, or
`black --check . && ruff check .`. Before opening a PR, run the slow gate or
`pytest` in the full environment. Run `mypy .` to match the current quality workflow.
Do not bypass hooks, force-push without explicit approval, disable checks, or label
unexecuted checks as passing. Infrastructure failures are not software passes.

Use conventional commit subjects, for example `fix(data): reject duplicate dates`
or `docs(research): record an unsuccessful replication`.

## Research submissions

Read the [research guide](docs/contributing/research.md) before a material experiment.
Use the [proposal and result templates](docs/templates/README.md). Preserve strategy
IDs and use the existing lifecycle records under `data/strategies/`; do not create
a second strategy registry or overwrite a frozen result. Every tried variant counts.

Code uses type hints. Use Polars in the core and pandas only where an existing
provider boundary requires it. Currency must be explicit: the legacy paper book is
USD; Decision Lab supports an explicit AUD lane. Never silently mix units.

## Data, safety and attribution

Keep downloaded data, private financial inputs, credentials and account logs out of
public commits. Use `research-local/` for local work and review `git diff --cached`
before pushing. Gitignore is not access control. Add retrieval instructions and
source hashes instead of uploading a dataset without redistribution permission.

Declare AI assistance and verify its code, numbers and citations. Credit original
research and other contributors. Review claims rather than personalities. Use a
private security-reporting route where available; never put a live secret in an issue.

A code merge is not approval for live trading, paid data, promotion or capital use.
Contributed code is covered by [MIT](LICENSE); dataset rights remain separate.

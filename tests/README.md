# Testing scope

[Contributing](../CONTRIBUTING.md) · [Evidence catalogue](../docs/evidence/README.md)

## Offline research checks

From the repository root with `PYTHONPATH=src`:

```sh
python -m pytest -q tests/test_decision_lab.py tests/test_decision_lab_study.py tests/test_decision_lab_evidence.py
python -m unittest discover -s tests -p test_research_docs.py
python scripts/check_research_docs.py
python -m llm_quant.decision_lab demo
```

The first command requires NumPy and pytest. Documentation checks use only the
standard library. No test here needs brokerage credentials or a paid data call.
Synthetic observations exercise contracts; they are not market validation.

## Full repository checks

Install the full development environment described in [CONTRIBUTING.md](../CONTRIBUTING.md),
then run `python -m pytest`, Black, Ruff and mypy as required by existing gates.
A focused pass is not a substitute. Record versions, exact commands, scope, test
counts, skips, failures and blocked steps in a dated evidence receipt.

## Add regression coverage

Include invalid/missing/stale data, date/currency/basis mismatches, future-information
leakage and deterministic seeds where relevant. A failed model is not repaired by
weakening its test. Network/paid experiments must be explicit opt-in and must not
run automatically for an untrusted pull request.

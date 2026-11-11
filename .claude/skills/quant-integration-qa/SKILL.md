---
name: quant-integration-qa
description: Verify research adapters, skills, native dependencies and reproducibility in llm-quant; use Hypothesis, contract tests and native integration checks.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Integration Qa

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

Inventory installed versions without importing optional backends. Run focused contract tests, then the selected native test. Distinguish mocks, skipped dependencies, actual runtime passes and full-repository checks. Run generated accounting properties through Hypothesis when installed.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations doctor
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

A selected CI job must fail if no selected tests run or any selected test skips. Keep the full quality gates and billing boundaries intact. No automatic installs, network calls, live trades or permission changes are authorized by this skill.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

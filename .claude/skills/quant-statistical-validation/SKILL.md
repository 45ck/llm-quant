---
name: quant-statistical-validation
description: Assess probability calibration, multiple-testing corrections, sparse tails, dependent observations and uncertainty using statsmodels and arch.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Statistical Validation

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

State the estimand, dependence assumptions and complete hypothesis family. Reuse the validation module and compare against a training-only base rate. Use HAC for a declared mean test, Holm for the complete tested family, and arch for dependence-aware resampling.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe statsmodels
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

Do not pool overlapping outcomes as independent observations. A small Brier difference or one tail event cannot establish a reliable predictor. Refuse exact independent-trial intervals when independence is not justified.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

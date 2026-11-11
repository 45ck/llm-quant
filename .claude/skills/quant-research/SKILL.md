---
name: quant-research
description: Route quantitative research, replication, model evaluation and integration work in llm-quant. Use before choosing a library or changing a research claim.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Research

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

Read RESEARCH_POLICY.md, the research register and the relevant model audit. Define a falsifiable question and a simple baseline. Separate engineering validation, retrospective evaluation and prospective evidence. Choose the smallest library that answers the question; do not install every profile.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations doctor
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

Record the hypothesis, source rights, time/currency basis, validation design, trial family, resource limit and stopping rule before evaluation. Keep failures and inconclusive results. Never optimize the narrative toward a return target.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

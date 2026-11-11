---
name: quant-experiments
description: Record experiment evidence, reproduce runs and maintain the llm-quant result register or forward forecast journal. Use Qlib only for explicit local metadata recording.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Experiments

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

Bind every result to code, input hashes, versions and settings. Keep observed, synthetic, retrospective and prospective evidence separate. Use the local Qlib adapter only after explicit selection of a research-local destination; never export private rows or credentials.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations doctor
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

A local hash chain or timestamp is not independent attestation. Forecast outcomes are settled only after maturity. Never edit a previous prediction to match the outcome. Do not initialize Qlib trading/data services or load untrusted pickle artifacts.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

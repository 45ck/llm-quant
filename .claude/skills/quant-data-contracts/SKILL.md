---
name: quant-data-contracts
description: Validate financial datasets, OHLC frames, timestamps, currency, corporate actions and licensing; use Pandera and the llm-quant snapshot contracts.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Data Contracts

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

Read the source contract before using observations. Preserve nulls, duplicate detection, native frequency, currencies and retrieval/vintage metadata. Use pandera_ohlc only at the pandas boundary; do not silently fill, coerce, resample or repair data.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe pandera
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

Stop for unknown distribution adjustments, mixed return bases or unlicensed histories. Preserve original snapshots and hashes. Historical labels alone do not prove the data was available at the forecast origin.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

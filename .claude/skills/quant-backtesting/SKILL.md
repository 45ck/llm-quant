---
name: quant-backtesting
description: Run cost-aware chronological backtests, execution-timing checks, baselines and future-data leakage tests in llm-quant; use scikit-learn and optional Backtesting.py.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Backtesting

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

Freeze the split plan, execution lag, costs and holdout before seeing scores. Use compare-synthetic for engine validation, then permitted observed snapshots with the same plan. Backtesting.py requires explicit AGPL acknowledgement and is a separate single-instrument timing diagnostic.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations compare-synthetic --method sklearn_inverse_volatility
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

Perturb future outcomes and verify fitted allocations do not change. Check fees against hand calculations. Never retune on the reserved holdout or present synthetic results as market returns.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

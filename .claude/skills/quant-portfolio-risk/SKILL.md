---
name: quant-portfolio-risk
description: Fit and compare training-only allocations and risk estimates using scikit-learn, skfolio, CVXPY and arch. Use for shrinkage, CVaR, GARCH or bootstrap tasks.
license: MIT
compatibility: Python 3.12+; run from the llm-quant repository root. Optional integrations require separate dependency installation.
metadata:
  version: "1.0"
  repository: "45ck/llm-quant"
---

# Quant Portfolio Risk

Read [RESEARCH_POLICY.md](../../../RESEARCH_POLICY.md) first. Research only;
no validated investment edge or live-trading approval.

## Procedure

Start with equal weight and cash. Fit covariances and allocations only on the supplied training slice. Use skfolio for mean-risk fits, CVXPY for an independent empirical CVaR objective, and arch for dependence-aware mean intervals or GARCH variance.

## Executable entry point

Run from the repository root; this example performs no market-data download:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe sklearn
```

Missing optional dependencies are a limitation, not a successful run. Obtain
explicit approval before any package installation or paid resource use. These
instructions do not grant permission for trades, account changes or publication.

## Required evidence and stop conditions

Reject nonconvergence, invalid weights and degenerate covariance. Report solver, alpha, block length, sample size and numerical tolerance. Conditional variance, mean intervals and empirical tail risk are not crash forecasts.

Return the actual commands executed, versions, input/code identities, results,
failures/skips and the interpretation limits. Do not fabricate a source or claim a
mocked library call demonstrates a working native integration.

## Reference

[Integration APIs, source links and QA limits](../../../docs/research/integrations.md)

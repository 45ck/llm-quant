# Established-library integrations

[Home](../../README.md) · [Current research policy](../../RESEARCH_POLICY.md) ·
[Skills](../skills/README.md) · [Attribution](../../THIRD_PARTY_RESEARCH.md)

**Callable adapters, not just references. No validated investment edge or live-trading approval.**
All imports are explicit. Missing libraries fail rather than substitute a different
model. Most backends are optional so a simple offline check does not install a
complete ML or solver platform. Integration tests use synthetic observations.

| Project | Implemented use | Verification boundary |
|---|---|---|
| scikit-learn | Time-ordered splits, shrinkage covariance, inverse-volatility allocation | Executed with the real installed library |
| statsmodels | HAC mean inference and Holm correction; SciPy supplies exact event-rate intervals | Executed with the real installed library |
| skfolio | Minimum-variance and CVaR fits on training slices | Adapter contracts tested; native dependency absent locally |
| arch | Stationary bootstrap and zero-mean Student-t GARCH(1,1) | Adapter contracts tested; native dependency absent locally |
| Backtesting.py | Next-open entry/exit and commission diagnostic | Runtime test supplied; AGPL opt-in; native dependency absent locally |
| Microsoft Qlib | Local QlibRecorder/MLflow experiment metadata | Adapter contracts tested; native dependency absent locally; no Qlib data/model training pipeline claimed |
| CVXPY | Long-only, capped empirical CVaR linear program | Runtime and cross-engine tests supplied; native dependency absent locally |
| Pandera | Strict typed OHLC schema | Native schema test supplied; prerequisite OHLC checks executed |
| Hypothesis | Generated accounting/conservation properties | Property test supplied; generator dependency absent locally |

## Run a core integration

From the repository root, in the core research environment:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.integrations doctor
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe sklearn
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe statsmodels
PYTHONPATH=src python -m llm_quant.decision_lab.integrations compare-synthetic \
  --method sklearn_inverse_volatility
```

`doctor` checks distribution metadata, not successful imports or tests. A probe
executes real library code and reports a version and result. `compare-synthetic`
feeds an upstream allocator into the cost-aware engine, with a gap, chronological
folds and a separately reserved holdout. It also reports the existing simple
baselines using the same accounting. It never selects the best-looking method.
The default fixture is explicitly artificial, equally spaced and denominated AUD.
No download or real market forecast occurs.

## Run optional native checks

Install only the desired profile in its own environment. These are bounded
compatibility requirements, not a fully locked all-library environment:

```sh
python -m pip install -r requirements/integrations/skfolio.txt
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe skfolio
PYTHONPATH=src python -m pytest -q tests/integrations/test_native.py -k skfolio

PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe arch
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe cvxpy
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe pandera
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe backtesting --acknowledge-agpl
PYTHONPATH=src python -m llm_quant.decision_lab.integrations probe qlib \
  --qlib-directory research-local/qlib-runs
```

Install the matching profile before each command. Qlib writes only explicitly
selected scalar run metadata to a local filesystem experiment store. It refuses
an external MLflow URI and paths outside `research-local/`; it neither initializes
a market-data provider nor loads untrusted pickle objects. No service is started.
Backtesting.py is a single-instrument execution diagnostic, not a multi-asset
allocation simulator. It does not share identical fill/rounding conventions with
the return-series engine; never compare their results as though they do.

## QA and the meaning of a pass

The core contract suite checks future-data isolation, finite inputs, covariance
failures, weight constraints, solver failures, source identities, OHLC envelopes,
local-only recording and licensing acknowledgement. Optional mocked API contracts
are **not native execution evidence**. Missing native libraries are reported as
skipped, never as passed. The manual `research-integrations` workflow requires its
selected backend to import and rejects a selected test run with zero tests or any
skip, failure or error. It has no schedule, secrets, paid feeds or privileged PR
execution. It is not dispatched automatically by this change.

Two separate implementations of empirical CVaR (CVXPY and skfolio) have a supplied
cross-engine objective-agreement test. It remains unexecuted until both libraries
are installed. An objective match would validate arithmetic, not investment value.

The prior hardening's result register, forward journal and leakage/accounting
tests are included unchanged except for explicit integration routing. Existing
legacy strategies, numeric risk settings and historical results are preserved.

## Interpretation boundaries

- Training dates and hashes establish the supplied slice, not point-in-time
  availability, correct adjusted prices, data ownership or an exchange calendar.
- A bootstrap mean interval is not a future-price prediction interval.
- GARCH conditional variance is not expected return or crash probability.
- CVaR alpha is a historical tail definition, not a calibrated future loss chance.
- HAC inference is asymptotic; it does not remove selection or multiple-testing bias.
- Neither mocked tests, successful imports nor high backtest returns justify promotion.

## Source catalogue

Machine-readable attribution and scope: `config/decision_lab/integrations.json`.
Official APIs and licenses: [third-party inventory](../../THIRD_PARTY_RESEARCH.md).

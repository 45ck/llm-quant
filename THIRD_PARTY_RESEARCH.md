# Third-party research integrations

All adapters and skills in this increment are original repository code. They call
public library APIs; no third-party source tree, dataset, model checkpoint or
copied skill corpus is vendored. Upstream license notices remain with upstream
packages. This file is an attribution and integration inventory, not legal advice.

- **scikit-learn** — BSD-3-Clause. [Source](https://github.com/scikit-learn/scikit-learn) · [API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). TimeSeriesSplit, LedoitWolf and training-only allocations.
- **statsmodels** — BSD-3-Clause. [Source](https://github.com/statsmodels/statsmodels) · [API documentation](https://www.statsmodels.org/stable/stats.html). HAC mean inference and Holm multiple-testing corrections. SciPy supplies the separate exact event-rate interval.
- **skfolio** — BSD-3-Clause. [Source](https://github.com/skfolio/skfolio) · [API documentation](https://skfolio.org/generated/skfolio.optimization.MeanRisk.html). Long-only minimum variance and empirical CVaR allocation.
- **arch** — NCSA. [Source](https://github.com/bashtage/arch) · [API documentation](https://bashtage.github.io/arch/). Stationary-bootstrap mean intervals and one-period GARCH-t variance.
- **Backtesting.py** — AGPL-3.0-or-later; review upstream terms. [Source](https://github.com/kernc/backtesting.py) · [API documentation](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html). Single-instrument next-open execution/commission diagnostic.
- **Microsoft Qlib** — MIT. [Source](https://github.com/microsoft/qlib) · [API documentation](https://qlib.readthedocs.io/en/stable/component/recorder.html). Local experiment recording through QlibRecorder and MLflowExpManager.
- **CVXPY** — Apache-2.0. [Source](https://github.com/cvxpy/cvxpy) · [API documentation](https://www.cvxpy.org/). Explicit empirical CVaR linear program with weight caps.
- **Pandera** — MIT. [Source](https://github.com/unionai-oss/pandera) · [API documentation](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html). Strict OHLC DataFrame schema; no silent coercion or repair.
- **Hypothesis** — MPL-2.0. [Source](https://github.com/HypothesisWorks/hypothesis) · [API documentation](https://hypothesis.readthedocs.io/en/latest/). Generated self-financing accounting properties with deterministic CI settings.

Backtesting.py is AGPL-licensed and excluded from default installation. Running
its adapter requires `--acknowledge-agpl`. This acknowledgement and optional import
boundary do **not** remove license obligations; assess distribution and deployment
terms before using a combined application commercially. Hypothesis retains MPL
terms; it is a test dependency, not part of the trading runtime.

The catalogue is dated 11 November 2026. Record resolved versions, review their
actual distribution licenses and transitive dependencies before redistribution.
No claim of endorsement by the upstream projects is made.

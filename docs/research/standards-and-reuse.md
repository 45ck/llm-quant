# Research standards and library reuse

[Repository home](../../README.md) · [Current policy](../../RESEARCH_POLICY.md)

## Integration update

The earlier hardening below distinguished direct reuse from design references.
The follow-up now adds callable skfolio, arch, Backtesting.py and Qlib adapters,
plus CVXPY, Pandera and Hypothesis QA. See [the current integration guide](integrations.md)
for exact scopes and runtime status. The original design-reference descriptions
below describe the earlier increment, not an assertion that the new optional
adapters were natively executed. No upstream source has been vendored.

## What was reused, and what was only studied in the first increment

| Upstream project | Practice taken | Implementation here |
| --- | --- | --- |
| scikit-learn | Time-ordered validation with a gap; Ledoit-Wolf covariance shrinkage | Direct calls to `TimeSeriesSplit` and `LedoitWolf` in `decision_lab/benchmark.py` |
| SciPy | Tested constrained solvers and exact statistical intervals | `minimize(SLSQP)`, `brentq` and `binomtest(...).proportion_ci(method="exact")` |
| statsmodels | Multiple-testing correction over the complete trial family | Direct `multipletests(method="holm")` call; refuses incomplete declared trial counts |
| skfolio | Explicit portfolio baselines, walk-forward assessment, costs and held-out evaluation | Design reference; skfolio itself is not installed or vendored by this increment |
| arch | Dependence-aware bootstrap choices and multiple-comparison discipline | Review reference for the existing historical stationary-bootstrap lane; no claim of a new arch integration |
| Microsoft Qlib | Separate experiment parameters, metrics, artifacts and run identity | A small hash-checked public register; no Qlib/MLflow dependency or service |
| Backtesting.py | Explicit signal/execution timing and transaction costs | Conceptual reference only; no source copied, engine vendored or AGPL code relicensed |

Use existing numerical libraries rather than copying their implementations.
This is not an endorsement by any upstream project. Code and dataset licensing
are separate; no upstream datasets or proprietary market histories are included.

## Five distinct benchmark methods

`cash` explicitly earns zero, not the current cash rate. `equal_weight` rebalances
on the frozen fold schedule. `equal_weight_buy_and_hold` invests once per evaluation
segment. `inverse_volatility_lw` uses the diagonal of a Ledoit-Wolf covariance
estimate; it is NOT equal-risk-contribution optimization. `minimum_variance_lw`
solves a long-only, fully invested minimum-variance problem on that shrinkage
estimate with SciPy SLSQP. Failed/degenerate fits stop rather than substitute a
more favourable model. None uses expected-return optimization or leverage.

Each candidate is declared before evaluation; the routine does not select a winner.
Every fitted covariance sees only its training slice. `TimeSeriesSplit` uses the
configured gap, at least one full observation and at least the declared label
horizon. The final reserved holdout never enters the development fits. It is
fitted/evaluated separately, with its own initial cash unit. Seeing the output
consumes that holdout for any subsequent human tuning: code cannot prove blindness.

**This is not combinatorial purged cross-validation.** Variable-duration event
labels require label-end-aware purging, not merely a fixed row gap. The current
harness accepts equally spaced close-to-close returns, not OHLC bars; it models
bar-close execution after the gap and does not claim next-open or intrabar fills.
The data owner must check exchange calendars and publication availability.

## Self-financing costs

For pre-trade value V, risky holdings h, target weights w and proportional charge c:

`V_after + c * sum(abs(w * V_after - h)) = V`

SciPy solves the scalar equation, including the effect of fees on target holdings.
An initial full investment with a 1% charge leaves `1 / 1.01` invested, not a
fee-free dollar. A cash leg is not charged again as another asset. Selling risky
holdings is charged. No rebalancing means no extra fee. Tests check conservation,
scale invariance and hand-calculated paths. Terminal positions are marked to
market, not forcibly liquidated. There are no taxes, fixed fees, market impact,
borrowing, liquidity constraints or real execution; those limitations stay visible.

## Statistical checks do not manufacture confidence

The Holm adapter requires the complete declared family of p-values. It cannot
rescue invalid p-values, an incomplete experiment register, data snooping or
selected hypotheses. It does not implement SPA/Reality Check/DSR and does not
interpret an adjusted p-value as a probability of profitability.

The Clopper-Pearson adapter requires an explicit independent constant-rate
Bernoulli assumption. It refuses by default. Overlapping housing windows do not
satisfy that assumption automatically; the historical housing chart has NOT been
silently assigned an invalid confidence interval. Zero events does not imply zero
underlying risk under the stated model.

## Reproduce the offline reference

From the repository root, with Python 3.13.5 in an isolated environment:

```sh
python -m pip install -r requirements/research-py313.lock
PYTHONPATH=src python -m llm_quant.decision_lab.benchmark --reference
PYTHONPATH=src python -m pytest -q tests/test_decision_lab*.py
python scripts/check_research_standards.py --environment
PYTHONPATH=src python -m llm_quant.decision_lab.catalog --check
```

The reference uses a fixed seed and **artificial** returns whose generator is in
source. The fixture, configuration, library versions and outputs are recorded in
`docs/evidence/synthetic-reference-2026-11-11.json` as a compact metrics summary.
The summary records the full-output SHA-256; the command above regenerates the
full NAV, fee and turnover arrays in local storage. It is a test of machinery, not a
new market experiment. Outcomes are identical in repeated runs of the pinned
local environment; other numerical platforms may need a declared tolerance.

For compatible licensed snapshots already obtained through the data contract:

```sh
PYTHONPATH=src python -m llm_quant.decision_lab.benchmark \
  --snapshot FIRST.json --snapshot SECOND.json \
  --plan config/decision_lab/benchmark.example.json
```

Histories must have identical calendars, frequencies, currencies and return bases.
The benchmark command is retrospective, not a current forecast. A quarterly
housing price index is not accepted as a tradable total-return portfolio.
The example plan has daily annualization; never apply it unchanged to monthly data.

## Prospective record, not retrospective invention

`decision_lab.forward` exposes `issue_forecast`, `settle_forecast`, `read_journal`
and `score_journal`. Issue time is assigned by the local wall clock, not a
backdating CLI argument. Forecasts must precede maturity and follow fresh source
observations. Outcomes append separately and must match the exact registered
maturity. Duplicate entries, premature outcomes, changed rows and concurrent
writers are rejected. Missing mature outcomes are counted, not silently dropped.

A SHA-256 chain is **not** an external timestamp or signature. Retain its head
independently to detect truncation; even a self-consistent chain cannot prove the
source prices or the local clock. No actual forward forecasts have been issued
as part of this engineering update. There is no scheduler, brokerage connection,
external timestamp service or automatic promotion.

## Environment and CI scope

`research-py313.lock` captures the exact installed dependency closure used for the
focused reference and tests on Linux/Python 3.13.5. It is version-pinned, not a
wheel-hash-verified or cross-platform lock. `quality-tools.txt` pins reviewed tool
releases separately; they were not available in this runtime for execution.

The full CI workflow now targets `master`, installs the package and dev dependencies
before tests, retains Black/Ruff/mypy/spelling/full pytest, and prints its complete
resolved environment. This repairs the missing-install and wrong-branch defects;
it is **not a claim that the full legacy dependency graph has been locked or run**.
A full resolver lock and hosted execution remain acceptance work. The focused job
retains its original three-minute cap; full quality has a ten-minute cap.
Read-only permissions, commit-pinned actions, cancellation and no new schedules
limit CI exposure. Fork PRs receive no secrets or write credentials.

## Primary references inspected on 11 November 2026

- scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn LedoitWolf: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
- skfolio walk-forward: https://skfolio.org/generated/skfolio.model_selection.WalkForward.html
- skfolio evaluation: https://skfolio.org/user_guide/backtesting_and_evaluation.html
- skfolio license (BSD-3-Clause): https://github.com/skfolio/skfolio/blob/main/LICENSE
- arch time-series bootstrap: https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html
- arch multiple comparisons: https://bashtage.github.io/arch/multiple-comparison/multiple-comparisons.html
- Qlib recorder: https://qlib.readthedocs.io/en/latest/component/recorder.html
- Qlib license (MIT): https://github.com/microsoft/qlib/blob/main/LICENSE
- Backtesting.py timing/cost API: https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html
- Backtesting.py license (AGPL-3.0): https://github.com/kernc/backtesting.py/blob/master/LICENSE.md
- statsmodels multiple tests: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
- SciPy exact proportion interval: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
- GitHub Python CI: https://docs.github.com/en/actions/tutorials/build-and-test-code/python

No upstream implementation source was copied into this change. The direct runtime
libraries keep their own distribution licenses; the new orchestration code remains
under the repository's license. Version-specific dependency behavior is verified
by the focused tests, not assumed from a newer upstream documentation version.

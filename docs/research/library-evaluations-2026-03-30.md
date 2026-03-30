# Library Evaluations — 2026-03-30

Evaluated 6 libraries cloned in `C:/Projects/quant-refs/`. None are pip-installed; all are
available as local source. Evaluations are against current project code and the research
lifecycle gate requirements.

---

## quant-trading
**Verdict**: adopt (reference only — mine for patterns)
**Why**: Contains a working Engle-Granger cointegration + pairs trading backtest
(`Pair trading backtest.py`) with signal generation, z-score entry/exit logic, and
rolling cointegration testing using statsmodels — directly applicable to GLD-SLV
(Family 2) and any Family 8 expansion. The implementation is educational-grade Python
(no packaging), not a drop-in library, but the cointegration pattern
(`EG_method` → `signal_generation`) is cleaner than rolling our own from scratch.
Kalman-filter spread tracking is not present but Johansen test usage is noted in comments.
**Integration point**: Use as a reference blueprint when implementing new pairs strategies
under Family 2. Extract the `EG_method` + rolling-cointegration pattern into
`src/llm_quant/backtest/strategies.py` for any new cointegrated-pair strategy.

---

## mlfinlab
**Verdict**: skip
**Why**: The cloned repo (`C:/Projects/quant-refs/mlfinlab/`) is the community edition —
virtually all key functions are stubs (`pass`). The CPCV class
(`mlfinlab/cross_validation/combinatorial.py`) has 9 `pass` stubs; `deflated_sharpe_ratio`
in `backtest_statistics/statistics.py` is also `pass`; 420 stub occurrences across the
package. The paid Hudson & Thames version has implementations, but that is not what is
available here. Our own `backtest/robustness.py` has working CPCV and DSR implementations
that are correctly derived from Lopez de Prado Chapter 12. There is nothing to replace.
**Integration point**: None. Keep our `backtest/robustness.py` as the authoritative
implementation. Consider referencing `machine-learning-for-trading` (also in quant-refs)
for the actual AFML chapter implementations.

---

## Riskfolio-Lib
**Verdict**: adopt (high priority for portfolio allocation upgrade)
**Why**: The `HCPortfolio` class in `riskfolio/src/HCPortfolio.py` provides HRP,
HERC, HERC2, and NCO optimization via a single `.optimization(model="HRP")` call.
The implementation is fully realized (not stubs), supports drawdown-based risk measures
(MDD, CDaR, EDaR), and accepts pandas DataFrames of asset returns — a thin Polars-to-pandas
adapter at the boundary is all that is needed. This directly addresses the portfolio
Sharpe math problem identified in `docs/governance/alpha-hunting-framework.md`: our current
`scripts/portfolio_optimizer.py` uses simple ATR-based sizing and scipy hierarchical
clustering, while Riskfolio provides correlation-aware recursive bisection with proper
tail risk measures. With 16 strategies across 6 mechanism families, correlation-aware
weighting is the highest-leverage improvement available.
**Integration point**: Replace the `scipy.cluster.hierarchy` + equal-weight logic in
`scripts/portfolio_optimizer.py` with `HCPortfolio.optimization(model="HRP")`. Add a
`pl.DataFrame.to_pandas()` adapter for the returns matrix input.

---

## alphalens-reloaded
**Verdict**: adopt (fills a genuine lifecycle gap)
**Why**: Provides `factor_information_coefficient` (Spearman IC), `mean_information_coefficient`
(IC by time window), IC decay curves, quantile return spreads, and turnover analysis — none
of which exist in our current `src/llm_quant/backtest/` stack. Python 3.12 compatible
(original quantopian/alphalens is not). Input is a pandas MultiIndex DataFrame
(date × asset), requiring a Polars-to-pandas adapter, but the computation is validated
and well-tested. Directly answers the `llm-quant-gidf` architecture question: IC analysis
belongs in `/hypothesis` as an optional validation step before freezing the research-spec,
not in `/robustness` (which is anti-overfitting, not signal predictiveness). The lazy
conversion pattern — convert Polars to pandas only at the alphalens boundary, keep all
upstream processing in Polars — is the correct adapter approach.
**Integration point**: Add an `alphalens_ic_check()` helper to `src/llm_quant/backtest/`
that wraps `factor_information_coefficient` and `mean_information_coefficient`. Call it
as an optional gate in `/hypothesis` command before signal → research-spec advancement.
IC mean > 0.05 and IC t-stat > 2.0 as soft threshold for signal quality.

---

## quantstats
**Verdict**: adopt (wire into /evaluate immediately)
**Why**: `qs.reports.full(returns, benchmark=benchmark_series)` and `qs.reports.html()`
generate complete tearsheets with rolling Sharpe, drawdown periods, monthly heatmaps,
and benchmark-relative metrics in one call. Both functions accept `pd.Series` with a
datetime index — a straightforward `pl.Series.to_pandas()` conversion from our
`portfolio_snapshots` NAV series. The benchmark can be a ticker string (auto-downloads)
or a pre-computed `pd.Series` (use the latter to avoid live data dependency during
evaluation). This replaces the manual metric assembly in `/evaluate` with a standardized,
visually rich output comparable to what institutional PMs use to review strategy
performance against 60/40 SPY/TLT.
**Integration point**: In the `/evaluate` command flow, after `compute_performance()`,
convert the daily returns list to a pandas Series and call
`qs.reports.full(returns_series, benchmark=benchmark_series, periods_per_year=252)`.
Output HTML tearsheet to `reports/tearsheet-{date}.html`.

---

## empyrical
**Verdict**: defer
**Why**: Empyrical provides correctly implemented Sharpe, Sortino, Calmar, max drawdown,
alpha/beta, and annualization with flexible period conventions (`DAILY`/`WEEKLY`/`MONTHLY`).
Annualization uses 252 trading days for daily returns — matching our convention exactly.
However, our `src/llm_quant/trading/performance.py` and `src/llm_quant/backtest/metrics.py`
already implement these metrics correctly in native Polars/numpy, and the DSR computation
in `backtest/metrics.py` relies on unannualized per-period Sharpe that empyrical does not
expose directly. Replacing working code with an external dependency adds risk without
meaningful gain. The correct future trigger: if `quantstats` is adopted (above),
quantstats already depends on empyrical internally for most metric calculations, so
coverage arrives transitively.
**Integration point**: If we ever need rolling alpha/beta vs factor returns
(e.g., Fama-French exposure monitoring), `empyrical.roll_alpha_beta()` is the right
call. Defer until that use case arises.

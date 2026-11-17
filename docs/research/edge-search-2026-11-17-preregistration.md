# Pre-registration: frozen-family forward test and mechanical edge search

[Research index](README.md) · [Research policy](../../RESEARCH_POLICY.md) ·
[Thesis under test](frontier-ai-edge-thesis.md)

**Status when committed: plan only. No outcome on the evaluation windows had been
computed.** Research only; no validated investment edge or live-trading approval.

This file, the code in `src/llm_quant/decision_lab/edge_search.py` and its tests
were committed before the experiment was run on market data. The data snapshot had
been downloaded and checked for completeness (first date, row count, gaps) only.
A Git timestamp is evidence of ordering, not proof that nobody looked earlier.

## Questions

1. **Frozen family (confirmatory).** The 23 lead-lag strategies in
   `scripts/run_paper_batch.py` had their parameters last changed in commit
   `bb35be5` on 24 April 2026. The repository last inspected their paper results
   with data through 28 April. Did they add anything, after that date, beyond
   simply holding the follower asset at the same average exposure?
2. **Search (retrospective with an untouched holdout).** If an agent mechanically
   searches tens of thousands of lead-lag rules on ten years of data and keeps
   the best, does the selected set keep a timing advantage in the following five
   years, and how does its in-sample Sharpe compare with what the same search
   finds when the lead-lag link is destroyed?

## Data

Yahoo Finance daily closes via `yfinance` 1.8.0 with `auto_adjust=True` (split and
distribution adjusted), 2 January 2009 to 13 November 2026, 4,495 rows, 44 US-listed
ETFs, USD. SHA-256 of the local CSV:
`e62c6e509a1cc722a779176f06cc48b0f32ef0ff734e6a1160e3dbb3a2efe36e`. The file stays
in gitignored `research-local/edge-20261117/`; only aggregate statistics are
published. Latest-vintage adjusted prices, not point-in-time. All 44 tickers still
trade, so the universe has survivorship bias. Free personal-use source; no paid feed.

Universe, fixed before any return was computed: SPY QQQ IWM DIA EFA EEM VNQ XLK XLF
XLE XLV XLI XLY XLP XLU XLB SOXX SMH IBB KRE XHB TLT IEF SHY LQD HYG AGG TIP EMB VCIT
MUB GLD SLV USO DBC UUP FXE FXY UNG CPER TQQQ UPRO SOXL TMF.

## Common definitions

`returns[t]` is close(t-1) to close(t). A signal at close(t) is the leader's trailing
`window`-bar return. `held[t] = state[t - delay]` earns `returns[t]`.

**Metric: timing excess.** Strategy return minus a buy-and-hold of the follower at
the strategy's own average exposure over the window, net of the strategy's costs.
A long-only rule on equities earns a positive Sharpe in a rising market by being
long; this metric removes that and measures only whether the rule was long on the
right days. The exposure match uses the evaluation window itself, the usual
market-timing construction.

**Inference.** One-sided test that the mean daily timing excess is positive,
Newey-West standard error with 10 lags (statsmodels). Sharpe is annualised with 252.
Alpha 0.05. Cash earns zero, taxes, borrowing and market impact are omitted, fills
are at the close.

## Experiment 1 — frozen family

The repository holds three different implementations of "the" rule: the frozen
specs add a `lag_days` delay and the backtest class adds hysteresis and a stop,
while the paper logger, which produced the April paper figures, is long only while
the signal is above the entry threshold and flat otherwise. Both readings below are
declared now; no other variant will be run.

| Item | Choice |
| --- | --- |
| Family | All 23 members; none dropped |
| Confirmatory window | Returns dated 29 April 2026 to 13 November 2026 |
| Primary rule | Paper logger as coded: long while signal >= entry, else flat |
| Secondary rule | Hysteresis: long at entry, flat at exit, otherwise unchanged |
| Timing | `delay = 1`: decided and traded at the same close, as coded |
| Cost | 3 bps of NAV per position change, as coded |
| Sensitivity | `delay = 2` and 10 bps per unit of turnover, primary rule |
| **Primary endpoint H1** | Equal-weight average timing excess across the 23 has mean > 0 |
| Secondary | Per-member tests with Holm correction over all 23 |

Declared deviations from the legacy logger: adjusted instead of unadjusted closes,
and cost charged on a position change instead of a regime-label change.

**Power, declared in advance.** The window is about 0.55 years. The standard error
of an annualised Sharpe over that span is roughly 1.35, so only a timing Sharpe
above about 2.2 would be detectable. A null result is therefore "no evidence", not
"disproven". A clearly negative result would count against the family.

**Exploratory replication.** The same two rules over 29 April 2021 to 28 April 2026,
the approximate span of the original backtests, to see how much of the recorded
performance was timing and how much was exposure. Labelled retrospective; this
window was used to choose the parameters.

## Experiment 2 — mechanical search

| Item | Choice |
| --- | --- |
| Rules | Leader x follower over all 44 x 44 pairs (own-asset momentum included) |
| Windows | 5, 10, 20 bars |
| Thresholds | Entry +k sigma, exit -k sigma, k in {0, 0.5, 1}; sigma is the leader signal's standard deviation in the selection window only |
| Direction | Long when the leader rose, or long when it fell |
| Total | 44 x 3 x 3 x 2 = 792 position series x 44 followers = 34,848 configurations |
| Rule | Hysteresis, `delay = 2` (one full bar between signal and trade) |
| Cost | 5 bps per unit of turnover; sensitivity 10 bps |
| Selection window | 3 January 2012 to 31 December 2021 |
| Holdout | 3 January 2022 to 13 November 2026, used once |
| Eligibility | Selection window only: at least 40 position changes, average exposure 10%-90% |
| Ranking | Selection-window timing-excess Sharpe, net of cost |
| **Primary endpoint H2** | Equal-weight portfolio of the top 100 has mean holdout timing excess > 0 |
| Also reported | Top 1 and top 10, share of the top 100 positive in holdout, rank correlation between selection and holdout Sharpe across all eligible configurations, the 20 best named |

**Placebo null.** 200 draws, seed 20261117. In each, every position series is
shifted circularly in time by one random offset of at least 252 bars, the same
offset for all series. Each series keeps its own persistence and turnover and the
followers keep their real returns, but any lead-lag relation is broken. The whole
search is repeated on each draw. Reported: the distribution of the best selection
Sharpe found on placebo data, the placebo distribution of the top-100 holdout
Sharpe, exceedance p-values for the real results, and the expected best placebo
Sharpe as a function of the number of rules tried (10, 100, 1,000, 10,000).

**Candidate criterion.** H2 significant at 5%, top-100 holdout Sharpe at least 0.5,
and mean still positive at 10 bps. Meeting it would make the set a *retrospective
candidate that needs a prospective forward test*, nothing more. Otherwise H2 is
not supported.

## Limits declared before running

- The rule family was chosen knowing the repository's earlier lead-lag backtests,
  which covered roughly 2021-2026, and the author has general knowledge of market
  history. The search itself involves no discretionary choice after seeing data.
- Daily closes cannot represent intraday execution, auction slippage or the
  financing embedded in leveraged ETFs beyond what their prices already show.
- Overlapping configurations are dependent; the placebo exceedance p-value, not a
  per-configuration count, is the multiple-testing control for the search.
- One run. Any change to universe, windows, costs or criteria after seeing results
  is a new trial and will be recorded as such.

## Commands

```sh
PYTHONPATH=src python -m pytest -q tests/test_decision_lab_edge_search.py
PYTHONPATH=src python -m llm_quant.decision_lab.edge_search \
  --prices research-local/edge-20261117/adj_close.csv \
  --snapshot research-local/edge-20261117/snapshot.json \
  --out docs/evidence/edge-search-2026-11-17.json
```

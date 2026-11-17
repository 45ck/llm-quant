# Results: frozen-family forward test and mechanical edge search

[Research index](README.md) · [Pre-registration](edge-search-2026-11-17-preregistration.md) ·
[Thesis](frontier-ai-edge-thesis.md) · [Machine-readable results](../evidence/edge-search-2026-11-17.json)

**Conclusion: no edge found. Both pre-registered hypotheses failed.** Research only;
no validated investment edge or live-trading approval.

The plan, code and tests were committed in `18fe3aa` before this run. The run was
executed once on 17 November 2026 with no change to the plan.

## Experiment 1 — the repository's own 23 lead-lag strategies after their freeze

Window: 29 April to 13 November 2026, 139 trading days, none of it seen when the
parameters were set. Metric: return above a buy-and-hold of the follower at the
strategy's own average exposure ("timing excess"), net of cost.

| Variant (all declared in advance) | Timing excess per year | Sharpe | t | Members positive |
| --- | ---: | ---: | ---: | ---: |
| **Primary**: paper logger as coded | -12.8% | -1.75 | -1.94 | 0 of 23 |
| Hysteresis rule | -14.4% | -1.70 | -2.12 | 0 of 23 |
| One-bar delay, 10 bps cost | -12.1% | -1.92 | -1.87 | 5 of 23 |

H1 (mean timing excess > 0): **not supported**, one-sided p = 0.97. No member
passes the Holm-corrected test. The median strategy had a Sharpe of -0.92 while
simply holding its follower at the same exposure had +0.39.

The point estimate is negative and borderline (two-sided p about 0.05). The window
is short and the 23 members share leaders and followers, so "0 of 23" is far fewer
than 23 independent failures. It is not evidence that the reverse rule works:
that would be a new hypothesis chosen after seeing the data.

**Exploratory replication, 29 April 2021 to 28 April 2026.** On the five years the
parameters were tuned on, the same rules show a timing-excess Sharpe of 1.13
(t = 2.6) and 22 of 23 members positive; the median strategy Sharpe was 1.18
against 0.73 for the matched buy-and-hold. So the recorded backtests reproduce in
sample, part of their headline Sharpe was simply being long a rising asset, and
the timing component reversed sign as soon as the tuning window ended.

## Experiment 2 — what a mechanical search finds, and what survives

34,848 lead-lag rules across 44 ETFs, ranked on 2012-2021, evaluated once on
2022 to 13 November 2026. 32,824 rules were eligible.

**In sample, the search looks like it works.**

| Selection-window timing Sharpe | Real data | Placebo (lead-lag link destroyed) |
| --- | ---: | ---: |
| Best single rule | 0.98 | 1.10 average (0.94-1.28) |
| Portfolio of the top 100 | 3.33 | 2.26 average |

The best real rule is *below* the average best rule found on placebo data
(exceedance p = 0.83). A search of this size produces a Sharpe near 1 from a
single rule, and above 2 from a portfolio of its winners, with no predictive
relation in the data at all.

**How the best placebo Sharpe grows with the number of rules tried:**

| Rules tried | 10 | 100 | 1,000 | 10,000 |
| --- | ---: | ---: | ---: | ---: |
| Expected best placebo Sharpe | 0.33 | 0.63 | 0.85 | 1.02 |

**Out of sample, it mostly disappears.**

| Portfolio | Selection Sharpe | Holdout Sharpe | Holdout p | Holdout Sharpe at 10 bps | Members positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top 1 | 0.98 | 0.28 | 0.28 | 0.22 | 1 of 1 |
| Top 10 | 2.14 | 0.83 | 0.023 | 0.67 | 8 of 10 |
| **Top 100 (primary)** | 3.33 | 0.28 | 0.26 | -0.04 | 46 of 100 |

H2 (top-100 holdout timing excess > 0): **not supported**. The top-100 portfolio
kept under a tenth of its selection Sharpe, earned about 0.9% a year before the
higher cost assumption and nothing after it, and its placebo exceedance p-value is
0.11. Across all eligible rules 38% were positive in the holdout; among the top
100, 46%.

### Two things that could be mistaken for an edge

- **The top 10 held up (Sharpe 0.83, p = 0.023).** This was not the primary
  endpoint. Three portfolio sizes were examined, so a simple correction puts it
  near 0.07, and no placebo distribution was pre-registered for it. Its members
  are mostly credit ETFs leading commodity ETFs and oil moving inversely to long
  Treasuries, which are economically ordinary relations. It is a lead for a
  future pre-registered forward test, not a result.
- **Selection and holdout Sharpe are rank-correlated (0.27).** Much of this is
  mechanical: timing excess is net of cost, and a rule that trades often is
  dragged down by cost in both periods. It does not show that selection skill
  persists.

## Scorecard for the predictions written beforehand

| Prediction | Outcome |
| --- | --- |
| P1. Best selection Sharpe well above 1 | **Wrong in size.** Best single rule was 0.98; the top-100 portfolio reached 3.33. |
| P2. Placebo search finds nearly as high | **Right.** Placebo best averaged 1.10, above the real 0.98. |
| P3. Top 100 loses most of it out of sample | **Right** for the pre-registered top 100 (3.33 to 0.28). The top 10 kept more than predicted. |
| P4. Frozen family shows no timing excess after the freeze | **Right**, and the estimate was negative. |

## Defects found while preparing the test

- "The" lead-lag rule exists in three inconsistent forms: frozen specs carry a
  `lag_days` delay, the backtest class adds hysteresis and a 5% stop, and the
  paper logger is flat in the neutral zone with neither. The April paper figures
  and the recorded backtests therefore measured different strategies.
- The paper logger computes signals and returns on unadjusted closes, so bond
  distributions appear as price drops, and it charges cost on a change of regime
  label, not of position.
- The offline quickstart lock omitted packages the test suite imports; fixed in
  the same change as this report.

## Limits

Latest-vintage adjusted Yahoo closes; survivorship in the ETF list; daily closes
with fills assumed at the close; no financing, tax or impact; one market, one
rule family, one run. The holdout is now used and is no longer fresh for any
related hypothesis. An independent pandas recomputation of three family members
matched the module to four decimals; the placebo and search were not
independently reimplemented.

## Reproduce

Snapshot SHA-256 `e62c6e509a1cc722a779176f06cc48b0f32ef0ff734e6a1160e3dbb3a2efe36e`
(Yahoo Finance via yfinance 1.8.0, retrieved 17 November 2026; a later download
may differ because adjusted prices are revised). Python 3.13.18, numpy 2.3.5,
scipy 1.17.0, statsmodels 0.14.6. Commands are in the pre-registration.

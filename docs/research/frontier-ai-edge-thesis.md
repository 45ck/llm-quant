# Can frontier AI just go and find infinite Sharpe?

[Research index](README.md) · [Research policy](../../RESEARCH_POLICY.md) ·
[Pre-registration](edge-search-2026-11-17-preregistration.md) ·
[Results](edge-search-2026-11-17-results.md) · [Comparable projects](external-projects.md)

**Research only. No validated investment edge or live-trading approval.**

The sections from "The thesis" to "Predictions" were committed before the
experiment ran and are not edited afterwards. Results are appended below them.

## The thesis

A frontier model can read everything, propose hypotheses without limit, and write
and run a backtest in minutes at almost no cost. So, the argument goes, pointing
one at markets should produce strategies with arbitrarily high risk-adjusted
return, and the more it searches the more it finds.

## What is right about it

- **Search became nearly free.** Hypothesis to code to backtest is a loop an agent
  runs unattended. Microsoft's RD-Agent(Q) demonstrates it as engineering, and this
  repository produced about 170 strategy ideas in three weeks the same way.
- **Breadth of reading.** Models turn filings, transcripts and news in any language
  into features. That is a real capability humans cannot match at scale.
- **Diligence.** An agent can audit its own pipeline. This repository's D3 signal
  bug, the mislabelled validation metrics and the three inconsistent lead-lag
  implementations recorded in the pre-registration were all found that way.
- **More independent bets do raise Sharpe.** The fundamental law of active
  management says information ratio is roughly skill times the square root of the
  number of independent bets. If AI really adds independent, skilled bets, Sharpe
  rises.

## Why "infinite Sharpe" does not follow

1. **An infinite Sharpe is an arbitrage.** Riskless profit exists in markets only
   in small sizes, briefly, for whoever has the lowest costs and fastest access.
   The very high Sharpe ratios that do exist, in market making, are small in
   capacity and bought with infrastructure, not with a cleverer idea.
2. **Search raises the measured Sharpe, not the true one.** The best of N
   strategies with no skill has an expected Sharpe of about
   sqrt(2 ln N) / sqrt(years). For 34,848 independent tries on ten years that is
   about 1.45 from pure noise. Cheaper search means a larger N and a more
   impressive false result. The resource AI makes abundant, trials, is exactly
   what makes a backtest worth less. Correlated trials lower the figure; the
   placebo in the pre-registration measures it directly for this search.
3. **Markets adapt.** McLean and Pontiff found published anomalies earned about a
   quarter less out of sample and more than half less after publication. Anything
   every agent can find from the same public data is, for that reason, not an
   edge. An edge is something others cannot or will not do: data, speed, capital,
   mandate, or risk they refuse to carry.
4. **There is one history.** Models are superhuman where rules are fixed and
   examples unlimited. Fifteen years of daily bars is about 3,800 observations of
   a process that changes. No amount of intelligence adds information that is not
   in the data.
5. **Language models remember the test period.** A model trained through 2025
   already knows what happened in 2020, so its backtest is partly recall. Live
   and contamination-controlled tests are where this shows: Nof1's real-money
   Alpha Arena, StockBench, FINSABER and the "Alpha Illusion" reproduction all
   report results near or below buy-and-hold once costs and clean windows apply.
6. **Costs and capacity are not optional.** Spreads, impact, financing and tax
   scale with turnover, and the small anomalies that survive do so because they
   are too small for large capital to bother with.

## Where an agent-era edge could plausibly exist

Not in a better pattern on public daily prices. More plausible, and each with a
reason it might persist:

- **Capacity-limited corners**: small or illiquid instruments where the profit is
  too small for a fund. This repository's prediction-market scans found exactly
  this, with position sizes of a few dollars.
- **Comprehension speed on new documents**: minutes, not microseconds. Real, but
  every well-funded desk has the same models.
- **Process discipline**: not fooling yourself is rarer than a good idea, and an
  agent that enforces pre-registration and holdouts is doing something most
  retail quants do not.
- **Risk premia**: being paid to hold risk others shed. Legitimate, but it is
  compensation, not skill, and it has drawdowns.

There is no public, verifiable example of an edge "only agents have discovered".
If agents reach it by the same public loop, they all reach it.

## Predictions

Written before the run, for the experiment in the pre-registration:

- **P1.** Among 34,848 lead-lag rules the search will find a best selection-window
  timing Sharpe well above 1.
- **P2.** The same search on placebo data, with the lead-lag link destroyed, will
  find a best Sharpe nearly as high.
- **P3.** The top 100 rules will lose most of their selection Sharpe in the
  holdout and will not be distinguishable from zero.
- **P4.** The repository's frozen 23-strategy family will show no detectable
  timing excess after its freeze date.

If P2 and P3 fail, so that the real search clearly beats the placebo and the
advantage holds out of sample, that is evidence for the thesis in this corner of
the market and will be reported as such.

## Results (added 17 November 2026, after the run)

Full tables are in the [results report](edge-search-2026-11-17-results.md).

| Prediction | What happened |
| --- | --- |
| P1. Best selection Sharpe well above 1 | Wrong in size: best single rule 0.98. A portfolio of the top 100 showed 3.33 in sample. |
| P2. Placebo nearly as high | Right: the best rule on placebo data averaged 1.10, higher than the real best. |
| P3. Top 100 fades out of sample | Right: 3.33 fell to 0.28 and was not distinguishable from zero. The top 10 kept 0.83, a secondary and uncorrected observation. |
| P4. Frozen family shows nothing after its freeze | Right: timing excess was negative, 0 of 23 members positive. |

What this run adds to the argument:

- **The search produced its most impressive number from nothing.** An in-sample
  Sharpe of 3.3 for the selected portfolio sits beside 2.3 for the same procedure
  on data with no lead-lag relation in it. An agent reporting the first figure
  alone would be reporting mostly the size of its own search.
- **More trials, higher fake Sharpe, on schedule.** The best placebo rule rose
  from 0.33 with 10 tries to 1.02 with 10,000. That is the logarithmic growth the
  theory predicts, lower in level because the rules overlap and pay costs.
- **A real-looking in-sample effect reversed.** This repository's own family had a
  timing Sharpe above 1 on its tuning window and about -1.7 in the months after.
- **Nothing here refutes the weaker claim** that AI lowers the cost of honest
  research. The whole experiment, including the placebo and the tests, took one
  session. That is the usable part of the thesis: cheap falsification, not cheap
  alpha.

One corner of the market, one rule family and one run cannot show that no edge
exists anywhere. They show that this search, done the way an unattended agent
would do it, did not find one, and that its in-sample output would have looked
like success.

## Sources

- [RD-Agent(Q), arXiv 2505.15155](https://arxiv.org/abs/2505.15155)
- [FINSABER, arXiv 2505.07078](https://arxiv.org/abs/2505.07078)
- [StockBench, arXiv 2510.02209](https://arxiv.org/abs/2510.02209)
- [The Alpha Illusion, arXiv 2605.16895](https://arxiv.org/abs/2605.16895)
- [Wiecki et al., "All that Glitters Is Not Gold"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220)
- [Bailey and López de Prado, "The Deflated Sharpe Ratio"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Harvey, Liu and Zhu, "...and the Cross-Section of Expected Returns"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314)
- [McLean and Pontiff, "Does Academic Research Destroy Stock Return Predictability?"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623)

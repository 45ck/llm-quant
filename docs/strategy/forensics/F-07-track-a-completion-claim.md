# Forensic Finding F-07 — The "Track A Complete, SR=2.205" Headline Doesn't Hold

**Question:** CLAUDE.md (v5.0, 2026-04-22) declares "Track A research complete... 35 strategies validated... portfolio SR=2.205." Verify against primary evidence.

## Sources Quoting the SR Number

| Source | Strategy count | Cluster reps | SR claimed | MaxDD |
|---|---|---|---|---|
| `CLAUDE.md` (current) | 35 | n/a | **2.205** | 6.3% |
| `research-tracks.md` v5.0 changelog | 35, 21 families | n/a | **2.205** | 6.3% |
| `research-tracks.md` body text | 23 | 15 | **2.184** | 5.1% |
| `alpha-hunting-framework.md` v2.0 | 23 (13 passing) | n/a | **2.184** | n/a |
| `alpha-hunting-framework.md` v3.0 | 34 (20 passing) | 18 | **2.205** | 6.3% |
| `portfolio-correlation-analysis-2026-03-31.md` (deployed reality) | **11** | n/a | **~1.35** (estimated, ρ=0.584) | n/a |

**The SR=2.205 number refers to a portfolio of 18 cluster representatives**, not the 35 nominal strategies. The cluster-rep methodology selects one strategy per correlation cluster to maximize effective diversification — an analytical aggregation, not a tradeable book.

## What the Empirical Reality Says

The 2026-03-31 portfolio correlation analysis (not the spec, but the post-hoc measurement) is unambiguous:

> **The current portfolio has a critical concentration problem: 10 of 11 deployed strategies belong to Family 1 (credit lead-lag), producing an estimated average pairwise correlation of ρ=0.584. This yields a corrected portfolio Sharpe of ~1.35 despite 11 strategies each averaging SR ~1.0.**

That's the deployed reality as of Mar 31. Since then, paper trading has added more strategies — but the F-03 finding shows **0 strategies have ≥10 trades**. The deployed reality is a portfolio of credit-lead-lag variants that haven't accumulated enough paper history to compute a meaningful SR.

The **actual portfolio NAV** (per the 2026-04-17 capital deployment diagnosis):
> Portfolio NAV grew only **+1.27%** (from $100,000 to $101,268.55) over 30 days while SPY rallied **+7.42%**.

A SR=2.184 portfolio compounding for 30 days at 22% annualized vol target should have produced roughly **+1.7% to +2.2%** mean return with low variance. The actual +1.27% (with dispersion across days suggesting much higher noise) is **directionally consistent with a SR ~0.5-0.8 portfolio**, not SR ~2.2.

## The D3 Smoking Gun

`alpha-hunting-framework.md` v3.0 changelog cites:
> Track D review: D3 TQQQ/TMF (Sharpe=2.21) highest individual strategy.

CLAUDE.md (current) says:
> D3 TQQQ/TMF RETIRED (independent replication shows Sharpe=-1.08)

**The single highest-Sharpe contributor to the validated portfolio was a signal bug.** The retire-on-replication discipline caught it, which is good. But the SR=2.205 number was computed *with D3 included*, and has not been re-computed since D3 was retired. The cluster-rep methodology might absorb the loss (D3 was probably grouped into a Track D leveraged cluster with TLT-TQQQ), but the headline number was never re-published with the corrected basis.

## What "Track A Research Complete" Actually Means

"Complete" in the v5.0 changelog reads as a strategic decision, not a measurement claim:
> moved to backlog per solo-trader CAGR priority

Translation: attention shifted to Track D, so Track A research stopped being added to. Not because the SR target was achieved with measured paper PnL, but because the **research throughput** had saturated (correlation analysis: "commodity/macro mechanism space saturated").

This is a valid research-management decision. It's NOT the same as "the Track A book is validated and ready to deploy capital against," which is how CLAUDE.md presents it.

## Three Interlocking Issues

1. **Backtest portfolio SR vs deployed portfolio SR are different things.** The platform consistently reports backtest SR (~2.2) as if it were the deployed PnL. The deployed portfolio at +1.27% / 30 days is more consistent with SR ~0.5-0.8.
2. **The cluster-rep methodology is being used as both an analytical tool and a deployment proxy.** "18 cluster reps with SR=2.205" is a way to estimate the diversification ceiling — not a tradeable portfolio (you can't actually trade the cluster centroid; you trade individual strategies which have higher correlations than the centroids suggest).
3. **The retirement of D3 has not been propagated through the headline metric.** Anyone reading "Track A SR=2.205" in CLAUDE.md today is reading a number that includes a signal bug.

## Implications

- **CLAUDE.md's identity claim ("$100k portfolio... SR=2.205") is structurally misleading.** The implied capital efficiency does not match observed paper performance.
- The Option A (research lab) framing actually fits Track A's status better — the *research* on Track A is complete in the sense of "we've explored the mechanism space"; the *deployment* is nowhere near complete.
- Any decision to allocate real capital to Track A based on the SR=2.205 number should require **first re-computing the metric with D3 removed** and **second verifying the metric against actual paper PnL**.

## Recommended Actions

1. Re-compute the cluster-rep portfolio SR with D3 removed. Publish as `docs/research/portfolio-sr-recomputation-2026-04-24.md`.
2. Add a measured-paper-SR column to the strategy inventory once any strategy clears the 50-trade gate (which under F-03 will require lowering the gate first).
3. Update CLAUDE.md to distinguish between "backtest SR" and "measured paper SR" anywhere a number appears.

## Confidence

**High** that the headline number is composed differently than CLAUDE.md presents it. **Medium** on the corrected SR — the cluster-rep methodology requires the underlying daily returns matrix, which is not directly accessible from the artifacts I read. The deployed-reality estimate of SR ~0.5-0.8 is back-of-envelope from total return / 30 days; an actual computation could go either way within a wide error band.

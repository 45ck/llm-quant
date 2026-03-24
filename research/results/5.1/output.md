# Literature survey for systematic macro trading: regime detection, momentum, and inflation allocation

**Three research briefs examined against the academic and practitioner literature reveal a mixed verdict: the core hypotheses draw on well-established foundations but several specific claims—particularly around HMM Sharpe improvement magnitudes, short-term crash prediction lead times, and commodity inflation-hedging reliability—face significant contradictory evidence.** The strongest empirical support exists for time-series momentum diversified across lookback windows and asset classes, the Bridgewater-style 2x2 growth/inflation allocation framework, and credit spreads as leading stress indicators. The weakest support exists for 4-state HMMs outperforming simpler models in portfolio performance (despite statistical fit), crowding indicators predicting crashes 2-4 weeks ahead, and passive commodities as reliable inflation hedges.

---

## Brief 5.1: Four-factor HMM regime detection finds academic grounding but practitioner skepticism

### The foundational regime-switching literature validates the concept but debates the details

Hamilton's (1989) seminal Markov-switching model in *Econometrica* demonstrated that GDP growth follows distinct AR processes in recessions versus expansions, with turning points closely matching NBER dates. This paper—now with over 20,000 citations—established the theoretical foundation for all subsequent regime-switching work. Ang & Bekaert (2002, *JBES*) extended this to interest rates, finding regime-switching models **forecast better out-of-sample** than single-regime models including affine multifactor models, and that incorporating international short-rate and term spread information improves regime classification. Ang & Timmermann (2012, NBER WP 17182) provide a comprehensive survey documenting that regime switching captures skewness, fat tails, downside risk, and time-varying correlations—and that **the cost of ignoring regimes is considerable** even after accounting for parameter uncertainty (Tu, 2010).

Guidolin & Timmermann (2007, *J. Economic Dynamics and Control*) provide the strongest academic support for **four regime states**. Using monthly stock and bond returns from 1954-1999, they identified crash, slow growth, bull, and recovery states as necessary to capture the joint return distribution. Optimal allocations varied considerably across these four states, and welfare costs from ignoring regime switching were substantial even out-of-sample. However, this finding is contested in practice.

### Credit spreads do lead equity volatility, but the relationship is unstable

The hypothesis that IG credit spreads lead equity volatility by 2-4 weeks during stress episodes finds meaningful support. Gilchrist & Zakrajsek (2012, *American Economic Review*) constructed the GZ credit spread index and decomposed it into an expected default component and an **excess bond premium (EBP)**. The EBP—reflecting reduction in risk-bearing capacity of the financial sector—has "considerable predictive power" for future economic activity, significantly exceeding that of the standard Baa-Aaa spread. Favara et al. (2016) showed the EBP predicts NBER-dated recessions **12 months ahead**.

Practitioner observations support shorter-term lead-lag dynamics. Wellington Management (2022) documented that during early 2022, credit spreads widened considerably while VIX remained "anomalously stable throughout the 20%+ drawdown." Market data providers note "US Credit Spread tends to lead VIX by a few weeks," and CCC spreads tracked against VIX 4-week averages show credit leading. The general pattern—"quiet VIX with creeping HY widening is a classic early warning"—aligns with the Merton framework where credit markets sit closer to refinancing risk. **However, this relationship broke down for approximately two years post-COVID 2020**, when spreads normalized due to Fed intervention while VIX stayed elevated. The lead-lag is regime-dependent, not constant.

### Four states may fit better statistically but two states often win in portfolio performance

The optimal number of HMM states is the most contentious parameter question. BIC analysis by Nguyen & Nguyen (2021) on S&P 500 monthly data across 120 rolling calibrations found the **four-state HMM was optimal** across AIC, BIC, HQC, and CAIC criteria. However, Fons et al. (2021) tested 200 random 5-asset combinations and found that while BIC was lowest for 4 states, **portfolio performance was significantly better with 2 states**. This creates a fundamental tension: statistical fit favors complexity while investment performance favors parsimony.

Practitioners overwhelmingly favor two states. Kritzman, Page & Turkington (2012, *Financial Analysts Journal*) used a 2-state HMM ("normal" vs. "event") applied to turbulence, inflation, and growth. First Sentier Investors, Nystrup et al. (2015-2019), and Shu & Mulvey (2024) all use 2-state models, citing Occam's razor. The practical recommendation from the literature: **2 states for risk-on/risk-off, 3 for bull/sideways/bear, 4 for comprehensive regime modeling**, with interpretability degrading rapidly beyond 4.

### The DXY as a distinct regime factor represents a genuine literature gap

No published paper was found that jointly models regime detection using a multivariate HMM with credit spreads, VIX, yield curve, **and** DXY momentum as observation variables. BIS (2024) shows the dollar is a "dominant driver" of EME capital flows with information content distinct from VIX (correlation between FCI and dollar is 0.45 vs. FCI and VIX at 0.62). Engel & Hamilton (1990) documented Markov-switching behavior in exchange rates ("long swings in the dollar"). Hansen (2024, *International Journal of Forecasting*) showed VIX and yield curve spread co-move in counterclockwise cycles aligned with the business cycle, significantly outperforming the yield curve alone for recession prediction. But the specific hypothesis that DXY 3-month momentum captures a **distinct** macro channel alongside VIX, yield curve, and credit spreads remains **untested in formal academic work**—this is an opportunity for original contribution.

### Weekly frequency, Gaussian emissions, and the HMM-vs-heuristic Sharpe gap

**Weekly observation frequency** appears to be the practitioner sweet spot. Calvet & Fisher (2004) found their multifractal model provided "considerable gains in accuracy at horizons of 10 to 50 days," and Nystrup et al. (2015, 2017) cautioned that daily data causes excessive regime flipping with costly turnover. Academic literature defaults to monthly data (Guidolin & Timmermann 2007, Ang & Bekaert 2002), while Hamilton (1989) used quarterly GDP. Weekly reduces noise versus daily while maintaining adequate responsiveness, with Nystrup et al. (2019) finding an **optimal lookback of ~130 days for 2-regime models**.

On emission distributions, Student-t emissions clearly outperform Gaussian statistically (fHMM package comparisons on DAX data show AIC of -35,270 for 3-state Student-t vs. -34,795 for 2-state Gaussian). Stock returns exhibit kurtosis consistent with Student-t distributions with **2-5 degrees of freedom** (Welch, 2024, *Financial Analysts Journal*). However, Gaussian HMMs remain standard in practice because the regime-switching structure itself generates fat tails through the mixture of state-dependent distributions.

**The claimed 0.15-0.30 Sharpe improvement of HMM over heuristics lacks direct empirical support.** Kritzman et al. (2012) claimed HMM outperforms "simple data partitions based on thresholds" without specific Sharpe differentials. Shu & Mulvey (2024) showed that Statistical Jump Models (JMs) **outperform HMMs** due to better state persistence—HMMs suffer from frequent switching, label instability, and out-of-sample degradation (Nystrup et al. 2020a, 2020b). Nystrup (2019) found multivariate HMM fitted to all indices resulted in "low persistence and frequent switches, leading to excessive portfolio turnover and poor results." The improvement from any regime model versus static allocation is meaningful (Sharpe improvements of 0.2-0.5 in various studies), but **the marginal HMM advantage over well-designed heuristics is modest and context-dependent**.

### Key parameters and contradictions for Brief 5.1

| Parameter | Literature consensus | Key source |
|-----------|---------------------|------------|
| Optimal states | 2 (practice) vs. 4 (statistical fit) | Fons et al. 2021; Guidolin & Timmermann 2007 |
| Observation frequency | Weekly (practice); monthly (academic) | Nystrup et al. 2015-2019 |
| Lookback window | ~130 days for 2 states | Nystrup et al. 2019 |
| Credit lead on VIX | 2-4 weeks during stress (unstable) | Wellington 2022; Gilchrist & Zakrajsek 2012 |
| HMM vs. heuristic Sharpe gain | Not documented at 0.15-0.30 | No direct evidence found |
| Best alternative to HMM | Statistical Jump Models | Shu, Yu & Mulvey 2024 |
| Emission distribution | Student-t statistically superior; Gaussian practical standard | Welch 2024 |

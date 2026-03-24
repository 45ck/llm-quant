# Literature survey for systematic macro trading: regime detection, momentum, and inflation allocation

**Three research briefs examined against the academic and practitioner literature reveal a mixed verdict: the core hypotheses draw on well-established foundations but several specific claims—particularly around HMM Sharpe improvement magnitudes, short-term crash prediction lead times, and commodity inflation-hedging reliability—face significant contradictory evidence.** The strongest empirical support exists for time-series momentum diversified across lookback windows and asset classes, the Bridgewater-style 2x2 growth/inflation allocation framework, and credit spreads as leading stress indicators. The weakest support exists for 4-state HMMs outperforming simpler models in portfolio performance (despite statistical fit), crowding indicators predicting crashes 2-4 weeks ahead, and passive commodities as reliable inflation hedges.

---

## Brief 5.2: Time-series momentum is robust but crash prediction at 2-4 week horizons remains unproven

### TSMOM generates strong diversified returns, but individual-asset evidence is weaker than pooled

Moskowitz, Ooi & Pedersen (2012, *JFE*) tested 58 liquid futures/forwards across commodities, currencies, equity indices, and government bonds from 1985-2009. All 58 contracts exhibited positive TSMOM returns, with 52 significant at 5%. The diversified TSMOM portfolio generated alpha of **1.26% per month** (t-stat ~ 7.55) with an estimated Sharpe ratio of **~1.31**. Each position was scaled to 40% annualized volatility using exponentially weighted realized vol (center of mass = 60 days). Return persistence held for 1-12 months with partial reversal at longer horizons.

Hurst, Ooi & Pedersen (2013, *JOIM*) extended this to a diversified portfolio across all assets and three lookback horizons (1, 3, 12 months), achieving a gross Sharpe ratio of **1.8** at 10% vol target. Their 2017 *JPM* paper ("A Century of Evidence on Trend-Following Investing") pushed the sample back to 1880 across 67 markets, finding an average net Sharpe of **~0.4 per individual market** but positive returns in every decade since 1880.

However, Huang, Li, Wang & Zhou (2020, *JFE*) deliver a major challenge: asset-by-asset regressions reveal **little evidence of time-series momentum** (47 of 55 assets have t-stat < 1.65). The pooled regression t-statistic falls below bootstrap critical values. They conclude TSMOM strategy profitability is "virtually the same" as a strategy based on the historical sample mean, which requires no predictability. Kim, Tse & Wald (2016, *J. Financial Markets*) reinforce this, finding TSMOM profits are **driven by volatility scaling** rather than time-series predictability per se—without vol scaling, performance is no better than buy-and-hold.

### Asset-class Sharpe ratios and the critical role of multi-lookback diversification

The brief's hypothesized Sharpe ratios by asset class (equities 0.4-0.6, fixed income 0.3-0.5, commodities 0.5-0.8, crypto 0.6-1.0, forex 0.3-0.5, combined 0.6-0.9) are broadly consistent with the literature at the diversified level but likely optimistic for individual asset classes.

From Hurst et al. (2013), average individual instrument Sharpe ratios are modest: **0.29 (1-month), 0.36 (3-month), 0.38 (12-month)**. The 12-month lookback is the strongest single signal, but combining all three horizons is dramatically beneficial due to low cross-correlations. Baltas & Kosowski (2013/2015) confirmed this with 71 contracts from 1974-2002, achieving **Sharpe ratios above 1.20** using comprehensive multi-frequency TSMOM portfolios. The key insight: diversification across both assets and lookback windows drives portfolio-level Sharpe from ~0.3-0.4 to above 1.0.

For cryptocurrency, evidence is striking but unreliable due to short samples. Huang, Sangiorgi & Urquhart (SSRN, 2024) report a volume-weighted crypto TSMOM Sharpe of **2.17**. An Auckland University study (2023) found **1.51** with 28-day lookback/5-day holding. Risk-managed crypto TSMOM (applying Barroso & Santa-Clara scaling) improves Sharpe from **1.12 to 1.42** (ScienceDirect, 2025). Liu & Tsyvinski (NBER WP 24877) found Bitcoin weekly returns in the top momentum quintile averaged **11.22% per week**. These results likely overstate achievable returns due to survivorship bias, illiquidity, and the extraordinary bull market sample.

### Volatility scaling is essential and may be the primary driver of returns

Barroso & Santa-Clara (2015, *JFE*) demonstrated that volatility targeting **nearly doubles the Sharpe ratio** for equity momentum: from **0.53 to 0.97**. Their method scales by inverse of trailing **126-day (6-month) realized variance**, targeting **12% constant volatility**. This reduced excess kurtosis from 18.24 to 2.68, improved skewness from -2.47 to -0.42, and cut the worst monthly return from -78.96% to -28.40%. Daniel & Moskowitz (2016, *JFE*) improved further by forecasting both conditional mean and variance, scaling proportional to the conditional Sharpe ratio, approximately **doubling alpha and Sharpe** versus static momentum.

The combined lookback hypothesis is well-supported. Hurst et al. (2013) showed 1-month, 3-month, and 12-month strategies have **low cross-correlation**, meaning they capture distinct return continuation phenomena. Levine & Pedersen (2016, *FAJ*) proved that TSMOM and moving-average crossovers are mathematically equivalent in their general forms—both reduce to linear filters—so the specific signal construction matters less than the frequency diversification.

### Momentum crashes cluster at regime transitions but short-term prediction remains elusive

Daniel & Moskowitz (2016) documented that momentum crashes are partly forecastable and occur in "panic states"—following market declines when volatility is high, contemporaneous with market rebounds. The mechanism: past losers acquire call-option-like characteristics via the Merton (1974) framework, surging when markets rebound. **14 of the 15 worst momentum returns** occurred when past 2-year market return was negative and contemporaneous market return was positive. The worst crashes were devastating: **-91.59%** (July-August 1932), **-73.42%** (March-May 2009).

Crashes clearly cluster at bear-to-bull regime transitions. Of 1,107 months from 1930, 157 (~14%) were classified as "market rebounds"—these contain 8 of the 10 largest momentum losses. This strongly supports the hypothesis that regime detection can improve momentum risk management.

However, **no robust academic evidence supports crowding indicators predicting crashes specifically 2-4 weeks ahead**. The most rigorous crowding research operates on longer horizons:

- **Lou & Polk (2021, *RFS*)**: Comomentum (average pairwise correlation among momentum stocks) predicts returns over **12-24 month horizons**. High comomentum predicts negative long-run momentum returns, with a spread exceeding **25 percentage points** over Years 1-2 (t-stat = -3.35).
- **Baltas (2019, *FAJ*)**: Pairwise correlation-based crowding predicts momentum underperformance over **6 months to 1 year** (t-statistics of -5.30 at 6 months, -9.12 at 1 year), but the first-month difference is statistically insignificant.
- **MSCI Integrated Factor Crowding Model**: Uses five metrics (valuation spread, short interest, pairwise correlation, factor volatility, factor reversal) with a composite z-score. Factors scoring above **+1.0** experience significantly higher drawdowns, but timing precision is monthly to quarterly, not weekly.
- **Return dispersion**: Stivers & Sun (2010, *JFQA*) found cross-sectional return dispersion is negatively related to subsequent momentum premium. Liu et al. (2025, SSRN) showed IQR of return dispersion forecasts momentum crashes across 52 international markets, but at **monthly or longer** horizons.

The shortest-term signal found is Verdad's high-yield spread threshold of **>700 bps** as a monthly-frequency exit signal for momentum. Barroso & Edelen (2022, *JFQA*), using 13F institutional holdings, cast "theoretical and empirical doubt on crowding as a stand-alone source of tail risk." The composite crowding index described in the brief—combining return dispersion collapse (z < -1.5), volume anomalies (z > 2.0), and pairwise correlation surges—would require significant adaptation beyond current published research to achieve 2-4 week prediction.

### Post-publication alpha decay is a real concern

McLean & Pontiff (2016, *JF*) found returns across 97 anomaly predictors were **26% lower out-of-sample and 58% lower post-publication**, with the largest degradation occurring 3-4 years after publication. Satchell & Grant (2020) and Huang et al. (2020) document declining TSMOM profitability since 2008. Man Group (2025) addressed this directly by examining whether trend-following AUM has become too large a share of futures markets, finding **no evidence of capacity constraints**—but acknowledged drawdown periods are inherent to the strategy.

### Key parameters and contradictions for Brief 5.2

| Parameter | Literature value | Key source |
|-----------|-----------------|------------|
| Diversified TSMOM Sharpe (gross) | 1.3-1.8 | Moskowitz et al. 2012; Hurst et al. 2013 |
| Individual instrument Sharpe | 0.29-0.38 | Hurst et al. 2013 |
| Vol-managed momentum Sharpe | 0.97 (equity); higher for multi-asset | Barroso & Santa-Clara 2015 |
| Crypto TSMOM Sharpe | 1.12-2.17 (unreliable short samples) | Various 2023-2025 |
| Signal half-life | ~10-12 months before reversal | Moskowitz et al. 2012 |
| Vol scaling window | 126 days (~6 months) | Barroso & Santa-Clara 2015 |
| Vol target | 12% (equity) or 40% per instrument | Multiple |
| Crash clustering | Bear-to-bull transitions (14/15 worst in rebounds) | Daniel & Moskowitz 2016 |
| Crowding prediction horizon | 6-24 months (NOT 2-4 weeks) | Lou & Polk 2021; Baltas 2019 |
| MSCI crowding z-score threshold | >1.0 (elevated risk) | MSCI |

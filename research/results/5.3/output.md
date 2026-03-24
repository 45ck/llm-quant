# Literature survey for systematic macro trading: regime detection, momentum, and inflation allocation

**Three research briefs examined against the academic and practitioner literature reveal a mixed verdict: the core hypotheses draw on well-established foundations but several specific claims—particularly around HMM Sharpe improvement magnitudes, short-term crash prediction lead times, and commodity inflation-hedging reliability—face significant contradictory evidence.** The strongest empirical support exists for time-series momentum diversified across lookback windows and asset classes, the Bridgewater-style 2x2 growth/inflation allocation framework, and credit spreads as leading stress indicators. The weakest support exists for 4-state HMMs outperforming simpler models in portfolio performance (despite statistical fit), crowding indicators predicting crashes 2-4 weeks ahead, and passive commodities as reliable inflation hedges.

---

## Brief 5.3: The 2x2 inflation-risk allocation matrix has strong foundations but stagflation detection remains the Achilles' heel

### Bridgewater's four-environment framework is the intellectual bedrock

The All-Weather framework, created by Ray Dalio, Bob Prince, and Greg Jensen and launched in 1996, defines four economic environments based on **changes relative to market expectations** in growth and inflation. The design principle is elegant: equal risk allocated to each quadrant (not equal capital), passive and non-predictive, with environmental exposures canceling each other out to leave risk premium. Bridgewater claims a **0.6 return-to-risk ratio** versus 0.4 for 60/40. The ALLW ETF (launched March 2025, SPDR/Bridgewater partnership) runs at ~1.88x leverage with allocations of nominal government bonds 73.1%, equities 43.6%, commodities 34%, and inflation-linked bonds 36.5%.

FTSE Russell's "Balanced Macro" framework (Schroeder, Allegrucci, Vilaboa, June 2025) provides the most rigorous recent empirical test. Over the post-Bretton Woods period (~1971-2025), the Balanced Macro portfolio achieved **Sharpe of 0.58** versus 0.51 for 60/40 and 0.45 for US equities, with maximum drawdown of just **-16.7%** versus -29.8% (60/40) and -51.1% (equities). Their critical methodological innovation: risk parity **within** quadrants (where correlations are stable) but **no correlation assumptions across** quadrants (because cross-quadrant correlations range from -40% to +40%). ReSolve Asset Management's structurally diversified portfolio achieved ~2x the 60/40 Sharpe (8.4% return at 6.8% vol versus 6.4% at 10.7% vol).

### Empirical evidence validates four economically distinct quadrants

Each cell of the 2x2 matrix does produce distinct optimal allocations:

- **Reflationary Boom** (rising growth + rising inflation): Commodities, energy, industrial metals, EM equities thrive. The energy sector's correlation to inflation is **0.6** (S&P Dow Jones Indices), triple the broad equity correlation of 0.2.
- **Disinflationary Boom** (rising growth + falling inflation): Developed market equities, corporate credit, and risk assets outperform. This is the "Goldilocks" quadrant where most time has been spent since 1980 (Fidenza Macro).
- **Stagflation** (falling growth + rising inflation): Gold dominated with **+9.2% real annual returns** during 1973-82 and **19.16% annualized** across all stagflation periods since 1973. TIPS (back-tested) returned +3.0% real. Energy stocks were the best-performing sector. Man Group's research (2020) finds **all traditional long-only asset classes deliver negative real returns on average** in inflationary regimes since 1926.
- **Deflationary Bust** (falling growth + falling inflation): Nominal government bonds rally; everything else sells off. The 2008 GFC is the canonical example.

AQR's work (Ilmanen, Maloney, Ross, *JPM* 2014) adds a crucial nuance: dynamic systematic strategies (value, momentum, carry) have "meaningfully less macro exposure" than asset classes, making them more resilient across environments. Diversified portfolios combining asset class premia and style premia show the **highest Sharpe ratios across all environments**.

### TIPS breakevens, CPI surprises, and commodity momentum each contribute partial but imperfect inflation signals

**TIPS breakevens** are reasonably accurate over longer horizons. BLS analysis (Church, 2019) comparing Treasury breakeven inflation with CPI-U from 2003-2018 found average deviations **never exceeded 80 bps** for any maturity, and **never exceeded 55 bps** for horizons >=2 years. But breakevens are distorted by two offsetting premia: an inflation risk premium (biases up) and a liquidity premium of **~100 bps** that biases down (D'Amico, Kim & Wei, 2010, Fed Working Paper). The Fed explicitly warns breakevens "should not be interpreted as estimates of inflation expectations." Detection lag is **2-4 months** during regime shifts, as seen in the 2020-2021 inflation surge. The 5y5y forward breakeven provides a more stable long-term signal.

**CPI surprises** provide the highest-frequency confirmation. Recent research on S&P 500 reactions (2021-2025) found a striking asymmetry: positive surprises (CPI below expectations) generated CARs exceeding **+1%** that were highly significant, while negative surprises produced similar magnitude effects **without statistical significance**. LSEG research (2009-2022) found core CPI had the biggest impact on Nasdaq 100 E-Minis (quintile spread of **-0.70**), larger than employment data. Effects are concentrated in 5-minute to 1-day windows but cascade through monetary policy expectations for weeks.

**Commodity momentum** as an inflation signal draws on Gorton & Rouwenhorst (2006, *NBER*), who found commodity futures are positively correlated with inflation (**~0.33**), unexpected inflation, and changes in expected inflation, while negatively correlated with equity and bond returns. Erb & Harvey (2006) paint a more skeptical picture: average annualized excess return of individual commodity futures was approximately **zero**, with portfolio-level returns attributable to rebalancing effects and momentum/term-structure tilts. Their regression of GSCI excess return on inflation changes yielded R^2 = **0.43**. Miffre & Rallis (2007) identified 13 profitable commodity momentum strategies averaging **9.38% per year**. The reconciliation: commodities hedge best against **unexpected supply-shock-driven inflation** in a tactical framework, not as passive allocations.

### Stagflation is indeed the hardest quadrant—and the most valuable for capital preservation

The literature strongly validates this hypothesis. Cliffwater/CAIA (Nesbitt, 2022) states: "Predicting [stagflation's] onset is **nearly impossible** but perhaps investment adaptation once the regime materializes can be worthwhile." Fidelity notes that "long periods of stagflation have been uncommon means that history can't offer foolproof guidance." Meketa (2025) finds "the vast majority of asset classes produce flat or negative returns" in stagflation.

The asset returns data from the 1970s stagflation are stark. During 1973-82, with CPI averaging 8.7% and real GDP growth of just 1.5%: stocks returned ~-2% real annually, bonds ~-3% real, cash ~0% real, while gold earned +9.2% real, REITs +4.5% real, and TIPS (back-tested) +3.0% real. Cliffwater's inflation betas reveal the magnitude: gold at **3.37**, commodities at **2.86**, EM at **4.73**, real estate at **2.76**, versus US stocks at **-0.77** and private equity at **-0.91**.

2022 provided a powerful out-of-sample validation of the 2x2 framework. Stock-bond positive correlation destroyed 60/40 (exactly as the framework predicts for rising-inflation regimes), while commodities were the only major hedge. CAIA (January 2024) noted most risk parity products "significantly underperformed" 60/40's -16.1% in 2022—"algorithms had nowhere to hide amid unprecedented inflation." The wide dispersion between funds (PanAgora -0.6% to Man AHL +8.6% through November 2023) suggests implementation matters as much as framework design.

### Gold's inflation-hedging ability is regime-dependent, not universal

Gold's role is more nuanced than commonly assumed. A threshold study (ScienceDirect, 2022) found gold exhibits significant inflation responses only when monthly US inflation **exceeds 0.55%** (~6.6% annualized)—below this threshold, gold is **not** an effective hedge. CFA Institute (2024) found changes in PCE inflation are "not meaningfully correlated" with gold spot prices (correlation confidence interval: -0.004 to 0.162). The Chicago Fed (2021) identified real interest rates as the primary gold price driver post-2000, with a negative relationship confirmed at annual, quarterly, and daily frequencies. A Markov-switching VAR (Valadkhani & O'Mahony, 2024) found gold provides a **4-5 month window** of inflation hedging following onset of intensified inflationary pressures. Over very long periods (decades), gold rises at the general rate of inflation, but with enormous short/medium-term deviations.

### Adding inflation regime improves risk-adjusted returns by a meaningful but modest margin

The evidence supports the claimed 0.1-0.2 Sharpe improvement, though the magnitude depends heavily on methodology and sample period. FTSE Russell's Balanced Macro improved Sharpe from **0.51 (60/40) to 0.58**—a gain of 0.07 that is somewhat below the hypothesized range but directionally consistent. AQR's risk parity research shows Sharpe of **0.75 versus 0.52** for 60/40, a larger gap of 0.23, though this reflects the full risk parity benefit (leverage + diversification + inflation balance), not just the inflation dimension.

ReSolve Asset Management's comparison is most direct: structurally diversified policy Sharpe of ~1.2 versus ~0.6 for 60/40, though this also reflects leverage. Dynamic risk parity (with rolling 12-month covariance updates) achieved Sharpe of **1.418** versus 1.368 for static risk parity, confirming modest but consistent gains from dynamic adjustment (Wattanasin et al., *JRFM* 2026). The key caveat from ReSolve: inverse-variance methods exhibit "large systematic bias toward bonds" that performed well during the 40-year disinflationary period but implies "vulnerability to inflationary regimes." This bias is precisely what adding the inflation dimension corrects.

### Key parameters and contradictions for Brief 5.3

| Parameter | Literature value | Key source |
|-----------|-----------------|------------|
| Balanced Macro Sharpe | 0.58 vs. 0.51 (60/40) | FTSE Russell 2025 |
| Risk Parity Sharpe | 0.75 vs. 0.52 (60/40) | AQR (Asness) |
| Balanced Macro max drawdown | -16.7% vs. -29.8% (60/40) | FTSE Russell 2025 |
| TIPS breakeven accuracy | Within 55-80 bps for >=2yr horizons | BLS Church 2019 |
| TIPS liquidity premium | ~100 bps | D'Amico, Kim & Wei 2010 |
| Gold real return in stagflation | +9.2%/yr (1973-82) | Cliffwater/CAIA 2022 |
| Gold inflation threshold | >0.55%/month (~6.6% ann.) | ScienceDirect 2022 |
| Commodity-inflation correlation | +0.33 (1959-2004) | Gorton & Rouwenhorst 2006 |
| Energy-inflation correlation | +0.6 | S&P Dow Jones Indices |
| Gold inflation beta | 3.37 | Cliffwater 2022 |
| Stagflation frequency since 1973 | ~11% of time | WealthGen Advisors 2025 |

---

## Cross-cutting themes: what the literature tells us about these three briefs together

Several findings recur across all three briefs and merit attention as a system design consideration.

**Simplicity often outperforms complexity in portfolio performance.** While 4-state HMMs, composite crowding indices, and multi-factor inflation models all show superior in-sample statistical fit, simpler alternatives frequently dominate out-of-sample. Two-state regime models outperform four-state models in portfolio returns (Fons et al. 2021). Vol-scaled momentum with simple inverse-variance weighting nearly doubles Sharpe without requiring crash prediction (Barroso & Santa-Clara 2015). Static risk parity captures most of the diversification benefit without dynamic adjustment (Sharpe of 1.368 vs. 1.418 for dynamic). The **overfitting risk** escalates with model complexity—Bailey et al. (2016) show that with only 5 years of daily data, no more than 45 strategy variations should be tested before a Sharpe >=1.0 appears by chance.

**Volatility scaling is the most consistently valuable technique across all three domains.** It is integral to TSMOM (Kim et al. 2016 argue it *is* the entire source of profits), essential for managing momentum crash risk (Barroso & Santa-Clara), central to risk parity construction, and implicitly present in the 2x2 allocation matrix through equal-risk-not-equal-capital design. Any systematic macro system should treat vol scaling as foundational infrastructure rather than an optional enhancement.

**Regime detection adds value primarily through avoiding catastrophic drawdowns rather than capturing upside.** The FTSE Russell Balanced Macro framework cut max drawdown from -51.1% (equities) to -16.7% while accepting lower absolute returns (7.8% vs. 11.3%). Daniel & Moskowitz's dynamic momentum strategy doubled Sharpe primarily by avoiding crash periods. The 2x2 framework's greatest value is stagflation protection. This suggests the system should be designed asymmetrically—focused more on loss avoidance during regime transitions than on alpha generation during stable regimes.

## Conclusion

The three briefs rest on foundations of varying solidity. **Brief 5.2 (TSMOM) has the strongest empirical support**, with robust evidence spanning 140+ years across dozens of markets, though the specific crash prediction mechanism at 2-4 week horizons lacks academic validation—practitioners should expect monthly-to-quarterly horizons for crowding signals. **Brief 5.3 (2x2 inflation-risk matrix) has strong conceptual and empirical backing** from Bridgewater's framework, FTSE Russell's recent rigorous testing, and 1970s stagflation data, though the incremental Sharpe improvement from adding the inflation dimension may be closer to 0.07 than 0.10-0.20. **Brief 5.1 (four-factor HMM) is the most novel but least validated**: the individual components (credit spread leading indicator, yield curve, VIX) are well-documented, but their joint use in a 4-state multivariate HMM is untested in the published literature, and practitioners increasingly favor Statistical Jump Models over HMMs for regime detection. The DXY momentum factor as a distinct regime channel represents a genuine literature gap and opportunity for original contribution. Across all three briefs, the literature consistently favors volatility scaling, multi-horizon diversification, and parsimony over model complexity.

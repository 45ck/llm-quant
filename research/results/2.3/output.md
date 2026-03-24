# COT positioning data as a leading signal for commodities and forex

**Bottom line: CFTC Commitments of Traders data contains statistically meaningful but fragile predictive content that works best as a regime filter or crowding indicator rather than a standalone signal.** The academic evidence is decidedly mixed—four foundational papers split roughly 2-against-2 on whether COT positioning predicts returns. More critically, the most comprehensive recent replication (Dreesmann et al. 2023, covering 1986–2020) finds that while COT signals work in isolated markets, they **fail to deliver excess returns at the portfolio level**, suggesting partial arbitrage of the signal. For the llm-quant system's 39-asset universe, COT data is best deployed as a secondary overlay—particularly for gold, crude oil, and FX—combined with regime classification, not as a primary alpha source.

Two critical data issues were uncovered: the CFTC commodity codes for **Gold COMEX should be 088691** (not 084691) and **Silver COMEX should be 084691** (not 084651). Additionally, FX instruments require the Traders in Financial Futures (TFF) report, not the Disaggregated report, which covers only physical commodities.

---

## Paper 1: Sanders, Irwin & Merrin (2009) — the definitive skeptical result

**"Smart Money? The Forecasting Ability of CFTC Large Traders in Agricultural Futures Markets"**
*Journal of Agricultural and Resource Economics, 34(2), 276–296*

This paper applies rigorous Granger causality tests to weekly COT data across 10 agricultural futures markets (1995–2006, 616 observations) and delivers the sharpest negative verdict on COT forecasting ability in the literature. Sanders, Irwin, and Merrin compute two position indicators for each trader category (non-commercial, commercial, non-reportable). The first is **Percent Net Long (PNL)**: PNL_t = (Long_t − Short_t) / (Long_t + Short_t), normalizing net position by group size. The second is **Wang's Sentiment Index (SI)**: SI_t = (S_t − Min) / (Max − Min), where Min and Max span a rolling **3-year window**—this is the COT Index formula the llm-quant system proposes. The two measures correlate at **0.95–0.97**, so choice between them is immaterial.

The core methodology uses bivariate Granger causality with 1–12 week lags selected by AIC, White heteroskedasticity-robust standard errors, and Breusch-Godfrey serial correlation tests. Three test frameworks are employed: (1) standard Granger causality asking whether positions lead returns, (2) long-horizon "fads" regressions averaging positions over 1–104 weeks, and (3) Cumby-Modest tests for extreme positions using **upper/lower 20th percentile thresholds** over a 3-year rolling window.

The results are stark. For the key question "do positions lead returns?", the null hypothesis is rejected at 5% in only **1 of 10 markets** for non-commercial traders, **3 of 10** for commercials (with mixed directional impact), and **0 of 10** for non-reportable traders. The Cumby-Modest extreme-position tests find no systematic pattern. Long-horizon fads regressions reject the null in only **2 of 10** markets for commercials. By contrast, the reverse direction—returns leading positions—is rejected at 5% in **8 of 10 markets** for non-commercials with positive cumulative impact, confirming clear trend-following behavior. Quandt likelihood-ratio tests find **no structural breaks** despite the post-2003 open interest surge. The paper's conclusion is unambiguous: "The results generally do not support use of the COT data in predicting price movements."

**Key limitations**: The study covers only agricultural markets during 1995–2006. Metals, energy, and FX are not tested. The Tuesday-to-Tuesday measurement window means results use data before public release—if no forecasting power exists at measurement, it certainly does not exist after the 3-day publication delay.

---

## Paper 2: Wang (2003) — hedging pressure and speculator profitability

**"The Behavior and Performance of Major Types of Futures Traders"**
*Journal of Futures Markets, 23(1), 1–31*

Wang's study spans **15 futures markets across four asset classes** (financials, agriculture, commodities, currencies) using monthly (4-week) data from October 1992 to March 2000. This broader asset coverage makes it more relevant to the llm-quant system than the agriculture-only Sanders paper.

The methodology involves OLS regressions of position changes on lagged returns, sentiment changes, and information variables (T-bill yield, default premium, dividend yield), followed by market-timing tests regressing future returns on lagged position changes. Wang also constructs explicit trading rules: go long (short) after a position increase (decrease) exceeding **1 standard deviation from the prior 3-year mean**.

The behavioral findings present a striking contrast with Sanders et al. Wang finds that speculators are **contrarians**—their coefficient on lagged returns is negative and significant in 12 of 15 markets—while hedgers engage in **positive feedback trading** (positive coefficient in 12 of 15 markets). This reversal likely stems from the monthly frequency and conditioning on information variables, which separates sentiment-driven from return-driven behavior.

The trading rule profitability results are the paper's strongest contribution. Following speculator position changes yields significant monthly abnormal returns: **Crude Oil +2.64%** (t=2.29), **Japanese Yen +2.88%** (t=2.75), **Corn +2.38%** (t=1.98), **Deutsche Mark +1.50%** (t=1.99). Annualized, these are substantial: the corn strategy implies ~28.6% annual abnormal returns. However, Wang's deeper analysis reveals this profitability primarily reflects **hedging pressure transfer** (risk premium compensation for providing liquidity to hedgers) rather than genuine forecasting ability. Financial futures show the weakest effects, while commodities and currencies show the strongest—a pattern directly relevant to the llm-quant asset universe.

Wang's sentiment index formula—SI = (Current − Min) / (Max − Min) over a 3-year window—became the standard COT Index adopted throughout the subsequent literature. An important clarification: the task description's reference to "1-week momentum vs 4-8 week contrarian reversal" likely conflates this paper with Wang (2001), which tested multiple horizons. The 2003 paper uses a 4-week primary horizon with robustness at 2 and 6 weeks, finding consistent (not flipping) contrarian speculator behavior across all tested horizons.

---

## Paper 3: Gorton, Hayashi & Rouwenhorst (2013) — basis trumps positioning

**"The Fundamentals of Commodity Futures Returns"**
*Review of Finance, 17(1), 35–105 (also NBER WP 13249)*

This is the most theoretically grounded paper of the four, establishing that **inventory levels and the basis (backwardation/contango) are the primary drivers of commodity futures risk premia**, while COT positioning data adds no incremental predictive power. The study covers 31 commodity futures from 1971–2010 using monthly data, with inventories normalized via Hodrick-Prescott filter (smoothness parameter 160,000).

The basis-inventory relationship is confirmed as negative and nonlinear for **30 of 31 commodities**: low inventories produce above-average backwardation. The nonlinearity is economically significant—copper's basis-inventory slope steepens from −3.2 at average inventory to **−15.3 at 25% below average** (t-stat difference: 5.64). Cross-sectional portfolio sorts deliver robust returns: the Low Inventory portfolio outperforms the High Inventory portfolio by **8.06% per annum** (t=3.19). The High−Low Basis spread generates **10.23% annualized** (t=3.73), positive in 58% of months. A simple cross-sectional regression of average basis vs. average excess return yields **R² = 52%**.

The paper's most important negative finding for the llm-quant system concerns COT data directly. Predictive regressions of futures returns on lagged hedging pressure (commercial net long / open interest) produce coefficients **insignificantly different from zero** with R² **below 1%**. The authors explicitly reject the Keynesian hedging pressure hypothesis as an independent predictor: "We reject the Keynesian 'hedging pressure' hypothesis that these positions are an important determinant of risk premiums." Positions respond to prices and inventories but do not predict future risk premia after controlling for the basis.

**Implications for the trading system**: The basis signal is far more powerful than COT positioning (R²=52% vs. <1%). For commodity ETFs, average contango of **−1.1% to −2.1% per annum** erodes returns. Crucially, gold and silver were **excluded** from this study as "essentially financial futures"—the inventory-based framework does not apply to precious metals, making the basis-only approach insufficient for GLD/SLV. This gap is precisely where COT data might still add value.

---

## Paper 4: Dewally, Ederington & Fernando (2013) — hedger concordance explains everything

**"Determinants of Trader Profits in Commodity Futures Markets"**
*Review of Financial Studies, 26(10), 2648–2683*

This paper uses proprietary CFTC Large Trader Reporting System data covering 382 individual traders in crude oil, gasoline, and heating oil futures (June 1993–March 1997). The granularity is unmatched—1,020,474 trader/day/contract observations classified into 11 categories including refiners, hedge funds, investment banks, and market makers.

The headline profitability results confirm the risk premium hypothesis. Likely hedgers lose an average **−3.80% annualized** (significant at p<0.0001); likely speculators earn **+10.40% annualized** (p<0.0001). Hedge funds are the most profitable at **+15.2% annualized** pre-cost (+12.4% after estimated $15/contract round-trip costs, p=0.038).

The paper's most powerful contribution is the **Hedger Concordance (HC)** measure—the fraction of time a trader's position sign matches aggregate hedgers. HC alone explains **14.8% of cross-sectional profit variation** (R²=0.148). The coefficient is −0.1853 (t=−7.686): a trader consistently opposing hedgers earns approximately **45.6 percentage points more annualized** than one always aligned with them. When HC is added to regressions including trader-type dummies, the hedge fund dummy becomes **completely insignificant** (t drops from 3.931 to 0.082). Hedge fund profits are entirely explained by their tendency to position opposite hedgers—not by information or skill advantages. Hedge funds had the lowest mean HC at **22.1%**, systematically taking the other side of hedger positions.

**Regime-conditional findings**: While the task asks about trending vs. mean-reverting regimes, this paper does not directly address regime conditioning. However, the published RFS version adds that long position profits vary inversely with inventories and directly with price volatility—consistent with the Gorton et al. framework. The paper also suggests that commodity momentum may be largely explained by hedging pressure, bridging the two frameworks.

---

## Recent literature reveals signal decay and regime dependence

The most consequential post-2013 finding comes from **Dreesmann, Herberger & Charifzadeh (2023)** in the *International Journal of Financial Markets and Derivatives*. Their comprehensive backtest of Williams Commercial Index strategies across all US futures markets from 1986–2020, benchmarked against Monte Carlo simulations with 100,000 randomized portfolios, finds that COT long-only strategies generate significant results in **only 6 individual markets**. When tested as a diversified portfolio strategy, the COT approach **underperforms**—the signal has been substantially arbitraged away at the portfolio level.

**Chen & Yang (2023)** in the *Journal of Commodity Markets* discover regime-dependent behavior in gold futures using a logistic smooth transition autoregressive (LSTAR) model: money managers adopt **positive feedback (trend-following)** below a gold price threshold (~$1,366) but **switch to negative feedback** above it. This directly supports regime-conditional signal interpretation for the llm-quant system's gold allocation.

**Tornell & Yuan (2012)** in the *Journal of Futures Markets* provide the strongest evidence for COT signals in FX markets. Peaks and troughs of net speculative positions predict spot exchange rate movements: subsequent returns after EUR speculator peaks averaged **0.48%/week, 1.44%/month, and 2.97%/quarter**. The peak/trough methodology outperforms level-based or change-based measures, suggesting that extreme positioning matters far more than moderate readings.

**Boos & Grob (2023)** in the *Journal of Financial Markets* take a reverse-engineering approach, showing that trend-following signals explain **>40% (R²)** of speculators' position changes across 23 commodities. The typical managed money signal is remarkably stable across markets and time. This insight is valuable for anticipating positioning unwinds rather than direct return prediction.

**Robe & Roberts (2024)** in the *Journal of Commodity Markets* document that just ~197 "permanent" large traders dominate agricultural futures positioning, and COT categories **mix rather dissimilar kinds of traders**—providing a structural explanation for signal degradation. Post-2008 hedge fund reclassification as "commercial" further undermines traditional smart money/dumb money framing. No major quantitative firm (AQR, Man Group) has published research advocating COT positioning as a primary trading signal; their published work focuses on price-based momentum.

---

## Data sources are accessible but contain critical errors

The CFTC COT data infrastructure remains fully operational. The main portal at `https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm` is active, and both the Legacy and Disaggregated bulk download URLs follow the confirmed patterns. However, the recommended access method since October 2022 is the **COT Public Reporting Environment (PRE) API** at `https://publicreporting.cftc.gov/`, which provides JSON/CSV output with no API token required.

**Two CFTC commodity codes in the task are incorrect.** Gold COMEX is **088691** (not 084691), and Silver COMEX is **084691** (not 084651). The remaining six codes—Crude Oil WTI (067651), Euro FX (099741), Japanese Yen (097741), British Pound (096742), Australian Dollar (232741), Swiss Franc (092741)—are all verified correct against March 2026 CFTC reports.

Nasdaq Data Link (formerly Quandl) continues to provide free pre-parsed COT data under database code `CFTC`, accessible via the `nasdaqdatalink` Python package with a free API key. Several open-source Python libraries actively download COT data: `cot_reports` and `cftc-cot` both pull directly from CFTC ZIP archives and return pandas DataFrames compatible with Polars conversion.

A critical operational risk emerged: the **2025 government shutdown** (October 1–November 12, 43 days) suspended all COT reporting, leaving traders without positioning data for over six weeks. The backlog was cleared by December 29, 2025. The 2023 ION cyber incident also disrupted reporting for approximately two weeks. Any production system depending on COT data needs a fallback mechanism.

For FX instruments, the system **must use the Traders in Financial Futures (TFF) report**, not the Disaggregated report. The Disaggregated report covers only physical commodities. The TFF report (available since June 2006) provides categories: Dealer/Intermediary, Asset Manager/Institutional, Leveraged Funds, and Other Reportable—offering more granular FX positioning data than the Legacy report's commercial/non-commercial split.

The 2021 position limits rule (effective March 15, 2021) expanded spot-month limits to 25 physical commodity derivatives and raised non-spot month limits to 10% of open interest for the first 50,000 contracts. This did **not** alter COT report format or categories but does affect the upper bounds of positions visible in the data.

---

## The COT Index methodology is standard but parameterization matters

The **min-max normalization formula**—COT_Index = (Current Net − Min over N weeks) / (Max over N weeks − Min over N weeks) × 100—is confirmed as the dominant standard in both academic and practitioner literature. Originated by Larry Williams (2005) and formalized academically by Wang (2001, 2003), it is used by Barchart, TradingView, and virtually all COT analysis platforms. Variations exist (percentile rank, z-score, normalization by open interest first) but are less common.

The **156-week (3-year) lookback** is the de facto industry standard, used by Wang (2001, 2003), MarketsMadeClear, and most TradingView implementations. Academic precedent supports this as "conventional." The 26-week lookback appears in Williams' original short-term reversal strategy. The key tradeoff: shorter lookbacks are more responsive but noisier; longer lookbacks are smoother but slower to adapt to structural regime changes.

The **20/80 threshold** is the most common standard across practitioners and was used in Sanders et al.'s Cumby-Modest extreme position tests. Academic studies tend to use wider zones (70–90 / 10–30) following Basu & Stremme (2009), which may improve robustness by generating signals at less extreme readings. No rigorous comparative study definitively identifies optimal thresholds; the choice should be calibrated to desired signal frequency and conviction level.

**Signal horizon consensus** clusters around **4–8 weeks**. Wang (2003) used 4-week intervals. InsiderWeek's practitioner analysis finds signals valid for approximately 6 weeks. Chen & Maher (2013) warn that the weekly COT publication lag "prevents timely public access," limiting short-horizon utility. The 2-week horizon is too noisy given the 3-day publication delay; the 12-week horizon risks signal decay.

Five **structural breaks** must be respected in backtesting: pre-1992 biweekly publication; 2006 introduction of disaggregated/TFF reports; 2007–2008 financialization spike (commodity-equity correlations jumped from near-zero to ~0.6); post-2008 trader reclassification issues; and the April 2020 USO restructuring / negative oil prices. Tang & Xiong (2012) and Cheng & Xiong (2014) document that commodity index trader growth during 2004–2006 fundamentally altered market dynamics. Using pre-2006 Legacy report data to calibrate signals applied post-2006 is methodologically questionable.

---

## GLD works well for COT signals but USO is fundamentally broken

**GLD** holds physical gold bullion (~880 metric tons as of late 2024) and tracks gold spot minus its 0.40% expense ratio. Tracking error to gold futures is negligible—CME Group analysis confirms "no measurable tracking error" for gold futures themselves, with GLD's deviation limited to the annual expense drag. COT signals derived from COMEX gold futures should apply cleanly to GLD positions.

**SLV** similarly holds physical silver with a 0.50% expense ratio and tight tracking to spot. Slightly wider deviations than GLD reflect silver's lower liquidity, but the physical-backed structure ensures COT applicability.

**USO is the critical problem.** Pre-April 2020, USO held only front-month WTI futures, rolling two weeks before expiration. The negative oil price event forced multiple structural changes: USO shifted to holding a mixture of multiple contract months, executed an 8-for-1 reverse split, and temporarily halted creation baskets. USO returned **−14.6% annualized** over 10 years ending January 2022 while crude oil spot moved very differently—contango roll costs alone can exceed 75% annualized during severe contango. Post-April 2020, COT data for front-month WTI no longer maps to USO's diversified contract holdings. **COT signals should not be applied to USO without substantial adjustment.** Better alternatives include USL (12-month oil fund) or direct futures.

For **forward-filling weekly COT data to daily frequency**: the data measures Tuesday close positions but is published Friday at 3:30 PM ET. The correct implementation to avoid look-ahead bias is to apply data starting **Friday close or Monday open** following publication, forward-filled through the next Thursday. Using Tuesday as the application date introduces 3 days of look-ahead bias that can meaningfully inflate backtest results. For conservative backtesting with the llm-quant system's DuckDB pipeline, assume Monday open execution following Friday release.

Transaction costs are minimal relative to expected signal alpha. GLD bid-ask spreads run **1–4 basis points**; EUR/USD institutional spreads are **0.5–1.0 pips**. With typical COT strategy turnover of 6–12 round-trip trades per year per instrument, annual friction for GLD is approximately 20 basis points and for EUR/USD approximately 10–20 pips. **Signal degradation and regime misidentification are far greater risks than transaction costs** for a system targeting Sharpe >0.8.

---

## Conclusion: actionable parameters and honest assessment of signal strength

The weight of evidence points to COT positioning as a **statistically real but economically modest and decaying signal**. The four foundational papers bracket the range of findings: Sanders et al. find essentially no predictive power in agriculture; Wang finds significant trading-rule profitability driven by hedging pressure; Gorton et al. show the basis dominates positioning as a predictor; Dewally et al. confirm that opposing hedgers is profitable but attribute this to risk premium capture rather than informational advantage. The post-2013 literature, especially Dreesmann et al.'s portfolio-level failure and Robe & Roberts' documentation of trader misclassification, suggests the signal has weakened.

For the llm-quant system, the recommended implementation parameters based on this literature review are:

- **COT Index formula**: Min-max normalization over **156-week lookback** (confirmed standard)
- **Thresholds**: **20/80** for primary signals; consider wider 25/75 zones for higher signal frequency in regime-transition states
- **Signal horizon**: **4–6 weeks** forward, aligned with the system's regime classification rebalancing
- **Interpretation**: Contrarian on commercial positioning (or hedger concordance) at extremes; momentum-confirming for managed money during established trends; regime-conditional switching supported by Chen & Yang (2023)
- **Asset-specific application**: Strongest evidence for gold (COMEX 088691), crude oil (067651), and FX pairs; weakest for agricultural and financial futures
- **Data source**: CFTC PRE API for production; TFF report for FX, Disaggregated for commodities, Legacy for longest history
- **Look-ahead bias prevention**: Apply Friday-published data starting Monday open; forward-fill through following Thursday
- **USO exclusion**: Remove USO from COT signal application or replace with direct WTI futures exposure

The honest assessment: targeting Sharpe >0.8 from COT signals alone is not supported by the evidence. COT data is best used as one input in the system's regime classification framework—elevating conviction when positioning confirms other signals, and triggering caution when crowding reaches extremes—rather than as a standalone alpha generator.
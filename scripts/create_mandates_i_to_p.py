#!/usr/bin/env python3
# ruff: noqa: S603, S607, PLW1510
"""Create Mandate I through P hypothesis beads."""

import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def bd(title, description, btype="feature", priority=2):
    r = subprocess.run(
        [
            "bd",
            "create",
            "--title",
            title,
            "--description",
            description,
            "--type",
            btype,
            "--priority",
            str(priority),
        ],
        capture_output=True,
        text=True,
        cwd="E:/llm-quant",
        encoding="utf-8",
    )
    out = (r.stdout.strip() or r.stderr.strip())[:100]
    print(out)
    return out


LIFECYCLE = "/mandate -> /hypothesis -> /data-contract -> /research-spec freeze -> /backtest -> /robustness -> /paper -> /promote"


def h(
    ml,
    mn,
    num,
    title,
    hypothesis,
    mechanism,
    universe,
    measurement,
    data,
    infra="existing",
    priority=2,
):
    desc = f"**Mandate {ml}: {mn}**\n\n**Hypothesis:** {hypothesis}\n**Mechanism:** {mechanism}\n**Universe:** {universe}\n**Measurement:** {measurement}\n**Data:** {data}\n\n**Lifecycle path:**\n{LIFECYCLE}\n\n**Infrastructure:** {infra}"
    return bd(f"[{ml}{num}] {title}", desc, "feature", priority)


# MANDATE I: Behavioral / Sentiment Signals (8)
h(
    "I",
    "Behavioral / Sentiment Signals",
    1,
    "AAII bearish sentiment extreme predicts SPY bounce",
    "AAII bearish reading >55% predicts positive SPY return over next 4 weeks in >70% of cases since 1990.",
    "AAII sentiment at extremes represents maximum retail fear; smart money buying into this fear drives subsequent recovery.",
    "SPY (AAII weekly survey)",
    "Fetch AAII sentiment from AAII.org (free); signal: bearish >55% -> long SPY; 4-week forward return event study.",
    "AAII.org (free weekly data), Yahoo Finance (SPY)",
    priority := 1,
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    2,
    "Put/call ratio spike (>1.3) predicts SPY bounce within 5 days",
    "CBOE equity put/call ratio >1.3 is followed by positive SPY return in next 5 days with >65% frequency.",
    "Extreme put buying represents peak fear and forced hedging; as fear subsides, hedges are unwound and prices recover.",
    "SPY, ^VIX (CBOE put/call ratio)",
    "Fetch CBOE equity P/C ratio from CBOE website; signal: >1.3 -> long SPY 5d; event study.",
    "CBOE.com (free P/C ratio), Yahoo Finance (SPY)",
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    3,
    "High short interest predicts squeeze risk and positive returns",
    "Stocks in top decile of short interest as % of float produce >5% return over next 20 days when stock is up 5% in 5 days.",
    "High short interest + upward price pressure = short squeeze dynamic; forced covering amplifies positive returns.",
    "Individual stocks (IWM components as proxy)",
    "Use Yahoo Finance short interest data; signal: high SI + 5d momentum -> long; forward 20d return.",
    "Yahoo Finance (short interest via yfinance)",
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    4,
    "Insider cluster buying predicts 3-month outperformance",
    "Stocks with >3 distinct insider purchases in 30 days outperform SPY by >3% over next 3 months.",
    "Insiders have superior information about firm prospects; cluster buying amplifies signal strength.",
    "S&P 500 components",
    "SEC Form 4 data from EDGAR (free); cluster: >3 buys/30d; forward 3m return vs SPY benchmark.",
    "SEC EDGAR Form 4 filings (free), Yahoo Finance",
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    5,
    "Analyst consensus downgrade cluster signals value opportunity",
    "Stocks receiving >3 analyst downgrades in 5 days produce positive excess return of >4% over subsequent 60 days.",
    "Consensus downgrades create overshooting; when multiple analysts downgrade simultaneously, overreaction likely.",
    "S&P 500 components",
    "Yahoo Finance analyst data; count downgrades per stock per 5d window; signal: >3 -> contrarian long.",
    "Yahoo Finance (analyst ratings)",
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    6,
    "IPO volume extreme predicts broad market peak",
    "Monthly IPO count >20 predicts SPY underperformance over next 6 months in >65% of occurrences.",
    "Extreme IPO activity signals market froth; companies rush to go public at peak valuations.",
    "SPY, IPO count data",
    "Monthly IPO count from Renaissance Capital or Wikipedia; signal: >20/month -> defensive posture.",
    "Renaissance Capital (free), Yahoo Finance (SPY)",
    2,
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    7,
    "Margin debt growth >20% YoY predicts increased correction probability",
    "When NYSE margin debt grows >20% YoY, SPY probability of >10% drawdown within 12 months is >80%.",
    "Margin debt measures leverage in the system; excessive leverage amplifies downside when margin calls cascade.",
    "SPY (FINRA margin debt data)",
    "FINRA monthly margin debt; compute YoY growth; signal: >20% -> reduce risk; forward 12m drawdown probability.",
    "FINRA.org (free margin data), Yahoo Finance",
    2,
)
h(
    "I",
    "Behavioral / Sentiment Signals",
    8,
    "Retail investor buy ratio extreme predicts contrarian 10d return",
    "When retail buy ratio (approximated by small-lot trade % from CBOE) >70%, SPY 10d return is negative with >60% frequency.",
    "Retail FOMO buying marks short-term tops; institutional selling into retail euphoria creates temporary overvaluation.",
    "SPY",
    "Small-lot trade volume as retail proxy (Yahoo Finance volume patterns); or CBOE retail flow data; forward 10d SPY.",
    "CBOE.com, Yahoo Finance (SPY)",
)

print("Mandate I done (8)")

# MANDATE J: Alternative Data Signals (9)
h(
    "J",
    "Alternative Data Signals",
    1,
    "Google Trends recession search spike predicts SPY underperformance",
    "Google Trends score >70 for 'recession' (US) predicts negative SPY 30d return in >65% of cases.",
    "Search behavior reflects mass anxiety; recession search spikes are leading indicators of consumer confidence drops.",
    "SPY (Google Trends: recession, stock market crash)",
    "Fetch Google Trends via pytrends; signal: score >70 -> defensive; forward 30d SPY return.",
    "Google Trends (pytrends library), Yahoo Finance (SPY)",
)
h(
    "J",
    "Alternative Data Signals",
    2,
    "TSA checkpoint passenger volume predicts airline/travel sector",
    "TSA daily checkpoint volume YoY growth >10% predicts JETS (airline ETF) outperformance vs SPY over next 30 days.",
    "Passenger volume is a real-time consumer spending indicator for travel sector; leads earnings by 3-6 months.",
    "JETS, SPY (TSA data: tsa.gov)",
    "TSA publishes daily checkpoint data on tsa.gov; compute YoY; signal: >10% -> long JETS.",
    "TSA.gov (free), Yahoo Finance (JETS, SPY)",
)
h(
    "J",
    "Alternative Data Signals",
    3,
    "US electricity consumption growth predicts industrial sector",
    "EIA weekly US electricity consumption YoY growth >3% predicts XLI (industrials) outperformance over next 60 days.",
    "Electricity consumption is a coincident indicator of industrial activity; rising consumption precedes industrial earnings.",
    "XLI, XLB (EIA: Weekly Electric Power Industry Report)",
    "EIA electricity data (free API); compute YoY; signal: >3% -> overweight XLI.",
    "EIA.gov (free API), Yahoo Finance (XLI, XLB)",
)
h(
    "J",
    "Alternative Data Signals",
    4,
    "OpenTable reservations YoY growth predicts consumer discretionary",
    "OpenTable seated diner YoY growth >15% predicts XLY outperformance vs XLP over next 30 days.",
    "Restaurant reservations are a real-time consumer sentiment indicator; dining out = discretionary spending confidence.",
    "XLY, XLP (OpenTable public data)",
    "OpenTable publishes daily restaurant data; compute YoY; signal: >15% -> long XLY/short XLP.",
    "OpenTable.com (public data), Yahoo Finance (XLY, XLP)",
)
h(
    "J",
    "Alternative Data Signals",
    5,
    "Google Trends stock market crash search predicts VIX spike within 10 days",
    "Google Trends score >60 for 'stock market crash' predicts VIX >25 within 10 days with >65% accuracy.",
    "Mass search for crash language signals elevated fear approaching inflection point; precedes volatility events.",
    "^VIX, SPY (Google Trends: stock market crash)",
    "pytrends daily data; signal: score >60 -> defensive; forward 10d VIX level; precision/recall.",
    "Google Trends (pytrends), Yahoo Finance (^VIX)",
    priority := 1,
)
h(
    "J",
    "Alternative Data Signals",
    6,
    "LinkedIn tech job postings growth predicts QQQ 60d outperformance",
    "LinkedIn tech job posting YoY growth >20% predicts QQQ outperforming SPY by >3% over next 60 days.",
    "Tech hiring reflects corporate R&D investment confidence; job postings lead capex and revenue growth.",
    "QQQ, SPY (LinkedIn job data via Indeed or BLS proxy)",
    "BLS JOLTS tech sector data (free) as LinkedIn proxy; signal: >20% -> overweight QQQ.",
    "BLS.gov JOLTS (free), Yahoo Finance (QQQ, SPY)",
)
h(
    "J",
    "Alternative Data Signals",
    7,
    "Wikipedia S&P 500 article view spike predicts index mean-reversion",
    "Weekly Wikipedia views of 'S&P 500' page >3 sigma above 52w average predicts SPY negative 10d return with >60% accuracy.",
    "Wikipedia views proxy for retail investor attention; attention spikes at market peaks as retail piles in.",
    "SPY (Wikipedia pageview data: wikimedia.org)",
    "Wikimedia Foundation API (free) for daily page views; compute z-score; signal: >3sigma -> short SPY 10d.",
    "Wikimedia Foundation API (free), Yahoo Finance (SPY)",
)
h(
    "J",
    "Alternative Data Signals",
    8,
    "Box office revenue growth predicts consumer discretionary sector",
    "Weekly US box office YoY growth >20% predicts XLY outperformance over next 30 days.",
    "Movie-going is a discretionary activity; strong box office signals consumer confidence and spending capacity.",
    "XLY, XLP (Box Office Mojo free data)",
    "Box Office Mojo weekly domestic gross; compute YoY; signal: >20% -> long XLY.",
    "Box Office Mojo (free), Yahoo Finance (XLY, XLP)",
)
h(
    "J",
    "Alternative Data Signals",
    9,
    "US housing permit YoY growth predicts XHB (homebuilder) momentum",
    "US monthly building permits YoY growth >10% predicts XHB (homebuilders ETF) outperforming SPY by >3% over next 60 days.",
    "Building permits are a leading indicator of construction activity and housing demand; leads homebuilder revenue.",
    "XHB, ITB (FRED: PERMIT)",
    "FRED PERMIT monthly data; compute YoY; signal: >10% -> overweight XHB.",
    "FRED API (PERMIT), Yahoo Finance (XHB, ITB)",
)

print("Mandate J done (9)")

# MANDATE K: Risk Premium / Spread-Based Signals (7)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    1,
    "Quality factor (QUAL vs SPY) outperforms during risk-off regimes",
    "QUAL (iShares MSCI USA Quality Factor) outperforms SPY by >2% during periods when VIX >20, confirmed over 5-year backtest.",
    "Quality companies (high ROE, stable earnings, low leverage) are defensive; outperform during uncertainty.",
    "QUAL, SPY, ^VIX",
    "Segment by VIX regime (>20 vs <20); measure QUAL vs SPY return differential; Sharpe by regime.",
    "Yahoo Finance (QUAL, SPY, ^VIX)",
)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    2,
    "Value-momentum combination reduces factor crash risk vs standalone",
    "Long VLUE + MTUM equal-weight portfolio Sharpe >0.9 and max drawdown <15% over 5 years; outperforms either factor alone.",
    "Value and momentum have low/negative correlation; combining reduces factor-specific drawdown risk.",
    "VLUE, MTUM, SPY",
    "5-year backtest equal-weight VLUE+MTUM; compare Sharpe, max DD, Calmar to standalone factors.",
    "Yahoo Finance (VLUE, MTUM, SPY)",
)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    3,
    "Low-volatility premium (USMV vs SPY) survives transaction costs",
    "USMV (low-vol ETF) achieves risk-adjusted return Sharpe >0.8 with max DD <12% over 10-year backtest on monthly rebalancing.",
    "Low-volatility anomaly: low-beta stocks earn higher risk-adjusted returns due to leverage constraints and institutional benchmarking.",
    "USMV, SPY",
    "10-year monthly rebalance backtest; Sharpe, max DD, Calmar for USMV vs SPY; with/without cost assumption.",
    "Yahoo Finance (USMV, SPY)",
)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    4,
    "Size premium (IWM vs SPY) conditional on economic expansion",
    "IWM outperforms SPY by >3% in the 6 months following ISM PMI crossing above 50 from below.",
    "Small caps are more economically sensitive; they benefit disproportionately from early-cycle expansion.",
    "IWM, SPY (FRED: ISM_PMI)",
    "FRED ISM; detect 50-crossing events; event study on forward 6m IWM vs SPY return.",
    "FRED API (ISM), Yahoo Finance (IWM, SPY)",
)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    5,
    "EM carry trade (high-yield EM vs low-yield DM) generates positive alpha",
    "A long EEM / short EFA position rebalanced monthly achieves Sharpe >0.5 when yield differential (EM-DM 10Y) >3%.",
    "Carry theory: higher-yielding assets attract capital flows; positive carry generates excess return.",
    "EEM, EFA (FRED: international 10Y yields)",
    "Compute EM-DM 10Y yield spread via FRED; signal: spread >3% -> long EEM/short EFA; monthly rebalance backtest.",
    "FRED API (international yields), Yahoo Finance (EEM, EFA)",
)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    6,
    "Equity risk premium extreme (ERP >6%) predicts 2-year positive SPY return",
    "When ERP (SPX earnings yield minus 10Y) >6%, SPY produces >15% return over next 24 months in >80% of cases since 1950.",
    "ERP >6% = equities cheap relative to bonds; fundamental mean reversion drives subsequent outperformance.",
    "SPY (FRED: DGS10, S&P 500 earnings yield)",
    "FRED DGS10; compute S&P 500 earnings yield from P/E; ERP = earnings_yield - DGS10; event study on forward 24m.",
    "FRED API (DGS10), Yahoo Finance (SPY), S&P 500 P/E data",
    priority := 1,
)
h(
    "K",
    "Risk Premium / Spread-Based Signals",
    7,
    "Risk parity timing via SPY-TLT correlation sign improves Sharpe",
    "Risk parity portfolio (60/40 SPY/TLT vol-adjusted) rebalanced when SPY-TLT 20d correlation changes sign achieves Sharpe >0.9.",
    "SPY-TLT correlation sign determines whether diversification benefit exists; rebalancing on regime change is optimal.",
    "SPY, TLT",
    "Compute 20d correlation SPY/TLT; rebalance risk parity weights on sign change; compare to static 60/40 Sharpe.",
    "Yahoo Finance (SPY, TLT)",
    "needs: correlation_regime strategy class",
    1,
)

print("Mandate K done (7)")

# MANDATE L: Technical / Microstructure / Flow Signals (7)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    1,
    "Opening gap frequency clusters predict mean-reversion",
    "SPY days following >5 gap-up opens in prior 10 days produce negative 1d returns with >60% frequency.",
    "Clustering of opening gaps signals elevated momentum that historically overshoots; mean-reversion follows.",
    "SPY",
    "Compute daily open-vs-prior-close gap; count +gap frequency in 10d window; signal: cluster -> fade; 1d forward.",
    "Yahoo Finance (SPY OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    2,
    "High-volume conviction candle predicts 5d trend continuation",
    "An up-candle on 2x average volume with close in top 25% of range predicts positive SPY return next 5 days with >65% accuracy.",
    "High-volume conviction candles signal institutional buying; strong close + volume = sustainable demand.",
    "SPY, QQQ",
    "Compute candle body position (close vs range), volume vs 20d average; signal: conviction candle -> long 5d.",
    "Yahoo Finance (SPY, QQQ OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    3,
    "VWAP deviation above 1% predicts intraday mean-reversion",
    "SPY trading >1% above daily VWAP at midday reverts to VWAP by close in >60% of sessions.",
    "VWAP is the fair value anchor for institutional execution algorithms; deviations are arbitraged back.",
    "SPY",
    "Compute daily VWAP from OHLCV (approximation); measure close-to-VWAP reversion; daily win rate.",
    "Yahoo Finance (SPY OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    4,
    "ATR breakout confirmation filter improves momentum entry Sharpe",
    "Adding ATR-based breakout confirmation (price breakout + ATR expansion) to simple SMA crossover improves Sharpe by >0.2.",
    "Breakouts with expanding ATR have higher probability of follow-through vs breakouts on contracting vol.",
    "SPY, QQQ",
    "Backtest SMA crossover with vs without ATR expansion filter; compare Sharpe, win rate, profit factor.",
    "Yahoo Finance (SPY, QQQ OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    5,
    "NR7 narrow range precedes volatility expansion",
    "SPY NR7 (narrowest 7-day range) days are followed by range expansion >1.5x average range within next 3 days in >65% of cases.",
    "Range contraction reflects market indecision; accumulated pressure releases as participants act on new information.",
    "SPY",
    "Identify NR7 days from OHLCV; measure next 3d range vs 20d average range; expansion frequency and magnitude.",
    "Yahoo Finance (SPY OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    6,
    "On-balance volume divergence from price predicts trend reversal",
    "Negative OBV divergence (price making new high, OBV falling) predicts SPY peak within 10 days with >60% accuracy.",
    "OBV captures volume weight of up vs down days; price/volume divergence signals weakening buying conviction.",
    "SPY, QQQ",
    "Compute OBV from OHLCV; detect price/OBV divergence; forward 10d peak-detection precision/recall.",
    "Yahoo Finance (SPY OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)
h(
    "L",
    "Technical / Microstructure / Flow Signals",
    7,
    "Relative volume spike (>3x) at support level signals high-probability entry",
    "SPY testing prior support on >3x average volume results in positive next-5d return in >70% of cases.",
    "High-volume support tests indicate institutional buyers absorbing supply; strong signal for trend continuation.",
    "SPY, QQQ",
    "Define support levels from 20/50d SMA; detect touches with volume >3x average; forward 5d return analysis.",
    "Yahoo Finance (SPY OHLCV)",
    "needs: microstructure_ohlcv strategy class",
)

print("Mandate L done (7)")

# MANDATE M: Correlation / Regime Signals (7)
h(
    "M",
    "Correlation / Regime Signals",
    1,
    "SPY-GLD correlation regime identifies risk-on/risk-off",
    "Periods with 20d SPY-GLD correlation <-0.3 are risk-off; SPY buy-and-hold conditioned on risk-on achieves Sharpe >0.8.",
    "SPY-GLD correlation measures equity vs safe-haven demand balance; negative correlation = flight-to-safety regime.",
    "SPY, GLD, ^VIX",
    "Compute 20d rolling corr; classify regime; measure SPY Sharpe in each regime; backtest risk-on-only strategy.",
    "Yahoo Finance (SPY, GLD)",
    "needs: correlation_regime strategy class",
)
h(
    "M",
    "Correlation / Regime Signals",
    2,
    "Cross-asset correlation dispersion predicts market stress",
    "When pairwise correlation of SPY/TLT/GLD/HYG exceeds 0.7 on average, SPY drawdown >3% occurs within 15 days with >65% frequency.",
    "High cross-asset correlation = all assets moving together = risk-off panic; diversification benefits collapse.",
    "SPY, TLT, GLD, HYG",
    "Rolling 10d pairwise correlation among 4 assets; average correlation as stress signal; forward 15d SPY drawdown.",
    "Yahoo Finance (SPY, TLT, GLD, HYG)",
    "needs: correlation_regime strategy class",
)
h(
    "M",
    "Correlation / Regime Signals",
    3,
    "Rolling bond-equity correlation sign change signals regime transition",
    "When 20d rolling TLT-SPY correlation changes sign, subsequent 20d SPY volatility is 30% higher than normal.",
    "Regime transitions (risk-on to risk-off) are characterized by correlation breakdowns; high vol follows.",
    "SPY, TLT",
    "Detect sign changes in 20d rolling SPY/TLT correlation; measure forward 20d SPY realized vol; statistical test.",
    "Yahoo Finance (SPY, TLT)",
    "needs: correlation_regime strategy class",
)
h(
    "M",
    "Correlation / Regime Signals",
    4,
    "Intra-sector correlation rise predicts sector crowding and mean-reversion",
    "When average pairwise correlation within XLK (technology sector) rises >0.7, XLK underperforms SPY by >3% over next 30 days.",
    "High intra-sector correlation signals momentum crowding; when all stocks in sector move together, factor risk is elevated.",
    "XLK components, SPY",
    "Compute pairwise correlation of top-10 XLK holdings; average; signal: >0.7 -> underweight XLK.",
    "Yahoo Finance (XLK components)",
    "needs: correlation_regime strategy class",
)
h(
    "M",
    "Correlation / Regime Signals",
    5,
    "Multi-asset Sharpe dispersion signals alpha opportunity window",
    "When cross-asset Sharpe dispersion (std dev of rolling 60d Sharpe across 8 assets) >1.0, rotation strategy alpha is 2x normal.",
    "High Sharpe dispersion = assets performing differently = rotation opportunity; rotation strategies thrive on dispersion.",
    "SPY, TLT, GLD, HYG, EEM, USO, BTC-USD, IWM",
    "Compute 60d Sharpe for each; standard deviation across assets; correlate dispersion with rotation strategy alpha.",
    "Yahoo Finance (8 assets)",
    "needs: asset_rotation strategy class",
)
h(
    "M",
    "Correlation / Regime Signals",
    6,
    "Correlation mean-reversion in asset pairs generates statistical arb",
    "Long-term cointegrated pairs (e.g., XLE/USO) with z-score >2 revert within 10 days in >65% of cases.",
    "Cointegrated pairs share common fundamental driver; temporary deviations are mean-reverting by construction.",
    "XLE/USO, GLD/GDXJ, SPY/IWM",
    "ADF cointegration test; compute z-score on spread; signal: z>2 -> long cheap/short expensive.",
    "Yahoo Finance (multiple pairs)",
    "needs: pairs_ratio strategy class",
)
h(
    "M",
    "Correlation / Regime Signals",
    7,
    "Correlation surprise (delta-corr >0.3 in 1 week) predicts drawdown",
    "When any major asset pair (SPY/TLT, SPY/HYG) correlation changes by >0.3 in 5 days, SPY drawdown >2% within 10 days with >70% precision.",
    "Rapid correlation shifts signal regime change; when equity-bond relationship suddenly breaks, systemic risk is elevated.",
    "SPY, TLT, HYG",
    "Compute 5d change in 10d rolling correlation; signal: >0.3 change -> defensive; forward 10d SPY return.",
    "Yahoo Finance (SPY, TLT, HYG)",
    "needs: correlation_regime strategy class",
    1,
)

print("Mandate M done (7)")

# MANDATE N: Fixed Income / Yield Curve (8)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    1,
    "3m/10y yield curve inversion depth predicts recession timing",
    "3m/10y spread (DGS10-DTB3) inversion depth >-100bps predicts recession within 12 months in 100% of post-1980 cases.",
    "3m/10y is the Fed's preferred recession indicator (more predictive than 2s10s); deeper inversion = closer recession.",
    "SPY, TLT (FRED: DGS10, DTB3)",
    "FRED DGS10-DTB3 spread; inversion depth analysis; event study on recessions and SPY forward returns.",
    "FRED API (DGS10, DTB3), Yahoo Finance (SPY, TLT)",
    "needs: yield_curve_regime strategy class",
    1,
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    2,
    "Fed funds vs 2-year spread predicts FOMC surprise direction",
    "When FEDFUNDS rate lags 2Y Treasury by >100bps, the probability of a hawkish Fed surprise in next meeting is >70%.",
    "2Y Treasury price in future policy; large gap between FF rate and 2Y suggests market expects hikes the Fed hasn't signaled.",
    "TLT, IEF (FRED: FEDFUNDS, DGS2)",
    "FRED FEDFUNDS vs DGS2; compute gap; correlate with direction of next FOMC decision; TLT reaction.",
    "FRED API (FEDFUNDS, DGS2), Yahoo Finance (TLT)",
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    3,
    "TLT 63d momentum signal beats buy-and-hold TLT",
    "Long TLT only when 63d return >0 (positive momentum) achieves Sharpe >0.5 and max DD <15% over 10 years.",
    "Bond momentum is persistent due to autocorrelation in inflation expectations and Fed policy cycles.",
    "TLT",
    "10-year backtest: long TLT when 63d return >0, else cash; Sharpe, max DD vs buy-and-hold TLT.",
    "Yahoo Finance (TLT)",
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    4,
    "Yield curve un-inversion signals recession confirmation and equity rotation",
    "When 2s10s rises from negative back above 0, SPY underperforms TLT by >5% over next 6 months in >75% of historical cases.",
    "Un-inversion = recession arriving; historical pattern shows equity underperformance after curve normalizes post-inversion.",
    "SPY, TLT (FRED: DGS2, DGS10)",
    "FRED 2s10s; detect zero-crossing from below; event study on forward 6m SPY vs TLT.",
    "FRED API (DGS2, DGS10), Yahoo Finance (SPY, TLT)",
    "needs: yield_curve_regime strategy class",
    1,
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    5,
    "TIPS breakeven inflation >2.5% predicts commodity outperformance",
    "5Y5Y forward breakeven inflation >2.5% predicts DJP (commodities) outperformance vs SPY over next 60 days.",
    "TIPS breakeven reflects institutional inflation expectations; rising breakevens signal commodity demand.",
    "TIP, DJP, GLD (FRED: T5YIFR)",
    "FRED T5YIFR breakeven; signal: >2.5% -> overweight DJP; forward 60d return comparison.",
    "FRED API (T5YIFR), Yahoo Finance (TIP, DJP, GLD)",
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    6,
    "Corporate-Treasury spread (LQD vs IEF) predicts equity credit cycle",
    "When LQD yield minus IEF yield spread widens >1% in 30 days, SPY 30d forward return is negative with >65% frequency.",
    "Credit spreads widen before equity downturn; corporate bond market is a leading indicator of equity stress.",
    "LQD, IEF, HYG, SPY",
    "Compute LQD-IEF yield spread proxy via price ratio; 30d change; signal: >1% widening -> underweight SPY.",
    "Yahoo Finance (LQD, IEF, HYG, SPY)",
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    7,
    "MOVE index (bond vol) extreme predicts TLT mean-reversion",
    "MOVE index >130 is followed by TLT positive return over next 10 days in >65% of cases (bond vol mean-reverts).",
    "MOVE extreme = institutional hedging demand peak in rates; overshoot reverts as uncertainty resolves.",
    "TLT (MERRILL LYNCH MOVE index proxy via VXTYN)",
    "Approximate MOVE via VXTYN or 30d realized vol of TLT; signal: extreme -> long TLT 10d; event study.",
    "Yahoo Finance (TLT, VXTYN)",
)
h(
    "N",
    "Fixed Income / Yield Curve Signals",
    8,
    "Global bond yield convergence predicts capital flow to US equities",
    "When spread between US 10Y and German 10Y Bund narrows >50bps in 60 days, EFA outperforms SPY by >2% over next 60d.",
    "Yield convergence reduces US rate advantage; capital flows from US to international equities for higher relative return.",
    "EFA, SPY (FRED: DGS10, German Bund via FRED)",
    "FRED international yields; compute US-German spread; 60d change; signal: narrowing -> overweight EFA.",
    "FRED API (DGS10, INTDSRDEM193N), Yahoo Finance (EFA, SPY)",
    "needs: yield_curve_regime strategy class",
)

print("Mandate N done (8)")

# MANDATE O: Commodities (8)
h(
    "O",
    "Commodity Signals",
    1,
    "EIA crude oil inventory surprise predicts USO direction",
    "EIA weekly crude inventory change >2M barrels vs consensus predicts USO negative 5d return with >65% frequency.",
    "Oil supply shock: larger-than-expected builds signal weak demand; immediate negative price pressure.",
    "USO, XLE (EIA weekly inventory report)",
    "EIA.gov weekly inventory vs consensus (Bloomberg median); signal: build >2M -> short USO 5d.",
    "EIA.gov (free), Yahoo Finance (USO, XLE)",
)
h(
    "O",
    "Commodity Signals",
    2,
    "Gold/Silver ratio extreme mean-reversion generates metals alpha",
    "Gold/Silver ratio >85 predicts silver (SLV) outperformance over gold (GLD) by >5% over next 30 days.",
    "High G/S ratio = silver undervalued relative to gold; mean-reversion driven by industrial vs safe-haven demand balance.",
    "GLD, SLV",
    "Compute GLD/SLV ratio daily; signal: z>2 or ratio >85 -> long SLV/short GLD; 5-year backtest.",
    "Yahoo Finance (GLD, SLV)",
    "needs: pairs_ratio strategy class",
    1,
)
h(
    "O",
    "Commodity Signals",
    3,
    "Commodity momentum rotation (GLD/USO/DJP) by 60d Sharpe beats equal-weight",
    "Monthly rotation to highest 60d Sharpe among GLD/USO/DJP produces Sharpe >0.5 and beats equal-weight baseline.",
    "Momentum persists in commodity markets due to supply/demand imbalances that take months to resolve.",
    "GLD, USO, DJP",
    "Monthly rotation by 60d Sharpe; compare to equal-weight GLD/USO/DJP and to SPY; 5-year backtest.",
    "Yahoo Finance (GLD, USO, DJP)",
    "needs: asset_rotation strategy class",
)
h(
    "O",
    "Commodity Signals",
    4,
    "WTI backwardation predicts positive USO return over 30 days",
    "When WTI front-month futures price exceeds 6m forward price by >5% (backwardation), USO produces positive 30d return with >65% frequency.",
    "Backwardation = near-term supply tightness; spot price premium signals demand exceeding current supply.",
    "USO, XLE (CME WTI futures data)",
    "WTI futures curve from CME (free delayed); compute front/6m spread; signal: backwardation -> long USO.",
    "CME Group (free delayed futures), Yahoo Finance (USO, XLE)",
)
h(
    "O",
    "Commodity Signals",
    5,
    "Copper/Gold ratio mean-reversion generates commodity pairs alpha",
    "Copper/Gold ratio (CPER/GLD) z-score >1.5 or <-1.5 predicts mean-reversion trade return of >3% over 20 days.",
    "Copper (industrial) and gold (safe-haven) are economically linked; ratio deviations reflect temporary sentiment extremes.",
    "CPER, GLD",
    "Compute 60d rolling mean/std of CPER/GLD ratio; signal: z>1.5 -> mean reversion; 3-year backtest.",
    "Yahoo Finance (CPER, GLD)",
    "needs: pairs_ratio strategy class",
)
h(
    "O",
    "Commodity Signals",
    6,
    "Agricultural weather anomaly predicts DBA (agriculture ETF) momentum",
    "El Nino/La Nina events in crop-sensitive regions predict DBA (agriculture ETF) outperformance vs SPY over next 90 days.",
    "Weather anomalies disrupt agricultural supply; supply shocks create sustained price momentum in agri commodities.",
    "DBA (NOAA: ENSO index)",
    "NOAA ENSO index (free); classify El Nino/La Nina; event study on forward 90d DBA vs SPY return.",
    "NOAA.gov (free ENSO data), Yahoo Finance (DBA)",
)
h(
    "O",
    "Commodity Signals",
    7,
    "Base metals vs precious metals rotation by yield curve regime",
    "Long LIT+CPER/short GLD when 2s10s >0 (steepening/growth); reverse when 2s10s <0 (inversion/recession).",
    "Base metals benefit from growth; precious metals from recession/inflation. Yield curve classifies economic regime.",
    "LIT, CPER, GLD (FRED: DGS2, DGS10)",
    "Yield curve regime from FRED; dynamic rotation LIT+CPER vs GLD; 5-year backtest vs equal-weight.",
    "FRED API (DGS2, DGS10), Yahoo Finance (LIT, CPER, GLD)",
    "needs: yield_curve_regime strategy class",
)
h(
    "O",
    "Commodity Signals",
    8,
    "Natural gas seasonal pattern (winter premium) generates USO/UNG calendar alpha",
    "UNG (natural gas ETF) produces positive return Oct-Feb in >70% of years driven by winter heating demand.",
    "Natural gas has strong seasonal demand patterns; winter premium is predictable and tradeable.",
    "UNG",
    "Monthly seasonality analysis: UNG return by calendar month; t-test for winter months vs summer; Sharpe.",
    "Yahoo Finance (UNG)",
    "needs: calendar_event strategy class",
)

print("Mandate O done (8)")

# MANDATE P: Factor / Meta-Strategy Signals (8)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    1,
    "Momentum factor crowding predicts subsequent factor crash",
    "When MTUM ETF 60d return exceeds 3 std dev above historical average, MTUM underperforms SPY by >5% over next 30 days.",
    "Momentum crowding = all investors long same stocks; when crowding unwinds (correlation spike), losses are concentrated.",
    "MTUM, SPY",
    "Compute MTUM 60d return z-score; signal: >3sigma -> underweight momentum; forward 30d MTUM vs SPY.",
    "Yahoo Finance (MTUM, SPY)",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    2,
    "Cross-sectional factor return dispersion predicts market regime",
    "When monthly return dispersion across size/value/momentum/quality factors >2 std dev, subsequent month SPY vol is 30% higher.",
    "High factor dispersion = conflicting market narratives; rotation across factors signals regime uncertainty.",
    "VLUE, MTUM, USMV, QUAL, SPY",
    "Compute monthly return for 4 factors; std dev; correlate dispersion with forward month SPY vol.",
    "Yahoo Finance (VLUE, MTUM, USMV, QUAL)",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    3,
    "Factor momentum (rank by 12m return) generates positive alpha",
    "Monthly rotation to best-performing factor (VLUE/MTUM/USMV/QUAL) by 12-1m return achieves Sharpe >0.7.",
    "Factor momentum: winning factors continue to win due to persistent regime effects and capital flows.",
    "VLUE, MTUM, USMV, QUAL",
    "Monthly rotation by 12-1m return; backtest vs equal-weight 4-factor portfolio; Sharpe comparison.",
    "Yahoo Finance (VLUE, MTUM, USMV, QUAL)",
    "needs: asset_rotation strategy class",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    4,
    "Defensive factor tilt during high VIX regime improves drawdown control",
    "Overweighting USMV+QUAL (defensive factors) when VIX >20 reduces max drawdown by >30% vs equal-weight factor portfolio.",
    "High VIX = equity stress; low-vol and quality factors outperform during stress due to flight-to-safety within equities.",
    "VLUE, MTUM, USMV, QUAL, ^VIX",
    "VIX-conditional factor weights; backtest vs static equal-weight; max DD comparison.",
    "Yahoo Finance (VLUE, MTUM, USMV, QUAL, ^VIX)",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    5,
    "Low beta anomaly generates superior Sharpe vs market",
    "SPLV (low beta ETF) achieves Sharpe >SPY Sharpe and max DD <SPY max DD over 10-year backtest.",
    "Low-beta anomaly: investors overpay for lottery-like high-beta stocks; low-beta stocks are undervalued on risk-adjusted basis.",
    "SPLV, SPY",
    "10-year backtest SPLV vs SPY; Sharpe, max DD, beta comparison; monthly rebalance.",
    "Yahoo Finance (SPLV, SPY)",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    6,
    "Factor correlation regime signals diversification opportunity",
    "When pairwise correlation among VLUE/MTUM/USMV/QUAL drops below 0.3, equal-weight factor portfolio Sharpe doubles vs high-correlation regime.",
    "Low factor correlation = each factor adding unique information; diversification benefit maximized in low-corr regime.",
    "VLUE, MTUM, USMV, QUAL",
    "Rolling 60d pairwise factor correlation; split by low/high correlation regime; compare equal-weight Sharpe.",
    "Yahoo Finance (VLUE, MTUM, USMV, QUAL)",
    "needs: correlation_regime strategy class",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    7,
    "Smart beta ETF flow momentum predicts factor continuation",
    "Sector/factor ETF with 3m net inflow >$1B generates above-average return over next 60 days with >60% frequency.",
    "Fund flows reflect institutional conviction; sustained inflows = demand support that drives continuation.",
    "MTUM, VLUE, USMV, sector ETFs",
    "ETF fund flows from ETF.com or Yahoo Finance proxy; signal: large inflow -> overweight factor.",
    "ETF.com, Yahoo Finance (factor ETFs)",
)
h(
    "P",
    "Factor / Meta-Strategy Signals",
    8,
    "Strategy winning streak predicts subsequent underperformance",
    "Systematic strategies with 5+ consecutive winning months underperform SPY by >3% over next 3 months in >65% of cases.",
    "Consecutive wins signal overfitting/crowding; strategies start to lose as edge gets arbitraged away or regime changes.",
    "Any systematic factor ETF",
    "Define consecutive win months; split by streak length (0-2, 3-4, 5+); measure forward 3m return vs SPY.",
    "Yahoo Finance (factor ETFs)",
    2,
)

print("Mandate P done (8)")
print("\nAll Mandates I-P complete. Total new beads: 64")

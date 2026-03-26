#!/usr/bin/env python3
# ruff: noqa: S603, S607, PLW1510
"""Create Mandate A through H hypothesis beads."""

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


# MANDATE A: Macro / Cross-Asset Signals (9)
A = "A"
AN = "Macro / Cross-Asset Signals"
h(
    A,
    AN,
    1,
    "Cu/Au ratio leads equity drawdowns by 15-25 days",
    "The copper/gold ratio declining below its 30d SMA predicts SPY drawdowns 15-25 trading days later with >55% directional accuracy.",
    "Copper reflects industrial growth; gold reflects risk-off demand. Cu/Au falling signals deceleration before equities reprice.",
    "CPER, GLD, SPY",
    "Granger causality at 15/20/25d lags; signal: Cu/Au 5d MA < 30d MA -> underweight SPY; 5-year backtest.",
    "Yahoo Finance (CPER, GLD, SPY)",
)
h(
    A,
    AN,
    2,
    "HYG-IEF credit spread widening predicts SPY correction within 10 days",
    "When HYG/IEF spread widens >0.5% in 5 days, SPY returns are negative over next 10 days with >60% frequency.",
    "Credit markets lead equity markets in risk-off episodes; HY spread is a leading stress indicator.",
    "HYG, IEF, SPY",
    "5d spread change >0.5% -> short SPY signal; event study on forward 10d returns.",
    "Yahoo Finance (HYG, IEF, SPY)",
)
h(
    A,
    AN,
    3,
    "DXY strength predicts commodity underperformance over 20 days",
    "When UUP rises >2% in 10 days, DJP underperforms SPY by >3% over subsequent 20 trading days.",
    "Dollar strength increases cost of dollar-denominated commodities for foreign buyers.",
    "UUP, DJP, GLD, USO",
    "Signal: UUP 10d return >2% -> underweight DJP; forward 20d return differential.",
    "Yahoo Finance (UUP, DJP, GLD, USO)",
)
h(
    A,
    AN,
    4,
    "VIX term structure contango (VXV/VIX >1.05) predicts positive SPY returns",
    "VXV/VIX >1.05 (contango) periods show above-average SPY returns over next 10-20 days vs backwardation.",
    "Contango indicates calm near-term expectations — historically bullish for equities.",
    "^VIX, ^VXV, SPY",
    "Split by VXV/VIX >1.05 vs <=1.05; measure forward 10/20d SPY returns; t-test.",
    "Yahoo Finance (^VIX, ^VXV, SPY)",
)
h(
    A,
    AN,
    5,
    "GLD/SPY ratio extreme signals mean-reversion trade opportunity",
    "GLD/SPY ratio >2 sigma above 252d mean -> long SPY/short GLD produces positive 20-30d returns.",
    "Extreme safe-haven demand is self-limiting; stress episodes revert when shock passes.",
    "GLD, SPY",
    "252d z-score on GLD/SPY ratio; signal: z>2 -> long SPY; 20d/30d return measurement.",
    "Yahoo Finance (GLD, SPY)",
)
h(
    A,
    AN,
    6,
    "EEM vs SPY 60d momentum divergence signals rotation opportunity",
    "EEM 60d return exceeding SPY by >10% predicts EEM underperformance over next 30 days with >60% frequency.",
    "EM/DM return differentials exhibit momentum-then-reversion cycles from capital flow dynamics.",
    "EEM, SPY, EFA",
    "60d return differential; signal: EEM-SPY spread >10% -> underweight EEM; 5-year backtest.",
    "Yahoo Finance (EEM, SPY, EFA)",
)
h(
    A,
    AN,
    7,
    "Cross-asset rotation by 60d Sharpe (TLT/GLD/SPY/BTC) generates alpha",
    "Monthly-rebalanced portfolio holding top-2 of {SPY,TLT,GLD,BTC-USD} by 60d Sharpe beats 60/40 benchmark with Sharpe >0.8.",
    "Risk-adjusted momentum persists across asset classes; rotating captures regime momentum.",
    "SPY, TLT, GLD, BTC-USD",
    "Rank by 60d Sharpe monthly; hold equal-weight top-2; compare to 60/40 benchmark.",
    "Yahoo Finance (SPY, TLT, GLD, BTC-USD)",
    "needs: asset_rotation strategy class",
    1,
)
h(
    A,
    AN,
    8,
    "SPY-TLT 5d correlation flip signals volatility spike within 5 days",
    "SPY-TLT 5d rolling correlation crossing from positive to negative predicts >2x SPY realized vol increase within 5 days in >65% of cases.",
    "Correlation turning negative = flight-to-safety breakdown; hallmark of liquidity stress.",
    "SPY, TLT, ^VIX",
    "Detect sign changes in 5d rolling corr; measure next 5d VIX change and SPY realized vol; precision/recall.",
    "Yahoo Finance (SPY, TLT, ^VIX)",
    "needs: correlation_regime strategy class",
    1,
)
h(
    A,
    AN,
    9,
    "Global macro breadth composite predicts SPY 20d return",
    "20d momentum z-score composite of EFA/EEM/TLT/GLD achieves Spearman rho >0.20 with SPY forward 20d return.",
    "Global breadth reflects macro environment; multiple trending asset classes signal positive SPY follow-through.",
    "EFA, EEM, TLT, GLD, SPY",
    "20d z-score for each asset; average; regress on SPY forward 20d return; OOS R2.",
    "Yahoo Finance (EFA, EEM, TLT, GLD, SPY)",
)

print("Mandate A done (9)")

# MANDATE B: Text/NLP (10, P3, blocked on INFRA-B)
B = "B"
BN = "Text / NLP / Language Signals"
h(
    B,
    BN,
    1,
    "10-K readability inversely predicts 12m stock return",
    "Stocks with above-median FK grade 10-K filings underperform by >3% annually over next 12 months.",
    "Complex filings obscure bad news; simpler filings signal management transparency.",
    "S&P 500 components",
    "FK grade on MD&A; median-split; 12m forward return differential; Fama-French adjusted.",
    "EDGAR full-text search API",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    2,
    "CEO letter I/we ratio predicts 12m underperformance",
    "Companies with CEO letter I/we ratio >2 underperform I/we ratio <1 companies by >4% over next 12 months.",
    "CEO overconfidence (high I usage) correlates with empire-building and excessive risk-taking.",
    "S&P 500 components",
    "Parse CEO letters; compute I/(I+we) ratio; split by ratio quartile; measure 12m return.",
    "EDGAR annual report text",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    3,
    "Forward-looking sentence density predicts positive earnings surprise",
    "Companies with >30% forward-looking sentences in MD&A beat earnings estimates in >60% of next quarters.",
    "Management signals confidence via forward-looking language; correlates with future beats.",
    "S&P 500 components",
    "Classify sentences via Claude API; compute density; correlate with Yahoo Finance earnings surprise.",
    "EDGAR 10-K text, Yahoo Finance",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    4,
    "Consecutive 10-K semantic similarity predicts negative returns",
    "Companies with >95% cosine similarity between consecutive 10-K filings underperform by >5% over 12 months.",
    "High similarity signals management inattention or hiding material changes (boilerplate copy-paste).",
    "S&P 500 components",
    "Embed MD&A via sentence-transformers; cosine similarity YoY; correlate with returns.",
    "EDGAR 10-K text",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    5,
    "Reddit WSB mention spike predicts mean-reverting -5% returns over 20 days",
    "Stocks with WSB daily mention volume >3 sigma above 30d average return -5% to -10% over subsequent 20 trading days.",
    "Retail attention-driven overpricing reverses as momentum traders exit.",
    "Meme stock universe (GME, AMC analogs)",
    "Track daily mention counts via Reddit API; signal: >3sigma -> short 20d.",
    "Reddit API, Yahoo Finance",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    6,
    "FOMC hedging language density predicts 5d TLT volatility",
    "FOMC statements with hedging density >2 sigma predict 5-day TLT realized vol increase >50% with >70% accuracy.",
    "Fed communication uncertainty transmits directly to rate expectations and bond vol.",
    "TLT, ^TNX",
    "Parse FOMC statements; count hedging words per sentence; signal: high hedging -> long TLT vol proxy.",
    "Federal Reserve website, Yahoo Finance (TLT)",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    7,
    "10-K risk factor word count growth predicts downside returns",
    "Companies adding >500 words to risk factors section YoY underperform by >3% over next 12 months.",
    "Expanding risk disclosures signal management awareness of emerging threats not yet priced.",
    "S&P 500 components",
    "Extract risk factors section length from EDGAR; compute delta; regress on forward returns.",
    "EDGAR 10-K text",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    8,
    "MD&A sentiment shift YoY predicts earnings direction",
    "Companies with positive YoY MD&A sentiment shift beat earnings estimates in >65% of subsequent quarters.",
    "Management tone shift anticipates operational improvements before they appear in financials.",
    "S&P 500 components",
    "Score MD&A sentiment via Claude API; compute YoY delta; correlate with next-quarter EPS beat/miss.",
    "EDGAR 10-K text, Yahoo Finance",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    9,
    "Earnings call Q&A uncertainty language predicts sell-side miss",
    "Earnings calls where CEO/CFO use >3 hedging words per Q&A answer predict consensus miss in >60% of subsequent quarters.",
    "Q&A exposes real uncertainty; management hedging signals lack of visibility.",
    "S&P 500 components",
    "Fetch earnings call transcripts; count hedging per answer; correlate with next quarter EPS miss.",
    "SEC EDGAR 8-K, Yahoo Finance",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)
h(
    B,
    BN,
    10,
    "SEC comment letter frequency predicts audit risk premium",
    "Companies receiving >2 comment letters in 2 years have -6% subsequent 12m returns vs peers.",
    "Comment letters signal accounting quality concerns before formal enforcement.",
    "S&P 500 components",
    "Query EDGAR EFTS for comment letter frequency by CIK; merge with returns.",
    "EDGAR EFTS API, Yahoo Finance",
    "needs: NLP pipeline (blocked on INFRA-B)",
    3,
)

print("Mandate B done (10)")

# MANDATE C: Volatility Signals (8)
C = "C"
CN = "Volatility Signals"
h(
    C,
    CN,
    1,
    "VoV (VIX 30d std dev) precedes fat tail equity return events",
    "VIX 30d rolling std dev crossing above its 80th percentile predicts SPY daily return >2sigma event within 10 days with >70% precision.",
    "Volatility-of-volatility measures the uncertainty in vol regime itself; VoV spikes precede crashes and large moves.",
    "^VIX, SPY",
    "Compute VIX 30d std dev daily; 80th pct threshold; precision/recall on forward 10d fat tails.",
    "Yahoo Finance (^VIX, SPY)",
    priority := 1,
)
h(
    C,
    CN,
    2,
    "SPX realized vs implied vol gap signals mean-reversion trade",
    "When SPX 5d realized vol exceeds VIX by >5 points, subsequent 10d SPY returns are positive with >60% frequency.",
    "Implied vol overprices realized vol on average; premium spikes revert as fear subsides.",
    "^VIX, SPY",
    "Compute 5d realized vol on SPY returns; compute VIX-realized spread; signal: spread >5 -> long SPY.",
    "Yahoo Finance (^VIX, SPY)",
)
h(
    C,
    CN,
    3,
    "VVIX/VIX ratio extreme (>1.5) signals imminent equity decline",
    "VVIX/VIX ratio above 1.5 predicts SPY negative 5d returns with >65% accuracy.",
    "VVIX measures the price of VIX options; extreme VVIX/VIX signals demand for crash protection exceeds base vol.",
    "^VVIX, ^VIX, SPY",
    "Compute VVIX/VIX ratio daily; signal: >1.5 -> defensive posture; forward 5d SPY return.",
    "Yahoo Finance (^VVIX, ^VIX, SPY)",
)
h(
    C,
    CN,
    4,
    "VIX futures curve steepness generates short-vol carry return",
    "Long SPY weighted by inverse VIX contango slope produces Sharpe >0.6 over 5-year backtest.",
    "Steep VIX term structure creates carry; holding equities during high carry is systematically rewarded.",
    "^VIX, ^VXV, SPY",
    "Compute VXV/VIX slope; scale SPY position by inverse slope; backtest vs equal-weight SPY.",
    "Yahoo Finance (^VIX, ^VXV, SPY)",
)
h(
    C,
    CN,
    5,
    "Single-day VIX spike >20pct predicts 5d SPY bounce",
    "When VIX rises >20% in one day, SPY produces >1% positive return over next 5 days with >70% frequency.",
    "Single-day vol spikes are overshoots driven by forced sellers; mean reversion follows as structural buyers emerge.",
    "^VIX, SPY",
    "Event study: detect VIX 1d change >20%; measure forward 5d SPY return; win rate and avg return.",
    "Yahoo Finance (^VIX, SPY)",
)
h(
    C,
    CN,
    6,
    "VIX weekday effect (Monday premium) creates day-of-week signal",
    "VIX closes higher on Mondays than Fridays in >60% of weeks; long SPY Friday close to Monday close beats buy-and-hold.",
    "Weekend uncertainty premium inflates VIX on Mondays; this premium decays through the week.",
    "^VIX, SPY",
    "Classify VIX changes by day-of-week; test Friday close vs Monday open SPY strategy; Sharpe.",
    "Yahoo Finance (^VIX, SPY)",
)
h(
    C,
    CN,
    7,
    "High VIX regime (>25) predicts equity momentum strategy failure",
    "Momentum strategy (12-1 month return signal) Sharpe drops below 0 when VIX >25 regime; regime-conditioning improves Sharpe.",
    "High vol regimes create mean-reversion not momentum; momentum crashes historically occur during vol spikes.",
    "^VIX, SPY, momentum ETF proxies",
    "Split backtest by VIX regime (>25 vs <25); compare momentum strategy Sharpe by regime.",
    "Yahoo Finance (^VIX, SPY, MTUM)",
)
h(
    C,
    CN,
    8,
    "SKEW index >135 predicts left tail event within 30 days",
    "CBOE SKEW index readings >135 are followed by SPY drawdowns >3% within 30 days with >60% frequency.",
    "SKEW measures demand for OTM puts; extreme SKEW = smart money buying crash protection.",
    "^SKEW, SPY",
    "Event study: SKEW >135 -> forward 30d SPY return distribution; compare to base rate.",
    "Yahoo Finance (^SKEW, SPY)",
    priority := 1,
)

print("Mandate C done (8)")

# MANDATE D: Crypto Signals (9)
D = "D"
DN = "Crypto / Digital Asset Signals"
h(
    D,
    DN,
    1,
    "BTC 200d SMA filter improves Sharpe vs buy-and-hold",
    "BTC-USD held only when price >200d SMA achieves Sharpe >0.8 and max drawdown <40% vs buy-and-hold over 5 years.",
    "Trend filter keeps investors in during bull runs and exits during bear markets; reduces catastrophic drawdowns.",
    "BTC-USD",
    "Backtest BTC-USD: long when >200d SMA, cash otherwise; vs buy-and-hold Sharpe and DD.",
    "Yahoo Finance (BTC-USD)",
)
h(
    D,
    DN,
    2,
    "ETH/BTC spread mean-reversion generates positive alpha",
    "When ETH/BTC ratio deviates >1.5 sigma from 60d mean, a mean-reversion trade produces Sharpe >0.5 over 3-year backtest.",
    "ETH and BTC are highly correlated; ratio deviations from mean are driven by temporary sector rotations that revert.",
    "ETH-USD, BTC-USD",
    "Compute 60d rolling mean/std of ETH/BTC ratio; signal: z>1.5 -> sell ETH/buy BTC; backtest.",
    "Yahoo Finance (ETH-USD, BTC-USD)",
    "needs: pairs_ratio strategy class",
)
h(
    D,
    DN,
    3,
    "Bitcoin halving event predicts 12m post-halving positive return",
    "BTC-USD 12m returns following halving events are positive in 100% of historical cases (2012, 2016, 2020, 2024).",
    "Halving reduces new supply issuance by 50%; supply shock combined with stable/growing demand creates upward price pressure.",
    "BTC-USD",
    "Event study: measure 12m return after each halving date; compute mean/median return; compare to random 12m periods.",
    "Yahoo Finance (BTC-USD), Bitcoin halving date list",
    "needs: event_study strategy class",
    1,
)
h(
    D,
    DN,
    4,
    "BTC hashrate recovery speed post-difficulty adjustment predicts 14d direction",
    "Faster hashrate recovery after difficulty adjustments (recovery within 7 days) predicts positive BTC-USD 14d returns with >65% accuracy.",
    "Hashrate recovery signals miner confidence and network health; slow recovery indicates miner capitulation and bearish conditions.",
    "BTC-USD, blockchain.com hashrate data",
    "Track hashrate vs difficulty; compute recovery days; correlate with 14d forward BTC return.",
    "blockchain.com API (free), Yahoo Finance (BTC-USD)",
)
h(
    D,
    DN,
    5,
    "Stablecoin dominance (USDT%) rise predicts crypto risk-off",
    "When USDT market cap as % of total crypto rises >3% in 7 days, BTC-USD returns over next 14 days are negative with >65% frequency.",
    "Rising stablecoin dominance = capital rotating to cash; equivalent to crypto VIX rising.",
    "BTC-USD, total crypto market cap",
    "USDT% from CoinGecko; signal: +3% in 7d -> underweight BTC; forward 14d return.",
    "CoinGecko API (free), Yahoo Finance (BTC-USD)",
)
h(
    D,
    DN,
    6,
    "BTC perpetual funding rate extreme predicts mean-reversion",
    "BTC funding rate >0.10% (extreme long bias) predicts -5% to -10% BTC return over next 7 days; negative funding predicts bounce.",
    "Funding rate = crowding in perp longs; extreme crowding creates squeeze conditions that revert.",
    "BTC-USD",
    "Fetch BTC perp funding rate from public exchanges (Binance/Bybit API); signal extremes -> counter-trend.",
    "Binance/Bybit public API, Yahoo Finance (BTC-USD)",
)
h(
    D,
    DN,
    7,
    "ETH/BTC ratio Bollinger Band mean-reversion generates alpha",
    "ETH/BTC ratio >2 sigma Bollinger Band reversal signal achieves Sharpe >0.5 over 3-year backtest.",
    "ETH and BTC high correlation makes ratio mean-reversion statistically robust across market regimes.",
    "ETH-USD, BTC-USD",
    "20d Bollinger Band on ETH/BTC ratio; signal: >2sigma -> short ETH/long BTC or vice versa.",
    "Yahoo Finance (ETH-USD, BTC-USD)",
    "needs: pairs_ratio strategy class",
    1,
)
h(
    D,
    DN,
    8,
    "Crypto Fear and Greed index extreme predicts contrarian returns",
    "Crypto Fear & Greed <20 (extreme fear) predicts >10% BTC return over next 30 days with >70% frequency; >80 (greed) predicts <0%.",
    "Sentiment extremes represent crowded positioning; contrarian trades against extremes historically profitable.",
    "BTC-USD",
    "Fetch Fear & Greed daily from alternative.me API; signal: <20 -> long BTC, >80 -> short/flat.",
    "alternative.me API (free), Yahoo Finance (BTC-USD)",
    priority := 1,
)
h(
    D,
    DN,
    9,
    "BTC MVRV ratio identifies long-term over/undervaluation",
    "BTC MVRV ratio >3.5 predicts negative 12m return; MVRV <1 predicts positive 12m return with >80% historical accuracy.",
    "MVRV compares market cap to realized cap; extreme overvaluation (MVRV>3.5) precedes major bear markets.",
    "BTC-USD",
    "Fetch MVRV from Glassnode or CoinMetrics public data; signal: >3.5 -> underweight, <1 -> overweight.",
    "Glassnode/CoinMetrics public API, Yahoo Finance (BTC-USD)",
)

print("Mandate D done (9)")

# MANDATE E: Macro / Economic Indicators (9)
E = "E"
EN = "Macro / Economic Indicators"
h(
    E,
    EN,
    1,
    "2s10s yield curve inversion predicts SPY underperformance 6-18 months out",
    "2s10s inversion (DGS10-DGS2 <0) predicts SPY 12m forward return <5% with >70% historical accuracy since 1980.",
    "Yield curve inversion reflects Fed overtightening and impending growth slowdown; equities lag by 6-18 months.",
    "SPY, TLT, IEF (FRED: DGS2, DGS10)",
    "Fetch FRED DGS2/DGS10 daily; signal: DGS10-DGS2 <0 -> underweight SPY; forward 6/12/18m return.",
    "FRED API (free): DGS2, DGS10; Yahoo Finance (SPY, TLT)",
    "needs: yield_curve_regime strategy class",
    1,
)
h(
    E,
    EN,
    2,
    "ISM PMI above/below 50 drives cyclical equity rotation",
    "ISM Manufacturing PMI rising above 50 predicts XLI (industrials) outperformance vs XLU (utilities) over next 60 days with >65% frequency.",
    "PMI >50 = expansion; cyclical sectors (XLI) outperform defensives (XLU) in expansion; reverse in contraction.",
    "XLI, XLU, SPY (FRED: ISM_PMI)",
    "Fetch ISM PMI from FRED; signal: >50 rising -> overweight XLI/underweight XLU.",
    "FRED API (ISM), Yahoo Finance (XLI, XLU)",
)
h(
    E,
    EN,
    3,
    "Unemployment rate YoY rise predicts XLY underperformance",
    "When UNRATE rises >0.5% YoY, XLY (consumer discretionary) underperforms XLP (consumer staples) by >5% over next 6 months.",
    "Rising unemployment = falling consumer confidence and spending; discretionary spending cut first.",
    "XLY, XLP, SPY (FRED: UNRATE)",
    "FRED UNRATE YoY change; signal: +0.5% -> underweight XLY/overweight XLP.",
    "FRED API (UNRATE), Yahoo Finance (XLY, XLP)",
)
h(
    E,
    EN,
    4,
    "CPI surprise predicts TIPS vs nominal Treasury positioning",
    "When CPI YoY > consensus estimate by >0.3%, TIPS (TIP) outperforms IEF over next 20 days with >65% frequency.",
    "Inflation surprise shifts real rate expectations immediately; TIPS (inflation-linked) outperform nominal.",
    "TIP, IEF, TLT (FRED/BLS: CPI)",
    "Fetch CPI releases; compute vs consensus (FRED); signal: beat -> long TIP/short IEF.",
    "FRED API (CPIAUCSL), Yahoo Finance (TIP, IEF, TLT)",
)
h(
    E,
    EN,
    5,
    "Fed funds rate hike cycle peak predicts regional bank rally",
    "Within 60 days of last Fed rate hike in a cycle, KRE (regional banks) outperforms SPY by >5% over next 6 months.",
    "Rate cycle peak removes policy uncertainty; NIM (net interest margin) benefits locked in; hike risk removed.",
    "KRE, XLF, SPY (FRED: FEDFUNDS)",
    "Identify Fed rate cycle peaks via FRED FEDFUNDS; event study: forward 6m KRE vs SPY return.",
    "FRED API (FEDFUNDS), Yahoo Finance (KRE, XLF, SPY)",
)
h(
    E,
    EN,
    6,
    "PCE deflator trend drives growth vs value rotation",
    "When PCE YoY rises above 3% and accelerating, IVE (value) outperforms IVW (growth) by >3% over next 60 days.",
    "Rising inflation hurts long-duration growth stocks (DCF denominator effect) more than value stocks.",
    "IVW, IVE, SPY (FRED: PCEPI)",
    "FRED PCEPI YoY rate; signal: >3% and rising -> overweight IVE/underweight IVW.",
    "FRED API (PCEPI), Yahoo Finance (IVW, IVE)",
)
h(
    E,
    EN,
    7,
    "Consumer confidence index level drives small-cap IWM momentum",
    "UMCSENT (U Michigan sentiment) above 85 predicts IWM outperforms SPY by >2% over next 30 days.",
    "High consumer confidence flows to small/domestic companies (IWM) which are more economically sensitive.",
    "IWM, SPY (FRED: UMCSENT)",
    "FRED UMCSENT level; signal: >85 -> overweight IWM; 30d forward IWM vs SPY return.",
    "FRED API (UMCSENT), Yahoo Finance (IWM, SPY)",
)
h(
    E,
    EN,
    8,
    "M2 money supply growth >8% YoY drives commodity momentum",
    "When M2 YoY growth >8%, DJP (commodities) produces above-average 60d forward return vs SPY.",
    "M2 growth represents monetary expansion; commodities are an inflation hedge that benefits from excess liquidity.",
    "DJP, GLD, USO, SPY (FRED: M2SL)",
    "FRED M2SL YoY; signal: >8% -> overweight DJP; compare 60d returns.",
    "FRED API (M2SL), Yahoo Finance (DJP, GLD)",
)
h(
    E,
    EN,
    9,
    "Yield curve re-steepening post-inversion drives financial sector rotation",
    "When 2s10s re-steepens from negative to +50bps, XLF outperforms SPY by >5% over next 60 days.",
    "Steeper yield curve = wider NIM for banks; financial sector profits expand as curve normalizes.",
    "XLF, KRE, SPY (FRED: DGS2, DGS10)",
    "FRED DGS10-DGS2; detect re-steepening events; event study on forward 60d XLF vs SPY.",
    "FRED API (DGS2, DGS10), Yahoo Finance (XLF, KRE)",
    "needs: yield_curve_regime strategy class",
)

print("Mandate E done (9)")

# MANDATE F: Seasonal / Calendar Effects (9)
F = "F"
FN = "Seasonal / Calendar Effects"
h(
    F,
    FN,
    1,
    "January small-cap effect generates IWM premium vs SPY",
    "IWM outperforms SPY in January in >65% of years; January return in IWM >2% in years when Dec IWM underperforms.",
    "Tax-loss selling in December creates oversold small-caps; January buying pressure reverses the discount.",
    "IWM, SPY",
    "Compute January monthly return for IWM and SPY each year; t-test for IWM premium; conditional on Dec underperformance.",
    "Yahoo Finance (IWM, SPY)",
    "needs: calendar_event strategy class",
)
h(
    F,
    FN,
    2,
    "Sell in May SPY underperformance strategy beats buy-and-hold",
    "Holding SPY only Nov-Apr and cash May-Oct produces higher Sharpe than buy-and-hold over 20 years.",
    "Summer months historically have lower equity returns; Halloween indicator is one of the most replicated calendar anomalies.",
    "SPY",
    "Monthly return comparison May-Oct vs Nov-Apr; Sharpe of seasonal vs buy-and-hold.",
    "Yahoo Finance (SPY)",
    "needs: calendar_event strategy class",
)
h(
    F,
    FN,
    3,
    "Turn-of-month effect captures >40% of annual SPY returns in 2 trading days",
    "Last trading day + first trading day of month combined return accounts for >40% of SPY annual return.",
    "Month-end institutional flows (pension/401k contributions), window dressing, and rebalancing create predictable demand.",
    "SPY",
    "Compute last-day + first-day return each month vs all-other-days; proportion of annual return; Sharpe decomposition.",
    "Yahoo Finance (SPY)",
    "needs: calendar_event strategy class",
    1,
)
h(
    F,
    FN,
    4,
    "Pre-FOMC TLT drift generates positive return in 3 days before meeting",
    "TLT returns positive in 2 of 3 days before FOMC meeting dates, producing an annualized 3-4% excess return.",
    "FOMC anticipation drives rates lower (bond prices higher) as the market prices in policy clarity reduction of uncertainty.",
    "TLT (FOMC calendar from Federal Reserve)",
    "Collect FOMC meeting dates; compute TLT 3-day pre-FOMC return; event study; compare to random 3-day windows.",
    "Yahoo Finance (TLT), Federal Reserve FOMC calendar",
    "needs: calendar_event strategy class",
    1,
)
h(
    F,
    FN,
    5,
    "OPEX volatility crush creates short-vol equity opportunity",
    "SPY realized vol during OPEX week (3rd Friday of month) is lower than non-OPEX weeks; short-vol strategy profitable.",
    "Options expiry reduces gamma exposure; market makers no longer need to hedge dynamically, reducing vol.",
    "SPY, ^VIX",
    "Classify OPEX vs non-OPEX weeks; compare realized vol; measure implied vs realized vol spread in OPEX week.",
    "Yahoo Finance (SPY, ^VIX)",
    "needs: calendar_event strategy class",
)
h(
    F,
    FN,
    6,
    "CPI release day creates directional TIPS drift",
    "TIP (TIPS ETF) returns positive on CPI release days in >60% of months over past 10 years.",
    "CPI day creates price discovery for inflation expectations; TIPS directly repriced on release day.",
    "TIP, IEF (BLS CPI release calendar)",
    "Collect CPI release dates; compute TIP return on release day; event study vs random days.",
    "Yahoo Finance (TIP, IEF), BLS CPI release calendar",
    "needs: calendar_event strategy class",
)
h(
    F,
    FN,
    7,
    "Quarter-end window dressing drives large-cap outperformance",
    "SPY outperforms IWM in last 5 trading days of each quarter in >60% of quarters.",
    "Institutional managers buy large-cap winners for quarterly reporting; this inflates large-cap prices temporarily.",
    "SPY, IWM",
    "Compute last-5-day return each quarter for SPY vs IWM; t-test for differential.",
    "Yahoo Finance (SPY, IWM)",
    "needs: calendar_event strategy class",
)
h(
    F,
    FN,
    8,
    "Summer doldrums (Jun-Aug) create lower-volatility environment",
    "SPY realized volatility in June-August is <80% of non-summer months in >65% of years.",
    "Reduced institutional participation in summer creates thinner markets but also less forced selling.",
    "SPY, ^VIX",
    "Compute monthly realized vol; compare Jun-Aug vs Sep-May; statistical test; VIX seasonality.",
    "Yahoo Finance (SPY, ^VIX)",
    "needs: calendar_event strategy class",
)
h(
    F,
    FN,
    9,
    "December tax-loss selling reversal creates January bounce",
    "Stocks with worst Dec returns (bottom decile) produce above-average January returns (>5% on average).",
    "Forced tax-loss selling depresses Dec prices; January rebound as selling pressure lifts.",
    "IWM, SPY (individual small-cap stocks proxy)",
    "Rank stocks by Dec return; bottom decile vs top decile Jan return; event study.",
    "Yahoo Finance (IWM components or small-cap ETFs)",
    "needs: calendar_event strategy class",
)

print("Mandate F done (9)")

# MANDATE G: ETF / Structural Signals (8)
G = "G"
GN = "ETF / Structural Signals"
h(
    G,
    GN,
    1,
    "ETF NAV premium/discount mean-reversion generates alpha",
    "SPY NAV premium >0.1% predicts -0.2% return over next 3 days; discount <-0.1% predicts +0.2% return.",
    "ETF arbitrage mechanism keeps NAV and price aligned; deviations are short-lived and mechanically traded by APs.",
    "SPY, GLD (ETF with published NAV)",
    "Compute SPY intraday NAV vs price spread daily; signal: deviation -> mean-reversion; 3d forward return.",
    "Yahoo Finance (SPY), ETF NAV data from fund website",
    "needs: event_study strategy class",
)
h(
    G,
    GN,
    2,
    "ETF AUM growth rate signals sector momentum continuation",
    "Sectors with 3m AUM growth >10% in their primary ETF outperform SPY by >2% over next 60 days.",
    "Flows into sector ETFs reflect institutional momentum; growing AUM = sustained demand.",
    "XLK, XLF, XLE, XLY, XLV, SPY",
    "Track ETF AUM from Yahoo Finance or ETF.com; signal: 3m AUM growth >10% -> overweight sector.",
    "Yahoo Finance (sector ETF prices as proxy), ETF.com",
)
h(
    G,
    GN,
    3,
    "Pre-OPEX pin risk creates predictable SPY price clustering",
    "SPY closes within 0.5% of round-number strikes on OPEX Fridays in >60% of monthly expiries.",
    "Max pain / pin risk: market makers hedge to minimize payout; this clusters prices near large OI strikes.",
    "SPY",
    "Collect monthly OPEX dates; compute distance of SPY close from nearest $5 strike; clustering statistic.",
    "Yahoo Finance (SPY), CBOE options data",
    "needs: calendar_event strategy class",
)
h(
    G,
    GN,
    4,
    "Leveraged ETF daily rebalancing creates predictable end-of-day momentum",
    "TQQQ (3x QQQ) rebalances at close; QQQ close-to-open return after large TQQQ flows is positive in >60% of cases.",
    "Leveraged ETF daily rebalancing requires buying winners and selling losers at close; creates follow-through momentum.",
    "TQQQ, QQQ, SPY",
    "Compute TQQQ daily return magnitude; correlate with QQQ close-to-next-open return; event study on large TQQQ moves.",
    "Yahoo Finance (TQQQ, QQQ)",
)
h(
    G,
    GN,
    5,
    "Index reconstitution buying pressure creates pre-announcement premium",
    "Stocks added to S&P 500 produce >3% excess return between announcement and effective date.",
    "Forced buying by passive index funds creates predictable demand; well-documented but implementable with free data.",
    "SPY components",
    "Collect S&P 500 reconstitution announcements from news/Wikipedia; event study on added/removed stocks.",
    "Yahoo Finance (stock prices), S&P press releases",
    "needs: event_study strategy class",
)
h(
    G,
    GN,
    6,
    "ETF fund flow divergence from underlying predicts reversal",
    "When SPY flows are negative (redemptions) but SPX futures are positive, SPY closes negative next day in >65% of cases.",
    "Institutional flow divergence signals informed selling into strength; a reliable short-term reversal signal.",
    "SPY, QQQ",
    "Approximate SPY flows via share count changes or volume/price proxy; divergence detection; 1d forward return.",
    "Yahoo Finance (SPY volume proxy)",
)
h(
    G,
    GN,
    7,
    "Closed-end fund discount to NAV mean-reversion generates alpha",
    "CEFs trading >15% discount to NAV produce >8% 12m excess return vs comparable open-end funds.",
    "CEF discounts reflect short-term investor sentiment; fundamental value eventually reasserts via buybacks or tender offers.",
    "Various CEF tickers (e.g., ADX, MSD)",
    "Screen for CEF discounts >15% from CEFConnect; measure forward 12m price return vs NAV.",
    "CEFConnect.com (free), Yahoo Finance",
    "needs: event_study strategy class",
    2,
)
h(
    G,
    GN,
    8,
    "Passive vs active fund flow ratio signals market microstructure regime",
    "When passive fund flows >70% of total equity fund flows, cross-sectional return dispersion drops; factor strategies underperform.",
    "Passive dominance reduces price discovery; all stocks move together, reducing alpha from selection signals.",
    "SPY, IWB, active fund proxies",
    "ICI fund flow data by passive/active; correlate with cross-sectional return dispersion; factor strategy Sharpe by regime.",
    "ICI.org (free fund flow data), Yahoo Finance",
    2,
)

print("Mandate G done (8)")

# MANDATE H: Lead-Lag / Cross-Market Signals (9)
H = "H"
HN = "Lead-Lag / Cross-Market Signals"
h(
    H,
    HN,
    1,
    "XLF (financials) leads SPY by 2-3 trading days",
    "XLF 2-day lagged return has positive Granger causality for SPY next-day return; signal achieves IC >0.05.",
    "Financial sector (XLF) is a leading indicator of broad market because credit conditions and bank health drive economy.",
    "XLF, SPY",
    "Granger causality XLF->SPY at 2/3d lags; construct signal: XLF 2d return -> SPY position; IC measurement.",
    "Yahoo Finance (XLF, SPY)",
    "needs: lead_lag strategy class",
    1,
)
h(
    H,
    HN,
    2,
    "HYG (high yield) leads SPY by 1-2 trading days",
    "HYG 1-day lagged return has statistically significant positive correlation with SPY next-day return (p<0.05).",
    "Credit leads equity in risk-on/risk-off transitions; HYG is a real-time credit barometer.",
    "HYG, LQD, SPY",
    "OLS regression: SPY_t ~ HYG_{t-1}; t-statistic; Granger test; signal implementation and backtest.",
    "Yahoo Finance (HYG, LQD, SPY)",
    "needs: lead_lag strategy class",
    1,
)
h(
    H,
    HN,
    3,
    "TLT leads SPY with inverse sign by 3 days",
    "TLT 3-day lagged return is negatively correlated with SPY return over next 3 days; inverse lead-lag relationship.",
    "Treasury safe-haven demand anticipates equity stress; rising TLT signals flight-to-safety that precedes equity weakness.",
    "TLT, IEF, SPY",
    "OLS: SPY_t ~ -TLT_{t-3}; Granger test; signal: TLT up 3d -> underweight SPY; backtest.",
    "Yahoo Finance (TLT, IEF, SPY)",
    "needs: lead_lag strategy class",
)
h(
    H,
    HN,
    4,
    "SOXX (semiconductors) leads QQQ by 5 trading days",
    "SOXX 5-day lagged return has IC >0.05 for QQQ forward 5-day return; semiconductors are a leading indicator for tech.",
    "Semiconductor demand reflects corporate tech spending 2-3 quarters ahead; chip orders lead device production.",
    "SOXX, QQQ, SMH",
    "Granger test SOXX->QQQ at 5d lag; signal: SOXX 5d return -> QQQ position; IC calculation.",
    "Yahoo Finance (SOXX, QQQ, SMH)",
    "needs: lead_lag strategy class",
)
h(
    H,
    HN,
    5,
    "USO (oil) leads XLE (energy sector) by 2 trading days",
    "USO 2-day lagged return explains >10% of XLE next-2d return variance; energy sector lags oil prices.",
    "Oil is the primary input cost and revenue driver for energy companies; stock market reprices with a lag.",
    "USO, XLE, XOP",
    "OLS: XLE_t ~ USO_{t-2}; R-squared; signal: USO 2d return -> XLE position; 5-year backtest.",
    "Yahoo Finance (USO, XLE, XOP)",
    "needs: lead_lag strategy class",
)
h(
    H,
    HN,
    6,
    "Copper ETF (CPER) leads XLB (materials) by 5 trading days",
    "CPER 5-day lagged return has IC >0.05 for XLB forward 5-day return.",
    "Copper is the primary industrial metal; materials sector earnings are driven by copper price with production lag.",
    "CPER, XLB",
    "Granger test CPER->XLB at 5d lag; IC measurement; backtest materials rotation signal.",
    "Yahoo Finance (CPER, XLB)",
    "needs: lead_lag strategy class",
)
h(
    H,
    HN,
    7,
    "China A-shares (ASHR) leads EEM by 3 trading days",
    "ASHR 3-day lagged return has positive correlation with EEM 3d forward return (p<0.05).",
    "China represents ~30% of EEM; A-share market movements lead EM broad index as capital flows propagate.",
    "ASHR, EEM, FXI",
    "OLS: EEM_t ~ ASHR_{t-3}; correlation and Granger test; signal: ASHR 3d return -> EEM position.",
    "Yahoo Finance (ASHR, EEM, FXI)",
    "needs: lead_lag strategy class",
)
h(
    H,
    HN,
    8,
    "Bitcoin leads risk-on assets (EEM, HYG) by 2-3 days",
    "BTC-USD 2d lagged return has positive correlation with EEM next-2d return (rho >0.10 at p<0.05).",
    "BTC as risk barometer: retail/institutional risk appetite in crypto propagates to other risk assets with a lag.",
    "BTC-USD, EEM, HYG",
    "Granger test BTC->EEM and BTC->HYG at 2/3d lags; signal implementation; Sharpe comparison.",
    "Yahoo Finance (BTC-USD, EEM, HYG)",
    "needs: lead_lag strategy class pairs_ratio",
    1,
)
h(
    H,
    HN,
    9,
    "XLU (utilities) inversely leads SPY by 2 trading days",
    "XLU 2-day lagged return is negatively correlated with SPY 2d forward return; utilities as early risk-off detector.",
    "Utilities are defensive safe havens; money rotating into XLU signals institutional de-risking that precedes broader equity weakness.",
    "XLU, SPY",
    "OLS: SPY_t ~ -XLU_{t-2}; Granger test; signal: XLU up 2d -> underweight SPY.",
    "Yahoo Finance (XLU, SPY)",
    "needs: lead_lag strategy class",
)

print("Mandate H done (9)")
print("Script part 1 complete: Mandates A-H (73 beads)")

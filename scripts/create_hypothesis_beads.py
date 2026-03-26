#!/usr/bin/env python3
# ruff: noqa: S603, S607, PLR0913, PLW1510, F401, I001, E401
"""Batch create all 133 hypothesis beads + 1 epic + 5 INFRA-B = 139 total."""

import subprocess, sys


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
    )
    out = r.stdout.strip() or r.stderr.strip()
    print(out[:120])
    return out


LIFECYCLE = "/mandate → /hypothesis → /data-contract → /research-spec freeze → /backtest → /robustness → /paper → /promote"


def h(
    mandate_letter,
    mandate_name,
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
    desc = f"""**Mandate {mandate_letter}: {mandate_name}**

**Hypothesis:** {hypothesis}
**Mechanism:** {mechanism}
**Universe:** {universe}
**Measurement:** {measurement}
**Data:** {data}

**Lifecycle path:**
{LIFECYCLE}

**Infrastructure:** {infra}"""
    return bd(f"[{mandate_letter}{num}] {title}", desc, "feature", priority)


# ── EPIC ──────────────────────────────────────────────────────────────────────
bd(
    "[EPIC] Process all 133 hypotheses through full lifecycle",
    "Master tracker for the 133-hypothesis quant lab project. 16 mandates A–P, "
    "each hypothesis goes through mandate→hypothesis→data-contract→research-spec→"
    "backtest→robustness→paper→promote. Target: 20-26 promotions (~20% pass rate).",
    "epic",
    1,
)

# ── INFRA-B ────────────────────────────────────────────────────────────────────
bd(
    "[INFRA-B] Install NLP Python dependencies",
    "pip install textstat sentence-transformers scikit-learn beautifulsoup4 into project venv. "
    "Verify: textstat.flesch_kincaid_grade, SentenceTransformer('all-MiniLM-L6-v2'), "
    "sklearn.decomposition.PCA all import cleanly. Required before any Mandate B hypothesis can run.",
    "task",
    1,
)

bd(
    "[INFRA-B] Build EDGAR 10-K text fetcher",
    "Fetch and cache full 10-K filings from EDGAR full-text search API by ticker+year. "
    "Cache to data/nlp/edgar/{ticker}/{year}.txt. Rate limit 10req/s per SEC guidelines. "
    "Extract MD&A and Risk Factors sections. Required for B1-B4, B7, B8.",
    "task",
    1,
)

bd(
    "[INFRA-B] Build Claude API text classifier for NLP signals",
    "Wrapper around anthropic SDK to classify 10-K/earnings call sentences as: "
    "forward_looking/backward_looking, causal/correlational, I_we_ratio. "
    "Batch sentences to minimize API calls. Cache results. Required for B2, B3, B8, B9.",
    "task",
    1,
)

bd(
    "[INFRA-B] Build FOMC transcript fetcher and hedging language scorer",
    "Fetch Federal Reserve press conference transcripts from federalreserve.gov. "
    "Score hedging language frequency (uncertain, approximately, roughly, might, could). "
    "Output: date + hedging_score float. Required for B6.",
    "task",
    1,
)

bd(
    "[INFRA-B] Integrate NLP signal outputs into backtest engine strategy class",
    "Add nlp_signal strategy class to strategies.py + STRATEGY_REGISTRY. "
    "Reads pre-computed NLP scores from data/nlp/signals/{ticker}.parquet as exogenous features. "
    "Integrates with existing backtest engine via standard indicator interface. Required for all B hypotheses.",
    "task",
    1,
)

print("\n✓ Epic + INFRA-B created (6 beads)")

# ── MANDATE A: Macro / Cross-Asset Signals (9) ────────────────────────────────
h(
    "A",
    "Macro / Cross-Asset Signals",
    1,
    "Cu/Au ratio leads equity drawdowns by 15-25 days",
    "The copper/gold ratio (industrial demand proxy) declining below its 30d SMA predicts SPY drawdowns 15-25 trading days later with >55% directional accuracy.",
    "Copper reflects industrial growth expectations; gold reflects risk-off demand. Cu/Au ratio falling signals global growth deceleration before equities fully reprice.",
    "CPER (copper ETF), GLD, SPY",
    "Granger causality test at 15/20/25d lags; signal: Cu/Au 5d MA < 30d MA → underweight SPY; backtest 5-year SPY return conditioned on signal.",
    "Yahoo Finance (CPER, GLD, SPY daily OHLCV)",
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    2,
    "HYG-IEF credit spread widening predicts equity correction within 10 days",
    "When HYG/IEF spread widens >0.5% in 5 days, SPY produces negative returns over the following 10 trading days with >60% frequency.",
    "Credit markets lead equity markets in risk-off episodes; HY spread is a leading indicator of financial stress.",
    "HYG, IEF, SPY",
    "Compute HYG yield - IEF yield approximation via price ratio; signal: 5d spread change >0.5% → short SPY; event study on forward 10d returns.",
    "Yahoo Finance (HYG, IEF, SPY)",
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    3,
    "DXY strength (UUP proxy) predicts commodity underperformance over 20 days",
    "When UUP rises >2% in 10 days (dollar strength), commodity ETF DJP underperforms SPY by >3% over subsequent 20 trading days.",
    "Dollar strength increases cost of dollar-denominated commodities for foreign buyers, reducing demand and commodity prices.",
    "UUP, DJP, GLD, USO",
    "Signal: UUP 10d return >2% → underweight DJP, GLD, USO; measure forward 20d return differential; OLS regression of DXY on commodity returns.",
    "Yahoo Finance (UUP, DJP, GLD, USO)",
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    4,
    "VIX term structure contango (VXV/VIX >1.05) predicts positive SPY returns",
    "When VXV/VIX ratio exceeds 1.05 (contango), SPY produces above-average returns over the next 10-20 days versus flat/backwardation periods.",
    "VIX term structure contango indicates calm short-term expectations vs elevated long-term uncertainty — historically bullish for equities.",
    "^VIX, ^VXV (CBOE 3m VIX), SPY",
    "Compute VXV/VIX ratio daily; split by >1.05 vs ≤1.05; measure forward 10/20d SPY returns; t-test for significance.",
    "Yahoo Finance (^VIX, ^VXV, SPY)",
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    5,
    "GLD/SPY ratio extreme (>2σ) signals imminent mean-reversion in safe-haven demand",
    "When the GLD/SPY price ratio exceeds its 252d mean by 2+ standard deviations, a mean-reversion trade (long SPY, short GLD) produces positive returns over 20-30 days.",
    "Extreme safe-haven demand is self-limiting; forced de-risking episodes revert when systemic shock passes.",
    "GLD, SPY, TLT",
    "Compute GLD/SPY 252d z-score; signal: z>2 → long SPY/short GLD; measure 20d/30d return; Sharpe of mean-reversion strategy.",
    "Yahoo Finance (GLD, SPY, TLT)",
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    6,
    "EEM vs SPY 60d momentum divergence signals rotation opportunity",
    "When EEM 60d return exceeds SPY by >10%, subsequent 30d EEM underperforms SPY with >60% frequency (mean reversion). When EEM trails SPY by >10%, EEM outperforms subsequently.",
    "EM/DM return differentials exhibit momentum-then-reversion dynamics driven by capital flow cycles.",
    "EEM, SPY, EFA",
    "Compute 60d return differential; signal: EEM-SPY spread >10% → underweight EEM; backtest 5 years; Sharpe comparison.",
    "Yahoo Finance (EEM, SPY, EFA)",
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    7,
    "Cross-asset rotation by 60d Sharpe (TLT/GLD/SPY/BTC) generates positive alpha",
    "A monthly-rebalanced portfolio holding the top-2 of {SPY, TLT, GLD, BTC-USD} ranked by 60d Sharpe produces Sharpe >0.8 and beats 60/40 SPY/TLT benchmark.",
    "Risk-adjusted momentum persists across asset classes; rotating into highest Sharpe assets captures regime momentum while managing drawdown.",
    "SPY, TLT, GLD, BTC-USD",
    "Rank 4 assets by rolling 60d Sharpe monthly; hold equal-weight top-2; rebalance monthly; compare to 60/40 benchmark.",
    "Yahoo Finance (SPY, TLT, GLD, BTC-USD)",
    "needs: asset_rotation strategy class",
    2,
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    8,
    "SPY-TLT 5d rolling correlation flip below zero signals volatility spike within 5 days",
    "When the 5d rolling correlation between SPY and TLT crosses from positive to negative (regime flip), realized SPY volatility increases >2x normal within 5 days in >65% of cases.",
    "SPY-TLT correlation turning negative signals flight-to-safety breakdown — equities and bonds falling together — a hallmark of liquidity stress episodes.",
    "SPY, TLT, ^VIX",
    "Compute 5d rolling correlation SPY/TLT; detect sign changes; measure VIX change and SPY realized vol over next 5 days; precision/recall on vol spike >2x.",
    "Yahoo Finance (SPY, TLT, ^VIX)",
    "needs: correlation_regime strategy class",
    1,
)

h(
    "A",
    "Macro / Cross-Asset Signals",
    9,
    "Global macro breadth composite (EFA/EEM/TLT/GLD momentum) predicts SPY 20d return",
    "A composite signal averaging the 20d momentum z-scores of EFA, EEM, TLT, and GLD produces Spearman rank correlation >0.20 with SPY forward 20d return.",
    "Global market breadth reflects the macro environment; when multiple asset classes are trending positively, SPY tends to follow.",
    "EFA, EEM, TLT, GLD, SPY",
    "Compute 20d z-score for each asset; average; regress on SPY forward 20d return; out-of-sample R² and Spearman ρ.",
    "Yahoo Finance (EFA, EEM, TLT, GLD, SPY)",
)

print("\n✓ Mandate A created (9 beads)")

# ── MANDATE B: Text / NLP / Language Signals (10, P3, blocked on INFRA-B) ─────
for num, title, hyp, mech, uni, meas, data_src in [
    (
        1,
        "10-K readability (FK grade) inversely predicts 12m stock return",
        "Stocks with above-median Flesch-Kincaid grade 10-K filings underperform below-median readability stocks by >3% annually over the subsequent 12 months.",
        "Complex 10-Ks obscure bad news; simpler filings signal management confidence and transparency.",
        "S&P 500 components (large-cap via SPY holdings)",
        "Compute FK grade for 10-K MD&A section; median-split; measure 12m forward return differential; Fama-French adjusted.",
        "EDGAR full-text search API",
    ),
    (
        2,
        "CEO letter I/we ratio predicts 12m underperformance",
        "Companies where the CEO letter has I/we ratio >2 (overconfident) underperform those with I/we ratio <1 by >4% over next 12 months.",
        "CEO overconfidence (high I usage) correlates with empire-building and excessive risk-taking.",
        "S&P 500 components",
        "Parse annual report CEO letters; compute I/(I+we) ratio; split by ratio quartile; measure 12m return.",
        "EDGAR annual report text via SEC API",
    ),
    (
        3,
        "Forward-looking sentence density predicts earnings surprise direction",
        "Companies with >30% forward-looking sentences in 10-K MD&A have positive earnings surprises in next quarter >60% of the time.",
        "Management signals confidence via forward-looking language; this correlates with future beats.",
        "S&P 500 components",
        "Classify sentences as forward/backward-looking via Claude API; compute density; correlate with FactSet/Yahoo earnings surprise.",
        "EDGAR 10-K text, Yahoo Finance earnings data",
    ),
    (
        4,
        "Semantic similarity of consecutive 10-K filings predicts negative returns",
        "Companies with >95% semantic similarity between consecutive 10-K filings (boilerplate copy-paste) underperform low-similarity companies by >5% over 12 months.",
        "High filing similarity signals management inattention or hiding material changes.",
        "S&P 500 components",
        "Embed 10-K MD&A via sentence-transformers all-MiniLM-L6-v2; compute cosine similarity year-over-year; correlate with returns.",
        "EDGAR 10-K text, sentence-transformers",
    ),
    (
        5,
        "Reddit/WSB mention spike predicts mean-reverting returns",
        "Stocks with WSB daily mention volume >3σ above 30d average produce mean-reverting returns of -5% to -10% over the subsequent 20 trading days.",
        "Retail attention-driven price discovery reverses as momentum traders exit and fundamentals reassert.",
        "GME, AMC, BBBY analogs; meme stock universe",
        "Track daily mention counts via Reddit API (pushshift); signal: >3σ spike → short 20d; measure forward return distribution.",
        "Reddit API / pushshift data, Yahoo Finance",
    ),
    (
        6,
        "FOMC hedging language frequency predicts 5d bond volatility",
        "FOMC statements with hedging language density (uncertain/approximately/might/could) >2σ predict 5-day TLT realized volatility increase >50% with >70% accuracy.",
        "Fed communication uncertainty transmits directly to rate expectations and bond market vol.",
        "TLT, ^TNX",
        "Parse FOMC statements from federalreserve.gov; count hedging words per sentence; signal: high hedging → long TLT vol (straddle equivalent).",
        "Federal Reserve website (FOMC statements), Yahoo Finance (TLT)",
    ),
    (
        7,
        "10-K risk factor word count growth predicts downside returns",
        "Companies adding >500 words to risk factors section YoY underperform those with stable/declining risk factor length by >3% over next 12 months.",
        "Expanding risk disclosures signal management awareness of emerging threats not yet priced.",
        "S&P 500 components",
        "Extract risk factors section length per year from EDGAR; compute delta; regress on forward returns.",
        "EDGAR 10-K text",
    ),
    (
        8,
        "MD&A sentiment shift YoY predicts earnings direction",
        "Companies with positive YoY MD&A sentiment shift (FinBERT or Claude API scoring) beat earnings estimates in >65% of subsequent quarters.",
        "Management tone shift anticipates operational improvements before they appear in financials.",
        "S&P 500 components",
        "Score MD&A sentiment via Claude API positive/neutral/negative; compute YoY delta; correlate with next-quarter EPS beat/miss.",
        "EDGAR 10-K text, Yahoo Finance earnings",
    ),
    (
        9,
        "Earnings call Q&A uncertainty language predicts sell-side miss",
        "Earnings calls where CEO/CFO use >3 hedging words per question in Q&A predict sell-side consensus miss in >60% of subsequent quarters.",
        "Analyst Q&A exposes real uncertainty; management hedging in responses signals lack of visibility.",
        "S&P 500 components",
        "Fetch earnings call transcripts; count hedging per Q&A answer; correlate with next quarter EPS miss.",
        "SEC EDGAR (8-K), Yahoo Finance earnings",
    ),
    (
        10,
        "SEC comment letter frequency predicts audit risk premium",
        "Companies receiving >2 SEC comment letters in a 2-year window have subsequent 12m returns -6% vs peers, reflecting higher audit/restatement risk.",
        "SEC comment letters signal accounting quality concerns before formal enforcement.",
        "S&P 500 components",
        "Query EDGAR EFTS for comment letter frequency by CIK; merge with returns; Fama-French adjusted alpha.",
        "EDGAR EFTS API, Yahoo Finance",
    ),
]:
    h(
        "B",
        "Text / NLP / Language Signals",
        num,
        title,
        hyp,
        mech,
        uni,
        meas,
        data_src,
        "needs: NLP pipeline (blocked on INFRA-B)",
        3,
    )

print("\n✓ Mandate B created (10 beads)")
print("\nScript complete through B. Run part 2 for C–P.")

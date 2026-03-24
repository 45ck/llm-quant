# Market microstructure for a $100k systematic macro book

**For a $100,000 portfolio trading $1k–$4k positions across 39 instruments, market impact is effectively zero — the binding execution costs are bid-ask spreads and broker minimums, not timing or impact.** This conclusion holds across all six asset classes, though the optimal execution approach differs meaningfully by asset class and volatility regime. The total round-trip friction ranges from **2–6 bps for US equity ETFs** to **12–33 bps for emerging market ETFs**, with crypto ETFs surprisingly efficient at 3–9 bps. ATR(14) systematically lags regime changes by ~10 days, creating dangerous sizing errors during volatility spikes. In 24-hour crypto and forex markets, daily momentum signals retain meaningful predictive power for 1–6 days, but alpha is concentrating around specific session windows as institutional participation reshapes microstructure.

---

## Brief 1.1: Execution timing barely matters at this scale, but asset class nuances do

### The closing auction has become the dominant liquidity event

The US equity closing auction now matches **$55.5 billion per day** and accounts for **9.44% of total notional volume** — both all-time highs as of Q2 2024. This represents a threefold increase from 3.1% in 2010, driven almost entirely by the growth of indexing and passive investing. On index rebalance or options expiration days, the closing auction captures roughly **20% of daily volume**. For liquid ETFs like SPY and QQQ, the closing auction share typically ranges 5–12% depending on the day.

Market impact within the closing auction is negligible for small orders. NYSE research from January 2024 confirms that most large auction orders submitted in the last 5 minutes have **less than 1x spread of impact**, with little difference between large and small orders. The auction's design pools all orders into a single clearing price, so $2,500 orders free-ride on the aggregate liquidity pool. Applying the square-root impact model (Almgren et al. 2005) to a $4,000 SPY order yields permanent impact of approximately **0.001 bps** — effectively zero. The van Kervel and Westerholm (2023) paper in the *Journal of Financial Markets* confirms that closing prices typically match pre-close bid or ask prices, and price impact is lower than during continuous trading.

### MOC is the simplest correct answer for US equity ETFs

For a $100k portfolio, the choice between MOC, TWAP, and next-open execution is dominated by noise rather than systematic cost differences. Almgren and Chriss (2000) state explicitly that "a broker given a small order to work over the course of a day will almost always execute the entire order immediately" because market impact from rapid trading is negligible compared with the opportunity cost of delay. At $2,500 per position in SPY (daily volume ~$30B+), the order represents less than **0.00001%** of daily volume.

The binding cost is the bid-ask spread itself: **1–2 bps** for SPY/QQQ, **3–5 bps** for TLT, and **5–15 bps** for international developed-market ETFs. MOC orders offer the advantage of executing at the official closing price — the benchmark for index funds and NAV calculations — eliminating tracking error. For a systematic strategy generating end-of-day signals, MOC execution is the natural choice for US equity and fixed income ETFs.

Next-open execution carries measurable overnight gap risk. SPY's cumulative close-to-open return over five years (Q3 2020–Q3 2025) was **+47.1%** versus only **+29.9%** for open-to-close. The average absolute overnight gap is ~0.25–0.30% for US equity ETFs, rising to **0.64%** for GLD (because gold trades 24 hours globally) and **0.5–1.0%** for USO. When a momentum signal is correct, waiting until the next open means forfeiting the overnight move — a cost that far exceeds any spread savings. For fixed income ETFs, overnight gaps are smaller (0.15–0.25% for TLT) because rate expectations drive pricing rather than idiosyncratic news.

### International ETFs demand execution during the overlap window

The stale NAV problem for international equity ETFs is well-documented and economically significant. When US markets close at 4:00 PM ET, European markets have been closed 5–6 hours and Asian markets 12–14 hours. Petajisto (2017, *Financial Analysts Journal*) found ETF prices deviate from NAV within a band of **~200 bps** on average, with international and illiquid-asset ETFs showing the largest deviations. After controlling for stale pricing, a residual pricing band of **~100 bps** persists. A 10 percentage-point VIX increase widens ETF premium dispersion by an additional **18 bps**.

Atanasova et al. (2020) studied 584 US-listed international equity ETFs and measured equal-weighted average bid-ask spreads of **48 bps** and premium volatility of **35 bps**. Vanguard notes that a round-trip cost from volatile premiums can reach **125 bps** (buying at a 64 bps premium, selling at a 61 bps discount). The practical recommendation from practitioners is to trade international ETFs during the overlap window when underlying markets are still open — for European ETFs (VGK, EFA), this is roughly **8:00–11:30 AM ET**. This is the one asset class where execution timing generates meaningful cost differences at any portfolio size.

### Crypto reference pricing has converged on institutional standards

The **CME CF Bitcoin Reference Rate — New York Variant (BRRNY)** at 4:00 PM ET has become the institutional standard, used by six spot Bitcoin ETFs representing over **$55 billion in AUM**. The methodology — a one-hour observation window divided into 12 five-minute partitions using volume-weighted medians — is designed for manipulation resistance. Replication analysis shows slippage of **no more than 1 basis point** on most days, with only 6.76% of days exceeding 5 bps slippage.

For systematic strategies, two conventions coexist: the **00:00 UTC candle close** (universal across Binance, Coinbase, Kraken) for charting and backtesting, and the BRRNY 4:00 PM ET close for institutional benchmarking. For a US-based systematic macro strategy executing via spot BTC ETFs, the BRRNY aligns naturally with the US equity close, creating a unified execution window. Direct crypto exchange execution at 00:00 UTC is a valid alternative for signal generation but introduces a timing mismatch with ETF NAV calculations.

### Forex execution should target the London-New York overlap

Forex spreads follow a predictable intraday pattern dictated by session overlaps. During the **London-New York overlap (13:00–17:00 UTC / 8:00 AM–12:00 PM ET)**, EUR/USD raw spreads on ECN accounts compress to **0.0–0.3 pips**, versus 1.0–2.0+ pips during the Asian session. This window captures roughly **55–60 pips** of EUR/USD's average 58-pip daily range. GBP/USD shows 60–100 pips in the European session expanding to 70–110 pips during the overlap.

The WM/Reuters 4 PM London fix remains the primary FX benchmark despite the 2013 manipulation scandal (total fines exceeding **$10 billion**). Post-reform, the fix window expanded from 1 minute to 5 minutes, roughly halving the impact of outlier trades. However, the fix window is "one of the most volatile intraday periods to trade," with average price impacts of **10–25 bps** — making it unsuitable for execution unless specifically benchmarked to the fix. Krohn (2024, *Journal of Finance*) documents a W-shaped USD return pattern around daily fixes, with systematic USD appreciation before each fix and depreciation afterward — a pattern that implies daily swings exceeding **$1 billion**.

For the llm-quant system, optimal forex execution is straightforward: execute during the London-New York overlap, avoid the 4 PM London fix window, and use limit orders near the quote. At $2,500 position sizes, the binding constraint is IBKR's **$2 minimum commission per order** (equivalent to 16 bps round-trip on $2,500), which makes small forex positions disproportionately expensive.

### VIX regime effects are real but manageable at this size

Bid-ask spreads scale nonlinearly with VIX across all asset classes. SPY spreads widen from **1–2 bps** when VIX is below 15 to approximately **20 bps** during extreme stress (VIX > 50, as in March 2020) — a 10–15x multiplier. For HYG, the widening is even more dramatic: from **3–10 bps** in calm markets to **30–100+ bps** during stress, with NAV discounts reaching 3–5% in March 2020 when underlying bond markets froze. Crypto spreads on major exchanges move from **1–5 bps** normally to **20–100+ bps** during high-volatility events. EUR/USD widens from **0.1–0.5 pips** to **5–15+ pips** in extreme conditions.

The relationship is convex: during March 2020, SPY experienced a roughly 10x spread widening for a ~6x VIX increase, and the effect is more pronounced for less liquid instruments. This convexity matters for position sizing — ATR-based models that assume linear cost scaling will underestimate execution costs during regime transitions.

| Asset class | VIX < 15 | VIX 15–25 | VIX 25–40 | VIX > 40 |
|---|---|---|---|---|
| SPY/QQQ | 1–2 bps | 2–5 bps | 5–15 bps | 15–20+ bps |
| International equity ETFs | 15–30 bps | 25–50 bps | 40–80 bps | 80–200+ bps |
| HYG (high yield) | 5–10 bps | 10–30 bps | 30–60 bps | 60–200+ bps |
| BTC/ETH (exchanges) | 1–5 bps | 3–10 bps | 10–30 bps | 30–100+ bps |
| EUR/USD | 0.1–0.5 pip | 0.5–1.5 pip | 1.5–5 pip | 5–15+ pip |

### Spread estimators perform poorly for liquid assets

The Roll (1984) effective spread estimator — derived from return autocovariance — fails approximately **50% of the time** when applied to daily data, producing undefined values when autocovariance is positive. Its cross-sectional correlation with TAQ effective spreads is only **0.560** (Goyenko, Holden & Trzcinka, 2009), the lowest among major estimators. The Corwin and Schultz (2012) high-low estimator performs better overall (average correlation ~0.62) but this figure is inflated by illiquid stocks; after removing the top decile by spread size, correlation drops to **0.48**. For high-cap, liquid securities — precisely the ETFs in this portfolio — the CS estimator correlation with actual spreads falls to just **0.18**. Tremacoldi-Rossi (2022) showed that with full bias correction, the estimator achieves ~95% cross-sectional correlation, confirming the structural bias is the primary issue. For commodities, the CS estimator has **no correlation** with actual trading costs (Marshall et al. 2011). For a portfolio of liquid ETFs, direct NBBO data from exchanges is strongly preferred over any estimation method.

# Market microstructure for a $100k systematic macro book

**For a $100,000 portfolio trading $1k–$4k positions across 39 instruments, market impact is effectively zero — the binding execution costs are bid-ask spreads and broker minimums, not timing or impact.** This conclusion holds across all six asset classes, though the optimal execution approach differs meaningfully by asset class and volatility regime. The total round-trip friction ranges from **2–6 bps for US equity ETFs** to **12–33 bps for emerging market ETFs**, with crypto ETFs surprisingly efficient at 3–9 bps. ATR(14) systematically lags regime changes by ~10 days, creating dangerous sizing errors during volatility spikes. In 24-hour crypto and forex markets, daily momentum signals retain meaningful predictive power for 1–6 days, but alpha is concentrating around specific session windows as institutional participation reshapes microstructure.

---

## Brief 1.3: Crypto momentum persists 1–6 days; forex alpha concentrates around session transitions

### Daily momentum signals in crypto decay slowly by financial-market standards

Liu and Tsyvinski (2021, *Review of Financial Studies*) established the definitive evidence on cryptocurrency time-series momentum. Bitcoin daily returns positively and significantly predict returns **1, 3, 5, and 6 days ahead** — a 1-standard-deviation increase in today's return predicts a 0.33% increase at 1 day, rising to 0.50% at 6 days. At weekly frequency, top-quintile momentum portfolios earn **11.22% per week** (Sharpe 0.45) versus 2.60% for the bottom quintile. Liu, Tsyvinski, and Wu (2022, *Journal of Finance*) confirmed this with a three-factor model where long/short momentum yields ~**3% excess weekly returns**.

Borgards (2021) found that crypto exhibits significantly **larger and longer momentum periods than equities**, attributed to higher noise-trader participation and difficulty deriving intrinsic value. This suggests that while equity momentum has been crowded from 15–20% annual returns down to 3–5%, crypto momentum retains substantially more alpha — though this gap is narrowing as markets mature. Shen, Urquhart, and Wang (2022, *Financial Review*) documented intraday time-series momentum in Bitcoin, where the first half-hour return positively predicts the last half-hour return, driven by liquidity provision rather than late-informed trading.

### Bitcoin returns concentrate in specific session windows

The most economically significant hourly returns for Bitcoin occur at **22:00 and 23:00 UTC** (5–7 PM Eastern), averaging approximately **0.07% per hour**. A simple strategy buying BTC at 21:00 UTC and selling at 23:00 UTC generates annualized returns of **37–41%** with a maximum drawdown of -18.87% and Calmar ratio of 1.97. The strongest day for this effect is Friday, followed by Thursday.

A separate and potentially more important pattern is the **Monday Asia Open Effect** documented by Concretum Group (2025). High-frequency trend-following benchmarks show strongly positive returns starting Sunday ~7:00 PM New York time, elevated for roughly 24 hours into Monday. This effect coincides with the Monday open of Asian cash equity markets and became **substantially more pronounced after mid-2020** as institutional participation increased. The high-frequency trend benchmark delivers a gross Sharpe ratio of ~**1.6** versus ~0.8 for long-only volatility-targeted Bitcoin.

A microstructure explanation has emerged: since the January 2024 spot ETF launch, Bitcoin ETFs (IBIT, FBTC, GBTC) dominate price discovery over spot markets approximately **85% of the time** (measured by Information Leadership Share on 5-minute data). This concentrates price discovery around US trading hours and makes the US market close increasingly important as a reference point.

### Forex returns follow a W-shaped pattern around daily fixes

Krohn (2024, *Journal of Finance*) documented that USD appreciates systematically in the run-up to FX fixes and depreciates afterward, creating a W-shaped return pattern around the clock. Before the Tokyo fix (9:55 local time), USD appreciates approximately **2 bps** with positive reversal returns **55–60%** of the time. Smaller but statistically significant effects occur around the Frankfurt (14:15) and London (16:00) fixes. These patterns imply exploitable daily swings exceeding $1 billion based on 2019 BIS turnover data.

The Engle-Ito-Lin (1990) "meteor showers" framework — where volatility transmits across geographical sessions — remains empirically valid. Cross-market volatility spillovers among risk-neutral volatility innovations account for **59% (JPY) to 77% (EUR)** of 10-day forecast error variance. For a daily momentum strategy, this means that strong US-session moves in EUR/USD will propagate into the Asian session with meaningful probability, creating both opportunity (if the signal is correct) and risk (if the Asian session reverses).

Ranaldo and Söderlind (2010, *Review of Finance*) established that CHF and JPY appreciate against USD when stock prices decrease and volatility increases, with safe-haven properties materializing across time granularities from hours to days. This is directly relevant for the llm-quant system's forex component: safe-haven currency momentum signals may exhibit different decay profiles than risk-currency signals, particularly during equity selloffs.

### Weekend crypto patterns are noisy but not random

There is **no statistically significant weekend-weekday gap in average Bitcoin returns** (2014–2024), but weekend volume is typically **20–40% lower** than weekday volume. Caporale and Plastun (2019) found that Bitcoin shows higher returns on Mondays (>0.4% average daily return) and lower on Thursdays, with a trading simulation showing a 60% win rate. Kumari, Wasan, and Chhimwal (2025) found that weekend momentum returns significantly exceed weekday momentum returns for altcoins, with higher Sharpe ratios and lower drawdowns — though this paper is from a less established journal and warrants skepticism.

Mourey et al. (2025, *ScienceDirect*) documented an asymmetric crypto-equity weekend spillover: negative weekend crypto returns significantly predict **Monday equity declines**, while positive weekend returns have no effect on Monday equities. This effect strengthened after the May 2022 LUNA collapse. For the llm-quant system, this means Friday close crypto positions carry asymmetric risk: weekend crypto losses may amplify Monday equity losses across the portfolio, while weekend crypto gains provide no portfolio-level benefit.

### Crypto microstructure has improved dramatically since 2020

Bitcoin market quality has transformed since the pre-institutional era. Bid-ask spreads on Coinbase fell from over **1 basis point to 0.3 bps**, and on Kraken from **0.4 to 0.1 bps** (Kaiko Research, 2024). Bitcoin 1% market depth on US exchanges reached an all-time high of **$290 million** in early 2025, definitively closing the Alameda Gap that persisted for over a year after FTX's collapse. BTC 60-day historical volatility has remained below **50%** since early 2023, compared to over 100% in 2022.

Cross-exchange arbitrage opportunities, which Makarov and Schoar (2020) documented as persisting for days to weeks, have **diminished substantially since 2018**. Settlement latency remains an impediment (Hautsch et al. 2024), but the practical implication for signal decay is that crypto markets are becoming more efficient, meaning momentum signals will likely decay faster over time. The equity market trajectory — momentum returns compressing from 15–20% annually to 3–5% — may presage crypto's future, though the timeline remains uncertain.

---

## Cross-cutting: the $100k portfolio sits in a sweet spot where simplicity dominates

### Execution costs are real but modest

For the complete 39-instrument portfolio at $2,500 average position size, realistic total round-trip execution costs (spread + commission + impact) are:

| Asset class | Typical spread | Commission (IBKR) | Total round-trip |
|---|---|---|---|
| US equity ETFs (SPY, QQQ) | 1–2 bps | 0 bps (Lite) | **2–6 bps** |
| International ETFs (VGK, EFA) | 5–10 bps | 2–8 bps | **7–18 bps** |
| Emerging market ETFs (EEM) | 10–25 bps | 2–8 bps | **12–33 bps** |
| Treasury ETFs (TLT) | 3–5 bps | 0 bps (Lite) | **3–9 bps** |
| High-yield ETFs (HYG) | 8–20 bps | 0 bps (Lite) | **8–24 bps** |
| Bitcoin ETF (IBIT) | 3–5 bps | 0 bps (Lite) | **3–9 bps** |
| Crypto direct (Binance) | 3–5 bps | 15–20 bps | **18–25 bps** |
| Forex (IBKR, $2,500) | 1–2 bps | 16 bps* | **17–18 bps** |
| Forex (IBKR, $25k+) | 1–2 bps | 0.4 bps | **1.4–2.4 bps** |

*IBKR's $2 minimum FX commission is binding on small positions, making $2,500 forex trades disproportionately expensive.

Bitcoin ETFs (IBIT at 3–9 bps round-trip) dominate direct exchange access (18–25 bps on Binance, 65–125 bps on Coinbase at low volume tiers) for cost efficiency at this position size. IBKR Lite's zero-commission structure for US-listed securities makes it the optimal broker for this portfolio.

### Backtest-to-live performance gaps are the real risk

The Quantopian study of 888 algorithmic strategies (Wiecki et al. 2016) found that backtest Sharpe ratio has an R² of **less than 0.025** with out-of-sample performance — essentially zero predictive power. More backtesting correlates with worse live performance, providing direct empirical evidence of overfitting harm. Live drawdowns typically run **1.5–2x backtested drawdowns**, and execution frictions erode **30–50% of theoretical returns**.

For the llm-quant system, this implies conservative execution cost assumptions should be embedded in all backtests: 5 bps for US ETFs, 15–30 bps for international/EM ETFs, 8–15 bps for fixed income, 8–10 bps for crypto via ETF, and 15–20 bps for small forex positions. Apply a **1.5–2x multiplier** for the backtest-to-live gap. Bailey and Lopez de Prado's Probability of Backtest Overfitting should target below 15%.

### Recent market structure changes create tailwinds

The SEC's tick-size reform (adopted September 2024) introduces a **half-penny tick ($0.005)** for stocks with time-weighted average quoted spread ≤ $0.015, affecting ~74% of listed stocks. The access fee cap drops from 30 mils to 10 mils — a 67% reduction. Compliance has been delayed to November 2026, but when implemented, tighter ticks on liquid ETFs will marginally reduce spread costs. Round-lot definition changes went live in November 2025, improving displayed liquidity for higher-priced securities.

The spot Bitcoin ETF launch in January 2024 created the single most important structural improvement for crypto execution. IBIT accumulated over **$45 billion in AUM** within its first year, with institutional investors holding over 25% of total US BTC ETF exposure. Options on IBIT launched in November 2024. This provides highly liquid, low-cost crypto exposure within traditional brokerage infrastructure, eliminating the need for direct exchange access.

## Conclusion

Three actionable findings emerge from this research that directly inform the llm-quant system's execution architecture. First, **execution timing optimization has near-zero marginal value at $100k** for US equity and fixed income ETFs — MOC orders are the correct default, and the engineering effort of TWAP or algorithmic execution cannot be justified. The one exception is international ETFs, where trading during the European-US overlap window (8:00–11:30 AM ET) reduces stale-NAV-driven costs by an estimated 20–50 bps per round-trip.

Second, **ATR(14) should be replaced with EMA-based ATR(10–14) plus a conditional VIX overlay** for position sizing. The Wilder smoothing's 9.55-day half-life is unacceptably slow for regime transitions, and the evidence from Bongaerts et al. (2020) shows that conditional volatility targeting — adjusting only at regime extremes — outperforms continuous scaling. A 126-day VIX percentile rank provides the optimal memory for regime classification.

Third, **crypto and forex signal decay is slower than expected at daily frequency**, with Bitcoin momentum remaining statistically significant for 1–6 days. The concentration of Bitcoin returns at 22:00–23:00 UTC and the Monday Asia Open Effect suggest that a daily signal-generation cadence is adequate — the alpha loss from not updating signals intraday is modest relative to the implementation complexity. The more important finding is the asymmetric weekend spillover: negative weekend crypto returns predict Monday equity losses, creating a portfolio-level risk that warrants position-level awareness if not explicit hedging. For forex, executing during the London-New York overlap and avoiding the WM/Reuters fix window captures the tightest spreads available, and the W-shaped return pattern around fixes offers a small but persistent edge for signal timing.

# Market microstructure for a $100k systematic macro book

**For a $100,000 portfolio trading $1k–$4k positions across 39 instruments, market impact is effectively zero — the binding execution costs are bid-ask spreads and broker minimums, not timing or impact.** This conclusion holds across all six asset classes, though the optimal execution approach differs meaningfully by asset class and volatility regime. The total round-trip friction ranges from **2–6 bps for US equity ETFs** to **12–33 bps for emerging market ETFs**, with crypto ETFs surprisingly efficient at 3–9 bps. ATR(14) systematically lags regime changes by ~10 days, creating dangerous sizing errors during volatility spikes. In 24-hour crypto and forex markets, daily momentum signals retain meaningful predictive power for 1–6 days, but alpha is concentrating around specific session windows as institutional participation reshapes microstructure.

---

## Brief 1.2: ATR(14) lags regime changes by 10 days and misses liquidity risk entirely

### The 9.55-day half-life creates dangerous blind spots

Wilder's smoothing formula for ATR(14) — ATR_t = (13/14) × ATR_{t-1} + (1/14) × TR_t — assigns a decay factor of (13/14)^k per period. The half-life solves to **~9.55 days** and 80% convergence to **~22 days**. This is mathematically exact and empirically consequential.

During the COVID crash, VIX moved from 15 to 82.69 in 17 trading days (February 20 to March 16, 2020). ATR(14) could not have reflected more than 50% of the new volatility regime until roughly March 6–10. A position-sizing model using ATR(14) on February 28 would have computed sizes based on the calm January–February regime, dramatically underestimating risk just as the market entered its most violent drawdown. Full 80% convergence would not have occurred until late March — after the worst of the crash was over.

The August 2024 VIX spike illustrates a different failure mode. VIX surged 180% to ~66 pre-market on August 5, then fell to ~39 by close and continued declining. ATR(14) absorbed only **~7%** (1/14) of the one-day spike, and subsequent lower-volatility days prevented ATR from ever reflecting the true intraday risk. This demonstrates ATR's fundamental inability to capture flash-crash-style events. In contrast, the 2022 rate-hiking cycle — where volatility increased gradually over months — is the one scenario where ATR(14) performs adequately, because exponential smoothing can track slow drift.

**EMA-based ATR** with the same period is approximately **twice as fast**, with a half-life of ~5 days versus Wilder's ~9.55 days. The difference arises because Wilder's smoothing factor is α = 1/n = 7.14%, while EMA uses α = 2/(n+1) = 13.3%. An EMA(14) ATR captures roughly 77% of weight in the last 14 bars versus Wilder's 64%. For regime-sensitive applications, this is a free improvement.

### ATR correlates with spreads but misses the liquidity dimension

Bouchaud et al. (2008, *Quantitative Finance*) established that bid-ask spreads are strongly correlated with per-trade volatility, with **R² exceeding 0.90** across 68 Paris Stock Exchange stocks and 155 NYSE stocks. The linear relationship S ≈ 2σ₁ confirms that the volatility captured by ATR is indeed a reasonable proxy for spread levels under normal conditions.

However, ATR misses critical dimensions of execution risk. It cannot detect **intraday spread dynamics**, **market depth or order-book thickness**, or **asymmetric spread widening** during stress. The BIS documented during the August 2024 VIX spike that OTM put option bid-ask spreads widened to **80%+ of mid-price** (versus ~25% normally), contributing to over 85% of the VIX spike — yet this was driven by market-maker quote adjustment, not fundamental volatility. This represents a case where spread risk can diverge dramatically from the volatility ATR captures. The Amihud (2002) illiquidity ratio (|return| / dollar volume) captures the price-impact dimension ATR cannot, but requires volume data and is more suited for screening than real-time sizing.

### VIX-adjusted sizing outperforms ATR-only models

Three lines of evidence support incorporating VIX directly into position sizing:

Wysocki (2025, *arXiv*) evaluated Kelly Criterion sizing, VIX-Rank sizing, and hybrid approaches for S&P 500 option strategies. The **hybrid sizing method consistently balanced return generation with robust drawdown control**, particularly in low-volatility environments. A key finding: VIX9D (9-day implied volatility) outperformed the standard 30-day VIX for short-dated strategies, as the 30-day measure "incorporates excessive noise or irrelevant longer-term expectations." Optimal VIX memory for risk-adjusted returns was **126 days** for regime classification.

Bongaerts, Kang, and Van Dijk (2020, *Financial Analysts Journal*) demonstrated that conventional (continuous) volatility targeting **fails to consistently improve performance** in global equity markets and can increase drawdowns due to overshooting the volatility target — the ratio of realized-to-target volatility was 1.16 in the US market. Their **conditional volatility targeting** — adjusting only in extreme high- and low-volatility states — consistently enhanced Sharpe ratios and reduced drawdowns with low turnover. This is the key insight for the llm-quant system: don't continuously scale positions with ATR; instead, establish VIX-based regime thresholds that trigger discrete sizing adjustments.

Man Group's practical implementation used a ~1-year half-life for volatility estimation and achieved 500 bps excess return over a passive benchmark at 10% volatility target, reducing maximum drawdown from 40% to 25%. But they acknowledged the measure "changes quite slowly… too slowly to cope with sudden violent changes, such as 2008."

### Spread blowouts during regime transitions are asymmetric and violent

During March 2020, fixed income ETFs experienced historically unprecedented dislocations. HYG dropped ~20% from peak to trough while trading at **multi-percent discounts to NAV** as underlying bond markets froze. Investment-grade bond spreads widened by **~400 bps** at peak. The MSCI LiquidityMetrics model, using 60-day historical volatility, continued showing rising transaction costs even after the market rebounded — directly demonstrating the lagging-indicator problem for ATR-style risk models.

Crypto markets showed similar but even more volatile dynamics. During the March 2020 crash, Binance BTC futures bid-ask spreads spiked to **7.95%** before declining to a record low of 0.25% by August 2020. The FTX/Alameda collapse in November 2022 created an "Alameda Gap" in Bitcoin market depth that persisted for over a year, only fully closing when US exchange 1% market depth hit an all-time high of **$290 million** in May 2025. In forex, flash crashes in thin liquidity — the CHF shock of January 2015 (30%+ in minutes), the JPY flash crash of January 2019 (7% in minutes), and the GBP flash crash of October 2016 (>6% in seconds) — illustrate that spread risk can decouple entirely from realized volatility.

### Practical position-sizing framework recommendation

A three-component model outperforms ATR-only sizing for this portfolio:

- **Base sizing**: EMA-based ATR (not Wilder's) with period 10–14, providing faster regime response (~5-day half-life)
- **VIX overlay**: Conditional regime adjustment using 126-day VIX percentile rank. Full sizing when VIX is in the 20th–80th percentile; reduce by 25–30% when VIX exceeds the 80th percentile; consider increasing by 10–15% when VIX is below the 20th percentile (to capture the low-vol undersizing bias)
- **Liquidity monitor**: Track Amihud ILLIQ ratio or direct spread data for each instrument; flag when ILLIQ exceeds its 90th percentile as a signal to reduce position size independently of ATR

Kyle's lambda (price sensitivity to order flow) and the Glosten-Milgrom adverse selection framework provide the theoretical foundation — adverse selection costs rise during high-VIX regimes as information asymmetry increases — but at $2,500 per position, these effects are not directly measurable. The practical impact is indirect: wider spreads reflecting higher adverse selection reduce the effective capture of theoretical alpha.

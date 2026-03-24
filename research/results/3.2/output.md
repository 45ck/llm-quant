# Module 3.2: ATR-Based Risk Budgeting

> **Summary from cross-module analysis**: Volatility-based position sizing (Module 3.2) stands on the firmest empirical ground of the three proposed modules. Across all three specs, the directional hypotheses are well-supported by decades of academic and practitioner research, but the specific quantitative targets are generally more optimistic than the evidence warrants.

---

## Module 3.2: ATR risk budgeting has the strongest empirical foundation

Volatility-based position sizing is the **most robustly supported** of the three approaches. Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & Van Hemert (2018) from Man AHL examined 60+ assets with daily data from 1926–2017 and found that volatility targeting reduced the volatility-of-volatility from **4.6% to 1.8%** for US equities — a 61% reduction. Maximum drawdowns for balanced and risk parity portfolios fell by **35–50%** over the full sample. Asness, Frazzini & Pedersen (2012) showed risk parity portfolios outperformed the market portfolio by approximately **4% annually** over 1926–2010. Research Affiliates (2024) demonstrated that during the COVID-19 crash, a volatility-targeted portfolio lost less than 5% versus approximately 25% for an unmanaged portfolio.

The spec's claimed 30–40% reduction in annualized portfolio volatility is **well within the documented range**, and the 25–35% max drawdown reduction is similarly supported. Sharpe preservation within +/-10% is plausible given that Moreira & Muir (2017) found volatility-managed portfolios produced alpha of **4.9%** and a 25% Sharpe improvement for equity portfolios. However, Cederburg et al. (2020) showed that across 103 equity strategies, volatility management outperformed in only 53 cases — barely above chance — cautioning against overconfidence in out-of-sample Sharpe improvement.

ATR(14) is the **industry-standard volatility proxy** for position sizing, used by the Turtle Trading system (which employed 20-day ATR and 2N stops, generating reportedly over $175 million in five years) and recommended by Robert Carver (ex-Man AHL). ATR captures gaps and limit moves better than standard deviation, making it superior for stop-loss calibration. However, practitioners like Carver prefer EWMA volatility for its faster adaptation to regime changes. **ATR(14) with Wilder smoothing gives each new bar only ~7% weight**, making it slow to adapt during sudden regime shifts. For crypto assets with annualized volatility of 60–80%, a shorter lookback (ATR(7) or ATR(10)) is advisable.

### The 2x ATR stop-loss multiplier is well-calibrated but asset-class-dependent

The 2x ATR multiplier is the **exact multiplier used by the Turtle Traders** and sits squarely in the recommended range for swing/position trading. A LuxAlgo (2024) study found that 2x ATR stops reduced maximum drawdown by **32%** compared to fixed stop-loss levels. The practitioner consensus spans 1.5x (day trading) to 3–4x (long-term position trading), with 2x as the modal recommendation. For crypto and volatile commodities, 2.5–3x may perform better, and the spec should backtest across the full range.

Turnover costs from volatility-driven rebalancing are **manageable but require buffer rules**. Harvey et al. (2018) found that despite 2.4–4.4x higher turnover, estimated transaction costs were only ~8 basis points annually — but they traded futures, not ETFs. For a $100K ETF portfolio averaging ~$2,500 per position, ETF spreads of 0.05–0.10% on frequent rebalancing could accumulate meaningfully. The literature strongly recommends rebalancing only when position deviation exceeds **20–25% of target weight**, and considering weekly rather than daily recalculation for less liquid assets.

A meaningful limitation: the spec's ATR-based sizing is **inverse-volatility weighting (naive risk parity), not true equal-risk-contribution**. The distinction matters. ERC accounts for correlations, giving more weight to assets with low correlation to the portfolio (like crypto), while inverse-vol systematically underweights them. ReSolve Asset Management's empirical studies found the performance gap is "modest for asset class universes" where correlations are relatively diverse, but Carver calls ignoring correlations in many portfolio problems "madness." A two-layer approach — ATR-based sizing within asset class clusters, risk-weighting across clusters — would address this without requiring full covariance estimation.

An important nuance for high-ATR assets: Burggraf & Rudolf (2021) found **no evidence of the low-volatility anomaly in cryptocurrency markets**, unlike equities, bonds, and commodities where Frazzini & Pedersen (2014) documented it across all major asset classes. Sizing crypto by ATR appropriately controls risk contribution but does not exploit a low-vol premium — it simply reduces allocation to the noisiest assets.

---

## Concrete Recommendations for Module 3.2

**Module 3.2** is the strongest and needs the fewest changes. Consider EWMA volatility as a faster-adapting alternative to Wilder-smoothed ATR(14), especially for crypto (where ATR(7-10) may be more appropriate). Implement buffer rules requiring **20–25% deviation** before rebalancing to control turnover costs. Backtest the 2x ATR stop-loss multiplier against 1.5x and 3x across all six asset classes. Consider a two-layer sizing approach (ATR within asset classes, correlation-aware weighting across classes) to address the inverse-volatility-vs-ERC gap.

---

## Cross-Module Context: The Layered Architecture

The evidence-based ranking for "bang for the buck" (drawdown reduction per unit of return sacrifice):

1. **Volatility-based position sizing** — highest value, simplest implementation, most robust evidence, nearly universally adopted by systematic firms, improves Sharpe by 0.2–0.3 with minimal cost
2. **CVaR/tail risk constraints** — moderate value, catches extreme scenarios that vol sizing misses, but estimation error is high and can be overly conservative; best used as a binding constraint rather than primary sizing tool
3. **Correlation-based trade blocking** — lowest standalone value, useful as a guardrail but crude as primary mechanism; dynamic correlation estimation is noisy and the blocking mechanism is untested; better to use correlations implicitly through diversification-weighted sizing

AQR explicitly warns about the whipsaw cost: "A strategy that reduces exposures before the worst of a tail event by definition begins to cut risk before a full-blown crisis. At times there will be false alarms, where positions are cut but the market quickly recovers." For a $100K portfolio, this cost is amplified by transaction costs. Carver notes that $100K is the **minimum viable size** for diversified systematic trading, and recommends binary or thresholded forecasting when position sizes are this small.

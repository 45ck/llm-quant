# Module 3.1: Dynamic Correlation Risk Limits

> **Summary from cross-module analysis**: Volatility-based position sizing (Module 3.2) stands on the firmest empirical ground of the three proposed modules, while the dynamic correlation limits (Module 3.1) face the most serious methodological challenges. Across all three specs, the directional hypotheses are well-supported by decades of academic and practitioner research, but the specific quantitative targets are generally more optimistic than the evidence warrants.

---

## Module 3.1: Correlation convergence is real but slower and messier than hypothesized

The spec's central claim — that pairwise correlations converge to **0.85–0.95 within 3–5 trading days** during stress — finds partial but not full support. Longin & Solnik (2001) confirmed that correlations increase in bear markets but not bull markets, using extreme value theory on five major equity markets from 1958–1996. Ang & Chen (2002) documented that downside correlations exceed normal-distribution-implied levels by **11.6%** for US equities. During COVID-19, US-UK stock market correlations reached approximately **0.95**, but this took roughly 2–3 weeks (late February to mid-March 2020), not 3–5 days. The 0.85–0.95 range is plausible for within-class equity correlations during extreme stress but does not hold for cross-asset pairs. Gold-equity correlations, for instance, flipped from strongly negative to positive during March 2020's "dash for cash" phase, then reverted within two months.

A critical caveat comes from Forbes & Rigobon (2002), who demonstrated that **apparent correlation increases during crises are partly a statistical artifact of heteroskedasticity**. Higher volatility mechanically biases Pearson correlations upward. After adjusting for this bias, they found "virtually no increase in unconditional correlation coefficients" during the 1997 Asian crisis, 1994 Mexican devaluation, and 1987 crash. The spec's rolling window approach is fully exposed to this bias; DCC-GARCH, which filters through GARCH-standardized residuals, inherently corrects for it.

The most consequential finding for the spec concerns **lead-time**. Evidence from Munnix et al. (2012) and a 72-year DJIA study shows that average correlation scales *linearly with concurrent market stress*, meaning correlations move simultaneously with drawdowns, not before them. Pharasi et al. (2018) identified "long-term precursor" correlation states using random matrix theory, but these operate on much longer timescales than 3–5 days. Since the spec uses VIX (forward-looking implied volatility) as the primary regime classifier and correlation as a secondary filter, this partially mitigates the lag problem — VIX spikes typically precede or coincide with drawdown onset, providing the lead time that rolling correlations cannot.

**The proposed VIX thresholds of 20 and 25 may not be optimal.** A 2025 RegimeFolio study directly compared three VIX regime classification methods: the spec's institutional fixed thresholds (VIX < 20, 20–25, > 25) achieved a Sharpe of 0.94, while data-driven terciles (breakpoints at 17.8 and 23.1) achieved **1.17** — a 24% improvement. The three-regime structure itself is well-motivated, but adaptive percentile-based boundaries would likely outperform fixed ones.

### Rolling windows are the weakest link in Module 3.1

Engle's (2002) foundational DCC-GARCH paper compared multiple dynamic correlation approaches and found that **"without exception, the worst performer on all of the tests and data sets is the moving average model"** — i.e., rolling window correlation. DCC-GARCH, EWMA, and even diagonal BEKK all outperformed rolling windows on statistical accuracy. The 21-day window is particularly problematic: with only 21 observations estimating 741 pairwise correlations, the estimates are extremely noisy. Ghost effects (artificial correlation changes when extreme observations enter or exit the window) compound this problem, though the spec's use of log returns rather than prices partially mitigates it.

DCC-GARCH with 39 assets is computationally feasible. The two-step estimation procedure (39 univariate GARCH models plus 2 scalar DCC parameters) can run in 30–60 seconds on modern hardware. Engle, Ledoit & Wolf (2019) specifically addressed high-dimensional DCC using nonlinear shrinkage estimators, and the DECO (Dynamic Equicorrelation) model of Engle & Kelly (2012) provides consistent estimates even in large systems. Lopez de Prado's (2016) hierarchical risk parity framework for correlation clustering is well-validated and naturally complements the spec's hierarchical clustering proposal — HRP achieved lower out-of-sample variance than classical Markowitz optimization in Monte Carlo simulations.

**The correlation-based trade-blocking mechanism itself is novel and has no direct empirical validation in the literature.** While regime-aware portfolio allocation broadly outperforms regime-agnostic approaches, no published study tests the specific mechanism of blocking BUY trades when average pairwise correlation exceeds a threshold. The claimed 20–30% average drawdown reduction and 15–25% max drawdown reduction are untested quantities requiring careful backtesting.

---

## Concrete Recommendations for Module 3.1

**Module 3.1** requires the most significant revisions. Replace rolling window correlations with DCC-GARCH or at minimum EWMA (lambda ~ 0.94–0.97), which Engle (2002) showed dramatically outperforms rolling windows. Adjust the convergence timeline expectation from 3–5 days to 1–3 weeks. Consider adaptive VIX thresholds based on historical percentiles (~17.8, 23.1) rather than fixed 20/25. Account for Forbes-Rigobon heteroskedasticity bias by using volatility-standardized returns. Recognize that the trade-blocking mechanism is novel and untested — it needs particularly rigorous out-of-sample validation.

---

## Cross-Module Context: The Layered Architecture

The evidence-based ranking for "bang for the buck" (drawdown reduction per unit of return sacrifice):

1. **Volatility-based position sizing** — highest value, simplest implementation, most robust evidence, nearly universally adopted by systematic firms, improves Sharpe by 0.2–0.3 with minimal cost
2. **CVaR/tail risk constraints** — moderate value, catches extreme scenarios that vol sizing misses, but estimation error is high and can be overly conservative; best used as a binding constraint rather than primary sizing tool
3. **Correlation-based trade blocking** — lowest standalone value, useful as a guardrail but crude as primary mechanism; dynamic correlation estimation is noisy and the blocking mechanism is untested; better to use correlations implicitly through diversification-weighted sizing

The key risk with layered constraints is **correlated model failure**. If all three layers rely on the same underlying covariance estimates, a single estimation failure propagates through the entire system. Each layer should use somewhat independent information sources — different lookback windows, different estimation methods, or different data inputs — to reduce correlated failure risk.

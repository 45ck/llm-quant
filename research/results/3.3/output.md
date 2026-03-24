# Module 3.3: Portfolio-Level CVaR Constraints

> **Summary from cross-module analysis**: Volatility-based position sizing (Module 3.2) stands on the firmest empirical ground of the three proposed modules, while the dynamic correlation limits (Module 3.1) face the most serious methodological challenges. The literature strongly endorses a layered risk management architecture in principle, but the research reveals critical implementation vulnerabilities including that historical simulation CVaR with only 252 observations produces unstable tail estimates.

---

## Module 3.3: CVaR constraints are theoretically sound but estimation is the Achilles' heel

Rockafellar & Uryasev's (2000, 2002) foundational work established that CVaR can be minimized via linear programming, making it computationally tractable for large portfolios. CVaR is coherent (sub-additive, convex, translation-invariant), unlike VaR, which can penalize diversification. Krokhmal, Uryasev & Zrazhevsky (2002) demonstrated in hedge fund portfolio optimization that **imposing CVaR risk constraints improved out-of-sample performance**, and that "it is beneficial to combine several types of risk constraints that control different sources of risk." A 2024 arxiv study found CVaR constraints improved Sharpe ratios by more than **0.2** (from 1.25 to 1.45) and reduced max drawdown by ~4% below market benchmarks.

The proportional scaling enforcement mechanism is well-supported by the CPPI (Constant Proportion Portfolio Insurance) literature. Black & Jones (1987) and Black & Perold (1992) showed that dynamic, proportional reduction of exposure outperforms binary stop-loss (hard blocking) in terms of risk-adjusted returns under moderate volatility regimes. Proportional scaling avoids the whipsaw problem inherent in hard blocking, where positions are fully cut and cannot participate in recoveries.

### Statistical stability is the critical vulnerability

The spec's historical simulation CVaR at 95% confidence with a 252-day lookback uses **only ~12–13 tail observations** to estimate expected shortfall. This is dangerously few. Kondor et al. (2015) and Caccioli et al. (2015) found that **approximately 4,000 daily observations (~16 years)** are required for reliable ES estimation in portfolio optimization settings. Yamai & Yoshiba (2002) showed that ES estimation error has a relative standard deviation of **20–30% with 250 observations** under normality, worsening significantly under fat tails. Sarykalin, Serraino & Uryasev (2008) explicitly warn: "If there is no good model for the tail of the distribution, CVaR value may be quite misleading."

The **cliff effect** — when a major stress event drops out of the 252-day rolling window — is well-documented and could cause CVaR estimates to drop discontinuously, precisely when the risk memory is most valuable. The Basel FRTB framework addresses this by requiring a separate "stressed ES" calibrated to the worst historical 250-day period, which the spec's stress-scenario conditional CVaR partially mirrors.

The literature converges on several mitigations: **Filtered Historical Simulation** (Barone-Adesi et al., 1999) — using GARCH models to rescale historical residuals to current volatility — is widely considered the best compromise between parametric and non-parametric approaches. Extending the lookback to 500+ days, applying EWMA volatility weighting (lambda = 0.94 per RiskMetrics), or fitting Extreme Value Theory / Generalized Pareto distributions to the tail would all improve stability. The spec's inclusion of named stress scenarios (COVID, 2022 rates, crypto winter, SVB) provides a critical safety net that the literature strongly endorses.

A rank correlation of 0.30 between predicted and realized CVaR is **reasonable and perhaps optimistic** for plain historical simulation. CVaR is not elicitable as a standalone forecast — there exists no simple scoring rule for evaluating CVaR predictions in isolation — making backtesting inherently more difficult than for VaR. The prediction has directional value but substantial noise.

---

## Concrete Recommendations for Module 3.3

**Module 3.3** should augment the 252-day historical simulation with Filtered Historical Simulation (GARCH-filtered residuals rescaled to current volatility), extend the base lookback to at minimum 500 days, and consider fitting Generalized Pareto distributions to the tail for more stable estimates. The stress-scenario overlay is essential, not optional — it directly addresses the cliff effect and aligns with FRTB best practices. Bootstrap confidence intervals on CVaR estimates should be computed and reported to quantify estimation uncertainty. The Sharpe preservation claim is actually conservative — CVaR constraints often *improve* risk-adjusted returns in the literature.

---

## Cross-Module Context: Walk-Forward Validation

With only 3–5 major crises in 20 years of daily data, risk parameters are essentially being fit to 3–5 data points. Lopez de Prado's Probability of Backtest Overfitting (PBO) framework formalized this concern, and Bailey et al. (2015) demonstrated that "such an 'optimal' strategy often performs very poorly out of sample." A 2024 comparative study found that Combinatorial Purged Cross-Validation (CPCV) is **"markedly superior in mitigating overfitting risks"** compared to standard walk-forward analysis, producing distributions of performance estimates rather than point estimates.

For risk management parameters specifically, the literature recommends: regime-aware block bootstrap (classify history into crisis/normal, bootstrap within regimes to create synthetic histories with more crisis periods); sensitivity analysis over parameter ranges rather than point optimization (ensure the strategy works across correlation limits of 0.55–0.80, not just at 0.70); and cross-asset/cross-time transfer tests (parameters calibrated on equities should show some validity on commodities). J.P. Morgan's systematic strategies research states bluntly: "Risk management signals may not have statistical significance — i.e., will look good in a backtest but may not work in the future."

## Cross-Module Context: The Layered Architecture

The evidence-based ranking for "bang for the buck" (drawdown reduction per unit of return sacrifice):

1. **Volatility-based position sizing** — highest value, simplest implementation, most robust evidence, nearly universally adopted by systematic firms, improves Sharpe by 0.2–0.3 with minimal cost
2. **CVaR/tail risk constraints** — moderate value, catches extreme scenarios that vol sizing misses, but estimation error is high and can be overly conservative; best used as a binding constraint rather than primary sizing tool
3. **Correlation-based trade blocking** — lowest standalone value, useful as a guardrail but crude as primary mechanism; dynamic correlation estimation is noisy and the blocking mechanism is untested; better to use correlations implicitly through diversification-weighted sizing

Man Numeric advocates using Expected Shortfall "in combination with other risk estimation techniques as part of an ensemble," noting that CVaR alone "does little to protect against long, slow drawdowns characterised by serially correlated returns."

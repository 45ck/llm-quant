# Domain 3 Analysis: Risk Management & Portfolio Construction

> Synthesis of Modules 3.1 (Dynamic Correlation Limits), 3.2 (ATR Risk Budgeting), and 3.3 (CVaR Constraints) against the current 8-check risk system in `risk/limits.py` and `risk/manager.py`.

---

## Key Findings

### What the research confirmed

- **Correlation convergence during stress is real.** Longin & Solnik (2001), Ang & Chen (2002), and COVID-19 evidence all confirm that pairwise correlations rise in bear markets. Within-class equity correlations can reach 0.85-0.95 during extreme stress. The gap in our risk system -- no awareness of correlation dynamics -- is a genuine vulnerability.
- **ATR-based volatility sizing has the strongest empirical foundation of all three proposals.** Man AHL's Harvey et al. (2018) documented 35-50% max drawdown reduction across 60+ assets over 90 years. Research Affiliates (2024) showed volatility-targeted portfolios lost <5% during COVID vs ~25% unmanaged. This is the single highest-value enhancement available.
- **CVaR is theoretically superior to VaR** (coherent, sub-additive, convex). Rockafellar & Uryasev (2000, 2002) proved it can be minimized via LP. A 2024 study found CVaR constraints improved Sharpe by >0.2 and reduced max drawdown by ~4%.
- **The 2x ATR stop-loss multiplier is well-calibrated** -- matches the Turtle Traders' exact multiplier and sits at the modal recommendation for swing/position trading. LuxAlgo (2024) showed 32% drawdown reduction vs fixed stops.
- **The three-regime VIX classification structure (low/mid/high) is well-motivated** and broadly supported. Our current system has no regime awareness at all.
- **Proportional scaling beats binary blocking** for risk enforcement. CPPI literature (Black & Jones 1987) shows proportional exposure reduction outperforms hard stop-loss on risk-adjusted returns.

### What the research challenged

- **Rolling window correlations are the worst performer.** Engle (2002) found that "without exception, the worst performer on all of the tests and data sets is the moving average model." Our spec's 21-day rolling window estimating 741 pairwise correlations from 21 observations is extremely noisy and exposed to Forbes-Rigobon heteroskedasticity bias. This was originally tagged P0 CRITICAL but the proposed implementation method was fundamentally flawed.
- **Correlation convergence is slower than hypothesized.** The spec claimed 3-5 trading days; evidence shows 1-3 weeks (COVID took late February to mid-March 2020). Correlations move simultaneously with drawdowns, not before them.
- **Fixed VIX thresholds of 20/25 are suboptimal.** A 2025 RegimeFolio study showed data-driven terciles (17.8/23.1) achieved Sharpe of 1.17 vs 0.94 for fixed thresholds -- a 24% improvement.
- **Historical simulation CVaR with 252 days is dangerously unstable.** Only ~12-13 tail observations at 95% confidence. Kondor et al. (2015) showed ~4,000 observations (~16 years) are needed for reliable ES estimation. Relative standard deviation of 20-30% with 250 observations under normality, worse under fat tails.
- **The correlation-based trade-blocking mechanism is entirely novel and untested.** No published study validates blocking BUY trades when average pairwise correlation exceeds a threshold. The claimed 20-30% drawdown reduction is an untested quantity.
- **ATR(14) with Wilder smoothing gives each new bar only ~7% weight**, making it slow to adapt during sudden regime shifts. EWMA volatility adapts faster. For crypto (60-80% annualized vol), ATR(7-10) is more appropriate.
- **Inverse-volatility weighting is not true equal-risk-contribution.** The spec's approach ignores correlations, which Carver calls "madness" in many portfolio contexts. The performance gap is modest for diverse asset class universes but exists.

### Current risk system gaps identified

Our 8 existing checks (position size, position weight, gross/net exposure, sector concentration, cash reserve, stop-loss, drawdown circuit breaker) are all **static, threshold-based** checks. They have zero awareness of:
1. Market volatility regime (no ATR/vol-based sizing)
2. Cross-asset correlation dynamics (no diversification measurement)
3. Tail risk distribution (no CVaR or expected shortfall)
4. Dynamic position sizing based on current conditions

The system enforces hard limits but does not adapt those limits to market conditions.

---

## Recommendations (Priority-Ordered)

### 1. ATR Risk Budgeting -- Implement FIRST (Priority 1)

**Why first:** Strongest empirical evidence, simplest implementation, highest bang-for-buck. Nearly universally adopted by systematic firms. Expected Sharpe improvement of 0.2-0.3 with minimal return sacrifice.

**What to build:**
- ATR-based position sizing: size positions inversely proportional to ATR so each contributes roughly equal risk
- ATR(14) as default, ATR(7-10) for crypto assets
- Consider EWMA volatility as faster-adapting alternative, especially for regime shifts
- 2x ATR stop-loss as baseline, backtest 1.5x-3x across all six asset classes (2.5-3x for crypto)
- Buffer rules: rebalance only when position deviation exceeds 20-25% of target weight (controls turnover on a $100k portfolio where transaction costs matter)
- Two-layer approach: ATR-based sizing within asset class clusters, correlation-aware weighting across clusters

**Integration point:** New `check_volatility_sizing()` in `limits.py` that validates proposed position sizes against ATR-derived targets. Modify `risk.toml` to add ATR parameters. This enhances rather than replaces existing checks.

### 2. Dynamic Correlation Monitoring -- Implement SECOND (Priority 2)

**Why not first despite original P0 tag:** The original spec's rolling window approach was fundamentally flawed (Engle proved it worst-in-class). Requires DCC-GARCH or at minimum EWMA (lambda 0.94-0.97), which is more complex to implement correctly.

**What to build:**
- DCC-GARCH correlation estimation (two-step: 39 univariate GARCH + 2 scalar DCC parameters, feasible in 30-60 seconds)
- EWMA correlation as fallback/comparison (simpler, still dramatically outperforms rolling windows)
- Use correlations to inform diversification scoring of proposed trades, NOT as a binary trade blocker (the blocking mechanism is untested)
- Adaptive VIX regime thresholds based on historical percentiles rather than fixed 20/25
- Forbes-Rigobon adjustment: use volatility-standardized returns to avoid heteroskedasticity bias
- HRP (Hierarchical Risk Parity) clustering for portfolio construction -- complements ATR sizing from step 1

**Integration point:** New `check_correlation_risk()` in `limits.py` that scores portfolio diversification and warns (but does not hard-block) when correlation concentration is elevated. Feed regime classification into the LLM context for decision-making.

### 3. CVaR Constraints -- Implement THIRD (Priority 3)

**Why third:** Estimation instability with 252 days is a real problem. Needs Filtered Historical Simulation (GARCH-filtered residuals rescaled to current volatility) at minimum, not raw historical sim. Best used as a binding constraint layered on top of ATR sizing, not as primary sizing tool.

**What to build:**
- Filtered Historical Simulation (Barone-Adesi et al. 1999) -- GARCH-filter residuals, rescale to current vol
- Extend lookback to 500+ days minimum (or use EWMA decay weighting, lambda=0.94)
- Named stress scenarios as mandatory overlay (COVID, 2022 rates, crypto winter) -- addresses cliff effect
- Bootstrap confidence intervals on CVaR estimates to quantify uncertainty
- Proportional scaling enforcement (CPPI-style), not binary blocking
- Consider Generalized Pareto Distribution fitting for tail stability

**Integration point:** New `check_portfolio_cvar()` in `limits.py` as portfolio-level constraint. Triggers proportional position reduction when portfolio CVaR exceeds threshold. Complementary to per-position ATR sizing.

### Cross-cutting concerns

- **Avoid correlated model failure:** Each layer should use somewhat independent information sources -- different lookback windows, different estimation methods. If all three rely on the same covariance estimates, a single estimation failure propagates through the entire system.
- **$100k minimum viable size:** Carver notes this is the minimum for diversified systematic trading. Consider binary/thresholded forecasting for position sizes. Transaction costs from frequent rebalancing accumulate meaningfully at this scale.
- **Anti-overfitting:** With only 3-5 major crises in 20 years, risk parameters are fit to 3-5 data points. Use regime-aware block bootstrap and sensitivity analysis over parameter ranges rather than point optimization. Test parameters calibrated on one asset class against others.

---

## Implementation Sequence

| Phase | Module | Effort | Expected Impact | Dependencies |
|-------|--------|--------|-----------------|--------------|
| **Phase 1** | ATR Risk Budgeting (3.2) | Medium | 35-50% drawdown reduction, +0.2-0.3 Sharpe | ATR indicators already computed in data pipeline |
| **Phase 2** | Correlation Monitoring (3.1) | High | Diversification awareness, regime-adaptive limits | Requires DCC-GARCH or EWMA library (arch package) |
| **Phase 3** | CVaR Constraints (3.3) | High | Tail risk protection, stress scenario coverage | Requires FHS implementation, extended data history |
| **Phase 4** | Integration & Validation | Medium | Layered system with independent failure modes | All three modules operational |

Phase 1 can be implemented immediately -- ATR(14) is already computed in the indicator pipeline. Phases 2 and 3 require new dependencies and more careful validation. Phase 4 ties the layers together with walk-forward testing.

---

## Beads Issues to Create

1. **Implement ATR-based position sizing in risk system** (Priority 1, Task)
   - Add `check_volatility_sizing()` to `limits.py`
   - ATR(14) default, ATR(7-10) for crypto
   - 20-25% deviation buffer before rebalancing
   - Integrate into `RiskManager.check_trade()`
   - Update `risk.toml` with ATR sizing parameters

2. **Implement ATR-calibrated stop-loss multipliers** (Priority 1, Task)
   - Replace static `default_stop_loss_pct` with ATR-multiplied stops
   - 2x ATR baseline, asset-class-dependent (2.5-3x for crypto)
   - Backtest 1.5x, 2x, 3x across all six asset classes
   - Update `check_stop_loss()` to validate ATR-relative stops

3. **Implement DCC-GARCH correlation estimation** (Priority 2, Task)
   - Replace rolling window correlation with DCC-GARCH (or EWMA fallback)
   - 39 univariate GARCH models + 2 scalar DCC parameters
   - Forbes-Rigobon adjustment using volatility-standardized returns
   - Add diversification scoring to risk checks (warn, not block)
   - Requires `arch` Python package dependency

4. **Implement adaptive VIX regime classification** (Priority 2, Task)
   - Replace fixed VIX thresholds (20/25) with historical percentile-based boundaries
   - Data-driven terciles (~17.8/23.1) showed 24% Sharpe improvement
   - Feed regime classification into LLM context and risk parameters
   - Regime drives position sizing and sector tilts

5. **Implement Filtered Historical Simulation CVaR** (Priority 3, Task)
   - GARCH-filter residuals, rescale to current volatility (Barone-Adesi 1999)
   - Extend lookback to 500+ days with EWMA decay weighting
   - Named stress scenario overlay (COVID, 2022 rates, crypto winter)
   - Proportional scaling enforcement (CPPI-style), not binary blocking
   - Bootstrap confidence intervals on estimates

# Quant Reference Repos

Cloned to `C:/Projects/quant-refs/` on 2026-03-30.
Source: https://github.com/wilsonfreitas/awesome-quant and https://github.com/georgezouq/awesome-ai-in-finance

## Immediate Use (HIGH priority)

### empyrical — metrics primitives
`C:/Projects/quant-refs/empyrical/` | 599K | quantopian/empyrical
Drop-in replacements for Sharpe, Sortino, Calmar, max drawdown, alpha/beta.
Works on plain return arrays. **Replace hand-rolled metrics in `trading/performance.py`.**

### alphalens-reloaded — signal IC validation
`C:/Projects/quant-refs/alphalens-reloaded/` | 697K | stefan-jansen/alphalens-reloaded
Information coefficient (IC), IC decay, quantile return spreads, turnover analysis.
Python 3.12 compatible (original quantopian/alphalens is not).
**Fill the Hunt→Validate gap — evaluate signal predictive content before backtest.**

### quantstats — tearsheets
`C:/Projects/quant-refs/quantstats/` | 3.9M | ranaroussi/quantstats
One-call tearsheet generation against arbitrary benchmark (SPY, 60/40).
**Drop into `/evaluate` to replace manual metric reporting.**

### mlfinlab — anti-overfitting implementations
`C:/Projects/quant-refs/mlfinlab/` | 2.8M | hudson-and-thames/mlfinlab
Combinatorial purged cross-validation (CPCV), Deflated Sharpe Ratio (DSR), triple-barrier
labeling, fractional differentiation, stationary bootstrap.
**Replace our hand-rolled CPCV/DSR in `backtest/robustness.py`.**
Key modules: `mlfinlab/cross_validation/`, `mlfinlab/backtest_statistics/`, `mlfinlab/labeling/`

### quant-trading — strategy reference implementations
`C:/Projects/quant-refs/quant-trading/` | 20M | je-suis-tm/quant-trading
9,554 stars. Pure Python. Pairs trading, Kalman filter spread, momentum, mean reversion.
**Directly relevant to GLD-SLV (Family 2) and SOXX-QQQ (Family 8) expansion.**

## Portfolio Construction (HIGH priority)

### Riskfolio-Lib — cross-strategy allocation
`C:/Projects/quant-refs/Riskfolio-Lib/` | 150M | dcajasn/Riskfolio-Lib
Hierarchical Risk Parity (HRP), Nested Clustered Optimization (NCO), Mean-CVaR,
Black-Litterman, risk budgeting.
**Implement proper correlation-aware allocation across 16 strategies (Track A/B split).**
Replaces the simple ATR-based sizing in current portfolio_optimizer.py.

### pysystemtrade — governance + FDM
`C:/Projects/quant-refs/pysystemtrade/` | 894M | robcarver17/pysystemtrade
Robert Carver's Systematic Trading framework in Python. Forecast Diversification Multiplier
(FDM), instrument weights, Sharpe-based position sizing, daily governance layer.
**Study the FDM for combining 16 strategies with realistic correlation-adjusted SR estimates.**
Also: working governance layer used in live IB trading — reference for our surveillance module.

## Architecture References (MEDIUM priority)

### qlib — LLM+ML alpha pipeline
`C:/Projects/quant-refs/qlib/` | 15M | microsoft/qlib
39,489 stars. Full ML pipeline: data → alpha factors → backtest → portfolio construction.
First-class LLM signal generation support. RD-Agent for automated R&D loops.
**Study for: where our research track could go, LLM signal integration patterns.**

### nautilus_trader — production engine
`C:/Projects/quant-refs/nautilus_trader/` | 165M | nautechsystems/nautilus_trader
21,492 stars. Rust-native, deterministic event-driven backtester + live engine.
Supports equities, crypto, forex, futures (our 39-asset universe).
**Study for: architectural patterns, deterministic replay, production-grade risk rules.**

### machine-learning-for-trading — Families 2-7 research
`C:/Projects/quant-refs/machine-learning-for-trading/` | 421M | stefan-jansen/machine-learning-for-trading
16,865 stars. Code for "ML for Algorithmic Trading" 2nd ed. Factor analysis, NLP signals,
HMM regime classification, CPCV examples.
**Primary reference for expanding into untested mechanism Families 2-7.**

### pyfolio — performance tearsheets
`C:/Projects/quant-refs/pyfolio/` | 28M | quantopian/pyfolio
Rolling Sharpe, drawdown periods, regime-split performance, factor exposures.
Note: uses pandas internally — needs thin Polars→pandas adapter at boundary.
**Supplement to quantstats for detailed post-trade analysis.**

## Recommended Integration Order

1. **empyrical** — smallest, highest leverage. Replace metrics in `trading/performance.py`.
2. **quantstats** — wire into `/evaluate` for benchmark-relative tearsheets.
3. **alphalens-reloaded** — add IC analysis to `/hypothesis` signal validation step.
4. **mlfinlab CPCV/DSR** — upgrade `backtest/robustness.py` with battle-tested implementations.
5. **Riskfolio-Lib HRP** — upgrade `scripts/portfolio_optimizer.py` allocation logic.
6. **pysystemtrade FDM** — study and adapt for multi-strategy combination math.
7. **quant-trading** — mine for Family 2 (mean reversion) and Family 8 expansion ideas.
8. **machine-learning-for-trading** — systematic reference for Families 2-7 research.

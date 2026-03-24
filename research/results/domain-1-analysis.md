# Domain 1 Analysis: Market Microstructure

**Date**: 2026-03-24
**Scope**: Briefs 1.1 (Order Timing), 1.2 (ATR vs Execution Risk), 1.3 (Signal Decay 24h Markets)
**Portfolio**: $100k paper, 39 instruments, ~$2,500 avg position size

---

## Key Findings

### 1. ATR(14) with Wilder's smoothing has a 9.55-day half-life -- dangerously slow for regime transitions

The current `_compute_atr()` in `indicators.py` uses `ewm_mean(alpha=1/14)` (Wilder's smoothing). During the COVID crash, this meant ATR could not reflect >50% of the new volatility regime until ~10 days in. An EMA-based ATR with `alpha=2/(n+1)` is **twice as fast** (~5-day half-life) with zero additional complexity. This is the single highest-impact code change identified.

### 2. Market impact is zero at this scale -- spread cost is the binding constraint

At $2,500 per position in SPY ($30B+ daily volume), our order represents <0.00001% of daily volume. Permanent impact: ~0.001 bps. The real cost is the bid-ask spread itself: 2-6 bps round-trip for US equity ETFs, rising to 12-33 bps for EM ETFs. No TWAP or algorithmic execution is justified -- MOC is the correct default for US instruments.

### 3. Conditional VIX-based sizing beats continuous ATR scaling

Bongaerts et al. (2020) showed continuous volatility targeting **fails** to improve performance and can increase drawdowns (realized-to-target vol ratio of 1.16 in US markets). Conditional targeting -- adjusting only at regime extremes using a 126-day VIX percentile rank -- consistently enhances Sharpe and reduces drawdowns. The current system fetches VIX (`context.py:_get_vix()`) but does not compute a percentile rank or use it for position sizing.

### 4. Crypto momentum decays slowly (1-6 days) but weekend positions carry asymmetric risk

Bitcoin daily returns predict forward returns 1-6 days ahead. The 15% crypto stop-loss in `risk.toml` is appropriate given this holding period. However, negative weekend crypto returns significantly predict Monday equity declines, while positive weekend returns have no cross-asset benefit. This creates an asymmetric risk for Friday-entered crypto positions that the risk manager does not account for.

### 5. International ETFs are the one asset class where execution timing matters

Stale NAV creates ~200 bps pricing deviations for international ETFs. Trading during the European-US overlap window (8:00-11:30 AM ET) reduces round-trip costs by 20-50 bps. This is the only execution timing optimization worth implementing.

### Confirmed vs Challenged

| Hypothesis | Verdict |
|---|---|
| ATR(14) is adequate for position sizing | **Challenged** -- 9.55-day half-life is too slow; EMA-ATR + VIX overlay recommended |
| Spread estimators (Roll, Corwin-Schultz) can proxy bid-ask spreads | **Challenged** -- correlation drops to 0.18 for liquid ETFs; direct NBBO data preferred |
| VIX regime classification drives sizing | **Confirmed** -- but conditional (threshold) approach beats continuous scaling |
| Crypto needs wider stops | **Confirmed** -- 15% stop-loss is consistent with 1-6 day momentum holding periods |
| Forex is cheap to trade | **Challenged** -- IBKR's $2 minimum commission = 16 bps round-trip on $2,500, making small forex positions disproportionately expensive |

---

## Recommendations (Priority-Ordered)

### R1. Switch ATR from Wilder's to EMA smoothing

- **What**: Change `_compute_atr()` from `alpha=1/period` to `alpha=2/(period+1)` (or equivalently use `span=period`). Halves the half-life from ~9.55 to ~5 days.
- **Expected impact**: Faster regime detection during vol spikes. Reduces the window where position sizes are dangerously stale. Direct improvement to drawdown control during transitions.
- **Complexity**: Low -- single-line change in `indicators.py`.
- **Files changed**: `src/llm_quant/data/indicators.py` (line 243, `_compute_atr` function)

### R2. Add VIX percentile rank to market context + conditional sizing overlay

- **What**: Compute 126-day VIX percentile rank in `build_market_context()`. Add it to `MarketContext`. Use it in the decision prompt for conditional position sizing: full size when VIX is 20th-80th percentile; reduce 25-30% above 80th; increase 10-15% below 20th.
- **Expected impact**: Bongaerts et al. showed conditional vol targeting consistently improves Sharpe and reduces max drawdown. Man Group achieved 500 bps excess return with similar approach.
- **Complexity**: Medium -- requires storing VIX history, computing percentile, adding field to MarketContext dataclass, updating prompt templates.
- **Files changed**: `src/llm_quant/brain/context.py`, `src/llm_quant/brain/models.py`, `config/prompts/decision.md.j2`

### R3. Add execution cost estimates to market context by asset class

- **What**: Embed realistic round-trip cost estimates per asset class in the decision context so the LLM can weigh friction against expected alpha. Use the research-derived table: US equity 5 bps, international 15-30 bps, EM 20-33 bps, fixed income 8-15 bps, crypto ETF 8-10 bps, forex 17-18 bps (at $2,500 size).
- **Expected impact**: Prevents the LLM from entering low-conviction trades where expected alpha is below friction. Particularly important for EM and forex where costs eat 20-50% of a typical daily move.
- **Complexity**: Low -- add a static lookup dict in config or context builder.
- **Files changed**: `src/llm_quant/brain/context.py`, `config/risk.toml` (new `[execution_costs]` section)

### R4. Add backtest friction assumptions to prevent overfitting

- **What**: When backtesting, apply minimum friction assumptions: 5 bps US equity, 20 bps international, 10 bps crypto ETF, 18 bps forex. Apply 1.5x multiplier for backtest-to-live gap (Quantopian study: backtest Sharpe has R^2 < 0.025 with live).
- **Expected impact**: Prevents deploying strategies that look good in backtest but fail live. Live drawdowns typically run 1.5-2x backtested.
- **Complexity**: Medium -- depends on backtest infrastructure being built.
- **Files changed**: Future backtesting module (not yet built)

---

## Risk Warnings

### Do NOT implement TWAP or algorithmic execution
At $2,500 position sizes, TWAP/VWAP adds engineering complexity with zero cost benefit. MOC is the correct default. The marginal cost difference is noise.

### Do NOT use spread estimators (Roll, Corwin-Schultz) for liquid ETFs
Cross-sectional correlation with actual spreads drops to 0.18 for liquid securities. For commodities, correlation is zero. If spread data is needed, use direct NBBO quotes.

### Do NOT assume linear spread scaling with VIX
The VIX-spread relationship is **convex**: March 2020 showed ~10x spread widening for ~6x VIX increase. ATR-based models that assume linear cost scaling will underestimate execution costs precisely when it matters most.

### Watch the forex minimum commission trap
IBKR's $2 minimum commission creates 16 bps round-trip friction on $2,500 forex positions. This makes small forex trades ~7x more expensive per notional than US equity ETFs. Either size forex positions larger ($10k+, where commission drops to ~3 bps) or accept the drag.

### Weekend crypto-equity spillover is asymmetric
Negative weekend crypto returns predict Monday equity declines; positive weekend crypto returns do NOT predict Monday equity gains. Holding crypto over weekends has a negatively skewed cross-portfolio risk profile.

---

## Beads Issues to Create

### Issue 1: Switch ATR computation to EMA smoothing
- **Title**: Switch ATR(14) from Wilder's to EMA smoothing for faster regime response
- **Description**: Current `_compute_atr()` in `indicators.py` uses `alpha=1/period` (Wilder's, half-life ~9.55 days). Change to `span=period` which uses `alpha=2/(period+1)` (EMA, half-life ~5 days). Research brief 1.2 shows Wilder's ATR missed >50% of COVID vol regime for ~10 days. Single-line change: replace `ewm_mean(alpha=alpha, ...)` with `ewm_mean(span=period, ...)` in `_compute_atr`. Also add `atr_14_ema` as the column name or replace `atr_14` in-place. Run existing tests to verify no regressions.
- **Priority**: P1
- **Type**: task

### Issue 2: Add VIX percentile rank to market context
- **Title**: Compute 126-day VIX percentile rank for conditional position sizing
- **Description**: Research brief 1.2 recommends conditional VIX-based sizing (Bongaerts et al. 2020): full size at 20th-80th percentile, reduce 25-30% above 80th, increase 10-15% below 20th. Implementation: (1) fetch 126 days of VIX history in `context.py`, (2) compute percentile rank, (3) add `vix_percentile` field to `MarketContext` in `models.py`, (4) update decision prompt to reference VIX percentile for sizing guidance. The 126-day lookback is the optimal memory per Wysocki (2025).
- **Priority**: P2
- **Type**: feature

### Issue 3: Add per-asset-class execution cost estimates to decision context
- **Title**: Embed execution cost estimates in market context for friction-aware decisions
- **Description**: Research domain 1 established round-trip friction by asset class: US equity 2-6 bps, international 7-18 bps, EM 12-33 bps, treasury 3-9 bps, HY 8-24 bps, crypto ETF 3-9 bps, forex 17-18 bps (at $2,500). Add an `[execution_costs]` section to `risk.toml` with these estimates. Include them in `MarketContext` so the LLM can compare expected alpha against friction before entering trades. Particularly important for EM/forex where friction can exceed daily alpha.
- **Priority**: P2
- **Type**: feature

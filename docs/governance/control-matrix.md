# Control Matrix Reference

Comprehensive failure-mode detection, escalation thresholds, and recovery criteria for the llm-quant portfolio management strategy.

Every failure mode follows the same escalation ladder:

1. **Detector** -- what metric or check to monitor
2. **Warning trigger** -- alert threshold; heightened scrutiny, no automatic action
3. **Hard stop trigger** -- halt new buys; sells and risk-reduction only
4. **Immediate action** -- mandatory steps when hard stop fires
5. **Full reset condition** -- criteria required before normal trading resumes

---

## Failure Modes

### 1. Regime Change

Market regime shifts that invalidate current positioning. The strategy is regime-conditional; operating under the wrong regime classification is among the highest-impact errors.

| Layer | Specification |
|-------|---------------|
| **Detector** | Rolling 21-day Sharpe ratio, trailing 21-day win rate, 21-day realized volatility -- all compared against the regime-specific baseline established at strategy deployment. |
| **Warning** | Rolling Sharpe drops 30% vs baseline **OR** win rate drops 15 percentage points **OR** realized vol spikes to 1.5x baseline. |
| **Hard stop** | Rolling Sharpe drops 50% vs baseline **OR** win rate drops 25 percentage points **OR** realized vol spikes to 2x baseline. |
| **Immediate action** | Reduce all position sizes to 50% of target weight. Tighten trailing stops by 25%. Re-run regime classification with current macro inputs (VIX, yield curve slope, SPY momentum). Log regime re-classification to strategy_changelog. |
| **Full reset** | Rolling Sharpe, win rate, and realized vol all return within 1 standard deviation of their regime baselines for 5 consecutive trading sessions. |

**Why it matters:** A mis-classified regime leads to systematically wrong position sizing and directional bias. The strategy assumes regime drives allocation; if the regime label is stale, every downstream decision inherits that error.

---

### 2. Alpha Decay

Strategy edge eroding over time. All quantitative edges decay as markets adapt; this detector catches gradual erosion before it becomes a drawdown event.

| Layer | Specification |
|-------|---------------|
| **Detector** | Rolling 63-day (one quarter) Sharpe ratio divided by full-history Sharpe ratio. This ratio measures whether recent performance is keeping pace with historical expectations. |
| **Warning** | Rolling/full-history Sharpe ratio falls below 0.60. |
| **Hard stop** | Rolling/full-history Sharpe ratio falls below 0.40. |
| **Immediate action** | Flag the strategy for formal review. Reduce new position sizing by 50%. Document the decay onset date and magnitude in strategy_changelog. Do not enter new high-conviction positions until review is complete. |
| **Full reset** | Rolling 63-day Sharpe recovers to within 20% of the historical average and sustains that level for 21 or more consecutive trading days. |

**Why it matters:** Alpha decay is slow and insidious. By the time it shows up in cumulative P&L, the drawdown is already material. Monitoring the ratio of recent-to-historical risk-adjusted returns provides early warning.

---

### 3. Crowding / Capacity

Too many market participants crowding into the same positions, eroding expected returns and amplifying drawdowns during unwinds.

| Layer | Specification |
|-------|---------------|
| **Detector** | Would monitor ETF flow data, short interest changes, options open interest concentration, and correlation clustering among portfolio holdings. |
| **Status** | **DEFERRED** -- not implemented. Paper trading executes with perfect fills and zero market impact. Crowding detection is meaningless without real order flow interaction. |
| **When to implement** | Before any transition to live trading. At minimum: track ETF creation/redemption data, monitor short interest spikes in held positions, flag when portfolio holdings cluster in the top decile of crowded names. |

---

### 4. Execution Drift / TCA (Transaction Cost Analysis)

Slippage between intended and actual fill prices, measuring implementation shortfall.

| Layer | Specification |
|-------|---------------|
| **Detector** | Would compare decision-time price (mid-quote at signal generation) vs actual fill price. Measure implementation shortfall per trade and as a rolling aggregate. |
| **Status** | **DEFERRED** -- not implemented. Paper trading assumes perfect fills at the last available close price. There is no slippage by definition. |
| **When to implement** | Before any transition to live trading. At minimum: log intended vs fill price, compute per-trade and rolling implementation shortfall, alert when cumulative slippage exceeds 50bps per month. |

---

### 5. Hidden Data Issues

Stale data, price gaps, feed problems, and data corruption that could lead to trading on false signals.

| Layer | Specification |
|-------|---------------|
| **Detector** | For each symbol in the 39-asset universe: check data freshness (age of most recent price), scan for single-day price gaps exceeding 20%, verify prices remain above plausibility floor ($0.01). |
| **Warning** | Any symbol has data older than 1 trading day **OR** a single-day price gap >20% is detected (could be legitimate event or data error). |
| **Hard stop** | Any symbol has data older than 3 trading days **OR** any price falls below $0.01 (data corruption). |
| **Immediate action** | Exclude all affected symbols from the tradeable universe. Do not generate signals for excluded symbols. Flag affected symbols for manual investigation. If affected symbols have open positions, tighten stops but do not force-close without valid price data. |
| **Full reset** | All 39 symbols have price data fresher than 24 hours. No plausibility violations detected. Any previously flagged gaps have been investigated and documented. |

**Why it matters:** The strategy cannot distinguish between a real price move and a data error. Trading on corrupt data can trigger false signals in either direction. Data quality is a prerequisite for every other check.

---

### 6. Process Drift

Unauthorized or undocumented changes to configuration files, prompt templates, or strategy parameters.

| Layer | Specification |
|-------|---------------|
| **Detector** | SHA-256 hashes of all files in `config/` are tracked in DuckDB. On each trading session, current hashes are compared against stored values. |
| **Warning** | Any config file hash has changed since the last session without a corresponding `strategy_changelog` entry documenting the change. |
| **Hard stop** | Multiple config file changes detected in a single session without documentation **OR** changes to risk limit parameters (`max_position_pct`, `max_trade_pct`, `max_gross_exposure`, etc.) without explicit review. |
| **Immediate action** | Log all detected changes to `strategy_changelog` with before/after values. Require human review of changes before resuming trading. Revert any risk limit changes that were not documented. |
| **Full reset** | All configuration changes documented in `strategy_changelog` with rationale. Changes reviewed and approved. Hash database updated to reflect new approved state. |

**Why it matters:** Configuration changes are strategy changes. An undocumented parameter tweak is indistinguishable from overfitting or unauthorized risk-taking. The hash-based audit trail ensures every change is intentional and traceable.

---

### 7. Risk Drift

Post-trade portfolio exposure or concentration exceeding the pre-trade risk limits. This should be caught by pre-trade checks, but market moves between sessions can push exposures beyond limits.

| Layer | Specification |
|-------|---------------|
| **Detector** | After each session, compare actual portfolio metrics against configured limits: gross exposure vs 200% cap, net exposure vs 100% cap, per-position weight vs 10% cap (5% crypto, 8% forex), sector concentration vs 30% cap, cash reserve vs 5% floor. |
| **Warning** | Any metric reaches within 10% of its limit (e.g., sector concentration at 27% when limit is 30%, or cash reserve at 5.5% when floor is 5%). |
| **Hard stop** | Any metric exceeds its configured limit. This should not happen if pre-trade risk checks are functioning correctly -- an exceedance indicates either a bug in risk checks or an inter-session market move. |
| **Immediate action** | Flag the breaching position(s) for immediate rebalancing. Investigate why pre-trade checks did not prevent the breach. If caused by market moves, rebalance at next session open. If caused by a risk check bug, halt trading until the bug is fixed. |
| **Full reset** | All exposure and concentration metrics within 90% of their respective limits after rebalancing. Root cause documented if the breach was due to a system issue. |

**Why it matters:** Risk limits exist to bound losses. A limit that can be silently breached provides false confidence. Post-trade verification is the backstop that catches what pre-trade checks miss.

---

### 8. Operational Fragility

System health issues that compromise the ability to trade, record, or verify portfolio state.

| Layer | Specification |
|-------|---------------|
| **Detector** | Check for: snapshot recency (days since last portfolio snapshot), price data freshness (age of most recent market data), hash chain integrity (DuckDB ledger verification via `pq verify`). |
| **Warning** | No portfolio snapshot saved for >3 calendar days **OR** market data prices are >48 hours old. |
| **Hard stop** | Hash chain verification fails (ledger integrity compromised -- snapshots may have been tampered with or corrupted). |
| **Immediate action** | Investigate system state immediately. For stale snapshots: determine if sessions were skipped or if writes are failing. For stale prices: check Yahoo Finance connectivity and data pipeline. For hash chain failure: **do not trade** -- the integrity of the historical record is in question. |
| **Full reset** | Hash chain verifies clean (`pq verify` passes). Portfolio snapshots are current (today or last trading day). All price data is fresh (<24 hours old on trading days). |

**Why it matters:** The portfolio ledger is the single source of truth. If the hash chain is broken, historical NAV calculations, performance metrics, and risk measurements are all suspect. Operational health is not optional.

---

## Kill Switches

Kill switches are binary circuit breakers. **Any single kill switch triggers a full halt** -- no new buys are permitted; only sells and risk-reduction trades are allowed. Kill switches are independent of the 8 failure modes above and operate as a final safety net.

| # | Kill Switch | Trigger Condition | Rationale |
|---|------------|-------------------|-----------|
| 1 | **NAV Drawdown** | NAV drawdown from peak exceeds 15% | Hard constraint from strategy mandate. A 15% drawdown at the portfolio level indicates systematic failure, not a single bad trade. |
| 2 | **Single-Day Loss** | Single-day portfolio loss exceeds 5% of NAV | A 5% single-day move implies either extreme market dislocation or a catastrophic position sizing error. Either warrants a full stop. |
| 3 | **Consecutive Losses** | 5 consecutive losing trades | Five losses in a row indicates the signal generation process is miscalibrated for current market conditions. Stop and reassess. |
| 4 | **Concentration Correlation** | Portfolio correlation to any single asset exceeds 85% | Effective diversification has collapsed. The portfolio is behaving as a single bet regardless of how many positions it holds. |
| 5 | **Data Blackout** | No fresh market data for more than 72 hours | Trading without data is speculation, not systematic trading. A 72-hour blackout means multiple sessions have passed without price updates. |
| 6 | **Risk Check Cascade** | 3 consecutive risk check failures across sessions | The risk management system itself is malfunctioning. If pre-trade checks are repeatedly failing, either the portfolio is in an extreme state or the checks have bugs. |

### Kill Switch Recovery

When a kill switch fires:

1. **Immediate**: All pending buy orders are cancelled. Only sell and risk-reduction trades are permitted.
2. **Investigation**: Document the trigger event, root cause, and current portfolio state in `strategy_changelog`.
3. **Remediation**: Address the root cause (not the symptom). A drawdown kill switch fired because of bad positioning, not because of the drawdown itself.
4. **Reset criteria**: The specific kill switch condition must be resolved (e.g., drawdown recovers below 12%, data feeds restored, risk checks pass cleanly for 3 consecutive sessions).
5. **Re-entry**: Resume trading at 50% of normal position sizing for the first 5 sessions after reset. Ramp back to full sizing only if no further triggers occur.

---

## Summary Matrix

| # | Failure Mode | Warning Trigger | Hard Stop Trigger | Implemented |
|---|-------------|-----------------|-------------------|-------------|
| 1 | Regime Change | Sharpe -30%, WR -15pp, vol 1.5x | Sharpe -50%, WR -25pp, vol 2x | Yes |
| 2 | Alpha Decay | Rolling/full Sharpe < 0.60 | Rolling/full Sharpe < 0.40 | Yes |
| 3 | Crowding / Capacity | -- | -- | Deferred |
| 4 | Execution Drift / TCA | -- | -- | Deferred |
| 5 | Hidden Data Issues | Stale >1d, gaps >20% | Stale >3d, price < $0.01 | Yes |
| 6 | Process Drift | Undocumented hash change | Multiple undocumented changes | Yes |
| 7 | Risk Drift | Within 10% of any limit | Any limit exceeded | Yes |
| 8 | Operational Fragility | No snapshot >3d, prices >48h | Hash chain failure | Yes |

| # | Kill Switch | Trigger |
|---|------------|---------|
| 1 | NAV Drawdown | >15% from peak |
| 2 | Single-Day Loss | >5% of NAV |
| 3 | Consecutive Losses | 5 in a row |
| 4 | Concentration Correlation | >85% to single asset |
| 5 | Data Blackout | >72 hours stale |
| 6 | Risk Check Cascade | 3 consecutive failures |

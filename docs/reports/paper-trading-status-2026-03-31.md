# Paper Trading Status Report — 2026-03-31

**Report date:** 2026-03-31
**Reporting agent:** Paper Trading Monitor
**Strategies monitored:** 14 (12 Track A, 2 Track D)
**DuckDB status:** No database file found — all tracking is via on-disk YAML artifacts only

---

## Executive Summary

Fourteen strategies have active paper-trading.yaml files across two tracks. No trades have been executed on any strategy. The program is in an early accumulation phase with the entire book flat, which is regime-appropriate behavior given the risk-off market environment (SPY -1.79% on 2026-03-26, -1.71% on 2026-03-27).

**Key concern:** Zero trades across all 14 strategies after up to 5 calendar days. The 50-trade gate (Track A) and 30-trade gate (Track D) are the critical bottleneck. At the current pace, trade count gates will not be met within the minimum calendar day windows. This is not a defect — strategies are correctly flat in a credit-stressed, risk-off environment — but it means paper trading periods will likely need to extend beyond the minimum calendar days.

**Governance gaps identified:**
1. Nine Batch 2 strategies (started 2026-03-30) have `backtest_baseline: null` — paper trading was started before backtest lifecycle artifacts were generated on disk.
2. Trade count gates vary inconsistently across strategies (10, 20, 50) — the deployment plan requires 50 for all Track A strategies per model-promotion-policy.md.
3. No DuckDB database exists — all portfolio state is tracked in YAML files only, meaning operational systems 6-7 (portfolio_persistence, hash_chain_integrity) cannot be tested until the database is initialized.

---

## Strategy Detail

### Batch 1 — Track A (started 2026-03-26)

```yaml
- slug: lqd-spy-credit-lead
  track: A
  batch: 1
  family: F1 (Cross-Asset Information Flow)
  mechanism: IG bond 5d return -> SPY entry/exit
  start_date: 2026-03-26
  days_elapsed: 5 (gate: 30)
  trades_executed: 0 (gate: 50)
  rolling_sharpe_21d: n/a (no trades)
  rolling_sharpe_63d: n/a
  max_drawdown: 0.0%
  cumulative_return: 0.0%
  current_nav: $100,000
  backtest_sharpe: 1.2502
  backtest_maxdd: 12.4%
  operations_tested: 4/8 (data_fetching, indicator_computation, signal_generation, performance_reporting)
  incidents: none
  gate_status: on_track
  estimated_promotion_date: 2026-04-25 (calendar) / TBD (trade count dependent)
  notes: >
    Strategy correctly flat. LQD 5d return was -1.20% on day 1 (exit signal) and
    -0.21% on day 2 (neutral). Entry requires LQD 5d >= +0.5%, which requires
    credit recovery. Daily log has 3 entries (2 complete, 1 pending for 2026-03-30).
    Gate file shows min_trades: 10 — this conflicts with model-promotion-policy.md
    which requires 50. Should be corrected to 50.

- slug: soxx-qqq-lead-lag
  track: A
  batch: 1
  family: F8 (Non-Credit Lead-Lag)
  mechanism: SOXX 5d return -> QQQ entry
  start_date: 2026-03-26
  days_elapsed: 5 (gate: 30)
  trades_executed: 0 (gate: 50)
  rolling_sharpe_21d: n/a
  rolling_sharpe_63d: n/a
  max_drawdown: 0.0%
  cumulative_return: 0.0%
  current_nav: $100,000
  backtest_sharpe: 0.8610
  backtest_maxdd: 14.4%
  operations_tested: 4/8 (data_fetching, indicator_computation, signal_generation, performance_reporting)
  incidents: none
  gate_status: on_track
  estimated_promotion_date: 2026-04-25 (calendar) / TBD (trade count dependent)
  notes: >
    Strategy correctly flat. SOXX 5d return was -3.34% and -2.72% on days 1-2,
    far below the +2% entry threshold. Entry requires SOXX positive momentum.
    Gate file shows min_trades: 50 — consistent with policy. Daily log has 3
    entries (2 complete, 1 pending).
```

### Batch 1 — Track A (started 2026-03-30)

```yaml
- slug: gld-slv-mean-reversion-v4
  track: A
  batch: 1
  family: F2 (Mean-Reversion Pairs)
  mechanism: GLD/SLV ratio z-score pairs trading
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 20)
  rolling_sharpe_21d: n/a
  rolling_sharpe_63d: n/a
  max_drawdown: 0.0%
  cumulative_return: 0.0%
  current_nav: $100,000
  backtest_sharpe: 1.1967
  backtest_maxdd: 9.6%
  operations_tested: 0/8
  incidents: none
  gate_status: on_track
  estimated_promotion_date: 2026-04-29 (calendar) / TBD (trade count dependent)
  notes: >
    Just started. Has full backtest baseline and spec_hash on file. Uses
    consensus windows [60, 90, 120] with 2.0 std Bollinger Bands. Most
    decorrelated strategy in the book (metals pairs, no equity exposure when flat).
    Gate file shows min_trades: 20. This is lower than the 50 required by
    model-promotion-policy.md — needs clarification. The deployment plan
    (Section 3) explicitly states ">= 50" and calls out the lqd-spy file's
    min_trades: 10 as a likely typo.
```

### Batch 2 — Track A (started 2026-03-30)

All 9 strategies in Batch 2 share identical characteristics: started 2026-03-30, zero trades, zero days of data, all 8 operational systems untested, and `backtest_baseline: null` (no on-disk backtest artifacts).

```yaml
- slug: agg-spy-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: AGG 5d return -> SPY
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 1.145)
  backtest_maxdd: null (reported: 8.4%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30 (calendar) / TBD
  notes: "No backtest artifacts on disk. Reported Sharpe from research-tracks.md summary table only."

- slug: agg-qqq-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: AGG 5d return -> QQQ
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 1.080)
  backtest_maxdd: null (reported: 11.2%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: vcit-qqq-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: VCIT 5d return -> QQQ
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 1.037)
  backtest_maxdd: null (reported: 14.5%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: lqd-qqq-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: LQD 5d return -> QQQ
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 1.023)
  backtest_maxdd: null (reported: 13.7%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: emb-spy-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: EMB 5d return -> SPY
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 1.005)
  backtest_maxdd: null (reported: 9.1%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: hyg-spy-5d-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: HYG 5d return -> SPY
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 0.913)
  backtest_maxdd: null (reported: 14.7%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: agg-efa-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: AGG 5d return -> EFA
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 0.860)
  backtest_maxdd: null (reported: 10.3%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: hyg-qqq-credit-lead
  track: A
  batch: 2
  family: F1
  mechanism: HYG 5d return -> QQQ
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 10*)
  backtest_sharpe: null (reported: 0.867)
  backtest_maxdd: null (reported: 13.4%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: "No backtest artifacts on disk."

- slug: spy-overnight-momentum
  track: A
  batch: 2
  family: F5 (Calendar/Seasonal)
  mechanism: Overnight gap momentum on SPY
  start_date: 2026-03-30
  days_elapsed: 1 (gate: 30)
  trades_executed: 0 (gate: 20)
  backtest_sharpe: null (reported: 1.043)
  backtest_maxdd: null (reported: 8.7%)
  operations_tested: 0/8
  incidents: none
  gate_status: at_risk
  estimated_promotion_date: 2026-04-30
  notes: >
    No backtest artifacts on disk. Family 5 mechanism — independent of credit
    lead-lag signals. Higher expected trade frequency (overnight gap fires more
    often than credit signals). Key data dependency: requires accurate SPY open
    prices from yfinance. Gate file shows min_trades: 20.
```

*\* Batch 2 strategies show `min_trades: 10` in their gate_criteria, which conflicts with the 50-trade requirement in model-promotion-policy.md. The deployment plan (Section 3) explicitly flags this as incorrect.*

### Track D — Sprint Alpha (started 2026-03-31)

```yaml
- slug: tlt-tqqq-sprint
  track: D
  batch: n/a (Track D experimental)
  family: F1 (re-expression of TLT->QQQ signal into TQQQ)
  mechanism: TLT lead-lag -> TQQQ (3x leveraged QQQ)
  start_date: 2026-03-31
  days_elapsed: 0 (gate: 60)
  trades_executed: 0 (gate: 30)
  rolling_sharpe_21d: n/a
  rolling_sharpe_63d: n/a
  max_drawdown: 0.0%
  cumulative_return: 0.0%
  current_nav: $100,000
  backtest_sharpe: 1.4345
  backtest_maxdd: 12.7%
  backtest_cpcv_oos_sharpe: 1.6277
  operations_tested: 0/8
  incidents: none
  gate_status: on_track
  estimated_promotion_date: 2026-05-30 (60-day clock)
  mar_kill_check_date: 2026-06-29 (90-day MAR >= 1.0 gate)
  notes: >
    Initialized today. Extraordinary backtest: OOS Sharpe (1.63) exceeds IS Sharpe
    (1.43), suggesting genuine signal rather than overfitting. Strategy params:
    TLT leader, TQQQ follower, 3-day lag, 10-day signal window, entry at +1%,
    exit at -0.5%, 30% target weight. Max hold 5 calendar days enforced.
    No daily log entries yet.

- slug: btc-momentum-v2
  track: D
  batch: n/a (Track D experimental)
  family: F5/F8 (time-series momentum)
  mechanism: BTC-USD multi-timeframe momentum with SMA50 trend filter
  start_date: 2026-03-31
  days_elapsed: 0 (gate: 60)
  trades_executed: 0 (gate: 20)
  rolling_sharpe_21d: n/a
  rolling_sharpe_63d: n/a
  max_drawdown: 0.0%
  cumulative_return: 0.0%
  current_nav: $100,000
  backtest_sharpe: 0.9604
  backtest_maxdd: 2.8%
  backtest_cpcv_oos_sharpe: 0.7119
  operations_tested: 0/8
  incidents: none
  gate_status: on_track
  estimated_promotion_date: 2026-05-30 (60-day clock)
  mar_kill_check_date: 2026-06-29 (90-day MAR >= 1.0 gate)
  notes: >
    Initialized today. DSR (0.9376) is below the Track D gate of 0.90 — very
    close but technically failing. This should be flagged for review. Low maxDD
    (2.76%) suggests conservative trend filter is effective but may limit trade
    count. Used $10M synthetic capital in backtest; real deployment uses
    fractional shares. No daily log entries yet.
```

---

## Summary Dashboard

```
report_date: 2026-03-31

summary:
  total_strategies: 14
  track_a: 12
  track_d: 2
  on_track: 4
  at_risk: 9
  gate_failed: 0
  flagged: 1 (btc-momentum-v2 DSR below gate)

trade_count_status:
  total_trades_all_strategies: 0
  strategies_with_trades: 0
  trade_count_bottleneck: true
  regime: risk_off (credit-stressed, equity selloff)

operational_readiness:
  batch_1_ops_tested: "4/8 (lqd-spy, soxx-qqq), 0/8 (gld-slv)"
  batch_2_ops_tested: "0/8 (all 9 strategies)"
  track_d_ops_tested: "0/8 (both strategies)"
  database_initialized: false
  systems_blocked_by_db: "portfolio_persistence, hash_chain_integrity"

next_milestones:
  - date: 2026-04-07
    action: "First weekly trade count review for Batch 1 (lqd-spy, soxx-qqq, gld-slv)"
  - date: 2026-04-14
    action: "Second weekly review — verify operational systems checklist progress"
  - date: 2026-04-25
    action: "Batch 1 30-day clock (lqd-spy, soxx-qqq) — promotion decision IF 50+ trades"
  - date: 2026-04-29
    action: "gld-slv-v4 30-day clock"
  - date: 2026-04-30
    action: "Batch 2 30-day clock — promotion decision IF 50+ trades"
  - date: 2026-05-30
    action: "Track D 60-day clock (tlt-tqqq-sprint, btc-momentum-v2)"
  - date: 2026-06-29
    action: "Track D 90-day MAR kill check"
```

---

## Issues and Action Items

### Critical

1. **No DuckDB database** — The database file (`data/llm_quant.duckdb`) does not exist. Run `pq init` to create schema + configs. Without this, operational systems 6-7 (portfolio_persistence, hash_chain_integrity) cannot be tested, and the 8/8 operational systems gate cannot pass.

2. **Batch 2 missing backtest artifacts** — All 9 Batch 2 strategies have `backtest_baseline: null`. Per deployment plan Section 2.2: "Before paper trading can formally start, each strategy needs (a) data/strategies/<slug>/ directory with lifecycle YAMLs, and (b) the /backtest and /robustness commands run to generate on-disk experiment artifacts." Paper trading clock is running but the lifecycle is incomplete.

### High

3. **Trade count gate inconsistency** — Six strategies show `min_trades: 10` in their gate_criteria YAML, which contradicts `model-promotion-policy.md` requirement of 50 trades. The deployment plan (Section 3) calls this out explicitly. All Track A strategies should show `min_trades: 50` for consistency with the promotion policy.

4. **btc-momentum-v2 DSR below gate** — Backtest DSR is 0.9376, below the Track D minimum of 0.90. It passes (barely), but this is the weakest integrity metric in the Track D pipeline. Watch closely.

### Medium

5. **Zero trades across all strategies** — Expected in risk-off regime, but creates a hard constraint on promotion timelines. The 50-trade gate (not the 30-day calendar gate) will determine actual promotion dates. If credit recovery takes 2+ weeks, the earliest Batch 1 promotion could slip from 2026-04-25 to May.

6. **Daily log updates stale** — Batch 1 strategies (lqd-spy, soxx-qqq) have pending entries for 2026-03-30 that were never completed. gld-slv has no daily log entries at all. Daily log discipline needs to be enforced.

7. **Track D max hold enforcement** — Both Track D strategies specify `max_hold_days: 5` in gate_criteria but there is no automated enforcement mechanism visible. This constraint must be enforced in the execution pipeline before Track D enters canary.

### Low

8. **spy-overnight-momentum data dependency** — Requires accurate SPY open prices from yfinance. This should be validated before first signal generation.

---

## Track D Special Attention

### tlt-tqqq-sprint (SR=1.43 in backtest)

- **Status:** Paper trading file created today (2026-03-31). Not yet running — no daily log entries.
- **Backtest quality:** Exceptional. CPCV OOS Sharpe (1.63) > IS Sharpe (1.43), which is the strongest anti-overfitting signal possible. 121 trades in backtest provides good statistical power.
- **Key risks:** Beta decay on TQQQ (3x daily reset), volatility drag on multi-day holds, path dependency in drawdowns. The 5-day max hold constraint is designed to mitigate these.
- **Gate timeline:** 60-day paper minimum = earliest promotion 2026-05-30. 90-day MAR kill check = 2026-06-29.
- **Action needed:** Begin daily signal computation and log entries starting 2026-03-31 (today).

### btc-momentum-v2 (SR=0.96 in backtest)

- **Status:** Paper trading file created today (2026-03-31). Not yet running — no daily log entries.
- **Backtest quality:** Acceptable but weaker. OOS Sharpe (0.71) shows significant IS/OOS degradation (0.96 -> 0.71). DSR (0.9376) barely passes Track D's 0.90 minimum. Only 42 trades in backtest — lower statistical confidence.
- **Key risks:** Low trade count in backtest (42) means paper trading sample will be critical. The very low maxDD (2.76%) may be an artifact of the 3-year window excluding 2022 crash — real deployment will face more volatile conditions.
- **Gate timeline:** Same as tlt-tqqq-sprint. 60-day clock, 90-day MAR check.
- **Action needed:** Begin daily signal computation. Monitor DSR closely — if paper DSR drops below 0.90, strategy should be paused for review.

---

## Beads Issues (Paper Trading Related)

| Issue ID | Title | Status |
|----------|-------|--------|
| llm-quant-7ud3 | Paper trade top 2 Track D strategies (60-day validation) | open |
| llm-quant-36sl | Promote Track D strategies to live portfolio (10% allocation) | open (blocked by 7ud3) |
| llm-quant-sxvd | Promote GLD-SLV v4 to production after paper track record | open |
| llm-quant-anf0 | Track C paper trading deployment | open |

---

## Recommendations

1. **Initialize DuckDB immediately** (`pq init`) — unblocks 2 of 8 operational system tests for all strategies.
2. **Run `/backtest` and `/robustness`** for each Batch 2 strategy to generate on-disk artifacts before the paper trading clock runs out. Prioritize `agg-spy-credit-lead` (highest reported Sharpe in Batch 2) and `spy-overnight-momentum` (independent mechanism, likely higher trade frequency).
3. **Correct min_trades gates** in all YAML files to match model-promotion-policy.md (50 for Track A).
4. **Establish daily logging discipline** — at minimum, update the Batch 1 strategies (lqd-spy, soxx-qqq, gld-slv) with end-of-day data each market day.
5. **Begin Track D daily monitoring** — both strategies need first daily log entries today.
6. **Monitor trade count weekly** — first review 2026-04-07. If zero trades persist through 2026-04-14, assess whether regime shift is likely before the 30-day window closes.

---

*Report generated: 2026-03-31 by Paper Trading Monitor agent.*
*Next report due: 2026-04-07 (first weekly review).*

# Dormant Paper-Trading Strategies — Investigation Report

**Date:** 2026-04-17
**Beads:** llm-quant-9kbq (P2 bug)
**Investigator:** dormant-strategies-debugger (apr17-cleanup team)

## Summary

Of the 45 `paper-trading.yaml` artifacts on disk, 37 are actively logging daily signals and 8 are dormant (zero entries in `daily_log` / `trades` as of 2026-04-17; most recent data would be 2026-04-16).

**Root cause of dormancy in every case:** the 8 dormant strategies are not registered in `scripts/run_paper_batch.py`. The batch runner only dispatches a hard-coded set of ~37 strategies via `LEAD_LAG_PARAMS`, `regime_configs`, `ratio_configs`, `pair_mr_configs`, and a few one-off special signal functions. There is no exception or data bug — the scripts simply do not iterate over these 8 slugs.

After investigation, **1 strategy was fixable via a small config-only change** (registered in the batch runner with params from its frozen spec). The remaining 7 cannot be revived without additional work — either their research prerequisites failed (blocked by governance) or their signal logic has no implementation in the batch runner and would require either a new signal generator function (bespoke strategy code — out of scope for this pass) or a retire/decommission decision.

## Dormant Strategies Table

| Slug | Status (YAML) | Last Signal | Cause | Category | Suggested Fix |
|------|---------------|-------------|-------|----------|---------------|
| `tlt-tqqq-sprint` | active | NONE | Not dispatched by `run_paper_batch.py`. Signal is a vanilla lead-lag identical in shape to existing entries in `LEAD_LAG_PARAMS`. | **Config** | **FIXED** — added to `LEAD_LAG_PARAMS` with params from frozen spec (TLT, TQQQ, 10, 0.01, -0.005, 0.30). |
| `btc-momentum-v2` | active | NONE | BTC-USD not in `ALL_SYMBOLS`; uses `trend_following` multi-timeframe consensus logic with no signal generator in batch runner. Also: fetcher not known to handle BTC-USD in current shared-data path. | **Code** | Add `BTC-USD` to `ALL_SYMBOLS` and write `signal_btc_momentum_v2()` generator (multi-timeframe SMA consensus + trend filter). Non-trivial (~40 LOC). Defer to a follow-up task. |
| `llm-alpha` | initialized | NONE | Requires FOMC/macro LLM-scored inputs (`fomc_semantic_diff_score`, `narrative_surprise_score`, `communication_complexity_score`). No LLM-scored cache in production and no proxy signal implemented. | **Code (hard)** | Needs LLM scoring pipeline OR Google Trends proxy implementation — both out of scope for batch runner. Recommend: retire paper track until scoring infrastructure is built; re-init when backtest is run against cached scores. |
| `risk-premium` | initialized | NONE | Requires VRP (VIX minus realized vol), HYG/IEF credit carry, and multi-asset risk parity construction. No signal generator in batch runner. | **Code** | Write `signal_risk_premium()` with VRP + credit carry + risk parity overlay (~80 LOC). Non-trivial. Defer to follow-up task. |
| `cross-asset-lead-lag` | initialized | NONE | Requires TLT lead + HYG/LQD credit lead + JPY carry unwind detection. JPY feed not in `ALL_SYMBOLS`. No signal generator. | **Code + Data** | Needs JPY data (JPY=X or similar) and a composite lead-lag signal. The two non-JPY channels overlap with existing credit-lead / rate-momentum strategies already in paper, so marginal value is limited. **Suggest retire** unless JPY channel is specifically desired. |
| `crypto-microstructure` | initialized | NONE | Requires BTC-SPY rolling correlation regime, ETH/BTC rotation signal, and 20-day crypto realized vol sizing. BTC-USD, ETH-USD not in `ALL_SYMBOLS`. No signal generator. | **Code + Data** | Needs crypto price feeds and a bespoke generator. Backtest maxDD was 34.9% (well above 15% mandate) — the robustness file explicitly flags high risk of gate failure during paper trading. **Suggest retire** unless a Track D/crypto-specific deployment is planned. |
| `momentum-regime` | **retired 2026-04-17** | NONE | Robustness gate FAILED (`DSR 0.713 < 0.95`, `PBO 0.657 > 0.10`, `parameter_stability 0% < 50%`). Paper trading cannot proceed. | **Config (by design)** | **RETIRED** — `retirement.yaml` written; status flipped to `retired` in `paper-trading.yaml`. Family dead — no v3 planned. |
| `momentum-regime-v2` | **retired 2026-04-17** | NONE | Robustness gate FAILED (`DSR 0.651 < 0.95`, `PBO 0.692 > 0.10`). Hypothesis falsified on 5 of 6 criteria. | **Config (by design)** | **RETIRED** — `retirement.yaml` written; status flipped to `retired` in `paper-trading.yaml`. Supersedes v1; family closed. |

## Categorized Breakdown

### Config issue (fixable in-place)
1. `tlt-tqqq-sprint` — **FIXED** in this session.

### Governance block (correctly retired by robustness gate)
2. `momentum-regime` — robustness FAIL. Recommend formal retirement.
3. `momentum-regime-v2` — robustness FAIL. Recommend formal retirement.

### Code + data issue (signal generator missing, defer to follow-up tasks)
4. `btc-momentum-v2` — needs BTC-USD symbol + trend-following generator.
5. `risk-premium` — needs VRP + credit carry + risk parity generator.
6. `cross-asset-lead-lag` — needs JPY feed + composite lead-lag generator.
7. `crypto-microstructure` — needs BTC-USD / ETH-USD feeds + microstructure generator.

### Blocked by infrastructure prerequisite
8. `llm-alpha` — requires LLM-scoring pipeline (FOMC/macro text scoring with caching). Out of scope for batch runner.

## Revive vs Retire Recommendations

### Safe to revive (now)
- **`tlt-tqqq-sprint`** — DONE this session. Frozen spec is clean, backtest is strong (Sharpe 1.43, CPCV OOS 1.63), Track D gates passed. Will begin logging on next `run_paper_batch.py` run.

### Revive with follow-up work (non-trivial, deferred)
- **`btc-momentum-v2`** — backtest is clean (Sharpe 0.96, CPCV 0.71, passes Track D). Worth the generator effort for Track D diversification.
- **`risk-premium`** — mandate-aligned, non-trivial but the VRP + credit carry signals are well-documented in the spec.

### Retire (recommend status change)
- **`momentum-regime`** — robustness FAIL. Blocking condition is permanent without a v3 redesign.
- **`momentum-regime-v2`** — robustness FAIL. Superseded; no path forward without material redesign.
- **`cross-asset-lead-lag`** — 2 of 3 channels already covered by deployed credit-lead / rate-momentum strategies; marginal value is low.
- **`crypto-microstructure`** — backtest maxDD 34.9% > 15% mandate. The robustness file itself warns paper gate passage is unlikely. Low conviction to revive.
- **`llm-alpha`** — requires LLM-scoring infrastructure not present in the current ops pipeline. Re-init after infra exists.

## Retirement Addendum (2026-04-17)

Formal retirement artifacts added for the two falsified momentum-regime strategies:

- `data/strategies/momentum-regime/retirement.yaml` — verdict: REJECTED, do_not_revive: true
- `data/strategies/momentum-regime-v2/retirement.yaml` — verdict: REJECTED, do_not_revive: true

Both `paper-trading.yaml` files updated: `status: blocked` → `status: retired` with `retired_date: 2026-04-17` and `retired_reason` pointing to the retirement artifact. Original `blocked_reason` text preserved for audit. Family memory: momentum-regime family retired (no v3 planned).

Related bead: `llm-quant-9kbq` (dormant-strategies investigation).

## Changes Applied This Session

1. `scripts/run_paper_batch.py`:
   - Added `"tlt-tqqq-sprint": ("TLT", "TQQQ", 10, 0.01, -0.005, 0.30)` to `LEAD_LAG_PARAMS`.
   - Added `"tlt-tqqq-sprint": "F6-D"` to `MECHANISM_FAMILIES`.
   - Verified via dry-run: strategy now logs alongside the existing 37 (total 38).

## Follow-Up Work

- File a P3 bead: implement `signal_btc_momentum_v2()` generator + add BTC-USD to `ALL_SYMBOLS`.
- File a P3 bead: implement `signal_risk_premium()` generator (VRP + credit carry + risk parity).
- Decide on formal retirement status for `momentum-regime`, `momentum-regime-v2`, `cross-asset-lead-lag`, `crypto-microstructure`, `llm-alpha`. Update `status:` field in each `paper-trading.yaml` from `"initialized"` / `"blocked"` to `"retired"` with a clear `retired_reason` field, so they no longer count against the 45-artifact denominator when measuring paper coverage.

## Verification

Dry-run of `scripts/run_paper_batch.py` after fix produced a line like:

```
tlt-tqqq-sprint                        F6-D neutral               0%   +0.00%     n/a  0.00%     1 DRY-RUN
```

The 38th strategy is now dispatched and will begin accumulating its `daily_log` on the next production run.

# Paper Trading Status Report — 2026-04-29

**Report date:** 2026-04-29
**Latest market data date:** 2026-04-28 (Tue close)
**Strategies in batch:** 52 (paper-trading.yaml files appended through 04-28)
**Strategies in dashboard registry:** 34 (33 active + 1 not started)

---

## Executive Summary

Today's session was operational, not research. Three things changed:

1. **Paper batch caught up.** 04-24, 04-27, 04-28 backfilled (the prior session
   logged most strategies through 04-23 only). All 52 paper yamls are now current
   through 2026-04-28 close. `tlt-tqqq-leveraged-lead-lag` was already current
   from a separate Track D run.
2. **Daily batch automation now lives in the repo.** `scripts/daily_paper_cron.py`
   (smart catch-up via SPY trading calendar) + `scripts/daily_paper_batch.bat`
   (Task Scheduler wrapper with ISO-date logs). User must register the
   `schtasks` entry — instructions inline in the .bat header.
3. **Track C NegRisk re-scan returned 0 actionable opportunities** at the $2k
   production threshold. 3 sub-threshold complement arbs found: us-gdp-q1-2026
   (net=7.3%, kelly=$6), maduro-prison-527 (net=7.5%, kelly=$5), texas-senate-877
   (net=1.5%, kelly=$1.4). Consistent with the 06:48 UTC scan committed earlier.

No promotion decisions are pending. Earliest paper strategies have logged 13–15
trading days against the 30-day calendar gate. The 50-trade gate remains the
binding constraint at current trade frequency (median 2 trades / 12 days).

---

## Days-elapsed leaderboard (top 10)

| Strategy | Days | Trades | Return | Sharpe | Days to 30-day gate |
|---|---:|---:|---:|---:|---:|
| soxx-qqq-lead-lag           | 14 | 1 | +1.9% |  2.855 | 16 |
| lqd-spy-credit-lead         | 14 | 5 | +0.1% |  0.225 | 16 |
| commodity-carry-v2          | 12 | 0 | +4.6% |  9.668 | 18 |
| tlt-qqq-rate-tech           | 12 | 2 | +1.4% |  4.675 | 18 |
| tlt-tqqq-leveraged-lead-lag | 12 | 2 | +1.6% |  4.688 | 18 |
| xlk-xle-sector-rotation-v1  | 12 | 2 | +2.2% |  3.461 | 18 |
| spy-overnight-momentum      | 12 | 4 | +0.2% |  0.679 | 18 |
| uso-xle-mean-reversion-v2   | 12 | 0 | +6.1% |  5.075 | 18 |
| ief-qqq-rate-tech           | 12 | 5 | +0.7% |  2.219 | 18 |
| credit-spread-regime-v1     | 12 | 0 | +2.0% |  4.207 | 18 |

Sample sizes still too small for Sharpe values to carry information. Track D
leveraged strategies (soxl/upro/tqqq couplings) remain the early-tape leaders
in the Track D book; `tlt-tqqq-leveraged-lead-lag` is the only Track D entry
currently rendered in the dashboard registry.

## Bottom 5

| Strategy | Days | Return | Sharpe |
|---|---:|---:|---:|
| agg-efa-credit-lead       | 11 | -2.5% | -5.917 |
| gld-slv-mean-reversion-v4 | 11 | -3.1% | -5.259 |
| tlt-shy-curve-momentum-v1 | 12 | -3.0% | -3.432 |
| global-yield-flow-v2      | 12 | -1.6% | -3.417 |
| behavioral-structural     | 12 | -0.7% | -2.679 |
| gdx-gld-mean-reversion-v1 | 12 | -4.6% | -4.357 |

None near a kill threshold. agg-efa-credit-lead remains the fastest bleeder
of the credit lead-lag family (full deployment, no exit signal), as flagged
in the 04-22 report.

---

## Gate Progress

| Gate | Strategies passing | Median elapsed | Median trades |
|---|---:|---:|---:|
| Paper ≥ 30 days   | 0 / 34 | 12 | 2 |
| Trades ≥ 50       | 0 / 34 | — | — |

At ~0.5 trade/strategy/week, the 50-trade gate is multi-month. This is the
real constraint, not the 30-day calendar window. Open beads
`llm-quant-yywp` ("Decide: drop 50-trade promotion gate to ~20 OR add
high-frequency mechanism") is the right place to resolve this — flagged for
PM-level decision, not deferred indefinitely.

---

## Automation Delivered

`scripts/daily_paper_cron.py`
- Reads `daily_log[].date` across all `paper-trading.yaml` files; identifies
  the latest logged date.
- Queries `market_data_daily` for SPY trading days strictly after that date.
- Calls `run_paper_batch.py --date <d>` for each missed day in chronological
  order, then a normal run for the most recent day.
- Idempotent (`run_paper_batch.py` already skips already-logged dates).
- Tested 2026-04-29: emits "Up to date" since the manual catch-up just
  finished.

`scripts/daily_paper_batch.bat`
- Sets `PYTHONPATH=src`, `cd`s to project root, captures stdout/stderr to
  `logs/paper_cron/YYYY-MM-DD.log`.
- ISO date via PowerShell, locale-independent.

To register (user runs):
```
schtasks /create /tn "llm-quant paper batch" /tr "E:\llm-quant\scripts\daily_paper_batch.bat" /sc daily /st 22:00 /f
```

To remove later:
```
schtasks /delete /tn "llm-quant paper batch" /f
```

This closes the recurring "manual run gap" problem flagged in the 04-22 and
04-24 reports — 5+ days backfilled twice in two weeks because the batch was
run by hand. Once the schtasks entry is live, missed days self-heal: if the
machine is off at 22:00 one evening, the next run sees the gap and backfills.

---

## Surveillance / Governance

`scripts/run_surveillance.py` returns **HALT** but for informational reasons:

| Check | Status | Notes |
|---|---|---|
| data_quality                    | HALT  | 4 stale macro symbols (^TNX, VIX, VIX3M, DOGE-USD) — Yahoo intermittency |
| operational_health (snapshot)   | WARN  | No portfolio snapshot for 12 days (no live `/trade` runs; system not deployed) |
| operational_health (data age)   | OK    | Refreshed via `pq fetch` 2026-04-29 — 14,271 rows through 04-28 |
| kill_switches (drawdown)        | OK    | 0% — no live exposure |
| kill_switches (correlation)     | DEFERRED | paper trading |
| track_d_vix_regime              | OK    | VIX=17.9, normal decay regime |
| track_d_hold_periods            | OK    | No leveraged ETF positions held |
| track_c_*                       | OK    | All Track C tables uninitialised — no exchange events to evaluate |

The HALT is structural (we're not deployed live), not a real signal. Will
remain HALT until a `/trade` cycle runs against a deployed strategy.

---

## Track C — NegRisk Scan

Run 2026-04-29 09:01 UTC against 1042 NegRisk events:

| Market | N | cost | net | volume | kelly |
|---|---:|---:|---:|---:|---:|
| us-gdp-growth-in-q1-2026     | 7 | 0.915 | +7.3% | $671 | $6.2 |
| maduro-prison-time-527       | 5 | 0.930 | +7.5% | $50  | $5.0 |
| texas-senate-election-877    | 7 | 0.985 | +1.5% | $100 | $1.4 |

All three are below the $2k per-venue concentration limit and would not
trigger a real fill at production sizing. Consistent with the 06:48 UTC
re-scan ("0 opps at production thresholds"). No action.

---

## Open Beads (paper-related)

- `llm-quant-93qr` — Run daily paper batch for 30 days (**in_progress**).
  Status: ~half there for earliest strategies. Days will accumulate
  automatically once schtasks is registered.
- `llm-quant-7ud3` — Paper trade top 2 Track D strategies for 60 days
  (**in_progress**).
- `llm-quant-twtl` — Track D paper trading 30-day validation (**ready**;
  blocked by elapsed time).
- `llm-quant-yywp` — Decide: drop 50-trade gate to ~20 OR add
  high-frequency mechanism (**ready**, P1).
- `llm-quant-h50t` — Wire up daily paper batch automation (**open**, P1).
  Code merged this session; awaiting `schtasks` registration to close.

---

## Recommendations

1. **Register the schtasks entry today.** That's the single highest-value
   change available right now — the automation code is ready and tested.
2. **Resolve `llm-quant-yywp`.** At current trade frequency, "30 days OR
   50 trades" is functionally a 2-year gate. Either accept that and
   document it, or relax the trade gate. Don't leave it ambiguous.
3. **Hold on promotion.** Earliest strategy is at 14 days. Re-check
   2026-05-15 when the 30-day calendar window first comes into play for
   `soxx-qqq-lead-lag` and `lqd-spy-credit-lead`.

---

*Previous report:* `docs/reports/paper-trading-status-2026-04-24.md`

# Paper Trading Status Report — 2026-04-22

**Report date:** 2026-04-22
**Strategies with daily logs:** 51 (Track A/B 36, Track D 15)
**Latest data date:** 2026-04-21 (Tue)
**Missed + backfilled:** 2026-04-17 (Fri), 2026-04-20 (Mon) — appended this session via `run_paper_batch.py --date`

---

## Executive Summary

Paper trading book has expanded from 14 to 51 strategies since the 2026-03-31 report.
Most strategies are in the first 4–10 trading days of paper validation — too early for
any promotion decisions, but the signal is broadly positive: **30 of 51 winners, 15 losers,
6 flat; average total return +0.49%; average Sharpe +1.79.**

The 2026-04-17 Friday rally and 2026-04-20/21 two-day pullback produced the cleanest
early divergence in the book:
- **Growth-coupled Track D (SOXL/TQQQ/UPRO rotations) led on the upside** — xlk-xle-soxl and
  soxx-soxl closed +5.5% and +6.9% respectively with Sharpe > 22.
- **Credit lead-lag F1 strategies split** — IG-to-QQQ variants modestly positive, AGG-EFA
  and AGG-SPY materially negative (the international-equity and SPY follower pairs caught
  the 04-20/21 drawdown without a matching credit signal).
- **Commodity mean-reversion the early laggards** — gdx-gld and gld-slv both -1.4 to -2.4%.

Sample sizes are too short (4–10 days) for Sharpe values to carry real information — the
extreme numbers (Sharpe 22+) reflect a single directional week, not an edge. The 30-day /
50-trade gates are the binding constraints; most strategies are ~1/7 of the way there.

---

## Top 10 by Sharpe (n ≥ 4 days)

| # | Slug | Days | NAV | Return | DD% | Sharpe | Trades |
|---|------|-----:|----:|-------:|----:|-------:|-------:|
| 1 | xlk-xle-soxl-rotation-v1       |  4 | 105,497 | +5.50% | 0.00 | 23.689 | 0 |
| 2 | soxx-soxl-lead-lag-v1          |  4 | 106,884 | +6.88% | 0.00 | 22.779 | 0 |
| 3 | commodity-carry-v2             |  7 | 102,818 | +2.82% | 0.00 |  8.198 | 0 |
| 4 | tlt-tqqq-leveraged-lead-lag    |  7 | 101,568 | +1.57% | 0.03 |  6.338 | 2 |
| 5 | tlt-qqq-rate-tech              |  7 | 101,392 | +1.39% | 0.03 |  6.321 | 2 |
| 6 | vcit-tqqq-sprint               |  4 | 100,953 | +0.95% | 0.61 |  6.287 | 0 |
| 7 | tlt-spy-rate-momentum          |  7 | 100,915 | +0.91% | 0.03 |  6.239 | 2 |
| 8 | d14-disinflation-tqqq          |  4 | 101,563 | +1.56% | 1.11 |  5.876 | 0 |
| 9 | d15-vol-regime-tqqq            |  4 | 101,563 | +1.56% | 1.11 |  5.876 | 0 |
| 10| d3-tqqq-tmf-ratio-mr           |  7 | 101,677 | +1.68% | 0.75 |  5.523 | 1 |

**Caveat:** d3-tqqq-tmf-ratio-mr paper track is RETIRED per independent replication
(Sharpe=-1.08 on clean rebuild). The +1.68% here is early noise on a stale yaml. Remove
from the monitored set at next housekeeping pass.

## Bottom 5 by Sharpe

| Slug | Days | NAV | Return | DD% | Sharpe |
|------|-----:|----:|-------:|----:|-------:|
| ief-tqqq-sprint        | 4 | 99,329 | -0.67% | 0.67 | -15.597 |
| agg-efa-credit-lead    | 7 | 97,606 | -2.39% | 2.39 |  -7.501 |
| gdx-gld-mean-reversion-v1 | 7 | 97,612 | -2.39% | 3.86 |  -3.572 |
| global-yield-flow-v2   | 7 | 99,032 | -0.97% | 1.77 |  -3.003 |
| gld-slv-mean-reversion-v4 | 7 | 98,636 | -1.36% | 2.15 |  -2.986 |

None near a kill threshold. agg-efa is the fastest bleeder (100% deployed, no credit
signal fired to exit); worth monitoring but too early for action.

## Flat (weight=0 all sessions, no trades)

agg-tqqq-sprint, agg-upro-sprint, lqd-tqqq-sprint, tlt-soxl-sprint, tlt-tqqq-sprint,
tlt-upro-sprint — six Track D sprints where the leader signal has stayed below the
entry threshold. Behaving as specified; not evidence of anything yet.

---

## Backfill Note — 2026-04-17 and 2026-04-20

The 2026-04-17 batch run committed on Apr 17 wrote entries for 2026-04-16 only (commit
message claimed 04-17 but data-availability cutoff prevented it). No batch ran 04-18–04-21.
A single catch-up run on 2026-04-22 appended the latest available close (04-21) but the
runner only wrote the *latest* date and skipped the two intervening sessions.

This session added a `--date YYYY-MM-DD` flag to `scripts/run_paper_batch.py` and
backfilled 26 tracked strategies for both missed days. Entries are inserted
chronologically; NAV/peak/drawdown/Sharpe are rebuilt from the full `daily_log`; an
`updated_at: 2026-04-22` and `backfilled_dates` trail is written for audit. See
commit `f6eb942`.

**Known minor drift:** the 2026-04-21 entries' `daily_return_pct` carry a switch-cost
adjustment computed against the prior-at-the-time regime (2026-04-16 value). Post-insert,
their true preceding regime is 2026-04-20's. In cases where 04-16 regime ≠ 04-21 regime
but 04-20 regime = 04-21 regime (or vice versa), the switch-cost adjustment is off by
one COST_PER_SWITCH = 0.03%. The NAV chain and all aggregates are correct.

---

## Gate Progress

None at the promotion threshold. Earliest is soxx-qqq-lead-lag (10 days, 1 trade) —
still 20 trading days and 49 trades short of the 30-day / 50-trade gate.

| Gate tier | Strategies | Median days elapsed | Median trades |
|-----------|-----------:|--------------------:|--------------:|
| Paper ≥ 30 days | 0/51 | 7 | 1 |
| Trades ≥ 50    | 0/51 | — | — |

At the current trade-generation pace (~0.5 trade/strategy/week), the 50-trade gate is
the binding constraint. Either (a) calendar windows will extend well past the 30-day
minimum, or (b) the program needs to add mechanisms with higher trade frequency (e.g.,
intraday signals, overnight gap momentum with lower thresholds).

---

## Open Beads (paper-trading related)

- `llm-quant-93qr` — Run daily paper batch for 30 days to accumulate paper track
  records (**in_progress**). Core blocker for all promotion decisions.
- `llm-quant-7ud3` — Paper-trade top 2 Track D strategies for 60-day validation
  (**in_progress**). Current top 2 by backtest CAGR are XLK-SOXL and TSMOM-UPRO, which
  are in the book.
- `llm-quant-twtl` — Track D paper trading — 30-day validation for all passing
  strategies (**ready**, blocked by elapsed time).
- `llm-quant-9kbq` — 8 paper strategies initialized but not logging daily signals
  (**ready**). May be partially resolved by today's catch-up; needs re-audit.

---

## Recommendations

1. **Schedule the daily batch** — running it manually creates gaps (two missed days in
   one week, repeatedly). A Claude Code cron/scheduled trigger or a Windows Task
   Scheduler entry firing `run_paper_batch.py` at ~22:00 ET each weekday would close
   this. Today's `--date` flag makes recovery possible when automation fails but doesn't
   replace the need for automation.
2. **Remove d3-tqqq-tmf-ratio-mr from the daily batch** — strategy is retired; paper
   entries are noise and distort aggregates. Delete its loop entry from
   `run_paper_batch.py`.
3. **Re-audit `llm-quant-9kbq`** — the "8 strategies not logging" bead was filed before
   the recent expansion to 51. Today's batch run touched 51 yamls; some of those eight
   may now be caught up. Close or re-scope.
4. **Hold on promotion decisions** — no strategy is close. Re-check 2026-05-15 when the
   earliest-started strategies (soxx-qqq, lqd-spy) hit 30 calendar days and the 50-trade
   gate becomes the only open question.

---

*Previous report: docs/reports/paper-trading-status-2026-03-31.md*

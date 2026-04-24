# Forensic Finding F-03 — The 50-Trade Promotion Gate Is Structurally Impossible

**Question:** CLAUDE.md and the model promotion policy require ≥50 trades over ≥30 days for promotion. Is this gate achievable at the current strategy throughput?

## Hard Numbers (executed 2026-04-24 against current `paper-trading.yaml` files)

| Metric | Value |
|---|---|
| Strategies in paper trading | **53** |
| Total trades across entire book | **46** |
| Mean trades per strategy | **0.87** |
| Median total return | **+0.000%** |
| Strategies with zero trades | **39 of 53 (74%)** |
| Strategies with ≥10 trades | **0** |
| Strategies with ≥50 trades | **0** |
| Top trade count (any strategy) | **4** (tied across 14 credit-lead variants) |

## The Math

Current pace: **46 trades across 53 strategies in ~3 weeks** = 15.3 trades/week, or **~0.29 trades per strategy per week.**

To reach the 50-trade gate at current pace:
- **One strategy** at 0.29 trades/wk needs **172 weeks ≈ 3.3 years** to hit 50 trades.
- **All 53 strategies** at the current aggregate rate need **2,650 / 15.3 = 173 weeks ≈ 3.3 years.**

This isn't a "binding constraint." It is a **promotion gate that, if applied literally, would block every strategy in the book indefinitely.**

## Why So Few Trades?

The 53 paper-trading strategies fall into two categories by mechanism family:

**Slow-signal strategies (~40 of 53):** Credit lead-lag (AGG/LQD/HYG → SPY/QQQ), rate momentum (TLT → various), regime-conditional rotations. These fire 0-2 trades per month *by design* — they wait for a multi-day signal cross. The 50-trade gate is incompatible with this signal class.

**Fast-signal strategies (~13 of 53):** Track D leveraged sprints, RSI-based mean reversion, overnight momentum. These should fire more frequently, but most are very recently registered (median 3 days in book) and haven't accumulated history.

The current top-of-book strategies all have exactly 4 trades each — that's not a coincidence. It's the cap of how many regime transitions have occurred in 3 weeks of price data for a 5-day rolling signal. The signal-firing rate is bounded by market regime changes, not by strategy parameters.

## What the Top Performers Look Like

```
agg-efa-credit-lead                        4 trades
agg-qqq-credit-lead                        4 trades
agg-spy-credit-lead                        4 trades
emb-spy-credit-lead                        4 trades
hyg-qqq-credit-lead                        4 trades
hyg-spy-5d-credit-lead                     4 trades
lqd-qqq-credit-lead                        4 trades
lqd-spy-credit-lead                        4 trades
spy-overnight-momentum                     4 trades
vcit-qqq-credit-lead                       4 trades
```

All credit-lead-lag F1 variants. Identical trade counts because they trade on similar regime crossings. **High inter-strategy correlation reduces effective diversification** — one regime change moves many of these together.

## Naming Drift Aside

The earlier agent claim that `cumulative_metrics` is silently dropped is **partially correct**:
- The dict returned by `compute_cumulative_metrics()` is NOT written to `paper["cumulative_metrics"]` (verified at `scripts/run_paper_batch.py:1891` and `:2028`).
- BUT the `current_sharpe` field of that dict IS extracted and written to `paper["performance"]["current_sharpe"]` (line 1898).

So the data exists under `performance.current_sharpe` (26 of 53 strategies have a Sharpe computed). The "all-flat" appearance in earlier surveys was a key-name mistake by the surveyor (me), not a code bug. **Real performance numbers are persisted; the secondary metrics dict (`max_dd`, `total_return`, `days`) is what's silently dropped.**

Mean Sharpe across the 26 strategies that have one computed: **1.466.** This number is not credible as a portfolio metric — it reflects 5-10 day windows containing a single directional event (the 04-17 rally + 04-21 pullback). Use as evidence of strategy life-signs, not edge.

## The Promotion Gate Has Three Possible Fixes

### Fix A — Lower the trade count to match signal class
Slow-signal strategies could use **15 trades + 95% confidence interval on Sharpe** as the gate. This is statistically defensible (n=15 gives reasonable Sharpe CI for many distributions) and achievable within the 30-day calendar.

Edit: `docs/governance/quant-lifecycle.md`, `docs/governance/model-promotion-policy.md`, `.claude/commands/promote.md`.

### Fix B — Time-replace trade-count
Use **90 days of paper trading + ≥1 entry-and-exit per 30-day window** as the gate. This trades the trade-count statistic for a longer time window, which is what slow signals actually need.

### Fix C — Two-tier promotion gates
- **Tier 1 (slow-signal):** 90 days, 5+ trades, Sharpe ≥ 0.40 with 95% CI.
- **Tier 2 (fast-signal):** 30 days, 50 trades, Sharpe ≥ 0.60.

Strategies declare their tier in the spec.

**Recommendation:** Fix C. Acknowledges the structural difference between credit-lead-lag (slow) and Track D leveraged sprints / overnight momentum (fast). Without it, the promotion gate is a dead letter.

## What This Means for the Platform

- **No strategy will pass the current promotion gate, ever, at current data scale.**
- The platform has been running paper trading for ~3 weeks with the implicit understanding that ~30 days will yield decisions. **It will not.** The understanding is wrong.
- Either the gate is changed, or paper trading becomes a perpetual research artifact — strategies stay in paper forever, decisions never get made.
- This is the most fixable item in the entire strategic review. **Editing the governance docs is a 30-minute change.** It just requires the PM to formally accept that 50 trades is not achievable for 75% of the book.

## Confidence

**Maximum.** Numbers were computed against current YAML state. The math (50 / 0.29 = 172 weeks) is arithmetic, not opinion.

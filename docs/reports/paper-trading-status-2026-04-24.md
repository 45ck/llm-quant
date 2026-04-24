# Paper Trading Status Report — 2026-04-24

**Report date:** 2026-04-24
**Strategies with paper-trading logs:** 53 (52 dated through 2026-04-23, 1 freshly registered)
**Latest market data date:** 2026-04-23
**New today:** lqd-soxl-sprint (registered to LEAD_LAG_PARAMS after passing robustness)

---

## Executive Summary

Two events today:
1. **Robustness sprint cleared 8 Track D leveraged strategies** (6 PASS, 2 KILL). The 6 passing strategies are all in paper trading; 5 were already there from prior sprints, and `lqd-soxl-sprint` was added today.
2. **Daily paper batch ran cleanly** — 52 strategies updated through 2026-04-23 close, no errors.

The Track D leveraged book continues to dominate the early-tape:
- **soxx-soxl-lead-lag-v1**: +10.46% in 5 days
- **xlk-xle-soxl-rotation-v1**: +8.34% in 5 days
- These are the SOXL-coupled momentum strategies catching the AI/semi rally that started 2026-04-17.

The remaining Track D sprints (agg-/tlt-/lqd-/vcit- → TQQQ/UPRO/SOXL) are mostly **flat** because their leader signals (credit and rate momentum) are in **neutral regime**. They are armed but not yet firing — the 10-day signal windows haven't crossed entry thresholds. This is the expected behavior of a regime-gated lead-lag strategy in a directionless macro tape.

---

## Today's Robustness Pass — Verdicts

Multi-agent parallel run on `scripts/robustness_*_sprint.py` for 8 candidates that had cleared backtest gates with 1 trial each.

| Strategy | Sharpe | DSR | Perturb | Shuffled p | Verdict |
|---|---|---|---|---|---|
| agg-tqqq-sprint | 1.078 | 0.969 | 56% | 0.0000 | **PASS** |
| tlt-soxl-sprint | 1.141 | 0.973 | 73% | 0.0000 | **PASS** (strongest) |
| lqd-soxl-sprint | 0.936 | 0.955 | 60% | 0.0040 | **PASS** |
| lqd-tqqq-sprint | 0.896 | 0.950 | 55% | 0.0000 | **PASS** (brittle) |
| tlt-upro-sprint | 0.849 | 0.943 | 59% | 0.0000 | **PASS** |
| agg-upro-sprint | 0.840 | 0.942 | 50% | 0.0010 | **PASS** (marginal) |
| agg-soxl-sprint | 0.731 | 0.921 | 48% | **0.527** | **KILL** |
| lqd-upro-sprint | 0.778 | 0.931 | 45% | 0.0030 | **KILL** |

**Critical finding:** `agg-soxl-sprint` would have advanced under the other 5 gates — its real Sharpe (0.731) is statistically indistinguishable from random shuffles (mean 0.735, p=0.527). The shuffled-signal test was the only thing separating "real lead-lag" from "drift exposure dressed up as signal." This is a vindication of the 6-gate Track D protocol.

Full report: [docs/research/results/track-d-robustness-results-2026-04-24.md](../research/results/track-d-robustness-results-2026-04-24.md)

---

## Active Paper Trading Book — Headline Numbers

- **Total strategies:** 53 (Track A/B baseline + Track D sprints)
- **Median days in book:** 3 (book is very fresh)
- **New (<5 days in paper):** 26 of 53 — too early to read tea leaves
- **Track D sprints in active paper:** 17 (added lqd-soxl today)

### Notable performers (5 days in book)
| Strategy | NAV change | Notes |
|---|---|---|
| soxx-soxl-lead-lag-v1 | +10.46% | F8-D, semi-lead-lag with leverage |
| xlk-xle-soxl-rotation-v1 | +8.34% | Tech vs Energy rotation into SOXL |
| tsmom-upro-trend-v1 | +1.42% (today) | TSMOM trend with UPRO |
| gdx-gld-mean-reversion-v1 | -1.19% | Mean-rev gold miners (lagging) |

### Sprints awaiting first entry (5+ days, 0% return)
agg-tqqq, tlt-soxl, tlt-upro, lqd-tqqq, vcit-tqqq, ief-tqqq — credit/rate signals in neutral regime; thresholds not crossed. These are healthy "waiting" states, not failures.

---

## Today's Batch Run — Regime Snapshot

```
Avg daily return: -0.084%
Risk-on:  4 strategies (xlk-xle-soxl, soxx-soxl, tsmom-upro, uso-xle)
Risk-off: 0 strategies
Neutral: 48 strategies (most of the book waiting for signals)
Total: 52
```

The market printed a -0.5% to -1% pullback on 2026-04-21/22 before stabilizing. The neutral-regime majority is consistent with the book's design: most lead-lag and credit signals require a week+ of confirming flow before entering.

---

## Sample Size Caveat

**No promotion decisions can be made yet.** Sharpe values currently shown for some strategies are >20 — those are 5-day artifacts of a single directional week (the 04-17 rally), not real edges. The binding gates per CLAUDE.md are:
- 30+ days in paper
- 50+ trades realised
- Sharpe ≥ 0.60 floor over the validation window

Median 3 days in book = ~10% of the way to the time gate. Most strategies have 0 trades because signals haven't fired.

---

## Open Items

- **overnight-tqqq-sprint** (bd llm-quant-9xnh) still broken — overnight_momentum strategy needs TQQQ ticker support. Infrastructure fix, not signal failure. F11 (overnight gap) is one of our strongest uncorrelated mechanisms; re-running on TQQQ should be high priority.
- **vcit-tqqq-sprint, ief-tqqq-sprint** — registered for paper trading but never had robustness run. Either run robustness or document why they were registered without the gate.
- **Track C arb research** — secondary focus per updated CLAUDE.md, no movement this session.

---

**Decision logged:** bd llm-quant-q952 (Track D Robustness Sprint).
**Commits:** 340e94d → bb35be5 on master, 3 commits (Track A KILLs, Track D robustness report, Track D robustness artifacts + paper registry).

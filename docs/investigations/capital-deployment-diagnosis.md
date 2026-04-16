# Capital Deployment Diagnosis — 61% Cash During Risk-On Rally

**Investigation date:** 2026-04-17
**Bead:** `llm-quant-bnry` (P1 bug)
**Investigator:** capital-deployment-debugger (team: apr17-cleanup)
**Status:** Diagnosis complete — bead remains open for follow-up fix

---

## Executive Summary

The paper portfolio sat at **60-69% cash** for three weeks (2026-03-24 to 2026-04-16) while SPY rallied **+7.42%** from `$653.18` to `$701.66`. Portfolio NAV grew only **+1.27%** (from `$100,000` to `$101,268.55`) — effectively matching `39% deployed × 7.42% ≈ 2.89%` minus drag from defensive positions (TLT, SHY, GLD).

**This is not a risk-manager rejection problem and not a strategy-deployment bug.** The root cause is the prompt template and the decision cadence: the LLM is correctly following a systematically over-conservative set of hard constraints written into `config/prompts/trader_system.md` and is only called on ~7 trading days out of 18, missing the entire risk-on repricing after VIX normalized below 22 on 2026-04-08.

---

## 1. Cash Trend — 30-Day Snapshot

Extracted from `portfolio_snapshots` table (pod_id=default). Duplicate rows per day are intra-day re-runs; latest per day shown:

| Date       | NAV         | Cash        | Cash %  | Gross % | Daily P&L | Note |
|------------|-------------|-------------|---------|---------|-----------|------|
| 2026-03-24 | 100,000.00  | 100,000.00  | 100.00  | 0.00    | 0.00      | Init |
| 2026-03-24 | 100,000.00  | 78,952.70   | 78.95   | 21.05   | 0.00      | First buys |
| 2026-03-25 |  99,958.95  | 46,205.65   | 46.22   | 53.78   | -41.05    | Largest deployment |
| 2026-04-01 | 100,526.55  | 51,227.04   | 50.96   | 49.04   | +526.55   | Still ~50% |
| 2026-04-02 | 100,646.36  | 65,288.79   | 64.87   | 35.13   | +120.00   | Cash spike (XLP close) |
| 2026-04-03 | 100,646.36  | 66,114.87   | 65.69   | 34.31   | 0.00      | USO trim |
| 2026-04-07 | 100,625.44  | 69,040.67   | 68.61   | 31.39   | -20.92    | XLV close (VIX=27) |
| 2026-04-15 | 101,155.07  | 61,726.52   | 61.02   | 38.98   | +528.71   | SPY/QQQ/XLK bought, but partial |
| 2026-04-16 | 101,268.55  | 61,726.52   | 60.95   | 39.05   | +113.48   | Flat vs SPY +0.25% |

**Key observation:** Cash % grew from 46% (peak deployment on 2026-03-25) to 69% on 2026-04-07, because positions were closed but not replaced. After VIX normalized (18.17 on 2026-04-15), one partial re-entry cycle deployed ~$7k but left $61k idle.

---

## 2. Signal Generation Breakdown

The `llm_decisions` table is **EMPTY (0 rows)** — the decision persistence path is not wired (the `decision_id` variable in `scripts/execute_decision.py` is hardcoded to `None` on line 107). Signal analytics must be inferred from the `trades` table.

### Trade action counts by session

| Date       | BUY | SELL | CLOSE | Total Notional |
|------------|-----|------|-------|----------------|
| 2026-03-24 |  16 |   0  |   0   | $33,024  (BUY) |
| 2026-03-25 |  14 |   0  |   7   | $28,575 / $17,618 |
| 2026-04-01 |   1 |   3  |   6   | $991 / $2,792 / $11,274 |
| 2026-04-02 |   0 |   0  |   1   | $4,000 (XLP close) |
| 2026-04-03 |   0 |   1  |   0   | $826 (USO trim) |
| 2026-04-07 |   0 |   0  |   1   | $2,926 (XLV close) |
| 2026-04-15 |   3 |   1  |   0   | $10,203 / $2,889 |
| **Totals** | **34** | **5** | **15** | — |

**Observation:** The April sessions are BUY-starved. After March 25, the system closed 15 positions (de-risking into the late-March VIX spike) but only opened 4 new positions in the entire April rally window. Even the 2026-04-15 re-entry (SPY, QQQ, XLK) averaged only **$3.4k per trade** — just above the 2% NAV floor.

### Trade cadence gap

**`/trade` was only executed on 7 distinct dates over a 22-trading-day window:**

```
2026-03-24, 2026-03-25, 2026-04-01, 2026-04-02, 2026-04-03, 2026-04-07, 2026-04-15
```

**Missing dates include 2026-04-08 through 2026-04-14 — precisely the days SPY ripped from $659 to $694 (+5.3%) as VIX collapsed from 25.78 to 18.36.** The system did not make decisions during the best part of the rally.

### Average BUY notional

34 BUY trades totalled $72,793 — **average $2,141 per trade, with 27/34 (79%) clustered near the 2% NAV floor (~$2,000)**. The LLM is sizing trades at the minimum of the stated hard constraint, never leaning into conviction.

---

## 3. Risk Rejection Analysis

**There is no `risk_rejections` table in the schema.** Rejections are only ephemeral JSON output from `scripts/execute_decision.py` (lines 144-155, written to stdout). None of the historical rejection data is persisted.

Code inspection of `src/llm_quant/risk/manager.py` confirms all 11 checks behave correctly:

- `check_cash_reserve` only triggers on buys that would drop cash below 5%. At 61% cash, this cannot be blocking trades.
- `check_position_weight` cap is 12% per position; largest current weight is SPY at 7.62%. Not binding.
- `check_gross_exposure` cap is 200%; current gross = 39%. Not binding.
- `check_position_size` cap is 4% of NAV per trade. The LLM is sizing at 2% — it is self-limiting, not rejected.
- `check_drawdown_limit` only activates at 15% DD; portfolio is at +1.27%, peak NAV = current NAV.

**Conclusion: the risk manager is not rejecting trades. The LLM is simply not generating BUY signals at meaningful size.**

---

## 4. Deployed Strategy Universe

### What is actually running

```
$ SELECT pod_id FROM pods
('default', 'Default Pod', 'regime_momentum', 100000.0, 'active', ...)
```

**Only the `default` pod exists in DuckDB.** This pod runs the discretionary LLM macro strategy via `/trade`.

### What is declared in `config/track-assignments.yaml`

- Track A: **29 strategies** (target 70% allocation)
- Track B: **3 strategies** (target 30% allocation)
- Track D: **1 strategy** (target 0% — experimental)
- Discretionary: 1 pod (target 0%)

### Deployment gap

**33 systematic strategies are specified but ZERO have live pods with capital deployed.** The 45 `paper-trading.yaml` files found in `data/strategies/*/paper-trading.yaml` are **log-only artifacts from `scripts/run_paper_batch.py`** — they generate signals and track hypothetical performance but **do not execute into the paper portfolio database.**

The `default` pod operates as a pure LLM-macro strategy with no wiring to:
- The SR=2.205 Track A portfolio (18 cluster reps, 35 validated strategies)
- The 3 Track B strategies (SOXX-QQQ, USO-XLE-MR, GDX-GLD-MR)
- Any Track D leveraged strategies (TLT-TQQQ, XLK-XLE-SOXL, TSMOM-UPRO, etc.)

This is the single largest structural contributor to underdeployment. The portfolio management headline number (SPY +7.4% vs us +1.3%) reflects **not a strategy failure but a strategy-NON-DEPLOYMENT**.

---

## 5. Prompt / Hard-Constraint Analysis

`config/prompts/trader_system.md` hardcodes constraints that are stricter than `config/risk.toml`:

| Constraint        | System prompt (told to LLM) | risk.toml (enforced) |
|-------------------|-----------------------------|----------------------|
| Max position      | **10%**                     | 12%                  |
| Max trade size    | **2%**                      | **4%**               |
| Cash reserve      | 5%                          | 5%                   |
| Max trades/sess   | 5                           | 5                    |
| Target weight range | `0.0-0.10`                | n/a                  |

**The LLM is being told `max trade = 2% of NAV`** even though the risk manager would accept 4%. The LLM religiously complies (27 of 34 buys clustered at ~$2,000). A 5-trade session at 2% each caps new deployment at **10% of NAV per session** — so even if the system ran daily, it would take ~6-10 sessions of nothing-but-buys to fully deploy from 61% cash to 5% cash. It never ran daily.

In addition, the decision prompt in `config/prompts/trader_decision.md` injects these directives every session:

- Line 64: *"VIX percentile sizing rule: when VIX percentile > 80, scale down target position sizes by **50%**"*
- Line 65: *"Silent stress alert: when silent_stress=True, treat as risk_off even if VIX appears benign"*

The 2026-04-07 snapshot (`data/trade_context_post.json`) shows VIX=27.38 **at 96th percentile** — that triggers the 50% size scale-down. The last `trade_decision.json` reads the LLM explicitly saying:

> *"Already 65.7% cash - no new entries. ... Wait for VIX to normalize below 22 before considering new entries."*

**VIX crossed 22 on 2026-04-08. The system was not run again until 2026-04-15 — seven trading days later.**

---

## 6. Root Cause — Ranked Hypotheses

| # | Hypothesis | Evidence | Severity | Verdict |
|---|-----------|----------|----------|---------|
| 1 | **Prompt mandates systematic under-sizing (2% cap & VIX-scaled sizing)** | `trader_system.md` line 9 hardcodes 2% trade cap vs 4% in risk.toml. 27/34 buys sized at ~$2k floor. VIX percentile >80 scales sizing by 0.5x further. | **CRITICAL** | **PRIMARY ROOT CAUSE** |
| 2 | **Operational cadence gap — 7-day black-out during best-of-rally** | 7 trade sessions across 22 trading days. Gap from 2026-04-07 to 2026-04-15 covered SPY +5.3% and VIX collapse from 25.8 to 18.4. | **CRITICAL** | **SECONDARY ROOT CAUSE** |
| 3 | **Deployment gap — 33 Track A/B/D strategies not wired to capital** | Only `default` pod exists in DB. Track A (SR=2.205) / Track B / Track D completely absent from `pods` table. Paper-batch artifacts are log-only, no execution path. | **CRITICAL** | **STRUCTURAL ROOT CAUSE** (largest dollar impact over time) |
| 4 | LLM regime stuck in "transition" / "risk_off" during low-VIX days | Last recorded decision 2026-04-07 was "transition" at VIX=27. Cannot confirm for 2026-04-15 (no llm_decisions persistence). Indirect evidence: only 3 small buys on 2026-04-15 (total $10k) with VIX at 18. | **HIGH** | **CONTRIBUTING** |
| 5 | `llm_decisions` table empty — no decision telemetry | 0 rows in `llm_decisions`; `execute_decision.py` passes `decision_id=None` at line 107. | MED | Auxiliary — blinds us to hypothesis 4 |
| 6 | Risk manager blocking trades | Code inspection: none of the 11 checks can bind at 61% cash with 39% gross | N/A | **REJECTED** — no evidence |
| 7 | Stop-loss whipsaw not replacing exits | Closes on 2026-04-01 (6 positions) and 2026-04-02/07 (XLP, XLV) were regime-driven defensive closes, not stop-outs. Stop-losses held. | LOW | **REJECTED** |
| 8 | Universe too narrow | 42-symbol universe covers SPY/QQQ/sector ETFs/bonds/commodities/crypto/forex — wider than deployed positions. Not the constraint. | LOW | **REJECTED** |

---

## 7. Concrete Fix Recommendations

Ranked by expected dollar impact per $100k NAV per month.

### Priority 1 — Raise per-trade size cap (low-risk, high-impact)

**Change** `config/prompts/trader_system.md` line 9 from:

> *"No single trade can exceed 2% of NAV. No position can exceed 10% of NAV."*

to:

> *"No single trade can exceed 4% of NAV. No position can exceed 12% of NAV."*

This aligns the prompt with `config/risk.toml` limits (`max_trade_size=0.04`, `max_position_weight=0.12`). Expected effect: doubles per-session deployment from max 10% to max 20% of NAV.

### Priority 2 — Close the cadence gap

Schedule `/trade` to run every trading day, not ad-hoc. Between 2026-04-07 and 2026-04-15 the system missed:
- VIX drop from 25.78 → 19.23 (from risk-off to risk-on)
- SPY rally +3.1%
- Natural re-entry window after the late-March de-risking

Adding automation (cron job, GitHub Actions, or requirement that the PM agent run `/trade` daily when markets are open) would have captured roughly half the missed alpha.

### Priority 3 — Lower cash-reserve floor, deploy idle cash to SHY/BIL placeholder

Per the bead's own suggestion list:
- (a) Lower `min_cash_reserve` in `config/risk.toml` from 5% → 2-3% (T-bill yields fund a near-cash placeholder)
- (b) Add an auto-deploy rule: any cash above target reserve is parked in SHY or BIL (3M T-bill ETF, ~5% yield) until allocated elsewhere. Eliminates the ~$60k in zero-yielding cash drag.

### Priority 4 — Persist `llm_decisions` rows

Fix `scripts/execute_decision.py` line 107 to actually record decisions before logging trades. This unblocks hypothesis 4 analysis (regime-driven HOLD loops) and enables future performance attribution.

### Priority 5 — Deploy the validated strategy portfolio

The largest structural fix — but most complex. Wire at least the 4-5 highest-Sharpe Track A cluster representatives (and the 3 Track B strategies) into real pods with real capital. The SR=2.205 portfolio is sitting on the shelf.

Minimal proof-of-concept: allocate $30k to a `track-a` pod running the top 3 Track A strategies via a simple daily signal-executor, and $20k to a `track-b` pod running SOXX-QQQ lead-lag. This alone would move 50% of the 61% cash into working strategies.

### Priority 6 — Soften the VIX >80th-pct sizing rule

The 50% size scale-down in `trader_decision.md` line 64 compounds on top of already-small 2% trades. When VIX spiked to the 96th percentile in late March, this drove trade sizes down to ~1% of NAV — essentially cosmetic positions. Consider scaling by 0.75x rather than 0.5x, or applying only to *new* positions (not rebalances of existing holdings).

---

## 8. Appendix — Key Numbers

- Sessions executed: 7 / 22 trading days (32%)
- Avg BUY notional: $2,141 (2.14% of initial NAV)
- BUY trades clustered at ~2% floor: 27 / 34 (79%)
- Gross exposure trajectory: 0% → 54% (peak 2026-03-25) → 31% (2026-04-07) → 39% (2026-04-16)
- Cash trajectory: 100% → 46% → 69% → 61%
- SPY trajectory (session window): $653.18 → $701.66 (+7.42%)
- Portfolio NAV: $100,000 → $101,268.55 (+1.27%)
- Implied deployed alpha: $1,268 / $39,542 gross ≈ 3.2% on working capital — not catastrophic, but the **$61,726 in idle cash returned ~0% in a +7% tape**, which is the entire shortfall.

---

*End of diagnosis. This report does not modify any code or configuration. Follow-up fixes should be filed as new beads linked to `llm-quant-bnry`.*

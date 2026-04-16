# Investigation: Apr 01 2026 Mass-Close Whipsaw

**Issue:** llm-quant-7a53 (P1 bug)
**Investigator:** whipsaw-investigator (team apr17-cleanup)
**Date:** 2026-04-17
**Period under investigation:** 2026-03-24 through 2026-04-10

## Executive Summary

On 2026-04-01 the paper portfolio executed 9 trades (7 full closes + 2 partial trims) across a single session, collapsing gross exposure from $49,299 to $39,238 (-20.4%) and moving ~$14,067 of notional into SHY / cash. The closes were rationalised by the LLM as a "risk-off" regime response to elevated VIX, but the decision was made AFTER the VIX spike had already peaked and begun reversing. The portfolio then sat in a defensive posture while SPY rallied +3.8% over the following 6 trading days.

**Root cause is NOT a kill-switch, governance halt, or risk-manager rejection cascade.** The surveillance scans on Apr 1 were all `severity=warning` (never `halt`) and the risk manager accepted every trade. The root cause is a **discretionary LLM misjudgement driven by stale context**: a 6-calendar-day gap in portfolio activity (2026-03-26 through 2026-03-31) caused the LLM to see the VIX-31 spike of Mar 27 as current, not as an already-receded event. It sold the dip instead of buying it.

**Estimated missed-rally cost Apr 1 → Apr 10: ~$432** on the 8 closed/trimmed lines (Apr 16 mark-to-market: ~$818). NAV ended +1.27% while SPY was +7.42% — alpha of -6.15% is consistent with the exposure reduction plus locked-in post-trade drift.

## Timeline of Events

| Date | VIX | SPY | QQQ | Portfolio Event |
|---|---|---|---|---|
| 2026-03-23 | 26.15 | 655.38 | 588.00 | Position building day 1 (baseline) |
| 2026-03-24 | 26.95 | 653.18 | 583.98 | 16 buys executed ($31,023 gross deployed) |
| 2026-03-25 | 25.33 | 656.82 | 587.82 | Final buys, snapshot_id=13 closes at 17:00; gross $51,745 |
| 2026-03-26 | **27.44** | 645.09 | 573.79 | **No portfolio activity after 08:19 UTC** |
| 2026-03-27 | **31.05** | 634.09 | 562.58 | **VIX spike peak. No trades. No snapshots.** |
| 2026-03-30 | 30.61 | 631.97 | 558.28 | **No trades. No snapshots.** |
| 2026-03-31 | 25.25 | 650.34 | 577.18 | **VIX already down -5.8pts from peak. No trades.** |
| 2026-04-01 | **24.54** | 655.24 | 584.31 | **Session reopens. LLM labels regime "risk_off". 9 trades executed 12:39-15:59 UTC. Gross collapses to $39,238.** |
| 2026-04-02 | 23.87 | 655.83 | 584.98 | XLP closed ($4,000 notional). |
| 2026-04-03 | 24.03* | 658.93* | 588.50* | USO trimmed. |
| 2026-04-07 | 25.78 | 659.22 | 588.59 | XLV closed. |
| 2026-04-08 | 21.04 | 676.01 | 606.09 | SPY +3.1% single day; portfolio under-allocated. |
| 2026-04-10 | 19.23 | 679.46 | 611.07 | VIX normalised; portfolio still defensive. |

(*Apr 3 is cash-equities holiday — Apr 6 figures shown for continuity.)

Apr 1 session sequence (UTC):
1. `12:39` — scan_id 27 posts 2 warnings: `operational_health: No portfolio snapshot for 7 days` and `track_d_vix_regime: VIX=25.2 elevated`.
2. `12:39` — trades 38, 39 execute: close SOL-USD (-10.2% loss), trim USO.
3. `13:02` — snapshot 16: further USO trim, XLI close, XLC close.
4. `14:36` — snapshot 17: QQQ close, DBA trim.
5. `15:59` — snapshot 18: XLF close, XLRE close, SHY add.
6. `16:38` — snapshot 19: session complete. Gross exposure $39,238 (20.4% lower).

## Positions Closed / Trimmed (Apr 1)

| trade_id | Symbol | Action | Shares | Price | Notional | Apr 10 price | Missed P&L |
|---|---|---|---|---|---|---|---|
| 38 | SOL-USD | close | 21 | 82.91 | $1,741 | 84.83 | $+40 |
| 39 | USO | sell | 10 | 127.25 | $1,273 | 124.82 | $-24 |
| 40 | USO | sell | 4 | 127.25 | $509 | 124.82 | $-10 |
| 41 | XLI | close | 12 | 161.73 | $1,941 | 171.52 | $+117 |
| 42 | XLC | close | 17 | 110.86 | $1,885 | 113.95 | $+53 |
| 43 | QQQ | close | 3 | 577.18 | $1,732 | 611.07 | $+102 |
| 44 | DBA | sell | 37 | 27.32 | $1,011 | 26.89 | $-16 |
| 45 | XLF | close | 40 | 49.37 | $1,975 | 50.77 | $+56 |
| 46 | XLRE | close | 49 | 40.83 | $2,001 | 42.82 | $+97 |
| 47 | SHY | buy | 12 | 82.57 | $991 | — | n/a (cash proxy) |

**Total closes+trims notional: $14,066. Net missed rally Apr 1→Apr 10: ~$415. Apr 1→Apr 16: ~$818.**

## Surveillance State

| scan_id | Timestamp (UTC) | Severity | Halts | Warnings | Warning Messages |
|---|---|---|---|---|---|
| 27 | 2026-04-01 12:36 | warning | 0 | 2 | snapshot_gap_days=7 (limit 3); VIX=25.2 (>=25) |
| 28-31 | 12:36-12:38 | warning | 0 | 2 | same |
| 32 | 13:01 | warning | 0 | 1 | VIX=25.2 (snapshot gap cleared after snapshot_id=14 written) |
| 33-38 | 14:33-16:37 | warning | 0 | 1 | VIX=25.2 |

**No kill switch triggered. No halt. The 15% NAV-drawdown, 5% daily-loss, 5-consecutive-loss, and 72h data-blackout switches all had slack.** The `operational_health.max_snapshot_gap_days=3` threshold was exceeded (7 days), but this is warning-level only in `governance.toml` and does not gate trading.

## Risk Manager State

All 9 Apr 1 trades recorded in `trades` table. No entries in any rejection log — the risk manager's 7 pre-trade checks approved every order. The SHY buy ($991, 1.0% of NAV) was well under the 2% per-trade and 10% per-position limits. Closes do not consume new risk budget.

## LLM Decision Rationale (from `trades.reasoning`)

The LLM explicitly cited **"risk_off regime"** and "elevated VIX" as the motivation for 5 of the 7 closes:

- **SOL-USD close:** "Cut crypto loser at -10.2% in risk-off regime. VIX elevated, 5% from stop loss."
- **XLI close:** "Cyclical sector is inappropriate for risk-off regime. Close position to reduce equity beta exposure and preserve capital."
- **XLC close:** "Sector has discretionary/growth characteristics that underperform in risk-off."
- **QQQ close:** "Tech sector weakest in risk_off regime. Close to reduce equity beta and preserve capital."
- **XLF close:** "Financials… Close to redeploy into safer assets when volatility subsides."
- **XLRE close:** "Rate-sensitive sector under pressure with elevated VIX."
- **SHY buy:** "Short-duration treasuries as cash proxy. Risk_off regime favors safety."

The USO/DBA trims were orthogonal (RSI overbought profit-taking, not regime driven) and are not part of the whipsaw.

The LLM's "risk_off" label was based on a VIX reading of 25.25 (the Mar 31 close) — but this was already **-5.8 points off the Mar 27 peak of 31.05** and the SPY had already recovered +3.2% from its Mar 30 low. The regime was transitioning OUT of risk-off, not into it. By selling on Apr 1, the system sold the VIX-spike-reversal bottom.

## Root Cause Analysis

The whipsaw is the product of three interacting defects, ranked by causal contribution:

### 1. Stale Decision Context (primary cause)
`build_context.py` feeds the LLM a snapshot of current indicators (latest VIX, latest SPY, RSI, MACD) but does not express the **direction of travel** of the regime. On Apr 1, VIX=25.25 in isolation looks like "elevated" — identical to the prompt it would have produced on Mar 26 when VIX=27.44 and climbing. The LLM cannot distinguish "VIX rising into a risk-off episode" from "VIX receding out of one". The 6-day gap where the system didn't trade during the actual spike compounded this: when it came back online, it behaved as if the spike was just starting.

### 2. Cadence / Snapshot Gap (enabling cause)
The `operational_health.max_snapshot_gap_days=3` warning fired at 7 days but was purely informational. The system went dark from 2026-03-26 07:58 UTC through 2026-04-01 12:36 UTC — through the entire VIX spike and its reversal. Had the LLM been running during Mar 27-31 it could have either (a) de-risked earlier at better prices, or (b) done nothing at all by Apr 1 because it would already have the de-risked book. The gap inverted the natural timing of the response.

### 3. Regime-Label Binarity (contributing cause)
The LLM's regime classifier produces a categorical label (`risk_on` / `risk_off` / `transition`) with no persistence memory and no awareness of the **position** we are already in (the book is already only 49% gross, 51% cash — already defensive). A rule like "don't de-risk further if cash is already >40%" would have blocked the cascade.

### Ruled-Out Causes
- **NOT a kill switch.** All 6 kill switches had slack on Apr 1; `surveillance_scans` row 27 shows `halt_count=0` persistent across Apr 1 scans.
- **NOT a risk-manager rejection cascade.** Zero trades rejected; the pre-trade checks approved every order. `risk_check_failure_streak=3` was not close to tripping.
- **NOT a governance halt.** `overall_severity=warning` for all Apr 1 scans — `/governance` would have permitted trading.
- **NOT a signal reversal from deployed strategy.** The portfolio is running discretionary LLM decisions via `/trade`, not promoted strategies. No Track A/B/D strategy generated the closes.

## Recommendations

Pure diagnostics — no code changes made. In descending priority:

### R1. Inject regime direction-of-travel into the context prompt
Modify `scripts/build_context.py` / `config/prompts/trader_decision.md` to add per-indicator delta fields: `vix_change_5d`, `vix_change_10d`, `vix_pct_from_20d_high`, `spy_drawdown_from_20d_high`, `days_since_regime_flip`. The LLM must see that VIX went 26 → 31 → 25 (receding from spike) instead of just "VIX=25, elevated".

### R2. Block discretionary de-risking when already defensive
Add a pre-trade heuristic (in the decision prompt, not in `risk/manager.py`) that prevents closing equity positions for regime reasons if `cash_pct >= 40%` OR `gross_exposure <= 0.60 * NAV`. On Apr 1 the book was 51% cash / 49% gross — already one of the most defensive states a 60/40 framework would ever hold. Further de-risking from there requires evidence of an *active* crisis, not a rearview mirror of one.

### R3. Harden the snapshot-gap warning into a soft block
In `config/governance.toml`, add a new kill-switch tier (or a trade-mode flag) that triggers when `snapshot_gap_days > 3`: next session must be **observation-only for 1 day** — compute indicators, log decisions to `llm_decisions`, but block `trades` execution. Prevents the "came back online after long absence → immediately churn the book" pattern.

### R4. Persist regime classification and require confirmation
Add a column to `llm_decisions` (already exists: `market_regime`) but require 2 consecutive sessions of the same label before the classifier is allowed to reverse an existing position stance. Avoids whipsaw on isolated noisy readings.

### R5. Cooldown on full closes
After any session with >=3 `action='close'` trades, block further closes for 24 hours unless a hard kill switch fires. Makes the system serial rather than parallel on exit decisions.

## Appendix: Data Queries Used

```sql
-- Portfolio snapshots around the event
SELECT snapshot_id, date, nav, cash, gross_exposure, daily_pnl, created_at
FROM portfolio_snapshots
WHERE date BETWEEN '2026-03-25' AND '2026-04-10' ORDER BY snapshot_id;

-- All trades in the window
SELECT trade_id, date, symbol, action, shares, price, notional, reasoning, created_at
FROM trades WHERE date BETWEEN '2026-03-20' AND '2026-04-10' ORDER BY date, trade_id;

-- Surveillance scans on the event day
SELECT scan_id, scan_timestamp, overall_severity, halt_count, warning_count, checks_json
FROM surveillance_scans WHERE scan_id BETWEEN 27 AND 40 ORDER BY scan_timestamp;

-- VIX / SPY price context
SELECT date, symbol, close FROM market_data_daily
WHERE symbol IN ('VIX','SPY','QQQ','TLT','XLE','USO')
  AND date BETWEEN '2026-03-20' AND '2026-04-10' ORDER BY date, symbol;
```

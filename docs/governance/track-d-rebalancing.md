# Track D Rebalancing Schedule — Sprint Alpha

Leveraged ETF strategies have fundamentally different rebalancing requirements from
standard equity strategies. This document defines the rebalancing schedule, position
sizing logic, transaction cost model, and operational rules for Track D.

---

## Why Rebalancing Frequency Matters More for Leveraged ETFs

Standard equity strategies tolerate weekly or monthly rebalancing. Leveraged ETFs do not.
Three structural forces make frequent rebalancing a survival requirement, not a
preference:

**1. Beta decay (volatility drag)**

A 3x leveraged ETF returns approximately `3 × daily_return - k × daily_variance²` where
`k` is the drag coefficient (roughly 4.5 for 3x products). Over a single day, drag is
negligible. Over a week of 1% daily moves, cumulative drag can consume 50-100bps of
return. Over a month, drag often exceeds the underlying signal.

Rule: signals that take longer than 5 calendar days to resolve are structurally
unsuitable for Track D. Don't force a slow signal into a fast vehicle.

**2. Volatility drag kills compounding**

For a 3x ETF, if the underlying moves +10% then -10%, the 3x ETF moves +30% then -30%:
- Underlying: $100 → $110 → $99 (-1% total)
- 3x ETF: $100 → $130 → $91 (-9% total)

The gap widens with volatility. High-volatility regimes (VIX > 25) make 3x ETFs
structurally losing even when the underlying trends slightly upward. Rebalancing
provides the exit mechanism when the regime shifts.

**3. TMF negative correlation decays fast**

TMF (3x TLT) is used as a hedge against TQQQ/UPRO in risk-off regimes. The
negative correlation between equities and long bonds is regime-dependent — it held
reliably from 2000-2021 but has broken down during inflationary episodes (2022, 2025).
Holding a TMF hedge for more than a few days without re-evaluating the correlation
regime risks holding two losing positions simultaneously. Daily correlation checks are
mandatory when TMF is in the portfolio.

---

## Rebalancing Schedule

### Tier 1: Daily checks (mandatory, always on)

Run every trading day regardless of position state:

| Check | Trigger | Action |
|-------|---------|--------|
| VIX spike | VIX intraday > 30 | Exit all leveraged long positions immediately |
| Regime flip | Overnight signal reversal | Exit affected position before market open |
| TMF correlation | Equity-bond 5-day rolling correlation > -0.1 | Exit TMF hedge, move to cash |
| 5-day holding timer | Any position held 5 calendar days | Force exit at next open |
| Single-day loss > 12% on any position | Intraday | Exit immediately, no waiting for close |

These checks run before the market open using prior-day close data. The VIX check
uses pre-market futures if available.

### Tier 2: Weekly rebalancing (signal-driven, Monday before open)

Run every Monday (or Tuesday if Monday is a holiday):

| Task | Logic |
|------|-------|
| Signal re-evaluation | Re-run Family 1 and Family 8 signals on fresh weekly data |
| Position sizing recalculation | Recompute ATR-based sizing with trailing 10-day volatility |
| MAR check | Compute rolling 90-day CAGR/MaxDD — flag if < 1.0 |
| Correlation matrix | Recompute TQQQ/SOXL/TMF correlations — flag if > 0.70 between two long positions |
| Transaction cost audit | Tally week's bid-ask costs — flag if > 0.5% of weekly P&L |

If Monday's signal re-evaluation shows no valid entry, hold cash. Do not manufacture
trades to stay invested. Cash is a position.

### Tier 3: Monthly review (first trading day of each month)

| Review | Decision |
|--------|----------|
| Beta decay audit | Measure actual 3x ETF return vs. 3x theoretical. If gap > 3% annualized, review signal holding period |
| Drawdown trajectory | If current drawdown > 25%, suspend new entries until recovery to -15% |
| MAR rolling window | If 90-day MAR < 1.0, file retirement issue and begin wind-down |
| Universe review | Check if any ETF in universe has had AUM drop below $500M (liquidity risk flag) |

---

## Position Sizing for Small Accounts ($5K-$25K)

Fractional share availability and minimum lot sizes create practical constraints
that backtests ignore. This section addresses the "paper math vs. real execution" gap.

### The fractional shares problem

Most brokers (Fidelity, Schwab) do not support fractional shares on leveraged ETFs.
At a $10K account:
- 30% position in TQQQ at $80/share = $3,000 = 37.5 shares → rounds to 37 shares = $2,960
- Rounding error: $40 (0.4% of account) — acceptable
- But at $5K account: 30% = $1,500 = 18.75 shares → rounds to 18 = $1,440 (4% rounding error)

Rule: For accounts under $10K, use 25% position sizing floor (not 30%) to reduce
rounding error impact. The position sizing model must always round down — never up.

### Account size tiers

| Account Size | Max Position | Max Simultaneous Positions | Cash Buffer |
|-------------|-------------|---------------------------|-------------|
| $5K-$10K | 25% | 2 | 15% |
| $10K-$25K | 35% | 2 | 12% |
| $25K+ | 50% | 2 | 10% |

Never run 3 simultaneous leveraged ETF positions regardless of account size. The
correlation between TQQQ and SOXL is typically > 0.85 during stress events — holding
both provides less diversification than it appears.

### Minimum viable position size

Below $1,000 per position, bid-ask spread costs consume signal alpha. For small
accounts where the 25% allocation yields less than $1,000:
- Skip the trade — log it as "below minimum viable size"
- Do not substitute with a smaller position — partial positions in leveraged ETFs
  produce asymmetric tracking error relative to the full-size backtest

---

## Transaction Cost Model

Backtests for Track D strategies must use realistic transaction cost assumptions.
Standard equity cost models (0.01% round-trip) understate leveraged ETF costs.

### Bid-ask spread estimates

| ETF | Typical Spread (normal) | Spread (high VIX > 25) | Notes |
|-----|------------------------|------------------------|-------|
| TQQQ | 0.05-0.10% | 0.15-0.25% | Most liquid 3x ETF |
| UPRO | 0.05-0.10% | 0.15-0.25% | Similar to TQQQ |
| SOXL | 0.10-0.20% | 0.30-0.50% | Semi sector — wider spread |
| TMF | 0.10-0.20% | 0.40-0.80% | Treasury 3x — wide in stress |
| TLTW | 0.15-0.30% | 0.50-1.00% | Covered call overlay — least liquid |

### Backtest cost assumptions

Apply the following round-trip costs in all Track D backtests:

| ETF | Round-trip cost (use in backtest) |
|-----|----------------------------------|
| TQQQ | 0.20% |
| UPRO | 0.20% |
| SOXL | 0.40% |
| TMF | 0.40% |
| TLTW | 0.60% |

These are pessimistic estimates — intentionally so. If a strategy doesn't survive
0.40% round-trip costs on SOXL, it won't survive real execution on a small account.

### Weekly rebalancing cost budget

With weekly rebalancing and 2 round-trips per week (in + out):
- TQQQ-only strategy: 2 × 0.20% × 52 weeks = 20.8% annualized cost drag
- Mixed TQQQ + SOXL: 2 × 0.30% × 52 = 31.2% annualized cost drag

This is the hurdle the signal must clear before a single dollar of alpha is earned.
Strategies with annualized gross return < 40% are unlikely to survive transaction
costs at weekly rebalancing frequency.

Implication: Track D strategies must demonstrate 60%+ gross CAGR in backtest to
have a realistic chance of 30%+ net CAGR in live trading.

---

## Recommended Operational Schedule

### Monday (weekly rebalance day)

1. Run Family 1 + Family 8 signal re-evaluation on Friday close data
2. Compute new position sizes using trailing 10-day ATR
3. Check VIX level — if > 25, scale position sizes to 50% of normal
4. Execute trades before 10:00 AM ET (avoid first 30-minute volatility)
5. Log entry prices, spread paid, intended exit date (max 5 days = Friday)

### Tuesday-Thursday (daily monitoring)

1. Run daily risk checks (Tier 1 list above)
2. No trades unless a Tier 1 trigger fires
3. Log daily P&L and running drawdown

### Friday (weekly close)

1. Force-exit any position held since Monday (5-day rule)
2. Compute week's MAR contribution
3. Update rolling 90-day MAR tracker
4. Prepare Monday's signal evaluation inputs

### End of month

1. Run Tier 3 monthly review
2. Update beta decay audit
3. Assess whether to continue, pause, or retire the strategy

---

## Integration with Track D Kill Conditions

The rebalancing schedule is the primary mechanism for enforcing Track D's kill
conditions. The daily checks catch fast failures; the weekly and monthly reviews
catch slow drift.

| Kill Condition | Detection Mechanism | Response Time |
|---------------|---------------------|---------------|
| MAR < 1.0 after 90 days | Monthly Tier 3 review | Next business day |
| Three consecutive weekly losses > 15% | Weekly MAR tracker | Monday of 4th week |
| Single-session loss > 20% | Daily Tier 1 check | Same day — intraday exit |
| Beta decay drag > 5% annualized | Monthly Tier 3 audit | Next monthly review |

When any kill condition triggers, file a beads issue with tag `track-d-retirement`
and begin wind-down. Do not restart the strategy without a full lifecycle reset.

---

## Automated Daily Monitoring

The Tier 1 daily checks described above are implemented in
`src/llm_quant/surveillance/track_d_monitor.py` and run automatically as part of
every `/governance` scan (pre-trade gate) via `SurveillanceScanner.run_full_scan()`.

Three detector functions are registered in `scanner.py`:

| Detector | Trigger | Severity |
|----------|---------|----------|
| `track_d_hold_periods` | Position held >= 4 days | WARNING; >= 5 days → HALT |
| `track_d_vix_regime` | VIX >= 25 | WARNING; VIX >= 30 with longs → HALT |
| `track_d_beta_decay` | Cumulative decay > 1% | WARNING (informational) |

When the scanner returns HALT from any Track D detector, `generate_forced_exit_signals()`
on `TrackDMonitor` produces SELL signals that must be executed before any new entries.

No manual invocation is required — these checks run every time `/governance` or
`/trade` is called.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-30 | Initial document — Track D rebalancing schedule, cost model, small account sizing. |
| 1.1 | 2026-03-30 | Add automated daily monitoring section; link to track_d_monitor.py. |

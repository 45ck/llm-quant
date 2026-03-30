# Track C Mandate vs Implementation Audit — 2026-03-30

**Auditor:** Governance Auditor Agent
**Issue:** llm-quant-nmnn
**Scope:** Verify gates in Track C mandate match implementation in `src/llm_quant/arb/`

---

## 1. Executive Summary

**The mandate file `data/strategies/niche-arbitrage/mandate.yaml` does not exist on disk.**

The mandate was described in beads issues (llm-quant-77th, llm-quant-m58r) and bd memory entries as having been created, but the file was never committed to git. The `data/strategies/niche-arbitrage/` directory does not exist.

The audit therefore proceeds against the mandate specification as documented in:
- `docs/governance/track-c-plan.md` (debate synthesis with risk framework)
- `docs/governance/research-tracks.md` (Track C section)
- Beads memory: `track-c-niche-arbitrage-launched` (gates: Sharpe>=1.5, MaxDD<10%, beta<0.15, DSR>=0.95, min 50 trades)
- Issue llm-quant-y7kg description (explicit list of gates)
- Issue llm-quant-tcdt description (risk manager integration requirements)
- Issue llm-quant-bfvd description (surveillance kill switches)

**Overall verdict: 5 of 13 gates are partially or fully implemented. 8 gates are missing or only stubs.**

---

## 2. Mandate Gate Specification (Reconstructed)

From the bd memory and issue corpus, the Track C mandate requires:

**Promotion Gates (robustness):**
1. Sharpe >= 1.5
2. MaxDD < 10%
3. Beta to SPY < 0.15
4. Min 50 trades
5. Cost stress test: survive 2x fees (Sharpe > 0 at 2x cost)

**Paper Validation Gates (PaperArbGate, for PM arb):**
6. Persistence: >= 50% of scan windows have >= 1 opportunity
7. Fill rate: >= 80% of opportunities estimated fillable
8. Capacity: position < 10% of avg market volume
9. Days elapsed: >= 30 calendar days of scan history

**Kill Switches:**
10. Exchange outage
11. Funding reversal (3 consecutive negative 8h funding periods)
12. Spread collapse (target spread < breakeven for 3 consecutive checks)
13. Counterparty alert (withdrawal delays > 24h)
14. Beta breach (rolling 30d beta to SPY > 0.15)

**Position Limits:**
15. Max 20% of Track C capital per strategy
16. Max $2K per individual prediction market position
17. Max $3K per individual CEF position
18. Max 25% on any single crypto exchange

---

## 3. Gate-by-Gate Implementation Audit

### 3.1 Sharpe >= 1.5 Gate

**Status: NOT IMPLEMENTED**

The PM arb paper gate (`src/llm_quant/arb/paper_gate.py`) does not include a Sharpe threshold gate. The 4 gates in `PaperArbGate` are: persistence, fill_rate, capacity, days_elapsed. Sharpe is not computed from paper trade results.

The main robustness module (`src/llm_quant/backtest/robustness.py`) applies DSR/PBO/CPCV gates designed for statistical strategies. Track C routing in `scripts/run_track_c_robustness.py` explicitly sets `dsr: None` and `pbo: None` for PM arb, replacing them with the 4 paper gates. No Sharpe >= 1.5 check appears anywhere in the Track C code path.

**Gap:** The mandate's headline Sharpe gate is not enforced in any promotion path.

---

### 3.2 MaxDD < 10% Gate

**Status: NOT IMPLEMENTED**

No MaxDD calculation or threshold check exists in `paper_gate.py`, `run_track_c_robustness.py`, or the arb execution engine. The CEF gate stub in `run_track_c_robustness.py` mentions `MaxDD<20%` (not 10%) as a note in a placeholder that returns REJECT immediately.

**Gap:** MaxDD gate is unimplemented. The CEF gate stub uses a different threshold (20%) than the mandate (10%).

---

### 3.3 Beta < 0.15 Gate

**Status: NOT IMPLEMENTED**

No beta calculation appears in any arb module. `src/llm_quant/risk/limits.py` and `src/llm_quant/risk/manager.py` contain no Track C beta check. The CEF gate stub mentions beta<0.15 in a comment but the gate returns REJECT without computing it. Issue llm-quant-y7kg explicitly tracks this as unimplemented.

**Gap:** Beta gate is entirely missing from the implementation.

---

### 3.4 Min 50 Trades Gate

**Status: NOT IMPLEMENTED**

No minimum trade count gate appears in any Track C code path. The PM arb paper gate counts scan windows (not individual trade executions). `paper_gate.py` `check_persistence()` uses scan-level data from `pm_scan_log`, not trade-level data from `pm_executions`. Issue llm-quant-y7kg explicitly tracks this as unimplemented.

**Gap:** Min 50 trades gate is missing.

---

### 3.5 Cost Stress Test (2x Fees)

**Status: PARTIALLY IMPLEMENTED (wrong path)**

`src/llm_quant/backtest/robustness.py` implements a `cost_2x_survives` check (line 69, line 424): `result.cost_2x_survives = cost_2x_sharpe > 0`. However, this is in the Track A/B robustness pipeline and is NOT wired into the Track C routing in `run_track_c_robustness.py`. The PM arb gate has no equivalent fee-stress test.

For PM arb, a cost stress test would need to re-run arb detection at 2x the fee rate (4% instead of 2% for Polymarket, 6% instead of 3% for Kalshi) and check whether net_spread remains positive. This is not implemented.

**Gap:** Cost stress test exists for Track A/B but is not applied to Track C strategies.

---

### 3.6 Paper Validation Gates (Persistence, Fill Rate, Capacity, Days Elapsed)

**Status: FULLY IMPLEMENTED (for PM arb / Kalshi only)**

`src/llm_quant/arb/paper_gate.py` implements all 4 gates with the correct thresholds:
- Persistence: >= 0.50 (fraction of scans with opportunities)
- Fill Rate: >= 0.80 (fraction of opps with sufficient volume)
- Capacity: <= 0.10 (position as fraction of avg volume, pass when BELOW threshold)
- Days Elapsed: >= 30 calendar days

These are wired into `scripts/run_track_c_robustness.py` for `pm-arb-*` slugs and output a `robustness-result.yaml` in the standard promote-compatible format.

**Compliant for PM arb. Not implemented for CEF or funding rate strategies.**

---

### 3.7 Kill Switch: Exchange Outage

**Status: NOT IMPLEMENTED**

No exchange health detector exists in `src/llm_quant/surveillance/detectors.py` or anywhere in the codebase. Issue llm-quant-bfvd explicitly tracks this as unbuilt. The main surveillance module monitors NAV drawdown, single-day loss, consecutive losing days, and data staleness — none of these are arb-specific.

**Gap:** Exchange outage kill switch does not exist.

---

### 3.8 Kill Switch: Funding Reversal

**Status: NOT IMPLEMENTED**

No funding reversal detector exists. Issue llm-quant-bfvd tracks "Funding rate reversal detector (3 consecutive negative 8h periods)" as unbuilt. The funding rate scanner (`src/llm_quant/arb/funding_scanner.py`) collects data but has no surveillance integration.

**Gap:** Funding reversal kill switch does not exist.

---

### 3.9 Kill Switch: Spread Collapse

**Status: NOT IMPLEMENTED**

No spread compression detector exists. Issue llm-quant-bfvd tracks "Spread compression detector (arb edge disappearing)" as unbuilt. The scanner detects current-moment spreads but does not track spread trends over time or trigger a halt.

**Gap:** Spread collapse kill switch does not exist.

---

### 3.10 Kill Switch: Counterparty Alert

**Status: NOT IMPLEMENTED**

No counterparty or exchange withdrawal monitoring exists. Issue llm-quant-bfvd tracks cross-strategy correlation spike detector but not specifically counterparty/withdrawal monitoring. Issue llm-quant-tcdt identifies this as a needed feature in the risk manager.

**Gap:** Counterparty alert kill switch does not exist.

---

### 3.11 Kill Switch: Beta Breach

**Status: NOT IMPLEMENTED**

No rolling beta-to-SPY monitor exists for Track C positions. Issue llm-quant-bfvd tracks "Beta drift detector (rolling 30d beta to SPY > 0.15)" as unbuilt. The existing kill switches in `src/llm_quant/risk/manager.py` (lines 791-1011) do not include a beta breach check.

**Gap:** Beta breach kill switch does not exist.

---

### 3.12 Position Limits (Risk Manager Integration)

**Status: NOT IMPLEMENTED**

`src/llm_quant/risk/manager.py` has no Track C position limits. The Track C execution engine (`src/llm_quant/arb/execution.py`) hardcodes its own limits independently:
- `NAV_USD = 100_000.0`
- `MAX_KELLY_FRACTION = 0.02` (2% of NAV = $2K cap per trade)
- `MIN_CONDITION_VOLUME = 100.0`

The $2K per prediction market cap matches the mandate. However, this is enforced only in the `KalshiArbExecution` class — it is NOT wired into the central risk manager (`src/llm_quant/risk/manager.py`). The Track A/B risk manager does not know about Track C positions. Issue llm-quant-tcdt explicitly tracks this integration as unbuilt.

**Partial compliance:** The execution engine self-enforces a $2K position cap, but this is isolated from the central risk framework. CEF position limits ($3K/CEF) and exchange concentration limits (25% max per exchange) are not enforced anywhere.

---

## 4. Summary Matrix

| Gate | Status | Location | Notes |
|------|--------|----------|-------|
| Sharpe >= 1.5 | NOT IMPLEMENTED | — | No Sharpe computed in Track C path |
| MaxDD < 10% | NOT IMPLEMENTED | — | CEF stub uses wrong threshold (20%) |
| Beta < 0.15 | NOT IMPLEMENTED | — | No beta calculation anywhere |
| Min 50 trades | NOT IMPLEMENTED | — | Paper gate counts scans, not trades |
| Cost stress (2x fees) | PARTIAL | backtest/robustness.py | Track A/B only, not wired to Track C |
| Persistence >= 50% | IMPLEMENTED | arb/paper_gate.py | PM arb only |
| Fill rate >= 80% | IMPLEMENTED | arb/paper_gate.py | PM arb only |
| Capacity < 10% | IMPLEMENTED | arb/paper_gate.py | PM arb only |
| Days elapsed >= 30 | IMPLEMENTED | arb/paper_gate.py | PM arb only |
| Kill switch: exchange outage | NOT IMPLEMENTED | — | Tracked in llm-quant-bfvd |
| Kill switch: funding reversal | NOT IMPLEMENTED | — | Tracked in llm-quant-bfvd |
| Kill switch: spread collapse | NOT IMPLEMENTED | — | Tracked in llm-quant-bfvd |
| Kill switch: counterparty | NOT IMPLEMENTED | — | Tracked in llm-quant-tcdt |
| Kill switch: beta breach | NOT IMPLEMENTED | — | Tracked in llm-quant-bfvd |
| Position limits (risk manager) | PARTIAL | arb/execution.py | $2K cap hardcoded, not centralized |
| CEF position limits | NOT IMPLEMENTED | — | $3K/CEF not enforced |
| Exchange concentration 25% | NOT IMPLEMENTED | — | Not enforced anywhere |

**Score: 4 of 17 gates fully implemented. 2 partially implemented. 11 not implemented.**

---

## 5. Critical Governance Findings

### Finding 1: Mandate file does not exist
`data/strategies/niche-arbitrage/mandate.yaml` was described in beads issues as having been created but was never committed to git. There is no canonical, machine-readable mandate document for Track C.

**Risk:** Any promotion check that reads the mandate file will fail silently or error. The `/promote` command cannot enforce mandate gates it cannot read.

### Finding 2: The 5 mandate promotion gates (Sharpe, MaxDD, Beta, Min Trades, Cost Stress) are unimplemented
The PM arb paper gate validates execution quality but not strategy-level performance gates. A PM arb strategy could pass paper gate validation while having Sharpe < 1.5 or beta > 0.15 and would still receive a PROMOTE recommendation.

**Risk:** Strategies could be promoted to live trading without meeting the mandate's minimum performance standards.

### Finding 3: All 5 kill switches are missing
No exchange outage, funding reversal, spread collapse, counterparty alert, or beta breach kill switch is implemented in the surveillance module. The existing kill switches cover NAV drawdown, single-day loss, and data staleness — which are relevant for Track A/B equity strategies but insufficient for arb strategies.

**Risk:** A funding reversal or exchange outage could result in directional exposure with no automatic halt.

### Finding 4: Track C is isolated from the central risk manager
The `KalshiArbExecution` engine enforces its own $2K position cap but does not report positions to `src/llm_quant/risk/manager.py`. Track C positions are invisible to the central risk framework. Portfolio-level exposure, gross/net exposure limits, and sector concentration checks do not include Track C.

---

## 6. Open Beads Issues Covering These Gaps

| Gate Gap | Tracking Issue |
|---------|----------------|
| Sharpe/MaxDD/Beta/Min Trades gates | llm-quant-y7kg (open) |
| Kill switches (all 5) | llm-quant-bfvd (open) |
| Risk manager integration | llm-quant-tcdt (open) |
| Paper trading deployment | llm-quant-anf0 (blocked on bfvd, tcdt) |

The open issues correctly identify the gaps. The mandate-vs-implementation divergence is a known technical debt, tracked and unresolved.

---

## 7. Recommended Actions

### Immediate
- [ ] Create `data/strategies/niche-arbitrage/mandate.yaml` with the authoritative gate thresholds and commit it to git.
- [ ] Block Track C promotion until at least the 5 mandate promotion gates are wired into `run_track_c_robustness.py`.

### Before any Track C strategy goes live
- [ ] Implement llm-quant-y7kg: add beta, Sharpe, MaxDD, and min-trades gates to the Track C robustness runner.
- [ ] Implement llm-quant-bfvd: build exchange outage, spread collapse, funding reversal, and beta breach kill switches in the surveillance module.
- [ ] Implement llm-quant-tcdt: integrate Track C position limits into the central risk manager.

### Acceptable to defer
- CEF and funding rate gate implementations (placeholders are honest stubs, strategies are not yet deployed).

---

*Audit date: 2026-03-30*

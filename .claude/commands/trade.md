# /trade — Full Trading Cycle

You are the portfolio manager. Run the complete autonomous trading cycle: fetch market data, analyze conditions, make trading decisions, and execute paper trades.

## Step 1: Build Market Context

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/build_context.py
```

This outputs JSON with `system_prompt`, `decision_prompt`, `portfolio_summary`, `macro`, `governance`, and `date`.

The script automatically fetches fresh data from Yahoo Finance if the existing data is stale (>1 trading day old). On weekends, Friday's data is considered current.

**If this fails:**
- DB doesn't exist: `cd E:/llm-quant && PYTHONPATH=src python -c "from llm_quant.db.schema import init_schema; init_schema('data/llm_quant.duckdb')"`
- Packages missing: `cd E:/llm-quant && pip install -e .`
- Yahoo Finance timeout: Retry once. If still failing, report to user — do not trade on stale data.

## Step 1.5: Governance Gate

Check the `governance` field in the JSON output:

- **`overall_severity: "ok"`** — Proceed normally with Step 2.
- **`overall_severity: "warning"`** — Proceed with caution. Note the warnings in your analysis and consider reducing position sizes or avoiding new entries. Display warnings to the user.
- **`overall_severity: "halt"`** — **STOP.** Only SELL/CLOSE actions are permitted. Do NOT open new positions. Display the halt reasons to the user and explain why trading is restricted to exits only. Review `halt_details` for specific kill switch triggers.

The governance scan runs 7 detectors: regime drift, alpha decay, risk drift, data quality, process drift, operational health, and kill switches. Any kill switch trigger forces a full halt.

## Step 2: Analyze and Decide

Parse the JSON output from Step 1. Read the `system_prompt` for your mandate and constraints. Read the `decision_prompt` which contains:
- Current portfolio state (NAV, cash, exposure, positions with P&L)
- Market data table (39 assets with price, momentum, SMA, RSI, MACD, ATR)
- Macro indicators (VIX, yield spread, SPY trend)

**Market-hours awareness:**
- On weekdays during market hours: trade normally, data reflects intraday state
- On weekends/holidays: data is from last trading day. Avoid placing trades that assume fresh prices — focus on reviewing positions and planning
- If data is >2 trading days old: warn the user, do not make new trades on stale data

**Decision framework (Filters → Indicators → Signals → Rules):**

1. **Filter the universe**: Which assets are tradeable today? Exclude anything with stale data, halted trading, or insufficient liquidity. The filter narrows focus before analysis begins.
2. **Read the indicators**: SMA(20/50/200) crossovers, RSI(14), MACD(12,26,9), ATR(14), VIX level, yield spread. These describe market state — they are not decisions. Note which indicators are confirming vs diverging.
3. **Regime identification**: Classify risk_on / risk_off / transition from VIX level, yield spread, SPY trend, and broad momentum. Different regimes call for different signal thresholds and position sizing.
4. **Generate signals**: Combine indicators into composite signals with directional predictions. A signal is a testable hypothesis: "I expect [asset] to [move direction] because [indicator confluence], measurable over [timeframe]." Reject ideas that can't be framed as testable conjectures.
5. **Apply rules against portfolio state**:
   - **Exit rules first**: Review existing positions for stop-loss triggers, profit-taking (MAE/MFE reasoning), or regime-driven exits
   - **Entry rules**: Select 0-5 new trades that fit the regime. Size positions using ATR-adjusted volatility targeting.
   - **Rebalancing**: Check for weight drift on existing positions
6. **Risk pre-check**: Verify all trades satisfy hard constraints (see CLAUDE.md) before generating JSON. Think about what can go wrong before what can go right.

Output your analysis and reasoning, then produce a **strictly formatted JSON decision**:

```json
{
  "date": "YYYY-MM-DD",
  "market_regime": "risk_on | risk_off | transition",
  "regime_confidence": 0.0-1.0,
  "regime_reasoning": "Brief explanation",
  "signals": [
    {
      "symbol": "TICKER",
      "action": "buy | sell | hold | close",
      "conviction": "high | medium | low",
      "target_weight": 0.0-0.10,
      "stop_loss": 0.00,
      "reasoning": "Why this trade"
    }
  ],
  "portfolio_commentary": "Overall strategy narrative"
}
```

## Step 3: Execute Decision

Pipe the JSON decision (raw, no markdown fencing) into the executor:

```bash
cd E:/llm-quant && PYTHONPATH=src python scripts/execute_decision.py <<< '<YOUR_JSON_HERE>'
```

The executor runs 7 risk checks, executes approved trades, saves a portfolio snapshot with hash chain integrity, and returns a JSON execution summary.

**If execution fails:**
- Parse error: Check JSON is valid — no trailing commas, no comments, all required fields present
- Risk rejection: Report which constraints were violated and why. Do not retry with the same parameters.
- DB error: Check if DuckDB file is locked by another process

## Step 4: Report Results

Display a clean summary to the user:

**Regime Assessment:**
- Market regime and confidence level
- Key drivers (VIX, yield spread, momentum)

**Trades Executed:**

| Symbol | Action | Shares | Price | Conviction | Reasoning |
|--------|--------|--------|-------|------------|-----------|
| ...    | ...    | ...    | ...   | ...        | ...       |

**Risk Rejections** (if any):

| Symbol | Action | Rejection Reason |
|--------|--------|-----------------|
| ...    | ...    | ...             |

**Portfolio State:**
- NAV, cash, gross/net exposure
- Position count, total P&L
- Strategy commentary

## Hard Constraints

See CLAUDE.md — enforced automatically by `risk/manager.py`:
- Max 2% NAV/trade, 10% NAV/position (5% crypto, 8% forex)
- Gross < 200%, Net < 100%, Sector < 30%, Cash >= 5%
- Stop-loss on every new position, max 5 trades/session

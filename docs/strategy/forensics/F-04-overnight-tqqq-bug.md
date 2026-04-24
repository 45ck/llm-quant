# Forensic Finding F-04 — overnight-tqqq-sprint Is a 30-Minute Bug

**Question:** Agent 3 claimed `overnight-tqqq-sprint` is a 30-minute param-alias bug, not a strategy redesign. Verify.

## The Bug — Confirmed

### Strategy class expects (`src/llm_quant/backtest/strategies.py:2097-2102`)
```python
params = self.config.parameters or {}
symbol: str = str(params.get("symbol", "SPY"))           # default "SPY"
window: int = int(params.get("window", 10))
entry_thresh: float = float(params.get("entry_thresh", 0.002))
exit_thresh: float = float(params.get("exit_thresh", -0.0005))
tgt_weight: float = float(params.get("target_weight", 0.90))
```

### Frozen spec writes (`data/strategies/overnight-tqqq-sprint/research-spec.yaml:9-14`)
```yaml
parameters:
  ticker: "TQQQ"                         # ← strategy reads "symbol", not "ticker"
  signal: "overnight_return"             # ← strategy doesn't read this
  rebalance_frequency_days: 1            # ← strategy doesn't read this
  target_position_weight: 0.30           # ← strategy reads "target_weight"
  stop_loss_pct: 0.05                    # ← strategy uses hardcoded 0.95 (5% stop)
```

### What happens at runtime
1. Spec writes `ticker: "TQQQ"`. Strategy reads `params.get("symbol", "SPY")` → falls back to `"SPY"`.
2. Strategy filters `indicators_df` for `symbol == "SPY"`.
3. But the fetcher only fetched `["TQQQ"]` (per `backtest_spec.symbols` and the experiment registry). No SPY rows in the indicators DataFrame.
4. `len(ind) < window + 1` → returns empty list → no trades.
5. Result: experiment recorded with `total_trades: 0, sharpe: 0.0, max_drawdown: 0.0`.

The experiment registry confirms the data was correctly fetched:
```json
"symbols": ["TQQQ"]    ← TQQQ data was loaded
"parameters": {"ticker": "TQQQ", ...}   ← spec params passed through verbatim
"total_trades": 0, "sharpe_ratio": 0.0  ← strategy returned empty signals
```

The strategy **silently failed** because no exception was raised — `return []` is a valid empty-signal response.

## Why This Happened

The spec was authored to a **fictional schema** that the strategy class doesn't implement. The keys (`ticker`, `signal`, `rebalance_frequency_days`, `target_position_weight`, `stop_loss_pct`) match a *general* leveraged-ETF parameter convention used elsewhere in the codebase (e.g., the Track D sprints in `LEAD_LAG_PARAMS` use `target_weight`, but other places use `target_position_weight`). There is **no spec → strategy contract validation** anywhere.

The spec was frozen and committed without ever being executed against the actual strategy. Trial 1 ran with the wrong params, produced 0 trades, and the failure was attributed to "F11 may not support TQQQ" (per `docs/research/results/track-d-sprint-results-2026-03-31.md`) rather than to a parameter-naming issue.

## The Fix — 30 Minutes

### Option A — Patch the strategy (recommended)
Add 5 lines of alias resolution in `src/llm_quant/backtest/strategies.py:2097`:

```python
params = self.config.parameters or {}
# Param aliasing for spec-naming drift
symbol: str = str(params.get("symbol", params.get("ticker", "SPY")))
window: int = int(params.get("window", 10))
entry_thresh: float = float(params.get("entry_thresh", 0.002))
exit_thresh: float = float(params.get("exit_thresh", -0.0005))
tgt_weight: float = float(params.get("target_weight",
                                     params.get("target_position_weight", 0.90)))
```

Re-run backtest, expect a non-zero result. Effort: 5 min code + 10 min re-run + 5 min verify trial 2 in registry.

### Option B — Re-freeze spec with corrected keys
Edit spec, set `frozen: false`, edit keys, set `frozen: true`, recompute hash. Per the lifecycle this requires a new version. Slightly more bureaucratic, equally fast.

## Larger Pattern

**There is no validation that a frozen spec can actually be consumed by its declared `strategy_type`.** This means any spec with a typo, a renamed key, or a key that drifts in convention will silently produce zero trades and be misattributed to "the strategy doesn't work."

**Recommended infrastructure addition:** add `Strategy.validate_params(params: dict) -> list[str]` returning the list of unrecognized or missing keys. Fail spec-freeze if any unknown keys are present.

This single check would have caught the `ticker` vs `symbol` drift at freeze time, before any wasted trial.

## Confidence

**Maximum.** The bug is mechanically reproducible by reading 50 lines of code. Fix is 5 lines.

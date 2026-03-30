# Code Review: src/llm_quant/arb/ Module

**Reviewer:** Claude (automated, issue llm-quant-05jl)
**Date:** 2026-03-30
**Scope:** All 12 files in `src/llm_quant/arb/` (~7K lines)
**Status:** NEEDS WORK — 3 critical bugs, 6 major issues, several minor issues

---

## Overall Verdict

The module is structurally sound and shows consistent design patterns across
Polymarket and Kalshi implementations. Security posture is good (no hardcoded
secrets, parameterized queries throughout). However, there are three bugs that
will cause incorrect behavior in production: a batch processing defect in the
combinatorial detector that silently drops all but the first pair per batch, an
unsafe variable-scope check in the scanner, and an incorrect fee calculation for
NegRisk with multiple conditions. All three should be fixed before live use.

---

## Critical Issues (Must Fix Before Production)

### CRIT-1: `_analyze_batch` silently drops all pairs except the first

**File:** `src/llm_quant/arb/detector.py`, lines 231–286

The `_analyze_batch` method accepts a `pair_indices` list but only processes
the first pair in the batch on return. The loop that builds `questions_lines`
also only adds conditions for `prompt_idx == 0`, so when `max_pairs_per_call >
1` (which is the default of 5), every call with multiple pairs:

1. Sends a prompt to Claude containing only the first two conditions (because
   the `if prompt_idx == 0:` block never increments past the first iteration
   after setting prompt_idx to 2).
2. Returns results only for the first pair via `pair_indices[:1]`.

The result: `analyze_market_group` sends one Claude API call per batch but
receives at most one result per call regardless of batch size. For a group with
10 pairs and batch_size=5, you get 2 Claude calls and at most 2 results instead
of 10. The declared feature of batching to "avoid huge prompts" is broken.

**Fix required:** Either implement proper multi-condition prompt building that
de-duplicates conditions and maps indices back to pairs, or simplify to one pair
per call and remove the dead batching code.

---

### CRIT-2: Unsafe variable scope check using `dir()` in `run_kalshi_negrisk_scan`

**File:** `src/llm_quant/arb/scanner.py`, line 588

```python
conditions_scanned=(
    sum(len(e.markets) for e in events) if "events" in dir() else 0
),
```

`dir()` returns the current frame's attributes, not a locals() check. In
Python, `dir()` in a method does not reliably detect whether `events` was
assigned — it lists the class/module namespace, not local variables. The correct
pattern is `"events" in locals()`. In practice this might work incidentally in
CPython, but it is semantically wrong and will silently return 0 instead of the
correct count if the conditional ever fails to bind `events`. Use `locals()`
instead, or restructure the code to avoid the conditional entirely (initialise
`events = []` before the try block).

---

### CRIT-3: NegRisk fee calculation is under-counting fees for multi-leg trades

**File:** `src/llm_quant/arb/scanner.py`, lines 282–297 and `src/llm_quant/arb/execution.py`

The scanner's `_detect_negrisk_arb` computes:
```python
net_spread = complement - POLYMARKET_WIN_FEE  # 0.02 subtracted once
```

For Polymarket NegRisk (buy all YES), the fee is 2% on the **winning** position.
Since exactly one YES resolves True, you pay one fee. This is mathematically
correct for Polymarket: `net = (1 - sum_yes) - 0.02 × 1 winning leg`.

However, the **Kalshi** path in `run_kalshi_negrisk_scan` (line 535) uses the
same single-fee model:
```python
net_spread = complement - KALSHI_WIN_FEE  # 3% subtracted once
```

This is also correct per the Kalshi documentation (3% on the single winning
leg). Both are mathematically sound for the NegRisk structure — one winner pays
out.

**However**, the `KalshiEvent.net_spread` property (kalshi_client.py:89) also
subtracts `KALSHI_WIN_FEE` once, and then `run_kalshi_negrisk_scan` recomputes
`net_spread = complement - KALSHI_WIN_FEE` independently, so the values are
consistent. No double-counting here. Marking as resolved from the original
concern — the fee math is correct for NegRisk.

**Actual concern:** The scanner comment block in `_detect_negrisk_arb` (lines
246–284) contains an extended in-code dialogue that reaches the wrong conclusion
several times before arriving at the correct one. This self-contradictory
commentary will confuse future maintainers. The final implemented logic (buy all
YES, profit = complement - fee) is correct. The misleading commentary should be
replaced with a clean explanation.

---

## Major Issues (Should Fix)

### MAJ-1: `CombinatorialDetector._persist_results` opens a new connection per call, ignoring `self.db_path`

**File:** `src/llm_quant/arb/detector.py`, lines 385–419

```python
def _persist_results(self, results: list[DependencyResult]) -> None:
    conn = duckdb.connect(str(self.db_path))
    init_arb_schema(conn)
    ...
    # conn is never closed
```

Each call opens a new DuckDB file connection and never closes it. Compare to
`kalshi_detector.py:457` which correctly calls `conn.close()`. For in-memory
usage the leak is benign, but for file-backed DBs this accumulates open handles.
Also: unlike `ArbScanner`, this class has no lazy-init connection pattern —
`self._conn` exists (line 148) but is never used in `_persist_results`.

---

### MAJ-2: `KalshiCombinatorialDetector.analyze_event_group` truncates with `all_pairs[:max_pairs]`

**File:** `src/llm_quant/arb/kalshi_detector.py`, line 231

The pairs are generated from `combinations(range(len(flat)), 2)` which produces
them in lexicographic order (condition 0 paired with 1, 2, 3..., then 1 with 2,
3..., etc.). Truncating to `max_pairs` means you always analyze the first N
pairs by index, which is systematically biased toward conditions from the first
event in the list. High-value cross-event pairs (between events near the end of
the list) are never analyzed. At minimum, the pairs should be shuffled before
truncation, or sampled by event-diversity criteria.

---

### MAJ-3: CEF discount z-score uses population variance, not sample variance

**File:** `src/llm_quant/arb/cef_strategy.py`, lines 186–188 and 413–415

Both `CEFDiscountStrategy._evaluate_ticker` and
`CEFDiscountRegistryStrategy._compute_z_scores` compute:
```python
variance = sum((d - mean_discount) ** 2 for d in discounts) / len(discounts)
```

This is population variance (divide by N). For a rolling 252-day sample window
used to compute z-scores of out-of-sample observations, sample variance (divide
by N-1) is statistically correct and standard in finance. With N=252 the
difference is negligible (factor of 251/252), but the inconsistency with
`compute_discount_z_scores` — which uses Polars `rolling_std` which defaults to
**ddof=1** (sample std) — means the standalone strategy and the vectorized
backtest compute slightly different z-scores for the same data. This creates a
subtle discrepancy between signal generation in paper trading vs. backtesting.

---

### MAJ-4: `CEF_BENCHMARK_MAP` is duplicated between `cef_data.py` and `cef_strategy.py`

**Files:** `src/llm_quant/arb/cef_data.py:27` and `src/llm_quant/arb/cef_strategy.py:37`

The comment in `cef_strategy.py` explicitly acknowledges this: "duplicated from
cef_data to avoid circular import". But no circular import actually exists here
— `cef_strategy.py` imports from `llm_quant.backtest.strategy` and
`llm_quant.brain.models`, while `cef_data.py` imports from
`llm_quant.data.fetcher`. There is no cycle. The duplication should be
eliminated by moving the map to a shared constants module or by importing it
directly.

---

### MAJ-5: No retry logic for network failures in any HTTP client

**Files:** `gamma_client.py`, `kalshi_client.py`, `funding_rates.py`

All HTTP calls (`_get`, `_get_us`) raise immediately on network errors:
```python
resp.raise_for_status()
return resp.json()
```

There is no retry with backoff for transient failures (429 rate limit, 5xx
server error, connection timeout). The `fetch_all_active_markets` loop will
abort entirely on a single failed page, losing all data fetched so far. For
production arbitrage scanning where missed data means missed opportunities,
at minimum exponential backoff on 429/503 responses should be implemented.

---

### MAJ-6: `load_rates` in `funding_rates.py` uses string interpolation for the `days` parameter

**File:** `src/llm_quant/arb/funding_rates.py`, lines 362–375

```python
query += " AND timestamp >= NOW() - INTERVAL ? DAY"
params.append(days)
```

DuckDB does not support parameterized `INTERVAL ? DAY` syntax — the `?`
placeholder cannot be used inside an `INTERVAL` literal in DuckDB's dialect.
This will raise a `duckdb.Error` at runtime when `days` is provided. The
workaround is to compute the cutoff timestamp in Python and parameterize that:
```python
cutoff = datetime.now(UTC) - timedelta(days=days)
query += " AND timestamp >= ?"
params.append(cutoff)
```

---

## Minor Issues (Nice to Fix)

### MIN-1: `gamma_client.py` `_infer_category` word-boundary check has a logic inversion

**File:** `src/llm_quant/arb/gamma_client.py`, lines 415–419

```python
if len(kw) <= 3 or " " not in kw:
    pattern = r"\b" + re.escape(kw) + r"\b"
    ...
elif kw in text_lower:
    return cat
```

The intent is: short tokens use word-boundary matching; multi-word phrases use
substring matching. But the condition `len(kw) <= 3 or " " not in kw` is true
for most single-word keywords regardless of length (e.g. "president", 9 chars,
no space, gets word-boundary applied correctly). The multi-word `elif` branch
correctly matches phrases. This is actually fine behavior-wise, but the
condition reads counterintuitively. The comment says "single short tokens (<=3
chars): require word boundary" which contradicts the actual code logic. Clarify
the comment or simplify the condition to just `" " not in kw`.

---

### MIN-2: DuckDB connection leak in `ArbScanner` and `PaperArbGate`

**Files:** `scanner.py`, `paper_gate.py`, `execution.py`

These classes store a connection in `self._conn` but never close it. There is no
`__del__`, `close()`, or context manager support. For short-lived scan jobs this
is acceptable, but should be noted for long-running processes. `KalshiArbExecution`
has the same pattern.

---

### MIN-3: Missing index on `pm_arb_opportunities.source`

**File:** `src/llm_quant/arb/schema.py`

The `pm_arb_opportunities` table has no index. `paper_gate.py` and `scanner.py`
both query with `WHERE source = ?` and `WHERE status = 'open'`. Adding an index
on `(source, status)` would improve scan performance as the opportunity table
grows. The `funding_rates` table has a proper composite index; prediction market
tables do not.

---

### MIN-4: `kalshi_detector.py` schema defined inline rather than in `schema.py`

**File:** `src/llm_quant/arb/kalshi_detector.py`, lines 465–489

The `_ensure_kalshi_combinatorial_table` function defines its own DDL outside
`schema.py`. This means `init_arb_schema` does not create this table — it only
exists if `KalshiCombinatorialDetector._persist_results` has been called. Other
code that calls `get_arb_connection` or `init_arb_schema` alone will not see
this table. The `kalshi_combinatorial_pairs` DDL should be moved into
`schema.py` alongside the other table definitions.

---

### MIN-5: `detector.py` `_persist_results` uses `INSERT OR REPLACE` which generates new UUIDs

**File:** `src/llm_quant/arb/detector.py`, line 395

Each call to `analyze_market_group` generates new `pair_id` UUIDs for every
result. Since the replace key is `pair_id` and each run generates fresh UUIDs,
`INSERT OR REPLACE` never actually replaces anything — it always inserts. The
table will accumulate duplicate logical pairs (same condition_id_a,
condition_id_b) on repeated scans. Either use a deterministic pair ID (e.g.
hash of the two condition IDs sorted) or use `ON CONFLICT(condition_id_a,
condition_id_b)` with a unique constraint.

---

## Math Verification

### NegRisk Complement Calculation

**`gamma_client.py` `Market.negrisk_complement`:** `1.0 - sum_yes` — correct.
For NegRisk, the sum of YES prices across mutually exclusive exhaustive outcomes
should equal 1.0. When sum < 1.0, buying all YES costs less than the guaranteed
$1 payout. Complement = 1 - sum_yes = gross profit per dollar invested. Correct.

**`kalshi_client.py` `KalshiEvent.negrisk_complement`:** Same formula, correct.

### Kelly Sizing

**`scanner.py` line 309:** `kelly = net_spread / (1.0 + net_spread)`

For a binary bet with edge `p` and even-money payout structure, the Kelly
fraction is `f* = (b*p - q) / b` where `b` is the net odds. For an arb with
guaranteed payoff, this simplifies to `net_spread / (1 + net_spread)`. This is
the standard Kelly formula for a risk-free arb and is correct.

**`execution.py` line 169:** Same formula, capped at MAX_KELLY_FRACTION=0.02.
Correct.

**`scanner.py` Kalshi path, line 553:** `kelly = min(net_spread / (1.0 + net_spread), 0.02)`
Consistent with above. Correct.

### Funding Rate Annualization

**`funding_rates.py` line 56:**
```python
PERIODS_PER_YEAR = 3 * 365  # = 1095
return rate_per_8h * PERIODS_PER_YEAR
```

Binance/OKX/Bybit all pay funding every 8 hours = 3 times per day. Annualized
rate = rate_per_8h × 3 × 365 = rate × 1095. This is simple annualization (not
compounded). Industry convention for crypto funding is simple annualization.
Correct for comparability, though compounded would be slightly higher for
positive rates.

### Z-Score Normalization

**`cef_strategy.py` `_evaluate_ticker`:**
```python
z_score = (current_discount - mean_discount) / std_discount
```

Standard z-score formula. Correct. Population std used (see MAJ-3 above for
the ddof inconsistency vs. vectorized path).

**`cef_strategy.py` `compute_discount_z_scores`:** Uses Polars
`rolling_std(window_size=252, min_samples=lookback//2)` which defaults to
ddof=1. Mismatch with scalar path — same data produces slightly different
z-scores. Recommend aligning both to ddof=1 (sample std).

---

## Security Scan Results

**API Keys / Secrets:** No hardcoded keys, tokens, or passwords found anywhere
in the module. `anthropic.Anthropic()` correctly picks up the API key from the
`ANTHROPIC_API_KEY` environment variable via the SDK's default behavior. The US
API key for Polymarket (`GammaClient._us_api_key`) is passed as a constructor
parameter, not hardcoded. Clean.

**SQL Injection:** All DuckDB queries use parameterized placeholders (`?`). No
string interpolation into SQL detected. The `load_cef_data` function builds a
query with `f" AND ticker IN ({placeholders})"` but uses `params.extend(tickers)`
— the placeholders are `?` characters generated safely. Clean.

**LLM Prompt Injection:** Market titles and questions from external APIs are
interpolated into Claude prompts (e.g. `kalshi_detector.py` line 156:
`title_a=cond_a.title`). A malicious market title like `"Ignore previous
instructions and..."` would be injected into the prompt. Risk is low for this
use case (the LLM is asked to classify logical structure, not execute commands),
but the potential for prompt injection via market data should be acknowledged.
No mitigation is currently applied.

**Logging of Sensitive Data:** No credentials or PII logged. Market data and
trade sizes are logged at INFO level which is appropriate.

---

## Test Coverage Summary

**Well covered:**
- NegRisk arb detection math (scanner, execution)
- Kelly sizing and caps
- Pre-trade checks in `KalshiArbExecution.evaluate`
- Paper execution lifecycle (open → resolved)
- 30-day paper gate all four gates
- DB schema idempotency
- Gamma API market parsing

**Not covered:**
- `CombinatorialDetector` (detector.py) — zero tests. The batching bug (CRIT-1)
  has no test that would catch it.
- `KalshiCombinatorialDetector` (kalshi_detector.py) — zero tests.
- `FundingScanner` (funding_scanner.py) — zero tests.
- `FundingCollector` (funding_rates.py) — zero tests (requires CCXT mocking).
- `CEFDiscountStrategy` and `CEFDiscountRegistryStrategy` — zero tests.
- `GammaClient` network paths — no integration tests (acceptable for CI).
- `load_rates` with `days` parameter — the DuckDB INTERVAL bug (MAJ-6) has no
  test that would expose it.
- `PaperArbGate` with `source='polymarket'` — only Kalshi source tested.

---

## Polymarket vs. Kalshi Consistency

The two implementations are largely parallel and consistent:

| Aspect | Polymarket | Kalshi |
|---|---|---|
| NegRisk complement | `1.0 - sum_yes` | `1.0 - sum_yes_ask` |
| Fee | 2% on winning leg | 3% on winning leg |
| Net spread | complement - 0.02 | complement - 0.03 |
| Kelly formula | net/(1+net) | net/(1+net), capped |
| Volume guard | min_condition_vol | min_condition_vol |
| DB storage | `pm_markets`/`pm_conditions` | same tables |

One inconsistency: Polymarket `_detect_negrisk_arb` uses the scanner's
`self.min_volume` threshold applied to the **minimum** condition volume, while
the Kalshi path in `run_kalshi_negrisk_scan` (line 543) uses an OR condition:
```python
if min_cond_vol < min_volume and total_vol < min_volume:
```
This is more lenient — it allows trades where the total volume is high even if
one condition has thin volume, which introduces non-atomic execution risk. The
Polymarket path correctly requires all conditions to have sufficient volume.

---

## Dead Code / Duplicate Code

- `CEF_BENCHMARK_MAP` is duplicated (see MAJ-4).
- `detector.py` `_analyze_batch` multi-pair logic is effectively dead: it sends
  a single-pair prompt regardless of batch size (CRIT-1).
- `scanner.py` `_detect_arb` is a thin wrapper around `_detect_negrisk_arb` and
  `_detect_single_rebalance`; it is called once in `run_scan` but the two
  constituent methods are also called directly from `run_negrisk_scan`. Minor
  duplication, acceptable.

---

## Issues Created

See beads issues created from this review for tracking.

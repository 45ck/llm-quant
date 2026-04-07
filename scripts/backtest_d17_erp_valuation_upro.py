#!/usr/bin/env python3
"""Backtest + Robustness: d17-erp-valuation-upro (Track D - Sprint Alpha).

Hypothesis: The F30 Equity Risk Premium valuation regime signal (SPY 1yr return
minus TNX yield, z-scored) can be re-expressed through UPRO (3x S&P 500) for
higher CAGR. When equities are CHEAP (ERP z-score > 1), allocate to UPRO.
When equities are EXPENSIVE (ERP z-score < -1), go defensive.

Causal thesis: ERP measures whether equity returns compensate investors for risk
relative to risk-free rates. High ERP (cheap equities) historically precedes
above-average equity returns. UPRO amplifies this signal 3x.

Signal logic (F30 → Track D):
  1. Compute SPY trailing 252-day return (annualized)
  2. Get TNX yield (decimal, /100 since Yahoo reports %)
  3. ERP = SPY_1yr_return - TNX_yield
  4. Z-score ERP over rolling 252 days
  5. Regime:
     - EQUITY_CHEAP (z > 1.0): target_weight% UPRO + rest SHY
     - EQUITY_EXPENSIVE (z < -1.0): 40% TLT + 30% GLD + 30% SHY
     - NEUTRAL (-1 <= z <= 1): 50% SPY + 50% SHY
     - VIX > 30 override: 100% SHY
  6. Rebalance every 5 days
  7. Cost: 20 bps per regime switch

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/backtest_d17_erp_valuation_upro.py
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, "src")

import polars as pl

from llm_quant.backtest.robustness import shuffled_signal_test
from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "d17-erp-valuation-upro"
SYMBOLS = ["SPY", "UPRO", "GLD", "TLT", "SHY"]
TNX_SYMBOL = "^TNX"
VIX_SYMBOL = "^VIX"
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 300  # need 252 for 1yr return + 252 for z-score + buffer

BASE_PARAMS = {
    "erp_lookback": 252,  # 1yr return window
    "zscore_window": 252,  # z-score rolling window
    "cheap_threshold": 1.0,  # z > 1 = cheap
    "expensive_threshold": -1.0,  # z < -1 = expensive
    "target_weight": 0.35,  # UPRO allocation in cheap regime
    "neutral_spy_weight": 0.50,  # SPY weight in neutral regime
    "tlt_defensive_weight": 0.40,  # TLT in expensive regime
    "gld_defensive_weight": 0.30,  # GLD in expensive regime
    "shy_defensive_weight": 0.30,  # SHY in expensive regime
    "vix_crash_threshold": 30,  # VIX crash override
    "rebalance_days": 5,  # rebalance frequency
    "cost_bps": 20,  # cost per regime switch
}

print("=" * 70)
print(f"BACKTEST + ROBUSTNESS: {SLUG} (Track D - Sprint Alpha)")
print("=" * 70)

# ==============================================================================
# FETCH DATA
# ==============================================================================
print("\nFetching data...")
prices = fetch_ohlcv([*SYMBOLS, TNX_SYMBOL, VIX_SYMBOL], lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price lookup by symbol
sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# TNX data -- stored as "TNX" in fetcher (YAHOO_SYMBOL_MAP maps TNX -> ^TNX)
tnx_key = "TNX"
tnx_df = prices.filter(pl.col("symbol") == tnx_key).sort("date")
if len(tnx_df) == 0:
    tnx_df = prices.filter(pl.col("symbol") == TNX_SYMBOL).sort("date")
    tnx_key = TNX_SYMBOL
tnx_data: dict = dict(
    zip(tnx_df["date"].to_list(), tnx_df["close"].to_list(), strict=False)
)
print(f"TNX data points: {len(tnx_data)}")

# VIX data
vix_key = "VIX"
vix_df = prices.filter(pl.col("symbol") == vix_key).sort("date")
if len(vix_df) == 0:
    vix_df = prices.filter(pl.col("symbol") == VIX_SYMBOL).sort("date")
    vix_key = VIX_SYMBOL
vix_data: dict = dict(
    zip(vix_df["date"].to_list(), vix_df["close"].to_list(), strict=False)
)
print(f"VIX data points: {len(vix_data)}")

# Use SPY as date backbone
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Precompute daily returns for each asset
daily_rets: dict[str, list[float]] = {}
for sym in SYMBOLS:
    rets = [0.0]
    data = sym_data[sym]
    for i in range(1, n):
        d, dp = dates[i], dates[i - 1]
        if d in data and dp in data and data[dp] > 0:
            rets.append(data[d] / data[dp] - 1)
        else:
            rets.append(0.0)
    daily_rets[sym] = rets

# Precompute TNX yield array aligned to dates (Yahoo reports as %, convert to decimal)
tnx_yields = []
for d in dates:
    val = tnx_data.get(d)
    tnx_yields.append(val / 100.0 if val is not None else None)


def _compute_metrics(returns):
    """Compute Sharpe, MaxDD, total return, CAGR from daily returns."""
    if not returns or len(returns) < 60:
        return {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "daily_returns": [],
        }

    nav = [1.0]
    for r in returns:
        nav.append(nav[-1] * (1.0 + r))

    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    mean_r = sum(returns) / len(returns)
    std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
    total_ret = nav[-1] / nav[0] - 1.0
    years = len(returns) / 252
    cagr = ((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_ret,
        "cagr": cagr,
        "daily_returns": returns,
    }


def run_single(params):  # noqa: PLR0912
    """Run a single backtest with given parameters.

    Signal logic (lag-1, no look-ahead):
    1. Compute SPY trailing 252-day return ending YESTERDAY
    2. Get TNX yield as of YESTERDAY (converted from % to decimal)
    3. ERP = SPY_1yr_return - TNX_yield
    4. Z-score ERP over rolling 252-day window ending YESTERDAY
    5. Regime classification based on z-score thresholds
    6. VIX crash override
    """
    erp_lookback = int(params.get("erp_lookback", 252))
    zscore_window = int(params.get("zscore_window", 252))
    cheap_thresh = float(params.get("cheap_threshold", 1.0))
    expensive_thresh = float(params.get("expensive_threshold", -1.0))
    target_weight = float(params.get("target_weight", 0.35))
    neutral_spy_w = float(params.get("neutral_spy_weight", 0.50))
    tlt_def_w = float(params.get("tlt_defensive_weight", 0.40))
    gld_def_w = float(params.get("gld_defensive_weight", 0.30))
    shy_def_w = float(params.get("shy_defensive_weight", 0.30))
    vix_thresh = float(params.get("vix_crash_threshold", 30))
    rebalance_freq = int(params.get("rebalance_days", 5))
    cost_bps = float(params.get("cost_bps", 20))

    cost_per_switch = cost_bps / 10000.0
    returns = []
    prev_regime = None
    current_regime = None
    days_since_rebal = 0

    min_start = max(WARMUP, erp_lookback + zscore_window + 1)

    # Precompute ERP series for z-score calculation
    erp_history: list[float | None] = [None] * n
    for i in range(erp_lookback, n):
        # SPY 1yr return ending yesterday (i-1)
        idx_start = i - erp_lookback
        if idx_start < 0 or spy_close[idx_start] <= 0:
            continue
        spy_1yr_ret = spy_close[i - 1] / spy_close[idx_start] - 1.0
        # TNX yield as of yesterday
        tnx_y = tnx_yields[i - 1]
        if tnx_y is None:
            continue
        erp_history[i] = spy_1yr_ret - tnx_y

    for i in range(min_start, n):
        days_since_rebal += 1
        evaluate_signal = (days_since_rebal >= rebalance_freq) or (
            current_regime is None
        )

        if evaluate_signal:
            days_since_rebal = 0

            # VIX crash filter
            d = dates[i]
            vix_level = vix_data.get(d, 0.0)

            if vix_level > vix_thresh:
                current_regime = "crash"
            else:
                # Compute z-score of ERP using history ending at i
                erp_window = []
                for j in range(i - zscore_window, i + 1):
                    if 0 <= j < n and erp_history[j] is not None:
                        erp_window.append(erp_history[j])

                if len(erp_window) < zscore_window // 2:
                    current_regime = current_regime or "neutral"
                else:
                    current_erp = erp_history[i]
                    if current_erp is None:
                        current_regime = current_regime or "neutral"
                    else:
                        erp_mean = sum(erp_window) / len(erp_window)
                        erp_std = (
                            sum((x - erp_mean) ** 2 for x in erp_window)
                            / len(erp_window)
                        ) ** 0.5
                        if erp_std > 0:
                            z = (current_erp - erp_mean) / erp_std
                        else:
                            z = 0.0

                        if z > cheap_thresh:
                            current_regime = "cheap"
                        elif z < expensive_thresh:
                            current_regime = "expensive"
                        else:
                            current_regime = "neutral"

        # Compute daily return based on current regime
        if current_regime == "crash":
            day_ret = daily_rets["SHY"][i]
        elif current_regime == "cheap":
            shy_w = 1.0 - target_weight
            day_ret = (
                daily_rets["UPRO"][i] * target_weight + daily_rets["SHY"][i] * shy_w
            )
        elif current_regime == "expensive":
            day_ret = (
                daily_rets["TLT"][i] * tlt_def_w
                + daily_rets["GLD"][i] * gld_def_w
                + daily_rets["SHY"][i] * shy_def_w
            )
        elif current_regime == "neutral":
            spy_w = neutral_spy_w
            shy_w = 1.0 - spy_w
            day_ret = daily_rets["SPY"][i] * spy_w + daily_rets["SHY"][i] * shy_w
        else:
            day_ret = daily_rets["SHY"][i]

        # Switching cost
        if prev_regime is not None and current_regime != prev_regime:
            day_ret -= cost_per_switch
        prev_regime = current_regime

        returns.append(day_ret)

    return _compute_metrics(returns)


def cpcv_sharpe(returns, n_groups=15, k=3, purge=5):
    """Combinatorial Purged Cross-Validation."""
    n_r = len(returns)
    if n_r < n_groups:
        return 0.0, 0.0, 0.0
    group_size = n_r // n_groups
    oos_sharpes = []
    for test_idx in combinations(range(n_groups), k):
        test_rets = []
        for g in test_idx:
            s, e = g * group_size + purge, (g + 1) * group_size - purge
            if s < e:
                test_rets.extend(returns[s:e])
        if len(test_rets) < 20:
            continue
        mean = sum(test_rets) / len(test_rets)
        std = (sum((r - mean) ** 2 for r in test_rets) / len(test_rets)) ** 0.5
        if std > 0:
            oos_sharpes.append(mean / std * math.sqrt(252))
    if not oos_sharpes:
        return 0.0, 0.0, 0.0
    m = sum(oos_sharpes) / len(oos_sharpes)
    s = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    pct_positive = sum(1 for x in oos_sharpes if x > 0) / len(oos_sharpes)
    return m, s, pct_positive


# ==============================================================================
# RUN BASE BACKTEST
# ==============================================================================
print("\n" + "-" * 70)
print("RUNNING BASE BACKTEST")
print("-" * 70)
print("Base parameters:")
for k, v in BASE_PARAMS.items():
    print(f"  {k} = {v}")

base = run_single(BASE_PARAMS)
print(f"\nBase Sharpe:       {base['sharpe']:.4f}")
print(f"Base MaxDD:        {base['max_dd']:.4f}")
print(f"Base Total Return: {base['total_return']:.4f}")
print(f"Base CAGR:         {base['cagr']:.4f}")
print(f"Trading days:      {len(base['daily_returns'])}")

passes_sharpe = base["sharpe"] >= 0.80
passes_dd = base["max_dd"] < DD_THRESHOLD
print(f"\nSharpe gate (>= 0.80): {'PASS' if passes_sharpe else 'FAIL'}")
print(f"MaxDD gate (< 40%):    {'PASS' if passes_dd else 'FAIL'}")

if base["sharpe"] < 0.70:
    print("\n*** SHARPE < 0.70 -- SKIPPING ROBUSTNESS (obvious failure) ***")
    # Save minimal results and exit
    output = {
        "strategy_slug": SLUG,
        "track": "D",
        "verdict": "FAIL - SHARPE TOO LOW FOR ROBUSTNESS",
        "base_sharpe": round(base["sharpe"], 4),
        "base_max_dd": round(base["max_dd"], 4),
        "base_cagr": round(base["cagr"], 4),
        "base_total_return": round(base["total_return"], 4),
        "trading_days": len(base["daily_returns"]),
    }
    out_path = Path(f"data/strategies/{SLUG}/robustness.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    print(f"Saved to {out_path}")
    sys.exit(0)

# ==============================================================================
# CPCV (Combinatorial Purged Cross-Validation)
# ==============================================================================
print("\n" + "-" * 70)
print("CPCV (15 groups, k=3, purge=5)")
print("-" * 70)
cpcv_mean, cpcv_std, cpcv_pct_pos = cpcv_sharpe(base["daily_returns"])
oos_is_ratio = cpcv_mean / base["sharpe"] if base["sharpe"] != 0 else 0.0
print(f"CPCV OOS Mean Sharpe:  {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS Ratio:     {oos_is_ratio:.4f}")
print(f"CPCV % Positive Folds: {cpcv_pct_pos:.1%}")

# ==============================================================================
# PERTURBATION TESTS
# ==============================================================================
perturbations = [
    # ERP lookback variations (±20-50%)
    ("erp_lookback=189", {**BASE_PARAMS, "erp_lookback": 189}),  # -25%
    ("erp_lookback=210", {**BASE_PARAMS, "erp_lookback": 210}),  # -17%
    ("erp_lookback=315", {**BASE_PARAMS, "erp_lookback": 315}),  # +25%
    ("erp_lookback=378", {**BASE_PARAMS, "erp_lookback": 378}),  # +50%
    # Z-score window variations
    ("zscore_window=189", {**BASE_PARAMS, "zscore_window": 189}),
    ("zscore_window=315", {**BASE_PARAMS, "zscore_window": 315}),
    # Cheap/expensive thresholds
    ("cheap_threshold=0.75", {**BASE_PARAMS, "cheap_threshold": 0.75}),
    ("cheap_threshold=1.25", {**BASE_PARAMS, "cheap_threshold": 1.25}),
    ("expensive_threshold=-0.75", {**BASE_PARAMS, "expensive_threshold": -0.75}),
    ("expensive_threshold=-1.25", {**BASE_PARAMS, "expensive_threshold": -1.25}),
    # target_weight variations
    ("target_weight=0.25", {**BASE_PARAMS, "target_weight": 0.25}),
    ("target_weight=0.30", {**BASE_PARAMS, "target_weight": 0.30}),
    ("target_weight=0.40", {**BASE_PARAMS, "target_weight": 0.40}),
    ("target_weight=0.45", {**BASE_PARAMS, "target_weight": 0.45}),
    # VIX threshold
    ("vix_crash_threshold=25", {**BASE_PARAMS, "vix_crash_threshold": 25}),
    ("vix_crash_threshold=35", {**BASE_PARAMS, "vix_crash_threshold": 35}),
    # Rebalance frequency
    ("rebalance_days=3", {**BASE_PARAMS, "rebalance_days": 3}),
    ("rebalance_days=10", {**BASE_PARAMS, "rebalance_days": 10}),
]

print("\n" + "-" * 70)
print("PERTURBATION TESTS")
print("-" * 70)

perturbation_results = []
stable_count = 0
for name, params in perturbations:
    print(f"  Running: {name}...", end=" ", flush=True)
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = "STABLE" if abs(pct) <= 25 else "UNSTABLE"
    if abs(pct) <= 25:
        stable_count += 1
    perturbation_results.append(
        {
            "variant": name,
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "cagr": round(r["cagr"], 4),
            "change_pct": round(pct, 1),
            "status": stable,
        }
    )
    print(
        f"Sharpe={r['sharpe']:.4f} MaxDD={r['max_dd']:.4f} "
        f"CAGR={r['cagr']:.4f} ({pct:+.1f}%) {stable}"
    )

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# ==============================================================================
# SHUFFLED SIGNAL TEST
# ==============================================================================
print("\n" + "-" * 70)
print("SHUFFLED SIGNAL TEST (1000 shuffles)")
print("-" * 70)

upro_daily_rets = daily_rets["UPRO"][WARMUP:]
strat_returns = base["daily_returns"]
n_min = min(len(strat_returns), len(upro_daily_rets))
aligned_strat = strat_returns[-n_min:]
aligned_upro = upro_daily_rets[-n_min:]

print(f"  Strategy returns: {len(aligned_strat)} days")
print(f"  UPRO returns:     {len(aligned_upro)} days")

shuffled_result = shuffled_signal_test(
    daily_returns=aligned_strat,
    asset_returns=aligned_upro,
    n_shuffles=1000,
    seed=42,
)
print(f"  Real Sharpe:     {shuffled_result.real_sharpe:.4f}")
print(f"  Shuffled Mean:   {shuffled_result.shuffled_mean:.4f}")
print(f"  Shuffled 95th:   {shuffled_result.shuffled_95th:.4f}")
print(f"  Shuffled 99th:   {shuffled_result.shuffled_99th:.4f}")
print(f"  p-value:         {shuffled_result.p_value:.4f}")
print(f"  PASSED:          {shuffled_result.passed}")

# ==============================================================================
# DSR (Deflated Sharpe Ratio)
# ==============================================================================
print("\n" + "-" * 70)
print("DSR (Deflated Sharpe Ratio)")
print("-" * 70)
sr = base["sharpe"]
T = len(base["daily_returns"])
T_years = T / 252
se_sr = math.sqrt((1 + sr**2 / 2) / T_years) if T_years > 0 else 1.0
dsr_value = float(stats.norm.cdf(sr / se_sr)) if se_sr > 0 else 0.0
print(f"  Sharpe Ratio:     {sr:.4f}")
print(f"  T (trading days): {T}")
print(f"  T (years):        {T_years:.2f}")
print(f"  SE(SR):           {se_sr:.4f}")
print(f"  DSR:              {dsr_value:.4f}")

# ==============================================================================
# GATE ASSESSMENT
# ==============================================================================
print(f"\n{'=' * 70}")
print("GATE ASSESSMENT (Track D -- Leveraged)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] >= 0.80
gate2 = base["max_dd"] < DD_THRESHOLD
gate3 = dsr_value >= 0.90
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 40
gate6 = shuffled_result.passed

gates = [
    ("Gate 1: Sharpe >= 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    ("Gate 5: Perturbation >= 40% stable", gate5, f"{pct_stable:.0f}%"),
    (
        "Gate 6: Shuffled Signal p < 0.05",
        gate6,
        f"p={shuffled_result.p_value:.4f}",
    ),
]

for name, passed, val in gates:
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} ({val})")

all_pass = all(g[1] for g in gates)
verdict = "PASS - ALL GATES CLEARED" if all_pass else "FAIL"
print(f"\n  VERDICT: {verdict}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output = {
    "strategy_slug": SLUG,
    "strategy_type": "erp_valuation_leveraged",
    "track": "D",
    "mechanism": "F30 ERP valuation regime (SPY 1yr return - TNX yield, z-scored) -> UPRO",
    "hypothesis": (
        "F30 ERP signal (SPY trailing return minus TNX yield, z-scored) "
        "re-expressed through UPRO. When equities are cheap (z>1), allocate "
        "to UPRO. When expensive (z<-1), go defensive (TLT/GLD/SHY)."
    ),
    "base_params": BASE_PARAMS,
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "base_cagr": round(base["cagr"], 4),
    "trading_days": len(base["daily_returns"]),
    "dsr": round(dsr_value, 4),
    "cpcv": {
        "oos_mean_sharpe": round(cpcv_mean, 4),
        "oos_std": round(cpcv_std, 4),
        "oos_is_ratio": round(oos_is_ratio, 4),
        "pct_positive_folds": round(cpcv_pct_pos, 4),
    },
    "perturbation": {
        "variants": perturbation_results,
        "pct_stable": round(pct_stable, 1),
        "stable_count": stable_count,
        "total_variants": len(perturbations),
    },
    "shuffled_signal": {
        "real_sharpe": round(shuffled_result.real_sharpe, 4),
        "shuffled_mean": round(shuffled_result.shuffled_mean, 4),
        "shuffled_95th": round(shuffled_result.shuffled_95th, 4),
        "shuffled_99th": round(shuffled_result.shuffled_99th, 4),
        "p_value": round(shuffled_result.p_value, 4),
        "n_shuffles": shuffled_result.n_shuffles,
        "passed": shuffled_result.passed,
    },
    "gates": {
        "sharpe_gte_0.80": gate1,
        "maxdd_lt_40pct": gate2,
        "dsr_gte_0.90": gate3,
        "cpcv_oos_positive": gate4,
        "perturbation_gte_40pct": gate5,
        "shuffled_signal_passed": gate6,
    },
    "verdict": "PASS" if all_pass else "FAIL",
}

out_path = Path(f"data/strategies/{SLUG}/robustness.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")

# Save experiment registry
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
entry = {
    "strategy_slug": SLUG,
    "track": "D",
    "sharpe": round(base["sharpe"], 4),
    "max_dd": round(base["max_dd"], 4),
    "cagr": round(base["cagr"], 4),
    "total_return": round(base["total_return"], 4),
    "dsr": round(dsr_value, 4),
    "verdict": verdict,
    "params": BASE_PARAMS,
}
with open(registry_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"Saved experiment registry to {registry_path}")

#!/usr/bin/env python3
"""Robustness analysis for tsmom-upro-trend-v1 (Track D - Sprint Alpha).

Skip-month TSMOM expressed through UPRO (3x S&P 500).

Track D gates (leveraged):
  - Gate 1: Sharpe > 0.80
  - Gate 2: MaxDD < 40%
  - Gate 3: DSR >= 0.90
  - Gate 4: CPCV OOS Sharpe > 0
  - Gate 5: Perturbation >= 40% stable (2/5 minimum)
  - Gate 6: Shuffled signal p < 0.05

Run: cd E:/llm-quant && PYTHONPATH=src python scripts/robustness_d13_tsmom_upro.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml
from scipy import stats

sys.path.insert(0, "src")

import polars as pl

from llm_quant.backtest.robustness import shuffled_signal_test
from llm_quant.data.fetcher import fetch_ohlcv

SLUG = "tsmom-upro-trend-v1"
SYMBOLS = ["SPY", "UPRO", "GLD", "TLT", "SHY"]
VIX_SYMBOL = "^VIX"
DD_THRESHOLD = 0.40  # Track D: 40% max drawdown
LOOKBACK_DAYS = 5 * 365
WARMUP = 300

# ---------------------------------------------------------------------------
# Load best params from backtest results, or use defaults
# ---------------------------------------------------------------------------
backtest_results_path = Path(f"data/strategies/{SLUG}/backtest_results.yaml")
if backtest_results_path.exists():
    with open(backtest_results_path) as f:
        bt_results = yaml.safe_load(f)
    BASE_PARAMS = bt_results.get("best_params", {})
    print(f"Loaded best params from backtest results: {bt_results.get('best_variant')}")
else:
    BASE_PARAMS = {
        "momentum_lookback": 252,
        "skip_period": 21,
        "base_upro_weight": 0.40,
        "max_upro_weight": 0.50,
        "shy_bullish_weight": 0.10,
        "gld_bearish_weight": 0.30,
        "tlt_bearish_weight": 0.30,
        "shy_bearish_weight": 0.40,
        "vol_target": 0.15,
        "vol_window": 20,
        "vix_crash_threshold": 30,
        "rebalance_frequency_days": 5,
        "cost_bps": 10,
    }
    print("Using default params (no backtest results found)")

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
print("\nFetching data...")
prices = fetch_ohlcv([*SYMBOLS, VIX_SYMBOL], lookback_days=LOOKBACK_DAYS)
print(
    f"Data: {len(prices)} rows, "
    f"date range: {prices['date'].min()} to {prices['date'].max()}"
)

# Build price lookup
sym_data: dict[str, dict] = {}
for sym in SYMBOLS:
    sdf = prices.filter(pl.col("symbol") == sym).sort("date")
    sym_data[sym] = dict(
        zip(sdf["date"].to_list(), sdf["close"].to_list(), strict=False)
    )

# VIX data
vix_key = "VIX"
vix_df = prices.filter(pl.col("symbol") == vix_key).sort("date")
if len(vix_df) == 0:
    vix_df = prices.filter(pl.col("symbol") == VIX_SYMBOL).sort("date")
vix_data: dict = dict(
    zip(vix_df["date"].to_list(), vix_df["close"].to_list(), strict=False)
)

# SPY as date backbone
spy_df = prices.filter(pl.col("symbol") == "SPY").sort("date")
dates = spy_df["date"].to_list()
spy_close = spy_df["close"].to_list()
n = len(dates)
print(f"Trading days: {n}")

# Precompute daily returns
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


def _vol_scale_weight(spy_daily_rets, i, base_w, max_w, vol_target, vol_window):
    """Compute vol-scaled UPRO weight using realized SPY vol."""
    start_idx = max(0, i - 1 - vol_window)
    end_idx = i - 1
    if end_idx <= start_idx or end_idx > len(spy_daily_rets):
        return base_w
    window_rets = spy_daily_rets[start_idx:end_idx]
    if len(window_rets) < 10:
        return base_w
    mean_wr = sum(window_rets) / len(window_rets)
    var_wr = sum((r - mean_wr) ** 2 for r in window_rets) / len(window_rets)
    realized_vol = math.sqrt(var_wr) * math.sqrt(252)
    if realized_vol <= 0:
        return base_w
    scaled = min(max_w, base_w * (vol_target / realized_vol))
    return max(0.10, scaled)


def _classify_regime(i, lookback, skip, vix_thresh):
    """Classify regime as crash/bullish/bearish. Returns (regime, valid)."""
    d = dates[i]
    vix_level = vix_data.get(d, 0.0)
    if vix_level > vix_thresh:
        return "crash", True
    idx_lb = i - lookback
    idx_skip = i - skip
    if idx_lb < 0 or idx_skip < 0:
        return "shy_only", False
    price_lb = spy_close[idx_lb]
    price_skip = spy_close[idx_skip]
    if price_lb <= 0:
        return "shy_only", False
    skip_mom = price_skip / price_lb - 1.0
    return ("bullish" if skip_mom > 0 else "bearish"), True


def run_single(params):
    """Run a single backtest with given parameters."""
    lookback = int(params.get("momentum_lookback", 252))
    skip = int(params.get("skip_period", 21))
    base_w = float(params.get("base_upro_weight", 0.40))
    max_w = float(params.get("max_upro_weight", 0.50))
    gld_bear_w = float(params.get("gld_bearish_weight", 0.30))
    tlt_bear_w = float(params.get("tlt_bearish_weight", 0.30))
    shy_bear_w = float(params.get("shy_bearish_weight", 0.40))
    vol_target = float(params.get("vol_target", 0.15))
    vol_window = int(params.get("vol_window", 20))
    vix_thresh = float(params.get("vix_crash_threshold", 30))
    cost_bps = float(params.get("cost_bps", 10))

    cost_per_switch = cost_bps / 10000.0
    returns = []
    prev_regime = None

    spy_daily_rets = [
        (spy_close[i] / spy_close[i - 1] - 1) if spy_close[i - 1] > 0 else 0.0
        for i in range(1, n)
    ]

    min_start = max(WARMUP, lookback + 1)

    for i in range(min_start, n):
        regime, valid = _classify_regime(i, lookback, skip, vix_thresh)
        if not valid:
            returns.append(daily_rets["SHY"][i])
            continue

        upro_w = base_w
        if regime == "bullish" and vol_window > 0:
            upro_w = _vol_scale_weight(
                spy_daily_rets, i, base_w, max_w, vol_target, vol_window
            )

        if regime == "crash":
            day_ret = daily_rets["SHY"][i]
        elif regime == "bullish":
            shy_w = 1.0 - upro_w
            day_ret = daily_rets["UPRO"][i] * upro_w + daily_rets["SHY"][i] * shy_w
        else:
            day_ret = (
                daily_rets["GLD"][i] * gld_bear_w
                + daily_rets["TLT"][i] * tlt_bear_w
                + daily_rets["SHY"][i] * shy_bear_w
            )

        if prev_regime is not None and regime != prev_regime:
            day_ret -= cost_per_switch

        prev_regime = regime
        returns.append(day_ret)

    return _compute_metrics(returns)


def cpcv_sharpe(returns, n_groups=6, k=2, purge=5):
    """Combinatorial Purged Cross-Validation (inline)."""
    from itertools import combinations

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
# RUN BASE
# ==============================================================================
print("=" * 70)
print(f"ROBUSTNESS ANALYSIS: {SLUG}")
print("=" * 70)
print("Base parameters:")
for k, v in BASE_PARAMS.items():
    print(f"  {k} = {v}")

print("\n--- RUNNING BASE BACKTEST ---")
base = run_single(BASE_PARAMS)
print(f"Base Sharpe:  {base['sharpe']:.4f}")
print(f"Base MaxDD:   {base['max_dd']:.4f}")
print(f"Base Return:  {base['total_return']:.4f}")
print(f"Base CAGR:    {base['cagr']:.4f}")

# ==============================================================================
# CPCV
# ==============================================================================
print("\n--- CPCV (Combinatorial Purged Cross-Validation) ---")
cpcv_mean, cpcv_std, cpcv_pct_pos = cpcv_sharpe(base["daily_returns"])
oos_is_ratio = cpcv_mean / base["sharpe"] if base["sharpe"] != 0 else 0.0
print(f"CPCV OOS Mean Sharpe:  {cpcv_mean:.4f} +/- {cpcv_std:.4f}")
print(f"CPCV OOS/IS Ratio:     {oos_is_ratio:.4f}")
print(f"CPCV % Positive Folds: {cpcv_pct_pos:.1%}")

# ==============================================================================
# PERTURBATION TESTS
# ==============================================================================
perturbations = [
    ("momentum_lookback=180", {**BASE_PARAMS, "momentum_lookback": 180}),
    ("momentum_lookback=360", {**BASE_PARAMS, "momentum_lookback": 360}),
    ("skip_period=10", {**BASE_PARAMS, "skip_period": 10}),
    ("skip_period=42", {**BASE_PARAMS, "skip_period": 42}),
    ("base_upro_weight=0.30", {**BASE_PARAMS, "base_upro_weight": 0.30}),
    ("base_upro_weight=0.50", {**BASE_PARAMS, "base_upro_weight": 0.50}),
    ("vol_target=0.12", {**BASE_PARAMS, "vol_target": 0.12}),
    ("vix_crash_threshold=25", {**BASE_PARAMS, "vix_crash_threshold": 25}),
]

print("\n--- PERTURBATION RESULTS ---")
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
        f"sharpe={r['sharpe']:.4f} max_dd={r['max_dd']:.4f} "
        f"cagr={r['cagr']:.4f} ({pct:+.1f}%) {stable}"
    )

pct_stable = stable_count / len(perturbations) * 100
print(f"\n  Stable: {stable_count}/{len(perturbations)} ({pct_stable:.0f}%)")

# ==============================================================================
# SHUFFLED SIGNAL TEST
# ==============================================================================
print("\n--- SHUFFLED SIGNAL TEST ---")
# Use UPRO returns as asset baseline (what you'd get with random timing)
upro_daily_rets = daily_rets["UPRO"][
    max(WARMUP, int(BASE_PARAMS.get("momentum_lookback", 252)) + 1) :
]
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
# DSR
# ==============================================================================
print("\n--- DSR ---")
sr = base["sharpe"]
T = len(base["daily_returns"])
T_years = T / 252
se_sr = math.sqrt((1 + sr**2 / 2) / T_years) if T_years > 0 else 1.0
dsr_value = float(stats.norm.cdf(sr / se_sr)) if se_sr > 0 else 0.0
print(f"  Sharpe Ratio:   {sr:.4f}")
print(f"  T (days):       {T}")
print(f"  SE(SR):         {se_sr:.4f}")
print(f"  DSR:            {dsr_value:.4f}")

# ==============================================================================
# GATE ASSESSMENT -- Track D gates
# ==============================================================================
print(f"\n{'=' * 70}")
print("GATE ASSESSMENT (Track D -- Leveraged)")
print(f"{'=' * 70}")

gate1 = base["sharpe"] > 0.80
gate2 = base["max_dd"] < DD_THRESHOLD  # 40% for Track D
gate3 = dsr_value >= 0.90  # Relaxed for Track D
gate4 = cpcv_mean > 0
gate5 = pct_stable >= 40  # 2/5 minimum for Track D (here 40% of 8 = 3.2, need 4)
gate6 = shuffled_result.passed

gates = [
    ("Gate 1: Sharpe > 0.80", gate1, f"{base['sharpe']:.4f}"),
    ("Gate 2: MaxDD < 40%", gate2, f"{base['max_dd']:.4f}"),
    ("Gate 3: DSR >= 0.90", gate3, f"{dsr_value:.4f}"),
    ("Gate 4: CPCV OOS Sharpe > 0", gate4, f"{cpcv_mean:.4f}"),
    (
        "Gate 5: Perturbation >= 40% stable",
        gate5,
        f"{pct_stable:.0f}% ({stable_count}/{len(perturbations)})",
    ),
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
    "strategy_type": "tsmom_leveraged",
    "track": "D",
    "mechanism": "Skip-month TSMOM (SPY t-252 to t-21) -> UPRO (3x S&P 500)",
    "base_params": dict(BASE_PARAMS),
    "base_sharpe": round(base["sharpe"], 4),
    "base_max_dd": round(base["max_dd"], 4),
    "base_total_return": round(base["total_return"], 4),
    "base_cagr": round(base["cagr"], 4),
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
        "sharpe_gt_0.80": gate1,
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

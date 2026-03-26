"""Robustness analysis for D7 eth-btc-ratio-mean-reversion."""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "eth-btc-ratio-mean-reversion"
STRATEGY = "pairs_ratio"
SYMBOLS = ["ETH-USD", "BTC-USD"]
BASE_PARAMS = {
    "symbol_a": "ETH-USD",
    "symbol_b": "BTC-USD",
    "bb_window": 20,
    "bb_std": 2.0,
    "target_weight": 0.90,
    "rebalance_frequency_days": 1,
}

# Higher crypto cost model
cost_model = CostModel(spread_bps=20.0, flat_slippage_bps=10.0)

print("Fetching data...")
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365 + 30)
print("Computing indicators...")
indicators_df = compute_indicators(prices_df)


def run_single(params: dict) -> dict:
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=params.get("rebalance_frequency_days", 1),
        max_positions=2,
        target_position_weight=params.get("target_weight", 0.90),
        stop_loss_pct=0.10,
        parameters=dict(params),
    )
    engine = BacktestEngine(
        create_strategy(STRATEGY, config),
        initial_capital=100000.0,
    )
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=cost_model,
        warmup_days=30,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "total_trades": m.total_trades if m else 0,
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(
    returns: list[float], n_groups: int = 6, k: int = 2, purge: int = 5
) -> tuple[float, float]:
    n = len(returns)
    if n < n_groups:
        return 0.0, 0.0
    group_size = n // n_groups
    oos: list[float] = []
    for test_idx in combinations(range(n_groups), k):
        tr: list[float] = []
        for i in test_idx:
            s, e = i * group_size + purge, (i + 1) * group_size - purge
            if s < e:
                tr.extend(returns[s:e])
        if len(tr) < 20:
            continue
        mu = sum(tr) / len(tr)
        std = (sum((r - mu) ** 2 for r in tr) / len(tr)) ** 0.5
        if std > 0:
            oos.append(mu / std * math.sqrt(252))
    if not oos:
        return 0.0, 0.0
    m = sum(oos) / len(oos)
    s = (sum((x - m) ** 2 for x in oos) / len(oos)) ** 0.5
    return m, s


# --- Base run ---
print("\n=== Base Configuration ===")
base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std = cpcv_sharpe(base["daily_returns"])
print(
    f"Base: Sharpe={base['sharpe']:.4f} MaxDD={base['max_dd']:.4f} Trades={base['total_trades']}"
)
print(f"CPCV: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")

# --- Perturbation stability ---
print("\n=== Perturbation Stability ===")
perturbations = [
    ("bb_window=15", {**BASE_PARAMS, "bb_window": 15}),
    ("bb_window=25", {**BASE_PARAMS, "bb_window": 25}),
    ("bb_std=1.5", {**BASE_PARAMS, "bb_std": 1.5}),
    ("bb_std=2.5", {**BASE_PARAMS, "bb_std": 2.5}),
    ("weight=0.70", {**BASE_PARAMS, "target_weight": 0.70}),
]
perturb_results = []
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = abs(pct) <= 30
    print(
        f"  {name}: Sharpe={r['sharpe']:.4f} ({pct:+.1f}%) {'STABLE' if stable else 'UNSTABLE'}"
    )
    perturb_results.append(
        {
            "name": name,
            "sharpe": r["sharpe"],
            "pct_change": round(pct, 1),
            "stable": stable,
        }
    )

# --- Read DSR from registry ---
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
dsr_value = exps[-1].get("dsr", 0.0)

n_stable = sum(1 for r in perturb_results if r["stable"])

print("\n=== Gate Summary ===")
print(
    f"DSR: {dsr_value:.4f} (threshold >= 0.95) -> {'PASS' if dsr_value >= 0.95 else 'FAIL'}"
)
print(
    f"CPCV OOS Sharpe: {cpcv_mean:.4f} (threshold > 0) -> {'PASS' if cpcv_mean > 0 else 'FAIL'}"
)
print(
    f"Perturbation: {n_stable}/{len(perturb_results)} stable (threshold >= 60%) -> {'PASS' if n_stable / len(perturb_results) >= 0.60 else 'FAIL'}"
)
print(
    f"Max DD: {base['max_dd'] * 100:.2f}% (threshold < 20%) -> {'PASS' if base['max_dd'] < 0.20 else 'FAIL'}"
)
print(f"Overall: {'PASS' if (dsr_value >= 0.95 and cpcv_mean > 0) else 'FAIL'}")

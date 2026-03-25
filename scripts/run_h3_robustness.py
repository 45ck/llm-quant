#!/usr/bin/env python3
"""Robustness analysis for H3: tlt-spy-inverse-lead-lag."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "src")

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "tlt-spy-inverse-lead-lag"
STRATEGY = "lead_lag"
SYMBOLS = ["TLT", "SPY"]
BASE_PARAMS = {
    "leader_symbol": "TLT",
    "follower_symbol": "SPY",
    "lag_days": 3,
    "signal_window": 3,
    "entry_threshold": 0.008,
    "exit_threshold": -0.003,
    "target_weight": 0.90,
    "inverse": True,
    "rebalance_frequency_days": 1,
}
DD_THRESHOLD = 0.15

prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365 + 30)
indicators_df = compute_indicators(prices_df)


def run_single(params):
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=1,
        max_positions=2,
        target_position_weight=0.90,
        stop_loss_pct=0.05,
        parameters=dict(params),
    )
    strategy = create_strategy(STRATEGY, config)
    engine = BacktestEngine(strategy, initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=CostModel(),
        warmup_days=30,
        cost_multiplier=1.0,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "daily_returns": result.daily_returns or [],
    }


def cpcv_sharpe(returns, n_groups=6, k=2, purge=5):
    from itertools import combinations

    n = len(returns)
    if n < n_groups:
        return 0.0, 0.0
    group_size = n // n_groups
    oos_sharpes = []
    for test_idx in combinations(range(n_groups), k):
        test_rets = []
        for i in test_idx:
            s, e = i * group_size + purge, (i + 1) * group_size - purge
            if s < e:
                test_rets.extend(returns[s:e])
        if len(test_rets) < 20:
            continue
        mean = sum(test_rets) / len(test_rets)
        std = (sum((r - mean) ** 2 for r in test_rets) / len(test_rets)) ** 0.5
        if std > 0:
            oos_sharpes.append(mean / std * math.sqrt(252))
    if not oos_sharpes:
        return 0.0, 0.0
    m = sum(oos_sharpes) / len(oos_sharpes)
    s = (sum((x - m) ** 2 for x in oos_sharpes) / len(oos_sharpes)) ** 0.5
    return m, s


base = run_single(BASE_PARAMS)
cpcv_mean, cpcv_std = cpcv_sharpe(base["daily_returns"])
print(f"Base Sharpe: {base['sharpe']:.4f}, Max DD: {base['max_dd']:.4f}")
print(f"CPCV: {cpcv_mean:.4f} +/- {cpcv_std:.4f}")

perturbations = [
    ("lag_days=2", {**BASE_PARAMS, "lag_days": 2}),
    ("lag_days=4", {**BASE_PARAMS, "lag_days": 4}),
    ("entry_threshold=0.004", {**BASE_PARAMS, "entry_threshold": 0.004}),
    ("entry_threshold=0.016", {**BASE_PARAMS, "entry_threshold": 0.016}),
    ("target_weight=0.70", {**BASE_PARAMS, "target_weight": 0.70}),
]
print("\nPerturbation results:")
for name, params in perturbations:
    r = run_single(params)
    pct = (r["sharpe"] - base["sharpe"]) / (abs(base["sharpe"]) + 1e-8) * 100
    stable = "STABLE" if abs(pct) <= 30 else "UNSTABLE"
    print(
        f"  {name}: sharpe={r['sharpe']:.4f} "
        f"max_dd={r['max_dd']:.4f} ({pct:+.1f}%) {stable}"
    )

# DSR from registry
registry_path = Path(f"data/strategies/{SLUG}/experiment-registry.jsonl")
with registry_path.open() as f:
    exps = [json.loads(line) for line in f if line.strip()]
dsr_value = exps[-1].get("dsr", 0.0)
print(f"\nDSR: {dsr_value:.4f}")
print(f"DD_THRESHOLD: {DD_THRESHOLD:.2f}")
dd_result = "PASS" if base["max_dd"] < DD_THRESHOLD else "FAIL"
print(
    f"Max DD check: {base['max_dd']:.4f} vs threshold {DD_THRESHOLD:.2f} -> {dd_result}"
)

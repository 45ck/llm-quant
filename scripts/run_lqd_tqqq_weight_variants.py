#!/usr/bin/env python3
"""Weight variant backtest for lqd-tqqq-sprint (Track D — D6).

Tests 3 weight levels: 50%, 70%, 90% target_weight.
Track D gates: Sharpe >= 0.80, MaxDD < 40%, DSR >= 0.90
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

SLUG = "lqd-tqqq-sprint"
STRATEGY = "lead_lag"
SYMBOLS = ["LQD", "TQQQ"]
DD_THRESHOLD = 0.40  # Track D: relaxed to 40%
SHARPE_THRESHOLD = 0.80
DSR_THRESHOLD = 0.90

BASE_PARAMS = {
    "leader_symbol": "LQD",
    "follower_symbol": "TQQQ",
    "lag_days": 3,
    "signal_window": 10,
    "entry_threshold": 0.01,
    "exit_threshold": -0.005,
    "target_weight": 0.30,
    "inverse": False,
    "rebalance_frequency_days": 1,
}

cost_model = CostModel(
    spread_bps=10.0, flat_slippage_bps=5.0, slippage_volatility_factor=0.2
)

print("Fetching data (LQD + TQQQ, 5 years)...")
prices_df = fetch_ohlcv(SYMBOLS, lookback_days=5 * 365 + 30)
print("Computing indicators...")
indicators_df = compute_indicators(prices_df)


def run_single(params: dict) -> dict:
    weight = params.get("target_weight", 0.30)
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=params.get("rebalance_frequency_days", 1),
        max_positions=2,
        target_position_weight=weight,
        stop_loss_pct=0.10,
        parameters=dict(params),
    )
    strategy = create_strategy(STRATEGY, config)
    engine = BacktestEngine(strategy, initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=SLUG,
        cost_model=cost_model,
        warmup_days=30,
        cost_multiplier=1.0,
    )
    m = result.metrics.get("1.0x")
    return {
        "sharpe": m.sharpe_ratio if m else 0.0,
        "max_dd": m.max_drawdown if m else 0.0,
        "total_return": m.total_return if m else 0.0,
        "annualized_return": m.annualized_return if m else 0.0,
        "sortino": m.sortino_ratio if m else 0.0,
        "calmar": m.calmar_ratio if m else 0.0,
        "total_trades": m.total_trades if m else 0,
        "win_rate": m.win_rate if m else 0.0,
        "dsr": m.dsr if m else 0.0,
        "daily_returns": result.daily_returns or [],
    }


def compute_dsr(returns: list[float], sharpe: float) -> float:
    """Approximate DSR from daily returns."""
    if not returns or len(returns) < 20:
        return 0.0

    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    std = variance**0.5
    if std == 0:
        return 0.0
    # Skew and kurtosis
    skew = sum((r - mean) ** 3 for r in returns) / (n * std**3)
    kurt = sum((r - mean) ** 4 for r in returns) / (n * std**4) - 3
    # DSR formula (De Prado)
    sr_hat = sharpe
    z = (
        sr_hat
        * math.sqrt(n - 1)
        / math.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
    )
    # Normal CDF approximation
    t = z / (1 + abs(z) * (0.2316419))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    cdf = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z**2) * poly
    return max(0.0, min(1.0, cdf))


print("\n" + "=" * 60)
print("LQD->TQQQ WEIGHT VARIANT ANALYSIS (Track D - D6)")
print("=" * 60)

weight_configs = [
    ("base (30%)", {**BASE_PARAMS, "target_weight": 0.30}),
    ("50%", {**BASE_PARAMS, "target_weight": 0.50}),
    ("70%", {**BASE_PARAMS, "target_weight": 0.70}),
    ("90%", {**BASE_PARAMS, "target_weight": 0.90}),
]

results = []
for label, params in weight_configs:
    print(f"\nRunning weight={label}...")
    r = run_single(params)
    passes = (
        r["sharpe"] >= SHARPE_THRESHOLD
        and r["max_dd"] < DD_THRESHOLD
        and r["dsr"] >= DSR_THRESHOLD
    )
    results.append(
        {
            "weight": label,
            "target_weight": params["target_weight"],
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "annualized_return": round(r["annualized_return"], 4),
            "total_return": round(r["total_return"], 4),
            "dsr": round(r["dsr"], 4),
            "sortino": round(r["sortino"], 4),
            "calmar": round(r["calmar"], 4),
            "total_trades": r["total_trades"],
            "win_rate": round(r["win_rate"], 4),
            "passes_track_d": passes,
        }
    )
    print(
        f"  Sharpe={r['sharpe']:.4f}  MaxDD={r['max_dd'] * 100:.1f}%  CAGR={r['annualized_return'] * 100:.1f}%  DSR={r['dsr']:.4f}  {'PASS' if passes else 'FAIL'}"
    )

print("\n\n=== SUMMARY TABLE ===")
print(f"{'Weight':<12} {'Sharpe':<8} {'MaxDD':<10} {'CAGR':<10} {'DSR':<8} {'Pass'}")
print("-" * 60)
for r in results:
    print(
        f"{r['weight']:<12} {r['sharpe']:<8.4f} {r['max_dd'] * 100:<10.1f} {r['annualized_return'] * 100:<10.1f} {r['dsr']:<8.4f} {'YES' if r['passes_track_d'] else 'NO'}"
    )

# Save to YAML
out_path = Path(f"data/strategies/{SLUG}/weight_variants.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
output = {
    "strategy_slug": SLUG,
    "run_date": "2026-03-30",
    "track": "D",
    "gates": {
        "sharpe_min": SHARPE_THRESHOLD,
        "maxdd_max": DD_THRESHOLD,
        "dsr_min": DSR_THRESHOLD,
    },
    "weight_variants": results,
}
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)

print(f"\nSaved to {out_path}")

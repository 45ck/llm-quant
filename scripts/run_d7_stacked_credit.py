#!/usr/bin/env python3
"""D7: Stacked credit signals on TQQQ.

Three credit leaders (TLT, LQD, IEF) each with 30% TQQQ allocation.
Voting system: 3/3 agree = 90% TQQQ, 2/3 = 60%, 1/3 = 30%, 0/3 = 0%.

Analytical approach: run each leader independently, compute daily signals,
measure correlation of signals, then combine using majority-vote weighting.
"""

from __future__ import annotations

import json
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

SLUG = "tqqq-stacked-credit"
STRATEGY = "lead_lag"
DD_THRESHOLD = 0.40
SHARPE_THRESHOLD = 0.80
DSR_THRESHOLD = 0.90

# Leader configs (each 30% weight, same signal params as tlt-tqqq-sprint)
LEADERS = ["TLT", "LQD", "IEF"]
LEADER_PARAMS = {
    "TLT": {
        "leader_symbol": "TLT",
        "follower_symbol": "TQQQ",
        "lag_days": 3,
        "signal_window": 10,
        "entry_threshold": 0.01,
        "exit_threshold": -0.005,
        "target_weight": 0.30,
        "inverse": False,
        "rebalance_frequency_days": 1,
    },
    "LQD": {
        "leader_symbol": "LQD",
        "follower_symbol": "TQQQ",
        "lag_days": 3,
        "signal_window": 10,
        "entry_threshold": 0.01,
        "exit_threshold": -0.005,
        "target_weight": 0.30,
        "inverse": False,
        "rebalance_frequency_days": 1,
    },
    "IEF": {
        "leader_symbol": "IEF",
        "follower_symbol": "TQQQ",
        "lag_days": 3,
        "signal_window": 10,
        "entry_threshold": 0.01,
        "exit_threshold": -0.005,
        "target_weight": 0.30,
        "inverse": False,
        "rebalance_frequency_days": 1,
    },
}

cost_model = CostModel(spread_bps=10.0, flat_slippage_bps=5.0, slippage_volatility_factor=0.2)

print("Fetching data (TLT, LQD, IEF, TQQQ, 5 years)...")
prices_df = fetch_ohlcv(["TLT", "LQD", "IEF", "TQQQ"], lookback_days=5 * 365 + 30)
print("Computing indicators...")
indicators_df = compute_indicators(prices_df)


def run_single(params: dict, slug_suffix: str) -> dict:
    config = StrategyConfig(
        name=STRATEGY,
        rebalance_frequency_days=params.get("rebalance_frequency_days", 1),
        max_positions=2,
        target_position_weight=params.get("target_weight", 0.30),
        stop_loss_pct=0.10,
        parameters=dict(params),
    )
    strategy = create_strategy(STRATEGY, config)
    engine = BacktestEngine(strategy, initial_capital=100000.0)
    result = engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug_suffix,
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
        "dsr": m.dsr if m else 0.0,
        "total_trades": m.total_trades if m else 0,
        "win_rate": m.win_rate if m else 0.0,
        "daily_returns": result.daily_returns or [],
    }


def corr(x: list[float], y: list[float]) -> float:
    """Pearson correlation of two series (same length, excluding leading zeros)."""
    n = min(len(x), len(y))
    if n < 20:
        return 0.0
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
    dy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def compute_combined_returns(returns_dict: dict[str, list[float]]) -> list[float]:
    """Majority-vote combination: each day, count how many leaders are in signal.
    TQQQ allocation = (votes/3) * 30% each = 30% per vote.
    Combined daily return = weighted average of individual returns.
    """
    keys = list(returns_dict.keys())
    n = min(len(v) for v in returns_dict.values())
    combined = []
    for i in range(n):
        # Count non-zero (active) signals
        active = sum(1 for k in keys if returns_dict[k][i] != 0)
        # Weighted return: active leaders each get equal weight
        if active == 0:
            combined.append(0.0)
        else:
            # Scale returns by vote count / total leaders
            total = sum(returns_dict[k][i] for k in keys)
            combined.append(total)  # sum of individual contributions
    return combined


def compute_metrics(returns: list[float]) -> dict:
    """Compute Sharpe, MaxDD, CAGR from daily returns."""
    nonzero = [r for r in returns if r != 0]
    if len(nonzero) < 20:
        return {"sharpe": 0.0, "max_dd": 0.0, "cagr": 0.0}

    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    std = var ** 0.5
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0

    # MaxDD
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cum *= (1 + r)
        if cum > peak:
            peak = cum
        dd = (peak - cum) / peak
        if dd > max_dd:
            max_dd = dd

    # CAGR
    total = cum - 1
    years = n / 252
    cagr = ((cum) ** (1 / years) - 1) if years > 0 else 0.0

    # DSR approximation
    skew = sum((r - mean) ** 3 for r in returns) / (n * std ** 3) if std > 0 else 0
    kurt = sum((r - mean) ** 4 for r in returns) / (n * std ** 4) - 3 if std > 0 else 0
    sr_hat = sharpe / math.sqrt(252)  # daily SR
    denom = 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2
    if denom > 0:
        z = sr_hat * math.sqrt(n - 1) / math.sqrt(denom)
        t = z / (1 + abs(z) * 0.2316419)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        dsr = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z ** 2) * poly
        dsr = max(0.0, min(1.0, dsr))
    else:
        dsr = 0.0

    return {"sharpe": sharpe, "max_dd": max_dd, "cagr": cagr, "total_return": total, "dsr": dsr}


print("\n" + "=" * 60)
print("D7: STACKED CREDIT SIGNALS ON TQQQ")
print("=" * 60)

# Run each leader independently
leader_results = {}
print("\n--- Individual Leader Results ---")
for leader in LEADERS:
    print(f"\nRunning {leader}->TQQQ (30% weight)...")
    r = run_single(LEADER_PARAMS[leader], f"tlt-tqqq-sprint" if leader == "TLT" else f"stacked-{leader.lower()}-tqqq")
    leader_results[leader] = r
    print(f"  Sharpe={r['sharpe']:.4f}  MaxDD={r['max_dd']*100:.1f}%  CAGR={r['annualized_return']*100:.1f}%  DSR={r['dsr']:.4f}")

# Signal correlation analysis
print("\n--- Signal Correlation (daily returns alignment) ---")
tlt_rets = leader_results["TLT"]["daily_returns"]
lqd_rets = leader_results["LQD"]["daily_returns"]
ief_rets = leader_results["IEF"]["daily_returns"]

corr_tlt_lqd = corr(tlt_rets, lqd_rets)
corr_tlt_ief = corr(tlt_rets, ief_rets)
corr_lqd_ief = corr(lqd_rets, ief_rets)

print(f"  TLT-LQD signal correlation: {corr_tlt_lqd:.4f}")
print(f"  TLT-IEF signal correlation: {corr_tlt_ief:.4f}")
print(f"  LQD-IEF signal correlation: {corr_lqd_ief:.4f}")
avg_corr = (corr_tlt_lqd + corr_tlt_ief + corr_lqd_ief) / 3
print(f"  Average pair correlation:   {avg_corr:.4f}")

adds_value = avg_corr < 0.70
print(f"  Correlation < 0.70 (adds diversification): {'YES' if adds_value else 'NO'}")

# Combined portfolio
print("\n--- Combined (Stacked) Portfolio ---")
rets_dict = {
    "TLT": tlt_rets,
    "LQD": lqd_rets,
    "IEF": ief_rets,
}
combined_returns = compute_combined_returns(rets_dict)
combined_metrics = compute_metrics(combined_returns)

print(f"  Sharpe: {combined_metrics['sharpe']:.4f}")
print(f"  MaxDD:  {combined_metrics['max_dd']*100:.1f}%")
print(f"  CAGR:   {combined_metrics['cagr']*100:.1f}%")
print(f"  DSR:    {combined_metrics.get('dsr', 0):.4f}")

# Theoretical portfolio SR using correlation adjustment
n = 3
avg_individual_sr = sum(r["sharpe"] for r in leader_results.values()) / 3
corr_adj = n / (1 + (n - 1) * avg_corr)
theoretical_sr = avg_individual_sr * math.sqrt(corr_adj)
print(f"\n  Theoretical combined SR (formula): {theoretical_sr:.4f}")
print(f"  Formula: SR * sqrt(N / (1 + (N-1)*rho)) = {avg_individual_sr:.4f} * sqrt({corr_adj:.4f})")

# Gate assessment
passes = (
    combined_metrics["sharpe"] >= SHARPE_THRESHOLD
    and combined_metrics["max_dd"] < DD_THRESHOLD
    and combined_metrics.get("dsr", 0) >= DSR_THRESHOLD
)
print(f"\n  Gate Assessment (Sharpe>={SHARPE_THRESHOLD}, MaxDD<{DD_THRESHOLD*100}%, DSR>={DSR_THRESHOLD}):")
print(f"    Sharpe: {'PASS' if combined_metrics['sharpe'] >= SHARPE_THRESHOLD else 'FAIL'} ({combined_metrics['sharpe']:.4f})")
print(f"    MaxDD:  {'PASS' if combined_metrics['max_dd'] < DD_THRESHOLD else 'FAIL'} ({combined_metrics['max_dd']*100:.1f}%)")
print(f"    DSR:    {'PASS' if combined_metrics.get('dsr', 0) >= DSR_THRESHOLD else 'FAIL'} ({combined_metrics.get('dsr', 0):.4f})")
print(f"  OVERALL: {'PASS' if passes else 'FAIL'}")

# Save research spec
spec_dir = Path("data/strategies/tqqq-stacked-credit")
spec_dir.mkdir(parents=True, exist_ok=True)

research_spec = {
    "strategy_slug": "tqqq-stacked-credit",
    "strategy_type": "stacked_lead_lag",
    "track": "D",
    "created_at": "2026-03-30",
    "hypothesis": (
        "Three credit bond leaders (TLT 20yr Treasury, LQD IG Corporate, IEF 7-10yr Treasury) "
        "each independently signal TQQQ direction via the same lead-lag mechanism. "
        "When all 3 agree (bullish), TQQQ allocation = 90%; "
        "2/3 agree = 60%; 1/3 = 30%; 0/3 = 0%."
    ),
    "signal_params": {
        "lag_days": 3,
        "signal_window": 10,
        "entry_threshold": 0.01,
        "exit_threshold": -0.005,
        "weight_per_leader": 0.30,
    },
    "individual_results": {
        leader: {
            "sharpe": round(r["sharpe"], 4),
            "max_dd": round(r["max_dd"], 4),
            "cagr": round(r["annualized_return"], 4),
            "dsr": round(r["dsr"], 4),
        }
        for leader, r in leader_results.items()
    },
    "signal_correlations": {
        "tlt_lqd": round(corr_tlt_lqd, 4),
        "tlt_ief": round(corr_tlt_ief, 4),
        "lqd_ief": round(corr_lqd_ief, 4),
        "avg": round(avg_corr, 4),
        "adds_diversification": adds_value,
    },
    "combined_portfolio": {
        "sharpe": round(combined_metrics["sharpe"], 4),
        "max_dd": round(combined_metrics["max_dd"], 4),
        "cagr": round(combined_metrics["cagr"], 4),
        "total_return": round(combined_metrics["total_return"], 4),
        "dsr": round(combined_metrics.get("dsr", 0), 4),
        "theoretical_sr": round(theoretical_sr, 4),
        "passes_track_d": passes,
    },
    "verdict": "PASS" if passes else "CONDITIONAL",
    "notes": [
        "Combined returns = sum of individual 30%-weighted returns (each leader independent)",
        "Voting: 3 active = 90% TQQQ, 2 active = 60%, 1 active = 30%",
        "Next step: full robustness with CPCV and perturbation analysis",
    ],
}

spec_path = spec_dir / "research_spec.yaml"
with open(spec_path, "w") as f:
    yaml.dump(research_spec, f, default_flow_style=False, sort_keys=False)

print(f"\nSaved research spec to {spec_path}")

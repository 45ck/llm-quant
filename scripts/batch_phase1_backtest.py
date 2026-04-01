#!/usr/bin/env python3
"""Phase 1 batch backtest: Run 6 untested strategy classes.

Uses run_backtest.py CLI for each strategy with --no-spec-check.
Creates research specs in proper format first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

DATA_DIR = Path("data")
YEARS = 5

STRATEGIES = [
    {
        "slug": "vrp-timing-spy",
        "strategy": "vrp_timing",
        "symbols": "VIX,SPY",
        "family": "F4",
        "params": {
            "vix_symbol": "VIX",
            "equity_symbol": "SPY",
            "rv_window": 20,
            "vrp_smoothing": 5,
            "vrp_entry_threshold": 3.0,
            "vrp_exit_threshold": 0.0,
            "vix_min": 16,
            "vix_max": 35,
            "target_weight": 0.90,
            "reduced_weight": 0.45,
            "time_stop_days": 20,
        },
        "rebalance": 1,
    },
    {
        "slug": "multi-asset-tsmom-v1",
        "strategy": "vol_scaled_tsmom",
        "symbols": "SPY,TLT,GLD,EFA",
        "family": "F3",
        "params": {
            "lookbacks": [21, 63, 252],
            "blend_weights": [0.33, 0.33, 0.34],
            "vol_target": 0.10,
            "vol_window": 126,
            "max_vol_scalar": 2.0,
            "flat_threshold": 0.2,
            "allow_short": False,
        },
        "rebalance": 5,
    },
    {
        "slug": "skip-month-tsmom-v1",
        "strategy": "skip_month_tsmom",
        "symbols": "SPY,TLT,GLD,EFA",
        "family": "F3",
        "params": {
            "momentum_lookback": 252,
            "skip_period": 21,
            "vol_target": 0.10,
            "vol_window": 63,
            "max_vol_scalar": 2.0,
            "allow_short": False,
        },
        "rebalance": 21,
    },
    {
        "slug": "opex-week-spy",
        "strategy": "opex_week",
        "symbols": "SPY,VIX",
        "family": "F5",
        "params": {
            "target_symbol": "SPY",
            "vix_symbol": "VIX",
            "vix_threshold": 25.0,
            "target_weight": 0.95,
            "exclude_quarterly": True,
        },
        "rebalance": 1,
    },
    {
        "slug": "rsi2-contrarian-spy",
        "strategy": "rsi2_contrarian",
        "symbols": "SPY,QQQ",
        "family": "F7",
        "params": {
            "rsi_period": 2,
            "oversold_threshold": 10,
            "overbought_threshold": 70,
            "ema_short": 50,
            "ema_long": 200,
            "time_stop_days": 10,
            "consecutive_up_exit": 2,
            "sma_exit_period": 5,
        },
        "rebalance": 1,
    },
    {
        "slug": "macro-barometer-v1",
        "strategy": "macro_barometer",
        "symbols": "SPY,TLT,DBC,EFA,GLD,HYG,IEF,AGG",
        "family": "F6",
        "params": {
            "barometer_symbols": ["SPY", "TLT", "DBC", "EFA"],
            "z_lookback_short": 21,
            "z_lookback_long": 63,
            "z_window": 63,
            "risk_on_threshold": 3,
            "risk_off_threshold": 1,
        },
        "rebalance": 21,
    },
]


def create_spec(strat: dict) -> None:
    """Create a frozen research spec YAML."""
    sdir = DATA_DIR / "strategies" / strat["slug"]
    sdir.mkdir(parents=True, exist_ok=True)
    spec_path = sdir / "research-spec.yaml"
    if spec_path.exists():
        print(f"  Spec exists: {strat['slug']}")
        return

    today = date.today().isoformat()
    params_yaml = "\n".join(
        f"  {k}: {json.dumps(v)}" for k, v in strat["params"].items()
    )

    content = f"""# Research Spec: {strat["slug"]} (Family {strat["family"]})
# FROZEN — do not modify after backtest is run

strategy_slug: "{strat["slug"]}"
strategy_name: "{strat["strategy"]}"
strategy_type: "{strat["strategy"]}"
created_at: "{today}"
frozen: true
frozen_at: "{today}"
frozen_hash: "placeholder-computed-on-freeze"

strategy_class: "{strat["strategy"]}"
parameters:
{params_yaml}
  rebalance_frequency_days: {strat["rebalance"]}

rebalance_frequency_days: {strat["rebalance"]}
warmup_days: 60
fill_delay: 1
initial_capital: 100000.0

cost_model:
  spread_bps: 3.0
  flat_slippage_bps: 1.0
  slippage_volatility_factor: 0.05

acceptance_criteria:
  - "Sharpe > 0.8"
  - "MaxDD < 0.15"
  - "DSR >= 0.95"

spec_version: "1.0"
"""
    spec_path.write_text(content)
    print(f"  Created spec: {strat['slug']}")


def run_backtest(strat: dict) -> str:
    """Run backtest via run_backtest.py CLI."""
    cmd = [
        sys.executable,
        "scripts/run_backtest.py",
        "--slug",
        strat["slug"],
        "--strategy",
        strat["strategy"],
        "--symbols",
        strat["symbols"],
        "--years",
        str(YEARS),
        "--no-spec-check",
    ]
    print(f"\n{'=' * 70}")
    print(f"RUNNING: {strat['slug']} ({strat['family']})")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        timeout=300,
    )

    output = result.stdout + "\n" + result.stderr
    print(output[-3000:])  # Last 3000 chars
    return output


def main() -> None:
    print("=" * 70)
    print("SR 2.0+ RESEARCH SPRINT — PHASE 1 BATCH BACKTEST")
    print("=" * 70)

    # Create all specs first
    print("\nCreating research specs...")
    for s in STRATEGIES:
        create_spec(s)

    # Run backtests sequentially
    results = {}
    for s in STRATEGIES:
        try:
            output = run_backtest(s)
            results[s["slug"]] = output
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {s['slug']}")
            results[s["slug"]] = "TIMEOUT"
        except Exception as e:
            print(f"ERROR: {s['slug']}: {e}")
            results[s["slug"]] = f"ERROR: {e}"

    print("\n" + "=" * 70)
    print("ALL BACKTESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

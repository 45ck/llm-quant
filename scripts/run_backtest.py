#!/usr/bin/env python3
"""Run a backtest against a frozen research spec.

Usage:
    python scripts/run_backtest.py --slug test_sma --strategy sma_crossover \
        --symbols SPY,QQQ,TLT --years 3

The script:
1. Validates the frozen research spec exists
2. Fetches historical data
3. Computes indicators
4. Runs the backtest at 1x, 1.5x, 2x, 3x cost multipliers
5. Appends to experiment-registry.jsonl
6. Persists the experiment artifact
7. Outputs a markdown report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from llm_quant.backtest.artifacts import (
    ExperimentRegistry,
    ensure_frozen_spec,
    save_artifact,
    strategy_dir,
)
from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.report import generate_backtest_report
from llm_quant.backtest.strategies import STRATEGY_REGISTRY, create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_strategy_config(strategy_name: str, spec: dict) -> StrategyConfig:
    """Build a StrategyConfig from frozen spec, mapping parameter names."""
    params = spec.get("parameters", {})

    # Map frozen-spec parameter names to what strategies expect
    mapped_params = dict(params)
    if "top_n_momentum" in params and "top_n" not in params:
        mapped_params["top_n"] = params["top_n_momentum"]
    if "momentum_lookback" in params and "lookback_days" not in params:
        mapped_params["lookback_days"] = params["momentum_lookback"]
    if "rebalance_frequency_days" in params and "rebalance_frequency" not in params:
        mapped_params["rebalance_frequency"] = params["rebalance_frequency_days"]

    # Multi-timeframe momentum parameters (TrendFollowingStrategy v3)
    spec_params = spec.get("parameters", {})
    if "lookback_short" in spec_params:
        mapped_params["lookback_short"] = spec_params["lookback_short"]
    if "lookback_long" in spec_params:
        mapped_params["lookback_long"] = spec_params["lookback_long"]
    if "lookback_medium" in spec_params:
        mapped_params["lookback_medium"] = spec_params["lookback_medium"]
    if "min_timeframes_positive" in spec_params:
        mapped_params["min_timeframes_positive"] = spec_params[
            "min_timeframes_positive"
        ]

    return StrategyConfig(
        name=strategy_name,
        rebalance_frequency_days=params.get(
            "rebalance_frequency_days",
            spec.get("rebalance_frequency_days", 5),
        ),
        max_positions=params.get(
            "top_n_momentum",
            spec.get("max_positions", 10),
        ),
        target_position_weight=params.get(
            "target_position_weight",
            spec.get("target_position_weight", 0.05),
        ),
        stop_loss_pct=params.get(
            "stop_loss_pct",
            spec.get("stop_loss_pct", 0.05),
        ),
        parameters=mapped_params,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--slug", required=True, help="Strategy slug")
    parser.add_argument(
        "--strategy",
        default=None,
        help=f"Strategy name. Available: {list(STRATEGY_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,TLT,GLD,IEF",
        help="Comma-separated symbols",
    )
    parser.add_argument("--years", type=int, default=3, help="Years of history")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--no-spec-check",
        action="store_true",
        help="Skip frozen spec check (for quick testing only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    strat_dir = strategy_dir(data_dir, args.slug)
    symbols = [s.strip() for s in args.symbols.split(",")]
    lookback_days = args.years * 365

    # Load or create research spec
    spec: dict = {}
    strategy_name = args.strategy or "sma_crossover"

    if not args.no_spec_check:
        try:
            spec = ensure_frozen_spec(strat_dir)
            # Only use spec strategy_type if --strategy was not explicitly provided
            if args.strategy is None:
                strategy_name = spec.get("strategy_type", strategy_name)
            logger.info("Loaded frozen research spec for %s", args.slug)
        except (FileNotFoundError, ValueError):
            logger.exception("Spec check failed")
            sys.exit(1)
    else:
        logger.warning("Skipping spec check — results are exploratory only")

    config = _build_strategy_config(strategy_name, spec)

    strategy = create_strategy(strategy_name, config)
    cost_model = CostModel.from_spec(spec)
    fill_delay = spec.get("fill_delay", 1)
    warmup_days = spec.get("warmup_days", 200)

    # Benchmark from spec or default
    benchmark_weights = {"SPY": 0.60, "TLT": 0.40}
    benchmark = spec.get("benchmark", {})
    if benchmark:
        benchmark_weights = benchmark.get("symbols", benchmark_weights)

    # Fetch data
    logger.info("Fetching %d symbols (%d days)...", len(symbols), lookback_days)
    prices_df = fetch_ohlcv(symbols, lookback_days=lookback_days)
    if len(prices_df) == 0:
        logger.error("No data fetched — aborting")
        sys.exit(1)

    # Compute indicators
    logger.info("Computing indicators...")
    indicators_df = compute_indicators(prices_df)

    # Run backtest with cost sensitivity
    logger.info("Running backtest with cost sensitivity...")
    engine = BacktestEngine(
        strategy=strategy,
        data_dir=str(data_dir),
        initial_capital=args.initial_capital,
    )

    result = engine.run_with_cost_sensitivity(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=args.slug,
        cost_model=cost_model,
        fill_delay=fill_delay,
        warmup_days=warmup_days,
        benchmark_weights=benchmark_weights,
    )

    # Append to experiment registry
    registry = ExperimentRegistry(strat_dir)
    base_metrics = result.metrics.get("1.0x")

    registry_entry = {
        "experiment_id": result.experiment_id,
        "strategy_name": result.strategy_name,
        "slug": args.slug,
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
        "symbols": symbols,
        "total_return": base_metrics.total_return if base_metrics else 0,
        "sharpe_ratio": base_metrics.sharpe_ratio if base_metrics else 0,
        "max_drawdown": base_metrics.max_drawdown if base_metrics else 0,
        "dsr": base_metrics.dsr if base_metrics else 0,
        "total_trades": base_metrics.total_trades if base_metrics else 0,
        "spec_hash": spec.get("frozen_hash", ""),
        "parameters": config.parameters,
    }
    trial_number = registry.append(registry_entry)
    result.trial_number = trial_number

    # Persist experiment artifact
    experiments_dir = strat_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    if base_metrics:
        artifact = {
            "experiment_id": result.experiment_id,
            "trial_number": trial_number,
            "strategy_name": result.strategy_name,
            "start_date": str(result.start_date),
            "end_date": str(result.end_date),
            "initial_capital": result.initial_capital,
            "symbols": result.symbols_used,
            "spec_hash": spec.get("frozen_hash", ""),
            "cost_model": {
                "spread_bps": cost_model.spread_bps,
                "slippage_volatility_factor": cost_model.slippage_volatility_factor,
                "flat_slippage_bps": cost_model.flat_slippage_bps,
            },
            "metrics_1x": {
                "total_return": base_metrics.total_return,
                "annualized_return": base_metrics.annualized_return,
                "sharpe_ratio": base_metrics.sharpe_ratio,
                "sortino_ratio": base_metrics.sortino_ratio,
                "calmar_ratio": base_metrics.calmar_ratio,
                "max_drawdown": base_metrics.max_drawdown,
                "dsr": base_metrics.dsr,
                "psr": base_metrics.psr,
                "total_trades": base_metrics.total_trades,
                "win_rate": base_metrics.win_rate,
            },
            "daily_returns": result.daily_returns,
            "data_warnings": result.data_warnings,
        }
    else:
        artifact = {
            "experiment_id": result.experiment_id,
            "error": "No metrics computed",
        }

    save_artifact(experiments_dir / f"{result.experiment_id}.yaml", artifact)

    # Generate report
    report = generate_backtest_report(result)
    print(report)

    logger.info(
        "Experiment %s saved (trial #%d)",
        result.experiment_id,
        trial_number,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run robustness analysis for a strategy.

Usage:
    python scripts/run_robustness.py --slug behavioral-structural \
        --strategy behavioral_structural \
        --symbols XLK,XLF,XLE,XLV,XLI,XLY,XLP,XLU,XLRE,XLB,XLC,SPY,QQQ,IWM,GLD,TLT \
        --years 8

The script:
1. Loads the experiment artifact (daily returns) for CPCV
2. Generates parameter perturbations (+/-20%)
3. Re-runs backtest for each perturbation variant
4. Runs PBO on all return series
5. Runs CPCV on the base returns
6. Evaluates all gates and writes robustness.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.artifacts import (
    ExperimentRegistry,
    ensure_frozen_spec,
    save_artifact,
    strategy_dir,
)
from llm_quant.backtest.engine import BacktestEngine, CostModel
from llm_quant.backtest.metrics import compute_sharpe
from llm_quant.backtest.robustness import (
    PerturbationResult,
    run_robustness_gate,
)
from llm_quant.backtest.strategies import create_strategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.data.fetcher import fetch_ohlcv
from llm_quant.data.indicators import compute_indicators

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Shared context for backtest runs."""

    strategy_name: str
    spec: dict
    prices_df: object
    indicators_df: object
    slug: str
    cost_model: CostModel
    fill_delay: int
    warmup_days: int
    benchmark_weights: dict[str, float]
    initial_capital: float


def _build_config(
    strategy_name: str,
    spec: dict,
    param_overrides: dict | None = None,
) -> StrategyConfig:
    """Build StrategyConfig from spec with optional overrides."""
    params = dict(spec.get("parameters", {}))
    if param_overrides:
        params.update(param_overrides)

    mapped = dict(params)
    if "top_n_momentum" in params and "top_n" not in params:
        mapped["top_n"] = params["top_n_momentum"]
    if "momentum_lookback" in params and "lookback_days" not in params:
        mapped["lookback_days"] = params["momentum_lookback"]
    if "rebalance_frequency_days" in params:
        mapped.setdefault("rebalance_frequency", params["rebalance_frequency_days"])

    return StrategyConfig(
        name=strategy_name,
        rebalance_frequency_days=params.get(
            "rebalance_frequency_days",
            spec.get("rebalance_frequency_days", 5),
        ),
        max_positions=params.get("top_n_momentum", spec.get("max_positions", 10)),
        target_position_weight=params.get(
            "target_position_weight", spec.get("target_position_weight", 0.05)
        ),
        stop_loss_pct=params.get("stop_loss_pct", spec.get("stop_loss_pct", 0.05)),
        parameters=mapped,
    )


def _run_single(ctx: RunContext, param_overrides: dict | None = None) -> list[float]:
    """Run a single backtest and return daily returns."""
    config = _build_config(ctx.strategy_name, ctx.spec, param_overrides)
    strategy = create_strategy(ctx.strategy_name, config)
    engine = BacktestEngine(
        strategy=strategy,
        data_dir="data",
        initial_capital=ctx.initial_capital,
    )
    result = engine.run(
        prices_df=ctx.prices_df,
        indicators_df=ctx.indicators_df,
        slug=ctx.slug,
        cost_model=ctx.cost_model,
        fill_delay=ctx.fill_delay,
        warmup_days=ctx.warmup_days,
        cost_multiplier=1.0,
        benchmark_weights=ctx.benchmark_weights,
        trial_count=1,
    )
    return result.daily_returns


def _load_base_experiment(strat_dir, registry):
    """Load base experiment data (DSR and returns)."""
    returns_matrix = registry.get_returns_matrix()
    base_returns = returns_matrix[0] if returns_matrix else []

    entries = registry.load_all()
    base_entry = entries[-1] if entries else {}
    dsr = base_entry.get("dsr", 0.0)

    exp_id = base_entry.get("experiment_id", "")
    artifact_path = strat_dir / "experiments" / f"{exp_id}.yaml"
    if artifact_path.exists():
        with artifact_path.open() as f:
            art = yaml.safe_load(f) or {}
        m1x = art.get("metrics_1x", {})
        dsr = m1x.get("dsr", dsr)

    return returns_matrix, base_returns, dsr


def _run_perturbations(ctx, base_params, returns_matrix):
    """Generate and run parameter perturbation backtests."""
    pct = 0.20
    perturbations: list[tuple[str, dict]] = []

    for key, value in base_params.items():
        if isinstance(value, (int, float)) and value != 0:
            up_val = value * (1.0 + pct)
            down_val = value * (1.0 - pct)
            if isinstance(value, int):
                up_val = max(1, round(up_val))
                down_val = max(1, round(down_val))
            perturbations.append((f"{key}+20%", {key: up_val}))
            perturbations.append((f"{key}-20%", {key: down_val}))

    logger.info("Running %d perturbation backtests...", len(perturbations))
    results: list[PerturbationResult] = []
    all_returns = list(returns_matrix)

    for i, (desc, overrides) in enumerate(perturbations):
        logger.info("  [%d/%d] %s", i + 1, len(perturbations), desc)
        try:
            rets = _run_single(ctx, param_overrides=overrides)
            sharpe = compute_sharpe(rets)
            results.append(
                PerturbationResult(
                    name=desc,
                    parameter_change=str(overrides),
                    sharpe=sharpe,
                    profitable=sharpe > 0,
                )
            )
            if rets:
                all_returns.append(rets)
        except Exception:
            logger.exception("  Perturbation %s failed", desc)
            results.append(
                PerturbationResult(name=desc, parameter_change=str(overrides))
            )

    return perturbations, results, all_returns


def _print_results(  # noqa: PLR0913
    args,
    spec_hash,
    returns_matrix,
    perturbations,
    gate,
    cost_2x_sharpe,
    perturbation_results,
):
    """Print robustness analysis results."""
    print("\n" + "=" * 60)
    print(f"ROBUSTNESS ANALYSIS: {args.slug}")
    print("=" * 60)
    print(f"\nSpec hash: {spec_hash}")
    print(f"Experiments analyzed: {len(returns_matrix)}")
    print(f"Perturbation variants: {len(perturbations)}")
    print()

    dsr_s = "PASS" if gate.dsr_passed else "FAIL"
    pbo_s = "PASS" if gate.pbo_passed else "FAIL"
    cpcv_s = "PASS" if gate.cpcv_passed else "FAIL"
    cost_s = "PASS" if gate.cost_2x_survives else "FAIL"
    stab_s = "PASS" if gate.parameter_stability_passed else "FAIL"

    stable = sum(1 for p in perturbation_results if p.profitable)
    total = len(perturbation_results) or 1

    print("## Gate Results")
    print(f"  DSR >= 0.95:           {gate.dsr:.4f}  {dsr_s}")
    print(f"  PBO <= 0.10:           {gate.pbo.pbo:.4f}  {pbo_s}")
    print(f"  CPCV OOS Sharpe > 0:   {gate.cpcv.mean_oos_sharpe:.4f}  {cpcv_s}")
    print(f"  2x cost survive:       {cost_2x_sharpe:.4f}  {cost_s}")
    print(f"  Param stability >50%:  {stable}/{total}  {stab_s}")
    print()
    overall_s = "PASS" if gate.overall_passed else "FAIL"
    print(f"  OVERALL GATE: {overall_s}")

    print("\n## CPCV Details")
    print(f"  Combinations: {gate.cpcv.n_combinations}")
    print(f"  Mean OOS Sharpe: {gate.cpcv.mean_oos_sharpe:.4f}")
    print(f"  Std OOS Sharpe: {gate.cpcv.std_oos_sharpe:.4f}")

    print("\n## PBO Details")
    print(f"  Strategy variants: {gate.pbo.n_strategies}")
    print(f"  Combinations tested: {gate.pbo.n_combinations}")
    print(f"  PBO estimate: {gate.pbo.pbo:.4f}")

    print("\n## Perturbation Results")
    print(f"  {'Parameter':<30} {'Sharpe':>8} {'Status':>8}")
    print("  " + "-" * 48)
    for p in perturbation_results:
        status = "PASS" if p.profitable else "FAIL"
        print(f"  {p.name:<30} {p.sharpe:>8.4f} {status:>8}")

    return stable, total


def _write_artifact(  # noqa: PLR0913
    strat_dir,
    args,
    spec_hash,
    returns_matrix,
    perturbations,
    gate,
    cost_2x_sharpe,
    perturbation_results,
    stable,
    total,
):
    """Write robustness.yaml artifact."""
    artifact = {
        "strategy_slug": args.slug,
        "spec_hash": spec_hash,
        "experiments_analyzed": len(returns_matrix),
        "perturbation_variants": len(perturbations),
        "dsr": {
            "value": gate.dsr,
            "threshold": 0.95,
            "passed": gate.dsr_passed,
        },
        "pbo": {
            "method": "CSCV",
            "n_strategies": gate.pbo.n_strategies,
            "n_combinations": gate.pbo.n_combinations,
            "pbo_estimate": round(gate.pbo.pbo, 4),
            "threshold": 0.10,
            "passed": gate.pbo_passed,
        },
        "cpcv": {
            "n_combinations": gate.cpcv.n_combinations,
            "mean_oos_sharpe": round(gate.cpcv.mean_oos_sharpe, 4),
            "std_oos_sharpe": round(gate.cpcv.std_oos_sharpe, 4),
            "passed": gate.cpcv_passed,
        },
        "cost_sensitivity": {
            "cost_2x_sharpe": round(cost_2x_sharpe, 4),
            "survives": gate.cost_2x_survives,
        },
        "parameter_stability": {
            "stable_count": stable,
            "total_count": total,
            "stability_pct": round(gate.parameter_stability, 4),
            "passed": gate.parameter_stability_passed,
        },
        "perturbations": [
            {
                "name": p.name,
                "sharpe": round(p.sharpe, 4),
                "profitable": p.profitable,
            }
            for p in perturbation_results
        ],
        "overall_gate": "PASS" if gate.overall_passed else "FAIL",
        "gate_details": dict(gate.gate_details),
    }

    robustness_path = strat_dir / "robustness.yaml"
    save_artifact(robustness_path, artifact)
    logger.info("Robustness artifact saved to %s", robustness_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness analysis")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbols", default="SPY,QQQ,TLT,GLD,IEF")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    args = parser.parse_args()

    data_dir = Path("data")
    strat_d = strategy_dir(data_dir, args.slug)
    symbols = [s.strip() for s in args.symbols.split(",")]

    spec = ensure_frozen_spec(strat_d)
    spec_hash = spec.get("frozen_hash", "")
    logger.info("Loaded frozen spec %s (hash: %s)", args.slug, spec_hash)

    registry = ExperimentRegistry(strat_d)
    returns_matrix, base_returns, dsr = _load_base_experiment(strat_d, registry)

    if not base_returns:
        logger.error("No experiment with daily returns. Run backtest first.")
        sys.exit(1)

    # Fetch data
    logger.info("Fetching %d symbols (%d years)...", len(symbols), args.years)
    prices_df = fetch_ohlcv(symbols, lookback_days=args.years * 365)
    if len(prices_df) == 0:
        logger.error("No data fetched")
        sys.exit(1)

    logger.info("Computing indicators...")
    indicators_df = compute_indicators(prices_df)

    cost_model = CostModel.from_spec(spec)
    fill_delay = spec.get("fill_delay", 1)
    warmup_days = spec.get("warmup_days", 200)

    benchmark_weights = {"SPY": 0.60, "TLT": 0.40}
    benchmark = spec.get("benchmark", {})
    if benchmark:
        benchmark_weights = benchmark.get("symbols", benchmark_weights)

    ctx = RunContext(
        strategy_name=args.strategy,
        spec=spec,
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=args.slug,
        cost_model=cost_model,
        fill_delay=fill_delay,
        warmup_days=warmup_days,
        benchmark_weights=benchmark_weights,
        initial_capital=args.initial_capital,
    )

    # 2x cost backtest
    logger.info("Running 2x cost backtest...")
    config_2x = _build_config(args.strategy, spec)
    strategy_2x = create_strategy(args.strategy, config_2x)
    engine_2x = BacktestEngine(
        strategy=strategy_2x,
        data_dir="data",
        initial_capital=args.initial_capital,
    )
    result_2x = engine_2x.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=args.slug,
        cost_model=cost_model,
        fill_delay=fill_delay,
        warmup_days=warmup_days,
        cost_multiplier=2.0,
        benchmark_weights=benchmark_weights,
        trial_count=1,
    )
    metrics_2x = next(iter(result_2x.metrics.values()), None)
    cost_2x_sharpe = metrics_2x.sharpe_ratio if metrics_2x else 0.0
    logger.info("2x cost Sharpe: %.4f", cost_2x_sharpe)

    # Perturbation runs
    base_params = spec.get("parameters", {})
    perturbations, perturbation_results, all_returns = _run_perturbations(
        ctx, base_params, returns_matrix
    )

    # Robustness gate
    logger.info("Computing robustness gate...")
    gate = run_robustness_gate(
        dsr=dsr,
        returns_matrix=all_returns,
        best_returns=base_returns,
        cost_2x_sharpe=cost_2x_sharpe,
        perturbation_results=perturbation_results,
    )

    stable, total = _print_results(
        args,
        spec_hash,
        returns_matrix,
        perturbations,
        gate,
        cost_2x_sharpe,
        perturbation_results,
    )

    _write_artifact(
        strat_d,
        args,
        spec_hash,
        returns_matrix,
        perturbations,
        gate,
        cost_2x_sharpe,
        perturbation_results,
        stable,
        total,
    )


if __name__ == "__main__":
    main()

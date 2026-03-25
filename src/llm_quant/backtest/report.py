"""Markdown report generation for backtest results."""

from __future__ import annotations

import logging
import math
from typing import Any

from llm_quant.backtest.engine import BacktestResult
from llm_quant.backtest.robustness import RobustnessResult

logger = logging.getLogger(__name__)

GATE_DISPLAY = {
    "dsr_passed": ("DSR (Deflated Sharpe)", ">= 0.95"),
    "pbo_passed": ("PBO (Prob. Backtest Overfit)", "<= 0.10"),
    "cost_2x_passed": ("2x Cost Survival", "Profitable"),
    "cpcv_passed": ("CPCV Mean OOS Sharpe", "> 0"),
    "parameter_stability_passed": ("Parameter Stability", "> 50%"),
}


def _fmt_metric(value: Any, fmt: str = ".3f") -> str:
    """Format a metric value safely, handling None/inf/nan."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return "N/A"
    try:
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return str(value)


def generate_backtest_report(result: BacktestResult) -> str:
    """Generate a markdown report from a BacktestResult."""
    lines: list[str] = []

    lines.append(f"# Backtest Report: {result.strategy_name}")
    lines.append("")
    lines.append(f"**Experiment ID**: {result.experiment_id}")
    lines.append(f"**Strategy**: {result.strategy_name}")
    lines.append(f"**Slug**: {result.slug}")
    lines.append(f"**Period**: {result.start_date} to {result.end_date}")
    lines.append(f"**Initial Capital**: ${result.initial_capital:,.2f}")
    lines.append(f"**Symbols**: {', '.join(result.symbols_used)}")
    lines.append(f"**Trial #**: {result.trial_number}")
    lines.append("")

    # Cost sensitivity table
    lines.append("## Performance by Cost Multiplier")
    lines.append("")
    lines.append(
        "| Metric | " + " | ".join(f"{k}" for k in sorted(result.metrics.keys())) + " |"
    )
    lines.append("| --- | " + " | ".join("---" for _ in result.metrics) + " |")

    metric_rows = [
        ("Total Return", lambda m: _fmt_metric(m.total_return, ".2%")),
        ("Annualized Return", lambda m: _fmt_metric(m.annualized_return, ".2%")),
        ("Sharpe Ratio", lambda m: _fmt_metric(m.sharpe_ratio, ".3f")),
        ("Sortino Ratio", lambda m: _fmt_metric(m.sortino_ratio, ".3f")),
        ("Calmar Ratio", lambda m: _fmt_metric(m.calmar_ratio, ".3f")),
        ("Max Drawdown", lambda m: _fmt_metric(m.max_drawdown, ".2%")),
        (
            "DD Duration (days)",
            lambda m: _fmt_metric(m.max_drawdown_duration_days, "d"),
        ),
        ("Total Trades", lambda m: _fmt_metric(m.total_trades, "d")),
        ("Win Rate", lambda m: _fmt_metric(m.win_rate, ".1%")),
        ("Profit Factor", lambda m: _fmt_metric(m.profit_factor, ".2f")),
        ("DSR", lambda m: _fmt_metric(m.dsr, ".4f")),
        ("PSR", lambda m: _fmt_metric(m.psr, ".4f")),
    ]

    sorted_keys = sorted(result.metrics.keys())
    for label, fmt_fn in metric_rows:
        vals = []
        for key in sorted_keys:
            m = result.metrics[key]
            try:
                vals.append(fmt_fn(m))
            except (AttributeError, TypeError, ValueError):
                vals.append("N/A")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")

    # Benchmark comparison (from base run)
    base_metrics = result.metrics.get("1.0x")
    if base_metrics and base_metrics.benchmark_return is not None:
        lines.append("## Benchmark Comparison")
        lines.append("")
        lines.append("| Metric | Strategy | Benchmark |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| Total Return | {_fmt_metric(base_metrics.total_return, '.2%')} | "
            f"{_fmt_metric(base_metrics.benchmark_return, '.2%')} |"
        )
        lines.append(
            f"| Sharpe Ratio | {_fmt_metric(base_metrics.sharpe_ratio, '.3f')} | "
            f"{_fmt_metric(base_metrics.benchmark_sharpe, '.3f')} |"
        )
        lines.append(
            f"| Excess Return | {_fmt_metric(base_metrics.excess_return, '.2%')} | - |"
        )
        lines.append(
            f"| Information Ratio | "
            f"{_fmt_metric(base_metrics.information_ratio, '.3f')} | - |"
        )
        lines.append("")

    # Cost sensitivity warning
    if "2.0x" in result.metrics:
        m2x = result.metrics["2.0x"]
        if m2x.sharpe_ratio is not None and m2x.sharpe_ratio <= 0:
            lines.append(
                "> **WARNING**: Strategy is unprofitable at 2x costs "
                f"(Sharpe={_fmt_metric(m2x.sharpe_ratio, '.3f')}). "
                "This strategy may not survive real-world transaction costs."
            )
            lines.append("")

    # Data quality warnings
    if result.data_warnings:
        lines.append("## Data Quality Warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in result.data_warnings)
        lines.append("")

    return "\n".join(lines)


def generate_robustness_report(result: RobustnessResult) -> str:
    """Generate a markdown report from a RobustnessResult."""
    lines: list[str] = []

    lines.append("# Robustness Gate Report")
    lines.append("")

    # Gate summary
    status = "PASS" if result.overall_passed else "FAIL"
    lines.append(f"**Overall: {status}**")
    lines.append("")

    lines.append("## Gate Results")
    lines.append("")
    lines.append("| Gate | Value | Threshold | Status |")
    lines.append("| --- | --- | --- | --- |")

    gate_values = {
        "dsr_passed": lambda: _fmt_metric(result.dsr, ".4f"),
        "pbo_passed": lambda: _fmt_metric(result.pbo.pbo, ".4f"),
        "cost_2x_passed": lambda: "Yes" if result.cost_2x_survives else "No",
        "cpcv_passed": lambda: _fmt_metric(result.cpcv.mean_oos_sharpe, ".4f"),
        "parameter_stability_passed": lambda: _fmt_metric(
            result.parameter_stability, ".1%"
        ),
    }

    for gate, passed in result.gate_details.items():
        status_str = "PASS" if passed else "FAIL"
        if gate in GATE_DISPLAY:
            display_name, threshold = GATE_DISPLAY[gate]
            value = gate_values[gate]()
            lines.append(f"| {display_name} | {value} | {threshold} | {status_str} |")
        else:
            lines.append(f"| {gate} | - | - | {status_str} |")

    lines.append("")

    # PBO details
    if result.pbo.n_combinations > 0:
        lines.append("## PBO Details (CSCV)")
        lines.append("")
        lines.append(f"- Strategies tested: {result.pbo.n_strategies}")
        lines.append(f"- Combinations evaluated: {result.pbo.n_combinations}")
        lines.append(f"- PBO: {_fmt_metric(result.pbo.pbo, '.4f')}")
        lines.append("")

    # CPCV details
    if result.cpcv.n_combinations > 0:
        lines.append("## CPCV Details")
        lines.append("")
        lines.append(f"- Combinations: {result.cpcv.n_combinations}")
        lines.append(f"- Independent paths: {result.cpcv.n_paths}")
        lines.append(
            f"- Mean OOS Sharpe: {_fmt_metric(result.cpcv.mean_oos_sharpe, '.4f')}"
        )
        lines.append(
            f"- Std OOS Sharpe: {_fmt_metric(result.cpcv.std_oos_sharpe, '.4f')}"
        )
        lines.append("")

    # Perturbation details
    if result.perturbations:
        lines.append("## Perturbation Suite")
        lines.append("")
        lines.append("| Perturbation | Sharpe | Profitable |")
        lines.append("| --- | --- | --- |")
        for p in result.perturbations:
            status_str = "Yes" if p.profitable else "No"
            lines.append(
                f"| {p.name} | {_fmt_metric(p.sharpe, '.3f')} | {status_str} |"
            )
        lines.append("")

    return "\n".join(lines)

"""Dependency inventory and explicitly synthetic integration checks; no data downloads."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmark import EvaluationPlan
from .allocation import cvxpy_min_cvar, skfolio_min_risk, sklearn_inverse_volatility
from .contracts import MissingIntegration, TrainingBatch, doctor
from .dataframes import pandera_ohlc
from .evaluation import ALLOCATORS, evaluate
from .execution import next_open_buy_hold
from .tracking import record_qlib_summary
from .uncertainty import arch_garch_variance, arch_mean_interval, statsmodels_hac_mean


def fixture() -> TrainingBatch:
    rng = np.random.default_rng(1701)
    returns = rng.normal(0.0002, 0.012, (240, 3))
    dates = tuple((date(2000, 1, 1) + timedelta(days=i)).isoformat() for i in range(240))
    return TrainingBatch(returns, dates, dates[-1], "AUD", "synthetic")


def ohlc_fixture() -> pd.DataFrame:
    close = 100 * np.cumprod(1 + fixture().returns[:, 0])
    opened = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": opened, "High": np.maximum(opened, close) * 1.01,
                         "Low": np.minimum(opened, close) * 0.99, "Close": close},
                        index=pd.date_range("2000-01-01", periods=len(close), freq="D"))


def probe(backend: str, *, acknowledge_agpl: bool = False,
          qlib_directory: Path | None = None) -> dict:
    batch = fixture()
    single = TrainingBatch(batch.returns[:, :1], batch.dates, batch.as_of, "AUD", "synthetic")
    if backend == "sklearn":
        return sklearn_inverse_volatility(batch)
    if backend == "statsmodels":
        return statsmodels_hac_mean(single)
    if backend == "skfolio":
        return skfolio_min_risk(batch)
    if backend == "cvxpy":
        return cvxpy_min_cvar(batch)
    if backend == "arch":
        return {"status": "SYNTHETIC_INTEGRATION_CHECK",
                "bootstrap": arch_mean_interval(single, repetitions=200),
                "volatility": arch_garch_variance(single)}
    if backend == "pandera":
        checked = pandera_ohlc(ohlc_fixture())
        return {"status": "SYNTHETIC_INTEGRATION_CHECK", "backend": backend,
                "validated_rows": len(checked)}
    if backend == "backtesting":
        return next_open_buy_hold(ohlc_fixture(), acknowledge_agpl=acknowledge_agpl)
    if backend == "qlib":
        if qlib_directory is None:
            raise ValueError("Qlib recording requires explicit --qlib-directory under research-local/")
        return record_qlib_summary(sklearn_inverse_volatility(batch), qlib_directory)
    raise ValueError("Unsupported probe; Hypothesis runs through pytest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    smoke = commands.add_parser("probe")
    smoke.add_argument("backend", choices=("sklearn", "statsmodels", "skfolio", "arch",
                                          "cvxpy", "pandera", "backtesting", "qlib"))
    smoke.add_argument("--acknowledge-agpl", action="store_true")
    smoke.add_argument("--qlib-directory", type=Path)
    comparison = commands.add_parser("compare-synthetic")
    comparison.add_argument("--method", action="append", choices=ALLOCATORS, required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            output = doctor()
        elif args.command == "compare-synthetic":
            output = evaluate(fixture(), EvaluationPlan(), tuple(args.method))
        else:
            output = probe(args.backend, acknowledge_agpl=args.acknowledge_agpl,
                           qlib_directory=args.qlib_directory)
        sys.stdout.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    except (ValueError, MissingIntegration) as error:
        parser.exit(2, f"Integration refused: {error}\nNo fallback or automatic installation.\n")


if __name__ == "__main__":
    main()

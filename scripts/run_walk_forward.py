#!/usr/bin/env python3
"""CLI wrapper for the walk-forward validation engine.

Usage:
    python scripts/run_walk_forward.py --strategy sma_crossover --symbols SPY,QQQ,TLT

Exits 0 on PASS (OOS/IS ratio > 0.60), 1 on FAIL.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Ensure src is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    _build_parser,
    _print_aggregate,
    _print_fold_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else date(today.year - 5, today.month, today.day)
    )

    cfg = WalkForwardConfig(
        mode=args.mode,
        n_splits=args.n_splits,
        train_pct=args.train_pct,
        step_pct=args.step_pct,
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    strategy_params: dict[str, Any] = {"symbols": symbols}

    engine = WalkForwardEngine(strategy_name=args.strategy, config=cfg)

    print(f"\nWalk-Forward Validation — {args.strategy}")
    print(
        f"Mode: {args.mode}  |  Folds: {args.n_splits}  |  Period: {start_date} → {end_date}"
    )
    print()

    result = engine.run(
        strategy_name=args.strategy,
        start_date=start_date,
        end_date=end_date,
        strategy_params=strategy_params,
    )

    if result.n_folds == 0:
        print("ERROR: no folds completed — check date range, data, and strategy name.")
        sys.exit(1)

    _print_fold_table(result.folds)
    _print_aggregate(result)

    sys.exit(0 if result.passes_gate else 1)


if __name__ == "__main__":
    main()

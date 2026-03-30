"""CLI for IC (Information Coefficient) analysis on strategy signals.

Usage:
    python scripts/run_ic_analysis.py --strategy SLUG --signal COLUMN [--lookback 504]

Loads signal data from DuckDB (signals table) or falls back to a synthetic demo.
Runs IC analysis and prints a formatted report.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Ensure src is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_quant.analysis.ic_analysis import (
    IcAnalysisConfig,
    IcAnalyzer,
    format_ic_report,
)

logger = logging.getLogger(__name__)


def _load_signal_from_db(
    strategy_slug: str,
    signal_col: str,
    lookback: int,
) -> tuple[pl.Series, pl.Series, pl.Series] | None:
    """Load signal and price data from DuckDB.

    Returns (signal, price, dates) or None if not found.
    """
    try:
        import duckdb

        from llm_quant.config import load_config

        cfg = load_config()
        con = duckdb.connect(str(cfg.db_path), read_only=True)

        # Try indicators table first for signal column
        try:
            df = con.execute(
                f"""
                SELECT date, {signal_col}, close
                FROM indicators
                WHERE strategy_slug = ?
                ORDER BY date DESC
                LIMIT {lookback}
                """,
                [strategy_slug],
            ).pl()
        except Exception:
            # Try signals table
            df = con.execute(
                f"""
                SELECT date, {signal_col}, price
                FROM signals
                WHERE strategy_slug = ?
                ORDER BY date DESC
                LIMIT {lookback}
                """,
                [strategy_slug],
            ).pl()

        con.close()

        if df.is_empty():
            return None

        df = df.sort("date")
        return (
            df[signal_col].cast(pl.Float64),
            df.select(pl.last()).to_series().cast(pl.Float64),  # price/close col
            df["date"].cast(pl.Utf8),
        )

    except Exception as exc:
        logger.warning("Could not load from DB: %s", exc)
        return None


def _synthetic_demo(lookback: int) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Generate synthetic signal/price data for demo purposes."""
    np.random.seed(42)
    n = lookback
    forward_return = pl.Series("price", np.random.randn(n) * 0.01)
    signal = pl.Series("signal", forward_return.to_numpy() + np.random.randn(n) * 0.015)
    dates = pl.Series("date", [str(i) for i in range(n)])
    return signal, forward_return, dates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IC analysis on a strategy signal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy",
        default="demo",
        help="Strategy slug to load from DB (default: demo — uses synthetic data)",
    )
    parser.add_argument(
        "--signal",
        default="signal",
        help="Column name for the signal/factor in the DB table",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=504,
        help="Number of trading days to load (default: 504 = 2 years)",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=0.05,
        help="Advisory |IC| threshold (default: 0.05)",
    )
    parser.add_argument(
        "--icir-threshold",
        type=float,
        default=0.5,
        help="Advisory |ICIR| threshold (default: 0.5)",
    )
    args = parser.parse_args()

    config = IcAnalysisConfig(
        lookback_days=args.lookback,
        ic_threshold=args.ic_threshold,
        icir_threshold=args.icir_threshold,
    )

    signal, prices, dates = None, None, None

    if args.strategy != "demo":
        data = _load_signal_from_db(args.strategy, args.signal, args.lookback)
        if data is not None:
            signal, prices, dates = data
            print(f"Loaded {len(signal)} observations for strategy '{args.strategy}', signal '{args.signal}'")
        else:
            print(
                f"Warning: could not load strategy '{args.strategy}' signal '{args.signal}' from DB. "
                "Using synthetic demo data.",
                file=sys.stderr,
            )

    if signal is None:
        print("Using synthetic demo data (IC ~0.06 by construction)...")
        signal, prices, dates = _synthetic_demo(args.lookback)

    analyzer = IcAnalyzer(config)
    result = analyzer.analyze(signal, prices, dates)
    print(format_ic_report(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()

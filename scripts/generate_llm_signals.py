#!/usr/bin/env python3
"""Offline LLM signal pre-computation for backtesting.

Phase 1 of the LLM-backtest hybrid:
- For each sampled date, gathers market context available at that date
- Calls Claude with the PM agent prompt
- Stores result in data/strategies/{slug}/llm_signals/{date}.json

This is expensive ($10-40 per full run) — run once, cache forever.

Usage:
    python scripts/generate_llm_signals.py --slug my_strategy \
        --symbols SPY,QQQ,TLT --start 2023-01-01 --end 2024-01-01 \
        --sample-every 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute LLM signals for backtest consumption"
    )
    parser.add_argument("--slug", required=True, help="Strategy slug")
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,TLT,GLD,IEF",
        help="Comma-separated symbols",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="Sample every N trading days",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print dates that would be sampled without calling LLM",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    signal_dir = data_dir / "strategies" / args.slug / "llm_signals"
    signal_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in args.symbols.split(",")]
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC).date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC).date()

    # Fetch full historical data
    from llm_quant.data.fetcher import fetch_ohlcv
    from llm_quant.data.indicators import compute_indicators

    lookback_extra = 300  # extra days for warmup
    total_days = (end_date - start_date).days + lookback_extra
    logger.info("Fetching data for %d symbols (%d days)...", len(symbols), total_days)
    prices_df = fetch_ohlcv(symbols, lookback_days=total_days)

    if len(prices_df) == 0:
        logger.error("No data fetched — aborting")
        sys.exit(1)

    indicators_df = compute_indicators(prices_df)

    # Get trading dates in range
    all_dates = sorted(
        prices_df.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        .select("date")
        .unique()
        .to_series()
        .to_list()
    )

    # Sample dates
    sample_dates = all_dates[:: args.sample_every]
    logger.info(
        "Sampling %d dates from %s to %s (every %d days)",
        len(sample_dates),
        start_date,
        end_date,
        args.sample_every,
    )

    if args.dry_run:
        for d in sample_dates:
            existing = signal_dir / f"{d}.json"
            status = "cached" if existing.exists() else "pending"
            print(f"  {d}: {status}")
        logger.info("Dry run complete. %d dates would be sampled.", len(sample_dates))
        return

    # Process each date
    cached = 0
    generated = 0
    errors = 0

    for d in sample_dates:
        output_path = signal_dir / f"{d}.json"

        # Skip if already cached
        if output_path.exists():
            cached += 1
            continue

        try:
            signal_data = _generate_signal_for_date(
                d, indicators_df, prices_df, symbols
            )
            output_path.write_text(
                json.dumps(signal_data, indent=2, default=str),
                encoding="utf-8",
            )
            generated += 1
            logger.info("Generated signal for %s", d)
        except Exception:
            logger.exception("Failed to generate signal for %s", d)
            errors += 1

    logger.info(
        "Done: %d generated, %d cached, %d errors out of %d dates",
        generated,
        cached,
        errors,
        len(sample_dates),
    )


def _generate_signal_for_date(
    as_of_date: date,
    indicators_df: pl.DataFrame,
    _prices_df: pl.DataFrame,
    symbols: list[str],
) -> dict:
    """Generate an LLM signal for a specific date.

    This is a placeholder that builds the context but does NOT call
    the LLM API. In production, this would call Claude with the PM prompt.
    """
    # Filter to data available at as_of_date
    causal_data = indicators_df.filter(pl.col("date") <= as_of_date)
    today = causal_data.filter(pl.col("date") == as_of_date)

    # Build context
    context = {}
    for symbol in symbols:
        sym_row = today.filter(pl.col("symbol") == symbol)
        if len(sym_row) == 0:
            continue
        row = sym_row.row(0, named=True)
        context[symbol] = {
            "close": row.get("close"),
            "sma_20": row.get("sma_20"),
            "sma_50": row.get("sma_50"),
            "rsi_14": row.get("rsi_14"),
            "macd": row.get("macd"),
            "atr_14": row.get("atr_14"),
        }

    # Build input hash for reproducibility
    context_str = json.dumps(context, sort_keys=True, default=str)
    input_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]

    return {
        "date": str(as_of_date),
        "model_id": "placeholder",
        "prompt_hash": "placeholder",
        "input_hash": input_hash,
        "signals": [],
        "raw_response": "LLM signal generation not implemented",
        "cost_usd": 0.0,
        "context_summary": context,
    }


if __name__ == "__main__":
    main()

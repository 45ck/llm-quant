#!/usr/bin/env python3
"""Generate quantstats tearsheet for a strategy from its experiment artifact.

Loads daily returns from a strategy's best experiment artifact, fetches
benchmark returns (60/40 SPY/TLT for Track A, SPY for Track B), and writes
an HTML tearsheet + prints a key metrics table to stdout.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/generate_tearsheet.py <slug>
    cd E:/llm-quant && PYTHONPATH=src python scripts/generate_tearsheet.py <slug> --track b
    cd E:/llm-quant && PYTHONPATH=src python scripts/generate_tearsheet.py <slug> --exp-id <exp_id>
    cd E:/llm-quant && PYTHONPATH=src python scripts/generate_tearsheet.py <slug> --no-html

Output:
    data/strategies/<slug>/evaluate-tearsheet.html
    Key metrics table printed to stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.backtest.artifacts import load_artifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Strategy → experiment ID map (mirrors portfolio_optimizer.py)
# ---------------------------------------------------------------------------

STRATEGY_EXPERIMENTS: dict[str, str] = {
    "soxx-qqq-lead-lag": "57fba00d",
    "lqd-spy-credit-lead": "b0588e6d",
    "agg-spy-credit-lead": "66bec9a0",
    "hyg-spy-5d-credit-lead": "1736ac56",
    "agg-qqq-credit-lead": "eaf37299",
    "lqd-qqq-credit-lead": "ec8745f9",
    "vcit-qqq-credit-lead": "b99dac63",
    "hyg-qqq-credit-lead": "ba0c05a2",
    "emb-spy-credit-lead": "90e531d1",
    "agg-efa-credit-lead": "bef23aa4",
    "spy-overnight-momentum": "22cddf8c",
    "tlt-spy-rate-momentum": "9e14ce90",
    "tlt-qqq-rate-tech": "2338b9e5",
    "ief-qqq-rate-tech": "594c4f53",
    "behavioral-structural": "7cb2cace",
    "gld-slv-mean-reversion-v4": "14cdfaaf",
}

# Track B slugs: benchmark is SPY only. All others default to Track A (60/40 SPY/TLT).
TRACK_B_SLUGS: set[str] = {
    "btc-momentum-v2",
    "lqd-tqqq-sprint",
    "tlt-tqqq-sprint",
}


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------


def _fetch_yfinance_returns(symbols: list[str], start: date, end: date) -> dict[str, "pd.Series"]:  # type: ignore[name-defined]  # noqa: F821
    """Fetch daily returns via yfinance for benchmark construction.

    Returns dict of symbol -> pd.Series of daily returns (pct_change, dropna).
    """
    import pandas as pd
    import yfinance as yf  # type: ignore[import]

    result: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start.isoformat(),
                end=(end + timedelta(days=5)).isoformat(),
                auto_adjust=True,
            )
            if hist.empty or "Close" not in hist.columns:
                logger.warning("No price data for %s", symbol)
                continue
            prices = hist["Close"]
            # Normalize index: drop timezone so it aligns with tz-naive strategy returns
            if hasattr(prices.index, "tz") and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            rets = prices.pct_change().dropna()
            result[symbol] = rets
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s: %s", symbol, exc)
    return result


def build_benchmark_returns(
    track: str,
    start: date,
    end: date,
) -> "pd.Series | None":  # type: ignore[name-defined]  # noqa: F821
    """Build benchmark daily returns series.

    Track A: 60/40 blended SPY/TLT.
    Track B: 100% SPY.

    Returns a pd.Series with DatetimeIndex, or None if data unavailable.
    """
    import pandas as pd

    if track == "b":
        symbols = ["SPY"]
        weights = {"SPY": 1.0}
    else:
        symbols = ["SPY", "TLT"]
        weights = {"SPY": 0.60, "TLT": 0.40}

    logger.info("Fetching benchmark data for %s (%s)...", symbols, track.upper())
    price_returns = _fetch_yfinance_returns(symbols, start, end)

    if not price_returns:
        logger.warning("Could not fetch benchmark data — tearsheet will be benchmark-free")
        return None

    # Align all series on common dates
    frames = []
    for sym, w in weights.items():
        if sym not in price_returns:
            logger.warning("Missing benchmark component %s", sym)
            return None
        frames.append(price_returns[sym].rename(sym) * w)

    bench_df = pd.concat(frames, axis=1).dropna()
    bench_series = bench_df.sum(axis=1)
    bench_series.name = "60/40 SPY/TLT" if track == "a" else "SPY"
    return bench_series


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------


def print_metrics_table(
    strategy_returns: "pd.Series",  # type: ignore[name-defined]  # noqa: F821
    benchmark_returns: "pd.Series | None",  # type: ignore[name-defined]  # noqa: F821
    slug: str,
    tearsheet_path: Path | None,
) -> None:
    """Print a compact key-metrics table to stdout."""
    import quantstats as qs

    def _safe(fn, *args, default=float("nan")):
        try:
            val = fn(*args)
            return float(val) if val is not None else default
        except Exception:  # noqa: BLE001
            return default

    # Strategy metrics
    sharpe_s = _safe(qs.stats.sharpe, strategy_returns)
    sortino_s = _safe(qs.stats.sortino, strategy_returns)
    maxdd_s = _safe(qs.stats.max_drawdown, strategy_returns)
    calmar_s = _safe(qs.stats.calmar, strategy_returns)
    cagr_s = _safe(qs.stats.cagr, strategy_returns)

    header_line = f"{'Metric':<20} {'Strategy':>12}"

    if benchmark_returns is not None:
        bench_name = benchmark_returns.name or "Benchmark"
        # Align on common dates
        import pandas as pd
        aligned = pd.concat(
            [strategy_returns.rename("strat"), benchmark_returns.rename("bench")],
            axis=1,
        ).dropna()
        b_rets = aligned["bench"]

        sharpe_b = _safe(qs.stats.sharpe, b_rets)
        sortino_b = _safe(qs.stats.sortino, b_rets)
        maxdd_b = _safe(qs.stats.max_drawdown, b_rets)
        calmar_b = _safe(qs.stats.calmar, b_rets)
        cagr_b = _safe(qs.stats.cagr, b_rets)

        header_line = f"{'Metric':<20} {'Strategy':>12} {bench_name:>16}"
        data_lines = [
            f"{'-'*52}",
            f"{'Sharpe':<20} {sharpe_s:>12.3f} {sharpe_b:>16.3f}",
            f"{'Sortino':<20} {sortino_s:>12.3f} {sortino_b:>16.3f}",
            f"{'Max Drawdown':<20} {maxdd_s:>11.1%} {maxdd_b:>15.1%}",
            f"{'Calmar':<20} {calmar_s:>12.3f} {calmar_b:>16.3f}",
            f"{'CAGR':<20} {cagr_s:>11.1%} {cagr_b:>15.1%}",
            f"{'-'*52}",
        ]
    else:
        data_lines = [
            f"{'-'*34}",
            f"{'Sharpe':<20} {sharpe_s:>12.3f}",
            f"{'Sortino':<20} {sortino_s:>12.3f}",
            f"{'Max Drawdown':<20} {maxdd_s:>11.1%}",
            f"{'Calmar':<20} {calmar_s:>12.3f}",
            f"{'CAGR':<20} {cagr_s:>11.1%}",
            f"{'-'*34}",
        ]

    lines = [
        "",
        "=" * 70,
        f"EVALUATE TEARSHEET: {slug}",
        "=" * 70,
        "",
        header_line,
        *data_lines,
    ]

    if tearsheet_path:
        lines.append(f"\nTearsheet: {tearsheet_path}")

    lines.append("=" * 70)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Core tearsheet generation
# ---------------------------------------------------------------------------


def generate_tearsheet(
    slug: str,
    data_dir: Path,
    track: str = "a",
    exp_id: str | None = None,
    no_html: bool = False,
) -> Path | None:
    """Generate a quantstats tearsheet for the given strategy slug.

    Parameters
    ----------
    slug:
        Strategy slug (directory name under data/strategies/).
    data_dir:
        Root data directory.
    track:
        "a" (Track A, benchmark 60/40 SPY/TLT) or "b" (Track B, benchmark SPY).
    exp_id:
        Experiment ID to use. If None, uses STRATEGY_EXPERIMENTS map, then
        falls back to first .yaml in experiments/ directory.
    no_html:
        If True, skip HTML generation (metrics table only).

    Returns
    -------
    Path to the generated HTML file, or None if HTML was skipped.
    """
    import pandas as pd
    import quantstats as qs

    # Resolve experiment artifact
    if exp_id is None:
        exp_id = STRATEGY_EXPERIMENTS.get(slug)

    strategy_dir = data_dir / "strategies" / slug
    experiments_dir = strategy_dir / "experiments"

    if exp_id:
        artifact_path = experiments_dir / f"{exp_id}.yaml"
    else:
        # Try the first available experiment
        yaml_files = sorted(experiments_dir.glob("*.yaml")) if experiments_dir.exists() else []
        if not yaml_files:
            logger.error(
                "No experiment artifacts found for %s in %s",
                slug,
                experiments_dir,
            )
            sys.exit(1)
        artifact_path = yaml_files[0]
        exp_id = artifact_path.stem
        logger.warning(
            "No experiment ID registered for %s — using %s",
            slug,
            artifact_path,
        )

    if not artifact_path.exists():
        logger.error("Artifact not found: %s", artifact_path)
        sys.exit(1)

    # Load artifact
    logger.info("Loading artifact: %s", artifact_path)
    artifact = load_artifact(artifact_path)
    daily_returns = artifact.get("daily_returns", [])

    if not daily_returns:
        logger.error(
            "Artifact %s contains no daily_returns — re-run /backtest to regenerate",
            artifact_path,
        )
        sys.exit(1)

    start_date = artifact.get("start_date")
    end_date = artifact.get("end_date")
    logger.info(
        "Strategy %s: %d return observations (%s to %s)",
        slug,
        len(daily_returns),
        start_date,
        end_date,
    )

    # Build a daily DatetimeIndex spanning start→end (business days)
    if start_date and end_date:
        # start_date may be a date object or a string
        sd = pd.Timestamp(str(start_date))
        ed = pd.Timestamp(str(end_date))
        idx = pd.bdate_range(start=sd, end=ed, freq="B")
        # Trim/pad to match the number of returns
        n = len(daily_returns)
        if len(idx) > n:
            idx = idx[-n:]  # take most recent
        elif len(idx) < n:
            # Extend backwards
            idx = pd.bdate_range(end=ed, periods=n, freq="B")
    else:
        # No dates — generate backwards from today
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=len(daily_returns), freq="B")

    strategy_returns = pd.Series(daily_returns, index=idx, dtype=float)
    strategy_returns.name = slug

    # Determine benchmark
    if track == "b" or slug in TRACK_B_SLUGS:
        effective_track = "b"
    else:
        effective_track = "a"

    benchmark_returns = build_benchmark_returns(
        effective_track,
        start=idx[0].date(),
        end=idx[-1].date(),
    )

    # Print metrics table regardless of --no-html
    tearsheet_path: Path | None = None
    if not no_html:
        tearsheet_path = strategy_dir / "evaluate-tearsheet.html"

    print_metrics_table(strategy_returns, benchmark_returns, slug, tearsheet_path)

    # Generate HTML tearsheet
    if not no_html:
        strategy_dir.mkdir(parents=True, exist_ok=True)
        bench_label = "60/40 SPY/TLT" if effective_track == "a" else "SPY"
        title = f"{slug} vs {bench_label}"

        logger.info("Generating HTML tearsheet: %s", tearsheet_path)
        try:
            qs.reports.html(
                strategy_returns,
                benchmark=benchmark_returns,
                title=title,
                output=str(tearsheet_path),
                periods_per_year=TRADING_DAYS,
                match_dates=True,
            )
            logger.info("Tearsheet saved: %s", tearsheet_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to generate HTML tearsheet: %s", exc)
            return None

    return tearsheet_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quantstats tearsheet for a strategy from its experiment artifact",
    )
    parser.add_argument(
        "slug",
        help="Strategy slug (e.g. soxx-qqq-lead-lag)",
    )
    parser.add_argument(
        "--track",
        choices=["a", "b"],
        default=None,
        help=(
            "Track A = benchmark 60/40 SPY/TLT (default for most strategies), "
            "Track B = benchmark SPY. Auto-detected if omitted."
        ),
    )
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Experiment ID override. Defaults to registered best experiment for the slug.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        default=False,
        help="Print metrics table only; skip HTML tearsheet generation.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # If track not specified, auto-detect from slug
    if args.track is not None:
        track = args.track
    elif args.slug in TRACK_B_SLUGS:
        track = "b"
    else:
        track = "a"

    generate_tearsheet(
        slug=args.slug,
        data_dir=data_dir,
        track=track,
        exp_id=args.exp_id,
        no_html=args.no_html,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Multi-track execution runner: orchestrates daily signals across all tracks.

Runs the paper batch signal pipeline (shared data fetch), aggregates signals
by track using SignalAggregator, and reports per-track summaries side-by-side.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_multi_track.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_multi_track.py --dry-run
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_multi_track.py --track track-a
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.trading.signal_aggregator import SignalAggregator
from llm_quant.trading.track_router import TrackRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"
INITIAL_NAV = 100_000.0
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Track-level aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class TrackMetrics:
    """Aggregated metrics for a single track."""

    track_id: str
    display_name: str
    benchmark: str
    n_strategies: int
    n_active: int
    n_with_data: int
    aggregate_nav: float
    aggregate_return: float  # decimal
    aggregate_sharpe: float | None
    aggregate_max_dd: float  # decimal, positive
    today_return: float  # decimal
    regime_summary: str
    target_allocations: dict[str, float] = field(default_factory=dict)
    strategy_rows: list[dict] = field(default_factory=list)


def _load_paper_metrics(slug: str) -> dict | None:
    """Load paper-trading.yaml and extract latest metrics."""
    path = DATA_DIR / slug / "paper-trading.yaml"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    perf = data.get("performance", {})
    daily_log = data.get("daily_log", [])
    status = data.get("status", "unknown")

    # Extract latest day's return
    today_ret = 0.0
    last_regime = "unknown"
    last_position = "flat"
    if daily_log:
        last_entry = daily_log[-1]
        ret_pct = last_entry.get("daily_return_pct")
        if ret_pct is not None:
            today_ret = float(ret_pct) / 100.0
        last_regime = str(last_entry.get("regime", "unknown"))
        last_position = str(last_entry.get("position", "flat"))

    # Extract cumulative returns
    returns = []
    for entry in daily_log:
        r = entry.get("daily_return_pct")
        if r is not None:
            returns.append(float(r) / 100.0)

    # Compute metrics
    sharpe = None
    max_dd = 0.0
    total_return = 0.0
    nav = perf.get("current_nav", INITIAL_NAV)

    if len(returns) >= 5:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var)
        if std > 0:
            sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)

    if returns:
        nav_curve = [1.0]
        for r in returns:
            nav_curve.append(nav_curve[-1] * (1.0 + r))
        peak = nav_curve[0]
        for v in nav_curve:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        total_return = nav_curve[-1] / nav_curve[0] - 1.0

    return {
        "slug": slug,
        "status": status,
        "nav": nav,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "today_return": today_ret,
        "days": len(returns),
        "last_regime": last_regime,
        "last_position": last_position,
        "allocation": daily_log[-1].get("allocation", {}) if daily_log else {},
    }


def _aggregate_return_stats(
    all_daily_returns: list[list[float]],
) -> dict:
    """Compute aggregate Sharpe, MaxDD, total return from multiple return series."""
    if not all_daily_returns:
        return {"sharpe": None, "max_dd": 0.0, "total_return": 0.0}

    min_len = min(len(r) for r in all_daily_returns)
    if min_len == 0:
        return {"sharpe": None, "max_dd": 0.0, "total_return": 0.0}

    # Equal-weight daily returns
    agg_rets = []
    for i in range(min_len):
        day_avg = sum(r[len(r) - min_len + i] for r in all_daily_returns) / len(
            all_daily_returns
        )
        agg_rets.append(day_avg)

    # Sharpe
    sharpe = None
    if len(agg_rets) >= 5:
        mean = sum(agg_rets) / len(agg_rets)
        var = sum((r - mean) ** 2 for r in agg_rets) / len(agg_rets)
        std = math.sqrt(var)
        if std > 0:
            sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)

    # MaxDD
    nav_curve = [1.0]
    for r in agg_rets:
        nav_curve.append(nav_curve[-1] * (1.0 + r))
    peak = nav_curve[0]
    max_dd = 0.0
    for v in nav_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    total_return = nav_curve[-1] / nav_curve[0] - 1.0

    return {"sharpe": sharpe, "max_dd": max_dd, "total_return": total_return}


def compute_track_metrics(
    track_id: str,
    router: TrackRouter,
    aggregator: SignalAggregator,
) -> TrackMetrics:
    """Compute aggregated metrics for a single track."""
    track_info = router.get_track_info(track_id)
    slugs = router.get_strategies(track_id)

    strategy_rows = []
    all_daily_returns: list[list[float]] = []
    n_active = 0
    n_with_data = 0
    total_nav = 0.0
    today_returns = []
    regimes = []

    for slug in slugs:
        metrics = _load_paper_metrics(slug)
        if metrics is None:
            strategy_rows.append(
                {
                    "slug": slug,
                    "status": "Not Started",
                    "nav": INITIAL_NAV,
                    "total_return": 0.0,
                    "sharpe": None,
                    "max_dd": 0.0,
                    "today_return": 0.0,
                    "days": 0,
                    "last_regime": "--",
                }
            )
            total_nav += INITIAL_NAV
            continue

        strategy_rows.append(metrics)
        total_nav += metrics["nav"]

        if metrics["status"] == "active":
            n_active += 1
        if metrics["days"] > 0:
            n_with_data += 1
            today_returns.append(metrics["today_return"])
            regimes.append(metrics["last_regime"])

            # Load full return series for aggregate computation
            path = DATA_DIR / slug / "paper-trading.yaml"
            try:
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                daily_log = data.get("daily_log", [])
                rets = []
                for entry in daily_log:
                    r = entry.get("daily_return_pct")
                    if r is not None:
                        rets.append(float(r) / 100.0)
                if rets:
                    all_daily_returns.append(rets)
            except Exception:
                pass

    # Aggregate track returns (equal-weight across strategies with data)
    agg_stats = _aggregate_return_stats(all_daily_returns)
    aggregate_sharpe = agg_stats["sharpe"]
    aggregate_max_dd = agg_stats["max_dd"]
    aggregate_return = agg_stats["total_return"]
    avg_today = 0.0

    if today_returns:
        avg_today = sum(today_returns) / len(today_returns)

    # Regime summary
    if regimes:
        from collections import Counter

        regime_counts = Counter(regimes)
        top_3 = regime_counts.most_common(3)
        regime_summary = ", ".join(f"{r}({c})" for r, c in top_3)
    else:
        regime_summary = "no data"

    # Get aggregated target allocations from SignalAggregator
    try:
        track_signal = aggregator.aggregate_track(track_id)
        target_allocs = dict(track_signal.net_allocations)
    except Exception:
        target_allocs = {}

    return TrackMetrics(
        track_id=track_id,
        display_name=track_info.display_name,
        benchmark=track_info.benchmark,
        n_strategies=len(slugs),
        n_active=n_active,
        n_with_data=n_with_data,
        aggregate_nav=total_nav,
        aggregate_return=aggregate_return,
        aggregate_sharpe=aggregate_sharpe,
        aggregate_max_dd=aggregate_max_dd,
        today_return=avg_today,
        regime_summary=regime_summary,
        target_allocations=target_allocs,
        strategy_rows=strategy_rows,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_track_summary(tracks: list[TrackMetrics]) -> None:
    """Print the cross-track comparison table."""
    print()
    print("=" * 100)
    print("MULTI-TRACK EXECUTION SUMMARY")
    print("=" * 100)
    print()

    # Header
    print(
        f"{'Track':<30} {'Strats':>6} {'Active':>6} "
        f"{'AggNAV':>12} {'Return':>8} {'Sharpe':>7} "
        f"{'MaxDD':>7} {'Today':>8} {'Benchmark':<15}"
    )
    print("-" * 100)

    for tm in tracks:
        sharpe_str = f"{tm.aggregate_sharpe:.3f}" if tm.aggregate_sharpe else "  n/a"
        print(
            f"{tm.display_name:<30} {tm.n_strategies:>6} "
            f"{tm.n_with_data:>6} "
            f"${tm.aggregate_nav:>10,.0f} "
            f"{tm.aggregate_return:>+7.2%} "
            f"{sharpe_str:>7} "
            f"{tm.aggregate_max_dd:>6.2%} "
            f"{tm.today_return:>+7.3%} "
            f"{tm.benchmark:<15}"
        )

    print("-" * 100)

    # Totals
    total_nav = sum(t.aggregate_nav for t in tracks)
    total_strats = sum(t.n_strategies for t in tracks)
    total_active = sum(t.n_with_data for t in tracks)
    print(f"{'TOTAL':<30} {total_strats:>6} {total_active:>6} ${total_nav:>10,.0f}")
    print()


def render_track_detail(tm: TrackMetrics) -> None:
    """Print detailed strategy table for a track."""
    print(f"### {tm.display_name}")
    print(f"Regime distribution: {tm.regime_summary}")

    if tm.target_allocations:
        alloc_str = ", ".join(
            f"{s}: {w:.1%}"
            for s, w in sorted(tm.target_allocations.items(), key=lambda x: -x[1])[:5]
        )
        print(f"Aggregated target: {alloc_str}")

    print()
    print(
        f"  {'Strategy':<36} {'Days':>5} {'Return':>8} "
        f"{'Sharpe':>7} {'MaxDD':>7} {'Today':>8} {'Regime':<16}"
    )
    print("  " + "-" * 92)

    # Sort by Sharpe descending
    def sort_key(r):
        s = r.get("sharpe")
        if s is None:
            return (1, 0.0)
        return (0, -s)

    for row in sorted(tm.strategy_rows, key=sort_key):
        sharpe_str = f"{row['sharpe']:.3f}" if row.get("sharpe") else "  n/a"
        regime = str(row.get("last_regime", "--"))
        if len(regime) > 16:
            regime = regime[:14] + ".."
        print(
            f"  {row['slug']:<36} {row.get('days', 0):>5} "
            f"{row.get('total_return', 0.0):>+7.2%} "
            f"{sharpe_str:>7} "
            f"{row.get('max_dd', 0.0):>6.2%} "
            f"{row.get('today_return', 0.0):>+7.3%} "
            f"{regime:<16}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-track execution runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show aggregated state without running paper batch",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Filter to a specific track (e.g. track-a, track-b)",
    )
    parser.add_argument(
        "--run-batch",
        action="store_true",
        default=False,
        help="Run paper batch signals before aggregating (calls run_paper_batch.py)",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        default=False,
        help="Show per-strategy detail tables for each track",
    )
    args = parser.parse_args()

    today = datetime.date.today()
    logger.info("Multi-track execution run: %s", today)

    # Optionally run the paper batch first
    if args.run_batch and not args.dry_run:
        import subprocess

        logger.info("Running paper batch signals...")
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(Path(__file__).parent / "run_paper_batch.py")],
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("Paper batch failed: %s", result.stderr[-500:])
            return 1
        logger.info("Paper batch complete.")

    # Load router and aggregator
    router = TrackRouter.load_from_yaml()
    aggregator = SignalAggregator(
        strategies_dir=DATA_DIR,
        router=router,
        ref_date=today,
    )

    # Track order
    track_order = ["track-a", "track-b", "track-d", "discretionary"]
    if args.track:
        track_order = [args.track]

    # Compute per-track metrics
    track_metrics: list[TrackMetrics] = []
    for track_id in track_order:
        if track_id not in router.track_ids:
            continue
        tm = compute_track_metrics(track_id, router, aggregator)
        track_metrics.append(tm)

    if not track_metrics:
        logger.error("No tracks found.")
        return 1

    # Render summary
    print(f"\n## Multi-Track Report -- {today}")
    render_track_summary(track_metrics)

    # Render detail if requested
    if args.detail:
        for tm in track_metrics:
            render_track_detail(tm)

    return 0


if __name__ == "__main__":
    sys.exit(main())

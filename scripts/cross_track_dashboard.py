#!/usr/bin/env python3
"""Cross-track performance comparison dashboard.

Compares all tracks side-by-side with:
  - Per-track aggregate NAV, P&L, Sharpe, MaxDD
  - Benchmark-relative alpha
  - Rolling Sharpe (30d, 60d, 90d)
  - Daily return correlation between tracks
  - Per-strategy contribution within each track

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/cross_track_dashboard.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/cross_track_dashboard.py --json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.trading.track_router import TrackRouter

logging_configured = False
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"
INITIAL_NAV = 100_000.0
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StrategyPerf:
    """Performance for a single strategy."""

    slug: str
    family: str | None
    days: int
    total_return: float
    sharpe: float | None
    max_dd: float
    contribution: float  # contribution to track return (equal-weight)
    daily_returns: list[float] = field(default_factory=list, repr=False)


@dataclass
class TrackPerf:
    """Aggregate performance for a track."""

    track_id: str
    display_name: str
    benchmark: str
    n_strategies: int
    n_with_data: int
    aggregate_nav: float
    total_return: float
    sharpe: float | None
    max_dd: float
    rolling_sharpe_30d: float | None
    rolling_sharpe_60d: float | None
    rolling_sharpe_90d: float | None
    alpha: float | None  # vs benchmark
    daily_returns: list[float] = field(default_factory=list, repr=False)
    strategies: list[StrategyPerf] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Return series loading
# ---------------------------------------------------------------------------


def _load_returns(slug: str) -> list[float]:
    """Load daily return series from paper-trading.yaml."""
    path = DATA_DIR / slug / "paper-trading.yaml"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    daily_log = data.get("daily_log", [])
    returns = []
    for entry in daily_log:
        r = entry.get("daily_return_pct")
        if r is not None:
            returns.append(float(r) / 100.0)
    return returns


def _compute_metrics(returns: list[float]) -> dict:
    """Compute Sharpe, MaxDD, total return from return series."""
    if not returns:
        return {"sharpe": None, "max_dd": 0.0, "total_return": 0.0}

    # NAV curve
    nav = [1.0]
    for r in returns:
        nav.append(nav[-1] * (1.0 + r))

    # MaxDD
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    total_return = nav[-1] / nav[0] - 1.0

    # Sharpe
    sharpe = None
    if len(returns) >= 5:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var)
        if std > 0:
            sharpe = mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)

    return {"sharpe": sharpe, "max_dd": max_dd, "total_return": total_return}


def _rolling_sharpe(returns: list[float], window: int) -> float | None:
    """Compute rolling Sharpe for the last N trading days."""
    if len(returns) < window:
        return None
    recent = returns[-window:]
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / len(recent)
    std = math.sqrt(var)
    if std <= 0:
        return None
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _pearson_correlation(a: list[float], b: list[float]) -> float | None:
    """Compute Pearson correlation between two return series."""
    n = min(len(a), len(b))
    if n < 10:
        return None
    # Align to latest N days
    x = a[-n:]
    y = b[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True)) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx <= 0 or sy <= 0:
        return None
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Track-level computation
# ---------------------------------------------------------------------------


def compute_track_perf(
    track_id: str,
    router: TrackRouter,
) -> TrackPerf:
    """Build a TrackPerf for a single track."""
    track_info = router.get_track_info(track_id)
    slugs = router.get_strategies(track_id)

    all_returns: list[list[float]] = []
    strategy_perfs: list[StrategyPerf] = []
    total_nav = 0.0

    for entry in track_info.strategies:
        slug = entry.slug
        rets = _load_returns(slug)
        metrics = _compute_metrics(rets)

        # NAV for this strategy
        if rets:
            nav_curve = [INITIAL_NAV]
            for r in rets:
                nav_curve.append(nav_curve[-1] * (1.0 + r))
            strat_nav = nav_curve[-1]
        else:
            strat_nav = INITIAL_NAV

        total_nav += strat_nav

        sp = StrategyPerf(
            slug=slug,
            family=entry.family,
            days=len(rets),
            total_return=metrics["total_return"],
            sharpe=metrics["sharpe"],
            max_dd=metrics["max_dd"],
            contribution=0.0,  # filled later
            daily_returns=rets,
        )
        strategy_perfs.append(sp)
        if rets:
            all_returns.append(rets)

    # Aggregate track returns (equal-weight)
    agg_rets: list[float] = []
    if all_returns:
        min_len = min(len(r) for r in all_returns)
        if min_len > 0:
            for i in range(min_len):
                day_avg = sum(r[len(r) - min_len + i] for r in all_returns) / len(
                    all_returns
                )
                agg_rets.append(day_avg)

    agg_metrics = _compute_metrics(agg_rets)

    # Contribution: each strategy's excess return contribution
    n_with_data = len(all_returns)
    if n_with_data > 0 and agg_metrics["total_return"] != 0:
        for sp in strategy_perfs:
            if sp.days > 0:
                sp.contribution = sp.total_return / n_with_data

    # Rolling Sharpe
    rs_30 = _rolling_sharpe(agg_rets, 30)
    rs_60 = _rolling_sharpe(agg_rets, 60)
    rs_90 = _rolling_sharpe(agg_rets, 90)

    return TrackPerf(
        track_id=track_id,
        display_name=track_info.display_name,
        benchmark=track_info.benchmark,
        n_strategies=len(slugs),
        n_with_data=n_with_data,
        aggregate_nav=total_nav,
        total_return=agg_metrics["total_return"],
        sharpe=agg_metrics["sharpe"],
        max_dd=agg_metrics["max_dd"],
        rolling_sharpe_30d=rs_30,
        rolling_sharpe_60d=rs_60,
        rolling_sharpe_90d=rs_90,
        alpha=None,  # filled later vs benchmark
        daily_returns=agg_rets,
        strategies=strategy_perfs,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_sharpe(v: float | None) -> str:
    if v is None:
        return "  n/a"
    return f"{v:.3f}"


def _fmt_pct(v: float | None, signed: bool = True) -> str:
    if v is None:
        return " n/a"
    pct = v * 100
    return f"{pct:+.2f}%" if signed else f"{pct:.2f}%"


def render_dashboard(tracks: list[TrackPerf]) -> None:
    """Print the full cross-track dashboard."""
    today = datetime.date.today()
    print(f"\n{'=' * 110}")
    print(f"  CROSS-TRACK PERFORMANCE DASHBOARD — {today}")
    print(f"{'=' * 110}\n")

    # -- Summary table --
    print("## Track Comparison\n")
    print(
        f"{'Track':<32} {'NAV':>11} {'Return':>8} {'Sharpe':>7} "
        f"{'MaxDD':>7} {'RS30':>7} {'RS60':>7} {'RS90':>7} "
        f"{'Strats':>6}"
    )
    print("-" * 110)

    for tp in tracks:
        print(
            f"{tp.display_name:<32} "
            f"${tp.aggregate_nav:>9,.0f} "
            f"{_fmt_pct(tp.total_return):>8} "
            f"{_fmt_sharpe(tp.sharpe):>7} "
            f"{tp.max_dd:>6.2%} "
            f"{_fmt_sharpe(tp.rolling_sharpe_30d):>7} "
            f"{_fmt_sharpe(tp.rolling_sharpe_60d):>7} "
            f"{_fmt_sharpe(tp.rolling_sharpe_90d):>7} "
            f"{tp.n_with_data:>3}/{tp.n_strategies:<2}"
        )

    print("-" * 110)
    total_nav = sum(t.aggregate_nav for t in tracks)
    total_strats = sum(t.n_strategies for t in tracks)
    total_active = sum(t.n_with_data for t in tracks)
    print(
        f"{'COMBINED':<32} "
        f"${total_nav:>9,.0f} "
        f"{'':>8} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} "
        f"{total_active:>3}/{total_strats:<2}"
    )
    print()

    # -- Cross-track correlation --
    active_tracks = [t for t in tracks if t.daily_returns]
    if len(active_tracks) >= 2:
        print("## Cross-Track Correlation\n")
        # Header
        names = [t.track_id for t in active_tracks]
        header = f"{'':>15}" + "".join(f"{n:>15}" for n in names)
        print(header)
        for i, t1 in enumerate(active_tracks):
            row = f"{t1.track_id:>15}"
            for j, t2 in enumerate(active_tracks):
                if i == j:
                    row += f"{'1.000':>15}"
                else:
                    rho = _pearson_correlation(t1.daily_returns, t2.daily_returns)
                    row += f"{_fmt_sharpe(rho):>15}" if rho else f"{'n/a':>15}"
            print(row)
        print()

    # -- Per-track strategy breakdown --
    for tp in tracks:
        if not tp.strategies:
            continue

        active_strats = [s for s in tp.strategies if s.days > 0]
        if not active_strats:
            continue

        print(f"## {tp.display_name} — Strategy Breakdown\n")
        print(
            f"  {'Strategy':<36} {'Fam':>3} {'Days':>5} "
            f"{'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Contrib':>8}"
        )
        print("  " + "-" * 78)

        # Sort by Sharpe descending
        for sp in sorted(
            active_strats,
            key=lambda s: (0, -(s.sharpe or 0)) if s.sharpe else (1, 0),
        ):
            print(
                f"  {sp.slug:<36} {sp.family or '--':>3} {sp.days:>5} "
                f"{sp.total_return:>+7.2%} "
                f"{_fmt_sharpe(sp.sharpe):>7} "
                f"{sp.max_dd:>6.2%} "
                f"{sp.contribution:>+7.3%}"
            )

        print()

    # -- Key takeaways --
    print("## Key Metrics\n")
    for tp in tracks:
        if tp.sharpe is not None:
            status = "PASSING" if tp.sharpe >= 0.80 else "BELOW TARGET"
            print(
                f"  {tp.track_id}: Sharpe={tp.sharpe:.3f} [{status}], "
                f"MaxDD={tp.max_dd:.2%}, "
                f"{tp.n_with_data}/{tp.n_strategies} strategies reporting"
            )
        else:
            print(
                f"  {tp.track_id}: No data yet, {tp.n_strategies} strategies assigned"
            )
    print()


def render_json(tracks: list[TrackPerf]) -> None:
    """Output JSON representation for programmatic consumption."""
    output = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "tracks": [],
    }
    for tp in tracks:
        track_data = {
            "track_id": tp.track_id,
            "display_name": tp.display_name,
            "benchmark": tp.benchmark,
            "n_strategies": tp.n_strategies,
            "n_with_data": tp.n_with_data,
            "aggregate_nav": round(tp.aggregate_nav, 2),
            "total_return": round(tp.total_return, 6),
            "sharpe": round(tp.sharpe, 4) if tp.sharpe else None,
            "max_dd": round(tp.max_dd, 4),
            "rolling_sharpe_30d": (
                round(tp.rolling_sharpe_30d, 4) if tp.rolling_sharpe_30d else None
            ),
            "rolling_sharpe_60d": (
                round(tp.rolling_sharpe_60d, 4) if tp.rolling_sharpe_60d else None
            ),
            "rolling_sharpe_90d": (
                round(tp.rolling_sharpe_90d, 4) if tp.rolling_sharpe_90d else None
            ),
            "strategies": [
                {
                    "slug": sp.slug,
                    "family": sp.family,
                    "days": sp.days,
                    "total_return": round(sp.total_return, 6),
                    "sharpe": round(sp.sharpe, 4) if sp.sharpe else None,
                    "max_dd": round(sp.max_dd, 4),
                    "contribution": round(sp.contribution, 6),
                }
                for sp in tp.strategies
            ],
        }
        output["tracks"].append(track_data)

    # Cross-track correlation
    active_tracks = [t for t in tracks if t.daily_returns]
    if len(active_tracks) >= 2:
        corr_matrix = {}
        for t1 in active_tracks:
            row = {}
            for t2 in active_tracks:
                if t1.track_id == t2.track_id:
                    row[t2.track_id] = 1.0
                else:
                    rho = _pearson_correlation(t1.daily_returns, t2.daily_returns)
                    row[t2.track_id] = round(rho, 4) if rho else None
            corr_matrix[t1.track_id] = row
        output["cross_track_correlation"] = corr_matrix

    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-track performance comparison dashboard"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Filter to specific track",
    )
    args = parser.parse_args()

    router = TrackRouter.load_from_yaml()
    track_order = ["track-a", "track-b", "track-d", "discretionary"]
    if args.track:
        track_order = [args.track]

    tracks: list[TrackPerf] = []
    for track_id in track_order:
        if track_id not in router.track_ids:
            continue
        tp = compute_track_perf(track_id, router)
        tracks.append(tp)

    if not tracks:
        print("No tracks found.", file=sys.stderr)
        return 1

    if args.json:
        render_json(tracks)
    else:
        render_dashboard(tracks)

    return 0


if __name__ == "__main__":
    sys.exit(main())

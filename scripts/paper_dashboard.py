#!/usr/bin/env python3
"""Paper trading dashboard: displays current state of all strategies.

Reads paper-trading.yaml files from data/strategies/*/paper-trading.yaml,
computes key metrics for each strategy, and outputs a formatted markdown
table grouped by track assignment and sorted by Sharpe ratio.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/paper_dashboard.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/paper_dashboard.py --track track-a
    cd E:/llm-quant && PYTHONPATH=src python scripts/paper_dashboard.py --track track-b
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.trading.track_router import TrackRouter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StrategyMetrics:
    """Computed paper-trading metrics for a single strategy."""

    slug: str
    track: str
    track_display: str
    family: str | None
    days: int
    total_trades: int
    cumulative_return: float | None  # decimal (e.g. 0.021 = +2.1%)
    current_nav: float | None
    sharpe: float | None
    max_dd: float | None  # decimal, positive (e.g. 0.012 = 1.2%)
    last_signal: str  # last regime/position description
    status: str  # "Active", "Not Started", etc.
    start_date: str | None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics_from_log(daily_log: list[dict]) -> dict:
    """Compute Sharpe, MaxDD, cumulative return from a daily log.

    Returns dict with keys: sharpe, max_dd, total_return, days.
    Handles both percentage-based daily_return_pct entries and
    entries with missing/null return values gracefully.
    """
    returns: list[float] = []
    for entry in daily_log:
        ret_pct = entry.get("daily_return_pct")
        if ret_pct is None:
            continue
        try:
            returns.append(float(ret_pct) / 100.0)
        except (TypeError, ValueError):
            continue

    if len(returns) < 1:
        return {"sharpe": None, "max_dd": 0.0, "total_return": None, "days": 0}

    # Build NAV curve
    nav = [1.0]
    for r in returns:
        nav.append(nav[-1] * (1.0 + r))

    # Max drawdown
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Cumulative return
    total_return = nav[-1] / nav[0] - 1.0

    # Sharpe ratio (need >= 5 data points for meaningful estimate)
    sharpe = None
    if len(returns) >= 5:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(var_r)
        if std_r > 0:
            sharpe = round(mean_r / std_r * math.sqrt(TRADING_DAYS_PER_YEAR), 3)

    return {
        "sharpe": sharpe,
        "max_dd": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "days": len(returns),
    }


def _empty_metrics(
    slug: str,
    track_id: str,
    track_display: str,
    family: str | None,
    status: str,
) -> StrategyMetrics:
    """Return a blank StrategyMetrics for missing/unreadable files."""
    return StrategyMetrics(
        slug=slug,
        track=track_id,
        track_display=track_display,
        family=family,
        days=0,
        total_trades=0,
        cumulative_return=None,
        current_nav=None,
        sharpe=None,
        max_dd=None,
        last_signal="--",
        status=status,
        start_date=None,
    )


def _load_yaml_data(yaml_path: Path) -> dict | None:
    """Load YAML data from path, returning None on any failure."""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _extract_nav(perf: dict, daily_log: list[dict]) -> float | None:
    """Extract current NAV from performance dict or last log entry."""
    nav = perf.get("current_nav")
    if nav is not None:
        return nav
    for entry in reversed(daily_log):
        if entry.get("nav") is not None:
            return entry["nav"]
    return None


def _extract_cumulative_return(metrics: dict, perf: dict) -> float | None:
    """Extract cumulative return, preferring computed over stored."""
    ret = metrics["total_return"]
    if ret is not None and not (ret == 0 and metrics["days"] == 0):
        return ret
    stored = perf.get("total_return")
    if stored is None:
        return ret
    try:
        val = float(stored)
    except (TypeError, ValueError):
        return ret
    # performance.total_return is stored as percentage in some files
    return val / 100.0 if abs(val) > 1.0 else val


def _extract_last_signal(daily_log: list[dict]) -> str:
    """Extract the most recent signal description from the log."""
    if not daily_log:
        return "--"
    last = daily_log[-1]
    # Try regime first (batch runner), then signal (manual), then position
    for key in ("regime", "signal", "position"):
        val = last.get(key)
        if val:
            return str(val).upper()
    return "--"


def _determine_status(file_status: str, computed_days: int) -> str:
    """Map file status and data availability to a display status."""
    if file_status in {"retired", "halted"}:
        return file_status.capitalize()
    return "Active" if computed_days > 0 else "Pending"


def load_strategy_metrics(
    slug: str,
    track_id: str,
    track_display: str,
    family: str | None,
) -> StrategyMetrics:
    """Load a single strategy's paper-trading.yaml and compute metrics."""
    yaml_path = DATA_DIR / slug / "paper-trading.yaml"

    if not yaml_path.exists():
        return _empty_metrics(slug, track_id, track_display, family, "Not Started")

    data = _load_yaml_data(yaml_path)
    if data is None:
        return _empty_metrics(slug, track_id, track_display, family, "Error")

    daily_log = data.get("daily_log", [])
    perf = data.get("performance", {})
    start_date = data.get("start_date")
    file_status = data.get("status", "unknown")

    metrics = compute_metrics_from_log(daily_log)
    total_trades = perf.get("total_trades", 0) or 0

    # Fall back to calendar days if no return data points yet
    days = metrics["days"]
    if days == 0 and start_date:
        try:
            sd = datetime.date.fromisoformat(str(start_date))
            days_since = (datetime.date.today() - sd).days
            if days_since > 0:
                days = days_since
        except (ValueError, TypeError):
            pass

    return StrategyMetrics(
        slug=slug,
        track=track_id,
        track_display=track_display,
        family=family,
        days=metrics["days"],
        total_trades=total_trades,
        cumulative_return=_extract_cumulative_return(metrics, perf),
        current_nav=_extract_nav(perf, daily_log),
        sharpe=metrics["sharpe"],
        max_dd=metrics["max_dd"],
        last_signal=_extract_last_signal(daily_log),
        status=_determine_status(file_status, metrics["days"]),
        start_date=start_date,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_pct(value: float | None, signed: bool = True) -> str:
    """Format a decimal value as a percentage string."""
    if value is None:
        return "--"
    pct = value * 100
    if signed:
        return f"{pct:+.1f}%"
    return f"{pct:.1f}%"


def format_sharpe(value: float | None) -> str:
    """Format Sharpe ratio."""
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_dd(value: float | None) -> str:
    """Format max drawdown (show as negative percentage)."""
    if value is None:
        return "--"
    return f"-{value * 100:.1f}%"


def render_table(
    strategies: list[StrategyMetrics],
    track_display: str,
    benchmark: str,
) -> str:
    """Render a markdown table for a single track."""
    lines: list[str] = []
    lines.append(f"### {track_display}")
    lines.append(f"Benchmark: {benchmark}")
    lines.append("")
    lines.append(
        "| Strategy | Fam | Days | Trades | Return | Sharpe | MaxDD | Signal | Status |"
    )
    lines.append(
        "|----------|-----|------|--------|--------|--------|-------|--------|--------|"
    )

    # Sort: Sharpe descending (None values last)
    def sort_key(s: StrategyMetrics) -> tuple[int, float]:
        if s.sharpe is None:
            return (1, 0.0)
        return (0, -s.sharpe)

    for sm in sorted(strategies, key=sort_key):
        fam = sm.family or "--"
        ret_str = format_pct(sm.cumulative_return)
        sharpe_str = format_sharpe(sm.sharpe)
        dd_str = format_dd(sm.max_dd)
        signal_str = sm.last_signal
        # Truncate long signal strings
        if len(signal_str) > 18:
            signal_str = signal_str[:16] + ".."

        lines.append(
            f"| {sm.slug:<34} | {fam:<3} | {sm.days:>4} | {sm.total_trades:>6} "
            f"| {ret_str:>6} | {sharpe_str:>6} | {dd_str:>5} "
            f"| {signal_str:<18} | {sm.status:<7} |"
        )

    lines.append("")
    return "\n".join(lines)


def render_summary(all_strategies: list[StrategyMetrics]) -> str:
    """Render the summary section."""
    total = len(all_strategies)
    active = sum(1 for s in all_strategies if s.days >= 1)
    meeting_30d = sum(1 for s in all_strategies if s.days >= 30)
    has_sharpe = [s for s in all_strategies if s.sharpe is not None]

    # Sharpe gate depends on track
    meeting_sharpe = sum(
        1
        for s in has_sharpe
        if (s.track == "track-b" and s.sharpe is not None and s.sharpe >= 1.0)
        or (s.track == "track-d" and s.sharpe is not None and s.sharpe >= 0.80)
        or (
            s.track not in ("track-b", "track-d")
            and s.sharpe is not None
            and s.sharpe >= 0.80
        )
    )

    # Best / worst performers
    with_return = [
        s for s in all_strategies if s.cumulative_return is not None and s.days >= 1
    ]
    best = (
        max(with_return, key=lambda s: s.cumulative_return or -999)
        if with_return
        else None
    )  # type: ignore[arg-type]
    worst = (
        min(with_return, key=lambda s: s.cumulative_return or 999)
        if with_return
        else None
    )  # type: ignore[arg-type]

    lines: list[str] = []
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Total strategies**: {total}")
    lines.append(f"- **Active (>= 1 day)**: {active}")
    lines.append(f"- **Meeting 30-day minimum**: {meeting_30d}")
    lines.append(
        f"- **Meeting Sharpe gate**: {meeting_sharpe} / {len(has_sharpe)} with data"
    )
    if best:
        lines.append(
            f"- **Best performer**: {best.slug} "
            f"({format_pct(best.cumulative_return)}, "
            f"Sharpe={format_sharpe(best.sharpe)})"
        )
    if worst:
        lines.append(
            f"- **Worst performer**: {worst.slug} "
            f"({format_pct(worst.cumulative_return)}, "
            f"Sharpe={format_sharpe(worst.sharpe)})"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper trading dashboard for all strategies"
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Filter to a specific track (e.g. track-a, track-b, track-d)",
    )
    args = parser.parse_args()

    # Load track router
    router = TrackRouter.load_from_yaml()
    today = datetime.date.today()

    # Collect metrics for all strategies across all tracks
    all_strategies: list[StrategyMetrics] = []
    track_order = router.track_display_order

    for track_id in track_order:
        if track_id not in router.track_ids:
            continue
        if args.track and track_id != args.track:
            continue

        track_info = router.get_track_info(track_id)
        for entry in track_info.strategies:
            sm = load_strategy_metrics(
                slug=entry.slug,
                track_id=track_id,
                track_display=track_info.display_name,
                family=entry.family,
            )
            all_strategies.append(sm)

    if not all_strategies:
        print(f"No strategies found for track filter: {args.track}")
        sys.exit(1)

    # Render dashboard
    print(f"## Paper Trading Dashboard -- {today}")
    print()

    # Group by track, maintaining order
    seen_tracks: list[str] = []
    for sm in all_strategies:
        if sm.track not in seen_tracks:
            seen_tracks.append(sm.track)

    for track_id in seen_tracks:
        track_strategies = [s for s in all_strategies if s.track == track_id]
        if not track_strategies:
            continue
        track_info = router.get_track_info(track_id)
        table = render_table(
            track_strategies,
            track_info.display_name,
            track_info.benchmark,
        )
        print(table)

    # Summary
    summary = render_summary(all_strategies)
    print(summary)


if __name__ == "__main__":
    main()

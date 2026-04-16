#!/usr/bin/env python3
"""Track D batch performance monitor.

Generates a unified paper-trading snapshot for all 14 Track D passing strategies.

For each strategy:
  - Reads data/strategies/<slug>/paper-trading.yaml
  - Parses daily_log to compute actual NAV, return, running Sharpe, drawdown, hit rate
  - Compares actual vs proportional expected return from backtest CAGR
  - Flags CONCERN (return > 2 sigma below expected), WATCH (DD > 30% of backtest MaxDD),
    DORMANT (no signal in last 5 days), MISSING (no paper-trading.yaml)

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_batch_monitor.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_batch_monitor.py --out <path>
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"
DEFAULT_OUT = DATA_DIR / "sprint-alpha" / "batch-monitor-2026-04-17.md"
AS_OF_DATE = dt.date(2026, 4, 17)
TRADING_DAYS_PER_YEAR = 252

# -----------------------------------------------------------------------------
# Canonical Track D registry (as of 2026-04-17).
# slug: directory name under data/strategies/
# bt_sharpe: backtest Sharpe
# bt_cagr: backtest annualized return (decimal)
# bt_maxdd: backtest max drawdown (decimal)
# -----------------------------------------------------------------------------
TRACK_D_REGISTRY: list[dict] = [
    {
        "rank": 1,
        "slug": "tsmom-upro-trend-v1",  # D13
        "alias": "D13 TSMOM-UPRO",
        "bt_sharpe": 1.345,
        "bt_cagr": 0.201,
        "bt_maxdd": 0.159,
    },
    {
        "rank": 2,
        "slug": "xlk-xle-soxl-rotation-v1",  # D10
        "alias": "D10 XLK-XLE-SOXL",
        "bt_sharpe": 1.171,
        "bt_cagr": 0.277,
        "bt_maxdd": 0.226,
    },
    {
        "rank": 3,
        "slug": "agg-tqqq-sprint",
        "alias": "AGG-TQQQ Sprint",
        "bt_sharpe": 1.079,
        "bt_cagr": 0.109,
        "bt_maxdd": 0.106,
    },
    {
        "rank": 4,
        "slug": "tlt-tqqq-sprint",  # D1
        "alias": "TLT-TQQQ Sprint (D1)",
        "bt_sharpe": 1.030,
        "bt_cagr": 0.124,
        "bt_maxdd": 0.102,
    },
    {
        "rank": 5,
        "slug": "d15-vol-regime-tqqq",
        "alias": "D15 Vol-Regime-TQQQ",
        "bt_sharpe": 0.978,
        "bt_cagr": 0.153,
        "bt_maxdd": 0.199,
    },
    {
        "rank": 6,
        "slug": "tlt-soxl-sprint",
        "alias": "TLT-SOXL Sprint",
        "bt_sharpe": 0.936,
        "bt_cagr": 0.166,
        "bt_maxdd": 0.171,
    },
    {
        "rank": 7,
        "slug": "tlt-upro-sprint",
        "alias": "TLT-UPRO Sprint",
        "bt_sharpe": 0.903,
        "bt_cagr": 0.074,
        "bt_maxdd": 0.129,
    },
    {
        "rank": 8,
        "slug": "tip-tlt-upro-real-yield-v1",  # D12
        "alias": "D12 TIP-TLT-UPRO",
        "bt_sharpe": 0.895,
        "bt_cagr": 0.097,
        "bt_maxdd": 0.167,
    },
    {
        "rank": 9,
        "slug": "d14-disinflation-tqqq",
        "alias": "D14 Disinflation-TQQQ",
        "bt_sharpe": 0.883,
        "bt_cagr": 0.126,
        "bt_maxdd": 0.195,
    },
    {
        "rank": 10,
        "slug": "vcit-tqqq-sprint",
        "alias": "VCIT-TQQQ Sprint",
        "bt_sharpe": 0.880,
        "bt_cagr": 0.115,
        "bt_maxdd": 0.186,
    },
    {
        "rank": 11,
        "slug": "ief-tqqq-sprint",
        "alias": "IEF-TQQQ Sprint",
        "bt_sharpe": 0.852,
        "bt_cagr": 0.111,
        "bt_maxdd": 0.110,
    },
    {
        "rank": 12,
        "slug": "agg-upro-sprint",
        "alias": "AGG-UPRO Sprint",
        "bt_sharpe": 0.841,
        "bt_cagr": 0.061,
        "bt_maxdd": 0.074,
    },
    {
        "rank": 13,
        "slug": "soxx-soxl-lead-lag-v1",  # D11
        "alias": "D11 SOXX-SOXL",
        "bt_sharpe": 0.818,
        "bt_cagr": 0.251,
        "bt_maxdd": 0.372,
    },
    {
        "rank": 14,
        "slug": "lqd-tqqq-sprint",
        "alias": "LQD-TQQQ Sprint",
        "bt_sharpe": 0.803,
        "bt_cagr": 0.088,
        "bt_maxdd": 0.126,
    },
]


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class StrategySnapshot:
    rank: int
    slug: str
    alias: str
    bt_sharpe: float
    bt_cagr: float
    bt_maxdd: float
    # Paper metrics
    has_paper_file: bool = False
    start_date: dt.date | None = None
    last_signal_date: dt.date | None = None
    days_elapsed: int = 0
    num_entries: int = 0
    actual_cum_return: float = 0.0
    expected_cum_return: float = 0.0
    z_score: float | None = None
    running_sharpe: float | None = None
    current_drawdown: float = 0.0
    hit_rate: float | None = None
    status: str = "MISSING"
    notes: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _to_date(val) -> dt.date | None:
    if val is None:
        return None
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    if isinstance(val, str):
        try:
            return dt.date.fromisoformat(val.strip())
        except ValueError:
            return None
    return None


def _read_paper_yaml(slug: str) -> dict | None:
    path = DATA_DIR / slug / "paper-trading.yaml"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        print(f"WARN: failed to parse {path}: {exc}", file=sys.stderr)
        return None


def _parse_daily_log(daily_log: list[dict]) -> pl.DataFrame:
    """Convert paper-trading daily_log list into a Polars DataFrame."""
    rows: list[dict] = []
    for row in daily_log:
        d = _to_date(row.get("date"))
        if d is None:
            continue
        nav = row.get("nav")
        pnl = row.get("daily_pnl")
        ret_pct = row.get("daily_return_pct")
        if ret_pct is not None:
            try:
                daily_ret = float(ret_pct) / 100.0
            except (TypeError, ValueError):
                daily_ret = 0.0
        elif nav and pnl is not None:
            try:
                prev_nav = float(nav) - float(pnl)
                daily_ret = (float(pnl) / prev_nav) if prev_nav > 0 else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                daily_ret = 0.0
        else:
            daily_ret = 0.0

        rows.append(
            {
                "date": d,
                "nav": float(nav) if nav is not None else None,
                "daily_return": float(daily_ret),
                "position": (row.get("position") or "flat"),
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "date": [],
                "nav": [],
                "daily_return": [],
                "position": [],
            }
        )
    return pl.DataFrame(rows).sort("date")


def _compute_drawdown(nav_series: list[float], init_nav: float) -> float:
    """Current drawdown from peak NAV."""
    if not nav_series:
        return 0.0
    peak = max(nav_series[0], init_nav)
    latest = nav_series[-1]
    for nav in nav_series:
        peak = max(peak, nav)
    return ((peak - latest) / peak) if peak > 0 else 0.0


def _compute_running_sharpe(rets: list[float]) -> float | None:
    """Annualized Sharpe from daily return series (>= 5 obs with non-zero variance)."""
    if len(rets) < 5:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd <= 0:
        return None
    return (mu / sd) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _compute_metrics(entry: dict, snap: StrategySnapshot) -> None:
    """Populate snap fields from the paper-trading.yaml payload."""
    snap.has_paper_file = True
    snap.start_date = _to_date(entry.get("start_date"))

    daily_log = entry.get("daily_log") or []
    snap.num_entries = len(daily_log)

    if snap.start_date:
        snap.days_elapsed = max(0, (AS_OF_DATE - snap.start_date).days)

    perf = entry.get("performance") or {}
    init_nav = float(perf.get("initial_nav") or 100_000.0)

    if not daily_log:
        snap.status = "NO_TRADES"
        snap.notes.append("paper-trading.yaml exists but daily_log is empty")
        reported_total_return = perf.get("total_return")
        if reported_total_return is not None:
            try:
                val = float(reported_total_return)
                snap.actual_cum_return = val / 100.0 if abs(val) > 1.5 else val
            except (TypeError, ValueError):
                pass
        return

    df = _parse_daily_log(daily_log)
    if df.is_empty():
        snap.status = "NO_TRADES"
        snap.notes.append("daily_log present but no parseable rows")
        return

    snap.last_signal_date = df["date"].max()

    nav_series = [v for v in df["nav"].to_list() if v is not None]
    rets = [float(r) for r in df["daily_return"].to_list()]

    if nav_series:
        snap.actual_cum_return = (nav_series[-1] / init_nav) - 1.0
    else:
        compound = 1.0
        for r in rets:
            compound *= 1.0 + r
        snap.actual_cum_return = compound - 1.0

    active_rets = [r for r in rets if r != 0.0]
    snap.running_sharpe = _compute_running_sharpe(rets)
    snap.current_drawdown = _compute_drawdown(nav_series, init_nav)

    if active_rets:
        wins = sum(1 for r in active_rets if r > 0.0)
        snap.hit_rate = wins / len(active_rets)

    years_elapsed = snap.days_elapsed / 365.25 if snap.days_elapsed > 0 else 0.0
    snap.expected_cum_return = (
        (1.0 + snap.bt_cagr) ** years_elapsed - 1.0 if years_elapsed > 0 else 0.0
    )

    annual_vol = (snap.bt_cagr / snap.bt_sharpe) if snap.bt_sharpe > 0 else None
    if annual_vol and years_elapsed > 0:
        window_sigma = annual_vol * math.sqrt(years_elapsed)
        if window_sigma > 0:
            snap.z_score = (
                snap.actual_cum_return - snap.expected_cum_return
            ) / window_sigma


def _classify(snap: StrategySnapshot) -> None:
    """Assign snap.status based on collected metrics."""
    if not snap.has_paper_file:
        snap.status = "MISSING"
        snap.notes.append(
            "No paper-trading.yaml — strategy not yet initialized for paper trading"
        )
        return

    if snap.status == "NO_TRADES":
        return

    if snap.last_signal_date is not None:
        days_since_last = (AS_OF_DATE - snap.last_signal_date).days
        if days_since_last > 5:
            snap.status = "DORMANT"
            snap.notes.append(
                f"last signal {snap.last_signal_date} ({days_since_last}d ago) > 5d dormant gate"
            )
            return

    watch_triggered = False
    if snap.current_drawdown > 0.30 * snap.bt_maxdd and snap.current_drawdown > 0.02:
        snap.status = "WATCH"
        snap.notes.append(
            f"current DD {snap.current_drawdown:.2%} > 30% of backtest MaxDD ({snap.bt_maxdd:.2%})"
        )
        watch_triggered = True

    if snap.z_score is not None and snap.z_score < -2.0:
        snap.status = "CONCERN"
        snap.notes.append(
            f"cum return {snap.actual_cum_return:+.2%} is z={snap.z_score:.2f} "
            f"below proportional expected {snap.expected_cum_return:+.2%}"
        )
        return

    if not watch_triggered and snap.status not in {"DORMANT", "NO_TRADES"}:
        snap.status = "OK"


def _build_snapshot(entry: dict) -> StrategySnapshot:
    snap = StrategySnapshot(
        rank=entry["rank"],
        slug=entry["slug"],
        alias=entry["alias"],
        bt_sharpe=entry["bt_sharpe"],
        bt_cagr=entry["bt_cagr"],
        bt_maxdd=entry["bt_maxdd"],
    )

    payload = _read_paper_yaml(snap.slug)
    if payload is None:
        _classify(snap)
        return snap

    _compute_metrics(payload, snap)
    _classify(snap)
    return snap


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:+.2f}%"


def _fmt_float(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _render_markdown(snaps: list[StrategySnapshot]) -> str:
    lines: list[str] = []
    lines.append(
        f"# Track D Batch Paper-Performance Snapshot — {AS_OF_DATE.isoformat()}"
    )
    lines.append("")
    lines.append(
        "Monitors 14 Track D strategies currently in paper trading "
        "(beads llm-quant-twtl). Compares actual paper performance against "
        "proportional backtest expectations."
    )
    lines.append("")

    counts: dict[str, int] = {}
    for s in snaps:
        counts[s.status] = counts.get(s.status, 0) + 1

    lines.append("## Status Summary")
    lines.append("")
    for status in ["OK", "WATCH", "CONCERN", "DORMANT", "NO_TRADES", "MISSING"]:
        lines.append(f"- **{status}**: {counts.get(status, 0)}")
    lines.append("")

    lines.append("## Per-Strategy Snapshot")
    lines.append("")
    lines.append(
        "| # | Slug | Status | Days | Entries | Actual CumRet | Expected CumRet "
        "| Z | Paper Sharpe | Current DD | Hit Rate | Last Signal |"
    )
    lines.append(
        "|---|------|--------|------|---------|---------------|"
        "-----------------|---|--------------|------------|----------|-------------|"
    )
    for s in snaps:
        last_sig = s.last_signal_date.isoformat() if s.last_signal_date else "-"
        lines.append(
            f"| {s.rank} | `{s.slug}` | {s.status} | {s.days_elapsed} "
            f"| {s.num_entries} | {_fmt_pct(s.actual_cum_return)} "
            f"| {_fmt_pct(s.expected_cum_return)} | {_fmt_float(s.z_score)} "
            f"| {_fmt_float(s.running_sharpe)} | {_fmt_pct(s.current_drawdown)} "
            f"| {_fmt_pct(s.hit_rate)} | {last_sig} |"
        )
    lines.append("")

    lines.append("## Backtest Reference")
    lines.append("")
    lines.append("| # | Slug | Alias | BT Sharpe | BT CAGR | BT MaxDD |")
    lines.append("|---|------|-------|-----------|---------|----------|")
    for s in snaps:
        lines.append(
            f"| {s.rank} | `{s.slug}` | {s.alias} | {s.bt_sharpe:.2f} "
            f"| {s.bt_cagr * 100:+.1f}% | {s.bt_maxdd * 100:.1f}% |"
        )
    lines.append("")

    flagged = [
        s for s in snaps if s.status in {"CONCERN", "WATCH", "DORMANT", "NO_TRADES"}
    ]
    if flagged:
        lines.append("## Flagged Strategies")
        lines.append("")
        for s in flagged:
            lines.append(f"### [{s.status}] {s.alias} (`{s.slug}`)")
            lines.append("")
            lines.append(f"- Days elapsed: {s.days_elapsed}")
            lines.append(f"- Entries logged: {s.num_entries}")
            lines.append(f"- Actual cum return: {_fmt_pct(s.actual_cum_return)}")
            lines.append(
                f"- Expected cum return (proportional): {_fmt_pct(s.expected_cum_return)}"
            )
            lines.append(f"- Z-score: {_fmt_float(s.z_score)}")
            lines.append(
                f"- Current DD: {_fmt_pct(s.current_drawdown)} "
                f"(BT MaxDD {s.bt_maxdd * 100:.1f}%)"
            )
            for note in s.notes:
                lines.append(f"- Note: {note}")
            lines.append("")

    missing = [s for s in snaps if s.status == "MISSING"]
    if missing:
        lines.append("## Missing / Not Initialized for Paper Trading")
        lines.append("")
        lines.append("| # | Slug | Alias |")
        lines.append("|---|------|-------|")
        for s in missing:
            lines.append(f"| {s.rank} | `{s.slug}` | {s.alias} |")
        lines.append("")
        lines.append(
            "_These Track D strategies have passed all 6 robustness gates but do "
            "not yet have `paper-trading.yaml` files. They need to be wired into "
            "`scripts/run_paper_batch.py` dispatch tables and initialized before "
            "their paper track records can be validated._"
        )
        lines.append("")

    live = [s for s in snaps if s.has_paper_file and s.num_entries > 0]
    lines.append("## Top / Bottom Performers")
    lines.append("")
    if live:
        live_sorted = sorted(live, key=lambda s: s.actual_cum_return, reverse=True)
        lines.append("### Top 3 by actual cumulative return")
        lines.append("")
        lines.append("| Rank | Slug | Actual CumRet | Z | Days |")
        lines.append("|------|------|---------------|---|------|")
        for s in live_sorted[:3]:
            lines.append(
                f"| {s.rank} | `{s.slug}` | {_fmt_pct(s.actual_cum_return)} "
                f"| {_fmt_float(s.z_score)} | {s.days_elapsed} |"
            )
        lines.append("")
        lines.append("### Bottom 3 by actual cumulative return")
        lines.append("")
        lines.append("| Rank | Slug | Actual CumRet | Z | Days |")
        lines.append("|------|------|---------------|---|------|")
        for s in live_sorted[-3:][::-1]:
            lines.append(
                f"| {s.rank} | `{s.slug}` | {_fmt_pct(s.actual_cum_return)} "
                f"| {_fmt_float(s.z_score)} | {s.days_elapsed} |"
            )
        lines.append("")
    else:
        lines.append(
            "_No Track D strategies currently have live paper-trading entries. "
            "`tlt-tqqq-sprint` has a paper-trading.yaml with empty daily_log; "
            "the remaining 13 have not been wired into the batch runner yet._"
        )
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- **days_elapsed**: calendar days between `start_date` in "
        "paper-trading.yaml and as-of date (2026-04-17)"
    )
    lines.append(
        "- **Actual CumRet**: `(last_nav / initial_nav) - 1` from `daily_log` NAV"
    )
    lines.append(
        "- **Expected CumRet**: `(1 + bt_cagr)^(days_elapsed/365.25) - 1` — "
        "proportional scaling of backtest CAGR over elapsed window"
    )
    lines.append(
        "- **Z-score**: `(actual - expected) / (annual_vol * sqrt(years))`, "
        "annual_vol = `bt_cagr / bt_sharpe`"
    )
    lines.append(
        "- **Paper Sharpe**: annualized Sharpe from daily_return series "
        "(requires >=5 days with non-zero variance)"
    )
    lines.append("- **Flag rules**:")
    lines.append(
        "  - CONCERN: z-score < -2.0 (actual > 2 sigma below proportional expected)"
    )
    lines.append(
        "  - WATCH: current drawdown > 30% of backtest MaxDD (and > 2% absolute)"
    )
    lines.append("  - DORMANT: last signal date > 5 calendar days before as-of date")
    lines.append("  - NO_TRADES: paper-trading.yaml exists but daily_log is empty")
    lines.append("  - MISSING: no paper-trading.yaml file")
    lines.append("")

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track D batch paper-performance monitor"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output markdown path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the rendered markdown to stdout",
    )
    args = parser.parse_args()

    snaps = [_build_snapshot(entry) for entry in TRACK_D_REGISTRY]
    md = _render_markdown(snaps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")

    counts: dict[str, int] = {}
    for s in snaps:
        counts[s.status] = counts.get(s.status, 0) + 1
    print(f"Track D batch monitor — as of {AS_OF_DATE.isoformat()}")
    for status in ["OK", "WATCH", "CONCERN", "DORMANT", "NO_TRADES", "MISSING"]:
        print(f"  {status:10s} {counts.get(status, 0)}")
    print(f"Report: {args.out}")

    if args.stdout:
        print()
        print(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())

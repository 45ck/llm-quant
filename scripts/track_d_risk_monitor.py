#!/usr/bin/env python3
"""Track D daily risk monitor for leveraged ETF positions.

Monitors Track D (Sprint Alpha) strategies for:
  1. Drawdown vs -40% kill switch
  2. Volatility decay drag (TQQQ vs 3x QQQ)
  3. Signal-instrument correlation (TLT vs TQQQ)
  4. Holding period violations (max 5 calendar days)

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_risk_monitor.py
    cd E:/llm-quant && PYTHONPATH=src python scripts/track_d_risk_monitor.py --track-d-only
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

from llm_quant.data.fetcher import fetch_ohlcv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"
DRAWDOWN_KILL_SWITCH = 0.40  # -40% drawdown halts Track D
VOL_DECAY_ALERT_THRESHOLD = 0.05  # 5% drag over 30 days
CORRELATION_FLOOR = 0.50  # Min rolling correlation
MAX_HOLD_DAYS = 5  # Track D mandate
MIN_DAYS_FOR_CHECKS = 5  # Minimum data points for meaningful analysis
ROLLING_CORR_WINDOW = 20  # 20-day rolling correlation window

# Track D strategy slugs — extend as new strategies are added
TRACK_D_STRATEGIES = [
    {
        "slug": "tlt-tqqq-leveraged-lead-lag",
        "signal_symbol": "TLT",
        "follower_symbol": "TQQQ",
        "underlying_symbol": "QQQ",
        "leverage_factor": 3,
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DrawdownResult:
    """Result of drawdown check."""

    current_drawdown: float  # positive decimal (e.g. 0.05 = 5%)
    peak_nav: float
    current_nav: float
    kill_switch_triggered: bool
    status: str  # PASS / WARNING / HALT


@dataclass
class VolDecayResult:
    """Result of volatility decay check."""

    cumulative_follower_return: float  # TQQQ actual
    cumulative_leveraged_underlying_return: float  # 3x QQQ theoretical
    drag_pct: float  # divergence as decimal
    alert_triggered: bool
    days_measured: int
    status: str


@dataclass
class CorrelationResult:
    """Result of signal-instrument correlation check."""

    current_correlation: float | None
    window_days: int
    alert_triggered: bool
    status: str


@dataclass
class HoldingPeriodResult:
    """Result of holding period check."""

    max_consecutive_days: int
    current_streak_days: int
    current_direction: str | None  # "long" / "short" / None
    violation: bool
    status: str


@dataclass
class RiskMonitorSummary:
    """Aggregate risk monitor output for a single strategy."""

    slug: str
    date: datetime.date
    drawdown: DrawdownResult
    vol_decay: VolDecayResult
    correlation: CorrelationResult
    holding_period: HoldingPeriodResult
    overall_status: str  # PASS / WARNING / HALT
    notes: list[str]


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def load_paper_trading_yaml(slug: str) -> dict | None:
    """Load paper-trading.yaml for a given strategy slug.

    Returns None if file doesn't exist or can't be parsed.
    """
    yaml_path = DATA_DIR / slug / "paper-trading.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def compute_drawdown(daily_log: list[dict], performance: dict) -> DrawdownResult:
    """Compute current drawdown from paper trading data.

    Uses the NAV series from daily_log entries to compute drawdown
    from peak. Falls back to performance dict if log is sparse.
    """
    # Extract NAV series from daily log
    navs: list[float] = []
    for entry in daily_log:
        nav = entry.get("nav")
        if nav is not None:
            try:
                navs.append(float(nav))
            except (TypeError, ValueError):
                continue

    if not navs:
        # Fall back to performance dict
        current_nav = float(performance.get("current_nav", 100000.0))
        peak_nav = float(performance.get("peak_nav", 100000.0))
    else:
        current_nav = navs[-1]
        peak_nav = max(navs)

    if peak_nav <= 0:
        return DrawdownResult(
            current_drawdown=0.0,
            peak_nav=peak_nav,
            current_nav=current_nav,
            kill_switch_triggered=False,
            status="PASS",
        )

    dd = (peak_nav - current_nav) / peak_nav
    dd = max(dd, 0.0)  # Clamp to non-negative

    kill = dd >= DRAWDOWN_KILL_SWITCH
    if kill:
        status = "HALT"
    elif dd >= DRAWDOWN_KILL_SWITCH * 0.75:  # Warning at 75% of kill level (30%)
        status = "WARNING"
    else:
        status = "PASS"

    return DrawdownResult(
        current_drawdown=dd,
        peak_nav=peak_nav,
        current_nav=current_nav,
        kill_switch_triggered=kill,
        status=status,
    )


def compute_vol_decay(
    follower_prices: list[float],
    underlying_prices: list[float],
    leverage_factor: int = 3,
) -> VolDecayResult:
    """Compare cumulative leveraged ETF return vs leverage * underlying return.

    Parameters
    ----------
    follower_prices:
        Daily close prices for the leveraged ETF (e.g. TQQQ).
    underlying_prices:
        Daily close prices for the underlying (e.g. QQQ).
        Must be same length and aligned by date.
    leverage_factor:
        The leverage multiplier (e.g. 3 for TQQQ).

    Returns
    -------
    VolDecayResult with drag calculation.
    """
    if len(follower_prices) < 2 or len(underlying_prices) < 2:
        return VolDecayResult(
            cumulative_follower_return=0.0,
            cumulative_leveraged_underlying_return=0.0,
            drag_pct=0.0,
            alert_triggered=False,
            days_measured=0,
            status="INSUFFICIENT_DATA",
        )

    n = min(len(follower_prices), len(underlying_prices))
    follower_prices = follower_prices[:n]
    underlying_prices = underlying_prices[:n]

    # Cumulative return of the leveraged ETF
    cum_follower = follower_prices[-1] / follower_prices[0] - 1.0

    # Theoretical: leverage_factor * cumulative return of underlying
    cum_underlying = underlying_prices[-1] / underlying_prices[0] - 1.0
    cum_leveraged_theoretical = leverage_factor * cum_underlying

    # Drag = theoretical - actual (positive means decay hurt performance)
    drag = cum_leveraged_theoretical - cum_follower

    # Alert if drag exceeds threshold over 30+ day windows
    alert = abs(drag) >= VOL_DECAY_ALERT_THRESHOLD and n >= 30

    if alert:
        status = "WARNING"
    else:
        status = "PASS"

    return VolDecayResult(
        cumulative_follower_return=round(cum_follower, 6),
        cumulative_leveraged_underlying_return=round(cum_leveraged_theoretical, 6),
        drag_pct=round(drag, 6),
        alert_triggered=alert,
        days_measured=n,
        status=status,
    )


def compute_rolling_correlation(
    signal_returns: list[float],
    follower_returns: list[float],
    window: int = ROLLING_CORR_WINDOW,
) -> CorrelationResult:
    """Compute rolling correlation between signal and follower returns.

    Parameters
    ----------
    signal_returns:
        Daily returns of the signal source (e.g. TLT).
    follower_returns:
        Daily returns of the follower instrument (e.g. TQQQ).
    window:
        Rolling window size (default 20 days).

    Returns
    -------
    CorrelationResult with the most recent rolling correlation value.
    """
    if len(signal_returns) < window or len(follower_returns) < window:
        return CorrelationResult(
            current_correlation=None,
            window_days=window,
            alert_triggered=False,
            status="INSUFFICIENT_DATA",
        )

    n = min(len(signal_returns), len(follower_returns))
    sig = signal_returns[n - window : n]
    fol = follower_returns[n - window : n]

    corr = _pearson_correlation(sig, fol)

    if corr is None:
        return CorrelationResult(
            current_correlation=None,
            window_days=window,
            alert_triggered=False,
            status="INSUFFICIENT_DATA",
        )

    alert = corr < CORRELATION_FLOOR

    if alert:
        status = "WARNING"
    else:
        status = "PASS"

    return CorrelationResult(
        current_correlation=round(corr, 4),
        window_days=window,
        alert_triggered=alert,
        status=status,
    )


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation between two equal-length sequences.

    Returns None if computation is not possible (e.g. zero variance).
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n

    if var_x <= 0 or var_y <= 0:
        return None

    return cov / math.sqrt(var_x * var_y)


def check_holding_period(daily_log: list[dict]) -> HoldingPeriodResult:
    """Check for holding period violations in daily log.

    A position held in the same direction for more than MAX_HOLD_DAYS
    consecutive calendar days is a violation of the Track D mandate.

    The function looks at the 'position' field in each daily log entry.
    Consecutive entries with the same non-neutral position direction
    (long or short) count as a holding streak.
    """
    if not daily_log:
        return HoldingPeriodResult(
            max_consecutive_days=0,
            current_streak_days=0,
            current_direction=None,
            violation=False,
            status="PASS",
        )

    # Classify each entry's position direction
    directions: list[str | None] = []
    for entry in daily_log:
        pos = entry.get("position", "")
        direction = _classify_position_direction(str(pos))
        directions.append(direction)

    # Find streaks
    max_streak = 0
    current_streak = 0
    current_dir: str | None = None
    last_dir: str | None = None

    for d in directions:
        if d is None:
            # No position -- reset streak
            max_streak = max(max_streak, current_streak)
            current_streak = 0
            last_dir = None
        elif d == last_dir:
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 1
            last_dir = d

    # Final streak
    max_streak = max(max_streak, current_streak)
    current_dir = last_dir

    violation = max_streak > MAX_HOLD_DAYS

    if violation:
        status = "WARNING"
    else:
        status = "PASS"

    return HoldingPeriodResult(
        max_consecutive_days=max_streak,
        current_streak_days=current_streak,
        current_direction=current_dir,
        violation=violation,
        status=status,
    )


def _classify_position_direction(position: str) -> str | None:
    """Classify a position string into a direction.

    Returns 'long', 'short', or None (for neutral/flat/hold_prev with no position).
    """
    pos_lower = position.lower().strip()

    # Explicit long signals
    if pos_lower in ("long", "buy", "enter_long"):
        return "long"
    # Explicit short signals
    if pos_lower in ("short", "sell", "enter_short"):
        return "short"
    # Hold previous position — counts as continuing the streak
    if pos_lower in ("hold_prev", "hold"):
        return "hold"
    # Flat/neutral/exit — no position
    if pos_lower in ("flat", "neutral", "exit", "close", ""):
        return None

    # Default: if it contains "long" -> long, "short" -> short
    if "long" in pos_lower:
        return "long"
    if "short" in pos_lower:
        return "short"

    return None


def prices_to_returns(prices: list[float]) -> list[float]:
    """Convert a price series to simple daily returns.

    Returns a list of length len(prices) - 1.
    """
    if len(prices) < 2:
        return []
    return [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices))]


# ---------------------------------------------------------------------------
# Market data fetching
# ---------------------------------------------------------------------------


def fetch_market_prices(
    symbols: list[str],
    lookback_days: int = 90,
) -> dict[str, list[float]]:
    """Fetch daily close prices for symbols via yfinance.

    Returns a dict mapping symbol -> list of close prices (sorted by date).
    """
    import polars as pl

    df = fetch_ohlcv(symbols=symbols, lookback_days=lookback_days)
    if df is None or len(df) == 0:
        return {}

    result: dict[str, list[float]] = {}
    for sym in symbols:
        sym_df = df.filter(pl.col("symbol") == sym).sort("date")
        if len(sym_df) > 0:
            result[sym] = sym_df["close"].to_list()
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_strategy_monitor(
    strategy_config: dict,
    market_prices: dict[str, list[float]],
) -> RiskMonitorSummary:
    """Run all risk checks for a single Track D strategy.

    Parameters
    ----------
    strategy_config:
        Dict with keys: slug, signal_symbol, follower_symbol,
        underlying_symbol, leverage_factor.
    market_prices:
        Dict mapping symbol -> list of daily close prices.
    """
    slug = strategy_config["slug"]
    signal_sym = strategy_config["signal_symbol"]
    follower_sym = strategy_config["follower_symbol"]
    underlying_sym = strategy_config["underlying_symbol"]
    leverage = strategy_config["leverage_factor"]

    notes: list[str] = []
    today = datetime.date.today()

    # Load paper trading data
    pt_data = load_paper_trading_yaml(slug)
    if pt_data is None:
        return RiskMonitorSummary(
            slug=slug,
            date=today,
            drawdown=DrawdownResult(0, 0, 0, False, "NO_DATA"),
            vol_decay=VolDecayResult(0, 0, 0, False, 0, "NO_DATA"),
            correlation=CorrelationResult(None, ROLLING_CORR_WINDOW, False, "NO_DATA"),
            holding_period=HoldingPeriodResult(0, 0, None, False, "NO_DATA"),
            overall_status="NO_DATA",
            notes=[f"No paper-trading.yaml found for {slug}"],
        )

    daily_log = pt_data.get("daily_log", [])
    performance = pt_data.get("performance", {})

    # 1. Drawdown check
    dd_result = compute_drawdown(daily_log, performance)
    if dd_result.status != "PASS":
        notes.append(
            f"Drawdown: {dd_result.current_drawdown:.1%} "
            f"(kill switch at {DRAWDOWN_KILL_SWITCH:.0%})"
        )

    # 2. Vol decay check
    follower_prices = market_prices.get(follower_sym, [])
    underlying_prices = market_prices.get(underlying_sym, [])
    vd_result = compute_vol_decay(follower_prices, underlying_prices, leverage)
    if vd_result.status == "WARNING":
        notes.append(
            f"Vol decay drag: {vd_result.drag_pct:.2%} over {vd_result.days_measured} days"
        )

    # 3. Correlation check
    signal_prices = market_prices.get(signal_sym, [])
    follower_prices_for_corr = market_prices.get(follower_sym, [])
    signal_returns = prices_to_returns(signal_prices)
    follower_returns = prices_to_returns(follower_prices_for_corr)
    corr_result = compute_rolling_correlation(signal_returns, follower_returns)
    if corr_result.status == "WARNING":
        notes.append(
            f"Signal-instrument correlation: {corr_result.current_correlation:.3f} "
            f"(floor: {CORRELATION_FLOOR:.2f})"
        )

    # 4. Holding period check
    hp_result = check_holding_period(daily_log)
    if hp_result.violation:
        notes.append(
            f"Holding period violation: {hp_result.max_consecutive_days} days "
            f"(max: {MAX_HOLD_DAYS})"
        )

    # Determine overall status
    statuses = [
        dd_result.status,
        vd_result.status,
        corr_result.status,
        hp_result.status,
    ]
    if "HALT" in statuses:
        overall = "HALT"
    elif "WARNING" in statuses:
        overall = "WARNING"
    elif all(s in ("PASS", "INSUFFICIENT_DATA", "NO_DATA") for s in statuses):
        # Count insufficient data checks
        insufficient = sum(1 for s in statuses if s == "INSUFFICIENT_DATA")
        if insufficient >= 3:
            overall = "INSUFFICIENT_DATA"
        else:
            overall = "PASS"
    else:
        overall = "PASS"

    return RiskMonitorSummary(
        slug=slug,
        date=today,
        drawdown=dd_result,
        vol_decay=vd_result,
        correlation=corr_result,
        holding_period=hp_result,
        overall_status=overall,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


def format_summary(summary: RiskMonitorSummary) -> str:
    """Format a risk monitor summary as readable text output."""
    lines: list[str] = []
    sep = "-" * 65

    lines.append(sep)
    lines.append(f"  TRACK D RISK MONITOR -- {summary.slug}")
    lines.append(f"  Date: {summary.date}")
    lines.append(sep)
    lines.append("")

    # 1. Drawdown
    dd = summary.drawdown
    lines.append("  [1] DRAWDOWN CHECK")
    lines.append(f"      Current drawdown:   {dd.current_drawdown:>8.2%}")
    lines.append(f"      Kill switch level:  {DRAWDOWN_KILL_SWITCH:>8.0%}")
    lines.append(f"      Peak NAV:           {dd.peak_nav:>12,.2f}")
    lines.append(f"      Current NAV:        {dd.current_nav:>12,.2f}")
    lines.append(f"      Status:             {dd.status}")
    lines.append("")

    # 2. Vol decay
    vd = summary.vol_decay
    lines.append("  [2] VOLATILITY DECAY TRACKER")
    if vd.status in {"INSUFFICIENT_DATA", "NO_DATA"}:
        lines.append(f"      Status:             {vd.status}")
        lines.append(f"      Days measured:      {vd.days_measured}")
    else:
        lines.append(
            f"      Follower return:    {vd.cumulative_follower_return:>+8.2%}"
        )
        lines.append(
            f"      {summary.slug.split('-')[0].upper()} x"
            f"{TRACK_D_STRATEGIES[0]['leverage_factor']}"
            f" theoretical: {vd.cumulative_leveraged_underlying_return:>+8.2%}"
        )
        lines.append(f"      Vol decay drag:     {vd.drag_pct:>+8.2%}")
        lines.append(
            f"      Alert threshold:    {VOL_DECAY_ALERT_THRESHOLD:>8.0%} over 30d"
        )
        lines.append(f"      Days measured:      {vd.days_measured:>8d}")
        lines.append(f"      Status:             {vd.status}")
    lines.append("")

    # 3. Correlation
    corr = summary.correlation
    lines.append("  [3] SIGNAL-INSTRUMENT CORRELATION")
    if corr.status in {"INSUFFICIENT_DATA", "NO_DATA"}:
        lines.append(f"      Status:             {corr.status}")
        lines.append(
            f"      Need {corr.window_days} days of data for correlation check"
        )
    else:
        corr_val = (
            corr.current_correlation if corr.current_correlation is not None else 0
        )
        lines.append(f"      Rolling {corr.window_days}d corr:  {corr_val:>8.4f}")
        lines.append(f"      Floor:              {CORRELATION_FLOOR:>8.2f}")
        lines.append(f"      Status:             {corr.status}")
    lines.append("")

    # 4. Holding period
    hp = summary.holding_period
    lines.append("  [4] HOLDING PERIOD CHECK")
    lines.append(f"      Max consecutive:    {hp.max_consecutive_days:>8d} days")
    lines.append(f"      Current streak:     {hp.current_streak_days:>8d} days")
    lines.append(f"      Current direction:  {hp.current_direction or 'none':>8s}")
    lines.append(f"      Max allowed:        {MAX_HOLD_DAYS:>8d} days")
    lines.append(f"      Status:             {hp.status}")
    lines.append("")

    # Overall
    lines.append(sep)
    lines.append(f"  OVERALL STATUS: {summary.overall_status}")
    if summary.notes:
        lines.append("")
        lines.append("  Notes:")
        for note in summary.notes:
            lines.append(f"    - {note}")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track D daily risk monitor for leveraged ETF positions"
    )
    parser.add_argument(
        "--track-d-only",
        action="store_true",
        default=True,
        help="Monitor Track D strategies only (default: True)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,
        help="Lookback days for market data (default: 90)",
    )
    _args = parser.parse_args()

    print(f"Track D Risk Monitor -- {datetime.date.today()}")
    print(f"{'=' * 65}")
    print()

    # Collect all symbols needed
    all_symbols: list[str] = []
    for cfg in TRACK_D_STRATEGIES:
        for key in ("signal_symbol", "follower_symbol", "underlying_symbol"):
            sym = cfg[key]
            if sym not in all_symbols:
                all_symbols.append(sym)

    # Fetch market data
    print(f"Fetching market data for: {', '.join(all_symbols)}")
    market_prices = fetch_market_prices(all_symbols, lookback_days=_args.lookback)
    print(f"Received data for: {', '.join(market_prices.keys()) or 'none'}")
    print()

    # Run checks for each strategy
    has_halt = False
    for cfg in TRACK_D_STRATEGIES:
        summary = run_strategy_monitor(cfg, market_prices)
        print(format_summary(summary))
        if summary.overall_status == "HALT":
            has_halt = True

    if has_halt:
        print("*** HALT TRIGGERED -- Track D trading must stop. Sells only. ***")
        sys.exit(1)


if __name__ == "__main__":
    main()

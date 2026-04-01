"""Track D surveillance detectors — leveraged ETF position risk monitors.

Track D (Leveraged ETF Alpha) uses 3x leveraged products (TQQQ, UPRO, SOXL, TMF)
that carry three structural risks absent from standard equity strategies:

1. Beta decay / volatility drag — 3x ETFs lose to variance every day they are held
2. Path dependency — daily reset amplifies compounding losses in volatile regimes
3. Hard 5-day hold limit — positions held beyond 5 calendar days must be force-exited

Each detector follows the same interface as all other surveillance detectors:
    fn(conn, config) -> list[SurveillanceCheck]

The ``TrackDMonitor`` class is a higher-level wrapper for the ``/trade`` and
``/governance`` commands.  Individual detector functions are also usable directly
from ``scanner.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import duckdb

from llm_quant.config import AppConfig
from llm_quant.surveillance.models import SeverityLevel, SurveillanceCheck

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVERAGED_ETFS: frozenset[str] = frozenset(
    {"TQQQ", "UPRO", "SOXL", "TMF", "TLTW", "UVXY"}
)
MAX_HOLD_DAYS = 5  # Track D hard rule — force exit on day 5+
WARN_HOLD_DAYS = 4  # Warn when approaching limit
VIX_HIGH_THRESHOLD = 30.0  # VIX level that dramatically amplifies decay
VIX_ELEVATED_THRESHOLD = 25.0  # Elevated but not extreme
# Approximate annualised drag coefficient for a 3x daily-reset ETF.
# Source: track-d-rebalancing.md formula  decay ≈ 4.5 × σ²
_DECAY_COEFF = 4.5


# ---------------------------------------------------------------------------
# Structured report dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrackDRiskReport:
    """Full daily risk snapshot for Track D leveraged ETF positions."""

    date: date
    positions_at_risk: list[str] = field(default_factory=list)  # day 4 — WARNING
    forced_exits: list[str] = field(default_factory=list)  # day 5+ — HALT
    beta_decay_estimate: dict[str, float] = field(default_factory=dict)
    volatility_drag: dict[str, float] = field(default_factory=dict)
    daily_rebalance_needed: bool = False
    warnings: list[str] = field(default_factory=list)
    halt: bool = False

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "positions_at_risk": self.positions_at_risk,
            "forced_exits": self.forced_exits,
            "beta_decay_estimate": self.beta_decay_estimate,
            "volatility_drag": self.volatility_drag,
            "daily_rebalance_needed": self.daily_rebalance_needed,
            "warnings": self.warnings,
            "halt": self.halt,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_current_vix(conn: duckdb.DuckDBPyConnection) -> float | None:
    """Return the most recent VIX close price from market_data_daily, or None."""
    row = conn.execute(
        """
        SELECT close
        FROM market_data_daily
        WHERE symbol = 'VIX'
        ORDER BY date DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    return None


def _get_leveraged_positions(
    conn: duckdb.DuckDBPyConnection,
) -> list[str]:
    """Return symbols of currently held leveraged ETF positions (latest snapshot)."""
    latest = conn.execute(
        """
        SELECT snapshot_id
        FROM portfolio_snapshots
        ORDER BY date DESC, snapshot_id DESC
        LIMIT 1
        """
    ).fetchone()
    if not latest:
        return []
    snapshot_id = latest[0]

    rows = conn.execute(
        """
        SELECT symbol
        FROM positions
        WHERE snapshot_id = ?
          AND shares > 0
        """,
        [snapshot_id],
    ).fetchall()
    return [r[0] for r in rows if r[0] in LEVERAGED_ETFS]


def _hold_days_from_trades(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
) -> int:
    """Estimate calendar days the current open position in ``symbol`` has been held.

    Strategy: find the most recent BUY trade for this symbol that has not been
    offset by a subsequent SELL/CLOSE trade of equal or larger magnitude.  This is
    a simplified heuristic — for paper trading purposes it is sufficient.

    Returns 0 if no open BUY can be determined.
    """
    # Walk trades for symbol in chronological order, track net shares
    trade_rows = conn.execute(
        """
        SELECT date, action, shares
        FROM trades
        WHERE symbol = ?
        ORDER BY date ASC, trade_id ASC
        """,
        [symbol],
    ).fetchall()

    if not trade_rows:
        return 0

    # Track net open position and the date the current long was opened
    net_shares: float = 0.0
    last_entry_date: date | None = None

    for row in trade_rows:
        trade_date, action, shares = row
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)
        action_upper = action.upper()

        if action_upper in ("BUY", "LONG"):
            if net_shares <= 0:
                # Opening or reversing — record entry date
                last_entry_date = trade_date
            net_shares += shares
        elif action_upper in ("SELL", "SHORT", "CLOSE"):
            net_shares -= shares
            if net_shares <= 0:
                last_entry_date = None  # Position fully closed

    if net_shares <= 0 or last_entry_date is None:
        return 0

    today = datetime.now(tz=UTC).date()
    return (today - last_entry_date).days


def _estimate_beta_decay(days_held: int, vix: float, leverage: float = 3.0) -> float:
    """Estimate cumulative beta decay as a fraction of notional.

    Formula from track-d-rebalancing.md:
        daily_variance ≈ (vix / 100 / sqrt(252))^2
        daily_drag = leverage^2 / 2 × daily_variance
        cumulative_drag ≈ days × daily_drag

    At VIX=20, 3x ETF over 5 days → ~0.5% drag (consistent with doc reference).
    At VIX=30, drag roughly doubles.

    Returns a positive float representing fractional loss (e.g. 0.005 = 0.5%).
    """
    daily_vol = (vix / 100.0) / math.sqrt(252)
    daily_variance = daily_vol**2
    # For a leverage-times daily-reset ETF:
    #   rebalancing drag per day = (L^2 - L) / 2 × σ²  (exact for continuous time)
    daily_drag = (leverage**2 - leverage) / 2.0 * daily_variance
    return daily_drag * days_held


def _estimate_volatility_drag(vix: float, leverage: float = 3.0) -> float:
    """Estimate annualised volatility drag as a fraction.

    Returns annualised drag (e.g. 0.208 = 20.8% per year).
    """
    daily_vol = (vix / 100.0) / math.sqrt(252)
    daily_variance = daily_vol**2
    daily_drag = (leverage**2 - leverage) / 2.0 * daily_variance
    return daily_drag * 252.0


# ---------------------------------------------------------------------------
# Detector functions (scanner.py interface)
# ---------------------------------------------------------------------------


def check_track_d_hold_periods(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Detect Track D leveraged ETF positions approaching or exceeding 5-day limit.

    WARNING at day 4 (approaching limit).
    HALT at day 5+ (must force-exit).
    """
    checks: list[SurveillanceCheck] = []

    try:
        held_symbols = _get_leveraged_positions(conn)
    except Exception:
        checks.append(
            SurveillanceCheck(
                detector="track_d_hold_periods",
                severity=SeverityLevel.OK,
                message="No portfolio snapshots — no Track D positions to check.",
                metric_name="track_d_max_hold_days",
            )
        )
        return checks

    if not held_symbols:
        checks.append(
            SurveillanceCheck(
                detector="track_d_hold_periods",
                severity=SeverityLevel.OK,
                message="No leveraged ETF positions currently held.",
                metric_name="track_d_max_hold_days",
                current_value=0.0,
                threshold_value=float(MAX_HOLD_DAYS),
            )
        )
        return checks

    worst_days = 0

    for symbol in held_symbols:
        days = _hold_days_from_trades(conn, symbol)
        worst_days = max(worst_days, days)

        if days >= MAX_HOLD_DAYS:
            checks.append(
                SurveillanceCheck(
                    detector="track_d_hold_periods",
                    severity=SeverityLevel.HALT,
                    message=(
                        f"KILL SWITCH: {symbol} held {days} calendar days — "
                        f"exceeds Track D 5-day limit. FORCE EXIT required."
                    ),
                    metric_name="track_d_hold_days",
                    current_value=float(days),
                    threshold_value=float(MAX_HOLD_DAYS),
                    details={"symbol": symbol, "days_held": days},
                )
            )
        elif days >= WARN_HOLD_DAYS:
            checks.append(
                SurveillanceCheck(
                    detector="track_d_hold_periods",
                    severity=SeverityLevel.WARNING,
                    message=(
                        f"{symbol} held {days} calendar days — "
                        f"approaching Track D 5-day limit. Plan exit."
                    ),
                    metric_name="track_d_hold_days",
                    current_value=float(days),
                    threshold_value=float(MAX_HOLD_DAYS),
                    details={"symbol": symbol, "days_held": days},
                )
            )

    # If all positions are well within limits, emit a single OK
    if not checks:
        symbols_summary = ", ".join(
            f"{s}({_hold_days_from_trades(conn, s)}d)" for s in held_symbols
        )
        checks.append(
            SurveillanceCheck(
                detector="track_d_hold_periods",
                severity=SeverityLevel.OK,
                message=(
                    f"Leveraged ETF hold periods within limit: {symbols_summary} "
                    f"(max {MAX_HOLD_DAYS} days)."
                ),
                metric_name="track_d_max_hold_days",
                current_value=float(worst_days),
                threshold_value=float(MAX_HOLD_DAYS),
            )
        )

    return checks


def check_track_d_vix_regime(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Detect high-VIX regime that amplifies leveraged ETF beta decay.

    WARNING if VIX > 25 (elevated decay — scale position sizes to 50%).
    HALT if VIX > 30 with any leveraged long positions (exit leveraged longs immediately,
    per track-d-rebalancing.md Tier 1 daily checks).
    """
    checks: list[SurveillanceCheck] = []

    vix = _get_current_vix(conn)

    if vix is None:
        checks.append(
            SurveillanceCheck(
                detector="track_d_vix_regime",
                severity=SeverityLevel.OK,
                message=(
                    "VIX data not available — "
                    "cannot evaluate leveraged ETF decay regime."
                ),
                metric_name="vix_level",
            )
        )
        return checks

    held_symbols = _get_leveraged_positions(conn)
    long_etfs = [s for s in held_symbols if s != "TMF"]  # TMF is the hedge leg

    if vix >= VIX_HIGH_THRESHOLD and long_etfs:
        checks.append(
            SurveillanceCheck(
                detector="track_d_vix_regime",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: VIX={vix:.1f} >= {VIX_HIGH_THRESHOLD:.0f} "
                    f"with leveraged long positions {long_etfs}. "
                    "Exit all leveraged longs immediately (Track D Tier 1 rule)."
                ),
                metric_name="vix_level",
                current_value=vix,
                threshold_value=VIX_HIGH_THRESHOLD,
                details={"vix": vix, "long_etfs": long_etfs},
            )
        )
    elif vix >= VIX_ELEVATED_THRESHOLD:
        checks.append(
            SurveillanceCheck(
                detector="track_d_vix_regime",
                severity=SeverityLevel.WARNING,
                message=(
                    f"VIX={vix:.1f} elevated (>= {VIX_ELEVATED_THRESHOLD:.0f}). "
                    "Scale Track D position sizes to 50% of normal. "
                    "Decay is approximately 2x baseline."
                ),
                metric_name="vix_level",
                current_value=vix,
                threshold_value=VIX_ELEVATED_THRESHOLD,
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_d_vix_regime",
                severity=SeverityLevel.OK,
                message=(
                    f"VIX={vix:.1f} — normal decay regime "
                    f"(warn >{VIX_ELEVATED_THRESHOLD:.0f}, "
                    f"halt >{VIX_HIGH_THRESHOLD:.0f} with longs)."
                ),
                metric_name="vix_level",
                current_value=vix,
                threshold_value=VIX_HIGH_THRESHOLD,
            )
        )

    return checks


def check_track_d_beta_decay(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Estimate cumulative beta decay on open Track D positions.

    WARNING if estimated decay > 1% on any position.
    HALT is not triggered by decay alone — only hold period and VIX checks halt.
    This detector surfaces decay magnitude to inform exit timing.
    """
    checks: list[SurveillanceCheck] = []

    held_symbols = _get_leveraged_positions(conn)
    if not held_symbols:
        checks.append(
            SurveillanceCheck(
                detector="track_d_beta_decay",
                severity=SeverityLevel.OK,
                message="No leveraged ETF positions — no decay to estimate.",
                metric_name="track_d_max_decay_pct",
                current_value=0.0,
            )
        )
        return checks

    vix = _get_current_vix(conn)
    effective_vix = vix if vix is not None else 20.0  # fallback to moderate assumption

    worst_decay = 0.0
    worst_symbol: str | None = None

    for symbol in held_symbols:
        days = _hold_days_from_trades(conn, symbol)
        if days <= 0:
            continue
        decay = _estimate_beta_decay(days, effective_vix)
        if decay > worst_decay:
            worst_decay = decay
            worst_symbol = symbol

    if worst_decay > 0.01:  # > 1% cumulative decay
        checks.append(
            SurveillanceCheck(
                detector="track_d_beta_decay",
                severity=SeverityLevel.WARNING,
                message=(
                    f"Estimated beta decay on {worst_symbol}: "
                    f"{worst_decay:.2%} cumulative "
                    f"(VIX={effective_vix:.1f}). "
                    "Consider timing exit."
                ),
                metric_name="track_d_max_decay_pct",
                current_value=worst_decay,
                threshold_value=0.01,
                details={
                    "vix": effective_vix,
                    "symbol": worst_symbol,
                    "estimated_decay_pct": round(worst_decay * 100, 3),
                },
            )
        )
    elif worst_decay > 0:
        checks.append(
            SurveillanceCheck(
                detector="track_d_beta_decay",
                severity=SeverityLevel.OK,
                message=(
                    f"Estimated beta decay on {worst_symbol}: "
                    f"{worst_decay:.3%} cumulative "
                    f"(VIX={effective_vix:.1f}) — within tolerance."
                ),
                metric_name="track_d_max_decay_pct",
                current_value=worst_decay,
                threshold_value=0.01,
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_d_beta_decay",
                severity=SeverityLevel.OK,
                message="No beta decay estimated — positions held < 1 day.",
                metric_name="track_d_max_decay_pct",
                current_value=0.0,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# TrackDMonitor — higher-level class for /governance and /trade integration
# ---------------------------------------------------------------------------


class TrackDMonitor:
    """Daily risk monitor for Track D leveraged ETF positions.

    Wraps the three detector functions and provides:
    - ``check_hold_periods()`` — dict of symbol → days held
    - ``estimate_beta_decay()`` — per-position decay estimate
    - ``run_daily_check()`` — full TrackDRiskReport
    - ``generate_forced_exit_signals()`` — SELL signals for day 5+ positions
    """

    def __init__(self, db_conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = db_conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_hold_periods(self) -> dict[str, int]:
        """Return dict of leveraged ETF symbol → calendar days held.

        Reads current positions and traces back to the most recent BUY trade
        to determine hold duration.
        """
        held_symbols = _get_leveraged_positions(self.conn)
        return {
            symbol: _hold_days_from_trades(self.conn, symbol) for symbol in held_symbols
        }

    def estimate_beta_decay(
        self,
        symbol: str,
        days_held: int,
        vix: float,
    ) -> float:
        """Estimate fractional beta decay for a single position.

        Args:
            symbol: ETF ticker (used to determine leverage multiple — all
                    currently supported tickers are 3x products).
            days_held: Calendar days the position has been open.
            vix: Current VIX level (annualised, e.g. 20.0).

        Returns:
            Estimated fractional decay loss, e.g. 0.005 for 0.5%.
        """
        # All currently tracked leveraged ETFs in LEVERAGED_ETFS are 3x products.
        # UVXY is ~1.5x but modelling it as 3x is conservatively pessimistic.
        return _estimate_beta_decay(days_held, vix, leverage=3.0)

    def run_daily_check(self) -> TrackDRiskReport:
        """Execute the full Track D daily risk check.

        Steps:
        1. Identify current leveraged ETF positions.
        2. Check hold period — flag day 4 (WARNING) and day 5+ (FORCE EXIT).
        3. Read VIX — flag if elevated (>25) or extreme (>30 with longs).
        4. Estimate cumulative beta decay per position.
        5. Return structured TrackDRiskReport.
        """
        today = datetime.now(tz=UTC).date()
        report = TrackDRiskReport(date=today)

        held_symbols = _get_leveraged_positions(self.conn)
        if not held_symbols:
            return report  # nothing to check

        vix = _get_current_vix(self.conn)
        effective_vix = vix if vix is not None else 20.0

        # 1. Hold period check
        hold_periods = self.check_hold_periods()

        for symbol, days in hold_periods.items():
            decay = self.estimate_beta_decay(symbol, days, effective_vix)
            report.beta_decay_estimate[symbol] = decay
            report.volatility_drag[symbol] = _estimate_volatility_drag(effective_vix)

            if days >= MAX_HOLD_DAYS:
                report.forced_exits.append(symbol)
                report.halt = True
                report.warnings.append(
                    f"{symbol}: {days}d hold exceeds 5-day limit — FORCE EXIT"
                )
                report.daily_rebalance_needed = True
            elif days >= WARN_HOLD_DAYS:
                report.positions_at_risk.append(symbol)
                report.warnings.append(
                    f"{symbol}: {days}d hold — exit by end of day tomorrow"
                )
                report.daily_rebalance_needed = True

        # 2. VIX regime check
        if vix is not None:
            long_etfs = [s for s in held_symbols if s != "TMF"]

            if vix >= VIX_HIGH_THRESHOLD and long_etfs:
                report.halt = True
                report.daily_rebalance_needed = True
                report.warnings.append(
                    f"VIX={vix:.1f} >= {VIX_HIGH_THRESHOLD:.0f} with leveraged longs "
                    f"{long_etfs} — exit immediately"
                )
                for symbol in long_etfs:
                    if symbol not in report.forced_exits:
                        report.forced_exits.append(symbol)
            elif vix >= VIX_ELEVATED_THRESHOLD:
                report.daily_rebalance_needed = True
                report.warnings.append(
                    f"VIX={vix:.1f} elevated — scale Track D sizes to 50% of normal"
                )

        # 3. High-decay warning (>1% cumulative)
        for symbol, decay in report.beta_decay_estimate.items():
            if decay > 0.01 and symbol not in report.positions_at_risk:
                report.warnings.append(
                    f"{symbol}: cumulative beta decay {decay:.2%} — consider exiting"
                )

        return report

    def generate_forced_exit_signals(self) -> list[dict]:
        """Return SELL signals for all leveraged ETF positions exceeding the 5-day limit.

        Returns a list of signal dicts compatible with execute_decision.py format:
        [{"symbol": "TQQQ", "action": "SELL", "conviction": "high",
          "reasoning": "Track D 5-day hold limit exceeded — forced exit"}]
        """
        signals: list[dict] = []
        hold_periods = self.check_hold_periods()

        vix = _get_current_vix(self.conn)
        long_etfs_with_high_vix: list[str] = []
        if vix is not None and vix >= VIX_HIGH_THRESHOLD:
            held_symbols = _get_leveraged_positions(self.conn)
            long_etfs_with_high_vix = [s for s in held_symbols if s != "TMF"]

        for symbol, days in hold_periods.items():
            if days >= MAX_HOLD_DAYS:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "conviction": "high",
                        "reasoning": (
                            f"Track D forced exit: {symbol} held {days} calendar days, "
                            f"exceeds 5-day maximum. Beta decay estimated at "
                            f"{self.estimate_beta_decay(symbol, days, vix or 20.0):.2%}."
                        ),
                    }
                )

        for symbol in long_etfs_with_high_vix:
            if symbol not in {s["symbol"] for s in signals}:
                signals.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "conviction": "high",
                        "reasoning": (
                            f"Track D VIX kill switch: VIX={vix:.1f} >= {VIX_HIGH_THRESHOLD:.0f}. "
                            "Exit all leveraged long positions immediately "
                            "(Track D Tier 1 daily rule)."
                        ),
                    }
                )

        return signals

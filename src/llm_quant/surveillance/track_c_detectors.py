"""Track C surveillance detectors — structural arbitrage kill-switch monitors.

Track C (Structural Arbitrage) covers PM arb, CEF discount capture, and
funding rate strategies.  These detectors guard against the specific failure
modes of market-neutral structural arb: exchange outages, spread compression,
funding reversals, and inadvertent beta exposure.

Each detector follows the same interface as existing detectors:
    fn(conn, config) -> list[SurveillanceCheck]

Configuration dataclasses provide sensible defaults matching the Track C
mandate thresholds.  The detector functions accept optional config overrides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import duckdb

from llm_quant.config import AppConfig
from llm_quant.surveillance.models import SeverityLevel, SurveillanceCheck

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExchangeHealthConfig:
    """Configuration for ExchangeHealthDetector.

    Attributes:
        error_threshold: Max consecutive API errors before HALT (default 3).
        lookback_hours: Window for counting errors (default 1 hour).
        withdrawal_delay_lookback_hours: Window for withdrawal delay events
            (default 24 hours).
    """

    error_threshold: int = 3
    lookback_hours: int = 1
    withdrawal_delay_lookback_hours: int = 24


@dataclass
class SpreadCompressionConfig:
    """Configuration for SpreadCompressionDetector.

    Attributes:
        halt_ratio: HALT when 7d avg spread drops below this fraction of 30d
            baseline (default 0.25 = 25%).
        warn_ratio: WARNING when 7d avg approaches compression (default 0.50).
        rolling_window_days: Short-term window for rolling average (default 7).
        baseline_window_days: Long-term window for baseline average (default 30).
    """

    halt_ratio: float = 0.25
    warn_ratio: float = 0.50
    rolling_window_days: int = 7
    baseline_window_days: int = 30


@dataclass
class FundingRateReversalConfig:
    """Configuration for FundingRateReversalDetector.

    Attributes:
        halt_consecutive: HALT after this many consecutive negative 8h funding
            periods (default 3).
        warn_consecutive: WARNING threshold (default 2).
    """

    halt_consecutive: int = 3
    warn_consecutive: int = 2


@dataclass
class BetaDriftConfig:
    """Configuration for BetaDriftDetector.

    Attributes:
        window_days: Rolling window for beta calculation (default 30).
        halt_beta: HALT when abs(beta) exceeds this value (default 0.15).
        warn_beta: WARNING threshold (default 0.10).
    """

    window_days: int = 30
    halt_beta: float = 0.15
    warn_beta: float = 0.10


@dataclass
class CrossTrackCorrelationConfig:
    """Configuration for CrossTrackCorrelationDetector.

    Attributes:
        window_days: Rolling window for correlation calc (default 30).
        halt_corr: HALT when correlation exceeds this (default 0.30).
        warn_corr: WARNING threshold (default 0.20).
    """

    window_days: int = 30
    halt_corr: float = 0.30
    warn_corr: float = 0.20


# ---------------------------------------------------------------------------
# Track C-1. Exchange Health — withdrawal delays and API error spikes
# ---------------------------------------------------------------------------


def check_exchange_health(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    detector_config: ExchangeHealthConfig | None = None,
) -> list[SurveillanceCheck]:
    """Detect exchange withdrawal delays or API error spikes.

    HALT if: >error_threshold consecutive API errors within lookback window,
    or any withdrawal delay event detected.
    Reads from track_c_exchange_events table if present; degrades gracefully
    if the table does not yet exist.
    """
    cfg = detector_config or ExchangeHealthConfig()
    checks: list[SurveillanceCheck] = []

    # Graceful degradation — table may not exist in early deployment
    try:
        table_exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'track_c_exchange_events'
            """
        ).fetchone()
        has_table = table_exists is not None and table_exists[0] > 0
    except Exception:
        has_table = False

    if not has_table:
        checks.append(
            SurveillanceCheck(
                detector="track_c_exchange_health",
                severity=SeverityLevel.OK,
                message=(
                    "Exchange health table not yet initialised "
                    "— no exchange events to evaluate."
                ),
                metric_name="exchange_api_errors",
            )
        )
        return checks

    cutoff = datetime.now(tz=UTC) - timedelta(hours=cfg.lookback_hours)

    # Count consecutive API errors (most recent streak)
    recent_events = conn.execute(
        """
        SELECT event_type, occurred_at, exchange, details
        FROM track_c_exchange_events
        WHERE occurred_at >= ?
        ORDER BY occurred_at DESC
        LIMIT 50
        """,
        [cutoff],
    ).fetchall()

    # Count leading consecutive api_error rows
    consecutive_errors = 0
    for row in recent_events:
        event_type = row[0]
        if event_type == "api_error":
            consecutive_errors += 1
        else:
            break

    if consecutive_errors > cfg.error_threshold:
        checks.append(
            SurveillanceCheck(
                detector="track_c_exchange_health",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: {consecutive_errors} consecutive API errors "
                    f"detected (limit {cfg.error_threshold})."
                ),
                metric_name="exchange_api_errors",
                current_value=float(consecutive_errors),
                threshold_value=float(cfg.error_threshold),
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_c_exchange_health",
                severity=SeverityLevel.OK,
                message=(
                    f"{consecutive_errors} consecutive API errors "
                    f"in last {cfg.lookback_hours}h "
                    f"— within limit ({cfg.error_threshold})."
                ),
                metric_name="exchange_api_errors",
                current_value=float(consecutive_errors),
                threshold_value=float(cfg.error_threshold),
            )
        )

    # Check for withdrawal delay events
    wd_cutoff = datetime.now(tz=UTC) - timedelta(
        hours=cfg.withdrawal_delay_lookback_hours,
    )
    withdrawal_delays = conn.execute(
        """
        SELECT COUNT(*)
        FROM track_c_exchange_events
        WHERE event_type = 'withdrawal_delay'
        AND occurred_at >= ?
        """,
        [wd_cutoff],
    ).fetchone()

    delay_count = withdrawal_delays[0] if withdrawal_delays else 0
    if delay_count > 0:
        checks.append(
            SurveillanceCheck(
                detector="track_c_exchange_health",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: {delay_count} withdrawal delay event(s) "
                    f"detected in last {cfg.withdrawal_delay_lookback_hours}h."
                ),
                metric_name="exchange_withdrawal_delays",
                current_value=float(delay_count),
                threshold_value=0.0,
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_c_exchange_health",
                severity=SeverityLevel.OK,
                message=(
                    "No withdrawal delays detected "
                    f"in last {cfg.withdrawal_delay_lookback_hours}h."
                ),
                metric_name="exchange_withdrawal_delays",
                current_value=0.0,
                threshold_value=0.0,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Track C-2. Spread Compression — arb edge disappearing
# ---------------------------------------------------------------------------


def check_spread_compression(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    detector_config: SpreadCompressionConfig | None = None,
) -> list[SurveillanceCheck]:
    """Detect structural arb edge compression.

    HALT if: rolling 7d average arb spread < 25% of 30d baseline.
    Reads from track_c_arb_spreads table if present.
    """
    cfg = detector_config or SpreadCompressionConfig()
    checks: list[SurveillanceCheck] = []

    try:
        table_exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'track_c_arb_spreads'
            """
        ).fetchone()
        has_table = table_exists is not None and table_exists[0] > 0
    except Exception:
        has_table = False

    if not has_table:
        checks.append(
            SurveillanceCheck(
                detector="track_c_spread_compression",
                severity=SeverityLevel.OK,
                message=(
                    "Arb spreads table not yet initialised "
                    "— no spread data to evaluate."
                ),
                metric_name="arb_spread_ratio",
            )
        )
        return checks

    now = datetime.now(tz=UTC)
    cutoff_rolling = now - timedelta(days=cfg.rolling_window_days)
    cutoff_baseline = now - timedelta(days=cfg.baseline_window_days)

    # Baseline average spread
    baseline_row = conn.execute(
        """
        SELECT AVG(spread_bps)
        FROM track_c_arb_spreads
        WHERE recorded_at >= ? AND spread_bps IS NOT NULL
        """,
        [cutoff_baseline],
    ).fetchone()

    baseline_avg = baseline_row[0] if baseline_row and baseline_row[0] else None

    if baseline_avg is None or baseline_avg <= 0:
        checks.append(
            SurveillanceCheck(
                detector="track_c_spread_compression",
                severity=SeverityLevel.OK,
                message=(
                    f"Insufficient {cfg.baseline_window_days}d spread history "
                    "for compression check."
                ),
                metric_name="arb_spread_ratio",
            )
        )
        return checks

    # Rolling average spread
    rolling_row = conn.execute(
        """
        SELECT AVG(spread_bps)
        FROM track_c_arb_spreads
        WHERE recorded_at >= ? AND spread_bps IS NOT NULL
        """,
        [cutoff_rolling],
    ).fetchone()

    rolling_avg = rolling_row[0] if rolling_row and rolling_row[0] else 0.0

    # Ratio of current to baseline
    spread_ratio = rolling_avg / baseline_avg if baseline_avg > 0 else 1.0

    if spread_ratio < cfg.halt_ratio:
        checks.append(
            SurveillanceCheck(
                detector="track_c_spread_compression",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: Rolling {cfg.rolling_window_days}d arb spread "
                    f"{rolling_avg:.1f}bps "
                    f"is {spread_ratio:.0%} of {cfg.baseline_window_days}d "
                    f"baseline {baseline_avg:.1f}bps "
                    f"(halt below {cfg.halt_ratio:.0%})."
                ),
                metric_name="arb_spread_ratio",
                current_value=spread_ratio,
                threshold_value=cfg.halt_ratio,
            )
        )
    elif spread_ratio < cfg.warn_ratio:
        checks.append(
            SurveillanceCheck(
                detector="track_c_spread_compression",
                severity=SeverityLevel.WARNING,
                message=(
                    f"Arb spread compressing: rolling {cfg.rolling_window_days}d "
                    f"{rolling_avg:.1f}bps "
                    f"is {spread_ratio:.0%} of {cfg.baseline_window_days}d "
                    f"baseline {baseline_avg:.1f}bps."
                ),
                metric_name="arb_spread_ratio",
                current_value=spread_ratio,
                threshold_value=cfg.warn_ratio,
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_c_spread_compression",
                severity=SeverityLevel.OK,
                message=(
                    f"Arb spread healthy: rolling {cfg.rolling_window_days}d "
                    f"{rolling_avg:.1f}bps "
                    f"({spread_ratio:.0%} of {cfg.baseline_window_days}d "
                    f"baseline {baseline_avg:.1f}bps)."
                ),
                metric_name="arb_spread_ratio",
                current_value=spread_ratio,
                threshold_value=cfg.halt_ratio,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Track C-3. Funding Rate Reversal — 3 consecutive negative 8h funding periods
# ---------------------------------------------------------------------------


def check_funding_rate_reversal(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    detector_config: FundingRateReversalConfig | None = None,
) -> list[SurveillanceCheck]:
    """Detect sustained negative funding regime on tracked perpetual symbols.

    HALT if: halt_consecutive or more consecutive negative funding rates on
    any tracked symbol.
    Reads from track_c_funding_rates table if present.
    """
    cfg = detector_config or FundingRateReversalConfig()
    checks: list[SurveillanceCheck] = []

    try:
        table_exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'track_c_funding_rates'
            """
        ).fetchone()
        has_table = table_exists is not None and table_exists[0] > 0
    except Exception:
        has_table = False

    if not has_table:
        checks.append(
            SurveillanceCheck(
                detector="track_c_funding_rate_reversal",
                severity=SeverityLevel.OK,
                message=(
                    "Funding rates table not yet initialised "
                    "— no funding data to evaluate."
                ),
                metric_name="consecutive_negative_funding",
            )
        )
        return checks

    # Get all tracked symbols
    symbols_row = conn.execute(
        "SELECT DISTINCT symbol FROM track_c_funding_rates"
    ).fetchall()
    symbols = [r[0] for r in symbols_row]

    if not symbols:
        checks.append(
            SurveillanceCheck(
                detector="track_c_funding_rate_reversal",
                severity=SeverityLevel.OK,
                message="No funding rate symbols tracked yet.",
                metric_name="consecutive_negative_funding",
            )
        )
        return checks

    worst_symbol: str | None = None
    worst_streak = 0

    for symbol in symbols:
        recent_rates = conn.execute(
            """
            SELECT funding_rate
            FROM track_c_funding_rates
            WHERE symbol = ?
            ORDER BY period_end DESC
            LIMIT 20
            """,
            [symbol],
        ).fetchall()

        consecutive_neg = 0
        for row in recent_rates:
            rate = row[0]
            if rate is not None and rate < 0:
                consecutive_neg += 1
            else:
                break

        if consecutive_neg > worst_streak:
            worst_streak = consecutive_neg
            worst_symbol = symbol

    if worst_streak >= cfg.halt_consecutive:
        checks.append(
            SurveillanceCheck(
                detector="track_c_funding_rate_reversal",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: {worst_streak} consecutive negative "
                    f"8h funding periods on {worst_symbol} "
                    f"(halt at {cfg.halt_consecutive})."
                ),
                metric_name="consecutive_negative_funding",
                current_value=float(worst_streak),
                threshold_value=float(cfg.halt_consecutive),
                details={"symbol": worst_symbol},
            )
        )
    elif worst_streak >= cfg.warn_consecutive:
        checks.append(
            SurveillanceCheck(
                detector="track_c_funding_rate_reversal",
                severity=SeverityLevel.WARNING,
                message=(
                    f"{worst_streak} consecutive negative funding periods "
                    f"on {worst_symbol} — approaching halt threshold "
                    f"({cfg.halt_consecutive})."
                ),
                metric_name="consecutive_negative_funding",
                current_value=float(worst_streak),
                threshold_value=float(cfg.halt_consecutive),
                details={"symbol": worst_symbol},
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_c_funding_rate_reversal",
                severity=SeverityLevel.OK,
                message=(
                    f"Funding rate normal: max {worst_streak} consecutive "
                    f"negative periods across {len(symbols)} symbol(s)."
                ),
                metric_name="consecutive_negative_funding",
                current_value=float(worst_streak),
                threshold_value=float(cfg.halt_consecutive),
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Track C-4. Beta Drift — Track C beta to SPY exceeding mandate
# ---------------------------------------------------------------------------


def check_beta_drift(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    detector_config: BetaDriftConfig | None = None,
) -> list[SurveillanceCheck]:
    """Detect Track C portfolio beta to SPY exceeding the market-neutral mandate.

    HALT if: rolling 30d beta of Track C returns to SPY > 0.15.
    Computes beta from track_c_daily_returns and SPY prices in market_data_daily.
    """
    cfg = detector_config or BetaDriftConfig()
    checks: list[SurveillanceCheck] = []

    try:
        table_exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'track_c_daily_returns'
            """
        ).fetchone()
        has_table = table_exists is not None and table_exists[0] > 0
    except Exception:
        has_table = False

    if not has_table:
        checks.append(
            SurveillanceCheck(
                detector="track_c_beta_drift",
                severity=SeverityLevel.OK,
                message=(
                    "Track C returns table not yet initialised "
                    "— no beta calculation possible."
                ),
                metric_name="track_c_beta_to_spy",
            )
        )
        return checks

    cutoff = datetime.now(tz=UTC).date() - timedelta(days=cfg.window_days + 5)

    # Fetch Track C daily returns
    tc_rows = conn.execute(
        """
        SELECT date, daily_return
        FROM track_c_daily_returns
        WHERE date >= ?
        ORDER BY date ASC
        """,
        [cutoff],
    ).fetchall()

    # Fetch SPY daily returns
    spy_rows = conn.execute(
        """
        SELECT date, close,
               LAG(close) OVER (ORDER BY date) AS prev_close
        FROM market_data_daily
        WHERE symbol = 'SPY' AND date >= ?
        ORDER BY date ASC
        """,
        [cutoff],
    ).fetchall()

    if len(tc_rows) < cfg.window_days or len(spy_rows) < cfg.window_days:
        checks.append(
            SurveillanceCheck(
                detector="track_c_beta_drift",
                severity=SeverityLevel.OK,
                message=(
                    f"Insufficient history for beta calculation "
                    f"({len(tc_rows)} Track C, {len(spy_rows)} SPY rows, "
                    f"need {cfg.window_days})."
                ),
                metric_name="track_c_beta_to_spy",
            )
        )
        return checks

    # Build aligned date-keyed return dicts
    tc_dict: dict = {r[0]: r[1] for r in tc_rows if r[1] is not None}
    spy_dict: dict = {}
    for row in spy_rows:
        date_val, close, prev_close = row
        if prev_close and prev_close > 0 and close is not None:
            spy_dict[date_val] = (close - prev_close) / prev_close

    # Align on common dates, take most recent window_days
    common_dates = sorted(set(tc_dict) & set(spy_dict))[-cfg.window_days :]

    if len(common_dates) < cfg.window_days // 2:
        checks.append(
            SurveillanceCheck(
                detector="track_c_beta_drift",
                severity=SeverityLevel.OK,
                message=(
                    f"Insufficient aligned dates for beta check "
                    f"({len(common_dates)} common dates)."
                ),
                metric_name="track_c_beta_to_spy",
            )
        )
        return checks

    tc_rets = [tc_dict[d] for d in common_dates]
    spy_rets = [spy_dict[d] for d in common_dates]
    n = len(tc_rets)

    # OLS beta = Cov(tc, spy) / Var(spy)
    tc_mean = sum(tc_rets) / n
    spy_mean = sum(spy_rets) / n
    cov = sum(
        (tc_rets[i] - tc_mean) * (spy_rets[i] - spy_mean) for i in range(n)
    ) / max(n - 1, 1)
    spy_var = sum((r - spy_mean) ** 2 for r in spy_rets) / max(n - 1, 1)
    beta = cov / spy_var if spy_var > 0 else 0.0

    if abs(beta) >= cfg.halt_beta:
        checks.append(
            SurveillanceCheck(
                detector="track_c_beta_drift",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: Track C rolling {cfg.window_days}d "
                    f"beta to SPY = {beta:.3f} "
                    f"exceeds mandate limit {cfg.halt_beta:.2f}."
                ),
                metric_name="track_c_beta_to_spy",
                current_value=abs(beta),
                threshold_value=cfg.halt_beta,
            )
        )
    elif abs(beta) >= cfg.warn_beta:
        checks.append(
            SurveillanceCheck(
                detector="track_c_beta_drift",
                severity=SeverityLevel.WARNING,
                message=(
                    f"Track C beta to SPY = {beta:.3f} approaching "
                    f"halt limit {cfg.halt_beta:.2f} "
                    f"(warn at {cfg.warn_beta:.2f})."
                ),
                metric_name="track_c_beta_to_spy",
                current_value=abs(beta),
                threshold_value=cfg.warn_beta,
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_c_beta_drift",
                severity=SeverityLevel.OK,
                message=(
                    f"Track C beta to SPY = {beta:.3f} — "
                    f"within market-neutral mandate "
                    f"(limit {cfg.halt_beta:.2f})."
                ),
                metric_name="track_c_beta_to_spy",
                current_value=abs(beta),
                threshold_value=cfg.halt_beta,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Track C-5. Cross-Strategy Correlation — Track C vs Track A correlation spike
# ---------------------------------------------------------------------------


def check_cross_strategy_correlation(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    detector_config: CrossTrackCorrelationConfig | None = None,
) -> list[SurveillanceCheck]:
    """Detect Track C correlation spike with Track A returns.

    WARNING if: rolling 30d correlation Track C vs Track A > 0.20
    HALT if:    rolling 30d correlation Track C vs Track A > 0.30

    High correlation means Track C is no longer providing diversification —
    the market-neutral arb is picking up systematic beta from Track A.
    """
    cfg = detector_config or CrossTrackCorrelationConfig()
    checks: list[SurveillanceCheck] = []

    # Check both required tables
    for table in ("track_c_daily_returns", "portfolio_snapshots"):
        try:
            exists = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = '{table}'
                """
            ).fetchone()
            if not exists or exists[0] == 0:
                checks.append(
                    SurveillanceCheck(
                        detector="track_c_cross_strategy_correlation",
                        severity=SeverityLevel.OK,
                        message=(
                            f"Required table '{table}' not yet initialised "
                            "— skipping cross-strategy correlation check."
                        ),
                        metric_name="track_c_vs_track_a_correlation",
                    )
                )
                return checks
        except Exception:
            checks.append(
                SurveillanceCheck(
                    detector="track_c_cross_strategy_correlation",
                    severity=SeverityLevel.OK,
                    message=(
                        "Cannot access tables for cross-strategy correlation check."
                    ),
                    metric_name="track_c_vs_track_a_correlation",
                )
            )
            return checks

    cutoff = datetime.now(tz=UTC).date() - timedelta(days=cfg.window_days + 5)

    # Track C daily returns
    tc_rows = conn.execute(
        """
        SELECT date, daily_return
        FROM track_c_daily_returns
        WHERE date >= ?
        ORDER BY date ASC
        """,
        [cutoff],
    ).fetchall()

    # Track A daily returns from portfolio_snapshots NAV
    ta_rows = conn.execute(
        """
        SELECT date, nav
        FROM portfolio_snapshots
        WHERE date >= ?
        ORDER BY date ASC
        """,
        [cutoff],
    ).fetchall()

    if len(tc_rows) < cfg.window_days or len(ta_rows) < cfg.window_days:
        checks.append(
            SurveillanceCheck(
                detector="track_c_cross_strategy_correlation",
                severity=SeverityLevel.OK,
                message=(
                    f"Insufficient history for correlation check "
                    f"({len(tc_rows)} Track C, {len(ta_rows)} Track A rows, "
                    f"need {cfg.window_days})."
                ),
                metric_name="track_c_vs_track_a_correlation",
            )
        )
        return checks

    # Build Track A daily returns from NAV
    tc_dict: dict = {r[0]: r[1] for r in tc_rows if r[1] is not None}

    ta_dict: dict = {}
    ta_navs = [(r[0], r[1]) for r in ta_rows if r[1] is not None]
    for i in range(1, len(ta_navs)):
        date_val = ta_navs[i][0]
        prev_nav = ta_navs[i - 1][1]
        curr_nav = ta_navs[i][1]
        if prev_nav > 0:
            ta_dict[date_val] = (curr_nav - prev_nav) / prev_nav

    # Align on common dates, take most recent window_days
    common_dates = sorted(set(tc_dict) & set(ta_dict))[-cfg.window_days :]

    if len(common_dates) < cfg.window_days // 2:
        checks.append(
            SurveillanceCheck(
                detector="track_c_cross_strategy_correlation",
                severity=SeverityLevel.OK,
                message=(
                    f"Only {len(common_dates)} aligned dates — "
                    "insufficient for correlation check."
                ),
                metric_name="track_c_vs_track_a_correlation",
            )
        )
        return checks

    tc_rets = [tc_dict[d] for d in common_dates]
    ta_rets = [ta_dict[d] for d in common_dates]
    n = len(tc_rets)

    # Pearson correlation
    tc_mean = sum(tc_rets) / n
    ta_mean = sum(ta_rets) / n
    cov = sum((tc_rets[i] - tc_mean) * (ta_rets[i] - ta_mean) for i in range(n)) / max(
        n - 1, 1
    )
    tc_std = (sum((r - tc_mean) ** 2 for r in tc_rets) / max(n - 1, 1)) ** 0.5
    ta_std = (sum((r - ta_mean) ** 2 for r in ta_rets) / max(n - 1, 1)) ** 0.5
    corr = cov / (tc_std * ta_std) if tc_std > 0 and ta_std > 0 else 0.0

    if corr > cfg.halt_corr:
        checks.append(
            SurveillanceCheck(
                detector="track_c_cross_strategy_correlation",
                severity=SeverityLevel.HALT,
                message=(
                    f"KILL SWITCH: Track C vs Track A {cfg.window_days}d "
                    f"correlation = {corr:.3f} "
                    f"exceeds halt threshold {cfg.halt_corr:.2f}. "
                    "Market-neutral mandate breached."
                ),
                metric_name="track_c_vs_track_a_correlation",
                current_value=corr,
                threshold_value=cfg.halt_corr,
            )
        )
    elif corr > cfg.warn_corr:
        checks.append(
            SurveillanceCheck(
                detector="track_c_cross_strategy_correlation",
                severity=SeverityLevel.WARNING,
                message=(
                    f"Track C vs Track A correlation = {corr:.3f} "
                    f"exceeds warn threshold {cfg.warn_corr:.2f} "
                    f"(halt at {cfg.halt_corr:.2f})."
                ),
                metric_name="track_c_vs_track_a_correlation",
                current_value=corr,
                threshold_value=cfg.warn_corr,
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="track_c_cross_strategy_correlation",
                severity=SeverityLevel.OK,
                message=(
                    f"Track C vs Track A correlation = {corr:.3f} — "
                    f"diversification intact "
                    f"(warn >{cfg.warn_corr:.2f}, "
                    f"halt >{cfg.halt_corr:.2f})."
                ),
                metric_name="track_c_vs_track_a_correlation",
                current_value=corr,
                threshold_value=cfg.warn_corr,
            )
        )

    return checks

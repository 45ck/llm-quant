"""Surveillance detectors — 7 governance checks for production monitoring.

Each detector function takes a DuckDB connection + AppConfig and returns
a list of SurveillanceCheck results.  Detectors are stateless — they query
the DB and config on every invocation.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb

from llm_quant.config import AppConfig
from llm_quant.surveillance.models import SeverityLevel, SurveillanceCheck

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Regime Drift — rolling Sharpe / win-rate / vol vs baseline
# ---------------------------------------------------------------------------


def check_regime_drift(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Compare rolling-window performance metrics against full-history baseline."""
    checks: list[SurveillanceCheck] = []
    cfg = config.governance.regime_drift
    window = cfg.rolling_window_days

    # Fetch daily P&L from snapshots
    rows = conn.execute("""
        SELECT date, daily_pnl, nav
        FROM portfolio_snapshots
        ORDER BY date ASC
        """).fetchall()

    if len(rows) < window + 5:
        checks.append(
            SurveillanceCheck(
                detector="regime_drift",
                severity=SeverityLevel.OK,
                message=(
                    f"Insufficient history ({len(rows)} snapshots, need {window + 5})."
                ),
                metric_name="history_length",
                current_value=float(len(rows)),
                threshold_value=float(window + 5),
            )
        )
        return checks

    daily_pnls = [r[1] for r in rows if r[1] is not None]
    navs = [r[2] for r in rows]

    if not daily_pnls or len(daily_pnls) < window:
        checks.append(
            SurveillanceCheck(
                detector="regime_drift",
                severity=SeverityLevel.OK,
                message="Insufficient daily P&L data for regime drift check.",
            )
        )
        return checks

    # Compute daily returns from NAV
    returns = [
        (navs[i] - navs[i - 1]) / navs[i - 1]
        for i in range(1, len(navs))
        if navs[i - 1] > 0
    ]

    if len(returns) < window:
        checks.append(
            SurveillanceCheck(
                detector="regime_drift",
                severity=SeverityLevel.OK,
                message="Insufficient return data for regime drift check.",
            )
        )
        return checks

    # Full-history Sharpe (annualised)
    full_mean = sum(returns) / len(returns)
    full_var = sum((r - full_mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    full_std = full_var**0.5
    full_sharpe = (full_mean / full_std * (252**0.5)) if full_std > 0 else 0.0

    # Rolling-window Sharpe
    rolling = returns[-window:]
    roll_mean = sum(rolling) / len(rolling)
    roll_var = sum((r - roll_mean) ** 2 for r in rolling) / max(len(rolling) - 1, 1)
    roll_std = roll_var**0.5
    roll_sharpe = (roll_mean / roll_std * (252**0.5)) if roll_std > 0 else 0.0

    # Sharpe decay ratio
    decay = 1.0 - roll_sharpe / full_sharpe if full_sharpe > 0 else 0.0

    if decay >= cfg.sharpe_decay_halt:
        severity = SeverityLevel.HALT
        msg = (
            f"Rolling Sharpe decayed {decay:.0%} vs baseline "
            f"(halt threshold {cfg.sharpe_decay_halt:.0%})."
        )
    elif decay >= cfg.sharpe_decay_warn:
        severity = SeverityLevel.WARNING
        msg = (
            f"Rolling Sharpe decayed {decay:.0%} vs baseline "
            f"(warn threshold {cfg.sharpe_decay_warn:.0%})."
        )
    else:
        severity = SeverityLevel.OK
        msg = f"Rolling Sharpe decay {decay:.0%} within tolerance."

    checks.append(
        SurveillanceCheck(
            detector="regime_drift",
            severity=severity,
            message=msg,
            metric_name="sharpe_decay",
            current_value=decay,
            threshold_value=cfg.sharpe_decay_warn,
        )
    )

    # Volatility spike
    full_vol = full_std * (252**0.5)
    roll_vol = roll_std * (252**0.5)
    vol_ratio = (roll_vol / full_vol) if full_vol > 0 else 1.0

    if vol_ratio >= cfg.vol_spike_halt:
        severity = SeverityLevel.HALT
        msg = (
            f"Rolling vol {vol_ratio:.1f}x baseline "
            f"(halt threshold {cfg.vol_spike_halt:.1f}x)."
        )
    elif vol_ratio >= cfg.vol_spike_warn:
        severity = SeverityLevel.WARNING
        msg = (
            f"Rolling vol {vol_ratio:.1f}x baseline "
            f"(warn threshold {cfg.vol_spike_warn:.1f}x)."
        )
    else:
        severity = SeverityLevel.OK
        msg = f"Rolling vol {vol_ratio:.1f}x baseline — within tolerance."

    checks.append(
        SurveillanceCheck(
            detector="regime_drift",
            severity=severity,
            message=msg,
            metric_name="vol_spike_ratio",
            current_value=vol_ratio,
            threshold_value=cfg.vol_spike_warn,
        )
    )

    return checks


# ---------------------------------------------------------------------------
# 2. Alpha Decay — rolling vs full-history Sharpe
# ---------------------------------------------------------------------------


def check_alpha_decay(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Detect alpha decay by comparing rolling Sharpe to full-history Sharpe."""
    cfg = config.governance.alpha_decay
    window = cfg.rolling_window_days

    rows = conn.execute(
        "SELECT nav FROM portfolio_snapshots ORDER BY date ASC"
    ).fetchall()

    if len(rows) < window + 10:
        return [
            SurveillanceCheck(
                detector="alpha_decay",
                severity=SeverityLevel.OK,
                message=(
                    f"Insufficient history ({len(rows)} snapshots) "
                    "for alpha decay check."
                ),
            )
        ]

    navs = [r[0] for r in rows]
    returns = [
        (navs[i] - navs[i - 1]) / navs[i - 1]
        for i in range(1, len(navs))
        if navs[i - 1] > 0
    ]

    if len(returns) < window:
        return [
            SurveillanceCheck(
                detector="alpha_decay",
                severity=SeverityLevel.OK,
                message="Insufficient return data for alpha decay check.",
            )
        ]

    def _sharpe(rets: list[float]) -> float:
        if len(rets) < 2:
            return 0.0
        m = sum(rets) / len(rets)
        v = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
        s = v**0.5
        return (m / s * (252**0.5)) if s > 0 else 0.0

    full_sharpe = _sharpe(returns)
    roll_sharpe = _sharpe(returns[-window:])

    if full_sharpe > 0:
        ratio = roll_sharpe / full_sharpe
        decay = 1.0 - ratio
    else:
        decay = 0.0

    if decay >= cfg.decay_halt:
        severity = SeverityLevel.HALT
        msg = (
            f"Alpha decay {decay:.0%} — rolling Sharpe "
            f"{roll_sharpe:.2f} vs full {full_sharpe:.2f} "
            f"(halt at {cfg.decay_halt:.0%})."
        )
    elif decay >= cfg.decay_warn:
        severity = SeverityLevel.WARNING
        msg = (
            f"Alpha decay {decay:.0%} — rolling Sharpe "
            f"{roll_sharpe:.2f} vs full {full_sharpe:.2f} "
            f"(warn at {cfg.decay_warn:.0%})."
        )
    else:
        severity = SeverityLevel.OK
        msg = (
            f"Alpha decay {decay:.0%} within tolerance "
            f"(rolling {roll_sharpe:.2f}, full {full_sharpe:.2f})."
        )

    return [
        SurveillanceCheck(
            detector="alpha_decay",
            severity=severity,
            message=msg,
            metric_name="alpha_decay_ratio",
            current_value=decay,
            threshold_value=cfg.decay_warn,
        )
    ]


# ---------------------------------------------------------------------------
# 3. Risk Drift — post-trade exposure monitoring
# ---------------------------------------------------------------------------


def check_risk_drift(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Check current portfolio exposure against configured limits."""
    checks: list[SurveillanceCheck] = []
    cfg = config.governance.risk_drift
    limits = config.risk

    # Get latest snapshot
    row = conn.execute("""
        SELECT nav, gross_exposure, net_exposure
        FROM portfolio_snapshots
        ORDER BY date DESC, snapshot_id DESC
        LIMIT 1
        """).fetchone()

    if row is None:
        return [
            SurveillanceCheck(
                detector="risk_drift",
                severity=SeverityLevel.OK,
                message="No portfolio snapshots — skipping risk drift check.",
            )
        ]

    nav, gross, net = row

    if nav <= 0:
        return [
            SurveillanceCheck(
                detector="risk_drift",
                severity=SeverityLevel.WARNING,
                message="NAV is zero or negative.",
            )
        ]

    # Gross exposure ratio
    gross_ratio = gross / nav
    gross_limit = limits.max_gross_exposure
    gross_warn = gross_limit * (1.0 - cfg.exposure_warn_buffer)

    if gross_ratio >= gross_limit:
        severity = SeverityLevel.HALT
        msg = f"Gross exposure {gross_ratio:.0%} exceeds limit {gross_limit:.0%}."
    elif gross_ratio >= gross_warn:
        severity = SeverityLevel.WARNING
        msg = (
            f"Gross exposure {gross_ratio:.0%} approaching "
            f"limit {gross_limit:.0%} (warn at {gross_warn:.0%})."
        )
    else:
        severity = SeverityLevel.OK
        msg = f"Gross exposure {gross_ratio:.0%} within limits."

    checks.append(
        SurveillanceCheck(
            detector="risk_drift",
            severity=severity,
            message=msg,
            metric_name="gross_exposure_ratio",
            current_value=gross_ratio,
            threshold_value=gross_limit,
        )
    )

    # Net exposure ratio
    net_ratio = abs(net) / nav
    net_limit = limits.max_net_exposure
    net_warn = net_limit * (1.0 - cfg.exposure_warn_buffer)

    if net_ratio >= net_limit:
        severity = SeverityLevel.HALT
        msg = f"Net exposure {net_ratio:.0%} exceeds limit {net_limit:.0%}."
    elif net_ratio >= net_warn:
        severity = SeverityLevel.WARNING
        msg = f"Net exposure {net_ratio:.0%} approaching limit {net_limit:.0%}."
    else:
        severity = SeverityLevel.OK
        msg = f"Net exposure {net_ratio:.0%} within limits."

    checks.append(
        SurveillanceCheck(
            detector="risk_drift",
            severity=severity,
            message=msg,
            metric_name="net_exposure_ratio",
            current_value=net_ratio,
            threshold_value=net_limit,
        )
    )

    # Sector concentration
    # Build sector map from universe config
    sector_map: dict[str, str] = {}
    for asset in config.universe.assets:
        sector_map[asset.symbol] = asset.sector

    # Get latest snapshot positions
    latest_snap = conn.execute("""
        SELECT snapshot_id
        FROM portfolio_snapshots
        ORDER BY date DESC, snapshot_id DESC
        LIMIT 1
        """).fetchone()

    if latest_snap:
        snapshot_id = latest_snap[0]
        pos_rows = conn.execute(
            "SELECT symbol, weight FROM positions WHERE snapshot_id = ?",
            [snapshot_id],
        ).fetchall()

        # Aggregate weights by sector
        sector_weights: dict[str, float] = {}
        for symbol, weight in pos_rows:
            sector = sector_map.get(symbol, "unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(weight)

        if sector_weights:
            max_sector = max(sector_weights.values())
            max_sector_name = max(sector_weights, key=sector_weights.get)  # type: ignore[arg-type]
            sector_limit = limits.max_sector_concentration
            sector_warn = sector_limit * (1.0 - cfg.concentration_warn_buffer)

            if max_sector >= sector_limit:
                severity = SeverityLevel.HALT
                msg = (
                    f"Sector '{max_sector_name}' concentration "
                    f"{max_sector:.0%} exceeds "
                    f"limit {sector_limit:.0%}."
                )
            elif max_sector >= sector_warn:
                severity = SeverityLevel.WARNING
                msg = (
                    f"Sector '{max_sector_name}' concentration "
                    f"{max_sector:.0%} approaching "
                    f"limit {sector_limit:.0%} "
                    f"(warn at {sector_warn:.0%})."
                )
            else:
                severity = SeverityLevel.OK
                msg = (
                    f"Max sector concentration {max_sector:.0%} "
                    f"('{max_sector_name}') within limits."
                )

            checks.append(
                SurveillanceCheck(
                    detector="risk_drift",
                    severity=severity,
                    message=msg,
                    metric_name="sector_concentration",
                    current_value=max_sector,
                    threshold_value=sector_limit,
                )
            )

    return checks


# ---------------------------------------------------------------------------
# 4. Data Quality — stale symbols, price gaps, plausibility
# ---------------------------------------------------------------------------


def check_data_quality(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Check for stale data, price gaps, and implausible prices."""
    checks: list[SurveillanceCheck] = []
    cfg = config.governance.data_quality

    # Check for stale symbols (no data in max_stale_days)
    cutoff_date = datetime.now(tz=UTC).date() - timedelta(days=cfg.max_stale_days + 2)
    stale_rows = conn.execute(
        """
        SELECT symbol, MAX(date) as last_date
        FROM market_data_daily
        GROUP BY symbol
        HAVING MAX(date) < ?
        """,
        [cutoff_date],
    ).fetchall()

    if stale_rows:
        stale_symbols = [r[0] for r in stale_rows]
        checks.append(
            SurveillanceCheck(
                detector="data_quality",
                severity=SeverityLevel.HALT,
                message=(
                    f"Stale data for {len(stale_symbols)} symbols: "
                    f"{', '.join(stale_symbols[:5])}"
                    f"{'...' if len(stale_symbols) > 5 else ''}"
                ),
                metric_name="stale_symbol_count",
                current_value=float(len(stale_symbols)),
                threshold_value=0.0,
                details={"stale_symbols": stale_symbols},
            )
        )
    else:
        checks.append(
            SurveillanceCheck(
                detector="data_quality",
                severity=SeverityLevel.OK,
                message="No stale symbols detected.",
                metric_name="stale_symbol_count",
                current_value=0.0,
            )
        )

    # Check for implausible prices
    implausible = conn.execute(
        """
        SELECT symbol, date, close
        FROM market_data_daily
        WHERE close < ? AND close IS NOT NULL
        AND date >= CURRENT_DATE - INTERVAL '7 days'
        """,
        [cfg.plausibility_min_price],
    ).fetchall()

    if implausible:
        symbols = list({r[0] for r in implausible})
        checks.append(
            SurveillanceCheck(
                detector="data_quality",
                severity=SeverityLevel.HALT,
                message=(
                    f"Implausible prices "
                    f"(<${cfg.plausibility_min_price}) for: "
                    f"{', '.join(symbols)}"
                ),
                metric_name="implausible_price_count",
                current_value=float(len(implausible)),
                threshold_value=0.0,
            )
        )

    # Check for large single-day gaps
    gap_rows = conn.execute(f"""
        SELECT symbol, date, close,
               LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
        FROM market_data_daily
        WHERE date >= CURRENT_DATE - INTERVAL '{cfg.max_stale_days + 7} days'
        QUALIFY prev_close IS NOT NULL
            AND prev_close > 0
            AND ABS(close - prev_close) / prev_close > {cfg.gap_threshold_pct}
        ORDER BY date DESC
        LIMIT 10
        """).fetchall()

    if gap_rows:
        checks.append(
            SurveillanceCheck(
                detector="data_quality",
                severity=SeverityLevel.WARNING,
                message=(
                    f"Large price gaps (>{cfg.gap_threshold_pct:.0%}) "
                    f"detected in {len(gap_rows)} recent data points."
                ),
                metric_name="price_gap_count",
                current_value=float(len(gap_rows)),
                threshold_value=0.0,
            )
        )

    if not checks:
        checks.append(
            SurveillanceCheck(
                detector="data_quality",
                severity=SeverityLevel.OK,
                message="Data quality checks passed.",
            )
        )

    return checks


# ---------------------------------------------------------------------------
# 5. Process Drift — SHA-256 hashes of config files
# ---------------------------------------------------------------------------


def check_process_drift(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Detect unauthorized config changes via SHA-256 hash comparison."""
    checks: list[SurveillanceCheck] = []
    cfg = config.governance.process_drift

    # Find project root (walk up from config module)
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    for file_rel in cfg.tracked_files:
        file_path = project_root / file_rel
        if not file_path.exists():
            checks.append(
                SurveillanceCheck(
                    detector="process_drift",
                    severity=SeverityLevel.WARNING,
                    message=f"Tracked file missing: {file_rel}",
                    metric_name="missing_config_file",
                )
            )
            continue

        # Compute current hash
        content = file_path.read_bytes()
        current_hash = hashlib.sha256(content).hexdigest()

        # Check against stored hash
        stored = conn.execute(
            """
            SELECT hash_sha256
            FROM config_hashes
            WHERE file_path = ?
            ORDER BY recorded_at DESC
            LIMIT 1
            """,
            [file_rel],
        ).fetchone()

        if stored is None:
            # First time — record baseline
            conn.execute(
                "INSERT INTO config_hashes (file_path, hash_sha256) VALUES (?, ?)",
                [file_rel, current_hash],
            )
            conn.commit()
            checks.append(
                SurveillanceCheck(
                    detector="process_drift",
                    severity=SeverityLevel.OK,
                    message=f"Baseline hash recorded for {file_rel}.",
                    metric_name="config_hash_baseline",
                )
            )
        elif stored[0] != current_hash:
            checks.append(
                SurveillanceCheck(
                    detector="process_drift",
                    severity=SeverityLevel.WARNING,
                    message=f"Config file changed: {file_rel} (hash mismatch).",
                    metric_name="config_hash_mismatch",
                    details={
                        "file": file_rel,
                        "stored_hash": stored[0][:12],
                        "current_hash": current_hash[:12],
                    },
                )
            )
            # Update stored hash
            conn.execute(
                "INSERT INTO config_hashes (file_path, hash_sha256) VALUES (?, ?)",
                [file_rel, current_hash],
            )
            conn.commit()
        else:
            checks.append(
                SurveillanceCheck(
                    detector="process_drift",
                    severity=SeverityLevel.OK,
                    message=f"Config unchanged: {file_rel}.",
                )
            )

    if not checks:
        checks.append(
            SurveillanceCheck(
                detector="process_drift",
                severity=SeverityLevel.OK,
                message="No tracked config files configured.",
            )
        )

    return checks


# ---------------------------------------------------------------------------
# 6. Operational Health — snapshot gaps, stale prices, hash chain
# ---------------------------------------------------------------------------


def check_operational_health(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Check for snapshot gaps, stale prices, and hash chain integrity."""
    checks: list[SurveillanceCheck] = []
    cfg = config.governance.operational_health

    # Check for snapshot gaps
    latest_snapshot = conn.execute(
        "SELECT MAX(date) FROM portfolio_snapshots"
    ).fetchone()

    if latest_snapshot and latest_snapshot[0]:
        last_date = latest_snapshot[0]
        if isinstance(last_date, str):
            last_date = date.fromisoformat(last_date)
        today = datetime.now(tz=UTC).date()
        gap_days = (today - last_date).days

        if gap_days > cfg.max_snapshot_gap_days:
            checks.append(
                SurveillanceCheck(
                    detector="operational_health",
                    severity=SeverityLevel.WARNING,
                    message=(
                        f"No portfolio snapshot for {gap_days} "
                        f"days (limit {cfg.max_snapshot_gap_days})."
                    ),
                    metric_name="snapshot_gap_days",
                    current_value=float(gap_days),
                    threshold_value=float(cfg.max_snapshot_gap_days),
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="operational_health",
                    severity=SeverityLevel.OK,
                    message=f"Latest snapshot {gap_days} day(s) ago.",
                    metric_name="snapshot_gap_days",
                    current_value=float(gap_days),
                )
            )
    else:
        checks.append(
            SurveillanceCheck(
                detector="operational_health",
                severity=SeverityLevel.WARNING,
                message="No portfolio snapshots found.",
                metric_name="snapshot_gap_days",
            )
        )

    # Check for stale market data
    latest_market = conn.execute("SELECT MAX(date) FROM market_data_daily").fetchone()

    if latest_market and latest_market[0]:
        last_market_date = latest_market[0]
        if isinstance(last_market_date, str):
            last_market_date = date.fromisoformat(last_market_date)
        today_mkt = datetime.now(tz=UTC).date()
        hours_stale = (today_mkt - last_market_date).days * 24

        if hours_stale > cfg.max_price_staleness_hours:
            checks.append(
                SurveillanceCheck(
                    detector="operational_health",
                    severity=SeverityLevel.WARNING,
                    message=(
                        f"Market data {hours_stale}h stale "
                        f"(limit {cfg.max_price_staleness_hours}h)."
                    ),
                    metric_name="price_staleness_hours",
                    current_value=float(hours_stale),
                    threshold_value=float(cfg.max_price_staleness_hours),
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="operational_health",
                    severity=SeverityLevel.OK,
                    message=f"Market data {hours_stale}h old — within tolerance.",
                    metric_name="price_staleness_hours",
                    current_value=float(hours_stale),
                )
            )

    # Hash chain verification
    if cfg.hash_chain_required:
        try:
            from llm_quant.db.integrity import verify_chain

            chain_ok, _last_id, chain_msg = verify_chain(conn)
            if not chain_ok:
                checks.append(
                    SurveillanceCheck(
                        detector="operational_health",
                        severity=SeverityLevel.HALT,
                        message=(
                            f"Trade ledger hash chain verification FAILED: {chain_msg}"
                        ),
                        metric_name="hash_chain_valid",
                        current_value=0.0,
                        threshold_value=1.0,
                    )
                )
            else:
                checks.append(
                    SurveillanceCheck(
                        detector="operational_health",
                        severity=SeverityLevel.OK,
                        message=f"Hash chain verified: {chain_msg}",
                        metric_name="hash_chain_valid",
                        current_value=1.0,
                    )
                )
        except Exception as exc:
            checks.append(
                SurveillanceCheck(
                    detector="operational_health",
                    severity=SeverityLevel.WARNING,
                    message=f"Hash chain check failed: {exc}",
                    metric_name="hash_chain_valid",
                )
            )

    return checks


# ---------------------------------------------------------------------------
# 7. Kill Switches — 6 automatic halt triggers
# ---------------------------------------------------------------------------


def check_kill_switches(  # noqa: C901, PLR0912
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> list[SurveillanceCheck]:
    """Evaluate 6 kill switch conditions.  Any one triggers full halt."""
    checks: list[SurveillanceCheck] = []
    ks = config.governance.kill_switches

    # 1. NAV drawdown from peak
    snapshots = conn.execute(
        "SELECT nav FROM portfolio_snapshots ORDER BY date ASC"
    ).fetchall()

    if snapshots:
        navs = [r[0] for r in snapshots]
        peak = max(navs)
        current = navs[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0.0

        if drawdown >= ks.max_drawdown_pct:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.HALT,
                    message=(
                        f"KILL SWITCH: Drawdown {drawdown:.1%} "
                        f"exceeds {ks.max_drawdown_pct:.0%} limit."
                    ),
                    metric_name="drawdown_pct",
                    current_value=drawdown,
                    threshold_value=ks.max_drawdown_pct,
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.OK,
                    message=(
                        f"Drawdown {drawdown:.1%} within "
                        f"{ks.max_drawdown_pct:.0%} limit."
                    ),
                    metric_name="drawdown_pct",
                    current_value=drawdown,
                    threshold_value=ks.max_drawdown_pct,
                )
            )

    # 2. Single-day loss
    daily_row = conn.execute("""
        SELECT daily_pnl, nav
        FROM portfolio_snapshots
        WHERE daily_pnl IS NOT NULL AND nav > 0
        ORDER BY date DESC
        LIMIT 1
        """).fetchone()

    if daily_row:
        daily_pnl, nav = daily_row
        daily_loss_pct = abs(min(daily_pnl, 0.0)) / nav if nav > 0 else 0.0

        if daily_loss_pct >= ks.max_daily_loss_pct:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.HALT,
                    message=(
                        f"KILL SWITCH: Daily loss "
                        f"{daily_loss_pct:.1%} exceeds "
                        f"{ks.max_daily_loss_pct:.0%} limit."
                    ),
                    metric_name="daily_loss_pct",
                    current_value=daily_loss_pct,
                    threshold_value=ks.max_daily_loss_pct,
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.OK,
                    message=f"Daily loss {daily_loss_pct:.1%} within limit.",
                    metric_name="daily_loss_pct",
                    current_value=daily_loss_pct,
                    threshold_value=ks.max_daily_loss_pct,
                )
            )

    # 3. Consecutive losing trades
    recent_trades = conn.execute("""
        SELECT trade_id, symbol, action, price, notional
        FROM trades
        ORDER BY date DESC, trade_id DESC
        LIMIT 20
        """).fetchall()

    if recent_trades:
        # Count consecutive losses from most recent
        # A trade is "losing" if it's a SELL/CLOSE with negative P&L
        # For simplicity, check consecutive days with negative daily_pnl
        recent_pnl = conn.execute("""
            SELECT daily_pnl
            FROM portfolio_snapshots
            WHERE daily_pnl IS NOT NULL
            ORDER BY date DESC
            LIMIT 20
            """).fetchall()

        consecutive_losses = 0
        for row in recent_pnl:
            if row[0] is not None and row[0] < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= ks.max_consecutive_losses:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.HALT,
                    message=(
                        f"KILL SWITCH: {consecutive_losses} "
                        "consecutive losing days "
                        f"(limit {ks.max_consecutive_losses})."
                    ),
                    metric_name="consecutive_losses",
                    current_value=float(consecutive_losses),
                    threshold_value=float(ks.max_consecutive_losses),
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.OK,
                    message=(
                        f"{consecutive_losses} consecutive losing "
                        f"days (limit {ks.max_consecutive_losses})."
                    ),
                    metric_name="consecutive_losses",
                    current_value=float(consecutive_losses),
                    threshold_value=float(ks.max_consecutive_losses),
                )
            )

    # 4. Data blackout
    latest_data = conn.execute("SELECT MAX(date) FROM market_data_daily").fetchone()

    if latest_data and latest_data[0]:
        last_date = latest_data[0]
        if isinstance(last_date, str):
            last_date = date.fromisoformat(last_date)
        today_ks = datetime.now(tz=UTC).date()
        hours_since = (today_ks - last_date).days * 24

        if hours_since >= ks.data_blackout_hours:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.HALT,
                    message=(
                        f"KILL SWITCH: No fresh data for "
                        f"{hours_since}h "
                        f"(limit {ks.data_blackout_hours}h)."
                    ),
                    metric_name="data_blackout_hours",
                    current_value=float(hours_since),
                    threshold_value=float(ks.data_blackout_hours),
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.OK,
                    message=f"Data freshness {hours_since}h — within limit.",
                    metric_name="data_blackout_hours",
                    current_value=float(hours_since),
                    threshold_value=float(ks.data_blackout_hours),
                )
            )

    # 5. Correlation breach — skip in paper trading (requires position-level returns)
    checks.append(
        SurveillanceCheck(
            detector="kill_switches",
            severity=SeverityLevel.OK,
            message="Correlation breach check deferred (paper trading).",
            metric_name="correlation_breach",
        )
    )

    # 6. Risk check failure streak
    recent_rejections = conn.execute("""
        SELECT COUNT(*)
        FROM surveillance_scans
        WHERE overall_severity = 'halt'
        AND scan_timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
        """).fetchone()

    if recent_rejections:
        streak = recent_rejections[0]
        if streak >= ks.risk_check_failure_streak:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.HALT,
                    message=(
                        f"KILL SWITCH: {streak} halt-level scans "
                        f"in 7 days "
                        f"(limit {ks.risk_check_failure_streak})."
                    ),
                    metric_name="risk_check_failure_streak",
                    current_value=float(streak),
                    threshold_value=float(ks.risk_check_failure_streak),
                )
            )
        else:
            checks.append(
                SurveillanceCheck(
                    detector="kill_switches",
                    severity=SeverityLevel.OK,
                    message=(
                        f"{streak} halt scans in 7 days "
                        f"(limit {ks.risk_check_failure_streak})."
                    ),
                    metric_name="risk_check_failure_streak",
                    current_value=float(streak),
                    threshold_value=float(ks.risk_check_failure_streak),
                )
            )

    return checks

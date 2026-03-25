"""Cross-pod risk monitoring and fund-level analytics.

Provides a read-only view across all trading pods, computing aggregate
metrics, correlation, and enforcing fund-level circuit breakers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PodSnapshot:
    """Point-in-time snapshot of a single pod's state."""

    pod_id: str
    strategy_type: str
    display_name: str
    nav: float
    initial_capital: float
    cash: float
    gross_exposure: float
    net_exposure: float
    total_pnl: float
    daily_pnl: float
    peak_nav: float
    drawdown_pct: float  # negative number
    num_positions: int
    total_trades: int
    status: str  # active, paused, retired


@dataclass
class MetaPortfolioMetrics:
    """Fund-level aggregate metrics across all pods."""

    fund_nav: float
    fund_initial_capital: float
    fund_total_return_pct: float
    fund_daily_pnl: float
    fund_drawdown_pct: float
    total_capital_at_risk: float  # sum of pod gross exposures
    num_active_pods: int
    num_total_positions: int
    pod_snapshots: list[PodSnapshot] = field(default_factory=list)
    pod_rankings: list[str] = field(default_factory=list)  # pod_ids by P&L
    alerts: list[str] = field(default_factory=list)


# Fund-level hard limits (immutable)
FUND_MAX_DRAWDOWN_PCT = 0.20
FUND_MAX_GROSS_EXPOSURE_RATIO = 1.8
SINGLE_SYMBOL_FUND_CAP = 0.15


class MetaRiskMonitor:
    """Read-only cross-pod risk monitor.

    Queries all pod snapshots and computes fund-level metrics.
    Can enforce fund-level circuit breakers that override pod-level decisions.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def _table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return row is not None and row[0] > 0

    def get_pod_snapshots(self) -> list[PodSnapshot]:
        """Get latest snapshot for each active pod."""
        if not self._table_exists("pods") or not self._table_exists(
            "portfolio_snapshots"
        ):
            return []

        # For each pod, get the latest portfolio_snapshot
        # (by date DESC, snapshot_id DESC).
        # Also count positions and trades per pod.
        rows = self.conn.execute("""
            WITH latest_snap AS (
                SELECT
                    ps.pod_id,
                    ps.snapshot_id,
                    ps.nav,
                    ps.cash,
                    ps.gross_exposure,
                    ps.net_exposure,
                    ps.total_pnl,
                    ps.daily_pnl,
                    ROW_NUMBER() OVER (
                        PARTITION BY ps.pod_id
                        ORDER BY ps.date DESC, ps.snapshot_id DESC
                    ) AS rn
                FROM portfolio_snapshots ps
            ),
            peak_navs AS (
                SELECT
                    pod_id,
                    MAX(nav) AS peak_nav
                FROM portfolio_snapshots
                GROUP BY pod_id
            ),
            pos_counts AS (
                SELECT
                    ls.pod_id,
                    COUNT(p.symbol) AS num_positions
                FROM latest_snap ls
                LEFT JOIN positions p ON p.snapshot_id = ls.snapshot_id
                WHERE ls.rn = 1
                GROUP BY ls.pod_id
            ),
            trade_counts AS (
                SELECT
                    pod_id,
                    COUNT(*) AS total_trades
                FROM trades
                GROUP BY pod_id
            )
            SELECT
                pods.pod_id,
                pods.strategy_type,
                pods.display_name,
                COALESCE(ls.nav, pods.initial_capital) AS nav,
                pods.initial_capital,
                COALESCE(ls.cash, pods.initial_capital) AS cash,
                COALESCE(ls.gross_exposure, 0.0) AS gross_exposure,
                COALESCE(ls.net_exposure, 0.0) AS net_exposure,
                COALESCE(ls.total_pnl, 0.0) AS total_pnl,
                COALESCE(ls.daily_pnl, 0.0) AS daily_pnl,
                COALESCE(pn.peak_nav, pods.initial_capital) AS peak_nav,
                COALESCE(pc.num_positions, 0) AS num_positions,
                COALESCE(tc.total_trades, 0) AS total_trades,
                pods.status
            FROM pods
            LEFT JOIN latest_snap ls
                ON ls.pod_id = pods.pod_id AND ls.rn = 1
            LEFT JOIN peak_navs pn
                ON pn.pod_id = pods.pod_id
            LEFT JOIN pos_counts pc
                ON pc.pod_id = pods.pod_id
            LEFT JOIN trade_counts tc
                ON tc.pod_id = pods.pod_id
            WHERE pods.status IN ('active', 'paused')
            ORDER BY pods.pod_id
            """).fetchall()

        snapshots: list[PodSnapshot] = []
        for r in rows:
            nav = float(r[3])
            peak = float(r[10])
            dd_pct = (nav - peak) / peak if peak > 0 else 0.0
            snapshots.append(
                PodSnapshot(
                    pod_id=r[0],
                    strategy_type=r[1],
                    display_name=r[2],
                    nav=nav,
                    initial_capital=float(r[4]),
                    cash=float(r[5]),
                    gross_exposure=float(r[6]),
                    net_exposure=float(r[7]),
                    total_pnl=float(r[8]),
                    daily_pnl=float(r[9]),
                    peak_nav=peak,
                    drawdown_pct=dd_pct,
                    num_positions=int(r[11]),
                    total_trades=int(r[12]),
                    status=r[13],
                )
            )

        return snapshots

    def compute_fund_metrics(self) -> MetaPortfolioMetrics:
        """Compute aggregate fund-level metrics."""
        snapshots = self.get_pod_snapshots()

        fund_nav = sum(s.nav for s in snapshots)
        fund_initial = sum(s.initial_capital for s in snapshots)
        fund_return = (
            (fund_nav - fund_initial) / fund_initial if fund_initial > 0 else 0.0
        )
        fund_daily_pnl = sum(s.daily_pnl for s in snapshots)
        total_car = sum(s.gross_exposure for s in snapshots)

        # Fund drawdown: use peak of aggregate NAV
        fund_peak = self._get_fund_peak_nav()
        fund_dd = (fund_nav - fund_peak) / fund_peak if fund_peak > 0 else 0.0

        # Rankings by total P&L
        ranked = sorted(snapshots, key=lambda s: s.total_pnl, reverse=True)
        rankings = [s.pod_id for s in ranked]

        # Alerts
        alerts: list[str] = []
        if fund_dd < -FUND_MAX_DRAWDOWN_PCT:
            alerts.append(
                f"CRITICAL: Fund drawdown {fund_dd:.1%} "
                f"exceeds {FUND_MAX_DRAWDOWN_PCT:.0%} limit"
            )
        if (
            fund_initial > 0
            and fund_nav > 0
            and total_car / fund_nav > FUND_MAX_GROSS_EXPOSURE_RATIO
        ):
            exp_ratio = total_car / fund_nav
            alerts.append(
                f"WARNING: Total gross exposure ratio "
                f"{exp_ratio:.2f}x exceeds "
                f"{FUND_MAX_GROSS_EXPOSURE_RATIO}x limit"
            )

        alerts.extend(
            f"WARNING: Pod '{s.pod_id}' drawdown {s.drawdown_pct:.1%}"
            for s in snapshots
            if s.drawdown_pct < -0.15
        )

        return MetaPortfolioMetrics(
            fund_nav=fund_nav,
            fund_initial_capital=fund_initial,
            fund_total_return_pct=fund_return,
            fund_daily_pnl=fund_daily_pnl,
            fund_drawdown_pct=fund_dd,
            total_capital_at_risk=total_car,
            num_active_pods=len(snapshots),
            num_total_positions=sum(s.num_positions for s in snapshots),
            pod_snapshots=snapshots,
            pod_rankings=rankings,
            alerts=alerts,
        )

    def check_fund_level_limits(self, _pod_id: str, is_buy: bool) -> list[str]:
        """Check fund-level limits before allowing a pod to trade.

        Returns list of rejection reasons (empty = OK to trade).
        """
        if not is_buy:
            return []  # sells always allowed

        metrics = self.compute_fund_metrics()
        rejections: list[str] = []

        if metrics.fund_drawdown_pct < -FUND_MAX_DRAWDOWN_PCT:
            rejections.append(
                f"Fund drawdown {metrics.fund_drawdown_pct:.1%} exceeds "
                f"{FUND_MAX_DRAWDOWN_PCT:.0%} — all buys blocked"
            )

        if (
            metrics.fund_initial_capital > 0
            and metrics.fund_nav > 0
            and metrics.total_capital_at_risk / metrics.fund_nav
            > FUND_MAX_GROSS_EXPOSURE_RATIO
        ):
            rejections.append(
                "Fund gross exposure ratio exceeds "
                f"{FUND_MAX_GROSS_EXPOSURE_RATIO}x — buys blocked"
            )

        return rejections

    def get_cross_pod_symbol_exposure(self, symbol: str) -> float:
        """Get total weight of a symbol across all pods.

        Returns the sum of market values for this symbol across all pods,
        divided by total fund NAV. Returns 0.0 if no data exists.
        """
        if not self._table_exists("portfolio_snapshots") or not self._table_exists(
            "positions"
        ):
            return 0.0

        # Get the latest snapshot_id per pod
        rows = self.conn.execute(
            """
            WITH latest_snap AS (
                SELECT
                    pod_id,
                    snapshot_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY pod_id
                        ORDER BY date DESC, snapshot_id DESC
                    ) AS rn
                FROM portfolio_snapshots
            )
            SELECT
                COALESCE(SUM(p.market_value), 0.0) AS total_mv,
                COALESCE(SUM(ls_nav.nav), 0.0) AS total_nav
            FROM latest_snap ls
            JOIN positions p
                ON p.snapshot_id = ls.snapshot_id AND p.symbol = ?
            JOIN portfolio_snapshots ls_nav
                ON ls_nav.snapshot_id = ls.snapshot_id
            WHERE ls.rn = 1
            """,
            [symbol],
        ).fetchone()

        if rows is None:
            return 0.0

        total_mv = float(rows[0])
        # For fund NAV, sum across all pods' latest snapshots
        fund_nav_row = self.conn.execute("""
            WITH latest_snap AS (
                SELECT
                    snapshot_id,
                    nav,
                    ROW_NUMBER() OVER (
                        PARTITION BY pod_id
                        ORDER BY date DESC, snapshot_id DESC
                    ) AS rn
                FROM portfolio_snapshots
            )
            SELECT COALESCE(SUM(nav), 0.0)
            FROM latest_snap
            WHERE rn = 1
            """).fetchone()

        fund_nav = float(fund_nav_row[0]) if fund_nav_row else 0.0
        if fund_nav <= 0.0:
            return 0.0

        return total_mv / fund_nav

    def _get_fund_peak_nav(self) -> float:
        """Get the historical peak of aggregate fund NAV.

        Computes the sum of NAV across all pods for each date, then
        returns the maximum of those sums.
        """
        if not self._table_exists("portfolio_snapshots"):
            return 0.0

        # For each date, take the last snapshot per pod, sum NAVs, then take max
        row = self.conn.execute("""
            WITH daily_pod_nav AS (
                SELECT
                    date,
                    pod_id,
                    nav,
                    ROW_NUMBER() OVER (
                        PARTITION BY date, pod_id
                        ORDER BY snapshot_id DESC
                    ) AS rn
                FROM portfolio_snapshots
            ),
            daily_fund_nav AS (
                SELECT
                    date,
                    SUM(nav) AS fund_nav
                FROM daily_pod_nav
                WHERE rn = 1
                GROUP BY date
            )
            SELECT COALESCE(MAX(fund_nav), 0.0)
            FROM daily_fund_nav
            """).fetchone()

        return float(row[0]) if row else 0.0

    def get_pod_correlation_matrix(self, lookback_days: int = 60) -> pl.DataFrame:
        """Compute correlation matrix of daily returns across pods.

        Returns a Polars DataFrame with pod_ids as both column names and
        a ``pod_id`` identifier column, containing pairwise correlations.
        Returns an empty DataFrame if insufficient data.
        """
        if not self._table_exists("portfolio_snapshots"):
            return pl.DataFrame()

        # Get daily P&L per pod over the lookback window
        rows = self.conn.execute(
            """
            WITH daily_pod AS (
                SELECT
                    date,
                    pod_id,
                    daily_pnl,
                    nav,
                    ROW_NUMBER() OVER (
                        PARTITION BY date, pod_id
                        ORDER BY snapshot_id DESC
                    ) AS rn
                FROM portfolio_snapshots
                WHERE date >= CURRENT_DATE - INTERVAL ? DAY
            )
            SELECT date, pod_id, daily_pnl, nav
            FROM daily_pod
            WHERE rn = 1
              AND daily_pnl IS NOT NULL
              AND nav > 0
            ORDER BY date ASC, pod_id ASC
            """,
            [lookback_days],
        ).fetchall()

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(
            {
                "date": [r[0] for r in rows],
                "pod_id": [r[1] for r in rows],
                "daily_return": [
                    float(r[2]) / float(r[3]) if float(r[3]) > 0 else 0.0 for r in rows
                ],
            }
        )

        # Pivot to wide format: date x pod_id
        wide = df.pivot(on="pod_id", index="date", values="daily_return")

        pod_ids = [c for c in wide.columns if c != "date"]
        if len(pod_ids) < 2:
            return pl.DataFrame()

        # Compute pairwise correlation
        numeric = wide.select(pod_ids).fill_null(0.0)
        corr_data: dict[str, list[float]] = {"pod_id": pod_ids}
        for pid in pod_ids:
            col_corrs: list[float] = []
            for other in pod_ids:
                corr = numeric[pid].pearson_corr(numeric[other])
                col_corrs.append(round(corr if corr is not None else 0.0, 4))
            corr_data[pid] = col_corrs

        return pl.DataFrame(corr_data)

    def format_dashboard(self) -> str:
        """Format a text dashboard of all pod metrics."""
        metrics = self.compute_fund_metrics()
        lines: list[str] = []
        lines.append(
            f"Fund NAV: ${metrics.fund_nav:,.0f} ({metrics.fund_total_return_pct:+.2%})"
        )
        lines.append(f"Daily P&L: ${metrics.fund_daily_pnl:+,.0f}")
        lines.append(f"Drawdown: {metrics.fund_drawdown_pct:.2%}")
        lines.append(f"Active pods: {metrics.num_active_pods}")
        lines.append(f"Total positions: {metrics.num_total_positions}")
        lines.append("")

        # Per-pod table
        header = (
            f"{'Pod':<20} {'NAV':>10} {'P&L':>10} {'DD':>8} {'Pos':>5} {'Status':<8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        lines.extend(
            f"{s.pod_id:<20} ${s.nav:>9,.0f} ${s.total_pnl:>+9,.0f} "
            f"{s.drawdown_pct:>7.1%} {s.num_positions:>5} {s.status:<8}"
            for s in metrics.pod_snapshots
        )

        if metrics.alerts:
            lines.append("")
            lines.append("ALERTS:")
            lines.extend(f"  ! {a}" for a in metrics.alerts)

        return "\n".join(lines)

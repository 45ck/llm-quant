"""Tests for MetaRiskMonitor (cross-pod fund-level analytics)."""

from datetime import UTC, datetime, timedelta

from llm_quant.risk.meta_monitor import MetaRiskMonitor


def test_meta_empty_db(pod_db):
    """MetaRiskMonitor on a DB with only the default pod (no snapshots)."""
    monitor = MetaRiskMonitor(pod_db)
    metrics = monitor.compute_fund_metrics()
    # Default pod exists but has no snapshots, so NAV falls back to initial_capital
    assert metrics.fund_nav == 100_000.0
    assert metrics.num_active_pods == 1
    assert metrics.num_total_positions == 0


def test_meta_single_pod(pod_db):
    """With one pod snapshot, fund_nav equals that pod's NAV."""
    conn = pod_db
    today = datetime.now(tz=UTC).date()

    sid = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', 105000.0, 90000.0, 15000.0, 15000.0, 5000.0, 200.0)",
        [sid, today],
    )
    conn.commit()

    monitor = MetaRiskMonitor(conn)
    metrics = monitor.compute_fund_metrics()
    assert metrics.fund_nav == 105_000.0
    assert metrics.fund_daily_pnl == 200.0
    assert metrics.num_active_pods == 1


def test_meta_two_pods(two_pod_db):
    """With two pod snapshots, fund_nav is the sum of both."""
    conn = two_pod_db
    today = datetime.now(tz=UTC).date()

    # Default pod snapshot: NAV=110k
    sid1 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', 110000.0, 90000.0, 20000.0, 20000.0, 10000.0, 500.0)",
        [sid1, today],
    )

    # Benchmark pod snapshot: NAV=95k
    sid2 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'benchmark', 95000.0, 80000.0, "
        "15000.0, 15000.0, -5000.0, -100.0)",
        [sid2, today],
    )
    conn.commit()

    monitor = MetaRiskMonitor(conn)
    metrics = monitor.compute_fund_metrics()
    assert metrics.fund_nav == 110_000.0 + 95_000.0
    assert metrics.num_active_pods == 2
    assert metrics.fund_daily_pnl == 500.0 + (-100.0)


def test_fund_level_drawdown_alert(two_pod_db):
    """When fund drawdown > 20%, alerts list is non-empty."""
    conn = two_pod_db
    today = datetime.now(tz=UTC).date()
    yesterday = today - timedelta(days=1)

    # Day 1: peak NAV for both pods = 100k + 100k = 200k
    sid1 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', 100000.0, 100000.0, 0.0, 0.0, 0.0, 0.0)",
        [sid1, yesterday],
    )
    sid2 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'benchmark', 100000.0, 100000.0, 0.0, 0.0, 0.0, 0.0)",
        [sid2, yesterday],
    )

    # Day 2: both pods drop ~25% each => fund NAV 150k, peak was 200k => 25% drawdown
    sid3 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', 75000.0, 75000.0, 0.0, 0.0, -25000.0, -25000.0)",
        [sid3, today],
    )
    sid4 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'benchmark', 75000.0, 75000.0, 0.0, 0.0, -25000.0, -25000.0)",
        [sid4, today],
    )
    conn.commit()

    monitor = MetaRiskMonitor(conn)
    metrics = monitor.compute_fund_metrics()
    # fund_nav=150k, peak=200k => drawdown = -25%
    assert metrics.fund_drawdown_pct < -0.20
    assert len(metrics.alerts) > 0
    assert any("drawdown" in a.lower() for a in metrics.alerts)


def test_format_dashboard_runs(two_pod_db):
    """format_dashboard() doesn't crash on valid data."""
    conn = two_pod_db
    today = datetime.now(tz=UTC).date()

    sid1 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', 102000.0, 90000.0, 12000.0, 12000.0, 2000.0, 150.0)",
        [sid1, today],
    )
    sid2 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'benchmark', 99000.0, 85000.0, "
        "14000.0, 14000.0, -1000.0, -50.0)",
        [sid2, today],
    )
    conn.commit()

    monitor = MetaRiskMonitor(conn)
    dashboard = monitor.format_dashboard()
    assert isinstance(dashboard, str)
    assert "Fund NAV" in dashboard
    assert "default" in dashboard
    assert "benchmark" in dashboard

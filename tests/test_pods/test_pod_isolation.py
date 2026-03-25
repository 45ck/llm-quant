"""Tests for pod data isolation -- pods must not bleed into each other."""

from datetime import UTC, date, datetime

from llm_quant.trading.ledger import get_portfolio_history, get_recent_trades
from llm_quant.trading.portfolio import Portfolio, Position


def test_portfolio_from_db_isolation(two_pod_db):
    """Snapshots saved for different pods restore independently."""
    conn = two_pod_db
    today = date(2025, 6, 1)

    # Save snapshot for 'default' pod: NAV=100k, cash=90k
    p_default = Portfolio(initial_capital=100_000.0, pod_id="default")
    p_default.cash = 90_000.0
    p_default.positions["SPY"] = Position(
        symbol="SPY", shares=20, avg_cost=450.0, current_price=500.0
    )
    sid1 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', ?, ?, ?, ?, ?, ?)",
        [
            sid1,
            today,
            p_default.nav,
            p_default.cash,
            p_default.gross_exposure,
            p_default.net_exposure,
            p_default.total_pnl,
            0.0,
        ],
    )
    conn.execute(
        "INSERT INTO positions "
        "(snapshot_id, symbol, shares, avg_cost, "
        "current_price, market_value, unrealized_pnl, "
        "weight, stop_loss) "
        "VALUES (?, 'SPY', 20, 450.0, 500.0, 10000.0, 1000.0, 0.1, 0.0)",
        [sid1],
    )

    # Save snapshot for 'benchmark' pod: NAV=50k, cash=50k (no positions)
    sid2 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'benchmark', 50000.0, 50000.0, 0.0, 0.0, -50000.0, 0.0)",
        [sid2, today],
    )
    conn.commit()

    # Load each pod — they should get their own data
    restored_default = Portfolio.from_db(
        conn, initial_capital=100_000.0, pod_id="default"
    )
    assert restored_default.cash == 90_000.0
    assert "SPY" in restored_default.positions
    assert restored_default.pod_id == "default"

    restored_bench = Portfolio.from_db(
        conn, initial_capital=100_000.0, pod_id="benchmark"
    )
    assert restored_bench.cash == 50_000.0
    assert len(restored_bench.positions) == 0
    assert restored_bench.pod_id == "benchmark"


def test_trades_isolation(two_pod_db):
    """get_recent_trades(pod_id=X) only returns that pod's trades."""
    conn = two_pod_db
    today = date(2025, 6, 1)

    # Insert trade for 'default'
    tid1 = conn.execute("SELECT nextval('seq_trade_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO trades "
        "(trade_id, date, pod_id, symbol, action, "
        "shares, price, notional, conviction, reasoning) "
        "VALUES (?, ?, 'default', 'SPY', 'BUY', 10, 450.0, 4500.0, 'high', 'test')",
        [tid1, today],
    )

    # Insert trade for 'benchmark'
    tid2 = conn.execute("SELECT nextval('seq_trade_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO trades "
        "(trade_id, date, pod_id, symbol, action, "
        "shares, price, notional, conviction, reasoning) "
        "VALUES (?, ?, 'benchmark', 'TLT', 'BUY', 50, 95.0, 4750.0, 'medium', 'bench')",
        [tid2, today],
    )
    conn.commit()

    default_trades = get_recent_trades(conn, pod_id="default")
    assert len(default_trades) == 1
    assert default_trades[0]["symbol"] == "SPY"
    assert default_trades[0]["pod_id"] == "default"

    bench_trades = get_recent_trades(conn, pod_id="benchmark")
    assert len(bench_trades) == 1
    assert bench_trades[0]["symbol"] == "TLT"
    assert bench_trades[0]["pod_id"] == "benchmark"

    # Without pod_id filter, both trades returned
    all_trades = get_recent_trades(conn)
    assert len(all_trades) == 2


def test_snapshot_isolation(two_pod_db):
    """get_portfolio_history(pod_id=X) only returns that pod's snapshots."""
    conn = two_pod_db
    today = datetime.now(tz=UTC).date()

    # Insert snapshot for 'default'
    sid1 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'default', 100000.0, 90000.0, 10000.0, 10000.0, 0.0, 100.0)",
        [sid1, today],
    )

    # Insert snapshot for 'benchmark'
    sid2 = conn.execute("SELECT nextval('seq_snapshot_id')").fetchone()[0]
    conn.execute(
        "INSERT INTO portfolio_snapshots "
        "(snapshot_id, date, pod_id, nav, cash, "
        "gross_exposure, net_exposure, total_pnl, daily_pnl) "
        "VALUES (?, ?, 'benchmark', 50000.0, 50000.0, 0.0, 0.0, -50000.0, -200.0)",
        [sid2, today],
    )
    conn.commit()

    default_hist = get_portfolio_history(conn, days=30, pod_id="default")
    assert len(default_hist) == 1
    assert default_hist[0]["pod_id"] == "default"
    assert default_hist[0]["nav"] == 100_000.0

    bench_hist = get_portfolio_history(conn, days=30, pod_id="benchmark")
    assert len(bench_hist) == 1
    assert bench_hist[0]["pod_id"] == "benchmark"
    assert bench_hist[0]["nav"] == 50_000.0

    # Without filter, both returned
    all_hist = get_portfolio_history(conn, days=30)
    assert len(all_hist) == 2

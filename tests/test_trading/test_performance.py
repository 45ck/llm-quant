"""Tests for performance analytics module."""

from datetime import date, timedelta

import pytest

try:
    import empyrical as _empyrical
except ImportError:
    _empyrical = None

pytestmark = pytest.mark.skipif(_empyrical is None, reason="empyrical not installed")

from llm_quant.trading.performance import compute_performance


def _insert_snapshots(
    conn,
    start_date: date,
    navs: list[float],
) -> None:
    """Insert portfolio snapshots with sequential dates and NAV values."""
    for i, nav in enumerate(navs):
        snap_id = i + 1
        d = start_date + timedelta(days=i)
        daily_pnl = nav - navs[i - 1] if i > 0 else 0.0
        conn.execute(
            """
            INSERT INTO portfolio_snapshots
                (snapshot_id, date, nav, cash, gross_exposure,
                 net_exposure, total_pnl, daily_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snap_id,
                d,
                nav,
                nav * 0.5,
                nav * 0.5,
                nav * 0.5,
                nav - 100_000.0,
                daily_pnl,
            ],
        )


def _insert_positions(conn, snapshot_id: int, positions: list[dict]) -> None:
    """Insert positions for a given snapshot."""
    for pos in positions:
        conn.execute(
            """
            INSERT INTO positions
                (snapshot_id, symbol, shares, avg_cost, current_price,
                 market_value, unrealized_pnl, weight, stop_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_id,
                pos["symbol"],
                pos["shares"],
                pos["avg_cost"],
                pos["current_price"],
                pos["market_value"],
                pos["unrealized_pnl"],
                pos["weight"],
                pos.get("stop_loss", 0.0),
            ],
        )


def _insert_benchmark_data(
    conn,
    start_date: date,
    spy_prices: list[float],
    tlt_prices: list[float],
) -> None:
    """Insert SPY and TLT market data for benchmark calculation."""
    for i, (spy_p, tlt_p) in enumerate(zip(spy_prices, tlt_prices, strict=True)):
        d = start_date + timedelta(days=i)
        for symbol, price in [("SPY", spy_p), ("TLT", tlt_p)]:
            conn.execute(
                """
                INSERT INTO market_data_daily (symbol, date, close)
                VALUES (?, ?, ?)
                """,
                [symbol, d, price],
            )


# -- Tests -----------------------------------------------------------------


class TestComputePerformanceDefaults:
    """When no data exists, defaults are returned."""

    def test_empty_snapshots(self, tmp_db):
        metrics = compute_performance(tmp_db)
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] is None
        assert metrics["calmar_ratio"] is None
        assert metrics["annualized_return"] == 0.0
        assert metrics["benchmark_return"] is None
        assert metrics["excess_return"] is None
        assert metrics["daily_returns"] == []
        assert metrics["best_positions"] == []
        assert metrics["worst_positions"] == []
        assert metrics["total_trades"] == 0
        assert metrics["latest_nav"] == 100_000.0


class TestExistingMetrics:
    """Existing metrics still compute correctly after enhancement."""

    def test_total_return(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 102_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["total_return"] == pytest.approx(0.02, abs=1e-4)
        assert metrics["latest_nav"] == 102_000.0
        assert metrics["total_pnl"] == 2_000.0

    def test_single_snapshot(self, tmp_db):
        _insert_snapshots(tmp_db, date(2025, 1, 1), [100_000.0])
        metrics = compute_performance(tmp_db)
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0

    def test_max_drawdown(self, tmp_db):
        # NAV goes 100k -> 110k -> 99k -> 105k
        # peak=110k, drawdown from 110k to 99k = -10%
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 110_000.0, 99_000.0, 105_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["max_drawdown"] == pytest.approx(-0.1, abs=1e-4)


class TestSortinoRatio:
    """Sortino ratio uses only downside deviation."""

    def test_no_negative_returns(self, tmp_db):
        # All positive returns -> no downside deviation -> None
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 102_000.0, 103_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["sortino_ratio"] is None

    def test_with_negative_returns(self, tmp_db):
        # Mix of positive and negative returns
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 99_000.0, 100_500.0, 98_000.0, 101_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["sortino_ratio"] is not None
        # Sortino is a real number
        assert isinstance(metrics["sortino_ratio"], float)


class TestCalmarRatio:
    """Calmar ratio = annualized_return / abs(max_drawdown)."""

    def test_zero_drawdown(self, tmp_db):
        # Monotonically increasing NAV -> no drawdown -> None
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 102_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["calmar_ratio"] is None

    def test_with_drawdown(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 110_000.0, 99_000.0, 105_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["calmar_ratio"] is not None
        assert isinstance(metrics["calmar_ratio"], float)


class TestAnnualizedReturn:
    """Annualized return scales total return to 252 trading days."""

    def test_basic(self, tmp_db):
        # 3 data points = 3 trading days
        # total_return = 2%, annualized = (1.02)^(252/3) - 1
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 102_000.0],
        )
        metrics = compute_performance(tmp_db)
        expected = (1.02) ** (252 / 3) - 1.0
        assert metrics["annualized_return"] == pytest.approx(expected, rel=1e-3)

    def test_single_snapshot_is_zero(self, tmp_db):
        _insert_snapshots(tmp_db, date(2025, 1, 1), [100_000.0])
        metrics = compute_performance(tmp_db)
        assert metrics["annualized_return"] == 0.0


class TestBenchmarkReturn:
    """60/40 SPY/TLT benchmark calculation."""

    def test_with_market_data(self, tmp_db):
        start = date(2025, 1, 1)
        _insert_snapshots(tmp_db, start, [100_000.0, 101_000.0, 102_000.0])

        # SPY: 400 -> 410 = +2.5%, TLT: 100 -> 102 = +2%
        _insert_benchmark_data(
            tmp_db,
            start,
            spy_prices=[400.0, 405.0, 410.0],
            tlt_prices=[100.0, 101.0, 102.0],
        )
        metrics = compute_performance(tmp_db)
        # 0.6 * 0.025 + 0.4 * 0.02 = 0.015 + 0.008 = 0.023
        assert metrics["benchmark_return"] == pytest.approx(0.023, abs=1e-4)
        assert metrics["excess_return"] is not None
        expected_excess = metrics["total_return"] - metrics["benchmark_return"]
        assert metrics["excess_return"] == pytest.approx(expected_excess, abs=1e-4)

    def test_no_market_data(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 102_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["benchmark_return"] is None
        assert metrics["excess_return"] is None

    def test_single_snapshot_no_benchmark(self, tmp_db):
        _insert_snapshots(tmp_db, date(2025, 1, 1), [100_000.0])
        metrics = compute_performance(tmp_db)
        assert metrics["benchmark_return"] is None


class TestDailyReturns:
    """daily_returns should be a list of (date, float) tuples."""

    def test_populated(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 102_000.0],
        )
        metrics = compute_performance(tmp_db)
        dr = metrics["daily_returns"]
        assert len(dr) == 2  # n-1 returns for n snapshots
        for _d, r in dr:
            assert isinstance(r, float)

    def test_empty_for_single_snapshot(self, tmp_db):
        _insert_snapshots(tmp_db, date(2025, 1, 1), [100_000.0])
        metrics = compute_performance(tmp_db)
        assert metrics["daily_returns"] == []


class TestBestWorstPositions:
    """Top/bottom 3 positions by unrealized_pnl."""

    def test_with_positions(self, tmp_db):
        start = date(2025, 1, 1)
        _insert_snapshots(tmp_db, start, [100_000.0, 101_000.0])

        # Snapshot 2 is the latest (index 1 -> snapshot_id=2)
        _insert_positions(
            tmp_db,
            snapshot_id=2,
            positions=[
                {
                    "symbol": "SPY",
                    "shares": 10,
                    "avg_cost": 400.0,
                    "current_price": 420.0,
                    "market_value": 4200.0,
                    "unrealized_pnl": 200.0,
                    "weight": 0.4,
                },
                {
                    "symbol": "QQQ",
                    "shares": 5,
                    "avg_cost": 380.0,
                    "current_price": 370.0,
                    "market_value": 1850.0,
                    "unrealized_pnl": -50.0,
                    "weight": 0.2,
                },
                {
                    "symbol": "TLT",
                    "shares": 20,
                    "avg_cost": 100.0,
                    "current_price": 105.0,
                    "market_value": 2100.0,
                    "unrealized_pnl": 100.0,
                    "weight": 0.2,
                },
                {
                    "symbol": "GLD",
                    "shares": 8,
                    "avg_cost": 180.0,
                    "current_price": 175.0,
                    "market_value": 1400.0,
                    "unrealized_pnl": -40.0,
                    "weight": 0.1,
                },
            ],
        )

        metrics = compute_performance(tmp_db)
        best = metrics["best_positions"]
        worst = metrics["worst_positions"]

        assert len(best) == 3
        assert len(worst) == 3

        # Best should be ordered: SPY (200), TLT (100), then GLD(-40) or QQQ(-50)
        assert best[0]["symbol"] == "SPY"
        assert best[0]["unrealized_pnl"] == 200.0

        # Worst should have most negative first
        assert worst[0]["symbol"] == "QQQ"
        assert worst[0]["unrealized_pnl"] == -50.0

    def test_no_positions(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0],
        )
        metrics = compute_performance(tmp_db)
        assert metrics["best_positions"] == []
        assert metrics["worst_positions"] == []


class TestBackwardCompatibility:
    """All original keys must still be present."""

    def test_all_original_keys_present(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0, 99_000.0, 102_000.0],
        )
        metrics = compute_performance(tmp_db)
        original_keys = {
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
            "avg_trade_pnl",
            "latest_nav",
            "total_pnl",
        }
        assert original_keys.issubset(metrics.keys())

    def test_new_keys_present(self, tmp_db):
        _insert_snapshots(
            tmp_db,
            date(2025, 1, 1),
            [100_000.0, 101_000.0],
        )
        metrics = compute_performance(tmp_db)
        new_keys = {
            "sortino_ratio",
            "calmar_ratio",
            "annualized_return",
            "benchmark_return",
            "excess_return",
            "daily_returns",
            "best_positions",
            "worst_positions",
        }
        assert new_keys.issubset(metrics.keys())

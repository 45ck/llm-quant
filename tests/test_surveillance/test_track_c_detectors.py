"""Tests for Track C surveillance detectors — structural arbitrage kill switches.

At least 2 tests per detector (10+ total) covering OK, WARNING, and HALT
severity paths, plus edge cases like missing tables and insufficient data.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import duckdb

from llm_quant.config import AppConfig
from llm_quant.surveillance.models import SeverityLevel
from llm_quant.surveillance.track_c_detectors import (
    BetaDriftConfig,
    CrossTrackCorrelationConfig,
    ExchangeHealthConfig,
    FundingRateReversalConfig,
    SpreadCompressionConfig,
    check_beta_drift,
    check_cross_strategy_correlation,
    check_exchange_health,
    check_funding_rate_reversal,
    check_spread_compression,
)

# ---------------------------------------------------------------------------
# Helpers — create Track C tables and insert test data
# ---------------------------------------------------------------------------


def _create_exchange_events_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the track_c_exchange_events table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS track_c_exchange_events (
            event_type VARCHAR NOT NULL,
            occurred_at TIMESTAMP NOT NULL,
            exchange VARCHAR,
            details VARCHAR
        )
    """)
    conn.commit()


def _create_arb_spreads_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the track_c_arb_spreads table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS track_c_arb_spreads (
            recorded_at TIMESTAMP NOT NULL,
            spread_bps DOUBLE,
            strategy VARCHAR
        )
    """)
    conn.commit()


def _create_funding_rates_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the track_c_funding_rates table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS track_c_funding_rates (
            symbol VARCHAR NOT NULL,
            period_end TIMESTAMP NOT NULL,
            funding_rate DOUBLE
        )
    """)
    conn.commit()


def _create_track_c_daily_returns_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the track_c_daily_returns table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS track_c_daily_returns (
            date DATE NOT NULL,
            daily_return DOUBLE
        )
    """)
    conn.commit()


def _insert_exchange_events(
    conn: duckdb.DuckDBPyConnection,
    events: list[tuple[str, datetime, str]],
) -> None:
    """Insert exchange events: (event_type, occurred_at, exchange)."""
    for event_type, occurred_at, exchange in events:
        conn.execute(
            "INSERT INTO track_c_exchange_events "
            "(event_type, occurred_at, exchange, details) "
            "VALUES (?, ?, ?, '')",
            [event_type, occurred_at, exchange],
        )
    conn.commit()


def _insert_arb_spreads(
    conn: duckdb.DuckDBPyConnection,
    spreads: list[tuple[datetime, float]],
) -> None:
    """Insert arb spread records: (recorded_at, spread_bps)."""
    for recorded_at, spread_bps in spreads:
        conn.execute(
            "INSERT INTO track_c_arb_spreads "
            "(recorded_at, spread_bps, strategy) VALUES (?, ?, 'test')",
            [recorded_at, spread_bps],
        )
    conn.commit()


def _insert_funding_rates(
    conn: duckdb.DuckDBPyConnection,
    rates: list[tuple[str, datetime, float]],
) -> None:
    """Insert funding rates: (symbol, period_end, funding_rate)."""
    for symbol, period_end, rate in rates:
        conn.execute(
            "INSERT INTO track_c_funding_rates "
            "(symbol, period_end, funding_rate) VALUES (?, ?, ?)",
            [symbol, period_end, rate],
        )
    conn.commit()


def _insert_track_c_returns(
    conn: duckdb.DuckDBPyConnection,
    returns: list[tuple[date, float]],
) -> None:
    """Insert Track C daily returns: (date, daily_return)."""
    for d, ret in returns:
        conn.execute(
            "INSERT INTO track_c_daily_returns (date, daily_return) VALUES (?, ?)",
            [d, ret],
        )
    conn.commit()


def _insert_spy_prices(
    conn: duckdb.DuckDBPyConnection,
    prices: list[tuple[date, float]],
) -> None:
    """Insert SPY market data."""
    for d, close in prices:
        conn.execute(
            "INSERT INTO market_data_daily "
            "(symbol, date, open, high, low, close, volume) "
            "VALUES ('SPY', ?, ?, ?, ?, ?, 1000000)",
            [d, close, close * 1.01, close * 0.99, close],
        )
    conn.commit()


def _insert_portfolio_snapshots(
    conn: duckdb.DuckDBPyConnection,
    navs: list[tuple[date, float]],
) -> None:
    """Insert portfolio snapshots with NAV series."""
    for i, (d, nav) in enumerate(navs):
        daily_pnl = (nav - navs[i - 1][1]) if i > 0 else 0.0
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(snapshot_id, date, nav, cash, gross_exposure, net_exposure, "
            "total_pnl, daily_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [i + 1, d, nav, nav * 0.5, nav * 0.3, nav * 0.2, 0.0, daily_pnl],
        )
    conn.commit()


# ---------------------------------------------------------------------------
# A. Configuration Dataclass Tests
# ---------------------------------------------------------------------------


class TestConfigDataclasses:
    """Verify configuration dataclasses have correct defaults."""

    def test_exchange_health_config_defaults(self):
        cfg = ExchangeHealthConfig()
        assert cfg.error_threshold == 3
        assert cfg.lookback_hours == 1
        assert cfg.withdrawal_delay_lookback_hours == 24

    def test_spread_compression_config_defaults(self):
        cfg = SpreadCompressionConfig()
        assert cfg.halt_ratio == 0.25
        assert cfg.warn_ratio == 0.50
        assert cfg.rolling_window_days == 7
        assert cfg.baseline_window_days == 30

    def test_funding_rate_reversal_config_defaults(self):
        cfg = FundingRateReversalConfig()
        assert cfg.halt_consecutive == 3
        assert cfg.warn_consecutive == 2

    def test_beta_drift_config_defaults(self):
        cfg = BetaDriftConfig()
        assert cfg.window_days == 30
        assert cfg.halt_beta == 0.15
        assert cfg.warn_beta == 0.10

    def test_cross_track_correlation_config_defaults(self):
        cfg = CrossTrackCorrelationConfig()
        assert cfg.window_days == 30
        assert cfg.halt_corr == 0.30
        assert cfg.warn_corr == 0.20

    def test_config_override(self):
        cfg = ExchangeHealthConfig(error_threshold=5, lookback_hours=2)
        assert cfg.error_threshold == 5
        assert cfg.lookback_hours == 2


# ---------------------------------------------------------------------------
# B. ExchangeHealthDetector Tests
# ---------------------------------------------------------------------------


class TestCheckExchangeHealth:
    """Tests for check_exchange_health detector."""

    def test_no_table_returns_ok(self, tmp_db):
        """When the exchange events table does not exist, degrade gracefully."""
        config = AppConfig()
        checks = check_exchange_health(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "not yet initialised" in checks[0].message

    def test_no_errors_returns_ok(self, tmp_db):
        """When no API errors exist, return OK."""
        _create_exchange_events_table(tmp_db)
        now = datetime.now(tz=UTC)
        # Insert only successful events
        _insert_exchange_events(
            tmp_db,
            [
                ("api_success", now - timedelta(minutes=10), "kalshi"),
                ("api_success", now - timedelta(minutes=20), "kalshi"),
            ],
        )
        config = AppConfig()
        checks = check_exchange_health(tmp_db, config)
        api_checks = [c for c in checks if c.metric_name == "exchange_api_errors"]
        assert len(api_checks) == 1
        assert api_checks[0].severity == SeverityLevel.OK

    def test_consecutive_errors_triggers_halt(self, tmp_db):
        """4+ consecutive API errors should trigger HALT."""
        _create_exchange_events_table(tmp_db)
        now = datetime.now(tz=UTC)
        # Insert 5 consecutive API errors (most recent first in time)
        events = [("api_error", now - timedelta(minutes=i), "kalshi") for i in range(5)]
        _insert_exchange_events(tmp_db, events)

        config = AppConfig()
        checks = check_exchange_health(tmp_db, config)
        api_checks = [c for c in checks if c.metric_name == "exchange_api_errors"]
        assert len(api_checks) == 1
        assert api_checks[0].severity == SeverityLevel.HALT
        assert "KILL SWITCH" in api_checks[0].message
        assert api_checks[0].current_value == 5.0

    def test_errors_below_threshold_returns_ok(self, tmp_db):
        """2 consecutive errors (below threshold of 3) should return OK."""
        _create_exchange_events_table(tmp_db)
        now = datetime.now(tz=UTC)
        _insert_exchange_events(
            tmp_db,
            [
                ("api_error", now - timedelta(minutes=1), "kalshi"),
                ("api_error", now - timedelta(minutes=2), "kalshi"),
                ("api_success", now - timedelta(minutes=3), "kalshi"),
            ],
        )
        config = AppConfig()
        checks = check_exchange_health(tmp_db, config)
        api_checks = [c for c in checks if c.metric_name == "exchange_api_errors"]
        assert len(api_checks) == 1
        assert api_checks[0].severity == SeverityLevel.OK

    def test_withdrawal_delay_triggers_halt(self, tmp_db):
        """Any withdrawal delay event should trigger HALT."""
        _create_exchange_events_table(tmp_db)
        now = datetime.now(tz=UTC)
        _insert_exchange_events(
            tmp_db,
            [
                ("withdrawal_delay", now - timedelta(hours=2), "binance"),
            ],
        )
        config = AppConfig()
        checks = check_exchange_health(tmp_db, config)
        wd_checks = [c for c in checks if c.metric_name == "exchange_withdrawal_delays"]
        assert len(wd_checks) == 1
        assert wd_checks[0].severity == SeverityLevel.HALT
        assert "withdrawal delay" in wd_checks[0].message.lower()

    def test_no_withdrawal_delays_returns_ok(self, tmp_db):
        """No withdrawal delays should return OK."""
        _create_exchange_events_table(tmp_db)
        now = datetime.now(tz=UTC)
        _insert_exchange_events(
            tmp_db,
            [("api_success", now - timedelta(minutes=5), "kalshi")],
        )
        config = AppConfig()
        checks = check_exchange_health(tmp_db, config)
        wd_checks = [c for c in checks if c.metric_name == "exchange_withdrawal_delays"]
        assert len(wd_checks) == 1
        assert wd_checks[0].severity == SeverityLevel.OK

    def test_custom_config_threshold(self, tmp_db):
        """Custom error_threshold=5 should allow 4 errors without HALT."""
        _create_exchange_events_table(tmp_db)
        now = datetime.now(tz=UTC)
        # 4 consecutive errors
        events = [("api_error", now - timedelta(minutes=i), "kalshi") for i in range(4)]
        _insert_exchange_events(tmp_db, events)

        config = AppConfig()
        custom_cfg = ExchangeHealthConfig(error_threshold=5)
        checks = check_exchange_health(tmp_db, config, detector_config=custom_cfg)
        api_checks = [c for c in checks if c.metric_name == "exchange_api_errors"]
        assert len(api_checks) == 1
        assert api_checks[0].severity == SeverityLevel.OK


# ---------------------------------------------------------------------------
# C. SpreadCompressionDetector Tests
# ---------------------------------------------------------------------------


class TestCheckSpreadCompression:
    """Tests for check_spread_compression detector."""

    def test_no_table_returns_ok(self, tmp_db):
        """When the arb spreads table does not exist, degrade gracefully."""
        config = AppConfig()
        checks = check_spread_compression(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "not yet initialised" in checks[0].message

    def test_healthy_spread_returns_ok(self, tmp_db):
        """When 7d avg is near 30d baseline, return OK."""
        _create_arb_spreads_table(tmp_db)
        now = datetime.now(tz=UTC)
        # Insert 30 days of stable spreads around 100 bps
        spreads = [(now - timedelta(days=i), 100.0 + (i % 3) * 2) for i in range(30)]
        _insert_arb_spreads(tmp_db, spreads)

        config = AppConfig()
        checks = check_spread_compression(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "healthy" in checks[0].message.lower()

    def test_compressed_spread_triggers_halt(self, tmp_db):
        """When 7d avg < 25% of 30d baseline, trigger HALT."""
        _create_arb_spreads_table(tmp_db)
        now = datetime.now(tz=UTC)
        # 30d baseline: 100 bps (days 8-30)
        old_spreads = [(now - timedelta(days=i), 100.0) for i in range(8, 31)]
        # Recent 7d: 10 bps (10% of baseline, well below 25% halt)
        recent_spreads = [(now - timedelta(days=i), 10.0) for i in range(7)]
        _insert_arb_spreads(tmp_db, old_spreads + recent_spreads)

        config = AppConfig()
        checks = check_spread_compression(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.HALT
        assert "KILL SWITCH" in checks[0].message

    def test_moderately_compressed_spread_triggers_warning(self, tmp_db):
        """When 7d avg between 25-50% of baseline, trigger WARNING."""
        _create_arb_spreads_table(tmp_db)
        now = datetime.now(tz=UTC)
        # 30d baseline: 100 bps (days 8-30)
        old_spreads = [(now - timedelta(days=i), 100.0) for i in range(8, 31)]
        # Recent 7d: 35 bps (~35% of baseline — between halt=25% and warn=50%)
        recent_spreads = [(now - timedelta(days=i), 35.0) for i in range(7)]
        _insert_arb_spreads(tmp_db, old_spreads + recent_spreads)

        config = AppConfig()
        checks = check_spread_compression(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.WARNING
        assert "compressing" in checks[0].message.lower()

    def test_insufficient_history_returns_ok(self, tmp_db):
        """When there is no 30d spread history, return OK."""
        _create_arb_spreads_table(tmp_db)
        # Empty table — no spreads at all
        config = AppConfig()
        checks = check_spread_compression(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "Insufficient" in checks[0].message


# ---------------------------------------------------------------------------
# D. FundingRateReversalDetector Tests
# ---------------------------------------------------------------------------


class TestCheckFundingRateReversal:
    """Tests for check_funding_rate_reversal detector."""

    def test_no_table_returns_ok(self, tmp_db):
        """When the funding rates table does not exist, degrade gracefully."""
        config = AppConfig()
        checks = check_funding_rate_reversal(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "not yet initialised" in checks[0].message

    def test_positive_rates_returns_ok(self, tmp_db):
        """When all funding rates are positive, return OK."""
        _create_funding_rates_table(tmp_db)
        now = datetime.now(tz=UTC)
        rates = [("BTC-PERP", now - timedelta(hours=i * 8), 0.01) for i in range(5)]
        _insert_funding_rates(tmp_db, rates)

        config = AppConfig()
        checks = check_funding_rate_reversal(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "normal" in checks[0].message.lower()
        assert checks[0].current_value == 0.0

    def test_three_consecutive_negative_triggers_halt(self, tmp_db):
        """3 consecutive negative funding periods should trigger HALT."""
        _create_funding_rates_table(tmp_db)
        now = datetime.now(tz=UTC)
        rates = [
            ("BTC-PERP", now - timedelta(hours=0), -0.005),
            ("BTC-PERP", now - timedelta(hours=8), -0.003),
            ("BTC-PERP", now - timedelta(hours=16), -0.002),
            ("BTC-PERP", now - timedelta(hours=24), 0.01),  # breaks streak
        ]
        _insert_funding_rates(tmp_db, rates)

        config = AppConfig()
        checks = check_funding_rate_reversal(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.HALT
        assert "KILL SWITCH" in checks[0].message
        assert checks[0].current_value == 3.0
        assert checks[0].details.get("symbol") == "BTC-PERP"

    def test_two_consecutive_negative_triggers_warning(self, tmp_db):
        """2 consecutive negative periods should trigger WARNING."""
        _create_funding_rates_table(tmp_db)
        now = datetime.now(tz=UTC)
        rates = [
            ("ETH-PERP", now - timedelta(hours=0), -0.005),
            ("ETH-PERP", now - timedelta(hours=8), -0.003),
            ("ETH-PERP", now - timedelta(hours=16), 0.01),
        ]
        _insert_funding_rates(tmp_db, rates)

        config = AppConfig()
        checks = check_funding_rate_reversal(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.WARNING
        assert checks[0].current_value == 2.0

    def test_no_symbols_returns_ok(self, tmp_db):
        """Empty funding rates table should return OK."""
        _create_funding_rates_table(tmp_db)
        config = AppConfig()
        checks = check_funding_rate_reversal(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "No funding rate symbols" in checks[0].message

    def test_multiple_symbols_worst_streak_counts(self, tmp_db):
        """Detector should flag the worst symbol across all tracked."""
        _create_funding_rates_table(tmp_db)
        now = datetime.now(tz=UTC)
        # BTC: 1 negative (OK)
        rates = [
            ("BTC-PERP", now - timedelta(hours=0), -0.001),
            ("BTC-PERP", now - timedelta(hours=8), 0.01),
        ]
        # ETH: 4 negative (HALT — worst)
        rates += [("ETH-PERP", now - timedelta(hours=i * 8), -0.002) for i in range(4)]
        _insert_funding_rates(tmp_db, rates)

        config = AppConfig()
        checks = check_funding_rate_reversal(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.HALT
        assert checks[0].details.get("symbol") == "ETH-PERP"
        assert checks[0].current_value == 4.0


# ---------------------------------------------------------------------------
# E. BetaDriftDetector Tests
# ---------------------------------------------------------------------------


class TestCheckBetaDrift:
    """Tests for check_beta_drift detector."""

    def test_no_table_returns_ok(self, tmp_db):
        """When track_c_daily_returns table does not exist, degrade gracefully."""
        config = AppConfig()
        checks = check_beta_drift(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "not yet initialised" in checks[0].message

    def test_insufficient_data_returns_ok(self, tmp_db):
        """When not enough data for beta calc, return OK."""
        _create_track_c_daily_returns_table(tmp_db)
        today = datetime.now(tz=UTC).date()
        # Only 5 days of data (need 30)
        returns = [(today - timedelta(days=i), 0.001) for i in range(5)]
        _insert_track_c_returns(tmp_db, returns)

        config = AppConfig()
        checks = check_beta_drift(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "Insufficient" in checks[0].message

    def test_zero_beta_returns_ok(self, tmp_db):
        """Market-neutral returns (uncorrelated with SPY) should return OK."""
        _create_track_c_daily_returns_table(tmp_db)
        today = datetime.now(tz=UTC).date()

        # Use a shorter window for the test
        window = 20
        cfg = BetaDriftConfig(window_days=window)

        # Generate uncorrelated returns: Track C has constant tiny returns
        # while SPY has realistic variance from alternating up/down days.
        # Constant Track C returns produce zero covariance with any SPY
        # pattern, resulting in beta = 0.
        tc_returns = []
        spy_prices = []
        base_price = 450.0

        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            # Track C: constant return (zero covariance with anything)
            tc_ret = 0.001
            tc_returns.append((d, tc_ret))
            # SPY: alternating up/down to create realistic variance
            if i % 2 == 0:
                base_price += 3.0
            else:
                base_price -= 2.0
            spy_prices.append((d, base_price))

        _insert_track_c_returns(tmp_db, tc_returns)
        _insert_spy_prices(tmp_db, spy_prices)

        config = AppConfig()
        checks = check_beta_drift(tmp_db, config, detector_config=cfg)
        assert len(checks) == 1
        # With constant Track C returns, beta should be exactly zero
        assert checks[0].severity == SeverityLevel.OK
        assert checks[0].metric_name == "track_c_beta_to_spy"

    def test_high_beta_triggers_halt(self, tmp_db):
        """When Track C returns track SPY closely, beta > 0.15 triggers HALT."""
        _create_track_c_daily_returns_table(tmp_db)
        today = datetime.now(tz=UTC).date()

        window = 20
        cfg = BetaDriftConfig(window_days=window, halt_beta=0.15)

        # Generate highly correlated returns: Track C moves with SPY
        tc_returns = []
        spy_prices = []
        base_price = 450.0

        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            # SPY has a clear trend
            spy_close = base_price + i * 2.0
            spy_prices.append((d, spy_close))

        _insert_spy_prices(tmp_db, spy_prices)

        # Track C returns that correlate strongly with SPY returns
        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            if i > 0:
                spy_ret = (spy_prices[i][1] - spy_prices[i - 1][1]) / spy_prices[i - 1][
                    1
                ]
                # Scale SPY return to create beta > 0.15
                tc_ret = spy_ret * 0.5
            else:
                tc_ret = 0.001
            tc_returns.append((d, tc_ret))

        _insert_track_c_returns(tmp_db, tc_returns)

        config = AppConfig()
        checks = check_beta_drift(tmp_db, config, detector_config=cfg)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.HALT
        assert "KILL SWITCH" in checks[0].message
        assert checks[0].current_value >= 0.15


# ---------------------------------------------------------------------------
# F. CrossTrackCorrelationDetector Tests
# ---------------------------------------------------------------------------


class TestCheckCrossStrategyCorrelation:
    """Tests for check_cross_strategy_correlation detector."""

    def test_no_tables_returns_ok(self, tmp_db):
        """When required tables do not exist, degrade gracefully."""
        config = AppConfig()
        checks = check_cross_strategy_correlation(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        # portfolio_snapshots exists (from schema init) but
        # track_c_daily_returns does not
        assert "not yet initialised" in checks[0].message

    def test_insufficient_data_returns_ok(self, tmp_db):
        """When not enough aligned data, return OK."""
        _create_track_c_daily_returns_table(tmp_db)
        today = datetime.now(tz=UTC).date()

        # Only a few days of data
        tc_returns = [(today - timedelta(days=i), 0.001) for i in range(5)]
        _insert_track_c_returns(tmp_db, tc_returns)

        navs = [(today - timedelta(days=i), 100_000 + i * 100) for i in range(5)]
        _insert_portfolio_snapshots(tmp_db, navs)

        config = AppConfig()
        checks = check_cross_strategy_correlation(tmp_db, config)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "Insufficient" in checks[0].message

    def test_low_correlation_returns_ok(self, tmp_db):
        """Uncorrelated Track C and Track A returns should return OK."""
        _create_track_c_daily_returns_table(tmp_db)
        today = datetime.now(tz=UTC).date()
        window = 20
        cfg = CrossTrackCorrelationConfig(window_days=window)

        # Track C: alternating returns (mean-reverting arb)
        tc_returns = []
        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            tc_ret = 0.002 * (1 if i % 2 == 0 else -1)
            tc_returns.append((d, tc_ret))
        _insert_track_c_returns(tmp_db, tc_returns)

        # Track A: steady uptrend (different pattern)
        navs = []
        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            nav = 100_000 + i * 200
            navs.append((d, nav))
        _insert_portfolio_snapshots(tmp_db, navs)

        config = AppConfig()
        checks = check_cross_strategy_correlation(tmp_db, config, detector_config=cfg)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.OK
        assert "diversification intact" in checks[0].message

    def test_high_correlation_triggers_halt(self, tmp_db):
        """When Track C and Track A are highly correlated, trigger HALT."""
        _create_track_c_daily_returns_table(tmp_db)
        today = datetime.now(tz=UTC).date()
        window = 20
        cfg = CrossTrackCorrelationConfig(
            window_days=window, halt_corr=0.30, warn_corr=0.20
        )

        # Both tracks with highly correlated returns
        # Track A: NAV with consistent uptrend
        navs = []
        base_nav = 100_000.0
        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            # Deterministic pattern for Track A
            nav = base_nav + i * 300 + (i % 3) * 100
            navs.append((d, nav))
        _insert_portfolio_snapshots(tmp_db, navs)

        # Compute Track A returns, then make Track C mirror them
        tc_returns = []
        for i in range(window + 6):
            d = today - timedelta(days=window + 5 - i)
            if i > 0:
                ta_ret = (navs[i][1] - navs[i - 1][1]) / navs[i - 1][1]
                # Track C mimics Track A with slight scaling
                tc_ret = ta_ret * 0.8 + 0.0001
            else:
                tc_ret = 0.003
            tc_returns.append((d, tc_ret))
        _insert_track_c_returns(tmp_db, tc_returns)

        config = AppConfig()
        checks = check_cross_strategy_correlation(tmp_db, config, detector_config=cfg)
        assert len(checks) == 1
        assert checks[0].severity == SeverityLevel.HALT
        assert "KILL SWITCH" in checks[0].message
        assert checks[0].current_value > 0.30

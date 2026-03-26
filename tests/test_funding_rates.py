"""Tests for crypto funding rate pipeline.

Covers:
  - Rate annualization math
  - Opportunity detection (high rate + cross-exchange)
  - Mock CCXT responses
  - DuckDB persistence
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import duckdb
import pytest

from llm_quant.arb.funding_rates import (
    PERIODS_PER_YEAR,
    FundingCollector,
    FundingRecord,
    annualize_funding_rate,
    get_funding_connection,
    init_funding_schema,
    load_rates,
    persist_records,
)
from llm_quant.arb.funding_scanner import (
    FundingScanner,
    format_scan_report,
    rates_to_polars,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_records() -> list[FundingRecord]:
    """Records with varied rates across exchanges."""
    return [
        FundingRecord(
            timestamp=NOW,
            exchange="binance",
            symbol="BTC/USDT",
            funding_rate=0.0003,  # 0.03% per 8h
            annualized_rate=annualize_funding_rate(0.0003),
            mark_price=87000.0,
        ),
        FundingRecord(
            timestamp=NOW,
            exchange="okx",
            symbol="BTC/USDT",
            funding_rate=0.0001,  # 0.01% per 8h
            annualized_rate=annualize_funding_rate(0.0001),
            mark_price=87010.0,
        ),
        FundingRecord(
            timestamp=NOW,
            exchange="bybit",
            symbol="BTC/USDT",
            funding_rate=-0.0002,  # negative: shorts pay longs
            annualized_rate=annualize_funding_rate(-0.0002),
            mark_price=86990.0,
        ),
        FundingRecord(
            timestamp=NOW,
            exchange="binance",
            symbol="ETH/USDT",
            funding_rate=0.00005,  # below default threshold
            annualized_rate=annualize_funding_rate(0.00005),
            mark_price=2050.0,
        ),
        FundingRecord(
            timestamp=NOW,
            exchange="binance",
            symbol="SOL/USDT",
            funding_rate=0.0005,  # 0.05% — high
            annualized_rate=annualize_funding_rate(0.0005),
            mark_price=135.0,
        ),
    ]


@pytest.fixture
def db_conn(tmp_path) -> duckdb.DuckDBPyConnection:
    """Create a temporary DuckDB connection with funding schema."""
    return get_funding_connection(tmp_path / "test_funding.db")


# ------------------------------------------------------------------
# Annualization math
# ------------------------------------------------------------------


class TestAnnualization:
    def test_positive_rate(self):
        rate = 0.0001  # 0.01% per 8h
        expected = 0.0001 * 3 * 365
        assert annualize_funding_rate(rate) == pytest.approx(expected)

    def test_negative_rate(self):
        rate = -0.0002
        expected = -0.0002 * 3 * 365
        assert annualize_funding_rate(rate) == pytest.approx(expected)

    def test_zero_rate(self):
        assert annualize_funding_rate(0.0) == 0.0

    def test_typical_high_rate(self):
        """0.01% per 8h should be ~10.95% annualized."""
        rate = 0.0001
        ann = annualize_funding_rate(rate)
        assert abs(ann - 0.1095) < 0.001

    def test_periods_per_year(self):
        assert PERIODS_PER_YEAR == 1095


# ------------------------------------------------------------------
# Opportunity detection — high rates
# ------------------------------------------------------------------


class TestHighRateDetection:
    def test_detects_above_threshold(self, sample_records):
        scanner = FundingScanner(rate_threshold=0.0001)
        opps = scanner.scan_high_rates(sample_records)
        # BTC binance (0.03%), BTC okx (0.01%), BTC bybit (-0.02%), SOL binance (0.05%)
        symbols = [(o.exchange, o.symbol) for o in opps]
        assert ("binance", "BTC/USDT") in symbols
        assert ("okx", "BTC/USDT") in symbols
        assert ("bybit", "BTC/USDT") in symbols
        assert ("binance", "SOL/USDT") in symbols
        # ETH at 0.005% is below 0.01% threshold
        assert ("binance", "ETH/USDT") not in symbols

    def test_sorted_by_absolute_rate(self, sample_records):
        scanner = FundingScanner(rate_threshold=0.0001)
        opps = scanner.scan_high_rates(sample_records)
        rates = [abs(o.annualized_rate) for o in opps]
        assert rates == sorted(rates, reverse=True)

    def test_no_opps_with_high_threshold(self, sample_records):
        scanner = FundingScanner(rate_threshold=0.01)  # 1% per 8h — very high
        opps = scanner.scan_high_rates(sample_records)
        assert len(opps) == 0

    def test_all_pass_with_zero_threshold(self, sample_records):
        scanner = FundingScanner(rate_threshold=0.0)
        opps = scanner.scan_high_rates(sample_records)
        # ETH has rate=0.00005, abs > 0
        assert len(opps) == len(sample_records)

    def test_opportunity_fields(self, sample_records):
        scanner = FundingScanner(rate_threshold=0.0001)
        opps = scanner.scan_high_rates(sample_records)
        sol_opp = next(o for o in opps if o.symbol == "SOL/USDT")
        assert sol_opp.opp_type == "high_rate"
        assert sol_opp.exchange == "binance"
        assert sol_opp.mark_price == 135.0
        assert sol_opp.funding_rate == 0.0005


# ------------------------------------------------------------------
# Opportunity detection — cross-exchange
# ------------------------------------------------------------------


class TestCrossExchangeDetection:
    def test_detects_btc_differential(self, sample_records):
        scanner = FundingScanner(differential_threshold=0.00005)
        opps = scanner.scan_cross_exchange(sample_records)
        # BTC has 3 exchanges with different rates, so 3 pairs:
        # binance(0.03%) vs okx(0.01%), binance(0.03%) vs bybit(-0.02%),
        # okx(0.01%) vs bybit(-0.02%)
        btc_opps = [o for o in opps if o.symbol == "BTC/USDT"]
        assert len(btc_opps) == 3

    def test_high_rate_exchange_listed_first(self, sample_records):
        scanner = FundingScanner(differential_threshold=0.00005)
        opps = scanner.scan_cross_exchange(sample_records)
        # binance has highest BTC rate, should be listed first for its pairs
        binance_okx = [
            o
            for o in opps
            if o.symbol == "BTC/USDT"
            and o.exchange == "binance"
            and o.counter_exchange == "okx"
        ]
        assert len(binance_okx) == 1
        assert binance_okx[0].differential is not None
        assert binance_okx[0].differential > 0

    def test_no_cross_exchange_for_single_exchange_symbol(self, sample_records):
        scanner = FundingScanner(differential_threshold=0.00005)
        opps = scanner.scan_cross_exchange(sample_records)
        # SOL and ETH only on binance, no cross-exchange possible
        sol_opps = [o for o in opps if o.symbol == "SOL/USDT"]
        assert len(sol_opps) == 0

    def test_differential_annualized(self, sample_records):
        scanner = FundingScanner(differential_threshold=0.00005)
        opps = scanner.scan_cross_exchange(sample_records)
        # binance-okx BTC: 0.0003 - 0.0001 = 0.0002 differential
        binance_okx = [
            o for o in opps if o.exchange == "binance" and o.counter_exchange == "okx"
        ]
        assert len(binance_okx) == 1
        assert binance_okx[0].differential == pytest.approx(0.0002)
        assert binance_okx[0].differential_annualized == pytest.approx(0.0002 * 3 * 365)


# ------------------------------------------------------------------
# Scanner scan_all
# ------------------------------------------------------------------


class TestScanAll:
    def test_combines_both_types(self, sample_records):
        scanner = FundingScanner(
            rate_threshold=0.0001,
            differential_threshold=0.00005,
        )
        opps = scanner.scan_all(sample_records)
        types = {o.opp_type for o in opps}
        assert "high_rate" in types
        assert "cross_exchange" in types


# ------------------------------------------------------------------
# DuckDB persistence
# ------------------------------------------------------------------


class TestPersistence:
    def test_persist_and_load(self, db_conn, sample_records):
        count = persist_records(db_conn, sample_records)
        assert count == len(sample_records)

        df = load_rates(db_conn)
        assert len(df) == len(sample_records)

    def test_persist_deduplicates(self, db_conn, sample_records):
        persist_records(db_conn, sample_records)
        # Insert again — should not duplicate
        persist_records(db_conn, sample_records)

        df = load_rates(db_conn)
        assert len(df) == len(sample_records)

    def test_load_with_symbol_filter(self, db_conn, sample_records):
        persist_records(db_conn, sample_records)

        df = load_rates(db_conn, symbol="BTC/USDT")
        assert len(df) == 3  # binance, okx, bybit

    def test_load_with_exchange_filter(self, db_conn, sample_records):
        persist_records(db_conn, sample_records)

        df = load_rates(db_conn, exchange="binance")
        assert len(df) == 3  # BTC, ETH, SOL

    def test_schema_creation(self, tmp_path):
        conn = duckdb.connect(str(tmp_path / "fresh.db"))
        init_funding_schema(conn)
        # Table should exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        assert "funding_rates" in tables
        conn.close()

    def test_empty_persist(self, db_conn):
        count = persist_records(db_conn, [])
        assert count == 0


# ------------------------------------------------------------------
# Mock CCXT tests
# ------------------------------------------------------------------


class TestMockCCXT:
    def test_fetch_current_rates_success(self):
        mock_exchange = MagicMock()
        mock_exchange.id = "binance"
        mock_exchange.markets = {"BTC/USDT:USDT": {}}
        mock_exchange.load_markets.return_value = None
        mock_exchange.fetch_funding_rate.return_value = {
            "fundingRate": 0.0002,
            "fundingTimestamp": 1711540800000,
            "markPrice": 87000.0,
            "timestamp": 1711540800000,
        }

        collector = FundingCollector(
            exchanges=["binance"],
            symbols=["BTC/USDT:USDT"],
            api_delay=0,
        )

        with patch.object(collector, "_create_exchange", return_value=mock_exchange):
            records = collector.fetch_current_rates()

        assert len(records) == 1
        assert records[0].funding_rate == 0.0002
        assert records[0].exchange == "binance"
        assert records[0].symbol == "BTC/USDT"
        assert records[0].annualized_rate == pytest.approx(0.0002 * 1095)

    def test_fetch_current_rates_handles_failure(self):
        mock_exchange = MagicMock()
        mock_exchange.id = "binance"
        mock_exchange.markets = {"BTC/USDT:USDT": {}}
        mock_exchange.load_markets.return_value = None
        mock_exchange.fetch_funding_rate.side_effect = Exception("Network error")

        collector = FundingCollector(
            exchanges=["binance"],
            symbols=["BTC/USDT:USDT"],
            api_delay=0,
        )

        with patch.object(collector, "_create_exchange", return_value=mock_exchange):
            records = collector.fetch_current_rates()

        assert len(records) == 0  # Graceful failure

    def test_fetch_history_not_supported(self):
        """Exchanges that don't support historical rates should not crash."""
        import ccxt as ccxt_mod

        mock_exchange = MagicMock()
        mock_exchange.id = "bybit"
        mock_exchange.markets = {"BTC/USDT:USDT": {}}
        mock_exchange.load_markets.return_value = None
        mock_exchange.fetch_funding_rate_history.side_effect = ccxt_mod.NotSupported(
            "Not supported"
        )

        collector = FundingCollector(
            exchanges=["bybit"],
            symbols=["BTC/USDT:USDT"],
            api_delay=0,
        )

        with patch.object(collector, "_create_exchange", return_value=mock_exchange):
            records = collector.fetch_history(days=7)

        assert len(records) == 0

    def test_symbol_normalization(self):
        mock_exchange = MagicMock()
        mock_exchange.id = "okx"
        mock_exchange.markets = {"BTC/USDT:USDT": {}}
        mock_exchange.load_markets.return_value = None

        collector = FundingCollector(api_delay=0)
        result = collector._normalize_symbol("BTC/USDT:USDT", mock_exchange)
        assert result == "BTC/USDT:USDT"

    def test_symbol_not_found(self):
        mock_exchange = MagicMock()
        mock_exchange.id = "okx"
        mock_exchange.markets = {"ETH/USDT:USDT": {}}
        mock_exchange.load_markets.return_value = None

        collector = FundingCollector(api_delay=0)
        result = collector._normalize_symbol("BTC/USDT:USDT", mock_exchange)
        assert result is None

    def test_unsupported_exchange(self):
        """Non-existent exchange should be skipped gracefully."""
        collector = FundingCollector(
            exchanges=["nonexistent_exchange_xyz"],
            symbols=["BTC/USDT:USDT"],
            api_delay=0,
        )
        records = collector.fetch_current_rates()
        assert len(records) == 0


# ------------------------------------------------------------------
# Report formatting
# ------------------------------------------------------------------


class TestReportFormatting:
    def test_format_empty_report(self):
        report = format_scan_report([], [])
        assert "FUNDING RATE SCANNER" in report
        assert "No rates above threshold" in report

    def test_format_with_high_rates(self, sample_records):
        scanner = FundingScanner(rate_threshold=0.0001)
        opps = scanner.scan_high_rates(sample_records)
        report = format_scan_report(opps, [])
        assert "HIGH FUNDING RATES" in report
        assert "BTC/USDT" in report

    def test_format_with_cross_exchange(self, sample_records):
        scanner = FundingScanner(differential_threshold=0.00005)
        cross = scanner.scan_cross_exchange(sample_records)
        report = format_scan_report([], cross)
        assert "CROSS-EXCHANGE DIFFERENTIALS" in report


# ------------------------------------------------------------------
# Polars conversion
# ------------------------------------------------------------------


class TestPolarsConversion:
    def test_records_to_polars(self, sample_records):
        df = rates_to_polars(sample_records)
        assert len(df) == len(sample_records)
        assert set(df.columns) == {
            "timestamp",
            "exchange",
            "symbol",
            "funding_rate",
            "annualized_rate",
            "mark_price",
        }

    def test_empty_to_polars(self):
        df = rates_to_polars([])
        assert len(df) == 0
        assert "funding_rate" in df.columns

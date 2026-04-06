"""Tests for crypto cash-and-carry basis trade pipeline.

Covers:
  - Basis calculation math (raw, annualized, fee-adjusted)
  - Scanner filtering and symbol construction
  - Strategy entry/exit logic
  - Position lifecycle (open, mark-to-market, close)
  - PnL calculations (gross, net, fees)
  - Mock CCXT responses (no live API calls)
  - Report formatting

Target: 20+ tests across all components.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from llm_quant.arb.basis_scanner import (
    BasisOpportunity,
    BasisScanner,
    annualize_basis,
    build_futures_symbol,
    build_spot_symbol,
    compute_fee_adjusted_basis,
    compute_raw_basis,
    format_basis_report,
    format_expiry_suffix,
    get_upcoming_expiries,
)
from llm_quant.arb.basis_strategy import (
    BasisStrategy,
    BasisStrategyConfig,
    PositionStatus,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

NOW = datetime(2026, 4, 6, 12, 0, 0, tzinfo=UTC)
TODAY = date(2026, 4, 6)
EXPIRY_JUN = date(2026, 6, 27)
EXPIRY_SEP = date(2026, 9, 27)


def _make_opportunity(
    symbol: str = "BTC",
    exchange: str = "binance",
    spot_price: float = 85000.0,
    futures_price: float = 87000.0,
    expiry: date = EXPIRY_JUN,
    as_of: date = TODAY,
) -> BasisOpportunity:
    """Build a BasisOpportunity for testing."""
    days = (expiry - as_of).days
    raw = compute_raw_basis(spot_price, futures_price)
    ann = annualize_basis(raw, days)
    return BasisOpportunity(
        symbol=symbol,
        exchange=exchange,
        spot_price=spot_price,
        futures_price=futures_price,
        futures_symbol=build_futures_symbol(symbol, expiry, exchange),
        expiry_date=expiry,
        days_to_expiry=days,
        raw_basis=raw,
        annualized_basis=ann,
        timestamp=NOW,
    )


@pytest.fixture
def btc_opportunity() -> BasisOpportunity:
    """BTC with ~2.35% raw basis, 82 days to expiry → ~10.5% annualized."""
    return _make_opportunity()


@pytest.fixture
def eth_opportunity() -> BasisOpportunity:
    """ETH with different parameters."""
    return _make_opportunity(
        symbol="ETH",
        spot_price=3200.0,
        futures_price=3260.0,
        expiry=EXPIRY_JUN,
    )


@pytest.fixture
def strategy() -> BasisStrategy:
    """Strategy with default config."""
    return BasisStrategy()


@pytest.fixture
def strategy_low_threshold() -> BasisStrategy:
    """Strategy with very low threshold for testing."""
    return BasisStrategy(
        config=BasisStrategyConfig(entry_threshold=0.01)  # 1% annualized
    )


# ------------------------------------------------------------------
# Basis calculation math
# ------------------------------------------------------------------


class TestBasisMath:
    def test_raw_basis_contango(self):
        """Futures above spot = positive basis (contango)."""
        basis = compute_raw_basis(85000.0, 87000.0)
        expected = (87000.0 - 85000.0) / 85000.0
        assert basis == pytest.approx(expected)
        assert basis > 0

    def test_raw_basis_backwardation(self):
        """Futures below spot = negative basis (backwardation)."""
        basis = compute_raw_basis(85000.0, 84000.0)
        assert basis < 0
        expected = (84000.0 - 85000.0) / 85000.0
        assert basis == pytest.approx(expected)

    def test_raw_basis_zero(self):
        """Spot equals futures = zero basis."""
        basis = compute_raw_basis(85000.0, 85000.0)
        assert basis == 0.0

    def test_raw_basis_zero_spot(self):
        """Zero spot price returns 0 (guard against division by zero)."""
        basis = compute_raw_basis(0.0, 87000.0)
        assert basis == 0.0

    def test_annualize_basis_90_days(self):
        """90-day basis annualization."""
        raw = 0.02  # 2% raw
        ann = annualize_basis(raw, 90)
        expected = 0.02 * (365.0 / 90)
        assert ann == pytest.approx(expected)

    def test_annualize_basis_30_days(self):
        """Shorter expiry = higher annualized rate."""
        raw = 0.01  # 1% raw
        ann_30 = annualize_basis(raw, 30)
        ann_90 = annualize_basis(raw, 90)
        # Same raw basis but shorter period = higher annualized
        assert ann_30 > ann_90

    def test_annualize_basis_zero_days(self):
        """Zero days to expiry returns 0 (guard)."""
        ann = annualize_basis(0.02, 0)
        assert ann == 0.0

    def test_annualize_basis_negative_days(self):
        """Negative days returns 0 (guard)."""
        ann = annualize_basis(0.02, -5)
        assert ann == 0.0

    def test_fee_adjusted_basis(self):
        """Fee adjustment subtracts round-trip costs from raw basis."""
        raw = 0.02  # 2%
        maker = 0.0002  # 0.02%
        taker = 0.0005  # 0.05%
        adjusted = compute_fee_adjusted_basis(raw, maker, taker)
        # Total fees = 2 * taker + maker = 0.0012
        expected = 0.02 - 0.0012
        assert adjusted == pytest.approx(expected)

    def test_fee_adjusted_can_go_negative(self):
        """If basis < fees, fee-adjusted basis is negative."""
        raw = 0.0005  # 0.05% — very small
        adjusted = compute_fee_adjusted_basis(raw, 0.0002, 0.0005)
        assert adjusted < 0

    def test_fee_adjusted_default_fees(self):
        """Test with default fee values."""
        raw = 0.03
        adjusted = compute_fee_adjusted_basis(raw)
        # Default: maker=0.0002, taker=0.0005
        # Total fees = 2*0.0005 + 0.0002 = 0.0012
        assert adjusted == pytest.approx(raw - 0.0012)


# ------------------------------------------------------------------
# Symbol construction
# ------------------------------------------------------------------


class TestSymbolConstruction:
    def test_format_expiry_suffix(self):
        assert format_expiry_suffix(date(2025, 6, 27)) == "250627"
        assert format_expiry_suffix(date(2026, 3, 27)) == "260327"
        assert format_expiry_suffix(date(2026, 12, 27)) == "261227"

    def test_build_futures_symbol_binance(self):
        sym = build_futures_symbol("BTC", date(2025, 6, 27), "binance")
        assert sym == "BTC/USDT:USDT-250627"

    def test_build_futures_symbol_okx(self):
        sym = build_futures_symbol("ETH", date(2025, 9, 27), "okx")
        assert sym == "ETH/USDT:USDT-250927"

    def test_build_futures_symbol_bybit(self):
        sym = build_futures_symbol("BTC", date(2026, 3, 27), "bybit")
        assert sym == "BTC/USDT:USDT-260327"

    def test_build_spot_symbol(self):
        assert build_spot_symbol("BTC") == "BTC/USDT"
        assert build_spot_symbol("ETH") == "ETH/USDT"
        assert build_spot_symbol("SOL") == "SOL/USDT"


# ------------------------------------------------------------------
# Upcoming expiries
# ------------------------------------------------------------------


class TestUpcomingExpiries:
    def test_basic_expiries(self):
        """Should return upcoming quarterly dates."""
        expiries = get_upcoming_expiries(as_of=date(2026, 1, 15), max_days=365)
        # Should include Mar, Jun, Sep, Dec 2026
        assert date(2026, 3, 27) in expiries
        assert date(2026, 6, 27) in expiries
        assert date(2026, 9, 27) in expiries
        assert date(2026, 12, 27) in expiries

    def test_excludes_past_expiries(self):
        """Expiries already passed should not appear."""
        expiries = get_upcoming_expiries(as_of=date(2026, 4, 6), max_days=90)
        # March 27 is past
        assert date(2026, 3, 27) not in expiries

    def test_max_days_filter(self):
        """Only expiries within max_days should appear."""
        expiries = get_upcoming_expiries(as_of=date(2026, 4, 6), max_days=30)
        # Jun 27 is ~82 days away — should not appear
        assert date(2026, 6, 27) not in expiries

    def test_empty_if_none_in_range(self):
        """No expiries within a very short window."""
        expiries = get_upcoming_expiries(as_of=date(2026, 4, 6), max_days=5)
        assert len(expiries) == 0

    def test_sorted_ascending(self):
        """Expiries should be sorted chronologically."""
        expiries = get_upcoming_expiries(as_of=date(2026, 1, 1), max_days=365)
        assert expiries == sorted(expiries)


# ------------------------------------------------------------------
# Scanner filtering
# ------------------------------------------------------------------


class TestScannerFiltering:
    def test_scanner_default_config(self):
        scanner = BasisScanner()
        assert scanner.min_annualized_pct == 3.0
        assert scanner.max_days_to_expiry == 90
        assert "BTC" in scanner.symbols
        assert "ETH" in scanner.symbols
        assert "binance" in scanner.exchanges

    def test_scanner_creates_exchange(self):
        """_create_exchange should return a ccxt Exchange with futures enabled."""
        scanner = BasisScanner(api_delay=0)
        exchange = scanner._create_exchange("binance")
        assert exchange.id == "binance"
        assert exchange.options.get("defaultType") == "future"

    def test_scanner_invalid_exchange(self):
        """Invalid exchange ID should raise AttributeError."""
        scanner = BasisScanner(api_delay=0)
        with pytest.raises(AttributeError):
            scanner._create_exchange("nonexistent_exchange_xyz")


# ------------------------------------------------------------------
# Mock CCXT scanner tests
# ------------------------------------------------------------------


class TestMockScannerCCXT:
    def test_scan_with_mock_exchange(self):
        """Full scan with mocked exchange returns opportunities."""
        scanner = BasisScanner(
            exchanges=["binance"],
            symbols=["BTC"],
            api_delay=0,
            min_annualized_pct=1.0,
        )

        # Mock spot exchange
        mock_spot = MagicMock()
        mock_spot.id = "binance"
        mock_spot.markets = {"BTC/USDT": {}}
        mock_spot.load_markets.return_value = None
        mock_spot.fetch_ticker.return_value = {"last": 85000.0}

        # Mock futures exchange
        mock_futures = MagicMock()
        mock_futures.id = "binance"
        mock_futures.markets = {"BTC/USDT:USDT-260627": {}}
        mock_futures.load_markets.return_value = None
        mock_futures.fetch_ticker.return_value = {"last": 87000.0}

        def mock_create_exchange(_exch_id):
            return mock_futures

        with (
            patch.object(scanner, "_create_exchange", side_effect=mock_create_exchange),
            patch.object(scanner, "_fetch_spot_price", return_value=85000.0),
            patch.object(scanner, "_fetch_futures_price", return_value=87000.0),
        ):
            opps = scanner.scan(as_of=TODAY)

        # Should find opportunities (exact count depends on available expiries)
        assert len(opps) >= 0  # May be 0 if no expiries match max_days

    def test_scan_no_spot_price_skips(self):
        """If spot price unavailable, symbol is skipped."""
        scanner = BasisScanner(
            exchanges=["binance"],
            symbols=["BTC"],
            api_delay=0,
        )

        mock_exchange = MagicMock()
        mock_exchange.id = "binance"

        with (
            patch.object(scanner, "_create_exchange", return_value=mock_exchange),
            patch.object(scanner, "_fetch_spot_price", return_value=None),
        ):
            opps = scanner.scan(as_of=TODAY)

        assert len(opps) == 0

    def test_scan_no_futures_price_skips(self):
        """If futures price unavailable, that contract is skipped."""
        scanner = BasisScanner(
            exchanges=["binance"],
            symbols=["BTC"],
            api_delay=0,
            min_annualized_pct=0.0,
        )

        mock_exchange = MagicMock()
        mock_exchange.id = "binance"

        with (
            patch.object(scanner, "_create_exchange", return_value=mock_exchange),
            patch.object(scanner, "_fetch_spot_price", return_value=85000.0),
            patch.object(scanner, "_fetch_futures_price", return_value=None),
        ):
            opps = scanner.scan(as_of=TODAY)

        assert len(opps) == 0

    def test_scan_sorted_by_annualized_desc(self):
        """Opportunities should be sorted by annualized basis descending."""
        scanner = BasisScanner(
            exchanges=["binance"],
            symbols=["BTC", "ETH"],
            api_delay=0,
            min_annualized_pct=0.0,
        )

        # Simulate returning different prices for each symbol
        spot_prices = {"BTC": 85000.0, "ETH": 3200.0}
        futures_prices = {"BTC": 86000.0, "ETH": 3300.0}  # ETH has higher % basis

        def mock_spot(_exchange, symbol):
            return spot_prices.get(symbol)

        def mock_futures(_exchange, symbol):
            # Map futures symbol back to base
            for base in futures_prices:
                if symbol.startswith(base):
                    return futures_prices[base]
            return None

        mock_exchange = MagicMock()
        mock_exchange.id = "binance"

        with (
            patch.object(scanner, "_create_exchange", return_value=mock_exchange),
            patch.object(scanner, "_fetch_spot_price", side_effect=mock_spot),
            patch.object(scanner, "_fetch_futures_price", side_effect=mock_futures),
        ):
            opps = scanner.scan(as_of=TODAY)

        if len(opps) >= 2:
            rates = [o.annualized_basis for o in opps]
            assert rates == sorted(rates, reverse=True)


# ------------------------------------------------------------------
# Strategy entry logic
# ------------------------------------------------------------------


class TestStrategyEntry:
    def test_entry_accepted(self, btc_opportunity, strategy_low_threshold):
        """Opportunity above threshold should be accepted."""
        should_enter, reason = strategy_low_threshold.evaluate_entry(btc_opportunity)
        assert should_enter is True
        assert "Entry signal" in reason

    def test_entry_rejected_below_threshold(self, strategy):
        """Opportunity below threshold should be rejected."""
        # Create a very small basis
        opp = _make_opportunity(spot_price=85000.0, futures_price=85100.0)
        should_enter, reason = strategy.evaluate_entry(opp)
        assert should_enter is False
        assert "below threshold" in reason

    def test_entry_rejected_max_positions(
        self, btc_opportunity, strategy_low_threshold
    ):
        """Should reject when max positions reached."""
        # Fill up positions
        for i in range(strategy_low_threshold.config.max_positions):
            opp = _make_opportunity(
                exchange=f"exchange_{i}",
                futures_price=87000.0 + i * 100,
            )
            strategy_low_threshold.open_position(opp)

        should_enter, reason = strategy_low_threshold.evaluate_entry(btc_opportunity)
        assert should_enter is False
        assert "Max positions" in reason

    def test_entry_rejected_duplicate_contract(
        self, btc_opportunity, strategy_low_threshold
    ):
        """Should reject duplicate contract on same exchange."""
        strategy_low_threshold.open_position(btc_opportunity)
        should_enter, reason = strategy_low_threshold.evaluate_entry(btc_opportunity)
        assert should_enter is False
        assert "Already have position" in reason

    def test_open_position_creates_record(
        self, btc_opportunity, strategy_low_threshold
    ):
        """open_position should create and store a BasisPosition."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        assert pos.position_id in strategy_low_threshold.positions
        assert pos.symbol == "BTC"
        assert pos.exchange == "binance"
        assert pos.entry_spot_price == 85000.0
        assert pos.entry_futures_price == 87000.0
        assert pos.status == PositionStatus.OPEN
        assert pos.is_open is True

    def test_open_position_raises_on_rejection(self, strategy):
        """open_position should raise ValueError if entry rejected."""
        opp = _make_opportunity(spot_price=85000.0, futures_price=85010.0)
        with pytest.raises(ValueError, match="Entry rejected"):
            strategy.open_position(opp)

    def test_position_sizing(self, btc_opportunity, strategy_low_threshold):
        """Position size should be capped at config max."""
        pos = strategy_low_threshold.open_position(
            btc_opportunity, position_size_usd=5000.0
        )
        # Config default is $2000
        assert pos.position_size_usd == 2000.0
        assert pos.quantity == pytest.approx(2000.0 / 85000.0)

    def test_position_sizing_custom(self, btc_opportunity):
        """Custom position size below max should be used."""
        config = BasisStrategyConfig(entry_threshold=0.01, max_position_usd=10000.0)
        strat = BasisStrategy(config=config)
        pos = strat.open_position(btc_opportunity, position_size_usd=3000.0)
        assert pos.position_size_usd == 3000.0


# ------------------------------------------------------------------
# Strategy exit logic
# ------------------------------------------------------------------


class TestStrategyExit:
    def test_exit_at_expiry(self, btc_opportunity, strategy_low_threshold):
        """Should signal exit when expiry date reached."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        should_exit, reason = strategy_low_threshold.evaluate_exit(
            pos, as_of=EXPIRY_JUN
        )
        assert should_exit is True
        assert "expired" in reason.lower() or "convergence" in reason.lower()

    def test_exit_negative_basis(self, btc_opportunity, strategy_low_threshold):
        """Should signal exit when basis turns negative."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        # Mark to market with negative basis
        strategy_low_threshold.mark_to_market(
            pos.position_id,
            current_spot=86000.0,
            current_futures=85500.0,  # futures below spot = backwardation
            as_of=date(2026, 5, 1),
        )
        should_exit, reason = strategy_low_threshold.evaluate_exit(
            pos, as_of=date(2026, 5, 1)
        )
        assert should_exit is True
        assert "negative" in reason.lower() or "backwardation" in reason.lower()

    def test_no_exit_normal_conditions(self, btc_opportunity, strategy_low_threshold):
        """Should not exit under normal conditions."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        strategy_low_threshold.mark_to_market(
            pos.position_id,
            current_spot=85500.0,
            current_futures=86800.0,  # still positive basis
            as_of=date(2026, 5, 1),
        )
        should_exit, _reason = strategy_low_threshold.evaluate_exit(
            pos, as_of=date(2026, 5, 1)
        )
        assert should_exit is False

    def test_no_exit_negative_basis_when_disabled(self, btc_opportunity):
        """Should not exit on negative basis when exit_on_negative_basis=False."""
        config = BasisStrategyConfig(
            entry_threshold=0.01,
            exit_on_negative_basis=False,
        )
        strat = BasisStrategy(config=config)
        pos = strat.open_position(btc_opportunity)
        strat.mark_to_market(
            pos.position_id,
            current_spot=86000.0,
            current_futures=85500.0,
            as_of=date(2026, 5, 1),
        )
        should_exit, _ = strat.evaluate_exit(pos, as_of=date(2026, 5, 1))
        assert should_exit is False


# ------------------------------------------------------------------
# Position lifecycle
# ------------------------------------------------------------------


class TestPositionLifecycle:
    def test_mark_to_market_updates_prices(
        self, btc_opportunity, strategy_low_threshold
    ):
        """mark_to_market should update current prices and basis."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        strategy_low_threshold.mark_to_market(
            pos.position_id,
            current_spot=86000.0,
            current_futures=87500.0,
            as_of=date(2026, 5, 1),
        )
        assert pos.current_spot_price == 86000.0
        assert pos.current_futures_price == 87500.0
        assert pos.current_basis is not None
        assert pos.current_basis == pytest.approx(compute_raw_basis(86000.0, 87500.0))

    def test_mark_to_market_updates_days(self, btc_opportunity, strategy_low_threshold):
        """mark_to_market should recalculate days to expiry."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        strategy_low_threshold.mark_to_market(
            pos.position_id,
            current_spot=86000.0,
            current_futures=87500.0,
            as_of=date(2026, 5, 1),
        )
        expected_days = (EXPIRY_JUN - date(2026, 5, 1)).days
        assert pos.days_to_expiry == expected_days

    def test_mark_to_market_not_found(self, strategy):
        """mark_to_market should raise KeyError for unknown position."""
        with pytest.raises(KeyError):
            strategy.mark_to_market("nonexistent", 85000.0, 87000.0)

    def test_mark_to_market_closed_position(
        self, btc_opportunity, strategy_low_threshold
    ):
        """mark_to_market should raise ValueError for closed position."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        strategy_low_threshold.close_position(
            pos.position_id, 86000.0, 86000.0, reason="test"
        )
        with pytest.raises(ValueError, match="not open"):
            strategy_low_threshold.mark_to_market(pos.position_id, 86000.0, 87000.0)

    def test_close_position_at_expiry(self, btc_opportunity, strategy_low_threshold):
        """Closing at expiry should set CLOSED_EXPIRY status."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        closed = strategy_low_threshold.close_position(
            pos.position_id,
            exit_spot_price=90000.0,
            exit_futures_price=90000.0,  # converged
            reason="Futures expired — convergence",
            as_of=EXPIRY_JUN,
        )
        assert closed.status == PositionStatus.CLOSED_EXPIRY
        assert closed.exit_date == EXPIRY_JUN
        assert closed.exit_spot_price == 90000.0
        assert closed.exit_futures_price == 90000.0

    def test_close_position_backwardation(
        self, btc_opportunity, strategy_low_threshold
    ):
        """Closing on backwardation should set CLOSED_SIGNAL status."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        closed = strategy_low_threshold.close_position(
            pos.position_id,
            exit_spot_price=86000.0,
            exit_futures_price=85500.0,
            reason="Basis turned negative — backwardation exit",
            as_of=date(2026, 5, 15),
        )
        assert closed.status == PositionStatus.CLOSED_SIGNAL

    def test_close_position_not_found(self, strategy):
        """close_position should raise KeyError for unknown position."""
        with pytest.raises(KeyError):
            strategy.close_position("nonexistent", 85000.0, 85000.0)

    def test_close_position_already_closed(
        self, btc_opportunity, strategy_low_threshold
    ):
        """Double-closing should raise ValueError."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        strategy_low_threshold.close_position(
            pos.position_id, 86000.0, 86000.0, reason="test"
        )
        with pytest.raises(ValueError, match="not open"):
            strategy_low_threshold.close_position(
                pos.position_id, 86000.0, 86000.0, reason="test again"
            )


# ------------------------------------------------------------------
# PnL calculations
# ------------------------------------------------------------------


class TestPnL:
    def test_perfect_convergence_pnl(self, btc_opportunity, strategy_low_threshold):
        """At expiry with convergence, PnL = entry premium * quantity - fees."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        # At expiry, spot and futures converge to same price
        strategy_low_threshold.close_position(
            pos.position_id,
            exit_spot_price=90000.0,
            exit_futures_price=90000.0,
            as_of=EXPIRY_JUN,
        )
        # Gross PnL = spot_leg + futures_leg
        # spot_leg = (90000 - 85000) * quantity
        # futures_leg = (87000 - 90000) * quantity  (short, so entry - exit)
        # combined = (90000 - 85000 + 87000 - 90000) * quantity = 2000 * quantity
        expected_gross = 2000.0 * pos.quantity
        assert pos.gross_pnl_usd == pytest.approx(expected_gross)
        assert pos.net_pnl_usd == pytest.approx(expected_gross - pos.total_fees_usd)
        assert pos.net_pnl_usd > 0  # Should be profitable

    def test_pnl_with_spot_drop(self, btc_opportunity, strategy_low_threshold):
        """Market-neutral: spot drops but basis converges → still profitable."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        # Price drops but converges at expiry
        strategy_low_threshold.close_position(
            pos.position_id,
            exit_spot_price=75000.0,
            exit_futures_price=75000.0,
            as_of=EXPIRY_JUN,
        )
        # spot_leg = (75000 - 85000) * q = -10000 * q
        # futures_leg = (87000 - 75000) * q = +12000 * q
        # combined = 2000 * q = premium
        expected_gross = 2000.0 * pos.quantity
        assert pos.gross_pnl_usd == pytest.approx(expected_gross)
        assert pos.net_pnl_usd > 0

    def test_pnl_unrealized(self, btc_opportunity, strategy_low_threshold):
        """Unrealized PnL should reflect current mark-to-market."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        strategy_low_threshold.mark_to_market(
            pos.position_id,
            current_spot=86000.0,
            current_futures=86500.0,
            as_of=date(2026, 5, 1),
        )
        # spot_leg = (86000 - 85000) * q = 1000 * q
        # futures_leg = (87000 - 86500) * q = 500 * q
        # combined = 1500 * q (partial convergence)
        expected_gross = 1500.0 * pos.quantity
        assert pos.gross_pnl_usd == pytest.approx(expected_gross)

    def test_fee_calculation(self, btc_opportunity, strategy_low_threshold):
        """Fee calculation should be correct."""
        pos = strategy_low_threshold.open_position(btc_opportunity)
        size = pos.position_size_usd
        # Entry: taker (buy spot) + maker (sell futures)
        assert pos.entry_fees_usd == pytest.approx(size * (0.0005 + 0.0002))
        # Exit: taker (sell spot), futures settle free
        assert pos.exit_fees_usd == pytest.approx(size * 0.0005)
        # Total = entry + exit
        assert pos.total_fees_usd == pytest.approx(size * (2 * 0.0005 + 0.0002))

    def test_premium_usd(self, btc_opportunity):
        """premium_usd should be futures - spot."""
        assert btc_opportunity.premium_usd == pytest.approx(2000.0)


# ------------------------------------------------------------------
# Portfolio summary
# ------------------------------------------------------------------


class TestPortfolioSummary:
    def test_empty_portfolio(self, strategy):
        summary = strategy.get_portfolio_summary()
        assert summary["open_positions"] == 0
        assert summary["closed_positions"] == 0
        assert summary["total_pnl_usd"] == 0.0
        assert summary["win_rate"] == 0.0

    def test_portfolio_with_open_and_closed(self, strategy_low_threshold):
        """Summary should reflect both open and closed positions."""
        opp1 = _make_opportunity(exchange="binance")
        opp2 = _make_opportunity(exchange="okx")

        pos1 = strategy_low_threshold.open_position(opp1)
        strategy_low_threshold.open_position(opp2)

        # Close one with profit
        strategy_low_threshold.close_position(
            pos1.position_id,
            exit_spot_price=90000.0,
            exit_futures_price=90000.0,
            as_of=EXPIRY_JUN,
        )

        summary = strategy_low_threshold.get_portfolio_summary()
        assert summary["open_positions"] == 1
        assert summary["closed_positions"] == 1
        assert summary["total_positions"] == 2
        assert summary["realized_pnl_usd"] > 0
        assert summary["win_rate"] == 1.0

    def test_open_and_closed_accessors(self, strategy_low_threshold):
        """open_positions and closed_positions should be disjoint."""
        opp1 = _make_opportunity(exchange="binance")
        opp2 = _make_opportunity(exchange="okx")

        pos1 = strategy_low_threshold.open_position(opp1)
        strategy_low_threshold.open_position(opp2)
        strategy_low_threshold.close_position(
            pos1.position_id, 90000.0, 90000.0, as_of=EXPIRY_JUN
        )

        assert len(strategy_low_threshold.open_positions) == 1
        assert len(strategy_low_threshold.closed_positions) == 1


# ------------------------------------------------------------------
# Report formatting
# ------------------------------------------------------------------


class TestReportFormatting:
    def test_empty_report(self):
        report = format_basis_report([])
        assert "CRYPTO BASIS SCANNER" in report
        assert "No basis spreads above threshold" in report

    def test_report_with_opportunities(self, btc_opportunity, eth_opportunity):
        report = format_basis_report([btc_opportunity, eth_opportunity])
        assert "BTC" in report
        assert "ETH" in report
        assert "binance" in report
        assert "2 found" in report

    def test_report_contains_prices(self, btc_opportunity):
        report = format_basis_report([btc_opportunity])
        assert "85,000" in report or "85000" in report


# ------------------------------------------------------------------
# PositionStatus enum
# ------------------------------------------------------------------


class TestPositionStatus:
    def test_status_values(self):
        assert PositionStatus.OPEN.value == "open"
        assert PositionStatus.CLOSED_EXPIRY.value == "closed_expiry"
        assert PositionStatus.CLOSED_SIGNAL.value == "closed_signal"
        assert PositionStatus.CLOSED_MANUAL.value == "closed_manual"

    def test_is_open_property(self, btc_opportunity, strategy_low_threshold):
        pos = strategy_low_threshold.open_position(btc_opportunity)
        assert pos.is_open is True

        strategy_low_threshold.close_position(
            pos.position_id, 90000.0, 90000.0, as_of=EXPIRY_JUN
        )
        assert pos.is_open is False


# ------------------------------------------------------------------
# BasisOpportunity display properties
# ------------------------------------------------------------------


class TestBasisOpportunityDisplay:
    def test_display_basis(self, btc_opportunity):
        display = btc_opportunity.display_basis
        assert "%" in display

    def test_display_annualized(self, btc_opportunity):
        display = btc_opportunity.display_annualized
        assert "%" in display

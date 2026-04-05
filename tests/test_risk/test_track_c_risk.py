"""Tests for Track C (Structural Arbitrage) risk constraints.

Validates that:
- Track C position limits (20% vs Track A's 10%) are correctly applied
- Track C trade size limits (5% vs Track A's 2%) are correctly applied
- Exchange concentration check blocks trades exceeding 25% on one exchange
- Net exposure check enforces the 30% cap for market-neutral arbs
- Cash reserve check enforces the 10% floor for event-driven staging
- Track A limits remain unchanged (backwards compatibility)
- Track C kill-switch checks are wired in but Track A gets pass placeholders
"""

import pytest

from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.config import (
    AppConfig,
    AssetEntry,
    DataConfig,
    GeneralConfig,
    LLMConfig,
    RiskLimits,
    TrackBLimits,
    TrackCLimits,
    UniverseConfig,
)
from llm_quant.risk.limits import check_exchange_concentration, check_net_exposure
from llm_quant.risk.manager import RiskManager
from llm_quant.trading.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def track_c_config(tmp_path):
    """AppConfig with explicit Track C limits for testing."""
    return AppConfig(
        general=GeneralConfig(db_path=str(tmp_path / "test.duckdb")),
        llm=LLMConfig(),
        data=DataConfig(lookback_days=60),
        risk=RiskLimits(),  # Track A defaults
        track_b=TrackBLimits(),
        track_c=TrackCLimits(),  # 20% weight, 5% trade, 30% net, 10% cash
        universe=UniverseConfig(
            name="Test Universe",
            assets=[
                AssetEntry(
                    symbol="SPY",
                    name="S&P 500",
                    category="equity",
                    sector="broad_market",
                ),
                AssetEntry(
                    symbol="TLT",
                    name="20Y Treasury",
                    category="fixed_income",
                    sector="bonds",
                ),
                AssetEntry(
                    symbol="GLD",
                    name="Gold",
                    category="commodity",
                    sector="precious_metals",
                ),
                AssetEntry(
                    symbol="BTC-PERP",
                    name="BTC Perpetual",
                    category="crypto",
                    sector="layer1",
                    asset_class="crypto",
                ),
                AssetEntry(
                    symbol="ARB-CEF",
                    name="CEF Discount Arb",
                    category="equity",
                    sector="closed_end_funds",
                ),
            ],
        ),
    )


@pytest.fixture
def empty_portfolio():
    """Fresh portfolio with no positions."""
    return Portfolio(initial_capital=100_000.0)


@pytest.fixture
def track_c_portfolio():
    """Portfolio with existing Track C positions for concentration tests."""
    p = Portfolio(initial_capital=100_000.0)
    p.cash = 70_000.0
    p.positions = {
        "SPY": Position(
            symbol="SPY",
            shares=50,
            avg_cost=400.0,
            current_price=400.0,
            stop_loss=380.0,
        ),
        "TLT": Position(
            symbol="TLT",
            shares=100,
            avg_cost=100.0,
            current_price=100.0,
            stop_loss=95.0,
        ),
    }
    return p


@pytest.fixture
def sample_prices():
    """Price map for test assets."""
    return {
        "SPY": 400.0,
        "TLT": 100.0,
        "GLD": 185.0,
        "BTC-PERP": 60_000.0,
        "ARB-CEF": 25.0,
    }


# ---------------------------------------------------------------------------
# 1. Track C position weight limits (20% vs Track A 10%)
# ---------------------------------------------------------------------------


class TestTrackCPositionWeight:
    def test_track_c_allows_15pct_position(self, track_c_config, sample_prices):
        """Track C permits 15% position weight (exceeds Track A's 10%).

        We simulate a portfolio that already holds 11% in GLD so the
        incremental trade is only 4% (within the 5% per-trade limit),
        but the target weight of 15% exceeds Track A's 10% cap.
        """
        mgr = RiskManager(track_c_config)
        # Build portfolio with existing 11% GLD position
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 89_000.0
        p.positions = {
            "GLD": Position(
                symbol="GLD",
                shares=59,  # 59 * 185 = 10,915 ~= 11% of 100k NAV
                avg_cost=185.0,
                current_price=185.0,
                stop_loss=170.0,
            ),
        }
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.HIGH,
            target_weight=0.15,  # 15% target => ~4% incremental
            stop_loss=170.0,
            reasoning="Arb leg",
        )
        approved, rejected = mgr.filter_signals([signal], p, sample_prices, track="C")
        assert len(approved) == 1
        assert len(rejected) == 0

    def test_track_c_rejects_25pct_position(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track C rejects 25% position weight (exceeds 20% limit)."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.HIGH,
            target_weight=0.25,
            stop_loss=170.0,
            reasoning="Oversized arb leg",
        )
        approved, rejected = mgr.filter_signals(
            [signal], empty_portfolio, sample_prices, track="C"
        )
        assert len(approved) == 0
        assert len(rejected) == 1
        checks = rejected[0][1]
        weight_fail = [
            c for c in checks if c.rule == "position_weight" and not c.passed
        ]
        assert len(weight_fail) == 1

    def test_track_a_rejects_15pct_position(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track A rejects 15% position weight (exceeds 10% limit)."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.HIGH,
            target_weight=0.15,
            stop_loss=170.0,
            reasoning="Oversized for Track A",
        )
        approved, rejected = mgr.filter_signals(
            [signal], empty_portfolio, sample_prices, track="A"
        )
        assert len(approved) == 0
        assert len(rejected) == 1


# ---------------------------------------------------------------------------
# 2. Track C trade size limits (5% vs Track A 2%)
# ---------------------------------------------------------------------------


class TestTrackCTradeSize:
    def test_track_c_allows_4pct_trade(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track C permits a 4% NAV trade (within 5% limit)."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.04,
            stop_loss=170.0,
            reasoning="Within Track C trade limit",
        )
        checks = mgr.check_trade(signal, empty_portfolio, sample_prices, track="C")
        size_check = [c for c in checks if c.rule == "position_size"]
        assert len(size_check) == 1
        assert size_check[0].passed

    def test_track_c_rejects_6pct_trade(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track C rejects a 6% NAV trade (exceeds 5% limit)."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.HIGH,
            target_weight=0.06,
            stop_loss=170.0,
            reasoning="Oversized for Track C",
        )
        checks = mgr.check_trade(signal, empty_portfolio, sample_prices, track="C")
        size_check = [c for c in checks if c.rule == "position_size"]
        assert len(size_check) == 1
        assert not size_check[0].passed

    def test_track_a_rejects_3pct_trade(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track A rejects a 3% NAV trade (exceeds 2% limit)."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.HIGH,
            target_weight=0.03,
            stop_loss=170.0,
            reasoning="Oversized for Track A",
        )
        checks = mgr.check_trade(signal, empty_portfolio, sample_prices, track="A")
        size_check = [c for c in checks if c.rule == "position_size"]
        assert len(size_check) == 1
        assert not size_check[0].passed


# ---------------------------------------------------------------------------
# 3. Exchange concentration check (25% limit)
# ---------------------------------------------------------------------------


class TestExchangeConcentration:
    def test_exchange_concentration_pass(self):
        """Trade within exchange limit passes."""
        result = check_exchange_concentration(
            exchange="KALSHI",
            exchange_weights={"KALSHI": 0.10, "NYSE": 0.15},
            trade_weight=0.10,
            max_exchange_concentration=0.25,
        )
        assert result.passed
        assert result.rule == "exchange_concentration"

    def test_exchange_concentration_fail(self):
        """Trade pushing exchange above 25% fails."""
        result = check_exchange_concentration(
            exchange="KALSHI",
            exchange_weights={"KALSHI": 0.20},
            trade_weight=0.10,
            max_exchange_concentration=0.25,
        )
        assert not result.passed
        assert "KALSHI" in result.message

    def test_exchange_concentration_new_exchange(self):
        """Trade on a new exchange (0% current) passes within limit."""
        result = check_exchange_concentration(
            exchange="CME",
            exchange_weights={"NYSE": 0.20},
            trade_weight=0.15,
            max_exchange_concentration=0.25,
        )
        assert result.passed

    def test_exchange_concentration_exactly_at_limit(self):
        """Trade reaching exactly 25% passes (<= comparison)."""
        result = check_exchange_concentration(
            exchange="BINANCE",
            exchange_weights={"BINANCE": 0.15},
            trade_weight=0.10,
            max_exchange_concentration=0.25,
        )
        assert result.passed

    def test_exchange_concentration_wired_into_track_c(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Exchange concentration check fires for Track C trades."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.04,
            stop_loss=170.0,
            reasoning="Arb trade on overloaded exchange",
        )
        # 22% already on KALSHI + 4% new trade => 26% > 25%
        checks = mgr.check_trade(
            signal,
            empty_portfolio,
            sample_prices,
            track="C",
            exchange="KALSHI",
            exchange_weights={"KALSHI": 0.22},
        )
        exch_check = [c for c in checks if c.rule == "exchange_concentration"]
        assert len(exch_check) == 1
        assert not exch_check[0].passed

    def test_exchange_concentration_skipped_for_track_a(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Exchange concentration check emits pass placeholder for Track A."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.02,
            stop_loss=170.0,
            reasoning="Track A trade",
        )
        checks = mgr.check_trade(
            signal,
            empty_portfolio,
            sample_prices,
            track="A",
            exchange="KALSHI",
            exchange_weights={"KALSHI": 0.50},  # would fail if checked
        )
        exch_check = [c for c in checks if c.rule == "exchange_concentration"]
        assert len(exch_check) == 1
        assert exch_check[0].passed  # placeholder pass, not enforced

    def test_exchange_concentration_sell_passes(self, track_c_config, sample_prices):
        """Sells should not be blocked by exchange concentration (reduces exposure)."""
        mgr = RiskManager(track_c_config)
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 80_000.0
        p.positions = {
            "GLD": Position(
                symbol="GLD",
                shares=100,
                avg_cost=185.0,
                current_price=185.0,
                stop_loss=170.0,
            ),
        }
        signal = TradeSignal(
            symbol="GLD",
            action=Action.SELL,
            conviction=Conviction.MEDIUM,
            target_weight=0.05,
            stop_loss=170.0,
            reasoning="Reducing position",
        )
        checks = mgr.check_trade(
            signal,
            p,
            sample_prices,
            track="C",
            exchange="NYSE",
            exchange_weights={"NYSE": 0.50},  # high weight, but selling
        )
        exch_check = [c for c in checks if c.rule == "exchange_concentration"]
        assert len(exch_check) == 1
        assert exch_check[0].passed


# ---------------------------------------------------------------------------
# 4. Net exposure check (30% cap for market-neutral arbs)
# ---------------------------------------------------------------------------


class TestTrackCNetExposure:
    def test_net_exposure_within_30pct(self):
        """Net exposure within 30% passes for Track C."""
        result = check_net_exposure(
            current_net=20_000.0,
            trade_notional=5_000.0,
            nav=100_000.0,
            max_net=0.30,
        )
        assert result.passed

    def test_net_exposure_exceeds_30pct(self):
        """Net exposure above 30% fails for Track C."""
        result = check_net_exposure(
            current_net=25_000.0,
            trade_notional=10_000.0,
            nav=100_000.0,
            max_net=0.30,
        )
        assert not result.passed

    def test_track_c_net_exposure_via_manager(self, track_c_config, sample_prices):
        """Track C manager enforces 30% net exposure cap."""
        mgr = RiskManager(track_c_config)
        # Build a portfolio with 28% net exposure already
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 72_000.0
        p.positions = {
            "SPY": Position(
                symbol="SPY",
                shares=70,
                avg_cost=400.0,
                current_price=400.0,
                stop_loss=380.0,
            ),
        }
        # NAV = 72,000 + 70*400 = 100,000; net = 28,000 = 28%
        # Adding 5% more would push to ~33%
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.05,
            stop_loss=170.0,
            reasoning="Adding arb leg",
        )
        checks = mgr.check_trade(signal, p, sample_prices, track="C")
        net_check = [c for c in checks if c.rule == "net_exposure"]
        assert len(net_check) == 1
        assert not net_check[0].passed


# ---------------------------------------------------------------------------
# 5. Cash reserve (10% for Track C vs 5% for Track A)
# ---------------------------------------------------------------------------


class TestTrackCCashReserve:
    def test_track_c_enforces_10pct_cash_reserve(self, track_c_config, sample_prices):
        """Track C requires 10% cash reserve."""
        mgr = RiskManager(track_c_config)
        # Portfolio with exactly 12% cash
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 12_000.0
        p.positions = {
            "SPY": Position(
                symbol="SPY",
                shares=220,
                avg_cost=400.0,
                current_price=400.0,
                stop_loss=380.0,
            ),
        }
        # NAV = 12,000 + 88,000 = 100,000; cash = 12%
        # Trade of 4% = $4,000 => remaining cash 8% < 10% => FAIL
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.04,
            stop_loss=170.0,
            reasoning="Arb leg",
        )
        checks = mgr.check_trade(signal, p, sample_prices, track="C")
        cash_check = [c for c in checks if c.rule == "cash_reserve"]
        assert len(cash_check) == 1
        assert not cash_check[0].passed

    def test_track_a_allows_same_trade_with_5pct_reserve(
        self, track_c_config, sample_prices
    ):
        """Same trade passes Track A's 5% cash reserve requirement."""
        mgr = RiskManager(track_c_config)
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 12_000.0
        p.positions = {
            "SPY": Position(
                symbol="SPY",
                shares=220,
                avg_cost=400.0,
                current_price=400.0,
                stop_loss=380.0,
            ),
        }
        # Same trade: 2% trade => remaining cash 10% > 5% => PASS
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.02,
            stop_loss=170.0,
            reasoning="Small hedge",
        )
        checks = mgr.check_trade(signal, p, sample_prices, track="A")
        cash_check = [c for c in checks if c.rule == "cash_reserve"]
        assert len(cash_check) == 1
        assert cash_check[0].passed


# ---------------------------------------------------------------------------
# 6. Backwards compatibility — Track A limits unchanged
# ---------------------------------------------------------------------------


class TestTrackABackwardsCompatibility:
    def test_track_a_max_position_weight_10pct(self, track_c_config):
        """Track A default position weight limit is 10%."""
        assert track_c_config.risk.max_position_weight == 0.10

    def test_track_a_max_trade_size_2pct(self, track_c_config):
        """Track A default trade size limit is 2%."""
        assert track_c_config.risk.max_trade_size == 0.02

    def test_track_a_min_cash_reserve_5pct(self, track_c_config):
        """Track A default cash reserve is 5%."""
        assert track_c_config.risk.min_cash_reserve == 0.05

    def test_track_a_max_net_exposure_100pct(self, track_c_config):
        """Track A default net exposure limit is 100%."""
        assert track_c_config.risk.max_net_exposure == 1.0

    def test_track_a_has_no_exchange_concentration(self, track_c_config):
        """Track A RiskLimits does not have exchange concentration."""
        assert not hasattr(track_c_config.risk, "max_exchange_concentration")

    def test_track_a_signal_approved_at_2pct(self, track_c_config, sample_prices):
        """A 2% trade passes all Track A checks."""
        mgr = RiskManager(track_c_config)
        p = Portfolio(initial_capital=100_000.0)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.02,
            stop_loss=170.0,
            reasoning="Small gold hedge",
        )
        approved, rejected = mgr.filter_signals([signal], p, sample_prices, track="A")
        assert len(approved) == 1
        assert len(rejected) == 0


# ---------------------------------------------------------------------------
# 7. Kill-switch placeholder consistency
# ---------------------------------------------------------------------------


class TestKillSwitchPlaceholders:
    def test_track_a_gets_5_tc_placeholders(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track A check_trade results include 5 TC placeholder passes."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.02,
            stop_loss=170.0,
            reasoning="Test",
        )
        checks = mgr.check_trade(signal, empty_portfolio, sample_prices, track="A")
        tc_rules = {
            "tc_exchange_outage",
            "tc_funding_reversal",
            "tc_spread_collapse",
            "tc_beta_breach",
            "exchange_concentration",
        }
        tc_checks = [c for c in checks if c.rule in tc_rules]
        assert len(tc_checks) == 5
        assert all(c.passed for c in tc_checks)

    def test_track_c_gets_5_tc_checks(
        self, track_c_config, empty_portfolio, sample_prices
    ):
        """Track C check_trade results include 5 real TC checks."""
        mgr = RiskManager(track_c_config)
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.04,
            stop_loss=170.0,
            reasoning="Track C trade",
        )
        checks = mgr.check_trade(signal, empty_portfolio, sample_prices, track="C")
        tc_rules = {
            "tc_exchange_outage",
            "tc_funding_reversal",
            "tc_spread_collapse",
            "tc_beta_breach",
            "exchange_concentration",
        }
        tc_checks = [c for c in checks if c.rule in tc_rules]
        assert len(tc_checks) == 5


# ---------------------------------------------------------------------------
# 8. Track C drawdown circuit breaker (tighter: 10% vs 15%)
# ---------------------------------------------------------------------------


class TestTrackCDrawdown:
    def test_track_c_drawdown_blocks_at_8pct(self, track_c_config, sample_prices):
        """Track C blocks buys when drawdown > 7% (threshold = 10% - 3%)."""
        mgr = RiskManager(track_c_config)
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 92_000.0  # NAV = 92k, peak = 100k => -8% drawdown
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.04,
            stop_loss=170.0,
            reasoning="Buying during drawdown",
        )
        checks = mgr.check_trade(signal, p, sample_prices, track="C")
        dd_check = [c for c in checks if c.rule == "drawdown_limit"]
        assert len(dd_check) == 1
        assert not dd_check[0].passed  # -8% < -7% threshold

    def test_track_a_allows_8pct_drawdown(self, track_c_config, sample_prices):
        """Track A allows buys at -8% drawdown (threshold = 15% - 3% = 12%)."""
        mgr = RiskManager(track_c_config)
        p = Portfolio(initial_capital=100_000.0)
        p.cash = 92_000.0  # NAV = 92k, peak = 100k => -8% drawdown
        signal = TradeSignal(
            symbol="GLD",
            action=Action.BUY,
            conviction=Conviction.MEDIUM,
            target_weight=0.02,
            stop_loss=170.0,
            reasoning="Buying during moderate drawdown",
        )
        checks = mgr.check_trade(signal, p, sample_prices, track="A")
        dd_check = [c for c in checks if c.rule == "drawdown_limit"]
        assert len(dd_check) == 1
        assert dd_check[0].passed  # -8% >= -12% threshold

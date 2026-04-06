"""Tests for NegRisk complement arbitrage strategy.

Covers:
  - Fee calculation (geopolitics 0%, standard 2%)
  - Kelly sizing for guaranteed payoffs
  - Position size caps (bankroll, liquidity)
  - Profit calculation (sum < 1.0 = profit, sum >= 1.0 = no opportunity)
  - Scanner filter tests (min profit, min volume, max outcomes)
  - Mock API response tests (Gamma + CLOB)
  - Paper logging

All tests use mocked HTTP responses -- no live API calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_quant.arb.clob_client import ClobClient
from llm_quant.arb.gamma_client import GammaClient
from llm_quant.arb.negrisk_arb import (
    NegRiskOpportunity,
    NegRiskScanner,
    calculate_fees,
    calculate_kelly,
    calculate_position_size,
    get_fee_rate,
    log_opportunity_to_paper,
)

# ------------------------------------------------------------------
# Fee calculation tests
# ------------------------------------------------------------------


class TestFeeRate:
    def test_politics_free(self):
        """Politics events have 0% fee."""
        assert get_fee_rate("politics") == 0.0

    def test_geopolitics_free(self):
        """Geopolitics events have 0% fee."""
        assert get_fee_rate("geopolitics") == 0.0

    def test_politics_case_insensitive(self):
        """Category matching is case-insensitive."""
        assert get_fee_rate("Politics") == 0.0
        assert get_fee_rate("GEOPOLITICS") == 0.0

    def test_standard_fee(self):
        """Non-politics categories have 2% fee."""
        assert get_fee_rate("sports") == 0.02
        assert get_fee_rate("crypto") == 0.02
        assert get_fee_rate("other") == 0.02
        assert get_fee_rate("finance") == 0.02

    def test_empty_category(self):
        """Empty category defaults to standard fee."""
        assert get_fee_rate("") == 0.02


class TestFeeCalculation:
    def test_zero_fee_rate(self):
        """No fees for fee-free categories."""
        prices = [0.30, 0.25, 0.35]
        assert calculate_fees(prices, 0.0) == 0.0

    def test_standard_fee(self):
        """Standard 2% fee: sum(0.02 * price_i)."""
        prices = [0.30, 0.25, 0.35]
        expected = 0.02 * (0.30 + 0.25 + 0.35)
        assert calculate_fees(prices, 0.02) == pytest.approx(expected)

    def test_single_outcome(self):
        """Fee on a single outcome."""
        prices = [0.50]
        assert calculate_fees(prices, 0.02) == pytest.approx(0.01)

    def test_many_outcomes(self):
        """Fee scales with number of outcomes."""
        prices = [0.10] * 10  # 10 outcomes at $0.10 each
        # Total = 0.02 * 1.0 = 0.02
        assert calculate_fees(prices, 0.02) == pytest.approx(0.02)

    def test_empty_prices(self):
        """Empty price list produces zero fee."""
        assert calculate_fees([], 0.02) == 0.0


# ------------------------------------------------------------------
# Kelly sizing tests
# ------------------------------------------------------------------


class TestKellySizing:
    def test_positive_profit(self):
        """Kelly for guaranteed payoff: f* = net_profit / (1 + net_profit)."""
        # net_profit = 0.10 -> f* = 0.10 / 1.10 = 0.0909
        assert calculate_kelly(0.10) == pytest.approx(0.10 / 1.10)

    def test_zero_profit(self):
        """Zero profit -> zero Kelly."""
        assert calculate_kelly(0.0) == 0.0

    def test_negative_profit(self):
        """Negative profit -> zero Kelly (don't bet)."""
        assert calculate_kelly(-0.05) == 0.0

    def test_large_profit(self):
        """Large profit -> Kelly approaches 1.0 but stays below."""
        # net_profit = 1.0 -> f* = 1.0 / 2.0 = 0.50
        assert calculate_kelly(1.0) == pytest.approx(0.50)

    def test_small_profit(self):
        """Small profit -> small Kelly fraction."""
        # net_profit = 0.01 -> f* = 0.01 / 1.01 = 0.0099
        assert calculate_kelly(0.01) == pytest.approx(0.01 / 1.01)


# ------------------------------------------------------------------
# Position size cap tests
# ------------------------------------------------------------------


class TestPositionSize:
    def test_basic_kelly_sizing(self):
        """Basic Kelly: fraction * bankroll."""
        size = calculate_position_size(
            kelly_fraction=0.10,
            bankroll=100.0,
            min_volume=1000.0,
        )
        assert size == 10.0

    def test_bankroll_cap(self):
        """Position capped at 20% of bankroll."""
        size = calculate_position_size(
            kelly_fraction=0.50,  # Would be $50 on $100 bankroll
            bankroll=100.0,
            min_volume=1000.0,
        )
        assert size == 20.0  # Capped at 20%

    def test_liquidity_cap(self):
        """Position capped at 10% of min volume."""
        size = calculate_position_size(
            kelly_fraction=0.10,
            bankroll=100.0,
            min_volume=50.0,  # 10% of $50 = $5
        )
        assert size == 5.0  # Capped by liquidity

    def test_below_min_bet(self):
        """Returns 0 if calculated size is below $1 min bet."""
        size = calculate_position_size(
            kelly_fraction=0.005,  # $0.50 on $100
            bankroll=100.0,
            min_volume=1000.0,
        )
        assert size == 0.0

    def test_zero_kelly(self):
        """Zero Kelly -> zero position."""
        size = calculate_position_size(
            kelly_fraction=0.0,
            bankroll=100.0,
            min_volume=1000.0,
        )
        assert size == 0.0

    def test_zero_volume_skips_liquidity_check(self):
        """Zero volume skips liquidity cap (no volume data available)."""
        size = calculate_position_size(
            kelly_fraction=0.10,
            bankroll=100.0,
            min_volume=0.0,
        )
        # When min_volume=0, liquidity check is skipped; Kelly sizing applies
        assert size == 10.0

    def test_tiny_volume_caps_position(self):
        """Very small volume caps position via liquidity constraint."""
        size = calculate_position_size(
            kelly_fraction=0.10,
            bankroll=100.0,
            min_volume=5.0,  # 10% of $5 = $0.50 < $1.00 min bet
        )
        assert size == 0.0  # Below min bet after liquidity cap

    def test_custom_max_position(self):
        """Custom max position percentage."""
        size = calculate_position_size(
            kelly_fraction=0.50,
            bankroll=100.0,
            min_volume=1000.0,
            max_position_pct=0.10,  # 10% cap
        )
        assert size == 10.0

    def test_rounded_to_cents(self):
        """Position size is rounded to 2 decimal places."""
        size = calculate_position_size(
            kelly_fraction=0.0333,
            bankroll=100.0,
            min_volume=1000.0,
        )
        assert size == 3.33


# ------------------------------------------------------------------
# Profit calculation tests
# ------------------------------------------------------------------


class TestProfitCalculation:
    def test_sum_below_one_is_profit(self):
        """Sum of YES prices < 1.0 means guaranteed profit."""
        prices = [0.20, 0.25, 0.30]  # sum = 0.75
        total_cost = sum(prices)
        gross_profit = 1.0 - total_cost
        assert gross_profit == pytest.approx(0.25)
        assert gross_profit > 0  # Profit exists

    def test_sum_equals_one_no_profit(self):
        """Sum of YES prices = 1.0 means no profit."""
        prices = [0.30, 0.30, 0.40]  # sum = 1.00
        total_cost = sum(prices)
        gross_profit = 1.0 - total_cost
        assert gross_profit == pytest.approx(0.0)

    def test_sum_above_one_no_opportunity(self):
        """Sum of YES prices > 1.0 means no opportunity."""
        prices = [0.40, 0.35, 0.30]  # sum = 1.05
        total_cost = sum(prices)
        gross_profit = 1.0 - total_cost
        assert gross_profit < 0

    def test_net_profit_after_fees(self):
        """Net profit = gross - fees."""
        prices = [0.20, 0.25, 0.30]  # sum = 0.75, gross = 0.25
        gross = 1.0 - sum(prices)
        fees = calculate_fees(prices, 0.02)  # 0.02 * 0.75 = 0.015
        net = gross - fees
        assert net == pytest.approx(0.25 - 0.015)

    def test_net_profit_fee_free(self):
        """Fee-free events: net = gross."""
        prices = [0.20, 0.25, 0.30]
        gross = 1.0 - sum(prices)
        fees = calculate_fees(prices, 0.0)
        net = gross - fees
        assert net == pytest.approx(0.25)

    def test_fees_eat_all_profit(self):
        """When fees exceed gross profit, net is negative."""
        prices = [0.32, 0.33, 0.34]  # sum = 0.99, gross = 0.01
        gross = 1.0 - sum(prices)
        fees = calculate_fees(prices, 0.02)  # 0.02 * 0.99 = 0.0198
        net = gross - fees
        assert net < 0  # Fees eat all profit


# ------------------------------------------------------------------
# NegRiskOpportunity dataclass tests
# ------------------------------------------------------------------


class TestNegRiskOpportunity:
    def test_display(self):
        """Display method produces formatted string."""
        opp = NegRiskOpportunity(
            condition_id="0xabc",
            event_slug="test-event-slug",
            question="Test?",
            n_outcomes=3,
            prices=[0.30, 0.25, 0.35],
            token_ids=["t1", "t2", "t3"],
            total_cost=0.90,
            gross_profit=0.10,
            fee_rate=0.0,
            net_profit=0.10,
            net_profit_pct=11.1,
            kelly_fraction=0.0909,
            suggested_size_usd=9.09,
            volumes=[100, 200, 150],
            min_volume=100,
            detected_at="2026-04-06T00:00:00",
        )
        display = opp.display()
        assert "test-event-slug" in display
        assert "N= 3" in display
        assert "11.1%" in display


# ------------------------------------------------------------------
# NegRiskScanner tests (mocked API)
# ------------------------------------------------------------------


def _make_event_data(
    slug: str = "test-event",
    prices: list[float] | None = None,
    category: str = "politics",
    *,
    neg_risk: bool = True,
    volumes: list[float] | None = None,
) -> dict:
    """Create a mock event data dict."""
    if prices is None:
        prices = [0.30, 0.25, 0.35]
    if volumes is None:
        volumes = [100.0] * len(prices)

    markets = []
    for i, (p, v) in enumerate(zip(prices, volumes, strict=True)):
        markets.append(
            {
                "id": f"mkt_{i}",
                "question": f"Outcome {i}?",
                "slug": f"outcome-{i}",
                "outcomes": '["Yes","No"]',
                "outcomePrices": f'["{p}","{1.0 - p}"]',
                "active": True,
                "negRisk": neg_risk,
                "category": category,
                "volume24hr": str(v),
                "clobTokenIds": f'["token_yes_{i}","token_no_{i}"]',
                "conditionId": f"cond_{i}",
            }
        )

    return {
        "id": f"event_{slug}",
        "slug": slug,
        "title": f"Test event: {slug}",
        "category": category,
        "negRisk": neg_risk,
        "markets": markets,
        "conditionId": f"cond_{slug}",
    }


@pytest.fixture
def scanner() -> NegRiskScanner:
    """NegRiskScanner with mocked clients."""
    gamma = GammaClient(ssl_verify=False)
    clob = ClobClient(ssl_verify=False)
    return NegRiskScanner(gamma_client=gamma, clob_client=clob, bankroll=100.0)


class TestNegRiskScanner:
    def test_scan_event_profitable(self, scanner):
        """Detect opportunity when sum(prices) < 1.0."""
        event = _make_event_data(
            prices=[0.20, 0.25, 0.30],  # sum=0.75, gross=0.25
            category="politics",  # fee-free
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            opp = scanner.scan_event("test-event")

        assert opp is not None
        assert opp.total_cost == pytest.approx(0.75)
        assert opp.gross_profit == pytest.approx(0.25)
        assert opp.fee_rate == 0.0  # politics = free
        assert opp.net_profit == pytest.approx(0.25)
        assert opp.net_profit_pct > 0
        assert opp.n_outcomes == 3

    def test_scan_event_no_opportunity(self, scanner):
        """No opportunity when sum(prices) >= 1.0."""
        event = _make_event_data(
            prices=[0.40, 0.35, 0.30],  # sum=1.05
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            opp = scanner.scan_event("no-arb-event")

        assert opp is None

    def test_scan_event_with_fees(self, scanner):
        """Fees reduce net profit for non-politics events."""
        event = _make_event_data(
            prices=[0.20, 0.25, 0.30],  # sum=0.75, gross=0.25
            category="sports",  # 2% fee
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            opp = scanner.scan_event("sports-event")

        assert opp is not None
        assert opp.fee_rate == 0.02
        expected_fees = 0.02 * 0.75
        assert opp.net_profit == pytest.approx(0.25 - expected_fees)
        # Net profit should be less than gross
        assert opp.net_profit < opp.gross_profit

    def test_scan_event_fees_eat_profit(self, scanner):
        """No opportunity when fees exceed gross profit."""
        event = _make_event_data(
            prices=[0.32, 0.33, 0.34],  # sum=0.99, gross=0.01
            category="sports",  # 2% fee -> fees = 0.02 * 0.99 = 0.0198
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            opp = scanner.scan_event("thin-margin-event")

        assert opp is None  # Fees eat the 1-cent profit

    def test_scan_event_min_profit_filter(self, scanner):
        """Filter out opportunities below min_profit_pct."""
        event = _make_event_data(
            prices=[0.30, 0.30, 0.35],  # sum=0.95, gross=0.05
            category="politics",  # fee-free, net=5%/0.95=5.26%
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            # With high threshold, should be filtered
            opp = scanner.scan_event("test", min_profit_pct=10.0)
            assert opp is None

            # With low threshold, should pass
            opp = scanner.scan_event("test", min_profit_pct=1.0)
            assert opp is not None

    def test_scan_event_min_volume_filter(self, scanner):
        """Filter out events with low volume outcomes."""
        event = _make_event_data(
            prices=[0.20, 0.25, 0.30],
            volumes=[100.0, 2.0, 50.0],  # Outcome 1 has very low volume
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            # min_volume=5 -> min across outcomes is $2 -> filtered
            opp = scanner.scan_event("low-vol", min_volume=5.0)
            assert opp is None

            # min_volume=1 -> passes
            opp = scanner.scan_event("low-vol", min_volume=1.0)
            assert opp is not None

    def test_scan_event_max_outcomes_filter(self, scanner):
        """Filter out events with too many outcomes."""
        event = _make_event_data(
            prices=[0.01] * 60,  # 60 outcomes
            volumes=[100.0] * 60,
        )
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            opp = scanner.scan_event("many-outcomes", max_outcomes=50)
            assert opp is None

    def test_scan_event_single_outcome_skipped(self, scanner):
        """Events with < 2 outcomes are skipped."""
        event = _make_event_data(prices=[0.50])
        # Only 1 market -> not valid for complement arb
        with patch.object(scanner._gamma, "fetch_event", return_value=event):
            opp = scanner.scan_event("single-outcome")
            assert opp is None

    def test_scan_event_uses_clob_prices(self):
        """Scanner should prefer CLOB prices over Gamma prices when enabled."""
        gamma = GammaClient(ssl_verify=False)
        clob = ClobClient(ssl_verify=False)
        clob_scanner = NegRiskScanner(
            gamma_client=gamma,
            clob_client=clob,
            bankroll=100.0,
            use_clob_prices=True,
        )
        event = _make_event_data(
            prices=[0.30, 0.30, 0.30],  # Gamma: sum=0.90
            category="politics",
        )
        # CLOB returns different (more accurate) prices
        clob_prices = {"token_yes_0": 0.32, "token_yes_1": 0.31, "token_yes_2": 0.29}

        def mock_clob_price(token_id: str, side: str = "BUY") -> float | None:  # noqa: ARG001
            return clob_prices.get(token_id)

        with (
            patch.object(clob_scanner._gamma, "fetch_event", return_value=event),
            patch.object(clob_scanner._clob, "get_price", side_effect=mock_clob_price),
        ):
            opp = clob_scanner.scan_event("clob-test")

        assert opp is not None
        # CLOB prices: 0.32 + 0.31 + 0.29 = 0.92 (not 0.90 from Gamma)
        assert opp.total_cost == pytest.approx(0.92)

    def test_scan_event_clob_failure_falls_back(self, scanner):
        """Falls back to Gamma prices when CLOB fails."""
        event = _make_event_data(
            prices=[0.30, 0.25, 0.30],  # sum=0.85
            category="politics",
        )
        with (
            patch.object(scanner._gamma, "fetch_event", return_value=event),
            patch.object(
                scanner._clob, "get_price", side_effect=Exception("CLOB down")
            ),
        ):
            opp = scanner.scan_event("fallback-test")

        assert opp is not None
        assert opp.total_cost == pytest.approx(0.85)  # Gamma prices used

    def test_scan_all_active(self, scanner):
        """scan_all_active fetches events and filters NegRisk ones."""
        events = [
            _make_event_data(
                slug="arb-1",
                prices=[0.20, 0.25, 0.30],
                category="politics",
            ),
            _make_event_data(
                slug="no-arb",
                prices=[0.40, 0.35, 0.30],
                category="politics",
            ),
            {
                "id": "non-negrisk",
                "slug": "non-neg",
                "negRisk": False,
                "markets": [],
            },
        ]
        with (
            patch.object(
                scanner._gamma, "fetch_all_active_events", return_value=events
            ),
            patch.object(scanner._clob, "get_price", return_value=None),
        ):
            opps = scanner.scan_all_active(min_profit_pct=1.0, min_volume=0.0)

        # Only arb-1 should pass (sum=0.75, profitable)
        # no-arb has sum=1.05, non-negrisk is filtered
        assert len(opps) == 1
        assert opps[0].event_slug == "arb-1"

    def test_scan_all_sorted_by_profit(self, scanner):
        """Results sorted by net_profit_pct descending."""
        events = [
            _make_event_data(
                slug="small-arb",
                prices=[0.30, 0.30, 0.30],  # sum=0.90, net=11.1%
                category="politics",
            ),
            _make_event_data(
                slug="big-arb",
                prices=[0.20, 0.20, 0.20],  # sum=0.60, net=66.7%
                category="politics",
            ),
        ]
        with (
            patch.object(
                scanner._gamma, "fetch_all_active_events", return_value=events
            ),
            patch.object(scanner._clob, "get_price", return_value=None),
        ):
            opps = scanner.scan_all_active(min_profit_pct=1.0, min_volume=0.0)

        assert len(opps) == 2
        assert opps[0].event_slug == "big-arb"  # Higher profit first
        assert opps[0].net_profit_pct > opps[1].net_profit_pct

    def test_scan_event_kelly_sizing(self, scanner):
        """Kelly fraction and position size are calculated correctly."""
        event = _make_event_data(
            prices=[0.30, 0.25, 0.35],  # sum=0.90, gross=0.10
            category="politics",  # fee-free, net=0.10
            volumes=[1000.0, 1000.0, 1000.0],
        )
        with (
            patch.object(scanner._gamma, "fetch_event", return_value=event),
            patch.object(scanner._clob, "get_price", return_value=None),
        ):
            opp = scanner.scan_event("kelly-test")

        assert opp is not None
        # Kelly: f* = 0.10 / 1.10 = 0.0909
        assert opp.kelly_fraction == pytest.approx(0.10 / 1.10)
        # Size: 0.0909 * 100 = $9.09 (within 20% cap and liquidity)
        assert opp.suggested_size_usd == pytest.approx(9.09, abs=0.01)

    def test_fetch_event_failure(self, scanner):
        """Graceful handling when Gamma API fails."""
        with patch.object(
            scanner._gamma,
            "fetch_event",
            side_effect=Exception("API down"),
        ):
            opp = scanner.scan_event("fail-event")
        assert opp is None


# ------------------------------------------------------------------
# Paper logging tests
# ------------------------------------------------------------------


class TestPaperLogging:
    def test_log_creates_file(self):
        """Paper log creates the file and parent dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "sub" / "paper_log.jsonl"
            opp = NegRiskOpportunity(
                condition_id="0xabc",
                event_slug="test-event",
                question="Test?",
                n_outcomes=3,
                prices=[0.30, 0.25, 0.35],
                token_ids=["t1", "t2", "t3"],
                total_cost=0.90,
                gross_profit=0.10,
                fee_rate=0.0,
                net_profit=0.10,
                net_profit_pct=11.1,
                kelly_fraction=0.09,
                suggested_size_usd=9.0,
                volumes=[100, 200, 150],
                min_volume=100,
                detected_at="2026-04-06T00:00:00",
            )
            log_opportunity_to_paper(opp, log_path, action="DETECTED")

            assert log_path.exists()
            with log_path.open() as f:
                record = json.loads(f.readline())
            assert record["action"] == "DETECTED"
            assert record["event_slug"] == "test-event"
            assert record["net_profit_pct"] == 11.1

    def test_log_appends(self):
        """Multiple logs append to same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "paper_log.jsonl"
            opp = NegRiskOpportunity(
                condition_id="0x1",
                event_slug="event-1",
                question="Q1?",
                n_outcomes=2,
                prices=[0.40, 0.40],
                token_ids=["t1", "t2"],
                total_cost=0.80,
                gross_profit=0.20,
                fee_rate=0.0,
                net_profit=0.20,
                net_profit_pct=25.0,
                kelly_fraction=0.166,
                suggested_size_usd=16.6,
                volumes=[500, 500],
                min_volume=500,
                detected_at="2026-04-06T00:00:00",
            )
            log_opportunity_to_paper(opp, log_path, action="DETECTED")
            log_opportunity_to_paper(opp, log_path, action="WOULD_BUY")

            with log_path.open() as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["action"] == "DETECTED"
            assert json.loads(lines[1])["action"] == "WOULD_BUY"

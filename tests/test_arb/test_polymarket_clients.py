"""Tests for Polymarket API clients -- Gamma and CLOB.

Covers:
  - GammaClient: market/event fetching, pagination, parsing, error handling
  - ClobClient: orderbook, pricing, price history, error handling
  - Data model properties (ConditionPrice, Market, Orderbook, PriceHistory)
  - Retry logic on rate-limit (429) and transient errors
  - Geo-block detection (HTML 404)

All tests use mocked HTTP responses -- no live API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_quant.arb.clob_client import (
    ClobClient,
    Orderbook,
    OrderbookLevel,
    PriceHistory,
    PricePoint,
)
from llm_quant.arb.gamma_client import (
    ConditionPrice,
    GammaClient,
    Market,
    _infer_category,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def gamma() -> GammaClient:
    """GammaClient with SSL disabled (for testing)."""
    return GammaClient(ssl_verify=False)


@pytest.fixture
def clob() -> ClobClient:
    """ClobClient with SSL disabled (for testing)."""
    return ClobClient(ssl_verify=False)


def _mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
    text: str = "",
    content_type: str = "application/json",
    headers: dict | None = None,
) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.headers = {"content-type": content_type}
    if headers:
        resp.headers.update(headers)
    resp.json.return_value = json_data if json_data is not None else {}
    resp.raise_for_status.side_effect = (
        None
        if status_code < 400
        else requests.HTTPError(f"HTTP {status_code}", response=resp)
    )
    return resp


# ------------------------------------------------------------------
# ConditionPrice dataclass tests
# ------------------------------------------------------------------


class TestConditionPrice:
    def test_spread_positive(self):
        """YES + NO > 1 means positive spread (no arb)."""
        cp = ConditionPrice("c1", "Q?", 0.55, 0.50, 1000, 5000)
        assert cp.spread == pytest.approx(0.05)
        assert cp.is_rebalance_arb is False

    def test_spread_negative(self):
        """YES + NO < 1 means negative spread (arb opportunity)."""
        cp = ConditionPrice("c1", "Q?", 0.45, 0.50, 1000, 5000)
        assert cp.spread == pytest.approx(-0.05)
        assert cp.is_rebalance_arb is True

    def test_spread_exact(self):
        """YES + NO = 1 means zero spread."""
        cp = ConditionPrice("c1", "Q?", 0.60, 0.40, 1000, 5000)
        assert cp.spread == pytest.approx(0.0)
        assert cp.is_rebalance_arb is False

    def test_small_negative_spread_not_arb(self):
        """Spread below -0.02 threshold is arb, above is not."""
        cp = ConditionPrice("c1", "Q?", 0.49, 0.50, 1000, 5000)
        assert cp.spread == pytest.approx(-0.01)
        assert cp.is_rebalance_arb is False  # below 2-cent threshold

    def test_clob_token_ids_default(self):
        """Default clob_token_ids is empty list."""
        cp = ConditionPrice("c1", "Q?", 0.50, 0.50, 1000, 5000)
        assert cp.clob_token_ids == []

    def test_clob_token_ids_provided(self):
        """clob_token_ids can be set explicitly."""
        cp = ConditionPrice(
            "c1",
            "Q?",
            0.50,
            0.50,
            1000,
            5000,
            clob_token_ids=["123", "456"],
        )
        assert cp.clob_token_ids == ["123", "456"]


# ------------------------------------------------------------------
# Market dataclass tests
# ------------------------------------------------------------------


class TestMarket:
    def test_sum_yes(self):
        c1 = ConditionPrice("c1", "Q1?", 0.30, 0.70, 1000, 5000)
        c2 = ConditionPrice("c2", "Q2?", 0.25, 0.75, 1000, 5000)
        m = Market("m1", "slug", "Q?", True, True, "politics", None, [c1, c2])
        assert m.sum_yes == pytest.approx(0.55)

    def test_negrisk_complement(self):
        c1 = ConditionPrice("c1", "Q1?", 0.30, 0.70, 1000, 5000)
        c2 = ConditionPrice("c2", "Q2?", 0.25, 0.75, 1000, 5000)
        c3 = ConditionPrice("c3", "Q3?", 0.35, 0.65, 1000, 5000)
        m = Market("m1", "slug", "Q?", True, True, "sports", None, [c1, c2, c3])
        assert m.negrisk_complement == pytest.approx(0.10)

    def test_is_negrisk_arb(self):
        """NegRisk arb requires negrisk=True and complement > 0.05."""
        c1 = ConditionPrice("c1", "Q1?", 0.30, 0.70, 1000, 5000)
        c2 = ConditionPrice("c2", "Q2?", 0.25, 0.75, 1000, 5000)
        m = Market("m1", "slug", "Q?", True, True, "politics", None, [c1, c2])
        # complement = 1 - 0.55 = 0.45 > 0.05
        assert m.is_negrisk_arb is True

    def test_not_negrisk_arb_if_not_negrisk(self):
        c1 = ConditionPrice("c1", "Q1?", 0.30, 0.70, 1000, 5000)
        m = Market("m1", "slug", "Q?", True, False, "sports", None, [c1])
        assert m.is_negrisk_arb is False


# ------------------------------------------------------------------
# GammaClient parsing tests
# ------------------------------------------------------------------


class TestGammaClientParsing:
    def test_parse_market_json_string_outcomes(self):
        """Parse market with JSON string outcomes/outcomePrices."""
        raw = {
            "id": "12345",
            "question": "Will X happen?",
            "slug": "will-x-happen",
            "conditionId": "0xabc",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.65","0.35"]',
            "active": True,
            "negRisk": False,
            "category": "politics",
            "endDate": "2026-12-31",
            "volume24hr": "5000",
            "clobTokenIds": '["111","222"]',
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.market_id == "12345"
        assert m.question == "Will X happen?"
        assert m.is_negrisk is False
        assert m.category == "politics"
        assert len(m.conditions) == 1
        assert m.conditions[0].outcome_yes == pytest.approx(0.65)
        assert m.conditions[0].outcome_no == pytest.approx(0.35)
        assert m.conditions[0].clob_token_ids == ["111", "222"]

    def test_parse_market_list_outcomes(self):
        """Parse market with list outcomes/outcomePrices."""
        raw = {
            "id": "999",
            "question": "Will Y happen?",
            "slug": "will-y-happen",
            "conditionId": "0xdef",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.40", "0.60"],
            "active": True,
            "negRisk": True,
            "volume24hr": 3000,
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.conditions[0].outcome_yes == pytest.approx(0.40)
        assert m.conditions[0].outcome_no == pytest.approx(0.60)
        assert m.is_negrisk is True

    def test_parse_market_tokens_override(self):
        """CLOB tokens should override outcomes/outcomePrices."""
        raw = {
            "id": "42",
            "question": "Token override?",
            "slug": "token-override",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.50","0.50"]',
            "tokens": [
                {"outcome": "Yes", "price": "0.70"},
                {"outcome": "No", "price": "0.30"},
            ],
            "active": True,
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.conditions[0].outcome_yes == pytest.approx(0.70)
        assert m.conditions[0].outcome_no == pytest.approx(0.30)

    def test_parse_market_neg_risk_field(self):
        """Gamma API uses 'negRisk' (not 'isNegRisk')."""
        raw = {
            "id": "neg1",
            "question": "NegRisk market",
            "slug": "negrisk",
            "negRisk": True,
            "active": True,
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.is_negrisk is True

    def test_parse_market_is_neg_risk_compat(self):
        """Also support 'isNegRisk' for backward compatibility."""
        raw = {
            "id": "neg2",
            "question": "NegRisk compat",
            "slug": "negrisk-compat",
            "isNegRisk": True,
            "active": True,
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.is_negrisk is True

    def test_parse_market_no_id(self):
        """Market with no ID should return None."""
        raw = {"question": "No ID?", "slug": "no-id"}
        m = GammaClient.parse_market(raw)
        assert m is None

    def test_parse_market_empty_dict(self):
        """Empty dict should return None."""
        m = GammaClient.parse_market({})
        assert m is None

    def test_parse_market_volume_24hr(self):
        """Should parse volume24hr from Gamma API."""
        raw = {
            "id": "vol1",
            "question": "Volume test",
            "slug": "volume-test",
            "volume24hr": "12345.67",
            "active": True,
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.conditions[0].volume_24h == pytest.approx(12345.67)

    def test_parse_market_category_inference(self):
        """When no category provided, infer from text."""
        raw = {
            "id": "cat1",
            "question": "Will bitcoin reach $100k?",
            "slug": "bitcoin-100k",
            "active": True,
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.category == "crypto"

    def test_parse_all_markets(self):
        """parse_all_markets should skip failures."""
        gamma = GammaClient(ssl_verify=False)
        raw_list = [
            {"id": "1", "question": "Q1", "slug": "q1", "active": True},
            {},  # should be skipped
            {"id": "2", "question": "Q2", "slug": "q2", "active": True},
            {"id": "3", "question": "", "slug": "q3", "active": True},  # empty question
        ]
        markets = gamma.parse_all_markets(raw_list)
        assert len(markets) == 2

    def test_parse_market_best_ask_fallback(self):
        """When outcomePrices are missing, use bestAsk."""
        raw = {
            "id": "ba1",
            "question": "Best ask test",
            "slug": "best-ask",
            "active": True,
            "bestAsk": "0.75",
        }
        m = GammaClient.parse_market(raw)
        assert m is not None
        assert m.conditions[0].outcome_yes == pytest.approx(0.75)
        assert m.conditions[0].outcome_no == pytest.approx(0.25)


# ------------------------------------------------------------------
# GammaClient HTTP tests
# ------------------------------------------------------------------


class TestGammaClientHTTP:
    def test_fetch_markets_page(self, gamma):
        """fetch_markets_page should return list of dicts."""
        mock_data = [
            {"id": "1", "question": "Q1"},
            {"id": "2", "question": "Q2"},
        ]
        with patch.object(gamma, "_get", return_value=mock_data) as mock:
            result = gamma.fetch_markets_page(offset=0, limit=2)
            assert len(result) == 2
            mock.assert_called_once_with(
                "/markets",
                params={
                    "limit": 2,
                    "offset": 0,
                    "active": "true",
                    "closed": "false",
                },
            )

    def test_fetch_markets_page_wrapped_response(self, gamma):
        """Handle wrapped response format."""
        mock_data = {"data": [{"id": "1"}]}
        with patch.object(gamma, "_get", return_value=mock_data):
            result = gamma.fetch_markets_page()
            assert len(result) == 1

    def test_fetch_events_page(self, gamma):
        """fetch_events_page should return list of event dicts."""
        mock_data = [
            {"id": "e1", "title": "Event 1", "markets": [{"id": "m1"}]},
        ]
        with patch.object(gamma, "_get", return_value=mock_data):
            result = gamma.fetch_events_page(offset=0, limit=1)
            assert len(result) == 1
            assert result[0]["title"] == "Event 1"
            assert len(result[0]["markets"]) == 1

    def test_fetch_all_active_markets_pagination(self, gamma):
        """Should paginate through multiple pages (Gamma-first path)."""
        gamma._prefer_us = False  # test Gamma pagination path
        page1 = [{"id": str(i)} for i in range(100)]
        page2 = [{"id": str(i)} for i in range(100, 150)]

        call_count = 0

        def mock_page(offset=0, **_kwargs):
            nonlocal call_count
            call_count += 1
            if offset == 0:
                return page1
            return page2

        with patch.object(gamma, "fetch_markets_page", side_effect=mock_page):
            result = gamma.fetch_all_active_markets(max_markets=200)
            assert len(result) == 150
            assert call_count == 2

    def test_fetch_all_active_markets_empty(self, gamma):
        """Empty first page should return empty list (Gamma-first path)."""
        gamma._prefer_us = False  # test Gamma path
        with patch.object(gamma, "fetch_markets_page", return_value=[]):
            result = gamma.fetch_all_active_markets()
            assert len(result) == 0

    def test_fetch_all_active_markets_us_first(self, gamma):
        """With prefer_us=True (default), should try US API first."""
        mock_markets = [{"id": "1"}, {"id": "2"}]
        with patch.object(gamma, "_fetch_us_markets", return_value=mock_markets):
            result = gamma.fetch_all_active_markets()
            assert len(result) == 2

    def test_fetch_market_single(self, gamma):
        """fetch_market should return single market dict."""
        mock_data = {"id": "m1", "question": "Single market"}
        with patch.object(gamma, "_get", return_value=mock_data):
            result = gamma.fetch_market("m1")
            assert result["id"] == "m1"

    def test_fetch_market_fallback_to_us(self, gamma):
        """Should fall back to US API on Gamma failure."""
        mock_us_data = {"id": "m1", "question": "From US API"}

        with (
            patch.object(
                gamma,
                "_get",
                side_effect=requests.ConnectionError("geo-blocked"),
            ),
            patch.object(gamma, "_get_us", return_value=mock_us_data),
        ):
            result = gamma.fetch_market("m1")
            assert result["question"] == "From US API"

    def test_geo_block_detection(self, gamma):
        """HTML 404 should raise ConnectionError (geo-block)."""
        mock_resp = _mock_response(
            status_code=404,
            text="<html>Azure Front Door</html>",
            content_type="text/html",
        )
        with (
            patch.object(gamma._session, "get", return_value=mock_resp),
            pytest.raises(requests.ConnectionError, match="geo-blocked"),
        ):
            gamma._get("/markets")

    def test_rate_limit_retry(self, gamma):
        """429 should trigger retry with backoff."""
        resp_429 = _mock_response(status_code=429)
        resp_ok = _mock_response(status_code=200, json_data={"ok": True})

        with patch.object(gamma._session, "get", side_effect=[resp_429, resp_ok]):
            result = gamma._get("/markets")
            assert result == {"ok": True}

    def test_all_retries_exhausted(self, gamma):
        """Should raise after all retries fail."""
        resp_500 = _mock_response(status_code=500)

        with (
            patch.object(gamma._session, "get", return_value=resp_500),
            pytest.raises(requests.HTTPError),
        ):
            gamma._get("/markets")


# ------------------------------------------------------------------
# Category inference tests
# ------------------------------------------------------------------


class TestCategoryInference:
    def test_sports(self):
        assert _infer_category("Will the NBA finals be exciting?") == "sports"

    def test_politics(self):
        assert _infer_category("Will Trump become president?") == "politics"

    def test_crypto(self):
        assert _infer_category("Will bitcoin reach $100k?") == "crypto"

    def test_finance(self):
        assert _infer_category("Will the fed cut rates?") == "finance"

    def test_geopolitics(self):
        assert _infer_category("Will there be a ceasefire in Ukraine?") == "geopolitics"

    def test_other(self):
        assert _infer_category("Will it rain tomorrow?") == "other"

    def test_word_boundary_short_tokens(self):
        """Short tokens like 'eth' should use word boundaries."""
        # "eth" inside "something" should NOT match crypto
        assert _infer_category("Is something happening?") == "other"
        # "eth" as standalone word should match
        assert _infer_category("Will ETH reach $5000?") == "crypto"


# ------------------------------------------------------------------
# Orderbook dataclass tests
# ------------------------------------------------------------------


class TestOrderbook:
    def test_best_bid_ask(self):
        book = Orderbook(
            market="0xabc",
            asset_id="123",
            bids=[OrderbookLevel(0.53, 100), OrderbookLevel(0.52, 200)],
            asks=[OrderbookLevel(0.54, 50), OrderbookLevel(0.55, 150)],
        )
        assert book.best_bid == 0.53
        assert book.best_ask == 0.54

    def test_spread(self):
        book = Orderbook(
            market="0xabc",
            asset_id="123",
            bids=[OrderbookLevel(0.53, 100)],
            asks=[OrderbookLevel(0.55, 50)],
        )
        assert book.spread == pytest.approx(0.02)

    def test_midpoint(self):
        book = Orderbook(
            market="0xabc",
            asset_id="123",
            bids=[OrderbookLevel(0.53, 100)],
            asks=[OrderbookLevel(0.55, 50)],
        )
        assert book.midpoint == pytest.approx(0.54)

    def test_empty_book(self):
        book = Orderbook(market="0xabc", asset_id="123")
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.spread is None
        assert book.midpoint is None
        assert book.bid_depth == 0
        assert book.ask_depth == 0

    def test_depth(self):
        book = Orderbook(
            market="0xabc",
            asset_id="123",
            bids=[OrderbookLevel(0.53, 100), OrderbookLevel(0.52, 200)],
            asks=[OrderbookLevel(0.54, 50), OrderbookLevel(0.55, 150)],
        )
        assert book.bid_depth == 300
        assert book.ask_depth == 200

    def test_one_sided_book(self):
        """Only bids, no asks."""
        book = Orderbook(
            market="0xabc",
            asset_id="123",
            bids=[OrderbookLevel(0.50, 100)],
        )
        assert book.best_bid == 0.50
        assert book.best_ask is None
        assert book.spread is None
        assert book.midpoint is None


# ------------------------------------------------------------------
# PriceHistory dataclass tests
# ------------------------------------------------------------------


class TestPriceHistory:
    def test_prices_and_timestamps(self):
        ph = PriceHistory(
            token_id="123",  # noqa: S106
            interval="1h",
            points=[
                PricePoint(1000, 0.50),
                PricePoint(2000, 0.55),
                PricePoint(3000, 0.53),
            ],
        )
        assert ph.prices == [0.50, 0.55, 0.53]
        assert ph.timestamps == [1000, 2000, 3000]

    def test_empty_history(self):
        ph = PriceHistory(token_id="123", interval="1d")  # noqa: S106
        assert ph.prices == []
        assert ph.timestamps == []


# ------------------------------------------------------------------
# ClobClient HTTP tests
# ------------------------------------------------------------------


class TestClobClientHTTP:
    def test_get_midpoint(self, clob):
        """Should parse midpoint response."""
        with patch.object(clob, "_get", return_value={"mid": "0.535"}):
            mid = clob.get_midpoint("token123")
            assert mid == pytest.approx(0.535)

    def test_get_midpoint_none(self, clob):
        """Should return None if no mid field."""
        with patch.object(clob, "_get", return_value={}):
            mid = clob.get_midpoint("token123")
            assert mid is None

    def test_get_spread(self, clob):
        with patch.object(clob, "_get", return_value={"spread": "0.01"}):
            spread = clob.get_spread("token123")
            assert spread == pytest.approx(0.01)

    def test_get_price(self, clob):
        with patch.object(clob, "_get", return_value={"price": "0.53"}):
            price = clob.get_price("token123", "BUY")
            assert price == pytest.approx(0.53)

    def test_get_tick_size(self, clob):
        with patch.object(clob, "_get", return_value={"minimum_tick_size": 0.01}):
            tick = clob.get_tick_size("token123")
            assert tick == pytest.approx(0.01)

    def test_get_last_trade_price(self, clob):
        with patch.object(clob, "_get", return_value={"price": "0.54"}):
            last = clob.get_last_trade_price("token123")
            assert last == pytest.approx(0.54)

    def test_get_book(self, clob):
        """Should parse orderbook correctly."""
        raw_book = {
            "market": "0xabc",
            "asset_id": "token123",
            "bids": [
                {"price": "0.52", "size": "200"},
                {"price": "0.53", "size": "100"},
            ],
            "asks": [
                {"price": "0.55", "size": "150"},
                {"price": "0.54", "size": "50"},
            ],
            "timestamp": "12345",
            "hash": "0xhash",
        }
        with patch.object(clob, "_get", return_value=raw_book):
            book = clob.get_book("token123")
            # Bids should be sorted descending
            assert book.bids[0].price == 0.53
            assert book.bids[1].price == 0.52
            # Asks should be sorted ascending
            assert book.asks[0].price == 0.54
            assert book.asks[1].price == 0.55
            assert book.market == "0xabc"
            assert book.timestamp == "12345"

    def test_get_books_multiple(self, clob):
        """get_books should fetch multiple orderbooks."""
        book1 = {
            "market": "0x1",
            "asset_id": "t1",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.52", "size": "50"}],
        }
        book2 = {
            "market": "0x2",
            "asset_id": "t2",
            "bids": [{"price": "0.60", "size": "80"}],
            "asks": [{"price": "0.62", "size": "40"}],
        }
        with patch.object(clob, "_get", side_effect=[book1, book2]):
            books = clob.get_books(["t1", "t2"])
            assert len(books) == 2
            assert books[0].market == "0x1"
            assert books[1].market == "0x2"

    def test_get_prices_history(self, clob):
        """Should parse price history correctly."""
        raw_history = {
            "history": [
                {"t": 3000, "p": 0.53},
                {"t": 1000, "p": 0.50},
                {"t": 2000, "p": 0.55},
            ]
        }
        with patch.object(clob, "_get", return_value=raw_history):
            ph = clob.get_prices_history("token123", interval="1h")
            # Should be sorted by timestamp ascending
            assert ph.points[0].timestamp == 1000
            assert ph.points[1].timestamp == 2000
            assert ph.points[2].timestamp == 3000
            assert ph.prices == [0.50, 0.55, 0.53]
            assert ph.interval == "1h"

    def test_get_prices_history_with_params(self, clob):
        """Should pass fidelity, startTs, endTs params."""
        with patch.object(clob, "_get", return_value={"history": []}) as mock:
            clob.get_prices_history(
                "token123",
                interval="1d",
                fidelity=60,
                start_ts=1000,
                end_ts=2000,
            )
            mock.assert_called_once_with(
                "/prices-history",
                params={
                    "market": "token123",
                    "interval": "1d",
                    "fidelity": 60,
                    "startTs": 1000,
                    "endTs": 2000,
                },
            )

    def test_get_market_snapshot(self, clob):
        """Snapshot should aggregate multiple endpoint calls."""
        with (
            patch.object(clob, "get_midpoint", return_value=0.535),
            patch.object(clob, "get_spread", return_value=0.01),
            patch.object(clob, "get_tick_size", return_value=0.01),
            patch.object(clob, "get_last_trade_price", return_value=0.54),
        ):
            snap = clob.get_market_snapshot("token123")
            assert snap["token_id"] == "token123"  # noqa: S105
            assert snap["midpoint"] == 0.535
            assert snap["spread"] == 0.01
            assert snap["tick_size"] == 0.01
            assert snap["last_trade_price"] == 0.54

    def test_get_market_snapshot_partial_failure(self, clob):
        """Snapshot should handle partial endpoint failures gracefully."""
        with (
            patch.object(clob, "get_midpoint", return_value=0.535),
            patch.object(
                clob,
                "get_spread",
                side_effect=requests.HTTPError("500"),
            ),
            patch.object(clob, "get_tick_size", return_value=0.01),
            patch.object(
                clob,
                "get_last_trade_price",
                side_effect=requests.ConnectionError("timeout"),
            ),
        ):
            snap = clob.get_market_snapshot("token123")
            assert snap["midpoint"] == 0.535
            assert snap["spread"] is None
            assert snap["tick_size"] == 0.01
            assert snap["last_trade_price"] is None

    def test_rate_limit_retry(self, clob):
        """429 should trigger retry with backoff."""
        resp_429 = _mock_response(status_code=429)
        resp_ok = _mock_response(status_code=200, json_data={"mid": "0.50"})

        with patch.object(clob._session, "get", side_effect=[resp_429, resp_ok]):
            result = clob._get("/midpoint", params={"token_id": "t1"})
            assert result == {"mid": "0.50"}

    def test_connection_error_retry(self, clob):
        """ConnectionError should retry."""
        with patch.object(
            clob._session,
            "get",
            side_effect=[
                requests.ConnectionError("conn fail"),
                _mock_response(200, {"mid": "0.50"}),
            ],
        ):
            result = clob._get("/midpoint")
            assert result == {"mid": "0.50"}

    def test_all_retries_exhausted(self, clob):
        """Should raise after all retries fail."""
        resp_500 = _mock_response(status_code=500)
        with (
            patch.object(clob._session, "get", return_value=resp_500),
            pytest.raises(requests.HTTPError),
        ):
            clob._get("/midpoint")

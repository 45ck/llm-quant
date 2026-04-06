"""Tests for Polymarket Parquet data store (pm_data module).

Covers:
  - Market metadata extraction and schema validation
  - Deduplication logic on condition_id
  - Incremental price history fetching
  - Orderbook snapshot ingestion
  - Data loading functions (load_markets, load_price_history, load_latest_orderbook)
  - Summary statistics (get_data_summary)

All tests use mocked API responses and temporary directories -- no live
API calls and no writes to real data directories.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from llm_quant.arb.clob_client import (
    ClobClient,
    Orderbook,
    OrderbookLevel,
    PriceHistory,
    PricePoint,
)
from llm_quant.arb.gamma_client import GammaClient
from llm_quant.arb.pm_data import (
    MARKETS_SCHEMA,
    ORDERBOOK_SCHEMA,
    PRICES_SCHEMA,
    _extract_market_row,
    _get_last_timestamp,
    get_data_summary,
    ingest_markets,
    ingest_orderbook_snapshot,
    ingest_price_history,
    load_latest_orderbook,
    load_markets,
    load_price_history,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all data paths to a temporary directory for each test."""
    tmp_data = tmp_path / "polymarket"
    tmp_markets = tmp_data / "markets.parquet"
    tmp_prices = tmp_data / "prices"
    tmp_orderbooks = tmp_data / "orderbooks"

    monkeypatch.setattr("llm_quant.arb.pm_data.DATA_DIR", tmp_data)
    monkeypatch.setattr("llm_quant.arb.pm_data.MARKETS_PATH", tmp_markets)
    monkeypatch.setattr("llm_quant.arb.pm_data.PRICES_DIR", tmp_prices)
    monkeypatch.setattr("llm_quant.arb.pm_data.ORDERBOOKS_DIR", tmp_orderbooks)
    # Disable rate limiting in tests
    monkeypatch.setattr("llm_quant.arb.pm_data._RATE_LIMIT_SLEEP", 0)


@pytest.fixture
def gamma() -> GammaClient:
    return GammaClient(ssl_verify=False)


@pytest.fixture
def clob() -> ClobClient:
    return ClobClient(ssl_verify=False)


def _raw_market(
    condition_id: str = "0xabc",
    question: str = "Will X happen?",
    slug: str = "will-x-happen",
    category: str = "politics",
    *,
    neg_risk: bool = False,
    volume: float = 1000.0,
    liquidity: float = 500.0,
    clob_tokens: list[str] | None = None,
) -> dict:
    """Build a raw market dict as returned by Gamma API."""
    tokens = clob_tokens or ["token_yes_1", "token_no_1"]
    return {
        "id": "12345",
        "conditionId": condition_id,
        "question": question,
        "slug": slug,
        "category": category,
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.65","0.35"]',
        "active": True,
        "negRisk": neg_risk,
        "endDate": "2026-12-31",
        "volume24hr": str(volume),
        "liquidity": str(liquidity),
        "clobTokenIds": json.dumps(tokens),
    }


# ------------------------------------------------------------------
# _extract_market_row tests
# ------------------------------------------------------------------


class TestExtractMarketRow:
    def test_basic_extraction(self):
        """Should extract all fields from a standard raw market dict."""
        raw = _raw_market()
        row = _extract_market_row(raw)
        assert row is not None
        assert row["condition_id"] == "0xabc"
        assert row["question"] == "Will X happen?"
        assert row["category"] == "politics"
        assert row["slug"] == "will-x-happen"
        assert row["volume"] == 1000.0
        assert row["liquidity"] == 500.0
        assert row["is_neg_risk"] is False
        assert row["end_date"] == "2026-12-31"

    def test_neg_risk(self):
        raw = _raw_market(neg_risk=True)
        row = _extract_market_row(raw)
        assert row is not None
        assert row["is_neg_risk"] is True

    def test_clob_token_ids_json_string(self):
        """Should parse clobTokenIds JSON string into tokens field."""
        raw = _raw_market(clob_tokens=["aaa", "bbb"])
        row = _extract_market_row(raw)
        assert row is not None
        tokens = json.loads(row["tokens"])
        assert tokens == ["aaa", "bbb"]

    def test_clob_token_ids_list(self):
        """Should handle clobTokenIds as native list."""
        raw = _raw_market()
        raw["clobTokenIds"] = ["aaa", "bbb"]
        row = _extract_market_row(raw)
        assert row is not None
        tokens = json.loads(row["tokens"])
        assert tokens == ["aaa", "bbb"]

    def test_missing_condition_id_uses_id(self):
        """Should fall back to 'id' field when conditionId is missing."""
        raw = _raw_market()
        del raw["conditionId"]
        raw["id"] = "fallback_id"
        row = _extract_market_row(raw)
        assert row is not None
        assert row["condition_id"] == "fallback_id"

    def test_no_id_returns_none(self):
        """Should return None when no ID is available."""
        raw = {"question": "No ID market"}
        row = _extract_market_row(raw)
        assert row is None

    def test_outcome_prices_string(self):
        raw = _raw_market()
        row = _extract_market_row(raw)
        assert row is not None
        assert row["outcome_prices"] == '["0.65","0.35"]'

    def test_outcome_prices_list(self):
        raw = _raw_market()
        raw["outcomePrices"] = ["0.70", "0.30"]
        row = _extract_market_row(raw)
        assert row is not None
        assert row["outcome_prices"] == '["0.70", "0.30"]'

    def test_volume_field_variants(self):
        """Should try volume, volume24hr, volumeNum in order."""
        raw = _raw_market()
        raw["volume"] = "9999"
        row = _extract_market_row(raw)
        assert row is not None
        assert row["volume"] == 9999.0


# ------------------------------------------------------------------
# Market ingestion tests
# ------------------------------------------------------------------


class TestIngestMarkets:
    def test_ingest_creates_parquet(self, gamma, tmp_path, monkeypatch):
        """Should create markets.parquet with correct schema."""
        raw_markets = [
            _raw_market(condition_id="0x111"),
            _raw_market(condition_id="0x222", question="Another market?"),
        ]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw_markets):
            df = ingest_markets(gamma, dry_run=False)

        assert len(df) == 2
        assert set(df.columns) == set(MARKETS_SCHEMA.keys())
        assert sorted(df["condition_id"].to_list()) == ["0x111", "0x222"]

    def test_deduplication(self, gamma):
        """Should deduplicate on condition_id, keeping last."""
        raw_markets = [
            _raw_market(condition_id="0x111", question="First"),
            _raw_market(condition_id="0x111", question="Updated"),
        ]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw_markets):
            df = ingest_markets(gamma, dry_run=True)

        assert len(df) == 1
        assert df["question"][0] == "Updated"

    def test_merge_with_existing(self, gamma, monkeypatch):
        """Should merge new data with existing parquet, deduplicating."""
        # First ingestion
        raw1 = [_raw_market(condition_id="0x111", question="First")]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw1):
            ingest_markets(gamma, dry_run=False)

        # Second ingestion with overlap + new
        raw2 = [
            _raw_market(condition_id="0x111", question="Updated"),
            _raw_market(condition_id="0x222", question="New"),
        ]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw2):
            df = ingest_markets(gamma, dry_run=False)

        assert len(df) == 2
        # Updated record should have new question
        updated = df.filter(pl.col("condition_id") == "0x111")
        assert updated["question"][0] == "Updated"

    def test_dry_run_no_write(self, gamma, monkeypatch):
        """Dry run should not create the parquet file."""
        # Get the current (monkeypatched) MARKETS_PATH
        import llm_quant.arb.pm_data as pm_mod

        raw = [_raw_market()]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw):
            df = ingest_markets(gamma, dry_run=True)

        assert len(df) == 1
        assert not pm_mod.MARKETS_PATH.exists()

    def test_empty_response(self, gamma):
        """Should return empty DataFrame on empty API response."""
        with patch.object(gamma, "fetch_all_active_markets", return_value=[]):
            df = ingest_markets(gamma, dry_run=True)

        assert df.is_empty()
        assert set(df.columns) == set(MARKETS_SCHEMA.keys())


# ------------------------------------------------------------------
# Price history ingestion tests
# ------------------------------------------------------------------


class TestIngestPriceHistory:
    def test_ingest_new_token(self, clob):
        """Should create price parquet for a new token."""
        history = PriceHistory(
            token_id="tok1",  # noqa: S106
            interval="1h",
            points=[
                PricePoint(1000, 0.50),
                PricePoint(2000, 0.55),
                PricePoint(3000, 0.53),
            ],
        )
        with patch.object(clob, "get_prices_history", return_value=history):
            df = ingest_price_history(clob, "tok1", dry_run=False)

        assert len(df) == 3
        assert df["timestamp"].to_list() == [1000, 2000, 3000]
        assert df["token_id"][0] == "tok1"
        assert set(df.columns) == set(PRICES_SCHEMA.keys())

    def test_incremental_fetch(self, clob, monkeypatch):
        """Should only fetch data after last stored timestamp."""
        import llm_quant.arb.pm_data as pm_mod

        # Create initial data
        initial = pl.DataFrame(
            [
                {"timestamp": 1000, "price": 0.50, "token_id": "tok1"},
                {"timestamp": 2000, "price": 0.55, "token_id": "tok1"},
            ],
            schema=PRICES_SCHEMA,
        )
        pm_mod.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        initial.write_parquet(pm_mod.PRICES_DIR / "tok1.parquet")

        # New data
        new_history = PriceHistory(
            token_id="tok1",  # noqa: S106
            interval="1h",
            points=[PricePoint(3000, 0.60)],
        )

        def mock_get_prices(**kwargs):
            # Verify incremental: start_ts should be last_ts + 1 = 2001
            assert kwargs.get("start_ts") == 2001
            return new_history

        with patch.object(clob, "get_prices_history", side_effect=mock_get_prices):
            df = ingest_price_history(clob, "tok1", dry_run=False)

        # Should have all 3 points combined
        assert len(df) == 3
        assert df["timestamp"].to_list() == [1000, 2000, 3000]

    def test_dedup_on_timestamp(self, clob, monkeypatch):
        """Should deduplicate on timestamp when merging."""
        import llm_quant.arb.pm_data as pm_mod

        # Create initial data
        initial = pl.DataFrame(
            [
                {"timestamp": 1000, "price": 0.50, "token_id": "tok1"},
                {"timestamp": 2000, "price": 0.55, "token_id": "tok1"},
            ],
            schema=PRICES_SCHEMA,
        )
        pm_mod.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        initial.write_parquet(pm_mod.PRICES_DIR / "tok1.parquet")

        # Overlapping data
        overlap_history = PriceHistory(
            token_id="tok1",  # noqa: S106
            interval="1h",
            points=[
                PricePoint(2000, 0.56),  # Updated price for same timestamp
                PricePoint(3000, 0.60),  # New
            ],
        )

        with patch.object(clob, "get_prices_history", return_value=overlap_history):
            df = ingest_price_history(clob, "tok1", dry_run=False)

        # Should have 3 unique timestamps
        assert len(df) == 3
        # The overlapping timestamp should use the new value
        ts_2000 = df.filter(pl.col("timestamp") == 2000)
        assert ts_2000["price"][0] == pytest.approx(0.56)

    def test_api_failure_returns_empty(self, clob):
        """Should return empty DataFrame on API failure."""
        with patch.object(
            clob,
            "get_prices_history",
            side_effect=Exception("API error"),
        ):
            df = ingest_price_history(clob, "tok_fail", dry_run=True)

        assert df.is_empty()

    def test_no_new_data(self, clob):
        """Should return existing data when API returns no new points."""
        import llm_quant.arb.pm_data as pm_mod

        # Create initial data
        initial = pl.DataFrame(
            [{"timestamp": 1000, "price": 0.50, "token_id": "tok1"}],
            schema=PRICES_SCHEMA,
        )
        pm_mod.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        initial.write_parquet(pm_mod.PRICES_DIR / "tok1.parquet")

        empty_history = PriceHistory(token_id="tok1", interval="1h", points=[])  # noqa: S106
        with patch.object(clob, "get_prices_history", return_value=empty_history):
            df = ingest_price_history(clob, "tok1", dry_run=False)

        assert len(df) == 1
        assert df["timestamp"][0] == 1000


# ------------------------------------------------------------------
# Orderbook snapshot tests
# ------------------------------------------------------------------


class TestIngestOrderbookSnapshot:
    def test_snapshot_creates_parquet(self, clob):
        """Should create orderbook parquet with correct schema."""
        book = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.53, 100), OrderbookLevel(0.52, 200)],
            asks=[OrderbookLevel(0.54, 50), OrderbookLevel(0.55, 150)],
        )
        with patch.object(clob, "get_book", return_value=book):
            df = ingest_orderbook_snapshot(clob, "0xabc", ["tok1"], dry_run=False)

        assert len(df) == 4  # 2 bids + 2 asks
        assert set(df.columns) == set(ORDERBOOK_SCHEMA.keys())
        assert df.filter(pl.col("side") == "bid").height == 2
        assert df.filter(pl.col("side") == "ask").height == 2

    def test_snapshot_multiple_tokens(self, clob):
        """Should handle multiple tokens for one condition."""
        book1 = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.53, 100)],
            asks=[OrderbookLevel(0.54, 50)],
        )
        book2 = Orderbook(
            market="0xabc",
            asset_id="tok2",
            bids=[OrderbookLevel(0.47, 80)],
            asks=[OrderbookLevel(0.48, 60)],
        )
        with patch.object(clob, "get_book", side_effect=[book1, book2]):
            df = ingest_orderbook_snapshot(
                clob, "0xabc", ["tok1", "tok2"], dry_run=False
            )

        assert len(df) == 4
        assert df.filter(pl.col("token_id") == "tok1").height == 2
        assert df.filter(pl.col("token_id") == "tok2").height == 2

    def test_snapshot_api_failure_partial(self, clob):
        """Should continue on per-token failure."""
        book1 = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.53, 100)],
            asks=[],
        )
        with patch.object(
            clob,
            "get_book",
            side_effect=[book1, Exception("API error")],
        ):
            df = ingest_orderbook_snapshot(
                clob, "0xabc", ["tok1", "tok2"], dry_run=False
            )

        # Should have data from tok1 only
        assert len(df) == 1
        assert df["token_id"][0] == "tok1"

    def test_dry_run_no_write(self, clob):
        """Dry run should not create files."""
        import llm_quant.arb.pm_data as pm_mod

        book = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.53, 100)],
            asks=[OrderbookLevel(0.54, 50)],
        )
        with patch.object(clob, "get_book", return_value=book):
            df = ingest_orderbook_snapshot(clob, "0xabc", ["tok1"], dry_run=True)

        assert len(df) == 2
        # No files should be written
        if pm_mod.ORDERBOOKS_DIR.exists():
            assert list(pm_mod.ORDERBOOKS_DIR.glob("*.parquet")) == []


# ------------------------------------------------------------------
# Data loading function tests
# ------------------------------------------------------------------


class TestLoadMarkets:
    def test_load_when_exists(self, gamma):
        """Should load markets from existing parquet."""
        raw = [_raw_market(condition_id="0x111")]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw):
            ingest_markets(gamma, dry_run=False)

        df = load_markets()
        assert len(df) == 1
        assert df["condition_id"][0] == "0x111"

    def test_load_when_missing(self):
        """Should return empty DataFrame with correct schema."""
        df = load_markets()
        assert df.is_empty()
        assert set(df.columns) == set(MARKETS_SCHEMA.keys())


class TestLoadPriceHistory:
    def test_load_existing(self, clob):
        """Should load price history sorted by timestamp."""
        history = PriceHistory(
            token_id="tok1",  # noqa: S106
            interval="1h",
            points=[
                PricePoint(3000, 0.53),
                PricePoint(1000, 0.50),
                PricePoint(2000, 0.55),
            ],
        )
        with patch.object(clob, "get_prices_history", return_value=history):
            ingest_price_history(clob, "tok1", dry_run=False)

        df = load_price_history("tok1")
        assert len(df) == 3
        assert df["timestamp"].to_list() == [1000, 2000, 3000]

    def test_load_missing(self):
        """Should return empty DataFrame for unknown token."""
        df = load_price_history("nonexistent_token")
        assert df.is_empty()
        assert set(df.columns) == set(PRICES_SCHEMA.keys())


class TestLoadLatestOrderbook:
    def test_load_latest(self, clob):
        """Should load the most recent orderbook snapshot."""
        book = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.53, 100)],
            asks=[OrderbookLevel(0.54, 50)],
        )
        with patch.object(clob, "get_book", return_value=book):
            ingest_orderbook_snapshot(clob, "0xabc", ["tok1"], dry_run=False)

        df = load_latest_orderbook("0xabc")
        assert len(df) == 2
        assert df["condition_id"][0] == "0xabc"

    def test_load_latest_multiple_snapshots(self, clob):
        """Should return the latest snapshot when multiple exist."""
        book1 = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.50, 100)],
            asks=[],
        )
        book2 = Orderbook(
            market="0xabc",
            asset_id="tok1",
            bids=[OrderbookLevel(0.60, 200)],
            asks=[OrderbookLevel(0.61, 50)],
        )

        with patch.object(clob, "get_book", return_value=book1):
            ingest_orderbook_snapshot(clob, "0xabc", ["tok1"], dry_run=False)

        # Small delay to ensure different timestamp in filename
        time.sleep(0.01)

        with patch.object(clob, "get_book", return_value=book2):
            ingest_orderbook_snapshot(clob, "0xabc", ["tok1"], dry_run=False)

        df = load_latest_orderbook("0xabc")
        # Should have data from the second (latest) snapshot
        assert len(df) == 2  # 1 bid + 1 ask
        bid = df.filter(pl.col("side") == "bid")
        assert bid["price"][0] == pytest.approx(0.60)

    def test_load_missing(self):
        """Should return empty DataFrame for unknown condition."""
        df = load_latest_orderbook("nonexistent_condition")
        assert df.is_empty()
        assert set(df.columns) == set(ORDERBOOK_SCHEMA.keys())


# ------------------------------------------------------------------
# Summary statistics tests
# ------------------------------------------------------------------


class TestGetDataSummary:
    def test_empty_summary(self):
        """Should return zeros when no data exists."""
        summary = get_data_summary()
        assert summary["markets_count"] == 0
        assert summary["markets_file_exists"] is False
        assert summary["price_tokens_count"] == 0
        assert summary["orderbook_snapshots_count"] == 0
        assert summary["oldest_price_ts"] is None
        assert summary["newest_price_ts"] is None

    def test_summary_with_data(self, gamma, clob):
        """Should reflect correct counts after ingestion."""
        raw = [
            _raw_market(condition_id="0x111", clob_tokens=["t1"]),
            _raw_market(condition_id="0x222", clob_tokens=["t2"]),
        ]
        with patch.object(gamma, "fetch_all_active_markets", return_value=raw):
            ingest_markets(gamma, dry_run=False)

        # Ingest prices for one token
        history = PriceHistory(
            token_id="t1",  # noqa: S106
            interval="1h",
            points=[PricePoint(1000, 0.50), PricePoint(2000, 0.55)],
        )
        with patch.object(clob, "get_prices_history", return_value=history):
            ingest_price_history(clob, "t1", dry_run=False)

        summary = get_data_summary()
        assert summary["markets_count"] == 2
        assert summary["markets_file_exists"] is True
        assert summary["price_tokens_count"] == 1
        assert summary["oldest_price_ts"] is not None
        assert summary["newest_price_ts"] is not None
        assert summary["markets_last_modified"] is not None


# ------------------------------------------------------------------
# Schema validation tests
# ------------------------------------------------------------------


class TestSchemas:
    def test_markets_schema_columns(self):
        """Markets schema should have all required columns."""
        expected_cols = {
            "condition_id",
            "question",
            "category",
            "slug",
            "outcome_prices",
            "volume",
            "liquidity",
            "end_date",
            "is_neg_risk",
            "tokens",
        }
        assert set(MARKETS_SCHEMA.keys()) == expected_cols

    def test_prices_schema_columns(self):
        """Prices schema should have timestamp, price, token_id."""
        expected_cols = {"timestamp", "price", "token_id"}
        assert set(PRICES_SCHEMA.keys()) == expected_cols

    def test_orderbook_schema_columns(self):
        """Orderbook schema should have all required columns."""
        expected_cols = {
            "timestamp",
            "condition_id",
            "token_id",
            "side",
            "price",
            "size",
        }
        assert set(ORDERBOOK_SCHEMA.keys()) == expected_cols

    def test_empty_dataframe_with_markets_schema(self):
        """Empty DataFrame with markets schema should be valid."""
        df = pl.DataFrame(schema=MARKETS_SCHEMA)
        assert df.is_empty()
        assert df.schema == MARKETS_SCHEMA

    def test_empty_dataframe_with_prices_schema(self):
        """Empty DataFrame with prices schema should be valid."""
        df = pl.DataFrame(schema=PRICES_SCHEMA)
        assert df.is_empty()
        assert df.schema == PRICES_SCHEMA

    def test_empty_dataframe_with_orderbook_schema(self):
        """Empty DataFrame with orderbook schema should be valid."""
        df = pl.DataFrame(schema=ORDERBOOK_SCHEMA)
        assert df.is_empty()
        assert df.schema == ORDERBOOK_SCHEMA


# ------------------------------------------------------------------
# _get_last_timestamp tests
# ------------------------------------------------------------------


class TestGetLastTimestamp:
    def test_no_file(self):
        """Should return None when no file exists."""
        assert _get_last_timestamp("nonexistent") is None

    def test_with_data(self, clob):
        """Should return max timestamp from existing file."""
        history = PriceHistory(
            token_id="tok1",  # noqa: S106
            interval="1h",
            points=[PricePoint(1000, 0.50), PricePoint(3000, 0.55)],
        )
        with patch.object(clob, "get_prices_history", return_value=history):
            ingest_price_history(clob, "tok1", dry_run=False)

        assert _get_last_timestamp("tok1") == 3000

    def test_empty_file(self):
        """Should return None for empty parquet file."""
        import llm_quant.arb.pm_data as pm_mod

        pm_mod.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        empty = pl.DataFrame(schema=PRICES_SCHEMA)
        empty.write_parquet(pm_mod.PRICES_DIR / "empty_tok.parquet")

        assert _get_last_timestamp("empty_tok") is None

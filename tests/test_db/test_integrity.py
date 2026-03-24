"""Tests for the SHA-256 hash chain on the trades table."""

from __future__ import annotations

from datetime import date, datetime

from llm_quant.db.integrity import (
    GENESIS_HASH,
    backfill_hashes,
    compute_trade_hash,
    get_latest_hash,
    verify_chain,
)

# ── helpers ──────────────────────────────────────────────────────────

SAMPLE_CREATED = datetime(2025, 6, 15, 10, 30, 0)  # noqa: DTZ001


def _insert_trade(
    conn,
    trade_id,
    *,
    prev_hash="",
    row_hash="",
    symbol="SPY",
    action="buy",
    shares=10.0,
    price=450.0,
    notional=4500.0,
    conviction="high",
    reasoning="test",
    decision_id=1,
    trade_date=None,
    created_at=None,
):
    """Insert a trade row directly (bypasses ledger to control hashes)."""
    trade_date = trade_date or date(2025, 6, 15)
    created_at = created_at or SAMPLE_CREATED
    conn.execute(
        """
        INSERT INTO trades (
            trade_id, date, symbol, action, shares, price, notional,
            conviction, reasoning, llm_decision_id, created_at,
            prev_hash, row_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            trade_id,
            trade_date,
            symbol,
            action,
            shares,
            price,
            notional,
            conviction,
            reasoning,
            decision_id,
            created_at,
            prev_hash,
            row_hash,
        ],
    )
    conn.commit()


def _insert_chained(conn, trade_id, prev_hash, **kwargs):
    """Insert a trade and compute valid hashes."""
    defaults = {
        "symbol": "SPY",
        "action": "buy",
        "shares": 10.0,
        "price": 450.0,
        "notional": 4500.0,
        "conviction": "high",
        "reasoning": "test",
        "decision_id": 1,
        "trade_date": date(2025, 6, 15),
        "created_at": SAMPLE_CREATED,
    }
    defaults.update(kwargs)
    row_hash = compute_trade_hash(
        prev_hash,
        trade_id,
        defaults["trade_date"],
        defaults["symbol"],
        defaults["action"],
        defaults["shares"],
        defaults["price"],
        defaults["notional"],
        defaults["conviction"],
        defaults["reasoning"],
        defaults["decision_id"],
        defaults["created_at"],
    )
    _insert_trade(conn, trade_id, prev_hash=prev_hash, row_hash=row_hash, **defaults)
    return row_hash


# ── tests ────────────────────────────────────────────────────────────


class TestComputeTradeHash:
    """compute_trade_hash is deterministic."""

    def test_deterministic(self):
        h1 = compute_trade_hash(
            GENESIS_HASH,
            1,
            date(2025, 6, 15),
            "SPY",
            "buy",
            10.0,
            450.0,
            4500.0,
            "high",
            "test",
            1,
            SAMPLE_CREATED,
        )
        h2 = compute_trade_hash(
            GENESIS_HASH,
            1,
            date(2025, 6, 15),
            "SPY",
            "buy",
            10.0,
            450.0,
            4500.0,
            "high",
            "test",
            1,
            SAMPLE_CREATED,
        )
        assert h1 == h2
        assert len(h1) == 64  # hex-encoded SHA-256

    def test_different_input_different_hash(self):
        h1 = compute_trade_hash(
            GENESIS_HASH,
            1,
            date(2025, 6, 15),
            "SPY",
            "buy",
            10.0,
            450.0,
            4500.0,
            "high",
            "test",
            1,
            SAMPLE_CREATED,
        )
        h2 = compute_trade_hash(
            GENESIS_HASH,
            1,
            date(2025, 6, 15),
            "QQQ",
            "buy",
            10.0,
            450.0,
            4500.0,
            "high",
            "test",
            1,
            SAMPLE_CREATED,
        )
        assert h1 != h2


class TestVerifyChain:
    """verify_chain validates the full chain."""

    def test_empty_table_is_valid(self, tmp_db):
        ok, last_id, _msg = verify_chain(tmp_db)
        assert ok is True
        assert last_id is None

    def test_single_trade_valid(self, tmp_db):
        _insert_chained(tmp_db, 1, GENESIS_HASH)
        ok, last_id, _msg = verify_chain(tmp_db)
        assert ok is True
        assert last_id == 1

    def test_three_trade_chain_valid(self, tmp_db):
        h1 = _insert_chained(tmp_db, 1, GENESIS_HASH, symbol="SPY")
        h2 = _insert_chained(
            tmp_db,
            2,
            h1,
            symbol="QQQ",
            price=380.0,
            notional=3800.0,
        )
        _insert_chained(
            tmp_db,
            3,
            h2,
            symbol="TLT",
            action="sell",
            price=95.0,
            notional=950.0,
        )

        ok, last_id, msg = verify_chain(tmp_db)
        assert ok is True
        assert last_id == 3
        assert "3 trade(s) verified" in msg

    def test_modified_field_breaks_chain(self, tmp_db):
        h1 = _insert_chained(tmp_db, 1, GENESIS_HASH)
        _insert_chained(tmp_db, 2, h1)

        # Tamper with price on trade 1
        tmp_db.execute("UPDATE trades SET price = 999.0 WHERE trade_id = 1")
        tmp_db.commit()

        ok, last_id, msg = verify_chain(tmp_db)
        assert ok is False
        assert last_id == 1
        assert "row_hash mismatch" in msg

    def test_deleted_row_breaks_chain(self, tmp_db):
        h1 = _insert_chained(tmp_db, 1, GENESIS_HASH)
        h2 = _insert_chained(tmp_db, 2, h1)
        _insert_chained(tmp_db, 3, h2)

        # Delete the middle trade
        tmp_db.execute("DELETE FROM trades WHERE trade_id = 2")
        tmp_db.commit()

        ok, last_id, msg = verify_chain(tmp_db)
        assert ok is False
        assert last_id == 3
        assert "prev_hash mismatch" in msg


class TestBackfill:
    """backfill_hashes fills in missing hashes."""

    def test_backfill_existing_rows(self, tmp_db):
        # Insert rows without hashes (simulating pre-v2 data)
        _insert_trade(tmp_db, 1, prev_hash="", row_hash="")
        _insert_trade(tmp_db, 2, prev_hash="", row_hash="", symbol="QQQ")

        count = backfill_hashes(tmp_db)
        assert count == 2

        # Chain should now be valid
        ok, _, _ = verify_chain(tmp_db)
        assert ok is True


class TestGetLatestHash:
    """get_latest_hash returns GENESIS or last row_hash."""

    def test_empty_table(self, tmp_db):
        assert get_latest_hash(tmp_db) == GENESIS_HASH

    def test_after_insert(self, tmp_db):
        h = _insert_chained(tmp_db, 1, GENESIS_HASH)
        assert get_latest_hash(tmp_db) == h

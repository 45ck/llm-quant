"""SHA-256 hash chain for tamper-evident trade ledger.

Each trade row includes a ``row_hash`` computed as::

    SHA256(prev_hash | trade_id | date | symbol | action | shares | price
           | notional | conviction | reasoning | decision_id | created_at)

Walking the chain from the genesis hash and recomputing every link
detects any modification, deletion, or reordering of rows.
"""

from __future__ import annotations

import hashlib
import logging

import duckdb

logger = logging.getLogger(__name__)

GENESIS_HASH = hashlib.sha256(b"GENESIS").hexdigest()


def compute_trade_hash(
    prev_hash: str,
    trade_id: int,
    date: object,
    symbol: str,
    action: str,
    shares: float,
    price: float,
    notional: float,
    conviction: str | None,
    reasoning: str | None,
    decision_id: int | None,
    created_at: object,
) -> str:
    """Compute the SHA-256 hash for a single trade row.

    Fields are pipe-delimited; floats use ``:.8f`` for deterministic
    representation.  ``None`` values are serialized as the literal
    string ``"None"``.
    """
    parts = "|".join([
        str(prev_hash),
        str(trade_id),
        str(date),
        str(symbol),
        str(action),
        f"{shares:.8f}",
        f"{price:.8f}",
        f"{notional:.8f}",
        str(conviction),
        str(reasoning),
        str(decision_id),
        str(created_at),
    ])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def get_latest_hash(conn: duckdb.DuckDBPyConnection) -> str:
    """Return the ``row_hash`` of the last trade, or :data:`GENESIS_HASH`."""
    row = conn.execute(
        "SELECT row_hash FROM trades ORDER BY trade_id DESC LIMIT 1"
    ).fetchone()
    if row is None or not row[0]:
        return GENESIS_HASH
    return row[0]


def verify_chain(conn: duckdb.DuckDBPyConnection) -> tuple[bool, int | None, str]:
    """Walk every trade and verify the hash chain.

    Returns
    -------
    tuple[bool, int | None, str]
        ``(ok, last_trade_id, message)``
    """
    rows = conn.execute(
        """
        SELECT trade_id, date, symbol, action, shares, price,
               notional, conviction, reasoning, llm_decision_id,
               created_at, prev_hash, row_hash
        FROM trades
        ORDER BY trade_id ASC
        """
    ).fetchall()

    if not rows:
        return True, None, "No trades in ledger – chain is trivially valid."

    prev = GENESIS_HASH
    for row in rows:
        (trade_id, dt, symbol, action, shares, price,
         notional, conviction, reasoning, decision_id,
         created_at, stored_prev, stored_hash) = row

        if stored_prev != prev:
            return (
                False,
                trade_id,
                f"prev_hash mismatch at trade_id={trade_id}: "
                f"expected {prev!r}, stored {stored_prev!r}",
            )

        expected = compute_trade_hash(
            prev, trade_id, dt, symbol, action,
            float(shares), float(price), float(notional),
            conviction, reasoning, decision_id, created_at,
        )

        if stored_hash != expected:
            return (
                False,
                trade_id,
                f"row_hash mismatch at trade_id={trade_id}: "
                f"expected {expected!r}, stored {stored_hash!r}",
            )

        prev = stored_hash

    return True, rows[-1][0], f"Chain intact – {len(rows)} trade(s) verified."


def backfill_hashes(conn: duckdb.DuckDBPyConnection) -> int:
    """Backfill ``prev_hash`` / ``row_hash`` for existing rows that lack them.

    Returns the number of rows updated.
    """
    rows = conn.execute(
        """
        SELECT trade_id, date, symbol, action, shares, price,
               notional, conviction, reasoning, llm_decision_id,
               created_at
        FROM trades
        WHERE row_hash = '' OR row_hash IS NULL
        ORDER BY trade_id ASC
        """
    ).fetchall()

    if not rows:
        return 0

    # Find the last valid hash before the gap
    first_id = rows[0][0]
    prev_row = conn.execute(
        "SELECT row_hash FROM trades WHERE trade_id < ? AND row_hash != '' "
        "ORDER BY trade_id DESC LIMIT 1",
        [first_id],
    ).fetchone()
    prev = prev_row[0] if prev_row else GENESIS_HASH

    count = 0
    for row in rows:
        (trade_id, dt, symbol, action, shares, price,
         notional, conviction, reasoning, decision_id, created_at) = row

        row_hash = compute_trade_hash(
            prev, trade_id, dt, symbol, action,
            float(shares), float(price), float(notional),
            conviction, reasoning, decision_id, created_at,
        )

        conn.execute(
            "UPDATE trades SET prev_hash = ?, row_hash = ? WHERE trade_id = ?",
            [prev, row_hash, trade_id],
        )
        prev = row_hash
        count += 1

    conn.commit()
    logger.info("Backfilled hashes for %d trade(s).", count)
    return count

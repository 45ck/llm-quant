"""DuckDB schema creation and management."""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS universe (
        symbol VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        sector VARCHAR NOT NULL,
        tradeable BOOLEAN DEFAULT TRUE,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_data_daily (
        symbol VARCHAR NOT NULL,
        date DATE NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        adj_close DOUBLE,
        sma_20 DOUBLE,
        sma_50 DOUBLE,
        rsi_14 DOUBLE,
        macd DOUBLE,
        macd_signal DOUBLE,
        macd_hist DOUBLE,
        atr_14 DOUBLE,
        PRIMARY KEY (symbol, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        snapshot_id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        nav DOUBLE NOT NULL,
        cash DOUBLE NOT NULL,
        gross_exposure DOUBLE NOT NULL,
        net_exposure DOUBLE NOT NULL,
        total_pnl DOUBLE NOT NULL,
        daily_pnl DOUBLE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE SEQUENCE IF NOT EXISTS seq_snapshot_id START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        snapshot_id INTEGER NOT NULL,
        symbol VARCHAR NOT NULL,
        shares DOUBLE NOT NULL,
        avg_cost DOUBLE NOT NULL,
        current_price DOUBLE NOT NULL,
        market_value DOUBLE NOT NULL,
        unrealized_pnl DOUBLE NOT NULL,
        weight DOUBLE NOT NULL,
        stop_loss DOUBLE,
        PRIMARY KEY (snapshot_id, symbol),
        FOREIGN KEY (snapshot_id) REFERENCES portfolio_snapshots(snapshot_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        symbol VARCHAR NOT NULL,
        action VARCHAR NOT NULL,
        shares DOUBLE NOT NULL,
        price DOUBLE NOT NULL,
        notional DOUBLE NOT NULL,
        conviction VARCHAR,
        reasoning TEXT,
        llm_decision_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        prev_hash VARCHAR NOT NULL DEFAULT '',
        row_hash VARCHAR NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE SEQUENCE IF NOT EXISTS seq_trade_id START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_decisions (
        decision_id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        model VARCHAR NOT NULL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        cost_usd DOUBLE,
        market_regime VARCHAR,
        regime_confidence DOUBLE,
        num_signals INTEGER,
        raw_response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE SEQUENCE IF NOT EXISTS seq_decision_id START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
]


def _migrate_v1_to_v2(conn: duckdb.DuckDBPyConnection) -> None:
    """Add hash-chain columns to trades and backfill existing rows."""
    cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'trades'"
        ).fetchall()
    }
    if "prev_hash" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN prev_hash VARCHAR DEFAULT ''")
    if "row_hash" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN row_hash VARCHAR DEFAULT ''")

    from llm_quant.db.integrity import backfill_hashes

    backfill_hashes(conn)
    logger.info("Migrated schema to v2: hash-chain columns added.")


def init_schema(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Create all tables in DuckDB. Returns the connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)

    # Run migrations for existing databases
    old_ver = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    old_version = int(old_ver[0]) if old_ver else 0
    if old_version < 2:
        _migrate_v1_to_v2(conn)

    conn.execute(
        "INSERT OR REPLACE INTO schema_meta VALUES ('version', ?)",
        [str(SCHEMA_VERSION)],
    )
    conn.commit()
    logger.info("DuckDB schema initialized at %s (v%d)", db_path, SCHEMA_VERSION)
    return conn


def get_connection(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open an existing DuckDB database."""
    return duckdb.connect(str(db_path))

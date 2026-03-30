"""DuckDB schema creation and management."""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5

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
    CREATE TABLE IF NOT EXISTS pods (
        pod_id VARCHAR PRIMARY KEY,
        display_name VARCHAR NOT NULL,
        strategy_type VARCHAR NOT NULL,
        initial_capital DOUBLE NOT NULL DEFAULT 100000.0,
        status VARCHAR NOT NULL DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        retired_at TIMESTAMP,
        config_path VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        snapshot_id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        pod_id VARCHAR NOT NULL DEFAULT 'default',
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
        pod_id VARCHAR NOT NULL DEFAULT 'default',
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
        pod_id VARCHAR NOT NULL DEFAULT 'default',
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
    # --- Governance / Surveillance tables (v3) ---
    """
    CREATE SEQUENCE IF NOT EXISTS seq_scan_id START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS surveillance_scans (
        scan_id INTEGER PRIMARY KEY DEFAULT nextval('seq_scan_id'),
        scan_timestamp TIMESTAMP NOT NULL,
        overall_severity VARCHAR NOT NULL,
        total_checks INTEGER NOT NULL,
        halt_count INTEGER NOT NULL,
        warning_count INTEGER NOT NULL,
        checks_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS config_hashes (
        file_path VARCHAR NOT NULL,
        hash_sha256 VARCHAR NOT NULL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (file_path, hash_sha256)
    )
    """,
    """
    CREATE SEQUENCE IF NOT EXISTS seq_change_id START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_changelog (
        change_id INTEGER PRIMARY KEY DEFAULT nextval('seq_change_id'),
        change_date DATE NOT NULL,
        change_type VARCHAR NOT NULL,
        description TEXT NOT NULL,
        config_diff TEXT,
        author VARCHAR DEFAULT 'system',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # --- COT (Commitments of Traders) weekly data (v5) ---
    """
    CREATE TABLE IF NOT EXISTS cot_weekly (
        symbol VARCHAR NOT NULL,
        report_date DATE NOT NULL,
        commercial_net DOUBLE,
        noncommercial_net DOUBLE,
        open_interest DOUBLE,
        cot_index DOUBLE,
        PRIMARY KEY (symbol, report_date)
    )
    """,
    # --- Pod indexes (v4) ---
    """
    CREATE INDEX IF NOT EXISTS idx_trades_pod_id
        ON trades (pod_id, trade_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_snapshots_pod_date
        ON portfolio_snapshots (pod_id, date DESC, snapshot_id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_decisions_pod_date
        ON llm_decisions (pod_id, date DESC)
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


def _migrate_v2_to_v3(conn: duckdb.DuckDBPyConnection) -> None:
    """Add governance/surveillance tables for production monitoring."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    if "surveillance_scans" not in tables:
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_scan_id START 1")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS surveillance_scans (
                scan_id INTEGER PRIMARY KEY DEFAULT nextval('seq_scan_id'),
                scan_timestamp TIMESTAMP NOT NULL,
                overall_severity VARCHAR NOT NULL,
                total_checks INTEGER NOT NULL,
                halt_count INTEGER NOT NULL,
                warning_count INTEGER NOT NULL,
                checks_json TEXT
            )
            """)
    if "config_hashes" not in tables:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_hashes (
                file_path VARCHAR NOT NULL,
                hash_sha256 VARCHAR NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (file_path, hash_sha256)
            )
            """)
    if "strategy_changelog" not in tables:
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_change_id START 1")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_changelog (
                change_id INTEGER PRIMARY KEY DEFAULT nextval('seq_change_id'),
                change_date DATE NOT NULL,
                change_type VARCHAR NOT NULL,
                description TEXT NOT NULL,
                config_diff TEXT,
                author VARCHAR DEFAULT 'system',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
    logger.info("Migrated schema to v3: governance/surveillance tables added.")


def _migrate_v3_to_v4(conn: duckdb.DuckDBPyConnection) -> None:
    """Add pods table and pod_id columns for multi-pod support."""
    # Create pods table
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    if "pods" not in tables:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pods (
                pod_id VARCHAR PRIMARY KEY,
                display_name VARCHAR NOT NULL,
                strategy_type VARCHAR NOT NULL,
                initial_capital DOUBLE NOT NULL DEFAULT 100000.0,
                status VARCHAR NOT NULL DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retired_at TIMESTAMP,
                config_path VARCHAR
            )
            """)

    # Insert default pod if not present
    existing = conn.execute(
        "SELECT pod_id FROM pods WHERE pod_id = 'default'"
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO pods "
            "(pod_id, display_name, strategy_type, "
            "initial_capital, status) "
            "VALUES ('default', 'Default Pod', "
            "'regime_momentum', 100000.0, 'active')"
        )

    # Add pod_id column to trades, portfolio_snapshots, llm_decisions
    for table in ("trades", "portfolio_snapshots", "llm_decisions"):
        cols = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?",
                [table],
            ).fetchall()
        }
        if "pod_id" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN pod_id VARCHAR DEFAULT 'default'"
            )
            conn.execute(f"UPDATE {table} SET pod_id = 'default' WHERE pod_id IS NULL")

    # Create indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_pod_id ON trades (pod_id, trade_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_snapshots_pod_date "
        "ON portfolio_snapshots (pod_id, date DESC, snapshot_id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_decisions_pod_date "
        "ON llm_decisions (pod_id, date DESC)"
    )

    logger.info("Migrated schema to v4: multi-pod support added.")


def _migrate_v4_to_v5(conn: duckdb.DuckDBPyConnection) -> None:
    """Add cot_weekly table for CFTC COT regime overlay."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    if "cot_weekly" not in tables:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cot_weekly (
                symbol VARCHAR NOT NULL,
                report_date DATE NOT NULL,
                commercial_net DOUBLE,
                noncommercial_net DOUBLE,
                open_interest DOUBLE,
                cot_index DOUBLE,
                PRIMARY KEY (symbol, report_date)
            )
            """)
    logger.info("Migrated schema to v5: cot_weekly table added.")


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
    if old_version < 3:
        _migrate_v2_to_v3(conn)
    if old_version < 4:
        _migrate_v3_to_v4(conn)
    if old_version < 5:
        _migrate_v4_to_v5(conn)

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

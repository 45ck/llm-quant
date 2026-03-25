"""Migration: add pod_id columns to trades, portfolio_snapshots, llm_decisions.

Idempotent — safe to run multiple times. Checks column existence before ALTER.
"""

import sys
from pathlib import Path

import duckdb

DB_PATH = Path("data/llm_quant.duckdb")


def get_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return set of column names for a table."""
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
        [table],
    ).fetchall()
    return {row[0] for row in rows}


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    actions: list[str] = []

    for table in ("trades", "portfolio_snapshots", "llm_decisions"):
        cols = get_columns(conn, table)
        if "pod_id" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN pod_id VARCHAR DEFAULT 'default'"
            )
            actions.append(f"  Added pod_id column to {table}")
        else:
            actions.append(f"  {table}.pod_id already exists — skipped")

        # Backfill NULLs regardless (cheap no-op if none exist)
        conn.execute(f"UPDATE {table} SET pod_id = 'default' WHERE pod_id IS NULL")
        actions.append(f"  Backfilled NULLs in {table}.pod_id")

    # Ensure pods table and default pod exist
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    if "pods" not in tables:
        conn.execute("""
            CREATE TABLE pods (
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
        actions.append("  Created pods table")

    existing = conn.execute(
        "SELECT pod_id FROM pods WHERE pod_id = 'default'"
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO pods (pod_id, display_name, strategy_type,"
            " initial_capital, status) "
            "VALUES ('default', 'Default Pod',"
            " 'regime_momentum', 100000.0, 'active')"
        )
        actions.append("  Inserted default pod row")

    conn.commit()

    print("Migration complete:")
    for a in actions:
        print(a)


def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        conn = duckdb.connect(str(DB_PATH))
        migrate(conn)
        conn.close()
    except (OSError, duckdb.Error) as exc:
        print(f"ERROR: Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

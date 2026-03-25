"""Tests for v4 schema migration (multi-pod support)."""


def test_init_schema_creates_pods_table(pod_db):
    """Verify the pods table exists after init_schema."""
    tables = {
        row[0]
        for row in pod_db.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert "pods" in tables


def test_default_pod_seeded(pod_db):
    """Verify 'default' pod row is inserted during migration."""
    row = pod_db.execute(
        "SELECT pod_id, display_name, strategy_type, initial_capital, status "
        "FROM pods WHERE pod_id = 'default'"
    ).fetchone()
    assert row is not None
    pod_id, display_name, strategy_type, initial_capital, status = row
    assert pod_id == "default"
    assert display_name == "Default Pod"
    assert strategy_type == "regime_momentum"
    assert initial_capital == 100_000.0
    assert status == "active"


def test_pod_id_columns_exist(pod_db):
    """Verify pod_id column exists on trades, portfolio_snapshots, llm_decisions."""
    for table in ("trades", "portfolio_snapshots", "llm_decisions"):
        cols = {
            row[0]
            for row in pod_db.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?",
                [table],
            ).fetchall()
        }
        assert "pod_id" in cols, f"pod_id column missing from {table}"


def test_indexes_created(pod_db):
    """Verify the 3 pod-related indexes exist."""
    indexes = {
        row[0]
        for row in pod_db.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    }
    assert "idx_trades_pod_id" in indexes
    assert "idx_snapshots_pod_date" in indexes
    assert "idx_decisions_pod_date" in indexes


def test_schema_version_is_4(pod_db):
    """Verify schema_meta shows version 4."""
    row = pod_db.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    assert row is not None
    assert row[0] == "4"

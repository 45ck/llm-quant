"""Fixtures for multi-pod tests."""

import pytest

from llm_quant.db.schema import init_schema


@pytest.fixture
def pod_db(tmp_path):
    """Fresh DuckDB with v4 schema (pods table + pod_id columns)."""
    db_path = str(tmp_path / "test_pods.duckdb")
    conn = init_schema(db_path)
    yield conn
    conn.close()


@pytest.fixture
def two_pod_db(pod_db):
    """DB with 'default' and 'benchmark' pods registered."""
    pod_db.execute(
        "INSERT INTO pods "
        "(pod_id, display_name, strategy_type, "
        "initial_capital, status) "
        "VALUES ('benchmark', 'Passive Benchmark', "
        "'passive_benchmark', 100000.0, 'active')"
    )
    return pod_db

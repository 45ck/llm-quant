"""Shared test fixtures."""

import os
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import pytest

from llm_quant.config import AppConfig, GeneralConfig, LLMConfig, DataConfig, RiskLimits, UniverseConfig, ETFEntry
from llm_quant.db.schema import init_schema
from llm_quant.trading.portfolio import Portfolio, Position


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DuckDB with schema initialized."""
    db_path = tmp_path / "test.duckdb"
    conn = init_schema(db_path)
    yield conn
    conn.close()


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal AppConfig for testing."""
    return AppConfig(
        general=GeneralConfig(db_path=str(tmp_path / "test.duckdb")),
        llm=LLMConfig(),
        data=DataConfig(lookback_days=60),
        risk=RiskLimits(),
        universe=UniverseConfig(
            name="Test Universe",
            etfs=[
                ETFEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad_market"),
                ETFEntry(symbol="QQQ", name="Nasdaq 100", category="equity", sector="tech"),
                ETFEntry(symbol="TLT", name="20Y Treasury", category="fixed_income", sector="bonds"),
                ETFEntry(symbol="GLD", name="Gold", category="commodity", sector="precious_metals"),
            ],
        ),
    )


@pytest.fixture
def sample_portfolio():
    """Create a Portfolio with some positions."""
    p = Portfolio(initial_capital=100_000.0)
    p.cash = 80_000.0
    p.positions = {
        "SPY": Position(symbol="SPY", shares=20, avg_cost=450.0, current_price=460.0, stop_loss=427.5),
        "QQQ": Position(symbol="QQQ", shares=15, avg_cost=380.0, current_price=390.0, stop_loss=361.0),
    }
    return p


@pytest.fixture
def sample_prices():
    """Sample price dict."""
    return {
        "SPY": 460.0,
        "QQQ": 390.0,
        "TLT": 95.0,
        "GLD": 185.0,
        "IWM": 200.0,
    }

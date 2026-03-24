"""Shared test fixtures."""


import pytest

from llm_quant.config import (
    AppConfig,
    AssetEntry,
    DataConfig,
    GeneralConfig,
    LLMConfig,
    RiskLimits,
    UniverseConfig,
)
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
            assets=[
                AssetEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad_market"),
                AssetEntry(symbol="QQQ", name="Nasdaq 100", category="equity", sector="tech"),
                AssetEntry(symbol="TLT", name="20Y Treasury", category="fixed_income", sector="bonds"),
                AssetEntry(symbol="GLD", name="Gold", category="commodity", sector="precious_metals"),
                AssetEntry(symbol="BTC-USD", name="Bitcoin", category="crypto", sector="layer1", asset_class="crypto"),
                AssetEntry(symbol="EURUSD=X", name="EUR/USD", category="forex", sector="major_pairs", asset_class="forex"),
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

"""Configuration loading and validation via Pydantic."""

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field


def _find_config_dir() -> Path:
    """Find the config directory relative to the project root."""
    # Check env var first
    env_path = os.environ.get("LLM_QUANT_CONFIG_DIR")
    if env_path:
        return Path(env_path)
    # Walk up from this file to find config/
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "config"
        if candidate.is_dir():
            return candidate
        current = current.parent
    # Fallback to CWD
    return Path.cwd() / "config"


CONFIG_DIR = _find_config_dir()


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
    max_trades_per_session: int = 5


class GeneralConfig(BaseModel):
    db_path: str = "data/llm_quant.duckdb"
    initial_capital: float = 100_000.0
    base_currency: str = "USD"


class DataConfig(BaseModel):
    lookback_days: int = 252
    fetch_timeout: int = 30


class RiskLimits(BaseModel):
    max_position_weight: float = 0.10
    max_trade_size: float = 0.02
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0
    max_sector_concentration: float = 0.30
    max_trades_per_session: int = 5
    min_cash_reserve: float = 0.05
    require_stop_loss: bool = True
    default_stop_loss_pct: float = 0.05


class ETFEntry(BaseModel):
    symbol: str
    name: str
    category: str
    sector: str
    tradeable: bool = True


class UniverseConfig(BaseModel):
    name: str = "US ETF Core 30"
    description: str = ""
    etfs: list[ETFEntry] = Field(default_factory=list)


class AppConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)


def _load_toml(path: Path) -> dict:
    """Load a TOML file and return as dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(config_dir: Path | None = None) -> AppConfig:
    """Load all TOML configs and merge into AppConfig."""
    config_dir = config_dir or CONFIG_DIR

    general_data: dict = {}
    llm_data: dict = {}
    data_data: dict = {}
    risk_data: dict = {}
    universe_data: dict = {}

    # Load default.toml
    default_path = config_dir / "default.toml"
    if default_path.exists():
        raw = _load_toml(default_path)
        general_data = raw.get("general", {})
        llm_data = raw.get("llm", {})
        data_data = raw.get("data", {})

    # Load risk.toml
    risk_path = config_dir / "risk.toml"
    if risk_path.exists():
        raw = _load_toml(risk_path)
        risk_data = raw.get("limits", {})

    # Load universe.toml
    universe_path = config_dir / "universe.toml"
    if universe_path.exists():
        raw = _load_toml(universe_path)
        universe_data = raw.get("universe", {})
        universe_data["etfs"] = raw.get("etfs", [])

    # Override db_path from env
    env_db = os.environ.get("LLM_QUANT_DB_PATH")
    if env_db:
        general_data["db_path"] = env_db

    # Override model from env
    env_model = os.environ.get("LLM_QUANT_MODEL")
    if env_model:
        llm_data["model"] = env_model

    return AppConfig(
        general=GeneralConfig(**general_data),
        llm=LLMConfig(**llm_data),
        data=DataConfig(**data_data),
        risk=RiskLimits(**risk_data),
        universe=UniverseConfig(**universe_data),
    )

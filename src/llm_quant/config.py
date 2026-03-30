"""Configuration loading and validation via Pydantic."""

import logging
import os
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

POD_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}[a-z0-9]$")


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
    max_drawdown_pct: float = 0.15  # Portfolio drawdown circuit breaker
    # ATR-calibrated stop-loss multipliers
    atr_stop_multiplier: float = 2.0          # 2x ATR for equities (Turtle Traders)
    atr_stop_multiplier_crypto: float = 2.5   # wider for crypto overnight gaps
    atr_stop_multiplier_commodity: float = 2.5  # wider for volatile commodities
    # ATR-based position sizing
    target_risk_pct: float = 0.01    # fraction of NAV to risk per trade
    deviation_buffer: float = 0.20   # buffer before triggering rebalance alert
    atr_period: int = 14             # ATR lookback for equities / fixed income
    atr_period_crypto: int = 7       # shorter ATR lookback for crypto
    # Per-asset-class overrides (crypto is more volatile, forex less so)
    crypto_max_position_weight: float = 0.05
    crypto_default_stop_loss_pct: float = 0.15
    forex_max_position_weight: float = 0.08
    forex_default_stop_loss_pct: float = 0.03


class TrackBLimits(BaseModel):
    """Risk limits for Track B — Aggressive Alpha strategies."""

    max_position_weight: float = 0.15
    max_trade_size: float = 0.03
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0
    max_sector_concentration: float = 0.30
    max_trades_per_session: int = 5
    min_cash_reserve: float = 0.03
    require_stop_loss: bool = True
    default_stop_loss_pct: float = 0.08
    atr_stop_multiplier: float = 2.0
    atr_stop_multiplier_crypto: float = 3.0
    atr_stop_multiplier_commodity: float = 2.5
    target_risk_pct: float = 0.015
    deviation_buffer: float = 0.25
    max_drawdown_pct: float = 0.30
    crypto_max_position_weight: float = 0.08
    crypto_default_stop_loss_pct: float = 0.20
    leveraged_etf_max_position_weight: float = 0.10


class TrackCLimits(BaseModel):
    """Risk limits for Track C — Structural Arb / Event-Driven strategies.

    Near-zero-beta arb strategies sit between Track A (defensive) and Track B
    (aggressive).  They allow wider single-leg sizing because legs partially
    offset each other, but enforce a tight *net* exposure cap to preserve
    market-neutrality.  A higher cash reserve supports event-driven staging.
    """

    max_position_weight: float = 0.20           # 20% per strategy
    max_trade_size: float = 0.05                # 5% per trade
    max_gross_exposure: float = 2.0             # 200% (both legs summed)
    max_net_exposure: float = 0.30              # 30% — enforces near-zero beta
    max_sector_concentration: float = 0.30
    max_exchange_concentration: float = 0.25    # 25% on any single exchange
    max_trades_per_session: int = 5
    min_cash_reserve: float = 0.10              # 10% — hold dry powder for events
    require_stop_loss: bool = True
    default_stop_loss_pct: float = 0.05
    atr_stop_multiplier: float = 2.0
    atr_stop_multiplier_crypto: float = 2.5
    atr_stop_multiplier_commodity: float = 2.5
    target_risk_pct: float = 0.01
    deviation_buffer: float = 0.20
    max_drawdown_pct: float = 0.10              # 10% — tighter; drawdown signals leg break

    # Kill-switch thresholds (Track C-specific)
    max_beta_to_spy: float = 0.25               # rolling-30d SPY beta limit
    min_spread_bps: float = 5.0                 # spread collapse threshold (bps)
    max_funding_rate_pct: float = 0.50          # funding reversal threshold (bps/day)


class RegimeDriftConfig(BaseModel):
    rolling_window_days: int = 21
    sharpe_decay_warn: float = 0.30
    sharpe_decay_halt: float = 0.50
    win_rate_decay_warn: float = 0.15
    win_rate_decay_halt: float = 0.25
    vol_spike_warn: float = 1.5
    vol_spike_halt: float = 2.0


class AlphaDecayConfig(BaseModel):
    rolling_window_days: int = 63
    decay_warn: float = 0.40
    decay_halt: float = 0.60


class RiskDriftConfig(BaseModel):
    exposure_warn_buffer: float = 0.10
    concentration_warn_buffer: float = 0.10


class DataQualityConfig(BaseModel):
    max_stale_days: int = 3
    gap_threshold_pct: float = 0.20
    plausibility_min_price: float = 0.01


class ProcessDriftConfig(BaseModel):
    tracked_files: list[str] = Field(
        default_factory=lambda: [
            "config/default.toml",
            "config/risk.toml",
            "config/universe.toml",
            "config/governance.toml",
            "config/prompts/trader_decision.md",
        ]
    )


class OperationalHealthConfig(BaseModel):
    max_snapshot_gap_days: int = 3
    max_price_staleness_hours: int = 48
    hash_chain_required: bool = True


class KillSwitchConfig(BaseModel):
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    max_consecutive_losses: int = 5
    max_correlation_breach: float = 0.85
    data_blackout_hours: int = 72
    risk_check_failure_streak: int = 3


class PromotionConfig(BaseModel):
    min_deflated_sharpe: float = 0.95
    max_pbo: float = 0.10
    max_spa_p_value: float = 0.05
    min_trl: int = 1
    scorecard_pass_threshold: int = 85
    min_paper_trades: int = 50
    min_paper_days: int = 30
    min_paper_sharpe: float = 0.60
    canary_allocation_pct: float = 0.10
    canary_min_days: int = 14
    canary_max_drawdown_pct: float = 0.10


class GovernanceConfig(BaseModel):
    regime_drift: RegimeDriftConfig = Field(default_factory=RegimeDriftConfig)
    alpha_decay: AlphaDecayConfig = Field(default_factory=AlphaDecayConfig)
    risk_drift: RiskDriftConfig = Field(default_factory=RiskDriftConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    process_drift: ProcessDriftConfig = Field(default_factory=ProcessDriftConfig)
    operational_health: OperationalHealthConfig = Field(
        default_factory=OperationalHealthConfig
    )
    kill_switches: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)


class AssetEntry(BaseModel):
    symbol: str
    name: str
    category: str
    sector: str
    asset_class: str = "equity"  # equity, crypto, forex
    tradeable: bool = True


# Backward-compatible alias
ETFEntry = AssetEntry


class UniverseConfig(BaseModel):
    name: str = "Multi-Asset Universe"
    description: str = ""
    assets: list[AssetEntry] = Field(default_factory=list)

    @property
    def etfs(self) -> list[AssetEntry]:
        """Backward-compatible alias for assets."""
        return self.assets


class AppConfig(BaseModel):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    track_b: TrackBLimits = Field(default_factory=TrackBLimits)
    track_c: TrackCLimits = Field(default_factory=TrackCLimits)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)


def _load_toml(path: Path) -> dict:
    """Load a TOML file and return as dict."""
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(config_dir: Path | None = None) -> AppConfig:
    """Load all TOML configs and merge into AppConfig."""
    config_dir = config_dir or CONFIG_DIR

    general_data: dict = {}
    llm_data: dict = {}
    data_data: dict = {}
    risk_data: dict = {}
    track_b_data: dict = {}
    track_c_data: dict = {}
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
        track_b_data = raw.get("track_b", {})
        track_c_data = raw.get("track_c", {})

    # Load universe.toml
    universe_path = config_dir / "universe.toml"
    if universe_path.exists():
        raw = _load_toml(universe_path)
        universe_data = raw.get("universe", {})
        # Accept both "assets" (new) and "etfs" (legacy) keys
        universe_data["assets"] = raw.get("assets", []) or raw.get("etfs", [])

    # Override db_path from env
    env_db = os.environ.get("LLM_QUANT_DB_PATH")
    if env_db:
        general_data["db_path"] = env_db

    # Override model from env
    env_model = os.environ.get("LLM_QUANT_MODEL")
    if env_model:
        llm_data["model"] = env_model

    # Load governance.toml
    governance_data: dict = {}
    governance_path = config_dir / "governance.toml"
    if governance_path.exists():
        governance_data = _load_toml(governance_path)

    return AppConfig(
        general=GeneralConfig(**general_data),
        llm=LLMConfig(**llm_data),
        data=DataConfig(**data_data),
        risk=RiskLimits(**risk_data),
        track_b=TrackBLimits(**track_b_data),
        track_c=TrackCLimits(**track_c_data),
        universe=UniverseConfig(**universe_data),
        governance=GovernanceConfig(**governance_data),
    )


def validate_pod_id(pod_id: str) -> bool:
    """Check whether *pod_id* is syntactically valid."""
    return pod_id == "default" or bool(POD_ID_RE.match(pod_id))


def load_config_for_pod(
    pod_id: str = "default",
    config_dir: Path | None = None,
) -> AppConfig:
    """Load base config, then overlay strategy-specific TOML if it exists."""
    base = load_config(config_dir)
    if pod_id == "default":
        return base

    cfg_dir = config_dir or _find_config_dir()
    strategy_path = cfg_dir / "strategies" / f"{pod_id}.toml"
    if not strategy_path.exists():
        logger.warning("No strategy overlay for pod '%s' at %s", pod_id, strategy_path)
        return base

    with strategy_path.open("rb") as f:
        overlay = tomllib.load(f)

    # Merge overlay into base config dict
    base_dict = base.model_dump()
    for section, values in overlay.items():
        if (
            section in base_dict
            and isinstance(values, dict)
            and isinstance(base_dict[section], dict)
        ):
            base_dict[section].update(values)

    return AppConfig(**base_dict)

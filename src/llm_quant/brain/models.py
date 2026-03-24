"""Domain models for the LLM brain module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class Conviction(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarketRegime(StrEnum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITION = "transition"


@dataclass
class TradeSignal:
    symbol: str
    action: Action
    conviction: Conviction
    target_weight: float
    stop_loss: float
    reasoning: str


@dataclass
class TradingDecision:
    date: date
    market_regime: MarketRegime
    regime_confidence: float
    regime_reasoning: str
    signals: list[TradeSignal]
    portfolio_commentary: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: str = ""


@dataclass
class MarketRow:
    """Single ETF's market data for the decision prompt."""

    symbol: str
    close: float
    change_pct: float
    sma_20: float
    sma_50: float
    rsi_14: float
    macd: float
    atr_14: float
    volume: int


@dataclass
class PositionRow:
    """Current position for the decision prompt."""

    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    pnl_pct: float
    weight_pct: float
    stop_loss: float


@dataclass
class MarketContext:
    """Full context assembled for the LLM decision prompt."""

    date: date
    nav: float
    cash: float
    cash_pct: float
    gross_exposure_pct: float
    net_exposure_pct: float
    positions: list[PositionRow] = field(default_factory=list)
    market_data: list[MarketRow] = field(default_factory=list)
    vix: float = 0.0
    yield_spread: float = 0.0
    spy_trend: str = "neutral"

"""Domain models for the LLM brain module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_quant.signals.tsmom import TsmomSignal


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
    # Task ev8: credit spread stress indicator
    credit_spread_oas: float | None = None
    credit_spread_zscore: float | None = None
    silent_stress: bool = False
    # Task 4a1: adaptive VIX regime classification
    vix_regime_thresholds: tuple[float, float] = (20.0, 25.0)
    market_regime: MarketRegime = MarketRegime.TRANSITION
    # Task vts: 126-day VIX percentile rank
    vix_percentile_126d: float = 50.0
    # Task llm-quant-56k: COT crowding overlay (symbol → signal string)
    # Signals: "crowded_long" | "crowded_short" | "neutral"
    # Applied Friday-published data on Monday open only; confirmation/warning,
    # never an independent trade signal.
    cot_crowding: dict[str, str] | None = None
    # Task llm-quant-bbt: execution cost lookup (symbol → round-trip bps)
    execution_costs: dict[str, float] | None = None
    # Task llm-quant-773: HMM regime detection confidence
    regime_confidence: float = 0.0  # HMM state probability (0–1)
    hmm_regime_fallback: bool = True  # True if HMM fell back to heuristic
    # Task llm-quant-rbu: multi-lookback TSMOM signals (symbol → TsmomSignal)
    tsmom_signals: dict[str, TsmomSignal] | None = None
    # Task llm-quant-t2s: 2x2 inflation regime overlay (Bridgewater / Dalio framework)
    macro_quadrant: str | None = None  # e.g. "disinflationary_boom"
    quadrant_tilts: dict | None = None  # keys: overweight, underweight, confidence
    # Whipsaw R1 — direction-of-travel fields (2026-04-01 whipsaw fix).
    # Added so the LLM can distinguish "VIX=25 rising INTO spike" from
    # "VIX=25 receding FROM spike". See docs/investigations/apr01-whipsaw.md.
    vix_change_5d: float | None = None  # current VIX minus VIX 5 sessions ago
    vix_pct_from_20d_high: float | None = None  # current / 20d max * 100
    spy_drawdown_from_20d_high: float | None = None  # SPY/20d_max - 1
    days_since_regime_flip: int | None = None  # sessions since current regime began

"""Tests for the LLM response parser."""

from datetime import date

import pytest

from llm_quant.brain.models import Action, Conviction, MarketRegime
from llm_quant.brain.parser import parse_trading_decision

VALID_RESPONSE = """{
    "date": "2026-03-24",
    "market_regime": "risk_on",
    "regime_confidence": 0.75,
    "regime_reasoning": "Broad market momentum is positive with low VIX",
    "signals": [
        {
            "symbol": "SPY",
            "action": "buy",
            "conviction": "high",
            "target_weight": 0.08,
            "stop_loss": 440.0,
            "reasoning": "Strong momentum and above both SMAs"
        },
        {
            "symbol": "TLT",
            "action": "sell",
            "conviction": "medium",
            "target_weight": 0.03,
            "stop_loss": 90.0,
            "reasoning": "Rising rates pressuring long bonds"
        }
    ],
    "portfolio_commentary": "Increasing equity exposure in risk-on environment"
}"""


def test_parse_valid_response():
    decision = parse_trading_decision(VALID_RESPONSE, date(2026, 3, 24))
    assert decision.market_regime == MarketRegime.RISK_ON
    assert decision.regime_confidence == 0.75
    assert len(decision.signals) == 2
    assert decision.signals[0].symbol == "SPY"
    assert decision.signals[0].action == Action.BUY
    assert decision.signals[0].conviction == Conviction.HIGH
    assert decision.signals[0].target_weight == 0.08
    assert decision.signals[0].stop_loss == 440.0


def test_parse_json_in_code_block():
    wrapped = f"```json\n{VALID_RESPONSE}\n```"
    decision = parse_trading_decision(wrapped, date(2026, 3, 24))
    assert decision.market_regime == MarketRegime.RISK_ON
    assert len(decision.signals) == 2


def test_parse_missing_optional_fields():
    minimal = """{
        "market_regime": "transition",
        "regime_confidence": 0.5,
        "signals": []
    }"""
    decision = parse_trading_decision(minimal, date(2026, 3, 24))
    assert decision.market_regime == MarketRegime.TRANSITION
    assert decision.signals == []
    assert decision.portfolio_commentary == ""


def test_parse_invalid_signal_skipped():
    """Signals with invalid enums should be skipped, not crash."""
    response = """{
        "market_regime": "risk_off",
        "regime_confidence": 0.6,
        "signals": [
            {
                "symbol": "SPY",
                "action": "invalid_action",
                "conviction": "high",
                "target_weight": 0.05,
                "stop_loss": 440.0,
                "reasoning": "Test"
            },
            {
                "symbol": "QQQ",
                "action": "buy",
                "conviction": "medium",
                "target_weight": 0.05,
                "stop_loss": 370.0,
                "reasoning": "Valid signal"
            }
        ]
    }"""
    decision = parse_trading_decision(response, date(2026, 3, 24))
    assert len(decision.signals) == 1
    assert decision.signals[0].symbol == "QQQ"


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_trading_decision("not json at all", date(2026, 3, 24))


def test_parse_clamps_target_weight():
    """target_weight > 0.10 should be clamped."""
    response = """{
        "market_regime": "risk_on",
        "regime_confidence": 0.8,
        "signals": [
            {
                "symbol": "SPY",
                "action": "buy",
                "conviction": "high",
                "target_weight": 0.25,
                "stop_loss": 440.0,
                "reasoning": "Over-sized"
            }
        ]
    }"""
    decision = parse_trading_decision(response, date(2026, 3, 24))
    assert decision.signals[0].target_weight == 0.10


def test_parse_clamps_confidence():
    response = """{
        "market_regime": "risk_on",
        "regime_confidence": 1.5,
        "signals": []
    }"""
    decision = parse_trading_decision(response, date(2026, 3, 24))
    assert decision.regime_confidence == 1.0

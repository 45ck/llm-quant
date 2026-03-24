"""Tests for risk limit checks and risk manager."""

from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.risk.limits import (
    check_cash_reserve,
    check_gross_exposure,
    check_position_size,
    check_position_weight,
    check_stop_loss,
)
from llm_quant.risk.manager import RiskManager


def test_position_size_pass():
    result = check_position_size(
        trade_notional=1_500.0, nav=100_000.0, max_trade_size=0.02
    )
    assert result.passed


def test_position_size_fail():
    result = check_position_size(
        trade_notional=3_000.0, nav=100_000.0, max_trade_size=0.02
    )
    assert not result.passed


def test_position_weight_pass():
    result = check_position_weight(
        current_weight=0.05, target_weight=0.08, max_weight=0.10
    )
    assert result.passed


def test_position_weight_fail():
    result = check_position_weight(
        current_weight=0.05, target_weight=0.12, max_weight=0.10
    )
    assert not result.passed


def test_gross_exposure_pass():
    result = check_gross_exposure(
        current_gross=100_000.0, trade_notional=5_000.0, nav=100_000.0, max_gross=2.0
    )
    assert result.passed


def test_gross_exposure_fail():
    result = check_gross_exposure(
        current_gross=195_000.0, trade_notional=10_000.0, nav=100_000.0, max_gross=2.0
    )
    assert not result.passed


def test_cash_reserve_pass():
    result = check_cash_reserve(
        cash=20_000.0, trade_notional=5_000.0, nav=100_000.0, min_reserve=0.05
    )
    assert result.passed


def test_cash_reserve_fail():
    result = check_cash_reserve(
        cash=6_000.0, trade_notional=2_000.0, nav=100_000.0, min_reserve=0.05
    )
    assert not result.passed


def test_stop_loss_required_present():
    result = check_stop_loss(has_stop_loss=True, require=True)
    assert result.passed


def test_stop_loss_required_missing():
    result = check_stop_loss(has_stop_loss=False, require=True)
    assert not result.passed


def test_stop_loss_not_required():
    result = check_stop_loss(has_stop_loss=False, require=False)
    assert result.passed


def test_risk_manager_approves_valid_signal(
    sample_portfolio,
    sample_prices,
    sample_config,
):
    mgr = RiskManager(sample_config)
    signal = TradeSignal(
        symbol="GLD",
        action=Action.BUY,
        conviction=Conviction.MEDIUM,
        target_weight=0.02,
        stop_loss=175.0,
        reasoning="Hedge",
    )
    approved, rejected = mgr.filter_signals([signal], sample_portfolio, sample_prices)
    assert len(approved) == 1
    assert len(rejected) == 0


def test_risk_manager_rejects_oversized(sample_portfolio, sample_prices, sample_config):
    mgr = RiskManager(sample_config)
    signal = TradeSignal(
        symbol="GLD",
        action=Action.BUY,
        conviction=Conviction.HIGH,
        target_weight=0.15,  # Exceeds 10% max weight
        stop_loss=175.0,
        reasoning="Too big",
    )
    approved, rejected = mgr.filter_signals([signal], sample_portfolio, sample_prices)
    assert len(approved) == 0
    assert len(rejected) == 1


def test_risk_manager_enforces_trade_limit(
    sample_portfolio,
    sample_prices,
    sample_config,
):
    """Only max_trades_per_session signals should be approved."""
    mgr = RiskManager(sample_config)
    signals = [
        TradeSignal(
            symbol=f"ETF{i}",
            action=Action.BUY,
            conviction=Conviction.LOW,
            target_weight=0.01,
            stop_loss=10.0,
            reasoning=f"Signal {i}",
        )
        for i in range(10)
    ]
    # Add all ETFs to prices
    for i in range(10):
        sample_prices[f"ETF{i}"] = 100.0

    approved, _rejected = mgr.filter_signals(signals, sample_portfolio, sample_prices)
    assert len(approved) <= sample_config.risk.max_trades_per_session

"""Tests for portfolio state management."""


from llm_quant.trading.portfolio import Portfolio, Position


def test_position_properties():
    pos = Position(symbol="SPY", shares=10, avg_cost=450.0, current_price=460.0, stop_loss=427.5)
    assert pos.market_value == 4600.0
    assert pos.unrealized_pnl == 100.0
    assert abs(pos.pnl_pct - 0.02222) < 0.001


def test_position_negative_pnl():
    pos = Position(symbol="SPY", shares=10, avg_cost=450.0, current_price=440.0)
    assert pos.unrealized_pnl == -100.0
    assert pos.pnl_pct < 0


def test_portfolio_nav(sample_portfolio):
    # cash=80000, SPY: 20*460=9200, QQQ: 15*390=5850
    expected_nav = 80_000.0 + 9_200.0 + 5_850.0
    assert sample_portfolio.nav == expected_nav


def test_portfolio_empty():
    p = Portfolio(initial_capital=100_000.0)
    assert p.nav == 100_000.0
    assert p.cash == 100_000.0
    assert p.gross_exposure == 0.0
    assert p.total_pnl == 0.0


def test_portfolio_gross_exposure(sample_portfolio):
    # |9200| + |5850| = 15050
    assert sample_portfolio.gross_exposure == 15_050.0


def test_portfolio_update_prices(sample_portfolio):
    sample_portfolio.update_prices({"SPY": 470.0, "QQQ": 400.0})
    assert sample_portfolio.positions["SPY"].current_price == 470.0
    assert sample_portfolio.positions["QQQ"].current_price == 400.0


def test_portfolio_position_weight(sample_portfolio):
    nav = sample_portfolio.nav
    spy_weight = sample_portfolio.get_position_weight("SPY")
    assert abs(spy_weight - (9200.0 / nav)) < 0.0001


def test_portfolio_position_weight_missing():
    p = Portfolio(initial_capital=100_000.0)
    assert p.get_position_weight("AAPL") == 0.0


def test_portfolio_to_snapshot(sample_portfolio):
    snap = sample_portfolio.to_snapshot_dict()
    assert snap["nav"] == sample_portfolio.nav
    assert snap["cash"] == sample_portfolio.cash
    assert len(snap["positions"]) == 2

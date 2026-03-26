"""Tests for prediction market arbitrage scanner."""

from __future__ import annotations

import pytest

from llm_quant.arb.gamma_client import (
    ConditionPrice,
    GammaClient,
    Market,
    _infer_category,
)
from llm_quant.arb.scanner import POLYMARKET_WIN_FEE, ArbScanner

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def negrisk_market() -> Market:
    """4-condition NegRisk market with sum_yes=0.93 (7% complement)."""
    return Market(
        market_id="nba-finals-2025",
        slug="nba-finals-winner",
        question="Who wins 2025 NBA Finals?",
        active=True,
        is_negrisk=True,
        category="sports",
        end_date="2025-06-25",
        conditions=[
            ConditionPrice("cond-celtics", "Celtics win", 0.35, 0.65, 50000, 200000),
            ConditionPrice("cond-lakers", "Lakers win", 0.28, 0.72, 40000, 180000),
            ConditionPrice("cond-warriors", "Warriors win", 0.20, 0.80, 30000, 120000),
            ConditionPrice("cond-heat", "Heat win", 0.10, 0.90, 10000, 50000),
        ],
    )


@pytest.fixture
def single_rebalance_market() -> Market:
    """Single-condition market with YES+NO=0.94 (6% spread)."""
    return Market(
        market_id="spx-mkt",
        slug="spx-above-5500",
        question="Will SPX close above 5500 on March 28?",
        active=True,
        is_negrisk=False,
        category="finance",
        end_date="2026-03-28",
        conditions=[
            ConditionPrice("cond-spx", "SPX above 5500", 0.43, 0.51, 25000, 80000),
        ],
    )


@pytest.fixture
def scanner(tmp_path) -> ArbScanner:
    return ArbScanner(
        db_path=tmp_path / "test.db",
        min_spread_pct=0.03,
        min_volume=5000,
    )


# ------------------------------------------------------------------
# Category inference
# ------------------------------------------------------------------


def test_category_sports():
    assert _infer_category("NBA game 7 Lakers beat Warriors") == "sports"


def test_category_politics():
    assert _infer_category("Will Trump win 2028 election?") == "politics"


def test_category_crypto():
    assert (
        _infer_category("Will Bitcoin price be above 100k by end of year?") == "crypto"
    )


def test_category_other():
    assert _infer_category("Something completely random") == "other"


# ------------------------------------------------------------------
# ConditionPrice math
# ------------------------------------------------------------------


def test_condition_spread_positive():
    c = ConditionPrice("id", "q", 0.60, 0.50, 1000, 5000)
    assert abs(c.spread - 0.10) < 1e-9


def test_condition_spread_negative_is_arb():
    c = ConditionPrice("id", "q", 0.45, 0.48, 1000, 5000)
    assert c.spread < 0
    assert c.is_rebalance_arb  # spread < -0.02


def test_condition_no_arb_near_parity():
    c = ConditionPrice("id", "q", 0.50, 0.51, 1000, 5000)
    assert not c.is_rebalance_arb  # spread = 0.01 > -0.02


# ------------------------------------------------------------------
# Market helpers
# ------------------------------------------------------------------


def test_negrisk_sum_yes(negrisk_market):
    assert abs(negrisk_market.sum_yes - 0.93) < 1e-9


def test_negrisk_complement(negrisk_market):
    assert abs(negrisk_market.negrisk_complement - 0.07) < 1e-9


def test_negrisk_arb_eligible(negrisk_market):
    assert negrisk_market.is_negrisk_arb


def test_negrisk_arb_not_eligible_when_overpriced():
    m = Market(
        market_id="test",
        slug="test",
        question="Test",
        active=True,
        is_negrisk=True,
        category="sports",
        end_date=None,
        conditions=[
            ConditionPrice("a", "A wins", 0.55, 0.45, 5000, 20000),
            ConditionPrice("b", "B wins", 0.50, 0.50, 5000, 20000),
            # sum = 1.05 > 1.0, complement = -0.05 < 0 → no arb
        ],
    )
    assert not m.is_negrisk_arb


# ------------------------------------------------------------------
# Scanner detection
# ------------------------------------------------------------------


def test_negrisk_detect(scanner, negrisk_market):
    opps = scanner._detect_negrisk_arb(negrisk_market)
    assert len(opps) == 1
    opp = opps[0]
    assert opp.arb_type == "negrisk_buy_yes"
    assert abs(opp.spread_pct - 0.07) < 1e-9
    assert abs(opp.net_spread_pct - (0.07 - POLYMARKET_WIN_FEE)) < 1e-9
    expected_kelly = opp.net_spread_pct / (1.0 + opp.net_spread_pct)
    assert abs(opp.kelly_fraction - expected_kelly) < 1e-9
    assert opp.total_volume == pytest.approx(130000)


def test_negrisk_below_threshold(scanner):
    """Market with complement < min_spread should not trigger."""
    m = Market(
        market_id="low-spread",
        slug="low",
        question="Test",
        active=True,
        is_negrisk=True,
        category="other",
        end_date=None,
        conditions=[
            ConditionPrice("a", "A", 0.50, 0.50, 5000, 20000),
            ConditionPrice("b", "B", 0.48, 0.52, 5000, 20000),
            # sum=0.98, complement=0.02 < 0.03 threshold
        ],
    )
    opps = scanner._detect_negrisk_arb(m)
    assert len(opps) == 0


def test_single_rebalance_detect(scanner, single_rebalance_market):
    opps = scanner._detect_single_rebalance(single_rebalance_market)
    assert len(opps) == 1
    opp = opps[0]
    assert opp.arb_type == "single_rebalance"
    # gross = 1 - (0.43 + 0.51) = 0.06
    assert abs(opp.spread_pct - 0.06) < 1e-9
    # net = 0.06 - 0.02 = 0.04
    assert abs(opp.net_spread_pct - 0.04) < 1e-9


def test_single_rebalance_no_arb_when_overpriced(scanner):
    m = Market(
        market_id="no-arb",
        slug="no-arb",
        question="Test",
        active=True,
        is_negrisk=False,
        category="other",
        end_date=None,
        conditions=[ConditionPrice("a", "A", 0.52, 0.50, 5000, 20000)],
        # YES + NO = 1.02 > 1.0, no arb
    )
    opps = scanner._detect_single_rebalance(m)
    assert len(opps) == 0


# ------------------------------------------------------------------
# DB persistence
# ------------------------------------------------------------------


def test_persist_and_query(scanner, negrisk_market, single_rebalance_market):
    opps_nr = scanner._detect_negrisk_arb(negrisk_market)
    opps_sr = scanner._detect_single_rebalance(single_rebalance_market)

    scanner._upsert_markets([negrisk_market, single_rebalance_market])
    scanner._persist_opportunities(opps_nr + opps_sr)

    results = scanner.get_open_opportunities(min_net_spread=0.03)
    assert len(results) == 2
    # Should be sorted by net_spread_pct descending
    assert results[0]["net_spread_pct"] >= results[1]["net_spread_pct"]


# ------------------------------------------------------------------
# Gamma API parser (unit — no network)
# ------------------------------------------------------------------


def test_parse_market_outcomes_string():
    """Gamma API returns outcomes/outcomePrices as JSON strings."""
    raw = {
        "id": "test-mkt",
        "slug": "test",
        "question": "Will it happen?",
        "active": True,
        "isNegRisk": False,
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.65","0.35"]',
        "volumeNum24hr": 12000.0,
        "openInterest": 50000.0,
    }
    client = GammaClient()
    m = client.parse_market(raw)
    assert m is not None
    assert m.market_id == "test-mkt"
    assert len(m.conditions) == 1
    c = m.conditions[0]
    assert abs(c.outcome_yes - 0.65) < 1e-9
    assert abs(c.outcome_no - 0.35) < 1e-9


def test_parse_market_invalid_missing_id():
    raw = {"question": "No ID market"}
    client = GammaClient()
    m = client.parse_market(raw)
    assert m is None

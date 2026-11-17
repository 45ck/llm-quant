"""Leakage, accounting and detection-power contracts for the edge search.

Synthetic inputs only: no network and no market data. A passing test shows the
machinery behaves, not that any strategy is profitable.
"""

from __future__ import annotations

import ast
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from llm_quant.decision_lab import edge_search as es

ROOT = Path(__file__).resolve().parents[1]


def business_dates(count: int) -> list[str]:
    day, out = dt.date(2015, 1, 1), []
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += dt.timedelta(days=1)
    return out


def random_prices(rng: np.random.Generator, days: int, assets: int) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, (days, assets)), axis=0)


def small_plan(dates: list[str]) -> dict:
    cut = int(len(dates) * 0.6)
    return {
        "selection_window": [dates[30], dates[cut - 1]],
        "holdout_window": [dates[cut], dates[-1]],
        "windows": [5],
        "threshold_sigmas": [0.0],
        "directions": [1, -1],
        "delay": 2,
        "cost_per_unit_turnover": 0.0005,
        "sensitivity_cost": 0.0010,
        "min_switches": 10,
        "exposure_bounds": [0.1, 0.9],
        "top_k": [1, 5],
        "primary_top_k": 5,
        "placebo_draws": 30,
        "placebo_min_shift": 50,
        "seed": 7,
    }


def test_frozen_family_is_the_committed_paper_batch_table():
    tree = ast.parse((ROOT / "scripts/run_paper_batch.py").read_text(encoding="utf-8"))
    found = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "LEAD_LAG_PARAMS"
        ):
            found = ast.literal_eval(node.value)
    assert found == es.FROZEN_FAMILY
    assert len(es.FROZEN_FAMILY) == 23


@pytest.mark.parametrize("hold_previous", [True, False])
@pytest.mark.parametrize("delay", [1, 2])
def test_positions_never_use_later_prices(hold_previous, delay):
    rng = np.random.default_rng(1)
    price = random_prices(rng, 300, 1)[:, 0]
    changed = price.copy()
    cut = 200
    changed[cut + 1 :] *= rng.uniform(0.5, 1.5, len(price) - cut - 1)

    def held(series):
        signal = es.window_signal(series, 5)[:, None]
        state = es.position_state(
            signal, np.array([0.01]), np.array([-0.01]), hold_previous=hold_previous
        )
        return es.held_positions(state, delay)[:, 0]

    before, after = held(price), held(changed)
    # The position earning returns[t] may depend on closes up to t - delay only.
    assert np.array_equal(before[: cut + delay + 1], after[: cut + delay + 1])
    assert not np.array_equal(before, after)


def test_threshold_rules_differ_only_in_the_neutral_zone():
    signal = np.array([[0.03], [0.0], [-0.03], [0.0], [math.nan]])
    entry, exit_ = np.array([0.02]), np.array([-0.02])
    hysteresis = es.position_state(signal, entry, exit_, hold_previous=True)[:, 0]
    threshold = es.position_state(signal, entry, exit_, hold_previous=False)[:, 0]
    assert hysteresis.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert threshold.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_excess_and_cost_match_a_hand_calculation():
    held = np.array([0.0, 1.0, 1.0, 0.0])
    returns = np.array([math.nan, 0.02, -0.01, 0.03])
    cost = 0.001 * es.switch_amount(held)
    assert cost.tolist() == [0.0, 0.001, 0.0, 0.001]
    mask = np.array([False, True, True, True])
    out = es.timing_excess(held, returns, cost, mask, weight=0.5)
    exposure = 2 / 3
    assert out["exposure"] == pytest.approx(exposure)
    assert out["switches"] == 2
    assert out["strategy"].tolist() == pytest.approx([0.009, -0.005, -0.001])
    assert out["excess"].tolist() == pytest.approx(
        [
            0.009 - 0.5 * exposure * 0.02,
            -0.005 + 0.5 * exposure * 0.01,
            -0.001 - 0.5 * exposure * 0.03,
        ]
    )


def test_constant_exposure_has_no_timing_excess_before_costs():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.001, 0.01, 50)
    out = es.timing_excess(np.ones(50), returns, np.zeros(50), np.ones(50, bool))
    assert np.allclose(out["excess"], 0.0)


def test_closed_form_grid_equals_explicit_series():
    rng = np.random.default_rng(3)
    prices = random_prices(rng, 400, 4)
    returns = es.simple_returns(prices)
    mask = np.zeros(400, bool)
    mask[50:300] = True
    state, _ = es.build_grid(prices, mask, [5, 10], [0.0, 0.5], [1, -1])
    held = es.held_positions(state, 2)
    cost = 0.0005 * es.switch_amount(held)
    grid = es.grid_statistics(held, cost, returns, mask)
    for config, follower in [(0, 0), (5, 3), (state.shape[1] - 1, 1)]:
        explicit = es.timing_excess(
            held[:, config], returns[:, follower], cost[:, config], mask
        )["excess"]
        assert grid["sharpe"][config, follower] == pytest.approx(es.sharpe(explicit))


def test_selection_statistics_do_not_change_when_the_holdout_changes():
    rng = np.random.default_rng(4)
    dates = business_dates(900)
    prices = random_prices(rng, 900, 5)
    plan = small_plan(dates)
    first = es.search(dates, [f"A{i}" for i in range(5)], prices, plan)
    cut = dates.index(plan["holdout_window"][0])
    altered = prices.copy()
    altered[cut:] *= np.cumprod(1.0 + rng.normal(0, 0.02, (900 - cut, 5)), axis=0)
    second = es.search(dates, [f"A{i}" for i in range(5)], altered, plan)
    assert first["selection_max_sharpe"] == second["selection_max_sharpe"]

    def picked(result):
        return [
            (c["leader"], c["follower"], c["direction"])
            for c in result["top_configurations"]
        ]

    assert picked(first) == picked(second)


def test_planted_lead_lag_survives_the_holdout_and_noise_does_not():
    rng = np.random.default_rng(5)
    days, assets = 1500, 5
    dates = business_dates(days)
    base = rng.normal(0.0002, 0.01, (days, assets))
    noise = 100.0 * np.cumprod(1.0 + base, axis=0)
    planted_returns = base.copy()
    leader_signal = es.window_signal(noise[:, 0], 5)
    for t in range(7, days):
        planted_returns[t, 1] += 0.003 * np.sign(leader_signal[t - 2])
    planted = 100.0 * np.cumprod(1.0 + planted_returns, axis=0)
    names = [f"A{i}" for i in range(assets)]
    plan = small_plan(dates)

    real = es.search(dates, names, planted, plan)
    best = real["top_configurations"][0]
    assert (best["leader"], best["follower"], best["direction"]) == ("A0", "A1", 1)
    assert real["top"]["1"]["holdout"]["p_one_sided"] < 0.01
    assert real["placebo"]["p_selection_max"] < 0.05

    null = es.search(dates, names, noise, plan)
    assert null["selection_max_sharpe"] > 0.3  # search always finds something
    assert null["top"]["1"]["holdout"]["p_one_sided"] > 0.05
    assert null["placebo"]["p_selection_max"] > 0.05


def test_price_loader_fails_closed_on_gaps(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text(
        "date,A,B\n2020-01-01,1.0,\n2020-01-02,1.1,2.0\n2020-01-03,1.2,2.1\n"
    )
    _, tickers, prices = es.load_prices(good)
    assert tickers == ["A", "B"] and math.isnan(prices[0, 1])
    bad = tmp_path / "bad.csv"
    bad.write_text("date,A\n2020-01-01,1.0\n2020-01-02,\n2020-01-03,1.2\n")
    with pytest.raises(ValueError, match="Gap"):
        es.load_prices(bad)


def test_family_reports_every_member_and_corrects_the_whole_family():
    rng = np.random.default_rng(6)
    tickers = sorted({t for row in es.FROZEN_FAMILY.values() for t in row[:2]})
    dates = business_dates(400)
    prices = random_prices(rng, 400, len(tickers))
    out = es.frozen_family(
        dates,
        tickers,
        prices,
        start=dates[100],
        end=dates[-1],
        hold_previous=False,
        delay=1,
        cost_per_switch=0.0003,
        cost_scales_with_weight=False,
    )
    assert set(out["members"]) == set(es.FROZEN_FAMILY)
    assert all("holm_adjusted_p" in m for m in out["members"].values())
    assert out["equal_weight_timing_excess"]["days"] == 300

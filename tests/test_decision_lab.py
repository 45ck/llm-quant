"""Offline tests. No tests access market-data providers or brokerage services."""

from dataclasses import replace
from datetime import date
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from llm_quant.decision_lab import providers
from llm_quant.decision_lab.__main__ import demo, main
from llm_quant.decision_lab.data import (
    Contract, Snapshot, aligned_returns, load_snapshot, save_snapshot, usd_to_aud,
)
from llm_quant.decision_lab.models import Simulation, bootstrap_paths, risk_report, summarize
from llm_quant.decision_lab.projects import ProjectCase, affordability_change, mortgage_payment
from llm_quant.decision_lab.validation import walk_forward


def example() -> Snapshot:
    contract = Contract(
        series_id="TEST", provider="Test fixture",
        source_url="https://example.invalid/data", terms_url="https://example.invalid/terms",
        geography="Fictitious", currency="AUD", frequency="M", basis="total_return",
        retrieved_at="2020-03-01T00:00:00+00:00", vintage="synthetic",
        evidence="synthetic", raw_sha256=hashlib.sha256(b"synthetic").hexdigest(),
    )
    return Snapshot(contract, ("2020-01-01", "2020-02-01", "2020-03-01"), (100., 110., 99.))


def test_snapshot_round_trip_and_hash(tmp_path: Path) -> None:
    snapshot = example()
    path = save_snapshot(snapshot, tmp_path)
    assert load_snapshot(path) == snapshot
    assert save_snapshot(snapshot, tmp_path) == path
    envelope = json.loads(path.read_text())
    envelope["snapshot"]["values"][0] = 50
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        save_snapshot(snapshot, tmp_path)


@pytest.mark.parametrize("values", [(100., 0., 2.), (1., float("nan"), 3.),
                                   (1., float("inf"), 3.), (-1., 2., 3.)])
def test_invalid_levels(values: tuple) -> None:
    with pytest.raises(ValueError, match="finite and strictly positive"):
        replace(example(), values=values).validate()


@pytest.mark.parametrize("dates", [
    ("2020-01-01", "2020-01-01", "2020-03-01"),
    ("2020-03-01", "2020-02-01", "2020-01-01"),
    ("2020-01-01", "2020-03-01", "2020-04-01"),
])
def test_invalid_dates(dates: tuple) -> None:
    with pytest.raises(ValueError):
        replace(example(), dates=dates).validate()


@pytest.mark.parametrize(("field", "value"), [
    ("series_id", "../../escape"), ("frequency", "H"), ("basis", "unknown"),
    ("retrieved_at", "2020-03-01T00:00:00"), ("vintage", "point_in_time"),
    ("source_url", "file:///private"), ("raw_sha256", "not-a-hash"),
    ("evidence", "observed"),
])
def test_invalid_contract(field: str, value: str) -> None:
    snapshot = example()
    with pytest.raises(ValueError):
        replace(snapshot.contract, **{field: value}).validate()


def test_stale_future_and_retrieved_after_origin() -> None:
    snapshot = example()
    with pytest.raises(ValueError, match="Stale"):
        snapshot.check_as_of(date(2020, 4, 1), 10)
    with pytest.raises(ValueError, match="Future"):
        snapshot.check_as_of(date(2020, 2, 1), 10)
    later = replace(snapshot.contract, retrieved_at="2020-04-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="retrieved after"):
        replace(snapshot, contract=later).check_as_of(date(2020, 3, 10), 30)
    snapshot.check_as_of(date(2020, 3, 1), 0)


def test_simple_returns_and_no_implicit_calendar_alignment() -> None:
    snapshot = example()
    np.testing.assert_allclose(aligned_returns([snapshot])[:, 0], [0.1, -0.1])
    other = replace(snapshot, dates=("2020-02-01", "2020-03-01", "2020-04-01"))
    with pytest.raises(ValueError, match="Dates differ"):
        aligned_returns([snapshot, other])


@pytest.mark.parametrize(("field", "value"), [("currency", "USD"), ("frequency", "D"),
                                              ("basis", "adjusted_close_proxy")])
def test_no_implicit_unit_mixing(field: str, value: str) -> None:
    snapshot = example()
    other = replace(snapshot, contract=replace(snapshot.contract, **{field: value}))
    with pytest.raises(ValueError, match="Mixed"):
        aligned_returns([snapshot, other])


def test_housing_is_not_a_tradable_total_return_portfolio() -> None:
    snapshot = example()
    index = replace(snapshot, contract=replace(snapshot.contract, basis="real_price_index"))
    with pytest.raises(ValueError, match="cannot be passed off"):
        aligned_returns([index])
    assert aligned_returns([index], portfolio=False).shape == (2, 1)
    with pytest.raises(ValueError, match="exactly one"):
        aligned_returns([index, index], portfolio=False)


def test_fx_quote_direction() -> None:
    np.testing.assert_allclose(usd_to_aud(np.array([100., 100.]), np.array([.5, 1.])), [200., 100.])
    with pytest.raises(ValueError, match="FX quotes"):
        usd_to_aud(np.array([100.]), np.array([0.]))


def test_stationary_bootstrap_seed_and_joint_rows() -> None:
    returns = np.random.default_rng(1).normal(.002, .02, (120, 1))
    config = Simulation(paths=500)
    single = bootstrap_paths(returns, np.array([1.]), config)
    duplicate = bootstrap_paths(np.repeat(returns, 2, axis=1), np.array([.4, .6]), config)
    np.testing.assert_allclose(single, duplicate)
    np.testing.assert_array_equal(single, bootstrap_paths(returns, np.array([1.]), config))
    assert not np.array_equal(single, bootstrap_paths(returns, np.array([1.]), replace(config, seed=9)))


def test_constant_returns_compound_analytically() -> None:
    wealth = bootstrap_paths(np.full((100, 1), .01), np.array([1.]), Simulation(paths=100))
    np.testing.assert_allclose(wealth[:, -1], 1.01 ** 12)
    assert np.all(wealth[:, 0] == 1)


@pytest.mark.parametrize("weights", [[.2], [-.1, 1.1], [float("nan"), 1.], [.4, .4]])
def test_invalid_weights(weights: list) -> None:
    with pytest.raises(ValueError):
        bootstrap_paths(np.zeros((100, 2)), np.array(weights), Simulation(paths=100))


@pytest.mark.parametrize("change", [
    {"horizon": 0}, {"horizon": 1.5}, {"horizon": True}, {"paths": 99},
    {"block_length": 0}, {"block_length": 101}, {"seed": -1},
    {"horizon": 1000, "paths": 100000},
])
def test_invalid_simulation(change: dict) -> None:
    with pytest.raises(ValueError):
        bootstrap_paths(np.zeros((100, 1)), np.ones(1), replace(Simulation(), **change))


def test_insufficient_history_and_impossible_returns() -> None:
    with pytest.raises(ValueError, match="Insufficient"):
        bootstrap_paths(np.zeros((10, 1)), np.ones(1), Simulation(paths=100))
    with pytest.raises(ValueError, match="Invalid simple"):
        bootstrap_paths(np.full((100, 1), -1.), np.ones(1), Simulation(paths=100))


def test_terminal_touch_and_drawdown_are_different() -> None:
    wealth = np.array([[1., .8, 1.1], [1., 1.3, 1.05]])
    result = summarize(wealth)["loss_events"]["0.1"]
    assert result["terminal_loss_probability"] == 0
    assert result["ever_below_initial_probability"] == .5
    assert result["peak_to_trough_probability"] == 1


def test_exact_threshold_and_initial_wealth_in_drawdown() -> None:
    event = summarize(np.array([[1., .9]]))["loss_events"]["0.1"]
    assert event["terminal_loss_probability"] == 1
    assert event["peak_to_trough_probability"] == 1


def test_risk_report_is_not_a_forecast_or_probability_of_success() -> None:
    result = risk_report(np.zeros((100, 1)), np.ones(1), Simulation(paths=100))
    assert result["status"] == "UNCALIBRATED_HISTORICAL_SCENARIOS"
    assert result["results"]["median_return"] == 0
    assert len(result["limitations"]) >= 5


def test_walk_forward_does_not_fit_future_and_has_nonoverlapping_labels() -> None:
    returns = np.random.default_rng(5).normal(0, .03, (120, 1))
    config = Simulation(paths=300)
    first = walk_forward(returns, np.ones(1), config, min_train=72)
    altered = returns.copy()
    altered[72:] = .2
    second = walk_forward(altered, np.ones(1), config, min_train=72)
    assert first["folds"][0]["predicted_loss_probability"] == second["folds"][0]["predicted_loss_probability"]
    assert first["folds"][0]["training_only_base_rate"] == second["folds"][0]["training_only_base_rate"]
    origins = [fold["origin_return_index"] for fold in first["folds"]]
    assert origins == [72, 84, 96, 108]
    assert first["status"] == "RETROSPECTIVE_DIAGNOSTIC_NOT_POINT_IN_TIME"
    assert 0 <= first["brier_score"] <= 1


def test_no_completed_validation_fold_is_an_error() -> None:
    with pytest.raises(ValueError, match="No complete"):
        walk_forward(np.zeros((80, 1)), np.ones(1), Simulation(paths=100), min_train=72)


def test_project_economics_analytic_break_even() -> None:
    case = ProjectCase(4000, 400, 6, 24, .4, 80, 100)
    result = case.evaluate()
    assert result["expected_cash_contribution"] == pytest.approx(1340)
    assert result["expected_founder_hours"] == pytest.approx(15.6)
    assert result["expected_economic_surplus"] == pytest.approx(92)
    probability = result["economic_break_even_win_probability"]
    assert probability == pytest.approx(580 / 1680)
    assert replace(case, probability_of_winning=probability).evaluate()["expected_economic_surplus"] == pytest.approx(0)


def test_project_unprofitable_even_when_won() -> None:
    result = ProjectCase(1000, 100, 10, 20, .5, 80).evaluate()
    assert result["economic_break_even_win_probability"] is None
    assert not result["positive_economic_surplus_feasible"]


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan")])
def test_project_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        ProjectCase(4000, 400, 6, 24, probability, 80).evaluate()


def test_mortgage_zero_rate_and_affordability_not_fair_value() -> None:
    assert mortgage_payment(360000, 0, 30) == 1000
    assert mortgage_payment(500000, .06, 30) == pytest.approx(2997.752626, rel=1e-7)
    assert affordability_change(1000000, 100000, 8) == pytest.approx(-.2)


def test_fred_parser_missing_and_schema_drift() -> None:
    raw = b"observation_date,QAUR628BIS\n2020-01-01,100\n2020-04-01,102\n"
    assert providers.parse_fred(raw, "QAUR628BIS") == (("2020-01-01", "2020-04-01"), (100., 102.))
    with pytest.raises(ValueError, match="Missing"):
        providers.parse_fred(raw.replace(b",102", b",."), "QAUR628BIS")
    with pytest.raises(ValueError, match="schema"):
        providers.parse_fred(b"<html>error</html>", "QAUR628BIS")


def test_fetch_fred_mocked_http_stores_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "https://fred.stlouisfed.org/data/QAUR628BIS"

    monkeypatch.setattr(providers, "urlopen", lambda *args, **kwargs: Response(
        b"<table><tr><td>Series ID</td><td>QAUR628BIS</td></tr>"
        b"<tr><td>Source</td><td>Bank for International Settlements</td></tr>"
        b"<tr><td>Frequency</td><td>Quarterly</td></tr>"
        b"<tr><td>Units</td><td>Index 2010=100</td></tr>"
        b"<tr><td>Last Updated</td><td>2020-08-01</td></tr>"
        b"<tr><td>Date Range</td><td>2020-01-01 to 2020-04-01</td></tr>"
        b"<tr><td>2020-01-01</td><td>100</td></tr>"
        b"<tr><td>2020-04-01</td><td>102</td></tr></table>",
    ))
    path = providers.fetch_fred("QAUR628BIS", tmp_path)
    snapshot = load_snapshot(path)
    assert snapshot.contract.frequency == "Q"
    assert snapshot.contract.basis == "real_price_index"
    assert snapshot.contract.vintage == "latest"
    assert len(list((tmp_path / "raw").iterdir())) == 1


def test_fetch_unknown_series_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        providers.fetch_fred("../PRIVATE", tmp_path)
    with pytest.raises(ValueError, match="registered"):
        providers.fetch_yahoo("UNKNOWN", "2020-01-01", "2021-01-01", tmp_path)


def test_cli_is_offline_without_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["decision-lab", "fetch", "fred", "QAUR628BIS"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_offline_demo_is_explicitly_synthetic(tmp_path: Path) -> None:
    target = tmp_path / "demo.json"
    demo(target)
    result = json.loads(target.read_text())
    assert result["status"] == "SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE"
    assert len(result["snapshot_sha256"]) == 3
    assert len(result["runtime"]["code_sha256"]) == 64

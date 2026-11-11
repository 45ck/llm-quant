"""Analytical, leakage and cost regressions for the reference benchmark."""

from dataclasses import replace

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from llm_quant.decision_lab.benchmark import (
    METHODS, EvaluationPlan, compare_baselines, fit_weights, performance,
    rebalance, reference_study, simulate,
)


def sample():
    return np.random.default_rng(8).normal(.001, [.01, .03, .07], size=(240, 3))


@pytest.mark.parametrize('method', METHODS)
def test_allocations_are_long_only_and_finite(method):
    weights = fit_weights(sample()[:80], method)
    assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
    assert weights.sum() == pytest.approx(0 if method == 'cash' else 1)


def test_library_shrinkage_minimum_variance_beats_equal_in_training_objective():
    training = sample()[:100]
    covariance = LedoitWolf().fit(training).covariance_
    weights = fit_weights(training, 'minimum_variance_lw')
    equal = np.full(3, 1 / 3)
    assert weights @ covariance @ weights <= equal @ covariance @ equal + 1e-10


def test_one_asset_and_degenerate_covariance():
    weights = fit_weights(sample()[:80, :1], 'minimum_variance_lw')
    np.testing.assert_allclose(weights, [1])
    with pytest.raises(ValueError, match='Degenerate'):
        fit_weights(np.zeros((80, 2)), 'inverse_volatility_lw')


@pytest.mark.parametrize('method', ['future_winner', '', 'leverage'])
def test_unregistered_models_fail(method):
    with pytest.raises(ValueError, match='Unregistered'):
        fit_weights(sample(), method)


@pytest.mark.parametrize('data', [np.array([]), np.array([.1, .2]),
                                 np.full((40, 2), np.nan), np.full((40, 2), -1)])
def test_invalid_data_fail_closed(data):
    with pytest.raises(ValueError):
        compare_baselines(data, EvaluationPlan())


@pytest.mark.parametrize('change', [
    {'splits': 1}, {'test_size': 0}, {'gap': 0}, {'gap': 1, 'label_horizon': 2},
    {'splits': True}, {'holdout_size': 0}, {'min_train': 10}, {'periods_per_year': 0},
    {'transaction_bps': -1}, {'transaction_bps': float('nan')}, {'transaction_bps': 501},
    {'test_size': 200}, {'holdout_size': 1},
])
def test_plan_constraints(change):
    with pytest.raises(ValueError):
        compare_baselines(sample(), replace(EvaluationPlan(), **change))


def test_training_and_holdout_are_separated_by_complete_bar():
    result = compare_baselines(sample(), EvaluationPlan(gap=2, label_horizon=2))
    for fold in result['folds']:
        assert fold['train_end_exclusive'] + 2 == fold['test_start']
        assert fold['test_end_exclusive'] <= result['holdout_start']
    assert result['holdout_train_end_exclusive'] + 2 == result['holdout_start']
    assert result['automatic_model_selection'] is False


def test_future_holdout_mutation_cannot_change_fitted_weights_or_development():
    data = sample()
    original = compare_baselines(data, EvaluationPlan())
    data[-20:] = .25
    mutated = compare_baselines(data, EvaluationPlan())
    for method in METHODS:
        assert original['results'][method]['development'] == mutated['results'][method]['development']
        assert original['results'][method]['holdout_weights'] == mutated['results'][method]['holdout_weights']
        assert original['results'][method]['development_allocations'] == mutated['results'][method]['development_allocations']


def test_gap_observations_not_seen_in_initial_fit():
    data = sample()
    original = compare_baselines(data, EvaluationPlan())
    start = original['evaluation_start']
    data[start - 1] = .5
    changed = compare_baselines(data, EvaluationPlan())
    for method in METHODS:
        assert original['results'][method]['development_allocations']['0'] == changed['results'][method]['development_allocations']['0']


def test_self_financing_entry_fee_has_analytic_solution():
    holdings, cash, fee, turnover = rebalance(np.array([0.]), 1., np.array([1.]), .01)
    assert holdings[0] == pytest.approx(1 / 1.01)
    assert fee == pytest.approx(.01 / 1.01)
    assert turnover == pytest.approx(1 / 1.01)
    assert holdings.sum() + cash + fee == pytest.approx(1)


def test_exit_fee_and_cash_are_not_double_counted():
    holdings, cash, fee, turnover = rebalance(np.array([1.]), 0., np.array([0.]), .01)
    assert cash == pytest.approx(.99)
    assert fee == pytest.approx(.01) and turnover == pytest.approx(1)
    np.testing.assert_array_equal(holdings, [0])


def test_zero_turnover_has_zero_fee():
    holdings, cash, fee, turnover = rebalance(np.array([.4, .6]), 0., np.array([.4, .6]), .02)
    assert fee == 0 and turnover == 0
    assert holdings.sum() + cash == pytest.approx(1)


@pytest.mark.parametrize('scale', [1e-6, 1, 1e6])
def test_accounting_is_scale_invariant(scale):
    holdings, cash, fee, turnover = rebalance(np.array([.3, .2]) * scale,
        .5 * scale, np.array([.2, .7]), .01)
    small = rebalance(np.array([.3, .2]), .5, np.array([.2, .7]), .01)
    np.testing.assert_allclose(holdings / scale, small[0])
    assert (holdings.sum() + cash + fee) / scale == pytest.approx(1)
    assert turnover / scale == pytest.approx(small[3])


@pytest.mark.parametrize('weights', [np.array([1.1]), np.array([-.1]), np.array([np.nan]),
                                   np.array([.5, .5])])
def test_invalid_portfolios_fail(weights):
    with pytest.raises(ValueError):
        rebalance(np.array([0.]), 1., weights, .01)


def test_golden_buy_and_hold_path():
    result = simulate(np.array([[.10], [-.10], [0.]]), {0: np.array([1.])}, 100)
    np.testing.assert_allclose(result['nav'], [1, 1.1 / 1.01, .99 / 1.01, .99 / 1.01])
    assert sum(result['fees_per_initial_unit']) == pytest.approx(.01 / 1.01)


def test_constant_returns_compound_without_rebalancing():
    result = simulate(np.full((20, 1), .01), {0: np.ones(1)}, 0)
    assert result['nav'][-1] == pytest.approx(1.01**20)


def test_fee_sensitivity_and_same_dates_for_baselines():
    low = compare_baselines(sample(), EvaluationPlan(transaction_bps=0))
    high = compare_baselines(sample(), EvaluationPlan(transaction_bps=50))
    assert low['folds'] == high['folds']
    for method in low['results']:
        assert high['results'][method]['development']['nav'][-1] <= low['results'][method]['development']['nav'][-1] + 1e-12


def test_cash_zero_sharpe_is_unavailable_not_a_perfect_strategy():
    result = performance([1., 1., 1.], 252)
    assert result['sharpe_zero_rate'] is None
    assert result['total_return'] == 0 and result['max_drawdown'] == 0


def test_geometric_annualization_and_initial_drawdown():
    result = performance([1., .5, .75], 2)
    assert result['annualized_compound_return'] == pytest.approx(-.25)
    assert result['max_drawdown'] == -.5


def test_reference_is_reproducible_and_explicitly_synthetic():
    first = reference_study()
    assert first == reference_study()
    assert first['status'] == 'SYNTHETIC_REFERENCE_NOT_MARKET_EVIDENCE'
    assert len(first['fixture_sha256']) == 64
    assert first['cost_sensitivity']['0'] > first['cost_sensitivity']['10'] > first['cost_sensitivity']['50']


def test_results_bind_plan_code_and_library_versions():
    original = compare_baselines(sample(), EvaluationPlan())
    changed = compare_baselines(sample(), EvaluationPlan(transaction_bps=20))
    identity = original['identity']
    assert len(identity['implementation_sha256']) == 64
    assert identity['implementation_sha256'] == changed['identity']['implementation_sha256']
    assert identity['plan_sha256'] != changed['identity']['plan_sha256']
    assert set(identity['library_versions']) == {'numpy', 'scipy', 'scikit-learn'}

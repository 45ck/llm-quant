"""Known-value and applicability checks for imported statistical methods."""

import pytest

from llm_quant.decision_lab.statistics import event_rate_interval, holm_family


def test_zero_events_does_not_imply_zero_underlying_risk():
    result = event_rate_interval(0, 35, independent_bernoulli_assumption=True)
    assert result['sample_rate'] == 0 and result['lower'] == 0
    assert result['upper'] == pytest.approx(1-.025**(1/35))


def test_all_events_has_lower_bound_below_one():
    result = event_rate_interval(35, 35, independent_bernoulli_assumption=True)
    assert result['upper'] == 1 and result['lower'] < 1


def test_dependence_assumption_is_not_silently_made():
    with pytest.raises(ValueError, match='Bernoulli assumption'):
        event_rate_interval(1, 36)


@pytest.mark.parametrize(('events','n'), [(1, 0), (-1, 10), (11, 10), (True, 10), (1, 1.5)])
def test_bad_counts(events, n):
    with pytest.raises(ValueError):
        event_rate_interval(events, n, independent_bernoulli_assumption=True)


def test_holm_matches_hand_calculation_in_original_order():
    result = holm_family([.04, .01, .03], declared_trials=3)
    assert result['adjusted_pvalues'] == pytest.approx([.06, .03, .06])
    assert result['reject_null'] == [False, True, False]


def test_missing_failed_trials_prevents_selective_correction():
    with pytest.raises(ValueError, match='all declared'):
        holm_family([.01], declared_trials=30)


@pytest.mark.parametrize('values', [[], [.01, float('nan')], [-.1], [1.1]])
def test_invalid_pvalues(values):
    with pytest.raises(ValueError):
        holm_family(values, declared_trials=len(values))

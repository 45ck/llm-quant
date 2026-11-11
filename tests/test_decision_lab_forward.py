"""Forward journal rejects backdating, premature scoring and corruption."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from llm_quant.decision_lab import forward

NOW = datetime(2026, 11, 11, 5, tzinfo=UTC)


class Clock(datetime):
    current = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz is None else cls.current.astimezone(tz)


@pytest.fixture(autouse=True)
def controlled_clock(monkeypatch):
    Clock.current = NOW
    monkeypatch.setattr(forward, 'datetime', Clock)


def forecast():
    return forward.Forecast('FIXTURE', 'a'*64, 'b'*64,
        (NOW-timedelta(hours=1)).isoformat(), (NOW+timedelta(days=7)).isoformat(),
        100., .1, .25)


def mature():
    Clock.current = NOW+timedelta(days=8)


def settle(path, key, level=90):
    return forward.settle_forecast(path, key, observed_at=forecast().matures_at,
                                  level=level, source_sha256='c'*64)


def test_issue_and_exact_maturity_outcome_brier(tmp_path):
    path = tmp_path/'journal.jsonl'
    key = forward.issue_forecast(path, forecast())
    before = path.read_bytes()
    mature()
    head = settle(path, key)
    assert path.read_bytes().startswith(before)
    result = forward.score_journal(path, head)
    assert result['forecasts'] == result['scored_outcomes'] == 1
    assert result['brier_score'] == pytest.approx((.25-1)**2)
    assert result['mature_unscored_forecasts'] == 0


def test_empty_is_not_zero_performance(tmp_path):
    result = forward.score_journal(tmp_path/'missing.jsonl')
    assert result['brier_score'] is None and result['head_sha256'] == forward.GENESIS


def test_premature_outcome_refused_and_journal_unchanged(tmp_path):
    path = tmp_path/'journal.jsonl'
    key = forward.issue_forecast(path, forecast())
    before = path.read_bytes()
    with pytest.raises(ValueError, match='not mature'):
        settle(path, key)
    assert path.read_bytes() == before
    assert not path.with_name(path.name+'.lock').exists()


def test_mature_missing_outcomes_are_visible(tmp_path):
    path = tmp_path/'journal.jsonl'
    forward.issue_forecast(path, forecast())
    mature()
    result = forward.score_journal(path)
    assert result['mature_unscored_forecasts'] == 1
    assert result['brier_score'] is None


@pytest.mark.parametrize('change', [
    {'data_through': (NOW+timedelta(hours=1)).isoformat()},
    {'matures_at': NOW.isoformat()}, {'data_through': '2026-11-11T00:00:00'},
    {'initial_level': 0}, {'initial_level': float('nan')}, {'loss_probability': 1.1},
    {'loss_probability': -.1}, {'loss_threshold': 0}, {'loss_threshold': 1},
    {'max_data_age_seconds': 10}, {'max_data_age_seconds': True},
    {'model_sha256': 'unhashed'}, {'snapshot_sha256': 'not-a-digest'},
    {'series_id': '../../secret'},
])
def test_invalid_forecasts_never_write(tmp_path, change):
    path = tmp_path/'journal.jsonl'
    with pytest.raises(ValueError):
        forward.issue_forecast(path, replace(forecast(), **change))
    assert not path.exists()


def test_no_duplicate_forecasts_or_outcomes(tmp_path):
    path = tmp_path/'journal.jsonl'
    key = forward.issue_forecast(path, forecast())
    with pytest.raises(ValueError, match='Duplicate'):
        forward.issue_forecast(path, forecast())
    mature()
    settle(path, key)
    with pytest.raises(ValueError, match='already settled'):
        settle(path, key)


def test_wrong_outcome_date_refused(tmp_path):
    path = tmp_path/'journal.jsonl'
    key = forward.issue_forecast(path, forecast())
    mature()
    with pytest.raises(ValueError, match='exact preregistered'):
        forward.settle_forecast(path, key, observed_at=Clock.current.isoformat(),
                               level=90, source_sha256='c'*64)


def test_mutation_of_past_record_detected(tmp_path):
    path = tmp_path/'journal.jsonl'
    forward.issue_forecast(path, forecast())
    value = json.loads(path.read_text())
    value['record']['payload']['loss_probability'] = .9
    path.write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError, match='hash chain'):
        forward.read_journal(path)


def test_partial_write_and_truncation_against_external_head_detected(tmp_path):
    path = tmp_path/'journal.jsonl'
    key = forward.issue_forecast(path, forecast())
    first = path.read_bytes()
    mature()
    head = settle(path, key)
    path.write_bytes(first)
    with pytest.raises(ValueError, match='retained head'):
        forward.read_journal(path, head)
    path.write_bytes(first[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        forward.read_journal(path)


def test_concurrent_writer_lock_fails_closed(tmp_path):
    path = tmp_path/'journal.jsonl'
    with forward.exclusive_lock(path):
        with pytest.raises(FileExistsError):
            forward.issue_forecast(path, forecast())
    assert not path.exists()


def test_clock_rollback_does_not_append(tmp_path):
    path = tmp_path/'journal.jsonl'
    forward.issue_forecast(path, forecast())
    Clock.current -= timedelta(minutes=10)
    with pytest.raises(ValueError, match='backwards'):
        forward.issue_forecast(path, replace(forecast(), loss_probability=.3))


def test_no_loss_scores_correctly(tmp_path):
    path = tmp_path/'journal.jsonl'
    key = forward.issue_forecast(path, forecast())
    mature()
    settle(path, key, 110)
    assert forward.score_journal(path)['brier_score'] == pytest.approx(.25**2)

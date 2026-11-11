"""Actual optional-library execution, separately selected from mock contract tests.

Absent dependencies are SKIPPED locally, not passed. The explicit CI job rejects
any skipped selected test, so a missing selected integration fails the job.
"""

from pathlib import Path

import numpy as np
import pytest

from llm_quant.decision_lab.integrations.__main__ import fixture, ohlc_fixture, probe
from llm_quant.decision_lab.integrations.allocation import cvxpy_min_cvar, skfolio_min_risk
from llm_quant.decision_lab.integrations.dataframes import pandera_ohlc
from llm_quant.decision_lab.integrations.execution import next_open_buy_hold
from llm_quant.decision_lab.integrations.uncertainty import arch_mean_interval


def test_arch_native():
    pytest.importorskip("arch", exc_type=ImportError)
    answer = probe("arch")
    assert answer["volatility"]["output"]["one_period_variance"] > 0
    first = arch_mean_interval(fixture(), repetitions=100, seed=9)
    second = arch_mean_interval(fixture(), repetitions=100, seed=9)
    assert first == second


def test_skfolio_native():
    pytest.importorskip("skfolio", exc_type=ImportError)
    for risk in ("variance", "cvar"):
        answer = skfolio_min_risk(fixture(), measure=risk, cap=.6)
        assert sum(answer["output"]["weights"]) == pytest.approx(1)
        assert max(answer["output"]["weights"]) <= .600001


def test_cvxpy_native():
    pytest.importorskip("cvxpy", exc_type=ImportError)
    answer = cvxpy_min_cvar(fixture(), cap=.6)
    assert answer["output"]["solver_status"] == "optimal"
    assert sum(answer["output"]["weights"]) == pytest.approx(1)


def test_cross_engine_cvar_native():
    pytest.importorskip("cvxpy", exc_type=ImportError)
    pytest.importorskip("skfolio", exc_type=ImportError)
    sample = fixture()
    direct = cvxpy_min_cvar(sample)
    library = skfolio_min_risk(sample, measure="cvar")
    # Independent implementations, same empirical objective. Allocation need not be unique.
    def cvar(weights):
        losses = -sample.returns @ np.asarray(weights)
        threshold = float(np.quantile(losses, .95))
        return threshold + float(np.maximum(losses - threshold, 0).mean()) / .05
    assert cvar(direct["output"]["weights"]) == pytest.approx(cvar(library["output"]["weights"]), abs=1e-5)


def test_pandera_native():
    pytest.importorskip("pandera", exc_type=ImportError)
    original = ohlc_fixture()
    assert pandera_ohlc(original).equals(original)
    original.iloc[0, 1] = .1
    with pytest.raises(ValueError):
        pandera_ohlc(original)


def test_backtesting_native():
    pytest.importorskip("backtesting", exc_type=ImportError)
    frame = ohlc_fixture()
    zero = next_open_buy_hold(frame, acknowledge_agpl=True, commission_bps=0)
    costly = next_open_buy_hold(frame, acknowledge_agpl=True, commission_bps=100)
    assert zero["entry_bar"] == 2 and zero["exit_bar"] == len(frame) - 1
    assert costly["final_equity"] < zero["final_equity"]


def test_qlib_native(tmp_path, monkeypatch):
    pytest.importorskip("qlib", exc_type=ImportError)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    answer = probe("qlib", qlib_directory=Path("research-local/qlib"))
    assert answer["recorder_id"] and answer["uri"].startswith("file:")


def test_hypothesis_accounting_native():
    hypothesis = pytest.importorskip("hypothesis", exc_type=ImportError)
    from hypothesis import strategies as st
    from llm_quant.decision_lab.benchmark import rebalance

    @hypothesis.settings(max_examples=75, derandomize=True, database=None, deadline=None)
    @hypothesis.given(cash=st.floats(min_value=1, max_value=1e7, allow_nan=False, allow_infinity=False),
                      rate=st.floats(min_value=0, max_value=.05, allow_nan=False, allow_infinity=False),
                      weight=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    def property_check(cash, rate, weight):
        holdings, remainder, fee, traded = rebalance(np.zeros(1), cash, np.array([weight]), rate)
        assert float(holdings.sum()) + remainder + fee == pytest.approx(cash, rel=1e-9)
        assert fee >= 0 and traded >= 0 and remainder >= 0
    property_check()


@pytest.fixture(autouse=True)
def no_network_in_native_tests(monkeypatch):
    import socket
    original = socket.socket.connect
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Network I/O is forbidden in native integration tests")
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", connect)

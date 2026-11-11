"""Executed core/contract checks; mocks do NOT certify optional library runtimes."""

from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from llm_quant.decision_lab.benchmark import EvaluationPlan
from llm_quant.decision_lab.integrations import allocation, contracts, tracking, uncertainty
from llm_quant.decision_lab.integrations.__main__ import fixture, main, ohlc_fixture, probe
from llm_quant.decision_lab.integrations.contracts import TrainingBatch, checked_weights
from llm_quant.decision_lab.integrations.dataframes import validate_ohlc
from llm_quant.decision_lab.integrations.evaluation import evaluate
from llm_quant.decision_lab.integrations.execution import next_open_buy_hold


def batch(values=None, **changes):
    original = fixture()
    fields = dict(returns=original.returns if values is None else values, dates=original.dates,
                  as_of=original.as_of, currency=original.currency, evidence=original.evidence)
    fields.update(changes)
    return TrainingBatch(**fields)


@pytest.mark.parametrize("changes", [
    {"currency": "INDEX"}, {"currency": ""}, {"evidence": "forecast"},
    {"as_of": "1900-01-01"}, {"dates": ("2000-01-01",)},
    {"dates": tuple(reversed(fixture().dates))},
    {"dates": tuple([fixture().dates[0]] * 240)},
])
def test_invalid_batch_metadata(changes):
    with pytest.raises(ValueError):
        batch(**changes)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1, -2])
def test_invalid_training_returns(value):
    values = fixture().returns.copy()
    values[4, 0] = value
    with pytest.raises(ValueError):
        batch(values)


@pytest.mark.parametrize("shape", [(2, 3), (20, 0), (20, 201), (20,), (2, 2, 2)])
def test_invalid_training_shape(shape):
    with pytest.raises(ValueError):
        batch(np.zeros(shape))


def test_training_batch_copies_and_binds_inputs():
    values = fixture().returns.copy()
    sample = batch(values)
    identity = sample.digest()
    values[:] = 0.5
    assert sample.digest() == identity
    assert sample.returns.flags.writeable is False
    assert batch(evidence="observed").digest() != identity
    assert batch(currency="USD").digest() != identity


def test_doctor_does_not_import_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optional module imported")
    monkeypatch.setattr(contracts.importlib, "import_module", forbidden)
    assert "NOT_EXECUTION" in contracts.doctor()["status"]


def test_missing_dependency_is_not_replaced(monkeypatch):
    def missing(*args):
        raise ModuleNotFoundError("missing arch", name="arch")
    monkeypatch.setattr(contracts.importlib, "import_module", missing)
    with pytest.raises(contracts.MissingIntegration, match="no automatic"):
        contracts.load("arch")


def test_broken_transitive_dependency_is_not_masked(monkeypatch):
    def broken(*args):
        raise ModuleNotFoundError("broken transitive", name="unrelated_dependency")
    monkeypatch.setattr(contracts.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="broken transitive") as caught:
        contracts.load("arch")
    assert not isinstance(caught.value, contracts.MissingIntegration)


@pytest.mark.parametrize("key,submodule", [("os", ""), ("arch", "../../os"), ("arch", "x/y")])
def test_loader_rejects_unregistered_modules(key, submodule):
    with pytest.raises(ValueError):
        contracts.load(key, submodule)


@pytest.mark.parametrize("weights", [[.2, .2, .2], [-.01, .51, .5], [np.nan, 0, 1], [1, 0], [[1, 0, 0]]])
def test_invalid_optimizer_results(weights):
    with pytest.raises(ValueError):
        checked_weights(weights, 3, 1)


def test_tiny_numerical_projection_is_disclosed():
    result = checked_weights([-.000000001, .5, .500000001], 3, 1)
    assert result["numerical_projection_l1"] > 0
    assert sum(result["weights"]) == pytest.approx(1)


def test_core_shrinkage_and_hac_actual_libraries():
    sample = fixture()
    allocation_result = allocation.sklearn_inverse_volatility(sample)
    assert allocation_result["status"] == "SYNTHETIC_INTEGRATION_CHECK"
    assert sum(allocation_result["output"]["weights"]) == pytest.approx(1)
    single = batch(sample.returns[:, :1])
    hac = uncertainty.statsmodels_hac_mean(single, maxlags=3)
    assert hac["output"]["mean_return"] == pytest.approx(float(single.returns.mean()))
    assert 0 <= hac["output"]["p_value_zero_mean"] <= 1
    assert hac["output"]["lower"] < hac["output"]["upper"]
    assert hac["libraries"]["statsmodels"] != "NOT_INSTALLED"


@pytest.mark.parametrize("lag", [-1, 120, 1.5, True])
def test_hac_invalid_lags(lag):
    with pytest.raises(ValueError):
        uncertainty.statsmodels_hac_mean(batch(fixture().returns[:, :1]), maxlags=lag)


def test_hac_rejects_multivariate():
    with pytest.raises(ValueError, match="one series"):
        uncertainty.statsmodels_hac_mean(fixture())


def test_constant_covariance_fails():
    with pytest.raises(ValueError, match="Degenerate"):
        allocation.sklearn_inverse_volatility(batch(np.zeros((240, 3))))


def test_skfolio_calls_declared_api_contract(monkeypatch):
    observed = {}
    class Estimator:
        def __init__(self, **kwargs):
            observed.update(kwargs)
        def fit(self, values):
            observed["shape"] = values.shape
            self.weights_ = np.array([.2, .3, .5])
    risk = SimpleNamespace(VARIANCE="variance", CVAR="cvar")
    opt = SimpleNamespace(MeanRisk=Estimator, ObjectiveFunction=SimpleNamespace(MINIMIZE_RISK="minimize"))
    monkeypatch.setattr(allocation, "load", lambda key, module="": opt if module else SimpleNamespace(RiskMeasure=risk))
    answer = allocation.skfolio_min_risk(fixture(), measure="cvar", cap=.6)
    assert observed["cvar_beta"] == .95
    assert observed["raise_on_failure"] is True
    assert observed["budget"] == 1
    assert observed["shape"] == (240, 3)
    assert answer["output"]["weights"] == [.2, .3, .5]


@pytest.mark.parametrize("cap", [0, -.1, float("nan"), .2, 1.5])
def test_bad_caps_fail_before_backend_import(cap):
    with pytest.raises(ValueError):
        allocation.skfolio_min_risk(fixture(), cap=cap)
    with pytest.raises(ValueError):
        allocation.cvxpy_min_cvar(fixture(), cap=cap)


@pytest.mark.parametrize("alpha", [.5, 1, 2, float("nan")])
def test_bad_tail_probability(alpha):
    with pytest.raises(ValueError):
        allocation.cvxpy_min_cvar(fixture(), alpha=alpha)


def test_arch_bootstrap_api_contract(monkeypatch):
    called = {}
    class Bootstrap:
        def __init__(self, block, values, seed):
            called.update(block=block, seed=seed, mean=values.mean(axis=0))
        def conf_int(self, fn, **kwargs):
            called.update(kwargs)
            assert fn(fixture().returns).shape == (3,)
            return np.array([[-.1] * 3, [.1] * 3])
    monkeypatch.setattr(uncertainty, "load", lambda *args: SimpleNamespace(StationaryBootstrap=Bootstrap))
    answer = uncertainty.arch_mean_interval(fixture(), repetitions=100, block=4, seed=8)
    assert called["method"] == "percentile"
    assert called["seed"] == 8
    assert answer["output"]["lower"] == [-.1] * 3


@pytest.mark.parametrize("kwargs", [{"block": 0}, {"block": 241}, {"block": True},
                                    {"repetitions": 10}, {"repetitions": 10001}, {"seed": -1}])
def test_arch_budget_guards(kwargs):
    with pytest.raises(ValueError):
        uncertainty.arch_mean_interval(fixture(), **kwargs)


def test_garch_percent_units_and_convergence_contract(monkeypatch):
    observed = {}
    def estimator(values, **kwargs):
        np.testing.assert_allclose(values, 100 * fixture().returns[:, 0])
        observed.update(kwargs)
        fitted = SimpleNamespace(convergence_flag=0, nobs=len(values), forecast=lambda **options:
                                 SimpleNamespace(variance=pd.DataFrame([[4.0]])))
        return SimpleNamespace(fit=lambda **options: fitted)
    monkeypatch.setattr(uncertainty, "load", lambda *args: SimpleNamespace(arch_model=estimator))
    answer = uncertainty.arch_garch_variance(batch(fixture().returns[:, :1]))
    assert observed["dist"] == "t" and observed["rescale"] is False
    assert answer["output"]["one_period_variance"] == pytest.approx(.0004)
    assert answer["output"]["one_period_volatility"] == pytest.approx(.02)


def test_garch_failure_does_not_create_forecast(monkeypatch):
    model = SimpleNamespace(fit=lambda **options: SimpleNamespace(convergence_flag=1))
    monkeypatch.setattr(uncertainty, "load", lambda *args: SimpleNamespace(arch_model=lambda *a, **k: model))
    with pytest.raises(ValueError, match="convergence"):
        uncertainty.arch_garch_variance(batch(fixture().returns[:, :1]))


def test_ohlc_validation_does_not_modify_input():
    original = ohlc_fixture()
    result = validate_ohlc(original)
    result.iloc[0, 0] = 1
    assert original.iloc[0, 0] != 1


@pytest.mark.parametrize("mutation", ["nan", "zero", "envelope", "extra", "duplicate", "reverse", "integer"])
def test_ohlc_rejects_broken_data(mutation):
    frame = ohlc_fixture()
    if mutation == "nan": frame.iloc[2, 0] = np.nan
    elif mutation == "zero": frame.iloc[2, 0] = 0
    elif mutation == "envelope": frame.iloc[2, 1] = .1
    elif mutation == "extra": frame["Volume"] = 1.
    elif mutation == "duplicate": frame.index = [frame.index[0]] * len(frame)
    elif mutation == "reverse": frame = frame.iloc[::-1]
    elif mutation == "integer": frame = frame.astype(int)
    with pytest.raises(ValueError):
        validate_ohlc(frame)


def test_agpl_ack_required_before_import(monkeypatch):
    monkeypatch.setattr(contracts, "load", lambda *args: pytest.fail("should not import"))
    with pytest.raises(ValueError, match="AGPL"):
        next_open_buy_hold(ohlc_fixture())


def test_no_future_data_in_fitted_allocations():
    sample = fixture()
    plan = EvaluationPlan()
    first = evaluate(sample, plan, ("sklearn_inverse_volatility",))
    modified = sample.returns.copy()
    modified[-plan.holdout_size:] = .4
    second = evaluate(batch(modified), plan, ("sklearn_inverse_volatility",))
    name = "sklearn_inverse_volatility"
    assert first["results"][name]["holdout_fit_receipt"] == second["results"][name]["holdout_fit_receipt"]
    assert first["results"][name]["fit_receipts"] == second["results"][name]["fit_receipts"]
    assert first["selected_winner"] is None
    assert first["results"][name]["reserved_holdout"]["metrics"] != second["results"][name]["reserved_holdout"]["metrics"]


@pytest.mark.parametrize("methods", [(), ("unknown",), ("skfolio_cvar", "skfolio_cvar")])
def test_evaluation_invalid_selection(methods):
    with pytest.raises(ValueError):
        evaluate(fixture(), EvaluationPlan(), methods)


def test_qlib_record_is_local_scalar_allowlist(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    captured = {}
    class Context:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Recorder:
        def __init__(self, manager): captured["manager"] = manager
        def start(self, **kwargs): captured.update(kwargs); return Context()
        def log_params(self, **kwargs): captured["logged"] = kwargs
        def get_recorder(self): return SimpleNamespace(id="fixture-id")
    manager = SimpleNamespace(MLflowExpManager=lambda **kwargs: kwargs)
    monkeypatch.setattr(tracking, "load", lambda key, sub: manager if sub.endswith("expm")
                        else SimpleNamespace(QlibRecorder=Recorder))
    summary = allocation.sklearn_inverse_volatility(fixture())
    summary["private_raw_data"] = "must not be logged"
    answer = tracking.record_qlib_summary(summary, Path("research-local/run"))
    assert answer["recorder_id"] == "fixture-id"
    assert captured["manager"]["uri"].startswith("file:")
    assert "private_raw_data" not in captured["logged"] and "output" not in captured["logged"]


@pytest.mark.parametrize("directory", ["../elsewhere", "public", "research-local"])
def test_qlib_path_guards(directory, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="subdirectory"):
        tracking.record_qlib_summary({}, Path(directory))


def test_qlib_external_environment_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    with pytest.raises(ValueError, match="external MLflow"):
        tracking.record_qlib_summary({}, Path("research-local/run"))


def test_cli_doctor_and_native_core(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["integrations", "doctor"])
    main()
    assert "DEPENDENCY_INVENTORY" in capsys.readouterr().out
    assert probe("sklearn")["status"] == "SYNTHETIC_INTEGRATION_CHECK"
    assert probe("statsmodels")["status"] == "SYNTHETIC_INTEGRATION_CHECK"


def test_cli_qlib_requires_path():
    with pytest.raises(ValueError, match="explicit"):
        probe("qlib")


ROOT = Path(__file__).resolve().parents[1]


def test_skill_and_catalog_contracts():
    import runpy
    check = runpy.run_path(str(ROOT / "scripts/check_quant_integrations.py"))["check"]
    assert check(ROOT) == []


@pytest.mark.parametrize("mutation", ["name", "description", "permission", "link", "policy", "command", "catalog"])
def test_skill_mutations_fail(tmp_path, mutation):
    import runpy
    import shutil
    check = runpy.run_path(str(ROOT / "scripts/check_quant_integrations.py"))["check"]
    for relative in (".claude/skills", "requirements/integrations", "config/decision_lab"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    for relative in ("RESEARCH_POLICY.md", "docs/research/integrations.md"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    target = tmp_path / ".claude/skills/quant-research/SKILL.md"
    text = target.read_text()
    if mutation == "name": text = text.replace("name: quant-research", "name: WRONG")
    elif mutation == "description": text = text.replace("description:", "unexpected:")
    elif mutation == "permission": text = text.replace("license: MIT", "allowed-tools: Bash(*)\nlicense: MIT")
    elif mutation == "link": text = text.replace("../../../RESEARCH_POLICY.md", "../../../missing.md")
    elif mutation == "policy": text = text.replace("RESEARCH_POLICY.md", "missing.md")
    elif mutation == "command": text = text.replace("decision_lab.integrations", "no_such_command")
    elif mutation == "catalog":
        catalog = tmp_path / "config/decision_lab/integrations.json"
        catalog.write_text(catalog.read_text().replace('"id": "arch"', '"id": "qlib"'))
    target.write_text(text)
    assert check(tmp_path)


@pytest.mark.parametrize("xml", [
    '<testsuites><testsuite tests="0"/></testsuites>',
    '<testsuite><testcase name="x"><skipped/></testcase></testsuite>',
    '<testsuite><testcase name="x"><failure/></testcase></testsuite>',
    '<testsuite><testcase name="x"><error/></testcase></testsuite>',
])
def test_native_job_rejects_skip_empty_or_failed(xml, tmp_path):
    import runpy
    check = runpy.run_path(str(ROOT / "scripts/check_native_integration_results.py"))["check"]
    path = tmp_path / "junit.xml"
    path.write_text(xml)
    assert check(path)


def test_native_job_accepts_executed_pass(tmp_path):
    import runpy
    check = runpy.run_path(str(ROOT / "scripts/check_native_integration_results.py"))["check"]
    path = tmp_path / "junit.xml"
    path.write_text('<testsuite><testcase name="actual-run"/></testsuite>')
    assert check(path) == []


def test_native_workflow_is_explicit_unprivileged_and_bounded():
    import yaml
    workflow = yaml.load((ROOT / ".github/workflows/research-integrations.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert int(workflow["jobs"]["native-library"]["timeout-minutes"]) <= 10
    text = (ROOT / ".github/workflows/research-integrations.yml").read_text()
    assert 'check_native_integration_results.py' in text
    assert '|| true' not in text and 'continue-on-error' not in text


def test_focused_workflow_uses_supported_checker_options():
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github/workflows/decision-lab.yml").read_text()
    assert "python scripts/check_quant_integrations.py --environment" not in text
    assert "python scripts/check_research_standards.py --environment" in text

"""QA tests for HRP weight allocation.

Validates:
1. Weights sum to 1.0 +/- 0.001
2. No single strategy weight > 0.15 (Track B limit)
3. Track A strategies sum to ~0.70 +/- 0.05
4. Track B strategies sum to ~0.30 +/- 0.05
5. No negative weights
6. HRP produces different weights from equal-weight (otherwise why use it)
7. validate_weights() passes valid input without raising
8. validate_weights() warns on sum > 1.0 (does not raise but logs warning)

Run: cd E:/llm-quant && PYTHONPATH=src pytest tests/test_hrp_weights.py -v
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Import the optimizer module directly without going through package machinery
# ---------------------------------------------------------------------------

_OPTIMIZER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "portfolio_optimizer.py"

spec = importlib.util.spec_from_file_location("portfolio_optimizer", _OPTIMIZER_PATH)
_optimizer = importlib.util.module_from_spec(spec)
sys.modules["portfolio_optimizer"] = _optimizer
spec.loader.exec_module(_optimizer)

compute_hrp_weights = _optimizer.compute_hrp_weights
validate_weights = _optimizer.validate_weights
TRACK_A_SLUGS = _optimizer.TRACK_A_SLUGS
TRACK_B_SLUGS = _optimizer.TRACK_B_SLUGS
MAX_STRATEGY_WEIGHT = _optimizer.MAX_STRATEGY_WEIGHT


# ---------------------------------------------------------------------------
# Helpers: synthetic return data
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)

# Use a representative subset: 3 Track A strategies + 1 Track B strategy.
# This avoids needing Riskfolio to handle 15+ series and keeps tests fast.
_A_SLUGS_SAMPLE = [
    "lqd-spy-credit-lead",
    "agg-spy-credit-lead",
    "spy-overnight-momentum",
]
_B_SLUGS_SAMPLE = ["soxx-qqq-lead-lag"]
_ALL_SLUGS_SAMPLE = _A_SLUGS_SAMPLE + _B_SLUGS_SAMPLE

_MIN_LEN = 252  # one year of daily returns


def _make_strategies(slugs: list[str], n: int = _MIN_LEN) -> dict[str, dict]:
    """Build a synthetic strategies dict with correlated return series."""
    # Introduce mild positive correlation (factor structure)
    common = _RNG.normal(0.0005, 0.008, size=n)
    strategies: dict[str, dict] = {}
    for i, slug in enumerate(slugs):
        idio = _RNG.normal(0.0002 * (i + 1), 0.005, size=n)
        returns = 0.6 * common + 0.4 * idio
        strategies[slug] = {"daily_returns": returns.tolist()}
    return strategies


def _run_hrp(slugs: list[str] | None = None) -> dict[str, float]:
    """Run compute_hrp_weights on synthetic data, skip if Riskfolio absent."""
    try:
        import Riskfolio  # noqa: F401
    except ImportError:
        pytest.skip("Riskfolio-Lib not installed — skipping HRP test")

    slugs = slugs or _ALL_SLUGS_SAMPLE
    strategies = _make_strategies(slugs)
    return compute_hrp_weights(strategies, slugs, _MIN_LEN)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hrp_weights_sum_to_one():
    """HRP weights must sum to 1.0 within tolerance."""
    weights = _run_hrp()
    total = sum(weights.values())
    assert abs(total - 1.0) <= 0.001, (
        f"Weights sum to {total:.6f}, expected 1.000 +/- 0.001"
    )


def test_no_weight_exceeds_track_b_limit():
    """No individual strategy weight may exceed the Track B cap (15%)."""
    weights = _run_hrp()
    for slug, w in weights.items():
        assert w <= MAX_STRATEGY_WEIGHT + 1e-9, (
            f"Strategy '{slug}' weight={w:.4f} exceeds Track B limit of "
            f"{MAX_STRATEGY_WEIGHT:.0%}"
        )


def test_track_a_allocation():
    """Track A slugs in the sample must collectively receive ~70% +/- 5%."""
    weights = _run_hrp()
    a_total = sum(weights.get(s, 0.0) for s in _A_SLUGS_SAMPLE)
    assert 0.65 <= a_total <= 0.75, (
        f"Track A allocation={a_total:.4f}, expected 0.65-0.75"
    )


def test_track_b_allocation():
    """Track B slugs in the sample must collectively receive ~30% +/- 5%."""
    weights = _run_hrp()
    b_total = sum(weights.get(s, 0.0) for s in _B_SLUGS_SAMPLE)
    assert 0.25 <= b_total <= 0.35, (
        f"Track B allocation={b_total:.4f}, expected 0.25-0.35"
    )


def test_no_negative_weights():
    """HRP is long-only — all weights must be >= 0."""
    weights = _run_hrp()
    for slug, w in weights.items():
        assert w >= 0.0, f"Strategy '{slug}' has negative weight {w:.6f}"


def test_hrp_differs_from_equal_weight():
    """HRP should produce non-uniform weights for correlated return series."""
    try:
        import Riskfolio  # noqa: F401
    except ImportError:
        pytest.skip("Riskfolio-Lib not installed — skipping HRP test")

    slugs = _ALL_SLUGS_SAMPLE
    weights = _run_hrp(slugs)

    # Equal-weight baseline (respecting 70/30 split)
    n_a = len(_A_SLUGS_SAMPLE)
    n_b = len(_B_SLUGS_SAMPLE)
    eq_a = 0.70 / n_a
    eq_b = 0.30 / n_b
    equal_weights = {s: eq_a for s in _A_SLUGS_SAMPLE}
    equal_weights.update({s: eq_b for s in _B_SLUGS_SAMPLE})

    # At least one weight should differ by more than 0.001 from equal-weight
    diffs = [abs(weights.get(s, 0.0) - equal_weights[s]) for s in slugs]
    assert max(diffs) > 0.001, (
        "HRP weights are indistinguishable from equal-weight — possible HRP "
        "implementation issue"
    )


def test_validate_weights_passes_valid_input(caplog):
    """validate_weights() should not log any warnings for a valid weight dict.

    Use 8 strategies so each weight stays well below the 15% cap while
    summing to exactly 1.0.
    """
    # 8 strategies at <= 15% each, total = 1.0
    weights = {
        "lqd-spy-credit-lead": 0.10,
        "agg-spy-credit-lead": 0.10,
        "hyg-spy-5d-credit-lead": 0.10,
        "agg-qqq-credit-lead": 0.10,
        "lqd-qqq-credit-lead": 0.10,
        "vcit-qqq-credit-lead": 0.10,
        "spy-overnight-momentum": 0.10,
        "soxx-qqq-lead-lag": 0.30,
    }
    # The soxx-qqq slug has weight 0.30 which is > 0.15 — drop it down
    # Use a 10-strategy dict where every weight is exactly 0.10
    weights = {
        "lqd-spy-credit-lead": 0.08,
        "agg-spy-credit-lead": 0.08,
        "hyg-spy-5d-credit-lead": 0.08,
        "agg-qqq-credit-lead": 0.08,
        "lqd-qqq-credit-lead": 0.08,
        "vcit-qqq-credit-lead": 0.08,
        "hyg-qqq-credit-lead": 0.08,
        "spy-overnight-momentum": 0.08,
        "tlt-spy-rate-momentum": 0.08,
        "soxx-qqq-lead-lag": 0.12,  # Track B at 12%, under 15% cap
        "behavioral-structural": 0.08,
        "gld-slv-mean-reversion-v4": 0.08,
    }
    # Verify test precondition: all weights <= 15% and sum = 1.0
    assert all(w <= MAX_STRATEGY_WEIGHT + 1e-9 for w in weights.values()), (
        "Test setup error: some weights exceed 15%"
    )
    assert abs(sum(weights.values()) - 1.0) < 0.001, (
        "Test setup error: weights do not sum to 1.0"
    )
    with caplog.at_level(logging.WARNING, logger="portfolio_optimizer"):
        validate_weights(weights, method="test")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0, (
        f"Expected no warnings for valid weights, got: {[w.message for w in warnings]}"
    )


def test_validate_weights_warns_sum_gt_one(caplog):
    """validate_weights() must log a warning when weights sum to > 1.001."""
    weights = {
        "lqd-spy-credit-lead": 0.60,
        "agg-spy-credit-lead": 0.60,
    }
    # sum = 1.20, well above threshold
    with caplog.at_level(logging.WARNING, logger="portfolio_optimizer"):
        validate_weights(weights, method="test")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) >= 1, "Expected at least one warning for weights summing to 1.2"
    assert any("1.2" in w.message or "sum" in w.message.lower() for w in warnings), (
        f"Warning did not mention the sum violation: {[w.message for w in warnings]}"
    )

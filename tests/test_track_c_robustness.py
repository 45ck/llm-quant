"""Tests for Track C structural arb robustness gate router.

Covers: detect_strategy_class() routing logic and YAML artifact format.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the scripts directory is importable (script uses relative imports via sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_track_c_robustness import _TRACK_C_PREFIXES, detect_strategy_class

# ---------------------------------------------------------------------------
# detect_strategy_class routing
# ---------------------------------------------------------------------------


def test_detect_pm_arb_kalshi():
    assert detect_strategy_class("pm-arb-kalshi") == "pm_arb"


def test_detect_pm_arb_polymarket():
    assert detect_strategy_class("pm-arb-polymarket") == "pm_arb"


def test_detect_pm_arb_generic():
    assert detect_strategy_class("pm-arb-sports") == "pm_arb"


def test_detect_cef_discount():
    assert detect_strategy_class("cef-pdi-discount") == "cef_discount"


def test_detect_cef_any():
    assert detect_strategy_class("cef-gof-mean-reversion") == "cef_discount"


def test_detect_funding_rate():
    assert detect_strategy_class("funding-btc-binance") == "funding_rate"


def test_detect_funding_eth():
    assert detect_strategy_class("funding-eth-bybit") == "funding_rate"


def test_detect_track_ab_returns_none():
    """Standard Track A/B slugs should not be detected as Track C."""
    for slug in [
        "soxx-qqq-lead-lag",
        "lqd-spy-credit-lead",
        "spy-overnight-momentum",
        "tlt-spy-rate-momentum",
        "ief-qqq-rate-tech",
        "agg-efa-credit-lead",
    ]:
        assert detect_strategy_class(slug) is None, f"Expected None for {slug!r}"


def test_detect_empty_string():
    assert detect_strategy_class("") is None


def test_detect_partial_prefix_no_match():
    """Slugs that contain but don't START with the prefix should not match."""
    assert detect_strategy_class("my-pm-arb-strategy") is None
    assert detect_strategy_class("no-cef-here") is None


def test_all_prefixes_registered():
    """Sanity: every prefix in the dict maps to a non-empty class string."""
    for prefix, cls in _TRACK_C_PREFIXES.items():
        assert prefix.endswith("-"), f"Prefix should end with '-': {prefix!r}"
        assert isinstance(cls, str) and cls, f"Class must be non-empty string: {cls!r}"

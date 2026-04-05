"""Tests for strategy-to-track routing."""

import pytest
import yaml

from llm_quant.trading.track_router import (
    StrategyEntry,
    TrackInfo,
    TrackRouter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_yaml(tmp_path):
    """Write a minimal track-assignments.yaml and return its path."""
    data = {
        "tracks": {
            "track-a": {
                "pod_id": "track-a",
                "display_name": "Track A -- Defensive Alpha",
                "benchmark": "60/40 SPY/TLT",
                "initial_capital": 100000,
                "target_allocation_pct": 70,
                "strategies": [
                    {
                        "slug": "lqd-spy-credit-lead",
                        "family": "F1",
                        "mechanism": "credit_lead",
                        "description": "LQD 5d return leads SPY",
                    },
                    {
                        "slug": "gld-slv-mean-reversion-v4",
                        "family": "F2",
                        "mechanism": "precious_metals_mr",
                        "description": "GLD/SLV ratio",
                    },
                ],
            },
            "track-b": {
                "pod_id": "track-b",
                "display_name": "Track B -- Aggressive Alpha",
                "benchmark": "100% SPY",
                "initial_capital": 100000,
                "target_allocation_pct": 30,
                "strategies": [
                    {
                        "slug": "soxx-qqq-lead-lag",
                        "family": "F8",
                        "mechanism": "semi_lead_lag",
                        "description": "SOXX 5d return leads QQQ",
                    },
                ],
            },
            "track-d": {
                "pod_id": "track-d",
                "display_name": "Track D -- Sprint Alpha",
                "benchmark": "100% TQQQ",
                "initial_capital": 100000,
                "target_allocation_pct": 0,
                "strategies": [
                    {
                        "slug": "tlt-tqqq-leveraged-lead-lag",
                        "family": "F6-leveraged",
                        "mechanism": "leveraged_rate_lead",
                        "description": "TLT -> TQQQ",
                    },
                ],
            },
            "discretionary": {
                "pod_id": "discretionary",
                "display_name": "Discretionary -- Manual Macro",
                "benchmark": "60/40 SPY/TLT",
                "initial_capital": 100000,
                "target_allocation_pct": 0,
                "strategies": [
                    {
                        "slug": "manual-macro",
                        "family": None,
                        "mechanism": "discretionary",
                        "description": "Manual macro trades",
                    },
                ],
            },
        }
    }
    yaml_path = tmp_path / "track-assignments.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return yaml_path


@pytest.fixture
def router(sample_yaml):
    """Load a TrackRouter from the sample YAML."""
    return TrackRouter.load_from_yaml(sample_yaml)


# ---------------------------------------------------------------------------
# load_from_yaml
# ---------------------------------------------------------------------------


class TestLoadFromYaml:
    def test_loads_all_tracks(self, router):
        """All 4 tracks are loaded."""
        assert sorted(router.track_ids) == [
            "discretionary",
            "track-a",
            "track-b",
            "track-d",
        ]

    def test_loads_all_strategies(self, router):
        """All 5 strategies are indexed."""
        assert len(router.all_slugs) == 5

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            TrackRouter.load_from_yaml(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# get_track
# ---------------------------------------------------------------------------


class TestGetTrack:
    def test_track_a_slug(self, router):
        assert router.get_track("lqd-spy-credit-lead") == "track-a"

    def test_track_b_slug(self, router):
        assert router.get_track("soxx-qqq-lead-lag") == "track-b"

    def test_track_d_slug(self, router):
        assert router.get_track("tlt-tqqq-leveraged-lead-lag") == "track-d"

    def test_discretionary_slug(self, router):
        assert router.get_track("manual-macro") == "discretionary"

    def test_unknown_slug_raises(self, router):
        with pytest.raises(KeyError, match="not-a-real-strategy"):
            router.get_track("not-a-real-strategy")


# ---------------------------------------------------------------------------
# get_strategies
# ---------------------------------------------------------------------------


class TestGetStrategies:
    def test_track_a_has_two(self, router):
        slugs = router.get_strategies("track-a")
        assert slugs == ["lqd-spy-credit-lead", "gld-slv-mean-reversion-v4"]

    def test_track_b_has_one(self, router):
        assert router.get_strategies("track-b") == ["soxx-qqq-lead-lag"]

    def test_unknown_track_raises(self, router):
        with pytest.raises(KeyError, match="track-z"):
            router.get_strategies("track-z")


# ---------------------------------------------------------------------------
# get_all_tracks
# ---------------------------------------------------------------------------


class TestGetAllTracks:
    def test_returns_all_tracks(self, router):
        all_tracks = router.get_all_tracks()
        assert len(all_tracks) == 4
        assert "track-a" in all_tracks
        assert "track-b" in all_tracks
        assert "track-d" in all_tracks
        assert "discretionary" in all_tracks

    def test_strategy_counts(self, router):
        all_tracks = router.get_all_tracks()
        assert len(all_tracks["track-a"]) == 2
        assert len(all_tracks["track-b"]) == 1
        assert len(all_tracks["track-d"]) == 1
        assert len(all_tracks["discretionary"]) == 1


# ---------------------------------------------------------------------------
# get_track_info / get_strategy_entry
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_track_info(self, router):
        info = router.get_track_info("track-a")
        assert isinstance(info, TrackInfo)
        assert info.pod_id == "track-a"
        assert info.display_name == "Track A -- Defensive Alpha"
        assert info.benchmark == "60/40 SPY/TLT"
        assert info.initial_capital == 100_000.0
        assert info.target_allocation_pct == 70.0

    def test_track_info_unknown_raises(self, router):
        with pytest.raises(KeyError, match="track-z"):
            router.get_track_info("track-z")

    def test_strategy_entry(self, router):
        entry = router.get_strategy_entry("soxx-qqq-lead-lag")
        assert isinstance(entry, StrategyEntry)
        assert entry.slug == "soxx-qqq-lead-lag"
        assert entry.family == "F8"
        assert entry.mechanism == "semi_lead_lag"

    def test_strategy_entry_unknown_raises(self, router):
        with pytest.raises(KeyError, match="nonexistent"):
            router.get_strategy_entry("nonexistent")


# ---------------------------------------------------------------------------
# Production YAML
# ---------------------------------------------------------------------------


class TestProductionYaml:
    """Smoke-test against the real config/track-assignments.yaml."""

    def test_loads_production_config(self):
        """The production YAML loads without errors."""
        router = TrackRouter.load_from_yaml()
        assert len(router.track_ids) >= 4
        assert len(router.all_slugs) >= 33

    def test_known_strategy_routes(self):
        """Spot-check a few known strategies."""
        router = TrackRouter.load_from_yaml()
        assert router.get_track("lqd-spy-credit-lead") == "track-a"
        assert router.get_track("soxx-qqq-lead-lag") == "track-b"
        assert router.get_track("tlt-tqqq-leveraged-lead-lag") == "track-d"
        assert router.get_track("manual-macro") == "discretionary"


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr(router):
    r = repr(router)
    assert "TrackRouter" in r
    assert "track-a" in r

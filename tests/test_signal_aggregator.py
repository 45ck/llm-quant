"""Tests for the signal aggregation engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_quant.trading.signal_aggregator import (
    SignalAggregator,
    TrackSignal,
    _extract_latest_signal,
    _parse_date,
)
from llm_quant.trading.track_router import (
    StrategyEntry,
    TrackInfo,
    TrackRouter,
)

# ── helpers ──────────────────────────────────────────────────────


def _make_router(
    tracks: dict[str, list[str]],
) -> TrackRouter:
    """Build a minimal TrackRouter from {track_id: [slug, ...]}."""
    track_infos: dict[str, TrackInfo] = {}
    for tid, slugs in tracks.items():
        entries = [
            StrategyEntry(
                slug=s,
                family=None,
                mechanism="test",
                description="test strategy",
            )
            for s in slugs
        ]
        track_infos[tid] = TrackInfo(
            pod_id=tid,
            display_name=tid,
            benchmark="test",
            initial_capital=100_000,
            target_allocation_pct=50,
            strategies=entries,
        )
    return TrackRouter(track_infos)


def _write_paper_yaml(
    strategies_dir: Path,
    slug: str,
    daily_log: list[dict],
    status: str = "active",
) -> None:
    """Write a paper-trading.yaml for a strategy."""
    strat_dir = strategies_dir / slug
    strat_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "strategy_slug": slug,
        "status": status,
        "daily_log": daily_log,
    }
    with (strat_dir / "paper-trading.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


# ── _parse_date ──────────────────────────────────────────────────


class TestParseDate:
    def test_date_object(self) -> None:
        d = date(2026, 4, 4)
        assert _parse_date(d) == d

    def test_iso_string(self) -> None:
        assert _parse_date("2026-04-04") == date(2026, 4, 4)

    def test_invalid_string(self) -> None:
        assert _parse_date("not-a-date") is None

    def test_none(self) -> None:
        assert _parse_date(None) is None

    def test_int(self) -> None:
        assert _parse_date(20260404) is None


# ── _extract_latest_signal ───────────────────────────────────────


class TestExtractLatestSignal:
    def test_returns_latest_entry(self) -> None:
        paper_data = {
            "daily_log": [
                {
                    "date": "2026-04-02",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.8, "SHY": 0.2},
                    "regime": "risk_on",
                    "signal_desc": "test signal",
                },
                {
                    "date": "2026-04-03",
                    "position": "risk_off",
                    "allocation": {"TLT": 0.9, "SHY": 0.1},
                    "regime": "risk_off",
                    "signal_desc": "newer signal",
                },
            ]
        }
        result = _extract_latest_signal(paper_data, date(2026, 4, 4))
        assert result is not None
        assert result["date"] == "2026-04-03"
        assert result["allocation"] == {
            "TLT": 0.9,
            "SHY": 0.1,
        }

    def test_stale_signal_returns_none(self) -> None:
        paper_data = {
            "daily_log": [
                {
                    "date": "2026-03-20",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.8},
                },
            ]
        }
        # ref_date is 2026-04-04, >5 days old
        result = _extract_latest_signal(paper_data, date(2026, 4, 4), staleness_days=5)
        assert result is None

    def test_empty_daily_log(self) -> None:
        result = _extract_latest_signal({"daily_log": []}, date(2026, 4, 4))
        assert result is None

    def test_no_daily_log_key(self) -> None:
        result = _extract_latest_signal({}, date(2026, 4, 4))
        assert result is None

    def test_skips_pending_entries(self) -> None:
        paper_data = {
            "daily_log": [
                {
                    "date": "2026-04-02",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.8},
                },
                {
                    "date": "2026-04-03",
                    "position": "pending_eod",
                    "allocation": {},
                },
            ]
        }
        result = _extract_latest_signal(paper_data, date(2026, 4, 4))
        assert result is not None
        assert result["date"] == "2026-04-02"

    def test_no_allocation_returns_empty_dict(self) -> None:
        """Older-format entries without allocation field."""
        paper_data = {
            "daily_log": [
                {
                    "date": "2026-04-03",
                    "position": "flat",
                    "signal": "exit",
                    "signal_reason": "credit weak",
                },
            ]
        }
        result = _extract_latest_signal(paper_data, date(2026, 4, 4))
        assert result is not None
        assert result["allocation"] == {}

    def test_future_date_skipped(self) -> None:
        paper_data = {
            "daily_log": [
                {
                    "date": "2026-04-10",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.8},
                },
                {
                    "date": "2026-04-03",
                    "position": "risk_on",
                    "allocation": {"TLT": 0.5},
                },
            ]
        }
        result = _extract_latest_signal(paper_data, date(2026, 4, 4))
        assert result is not None
        assert result["date"] == "2026-04-03"
        assert result["allocation"] == {"TLT": 0.5}


# ── SignalAggregator ─────────────────────────────────────────────


class TestSignalAggregatorLoadSignal:
    def test_load_existing_strategy(self, tmp_path: Path) -> None:
        router = _make_router({"t": ["strat-a"]})
        _write_paper_yaml(
            tmp_path,
            "strat-a",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.8},
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg._load_strategy_signal("strat-a")
        assert result is not None
        assert result["allocation"] == {"SPY": 0.8}

    def test_missing_yaml_returns_none(self, tmp_path: Path) -> None:
        router = _make_router({"t": ["no-yaml"]})
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        assert agg._load_strategy_signal("no-yaml") is None

    def test_paused_strategy_skipped(self, tmp_path: Path) -> None:
        router = _make_router({"t": ["paused"]})
        _write_paper_yaml(
            tmp_path,
            "paused",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.8},
                }
            ],
            status="paused",
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        assert agg._load_strategy_signal("paused") is None


# ── aggregate_track ──────────────────────────────────────────────


class TestAggregateTrack:
    def test_equal_weight_two_strategies(self, tmp_path: Path) -> None:
        router = _make_router({"track-a": ["s1", "s2"]})
        _write_paper_yaml(
            tmp_path,
            "s1",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.80, "SHY": 0.20},
                }
            ],
        )
        _write_paper_yaml(
            tmp_path,
            "s2",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_off",
                    "allocation": {"SPY": 0.20, "TLT": 0.60},
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("track-a")

        assert result.track_id == "track-a"
        assert result.n_active_strategies == 2
        assert result.n_total_strategies == 2
        # SPY: (0.80 + 0.20) / 2 = 0.50
        assert result.net_allocations["SPY"] == pytest.approx(0.50, abs=1e-6)
        # SHY: (0.20 + 0.0) / 2 = 0.10
        assert result.net_allocations["SHY"] == pytest.approx(0.10, abs=1e-6)
        # TLT: (0.0 + 0.60) / 2 = 0.30
        assert result.net_allocations["TLT"] == pytest.approx(0.30, abs=1e-6)

    def test_stale_strategy_excluded(self, tmp_path: Path) -> None:
        router = _make_router({"track-a": ["fresh", "stale"]})
        _write_paper_yaml(
            tmp_path,
            "fresh",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.80},
                }
            ],
        )
        _write_paper_yaml(
            tmp_path,
            "stale",
            [
                {
                    "date": "2026-03-15",
                    "position": "risk_on",
                    "allocation": {"SPY": 0.20},
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("track-a")

        assert result.n_active_strategies == 1
        assert result.n_total_strategies == 2
        # Only fresh signal's allocation
        assert result.net_allocations["SPY"] == pytest.approx(0.80, abs=1e-6)

    def test_missing_yaml_excluded(self, tmp_path: Path) -> None:
        router = _make_router({"track-a": ["exists", "missing"]})
        _write_paper_yaml(
            tmp_path,
            "exists",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_on",
                    "allocation": {"GLD": 0.50},
                }
            ],
        )
        # "missing" has no paper-trading.yaml
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("track-a")

        assert result.n_active_strategies == 1
        assert result.n_total_strategies == 2
        assert result.net_allocations["GLD"] == pytest.approx(0.50, abs=1e-6)

    def test_no_active_strategies(self, tmp_path: Path) -> None:
        router = _make_router({"track-a": ["empty"]})
        # No paper-trading.yaml at all
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("track-a")

        assert result.n_active_strategies == 0
        assert result.n_total_strategies == 1
        assert result.net_allocations == {}

    def test_cash_excluded_from_allocations(self, tmp_path: Path) -> None:
        router = _make_router({"track-a": ["s1"]})
        _write_paper_yaml(
            tmp_path,
            "s1",
            [
                {
                    "date": "2026-04-04",
                    "position": "risk_on",
                    "allocation": {
                        "SPY": 0.80,
                        "cash": 0.10,
                        "SHY": 0.10,
                    },
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("track-a")

        assert "cash" not in result.net_allocations
        assert "SPY" in result.net_allocations
        assert "SHY" in result.net_allocations

    def test_three_strategy_math(self, tmp_path: Path) -> None:
        """Verify equal-weight math with 3 strategies."""
        router = _make_router({"t": ["s1", "s2", "s3"]})
        _write_paper_yaml(
            tmp_path,
            "s1",
            [
                {
                    "date": "2026-04-04",
                    "position": "a",
                    "allocation": {"SPY": 0.90, "TLT": 0.10},
                }
            ],
        )
        _write_paper_yaml(
            tmp_path,
            "s2",
            [
                {
                    "date": "2026-04-04",
                    "position": "b",
                    "allocation": {"SPY": 0.30, "GLD": 0.60},
                }
            ],
        )
        _write_paper_yaml(
            tmp_path,
            "s3",
            [
                {
                    "date": "2026-04-04",
                    "position": "c",
                    "allocation": {"SPY": 0.60, "TLT": 0.40},
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("t")

        assert result.n_active_strategies == 3
        # SPY: (0.90 + 0.30 + 0.60) / 3 = 0.60
        assert result.net_allocations["SPY"] == pytest.approx(0.60, abs=1e-6)
        # TLT: (0.10 + 0 + 0.40) / 3 ≈ 0.166667
        assert result.net_allocations["TLT"] == pytest.approx(0.166667, abs=1e-4)
        # GLD: (0 + 0.60 + 0) / 3 = 0.20
        assert result.net_allocations["GLD"] == pytest.approx(0.20, abs=1e-6)


# ── aggregate_all_tracks ─────────────────────────────────────────


class TestAggregateAllTracks:
    def test_returns_all_tracks(self, tmp_path: Path) -> None:
        router = _make_router({"track-a": ["sa"], "track-b": ["sb"]})
        _write_paper_yaml(
            tmp_path,
            "sa",
            [
                {
                    "date": "2026-04-04",
                    "position": "on",
                    "allocation": {"SPY": 0.80},
                }
            ],
        )
        _write_paper_yaml(
            tmp_path,
            "sb",
            [
                {
                    "date": "2026-04-04",
                    "position": "on",
                    "allocation": {"TQQQ": 0.30},
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        results = agg.aggregate_all_tracks()

        assert set(results.keys()) == {"track-a", "track-b"}
        assert results["track-a"].net_allocations["SPY"] == pytest.approx(0.80)
        assert results["track-b"].net_allocations["TQQQ"] == pytest.approx(0.30)


# ── to_json_decision ─────────────────────────────────────────────


class TestToJsonDecision:
    def test_basic_format(self) -> None:
        ts = TrackSignal(
            track_id="track-a",
            date="2026-04-04",
            n_active_strategies=5,
            n_total_strategies=10,
            net_allocations={"SPY": 0.50, "TLT": 0.30},
            strategy_details=[
                {"slug": "s1", "active": True},
                {"slug": "s2", "active": True},
                {"slug": "s3", "active": False},
                {"slug": "s4", "active": True},
                {"slug": "s5", "active": True},
                {"slug": "s6", "active": True},
            ],
        )
        decision = SignalAggregator.to_json_decision(
            ts, regime="risk_on", regime_confidence=0.7
        )

        assert decision["market_regime"] == "risk_on"
        assert decision["regime_confidence"] == 0.7
        assert "track-a" in decision["regime_reasoning"]
        assert len(decision["signals"]) == 2

        spy_sig = next(s for s in decision["signals"] if s["symbol"] == "SPY")
        assert spy_sig["action"] == "buy"
        assert spy_sig["target_weight"] == pytest.approx(0.10)  # clamped to 0.10

        tlt_sig = next(s for s in decision["signals"] if s["symbol"] == "TLT")
        assert tlt_sig["target_weight"] == pytest.approx(0.10)  # clamped to 0.10

    def test_small_weights_not_clamped(self) -> None:
        ts = TrackSignal(
            track_id="track-a",
            date="2026-04-04",
            n_active_strategies=2,
            n_total_strategies=2,
            net_allocations={"GLD": 0.05},
            strategy_details=[
                {"slug": "s1", "active": True},
                {"slug": "s2", "active": True},
            ],
        )
        decision = SignalAggregator.to_json_decision(ts)

        assert len(decision["signals"]) == 1
        assert decision["signals"][0]["target_weight"] == pytest.approx(0.05)

    def test_zero_weight_excluded(self) -> None:
        ts = TrackSignal(
            track_id="track-a",
            date="2026-04-04",
            n_active_strategies=1,
            n_total_strategies=1,
            net_allocations={"SPY": 0.0, "TLT": 0.50},
            strategy_details=[
                {"slug": "s1", "active": True},
            ],
        )
        decision = SignalAggregator.to_json_decision(ts)

        assert len(decision["signals"]) == 1
        assert decision["signals"][0]["symbol"] == "TLT"

    def test_empty_allocations(self) -> None:
        ts = TrackSignal(
            track_id="track-a",
            date="2026-04-04",
            n_active_strategies=0,
            n_total_strategies=3,
            net_allocations={},
            strategy_details=[],
        )
        decision = SignalAggregator.to_json_decision(ts)

        assert decision["signals"] == []
        assert "portfolio_commentary" in decision

    def test_decision_has_required_keys(self) -> None:
        ts = TrackSignal(
            track_id="track-b",
            date="2026-04-04",
            n_active_strategies=1,
            n_total_strategies=1,
            net_allocations={"SOXX": 0.08},
            strategy_details=[
                {"slug": "soxx-qqq", "active": True},
            ],
        )
        decision = SignalAggregator.to_json_decision(ts)

        required = {
            "market_regime",
            "regime_confidence",
            "regime_reasoning",
            "signals",
            "portfolio_commentary",
        }
        assert required.issubset(decision.keys())

    def test_signal_has_required_fields(self) -> None:
        ts = TrackSignal(
            track_id="track-a",
            date="2026-04-04",
            n_active_strategies=1,
            n_total_strategies=1,
            net_allocations={"SPY": 0.06},
            strategy_details=[
                {"slug": "s1", "active": True},
            ],
        )
        decision = SignalAggregator.to_json_decision(ts)

        sig = decision["signals"][0]
        required = {
            "symbol",
            "action",
            "conviction",
            "target_weight",
            "stop_loss",
            "reasoning",
        }
        assert required.issubset(sig.keys())


# ── strategy_details ─────────────────────────────────────────────


class TestStrategyDetails:
    def test_details_include_all_strategies(self, tmp_path: Path) -> None:
        router = _make_router({"t": ["active-one", "missing-one"]})
        _write_paper_yaml(
            tmp_path,
            "active-one",
            [
                {
                    "date": "2026-04-04",
                    "position": "on",
                    "allocation": {"SPY": 0.5},
                }
            ],
        )
        agg = SignalAggregator(
            strategies_dir=tmp_path,
            router=router,
            ref_date=date(2026, 4, 4),
        )
        result = agg.aggregate_track("t")

        assert len(result.strategy_details) == 2
        slugs = {d["slug"] for d in result.strategy_details}
        assert slugs == {"active-one", "missing-one"}

        active_detail = next(
            d for d in result.strategy_details if d["slug"] == "active-one"
        )
        assert active_detail["active"] is True
        assert active_detail["allocation"] == {"SPY": 0.5}

        missing_detail = next(
            d for d in result.strategy_details if d["slug"] == "missing-one"
        )
        assert missing_detail["active"] is False

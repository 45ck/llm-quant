"""Strategy-to-track routing.

Reads config/track-assignments.yaml and provides lookup methods
to route strategy slugs to their owning track (pod_id) and vice versa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llm_quant.config import _find_config_dir

logger = logging.getLogger(__name__)

DEFAULT_ASSIGNMENTS_PATH = _find_config_dir() / "track-assignments.yaml"


@dataclass(frozen=True)
class StrategyEntry:
    """Metadata for a single strategy within a track."""

    slug: str
    family: str | None
    mechanism: str
    description: str


@dataclass
class TrackInfo:
    """Metadata for a track (pod)."""

    pod_id: str
    display_name: str
    benchmark: str
    initial_capital: float
    target_allocation_pct: float
    strategies: list[StrategyEntry] = field(default_factory=list)


class TrackRouter:
    """Routes strategy slugs to tracks and vice versa.

    Usage::

        router = TrackRouter.load_from_yaml()
        track = router.get_track("lqd-spy-credit-lead")   # "track-a"
        slugs = router.get_strategies("track-b")           # [...]
    """

    def __init__(self, tracks: dict[str, TrackInfo]) -> None:
        self._tracks = tracks
        # Build reverse index: slug -> pod_id
        self._slug_to_track: dict[str, str] = {}
        for track_id, info in tracks.items():
            for entry in info.strategies:
                if entry.slug in self._slug_to_track:
                    logger.warning(
                        "Duplicate slug '%s' — already assigned to '%s', "
                        "overwriting with '%s'",
                        entry.slug,
                        self._slug_to_track[entry.slug],
                        track_id,
                    )
                self._slug_to_track[entry.slug] = track_id

    # -- factory ---------------------------------------------------------------

    @classmethod
    def load_from_yaml(cls, path: Path | str | None = None) -> TrackRouter:
        """Load track assignments from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML file.  Defaults to
            ``config/track-assignments.yaml`` relative to the project root.
        """
        yaml_path = Path(path) if path is not None else DEFAULT_ASSIGNMENTS_PATH
        if not yaml_path.exists():
            raise FileNotFoundError(f"Track assignments file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        tracks_raw = raw.get("tracks", {})
        tracks: dict[str, TrackInfo] = {}

        for track_id, track_data in tracks_raw.items():
            strategies = [
                StrategyEntry(
                    slug=s["slug"],
                    family=s.get("family"),
                    mechanism=s.get("mechanism", ""),
                    description=s.get("description", ""),
                )
                for s in track_data.get("strategies", [])
            ]
            tracks[track_id] = TrackInfo(
                pod_id=track_data.get("pod_id", track_id),
                display_name=track_data.get("display_name", track_id),
                benchmark=track_data.get("benchmark", ""),
                initial_capital=float(track_data.get("initial_capital", 100_000)),
                target_allocation_pct=float(track_data.get("target_allocation_pct", 0)),
                strategies=strategies,
            )

        logger.info(
            "Loaded %d tracks with %d total strategies from %s",
            len(tracks),
            sum(len(t.strategies) for t in tracks.values()),
            yaml_path,
        )
        return cls(tracks)

    # -- queries ---------------------------------------------------------------

    def get_track(self, slug: str) -> str:
        """Return the track (pod_id) for a given strategy slug.

        Raises ``KeyError`` if the slug is not assigned to any track.
        """
        try:
            return self._slug_to_track[slug]
        except KeyError:
            raise KeyError(
                f"Strategy slug '{slug}' not found in any track. "
                f"Known slugs: {sorted(self._slug_to_track.keys())}"
            ) from None

    def get_strategies(self, track_id: str) -> list[str]:
        """Return all strategy slugs assigned to a track.

        Raises ``KeyError`` if the track_id is not known.
        """
        if track_id not in self._tracks:
            raise KeyError(
                f"Track '{track_id}' not found. "
                f"Known tracks: {sorted(self._tracks.keys())}"
            )
        return [e.slug for e in self._tracks[track_id].strategies]

    def get_all_tracks(self) -> dict[str, list[str]]:
        """Return a mapping of all track_ids to their strategy slugs."""
        return {
            track_id: [e.slug for e in info.strategies]
            for track_id, info in self._tracks.items()
        }

    def get_track_info(self, track_id: str) -> TrackInfo:
        """Return full metadata for a track.

        Raises ``KeyError`` if the track_id is not known.
        """
        if track_id not in self._tracks:
            raise KeyError(
                f"Track '{track_id}' not found. "
                f"Known tracks: {sorted(self._tracks.keys())}"
            )
        return self._tracks[track_id]

    def get_strategy_entry(self, slug: str) -> StrategyEntry:
        """Return the full StrategyEntry for a given slug.

        Raises ``KeyError`` if the slug is not found.
        """
        track_id = self.get_track(slug)
        for entry in self._tracks[track_id].strategies:
            if entry.slug == slug:
                return entry
        # Should never reach here if _slug_to_track is consistent
        raise KeyError(
            f"Strategy slug '{slug}' index inconsistency"
        )  # pragma: no cover

    @property
    def all_slugs(self) -> list[str]:
        """Return a sorted list of all known strategy slugs."""
        return sorted(self._slug_to_track.keys())

    @property
    def track_ids(self) -> list[str]:
        """Return a sorted list of all known track IDs."""
        return sorted(self._tracks.keys())

    @property
    def track_display_order(self) -> list[str]:
        """Return track IDs in YAML insertion order (for display)."""
        return list(self._tracks.keys())

    def __repr__(self) -> str:
        counts = {tid: len(info.strategies) for tid, info in self._tracks.items()}
        return f"TrackRouter(tracks={counts})"

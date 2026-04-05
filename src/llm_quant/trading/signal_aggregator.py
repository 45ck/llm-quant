"""Signal aggregation engine for multi-track portfolio.

Reads daily paper-trading signals from YAML files, groups them by
track using TrackRouter, and produces per-track aggregated positions
using equal-weight averaging.

Usage::

    agg = SignalAggregator()
    results = agg.aggregate_all_tracks()
    for track_id, track_signal in results.items():
        print(track_id, track_signal.net_allocations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from llm_quant.trading.track_router import TrackRouter

logger = logging.getLogger(__name__)

# Default location for strategy paper-trading data
_DEFAULT_STRATEGIES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "strategies"
)

# Strategies with no signal within this many calendar days
# are considered stale and excluded from aggregation.
_STALENESS_DAYS = 5


@dataclass
class TrackSignal:
    """Aggregated signal for a single track."""

    track_id: str
    date: str
    n_active_strategies: int
    n_total_strategies: int
    net_allocations: dict[str, float]
    strategy_details: list[dict[str, Any]] = field(default_factory=list)


def _parse_date(value: Any) -> date | None:
    """Parse a date from various YAML representations.

    Accepts ``datetime.date``, ``"YYYY-MM-DD"`` strings, or
    ``datetime.datetime`` objects.  Returns ``None`` on failure.
    """
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _extract_latest_signal(
    paper_data: dict[str, Any],
    ref_date: date,
    staleness_days: int = _STALENESS_DAYS,
) -> dict[str, Any] | None:
    """Extract the most recent non-stale daily_log entry.

    Returns a dict with keys ``date``, ``allocation``, ``regime``,
    and ``signal_desc`` if a valid entry is found within
    *staleness_days* of *ref_date*.  Returns ``None`` otherwise.
    """
    daily_log = paper_data.get("daily_log")
    if not daily_log or not isinstance(daily_log, list):
        return None

    # Iterate newest-first (entries are chronologically ordered)
    for entry in reversed(daily_log):
        entry_date = _parse_date(entry.get("date"))
        if entry_date is None:
            continue

        # Check staleness
        delta = (ref_date - entry_date).days
        if delta < 0:
            # Future date -- skip
            continue
        if delta > staleness_days:
            # Too old
            return None

        # Skip entries that are pending/incomplete
        position = entry.get("position", "")
        if position in ("pending", "pending_eod"):
            continue

        # Extract allocation dict
        allocation = entry.get("allocation")
        if allocation is None:
            # Older format: no allocation dict.  We can try to
            # infer from signal field, but conservatively return
            # empty allocation to indicate flat/no-position.
            allocation = {}

        return {
            "date": str(entry_date),
            "allocation": dict(allocation),
            "regime": str(entry.get("regime", "")),
            "signal_desc": str(entry.get("signal_desc", entry.get("signal", ""))),
            "position": str(position),
        }

    return None


class SignalAggregator:
    """Aggregate paper-trading signals across all strategies.

    Parameters
    ----------
    strategies_dir:
        Path to the ``data/strategies/`` directory containing
        per-strategy subdirectories with ``paper-trading.yaml``.
    router:
        A ``TrackRouter`` instance for mapping slugs to tracks.
        If ``None``, loads from the default YAML.
    ref_date:
        The reference date for staleness checks.  Defaults to
        today.
    staleness_days:
        Maximum calendar days since the last signal before a
        strategy is considered stale.
    """

    def __init__(
        self,
        strategies_dir: Path | str | None = None,
        router: TrackRouter | None = None,
        ref_date: date | None = None,
        staleness_days: int = _STALENESS_DAYS,
    ) -> None:
        self.strategies_dir = (
            Path(strategies_dir)
            if strategies_dir is not None
            else _DEFAULT_STRATEGIES_DIR
        )
        self.router = router if router is not None else TrackRouter.load_from_yaml()
        self.ref_date = ref_date or datetime.now(tz=UTC).date()
        self.staleness_days = staleness_days

    def _load_strategy_signal(self, slug: str) -> dict[str, Any] | None:
        """Load the latest signal for a single strategy.

        Returns ``None`` if the paper-trading YAML does not exist,
        is unparseable, or has no recent signal.
        """
        yaml_path = self.strategies_dir / slug / "paper-trading.yaml"
        if not yaml_path.exists():
            logger.debug("No paper-trading.yaml for '%s' — skipping", slug)
            return None

        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                paper_data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as exc:
            logger.warning(
                "Failed to load paper-trading.yaml for '%s': %s",
                slug,
                exc,
            )
            return None

        if not isinstance(paper_data, dict):
            logger.warning("paper-trading.yaml for '%s' is not a dict", slug)
            return None

        # Check that status is active
        status = paper_data.get("status", "")
        if status not in ("active", ""):
            logger.debug(
                "Strategy '%s' status is '%s' — skipping",
                slug,
                status,
            )
            return None

        return _extract_latest_signal(
            paper_data,
            self.ref_date,
            self.staleness_days,
        )

    def aggregate_track(self, track_id: str) -> TrackSignal:
        """Aggregate signals for a single track.

        Uses equal-weight averaging across all active strategies
        in the track.
        """
        slugs = self.router.get_strategies(track_id)
        strategy_details: list[dict[str, Any]] = []
        active_allocations: list[dict[str, float]] = []

        for slug in slugs:
            signal = self._load_strategy_signal(slug)
            detail: dict[str, Any] = {
                "slug": slug,
                "active": signal is not None,
            }
            if signal is not None:
                detail.update(signal)
                alloc = signal.get("allocation", {})
                if isinstance(alloc, dict):
                    active_allocations.append(alloc)
            strategy_details.append(detail)

        # Equal-weight average across active strategies
        n_active = len(active_allocations)
        net_alloc: dict[str, float] = {}

        if n_active > 0:
            # Collect all symbols across all active strategies
            all_symbols: set[str] = set()
            for alloc in active_allocations:
                all_symbols.update(alloc.keys())

            for sym in sorted(all_symbols):
                # Skip cash — it's implicit (1 - sum of weights)
                if sym.lower() == "cash":
                    continue
                total = sum(alloc.get(sym, 0.0) for alloc in active_allocations)
                net_alloc[sym] = round(total / n_active, 6)

        return TrackSignal(
            track_id=track_id,
            date=str(self.ref_date),
            n_active_strategies=n_active,
            n_total_strategies=len(slugs),
            net_allocations=net_alloc,
            strategy_details=strategy_details,
        )

    def aggregate_all_tracks(self) -> dict[str, TrackSignal]:
        """Aggregate signals for all tracks.

        Returns a mapping of track_id to ``TrackSignal``.
        """
        results: dict[str, TrackSignal] = {}
        for track_id in self.router.track_ids:
            results[track_id] = self.aggregate_track(track_id)
        return results

    @staticmethod
    def to_json_decision(
        track_signal: TrackSignal,
        regime: str = "transition",
        regime_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Format a TrackSignal into a JSON decision dict.

        The output is compatible with
        ``scripts/execute_decision.py`` which expects::

            {
                "market_regime": "risk_on",
                "regime_confidence": 0.7,
                "regime_reasoning": "...",
                "signals": [...],
                "portfolio_commentary": "..."
            }

        Parameters
        ----------
        track_signal:
            The aggregated track signal to convert.
        regime:
            Market regime string (risk_on, risk_off, transition).
        regime_confidence:
            Confidence in the regime classification (0-1).

        Returns
        -------
        dict
            JSON-serializable decision dict.
        """
        signals: list[dict[str, Any]] = []

        for sym, weight in sorted(track_signal.net_allocations.items()):
            if weight <= 0:
                continue
            # Determine action: buy if weight > 0
            signals.append(
                {
                    "symbol": sym.upper(),
                    "action": "buy",
                    "conviction": "medium",
                    "target_weight": round(min(weight, 0.10), 4),
                    "stop_loss": 0.0,
                    "reasoning": (
                        f"Aggregated signal from "
                        f"{track_signal.n_active_strategies}"
                        f"/{track_signal.n_total_strategies}"
                        f" active strategies in "
                        f"{track_signal.track_id}"
                    ),
                }
            )

        active_slugs = [
            d["slug"] for d in track_signal.strategy_details if d.get("active")
        ]

        return {
            "market_regime": regime,
            "regime_confidence": regime_confidence,
            "regime_reasoning": (
                f"Aggregated from {track_signal.track_id}: "
                f"{track_signal.n_active_strategies}"
                f"/{track_signal.n_total_strategies}"
                f" strategies active"
            ),
            "signals": signals,
            "portfolio_commentary": (
                f"Signal aggregation for {track_signal.track_id}"
                f" on {track_signal.date}. "
                f"Active strategies: "
                f"{', '.join(active_slugs) or 'none'}."
            ),
        }

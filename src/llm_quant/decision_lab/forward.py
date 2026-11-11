"""Local append-only forecast/outcome journal, not a broker or trusted timestamp service.

Publishing the head hash to an independent timestamped system is necessary to make
later truncation/rewrite detectable. A local hash chain alone is not authenticity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .data import require

GENESIS = "0" * 64


def utc_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    require(parsed.utcoffset() is not None, "Timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_digest(value: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Expected a SHA-256 digest")


@dataclass(frozen=True)
class Forecast:
    series_id: str
    model_sha256: str
    snapshot_sha256: str
    data_through: str
    matures_at: str
    initial_level: float
    loss_threshold: float
    loss_probability: float
    max_data_age_seconds: int = 604800

    def validate(self, recorded_at: datetime) -> None:
        require(re.fullmatch(r"[A-Za-z0-9_.=-]{1,80}", self.series_id) is not None,
                "Unsafe series identifier")
        checked_digest(self.model_sha256)
        checked_digest(self.snapshot_sha256)
        require(utc_time(self.data_through) <= recorded_at < utc_time(self.matures_at),
                "Forecast must precede maturity and follow all source observations")
        require(type(self.max_data_age_seconds) is int and self.max_data_age_seconds >= 0,
                "Invalid data freshness limit")
        require((recorded_at - utc_time(self.data_through)).total_seconds()
                <= self.max_data_age_seconds, "Stale forecast data")
        require(math.isfinite(self.initial_level) and self.initial_level > 0,
                "Invalid starting level")
        require(math.isfinite(self.loss_threshold) and 0 < self.loss_threshold < 1,
                "Invalid terminal loss threshold")
        require(math.isfinite(self.loss_probability) and 0 <= self.loss_probability <= 1,
                "Invalid probability")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Reject simultaneous writers. A crashed lock is reviewed, never silently ignored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def read_journal(path: Path, expected_head: str | None = None) -> list[dict]:
    if expected_head is not None:
        checked_digest(expected_head)
    raw = path.read_bytes() if path.exists() else b""
    require(not raw or raw.endswith(b"\n"), "Truncated journal record")
    rows, previous, timestamp = [], GENESIS, None
    forecasts: dict[str, dict] = {}
    outcomes: set[str] = set()
    for index, line in enumerate(raw.splitlines()):
        envelope = json.loads(line)
        record = envelope["record"]
        digest = hashlib.sha256(canonical(record)).hexdigest()
        require(digest == envelope["sha256"] and record["previous_sha256"] == previous,
                "Journal hash chain mismatch")
        require(record["sequence"] == index and record["schema_version"] == 1,
                "Journal order/schema mismatch")
        current = utc_time(record["recorded_at"])
        require(timestamp is None or current >= timestamp, "Journal clock moved backwards")
        payload = record["payload"]
        if record["kind"] == "forecast":
            forecast = Forecast(**payload)
            forecast.validate(current)
            forecasts[digest] = payload
        elif record["kind"] == "outcome":
            _check_outcome(payload, forecasts, outcomes, current)
            outcomes.add(payload["forecast_sha256"])
        else:
            raise ValueError("Unknown journal record type")
        rows.append(envelope)
        previous, timestamp = digest, current
    require(expected_head is None or previous == expected_head,
            "Journal differs from independently retained head hash")
    return rows


def _check_outcome(payload: dict, forecasts: dict[str, dict],
                   settled: set[str], now: datetime) -> None:
    key = payload["forecast_sha256"]
    require(key in forecasts and key not in settled, "Unknown or already settled forecast")
    forecast = forecasts[key]
    require(now >= utc_time(forecast["matures_at"]), "Outcome not mature")
    require(utc_time(payload["observed_at"]) == utc_time(forecast["matures_at"]),
            "Outcome must refer to the exact preregistered maturity")
    require(math.isfinite(payload["level"]) and payload["level"] > 0, "Invalid outcome level")
    checked_digest(payload["source_sha256"])


def _append(path: Path, rows: list[dict], kind: str, payload: dict,
            now: datetime) -> str:
    previous = rows[-1]["sha256"] if rows else GENESIS
    require(not rows or now >= utc_time(rows[-1]["record"]["recorded_at"]),
            "Journal clock moved backwards")
    record = {"schema_version": 1, "sequence": len(rows), "previous_sha256": previous,
              "recorded_at": now.isoformat(), "kind": kind, "payload": payload}
    digest = hashlib.sha256(canonical(record)).hexdigest()
    line = canonical({"record": record, "sha256": digest}) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def issue_forecast(path: Path, forecast: Forecast) -> str:
    """Use current wall-clock time; there is deliberately no backdating argument."""
    with exclusive_lock(path):
        rows = read_journal(path)
        now = datetime.now(UTC)
        forecast.validate(now)
        payload = asdict(forecast)
        require(not any(row["record"]["kind"] == "forecast"
                        and row["record"]["payload"] == payload for row in rows),
                "Duplicate forecast")
        return _append(path, rows, "forecast", payload, now)


def settle_forecast(path: Path, forecast_sha256: str, *, observed_at: str,
                    level: float, source_sha256: str) -> str:
    with exclusive_lock(path):
        rows = read_journal(path)
        forecasts = {row["sha256"]: row["record"]["payload"] for row in rows
                     if row["record"]["kind"] == "forecast"}
        settled = {row["record"]["payload"]["forecast_sha256"] for row in rows
                   if row["record"]["kind"] == "outcome"}
        payload = {"forecast_sha256": forecast_sha256, "observed_at": observed_at,
                   "level": level, "source_sha256": source_sha256}
        now = datetime.now(UTC)
        _check_outcome(payload, forecasts, settled, now)
        return _append(path, rows, "outcome", payload, now)


def score_journal(path: Path, expected_head: str | None = None) -> dict:
    rows = read_journal(path, expected_head)
    forecasts = {row["sha256"]: row["record"]["payload"] for row in rows
                 if row["record"]["kind"] == "forecast"}
    scores = []
    settled = set()
    for row in rows:
        if row["record"]["kind"] != "outcome":
            continue
        outcome = row["record"]["payload"]
        forecast = forecasts[outcome["forecast_sha256"]]
        observed = int(outcome["level"] <= forecast["initial_level"] * (1 - forecast["loss_threshold"]))
        scores.append((forecast["loss_probability"] - observed) ** 2)
        settled.add(outcome["forecast_sha256"])
    now = datetime.now(UTC)
    overdue = sum(key not in settled and utc_time(value["matures_at"]) <= now
                  for key, value in forecasts.items())
    return {
        "status": "LOCAL_FORWARD_RECORD_NOT_INDEPENDENTLY_ATTESTED",
        "forecasts": len(forecasts), "scored_outcomes": len(scores),
        "mature_unscored_forecasts": overdue,
        "brier_score": None if not scores else sum(scores) / len(scores),
        "head_sha256": rows[-1]["sha256"] if rows else GENESIS,
        "limitations": ["Local clock and metadata are not externally authenticated.",
                        "Retain the head hash independently to detect truncation or rewrite.",
                        "Missing mature outcomes must not be discarded to improve scores.",
                        "Hash validity establishes neither data accuracy nor an investment edge."],
    }

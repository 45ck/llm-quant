"""Immutable source snapshots and fail-closed, explicit data contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    """Raise a validation error rather than silently repairing financial data."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Contract:
    series_id: str
    provider: str
    source_url: str
    terms_url: str
    geography: str
    currency: str
    frequency: str
    basis: str
    retrieved_at: str
    vintage: str
    evidence: str
    raw_sha256: str

    def validate(self) -> None:
        require(
            re.fullmatch(r"[A-Za-z0-9_.=-]{1,80}", self.series_id) is not None,
            "Unsafe or empty series identifier",
        )
        require(self.frequency in {"D", "M", "Q"}, "Unsupported native frequency")
        require(
            self.basis in {
                "adjusted_close_proxy", "total_return", "nominal_price_index",
                "real_price_index", "fx_quote",
            },
            "An explicit supported value basis is required",
        )
        require(self.vintage in {"latest", "synthetic"}, "Unsupported vintage")
        require(self.evidence in {"observed", "synthetic"}, "Unsupported evidence")
        require(
            (self.vintage == "synthetic") == (self.evidence == "synthetic"),
            "Synthetic evidence and vintage must agree",
        )
        timestamp = datetime.fromisoformat(self.retrieved_at)
        require(timestamp.utcoffset() is not None, "Retrieval must be timezone-aware")
        require(bool(self.currency and self.geography and self.provider), "Empty metadata")
        require(
            self.source_url.startswith("https://")
            and self.terms_url.startswith("https://"),
            "HTTPS source and terms references are required",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", self.raw_sha256) is not None,
            "Invalid raw source hash",
        )


@dataclass(frozen=True)
class Snapshot:
    contract: Contract
    dates: tuple[str, ...]
    values: tuple[float, ...]

    def validate(self) -> None:
        self.contract.validate()
        require(len(self.dates) == len(self.values) >= 2, "Need matching observations")
        dates = tuple(date.fromisoformat(item) for item in self.dates)
        require(all(a < b for a, b in zip(dates[:-1], dates[1:], strict=True)), "Unsorted/duplicate dates")
        require(
            all(math.isfinite(value) and value > 0 for value in self.values),
            "Levels must be finite and strictly positive; missing values are not filled",
        )
        if self.contract.frequency in {"M", "Q"}:
            months = [item.year * 12 + item.month for item in dates]
            step = 1 if self.contract.frequency == "M" else 3
            require(
                all(b - a == step for a, b in zip(months[:-1], months[1:], strict=True)),
                "Missing or irregular native-frequency periods",
            )

    def check_as_of(self, as_of: date, max_age_days: int) -> None:
        self.validate()
        require(max_age_days >= 0, "Maximum data age must be nonnegative")
        last = date.fromisoformat(self.dates[-1])
        fetched = datetime.fromisoformat(self.contract.retrieved_at).date()
        require(last <= as_of, "Future observations are not permitted")
        require(fetched <= as_of, "Latest-vintage data was retrieved after this origin")
        require((as_of - last).days <= max_age_days, "Stale data: refuse current forecast")

    def payload(self) -> dict:
        self.validate()
        return {
            "contract": asdict(self.contract),
            "dates": list(self.dates),
            "values": list(self.values),
        }

    def digest(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()


def save_snapshot(snapshot: Snapshot, directory: Path) -> Path:
    """Content-addressed snapshots never overwrite a different observation vintage."""
    digest = snapshot.digest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{snapshot.contract.series_id}-{digest}.json"
    envelope = {"schema_version": 1, "sha256": digest, "snapshot": snapshot.payload()}
    if not target.exists():
        with target.open("x", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True, indent=2, allow_nan=False)
    else:
        require(load_snapshot(target).digest() == digest, "Existing snapshot is corrupt")
    return target


def load_snapshot(path: Path) -> Snapshot:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    require(envelope.get("schema_version") == 1, "Unsupported snapshot version")
    payload = envelope["snapshot"]
    result = Snapshot(
        Contract(**payload["contract"]), tuple(payload["dates"]),
        tuple(float(value) for value in payload["values"]),
    )
    require(result.digest() == envelope["sha256"], "Snapshot hash mismatch")
    return result


def aligned_returns(snapshots: list[Snapshot], *, portfolio: bool = True) -> np.ndarray:
    """No implicit FX, date joins, filling, resampling, or index splicing."""
    require(bool(snapshots), "At least one snapshot is required")
    for snapshot in snapshots:
        snapshot.validate()
    reference = snapshots[0]
    for snapshot in snapshots:
        require(snapshot.dates == reference.dates, "Dates differ; align explicitly")
        for field in ("currency", "frequency", "basis", "evidence"):
            require(
                getattr(snapshot.contract, field) == getattr(reference.contract, field),
                f"Mixed {field}; transform explicitly with provenance",
            )
    if portfolio:
        require(
            reference.contract.basis in {"adjusted_close_proxy", "total_return"},
            "Housing/FX price indices cannot be passed off as investable total returns",
        )
    else:
        require(len(snapshots) == 1, "Indicator mode accepts exactly one series")
    levels = np.column_stack([snapshot.values for snapshot in snapshots])
    return levels[1:] / levels[:-1] - 1.0


def usd_to_aud(usd_levels: np.ndarray, usd_per_aud: np.ndarray) -> np.ndarray:
    """P_AUD = P_USD / (USD per AUD); caller must align timestamps explicitly."""
    prices = np.asarray(usd_levels, dtype=float)
    quotes = np.asarray(usd_per_aud, dtype=float)
    require(prices.shape == quotes.shape, "FX and asset shapes differ")
    require(
        bool(np.all(np.isfinite(prices) & (prices > 0))), "Invalid USD levels",
    )
    require(
        bool(np.all(np.isfinite(quotes) & (quotes > 0))), "Invalid FX quotes",
    )
    return prices / quotes

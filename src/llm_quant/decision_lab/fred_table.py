"""Strict public-table ingestion, plus declared browser-export imports.

Browser exports are not original HTTP bodies. A receipt binds the exported bytes
and the extraction metadata; it is provenance, not a provider signature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from .data import Contract, Snapshot, require, save_snapshot

SERIES_BASIS = {
    "QAUR628BIS": "real_price_index",
    "QAUN628BIS": "nominal_price_index",
}
MAX_BYTES = 10_000_000


class _TableRows(HTMLParser):
    """Extract table cells, never scripts, generated numbers or arbitrary text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden:
            if tag == "tr":
                self.row = []
            elif tag in {"td", "th"} and self.row is not None:
                self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None and not self.hidden:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag in {"td", "th"} and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
            self.cell = None


@dataclass(frozen=True)
class TableData:
    dates: tuple[str, ...]
    values: tuple[float, ...]
    updated: str


def parse_table(raw: bytes, series_id: str) -> TableData:
    require(series_id in SERIES_BASIS, "FRED series is not allowlisted")
    require(0 < len(raw) <= MAX_BYTES, "Invalid source response size")
    parser = _TableRows()
    parser.feed(raw.decode("utf-8-sig"))
    metadata: dict[str, str] = {}
    dates: list[str] = []
    values: list[float] = []
    date_row = re.compile(r"\d{4}-\d{2}-\d{2}")
    for row in parser.rows:
        if not row:
            continue
        if date_row.fullmatch(row[0]):
            require(len(row) == 2, "Unexpected number of observation columns")
            require(row[1] not in {"", ".", "NA", "N/A"}, "Missing FRED value")
            dates.append(row[0])
            values.append(float(row[1]))
        elif len(row) == 2:
            key = row[0].rstrip(":").strip()
            require(key not in metadata, "Duplicate FRED metadata field")
            metadata[key] = row[1]
    require(metadata.get("Series ID") == series_id, "FRED series metadata mismatch")
    require(metadata.get("Frequency") == "Quarterly", "Unexpected FRED frequency")
    require(metadata.get("Units") == "Index 2010=100", "Unexpected FRED index basis")
    require(metadata.get("Source") == "Bank for International Settlements", "Unexpected source")
    require(bool(metadata.get("Last Updated")), "Missing FRED update metadata")
    require(len(dates) >= 2, "FRED table contains no usable observation history")
    require(
        metadata.get("Date Range") == f"{dates[0]} to {dates[-1]}",
        "FRED table range does not match metadata",
    )
    return TableData(tuple(dates), tuple(values), metadata["Last Updated"])


def import_table_export(csv_path: Path, receipt_path: Path, directory: Path) -> Path:
    """Import a complete, explicitly declared FRED browser table export.

    Does not download or certify the authenticity of the extraction. Validate
    independently that the receipt actually describes the source you inspected.
    """
    from .providers import parse_fred, store_raw

    require(csv_path.stat().st_size <= MAX_BYTES, "Export exceeds import budget")
    require(receipt_path.stat().st_size <= 64_000, "Receipt exceeds import budget")
    raw = csv_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(isinstance(receipt, dict), "Receipt must be an object")
    series_id = receipt.get("series_id")
    require(series_id in SERIES_BASIS, "FRED series is not allowlisted")
    require(receipt.get("acquisition") == "browser_table_export", "Declare acquisition")
    require(receipt.get("source_url") == f"https://fred.stlouisfed.org/data/{series_id}",
            "Receipt source URL mismatch")
    require(receipt.get("units") == "Index 2010=100", "Receipt units mismatch")
    require(receipt.get("frequency") == "Quarterly", "Receipt frequency mismatch")
    require(receipt.get("scope") == "complete_series", "Partial exports are not accepted")
    updated = receipt.get("source_last_updated")
    require(isinstance(updated, str) and bool(updated.strip()), "Missing source update")
    captured = datetime.fromisoformat(receipt["captured_at"])
    require(captured.utcoffset() is not None, "Capture timestamp requires timezone")
    require(captured <= datetime.now(UTC), "Future capture timestamp")
    digest = hashlib.sha256(raw).hexdigest()
    require(receipt.get("export_sha256") == digest, "Export hash mismatch")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    require(reader.fieldnames == ["observation_date", series_id], "Export schema mismatch")
    dates, values = parse_fred(raw, series_id)
    require(receipt.get("row_count") == len(dates), "Export row count mismatch")
    require(dates and dates[0] == receipt.get("first_date"), "Export first date mismatch")
    require(dates[-1] == receipt.get("last_date"), "Export last date mismatch")
    require(values[0] == receipt.get("first_value"), "Export first value mismatch")
    require(values[-1] == receipt.get("last_value"), "Export last value mismatch")
    contract = Contract(
        series_id=series_id,
        provider=f"BIS via FRED browser table export; source update {updated}",
        source_url=receipt["source_url"],
        terms_url="https://www.bis.org/terms_conditions.htm",
        geography="Australia: eight large cities; not a Sydney-specific series",
        currency="INDEX",
        frequency="Q",
        basis=SERIES_BASIS[series_id],
        retrieved_at=receipt["captured_at"],
        vintage="latest",
        evidence="observed",
        raw_sha256=digest,
    )
    snapshot = Snapshot(contract, dates, values)
    snapshot.validate()
    store_raw(raw, directory / "raw")
    receipt_bytes = json.dumps(receipt, sort_keys=True, allow_nan=False).encode()
    store_raw(receipt_bytes, directory / "receipts")
    return save_snapshot(snapshot, directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--directory", type=Path,
                        default=Path("research-local/decision-lab/snapshots"))
    args = parser.parse_args()
    try:
        path = import_table_export(args.csv, args.receipt, args.directory)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Import rejected: {error}\n")
    sys.stdout.write(f"Snapshot: {path}\nAcquisition: declared browser table export\n")


if __name__ == "__main__":
    main()

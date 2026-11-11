"""Opt-in free research adapters; no credentials, trades, or silent fallbacks."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from .data import Contract, Snapshot, require, save_snapshot
from .fred_table import parse_table

FRED_SERIES = {
    "QAUR628BIS": "real_price_index",
    "QAUN628BIS": "nominal_price_index",
}
ASX_SYMBOLS = frozenset({"VAS.AX", "VGS.AX", "NDQ.AX", "VAF.AX", "VAP.AX", "GOLD.AX"})
MAX_DOWNLOAD_BYTES = 10_000_000


def parse_fred(raw: bytes, series_id: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    date_column = "observation_date" if "observation_date" in fields else "DATE"
    require(date_column in fields and series_id in fields, "Unexpected FRED CSV schema")
    dates, values = [], []
    for row in reader:
        require(row[series_id] not in {"", "."}, "Missing FRED value; no filling allowed")
        dates.append(date.fromisoformat(row[date_column]).isoformat())
        values.append(float(row[series_id]))
    return tuple(dates), tuple(values)


def store_raw(raw: bytes, directory: Path) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.raw"
    if not target.exists():
        with target.open("xb") as handle:
            handle.write(raw)
    else:
        require(hashlib.sha256(target.read_bytes()).hexdigest() == digest, "Corrupt raw cache")
    return digest


def fetch_fred(series_id: str, directory: Path) -> Path:
    require(series_id in FRED_SERIES, "FRED series is not allowlisted")
    url = f"https://fred.stlouisfed.org/data/{series_id}"
    # URL is fixed HTTPS + an allowlisted identifier; arbitrary URLs are not accepted.
    with urlopen(url, timeout=30) as response:  # noqa: S310 -- fixed HTTPS allowlist above
        destination = urlsplit(response.geturl())
        require(
            destination.scheme == "https" and destination.hostname == "fred.stlouisfed.org",
            "Unexpected provider redirect",
        )
        raw = response.read(MAX_DOWNLOAD_BYTES + 1)
    require(len(raw) <= MAX_DOWNLOAD_BYTES, "Provider response exceeds download budget")
    table = parse_table(raw, series_id)
    contract = Contract(
        series_id=series_id,
        provider=f"BIS via FRED HTML table; source update {table.updated}",
        source_url=url,
        terms_url="https://www.bis.org/terms_conditions.htm",
        geography="Australia: eight large cities; not a Sydney-specific series",
        currency="INDEX", frequency="Q", basis=FRED_SERIES[series_id],
        retrieved_at=datetime.now(UTC).isoformat(), vintage="latest", evidence="observed",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    snapshot = Snapshot(contract, table.dates, table.values)
    snapshot.validate()
    store_raw(raw, directory / "raw")
    return save_snapshot(snapshot, directory)


def fetch_yahoo(symbol: str, start: str, end: str, directory: Path) -> Path:
    """End is exclusive. Optional yfinance dependency is already in the parent repo."""
    require(symbol in ASX_SYMBOLS, "Only explicitly registered AUD ASX ETFs are accepted")
    require(date.fromisoformat(start) < date.fromisoformat(end), "Invalid date interval")
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    frame = ticker.history(start=start, end=end, auto_adjust=False, actions=False,
                           keepna=True, repair=False, raise_errors=True, timeout=20)
    require(not frame.empty and "Adj Close" in frame.columns, "No adjusted-close evidence")
    metadata = ticker.get_history_metadata()
    require(metadata.get("currency") == "AUD", "Provider did not confirm AUD currency")
    require(frame.index.is_unique, "Duplicate provider timestamps")
    dates = tuple(item.date().isoformat() for item in frame.index)
    values = tuple(float(value) for value in frame["Adj Close"])
    # This is a provider-frame export, not the raw HTTP response body.
    raw = frame.to_csv().encode("utf-8")
    contract = Contract(
        series_id=symbol, provider="Yahoo via yfinance; provider-frame CSV export",
        source_url=f"https://finance.yahoo.com/quote/{symbol}/history/",
        terms_url="https://ranaroussi.github.io/yfinance/",
        geography="ASX-listed instrument; underlying exposure varies by fund",
        currency="AUD", frequency="D", basis="adjusted_close_proxy",
        retrieved_at=datetime.now(UTC).isoformat(), vintage="latest", evidence="observed",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    snapshot = Snapshot(contract, dates, values)
    snapshot.validate()
    store_raw(raw, directory / "raw")
    return save_snapshot(snapshot, directory)

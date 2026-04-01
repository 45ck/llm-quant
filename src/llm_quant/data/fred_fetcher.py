"""Fetch macroeconomic data from FRED (Federal Reserve Economic Data).

Supports two backends:
  1. ``fredapi`` — direct FRED API access (requires FRED_API_KEY env var).
  2. yfinance fallback — fetches FRED series that Yahoo Finance proxies,
     used when fredapi is unavailable or no API key is configured.

Data is stored in the ``fred_data_daily`` DuckDB table alongside
yfinance market data, and cached locally with weekly refresh.

Series fetched by default (Family 6 — Macro Regime):
  T10Y2Y  — 10-Year minus 2-Year Treasury yield spread (yield curve slope)
  UNRATE  — Civilian unemployment rate (monthly → forward-filled to daily)
  CPIAUCSL — CPI All Urban Consumers (monthly → forward-filled to daily)
  T10YIE  — 10-Year Breakeven Inflation Rate (daily)

yfinance proxy map (fallback when FRED_API_KEY not set):
  T10Y2Y  → ^TNX minus ^IRX approximation is unavailable directly;
             use "T10Y2Y=X" on Yahoo Finance (not always reliable).
             Better proxies: DGS10 and DGS2 via direct FRED download.
  Fallback uses FRED CSV download endpoint (no API key required).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series configuration
# ---------------------------------------------------------------------------

#: Core FRED series for Family 6 Macro Regime research.
#: Each entry: (series_id, description, frequency, fill_method)
FRED_SERIES: dict[str, dict[str, str]] = {
    "T10Y2Y": {
        "description": "10-Year minus 2-Year Treasury Yield Spread",
        "frequency": "daily",
        "fill_method": "forward",
    },
    "UNRATE": {
        "description": "Civilian Unemployment Rate",
        "frequency": "monthly",
        "fill_method": "forward",
    },
    "CPIAUCSL": {
        "description": "CPI All Urban Consumers, Seasonally Adjusted",
        "frequency": "monthly",
        "fill_method": "forward",
    },
    "T10YIE": {
        "description": "10-Year Breakeven Inflation Rate",
        "frequency": "daily",
        "fill_method": "forward",
    },
}

#: FRED series available via direct CSV (no API key required).
#: URL pattern: https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
FRED_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

#: Cache refresh threshold: re-fetch if data older than this many days.
CACHE_REFRESH_DAYS = 7

# ---------------------------------------------------------------------------
# DDL for fred_data_daily table
# ---------------------------------------------------------------------------

FRED_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS fred_data_daily (
    series_id VARCHAR NOT NULL,
    date      DATE    NOT NULL,
    value     DOUBLE,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series_id, date)
)
"""


def ensure_fred_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create ``fred_data_daily`` table if it does not exist."""
    conn.execute(FRED_TABLE_DDL)
    conn.commit()
    logger.debug("fred_data_daily table ensured")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class FredFetcher:
    """Fetch and cache FRED macroeconomic data.

    Usage
    -----
    >>> fetcher = FredFetcher(conn, cache_dir=Path("data/fred_cache"))
    >>> df = fetcher.fetch(["T10Y2Y", "UNRATE"])
    >>> df_from_db = fetcher.get("T10Y2Y", start_date="2020-01-01")

    Parameters
    ----------
    conn:
        Open DuckDB connection with schema initialised.
    cache_dir:
        Optional directory for local CSV cache files.  Defaults to
        ``data/fred_cache`` relative to the project root.
    api_key:
        FRED API key.  When *None*, falls back to ``FRED_API_KEY``
        environment variable, then to the public CSV download endpoint
        (no key required, but rate-limited).
    refresh_days:
        Number of days before cached data is considered stale.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        refresh_days: int = CACHE_REFRESH_DAYS,
    ) -> None:
        self._conn = conn
        self._api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._refresh_days = refresh_days
        self._cache_dir = cache_dir or _default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        ensure_fred_table(conn)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch(
        self,
        series_ids: list[str] | None = None,
        start_date: str | date | None = None,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Fetch FRED series, store in DuckDB, and return as Polars DataFrame.

        Parameters
        ----------
        series_ids:
            List of FRED series IDs to fetch.  Defaults to
            :data:`FRED_SERIES` keys (all configured series).
        start_date:
            Earliest date to fetch.  Defaults to 10 years ago.
        force_refresh:
            When *True*, bypass the cache age check and always re-fetch.

        Returns
        -------
        pl.DataFrame
            Columns: ``series_id, date, value``.  Sorted by
            ``(series_id, date)``.  Monthly series are forward-filled
            to daily frequency.
        """
        if series_ids is None:
            series_ids = list(FRED_SERIES.keys())

        if start_date is None:
            start_date = datetime.now(tz=UTC).date() - timedelta(days=10 * 365)
        elif isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)

        frames: list[pl.DataFrame] = []
        for sid in series_ids:
            try:
                df = self._fetch_series(sid, start_date, force_refresh)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception:
                logger.exception("Failed to fetch FRED series %s", sid)

        if not frames:
            logger.warning("No FRED data fetched for series: %s", series_ids)
            return pl.DataFrame(
                schema={"series_id": pl.Utf8, "date": pl.Date, "value": pl.Float64}
            )

        return pl.concat(frames, how="vertical").sort(["series_id", "date"])

    def get(
        self,
        series_id: str,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> pl.DataFrame:
        """Read a FRED series from DuckDB (must have been fetched first).

        Parameters
        ----------
        series_id:
            FRED series ID (e.g. ``"T10Y2Y"``).
        start_date:
            Earliest date (inclusive).
        end_date:
            Latest date (inclusive).  Defaults to today.

        Returns
        -------
        pl.DataFrame
            Columns: ``series_id, date, value``.
        """
        ensure_fred_table(self._conn)

        if end_date is None:
            end_date = datetime.now(tz=UTC).date()
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        params: list[Any] = [series_id]
        query = "SELECT series_id, date, value FROM fred_data_daily WHERE series_id = ?"
        if start_date is not None:
            query += " AND date >= ?"
            params.append(str(start_date))
        if end_date is not None:
            query += " AND date <= ?"
            params.append(str(end_date))
        query += " ORDER BY date"

        return self._conn.execute(query, params).pl()

    def get_latest_date(self, series_id: str) -> date | None:
        """Return the most recent date stored for *series_id*, or None."""
        ensure_fred_table(self._conn)
        result = self._conn.execute(
            "SELECT MAX(date) FROM fred_data_daily WHERE series_id = ?",
            [series_id],
        ).fetchone()
        if result is None or result[0] is None:
            return None
        val = result[0]
        if isinstance(val, str):
            return date.fromisoformat(val)
        if hasattr(val, "date"):
            return val.date()
        return val  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_series(
        self,
        series_id: str,
        start_date: date,
        force_refresh: bool,
    ) -> pl.DataFrame | None:
        """Fetch a single series, using cache if fresh enough."""
        # Check if DB cache is fresh
        if not force_refresh:
            latest = self.get_latest_date(series_id)
            if latest is not None:
                age_days = (datetime.now(tz=UTC).date() - latest).days
                if age_days <= self._refresh_days:
                    logger.info(
                        "FRED %s: cache is %d days old — skipping fetch",
                        series_id,
                        age_days,
                    )
                    return self.get(series_id, start_date=start_date)

        # Choose fetch backend
        if self._api_key:
            df = self._fetch_via_fredapi(series_id, start_date)
        else:
            logger.info(
                "FRED_API_KEY not set — using public CSV endpoint for %s", series_id
            )
            df = self._fetch_via_csv(series_id, start_date)

        if df is None or len(df) == 0:
            logger.warning("No data returned for FRED series %s", series_id)
            return None

        # Forward-fill monthly/low-frequency series to daily
        cfg = FRED_SERIES.get(series_id, {})
        if cfg.get("frequency") == "monthly" or cfg.get("fill_method") == "forward":
            df = _forward_fill_to_daily(df, series_id)

        # Upsert into DuckDB
        self._upsert(df)
        logger.info("Fetched and stored %d rows for FRED series %s", len(df), series_id)
        return df

    def _fetch_via_fredapi(
        self, series_id: str, start_date: date
    ) -> pl.DataFrame | None:
        """Fetch via fredapi library (requires FRED_API_KEY)."""
        try:
            from fredapi import Fred  # type: ignore[import]
        except ImportError:
            logger.error("fredapi not installed — run: pip install fredapi")
            return None

        try:
            fred = Fred(api_key=self._api_key)
            series = fred.get_series(
                series_id,
                observation_start=str(start_date),
            )
        except Exception:
            logger.exception("fredapi failed for series %s", series_id)
            return None

        if series is None or len(series) == 0:
            return None

        # Convert pandas Series to Polars DataFrame
        pdf = series.reset_index()
        pdf.columns = ["date", "value"]
        pdf = pdf.dropna(subset=["value"])
        pdf["series_id"] = series_id

        df = pl.from_pandas(pdf).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
            pl.col("series_id").cast(pl.Utf8),
        )
        return df.select(["series_id", "date", "value"])

    def _fetch_via_csv(self, series_id: str, start_date: date) -> pl.DataFrame | None:
        """Fetch via FRED public CSV endpoint (no API key required).

        Downloads the full series CSV and filters to start_date.
        This endpoint is rate-limited — use fredapi for production.
        """
        import urllib.request
        from io import StringIO

        url = f"{FRED_CSV_BASE_URL}{series_id}"
        logger.info("Downloading FRED CSV for %s from %s", series_id, url)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "llm-quant/1.0 (research; non-commercial)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")
        except Exception:
            logger.exception(
                "HTTP download failed for FRED series %s (url=%s)", series_id, url
            )
            return None

        try:
            df = pl.read_csv(
                StringIO(content),
                schema_overrides={"DATE": pl.Utf8, "VALUE": pl.Utf8},
            )
        except Exception:
            logger.exception("CSV parse failed for FRED series %s", series_id)
            return None

        # Normalise column names (FRED CSV uses DATE / VALUE or date / value)
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(col_map)

        if "date" not in df.columns or "value" not in df.columns:
            logger.error(
                "Unexpected FRED CSV columns for %s: %s", series_id, df.columns
            )
            return None

        # Filter out "." (missing value marker used by FRED)
        df = df.filter(pl.col("value") != ".")

        df = df.with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("value").cast(pl.Float64),
            pl.lit(series_id).alias("series_id"),
        )

        # Filter to start_date
        df = df.filter(pl.col("date") >= start_date)

        return df.select(["series_id", "date", "value"])

    def _upsert(self, df: pl.DataFrame) -> None:
        """Insert or replace rows in fred_data_daily."""
        if len(df) == 0:
            return

        # Add fetched_at timestamp
        df = df.with_columns(
            pl.lit(datetime.now(tz=UTC).replace(tzinfo=None)).alias("fetched_at")
        )

        rows = df.select(["series_id", "date", "value", "fetched_at"]).rows()
        self._conn.executemany(
            "INSERT OR REPLACE INTO fred_data_daily "
            "(series_id, date, value, fetched_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _forward_fill_to_daily(df: pl.DataFrame, series_id: str) -> pl.DataFrame:
    """Expand a low-frequency (monthly/weekly) FRED series to daily frequency.

    Creates a continuous date range from the first to last observation date,
    then forward-fills the value for each day.

    Parameters
    ----------
    df:
        Polars DataFrame with columns ``series_id, date, value``.
    series_id:
        The series ID (for logging).

    Returns
    -------
    pl.DataFrame
        Daily-frequency DataFrame with the same columns.
    """
    if len(df) == 0:
        return df

    min_date = df["date"].min()
    max_date = df["date"].max()

    if min_date is None or max_date is None:
        return df

    # Build a complete daily date spine
    all_dates = pl.date_range(min_date, max_date, interval="1d", eager=True)
    spine = pl.DataFrame({"date": all_dates})

    # Join observations onto the date spine and forward-fill
    filled = (
        spine.join(
            df.select(["date", "value"]),
            on="date",
            how="left",
        )
        .with_columns(pl.col("value").forward_fill())
        .with_columns(pl.lit(series_id).alias("series_id"))
        .select(["series_id", "date", "value"])
        .filter(pl.col("value").is_not_null())
    )

    logger.debug(
        "Forward-filled FRED %s: %d observations → %d daily rows",
        series_id,
        len(df),
        len(filled),
    )
    return filled


def _default_cache_dir() -> Path:
    """Return the default FRED cache directory (data/fred_cache/ from project root)."""
    # Walk up from this file to find data/
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "data" / "fred_cache"
        if (current / "data").is_dir():
            return candidate
        current = current.parent
    # Fallback
    return Path.cwd() / "data" / "fred_cache"

"""Timestamped market-data recorder for Polymarket research.

Records point-in-time snapshots of all active Polymarket markets into
DuckDB using the normalized experiment schema (n_experiment_snapshots +
n_experiment_market_states).  Each call to record_snapshot() fetches the
current market universe via GammaClient, parses every market, and stores
a single atomic snapshot that can be replayed for reconstruction research.

Usage:
    from llm_quant.arb.research.recorder import MarketDataRecorder

    recorder = MarketDataRecorder(db_path="data/pm_research.duckdb")
    snapshot_id = recorder.record_snapshot()
    print(recorder.get_snapshot(snapshot_id))
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from llm_quant.arb.gamma_client import GammaClient, Market

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — normalized experiment tables (from normalized-schemas.md, Schema 8)
# ---------------------------------------------------------------------------

_EXPERIMENT_SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS n_experiment_snapshots (
    snapshot_id         VARCHAR PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    source              VARCHAR NOT NULL,
    active_market_count INTEGER DEFAULT 0,
    total_volume_24h    DOUBLE DEFAULT 0.0,
    scan_duration_ms    INTEGER,
    data_quality        VARCHAR DEFAULT 'unknown',
    notes               VARCHAR
);
"""

_EXPERIMENT_SNAPSHOTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_n_exp_snapshots_ts
    ON n_experiment_snapshots(timestamp);
"""

_EXPERIMENT_MARKET_STATES_DDL = """
CREATE TABLE IF NOT EXISTS n_experiment_market_states (
    snapshot_id         VARCHAR NOT NULL
                        REFERENCES n_experiment_snapshots(snapshot_id),
    market_id           VARCHAR NOT NULL,
    condition_id        VARCHAR,
    question            VARCHAR NOT NULL,
    category            VARCHAR,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    is_negrisk          BOOLEAN NOT NULL DEFAULT FALSE,
    yes_price           DOUBLE NOT NULL,
    no_price            DOUBLE NOT NULL,
    spread              DOUBLE NOT NULL,
    volume_24h          DOUBLE DEFAULT 0.0,
    open_interest       DOUBLE DEFAULT 0.0,
    best_bid_yes        DOUBLE,
    best_ask_yes        DOUBLE,
    best_bid_no         DOUBLE,
    best_ask_no         DOUBLE,
    days_to_resolution  DOUBLE,
    fee_rate            DOUBLE,
    PRIMARY KEY (snapshot_id, market_id)
);
"""


def _init_experiment_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create normalized experiment tables if they don't exist."""
    conn.execute(_EXPERIMENT_SNAPSHOTS_DDL)
    conn.execute(_EXPERIMENT_SNAPSHOTS_INDEX)
    conn.execute(_EXPERIMENT_MARKET_STATES_DDL)


def _generate_snapshot_id(source: str, ts: datetime) -> str:
    """Generate a canonical snapshot ID: {source}_{iso}_{short_uuid}."""
    iso = ts.strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{source}_{iso}_{short}"


def _compute_days_to_resolution(end_date: str | None, now: datetime) -> float | None:
    """Parse end_date string and return days remaining, or None."""
    if not end_date:
        return None
    try:
        # Gamma API returns ISO-8601 strings
        if "T" in end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        delta = (end_dt - now).total_seconds() / 86400.0
        return max(delta, 0.0)
    except (ValueError, TypeError):
        return None


class MarketDataRecorder:
    """Records timestamped Polymarket snapshots into DuckDB.

    Uses GammaClient to fetch the current market universe and stores
    each snapshot atomically in the n_experiment_snapshots /
    n_experiment_market_states tables.

    Parameters
    ----------
    db_path : str | Path
        Path to the DuckDB database file.
    client : GammaClient | None
        Pre-configured GammaClient instance.  If None, a default client
        is created.
    """

    def __init__(
        self,
        db_path: str | Path = "data/pm_research.duckdb",
        client: GammaClient | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = client or GammaClient()
        self._conn = duckdb.connect(str(self._db_path))
        _init_experiment_schema(self._conn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_snapshot(self, source: str = "live_scan") -> str:
        """Fetch current market state and store a timestamped snapshot.

        Parameters
        ----------
        source : str
            How the snapshot was created.  Typically ``"live_scan"`` for
            real API calls or ``"synthetic"`` for tests.

        Returns
        -------
        str
            The snapshot_id of the newly stored snapshot.
        """
        now = datetime.now(tz=UTC)
        snapshot_id = _generate_snapshot_id(source, now)

        # -- Fetch --
        t0 = time.monotonic()
        try:
            raw_markets = self._client.fetch_all_active_markets()
        except Exception:
            logger.exception("API fetch failed during snapshot recording")
            # Store an empty snapshot so the failure is visible in history
            self._store_snapshot(
                snapshot_id=snapshot_id,
                ts=now,
                source=source,
                markets=[],
                scan_ms=0,
                data_quality="degraded",
                notes="API fetch failed",
            )
            return snapshot_id

        parsed: list[Market] = self._client.parse_all_markets(raw_markets)
        scan_ms = int((time.monotonic() - t0) * 1000)

        # -- Determine data quality --
        if not parsed:
            quality = "degraded"
        elif len(parsed) < 10:
            quality = "partial"
        else:
            quality = "complete"

        # -- Store --
        self._store_snapshot(
            snapshot_id=snapshot_id,
            ts=now,
            source=source,
            markets=parsed,
            scan_ms=scan_ms,
            data_quality=quality,
        )

        logger.info(
            "Recorded snapshot %s — %d markets, quality=%s, %d ms",
            snapshot_id,
            len(parsed),
            quality,
            scan_ms,
        )
        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> dict | None:
        """Retrieve a stored snapshot header and its market states.

        Returns
        -------
        dict | None
            A dict with keys ``"header"`` (snapshot metadata) and
            ``"markets"`` (list of market state dicts), or None if the
            snapshot_id does not exist.
        """
        header = self._conn.execute(
            "SELECT * FROM n_experiment_snapshots WHERE snapshot_id = ?",
            [snapshot_id],
        ).fetchone()
        if header is None:
            return None

        columns = [
            desc[0]
            for desc in self._conn.execute(
                "SELECT * FROM n_experiment_snapshots LIMIT 0"
            ).description
        ]
        header_dict = dict(zip(columns, header, strict=True))

        markets_raw = self._conn.execute(
            "SELECT * FROM n_experiment_market_states WHERE snapshot_id = ? "
            "ORDER BY market_id",
            [snapshot_id],
        ).fetchall()
        market_cols = [
            desc[0]
            for desc in self._conn.execute(
                "SELECT * FROM n_experiment_market_states LIMIT 0"
            ).description
        ]
        markets_list = [dict(zip(market_cols, row, strict=True)) for row in markets_raw]

        return {"header": header_dict, "markets": markets_list}

    def list_snapshots(self, limit: int = 20) -> list[dict]:
        """Return recent snapshot headers, newest first.

        Parameters
        ----------
        limit : int
            Maximum number of snapshots to return.

        Returns
        -------
        list[dict]
            List of snapshot header dicts.
        """
        rows = self._conn.execute(
            "SELECT * FROM n_experiment_snapshots ORDER BY timestamp DESC LIMIT ?",
            [limit],
        ).fetchall()
        columns = [
            desc[0]
            for desc in self._conn.execute(
                "SELECT * FROM n_experiment_snapshots LIMIT 0"
            ).description
        ]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def snapshot_count(self) -> int:
        """Return total number of stored snapshots."""
        result = self._conn.execute(
            "SELECT COUNT(*) FROM n_experiment_snapshots"
        ).fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store_snapshot(
        self,
        *,
        snapshot_id: str,
        ts: datetime,
        source: str,
        markets: list[Market],
        scan_ms: int,
        data_quality: str,
        notes: str | None = None,
    ) -> None:
        """Atomically write snapshot header + market states."""
        total_vol = 0.0
        active_count = 0

        for m in markets:
            if m.active:
                active_count += 1
            for c in m.conditions:
                total_vol += c.volume_24h

        # -- Insert header --
        self._conn.execute(
            "INSERT INTO n_experiment_snapshots "
            "(snapshot_id, timestamp, source, active_market_count, "
            " total_volume_24h, scan_duration_ms, data_quality, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                snapshot_id,
                ts,
                source,
                active_count,
                total_vol,
                scan_ms,
                data_quality,
                notes,
            ],
        )

        # -- Insert per-market states --
        for m in markets:
            if not m.conditions:
                continue
            cond = m.conditions[0]
            days_to_res = _compute_days_to_resolution(m.end_date, ts)

            self._conn.execute(
                "INSERT INTO n_experiment_market_states "
                "(snapshot_id, market_id, condition_id, question, category, "
                " active, is_negrisk, yes_price, no_price, spread, "
                " volume_24h, open_interest, days_to_resolution) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    snapshot_id,
                    m.market_id,
                    cond.condition_id,
                    m.question,
                    m.category,
                    m.active,
                    m.is_negrisk,
                    cond.outcome_yes,
                    cond.outcome_no,
                    cond.spread,
                    cond.volume_24h,
                    cond.open_interest,
                    days_to_res,
                ],
            )

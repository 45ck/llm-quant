"""Load asset universe from config and sync to DuckDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import duckdb

from llm_quant.config import AppConfig

logger = logging.getLogger(__name__)


def get_tradeable_symbols(config: AppConfig) -> list[str]:
    """Return a sorted list of tradeable ticker symbols from the universe config.

    Parameters
    ----------
    config:
        The application configuration containing the universe definition.

    Returns
    -------
    list[str]
        Sorted list of ticker symbol strings where ``tradeable`` is True.
    """
    symbols = [
        asset.symbol
        for asset in config.universe.assets
        if asset.tradeable
    ]
    symbols.sort()
    logger.info(
        "Resolved %d tradeable symbols from universe '%s'",
        len(symbols),
        config.universe.name,
    )
    return symbols


def sync_universe_to_db(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
) -> int:
    """Upsert the universe table in DuckDB from the current config.

    For each asset entry in ``config.universe.assets``, performs an INSERT OR REPLACE
    into the ``universe`` table.  This ensures newly added assets appear and existing
    entries are updated (e.g. if a name or sector changes).

    Parameters
    ----------
    conn:
        An open DuckDB connection (schema must already be initialised).
    config:
        The application configuration containing the universe definition.

    Returns
    -------
    int
        The number of rows upserted.
    """
    assets = config.universe.assets
    if not assets:
        logger.warning("No asset entries found in universe config — nothing to sync")
        return 0

    now = datetime.now(timezone.utc)
    count = 0

    for asset in assets:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO universe (symbol, name, category, sector, tradeable, added_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [asset.symbol, asset.name, asset.category, asset.sector, asset.tradeable, now],
            )
            count += 1
        except duckdb.Error:
            logger.exception("Failed to upsert symbol %s", asset.symbol)

    conn.commit()
    logger.info(
        "Synced %d / %d asset entries to universe table",
        count,
        len(assets),
    )
    return count

"""Polymarket Parquet data store for market metadata, prices, and orderbooks.

Provides ingestion from Gamma API (market metadata) and CLOB API (price history,
orderbooks), with Parquet storage under data/polymarket/.

Storage layout:
  data/polymarket/markets.parquet                         -- market metadata
  data/polymarket/prices/{token_id}.parquet               -- price history per token
  data/polymarket/orderbooks/{condition_id}_{ts}.parquet  -- orderbook snapshots

All DataFrames use Polars. Incremental price history fetching is supported:
only data after the last stored timestamp is fetched.

Rate limiting: max 2 requests/second to avoid API throttling.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from llm_quant.arb.clob_client import ClobClient
from llm_quant.arb.gamma_client import GammaClient

logger = logging.getLogger(__name__)

# Base directory for all Polymarket data (relative to project root)
DATA_DIR = Path("data/polymarket")
MARKETS_PATH = DATA_DIR / "markets.parquet"
PRICES_DIR = DATA_DIR / "prices"
ORDERBOOKS_DIR = DATA_DIR / "orderbooks"

# Rate limiting: max 2 req/s = 0.5s between requests
_RATE_LIMIT_SLEEP = 0.5


# ------------------------------------------------------------------
# Ensure directories exist
# ------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Create data directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    ORDERBOOKS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Market metadata ingestion
# ------------------------------------------------------------------

# Schema for markets parquet
MARKETS_SCHEMA = {
    "condition_id": pl.Utf8,
    "question": pl.Utf8,
    "category": pl.Utf8,
    "slug": pl.Utf8,
    "outcome_prices": pl.Utf8,  # JSON string of outcome prices
    "volume": pl.Float64,
    "liquidity": pl.Float64,
    "end_date": pl.Utf8,
    "is_neg_risk": pl.Boolean,
    "tokens": pl.Utf8,  # JSON string of token IDs list
}


def _extract_market_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a flat row dict from a raw Gamma API market dict.

    Returns None if the market has no condition ID.
    """
    condition_id = raw.get("conditionId") or raw.get("condition_id") or ""
    if not condition_id:
        # Try to use 'id' as fallback
        condition_id = str(raw.get("id", ""))
    if not condition_id:
        return None

    # Extract outcome prices as JSON string
    outcome_prices_raw = raw.get("outcomePrices", "")
    if isinstance(outcome_prices_raw, list):
        outcome_prices = json.dumps(outcome_prices_raw)
    elif isinstance(outcome_prices_raw, str):
        outcome_prices = outcome_prices_raw
    else:
        outcome_prices = ""

    # Extract CLOB token IDs
    clob_raw = raw.get("clobTokenIds", "")
    if isinstance(clob_raw, list):
        tokens_str = json.dumps([str(t) for t in clob_raw])
    elif isinstance(clob_raw, str) and clob_raw:
        # Already JSON string; validate and normalize
        try:
            parsed = json.loads(clob_raw)
            tokens_str = json.dumps([str(t) for t in parsed])
        except (ValueError, TypeError):
            tokens_str = "[]"
    else:
        tokens_str = "[]"

    # Volume: try multiple field names
    volume = float(
        raw.get("volume") or raw.get("volume24hr") or raw.get("volumeNum") or 0.0
    )

    # Liquidity
    liquidity = float(raw.get("liquidity") or 0.0)

    # NegRisk
    is_neg_risk = bool(
        raw.get("negRisk") or raw.get("isNegRisk") or raw.get("is_neg_risk", False)
    )

    # Category
    category = raw.get("category", "")

    return {
        "condition_id": condition_id,
        "question": raw.get("question") or raw.get("title", ""),
        "category": category,
        "slug": raw.get("slug") or raw.get("market_slug", ""),
        "outcome_prices": outcome_prices,
        "volume": volume,
        "liquidity": liquidity,
        "end_date": raw.get("endDate") or raw.get("end_date") or "",
        "is_neg_risk": is_neg_risk,
        "tokens": tokens_str,
    }


def ingest_markets(
    gamma: GammaClient,
    max_markets: int = 5000,
    dry_run: bool = False,
) -> pl.DataFrame:
    """Fetch all active markets from Gamma API and store as Parquet.

    Parameters
    ----------
    gamma : GammaClient
        Initialized Gamma API client.
    max_markets : int
        Maximum number of markets to fetch.
    dry_run : bool
        If True, fetch and parse but don't write to disk.

    Returns
    -------
    pl.DataFrame
        Market metadata DataFrame.
    """
    logger.info("Fetching active markets from Gamma API (max=%d)...", max_markets)
    raw_markets = gamma.fetch_all_active_markets(max_markets=max_markets)
    logger.info("Fetched %d raw markets", len(raw_markets))

    rows: list[dict[str, Any]] = []
    for raw in raw_markets:
        row = _extract_market_row(raw)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.warning("No valid markets extracted")
        return pl.DataFrame(schema=MARKETS_SCHEMA)

    df = pl.DataFrame(rows, schema=MARKETS_SCHEMA)

    # Deduplicate on condition_id, keeping the last occurrence (most recent data)
    df = df.unique(subset=["condition_id"], keep="last")
    logger.info("Parsed %d unique markets", len(df))

    if not dry_run:
        _ensure_dirs()
        # If existing file exists, merge with new data (dedup on condition_id)
        if MARKETS_PATH.exists():
            existing = pl.read_parquet(MARKETS_PATH)
            df = pl.concat([existing, df]).unique(subset=["condition_id"], keep="last")
            logger.info("Merged with existing data: %d total markets", len(df))

        df.write_parquet(MARKETS_PATH)
        logger.info("Wrote markets to %s", MARKETS_PATH)

    return df


# ------------------------------------------------------------------
# Price history ingestion
# ------------------------------------------------------------------

PRICES_SCHEMA = {
    "timestamp": pl.Int64,
    "price": pl.Float64,
    "token_id": pl.Utf8,
}


def _get_last_timestamp(token_id: str) -> int | None:
    """Get the last stored timestamp for a token's price history.

    Returns None if no data exists.
    """
    path = PRICES_DIR / f"{token_id}.parquet"
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path)
        if df.is_empty():
            return None
        return df["timestamp"].max()
    except Exception:
        logger.debug("Failed to read existing prices for %s", token_id)
        return None


def ingest_price_history(
    clob: ClobClient,
    token_id: str,
    interval: str = "1h",
    dry_run: bool = False,
) -> pl.DataFrame:
    """Fetch price history for a single token and store as Parquet.

    Performs incremental fetching: only fetches data after the last
    stored timestamp.

    Parameters
    ----------
    clob : ClobClient
        Initialized CLOB API client.
    token_id : str
        CLOB token ID.
    interval : str
        Time interval for price history (default: '1h').
    dry_run : bool
        If True, fetch but don't write to disk.

    Returns
    -------
    pl.DataFrame
        Price history DataFrame.
    """
    # Check for existing data to determine start timestamp
    last_ts = _get_last_timestamp(token_id)
    start_ts = (last_ts + 1) if last_ts is not None else None

    try:
        history = clob.get_prices_history(
            token_id=token_id,
            interval=interval,
            start_ts=start_ts,
        )
    except Exception as exc:
        logger.warning("Failed to fetch price history for %s: %s", token_id[:20], exc)
        return pl.DataFrame(schema=PRICES_SCHEMA)

    if not history.points:
        logger.debug("No new price data for token %s", token_id[:20])
        # Return existing data if available
        path = PRICES_DIR / f"{token_id}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return pl.DataFrame(schema=PRICES_SCHEMA)

    new_rows = [
        {"timestamp": pt.timestamp, "price": pt.price, "token_id": token_id}
        for pt in history.points
    ]
    new_df = pl.DataFrame(new_rows, schema=PRICES_SCHEMA)

    if not dry_run:
        _ensure_dirs()
        path = PRICES_DIR / f"{token_id}.parquet"
        if path.exists() and last_ts is not None:
            existing = pl.read_parquet(path)
            combined = pl.concat([existing, new_df])
            # Deduplicate on timestamp, keep last
            combined = combined.unique(subset=["timestamp"], keep="last")
            combined = combined.sort("timestamp")
            combined.write_parquet(path)
            logger.debug(
                "Appended %d points to %s (total: %d)",
                len(new_df),
                path,
                len(combined),
            )
            return combined
        new_df = new_df.sort("timestamp")
        new_df.write_parquet(path)
        logger.debug("Wrote %d points to %s", len(new_df), path)

    return new_df


def ingest_all_prices(
    gamma: GammaClient,
    clob: ClobClient,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Fetch price history for all tokens in the markets Parquet.

    Parameters
    ----------
    gamma : GammaClient
        Used only if markets haven't been fetched yet.
    clob : ClobClient
        CLOB API client for price data.
    limit : int or None
        Max number of markets to process (for testing).
    dry_run : bool
        If True, don't write to disk.

    Returns
    -------
    dict
        Summary with counts: markets_processed, tokens_fetched, errors.
    """
    # Load markets
    if not MARKETS_PATH.exists():
        logger.info("Markets file not found, fetching first...")
        ingest_markets(gamma, dry_run=dry_run)

    if not MARKETS_PATH.exists():
        logger.error("No markets data available")
        return {"markets_processed": 0, "tokens_fetched": 0, "errors": 0}

    markets_df = pl.read_parquet(MARKETS_PATH)

    if limit is not None:
        markets_df = markets_df.head(limit)

    summary = {"markets_processed": 0, "tokens_fetched": 0, "errors": 0}

    for row in markets_df.iter_rows(named=True):
        tokens_str = row["tokens"]
        try:
            token_ids = json.loads(tokens_str)
        except (ValueError, TypeError):
            token_ids = []

        if not token_ids:
            continue

        summary["markets_processed"] += 1

        for token_id in token_ids:
            try:
                df = ingest_price_history(clob, str(token_id), dry_run=dry_run)
                if not df.is_empty():
                    summary["tokens_fetched"] += 1
            except Exception as exc:
                logger.warning("Error fetching prices for token %s: %s", token_id, exc)
                summary["errors"] += 1

            # Rate limiting
            time.sleep(_RATE_LIMIT_SLEEP)

    logger.info(
        "Price ingestion complete: %d markets, %d tokens fetched, %d errors",
        summary["markets_processed"],
        summary["tokens_fetched"],
        summary["errors"],
    )
    return summary


# ------------------------------------------------------------------
# Orderbook snapshot ingestion
# ------------------------------------------------------------------

ORDERBOOK_SCHEMA = {
    "timestamp": pl.Utf8,
    "condition_id": pl.Utf8,
    "token_id": pl.Utf8,
    "side": pl.Utf8,  # "bid" or "ask"
    "price": pl.Float64,
    "size": pl.Float64,
}


def ingest_orderbook_snapshot(
    clob: ClobClient,
    condition_id: str,
    token_ids: list[str],
    dry_run: bool = False,
) -> pl.DataFrame:
    """Snapshot current orderbooks for a single market (all tokens).

    Parameters
    ----------
    clob : ClobClient
        CLOB API client.
    condition_id : str
        Market condition ID (used for file naming).
    token_ids : list[str]
        CLOB token IDs for this market.
    dry_run : bool
        If True, don't write to disk.

    Returns
    -------
    pl.DataFrame
        Orderbook snapshot DataFrame.
    """
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, Any]] = []

    for token_id in token_ids:
        try:
            book = clob.get_book(token_id)
        except Exception as exc:
            logger.warning(
                "Failed to fetch orderbook for token %s: %s", token_id[:20], exc
            )
            continue

        rows.extend(
            {
                "timestamp": ts,
                "condition_id": condition_id,
                "token_id": token_id,
                "side": "bid",
                "price": level.price,
                "size": level.size,
            }
            for level in book.bids
        )
        rows.extend(
            {
                "timestamp": ts,
                "condition_id": condition_id,
                "token_id": token_id,
                "side": "ask",
                "price": level.price,
                "size": level.size,
            }
            for level in book.asks
        )

        # Rate limiting between token requests
        time.sleep(_RATE_LIMIT_SLEEP)

    if not rows:
        return pl.DataFrame(schema=ORDERBOOK_SCHEMA)

    df = pl.DataFrame(rows, schema=ORDERBOOK_SCHEMA)

    if not dry_run:
        _ensure_dirs()
        # Sanitize condition_id for filename (replace non-alphanumeric)
        safe_id = "".join(c if c.isalnum() else "_" for c in condition_id)[:64]
        filename = f"{safe_id}_{ts}.parquet"
        path = ORDERBOOKS_DIR / filename
        df.write_parquet(path)
        logger.debug("Wrote orderbook snapshot to %s (%d levels)", path, len(df))

    return df


def ingest_all_orderbooks(
    gamma: GammaClient,
    clob: ClobClient,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Snapshot orderbooks for all markets in the markets Parquet.

    Parameters
    ----------
    gamma : GammaClient
        Used only if markets haven't been fetched yet.
    clob : ClobClient
        CLOB API client.
    limit : int or None
        Max number of markets to process.
    dry_run : bool
        If True, don't write to disk.

    Returns
    -------
    dict
        Summary: markets_processed, snapshots_taken, errors.
    """
    if not MARKETS_PATH.exists():
        logger.info("Markets file not found, fetching first...")
        ingest_markets(gamma, dry_run=dry_run)

    if not MARKETS_PATH.exists():
        logger.error("No markets data available")
        return {"markets_processed": 0, "snapshots_taken": 0, "errors": 0}

    markets_df = pl.read_parquet(MARKETS_PATH)

    if limit is not None:
        markets_df = markets_df.head(limit)

    summary = {"markets_processed": 0, "snapshots_taken": 0, "errors": 0}

    for row in markets_df.iter_rows(named=True):
        condition_id = row["condition_id"]
        tokens_str = row["tokens"]
        try:
            token_ids = json.loads(tokens_str)
        except (ValueError, TypeError):
            token_ids = []

        if not token_ids:
            continue

        summary["markets_processed"] += 1

        try:
            df = ingest_orderbook_snapshot(
                clob, condition_id, token_ids, dry_run=dry_run
            )
            if not df.is_empty():
                summary["snapshots_taken"] += 1
        except Exception as exc:
            logger.warning(
                "Error snapshotting orderbook for %s: %s", condition_id[:20], exc
            )
            summary["errors"] += 1

    logger.info(
        "Orderbook ingestion complete: %d markets, %d snapshots, %d errors",
        summary["markets_processed"],
        summary["snapshots_taken"],
        summary["errors"],
    )
    return summary


# ------------------------------------------------------------------
# Data access functions
# ------------------------------------------------------------------


def load_markets() -> pl.DataFrame:
    """Load market metadata from Parquet.

    Returns
    -------
    pl.DataFrame
        Market metadata, or empty DataFrame if no data exists.
    """
    if not MARKETS_PATH.exists():
        return pl.DataFrame(schema=MARKETS_SCHEMA)
    return pl.read_parquet(MARKETS_PATH)


def load_price_history(token_id: str) -> pl.DataFrame:
    """Load price history for a specific token.

    Parameters
    ----------
    token_id : str
        CLOB token ID.

    Returns
    -------
    pl.DataFrame
        Price history sorted by timestamp, or empty DataFrame if no data.
    """
    path = PRICES_DIR / f"{token_id}.parquet"
    if not path.exists():
        return pl.DataFrame(schema=PRICES_SCHEMA)
    df = pl.read_parquet(path)
    return df.sort("timestamp")


def load_latest_orderbook(condition_id: str) -> pl.DataFrame:
    """Load the most recent orderbook snapshot for a condition.

    Parameters
    ----------
    condition_id : str
        Market condition ID.

    Returns
    -------
    pl.DataFrame
        Orderbook snapshot, or empty DataFrame if no data.
    """
    if not ORDERBOOKS_DIR.exists():
        return pl.DataFrame(schema=ORDERBOOK_SCHEMA)

    # Sanitize condition_id for filename matching
    safe_id = "".join(c if c.isalnum() else "_" for c in condition_id)[:64]

    # Find all orderbook files for this condition
    matching = sorted(ORDERBOOKS_DIR.glob(f"{safe_id}_*.parquet"))
    if not matching:
        return pl.DataFrame(schema=ORDERBOOK_SCHEMA)

    # Return the latest (last in sorted order, since filenames contain timestamps)
    return pl.read_parquet(matching[-1])


def get_data_summary() -> dict[str, Any]:
    """Get summary statistics about stored Polymarket data.

    Returns
    -------
    dict
        Summary with counts, date ranges, and staleness info.
    """
    summary: dict[str, Any] = {
        "markets_count": 0,
        "markets_file_exists": MARKETS_PATH.exists(),
        "price_tokens_count": 0,
        "orderbook_snapshots_count": 0,
        "markets_last_modified": None,
        "price_files": [],
        "oldest_price_ts": None,
        "newest_price_ts": None,
    }

    # Markets summary
    if MARKETS_PATH.exists():
        try:
            df = pl.read_parquet(MARKETS_PATH)
            summary["markets_count"] = len(df)
            mtime = MARKETS_PATH.stat().st_mtime
            summary["markets_last_modified"] = datetime.fromtimestamp(
                mtime, tz=UTC
            ).isoformat()
        except Exception as exc:
            logger.debug("Error reading markets summary: %s", exc)

    # Price history summary
    if PRICES_DIR.exists():
        price_files = list(PRICES_DIR.glob("*.parquet"))
        summary["price_tokens_count"] = len(price_files)

        oldest_ts: int | None = None
        newest_ts: int | None = None
        for pf in price_files:
            try:
                df = pl.read_parquet(pf)
                if not df.is_empty():
                    ts_min = df["timestamp"].min()
                    ts_max = df["timestamp"].max()
                    if oldest_ts is None or ts_min < oldest_ts:
                        oldest_ts = ts_min
                    if newest_ts is None or ts_max > newest_ts:
                        newest_ts = ts_max
            except Exception as exc:
                logger.debug("Error reading price file %s: %s", pf, exc)

        if oldest_ts is not None:
            summary["oldest_price_ts"] = datetime.fromtimestamp(
                oldest_ts, tz=UTC
            ).isoformat()
        if newest_ts is not None:
            summary["newest_price_ts"] = datetime.fromtimestamp(
                newest_ts, tz=UTC
            ).isoformat()

    # Orderbook summary
    if ORDERBOOKS_DIR.exists():
        ob_files = list(ORDERBOOKS_DIR.glob("*.parquet"))
        summary["orderbook_snapshots_count"] = len(ob_files)

    return summary

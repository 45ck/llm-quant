"""DuckDB schema for prediction market arbitrage system.

Tables:
  pm_markets          — Market metadata (Polymarket / Kalshi)
  pm_conditions       — Individual binary conditions within a market
  pm_negrisk_groups   — NegRisk multi-condition groups (mutually exclusive)
  pm_arb_opportunities — Detected arb opportunities (NegRisk + combinatorial)
  pm_combinatorial_pairs — Claude-detected logically dependent pairs
  pm_scan_log         — Scanner run history
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# --- DDL ------------------------------------------------------------------

_PM_MARKETS_DDL = """
CREATE TABLE IF NOT EXISTS pm_markets (
    market_id       VARCHAR PRIMARY KEY,
    source          VARCHAR NOT NULL,          -- 'polymarket' | 'kalshi'
    slug            VARCHAR,
    question        VARCHAR NOT NULL,
    category        VARCHAR,
    end_date        DATE,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    is_negrisk      BOOLEAN NOT NULL DEFAULT FALSE,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_PM_CONDITIONS_DDL = """
CREATE TABLE IF NOT EXISTS pm_conditions (
    condition_id    VARCHAR PRIMARY KEY,
    market_id       VARCHAR NOT NULL REFERENCES pm_markets(market_id),
    question        VARCHAR NOT NULL,          -- the binary question text
    outcome_yes     DOUBLE,                    -- current YES price (0-1)
    outcome_no      DOUBLE,                    -- current NO price (0-1)
    spread          DOUBLE,                    -- outcome_yes + outcome_no - 1.0
    volume_24h      DOUBLE,
    open_interest   DOUBLE,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_PM_NEGRISK_GROUPS_DDL = """
CREATE TABLE IF NOT EXISTS pm_negrisk_groups (
    group_id        VARCHAR PRIMARY KEY,       -- market_id for NegRisk parent
    condition_ids   VARCHAR[],                 -- array of condition_ids in group
    sum_yes         DOUBLE,                    -- sum of YES prices across conditions
    complement      DOUBLE,                    -- 1.0 - sum_yes (arbitrage margin)
    arb_eligible    BOOLEAN NOT NULL DEFAULT FALSE,  -- complement > threshold
    last_checked    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_PM_ARB_OPPORTUNITIES_DDL = """
CREATE TABLE IF NOT EXISTS pm_arb_opportunities (
    opp_id          VARCHAR PRIMARY KEY,       -- UUID
    arb_type        VARCHAR NOT NULL,
    source          VARCHAR NOT NULL,          -- 'polymarket' | 'kalshi'
    market_id       VARCHAR,
    condition_ids   VARCHAR[],
    spread_pct      DOUBLE NOT NULL,           -- gross arb spread (0-1)
    net_spread_pct  DOUBLE,                    -- after 2% winning fee
    kelly_fraction  DOUBLE,                    -- f* = spread / (1 + spread)
    total_volume    DOUBLE,                    -- liquidity available
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,               -- market end date
    status          VARCHAR NOT NULL DEFAULT 'open',
    notes           VARCHAR
);
"""

_PM_COMBINATORIAL_PAIRS_DDL = """
CREATE TABLE IF NOT EXISTS pm_combinatorial_pairs (
    pair_id             VARCHAR PRIMARY KEY,
    condition_id_a      VARCHAR NOT NULL,
    condition_id_b      VARCHAR NOT NULL,
    question_a          VARCHAR NOT NULL,
    question_b          VARCHAR NOT NULL,
    dependency_type     VARCHAR,
    claude_confidence   DOUBLE,                -- 0-1 confidence from Claude
    valid_combos        JSON,                  -- Claude's valid_combinations output
    price_a             DOUBLE,
    price_b             DOUBLE,
    implied_arb_spread  DOUBLE,                -- spread implied by logical constraint
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed            BOOLEAN DEFAULT FALSE
);
"""

_PM_SCAN_LOG_DDL = """
CREATE TABLE IF NOT EXISTS pm_scan_log (
    scan_id         VARCHAR PRIMARY KEY,
    scan_type       VARCHAR NOT NULL,          -- 'negrisk' | 'combinatorial' | 'full'
    source          VARCHAR NOT NULL,
    markets_scanned INTEGER,
    conditions_scanned INTEGER,
    opps_found      INTEGER,
    pairs_detected  INTEGER,
    duration_secs   DOUBLE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    error           VARCHAR
);
"""

_PM_EXECUTIONS_DDL = """
CREATE TABLE IF NOT EXISTS pm_executions (
    exec_id         VARCHAR PRIMARY KEY,
    event_ticker    VARCHAR NOT NULL,
    event_title     VARCHAR,
    exec_dt         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conditions_json VARCHAR,
    sum_yes_ask     DOUBLE,
    gross_complement DOUBLE,
    net_spread      DOUBLE,
    kelly_fraction  DOUBLE,
    position_usd    DOUBLE,
    expected_pnl    DOUBLE,
    status          VARCHAR DEFAULT 'open',
    actual_pnl      DOUBLE,
    resolved_dt     TIMESTAMP,
    notes           VARCHAR
);
"""

_ALL_DDL = [
    _PM_MARKETS_DDL,
    _PM_CONDITIONS_DDL,
    _PM_NEGRISK_GROUPS_DDL,
    _PM_ARB_OPPORTUNITIES_DDL,
    _PM_COMBINATORIAL_PAIRS_DDL,
    _PM_SCAN_LOG_DDL,
    _PM_EXECUTIONS_DDL,
]


def init_arb_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all prediction market arb tables if they don't exist."""
    for ddl in _ALL_DDL:
        conn.execute(ddl)
    logger.info("Prediction market arb schema initialized.")


def get_arb_connection(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and ensure arb tables exist."""
    conn = duckdb.connect(str(db_path))
    init_arb_schema(conn)
    return conn

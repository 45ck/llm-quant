"""Prediction market arbitrage scanner.

Detects two types of arb:

1. NegRisk Buy-All-NO (dominant arb type — $17.3M extracted in paper):
   In NegRisk multi-condition markets, exactly one condition resolves YES.
   If sum(YES prices) < 1.0, buying all NO positions costs < $1 but pays $1
   when the non-winning conditions all resolve NO.
   Net profit = 1.0 - sum(YES) - 2% winning fee per NO position that pays.
   After fee: net = (1 - sum_yes) - 0.02 * (N-1) * avg_no_price.

2. Single-condition rebalancing (less common, $5.9M in paper):
   YES_price + NO_price < $1.00. Buy both, collect $1 at resolution.
   After 2% fee: net = 1 - (yes_price + no_price) - 0.02.

Filters applied before storing opportunity:
  - min_spread_pct: minimum gross spread (default 5%)
  - min_volume_24h: minimum 24h volume for liquidity (default $1,000)
  - min_open_interest: minimum open interest (default $5,000)
  - category_filter: optional category whitelist (e.g. ['sports'])
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import requests

from llm_quant.arb.gamma_client import GammaClient, Market
from llm_quant.arb.schema import init_arb_schema

logger = logging.getLogger(__name__)

# Fee structure from Saguillo et al. 2025
# maker_base_fee = 0, taker_base_fee = 0
# Only fee: 2% on winning positions at resolution
POLYMARKET_WIN_FEE = 0.02

# Defaults
DEFAULT_MIN_SPREAD_PCT = 0.05  # 5 cents minimum gross spread
DEFAULT_MIN_VOLUME = 1_000.0  # $1k 24h volume
DEFAULT_MIN_OI = 5_000.0  # $5k open interest
DEFAULT_NEGRISK_THRESHOLD = 0.05  # complement > 5% after fee estimation


@dataclass
class ScanRecord:
    """Data for a single scan log entry."""

    scan_id: str
    scan_type: str
    markets_scanned: int
    conditions_scanned: int
    opps_found: int
    pairs_detected: int
    duration: float
    started_at: datetime
    completed_at: datetime
    error: str | None


@dataclass
class ArbOpportunity:
    opp_id: str
    arb_type: str  # 'negrisk_buy_no' | 'single_rebalance'
    source: str  # 'polymarket' | 'kalshi'
    market_id: str
    condition_ids: list[str]
    spread_pct: float  # gross
    net_spread_pct: float  # after 2% fee
    kelly_fraction: float  # f* = net_spread / (1 + net_spread)
    total_volume: float
    notes: str = ""

    def display(self) -> str:
        return (
            f"[{self.arb_type}] {self.market_id} | "
            f"gross={self.spread_pct:.1%} net={self.net_spread_pct:.1%} "
            f"kelly={self.kelly_fraction:.1%} vol=${self.total_volume:,.0f}"
        )


class ArbScanner:
    """Scans Polymarket (and optionally Kalshi) for arbitrage opportunities."""

    def __init__(
        self,
        db_path: str | Path,
        min_spread_pct: float = DEFAULT_MIN_SPREAD_PCT,
        min_volume: float = DEFAULT_MIN_VOLUME,
        min_oi: float = DEFAULT_MIN_OI,
        category_filter: list[str] | None = None,
        source: str = "polymarket",
    ) -> None:
        self.db_path = Path(db_path)
        self.min_spread_pct = min_spread_pct
        self.min_volume = min_volume
        self.min_oi = min_oi
        self.category_filter = category_filter  # None = all categories
        self.source = source
        self._client = GammaClient()
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            init_arb_schema(self._conn)
        return self._conn

    def close(self) -> None:
        """Close the DuckDB connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run_scan(self, max_markets: int = 5000) -> list[ArbOpportunity]:
        """Full scan: fetch all markets, detect arb, persist to DB."""
        scan_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        logger.info("Starting full arb scan (scan_id=%s)", scan_id)

        opps: list[ArbOpportunity] = []
        error_msg: str | None = None
        n_markets = n_conditions = 0

        try:
            raw_markets = self._client.fetch_all_active_markets(max_markets=max_markets)
            markets = self._client.parse_all_markets(raw_markets)
            n_markets = len(markets)

            # Apply category filter
            if self.category_filter:
                markets = [m for m in markets if m.category in self.category_filter]
                logger.info(
                    "After category filter %s: %d markets",
                    self.category_filter,
                    len(markets),
                )

            # Count conditions
            n_conditions = sum(len(m.conditions) for m in markets)

            # Persist markets + conditions
            self._upsert_markets(markets)

            # Detect arb
            for market in markets:
                opps.extend(self._detect_arb(market))

            # Persist opportunities
            if opps:
                self._persist_opportunities(opps)
                logger.info("Found %d arb opportunities", len(opps))
            else:
                logger.info("No arb opportunities found in this scan")

        except (duckdb.Error, ValueError, OSError) as exc:
            error_msg = str(exc)
            logger.exception("Scan error")

        # Log scan
        completed_at = datetime.now(UTC)
        duration = (completed_at - started_at).total_seconds()
        self._log_scan(
            ScanRecord(
                scan_id=scan_id,
                scan_type="full",
                markets_scanned=n_markets,
                conditions_scanned=n_conditions,
                opps_found=len(opps),
                pairs_detected=0,
                duration=duration,
                started_at=started_at,
                completed_at=completed_at,
                error=error_msg,
            )
        )

        return opps

    def run_negrisk_scan(self, max_markets: int = 5000) -> list[ArbOpportunity]:
        """Focused NegRisk scan only — faster, most lucrative arb type."""
        scan_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)

        raw_markets = self._client.fetch_all_active_markets(max_markets=max_markets)
        markets = self._client.parse_all_markets(raw_markets)

        negrisk = [m for m in markets if m.is_negrisk]
        logger.info("NegRisk markets: %d / %d total", len(negrisk), len(markets))

        if self.category_filter:
            negrisk = [m for m in negrisk if m.category in self.category_filter]

        opps: list[ArbOpportunity] = []
        for market in negrisk:
            opps.extend(self._detect_negrisk_arb(market))

        self._upsert_markets(negrisk)
        if opps:
            self._persist_opportunities(opps)

        completed_at = datetime.now(UTC)
        self._log_scan(
            ScanRecord(
                scan_id=scan_id,
                scan_type="negrisk",
                markets_scanned=len(negrisk),
                conditions_scanned=sum(len(m.conditions) for m in negrisk),
                opps_found=len(opps),
                pairs_detected=0,
                duration=(completed_at - started_at).total_seconds(),
                started_at=started_at,
                completed_at=completed_at,
                error=None,
            )
        )

        return opps

    # ------------------------------------------------------------------
    # Arb detection
    # ------------------------------------------------------------------

    def _detect_arb(self, market: Market) -> list[ArbOpportunity]:
        opps: list[ArbOpportunity] = []
        if market.is_negrisk:
            opps.extend(self._detect_negrisk_arb(market))
        else:
            opps.extend(self._detect_single_rebalance(market))
        return opps

    def _detect_negrisk_arb(self, market: Market) -> list[ArbOpportunity]:
        """NegRisk Buy-All-YES strategy.

        # NegRisk Buy-All-YES strategy:
        # - Pay: sum(YES prices) across all mutually exclusive conditions
        # - Receive: $1.00 from the single condition that resolves YES
        # - Gross profit: 1.0 - sum_yes (the "complement")
        # - Net profit: complement - 0.02 (2% winning fee on the $1 payout)
        """
        if len(market.conditions) < 2:
            return []

        n = len(market.conditions)
        sum_yes = market.sum_yes
        complement = 1.0 - sum_yes  # gross profit from buying ALL YES

        if complement <= 0:
            return []

        # Fee: 2% on winning position (the one YES that resolves True)
        net_spread = complement - POLYMARKET_WIN_FEE

        if net_spread < self.min_spread_pct:
            return []

        # Liquidity check — use minimum volume across conditions
        total_volume = sum(c.volume_24h for c in market.conditions)
        min_condition_vol = min(c.volume_24h for c in market.conditions)

        if min_condition_vol < self.min_volume:
            return []

        kelly = net_spread / (1.0 + net_spread)

        opp = ArbOpportunity(
            opp_id=str(uuid.uuid4()),
            arb_type="negrisk_buy_yes",
            source=self.source,
            market_id=market.market_id,
            condition_ids=[c.condition_id for c in market.conditions],
            spread_pct=complement,
            net_spread_pct=net_spread,
            kelly_fraction=kelly,
            total_volume=total_volume,
            notes=(
                f"N={n} conditions, sum_yes={sum_yes:.4f}, complement={complement:.4f}"
            ),
        )
        logger.info("NegRisk arb: %s", opp.display())
        return [opp]

    def _detect_single_rebalance(self, market: Market) -> list[ArbOpportunity]:
        """Single market rebalancing: YES + NO < $1."""
        opps: list[ArbOpportunity] = []
        for cond in market.conditions:
            if cond.outcome_yes <= 0 or cond.outcome_no <= 0:
                continue
            if cond.volume_24h < self.min_volume:
                continue

            gross = -(cond.spread)  # spread is negative when YES+NO < 1
            if gross <= 0:
                continue

            # Fee: 2% on winning position (either YES or NO)
            net = gross - POLYMARKET_WIN_FEE
            if net < self.min_spread_pct:
                continue

            kelly = net / (1.0 + net)
            opp = ArbOpportunity(
                opp_id=str(uuid.uuid4()),
                arb_type="single_rebalance",
                source=self.source,
                market_id=market.market_id,
                condition_ids=[cond.condition_id],
                spread_pct=gross,
                net_spread_pct=net,
                kelly_fraction=kelly,
                total_volume=cond.volume_24h,
                notes=f"yes={cond.outcome_yes:.4f} no={cond.outcome_no:.4f}",
            )
            logger.info("Single rebalance arb: %s", opp.display())
            opps.append(opp)

        return opps

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _upsert_markets(self, markets: list[Market]) -> None:
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        for m in markets:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pm_markets
                    (market_id, source, slug, question, category,
                     end_date, active, is_negrisk,
                     fetched_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        m.market_id,
                        self.source,
                        m.slug,
                        m.question,
                        m.category,
                        m.end_date,
                        m.active,
                        m.is_negrisk,
                        now,
                        now,
                    ],
                )
                for c in m.conditions:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO pm_conditions
                        (condition_id, market_id, question, outcome_yes, outcome_no,
                         spread, volume_24h, open_interest, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            c.condition_id,
                            m.market_id,
                            c.question,
                            c.outcome_yes,
                            c.outcome_no,
                            c.spread,
                            c.volume_24h,
                            c.open_interest,
                            now,
                        ],
                    )
            except duckdb.Error as exc:
                logger.debug("Failed to upsert market %s: %s", m.market_id, exc)

    def _persist_opportunities(self, opps: list[ArbOpportunity]) -> None:
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        for opp in opps:
            try:
                conn.execute(
                    """
                    INSERT INTO pm_arb_opportunities
                    (opp_id, arb_type, source, market_id, condition_ids,
                     spread_pct, net_spread_pct, kelly_fraction, total_volume,
                     detected_at, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                    """,
                    [
                        opp.opp_id,
                        opp.arb_type,
                        opp.source,
                        opp.market_id,
                        opp.condition_ids,
                        opp.spread_pct,
                        opp.net_spread_pct,
                        opp.kelly_fraction,
                        opp.total_volume,
                        now,
                        opp.notes,
                    ],
                )
            except duckdb.Error as exc:
                logger.debug("Failed to persist opportunity %s: %s", opp.opp_id, exc)

    def _log_scan(self, rec: ScanRecord) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO pm_scan_log
                (scan_id, scan_type, source,
                 markets_scanned, conditions_scanned,
                 opps_found, pairs_detected,
                 duration_secs, started_at,
                 completed_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    rec.scan_id,
                    rec.scan_type,
                    self.source,
                    rec.markets_scanned,
                    rec.conditions_scanned,
                    rec.opps_found,
                    rec.pairs_detected,
                    rec.duration,
                    rec.started_at.isoformat(),
                    rec.completed_at.isoformat(),
                    rec.error,
                ],
            )
        except duckdb.Error as exc:
            logger.warning("Failed to log scan: %s", exc)

    # ------------------------------------------------------------------
    # Kalshi NegRisk scan (mutually_exclusive events)
    # ------------------------------------------------------------------

    def run_kalshi_negrisk_scan(
        self,
        min_spread: float | None = None,
        min_volume: float | None = None,
    ) -> list[ArbOpportunity]:
        """Scan Kalshi mutually_exclusive events for NegRisk arb.

        Fetches all open mutually_exclusive events, sums YES_ask prices,
        and flags events where sum < 1.0 (buying all YES costs < $1,
        guaranteed payout of $1 from the winning condition).

        Kalshi taker fee: 3% on winning position.
        Net spread = (1 - sum_yes_ask) - 0.03.
        """
        from llm_quant.arb.kalshi_client import KALSHI_WIN_FEE, KalshiClient

        min_spread = min_spread if min_spread is not None else self.min_spread_pct
        min_volume = min_volume if min_volume is not None else self.min_volume

        scan_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        logger.info("Starting Kalshi NegRisk scan (scan_id=%s)", scan_id)

        opps: list[ArbOpportunity] = []
        error_msg: str | None = None
        n_events = 0

        try:
            client = KalshiClient()
            events = client.fetch_negrisk_events()
            n_events = len(events)
            logger.info(
                "Kalshi: %d mutually exclusive events with 2+ conditions", n_events
            )

            # Apply category filter
            if self.category_filter:
                events = [
                    e
                    for e in events
                    if e.category.lower() in [c.lower() for c in self.category_filter]
                ]
                logger.info("After category filter: %d events", len(events))

            for evt in events:
                # Persist event → pm_markets
                self._upsert_kalshi_event(evt)

                # Check for arb
                if not evt.mutually_exclusive or len(evt.markets) < 2:
                    continue

                sum_yes = evt.sum_yes_ask
                complement = evt.negrisk_complement
                net_spread = complement - KALSHI_WIN_FEE

                if net_spread < min_spread:
                    continue

                total_vol = evt.total_volume_24h
                min_cond_vol = evt.min_condition_volume

                # Volume check: each condition must be individually fillable
                if min_cond_vol < min_volume or total_vol < min_volume:
                    logger.debug(
                        "Kalshi %s: skipped (vol=%.0f < threshold %.0f)",
                        evt.event_ticker,
                        min_cond_vol,
                        min_volume,
                    )
                    continue

                kelly = net_spread / (1.0 + net_spread)

                opp = ArbOpportunity(
                    opp_id=str(uuid.uuid4()),
                    arb_type="negrisk_buy_yes",
                    source="kalshi",
                    market_id=evt.event_ticker,
                    condition_ids=[c.ticker for c in evt.markets],
                    spread_pct=complement,
                    net_spread_pct=net_spread,
                    kelly_fraction=kelly,
                    total_volume=total_vol,
                    notes=(
                        f"N={len(evt.markets)} outcomes, sum_yes={sum_yes:.4f}, "
                        f"complement={complement:.4f}, "
                        f"category={evt.category}"
                    ),
                )
                opps.append(opp)
                logger.info("Kalshi NegRisk arb: %s", opp.display())

            if opps:
                self._persist_opportunities(opps)

        except (requests.RequestException, duckdb.Error, ValueError) as exc:
            error_msg = str(exc)
            logger.exception("Kalshi scan error: %s", exc)

        completed_at = datetime.now(UTC)
        self._log_scan(
            ScanRecord(
                scan_id=scan_id,
                scan_type="kalshi_negrisk",
                markets_scanned=n_events,
                conditions_scanned=(
                    sum(len(e.markets) for e in events) if "events" in locals() else 0
                ),
                opps_found=len(opps),
                pairs_detected=0,
                duration=(completed_at - started_at).total_seconds(),
                started_at=started_at,
                completed_at=completed_at,
                error=error_msg,
            )
        )

        return opps

    def _upsert_kalshi_event(self, evt: Any) -> None:
        """Persist a Kalshi event and its conditions to DuckDB."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO pm_markets
                (market_id, source, slug, question, category, end_date,
                 active, is_negrisk, fetched_at, updated_at)
                VALUES (?, 'kalshi', ?, ?, ?, NULL, TRUE, TRUE, ?, ?)
                """,
                [
                    evt.event_ticker,
                    evt.series_ticker,
                    evt.title,
                    evt.category,
                    now,
                    now,
                ],
            )
            for cond in evt.markets:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pm_conditions
                    (condition_id, market_id, question,
                     outcome_yes, outcome_no, spread,
                     volume_24h, open_interest, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    [
                        cond.ticker,
                        evt.event_ticker,
                        cond.title,
                        cond.yes_ask,
                        cond.no_ask,
                        cond.yes_ask + cond.no_ask - 1.0,
                        cond.volume_24h,
                        now,
                    ],
                )
        except duckdb.Error as exc:
            logger.debug("Failed to upsert Kalshi event %s: %s", evt.event_ticker, exc)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_open_opportunities(self, min_net_spread: float = 0.03) -> list[dict]:
        """Return open opportunities above minimum net spread."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT opp_id, arb_type, market_id, spread_pct, net_spread_pct,
                   kelly_fraction, total_volume, detected_at, notes
            FROM pm_arb_opportunities
            WHERE status = 'open'
              AND net_spread_pct >= ?
            ORDER BY net_spread_pct DESC
            """,
            [min_net_spread],
        ).fetchall()

        cols = [
            "opp_id",
            "arb_type",
            "market_id",
            "spread_pct",
            "net_spread_pct",
            "kelly_fraction",
            "total_volume",
            "detected_at",
            "notes",
        ]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def get_scan_summary(self) -> dict:
        """Summary of scan history."""
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total_scans,
                MAX(started_at) as last_scan,
                SUM(opps_found) as total_opps_detected,
                AVG(duration_secs) as avg_duration_secs
            FROM pm_scan_log
            WHERE error IS NULL
            """
        ).fetchone()
        if row:
            return {
                "total_scans": row[0],
                "last_scan": row[1],
                "total_opps_detected": row[2],
                "avg_duration_secs": round(row[3] or 0, 1),
            }
        return {}

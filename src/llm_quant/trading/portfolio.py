"""Portfolio state management.

Tracks cash, positions, and computes NAV / exposure metrics.
The Portfolio object is the single source of truth for current holdings
and is persisted to DuckDB via the ledger module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A single holding in the portfolio."""

    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    stop_loss: float = 0.0

    # -- derived properties ------------------------------------------------

    @property
    def market_value(self) -> float:
        """Current market value (signed – negative if short)."""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis for the position."""
        return self.shares * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Unrealised profit / loss in currency terms."""
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        """Unrealised P&L as a percentage of cost basis."""
        if self.avg_cost == 0.0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class Portfolio:
    """In-memory representation of the full portfolio state."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        pod_id: str = "default",
    ) -> None:
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.initial_capital: float = initial_capital
        self.pod_id: str = pod_id
        logger.info(
            "Portfolio initialised with capital=%.2f, pod_id=%s",
            initial_capital,
            pod_id,
        )

    # -- aggregate properties ----------------------------------------------

    @property
    def nav(self) -> float:
        """Net asset value: cash + sum of position market values."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        """Sum of absolute market values (long + short legs)."""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Net signed market value across all positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total profit / loss since inception."""
        return self.nav - self.initial_capital

    # -- helpers -----------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update ``current_price`` for every held position.

        Symbols not present in *prices* are left unchanged and a warning is
        emitted so callers can detect stale data.
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
            else:
                logger.warning(
                    "No price provided for %s – keeping stale price %.4f",
                    symbol,
                    pos.current_price,
                )

    def get_position_weight(self, symbol: str) -> float:
        """Return the weight of *symbol* as a fraction of NAV."""
        current_nav = self.nav
        if current_nav == 0.0:
            return 0.0
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return pos.market_value / current_nav

    def get_sector_exposure(self, sector_map: dict[str, str]) -> dict[str, float]:
        """Compute aggregate weight per sector.

        Parameters
        ----------
        sector_map:
            Mapping ``{symbol: sector}`` (typically built from the universe
            config).

        Returns
        -------
        dict[str, float]
            ``{sector: total_weight}`` where weight is fraction of NAV.
        """
        current_nav = self.nav
        if current_nav == 0.0:
            return {}

        sector_weights: dict[str, float] = {}
        for symbol, pos in self.positions.items():
            sector = sector_map.get(symbol, "Unknown")
            weight = pos.market_value / current_nav
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        return sector_weights

    def to_snapshot_dict(self) -> dict:
        """Serialise current state for DB persistence / prompt building.

        Returns a flat dict suitable for insertion into
        ``portfolio_snapshots`` and for constructing ``MarketContext``.
        """
        current_nav = self.nav
        return {
            "nav": current_nav,
            "cash": self.cash,
            "cash_pct": (self.cash / current_nav) if current_nav else 0.0,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "total_pnl": self.total_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "pnl_pct": p.pnl_pct,
                    "weight": (p.market_value / current_nav if current_nav else 0.0),
                    "stop_loss": p.stop_loss,
                }
                for p in self.positions.values()
            ],
        }

    # -- persistence -------------------------------------------------------

    @classmethod
    def from_db(
        cls,
        conn: duckdb.DuckDBPyConnection,
        initial_capital: float,
        pod_id: str = "default",
    ) -> Portfolio:
        """Restore portfolio from the latest snapshot stored in DuckDB.

        If no snapshot exists the method returns a *fresh* portfolio with the
        given ``initial_capital``.
        """
        # Find the most recent snapshot
        # Check if pod_id column exists in schema
        cols = [c[0] for c in conn.execute("DESCRIBE portfolio_snapshots").fetchall()]
        if "pod_id" in cols:
            row = conn.execute(
                """
                SELECT snapshot_id, nav, cash
                FROM portfolio_snapshots
                WHERE pod_id = ?
                ORDER BY date DESC, snapshot_id DESC
                LIMIT 1
                """,
                [pod_id],
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT snapshot_id, nav, cash
                FROM portfolio_snapshots
                ORDER BY date DESC, snapshot_id DESC
                LIMIT 1
                """,
            ).fetchone()

        if row is None:
            logger.info(
                "No existing snapshot found – returning fresh "
                "portfolio (capital=%.2f, pod_id=%s)",
                initial_capital,
                pod_id,
            )
            return cls(initial_capital=initial_capital, pod_id=pod_id)

        snapshot_id: int = row[0]
        nav_db: float = row[1]
        cash_db: float = row[2]

        portfolio = cls(initial_capital=initial_capital, pod_id=pod_id)
        portfolio.cash = cash_db

        # Load positions attached to this snapshot
        pos_rows = conn.execute(
            """
            SELECT symbol, shares, avg_cost, current_price, stop_loss
            FROM positions
            WHERE snapshot_id = ?
            """,
            [snapshot_id],
        ).fetchall()

        for pr in pos_rows:
            symbol, shares, avg_cost, current_price, stop_loss = pr
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=avg_cost,
                current_price=current_price,
                stop_loss=stop_loss if stop_loss is not None else 0.0,
            )

        logger.info(
            "Restored portfolio from snapshot %d: NAV=%.2f, cash=%.2f, %d position(s)",
            snapshot_id,
            nav_db,
            cash_db,
            len(portfolio.positions),
        )
        return portfolio

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Portfolio(nav={self.nav:.2f}, cash={self.cash:.2f}, "
            f"positions={len(self.positions)})"
        )

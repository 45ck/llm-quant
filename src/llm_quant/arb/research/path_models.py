"""Mathematical models for compounding path requirements."""

from __future__ import annotations

import math


class CompoundingPath:
    """Model required parameters for a compounding path."""

    @staticmethod
    def min_edge(
        start: float,
        target: float,
        days: int,
        trades_per_day: int,
    ) -> float:
        """Minimum per-trade edge to compound from start to target.

        Returns the fractional edge per trade (e.g. 0.005 = 0.5%).
        Formula: (target/start)^(1/(days*trades_per_day)) - 1
        """
        total_trades = days * trades_per_day
        if total_trades <= 0:
            msg = "total trades must be positive"
            raise ValueError(msg)
        if start <= 0:
            msg = "start bankroll must be positive"
            raise ValueError(msg)
        return (target / start) ** (1.0 / total_trades) - 1.0

    @staticmethod
    def required_win_rate(
        edge: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Required win rate given edge and avg win/loss sizes.

        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        Solving: win_rate = (edge + avg_loss) / (avg_win + avg_loss)
        """
        denom = avg_win + avg_loss
        if denom == 0:
            msg = "avg_win + avg_loss must be nonzero"
            raise ValueError(msg)
        return (edge + avg_loss) / denom

    @staticmethod
    def kelly_fraction(
        win_rate: float,
        win_loss_ratio: float,
    ) -> float:
        """Kelly criterion for optimal bet sizing.

        f* = (p * (b + 1) - 1) / b
        where p = win_rate, b = win_loss_ratio (avg_win / avg_loss).
        """
        if win_loss_ratio == 0:
            msg = "win_loss_ratio must be nonzero"
            raise ValueError(msg)
        return (win_rate * (win_loss_ratio + 1.0) - 1.0) / win_loss_ratio

    @staticmethod
    def simulate_path(
        bankroll: float,
        edge_per_trade: float,
        trades_per_day: int,
        days: int,
        fee_rate: float = 0.02,
        slippage: float = 0.005,
    ) -> list[float]:
        """Simulate deterministic bankroll path with given params.

        Each trade grows bankroll by (1 + edge - fee - slippage).
        Returns list of bankroll values, length = days * trades_per_day + 1.
        First element is starting bankroll.
        """
        net_edge = edge_per_trade - fee_rate - slippage
        growth = 1.0 + net_edge
        total_trades = days * trades_per_day
        path = [bankroll]
        current = bankroll
        for _ in range(total_trades):
            current *= growth
            path.append(current)
        return path

    @staticmethod
    def days_to_target(
        start: float,
        target: float,
        edge_per_trade: float,
        trades_per_day: int,
        fee_rate: float = 0.0,
        slippage: float = 0.0,
    ) -> float:
        """Days needed to compound from start to target.

        Returns fractional days.
        """
        net_edge = edge_per_trade - fee_rate - slippage
        if net_edge <= 0:
            return math.inf
        growth_per_trade = math.log(1.0 + net_edge)
        total_growth_needed = math.log(target / start)
        trades_needed = total_growth_needed / growth_per_trade
        return trades_needed / trades_per_day

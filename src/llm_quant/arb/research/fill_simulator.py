"""Models for fill probability and execution quality."""

from __future__ import annotations

import math


class FillSimulator:
    """Models fill probability based on orderbook depth and size."""

    @staticmethod
    def fill_probability(
        trade_size: float,
        depth_at_price: float,
        distance_from_mid: float,
    ) -> float:
        """Probability of full fill given depth and distance.

        Model: P(fill) = min(1, depth/size) * (1 - exp(-k*dist))
        where k controls how quickly fill prob increases with
        distance from mid. At mid (dist=0), P=0 (no passive fill).
        Far from mid, approaches min(1, depth/size).

        Parameters
        ----------
        trade_size : number of contracts to fill
        depth_at_price : available depth at the target price
        distance_from_mid : distance from midpoint in cents

        Returns
        -------
        Probability in [0, 1].
        """
        if trade_size <= 0:
            return 0.0
        if depth_at_price <= 0:
            return 0.0

        # Depth ratio — can we fill at all?
        depth_ratio = min(1.0, depth_at_price / trade_size)

        # Distance factor — passive fills more likely further from mid
        # k = 50 gives reasonable decay: at 1c away ~39%, at 2c ~63%
        k = 50.0
        distance_factor = 1.0 - math.exp(-k * abs(distance_from_mid))

        return depth_ratio * distance_factor

    @staticmethod
    def expected_slippage(
        trade_size: float,
        depth_at_price: float,
        tick_size: float = 0.01,
    ) -> float:
        """Expected slippage in cents given trade size vs depth.

        If trade_size <= depth, slippage = 0 (filled at price).
        If trade_size > depth, excess walks the book.
        Slippage = sum of (excess_at_level * tick_size) / trade_size.
        Simplified: (trade_size - depth) / trade_size * tick_size.

        Returns slippage in same units as tick_size (e.g. cents).
        """
        if trade_size <= 0:
            return 0.0
        if depth_at_price <= 0:
            return tick_size  # Full slippage
        if trade_size <= depth_at_price:
            return 0.0

        excess_ratio = (trade_size - depth_at_price) / trade_size
        return excess_ratio * tick_size

    @staticmethod
    def polymarket_fee(
        price: float,
        num_contracts: int,
        fee_rate: float = 0.02,
    ) -> float:
        """Polymarket fee per the documented formula.

        Fee = C * p * feeRate * (p * (1-p))^exponent
        where exponent depends on fee regime.

        Simplified (standard regime, exponent=1):
        Fee = num_contracts * price * fee_rate * price * (1 - price)

        Parameters
        ----------
        price : contract price (probability), 0-1
        num_contracts : number of contracts
        fee_rate : base fee rate (default 2%)

        Returns
        -------
        Fee in dollars.
        """
        if price <= 0 or price >= 1:
            return 0.0
        return num_contracts * price * fee_rate * price * (1.0 - price)

"""Inflation regime overlay using 2x2 growth/inflation matrix.

Based on Bridgewater's All Weather / Ray Dalio's economic machine framework.
Four quadrants:
- Reflationary Boom: rising growth + rising inflation → overweight commodities/energy
- Disinflationary Boom: rising growth + falling inflation → overweight equities/credit
- Stagflation: falling growth + rising inflation → overweight gold/TIPS
- Deflationary Bust: falling growth + falling inflation → overweight nominal bonds

Data sources:
- Growth proxy: SPY 3-month momentum + ISM PMI from FRED (if available)
- Inflation proxy: TIPS 5y breakeven (T5YIE from FRED) + CPI momentum

Key limitations per research notes:
- Stagflation onset detection lag: 2-4 months
- TIPS breakeven accuracy: ±55-80 bps, distorted by liquidity premium (~100 bps)
- Expected Sharpe improvement: 0.07-0.20 (modest but additive)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

logger = logging.getLogger(__name__)


class GrowthRegime(StrEnum):
    RISING = "rising"
    FALLING = "falling"
    NEUTRAL = "neutral"


class InflationRegime(StrEnum):
    RISING = "rising"
    FALLING = "falling"
    NEUTRAL = "neutral"


class MacroQuadrant(StrEnum):
    REFLATIONARY_BOOM = "reflationary_boom"     # rising growth + rising inflation
    DISINFLATIONARY_BOOM = "disinflationary_boom"  # rising growth + falling inflation
    STAGFLATION = "stagflation"                 # falling growth + rising inflation
    DEFLATIONARY_BUST = "deflationary_bust"     # falling growth + falling inflation
    TRANSITION = "transition"                   # neutral on one or both axes


@dataclass
class QuadrantTilts:
    """Suggested asset allocation tilts for a macro quadrant."""

    overweight: list[str]
    underweight: list[str]
    confidence: float  # 0-1, based on signal clarity


#: Per-quadrant tilt recommendations (all five states including TRANSITION).
QUADRANT_TILTS: dict[MacroQuadrant, QuadrantTilts] = {
    MacroQuadrant.REFLATIONARY_BOOM: QuadrantTilts(
        overweight=["commodities", "energy", "international_equity"],
        underweight=["nominal_bonds", "defensive_equity"],
        confidence=0.7,
    ),
    MacroQuadrant.DISINFLATIONARY_BOOM: QuadrantTilts(
        overweight=["equities", "credit", "tech"],
        underweight=["commodities", "gold"],
        confidence=0.8,
    ),
    MacroQuadrant.STAGFLATION: QuadrantTilts(
        overweight=["gold", "tips", "commodities"],
        underweight=["equities", "credit", "nominal_bonds"],
        confidence=0.5,  # detection lag makes this unreliable
    ),
    MacroQuadrant.DEFLATIONARY_BUST: QuadrantTilts(
        overweight=["nominal_bonds", "gold", "defensive_equity"],
        underweight=["equities", "credit", "commodities"],
        confidence=0.75,
    ),
    MacroQuadrant.TRANSITION: QuadrantTilts(
        overweight=[],
        underweight=[],
        confidence=0.3,
    ),
}

#: Alias for the transition tilt (convenience reference).
_TRANSITION_TILTS = QUADRANT_TILTS[MacroQuadrant.TRANSITION]

# Momentum threshold (fraction) to classify SPY growth regime.
_GROWTH_RISING_THRESHOLD = 0.02   # > +2% 63-day return → rising
_GROWTH_FALLING_THRESHOLD = -0.02  # < -2% 63-day return → falling

# Breakeven change threshold (percentage points) for inflation classification.
_BREAKEVEN_RISING_BPS = 0.10   # > +10 bps 63-day change → rising
_BREAKEVEN_FALLING_BPS = -0.10  # < -10 bps 63-day change → falling

# CPI momentum threshold (fraction) for fallback inflation classification.
_CPI_RISING_THRESHOLD = 0.001   # > +0.1% 3m change → rising
_CPI_FALLING_THRESHOLD = -0.001  # < -0.1% 3m change → falling


class InflationRegimeDetector:
    """Classify the current macro quadrant from SPY momentum and inflation proxies.

    The 2x2 matrix maps growth (rising/falling/neutral) and inflation
    (rising/falling/neutral) onto one of five ``MacroQuadrant`` values.
    When either axis is NEUTRAL the result is TRANSITION.

    Parameters
    ----------
    growth_lookback:
        Number of trading days for SPY momentum window (default 63 ≈ 3 months).
    inflation_lookback:
        Number of trading days for TIPS breakeven change window (default 63).
    """

    def __init__(
        self,
        growth_lookback: int = 63,
        inflation_lookback: int = 63,
    ) -> None:
        self.growth_lookback = growth_lookback
        self.inflation_lookback = inflation_lookback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        spy_prices: pl.Series,
        tips_breakeven: pl.Series | None = None,
        cpi_series: pl.Series | None = None,
    ) -> tuple[MacroQuadrant, QuadrantTilts]:
        """Classify the current macro quadrant.

        Parameters
        ----------
        spy_prices:
            Daily SPY close prices.  Must contain at least
            ``growth_lookback + 1`` values for a valid signal.
        tips_breakeven:
            Optional TIPS 5y breakeven rate series (T5YIE from FRED), as
            percentage (e.g. 2.35).  Must contain at least
            ``inflation_lookback + 1`` values for a valid signal.
        cpi_series:
            Optional CPI level series (CPIAUCSL from FRED).  Used as
            fallback when ``tips_breakeven`` is None or too short.

        Returns
        -------
        (MacroQuadrant, QuadrantTilts)
            The detected quadrant and its associated asset tilt recommendations.
        """
        growth = self._classify_growth(spy_prices)
        inflation = self._classify_inflation(tips_breakeven, cpi_series)

        logger.debug(
            "InflationRegimeDetector: growth=%s inflation=%s",
            growth,
            inflation,
        )

        # Any NEUTRAL axis → TRANSITION
        if growth == GrowthRegime.NEUTRAL or inflation == InflationRegime.NEUTRAL:
            logger.info("Macro quadrant=transition (one or both axes neutral)")
            return MacroQuadrant.TRANSITION, _TRANSITION_TILTS

        # Map the 2x2 to a quadrant
        quadrant = _QUADRANT_MAP[(growth, inflation)]
        tilts = QUADRANT_TILTS[quadrant]

        logger.info(
            "Macro quadrant=%s (growth=%s, inflation=%s, confidence=%.2f)",
            quadrant,
            growth,
            inflation,
            tilts.confidence,
        )
        return quadrant, tilts

    # ------------------------------------------------------------------
    # Internal classifiers
    # ------------------------------------------------------------------

    def _classify_growth(self, spy_prices: pl.Series) -> GrowthRegime:
        """Classify growth regime from SPY 63-day momentum.

        Returns
        -------
        GrowthRegime
            RISING if 63d return > +2%, FALLING if < -2%, NEUTRAL otherwise.
            Returns NEUTRAL when insufficient price history is available.
        """
        n = len(spy_prices)
        min_needed = self.growth_lookback + 1

        if n < min_needed:
            logger.warning(
                "SPY price series too short for growth classification "
                "(%d rows, need %d); defaulting to NEUTRAL",
                n,
                min_needed,
            )
            return GrowthRegime.NEUTRAL

        # Use the last growth_lookback+1 values
        recent = spy_prices.slice(n - min_needed, min_needed)
        price_now = float(recent[-1])
        price_lookback = float(recent[0])

        if price_lookback == 0.0:
            logger.warning("SPY lookback price is 0; defaulting growth to NEUTRAL")
            return GrowthRegime.NEUTRAL

        momentum = (price_now / price_lookback) - 1.0
        logger.debug("SPY 63d momentum=%.4f (now=%.2f, lookback=%.2f)", momentum, price_now, price_lookback)

        if momentum > _GROWTH_RISING_THRESHOLD:
            return GrowthRegime.RISING
        if momentum < _GROWTH_FALLING_THRESHOLD:
            return GrowthRegime.FALLING
        return GrowthRegime.NEUTRAL

    def _classify_inflation(
        self,
        tips_breakeven: pl.Series | None,
        cpi: pl.Series | None,
    ) -> InflationRegime:
        """Classify inflation regime from TIPS breakeven or CPI.

        Priority:
        1. TIPS breakeven 63-day change (preferred — higher frequency).
        2. CPI 3-month momentum (fallback when breakeven unavailable).
        3. NEUTRAL (graceful degradation when no data is available).

        Returns
        -------
        InflationRegime
            RISING / FALLING / NEUTRAL.
        """
        # --- Primary: TIPS breakeven ---
        if tips_breakeven is not None and len(tips_breakeven) >= self.inflation_lookback + 1:
            n = len(tips_breakeven)
            recent = tips_breakeven.slice(n - (self.inflation_lookback + 1), self.inflation_lookback + 1)
            be_now = float(recent[-1])
            be_lookback = float(recent[0])
            change_bps = be_now - be_lookback  # in percentage-point units matching T5YIE
            logger.debug(
                "TIPS breakeven 63d change=%.4f pp (now=%.4f, lookback=%.4f)",
                change_bps,
                be_now,
                be_lookback,
            )
            if change_bps > _BREAKEVEN_RISING_BPS:
                return InflationRegime.RISING
            if change_bps < _BREAKEVEN_FALLING_BPS:
                return InflationRegime.FALLING
            return InflationRegime.NEUTRAL

        # --- Fallback: CPI 3-month momentum ---
        if cpi is not None and len(cpi) >= 3:
            n = len(cpi)
            cpi_now = float(cpi[-1])
            cpi_3m_ago = float(cpi[max(0, n - 3)])
            if cpi_3m_ago == 0.0:
                logger.warning("CPI 3m-ago value is 0; defaulting inflation to NEUTRAL")
                return InflationRegime.NEUTRAL
            cpi_momentum = (cpi_now - cpi_3m_ago) / cpi_3m_ago
            logger.debug(
                "CPI 3m momentum=%.4f (now=%.2f, 3m_ago=%.2f)",
                cpi_momentum,
                cpi_now,
                cpi_3m_ago,
            )
            if cpi_momentum > _CPI_RISING_THRESHOLD:
                return InflationRegime.RISING
            if cpi_momentum < _CPI_FALLING_THRESHOLD:
                return InflationRegime.FALLING
            return InflationRegime.NEUTRAL

        # --- Graceful degradation ---
        logger.warning(
            "No usable inflation data (TIPS breakeven or CPI); defaulting to NEUTRAL. "
            "Run pq fetch to pull T5YIE/CPIAUCSL from FRED."
        )
        return InflationRegime.NEUTRAL


# ---------------------------------------------------------------------------
# Lookup table: (GrowthRegime, InflationRegime) → MacroQuadrant
# ---------------------------------------------------------------------------

_QUADRANT_MAP: dict[tuple[GrowthRegime, InflationRegime], MacroQuadrant] = {
    (GrowthRegime.RISING, InflationRegime.RISING): MacroQuadrant.REFLATIONARY_BOOM,
    (GrowthRegime.RISING, InflationRegime.FALLING): MacroQuadrant.DISINFLATIONARY_BOOM,
    (GrowthRegime.FALLING, InflationRegime.RISING): MacroQuadrant.STAGFLATION,
    (GrowthRegime.FALLING, InflationRegime.FALLING): MacroQuadrant.DEFLATIONARY_BUST,
}

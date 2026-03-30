"""CFTC COT data fetcher. No API key required.

Anti-overfitting rules (hard-coded, NOT to be optimised):
- 20/80 thresholds and 156-week lookback are fixed — do not tune them
- COT signals CONFIRM or WARN on existing signals only; they never generate
  independent trade entries
- CFTC releases data on Friday; we apply it on Monday open to avoid
  look-ahead bias (not Tuesday — Friday release is after close, Monday is
  the first tradeable session that can legally see the report)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# CFTC Disaggregated / Legacy COT report codes for our universe symbols.
# Codes taken from the CFTC published instrument reference.
# TFF report for FX instruments, Disaggregated for commodities.
CFTC_CODES: dict[str, str] = {
    "GLD": "088691",   # Gold
    "SLV": "084691",   # Silver
    "USO": "067651",   # Crude Oil (WTI)
    "EURUSD": "099741",  # Euro FX
    "USDJPY": "097741",  # Japanese Yen
    "GBPUSD": "096742",  # British Pound
    "AUDUSD": "232741",  # Australian Dollar
}

# Map from universe.toml symbols (which may include "=X" suffix for forex)
# to the CFTC_CODES keys above.
SYMBOL_TO_COT_KEY: dict[str, str] = {
    "GLD": "GLD",
    "SLV": "SLV",
    "USO": "USO",
    "EURUSD=X": "EURUSD",
    "USDJPY=X": "USDJPY",
    "GBPUSD=X": "GBPUSD",
    "AUDUSD=X": "AUDUSD",
}

# Fixed thresholds — do NOT optimise these (anti-overfitting discipline).
COT_CROWDED_LONG_THRESHOLD: int = 80   # COT index > 80 → commercials net short (crowded long)
COT_CROWDED_SHORT_THRESHOLD: int = 20  # COT index < 20 → commercials net long (crowded short)
COT_LOOKBACK_WEEKS: int = 156          # 3-year (156-week) min-max window — fixed

# Symbols explicitly excluded from COT analysis.
# USO: post-April 2020 fund restructuring broke the WTI futures COT mapping —
# the fund now rolls across multiple contract months rather than the front-month
# WTI contract that the CFTC code 067651 covers.
COT_INELIGIBLE_SYMBOLS: frozenset[str] = frozenset({"USO"})


class CotFetcher:
    """Fetch and process CFTC Commitments of Traders (COT) data.

    Uses the CFTC public Socrata API — no API key required.
    Rate limits are generous for weekly data volumes.
    """

    CFTC_API: str = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

    def fetch(
        self,
        cftc_code: str,
        lookback_weeks: int = COT_LOOKBACK_WEEKS,
    ) -> pl.DataFrame:
        """Fetch COT data for the given CFTC instrument code.

        Parameters
        ----------
        cftc_code:
            6-digit CFTC instrument code (e.g. ``"088691"`` for Gold).
        lookback_weeks:
            Number of weeks of history to request.  Fixed at 156 by
            anti-overfitting policy — only override in tests.

        Returns
        -------
        pl.DataFrame
            Columns: ``report_date, commercial_net, noncommercial_net,
            open_interest``.  Sorted ascending by ``report_date``.
            Returns an empty DataFrame with the correct schema on failure.
        """
        empty: pl.DataFrame = pl.DataFrame(
            schema={
                "report_date": pl.Date,
                "commercial_net": pl.Float64,
                "noncommercial_net": pl.Float64,
                "open_interest": pl.Float64,
            }
        )

        try:
            import urllib.parse
            import urllib.request
            import json

            # CFTC Socrata API: filter by cftc_contract_market_code, limit rows
            cutoff = (datetime.now(tz=UTC) - timedelta(weeks=lookback_weeks)).strftime(
                "%Y-%m-%dT00:00:00"
            )
            params = urllib.parse.urlencode(
                {
                    "$where": (
                        f"cftc_contract_market_code='{cftc_code}' "
                        f"AND report_date_as_yyyy_mm_dd >= '{cutoff}'"
                    ),
                    "$order": "report_date_as_yyyy_mm_dd ASC",
                    "$limit": str(lookback_weeks + 10),  # small buffer
                }
            )
            url = f"{self.CFTC_API}?{params}"
            logger.debug("COT fetch: GET %s", url)

            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                raw: list[dict[str, Any]] = json.loads(resp.read())

        except Exception:
            logger.exception("COT fetch failed for CFTC code %s", cftc_code)
            return empty

        if not raw:
            logger.warning("COT: no rows returned for CFTC code %s", cftc_code)
            return empty

        rows: list[dict[str, Any]] = []
        for rec in raw:
            try:
                report_date_str: str = rec.get(
                    "report_date_as_yyyy_mm_dd", rec.get("report_date", "")
                )
                if not report_date_str:
                    continue

                # Parse date — may come as "2024-01-02T00:00:00.000" or "2024-01-02"
                report_date_str = report_date_str[:10]

                # Commercial positions (hedgers — the "smart money")
                comm_long = float(rec.get("comm_positions_long_all", 0) or 0)
                comm_short = float(rec.get("comm_positions_short_all", 0) or 0)

                # Non-commercial (speculators)
                noncomm_long = float(rec.get("noncomm_positions_long_all", 0) or 0)
                noncomm_short = float(rec.get("noncomm_positions_short_all", 0) or 0)

                oi = float(rec.get("open_interest_all", 0) or 0)

                rows.append(
                    {
                        "report_date": report_date_str,
                        "commercial_net": comm_long - comm_short,
                        "noncommercial_net": noncomm_long - noncomm_short,
                        "open_interest": oi,
                    }
                )
            except Exception:
                logger.debug("COT: skipping malformed row: %s", rec)
                continue

        if not rows:
            logger.warning("COT: no parseable rows for CFTC code %s", cftc_code)
            return empty

        df = pl.DataFrame(rows).with_columns(
            pl.col("report_date").str.to_date("%Y-%m-%d"),
            pl.col("commercial_net").cast(pl.Float64),
            pl.col("noncommercial_net").cast(pl.Float64),
            pl.col("open_interest").cast(pl.Float64),
        )
        return df.sort("report_date")

    def compute_cot_index(
        self,
        df: pl.DataFrame,
        window: int = COT_LOOKBACK_WEEKS,
    ) -> pl.DataFrame:
        """Add a ``cot_index`` column: min-max normalisation of ``commercial_net``.

        COT Index = (current - rolling_min) / (rolling_max - rolling_min) * 100

        Values near 100 → commercials are net long (historically bullish signal).
        Values near 0   → commercials are net short (historically bearish signal).

        The ``window`` is fixed at 156 weeks by anti-overfitting policy.

        Parameters
        ----------
        df:
            Output of :meth:`fetch` — must contain ``commercial_net`` column.
        window:
            Rolling lookback in rows (weeks).  Fixed at 156 by policy.

        Returns
        -------
        pl.DataFrame
            Input DataFrame with an additional ``cot_index`` (Float64, 0–100)
            column.  Rows where the window is not yet full get ``null``.
        """
        if df.is_empty():
            return df.with_columns(pl.lit(None).cast(pl.Float64).alias("cot_index"))

        roll_min = pl.col("commercial_net").rolling_min(window_size=window)
        roll_max = pl.col("commercial_net").rolling_max(window_size=window)

        df = df.with_columns(
            pl.when(roll_max - roll_min == 0.0)
            .then(pl.lit(50.0))
            .otherwise(
                (pl.col("commercial_net") - roll_min)
                / (roll_max - roll_min)
                * 100.0
            )
            .alias("cot_index")
        )
        return df

    def get_regime_signal(self, symbol: str) -> dict[str, Any]:
        """Return the current COT index and crowding signal for a universe symbol.

        Parameters
        ----------
        symbol:
            Universe symbol (e.g. ``"GLD"``, ``"EURUSD=X"``).

        Returns
        -------
        dict with keys:
            ``symbol``, ``cot_key``, ``report_date``, ``cot_index``,
            ``commercial_net``, ``signal``

        The ``signal`` field is one of:
            - ``"crowded_long"``  — COT index > 80 (commercial shorts concentrated
              → longs crowded → mean-reversion risk for long positions)
            - ``"crowded_short"`` — COT index < 20 (commercial longs concentrated
              → shorts crowded → mean-reversion risk for short positions)
            - ``"neutral"``       — no crowding signal

        This signal is a CONFIRMATION / WARNING overlay only.  It must not be
        used to generate trade entries independently.
        """
        default: dict[str, Any] = {
            "symbol": symbol,
            "cot_key": None,
            "report_date": None,
            "cot_index": None,
            "commercial_net": None,
            "signal": "neutral",
        }

        # Skip symbols explicitly flagged as COT-ineligible
        if symbol in COT_INELIGIBLE_SYMBOLS:
            logger.info("Skipping %s: cot_eligible=False", symbol)
            return default

        cot_key = SYMBOL_TO_COT_KEY.get(symbol)
        if cot_key is None:
            logger.debug("COT: no CFTC code mapping for symbol %s", symbol)
            return default

        cftc_code = CFTC_CODES.get(cot_key)
        if cftc_code is None:
            logger.debug("COT: no CFTC code for cot_key %s", cot_key)
            return default

        default["cot_key"] = cot_key

        df = self.fetch(cftc_code)
        if df.is_empty():
            return default

        df = self.compute_cot_index(df)

        # Latest row with a valid COT index
        valid = df.filter(pl.col("cot_index").is_not_null())
        if valid.is_empty():
            return default

        latest = valid.tail(1).to_dicts()[0]
        cot_index: float = float(latest["cot_index"])
        commercial_net: float = float(latest["commercial_net"])
        report_date = latest["report_date"]

        if cot_index > COT_CROWDED_LONG_THRESHOLD:
            signal = "crowded_long"
        elif cot_index < COT_CROWDED_SHORT_THRESHOLD:
            signal = "crowded_short"
        else:
            signal = "neutral"

        logger.info(
            "COT %s (%s): index=%.1f, commercial_net=%.0f, signal=%s, "
            "report_date=%s (apply Monday open)",
            symbol,
            cot_key,
            cot_index,
            commercial_net,
            signal,
            report_date,
        )

        return {
            "symbol": symbol,
            "cot_key": cot_key,
            "report_date": str(report_date),
            "cot_index": round(cot_index, 1),
            "commercial_net": round(commercial_net, 0),
            "signal": signal,
        }

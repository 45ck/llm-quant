"""ML gate feature extraction from causal indicator data.

All features are backward-looking only — no future data ever touches this code.
The feature set is FIXED (pre-specified with economic priors) and must NOT be
modified between training and inference, or between strategy variants.

Economic rationale for each feature:
  vix_pct_rank_252   — Market fear level. High VIX = risk-off, signals less reliable.
  spy_vs_sma200      — Bull/bear regime. Below SMA200 = bear, correlations change.
  credit_stress_20d  — Credit widening (HYG falling) = stress spreading to equity.
  follower_rsi_14    — Entry price quality. Low RSI = oversold = better entry for long.
  follower_vol_rank  — Execution quality. High vol = wider slippage, noisier signals.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Fixed feature names — never reorder, add, or remove without versioning the gate.
FEATURE_NAMES: list[str] = [
    "vix_pct_rank_252",
    "spy_vs_sma200",
    "credit_stress_20d",
    "follower_rsi_14",
    "follower_vol_rank_21",
]

# Sentinel value for missing features (imputed as neutral/midpoint)
_MISSING = {
    "vix_pct_rank_252": 0.5,
    "spy_vs_sma200": 0.0,
    "credit_stress_20d": 0.0,
    "follower_rsi_14": 50.0,
    "follower_vol_rank_21": 0.5,
}


def _symbol_data(indicators_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Return sorted rows for *symbol*, or empty DataFrame if not present."""
    return indicators_df.filter(pl.col("symbol") == symbol).sort("date")


def _pct_rank(values: list[float], current: float) -> float:
    """Fraction of values in *values* that are <= *current* (percentile rank 0–1)."""
    if not values:
        return 0.5
    return sum(1 for v in values if v <= current) / len(values)


def extract_gate_features(
    as_of_date: date,  # noqa: ARG001 — reserved for future filtering validation
    indicators_df: pl.DataFrame,
    follower_symbol: str,
) -> np.ndarray:
    """Extract the 5 fixed gate features for a single prediction row.

    Parameters
    ----------
    as_of_date:
        The date for which signals are being generated.  The indicators_df
        passed to this function must already be filtered to ``date <= as_of_date``
        by the caller (the BacktestEngine enforces this via causal slicing).
    indicators_df:
        Causally-filtered indicator DataFrame (all dates up to as_of_date).
    follower_symbol:
        The strategy's primary asset (the one being entered/exited).

    Returns
    -------
    np.ndarray
        Shape (1, 5) feature array matching FEATURE_NAMES order.
    """
    features: dict[str, float] = dict(_MISSING)

    # ------------------------------------------------------------------
    # Feature 1: VIX percentile rank (252 days)
    # ------------------------------------------------------------------
    vix_df = _symbol_data(indicators_df, "VIX")
    if len(vix_df) >= 2:
        closes = vix_df["close"].to_list()
        current_vix = closes[-1]
        window = closes[-252:] if len(closes) >= 252 else closes
        features["vix_pct_rank_252"] = _pct_rank(window, current_vix)

    # ------------------------------------------------------------------
    # Feature 2: SPY distance from SMA-200
    # ------------------------------------------------------------------
    spy_df = _symbol_data(indicators_df, "SPY")
    if len(spy_df) >= 1 and "sma_200" in spy_df.columns:
        row = spy_df.tail(1).row(0, named=True)
        sma200 = row.get("sma_200")
        close = row.get("close")
        if sma200 and sma200 > 0 and close:
            features["spy_vs_sma200"] = close / sma200 - 1.0

    # ------------------------------------------------------------------
    # Feature 3: HYG 20-day return (credit stress proxy)
    # ------------------------------------------------------------------
    hyg_df = _symbol_data(indicators_df, "HYG")
    if len(hyg_df) >= 21:
        closes = hyg_df["close"].to_list()
        features["credit_stress_20d"] = closes[-1] / closes[-21] - 1.0
    elif len(hyg_df) >= 2:
        closes = hyg_df["close"].to_list()
        features["credit_stress_20d"] = closes[-1] / closes[0] - 1.0

    # ------------------------------------------------------------------
    # Feature 4: Follower asset RSI-14
    # ------------------------------------------------------------------
    fol_df = _symbol_data(indicators_df, follower_symbol)
    if len(fol_df) >= 1 and "rsi_14" in fol_df.columns:
        rsi = fol_df.tail(1).row(0, named=True).get("rsi_14")
        if rsi is not None:
            features["follower_rsi_14"] = float(rsi)

    # ------------------------------------------------------------------
    # Feature 5: Follower 21-day volatility percentile (ATR/close)
    # ------------------------------------------------------------------
    if len(fol_df) >= 22 and "atr_14" in fol_df.columns:
        rows = fol_df.tail(22)
        closes = rows["close"].to_list()
        atrs = rows["atr_14"].to_list()
        # Normalised vol series: ATR/close for each day
        norm_vols = [a / c for a, c in zip(atrs, closes, strict=False) if c and c > 0]
        if norm_vols:
            current_vol = norm_vols[-1]
            features["follower_vol_rank_21"] = _pct_rank(norm_vols[:-1], current_vol)

    return np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float64)

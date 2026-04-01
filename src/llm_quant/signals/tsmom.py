"""Time-Series Momentum (TSMOM) signals with multi-lookback blending.

Based on:
- Moskowitz, Ooi, Pedersen (2012): TSMOM — sign(r_{t-12,t}) × vol_scaled_signal
- Hurst, Ooi, Pedersen (2013): equal-weight blend across lookbacks near-optimal
- Barroso & Santa-Clara (2015): vol scaling nearly doubles Sharpe
- Levine & Pedersen (2016): TSMOM and SMA crossovers are mathematically equivalent

Three lookbacks: 1-month (21d), 3-month (63d), 12-month (252d)
Equal-weight blend (1/3 each)
Vol scaling: target 12% annualized using 126d realized variance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from llm_quant.data.indicators import compute_vol_scalar

logger = logging.getLogger(__name__)


@dataclass
class TsmomConfig:
    """Configuration for TSMOM signal computation."""

    lookbacks: list[int] = field(default_factory=lambda: [21, 63, 252])
    blend_weights: list[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    vol_target: float = 0.12
    vol_window: int = 126
    max_vol_scalar: float = 2.0
    flat_threshold: float = 0.2

    def __post_init__(self) -> None:
        if len(self.lookbacks) != len(self.blend_weights):
            raise ValueError(
                f"lookbacks length ({len(self.lookbacks)}) must match "
                f"blend_weights length ({len(self.blend_weights)})"
            )
        total_weight = sum(self.blend_weights)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"blend_weights must sum to 1.0, got {total_weight:.6f}")


@dataclass
class TsmomSignal:
    """TSMOM signal for a single symbol at a single point in time."""

    symbol: str
    date: date
    raw_signal: float  # blended sign score (-1.0 to +1.0)
    vol_scalar: float  # applied volatility scalar
    scaled_signal: float  # raw_signal × vol_scalar
    direction: str  # "long", "short", or "flat"
    lookback_signals: dict[int, float]  # {21: +1.0, 63: +1.0, 252: -1.0}
    signal_agreement: (
        float  # fraction of lookbacks agreeing with blend direction (0.0–1.0)
    )


class TsmomCalculator:
    """Compute multi-lookback blended TSMOM signals with volatility scaling.

    Usage
    -----
    calc = TsmomCalculator()
    sig = calc.compute(price_series, symbol="SPY")

    # Batch mode (prices_df must have columns: symbol, close, date)
    signals = calc.compute_batch(prices_df)
    """

    def __init__(self, config: TsmomConfig | None = None) -> None:
        self.config = config or TsmomConfig()

    def compute(self, prices: pl.Series, symbol: str = "") -> TsmomSignal:
        """Compute TSMOM signal for a single price series.

        Parameters
        ----------
        prices:
            Chronologically ordered price series (oldest first).
        symbol:
            Ticker label used in the returned TsmomSignal.

        Returns
        -------
        TsmomSignal
            Fully populated signal. When there is insufficient price history
            for a given lookback, that lookback's sign is 0.0 (excluded from
            the blend calculation but still recorded in lookback_signals).
        """
        cfg = self.config
        n = len(prices)
        price_now = prices[-1]

        # Infer date (use today's date as a fallback)
        from datetime import date as date_cls

        signal_date = date_cls.today()

        # -- Compute per-lookback sign signals ----------------------------------
        lookback_signals: dict[int, float] = {}
        for lb in cfg.lookbacks:
            if n > lb:
                price_lb_ago = prices[-(lb + 1)]
                ret = (price_now / price_lb_ago) - 1.0
                lookback_signals[lb] = 1.0 if ret > 0 else -1.0
            else:
                # Insufficient history — record as 0 (neutral, excluded from blend)
                lookback_signals[lb] = 0.0

        # -- Blend signals (only include lookbacks with sufficient history) -----
        valid_pairs = [
            (w, lookback_signals[lb])
            for lb, w in zip(cfg.lookbacks, cfg.blend_weights)
            if n > lb
        ]

        if not valid_pairs:
            # No lookback has enough history — return flat signal
            return TsmomSignal(
                symbol=symbol,
                date=signal_date,
                raw_signal=0.0,
                vol_scalar=1.0,
                scaled_signal=0.0,
                direction="flat",
                lookback_signals=lookback_signals,
                signal_agreement=0.0,
            )

        # Re-normalize weights for valid lookbacks only
        total_valid_weight = sum(w for w, _ in valid_pairs)
        raw_signal = sum(w * s for w, s in valid_pairs) / total_valid_weight

        # -- Volatility scalar --------------------------------------------------
        vol_scalars = compute_vol_scalar(
            prices,
            target_vol=cfg.vol_target,
            window=cfg.vol_window,
            max_scalar=cfg.max_vol_scalar,
        )

        # Use last non-null value; fall back to 1.0 if all null
        last_scalar: float = 1.0
        for v in reversed(vol_scalars.to_list()):
            if v is not None:
                last_scalar = float(v)
                break

        # -- Scaled signal and direction ----------------------------------------
        scaled_signal = raw_signal * last_scalar

        if scaled_signal > cfg.flat_threshold:
            direction = "long"
        elif scaled_signal < -cfg.flat_threshold:
            direction = "short"
        else:
            direction = "flat"

        # -- Signal agreement ---------------------------------------------------
        # Fraction of valid lookbacks whose sign matches the blend direction
        if direction == "flat":
            # Neutral blend — count lookbacks agreeing with sign of raw_signal
            reference_sign = raw_signal
        else:
            reference_sign = scaled_signal

        if reference_sign == 0.0:
            signal_agreement = 0.0
        else:
            blend_sign = 1.0 if reference_sign > 0 else -1.0
            agreeing = sum(
                1
                for lb, s in lookback_signals.items()
                if s != 0.0 and (1.0 if s > 0 else -1.0) == blend_sign
            )
            total_valid = sum(1 for s in lookback_signals.values() if s != 0.0)
            signal_agreement = agreeing / total_valid if total_valid > 0 else 0.0

        return TsmomSignal(
            symbol=symbol,
            date=signal_date,
            raw_signal=round(raw_signal, 6),
            vol_scalar=round(last_scalar, 6),
            scaled_signal=round(scaled_signal, 6),
            direction=direction,
            lookback_signals=lookback_signals,
            signal_agreement=round(signal_agreement, 4),
        )

    def compute_batch(self, prices_df: pl.DataFrame) -> list[TsmomSignal]:
        """Compute TSMOM signals for all symbols in a prices DataFrame.

        Parameters
        ----------
        prices_df:
            DataFrame with columns: symbol (str), date (date), close (float).
            Must be sorted by (symbol, date) ascending.

        Returns
        -------
        list[TsmomSignal]
            One TsmomSignal per symbol that has sufficient history.
            Symbols with zero rows are silently skipped.
        """
        required = {"symbol", "close"}
        missing = required - set(prices_df.columns)
        if missing:
            raise ValueError(f"prices_df is missing required columns: {missing}")

        signals: list[TsmomSignal] = []
        symbols = prices_df["symbol"].unique().to_list()

        for sym in sorted(symbols):
            sym_df = prices_df.filter(pl.col("symbol") == sym).sort("date")
            if sym_df.is_empty():
                continue
            price_series = sym_df["close"]
            try:
                sig = self.compute(price_series, symbol=sym)
                signals.append(sig)
            except Exception:
                logger.warning("TSMOM compute failed for %s", sym, exc_info=True)

        return signals


def compute_portfolio_tsmom(
    prices_df: pl.DataFrame,
    config: TsmomConfig | None = None,
) -> pl.DataFrame:
    """Compute TSMOM signals for all symbols in prices_df and return a summary DataFrame.

    Parameters
    ----------
    prices_df:
        DataFrame with columns: symbol (str), date (date), close (float).
        Must cover at least 252+ trading days for full signal coverage.
    config:
        Optional TsmomConfig; defaults to TsmomConfig() if not provided.

    Returns
    -------
    pl.DataFrame
        Columns: symbol, direction, scaled_signal, vol_scalar, signal_agreement.
        Sorted by abs(scaled_signal) descending.
    """
    calc = TsmomCalculator(config=config)
    signals = calc.compute_batch(prices_df)

    if not signals:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "direction": pl.Utf8,
                "scaled_signal": pl.Float64,
                "vol_scalar": pl.Float64,
                "signal_agreement": pl.Float64,
            }
        )

    rows = [
        {
            "symbol": s.symbol,
            "direction": s.direction,
            "scaled_signal": s.scaled_signal,
            "vol_scalar": s.vol_scalar,
            "signal_agreement": s.signal_agreement,
        }
        for s in signals
    ]

    df = pl.DataFrame(rows)
    df = df.with_columns(pl.col("scaled_signal").abs().alias("_abs_signal"))
    df = df.sort("_abs_signal", descending=True).drop("_abs_signal")
    return df

"""NLP Signal Strategy — trades on pre-computed NLP scores from parquet files.

Reads pre-computed NLP scores from ``data/nlp/signals/{ticker}.parquet`` and
generates long/neutral signals based on configurable thresholds.  Each parquet
file has the schema::

    date                  Date
    forward_looking_score Float64  — fraction of forward-looking sentences
    hedging_score         Float64  — hedging language density
    sentiment_score       Float64  — net sentiment score [0, 1]
    i_we_ratio            Float64  — first-person singular / plural pronoun ratio
    readability_score     Float64  — readability metric (higher = more readable)

NLP signal rules (using default ``signal_column="sentiment_score"``):

    LONG   when signal_column >= entry_threshold (after lag)
    EXIT   when signal_column <  exit_threshold  (after lag)

The ``signal_lag`` parameter (minimum 1) ensures no look-ahead bias: a score
computed from a filing on day T is only observable on day T + signal_lag.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from llm_quant.backtest.strategy import Strategy, StrategyConfig
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)

# Default directory for NLP signal parquet files
_DEFAULT_NLP_DIR = Path("data/nlp/signals")

# Valid NLP score columns that can be used as the primary signal
_VALID_SIGNAL_COLUMNS: frozenset[str] = frozenset(
    [
        "forward_looking_score",
        "hedging_score",
        "sentiment_score",
        "i_we_ratio",
        "readability_score",
    ]
)


class NlpSignalStrategy(Strategy):
    """Strategy that trades on pre-computed NLP scores from parquet files.

    Reads NLP scores from ``data/nlp/signals/{ticker}.parquet``, aligns them
    with price data, applies a configurable lag to prevent look-ahead bias,
    and generates long/neutral signals based on threshold crossings on a
    chosen NLP score column.

    **Parameters (in StrategyConfig.parameters)**

    - ``ticker`` : str — which ticker's NLP data to load (REQUIRED)
    - ``signal_column`` : str — NLP score to use as primary signal
      (default: ``"sentiment_score"``)
    - ``entry_threshold`` : float — score >= this generates long signal
      (default: 0.6)
    - ``exit_threshold`` : float — score < this generates exit signal
      (default: 0.3)
    - ``signal_lag`` : int — days to lag the NLP signal; minimum 1 to avoid
      look-ahead bias (default: 1)
    - ``target_weight`` : float — position weight when signal is active
      (default: 0.80)
    - ``nlp_dir`` : str — override directory for NLP parquet files
      (default: ``"data/nlp/signals"``)
    - ``nlp_df_override`` : pl.DataFrame | None — inject NLP data directly
      for testing (bypasses file I/O)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        params = config.parameters
        self._ticker: str = params.get("ticker", "")
        self._signal_column: str = params.get("signal_column", "sentiment_score")
        self._entry_threshold: float = params.get("entry_threshold", 0.6)
        self._exit_threshold: float = params.get("exit_threshold", 0.3)
        self._signal_lag: int = max(1, params.get("signal_lag", 1))
        self._target_weight: float = params.get("target_weight", 0.80)
        self._nlp_dir: Path = Path(params.get("nlp_dir", str(_DEFAULT_NLP_DIR)))
        # Allow injecting NLP data directly (for testing)
        self._nlp_df_override: pl.DataFrame | None = params.get("nlp_df_override")
        # Cache loaded NLP data
        self._nlp_cache: pl.DataFrame | None = None

    def _load_nlp_scores(self) -> pl.DataFrame | None:
        """Load NLP scores for the configured ticker.

        Returns the NLP DataFrame with a ``date`` column and signal columns,
        or None if the parquet file does not exist or is unreadable.
        """
        # Use injected override if available (testing path)
        if self._nlp_df_override is not None:
            return self._validate_nlp_df(self._nlp_df_override)

        # Return cached data if already loaded
        if self._nlp_cache is not None:
            return self._nlp_cache

        if not self._ticker:
            logger.warning(
                "NlpSignalStrategy: no ticker configured — returning neutral"
            )
            return None

        parquet_path = self._nlp_dir / f"{self._ticker}.parquet"
        if not parquet_path.exists():
            logger.warning(
                "NlpSignalStrategy: parquet not found at %s — returning neutral",
                parquet_path,
            )
            return None

        try:
            df = pl.read_parquet(parquet_path)
        except Exception:
            logger.exception(
                "NlpSignalStrategy: failed to read parquet at %s", parquet_path
            )
            return None

        validated = self._validate_nlp_df(df)
        if validated is not None:
            self._nlp_cache = validated
        return validated

    def _validate_nlp_df(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Validate and normalise an NLP scores DataFrame.

        Returns the cleaned DataFrame (date cast, sorted) or None if
        required columns are missing.
        """
        if "date" not in df.columns:
            logger.warning("NlpSignalStrategy: DataFrame missing 'date' column")
            return None

        if self._signal_column not in df.columns:
            logger.warning(
                "NlpSignalStrategy: DataFrame missing signal column '%s'",
                self._signal_column,
            )
            return None

        # Ensure date column is Date type
        if df.schema["date"] != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        # Sort by date for correct lag application
        return df.sort("date")

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate long/neutral signals based on NLP score thresholds.

        Steps:
        1. Load the NLP parquet for the configured ticker
        2. Apply signal_lag (shift scores forward by signal_lag days)
        3. Find the score for as_of_date (after lagging)
        4. Compare against entry/exit thresholds
        5. Return BUY or CLOSE signal as appropriate

        If no NLP data is available (missing parquet, missing dates), returns
        an empty list (neutral — no signal, no error).
        """
        nlp_df = self._load_nlp_scores()
        if nlp_df is None:
            return []

        # Determine the trading symbol — use ticker by default
        symbol = self._ticker
        if symbol not in prices or prices[symbol] <= 0:
            # Try to find the symbol in the indicators DataFrame
            available = indicators_df.select("symbol").unique().to_series().to_list()
            if symbol not in available:
                logger.debug(
                    "NlpSignalStrategy: ticker %s not in prices or indicators", symbol
                )
                return []

        # Apply signal lag: shift the signal column forward by signal_lag rows.
        # This means the score from day T becomes available on day T + signal_lag.
        lagged_df = nlp_df.with_columns(
            pl.col(self._signal_column).shift(self._signal_lag).alias("_lagged_signal")
        )

        # Filter to rows on or before as_of_date and get the latest available signal
        available = lagged_df.filter(
            (pl.col("date") <= as_of_date) & pl.col("_lagged_signal").is_not_null()
        )

        if len(available) == 0:
            logger.debug(
                "NlpSignalStrategy: no lagged NLP data for %s on or before %s",
                self._ticker,
                as_of_date,
            )
            return []

        # Take the most recent row
        latest = available.sort("date").tail(1).row(0, named=True)
        score = latest["_lagged_signal"]

        if score is None:
            return []

        signals: list[TradeSignal] = []
        has_position = symbol in portfolio.positions
        close = prices.get(symbol, 0.0)

        if close <= 0:
            return []

        # EXIT: score dropped below exit threshold while holding
        if has_position and score < self._exit_threshold:
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.CLOSE,
                    conviction=Conviction.MEDIUM,
                    target_weight=0.0,
                    stop_loss=0.0,
                    reasoning=(
                        f"NLP exit: {self._signal_column}={score:.3f} "
                        f"< exit_threshold={self._exit_threshold:.3f} "
                        f"(lag={self._signal_lag}, date={latest['date']})"
                    ),
                )
            )
            return signals

        # ENTRY: score above entry threshold and not already positioned
        if (
            not has_position
            and len(portfolio.positions) < self.config.max_positions
            and score >= self._entry_threshold
        ):
            stop_loss = close * (1.0 - self.config.stop_loss_pct)
            signals.append(
                TradeSignal(
                    symbol=symbol,
                    action=Action.BUY,
                    conviction=Conviction.MEDIUM,
                    target_weight=self._target_weight,
                    stop_loss=stop_loss,
                    reasoning=(
                        f"NLP entry: {self._signal_column}={score:.3f} "
                        f">= entry_threshold={self._entry_threshold:.3f} "
                        f"(lag={self._signal_lag}, date={latest['date']})"
                    ),
                )
            )

        return signals

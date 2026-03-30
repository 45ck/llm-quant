"""NLP Signal Strategy — uses pre-computed NLP scores as trading indicators.

Reads forward-looking / causal classification scores from the DuckDB
``nlp_classifications`` table and aggregates them into per-date signals for a
given ticker.  The resulting signals are overlaid on a standard price-based
strategy: the NLP scores act as an *entry gate* (BUY only when the filing
language is sufficiently forward-looking) and as an *exit trigger* (close when
hedging language dominates).

Schema consumed (``nlp_classifications`` table):
    doc_id          VARCHAR   — e.g. "AAPL_2024-01-15" (ticker_date)
    sentence_hash   VARCHAR
    sentence_text   VARCHAR
    forward_looking BOOLEAN   — True if sentence makes a prediction/forecast
    causal          BOOLEAN   — True if sentence asserts X causes Y
    confidence      DOUBLE    — classifier confidence [0, 1]
    classified_at   TIMESTAMP

Derived per-doc aggregates (computed at runtime, not stored):
    forward_looking_ratio  — fraction of sentences that are forward-looking
    causal_ratio           — fraction of sentences that are causal
    readability_score      — proxy: 1 - avg sentence length / 30  (simple Flesch proxy)
    hedging_density        — fraction containing hedging keywords

NLP signal rules:
    BUY  when forward_looking_ratio >= forward_looking_threshold
          AND causal_ratio >= causal_threshold
          AND hedging_density < hedging_threshold
    SELL when hedging_density >= hedging_threshold (reduce / exit)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl

from llm_quant.backtest.strategy import Strategy, StrategyConfig
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)

# Words commonly associated with management hedging language
_HEDGING_KEYWORDS: frozenset[str] = frozenset(
    [
        "may",
        "might",
        "could",
        "uncertain",
        "uncertainty",
        "risk",
        "risks",
        "challenging",
        "difficult",
        "volatile",
        "volatility",
        "headwind",
        "headwinds",
        "cautious",
        "caution",
        "concern",
        "concerns",
    ]
)


@dataclass
class NlpScoreRow:
    """Aggregated NLP scores for one document (ticker + date)."""

    ticker: str
    doc_date: date
    forward_looking_ratio: float
    causal_ratio: float
    readability_score: float  # 0–1; higher = more readable (shorter sentences)
    hedging_density: float  # 0–1; fraction of sentences with hedging words
    sentence_count: int


def _compute_hedging_density(sentences: list[str]) -> float:
    """Fraction of sentences containing at least one hedging keyword."""
    if not sentences:
        return 0.0
    hits = 0
    for s in sentences:
        words = set(re.findall(r"\b\w+\b", s.lower()))
        if words & _HEDGING_KEYWORDS:
            hits += 1
    return hits / len(sentences)


def _readability_proxy(sentences: list[str]) -> float:
    """Simple readability proxy: 1 - (mean_word_count / 30), clamped to [0, 1].

    Shorter sentences → higher score.  Threshold of 30 words represents a
    moderately complex sentence.
    """
    if not sentences:
        return 0.5
    word_counts = [len(s.split()) for s in sentences if s.strip()]
    if not word_counts:
        return 0.5
    mean_wc = sum(word_counts) / len(word_counts)
    return max(0.0, min(1.0, 1.0 - mean_wc / 30.0))


def aggregate_nlp_scores(
    nlp_df: pl.DataFrame,
    ticker: str,
    doc_date: date,
) -> NlpScoreRow | None:
    """Aggregate sentence-level classifications into document-level NLP scores.

    Parameters
    ----------
    nlp_df:
        DataFrame from ``nlp_classifications`` table (all rows, or pre-filtered).
    ticker:
        Ticker symbol (must match prefix of ``doc_id``, e.g. "AAPL").
    doc_date:
        Document date to match (matches ``doc_id`` suffix formatted as YYYY-MM-DD).

    Returns
    -------
    NlpScoreRow or None if no matching document found.
    """
    date_str = doc_date.strftime("%Y-%m-%d")
    doc_id_prefix = f"{ticker}_{date_str}"

    filtered = nlp_df.filter(pl.col("doc_id").str.starts_with(doc_id_prefix))
    if len(filtered) == 0:
        return None

    rows = filtered.to_dicts()
    sentences = [r["sentence_text"] for r in rows]
    n = len(rows)

    forward_count = sum(1 for r in rows if r.get("forward_looking") is True)
    causal_count = sum(1 for r in rows if r.get("causal") is True)

    return NlpScoreRow(
        ticker=ticker,
        doc_date=doc_date,
        forward_looking_ratio=forward_count / n if n else 0.0,
        causal_ratio=causal_count / n if n else 0.0,
        readability_score=_readability_proxy(sentences),
        hedging_density=_compute_hedging_density(sentences),
        sentence_count=n,
    )


class NlpSignalStrategy(Strategy):
    """Strategy that uses NLP-derived signals (readability, forward-looking, hedging).

    Reads pre-computed NLP scores from a Polars DataFrame backed by the
    ``nlp_classifications`` DuckDB table.  Generates BUY signals when the NLP
    profile of a recent filing is bullish, and SELL/CLOSE signals when hedging
    language dominates.

    **How to supply NLP data**

    Pass ``nlp_scores_df`` to ``generate_signals()`` via the strategy's
    ``parameters["nlp_scores"]`` key at construction time, or call
    ``set_nlp_scores()`` after construction.  The DataFrame must have at least
    these columns:

        doc_id          VARCHAR  (e.g. "AAPL_2024-01-15")
        sentence_text   VARCHAR
        forward_looking BOOLEAN
        causal          BOOLEAN

    **Parameters (in StrategyConfig.parameters)**

    - ``symbols`` : list[str] — tickers to trade (default: all in indicators_df)
    - ``forward_looking_threshold`` : float — min ratio to trigger BUY (default 0.4)
    - ``causal_threshold`` : float — min causal ratio for BUY (default 0.2)
    - ``hedging_threshold`` : float — max hedging density before SELL (default 0.35)
    - ``readability_threshold`` : float — min readability score for BUY (default 0.3)
    - ``lookback_days`` : int — max days before as_of_date to search for filings (default 30)
    - ``nlp_scores`` : pl.DataFrame | None — NLP scores dataframe (can be set later)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._nlp_df: pl.DataFrame | None = config.parameters.get("nlp_scores")

    def set_nlp_scores(self, nlp_df: pl.DataFrame) -> None:
        """Set the NLP scores dataframe after construction."""
        self._nlp_df = nlp_df

    def generate_signals(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate BUY/SELL signals based on NLP scores + price confirmation.

        BUY conditions (all must hold):
        - Most recent filing within lookback_days has:
          - forward_looking_ratio >= forward_looking_threshold
          - causal_ratio >= causal_threshold
          - hedging_density < hedging_threshold
          - readability_score >= readability_threshold
        - Price is available for the symbol

        SELL/CLOSE conditions:
        - Position held AND most recent filing has hedging_density >= hedging_threshold

        If no NLP data is available for a symbol, no signal is generated
        (strategy is silent, not erroneous).
        """
        if self._nlp_df is None or len(self._nlp_df) == 0:
            logger.debug(
                "NlpSignalStrategy: no NLP scores loaded — no signals on %s",
                as_of_date,
            )
            return []

        params = self.config.parameters
        forward_threshold: float = params.get("forward_looking_threshold", 0.4)
        causal_threshold: float = params.get("causal_threshold", 0.2)
        hedging_threshold: float = params.get("hedging_threshold", 0.35)
        readability_threshold: float = params.get("readability_threshold", 0.3)
        lookback_days: int = params.get("lookback_days", 30)

        # Symbols to consider: explicit list or all in indicators
        symbols: list[str] = params.get("symbols") or sorted(
            indicators_df.select("symbol").unique().to_series().to_list()
        )

        signals: list[TradeSignal] = []

        for symbol in symbols:
            if symbol not in prices or prices[symbol] <= 0:
                continue

            # Find most recent NLP filing within lookback window
            window_start = as_of_date - timedelta(days=lookback_days)
            scores = self._find_latest_scores(symbol, as_of_date, window_start)

            if scores is None:
                logger.debug(
                    "NlpSignalStrategy: no NLP doc for %s in [%s, %s]",
                    symbol,
                    window_start,
                    as_of_date,
                )
                continue

            has_position = symbol in portfolio.positions
            close = prices[symbol]

            # SELL: hedging dominates → exit existing position
            if has_position and scores.hedging_density >= hedging_threshold:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.MEDIUM,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            f"NLP: hedging_density={scores.hedging_density:.2f} "
                            f">= threshold={hedging_threshold:.2f} "
                            f"(doc_date={scores.doc_date})"
                        ),
                    )
                )
                continue

            # BUY: positive NLP profile + not already positioned
            if (
                not has_position
                and len(portfolio.positions) < self.config.max_positions
                and scores.forward_looking_ratio >= forward_threshold
                and scores.causal_ratio >= causal_threshold
                and scores.hedging_density < hedging_threshold
                and scores.readability_score >= readability_threshold
            ):
                conviction = (
                    Conviction.HIGH
                    if scores.forward_looking_ratio >= forward_threshold + 0.1
                    else Conviction.MEDIUM
                )
                stop_loss = close * (1.0 - self.config.stop_loss_pct)
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.BUY,
                        conviction=conviction,
                        target_weight=self.config.target_position_weight,
                        stop_loss=stop_loss,
                        reasoning=(
                            f"NLP: forward={scores.forward_looking_ratio:.2f} "
                            f"causal={scores.causal_ratio:.2f} "
                            f"hedging={scores.hedging_density:.2f} "
                            f"readability={scores.readability_score:.2f} "
                            f"(doc_date={scores.doc_date}, n={scores.sentence_count})"
                        ),
                    )
                )

        return signals

    def _find_latest_scores(
        self,
        ticker: str,
        as_of_date: date,
        window_start: date,
    ) -> NlpScoreRow | None:
        """Find the most recent NLP filing for *ticker* within [window_start, as_of_date].

        Searches doc_id column for entries matching ``{ticker}_YYYY-MM-DD`` pattern
        and returns the aggregate scores for the most recent one found.
        """
        if self._nlp_df is None:
            return None

        # Filter to rows where doc_id starts with ticker prefix
        ticker_rows = self._nlp_df.filter(
            pl.col("doc_id").str.starts_with(f"{ticker}_")
        )
        if len(ticker_rows) == 0:
            return None

        # Extract date from doc_id (format: TICKER_YYYY-MM-DD)
        # and find filings in [window_start, as_of_date]
        ticker_rows = ticker_rows.with_columns(
            pl.col("doc_id")
            .str.extract(r"_(\d{4}-\d{2}-\d{2})$", 1)
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("_doc_date")
        ).filter(
            (pl.col("_doc_date") >= window_start)
            & (pl.col("_doc_date") <= as_of_date)
        )
        if len(ticker_rows) == 0:
            return None

        # Use the most recent doc date
        latest_date = ticker_rows.select("_doc_date").max().item()
        if latest_date is None:
            return None

        return aggregate_nlp_scores(self._nlp_df, ticker, latest_date)

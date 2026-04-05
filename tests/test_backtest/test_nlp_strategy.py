"""Tests for NlpSignalStrategy — parquet-based NLP signal generation.

Covers:
- Signal generation with known NLP scores
- Lag application (no look-ahead bias)
- Missing data handling (no parquet, missing columns)
- Threshold crossing (entry and exit)
- Registration in STRATEGY_REGISTRY
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from llm_quant.backtest.nlp_signal_strategy import NlpSignalStrategy
from llm_quant.backtest.strategy import StrategyConfig
from llm_quant.brain.models import Action
from llm_quant.trading.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nlp_df(
    n_days: int = 30,
    start_date: date | None = None,
    sentiment_values: list[float] | None = None,
) -> pl.DataFrame:
    """Create a synthetic NLP scores DataFrame.

    If ``sentiment_values`` is provided, it must have length ``n_days``.
    Otherwise, sentiment oscillates between 0.2 and 0.8.
    """
    if start_date is None:
        start_date = date(2024, 1, 1)

    if sentiment_values is not None:
        assert len(sentiment_values) == n_days
    else:
        sentiment_values = [0.8 if i % 2 == 0 else 0.2 for i in range(n_days)]

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "date": dates,
            "forward_looking_score": [0.5] * n_days,
            "hedging_score": [0.1] * n_days,
            "sentiment_score": sentiment_values,
            "i_we_ratio": [1.2] * n_days,
            "readability_score": [0.7] * n_days,
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def _make_indicators_df(
    symbol: str = "AAPL",
    n_days: int = 30,
    start_date: date | None = None,
) -> pl.DataFrame:
    """Create a minimal indicators DataFrame with OHLCV columns."""
    if start_date is None:
        start_date = date(2024, 1, 1)

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n_days,
            "date": dates,
            "open": [100.0] * n_days,
            "high": [101.0] * n_days,
            "low": [99.0] * n_days,
            "close": [100.0] * n_days,
            "volume": [1_000_000] * n_days,
        }
    ).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("volume").cast(pl.Int64),
    )


def _make_portfolio(initial_capital: float = 100_000.0) -> Portfolio:
    """Create a fresh portfolio for testing."""
    return Portfolio(initial_capital=initial_capital)


def _make_config(
    ticker: str = "AAPL",
    signal_column: str = "sentiment_score",
    entry_threshold: float = 0.6,
    exit_threshold: float = 0.3,
    signal_lag: int = 1,
    target_weight: float = 0.80,
    nlp_df_override: pl.DataFrame | None = None,
) -> StrategyConfig:
    """Create a StrategyConfig for NlpSignalStrategy."""
    params: dict = {
        "ticker": ticker,
        "signal_column": signal_column,
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "signal_lag": signal_lag,
        "target_weight": target_weight,
    }
    if nlp_df_override is not None:
        params["nlp_df_override"] = nlp_df_override
    return StrategyConfig(
        name="test_nlp",
        max_positions=10,
        target_position_weight=0.05,
        stop_loss_pct=0.05,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Tests: Signal generation with known NLP scores
# ---------------------------------------------------------------------------


class TestNlpSignalGeneration:
    """Test basic signal generation from NLP scores."""

    def test_buy_signal_when_score_above_entry_threshold(self) -> None:
        """Score above entry_threshold should generate a BUY signal."""
        # All sentiment scores = 0.8, entry_threshold = 0.6
        nlp_df = _make_nlp_df(n_days=10, sentiment_values=[0.8] * 10)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=10)
        prices = {"AAPL": 100.0}

        # as_of_date = day 5 (with lag=1, uses score from day 4)
        as_of_date = date(2024, 1, 6)
        signals = strategy.generate_signals(as_of_date, indicators, portfolio, prices)

        assert len(signals) == 1
        assert signals[0].action == Action.BUY
        assert signals[0].symbol == "AAPL"
        assert signals[0].target_weight == 0.80

    def test_no_signal_when_score_between_thresholds(self) -> None:
        """Score between exit and entry thresholds produces no signal."""
        # All scores = 0.5 (between exit=0.3 and entry=0.6)
        nlp_df = _make_nlp_df(n_days=10, sentiment_values=[0.5] * 10)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=10)
        prices = {"AAPL": 100.0}

        as_of_date = date(2024, 1, 6)
        signals = strategy.generate_signals(as_of_date, indicators, portfolio, prices)

        assert len(signals) == 0

    def test_exit_signal_when_score_below_exit_threshold(self) -> None:
        """Score below exit_threshold should close existing position."""
        # All scores = 0.1 (below exit=0.3)
        nlp_df = _make_nlp_df(n_days=10, sentiment_values=[0.1] * 10)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        # Simulate an existing position
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", shares=10, avg_cost=100.0, current_price=100.0
        )
        indicators = _make_indicators_df(n_days=10)
        prices = {"AAPL": 100.0}

        as_of_date = date(2024, 1, 6)
        signals = strategy.generate_signals(as_of_date, indicators, portfolio, prices)

        assert len(signals) == 1
        assert signals[0].action == Action.CLOSE
        assert signals[0].symbol == "AAPL"
        assert signals[0].target_weight == 0.0

    def test_no_buy_when_already_positioned(self) -> None:
        """Should not generate BUY when already holding a position."""
        nlp_df = _make_nlp_df(n_days=10, sentiment_values=[0.8] * 10)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", shares=10, avg_cost=100.0, current_price=100.0
        )
        indicators = _make_indicators_df(n_days=10)
        prices = {"AAPL": 100.0}

        as_of_date = date(2024, 1, 6)
        signals = strategy.generate_signals(as_of_date, indicators, portfolio, prices)

        # Score is 0.8 > entry but position exists, so no BUY.
        # Score is 0.8 > exit threshold, so no CLOSE either.
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: Lag application (no look-ahead bias)
# ---------------------------------------------------------------------------


class TestLagApplication:
    """Verify signal_lag prevents look-ahead bias."""

    def test_lag_shifts_signal_correctly(self) -> None:
        """With lag=1, the signal on day T uses the score from day T-1."""
        # Day 0: sentiment=0.1 (low), Day 1: sentiment=0.9 (high)
        # With lag=1, day 1 should see day 0's score (0.1), not day 1's (0.9)
        nlp_df = _make_nlp_df(
            n_days=5,
            sentiment_values=[0.1, 0.9, 0.1, 0.9, 0.1],
        )
        config = _make_config(signal_lag=1, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        # On day 1 (2024-01-02), with lag=1, sees score from day 0 (0.1)
        # 0.1 < entry_threshold (0.6) → no BUY
        signals = strategy.generate_signals(
            date(2024, 1, 2), indicators, portfolio, prices
        )
        assert len(signals) == 0

        # On day 2 (2024-01-03), with lag=1, sees score from day 1 (0.9)
        # 0.9 >= entry_threshold (0.6) → BUY
        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].action == Action.BUY

    def test_lag_2_shifts_by_two_days(self) -> None:
        """With lag=2, day T uses the score from day T-2."""
        nlp_df = _make_nlp_df(
            n_days=5,
            sentiment_values=[0.9, 0.1, 0.1, 0.1, 0.1],
        )
        config = _make_config(signal_lag=2, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        # Day 2 (2024-01-03), lag=2 → sees score from day 0 (0.9) → BUY
        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].action == Action.BUY

        # Day 3 (2024-01-04), lag=2 → sees score from day 1 (0.1) → no BUY
        signals = strategy.generate_signals(
            date(2024, 1, 4), indicators, portfolio, prices
        )
        assert len(signals) == 0

    def test_minimum_lag_is_one(self) -> None:
        """signal_lag=0 should be clamped to 1 (minimum)."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.1, 0.9, 0.1, 0.1, 0.1])
        config = _make_config(signal_lag=0, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        # Internal lag should be clamped to 1
        assert strategy._signal_lag == 1

    def test_first_day_has_no_signal_due_to_lag(self) -> None:
        """On the first day, no lagged data is available → no signal."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.9] * 5)
        config = _make_config(signal_lag=1, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        # Day 0: lag=1 means we need data from day -1, which doesn't exist
        signals = strategy.generate_signals(
            date(2024, 1, 1), indicators, portfolio, prices
        )
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: Missing data handling
# ---------------------------------------------------------------------------


class TestMissingDataHandling:
    """Test graceful handling of missing NLP data."""

    def test_no_parquet_returns_neutral(self, tmp_path: object) -> None:
        """If no NLP parquet exists, return empty signals (not error)."""
        from pathlib import Path

        empty_dir = Path(str(tmp_path)) / "empty_nlp"
        empty_dir.mkdir()
        config = _make_config(
            ticker="NONEXISTENT",
            nlp_df_override=None,
        )
        # Point to an empty directory (no parquet files)
        config.parameters["nlp_dir"] = str(empty_dir)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"NONEXISTENT": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 0

    def test_no_ticker_returns_neutral(self) -> None:
        """If no ticker is configured, return empty signals."""
        config = StrategyConfig(
            name="test_nlp_no_ticker",
            parameters={"ticker": ""},
        )
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 0

    def test_missing_signal_column_returns_neutral(self) -> None:
        """If the signal column is missing from the parquet, return neutral."""
        # Create a DataFrame without the expected signal column
        nlp_df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "forward_looking_score": [0.5, 0.5],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        config = _make_config(
            signal_column="sentiment_score",
            nlp_df_override=nlp_df,
        )
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        # The override DataFrame is missing "sentiment_score"
        # _load_nlp_scores should detect this and return None
        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 0

    def test_no_price_for_ticker_returns_neutral(self) -> None:
        """If prices dict doesn't contain the ticker, return neutral."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.9] * 5)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5, symbol="AAPL")
        prices = {"MSFT": 100.0}  # No AAPL price

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 0

    def test_as_of_date_before_all_data_returns_neutral(self) -> None:
        """If as_of_date is before all NLP data, return neutral."""
        nlp_df = _make_nlp_df(
            n_days=5,
            start_date=date(2024, 6, 1),
            sentiment_values=[0.9] * 5,
        )
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        # as_of_date is way before NLP data starts
        signals = strategy.generate_signals(
            date(2024, 1, 1), indicators, portfolio, prices
        )
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: Threshold crossing
# ---------------------------------------------------------------------------


class TestThresholdCrossing:
    """Test entry and exit threshold logic."""

    def test_exact_entry_threshold_triggers_buy(self) -> None:
        """Score exactly at entry_threshold should trigger BUY."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.6] * 5)
        config = _make_config(entry_threshold=0.6, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].action == Action.BUY

    def test_just_below_entry_threshold_no_signal(self) -> None:
        """Score just below entry_threshold should not trigger BUY."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.599] * 5)
        config = _make_config(entry_threshold=0.6, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 0

    def test_exact_exit_threshold_no_close(self) -> None:
        """Score exactly at exit_threshold should NOT trigger CLOSE (< required)."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.3] * 5)
        config = _make_config(exit_threshold=0.3, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", shares=10, avg_cost=100.0, current_price=100.0
        )
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        # 0.3 is NOT < 0.3, so no CLOSE
        assert len(signals) == 0

    def test_just_below_exit_threshold_triggers_close(self) -> None:
        """Score just below exit_threshold should trigger CLOSE."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.299] * 5)
        config = _make_config(exit_threshold=0.3, nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", shares=10, avg_cost=100.0, current_price=100.0
        )
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].action == Action.CLOSE

    def test_custom_signal_column(self) -> None:
        """Strategy should use the configured signal_column."""
        nlp_df = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(1, 6)],
                "forward_looking_score": [0.9, 0.9, 0.9, 0.9, 0.9],
                "hedging_score": [0.1, 0.1, 0.1, 0.1, 0.1],
                "sentiment_score": [0.1, 0.1, 0.1, 0.1, 0.1],  # Low
                "i_we_ratio": [1.2, 1.2, 1.2, 1.2, 1.2],
                "readability_score": [0.7, 0.7, 0.7, 0.7, 0.7],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

        # Use forward_looking_score (0.9) instead of sentiment_score (0.1)
        config = _make_config(
            signal_column="forward_looking_score",
            entry_threshold=0.6,
            nlp_df_override=nlp_df,
        )
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 4), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].action == Action.BUY

    def test_stop_loss_is_set_on_buy(self) -> None:
        """BUY signal should include a stop loss."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.9] * 5)
        config = _make_config(nlp_df_override=nlp_df)
        config.stop_loss_pct = 0.05
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 4), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].stop_loss == pytest.approx(95.0)  # 100 * (1 - 0.05)


# ---------------------------------------------------------------------------
# Tests: Parquet file I/O (using tmp_path)
# ---------------------------------------------------------------------------


class TestParquetFileIO:
    """Test reading NLP scores from actual parquet files on disk."""

    def test_reads_from_parquet_file(self, tmp_path: object) -> None:
        """Strategy should read NLP scores from a parquet file."""
        from pathlib import Path

        tmp = Path(str(tmp_path))
        nlp_df = _make_nlp_df(n_days=10, sentiment_values=[0.9] * 10)
        parquet_path = tmp / "AAPL.parquet"
        nlp_df.write_parquet(parquet_path)

        config = _make_config(ticker="AAPL")
        config.parameters["nlp_dir"] = str(tmp)
        # Remove override so it reads from file
        config.parameters.pop("nlp_df_override", None)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=10)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 6), indicators, portfolio, prices
        )
        assert len(signals) == 1
        assert signals[0].action == Action.BUY

    def test_missing_parquet_returns_empty(self, tmp_path: object) -> None:
        """Missing parquet file should return empty signals, not error."""
        from pathlib import Path

        tmp = Path(str(tmp_path))
        config = _make_config(ticker="MISSING")
        config.parameters["nlp_dir"] = str(tmp)
        config.parameters.pop("nlp_df_override", None)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5, symbol="MISSING")
        prices = {"MISSING": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 3), indicators, portfolio, prices
        )
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Tests: STRATEGY_REGISTRY
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    """Verify NlpSignalStrategy is registered correctly."""

    def test_nlp_signal_in_registry(self) -> None:
        """'nlp_signal' key should exist in STRATEGY_REGISTRY."""
        from llm_quant.backtest.strategies import STRATEGY_REGISTRY

        assert "nlp_signal" in STRATEGY_REGISTRY

    def test_registry_maps_to_correct_class(self) -> None:
        """'nlp_signal' should map to NlpSignalStrategy."""
        from llm_quant.backtest.strategies import STRATEGY_REGISTRY

        assert STRATEGY_REGISTRY["nlp_signal"] is NlpSignalStrategy

    def test_create_strategy_factory(self) -> None:
        """create_strategy('nlp_signal', ...) should return NlpSignalStrategy."""
        from llm_quant.backtest.strategies import create_strategy

        config = StrategyConfig(
            name="test_nlp",
            parameters={"ticker": "AAPL"},
        )
        strategy = create_strategy("nlp_signal", config)
        assert isinstance(strategy, NlpSignalStrategy)


# ---------------------------------------------------------------------------
# Tests: Signal reasoning content
# ---------------------------------------------------------------------------


class TestSignalReasoning:
    """Verify signal reasoning strings contain useful information."""

    def test_buy_reasoning_contains_score_and_threshold(self) -> None:
        """BUY reasoning should include score value and threshold."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.85] * 5)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 4), indicators, portfolio, prices
        )
        assert len(signals) == 1
        reasoning = signals[0].reasoning
        assert "sentiment_score" in reasoning
        assert "0.850" in reasoning
        assert "entry_threshold" in reasoning

    def test_close_reasoning_contains_score_and_threshold(self) -> None:
        """CLOSE reasoning should include score value and threshold."""
        nlp_df = _make_nlp_df(n_days=5, sentiment_values=[0.15] * 5)
        config = _make_config(nlp_df_override=nlp_df)
        strategy = NlpSignalStrategy(config)
        portfolio = _make_portfolio()
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL", shares=10, avg_cost=100.0, current_price=100.0
        )
        indicators = _make_indicators_df(n_days=5)
        prices = {"AAPL": 100.0}

        signals = strategy.generate_signals(
            date(2024, 1, 4), indicators, portfolio, prices
        )
        assert len(signals) == 1
        reasoning = signals[0].reasoning
        assert "sentiment_score" in reasoning
        assert "exit_threshold" in reasoning

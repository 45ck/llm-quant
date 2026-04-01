"""Walk-forward validation engine.

Two modes:
- Anchored expanding-window: train on [0, T], test on [T, T+step], advance T by step
- Rolling-window: train on [T-window, T], test on [T, T+step], advance by step

For each fold: runs BacktestEngine.run() on train period, then test period.
Collects OOS equity curves and computes aggregate metrics.

Gate: WF-OOS SR / IS SR > 0.60
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes
    ----------
    mode : str
        "anchored" (expanding window) or "rolling" (fixed-size window).
    n_splits : int
        Number of OOS folds to generate.
    train_pct : float
        Fraction of total period used for the initial training window.
        Only meaningful for anchored mode; for rolling mode the train window
        is computed as total_days * train_pct.
    step_pct : float
        Step size as a fraction of total period.  Each fold advances the
        test window by step_pct * total_days calendar days.
    min_train_days : int
        Minimum calendar days in a training window.  Folds with less data
        are skipped with a warning.
    min_test_days : int
        Minimum calendar days in a test window.  Folds with less data are
        skipped with a warning.
    """

    mode: str = "anchored"
    n_splits: int = 5
    train_pct: float = 0.7
    step_pct: float = 0.1
    min_train_days: int = 252
    min_test_days: int = 63


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """Results for a single IS/OOS fold pair.

    Attributes
    ----------
    fold_idx : int
        Zero-based fold index.
    train_start, train_end : date
        In-sample window bounds (inclusive).
    test_start, test_end : date
        Out-of-sample window bounds (inclusive).
    is_sharpe : float
        Annualized Sharpe over the training window.
    oos_sharpe : float
        Annualized Sharpe over the test window.
    oos_cagr : float
        Annualized return over the test window.
    oos_max_dd : float
        Maximum drawdown fraction over the test window.
    oos_n_trades : int
        Number of trades executed during the test window.
    """

    fold_idx: int = 0
    train_start: date = field(default_factory=date.today)
    train_end: date = field(default_factory=date.today)
    test_start: date = field(default_factory=date.today)
    test_end: date = field(default_factory=date.today)
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    oos_cagr: float = 0.0
    oos_max_dd: float = 0.0
    oos_n_trades: int = 0


@dataclass
class WalkForwardResult:
    """Aggregate results for a complete walk-forward analysis.

    Attributes
    ----------
    folds : list[WalkForwardFold]
        Per-fold results (only folds that ran successfully).
    oos_sharpe : float
        Mean OOS Sharpe across folds.
    is_sharpe : float
        Mean IS Sharpe across folds.
    oos_is_ratio : float
        oos_sharpe / is_sharpe.  Values near 1.0 indicate low overfitting.
    oos_cagr : float
        Mean OOS CAGR across folds.
    oos_max_dd : float
        Mean OOS max drawdown across folds.
    passes_gate : bool
        True when oos_is_ratio > 0.60.
    n_folds : int
        Number of folds that completed successfully.
    """

    folds: list[WalkForwardFold] = field(default_factory=list)
    oos_sharpe: float = 0.0
    is_sharpe: float = 0.0
    oos_is_ratio: float = 0.0
    oos_cagr: float = 0.0
    oos_max_dd: float = 0.0
    passes_gate: bool = False
    n_folds: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WalkForwardEngine:
    """Walk-forward validation engine.

    Splits a historical date range into IS/OOS fold pairs, runs a
    BacktestEngine on each pair, and aggregates the OOS results to
    measure out-of-sample robustness.

    The OOS/IS Sharpe ratio gate is set at 0.60: strategies that
    degrade severely OOS (ratio < 0.60) are flagged as over-fit.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy as registered in STRATEGY_REGISTRY.
    config : WalkForwardConfig
        Walk-forward parameters (mode, splits, train/step fractions, …).
    """

    WF_GATE = 0.60  # OOS/IS Sharpe ratio threshold

    def __init__(self, strategy_name: str, config: WalkForwardConfig) -> None:
        self.strategy_name = strategy_name
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_folds(self, start_date: date, end_date: date) -> list[WalkForwardFold]:
        """Generate IS/OOS fold date ranges without running any backtest.

        Parameters
        ----------
        start_date : date
            First calendar day of the full analysis period.
        end_date : date
            Last calendar day of the full analysis period.

        Returns
        -------
        list[WalkForwardFold]
            Fold objects with date ranges filled in (IS/OOS metrics left at 0).
        """
        cfg = self.config
        total_days = (end_date - start_date).days
        if total_days <= 0:
            logger.error("start_date must be before end_date")
            return []

        step_days = max(1, int(total_days * cfg.step_pct))
        train_days = max(cfg.min_train_days, int(total_days * cfg.train_pct))

        folds: list[WalkForwardFold] = []
        for i in range(cfg.n_splits):
            # Test window
            test_start = start_date + timedelta(days=train_days + i * step_days)
            test_end = test_start + timedelta(days=step_days - 1)

            # Clamp test_end to overall end_date
            test_end = min(test_end, end_date)

            # Test must have enough data
            test_span = (test_end - test_start).days + 1
            if test_span < cfg.min_test_days:
                logger.warning(
                    "Fold %d: test window only %d days (min %d) — skipping",
                    i,
                    test_span,
                    cfg.min_test_days,
                )
                continue

            # Training window
            if cfg.mode == "rolling":
                train_start = test_start - timedelta(days=train_days)
            else:
                # anchored: always start from the global start
                train_start = start_date

            train_end = test_start - timedelta(days=1)

            # Train must have enough data
            train_span = (train_end - train_start).days + 1
            if train_span < cfg.min_train_days:
                logger.warning(
                    "Fold %d: training window only %d days (min %d) — skipping",
                    i,
                    train_span,
                    cfg.min_train_days,
                )
                continue

            fold = WalkForwardFold(
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

            # Stop once test window hits the end of history
            if test_end >= end_date:
                break

        return folds

    def run(
        self,
        strategy_name: str,
        start_date: date,
        end_date: date,
        strategy_params: dict[str, Any],
    ) -> WalkForwardResult:
        """Run a complete walk-forward analysis.

        Fetches data once for the full period, then slices it per fold so
        that each BacktestEngine run sees only in-period data.

        Parameters
        ----------
        strategy_name : str
            Strategy registry key (may differ from self.strategy_name if
            the caller overrides).
        start_date : date
            First day of the full analysis window.
        end_date : date
            Last day of the full analysis window.
        strategy_params : dict
            Parameter overrides forwarded to StrategyConfig.

        Returns
        -------
        WalkForwardResult
            Aggregate OOS metrics across all completed folds.
        """
        # Lazy import to avoid circular dependency at module load time
        from llm_quant.data.fetcher import fetch_ohlcv
        from llm_quant.data.indicators import compute_indicators

        folds = self.generate_folds(start_date, end_date)
        if not folds:
            logger.error("No valid folds generated — check date range and config")
            return WalkForwardResult()

        # Determine universe from strategy_params or fall back to a default
        symbols: list[str] = strategy_params.get(
            "symbols", ["SPY", "QQQ", "TLT", "GLD", "IEF"]
        )

        # Fetch enough data to cover the full window (plus indicator warmup)
        lookback_days = (end_date - start_date).days + 400  # 400-day warmup buffer
        logger.info(
            "Fetching %d days of data for %d symbols …", lookback_days, len(symbols)
        )
        prices_df = fetch_ohlcv(symbols, lookback_days=lookback_days)
        if len(prices_df) == 0:
            logger.error("No price data returned")
            return WalkForwardResult()

        indicators_df = compute_indicators(prices_df)

        completed_folds: list[WalkForwardFold] = []
        for fold in folds:
            completed = self._run_fold(
                fold=fold,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                prices_df=prices_df,
                indicators_df=indicators_df,
            )
            if completed is not None:
                completed_folds.append(completed)

        if not completed_folds:
            logger.error("All folds failed — no OOS results to aggregate")
            return WalkForwardResult()

        return self._aggregate(completed_folds)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold: WalkForwardFold,
        strategy_name: str,
        strategy_params: dict[str, Any],
        prices_df: pl.DataFrame,
        indicators_df: pl.DataFrame,
    ) -> WalkForwardFold | None:
        """Run IS and OOS backtests for a single fold.

        Parameters
        ----------
        fold : WalkForwardFold
            Fold with date ranges already set.
        strategy_name : str
            Strategy registry key.
        strategy_params : dict
            Strategy parameter overrides.
        prices_df : pl.DataFrame
            Full price history (will be filtered per fold).
        indicators_df : pl.DataFrame
            Full indicator history (will be filtered per fold).

        Returns
        -------
        WalkForwardFold | None
            Completed fold with IS/OOS metrics, or None on failure.
        """
        from llm_quant.backtest.engine import BacktestEngine, CostModel
        from llm_quant.backtest.strategies import create_strategy
        from llm_quant.backtest.strategy import StrategyConfig

        logger.info(
            "Fold %d: IS [%s → %s]  OOS [%s → %s]",
            fold.fold_idx,
            fold.train_start,
            fold.train_end,
            fold.test_start,
            fold.test_end,
        )

        # Build strategy config
        params = {k: v for k, v in strategy_params.items() if k != "symbols"}
        config = StrategyConfig(
            name=strategy_name,
            rebalance_frequency_days=params.get("rebalance_frequency_days", 5),
            max_positions=params.get("max_positions", 10),
            target_position_weight=params.get("target_position_weight", 0.05),
            stop_loss_pct=params.get("stop_loss_pct", 0.05),
            parameters=params,
        )

        try:
            strategy_is = create_strategy(strategy_name, config)
            strategy_oos = create_strategy(strategy_name, config)
        except Exception as exc:
            logger.error("Fold %d: failed to create strategy — %s", fold.fold_idx, exc)
            return None

        # Slice data for each window
        is_prices = _slice_df(prices_df, fold.train_start, fold.train_end)
        is_indicators = _slice_df(indicators_df, fold.train_start, fold.train_end)
        oos_prices = _slice_df(prices_df, fold.test_start, fold.test_end)
        oos_indicators = _slice_df(indicators_df, fold.test_start, fold.test_end)

        if len(is_prices) == 0 or len(oos_prices) == 0:
            logger.warning("Fold %d: empty data slice — skipping", fold.fold_idx)
            return None

        cost_model = CostModel()
        slug = f"wf_{strategy_name}_fold{fold.fold_idx}"

        # --- IS run ---
        is_engine = BacktestEngine(strategy=strategy_is, initial_capital=100_000.0)
        try:
            is_result = is_engine.run(
                prices_df=is_prices,
                indicators_df=is_indicators,
                slug=f"{slug}_is",
                cost_model=cost_model,
            )
        except Exception as exc:
            logger.error("Fold %d IS run failed: %s", fold.fold_idx, exc)
            return None

        is_metrics = is_result.metrics.get("1x")
        if is_metrics is None:
            logger.warning("Fold %d: no IS metrics at 1x cost", fold.fold_idx)
            is_sharpe = 0.0
        else:
            is_sharpe = is_metrics.sharpe_ratio

        # --- OOS run ---
        oos_engine = BacktestEngine(strategy=strategy_oos, initial_capital=100_000.0)
        try:
            oos_result = oos_engine.run(
                prices_df=oos_prices,
                indicators_df=oos_indicators,
                slug=f"{slug}_oos",
                cost_model=cost_model,
            )
        except Exception as exc:
            logger.error("Fold %d OOS run failed: %s", fold.fold_idx, exc)
            return None

        oos_metrics = oos_result.metrics.get("1x")
        if oos_metrics is None:
            logger.warning("Fold %d: no OOS metrics at 1x cost", fold.fold_idx)
            oos_sharpe = 0.0
            oos_cagr = 0.0
            oos_max_dd = 0.0
            oos_n_trades = 0
        else:
            oos_sharpe = oos_metrics.sharpe_ratio
            oos_cagr = oos_metrics.annualized_return
            oos_max_dd = oos_metrics.max_drawdown
            oos_n_trades = oos_metrics.total_trades

        fold.is_sharpe = is_sharpe
        fold.oos_sharpe = oos_sharpe
        fold.oos_cagr = oos_cagr
        fold.oos_max_dd = oos_max_dd
        fold.oos_n_trades = oos_n_trades

        logger.info(
            "Fold %d done: IS SR=%.2f  OOS SR=%.2f  OOS CAGR=%.1f%%  OOS MaxDD=%.1f%%",
            fold.fold_idx,
            is_sharpe,
            oos_sharpe,
            oos_cagr * 100,
            oos_max_dd * 100,
        )
        return fold

    def _aggregate(self, folds: list[WalkForwardFold]) -> WalkForwardResult:
        """Compute aggregate statistics from completed folds."""
        n = len(folds)
        mean_oos_sharpe = sum(f.oos_sharpe for f in folds) / n
        mean_is_sharpe = sum(f.is_sharpe for f in folds) / n
        mean_oos_cagr = sum(f.oos_cagr for f in folds) / n
        mean_oos_max_dd = sum(f.oos_max_dd for f in folds) / n

        if mean_is_sharpe != 0:
            oos_is_ratio = mean_oos_sharpe / mean_is_sharpe
        else:
            oos_is_ratio = 0.0

        passes_gate = oos_is_ratio > self.WF_GATE

        return WalkForwardResult(
            folds=folds,
            oos_sharpe=mean_oos_sharpe,
            is_sharpe=mean_is_sharpe,
            oos_is_ratio=oos_is_ratio,
            oos_cagr=mean_oos_cagr,
            oos_max_dd=mean_oos_max_dd,
            passes_gate=passes_gate,
            n_folds=n,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slice_df(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Filter a DataFrame to rows where 'date' is in [start, end]."""
    return df.filter((pl.col("date") >= start) & (pl.col("date") <= end))


# ---------------------------------------------------------------------------
# CLI formatting helpers
# ---------------------------------------------------------------------------


def _print_fold_table(folds: list[WalkForwardFold]) -> None:
    """Print a summary table of fold results to stdout."""
    header = (
        f"{'Fold':>4}  {'Train Start':>12}  {'Train End':>12}  "
        f"{'Test Start':>12}  {'Test End':>12}  "
        f"{'IS SR':>7}  {'OOS SR':>7}  {'OOS CAGR':>9}  {'OOS MaxDD':>9}  {'Trades':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for f in folds:
        print(
            f"{f.fold_idx:>4}  {f.train_start!s:>12}  {f.train_end!s:>12}  "
            f"{f.test_start!s:>12}  {f.test_end!s:>12}  "
            f"{f.is_sharpe:>7.2f}  {f.oos_sharpe:>7.2f}  "
            f"{f.oos_cagr * 100:>8.1f}%  {f.oos_max_dd * 100:>8.1f}%  {f.oos_n_trades:>6}"
        )
    print(sep)


def _print_aggregate(result: WalkForwardResult) -> None:
    """Print aggregate results and gate verdict."""
    print()
    print(f"  Folds completed : {result.n_folds}")
    print(f"  Mean IS Sharpe  : {result.is_sharpe:.3f}")
    print(f"  Mean OOS Sharpe : {result.oos_sharpe:.3f}")
    print(
        f"  OOS/IS ratio    : {result.oos_is_ratio:.3f}  (gate: > {WalkForwardEngine.WF_GATE})"
    )
    print(f"  Mean OOS CAGR   : {result.oos_cagr * 100:.1f}%")
    print(f"  Mean OOS MaxDD  : {result.oos_max_dd * 100:.1f}%")
    print()
    verdict = "PASS" if result.passes_gate else "FAIL"
    print(f"  Walk-forward gate: {verdict}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation on a strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", required=True, help="Strategy registry key")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date YYYY-MM-DD (default: 5 years ago)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--mode",
        choices=["anchored", "rolling"],
        default="anchored",
        help="Walk-forward mode",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of OOS folds",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.7,
        help="Fraction of total period for initial train window",
    )
    parser.add_argument(
        "--step-pct",
        type=float,
        default=0.1,
        help="Step size as fraction of total period",
    )
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,TLT,GLD,IEF",
        help="Comma-separated list of symbols",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args = parser.parse_args()

    today = date.today()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else date(today.year - 5, today.month, today.day)
    )

    cfg = WalkForwardConfig(
        mode=args.mode,
        n_splits=args.n_splits,
        train_pct=args.train_pct,
        step_pct=args.step_pct,
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    strategy_params: dict[str, Any] = {"symbols": symbols}

    engine = WalkForwardEngine(strategy_name=args.strategy, config=cfg)

    print(f"\nWalk-Forward Validation — {args.strategy}")
    print(
        f"Mode: {args.mode}  |  Folds: {args.n_splits}  |  Period: {start_date} → {end_date}"
    )
    print()

    result = engine.run(
        strategy_name=args.strategy,
        start_date=start_date,
        end_date=end_date,
        strategy_params=strategy_params,
    )

    if result.n_folds == 0:
        print("ERROR: no folds completed — check date range, data, and strategy name.")
        sys.exit(1)

    _print_fold_table(result.folds)
    _print_aggregate(result)

    sys.exit(0 if result.passes_gate else 1)

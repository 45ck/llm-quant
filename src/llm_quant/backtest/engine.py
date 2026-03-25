"""Backtest engine with fill-delay, cost model, and stop-loss enforcement.

The engine loop processes one trading day at a time, ensuring strict
temporal ordering (no look-ahead). Indicators are pre-computed once
using causal operations, then sliced per-day for the strategy.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from llm_quant.backtest.artifacts import (
    ExperimentRegistry,
    hash_content,
    strategy_dir,
)
from llm_quant.backtest.metrics import (
    BacktestMetrics,
    compute_all_metrics,
    compute_benchmark_returns,
)
from llm_quant.backtest.strategy import Strategy
from llm_quant.brain.models import Action, Conviction, TradeSignal
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Transaction cost model with square-root market impact.

    When volume data is available:
      impact = slippage_volatility_factor * daily_vol * sqrt(shares / daily_volume)
      cost_per_trade = notional * (spread_bps/10000 + impact)

    When volume is unavailable:
      cost_per_trade = notional * (spread_bps + flat_slippage_bps) / 10000
    """

    spread_bps: float = 5.0
    slippage_volatility_factor: float = 0.5
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    flat_slippage_bps: float = 2.0

    def compute_cost(
        self,
        notional: float,
        shares: float,
        daily_volume: float | None = None,
        daily_volatility: float | None = None,
        multiplier: float = 1.0,
    ) -> float:
        """Compute total transaction cost for a trade.

        Parameters
        ----------
        notional : float
            Absolute dollar value of the trade.
        shares : float
            Number of shares traded.
        daily_volume : float | None
            Average daily volume in shares.
        daily_volatility : float | None
            Daily return volatility (std dev).
        multiplier : float
            Cost multiplier for stress testing (1.0, 1.5, 2.0, 3.0).

        Returns
        -------
        float
            Total cost in dollars.
        """
        if notional == 0:
            return 0.0

        # Spread cost
        spread_cost = notional * self.spread_bps / 10_000.0

        # Market impact
        if (
            daily_volume is not None
            and daily_volume > 0
            and daily_volatility is not None
            and daily_volatility > 0
            and abs(shares) > 0
        ):
            impact = (
                self.slippage_volatility_factor
                * daily_volatility
                * math.sqrt(abs(shares) / daily_volume)
            )
            impact_cost = notional * impact
        else:
            impact_cost = notional * self.flat_slippage_bps / 10_000.0

        # Commission
        commission = max(
            abs(shares) * self.commission_per_share,
            self.min_commission,
        )

        return (spread_cost + impact_cost + commission) * multiplier

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> CostModel:
        """Create a CostModel from research-spec cost_model section."""
        cm = spec.get("cost_model", {})
        return cls(
            spread_bps=cm.get("spread_bps", 5.0),
            slippage_volatility_factor=cm.get("slippage_volatility_factor", 0.5),
            commission_per_share=cm.get("commission_per_share", 0.0),
            min_commission=cm.get("min_commission", 0.0),
            flat_slippage_bps=cm.get("flat_slippage_bps", 2.0),
        )


# ---------------------------------------------------------------------------
# Snapshots and results
# ---------------------------------------------------------------------------


@dataclass
class DailySnapshot:
    """State of the portfolio at end of a trading day."""

    date: date
    nav: float
    cash: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    trades_today: int = 0


@dataclass
class TradeRecord:
    """Record of a single executed trade in the backtest."""

    date: date
    symbol: str
    action: str
    shares: float
    price: float
    notional: float
    cost: float
    pnl: float = 0.0
    reasoning: str = ""


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    experiment_id: str
    strategy_name: str
    slug: str
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 100_000.0

    # Metrics at different cost multipliers
    metrics: dict[str, BacktestMetrics] = field(default_factory=dict)

    # Raw data
    snapshots: list[DailySnapshot] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    nav_series: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    # Config used
    cost_model: CostModel | None = None
    spec_hash: str = ""
    trial_number: int = 0

    # Data quality
    symbols_used: list[str] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven backtest engine with fill-delay and cost modeling.

    The engine enforces:
    1. Frozen research spec before backtesting
    2. Causal indicator computation (via compute_indicators)
    3. Fill delay (default: T+1 open)
    4. Square-root impact cost model
    5. Stop-loss enforcement at each day's close
    6. Cost multiplier stress tests (1x, 1.5x, 2x, 3x)
    7. Append-only experiment registry
    """

    def __init__(
        self,
        strategy: Strategy,
        data_dir: str | None = None,
        initial_capital: float = 100_000.0,
        *,
        risk_checks_enabled: bool = False,
        risk_manager: Any = None,
    ) -> None:
        self.strategy = strategy
        self.data_dir = data_dir or "data"
        self.initial_capital = initial_capital
        self.risk_checks_enabled = risk_checks_enabled
        self.risk_manager = risk_manager

    def run(
        self,
        prices_df: pl.DataFrame,
        indicators_df: pl.DataFrame,
        slug: str,
        cost_model: CostModel | None = None,
        fill_delay: int = 1,
        warmup_days: int = 200,
        cost_multiplier: float = 1.0,
        benchmark_weights: dict[str, float] | None = None,
        benchmark_rebalance_days: int = 21,
        trial_count: int | None = None,
    ) -> BacktestResult:
        """Run a single backtest pass.

        Parameters
        ----------
        prices_df : pl.DataFrame
            Full OHLCV data with columns: symbol, date, open, high, low, close,
            volume, adj_close.
        indicators_df : pl.DataFrame
            Pre-computed indicators (output of compute_indicators).
        slug : str
            Strategy slug for artifact storage.
        cost_model : CostModel | None
            Transaction cost model. Uses defaults if None.
        fill_delay : int
            Number of days between signal and execution (0=same day, 1=next day).
        warmup_days : int
            Number of trading days to skip for indicator warmup.
        cost_multiplier : float
            Cost multiplier for stress testing.
        benchmark_weights : dict[str, float] | None
            Benchmark portfolio weights for comparison.
        benchmark_rebalance_days : int
            Benchmark rebalance frequency.
        trial_count : int | None
            Override trial count for DSR. If None, reads from registry.

        Returns
        -------
        BacktestResult
            Complete backtest output with metrics.
        """
        if cost_model is None:
            cost_model = CostModel()

        experiment_id = str(uuid.uuid4())[:8]

        # Get sorted unique trading dates
        all_dates = sorted(prices_df.select("date").unique().to_series().to_list())
        if len(all_dates) <= warmup_days:
            logger.warning(
                "Not enough dates (%d) for warmup (%d)",
                len(all_dates),
                warmup_days,
            )
            return BacktestResult(
                experiment_id=experiment_id,
                strategy_name=self.strategy.config.name,
                slug=slug,
                data_warnings=["Insufficient data for warmup period"],
            )

        trading_dates = all_dates[warmup_days:]
        start_date = trading_dates[0]
        end_date = trading_dates[-1]

        # Check data quality
        data_warnings = self._check_data_quality(prices_df, trading_dates)
        symbols_used = sorted(prices_df.select("symbol").unique().to_series().to_list())

        # Initialize portfolio
        portfolio = Portfolio(initial_capital=self.initial_capital)

        # Compute volume stats for cost model
        volume_stats = self._compute_volume_stats(prices_df)

        # Pre-compute daily volatility per symbol
        vol_stats = self._compute_volatility_stats(prices_df)

        # Run the daily loop
        snapshots: list[DailySnapshot] = []
        trades: list[TradeRecord] = []
        nav_series: list[float] = [self.initial_capital]
        pending_signals: list[tuple[date, list[TradeSignal]]] = []
        rebalance_counter = 0
        # O(1) date-to-index lookup for fill-delay processing
        date_index_map = {d: idx for idx, d in enumerate(trading_dates)}

        for i, current_date in enumerate(trading_dates):
            # 4a. Get today's prices
            today_prices = self._get_prices_for_date(prices_df, current_date)
            if not today_prices:
                # D5: carry forward previous NAV so nav_series stays aligned
                nav_series.append(nav_series[-1])
                snapshots.append(
                    DailySnapshot(
                        date=current_date,
                        nav=nav_series[-1],
                        cash=portfolio.cash,
                        gross_exposure=portfolio.gross_exposure,
                        net_exposure=portfolio.net_exposure,
                        n_positions=len(portfolio.positions),
                        trades_today=0,
                    )
                )
                continue

            # 4b. Mark portfolio to market at today's close
            portfolio.update_prices(today_prices)

            # 4c. Check stop-losses at today's close
            stop_signals = self._check_stop_losses(portfolio, today_prices)

            # 4d. Generate signals on rebalance days
            rebalance_freq = self.strategy.config.rebalance_frequency_days
            is_rebalance_day = rebalance_counter % rebalance_freq == 0
            rebalance_counter += 1

            strategy_signals: list[TradeSignal] = []
            if is_rebalance_day:
                # Filter indicators to dates <= current_date (causal)
                causal_indicators = indicators_df.filter(pl.col("date") <= current_date)
                strategy_signals = self.strategy.generate_signals(
                    current_date, causal_indicators, portfolio, today_prices
                )

            # Deduplicate: stop-loss takes priority over strategy for same symbol
            stop_symbols = {s.symbol for s in stop_signals}
            deduped_strategy = [
                s for s in strategy_signals if s.symbol not in stop_symbols
            ]
            all_signals = stop_signals + deduped_strategy

            # 4e. Risk filtering
            if self.risk_checks_enabled and self.risk_manager and all_signals:
                approved, _rejected = self.risk_manager.filter_signals(
                    all_signals, portfolio, today_prices
                )
                all_signals = approved

            # Split: stop-losses execute immediately, strategy signals follow fill delay
            immediate_signals = [s for s in all_signals if s.symbol in stop_symbols]
            delayed_signals = [s for s in all_signals if s.symbol not in stop_symbols]

            # Execute stop-losses immediately at today's close (no fill delay)
            if immediate_signals:
                day_trades = self._execute_signals(
                    immediate_signals,
                    portfolio,
                    today_prices,
                    current_date,
                    cost_model,
                    cost_multiplier,
                    volume_stats,
                    vol_stats,
                )
                trades.extend(day_trades)

            # Handle fill delay for strategy signals
            if fill_delay == 0:
                if delayed_signals:
                    day_trades = self._execute_signals(
                        delayed_signals,
                        portfolio,
                        today_prices,
                        current_date,
                        cost_model,
                        cost_multiplier,
                        volume_stats,
                        vol_stats,
                    )
                    trades.extend(day_trades)
            else:
                # Queue for next day execution
                if delayed_signals:
                    pending_signals.append((current_date, delayed_signals))

                # Execute pending signals from fill_delay days ago
                # Use date-to-index map for O(1) lookup
                new_pending = []
                for signal_date, signals in pending_signals:
                    signal_idx = date_index_map.get(signal_date, -1)
                    if signal_idx >= 0 and i - signal_idx >= fill_delay:
                        # Execute at today's open
                        fill_prices = self._get_open_prices_for_date(
                            prices_df, current_date
                        )
                        if fill_prices:
                            day_trades = self._execute_signals(
                                signals,
                                portfolio,
                                fill_prices,
                                current_date,
                                cost_model,
                                cost_multiplier,
                                volume_stats,
                                vol_stats,
                            )
                            trades.extend(day_trades)
                            # Update prices after execution
                            portfolio.update_prices(today_prices)
                    else:
                        new_pending.append((signal_date, signals))
                pending_signals = new_pending

            # 4h. Record snapshot
            nav = portfolio.nav
            nav_series.append(nav)
            trades_today = sum(1 for t in trades if t.date == current_date)
            snapshots.append(
                DailySnapshot(
                    date=current_date,
                    nav=nav,
                    cash=portfolio.cash,
                    gross_exposure=portfolio.gross_exposure,
                    net_exposure=portfolio.net_exposure,
                    n_positions=len(portfolio.positions),
                    trades_today=trades_today,
                )
            )

        # Compute benchmark returns
        benchmark_returns = None
        if benchmark_weights:
            benchmark_returns = compute_benchmark_returns(
                prices_df,
                benchmark_weights,
                rebalance_frequency_days=benchmark_rebalance_days,
                use_adj_close=True,
            )

        # Get trial count from registry if not provided
        if trial_count is None:
            strat_d = strategy_dir(base_dir=self._resolve_data_dir(), slug=slug)
            registry = ExperimentRegistry(strat_d)
            trial_count = registry.trial_count + 1  # include current run

        # Compute metrics
        trade_dicts = [{"pnl": t.pnl, "notional": t.notional} for t in trades]
        metrics = compute_all_metrics(
            nav_series,
            trade_dicts,
            trial_count=trial_count,
            benchmark_returns=benchmark_returns,
        )
        metrics.warnings.extend(data_warnings)

        # W5: compute spec_hash from strategy config
        spec_hash = hash_content(
            yaml.dump(self.strategy.config.to_dict(), sort_keys=True)
        )

        return BacktestResult(
            experiment_id=experiment_id,
            strategy_name=self.strategy.config.name,
            slug=slug,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            metrics={"1.0x": metrics},
            snapshots=snapshots,
            trades=trades,
            nav_series=nav_series,
            daily_returns=metrics.daily_returns,
            cost_model=cost_model,
            spec_hash=spec_hash,
            trial_number=trial_count,
            symbols_used=symbols_used,
            data_warnings=data_warnings,
        )

    def run_with_cost_sensitivity(
        self,
        prices_df: pl.DataFrame,
        indicators_df: pl.DataFrame,
        slug: str,
        cost_model: CostModel | None = None,
        fill_delay: int = 1,
        warmup_days: int = 200,
        cost_multipliers: list[float] | None = None,
        benchmark_weights: dict[str, float] | None = None,
        benchmark_rebalance_days: int = 21,
    ) -> BacktestResult:
        """Run backtest at multiple cost multipliers.

        Returns a single BacktestResult with metrics for each multiplier.
        """
        if cost_multipliers is None:
            cost_multipliers = [1.0, 1.5, 2.0, 3.0]

        if cost_model is None:
            cost_model = CostModel()

        # Get trial count once
        strat_d = strategy_dir(base_dir=self._resolve_data_dir(), slug=slug)
        registry = ExperimentRegistry(strat_d)
        trial_count = registry.trial_count + 1

        # Run at base cost first
        base_result = self.run(
            prices_df=prices_df,
            indicators_df=indicators_df,
            slug=slug,
            cost_model=cost_model,
            fill_delay=fill_delay,
            warmup_days=warmup_days,
            cost_multiplier=1.0,
            benchmark_weights=benchmark_weights,
            benchmark_rebalance_days=benchmark_rebalance_days,
            trial_count=trial_count,
        )

        # Run at other multipliers
        for mult in cost_multipliers:
            if mult == 1.0:
                continue
            result = self.run(
                prices_df=prices_df,
                indicators_df=indicators_df,
                slug=slug,
                cost_model=cost_model,
                fill_delay=fill_delay,
                warmup_days=warmup_days,
                cost_multiplier=mult,
                benchmark_weights=benchmark_weights,
                benchmark_rebalance_days=benchmark_rebalance_days,
                trial_count=trial_count,
            )
            key = f"{mult}x"
            if result.metrics:
                base_result.metrics[key] = next(iter(result.metrics.values()))

        return base_result

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _resolve_data_dir(self) -> Path:
        return Path(self.data_dir)

    def _get_prices_for_date(self, df: pl.DataFrame, d: date) -> dict[str, float]:
        """Get close prices for all symbols on date *d*."""
        day_data = df.filter(pl.col("date") == d)
        if len(day_data) == 0:
            return {}
        return dict(
            zip(
                day_data.select("symbol").to_series().to_list(),
                day_data.select("close").to_series().to_list(),
                strict=False,
            )
        )

    def _get_open_prices_for_date(self, df: pl.DataFrame, d: date) -> dict[str, float]:
        """Get open prices for all symbols on date *d*."""
        day_data = df.filter(pl.col("date") == d)
        if len(day_data) == 0:
            return {}
        return dict(
            zip(
                day_data.select("symbol").to_series().to_list(),
                day_data.select("open").to_series().to_list(),
                strict=False,
            )
        )

    def _check_stop_losses(
        self, portfolio: Portfolio, prices: dict[str, float]
    ) -> list[TradeSignal]:
        """Check if any positions have breached their stop-loss."""
        signals: list[TradeSignal] = []
        for symbol, pos in portfolio.positions.items():
            if (
                pos.stop_loss > 0
                and symbol in prices
                and prices[symbol] <= pos.stop_loss
            ):
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action=Action.CLOSE,
                        conviction=Conviction.HIGH,
                        target_weight=0.0,
                        stop_loss=0.0,
                        reasoning=(
                            f"Stop-loss triggered at "
                            f"{prices[symbol]:.2f} <= "
                            f"{pos.stop_loss:.2f}"
                        ),
                    )
                )
        return signals

    def _execute_signals(
        self,
        signals: list[TradeSignal],
        portfolio: Portfolio,
        prices: dict[str, float],
        trade_date: date,
        cost_model: CostModel,
        cost_multiplier: float,
        volume_stats: dict[str, float],
        vol_stats: dict[str, float],
    ) -> list[TradeRecord]:
        """Execute signals and apply costs. Returns trade records."""
        records: list[TradeRecord] = []
        nav = portfolio.nav

        for signal in signals:
            price = prices.get(signal.symbol)
            if price is None or price <= 0:
                continue

            if signal.action == Action.HOLD:
                continue

            # Calculate shares
            if signal.action == Action.BUY:
                target_notional = signal.target_weight * nav
                current_notional = 0.0
                existing = portfolio.positions.get(signal.symbol)
                if existing:
                    current_notional = existing.market_value
                additional = target_notional - current_notional
                if additional <= 0:
                    continue
                shares = math.floor(additional / price)
                if shares <= 0:
                    continue

                # Check cash
                cost_estimate = cost_model.compute_cost(
                    shares * price,
                    shares,
                    volume_stats.get(signal.symbol),
                    vol_stats.get(signal.symbol),
                    cost_multiplier,
                )
                total_cost = shares * price + cost_estimate
                if total_cost > portfolio.cash:
                    shares = math.floor((portfolio.cash - cost_estimate) / price)
                    if shares <= 0:
                        continue

                notional = shares * price
                cost = cost_model.compute_cost(
                    notional,
                    shares,
                    volume_stats.get(signal.symbol),
                    vol_stats.get(signal.symbol),
                    cost_multiplier,
                )

                # Update portfolio
                portfolio.cash -= notional + cost
                if existing:
                    total_shares = existing.shares + shares
                    existing.avg_cost = (
                        existing.shares * existing.avg_cost + shares * price
                    ) / total_shares
                    existing.shares = total_shares
                    existing.current_price = price
                    existing.stop_loss = signal.stop_loss
                else:
                    from llm_quant.trading.portfolio import Position

                    portfolio.positions[signal.symbol] = Position(
                        symbol=signal.symbol,
                        shares=shares,
                        avg_cost=price,
                        current_price=price,
                        stop_loss=signal.stop_loss,
                    )

                records.append(
                    TradeRecord(
                        date=trade_date,
                        symbol=signal.symbol,
                        action="buy",
                        shares=shares,
                        price=price,
                        notional=notional,
                        cost=cost,
                        reasoning=signal.reasoning,
                    )
                )

            elif signal.action in (Action.SELL, Action.CLOSE):
                existing = portfolio.positions.get(signal.symbol)
                if existing is None or existing.shares <= 0:
                    continue

                if signal.action == Action.CLOSE:
                    shares = existing.shares
                else:
                    target_notional = signal.target_weight * nav
                    current_notional = existing.shares * price
                    reduce = current_notional - target_notional
                    if reduce <= 0:
                        continue
                    shares = min(math.floor(reduce / price), existing.shares)
                    if shares <= 0:
                        continue

                notional = shares * price
                cost = cost_model.compute_cost(
                    notional,
                    shares,
                    volume_stats.get(signal.symbol),
                    vol_stats.get(signal.symbol),
                    cost_multiplier,
                )

                # PnL for this trade
                pnl = (price - existing.avg_cost) * shares - cost

                # Update portfolio
                portfolio.cash += notional - cost
                existing.shares -= shares
                if existing.shares <= 0:
                    del portfolio.positions[signal.symbol]
                else:
                    existing.current_price = price

                records.append(
                    TradeRecord(
                        date=trade_date,
                        symbol=signal.symbol,
                        action=signal.action.value,
                        shares=shares,
                        price=price,
                        notional=notional,
                        cost=cost,
                        pnl=pnl,
                        reasoning=signal.reasoning,
                    )
                )

        return records

    def _compute_volume_stats(self, df: pl.DataFrame) -> dict[str, float]:
        """Compute average daily volume per symbol."""
        if "volume" not in df.columns:
            return {}
        stats = df.group_by("symbol").agg(pl.col("volume").mean().alias("avg_volume"))
        return dict(
            zip(
                stats.select("symbol").to_series().to_list(),
                stats.select("avg_volume").to_series().to_list(),
                strict=False,
            )
        )

    def _compute_volatility_stats(self, df: pl.DataFrame) -> dict[str, float]:
        """Compute daily return volatility per symbol."""
        result: dict[str, float] = {}
        symbols = df.select("symbol").unique().to_series().to_list()
        for symbol in symbols:
            sym_data = (
                df.filter(pl.col("symbol") == symbol)
                .sort("date")
                .select("close")
                .to_series()
            )
            if len(sym_data) < 20:
                continue
            returns = sym_data.pct_change().drop_nulls()
            if len(returns) > 0:
                result[symbol] = float(returns.std())
        return result

    def _check_data_quality(self, df: pl.DataFrame, trading_dates: list) -> list[str]:
        """Check data quality and return warnings."""
        warnings: list[str] = []
        total_dates = len(trading_dates)
        if total_dates == 0:
            return warnings

        first_trade_date = trading_dates[0]
        last_trade_date = trading_dates[-1]
        symbols = df.select("symbol").unique().to_series().to_list()
        for symbol in symbols:
            sym_dates = (
                df.filter(
                    (pl.col("symbol") == symbol)
                    & (pl.col("date") >= first_trade_date)
                    & (pl.col("date") <= last_trade_date)
                )
                .select("date")
                .to_series()
                .to_list()
            )
            coverage = len(sym_dates) / total_dates
            if coverage < 0.80:
                warnings.append(
                    f"Survivorship warning: {symbol} has {coverage:.0%} date coverage "
                    f"({len(sym_dates)}/{total_dates} dates)"
                )

        return warnings

"""Performance metrics, Deflated Sharpe Ratio, and benchmark computation.

All Sharpe ratios are computed unannualized (per-period) for DSR math.
Annualized versions are provided separately for display.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Core performance metrics
# ---------------------------------------------------------------------------


@dataclass
class BacktestMetrics:
    """Container for all backtest performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns: list[float] = field(default_factory=list)

    # Risk-adjusted (annualized for display)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Unannualized (for DSR math)
    sharpe_per_period: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0

    # DSR / statistical
    psr: float = 0.0
    dsr: float = 0.0
    trial_count: int = 0

    # Benchmark comparison
    benchmark_return: float = 0.0
    benchmark_sharpe: float = 0.0
    excess_return: float = 0.0
    information_ratio: float = 0.0

    # Data quality
    warnings: list[str] = field(default_factory=list)


def compute_returns(nav_series: list[float]) -> list[float]:
    """Compute daily returns from a NAV time series."""
    if len(nav_series) < 2:
        return []
    returns = []
    for i in range(1, len(nav_series)):
        if nav_series[i - 1] != 0:
            returns.append(nav_series[i] / nav_series[i - 1] - 1.0)
        else:
            returns.append(0.0)
    return returns


def compute_sharpe(returns: list[float], annualize: bool = True) -> float:
    """Compute Sharpe ratio (excess return / volatility).

    Assumes risk-free rate = 0 for simplicity.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))
    if std_ret == 0:
        return 0.0
    sr = mean_ret / std_ret
    if annualize:
        sr *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return sr


def compute_sortino(returns: list[float], annualize: bool = True) -> float:
    """Compute Sortino ratio (excess return / downside deviation).

    Downside deviation = sqrt(mean(min(r_i, 0)^2)) over ALL observations,
    not just the negative ones (standard Sortino formulation).
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_ret = float(np.mean(arr))
    # Full-sample downside deviation: sqrt(mean(min(r, 0)^2))
    downside_sq = np.minimum(arr, 0.0) ** 2
    downside_std = float(np.sqrt(np.mean(downside_sq)))
    if downside_std == 0:
        return float("inf") if mean_ret > 0 else 0.0
    sortino = mean_ret / downside_std
    if annualize:
        sortino *= math.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino


def compute_max_drawdown(nav_series: list[float]) -> tuple[float, int]:
    """Compute maximum drawdown and its duration in days.

    Returns (max_drawdown_fraction, duration_days).
    """
    if len(nav_series) < 2:
        return 0.0, 0

    peak = nav_series[0]
    max_dd = 0.0
    max_dd_duration = 0
    current_dd_start = 0

    for i, nav in enumerate(nav_series):
        if nav >= peak:
            # New peak — record duration of previous drawdown
            if max_dd > 0:
                duration = i - current_dd_start
                max_dd_duration = max(max_dd_duration, duration)
            peak = nav
            current_dd_start = i
        else:
            dd = (peak - nav) / peak
            max_dd = max(max_dd, dd)

    # Check final drawdown duration
    if nav_series[-1] < peak:
        duration = len(nav_series) - 1 - current_dd_start
        max_dd_duration = max(max_dd_duration, duration)

    return max_dd, max_dd_duration


def compute_calmar(annualized_return: float, max_drawdown: float) -> float:
    """Compute Calmar ratio (annualized return / max drawdown)."""
    if max_drawdown == 0:
        return float("inf") if annualized_return > 0 else 0.0
    return annualized_return / max_drawdown


def compute_annualized_return(total_return: float, trading_days: int) -> float:
    """Annualize a total return given number of trading days."""
    if trading_days <= 0:
        return 0.0
    years = trading_days / TRADING_DAYS_PER_YEAR
    if years == 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------


def compute_psr(
    sharpe_per_period: float,
    benchmark_sharpe_per_period: float,
    n: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Compute the Probabilistic Sharpe Ratio.

    PSR = Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - gamma3*SR + (gamma4-1)/4 * SR^2))

    Parameters
    ----------
    sharpe_per_period : float
        Unannualized Sharpe ratio (mean/std of per-period returns).
    benchmark_sharpe_per_period : float
        Benchmark Sharpe ratio (SR*), unannualized.
    n : int
        Number of observations (trading days).
    skewness : float
        Sample skewness of returns (gamma_3).
    kurtosis : float
        Sample excess kurtosis of returns (gamma_4).

    Returns
    -------
    float
        PSR in [0, 1].
    """
    if n <= 1:
        return 0.0

    sr = sharpe_per_period
    sr_star = benchmark_sharpe_per_period

    # Bailey & Lopez de Prado (2014): V(SR) = (1 - γ₃·SR + (γ₄-1)/4·SR²)/(n-1)
    # where γ₄ = regular kurtosis. Since scipy returns excess kurtosis (normal=0),
    # γ₄_regular = excess + 3, so (γ₄_regular - 1)/4 = (excess + 2)/4
    denominator_sq = 1.0 - skewness * sr + (kurtosis + 2.0) / 4.0 * sr**2
    if denominator_sq <= 0:
        return 0.0

    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denominator_sq)
    return float(stats.norm.cdf(z))


def compute_sr0(trial_count: int, variance_of_sr: float) -> float:
    """Compute the False Strategy Theorem SR_0.

    SR_0 = sqrt(V[SR]) * ((1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(Ne)))

    where gamma ≈ 0.5772 (Euler-Mascheroni constant), N = trial count.
    """
    if trial_count <= 1:
        return 0.0

    euler_gamma = 0.5772156649
    n = trial_count

    # Avoid numerical issues with very large N
    q1 = max(1.0 - 1.0 / n, 1e-10)
    q2 = max(1.0 - 1.0 / (n * math.e), 1e-10)

    term1 = (1.0 - euler_gamma) * float(stats.norm.ppf(q1))
    term2 = euler_gamma * float(stats.norm.ppf(q2))

    return math.sqrt(variance_of_sr) * (term1 + term2)


def compute_dsr(
    returns: list[float],
    trial_count: int,
    benchmark_sharpe_per_period: float = 0.0,
) -> tuple[float, float, float]:
    """Compute the Deflated Sharpe Ratio.

    Returns (dsr, psr, sr0).

    DSR = PSR with benchmark replaced by SR_0 from the False Strategy Theorem.
    """
    if len(returns) < 10 or trial_count < 1:
        return 0.0, 0.0, 0.0

    arr = np.array(returns)
    n = len(arr)
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))

    if std_ret == 0:
        return 0.0, 0.0, 0.0

    sr = mean_ret / std_ret  # unannualized
    skewness = float(stats.skew(arr, bias=False))
    kurtosis = float(stats.kurtosis(arr, bias=False))  # excess kurtosis

    # Variance of Sharpe ratio estimator
    var_sr = (1.0 + 0.5 * sr**2 - skewness * sr + (kurtosis / 4.0) * sr**2) / (n - 1)

    # SR_0 from False Strategy Theorem
    sr0 = compute_sr0(trial_count, max(var_sr, 1e-10))

    # PSR with benchmark = SR_0
    dsr = compute_psr(sr, sr0, n, skewness, kurtosis)

    # Also compute standard PSR vs provided benchmark
    psr = compute_psr(sr, benchmark_sharpe_per_period, n, skewness, kurtosis)

    return dsr, psr, sr0


# ---------------------------------------------------------------------------
# Benchmark computation (total return)
# ---------------------------------------------------------------------------


def compute_benchmark_returns(
    prices_df: pl.DataFrame,
    weights: dict[str, float],
    rebalance_frequency_days: int = 21,
    use_adj_close: bool = True,
) -> list[float]:
    """Compute benchmark total returns from a multi-asset portfolio.

    Parameters
    ----------
    prices_df : pl.DataFrame
        DataFrame with columns: symbol, date, close, adj_close.
    weights : dict[str, float]
        Target weights by symbol (must sum to ~1.0).
    rebalance_frequency_days : int
        How often to rebalance to target weights.
    use_adj_close : bool
        If True, uses adj_close (total return including dividends).

    Returns
    -------
    list[float]
        Daily benchmark returns.
    """
    price_col = "adj_close" if use_adj_close else "close"

    # Get sorted dates
    dates = sorted(prices_df.select("date").unique().to_series().to_list())
    if len(dates) < 2:
        return []

    # Build a price matrix: one column per symbol
    benchmark_symbols = list(weights.keys())
    daily_returns: list[float] = []

    # Current weights (start at target)
    current_weights = dict(weights)
    days_since_rebalance = 0

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        portfolio_return = 0.0
        new_weights: dict[str, float] = {}
        total_weight = 0.0

        for symbol in benchmark_symbols:
            sym_data = prices_df.filter(pl.col("symbol") == symbol)

            prev_rows = sym_data.filter(pl.col("date") == prev_date)
            curr_rows = sym_data.filter(pl.col("date") == curr_date)

            if len(prev_rows) == 0 or len(curr_rows) == 0:
                new_weights[symbol] = current_weights.get(symbol, 0.0)
                total_weight += new_weights[symbol]
                continue

            prev_price = prev_rows.select(price_col).item()
            curr_price = curr_rows.select(price_col).item()

            if prev_price is None or curr_price is None or prev_price == 0:
                new_weights[symbol] = current_weights.get(symbol, 0.0)
                total_weight += new_weights[symbol]
                continue

            asset_return = curr_price / prev_price - 1.0
            w = current_weights.get(symbol, 0.0)
            portfolio_return += w * asset_return

            # Drift weights
            new_weights[symbol] = w * (1.0 + asset_return)
            total_weight += new_weights[symbol]

        daily_returns.append(portfolio_return)

        # Normalize drifted weights
        if total_weight > 0:
            for s in new_weights:
                new_weights[s] /= total_weight

        days_since_rebalance += 1

        # Rebalance
        if days_since_rebalance >= rebalance_frequency_days:
            current_weights = dict(weights)
            days_since_rebalance = 0
        else:
            current_weights = new_weights

    return daily_returns


def compute_trade_stats(
    trades: list[dict],
) -> tuple[float, float, float]:
    """Compute win rate, profit factor, and average trade return.

    Returns (win_rate, profit_factor, avg_return).
    """
    if not trades:
        return 0.0, 0.0, 0.0

    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_return = 0.0

    for trade in trades:
        pnl = trade.get("pnl", 0.0)
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)
        total_return += pnl

    win_rate = wins / len(trades) if trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_return = total_return / len(trades) if trades else 0.0

    return win_rate, profit_factor, avg_return


def compute_all_metrics(
    nav_series: list[float],
    trades: list[dict],
    trial_count: int = 1,
    benchmark_returns: list[float] | None = None,
) -> BacktestMetrics:
    """Compute all metrics from a NAV series and trade list."""
    metrics = BacktestMetrics()

    if len(nav_series) < 2:
        metrics.warnings.append("NAV series too short for metrics computation")
        return metrics

    # Returns
    daily_rets = compute_returns(nav_series)
    metrics.daily_returns = daily_rets
    metrics.total_return = nav_series[-1] / nav_series[0] - 1.0

    trading_days = len(daily_rets)
    metrics.annualized_return = compute_annualized_return(
        metrics.total_return, trading_days
    )

    # Risk-adjusted
    metrics.sharpe_ratio = compute_sharpe(daily_rets, annualize=True)
    metrics.sharpe_per_period = compute_sharpe(daily_rets, annualize=False)
    metrics.sortino_ratio = compute_sortino(daily_rets, annualize=True)

    # Drawdown
    metrics.max_drawdown, metrics.max_drawdown_duration_days = compute_max_drawdown(
        nav_series
    )
    metrics.calmar_ratio = compute_calmar(
        metrics.annualized_return, metrics.max_drawdown
    )

    # Trade statistics
    metrics.total_trades = len(trades)
    if trades:
        metrics.win_rate, metrics.profit_factor, metrics.avg_trade_return = (
            compute_trade_stats(trades)
        )

    # DSR
    metrics.trial_count = trial_count
    benchmark_sr = 0.0
    if benchmark_returns and len(benchmark_returns) > 1:
        benchmark_sr = compute_sharpe(benchmark_returns, annualize=False)
        metrics.benchmark_sharpe = compute_sharpe(benchmark_returns, annualize=True)
        metrics.benchmark_return = float(
            np.prod([1 + r for r in benchmark_returns]) - 1.0
        )
        metrics.excess_return = metrics.total_return - metrics.benchmark_return

        # Information ratio
        if len(daily_rets) == len(benchmark_returns):
            excess = [
                d - b for d, b in zip(daily_rets, benchmark_returns, strict=False)
            ]
            excess_arr = np.array(excess)
            te = float(np.std(excess_arr, ddof=1))
            if te > 0:
                metrics.information_ratio = (
                    float(np.mean(excess_arr)) / te * math.sqrt(TRADING_DAYS_PER_YEAR)
                )

    dsr, psr, _sr0 = compute_dsr(daily_rets, trial_count, benchmark_sr)
    metrics.dsr = dsr
    metrics.psr = psr

    return metrics

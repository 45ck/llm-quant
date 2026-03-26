"""Robustness analysis: PBO via CSCV, CPCV, and perturbation suite.

Implements the robustness gate that must pass before promotion.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_quant.backtest.metrics import compute_sharpe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PBOResult:
    """Result of Probability of Backtest Overfitting analysis."""

    pbo: float = 1.0  # fraction of combos where IS-best underperforms OOS
    n_combinations: int = 0
    n_strategies: int = 0
    is_best_oos_ranks: list[int] = field(default_factory=list)
    passed: bool = False  # PBO <= 0.10


@dataclass
class CPCVResult:
    """Result of Combinatorial Purged Cross-Validation."""

    mean_oos_sharpe: float = 0.0
    std_oos_sharpe: float = 0.0
    n_paths: int = 0
    n_combinations: int = 0
    oos_sharpes: list[float] = field(default_factory=list)
    passed: bool = False  # mean OOS Sharpe > 0


@dataclass
class PerturbationResult:
    """Result of a single perturbation test."""

    name: str = ""
    parameter_change: str = ""
    sharpe: float = 0.0
    profitable: bool = False


@dataclass
class RobustnessResult:
    """Complete robustness gate result."""

    # Individual gate results
    dsr: float = 0.0
    dsr_passed: bool = False
    pbo: PBOResult = field(default_factory=PBOResult)
    pbo_passed: bool = False
    cpcv: CPCVResult = field(default_factory=CPCVResult)
    cpcv_passed: bool = False
    cost_2x_survives: bool = False
    parameter_stability: float = 0.0  # fraction of perturbations profitable
    parameter_stability_passed: bool = False
    perturbations: list[PerturbationResult] = field(default_factory=list)

    # Gate 6: Shuffled signal test
    shuffled_signal: object = None  # ShuffledSignalResult or None
    shuffled_signal_passed: bool = True  # default True (skipped = pass)

    # Overall
    overall_passed: bool = False
    gate_details: dict[str, bool] = field(default_factory=dict)

    def compute_overall(self) -> None:
        """Compute overall gate pass from individual results."""
        self.gate_details = {
            "dsr_>=_0.95": self.dsr_passed,
            "pbo_<=_0.10": self.pbo_passed,
            "cpcv_mean_oos_sharpe_>_0": self.cpcv_passed,
            "2x_costs_survive": self.cost_2x_survives,
            "parameter_stability_>_50%": self.parameter_stability_passed,
            "shuffled_signal_p_<_0.05": self.shuffled_signal_passed,
        }
        self.overall_passed = all(self.gate_details.values())


# ---------------------------------------------------------------------------
# PBO via Combinatorial Symmetric Cross-Validation (CSCV)
# ---------------------------------------------------------------------------


def compute_pbo(
    returns_matrix: list[list[float]],
    n_submatrices: int = 16,
) -> PBOResult:
    """Compute Probability of Backtest Overfitting using CSCV.

    Parameters
    ----------
    returns_matrix : list[list[float]]
        List of daily return series, one per strategy/parameter variant.
        All series must have the same length.
    n_submatrices : int
        Number of submatrices S to partition the time axis into.
        Default 16 → C(16,8) = 12,870 combinations.

    Returns
    -------
    PBOResult
        PBO value and diagnostic details.
    """
    if len(returns_matrix) < 2:
        logger.warning("PBO requires at least 2 strategy variants")
        return PBOResult(pbo=1.0, n_strategies=len(returns_matrix))

    # Convert to numpy array: rows = time, columns = strategies
    min_len = min(len(r) for r in returns_matrix)
    n_strategies = len(returns_matrix)

    if min_len < n_submatrices:
        logger.warning(
            "Not enough observations (%d) for %d submatrices",
            min_len,
            n_submatrices,
        )
        return PBOResult(pbo=1.0, n_strategies=n_strategies)

    # Truncate all to same length
    matrix = np.array([r[:min_len] for r in returns_matrix]).T  # (T, N)
    T, N = matrix.shape

    # Partition into S submatrices of roughly equal size
    S = n_submatrices
    block_size = T // S
    if block_size < 5:
        logger.warning("Block size too small (%d) for meaningful PBO", block_size)
        return PBOResult(pbo=1.0, n_strategies=N)

    # Trim to exact multiple of S
    matrix = matrix[: block_size * S, :]

    # Compute Sharpe per block per strategy
    block_sharpes = np.zeros((S, N))
    for s in range(S):
        start = s * block_size
        end = start + block_size
        block_data = matrix[start:end, :]
        for n in range(N):
            block_sharpes[s, n] = compute_sharpe(
                block_data[:, n].tolist(), annualize=False
            )

    # Generate all C(S, S/2) combinations
    half = S // 2
    combos = list(itertools.combinations(range(S), half))

    # Limit to avoid excessive computation
    max_combos = 5000
    if len(combos) > max_combos:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[i] for i in sorted(indices)]

    n_overfit = 0
    is_best_oos_ranks: list[int] = []

    for combo in combos:
        is_blocks = set(combo)
        oos_blocks = set(range(S)) - is_blocks

        # IS performance: sum of Sharpes across IS blocks per strategy
        is_perf = np.sum(block_sharpes[list(is_blocks), :], axis=0)  # (N,)
        oos_perf = np.sum(block_sharpes[list(oos_blocks), :], axis=0)  # (N,)

        # Find IS-optimal strategy
        is_best = int(np.argmax(is_perf))

        # Rank of IS-best in OOS (1-based; strict inequality avoids ties inflating rank)
        oos_rank = int(np.sum(oos_perf > oos_perf[is_best])) + 1
        is_best_oos_ranks.append(oos_rank)

        # IS-best ranks below OOS median?
        median_rank = N // 2
        if oos_rank > median_rank:
            n_overfit += 1

    pbo = n_overfit / len(combos) if combos else 1.0

    return PBOResult(
        pbo=pbo,
        n_combinations=len(combos),
        n_strategies=N,
        is_best_oos_ranks=is_best_oos_ranks,
        passed=False,  # caller decides threshold
    )


# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation (CPCV)
# ---------------------------------------------------------------------------


def run_cpcv(
    returns: list[float],
    strategy_fn: Any,
    n_groups: int = 6,
    k_test: int = 2,
    purge_days: int = 5,
    embargo_pct: float = 0.01,
) -> CPCVResult:
    """Run Combinatorial Purged Cross-Validation.

    Parameters
    ----------
    returns : list[float]
        Full daily return series from the strategy.
    strategy_fn : callable | None
        If provided, a function that takes a return series and returns
        the Sharpe ratio. If None, computes Sharpe directly from returns.
    n_groups : int
        Number of sequential groups (N). Default 6.
    k_test : int
        Number of test groups per split. Default 2.
    purge_days : int
        Number of observations to remove at train/test boundaries.
    embargo_pct : float
        Fraction of total observations to embargo at boundaries.

    Returns
    -------
    CPCVResult
        Distribution of OOS Sharpe ratios across all combinations.
    """
    T = len(returns)
    if n_groups * 10 > T:
        return CPCVResult()

    arr = np.array(returns)
    group_size = T // n_groups
    embargo_size = max(int(T * embargo_pct), 1)

    # Generate all C(n_groups, k_test) combinations
    combos = list(itertools.combinations(range(n_groups), k_test))

    # Number of independent backtest paths
    n_paths = n_groups - k_test  # approximate

    oos_sharpes: list[float] = []

    for combo in combos:
        test_groups = set(combo)
        train_groups = set(range(n_groups)) - test_groups

        # Build test indices
        test_indices: list[int] = []
        for g in sorted(test_groups):
            start = g * group_size
            end = min(start + group_size, T)
            test_indices.extend(range(start, end))

        # Build train indices with purging and embargo
        train_indices: list[int] = []
        for g in sorted(train_groups):
            start = g * group_size
            end = min(start + group_size, T)

            # Apply purge: remove observations near test boundaries
            purged_start = start
            purged_end = end

            for tg in sorted(test_groups):
                test_start = tg * group_size
                test_end = min(test_start + group_size, T)

                # If this train group is right before a test group
                if end > test_start - purge_days and end <= test_end:
                    purged_end = max(start, end - purge_days)

                # If this train group is right after a test group
                if start >= test_start and start < test_end + embargo_size:
                    purged_start = min(end, start + embargo_size)

            train_indices.extend(range(purged_start, purged_end))

        # Compute OOS Sharpe on test set
        if not test_indices:
            continue
        test_returns = arr[test_indices]

        if strategy_fn is not None:
            oos_sharpe = strategy_fn(test_returns.tolist())
        else:
            oos_sharpe = compute_sharpe(test_returns.tolist(), annualize=False)

        oos_sharpes.append(oos_sharpe)

    if not oos_sharpes:
        return CPCVResult()

    return CPCVResult(
        mean_oos_sharpe=float(np.mean(oos_sharpes)),
        std_oos_sharpe=float(np.std(oos_sharpes)),
        n_paths=n_paths,
        n_combinations=len(combos),
        oos_sharpes=oos_sharpes,
        passed=float(np.mean(oos_sharpes)) > 0,
    )


# ---------------------------------------------------------------------------
# Perturbation suite
# ---------------------------------------------------------------------------


def generate_perturbations(
    base_params: dict[str, Any],
    perturbation_pct: float = 0.20,
) -> list[tuple[str, dict[str, Any]]]:
    """Generate parameter perturbations for robustness testing.

    For each numeric parameter, generates +/- perturbation_pct variants.

    Returns list of (description, modified_params) tuples.
    """
    perturbations: list[tuple[str, dict[str, Any]]] = []

    for key, value in base_params.items():
        if isinstance(value, (int, float)) and value != 0:
            # +perturbation
            up_params = dict(base_params)
            up_val = value * (1.0 + perturbation_pct)
            if isinstance(value, int):
                up_val = round(up_val)
                up_val = max(1, up_val) if value > 0 else min(-1, up_val)
            up_params[key] = up_val
            perturbations.append((f"{key}+{perturbation_pct:.0%}", up_params))

            # -perturbation
            down_params = dict(base_params)
            down_val = value * (1.0 - perturbation_pct)
            if isinstance(value, int):
                down_val = round(down_val)
                down_val = max(1, down_val) if value > 0 else min(-1, down_val)
            down_params[key] = down_val
            perturbations.append((f"{key}-{perturbation_pct:.0%}", down_params))

    return perturbations


# ---------------------------------------------------------------------------
# Full robustness gate
# ---------------------------------------------------------------------------


def run_robustness_gate(
    dsr: float,
    returns_matrix: list[list[float]],
    best_returns: list[float],
    cost_2x_sharpe: float,
    perturbation_results: list[PerturbationResult],
    dsr_threshold: float = 0.95,
    pbo_threshold: float = 0.10,
    asset_returns: list[float] | None = None,
    n_shuffles: int = 1000,
) -> RobustnessResult:
    """Run the complete robustness gate.

    Parameters
    ----------
    dsr : float
        Deflated Sharpe Ratio from metrics.
    returns_matrix : list[list[float]]
        Daily returns from all experiments (for PBO).
    best_returns : list[float]
        Daily returns from the best experiment (for CPCV).
    cost_2x_sharpe : float
        Sharpe ratio at 2x cost multiplier.
    perturbation_results : list[PerturbationResult]
        Results from parameter perturbation tests.
    dsr_threshold : float
        Minimum DSR for pass.
    pbo_threshold : float
        Maximum PBO for pass.
    asset_returns : list[float] | None
        Buy-and-hold daily returns of the underlying asset (for shuffled
        signal test).  If None, Gate 6 is skipped (defaults to pass).
    n_shuffles : int
        Number of permutations for shuffled signal test.

    Returns
    -------
    RobustnessResult
        Complete gate result.
    """
    result = RobustnessResult()

    # 1. DSR gate
    result.dsr = dsr
    result.dsr_passed = dsr >= dsr_threshold

    # 2. PBO gate
    if len(returns_matrix) >= 2:
        result.pbo = compute_pbo(returns_matrix)
        result.pbo_passed = result.pbo.pbo <= pbo_threshold
    else:
        logger.warning("Insufficient experiments for PBO -- need >= 2")
        result.pbo = PBOResult(pbo=1.0, n_strategies=len(returns_matrix))
        result.pbo_passed = False

    # 3. CPCV gate
    if best_returns:
        result.cpcv = run_cpcv(best_returns, strategy_fn=None)
        result.cpcv_passed = result.cpcv.passed

    # 4. Cost survival
    result.cost_2x_survives = cost_2x_sharpe > 0

    # 5. Parameter stability
    result.perturbations = perturbation_results
    if perturbation_results:
        profitable = sum(1 for p in perturbation_results if p.profitable)
        result.parameter_stability = profitable / len(perturbation_results)
    result.parameter_stability_passed = result.parameter_stability > 0.50

    # 6. Shuffled signal test (Gate 6)
    if asset_returns is not None and best_returns:
        result.shuffled_signal = shuffled_signal_test(
            daily_returns=best_returns,
            asset_returns=asset_returns,
            n_shuffles=n_shuffles,
        )
        result.shuffled_signal_passed = result.shuffled_signal.passed
    else:
        # Skipped -- default to pass (backwards compatible)
        result.shuffled_signal_passed = True

    # Overall
    result.compute_overall()
    return result


# ---------------------------------------------------------------------------
# Fraud Detector 1: Shuffled Signal Test
# ---------------------------------------------------------------------------


@dataclass
class ShuffledSignalResult:
    """Result of shuffled signal permutation test."""

    real_sharpe: float = 0.0
    shuffled_mean: float = 0.0
    shuffled_95th: float = 0.0
    shuffled_99th: float = 0.0
    p_value: float = 1.0
    n_shuffles: int = 0
    passed: bool = False  # real Sharpe > 95th percentile of shuffled


def shuffled_signal_test(
    daily_returns: list[float],
    asset_returns: list[float],
    n_shuffles: int = 1000,
    seed: int = 42,
) -> ShuffledSignalResult:
    """Test whether signal timing adds value beyond random entry.

    Randomly selects which days to be invested (keeping the same number
    of invested days) using the ACTUAL asset returns for each randomly
    chosen day. This tests whether the signal picks better-than-random
    days to hold the asset.

    Controls for: time-in-market bias, volatility harvesting, bull
    market drift, and all non-timing sources of return.

    Parameters
    ----------
    daily_returns : list[float]
        Daily strategy returns (including 0.0 for cash days).
    asset_returns : list[float]
        Daily returns of the follower asset for ALL days (same length
        as daily_returns). Used to compute what random entry would yield.
    n_shuffles : int
        Number of random permutations to run.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ShuffledSignalResult
        Contains real Sharpe, shuffled distribution stats, and pass/fail.
    """
    strat_arr = np.array(daily_returns)
    asset_arr = np.array(asset_returns)
    n = len(strat_arr)
    if n < 30 or len(asset_arr) != n:
        return ShuffledSignalResult()

    # Real Sharpe
    real_sharpe = compute_sharpe(daily_returns, annualize=True)

    # Count invested days
    n_invested = int(np.sum(strat_arr != 0.0))
    if n_invested < 10:
        return ShuffledSignalResult(real_sharpe=real_sharpe)

    # Shuffle: randomly pick which N_invested days to be in the asset
    rng = np.random.default_rng(seed)
    shuffled_sharpes = np.zeros(n_shuffles)

    for i in range(n_shuffles):
        shuffled = np.zeros(n)
        random_days = rng.choice(n, size=n_invested, replace=False)
        shuffled[random_days] = asset_arr[random_days]
        shuffled_sharpes[i] = compute_sharpe(shuffled.tolist(), annualize=True)

    # Statistics
    mean_s = float(np.mean(shuffled_sharpes))
    p95 = float(np.percentile(shuffled_sharpes, 95))
    p99 = float(np.percentile(shuffled_sharpes, 99))
    p_value = float(np.mean(shuffled_sharpes >= real_sharpe))

    return ShuffledSignalResult(
        real_sharpe=real_sharpe,
        shuffled_mean=mean_s,
        shuffled_95th=p95,
        shuffled_99th=p99,
        p_value=p_value,
        n_shuffles=n_shuffles,
        passed=real_sharpe > p95,
    )


# ---------------------------------------------------------------------------
# Fraud Detector 2: Mechanism Inversion Test
# ---------------------------------------------------------------------------


@dataclass
class InversionResult:
    """Result of mechanism inversion test."""

    original_sharpe: float = 0.0
    inverted_sharpe: float = 0.0
    sharpe_differential: float = 0.0
    inverted_is_negative: bool = False
    passed: bool = False


def mechanism_inversion_test(
    original_returns: list[float],
    inverted_returns: list[float],
    min_differential: float = 0.25,
) -> InversionResult:
    """Test whether signal direction matters by comparing original vs inverted.

    For long-only strategies in trending markets, the inverted signal may
    still be positive (equity risk premium). The correct test is whether
    the DIFFERENTIAL (original - inverted) exceeds a meaningful threshold,
    confirming the signal has genuine directional content.

    Parameters
    ----------
    original_returns : list[float]
        Daily returns from the original strategy.
    inverted_returns : list[float]
        Daily returns from the strategy with inverted signals.
    min_differential : float
        Minimum Sharpe differential (original - inverted) for pass.
        Default 0.25 = signal direction adds at least 0.25 Sharpe.

    Returns
    -------
    InversionResult
    """
    orig_sr = compute_sharpe(original_returns, annualize=True)
    inv_sr = compute_sharpe(inverted_returns, annualize=True)
    diff = orig_sr - inv_sr

    return InversionResult(
        original_sharpe=orig_sr,
        inverted_sharpe=inv_sr,
        sharpe_differential=diff,
        inverted_is_negative=inv_sr < 0,
        passed=diff >= min_differential,
    )


# ---------------------------------------------------------------------------
# Fraud Detector 3: Time-in-Market Analysis
# ---------------------------------------------------------------------------


@dataclass
class TimeInMarketResult:
    """Result of time-in-market analysis."""

    total_days: int = 0
    invested_days: int = 0
    cash_days: int = 0
    pct_invested: float = 0.0
    passed: bool = False  # invested < 80% of the time


def time_in_market(daily_returns: list[float]) -> TimeInMarketResult:
    """Measure what fraction of days the strategy is invested.

    Strategies invested >80% of the time are likely capturing
    equity beta rather than providing genuine timing alpha.

    Parameters
    ----------
    daily_returns : list[float]
        Daily strategy returns (0.0 = cash day).

    Returns
    -------
    TimeInMarketResult
    """
    arr = np.array(daily_returns)
    total = len(arr)
    invested = int(np.sum(arr != 0.0))
    cash = total - invested
    pct = invested / total if total > 0 else 0.0

    return TimeInMarketResult(
        total_days=total,
        invested_days=invested,
        cash_days=cash,
        pct_invested=pct,
        passed=pct < 0.80,
    )

"""Robustness analysis: PBO via CSCV, CPCV, and perturbation suite.

Implements the robustness gate that must pass before promotion.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

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
class MinTRLResult:
    """Result of Minimum Track Record Length computation.

    Bailey & Lopez de Prado (2014): minimum number of monthly observations
    needed to determine with statistical confidence that a Sharpe ratio is
    genuine, accounting for non-normal return distributions.

    Formula:
        MinTRL = (1 - skew*SR + ((kurt-1)/4)*SR^2) * (z_alpha / (SR - SR*))^2
    where SR* = 0 (benchmark), z_alpha = 1.645 (95% confidence).
    """

    min_trl_months: float = 0.0
    backtest_months: float = 0.0
    sharpe: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0  # excess kurtosis
    confidence: float = 0.95
    min_trl_pass: bool = False  # True if backtest_months >= min_trl_months


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

    # MinTRL (informational — not a hard gate but emits WARNING if insufficient)
    min_trl: MinTRLResult = field(default_factory=MinTRLResult)

    # Portfolio admission gates (taff + r5j4)
    marginal_sr_contribution: float = 0.0  # ΔSR_P; gate: >= 0.05
    marginal_sr_passed: bool = True  # default True (skipped = pass)
    portfolio_correlation: float = 0.0  # rolling 60-day avg; gate: < 0.30
    portfolio_correlation_passed: bool = True  # default True (skipped = pass)

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
            "marginal_sr_contribution_>=_0.05": self.marginal_sr_passed,
            "portfolio_correlation_<_0.30": self.portfolio_correlation_passed,
        }
        self.overall_passed = all(self.gate_details.values())

        # Emit WARNING if backtest history is shorter than MinTRL
        if not self.min_trl.min_trl_pass and self.min_trl.min_trl_months > 0:
            logger.warning(
                "MinTRL WARNING: backtest has %.1f months but requires %.1f months "
                "for %.0f%% confidence that SR=%.3f is genuine. "
                "Results may not be statistically significant.",
                self.min_trl.backtest_months,
                self.min_trl.min_trl_months,
                self.min_trl.confidence * 100,
                self.min_trl.sharpe,
            )


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
# Minimum Track Record Length (MinTRL)
# ---------------------------------------------------------------------------


def compute_min_trl(
    sharpe: float,
    skew: float,
    kurtosis: float,
    n_observations: int,
    confidence: float = 0.95,
    trading_days_per_year: int = 252,
    sr_star: float = 0.0,
) -> MinTRLResult:
    """Compute Minimum Track Record Length.

    Bailey & Lopez de Prado (2014): minimum monthly observations needed to
    conclude with statistical confidence that a Sharpe ratio exceeds sr_star.

    Formula (in trading-day units):
        MinTRL = (1 - skew*SR + ((kurt-1)/4)*SR^2) * (z_alpha / (SR - SR*))^2

    where:
      - SR is the *annualized* Sharpe ratio
      - skew is sample skewness of daily returns
      - kurt is sample *excess* kurtosis (normal = 0)
      - z_alpha = norm.ppf(confidence)
      - SR* = benchmark annualized Sharpe (default 0)

    Parameters
    ----------
    sharpe : float
        Annualized Sharpe ratio of the strategy.
    skew : float
        Sample skewness of daily returns.
    kurtosis : float
        Sample excess kurtosis of daily returns (normal = 0).
    n_observations : int
        Number of daily observations actually available in the backtest.
    confidence : float
        Statistical confidence level (default 0.95 → z = 1.645).
    trading_days_per_year : int
        Annualization factor (default 252).
    sr_star : float
        Benchmark annualized Sharpe ratio (default 0 = cash).

    Returns
    -------
    MinTRLResult
        min_trl_months, backtest_months, and pass/fail.
    """
    backtest_months = n_observations / (trading_days_per_year / 12.0)

    if sharpe <= sr_star:
        # Strategy doesn't beat benchmark — MinTRL is infinite
        return MinTRLResult(
            min_trl_months=float("inf"),
            backtest_months=backtest_months,
            sharpe=sharpe,
            skew=skew,
            kurtosis=kurtosis,
            confidence=confidence,
            min_trl_pass=False,
        )

    z_alpha = float(stats.norm.ppf(confidence))
    sr_diff = sharpe - sr_star

    # Adjustment factor for non-normality
    # Uses *excess* kurtosis (normal = 0), so (kurt_excess + 3 - 1)/4 = (kurt_excess + 2)/4
    # Simplifies to: (1 - skew*SR + (kurt_excess + 2)/4 * SR^2)
    # But the original Bailey & de Prado formula uses (γ₄ - 1)/4 where γ₄ = regular kurtosis.
    # Regular kurtosis = excess kurtosis + 3, so (γ₄ - 1)/4 = (excess + 2)/4.
    adjustment = 1.0 - skew * sharpe + (kurtosis + 2.0) / 4.0 * sharpe**2

    # Guard against pathological distributions
    adjustment = max(adjustment, 0.01)

    # MinTRL in trading-day units
    min_trl_days = adjustment * (z_alpha / sr_diff) ** 2

    # Convert to months
    days_per_month = trading_days_per_year / 12.0
    min_trl_months = min_trl_days / days_per_month

    min_trl_pass = backtest_months >= min_trl_months

    logger.debug(
        "MinTRL: SR=%.3f, skew=%.3f, kurt=%.3f, adj=%.3f → "
        "%.1f months required, %.1f months available → %s",
        sharpe,
        skew,
        kurtosis,
        adjustment,
        min_trl_months,
        backtest_months,
        "PASS" if min_trl_pass else "FAIL",
    )

    return MinTRLResult(
        min_trl_months=min_trl_months,
        backtest_months=backtest_months,
        sharpe=sharpe,
        skew=skew,
        kurtosis=kurtosis,
        confidence=confidence,
        min_trl_pass=min_trl_pass,
    )


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
    annualized_sharpe: float | None = None,
    min_trl_confidence: float = 0.95,
    portfolio_returns: list[float] | None = None,
    portfolio_sr: float | None = None,
    correlation_window: int = 60,
    marginal_sr_threshold: float = 0.05,
    correlation_threshold: float = 0.30,
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
    annualized_sharpe : float | None
        Annualized Sharpe ratio of the best strategy (for MinTRL). If None,
        MinTRL is computed from best_returns.
    min_trl_confidence : float
        Statistical confidence for MinTRL computation (default 0.95).
    portfolio_returns : list[float] | None
        Daily portfolio NAV returns for portfolio admission gates.
        If None, both portfolio gates are skipped (default pass).
    portfolio_sr : float | None
        Current portfolio combined Sharpe ratio for marginal SR gate.
        Required when portfolio_returns is provided.  If None but
        portfolio_returns is given, marginal SR gate is skipped.
    correlation_window : int
        Rolling window (days) for portfolio correlation gate (default 60).
    marginal_sr_threshold : float
        Minimum ΔSR_P for admission (default 0.05).
    correlation_threshold : float
        Maximum rolling correlation to portfolio NAV for admission (default 0.30).

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

    # 7. MinTRL (informational — emits WARNING if insufficient)
    if best_returns:
        arr = np.array(best_returns)
        n_obs = len(arr)
        # Compute annualized Sharpe if not provided
        if annualized_sharpe is not None:
            sr_ann = annualized_sharpe
        else:
            from llm_quant.backtest.metrics import compute_sharpe

            sr_ann = compute_sharpe(best_returns, annualize=True)

        from scipy import stats as _stats

        skew = float(_stats.skew(arr, bias=False))
        kurt = float(_stats.kurtosis(arr, bias=False))  # excess kurtosis

        result.min_trl = compute_min_trl(
            sharpe=sr_ann,
            skew=skew,
            kurtosis=kurt,
            n_observations=n_obs,
            confidence=min_trl_confidence,
        )
    else:
        result.min_trl = MinTRLResult()

    # 8. Portfolio admission gates (taff: marginal SR, r5j4: correlation)
    if portfolio_returns is not None and best_returns:
        # Correlation gate (r5j4)
        result.portfolio_correlation = check_portfolio_correlation(
            strategy_returns=best_returns,
            portfolio_returns=portfolio_returns,
            window=correlation_window,
        )
        result.portfolio_correlation_passed = (
            result.portfolio_correlation < correlation_threshold
        )
        logger.info(
            "Portfolio correlation gate: rho=%.3f (threshold < %.2f) — %s",
            result.portfolio_correlation,
            correlation_threshold,
            "PASS" if result.portfolio_correlation_passed else "FAIL",
        )

        # Marginal SR gate (taff)
        if portfolio_sr is not None:
            if annualized_sharpe is not None:
                sr_k = annualized_sharpe
            else:
                from llm_quant.backtest.metrics import compute_sharpe

                sr_k = compute_sharpe(best_returns, annualize=True)

            result.marginal_sr_contribution = compute_marginal_sr_contribution(
                strategy_sr=sr_k,
                portfolio_sr=portfolio_sr,
                correlation=result.portfolio_correlation,
            )
            result.marginal_sr_passed = (
                result.marginal_sr_contribution >= marginal_sr_threshold
            )
            logger.info(
                "Marginal SR gate: ΔSR_P=%.4f (threshold >= %.2f) — %s",
                result.marginal_sr_contribution,
                marginal_sr_threshold,
                "PASS" if result.marginal_sr_passed else "FAIL",
            )
        else:
            # portfolio_sr not provided — skip marginal SR gate
            result.marginal_sr_passed = True
            logger.info("Marginal SR gate skipped (no portfolio_sr provided)")
    else:
        # No portfolio context — both admission gates default to pass
        result.marginal_sr_passed = True
        result.portfolio_correlation_passed = True

    # Overall
    result.compute_overall()
    return result


# ---------------------------------------------------------------------------
# Track C — Structural Arb Robustness Gate
# ---------------------------------------------------------------------------


@dataclass
class TrackCRobustnessResult:
    """Robustness gate result specific to Track C structural arb strategies.

    Track C applies tighter performance floors and adds market-neutrality
    and statistical-significance checks that are not required for Tracks A/B.

    Gates
    -----
    1. sharpe_gate       — annualized Sharpe >= 1.5  (vs 0.8 for Track A)
    2. maxdd_gate        — max drawdown < 10%        (vs 15% for Track A)
    3. beta_gate         — absolute SPY beta < 0.15  (market-neutrality)
    4. min_trades_gate   — at least 50 completed trades (stat significance)
    5. cost_stress_gate  — re-run at 2x fees; Sharpe still >= 1.0

    The ``overall_passed`` flag is True only when all five gates pass.
    """

    # Raw inputs (stored for audit trail)
    sharpe: float = 0.0
    max_drawdown: float = 0.0  # positive fraction, e.g. 0.08 for 8%
    beta_to_spy: float = 0.0  # absolute value
    n_trades: int = 0
    cost_stress_sharpe: float = 0.0

    # Individual gate results
    sharpe_gate: bool = False  # sharpe >= 1.5
    maxdd_gate: bool = False  # max_drawdown < 0.10
    beta_gate: bool = False  # beta_to_spy < 0.15
    min_trades_gate: bool = False  # n_trades >= 50
    cost_stress_gate: bool = False  # cost_stress_sharpe >= 1.0

    # Gate thresholds (informational)
    sharpe_threshold: float = 1.5
    maxdd_threshold: float = 0.10
    beta_threshold: float = 0.15
    min_trades_threshold: int = 50
    cost_stress_sharpe_threshold: float = 1.0

    # Overall
    overall_passed: bool = False
    gate_details: dict[str, bool] = field(default_factory=dict)

    def compute_overall(self) -> None:
        """Compute overall gate pass from individual gate results."""
        self.gate_details = {
            "sharpe_>=_1.5": self.sharpe_gate,
            "max_drawdown_<_10%": self.maxdd_gate,
            "beta_to_spy_<_0.15": self.beta_gate,
            "min_50_trades": self.min_trades_gate,
            "cost_2x_sharpe_>=_1.0": self.cost_stress_gate,
        }
        self.overall_passed = all(self.gate_details.values())


def run_track_c_robustness_gate(
    sharpe: float,
    max_drawdown: float,
    beta_to_spy: float,
    n_trades: int,
    cost_stress_sharpe: float,
    sharpe_threshold: float = 1.5,
    maxdd_threshold: float = 0.10,
    beta_threshold: float = 0.15,
    min_trades: int = 50,
    cost_stress_sharpe_threshold: float = 1.0,
) -> TrackCRobustnessResult:
    """Run the Track C structural arb robustness gate.

    All five gates must pass for the strategy to be eligible for promotion.

    Parameters
    ----------
    sharpe : float
        Annualized Sharpe ratio from the primary backtest.
    max_drawdown : float
        Maximum portfolio drawdown as a positive fraction (e.g. 0.08 = 8%).
    beta_to_spy : float
        Absolute rolling-30d beta of the strategy to SPY.  Arb strategies
        must remain near-zero beta; values >= 0.15 indicate leg imbalance.
    n_trades : int
        Total number of completed round-trip trades in the backtest.  Arb
        strategies need >= 50 trades for statistical significance.
    cost_stress_sharpe : float
        Sharpe ratio re-computed with transaction costs doubled (2x fees).
        Must remain >= 1.0 to confirm cost robustness.
    sharpe_threshold : float
        Minimum acceptable Sharpe.  Default 1.5 (vs 0.8 for Track A).
    maxdd_threshold : float
        Maximum acceptable drawdown.  Default 0.10 (vs 0.15 for Track A).
    beta_threshold : float
        Maximum acceptable absolute SPY beta.  Default 0.15.
    min_trades : int
        Minimum number of completed trades for stat significance.  Default 50.
    cost_stress_sharpe_threshold : float
        Minimum Sharpe under 2x-cost stress.  Default 1.0.

    Returns
    -------
    TrackCRobustnessResult
        Individual gate results and overall pass/fail flag.
    """
    result = TrackCRobustnessResult(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        beta_to_spy=abs(beta_to_spy),
        n_trades=n_trades,
        cost_stress_sharpe=cost_stress_sharpe,
        sharpe_threshold=sharpe_threshold,
        maxdd_threshold=maxdd_threshold,
        beta_threshold=beta_threshold,
        min_trades_threshold=min_trades,
        cost_stress_sharpe_threshold=cost_stress_sharpe_threshold,
    )

    # Gate 1: Sharpe >= threshold
    result.sharpe_gate = sharpe >= sharpe_threshold
    if not result.sharpe_gate:
        logger.warning(
            "Track C sharpe gate FAILED: %.3f < threshold %.3f",
            sharpe,
            sharpe_threshold,
        )

    # Gate 2: MaxDD < threshold
    result.maxdd_gate = max_drawdown < maxdd_threshold
    if not result.maxdd_gate:
        logger.warning(
            "Track C maxdd gate FAILED: %.2f%% >= threshold %.2f%%",
            max_drawdown * 100,
            maxdd_threshold * 100,
        )

    # Gate 3: Beta to SPY < threshold (market-neutrality check)
    result.beta_gate = abs(beta_to_spy) < beta_threshold
    if not result.beta_gate:
        logger.warning(
            "Track C beta gate FAILED: |beta|=%.3f >= threshold %.3f — "
            "strategy is not market-neutral; check leg sizing.",
            abs(beta_to_spy),
            beta_threshold,
        )

    # Gate 4: Min trades (statistical significance)
    result.min_trades_gate = n_trades >= min_trades
    if not result.min_trades_gate:
        logger.warning(
            "Track C min_trades gate FAILED: %d trades < required %d — "
            "insufficient sample size for arb stat significance.",
            n_trades,
            min_trades,
        )

    # Gate 5: Cost stress — 2x fees, Sharpe still >= cost_stress_sharpe_threshold
    result.cost_stress_gate = cost_stress_sharpe >= cost_stress_sharpe_threshold
    if not result.cost_stress_gate:
        logger.warning(
            "Track C cost_stress gate FAILED: 2x-fee Sharpe %.3f < threshold %.3f — "
            "strategy edge erodes under realistic cost assumptions.",
            cost_stress_sharpe,
            cost_stress_sharpe_threshold,
        )

    result.compute_overall()

    logger.info(
        "Track C robustness gate %s: sharpe=%.3f(%s) maxdd=%.2f%%(%s) "
        "beta=%.3f(%s) trades=%d(%s) cost_stress_sr=%.3f(%s)",
        "PASSED" if result.overall_passed else "FAILED",
        sharpe,
        "pass" if result.sharpe_gate else "FAIL",
        max_drawdown * 100,
        "pass" if result.maxdd_gate else "FAIL",
        abs(beta_to_spy),
        "pass" if result.beta_gate else "FAIL",
        n_trades,
        "pass" if result.min_trades_gate else "FAIL",
        cost_stress_sharpe,
        "pass" if result.cost_stress_gate else "FAIL",
    )

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
# Portfolio Admission Gates (Task taff + r5j4)
# ---------------------------------------------------------------------------


def compute_marginal_sr_contribution(
    strategy_sr: float,
    portfolio_sr: float,
    correlation: float,
) -> float:
    """Compute the marginal Sharpe Ratio contribution of a new strategy.

    Uses the Lopez de Prado portfolio SR approximation:
        ΔSR_P ≈ (SR_k - ρ_{kP} × SR_P) / sqrt(1 + 2×ρ_{kP}×SR_k/SR_P)

    A positive ΔSR_P means adding the strategy improves the portfolio's
    risk-adjusted return.  The admission gate requires ΔSR_P >= 0.05.

    Parameters
    ----------
    strategy_sr : float
        Annualized Sharpe ratio of the candidate strategy.
    portfolio_sr : float
        Current portfolio (combined) Sharpe ratio.
    correlation : float
        Rolling correlation between the candidate strategy returns and
        the portfolio NAV returns.  Should be in [-1, 1].

    Returns
    -------
    float
        Marginal SR contribution ΔSR_P.  Gate passes when >= 0.05.
    """
    if portfolio_sr <= 0:
        # Degenerate case: any positive-SR strategy improves a zero-SR portfolio
        return strategy_sr

    rho = float(correlation)
    sr_k = float(strategy_sr)
    sr_p = float(portfolio_sr)

    numerator = sr_k - rho * sr_p
    denominator_sq = 1.0 + 2.0 * rho * sr_k / sr_p
    # Guard against negative denominator (highly correlated, low-SR candidate)
    if denominator_sq <= 0:
        return -abs(numerator)
    denominator = math.sqrt(denominator_sq)
    return numerator / denominator


def check_portfolio_correlation(
    strategy_returns: list[float],
    portfolio_returns: list[float],
    window: int = 60,
) -> float:
    """Compute rolling 60-day average correlation between strategy and portfolio.

    Used as the portfolio admission correlation gate.  New strategies must
    have a rolling average correlation < 0.30 to the portfolio NAV returns.

    Parameters
    ----------
    strategy_returns : list[float]
        Daily strategy returns.
    portfolio_returns : list[float]
        Daily portfolio NAV returns (same length, aligned on dates).
    window : int
        Rolling window in days.  Default 60.

    Returns
    -------
    float
        Mean rolling correlation over the overlapping history.
        The admission gate requires this to be < 0.30.
    """
    n = min(len(strategy_returns), len(portfolio_returns))
    if n < window:
        logger.warning(
            "check_portfolio_correlation: only %d observations, need %d for window",
            n,
            window,
        )
        if n < 2:
            return 0.0
        # Fall back to full-period correlation
        s = np.array(strategy_returns[:n])
        p = np.array(portfolio_returns[:n])
        if np.std(s) == 0 or np.std(p) == 0:
            return 0.0
        return float(np.corrcoef(s, p)[0, 1])

    s = np.array(strategy_returns[-n:])
    p = np.array(portfolio_returns[-n:])

    rolling_corrs: list[float] = []
    for start in range(n - window + 1):
        end = start + window
        s_w = s[start:end]
        p_w = p[start:end]
        if np.std(s_w) == 0 or np.std(p_w) == 0:
            continue
        rolling_corrs.append(float(np.corrcoef(s_w, p_w)[0, 1]))

    if not rolling_corrs:
        return 0.0
    return float(np.mean(rolling_corrs))


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

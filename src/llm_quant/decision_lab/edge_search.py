"""Pre-registered lead-lag edge search: frozen-family test, holdout and placebo null.

Research only. The plan this module executes is frozen in
``docs/research/edge-search-2026-11-17-preregistration.md``. Nothing here downloads
data, places trades or approves a strategy. Inputs are a local adjusted-close CSV
and its snapshot receipt; outputs are aggregate statistics, never vendor rows.

Index convention for every array: ``returns[t]`` is the close(t-1) -> close(t)
return, ``signal[t]`` and ``state[t]`` use prices up to and including close(t),
and ``held[t]`` is the position that earns ``returns[t]``. ``held[t] = state[t-delay]``
so ``delay=1`` trades at the signal close and ``delay=2`` waits one further bar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import norm, spearmanr

from .data import require
from .statistics import holm_family

STATUS = "RETROSPECTIVE_RESEARCH_NOT_INVESTMENT_APPROVAL"
ANNUAL = 252.0
HAC_LAGS = 10

# Copied from scripts/run_paper_batch.py LEAD_LAG_PARAMS, last changed in commit
# bb35be5 (2026-04-24). slug: (leader, follower, window, entry, exit, weight).
FROZEN_FAMILY_COMMIT = "bb35be5"
FROZEN_FAMILY: dict[str, tuple[str, str, int, float, float, float]] = {
    "soxx-qqq-lead-lag": ("SOXX", "QQQ", 5, 0.02, -0.01, 0.90),
    "lqd-spy-credit-lead": ("LQD", "SPY", 5, 0.005, -0.005, 0.80),
    "agg-spy-credit-lead": ("AGG", "SPY", 5, 0.003, -0.003, 0.80),
    "hyg-spy-5d-credit-lead": ("HYG", "SPY", 5, 0.005, -0.005, 0.80),
    "agg-qqq-credit-lead": ("AGG", "QQQ", 5, 0.003, -0.003, 0.80),
    "lqd-qqq-credit-lead": ("LQD", "QQQ", 5, 0.005, -0.005, 0.80),
    "vcit-qqq-credit-lead": ("VCIT", "QQQ", 5, 0.004, -0.004, 0.80),
    "hyg-qqq-credit-lead": ("HYG", "QQQ", 5, 0.005, -0.005, 0.75),
    "emb-spy-credit-lead": ("EMB", "SPY", 5, 0.005, -0.005, 0.80),
    "agg-efa-credit-lead": ("AGG", "EFA", 5, 0.003, -0.003, 0.80),
    "tlt-spy-rate-momentum": ("TLT", "SPY", 10, 0.01, -0.01, 0.80),
    "tlt-qqq-rate-tech": ("TLT", "QQQ", 10, 0.01, -0.01, 0.80),
    "ief-qqq-rate-tech": ("IEF", "QQQ", 10, 0.005, -0.005, 0.80),
    "tlt-tqqq-leveraged-lead-lag": ("TLT", "TQQQ", 10, 0.01, -0.01, 0.30),
    "tlt-tqqq-sprint": ("TLT", "TQQQ", 10, 0.01, -0.005, 0.30),
    "tlt-soxl-sprint": ("TLT", "SOXL", 10, 0.01, -0.005, 0.30),
    "tlt-upro-sprint": ("TLT", "UPRO", 10, 0.01, -0.005, 0.30),
    "ief-tqqq-sprint": ("IEF", "TQQQ", 10, 0.005, -0.003, 0.30),
    "agg-tqqq-sprint": ("AGG", "TQQQ", 10, 0.01, -0.005, 0.30),
    "agg-upro-sprint": ("AGG", "UPRO", 10, 0.01, -0.005, 0.30),
    "vcit-tqqq-sprint": ("VCIT", "TQQQ", 10, 0.005, -0.003, 0.30),
    "lqd-tqqq-sprint": ("LQD", "TQQQ", 10, 0.01, -0.005, 0.30),
    "lqd-soxl-sprint": ("LQD", "SOXL", 10, 0.01, -0.005, 0.30),
}

PLAN = {
    "family": {
        "confirmatory_window": ["2026-04-29", "2026-11-13"],
        "replication_window": ["2021-04-29", "2026-04-28"],
        "primary_rule": "threshold_flat_in_neutral",
        "secondary_rule": "hysteresis_hold_previous",
        "delay": 1,
        "cost_per_switch": 0.0003,
        "sensitivity": {"delay": 2, "cost_per_unit_turnover": 0.0010},
    },
    "search": {
        "selection_window": ["2012-01-03", "2021-12-31"],
        "holdout_window": ["2022-01-03", "2026-11-13"],
        "windows": [5, 10, 20],
        "threshold_sigmas": [0.0, 0.5, 1.0],
        "directions": [1, -1],
        "delay": 2,
        "cost_per_unit_turnover": 0.0005,
        "sensitivity_cost": 0.0010,
        "min_switches": 40,
        "exposure_bounds": [0.1, 0.9],
        "top_k": [1, 10, 100],
        "primary_top_k": 100,
        "placebo_draws": 200,
        "placebo_min_shift": 252,
        "seed": 20261117,
    },
    "alpha": 0.05,
    "candidate_min_holdout_sharpe": 0.5,
}


# --------------------------------------------------------------------------- data


def load_prices(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read ``date,<ticker>...`` adjusted closes; blanks stay NaN, nothing is filled."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    require(len(rows) > 2 and rows[0][0] == "date", "Price file needs a date header")
    tickers = rows[0][1:]
    require(
        len(set(tickers)) == len(tickers) and bool(tickers), "Duplicate or no tickers"
    )
    dates = [row[0][:10] for row in rows[1:]]
    require(dates == sorted(set(dates)), "Dates must be unique and increasing")
    prices = np.array(
        [[float(v) if v else math.nan for v in row[1:]] for row in rows[1:]]
    )
    require(prices.shape == (len(dates), len(tickers)), "Ragged price table")
    finite = np.isfinite(prices)
    require(bool(np.all(prices[finite] > 0)), "Non-positive price")
    for col in range(prices.shape[1]):
        first = int(np.argmax(finite[:, col]))
        require(bool(finite[first:, col].all()), "Gap after first observation")
    return dates, tickers, prices


def simple_returns(prices: np.ndarray) -> np.ndarray:
    out = np.full_like(prices, math.nan)
    out[1:] = prices[1:] / prices[:-1] - 1.0
    return out


def window_mask(dates: list[str], start: str, end: str) -> np.ndarray:
    mask = np.array([start <= d <= end for d in dates])
    require(bool(mask.any()), "Empty evaluation window")
    return mask


# ------------------------------------------------------------------------- signals


def window_signal(price: np.ndarray, window: int) -> np.ndarray:
    """Trailing ``window``-bar return ending at each close; NaN until available."""
    require(type(window) is int and window >= 1, "Invalid signal window")
    out = np.full(price.shape, math.nan)
    out[window:] = price[window:] / price[:-window] - 1.0
    return out


def position_state(
    signal: np.ndarray, entry: np.ndarray, exit_: np.ndarray, *, hold_previous: bool
) -> np.ndarray:
    """Long/flat state per column from threshold rules, using no later rows.

    ``hold_previous=True``: 1 at ``signal >= entry``, 0 at ``signal <= exit``,
    otherwise unchanged. ``False``: 1 only while ``signal >= entry``.
    Missing signals keep the prior state (hysteresis) or are flat (threshold).
    """
    require(
        signal.ndim == 2 and entry.shape == exit_.shape == (signal.shape[1],),
        "Signal and threshold shapes disagree",
    )
    require(bool(np.all(entry >= exit_)), "Entry threshold below exit threshold")
    long_now = signal >= entry
    if not hold_previous:
        return long_now.astype(float)
    flat_now = signal <= exit_
    state = np.zeros(signal.shape)
    current = np.zeros(signal.shape[1])
    for row in range(signal.shape[0]):
        current = np.where(long_now[row], 1.0, np.where(flat_now[row], 0.0, current))
        state[row] = current
    return state


def held_positions(state: np.ndarray, delay: int) -> np.ndarray:
    require(type(delay) is int and delay >= 1, "Execution needs at least a one-bar lag")
    held = np.zeros(state.shape)
    held[delay:] = state[:-delay]
    return held


def switch_amount(held: np.ndarray) -> np.ndarray:
    out = np.zeros(held.shape)
    out[1:] = np.abs(held[1:] - held[:-1])
    return out


# ---------------------------------------------------------------------- statistics


def sharpe(series: np.ndarray) -> float:
    sd = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
    return float(np.mean(series) / sd * math.sqrt(ANNUAL)) if sd > 0 else math.nan


def hac_one_sided(series: np.ndarray) -> dict:
    """Newey-West mean test, H1: mean > 0. Not a profitability guarantee."""
    require(
        series.ndim == 1
        and len(series) > HAC_LAGS + 2
        and bool(np.all(np.isfinite(series))),
        "Series too short or non-finite",
    )
    if float(np.std(series)) == 0.0:
        return {"mean": float(series[0]), "t": math.nan, "p_one_sided": math.nan}
    fit = sm.OLS(series, np.ones(len(series))).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    t_value = float(fit.tvalues[0])
    return {
        "mean": float(fit.params[0]),
        "t": t_value,
        "p_one_sided": float(norm.sf(t_value)),
    }


def describe(series: np.ndarray) -> dict:
    return {
        "days": len(series),
        "annualised_mean": float(np.mean(series) * ANNUAL),
        "sharpe": sharpe(series),
        **hac_one_sided(series),
    }


def timing_excess(
    held: np.ndarray,
    follower: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
    weight: float = 1.0,
) -> dict:
    """Strategy minus same-average-exposure buy-and-hold of the follower.

    The baseline holds ``weight * mean(held)`` of the follower throughout the window
    and pays no cost, so the mean excess is ``weight * cov(held, return) - cost``.
    """
    h, r, c = held[mask], follower[mask], cost[mask]
    require(bool(np.all(np.isfinite(r))), "Follower returns missing inside the window")
    exposure = float(h.mean())
    strategy = weight * h * r - c
    return {
        "strategy": strategy,
        "baseline": weight * exposure * r,
        "excess": strategy - weight * exposure * r,
        "exposure": exposure,
        "switches": int(np.count_nonzero(switch_amount(held)[mask])),
    }


# ---------------------------------------------------------- experiment 1: family


def frozen_family(  # noqa: PLR0913 - every argument is a declared plan setting
    dates: list[str],
    tickers: list[str],
    prices: np.ndarray,
    *,
    start: str,
    end: str,
    hold_previous: bool,
    delay: int,
    cost_per_switch: float,
    cost_scales_with_weight: bool,
) -> dict:
    """Evaluate every frozen family member; no member is dropped after the fact."""
    returns, mask = simple_returns(prices), window_mask(dates, start, end)
    index = {t: i for i, t in enumerate(tickers)}
    members, excess, pvalues = {}, [], []
    for slug, (leader, follower, window, entry, exit_, weight) in FROZEN_FAMILY.items():
        require(leader in index and follower in index, f"Missing ticker for {slug}")
        signal = window_signal(prices[:, index[leader]], window)[:, None]
        state = position_state(
            signal, np.array([entry]), np.array([exit_]), hold_previous=hold_previous
        )[:, 0]
        held = held_positions(state, delay)
        unit = cost_per_switch * (weight if cost_scales_with_weight else 1.0)
        result = timing_excess(
            held, returns[:, index[follower]], unit * switch_amount(held), mask, weight
        )
        stats = describe(result["excess"])
        members[slug] = {
            "exposure": result["exposure"],
            "switches": result["switches"],
            "strategy_sharpe": sharpe(result["strategy"]),
            "matched_buy_hold_sharpe": sharpe(result["baseline"]),
            "strategy_annualised_return": float(result["strategy"].mean() * ANNUAL),
            "timing_excess": stats,
        }
        excess.append(result["excess"])
        pvalues.append(
            1.0 if math.isnan(stats["p_one_sided"]) else stats["p_one_sided"]
        )
    holm = holm_family(pvalues, declared_trials=len(FROZEN_FAMILY), alpha=PLAN["alpha"])
    for slug, adjusted, reject in zip(
        members, holm["adjusted_pvalues"], holm["reject_null"], strict=True
    ):
        members[slug]["holm_adjusted_p"] = adjusted
        members[slug]["holm_reject"] = bool(reject)
    portfolio = np.mean(np.vstack(excess), axis=0)
    return {
        "window": [start, end],
        "rule": "hysteresis_hold_previous"
        if hold_previous
        else "threshold_flat_in_neutral",
        "delay": delay,
        "cost_per_switch": cost_per_switch,
        "cost_scales_with_weight": cost_scales_with_weight,
        "equal_weight_timing_excess": describe(portfolio),
        "members_positive_excess": int(
            sum(m["timing_excess"]["mean"] > 0 for m in members.values())
        ),
        "members_holm_rejected": int(sum(holm["reject_null"])),
        "members": members,
    }


# ---------------------------------------------------------- experiment 2: search


def build_grid(
    prices: np.ndarray,
    selection: np.ndarray,
    windows: list[int],
    sigmas: list[float],
    directions: list[int],
) -> tuple[np.ndarray, list[tuple]]:
    """State for every (leader, window, sigma multiple, direction).

    Threshold scale is the leader signal's standard deviation in the selection
    window only, so the holdout never influences a threshold.
    """
    columns, thresholds, meta = [], [], []
    for leader in range(prices.shape[1]):
        for window in windows:
            signal = window_signal(prices[:, leader], window)
            scale = float(np.nanstd(signal[selection], ddof=1))
            require(
                math.isfinite(scale) and scale > 0, "Degenerate selection-window signal"
            )
            for sigma in sigmas:
                for direction in directions:
                    columns.append(direction * signal)
                    thresholds.append(sigma * scale)
                    meta.append((leader, window, sigma, direction))
    threshold = np.array(thresholds)
    state = position_state(
        np.column_stack(columns), threshold, -threshold, hold_previous=True
    )
    return state, meta


def grid_statistics(
    held: np.ndarray, cost: np.ndarray, returns: np.ndarray, mask: np.ndarray
) -> dict:
    """Timing-excess Sharpe for every (configuration, follower) pair in closed form."""
    h, c, r = held[mask], cost[mask], returns[mask]
    require(bool(np.all(np.isfinite(r))), "Follower returns missing inside the window")
    days = len(r)
    exposure = h.mean(axis=0)
    centred = h - exposure
    mean = centred.T @ r / days - c.mean(axis=0)[:, None]
    second = ((centred**2).T @ (r**2) - 2.0 * (centred * c).T @ r) / days + (c**2).mean(
        axis=0
    )[:, None]
    variance = np.maximum(second - mean**2, 0.0) * days / (days - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            variance > 0, mean / np.sqrt(variance) * math.sqrt(ANNUAL), math.nan
        )
    return {
        "sharpe": ratio,
        "exposure": exposure,
        "switches": np.count_nonzero(c, axis=0),
    }


def select_top(selection: dict, plan: dict, count: int) -> list[tuple[int, int]]:
    """Rank eligible pairs by selection-window Sharpe only; ties keep grid order."""
    low, high = plan["exposure_bounds"]
    eligible = (
        (selection["switches"] >= plan["min_switches"])
        & (selection["exposure"] >= low)
        & (selection["exposure"] <= high)
    )
    score = np.where(
        eligible[:, None] & np.isfinite(selection["sharpe"]),
        selection["sharpe"],
        -np.inf,
    )
    order = np.argsort(-score, axis=None, kind="stable")[:count]
    picks = [tuple(int(v) for v in np.unravel_index(i, score.shape)) for i in order]
    return [p for p in picks if np.isfinite(score[p])]


def portfolio_excess(
    picks: list[tuple[int, int]],
    held: np.ndarray,
    cost: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    series = [
        timing_excess(held[:, j], returns[:, f], cost[:, j], mask)["excess"]
        for j, f in picks
    ]
    return np.mean(np.vstack(series), axis=0)


def search(
    dates: list[str], tickers: list[str], prices: np.ndarray, plan: dict
) -> dict:
    """Mechanical grid search, untouched holdout and circular-shift placebo null."""
    returns = simple_returns(prices)
    sel = window_mask(dates, *plan["selection_window"])
    hold = window_mask(dates, *plan["holdout_window"])
    require(
        max(i for i, m in enumerate(sel) if m)
        < min(i for i, m in enumerate(hold) if m),
        "Selection must end before the holdout begins",
    )
    state, meta = build_grid(
        prices, sel, plan["windows"], plan["threshold_sigmas"], plan["directions"]
    )
    held = held_positions(state, plan["delay"])
    unit = plan["cost_per_unit_turnover"]
    cost = unit * switch_amount(held)
    in_sample = grid_statistics(held, cost, returns, sel)
    out_sample = grid_statistics(held, cost, returns, hold)
    largest = max(plan["top_k"])
    picks = select_top(in_sample, plan, largest)
    require(len(picks) == largest, "Too few eligible configurations")

    def name(pick: tuple[int, int]) -> dict:
        leader, window, sigma, direction = meta[pick[0]]
        return {
            "leader": tickers[leader],
            "follower": tickers[pick[1]],
            "window": window,
            "threshold_sigma": sigma,
            "direction": direction,
            "selection_sharpe": float(in_sample["sharpe"][pick]),
            "holdout_sharpe": float(out_sample["sharpe"][pick]),
        }

    low, high = plan["exposure_bounds"]
    eligible = (
        (in_sample["switches"] >= plan["min_switches"])
        & (in_sample["exposure"] >= low)
        & (in_sample["exposure"] <= high)
    )
    usable = (
        eligible[:, None]
        & np.isfinite(in_sample["sharpe"])
        & np.isfinite(out_sample["sharpe"])
    )
    sel_scores, hold_scores = in_sample["sharpe"][usable], out_sample["sharpe"][usable]
    rank = spearmanr(sel_scores, hold_scores)
    costly = plan["sensitivity_cost"] * switch_amount(held)
    top = {}
    for k in plan["top_k"]:
        chosen = picks[:k]
        top[str(k)] = {
            "selection": describe(portfolio_excess(chosen, held, cost, returns, sel)),
            "holdout": describe(portfolio_excess(chosen, held, cost, returns, hold)),
            "holdout_at_sensitivity_cost": describe(
                portfolio_excess(chosen, held, costly, returns, hold)
            ),
            "members_positive_in_holdout": int(
                sum(out_sample["sharpe"][p] > 0 for p in chosen)
            ),
        }

    rng = np.random.default_rng(plan["seed"])
    sizes = [10, 100, 1000, 10000]
    null_max, null_top_sel, null_top_hold = [], [], []
    null_by_size = {size: [] for size in sizes}
    primary = plan["primary_top_k"]
    span = len(dates)
    for _ in range(plan["placebo_draws"]):
        shift = int(
            rng.integers(plan["placebo_min_shift"], span - plan["placebo_min_shift"])
        )
        fake_held = np.roll(held, shift, axis=0)
        fake_cost = unit * switch_amount(fake_held)
        fake = grid_statistics(fake_held, fake_cost, returns, sel)
        fake_picks = select_top(fake, plan, primary)
        scores = np.sort(np.array([fake["sharpe"][p] for p in fake_picks]))[::-1]
        null_max.append(float(scores[0]))
        null_top_sel.append(
            sharpe(portfolio_excess(fake_picks, fake_held, fake_cost, returns, sel))
        )
        null_top_hold.append(
            sharpe(portfolio_excess(fake_picks, fake_held, fake_cost, returns, hold))
        )
        ok = (
            (fake["switches"] >= plan["min_switches"])
            & (fake["exposure"] >= low)
            & (fake["exposure"] <= high)
        )
        pool = fake["sharpe"][ok[:, None] & np.isfinite(fake["sharpe"])]
        for size in sizes:
            if len(pool) >= size:
                null_by_size[size].append(
                    float(rng.choice(pool, size, replace=False).max())
                )

    def exceed(null: list[float], observed: float) -> float:
        return (1 + sum(v >= observed for v in null)) / (len(null) + 1)

    actual_max = float(in_sample["sharpe"][picks[0]])
    actual_hold = top[str(primary)]["holdout"]["sharpe"]
    return {
        "plan": plan,
        "configurations_total": int(in_sample["sharpe"].size),
        "configurations_eligible": int(usable.sum()),
        "selection_sharpe_quantiles": {
            q: float(np.quantile(sel_scores, float(q)))
            for q in ("0.5", "0.9", "0.99", "0.999")
        },
        "selection_max_sharpe": actual_max,
        "eligible_positive_in_holdout_share": float(np.mean(hold_scores > 0)),
        "selection_holdout_spearman": {
            "rho": float(rank.statistic),
            "p_two_sided": float(rank.pvalue),
            "note": "configurations overlap; p-value is descriptive",
        },
        "top": top,
        "top_configurations": [name(p) for p in picks[:20]],
        "placebo": {
            "draws": plan["placebo_draws"],
            "null_max_selection_sharpe": {
                "mean": float(np.mean(null_max)),
                "p05": float(np.quantile(null_max, 0.05)),
                "p95": float(np.quantile(null_max, 0.95)),
            },
            "p_selection_max": exceed(null_max, actual_max),
            "null_primary_selection_sharpe_mean": float(np.mean(null_top_sel)),
            "null_primary_holdout_sharpe": {
                "mean": float(np.mean(null_top_hold)),
                "p05": float(np.quantile(null_top_hold, 0.05)),
                "p95": float(np.quantile(null_top_hold, 0.95)),
            },
            "p_primary_holdout": exceed(null_top_hold, actual_hold),
            "expected_null_max_by_trials": {
                str(size): float(np.mean(values))
                for size, values in null_by_size.items()
                if values
            },
        },
    }


# ------------------------------------------------------------------------- report


def verdicts(family: dict, found: dict) -> dict:
    alpha = PLAN["alpha"]
    primary = found["top"][str(PLAN["search"]["primary_top_k"])]
    fam_p = family["confirmatory_primary"]["equal_weight_timing_excess"]["p_one_sided"]
    hold = primary["holdout"]
    candidate = (
        hold["p_one_sided"] < alpha
        and hold["sharpe"] >= PLAN["candidate_min_holdout_sharpe"]
        and primary["holdout_at_sensitivity_cost"]["mean"] > 0
    )
    return {
        "h1_frozen_family_post_freeze": "SUPPORTED_AT_5PCT"
        if fam_p < alpha
        else "NOT_SUPPORTED",
        "h2_search_top100_holdout": "CANDIDATE_REQUIRES_FORWARD_TEST"
        if candidate
        else "NOT_SUPPORTED",
        "note": "Supported means retrospective evidence only, never approval to trade.",
    }


def run(prices_path: Path, snapshot_path: Path) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(prices_path.read_bytes()).hexdigest()
    require(
        digest == snapshot["sha256_adj_close_csv"], "Price file does not match receipt"
    )
    dates, tickers, prices = load_prices(prices_path)
    require(
        dates[-1] == PLAN["search"]["holdout_window"][1],
        "Snapshot end differs from plan",
    )
    plan = PLAN["family"]
    extra = plan["sensitivity"]

    def family(
        window: list[str], *, hold_previous: bool, sensitivity: bool = False
    ) -> dict:
        return frozen_family(
            dates,
            tickers,
            prices,
            start=window[0],
            end=window[1],
            hold_previous=hold_previous,
            delay=extra["delay"] if sensitivity else plan["delay"],
            cost_per_switch=(
                extra["cost_per_unit_turnover"]
                if sensitivity
                else plan["cost_per_switch"]
            ),
            cost_scales_with_weight=sensitivity,
        )

    confirm, replicate = plan["confirmatory_window"], plan["replication_window"]
    family_result = {
        "frozen_parameters_commit": FROZEN_FAMILY_COMMIT,
        "confirmatory_primary": family(confirm, hold_previous=False),
        "confirmatory_secondary_hysteresis": family(confirm, hold_previous=True),
        "confirmatory_sensitivity_delay_cost": family(
            confirm, hold_previous=False, sensitivity=True
        ),
        "replication_exploratory_threshold": family(replicate, hold_previous=False),
        "replication_exploratory_hysteresis": family(replicate, hold_previous=True),
    }
    found = search(dates, tickers, prices, PLAN["search"])
    return {
        "schema_version": 1,
        "status": STATUS,
        "preregistration": "docs/research/edge-search-2026-11-17-preregistration.md",
        "data": {
            k: snapshot[k]
            for k in (
                "source",
                "field",
                "retrieved_utc",
                "request",
                "tickers",
                "rows",
                "first",
                "last",
                "sha256_adj_close_csv",
            )
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "family": family_result,
        "search": found,
        "verdicts": verdicts(family_result, found),
    }


def _json_safe(value):
    """Strict JSON has no NaN; an undefined statistic is published as null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prices, args.snapshot)
    args.out.write_text(
        json.dumps(_json_safe(result), indent=1, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["verdicts"], indent=1))  # noqa: T201 - CLI receipt


if __name__ == "__main__":
    main()

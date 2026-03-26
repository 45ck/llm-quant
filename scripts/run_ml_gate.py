"""Walk-forward ML gate training and A/B comparison.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_ml_gate.py \
        --slug soxx-qqq-lead-lag \
        --years 5

What this script does:
  1. Loads the strategy's frozen research spec and creates the strategy
  2. Runs a BASE backtest (no ML gate) to establish the benchmark
  3. Builds training data: (features at signal date, label = profitable?)
  4. Walk-forward: for each quarterly window
       - Train LogisticRegressionGate on expanding data up to train_end - 10d purge
       - Apply gate on next quarter's signals
       - Stitch OOS windows into a single ML-gated return series
  5. Runs the ML-gated backtest and computes Sharpe, MaxDD, gate stats
  6. Prints A/B comparison table

Anti-overfitting discipline:
  - Labels are ONLY from trades within the current OOS window (no future)
  - Purge buffer = 10 trading days (prevents rolling-feature leakage)
  - The feature set is FIXED at 5 features — no selection performed here
  - Threshold is FIXED at 0.5 — not optimised
  - Gate coefficients are printed to inspect economic interpretability
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_spec_and_create_strategy(strat_dir: Path):
    """Load frozen research spec and instantiate strategy."""
    from llm_quant.backtest.artifacts import ensure_frozen_spec
    from llm_quant.backtest.engine import CostModel
    from scripts.run_backtest import (  # type: ignore[import]
        _build_strategy_config,
        create_strategy,
    )

    spec_path = strat_dir / "research-spec.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"No research-spec.yaml at {spec_path}")

    ensure_frozen_spec(strat_dir)

    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    cost_model = CostModel.from_spec(spec)
    strategy_config = _build_strategy_config(spec)
    strategy = create_strategy(spec["strategy_type"], strategy_config)
    return strategy, spec, cost_model


def _load_price_and_indicator_data(spec: dict, years: int = 5):
    """Load price + indicator data for the strategy's symbols."""
    from llm_quant.data.fetcher import DataFetcher
    from llm_quant.data.indicators import compute_indicators

    symbols = []
    params = spec.get("parameters", {})
    for key in ("symbols", "leader_symbol", "follower_symbol", "symbol"):
        val = params.get(key)
        if val:
            if isinstance(val, list):
                symbols.extend(val)
            else:
                symbols.append(val)

    # Add regime symbols for ML features
    for extra in ("SPY", "VIX", "HYG", "TLT", "AGG"):
        if extra not in symbols:
            symbols.append(extra)

    fetcher = DataFetcher(data_dir=str(ROOT / "data"))
    lookback_days = years * 365 + 60  # a bit extra for indicator warmup
    prices_df = fetcher.fetch_multiple(symbols, lookback_days=lookback_days)
    indicators_df = compute_indicators(prices_df)
    return prices_df, indicators_df


def _run_base_backtest(strategy, prices_df, indicators_df, spec, cost_model, slug):
    """Run the baseline backtest without ML gate."""
    from llm_quant.backtest.engine import BacktestEngine

    engine = BacktestEngine(strategy=strategy, data_dir=str(ROOT / "data"))
    warmup_days = spec.get("warmup_days", 200)
    return engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug,
        cost_model=cost_model,
        fill_delay=1,
        warmup_days=warmup_days,
        cost_multiplier=1.0,
    )


def _build_training_data(
    base_result,
    prices_df: pl.DataFrame,
    indicators_df: pl.DataFrame,
    follower_symbol: str,
    holding_period_days: int = 5,
):
    """Build (features, labels) from base backtest trades.

    For each BUY trade:
      - features = gate features extracted at the trade date
      - label = 1 if trade was profitable (positive forward return), else 0
    """
    from llm_quant.backtest.ml_features import extract_gate_features

    # Build price series for follower to compute forward returns
    fol_prices = (
        prices_df.filter(pl.col("symbol") == follower_symbol)
        .sort("date")
        .select(["date", "close"])
    )
    price_map: dict[date, float] = {row[0]: row[1] for row in fol_prices.iter_rows()}
    all_dates = sorted(price_map.keys())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    trade_dates: list[date] = []

    for trade in base_result.trades:
        if trade.action != "buy":
            continue

        # Compute forward return
        entry_date = trade.date
        idx = date_to_idx.get(entry_date)
        if idx is None:
            continue
        exit_idx = min(idx + holding_period_days, len(all_dates) - 1)
        exit_date = all_dates[exit_idx]

        entry_price = price_map.get(entry_date)
        exit_price = price_map.get(exit_date)
        if entry_price is None or exit_price is None or entry_price <= 0:
            continue

        fwd_return = exit_price / entry_price - 1.0
        # Cost-adjusted threshold: ~10bps round-trip
        label = 1 if fwd_return > 0.001 else 0

        # Extract features at entry date (causal: data up to entry_date)
        causal = indicators_df.filter(pl.col("date") <= entry_date)
        feats = extract_gate_features(entry_date, causal, follower_symbol)
        features_list.append(feats)
        labels_list.append(label)
        trade_dates.append(entry_date)

    if not features_list:
        return np.empty((0, 5)), np.empty(0), []

    return (
        np.vstack(features_list),
        np.array(labels_list, dtype=int),
        trade_dates,
    )


def _walk_forward_ml_backtest(
    strategy,
    prices_df,
    indicators_df,
    spec,
    cost_model,
    slug,
    all_features,
    all_labels,
    all_trade_dates,
    warmup_days: int = 200,
    initial_train_days: int = 504,
    retrain_frequency: int = 63,
    purge_days: int = 10,
):
    """Walk-forward: train gate on expanding window, backtest on OOS window."""
    from llm_quant.backtest.engine import BacktestEngine
    from llm_quant.backtest.ml_gate import LogisticRegressionGate

    params = spec.get("parameters", {})
    follower_symbol = params.get("follower_symbol") or params.get("symbol", "SPY")
    holding_period = spec.get("rebalance_frequency_days", 5)

    all_dates = sorted(prices_df.select("date").unique().to_series().to_list())
    trading_dates = all_dates[warmup_days:]

    if len(trading_dates) <= initial_train_days:
        print("Not enough trading dates for walk-forward training.")
        return None

    print("\nWalk-forward ML gate training:")
    print(f"  Initial train window: {initial_train_days} days")
    print(f"  Retrain frequency:    {retrain_frequency} days (quarterly)")
    print(f"  Purge buffer:         {purge_days} days")
    print(
        f"  Total trade samples:  {len(all_labels)} (pos={all_labels.sum()}, neg={len(all_labels) - all_labels.sum()})"
    )

    # Build walk-forward windows
    windows = []
    for start_idx in range(initial_train_days, len(trading_dates), retrain_frequency):
        train_end = trading_dates[start_idx - 1]
        test_start_idx = start_idx + purge_days
        test_end_idx = min(
            start_idx + retrain_frequency + purge_days, len(trading_dates)
        )
        if test_start_idx >= len(trading_dates):
            break
        test_start = trading_dates[test_start_idx]
        test_end = trading_dates[test_end_idx - 1]
        windows.append((train_end, test_start, test_end))

    # For each window: train gate on [start, train_end], note which test-period signals to filter
    gate_decisions: dict[date, bool] = {}  # date -> should_allow
    final_gate = None

    for train_end, test_start, test_end in windows:
        # Filter training samples to those before train_end - purge
        purge_cutoff = _subtract_trading_days(all_dates, train_end, purge_days)
        train_mask = np.array([d <= purge_cutoff for d in all_trade_dates])

        if train_mask.sum() < 10:
            # Not enough training data yet — pass all signals
            for d in _dates_in_range(trading_dates, test_start, test_end):
                gate_decisions[d] = True
            continue

        gate = LogisticRegressionGate(
            follower_symbol=follower_symbol,
            holding_period_days=holding_period,
        )
        train_meta = gate.train(
            all_features[train_mask],
            all_labels[train_mask],
            train_end_date=train_end,
        )
        if not train_meta.get("trained", False):
            for d in _dates_in_range(trading_dates, test_start, test_end):
                gate_decisions[d] = True
            continue

        # Predict for each test-period date that had a BUY signal in base run
        for signal_date in all_trade_dates:
            if test_start <= signal_date <= test_end:
                causal = indicators_df.filter(pl.col("date") <= signal_date)
                decision = gate.predict(signal_date, causal, follower_symbol)
                gate_decisions[signal_date] = decision.allow

        final_gate = gate
        print(
            f"  Window train_end={train_end} test={test_start}→{test_end} "
            f"n_train={train_mask.sum()} "
            f"acc={train_meta.get('train_accuracy', 0):.3f} "
            f"balance={train_meta.get('class_balance', 0):.2f}"
        )

    if final_gate and final_gate.is_trained():
        print("\n  Feature coefficients (positive = increases P(allow)):")
        for name, coef in final_gate.get_coefficients().items():
            print(f"    {name:30s}: {coef:+.4f}")

    # Run ML-gated backtest using a pre-built gate that replays the walk-forward decisions
    class _ReplayGate:
        """Gate that replays pre-computed walk-forward decisions."""

        _decisions = gate_decisions
        _follower = follower_symbol

        def is_trained(self):
            return True

        def predict(self, as_of_date, indicators_df, follower_symbol=None):
            from llm_quant.backtest.ml_gate import GateDecision

            allow = self._decisions.get(as_of_date, True)
            return GateDecision(
                allow=allow,
                confidence=0.7 if allow else 0.3,
                regime_label="risk_on" if allow else "risk_off",
            )

        def filter_signals(self, signals, decision):
            from llm_quant.brain.models import Action

            if decision.allow:
                return list(signals), []
            approved = [s for s in signals if s.action in (Action.CLOSE, Action.SELL)]
            rejected = [
                s for s in signals if s.action not in (Action.CLOSE, Action.SELL)
            ]
            return approved, rejected

    replay_gate = _ReplayGate()
    ml_engine = BacktestEngine(
        strategy=strategy,
        data_dir=str(ROOT / "data"),
        ml_gate=replay_gate,
    )
    ml_result = ml_engine.run(
        prices_df=prices_df,
        indicators_df=indicators_df,
        slug=slug,
        cost_model=cost_model,
        fill_delay=1,
        warmup_days=warmup_days,
        cost_multiplier=1.0,
    )

    # Gate statistics
    n_blocked = sum(1 for d, allow in gate_decisions.items() if not allow)
    n_allowed = sum(1 for d, allow in gate_decisions.items() if allow)
    block_rate = n_blocked / len(gate_decisions) if gate_decisions else 0

    return ml_result, block_rate, n_blocked, n_allowed


def _subtract_trading_days(all_dates: list, d: date, n: int) -> date:
    """Return the date n trading days before d."""
    try:
        idx = all_dates.index(d)
        return all_dates[max(0, idx - n)]
    except ValueError:
        return d - timedelta(days=n * 1.5)


def _dates_in_range(dates: list, start: date, end: date) -> list[date]:
    return [d for d in dates if start <= d <= end]


def _print_comparison(base_result, ml_result, block_rate: float) -> None:
    """Print A/B comparison table."""
    base_m = next(iter(base_result.metrics.values())) if base_result.metrics else None
    ml_m = next(iter(ml_result.metrics.values())) if ml_result.metrics else None

    if base_m is None or ml_m is None:
        print("Could not compute metrics.")
        return

    print("\n" + "=" * 60)
    print("ML GATE A/B COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Base':>10} {'ML Gate':>10} {'Delta':>10}")
    print("-" * 60)

    def _row(label, base_val, ml_val, fmt=".4f", pct=False):
        suffix = "%" if pct else ""
        bv = base_val * 100 if pct else base_val
        mv = ml_val * 100 if pct else ml_val
        delta = mv - bv
        sign = "+" if delta >= 0 else ""
        print(
            f"{label:<30} {bv:>10{fmt}}{suffix} {mv:>10{fmt}}{suffix} "
            f"{sign}{delta:>{8}{fmt}}{suffix}"
        )

    _row("Sharpe ratio", base_m.sharpe_ratio, ml_m.sharpe_ratio)
    _row("Max drawdown", base_m.max_drawdown * 100, ml_m.max_drawdown * 100, fmt=".2f")
    _row("CAGR", base_m.cagr * 100, ml_m.cagr * 100, fmt=".2f")
    _row("Calmar ratio", base_m.calmar_ratio, ml_m.calmar_ratio)
    _row("Win rate", base_m.win_rate * 100, ml_m.win_rate * 100, fmt=".1f")
    _row("DSR", base_m.dsr, ml_m.dsr)

    n_base_trades = len(base_result.trades)
    n_ml_trades = len(ml_result.trades)
    print(f"\n{'Trades (base):':<30} {n_base_trades:>10}")
    print(f"{'Trades (ML-gated):':<30} {n_ml_trades:>10}")
    print(f"{'Block rate:':<30} {block_rate * 100:>9.1f}%")

    sharpe_delta = ml_m.sharpe_ratio - base_m.sharpe_ratio
    dd_delta = ml_m.max_drawdown - base_m.max_drawdown

    print("\n" + "=" * 60)
    print("GATE VERDICT")
    print("=" * 60)
    passes = 0
    fails = 0

    def _check(label, passed, detail=""):
        nonlocal passes, fails
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        if passed:
            passes += 1
        else:
            fails += 1
        print(f"  [{status}] {symbol} {label}{' — ' + detail if detail else ''}")

    _check(
        "Sharpe improvement >= 0.10",
        sharpe_delta >= 0.10,
        f"delta={sharpe_delta:+.4f}",
    )
    _check(
        "MaxDD not worse",
        dd_delta <= 0.005,
        f"delta={dd_delta * 100:+.2f}%",
    )
    _check(
        "Block rate 10-50% (gate is informative but not too aggressive)",
        0.10 <= block_rate <= 0.50,
        f"{block_rate * 100:.1f}%",
    )
    _check(
        "ML Sharpe >= base strategy gate (>= 0.80)",
        ml_m.sharpe_ratio >= 0.80,
        f"{ml_m.sharpe_ratio:.4f}",
    )

    print(f"\n  Result: {passes}/{passes + fails} gates pass")
    if passes == passes + fails:
        print("  → PROCEED to full robustness testing (DSR, CPCV, perturbation)")
        print(
            "  → Register as new trial in experiment-registry.jsonl (ml_variant=true)"
        )
    elif sharpe_delta >= 0:
        print("  → MARGINAL — run full robustness before deciding")
    else:
        print("  → REJECT — ML gate degraded strategy performance")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward ML gate A/B comparison")
    parser.add_argument("--slug", required=True, help="Strategy slug")
    parser.add_argument("--years", type=int, default=5, help="Backtest years")
    parser.add_argument(
        "--initial-train-days",
        type=int,
        default=504,
        help="Initial training window (trading days)",
    )
    parser.add_argument(
        "--retrain-freq",
        type=int,
        default=63,
        help="Retrain frequency (trading days, default=63=quarterly)",
    )
    parser.add_argument(
        "--purge", type=int, default=10, help="Purge buffer (trading days)"
    )
    args = parser.parse_args()

    strat_dir = ROOT / "data" / "strategies" / args.slug
    if not strat_dir.exists():
        print(f"Error: strategy directory not found: {strat_dir}")
        sys.exit(1)

    print(f"Loading strategy: {args.slug}")
    strategy, spec, cost_model = _load_spec_and_create_strategy(strat_dir)

    print(f"Loading {args.years}yr price + indicator data...")
    prices_df, indicators_df = _load_price_and_indicator_data(spec, years=args.years)

    warmup_days = spec.get("warmup_days", 200)

    print("\nRunning BASE backtest (no ML gate)...")
    base_result = _run_base_backtest(
        strategy, prices_df, indicators_df, spec, cost_model, args.slug
    )
    base_m = next(iter(base_result.metrics.values())) if base_result.metrics else None
    if base_m:
        print(
            f"  Base: Sharpe={base_m.sharpe_ratio:.4f} "
            f"MaxDD={base_m.max_drawdown * 100:.2f}% "
            f"Trades={len(base_result.trades)}"
        )

    params = spec.get("parameters", {})
    follower_symbol = params.get("follower_symbol") or params.get("symbol", "SPY")
    holding_period = spec.get("rebalance_frequency_days", 5)

    print("\nBuilding training data from base backtest trades...")
    all_features, all_labels, all_trade_dates = _build_training_data(
        base_result, prices_df, indicators_df, follower_symbol, holding_period
    )

    if len(all_labels) < 20:
        print(
            f"Only {len(all_labels)} BUY trades found — insufficient for ML training."
        )
        sys.exit(1)

    print(
        f"  {len(all_labels)} BUY trades: {all_labels.sum()} profitable ({all_labels.mean() * 100:.1f}%)"
    )

    print("\nRunning walk-forward ML-gated backtest...")
    wf_result = _walk_forward_ml_backtest(
        strategy=strategy,
        prices_df=prices_df,
        indicators_df=indicators_df,
        spec=spec,
        cost_model=cost_model,
        slug=args.slug,
        all_features=all_features,
        all_labels=all_labels,
        all_trade_dates=all_trade_dates,
        warmup_days=warmup_days,
        initial_train_days=args.initial_train_days,
        retrain_frequency=args.retrain_freq,
        purge_days=args.purge,
    )

    if wf_result is None:
        print("Walk-forward failed.")
        sys.exit(1)

    ml_result, block_rate, _n_blocked, _n_allowed = wf_result
    _print_comparison(base_result, ml_result, block_rate)


if __name__ == "__main__":
    main()

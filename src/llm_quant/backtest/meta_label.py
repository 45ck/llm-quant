"""ML Meta-Labeling: XGBoost filter on top of hypothesis-driven strategies.

Implements Lopez de Prado's two-model architecture (AFML ch. 3):
- Primary model: existing lead-lag strategy (decides SIDE)
- Secondary model: XGBoost meta-labeler (decides SIZE / whether to act)

Also includes rule-based alternatives (regime filter, signal strength weighting)
that work without ML training data.

Design constraints:
- Pre-registered XGBoost config (no hyperparameter search)
- 24 fixed features in 5 groups (leader, follower, regime, calendar, strategy)
- Triple-barrier labeling with intraday high/low barrier checks
- Temporal train/test split with embargo (purge buffer)
- SHAP TreeExplainer for feature importance
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature names (frozen -- do not reorder without versioning)
# ---------------------------------------------------------------------------

ALL_FEATURE_NAMES: list[str] = [
    # Group A: Leader Signal (4)
    "leader_return",
    "leader_return_zscore",
    "leader_volume_ratio",
    "leader_rsi_14",
    # Group B: Follower Technical (10)
    "follower_rsi_14",
    "follower_macd_hist",
    "follower_bb_pct_b",
    "follower_atr_pct",
    "follower_sma20_dist",
    "follower_sma50_dist",
    "follower_sma200_dist",
    "follower_intraday_return",
    "follower_volume_ratio",
    "follower_trailing_20d_return",
    # Group C: Regime (4)
    "vix_level",
    "vix_zscore",
    "vix_change_5d",
    "spy_vs_sma200",
    # Group D: Calendar (3)
    "day_of_week",
    "month",
    "is_pre_fomc",
    # Group E: Strategy Self-Assessment (3)
    "rolling_win_rate_10",
    "avg_pnl_last_10",
    "days_since_last_trade",
]

# Pre-registered XGBoost hyperparameters (no search allowed)
META_MODEL_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}

# FOMC announcement dates (imported from strategies.py at runtime to avoid
# circular import -- we duplicate the set here for standalone use)
_FOMC_DATES: frozenset[date] = frozenset(
    date(int(y), int(m), int(d))
    for y, m, d in [
        (2021, 1, 27),
        (2021, 3, 17),
        (2021, 4, 28),
        (2021, 6, 16),
        (2021, 7, 28),
        (2021, 9, 22),
        (2021, 11, 3),
        (2021, 12, 15),
        (2022, 1, 26),
        (2022, 3, 16),
        (2022, 5, 4),
        (2022, 6, 15),
        (2022, 7, 27),
        (2022, 9, 21),
        (2022, 11, 2),
        (2022, 12, 14),
        (2023, 2, 1),
        (2023, 3, 22),
        (2023, 5, 3),
        (2023, 6, 14),
        (2023, 7, 26),
        (2023, 9, 20),
        (2023, 11, 1),
        (2023, 12, 13),
        (2024, 1, 31),
        (2024, 3, 20),
        (2024, 5, 1),
        (2024, 6, 12),
        (2024, 7, 31),
        (2024, 9, 18),
        (2024, 11, 7),
        (2024, 12, 18),
        (2025, 1, 29),
        (2025, 3, 19),
        (2025, 5, 7),
        (2025, 6, 18),
        (2025, 7, 30),
        (2025, 9, 17),
        (2025, 11, 5),
        (2025, 12, 17),
        (2026, 1, 28),
        (2026, 3, 18),
        (2026, 4, 29),
    ]
)


def _is_pre_fomc(d: date, window: int = 2) -> bool:
    """True if *d* is within *window* calendar days before an FOMC date."""
    for fomc in _FOMC_DATES:
        delta = (fomc - d).days
        if 0 < delta <= window:
            return True
    return False


# ---------------------------------------------------------------------------
# Triple-Barrier Labeling
# ---------------------------------------------------------------------------


def apply_triple_barrier(  # noqa: PLR0911
    entry_date: date,
    symbol: str,
    prices_df: pl.DataFrame,
    upper_pct: float = 0.03,
    lower_pct: float = 0.05,
    max_holding_days: int = 10,
) -> int:
    """Label a trade entry using triple-barrier method.

    Uses intraday high/low to check if barriers were touched within each day,
    falling back to close-only if high/low columns are not available.

    Returns 1 if upper barrier hit first, 0 if stop loss or time expiry.
    """
    future = (
        prices_df.filter((pl.col("symbol") == symbol) & (pl.col("date") > entry_date))
        .sort("date")
        .head(max_holding_days)
    )
    if len(future) == 0:
        return 0

    entry_row = prices_df.filter(
        (pl.col("symbol") == symbol) & (pl.col("date") == entry_date)
    )
    if len(entry_row) == 0:
        return 0
    entry_price = entry_row["close"][0]
    if entry_price <= 0:
        return 0

    upper = entry_price * (1 + upper_pct)
    lower = entry_price * (1 - lower_pct)

    has_high = "high" in future.columns
    has_low = "low" in future.columns

    for row in future.iter_rows(named=True):
        if has_high and has_low:
            high = row.get("high", row["close"])
            low = row.get("low", row["close"])
            # Check if both barriers hit on same day -- lower takes priority
            # (conservative: assume stop hit before target on volatile day)
            if low <= lower:
                return 0
            if high >= upper:
                return 1
        else:
            close = row["close"]
            if close >= upper:
                return 1
            if close <= lower:
                return 0

    return 0  # time expiry


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(  # noqa: PLR0912, PLR0913, PLR0915, C901
    signal_date: date,
    leader_symbol: str,
    follower_symbol: str,
    leader_return: float,
    indicators_df: pl.DataFrame,
    prices_df: pl.DataFrame,  # noqa: ARG001
    trade_history: list[dict] | None = None,
) -> dict[str, float]:
    """Extract 24-feature vector at signal time. All features causal.

    Returns dict of {feature_name: value}. NaN for unavailable features.
    """
    features: dict[str, float] = {}

    # Causal filter
    ind = indicators_df.filter(pl.col("date") <= signal_date)

    # --- Group A: Leader Signal Features ---
    features["leader_return"] = leader_return

    # Leader return z-score (normalized by 63-day rolling std)
    leader_data = ind.filter(pl.col("symbol") == leader_symbol).sort("date").tail(64)
    if len(leader_data) >= 20:
        closes = leader_data["close"].to_list()
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        if rets:
            mu = sum(rets) / len(rets)
            std = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
            features["leader_return_zscore"] = (
                (leader_return - mu) / std if std > 0 else 0.0
            )
        else:
            features["leader_return_zscore"] = float("nan")
    else:
        features["leader_return_zscore"] = float("nan")

    # Leader volume ratio
    leader_last = ind.filter(pl.col("symbol") == leader_symbol).sort("date").tail(1)
    if len(leader_last) > 0 and "vol_sma_20" in leader_last.columns:
        vol = leader_last["volume"][0]
        vsma = leader_last["vol_sma_20"][0]
        features["leader_volume_ratio"] = (
            vol / vsma if vsma and vsma > 0 else float("nan")
        )
    else:
        features["leader_volume_ratio"] = float("nan")

    # Leader RSI
    if len(leader_last) > 0 and "rsi_14" in leader_last.columns:
        features["leader_rsi_14"] = float(leader_last["rsi_14"][0] or float("nan"))
    else:
        features["leader_rsi_14"] = float("nan")

    # --- Group B: Follower Technical Indicators ---
    follower_last = ind.filter(pl.col("symbol") == follower_symbol).sort("date").tail(1)

    def _get_ind(col: str) -> float:
        if len(follower_last) > 0 and col in follower_last.columns:
            val = follower_last[col][0]
            return float(val) if val is not None else float("nan")
        return float("nan")

    features["follower_rsi_14"] = _get_ind("rsi_14")
    features["follower_macd_hist"] = _get_ind("macd_hist")
    features["follower_bb_pct_b"] = _get_ind("bb_pct_b")

    # ATR normalized by close
    atr = _get_ind("atr_14")
    close = _get_ind("close")
    features["follower_atr_pct"] = (
        atr / close if close > 0 and not math.isnan(atr) else float("nan")
    )

    # SMA distances
    sma20 = _get_ind("sma_20")
    sma50 = _get_ind("sma_50")
    sma200 = _get_ind("sma_200")
    features["follower_sma20_dist"] = (
        (close - sma20) / sma20 if sma20 > 0 and not math.isnan(sma20) else float("nan")
    )
    features["follower_sma50_dist"] = (
        (close - sma50) / sma50 if sma50 > 0 and not math.isnan(sma50) else float("nan")
    )
    features["follower_sma200_dist"] = (
        (close - sma200) / sma200
        if sma200 > 0 and not math.isnan(sma200)
        else float("nan")
    )

    features["follower_intraday_return"] = _get_ind("intraday_return")

    # Follower volume ratio
    fvol = _get_ind("volume")
    fvsma = _get_ind("vol_sma_20")
    features["follower_volume_ratio"] = (
        fvol / fvsma if fvsma > 0 and not math.isnan(fvsma) else float("nan")
    )

    # Trailing 20d return
    follower_20 = ind.filter(pl.col("symbol") == follower_symbol).sort("date").tail(21)
    if len(follower_20) >= 20:
        fc = follower_20["close"].to_list()
        features["follower_trailing_20d_return"] = (
            fc[-1] / fc[0] - 1.0 if fc[0] > 0 else float("nan")
        )
    else:
        features["follower_trailing_20d_return"] = float("nan")

    # --- Group C: Regime Features ---
    vix_data = ind.filter(pl.col("symbol") == "VIX").sort("date").tail(64)
    if len(vix_data) >= 5:
        vix_closes = vix_data["close"].to_list()
        features["vix_level"] = vix_closes[-1]
        if len(vix_closes) >= 63:
            vmu = sum(vix_closes[-63:]) / 63
            vstd = (sum((v - vmu) ** 2 for v in vix_closes[-63:]) / 63) ** 0.5
            features["vix_zscore"] = (vix_closes[-1] - vmu) / vstd if vstd > 0 else 0.0
        else:
            features["vix_zscore"] = float("nan")
        if len(vix_closes) >= 6:
            features["vix_change_5d"] = (
                vix_closes[-1] / vix_closes[-6] - 1.0
                if vix_closes[-6] > 0
                else float("nan")
            )
        else:
            features["vix_change_5d"] = float("nan")
    else:
        features["vix_level"] = float("nan")
        features["vix_zscore"] = float("nan")
        features["vix_change_5d"] = float("nan")

    # SPY vs SMA-200
    spy_data = ind.filter(pl.col("symbol") == "SPY").sort("date").tail(1)
    if len(spy_data) > 0 and "sma_200" in spy_data.columns:
        spy_close = spy_data["close"][0]
        spy_sma200 = spy_data["sma_200"][0]
        if spy_sma200 and spy_sma200 > 0 and spy_close:
            features["spy_vs_sma200"] = spy_close / spy_sma200 - 1.0
        else:
            features["spy_vs_sma200"] = float("nan")
    else:
        features["spy_vs_sma200"] = float("nan")

    # --- Group D: Calendar Features ---
    features["day_of_week"] = float(signal_date.weekday())
    features["month"] = float(signal_date.month)
    features["is_pre_fomc"] = 1.0 if _is_pre_fomc(signal_date) else 0.0

    # --- Group E: Strategy Self-Assessment ---
    if trade_history and len(trade_history) >= 5:
        recent = trade_history[-10:]
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        features["rolling_win_rate_10"] = wins / len(recent)
        pnls = [t.get("pnl", 0) for t in recent]
        features["avg_pnl_last_10"] = sum(pnls) / len(pnls) if pnls else float("nan")
    else:
        features["rolling_win_rate_10"] = float("nan")
        features["avg_pnl_last_10"] = float("nan")

    if trade_history:
        last_trade_date = trade_history[-1].get("date")
        if last_trade_date:
            features["days_since_last_trade"] = float(
                (signal_date - last_trade_date).days
            )
        else:
            features["days_since_last_trade"] = float("nan")
    else:
        features["days_since_last_trade"] = float("nan")

    return features


# ---------------------------------------------------------------------------
# Replay lead-lag signals (without full backtest engine)
# ---------------------------------------------------------------------------


def replay_lead_lag_signals(  # noqa: PLR0913
    indicators_df: pl.DataFrame,
    leader_symbol: str,
    follower_symbol: str,
    lag_days: int = 1,
    signal_window: int = 5,
    entry_threshold: float = 0.005,
    exit_threshold: float = -0.005,
    inverse: bool = False,  # noqa: FBT002
) -> list[dict]:
    """Replay a lead-lag strategy to extract signal dates and leader returns.

    Does NOT run through BacktestEngine (no fills, no portfolio, no costs).
    Only identifies dates where the strategy WOULD have generated a BUY signal.

    Returns list of {date, leader_return, action} dicts.
    """
    follower_dates = (
        indicators_df.filter(pl.col("symbol") == follower_symbol)
        .sort("date")
        .select("date")["date"]
        .to_list()
    )
    leader_df = (
        indicators_df.filter(pl.col("symbol") == leader_symbol)
        .sort("date")
        .select(["date", "close"])
    )
    leader_dates = leader_df["date"].to_list()
    leader_closes = leader_df["close"].to_list()

    if len(leader_closes) < signal_window + lag_days + 2:
        return []

    signals: list[dict] = []
    in_position = False

    for trade_date in follower_dates:
        # Get leader prices ending lag_days before trade_date
        # We need the leader close series up to the date
        leader_hist = [
            (d, c)
            for d, c in zip(leader_dates, leader_closes, strict=False)
            if d <= trade_date
        ]
        if len(leader_hist) < signal_window + lag_days:
            continue

        # The leader return is computed from prices:
        #   end_idx = len - lag_days, start_idx = end_idx - signal_window
        end_idx = len(leader_hist) - lag_days
        start_idx = end_idx - signal_window
        if start_idx < 0:
            continue

        p_start = leader_hist[start_idx][1]
        p_end = leader_hist[end_idx - 1][1]
        if p_start <= 0:
            continue
        leader_ret = p_end / p_start - 1.0

        signal_long = (
            (leader_ret >= entry_threshold)
            if not inverse
            else (leader_ret <= -entry_threshold)
        )
        signal_exit = (
            (leader_ret <= exit_threshold)
            if not inverse
            else (leader_ret >= -exit_threshold)
        )

        if signal_long and not in_position:
            signals.append(
                {
                    "date": trade_date,
                    "leader_return": leader_ret,
                    "action": "BUY",
                }
            )
            in_position = True
        elif signal_exit and in_position:
            signals.append(
                {
                    "date": trade_date,
                    "leader_return": leader_ret,
                    "action": "CLOSE",
                }
            )
            in_position = False

    return signals


# ---------------------------------------------------------------------------
# MetaLabelConfig
# ---------------------------------------------------------------------------


@dataclass
class MetaLabelConfig:
    """Pre-registered meta-labeling configuration."""

    upper_barrier_pct: float = 0.03
    lower_barrier_pct: float = 0.05
    max_holding_days: int = 10
    embargo_days: int = 10
    min_train_samples: int = 20
    min_test_samples: int = 10
    min_total_samples: int = 30
    probability_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "upper_barrier_pct": self.upper_barrier_pct,
            "lower_barrier_pct": self.lower_barrier_pct,
            "max_holding_days": self.max_holding_days,
            "embargo_days": self.embargo_days,
            "min_train_samples": self.min_train_samples,
            "min_test_samples": self.min_test_samples,
            "min_total_samples": self.min_total_samples,
            "probability_threshold": self.probability_threshold,
        }


# ---------------------------------------------------------------------------
# MetaLabelResult
# ---------------------------------------------------------------------------


@dataclass
class MetaLabelResult:
    """Results from meta-label training and evaluation."""

    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    test_auc: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    n_train: int = 0
    n_test: int = 0
    n_train_positive: int = 0
    n_test_positive: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    shap_importance: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Temporal split utility
# ---------------------------------------------------------------------------


def temporal_split(
    signal_dates: list[date],
    cutoff_date: date,
    embargo_days: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, test_mask) boolean arrays with embargo purge.

    Samples within embargo_days * 2 calendar days before the cutoff are
    excluded from both train and test (purge zone) to prevent triple-barrier
    label leakage across the boundary.
    """
    dates_arr = np.array(signal_dates)
    embargo_date = cutoff_date - timedelta(days=embargo_days * 2)

    train_mask = dates_arr < np.datetime64(embargo_date)
    test_mask = dates_arr >= np.datetime64(cutoff_date)
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# MetaLabelFilter (main class)
# ---------------------------------------------------------------------------


class MetaLabelFilter:
    """XGBoost meta-label filter for lead-lag strategies.

    Usage:
        mlf = MetaLabelFilter(config)
        dataset = mlf.build_labeled_dataset(indicators_df, prices_df, signals, ...)
        result = mlf.train(dataset, cutoff_date)
        proba = mlf.predict_proba(features_dict)
        approved, rejected = mlf.filter_signals(signals, as_of_date, indicators_df, ...)
    """

    def __init__(self, config: MetaLabelConfig | None = None) -> None:
        self.config = config or MetaLabelConfig()
        self._model: Any = None
        self._feature_names: list[str] = list(ALL_FEATURE_NAMES)
        self._result: MetaLabelResult | None = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def build_labeled_dataset(
        self,
        indicators_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        entry_signals: list[dict],
        leader_symbol: str,
        follower_symbol: str,
    ) -> dict[str, Any]:
        """Build feature matrix + labels from replayed entry signals.

        Parameters
        ----------
        indicators_df : pl.DataFrame
            Full indicator data.
        prices_df : pl.DataFrame
            Full OHLCV price data (needs high/low for triple barrier).
        entry_signals : list[dict]
            Output of replay_lead_lag_signals() filtered to action=="BUY".
        leader_symbol, follower_symbol : str
            Strategy's leader/follower assets.

        Returns
        -------
        dict with keys: features_list, labels, signal_dates, feature_names
        """
        features_list: list[dict[str, float]] = []
        labels: list[int] = []
        signal_dates: list[date] = []
        trade_history: list[dict] = []

        for sig in entry_signals:
            if sig.get("action") != "BUY":
                continue

            trade_date = sig["date"]
            leader_ret = sig["leader_return"]

            feats = extract_features(
                signal_date=trade_date,
                leader_symbol=leader_symbol,
                follower_symbol=follower_symbol,
                leader_return=leader_ret,
                indicators_df=indicators_df,
                prices_df=prices_df,
                trade_history=trade_history,
            )

            label = apply_triple_barrier(
                entry_date=trade_date,
                symbol=follower_symbol,
                prices_df=prices_df,
                upper_pct=self.config.upper_barrier_pct,
                lower_pct=self.config.lower_barrier_pct,
                max_holding_days=self.config.max_holding_days,
            )

            features_list.append(feats)
            labels.append(label)
            signal_dates.append(trade_date)

            # Update trade history for rolling features
            pnl = 1.0 if label == 1 else -1.0
            trade_history.append({"date": trade_date, "pnl": pnl})

        return {
            "features_list": features_list,
            "labels": labels,
            "signal_dates": signal_dates,
            "feature_names": self._feature_names,
        }

    def train(  # noqa: C901, PLR0915
        self,
        dataset: dict[str, Any],
        cutoff_date: date,
    ) -> MetaLabelResult:
        """Train XGBoost meta-labeler with temporal split.

        Pre-registered configuration (no hyperparameter search).
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
        except ImportError:
            logger.exception("xgboost and scikit-learn required")
            self._result = MetaLabelResult()
            return self._result

        features_list = dataset["features_list"]
        labels_list = dataset["labels"]
        signal_dates = dataset["signal_dates"]
        feature_names = dataset["feature_names"]

        if len(features_list) < self.config.min_total_samples:
            logger.warning(
                "Insufficient data for meta-labeling: %d samples (need %d)",
                len(features_list),
                self.config.min_total_samples,
            )
            self._result = MetaLabelResult()
            return self._result

        # Build feature matrix
        x_all = np.array(
            [[f.get(k, float("nan")) for k in feature_names] for f in features_list]
        )
        y = np.array(labels_list)

        # Temporal split with embargo
        train_mask, test_mask = temporal_split(
            signal_dates, cutoff_date, self.config.embargo_days
        )
        x_train, y_train = x_all[train_mask], y[train_mask]
        x_test, y_test = x_all[test_mask], y[test_mask]

        if len(x_train) < self.config.min_train_samples:
            logger.warning(
                "Insufficient train data: %d (need %d)",
                len(x_train),
                self.config.min_train_samples,
            )
            self._result = MetaLabelResult(n_train=len(x_train), n_test=len(x_test))
            return self._result

        if len(x_test) < self.config.min_test_samples:
            logger.warning(
                "Insufficient test data: %d (need %d)",
                len(x_test),
                self.config.min_test_samples,
            )
            self._result = MetaLabelResult(n_train=len(x_train), n_test=len(x_test))
            return self._result

        # Class balance weight
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        params = dict(META_MODEL_PARAMS)
        params["scale_pos_weight"] = scale_pos

        model = xgb.XGBClassifier(**params)

        # Replace NaN with -999 for XGBoost (native missing value handling)
        x_train_clean = np.nan_to_num(x_train, nan=-999.0)
        x_test_clean = np.nan_to_num(x_test, nan=-999.0)

        model.fit(x_train_clean, y_train)
        self._model = model
        self._feature_names = feature_names

        # Evaluate
        train_pred = model.predict(x_train_clean)
        test_pred = model.predict(x_test_clean)
        test_proba = model.predict_proba(x_test_clean)[:, 1]

        result = MetaLabelResult(
            train_accuracy=float(accuracy_score(y_train, train_pred)),
            test_accuracy=float(accuracy_score(y_test, test_pred)),
            test_precision=float(precision_score(y_test, test_pred, zero_division=0)),
            test_recall=float(recall_score(y_test, test_pred, zero_division=0)),
            n_train=len(x_train),
            n_test=len(x_test),
            n_train_positive=n_pos,
            n_test_positive=int(np.sum(y_test == 1)),
        )

        # AUC (may fail if only one class in test)
        try:
            result.test_auc = float(roc_auc_score(y_test, test_proba))
        except ValueError:
            result.test_auc = 0.5

        # Feature importance (gain-based)
        importance = model.get_booster().get_score(importance_type="gain")
        for fname_idx, score in importance.items():
            idx = int(fname_idx.replace("f", ""))
            if idx < len(feature_names):
                result.feature_importance[feature_names[idx]] = round(score, 4)
        result.feature_importance = dict(
            sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # SHAP feature importance
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test_clean)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            for i, name in enumerate(feature_names):
                if i < len(mean_abs_shap):
                    result.shap_importance[name] = round(float(mean_abs_shap[i]), 6)
            result.shap_importance = dict(
                sorted(
                    result.shap_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("SHAP failed: %s", e)

        # Skeptic guardrails
        gap = result.train_accuracy - result.test_accuracy
        if gap > 0.15:
            logger.warning(
                "OVERFIT WARNING: train-test accuracy gap = %.1f%%",
                gap * 100,
            )
        if result.test_auc < 0.55:
            logger.warning(
                "WEAK MODEL: test AUC = %.3f (barely above random)",
                result.test_auc,
            )
        top_fi = list(result.feature_importance.values())
        if top_fi and top_fi[0] > 0.5 * sum(top_fi):
            logger.warning("CONCENTRATION: top feature has >50%% of total importance")

        self._result = result
        return result

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return P(favorable) for a single feature dict.

        Returns 0.5 if model is not trained (no-op default).
        """
        if self._model is None:
            return 0.5

        x = np.array([[features.get(k, float("nan")) for k in self._feature_names]])
        x_clean = np.nan_to_num(x, nan=-999.0)
        return float(self._model.predict_proba(x_clean)[0][1])

    def filter_signals(  # noqa: PLR0913
        self,
        signals: list,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        leader_symbol: str,
        follower_symbol: str,
        leader_return: float,
        trade_history: list[dict] | None = None,
    ) -> tuple[list, list, float]:
        """Apply meta-label filter to trade signals.

        CLOSE/SELL signals always pass regardless of meta-label decision.
        BUY signals are filtered based on P(favorable) >= threshold.

        Returns (approved, rejected, probability).
        """
        from llm_quant.brain.models import Action

        if not self.is_trained:
            return list(signals), [], 0.5

        feats = extract_features(
            signal_date=as_of_date,
            leader_symbol=leader_symbol,
            follower_symbol=follower_symbol,
            leader_return=leader_return,
            indicators_df=indicators_df,
            prices_df=prices_df,
            trade_history=trade_history,
        )

        prob = self.predict_proba(feats)

        if prob >= self.config.probability_threshold:
            return list(signals), [], prob

        # Block BUY signals, pass CLOSE/SELL
        approved = []
        rejected = []
        for sig in signals:
            if sig.action in (Action.CLOSE, Action.SELL):
                approved.append(sig)
            else:
                rejected.append(sig)
        return approved, rejected, prob

    def get_result(self) -> MetaLabelResult | None:
        return self._result

    def save(self, path: Path) -> None:
        """Persist model and metadata to directory."""
        import joblib

        path.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            joblib.dump(self._model, path / "meta_model.joblib")

        meta = {
            "feature_names": self._feature_names,
            "config": self.config.to_dict(),
            "model_params": META_MODEL_PARAMS,
        }
        if self._result:
            meta["result"] = {
                "train_accuracy": self._result.train_accuracy,
                "test_accuracy": self._result.test_accuracy,
                "test_auc": self._result.test_auc,
                "test_precision": self._result.test_precision,
                "test_recall": self._result.test_recall,
                "n_train": self._result.n_train,
                "n_test": self._result.n_test,
                "feature_importance": self._result.feature_importance,
                "shap_importance": self._result.shap_importance,
            }

        with (path / "meta_label_meta.json").open("w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("MetaLabelFilter saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model and metadata from directory."""
        import joblib

        model_path = path / "meta_model.joblib"
        if model_path.exists():
            self._model = joblib.load(model_path)

        meta_path = path / "meta_label_meta.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            self._feature_names = meta.get("feature_names", list(ALL_FEATURE_NAMES))

        logger.info(
            "MetaLabelFilter loaded from %s (trained=%s)", path, self.is_trained
        )


# ---------------------------------------------------------------------------
# Standalone train function (backward-compatible wrapper)
# ---------------------------------------------------------------------------


def train_meta_labeler(
    features_list: list[dict[str, float]],
    labels: list[int],
    signal_dates: list[date],
    cutoff_date: date,
    embargo_days: int = 10,
) -> tuple[Any | None, MetaLabelResult]:
    """Train XGBoost meta-labeler with temporal split.

    Backward-compatible standalone function. Delegates to MetaLabelFilter.
    """
    config = MetaLabelConfig(embargo_days=embargo_days)
    mlf = MetaLabelFilter(config)

    dataset = {
        "features_list": features_list,
        "labels": labels,
        "signal_dates": signal_dates,
        "feature_names": list(ALL_FEATURE_NAMES),
    }

    result = mlf.train(dataset, cutoff_date)
    return mlf._model, result  # noqa: SLF001


# ---------------------------------------------------------------------------
# Rule-Based Alternatives (no ML needed)
# ---------------------------------------------------------------------------


def regime_filter(
    vix_level: float,
    vix_threshold: float = 25.0,
) -> bool:
    """Simple regime filter: suppress entries when VIX > threshold.

    Returns True if signal should be KEPT (VIX is calm).
    """
    return vix_level <= vix_threshold


def signal_strength_weight(
    leader_return: float,
    entry_threshold: float,
    max_multiplier: float = 1.5,
) -> float:
    """Scale position size by signal strength (conviction weighting).

    Returns multiplier [0.5, max_multiplier] based on how far the leader
    return exceeds the entry threshold.
    """
    if entry_threshold <= 0:
        return 1.0
    ratio = abs(leader_return) / abs(entry_threshold)
    return max(0.5, min(max_multiplier, ratio * 0.5 + 0.5))


def ensemble_vote(
    strategy_signals: dict[str, str],
    min_agreement: int = 2,
) -> bool:
    """Check if multiple strategies agree on direction.

    Returns True if enough strategies agree.
    """
    buy_count = sum(1 for v in strategy_signals.values() if v == "buy")
    sell_count = sum(1 for v in strategy_signals.values() if v == "sell")
    return buy_count >= min_agreement or sell_count >= min_agreement

"""ML Meta-Labeling: XGBoost filter on top of hypothesis-driven strategies.

Implements López de Prado's two-model architecture:
- Primary model: existing lead-lag strategy (decides direction)
- Secondary model: XGBoost meta-labeler (decides whether to act + bet size)

Also includes rule-based alternatives (regime filter, signal strength weighting)
that work without ML training data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triple-Barrier Labeling
# ---------------------------------------------------------------------------


def apply_triple_barrier(
    entry_date: date,
    symbol: str,
    prices_df: pl.DataFrame,
    upper_pct: float = 0.03,
    lower_pct: float = 0.05,
    max_holding_days: int = 10,
) -> int:
    """Label a trade entry using triple-barrier method.

    Returns 1 if upper barrier (profit target) hit first, 0 otherwise.
    Uses only prices AFTER entry_date (forward-looking label).

    Parameters
    ----------
    entry_date : date
        Date of trade entry signal.
    symbol : str
        Asset symbol.
    prices_df : pl.DataFrame
        Full price data with columns: symbol, date, close.
    upper_pct : float
        Profit target as fraction (e.g., 0.03 = +3%).
    lower_pct : float
        Stop loss as fraction (e.g., 0.05 = -5%).
    max_holding_days : int
        Maximum holding period in trading days.

    Returns
    -------
    int
        1 if profit target hit first, 0 if stop loss or time expiry.
    """
    future = (
        prices_df.filter((pl.col("symbol") == symbol) & (pl.col("date") > entry_date))
        .sort("date")
        .head(max_holding_days)
    )
    if len(future) == 0:
        return 0

    # Entry price = close on entry_date
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

    for row in future.iter_rows(named=True):
        close = row["close"]
        if close >= upper:
            return 1  # profit target hit first
        if close <= lower:
            return 0  # stop loss hit first

    return 0  # time expiry


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(  # noqa: PLR0912, PLR0915
    signal_date: date,
    leader_symbol: str,
    follower_symbol: str,
    leader_return: float,
    indicators_df: pl.DataFrame,
    _prices_df: pl.DataFrame,
    trade_history: list[dict] | None = None,
) -> dict[str, float]:
    """Extract feature vector at signal time. All features causal.

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

    # --- Group D: Calendar Features ---
    features["day_of_week"] = float(signal_date.weekday())
    features["month"] = float(signal_date.month)

    # --- Group E: Strategy Self-Assessment ---
    if trade_history and len(trade_history) >= 5:
        recent = trade_history[-10:]
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        features["rolling_win_rate_10"] = wins / len(recent)
    else:
        features["rolling_win_rate_10"] = float("nan")

    return features


# ---------------------------------------------------------------------------
# Meta-Label Filter (XGBoost)
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


def train_meta_labeler(
    features_list: list[dict[str, float]],
    labels: list[int],
    signal_dates: list[date],
    cutoff_date: date,
    embargo_days: int = 10,
) -> tuple[object | None, MetaLabelResult]:
    """Train XGBoost meta-labeler with temporal split.

    Pre-registered configuration (no hyperparameter search):
    max_depth=3, n_estimators=100, min_child_weight=5.

    Parameters
    ----------
    features_list : list of feature dicts
    labels : list of 0/1 labels
    signal_dates : list of dates for temporal split
    cutoff_date : date separating train/test
    embargo_days : int, purge buffer

    Returns
    -------
    (model, MetaLabelResult) or (None, MetaLabelResult) if insufficient data
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
        return None, MetaLabelResult()

    if len(features_list) < 30:
        logger.warning(
            "Insufficient data for meta-labeling: %d samples", len(features_list)
        )
        return None, MetaLabelResult()

    # Build feature matrix
    feature_names = sorted(features_list[0].keys())
    x_all = np.array(
        [[f.get(k, float("nan")) for k in feature_names] for f in features_list]
    )
    y = np.array(labels)
    dates_arr = np.array(signal_dates)

    # Temporal split with embargo
    from datetime import timedelta

    embargo_date = cutoff_date - timedelta(days=embargo_days * 2)
    train_mask = dates_arr < np.datetime64(embargo_date)
    test_mask = dates_arr >= np.datetime64(cutoff_date)

    x_train, y_train = x_all[train_mask], y[train_mask]
    x_test, y_test = x_all[test_mask], y[test_mask]

    if len(x_train) < 20 or len(x_test) < 10:
        logger.warning(
            "Insufficient train/test split: train=%d, test=%d",
            len(x_train),
            len(x_test),
        )
        return None, MetaLabelResult(n_train=len(x_train), n_test=len(x_test))

    # Pre-registered XGBoost config (NO hyperparameter search)
    scale_pos = (
        float(np.sum(y_train == 0)) / float(np.sum(y_train == 1))
        if np.sum(y_train == 1) > 0
        else 1.0
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    # Replace NaN with -999 for XGBoost
    x_train_clean = np.nan_to_num(x_train, nan=-999.0)
    x_test_clean = np.nan_to_num(x_test, nan=-999.0)

    model.fit(x_train_clean, y_train)

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
        n_train_positive=int(np.sum(y_train == 1)),
        n_test_positive=int(np.sum(y_test == 1)),
    )

    # AUC (may fail if only one class in test)
    try:
        result.test_auc = float(roc_auc_score(y_test, test_proba))
    except ValueError:
        result.test_auc = 0.5

    # Feature importance (gain-based)
    importance = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1... back to feature names
    for fname_idx, score in importance.items():
        idx = int(fname_idx.replace("f", ""))
        if idx < len(feature_names):
            result.feature_importance[feature_names[idx]] = round(score, 4)

    # Sort by importance
    result.feature_importance = dict(
        sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # SHAP (optional, may be slow)
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
    train_test_gap = result.train_accuracy - result.test_accuracy
    if train_test_gap > 0.15:
        logger.warning(
            "OVERFIT WARNING: train-test accuracy gap = %.1f%%",
            train_test_gap * 100,
        )

    if result.test_auc < 0.55:
        logger.warning(
            "WEAK MODEL: test AUC = %.3f (barely above random)",
            result.test_auc,
        )

    top_features = list(result.feature_importance.values())
    if top_features and top_features[0] > 0.5 * sum(top_features):
        logger.warning("CONCENTRATION: top feature has >50%% of total importance")

    return model, result


# ---------------------------------------------------------------------------
# Rule-Based Alternatives (no ML needed)
# ---------------------------------------------------------------------------


def regime_filter(
    vix_level: float,
    vix_threshold: float = 25.0,
) -> bool:
    """Simple regime filter: suppress entries when VIX > threshold.

    No training needed. Economically motivated: high VIX = high uncertainty
    = signal-to-noise degrades.

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
    return exceeds the entry threshold. Stronger signals → larger positions.

    No ML needed. Economically motivated: larger leader moves represent
    higher-conviction signals.
    """
    if entry_threshold <= 0:
        return 1.0
    ratio = abs(leader_return) / abs(entry_threshold)
    # Clamp to [0.5, max_multiplier]
    return max(0.5, min(max_multiplier, ratio * 0.5 + 0.5))


def ensemble_vote(
    strategy_signals: dict[str, str],
    min_agreement: int = 2,
) -> bool:
    """Check if multiple strategies agree on direction.

    Parameters
    ----------
    strategy_signals : dict[str, str]
        {strategy_slug: "buy" | "sell" | "none"} for each active strategy
        targeting the same follower asset.
    min_agreement : int
        Minimum number of strategies that must agree for signal to pass.

    Returns True if enough strategies agree.
    """
    buy_count = sum(1 for v in strategy_signals.values() if v == "buy")
    sell_count = sum(1 for v in strategy_signals.values() if v == "sell")
    return buy_count >= min_agreement or sell_count >= min_agreement

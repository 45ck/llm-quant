"""ML-based pre-trade gate for the backtest engine.

Architecture:
  - MLGate ABC defines the interface (predict / train / save / load)
  - LogisticRegressionGate is the only concrete implementation
  - The gate slots into BacktestEngine as an optional filter between
    strategy.generate_signals() and risk_manager.filter_signals()

Design constraints (enforced by quant-critic review):
  - Algorithm: Logistic Regression ONLY (fixed L2 regularization C=1.0)
  - Features: 5 pre-specified features from ml_features.py (no selection)
  - Output: binary pass/reject (no continuous position sizing)
  - Training: expanding walk-forward with 10-day purge buffer
  - Labels: forward return sign over strategy's holding period (pre-specified)
  - CLOSE/SELL signals always pass — you must always be able to exit

DSR accounting: every ML experiment (training run) MUST be appended to the
strategy's experiment-registry.jsonl under ml_variant=True. ML trials count
toward the same N as non-ML trials for DSR computation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from llm_quant.backtest.ml_features import FEATURE_NAMES, extract_gate_features
from llm_quant.brain.models import Action, TradeSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------


@dataclass
class GateDecision:
    """Output of a single ML gate evaluation."""

    allow: bool
    confidence: float  # P(favorable) from the model; 0.5 = no model
    regime_label: str  # descriptive label for logging
    feature_values: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class MLGate(ABC):
    """Pre-trade ML gate: filters BUY signals, always passes CLOSE/SELL."""

    @abstractmethod
    def predict(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        follower_symbol: str,
    ) -> GateDecision:
        """Predict whether current market conditions favour new entries."""
        ...

    @abstractmethod
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_end_date: date | None = None,
    ) -> dict[str, Any]:
        """Fit the model on (features, labels).

        Returns training metadata (accuracy, coefficients, etc.).
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefact to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model artefact from disk."""
        ...

    def is_trained(self) -> bool:
        """True if the model has been trained and can make predictions."""
        return False

    def filter_signals(
        self,
        signals: list[TradeSignal],
        decision: GateDecision,
    ) -> tuple[list[TradeSignal], list[TradeSignal]]:
        """Apply gate decision to signal list.

        CLOSE/SELL signals always pass regardless of gate decision.
        BUY signals are blocked when decision.allow is False.

        Returns
        -------
        (approved, rejected)
        """
        if decision.allow:
            return list(signals), []

        approved: list[TradeSignal] = []
        rejected: list[TradeSignal] = []
        for sig in signals:
            if sig.action in (Action.CLOSE, Action.SELL):
                approved.append(sig)
            else:
                rejected.append(sig)
        return approved, rejected


# ---------------------------------------------------------------------------
# Concrete implementation: Logistic Regression gate
# ---------------------------------------------------------------------------


class LogisticRegressionGate(MLGate):
    """Binary pre-trade gate using logistic regression.

    Model specification (frozen — do not change without versioning):
      - Algorithm: sklearn LogisticRegression
      - Penalty: L2 (C=1.0 — no tuning)
      - Solver: lbfgs
      - Max iterations: 500
      - Features: 5 fixed (see ml_features.FEATURE_NAMES)
      - Threshold: 0.5 (natural decision boundary, not optimised)

    Training discipline:
      - Expanding window with 10-day purge
      - Labels: sign of forward return over holding_period_days
      - Features extracted at signal date from causal indicator data only
    """

    THRESHOLD: float = 0.5

    def __init__(self, follower_symbol: str, holding_period_days: int = 5) -> None:
        self.follower_symbol = follower_symbol
        self.holding_period_days = holding_period_days
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._train_metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self._model is not None

    def predict(
        self,
        as_of_date: date,
        indicators_df: pl.DataFrame,
        follower_symbol: str | None = None,
    ) -> GateDecision:
        """Return gate decision for the current market state."""
        sym = follower_symbol or self.follower_symbol
        features = extract_gate_features(as_of_date, indicators_df, sym)
        feature_dict = dict(zip(FEATURE_NAMES, features[0], strict=False))

        if self._model is None or self._scaler is None:
            # No model yet — pass all signals through (conservative default)
            return GateDecision(
                allow=True,
                confidence=0.5,
                regime_label="no_model",
                feature_values=feature_dict,
                reasoning="ML gate not trained yet — passing all signals",
            )

        x_scaled = self._scaler.transform(features)
        prob = float(self._model.predict_proba(x_scaled)[0][1])
        allow = prob >= self.THRESHOLD

        if prob > 0.6:
            regime = "risk_on"
        elif prob < 0.4:
            regime = "risk_off"
        else:
            regime = "uncertain"

        return GateDecision(
            allow=allow,
            confidence=prob,
            regime_label=regime,
            feature_values=feature_dict,
            reasoning=f"P(favorable)={prob:.3f} threshold={self.THRESHOLD}",
        )

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_end_date: date | None = None,
    ) -> dict[str, Any]:
        """Fit on (features, labels).

        Parameters
        ----------
        features:
            Shape (N, 5) array of gate features.
        labels:
            Shape (N,) binary array — 1 if trade was profitable, 0 otherwise.
        train_end_date:
            Informational only (for logging / metadata).

        Returns
        -------
        dict with training metadata.
        """
        if len(features) < 10:
            logger.warning(
                "ML gate training: only %d samples — skipping (need >= 10)",
                len(features),
            )
            return {"trained": False, "reason": "insufficient samples"}

        # Drop NaN rows
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        labels = labels[mask]

        if len(features) < 10:
            return {"trained": False, "reason": "insufficient samples after NaN drop"}

        # Scale
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(features)

        # Fit logistic regression (fixed hyperparameters — no tuning allowed)
        self._model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )
        self._model.fit(x_scaled, labels)

        # Training accuracy (in-sample — for diagnostics only, not for gate evaluation)
        train_acc = float(self._model.score(x_scaled, labels))
        class_balance = float(labels.mean())
        coefs = dict(zip(FEATURE_NAMES, self._model.coef_[0], strict=False))

        self._train_metadata = {
            "trained": True,
            "n_samples": len(features),
            "train_end_date": str(train_end_date) if train_end_date else None,
            "train_accuracy": round(train_acc, 4),
            "class_balance": round(class_balance, 4),
            "coefficients": {k: round(float(v), 4) for k, v in coefs.items()},
            "intercept": round(float(self._model.intercept_[0]), 4),
        }
        logger.info(
            "ML gate trained: n=%d acc=%.3f balance=%.2f end=%s",
            len(features),
            train_acc,
            class_balance,
            train_end_date,
        )
        return self._train_metadata

    def save(self, path: Path) -> None:
        """Save model + scaler + metadata to *path* directory."""
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            joblib.dump(self._model, path / "model.joblib")
        if self._scaler is not None:
            joblib.dump(self._scaler, path / "scaler.joblib")
        meta = {
            "follower_symbol": self.follower_symbol,
            "holding_period_days": self.holding_period_days,
            "threshold": self.THRESHOLD,
            "feature_names": FEATURE_NAMES,
            "train_metadata": self._train_metadata,
        }
        with (path / "gate_meta.yaml").open("w") as f:
            yaml.dump(meta, f, sort_keys=False)
        logger.info("ML gate saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model + scaler from *path* directory."""
        model_path = path / "model.joblib"
        scaler_path = path / "scaler.joblib"
        if model_path.exists():
            self._model = joblib.load(model_path)
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
        meta_path = path / "gate_meta.yaml"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = yaml.safe_load(f)
            self._train_metadata = meta.get("train_metadata", {})
        logger.info("ML gate loaded from %s (trained=%s)", path, self.is_trained())

    # ------------------------------------------------------------------
    # Feature coefficient inspection
    # ------------------------------------------------------------------

    def get_coefficients(self) -> dict[str, float]:
        """Return feature coefficients (positive = increases P(allow))."""
        if self._model is None:
            return {}
        return dict(zip(FEATURE_NAMES, self._model.coef_[0], strict=False))

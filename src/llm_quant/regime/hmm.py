"""2-state Hidden Markov Model regime detection.

States: 0=risk_off, 1=risk_on
Based on Nystrup et al. (2019): weekly observations, ~130-day lookback.
Observation variables: VIX level (normalized), yield curve slope, SPY momentum.

Falls back to existing heuristic if insufficient data or hmmlearn unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class HmmRegimeConfig:
    n_states: int = 2
    lookback_days: int = 130
    covariance_type: str = "full"
    n_iter: int = 100
    min_observations: int = 20  # minimum weekly obs needed


@dataclass
class HmmRegimeResult:
    regime: str  # "risk_on" or "risk_off"
    confidence: float  # probability of the predicted state (0–1)
    state_probs: list[float] = field(
        default_factory=list
    )  # [prob_risk_off, prob_risk_on]
    fallback_used: bool = True  # True if fell back to heuristic


class HmmRegimeDetector:
    """2-state Gaussian HMM for market regime classification."""

    def __init__(self, config: HmmRegimeConfig | None = None) -> None:
        self.config = config or HmmRegimeConfig()

    def fit_predict(
        self,
        vix_series: pl.Series,
        yield_slope: pl.Series,
        spy_momentum: pl.Series,
    ) -> HmmRegimeResult:
        """Fit a 2-state Gaussian HMM and return the current regime.

        Parameters
        ----------
        vix_series:
            Daily VIX close prices (lookback_days length).
        yield_slope:
            Daily yield curve slope proxy (same length).
        spy_momentum:
            Daily SPY momentum (e.g. 20d return, same length).

        Returns
        -------
        HmmRegimeResult
            Regime classification with confidence. ``fallback_used=True``
            if hmmlearn is unavailable or any error occurs.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not available; using fallback regime detection")
            return HmmRegimeResult(
                regime="risk_on",
                confidence=0.5,
                state_probs=[0.5, 0.5],
                fallback_used=True,
            )

        try:
            observations = self._normalize_features(
                vix_series, yield_slope, spy_momentum
            )

            # Need at least min_observations weekly obs
            # Weekly sampling: every 5 rows from daily data
            weekly_obs = observations[::5]
            if len(weekly_obs) < self.config.min_observations:
                logger.warning(
                    "Insufficient observations for HMM (%d weekly, need %d); using fallback",
                    len(weekly_obs),
                    self.config.min_observations,
                )
                return HmmRegimeResult(
                    regime="risk_on",
                    confidence=0.5,
                    state_probs=[0.5, 0.5],
                    fallback_used=True,
                )

            model = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=42,
            )
            model.fit(weekly_obs)

            return self._decode_state(model, observations)

        except Exception:
            logger.warning(
                "HMM fitting failed; using fallback regime detection", exc_info=True
            )
            return HmmRegimeResult(
                regime="risk_on",
                confidence=0.5,
                state_probs=[0.5, 0.5],
                fallback_used=True,
            )

    def _normalize_features(self, *series: pl.Series) -> np.ndarray:
        """Z-score normalize each series and stack into observation matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_features), z-score normalized.
        """
        cols = []
        for s in series:
            arr = s.to_numpy(allow_copy=True).astype(float)
            # Drop NaN for stats, then normalize full array
            valid = arr[~np.isnan(arr)]
            if len(valid) < 2:
                # Can't normalize — fill with zeros
                cols.append(np.zeros_like(arr))
            else:
                mean = valid.mean()
                std = valid.std()
                if std == 0.0:
                    cols.append(np.zeros_like(arr))
                else:
                    cols.append((arr - mean) / std)

        obs = np.column_stack(cols)
        # Replace any remaining NaN with 0 (forward-fill would be better but
        # this is safe for HMM warm-up periods)
        return np.nan_to_num(obs, nan=0.0)

    def _decode_state(self, model: object, observations: np.ndarray) -> HmmRegimeResult:
        """Predict regime from the final observation using a fitted HMM.

        The state with the HIGHER mean VIX (feature index 0) is risk_off.
        Lower VIX mean = risk_on (state 1 in output convention).

        Parameters
        ----------
        model:
            Fitted GaussianHMM instance.
        observations:
            Full normalized observation array, shape (n_samples, n_features).

        Returns
        -------
        HmmRegimeResult
        """
        from hmmlearn.hmm import GaussianHMM

        assert isinstance(model, GaussianHMM)

        # Predict posterior probabilities for the last observation
        # We use the full sequence for Viterbi path, then take final state probs
        last_obs = observations[-1:, :]
        state_probs_last = model.predict_proba(last_obs)[0]  # shape (n_states,)

        # Identify which HMM state corresponds to risk_off: higher VIX mean
        # model.means_ shape: (n_components, n_features), feature 0 = VIX
        vix_means = model.means_[:, 0]  # normalized VIX mean per state
        risk_off_state = int(np.argmax(vix_means))
        risk_on_state = 1 - risk_off_state  # works for 2-state model

        # Build canonical state_probs = [prob_risk_off, prob_risk_on]
        prob_risk_off = float(state_probs_last[risk_off_state])
        prob_risk_on = float(state_probs_last[risk_on_state])
        state_probs = [round(prob_risk_off, 4), round(prob_risk_on, 4)]

        # Current regime = argmax of state probs
        if prob_risk_off >= prob_risk_on:
            regime = "risk_off"
            confidence = prob_risk_off
        else:
            regime = "risk_on"
            confidence = prob_risk_on

        confidence = round(float(confidence), 4)

        logger.debug(
            "HMM regime=%s confidence=%.3f [risk_off=%.3f, risk_on=%.3f]",
            regime,
            confidence,
            prob_risk_off,
            prob_risk_on,
        )

        return HmmRegimeResult(
            regime=regime,
            confidence=confidence,
            state_probs=state_probs,
            fallback_used=False,
        )

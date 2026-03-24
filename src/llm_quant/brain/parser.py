"""Parse and validate Claude's JSON response into domain dataclasses."""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from llm_quant.brain.models import (
    Action,
    Conviction,
    MarketRegime,
    TradeSignal,
    TradingDecision,
)

logger = logging.getLogger(__name__)

# Regex to extract JSON from markdown fenced code blocks
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def _extract_json_string(raw: str) -> str:
    """Extract a JSON string from raw text.

    Tries the following strategies in order:
    1. If the text contains a markdown code block, extract its content.
    2. If the text starts with ``{``, use it directly.
    3. Find the first ``{`` and last ``}`` and extract that substring.

    Parameters
    ----------
    raw:
        The raw text that may contain JSON.

    Returns
    -------
    str
        The extracted JSON string.

    Raises
    ------
    ValueError
        If no JSON-like structure can be found in the input.
    """
    stripped = raw.strip()

    # Strategy 1: fenced code block
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        return match.group(1).strip()

    # Strategy 2: already looks like JSON
    if stripped.startswith("{"):
        return stripped

    # Strategy 3: find outermost braces
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]

    raise ValueError(
        "Could not locate JSON object in response. "
        f"First 200 chars: {stripped[:200]!r}"
    )


def _parse_enum_safe(enum_cls: type, value: Any, default: Any = None) -> Any:
    """Safely parse a string into an enum, returning *default* on failure."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        # Try direct value match
        try:
            return enum_cls(value.lower().strip())
        except ValueError:
            pass
        # Try name match
        try:
            return enum_cls[value.upper().strip()]
        except KeyError:
            pass
    logger.warning("Invalid %s value: %r; using default %s", enum_cls.__name__, value, default)
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the range [lo, hi]."""
    return max(lo, min(hi, value))


def _parse_signal(raw_signal: dict[str, Any]) -> TradeSignal | None:
    """Parse a single trade signal dict into a TradeSignal.

    Returns None if the signal cannot be parsed into a valid state
    (e.g. missing required fields or unrecognizable enum values).
    """
    symbol = raw_signal.get("symbol")
    if not symbol or not isinstance(symbol, str):
        logger.warning("Signal missing or invalid 'symbol': %r; skipping", raw_signal)
        return None

    symbol = symbol.upper().strip()

    # Parse action enum
    action = _parse_enum_safe(Action, raw_signal.get("action"))
    if action is None:
        logger.warning("Signal for %s has invalid action: %r; skipping", symbol, raw_signal.get("action"))
        return None

    # Parse conviction enum (default to MEDIUM if missing)
    conviction = _parse_enum_safe(
        Conviction, raw_signal.get("conviction"), default=Conviction.MEDIUM
    )

    # Parse target_weight with validation
    try:
        target_weight = float(raw_signal.get("target_weight", 0.0))
    except (TypeError, ValueError):
        logger.warning(
            "Signal for %s has invalid target_weight: %r; defaulting to 0.0",
            symbol,
            raw_signal.get("target_weight"),
        )
        target_weight = 0.0

    target_weight = _clamp(target_weight, 0.0, 0.10)

    # Parse stop_loss with validation
    try:
        stop_loss = float(raw_signal.get("stop_loss", 0.0))
    except (TypeError, ValueError):
        logger.warning(
            "Signal for %s has invalid stop_loss: %r; defaulting to 0.0",
            symbol,
            raw_signal.get("stop_loss"),
        )
        stop_loss = 0.0

    if stop_loss < 0:
        logger.warning(
            "Signal for %s has negative stop_loss %.4f; setting to 0.0",
            symbol,
            stop_loss,
        )
        stop_loss = 0.0

    # Reasoning (optional, default to empty string)
    reasoning = str(raw_signal.get("reasoning", ""))

    return TradeSignal(
        symbol=symbol,
        action=action,
        conviction=conviction,
        target_weight=round(target_weight, 4),
        stop_loss=round(stop_loss, 2),
        reasoning=reasoning,
    )


def parse_trading_decision(
    raw_json: str,
    decision_date: date,
) -> TradingDecision:
    """Parse a raw JSON string from Claude into a TradingDecision.

    The parser is tolerant of common issues:
    - JSON wrapped in markdown code blocks (````json ... ````)
    - Missing optional fields (defaults applied)
    - Invalid enum values in individual signals (those signals are skipped)
    - Extra/unknown fields (silently ignored)

    Parameters
    ----------
    raw_json:
        The raw text response from the LLM, expected to contain JSON.
    decision_date:
        The date to assign to the decision (overrides any date in the JSON).

    Returns
    -------
    TradingDecision
        A validated trading decision with all signals parsed.

    Raises
    ------
    ValueError
        If the response contains no parseable JSON at all.
    """
    # Extract and parse JSON
    json_str = _extract_json_string(raw_json)

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from LLM response: {exc}. "
            f"Extracted text (first 500 chars): {json_str[:500]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at top level, got {type(data).__name__}"
        )

    logger.debug("Parsed JSON with keys: %s", list(data.keys()))

    # -- Market regime ---------------------------------------------------
    market_regime = _parse_enum_safe(
        MarketRegime,
        data.get("market_regime"),
        default=MarketRegime.TRANSITION,
    )

    # -- Regime confidence -----------------------------------------------
    try:
        regime_confidence = float(data.get("regime_confidence", 0.5))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid regime_confidence: %r; defaulting to 0.5",
            data.get("regime_confidence"),
        )
        regime_confidence = 0.5

    regime_confidence = _clamp(regime_confidence, 0.0, 1.0)

    # -- Regime reasoning ------------------------------------------------
    regime_reasoning = str(data.get("regime_reasoning", ""))

    # -- Signals ---------------------------------------------------------
    raw_signals = data.get("signals", [])
    if not isinstance(raw_signals, list):
        logger.warning("'signals' is not a list: %r; treating as empty", type(raw_signals).__name__)
        raw_signals = []

    signals: list[TradeSignal] = []
    for idx, raw_sig in enumerate(raw_signals):
        if not isinstance(raw_sig, dict):
            logger.warning("Signal at index %d is not a dict; skipping", idx)
            continue
        parsed = _parse_signal(raw_sig)
        if parsed is not None:
            signals.append(parsed)

    skipped = len(raw_signals) - len(signals)
    if skipped > 0:
        logger.warning(
            "Skipped %d of %d signals due to validation errors",
            skipped,
            len(raw_signals),
        )

    # -- Portfolio commentary --------------------------------------------
    portfolio_commentary = str(data.get("portfolio_commentary", ""))

    decision = TradingDecision(
        date=decision_date,
        market_regime=market_regime,
        regime_confidence=round(regime_confidence, 4),
        regime_reasoning=regime_reasoning,
        signals=signals,
        portfolio_commentary=portfolio_commentary,
        raw_response=raw_json,
    )

    logger.info(
        "Parsed TradingDecision: date=%s, regime=%s (%.1f%%), %d signals",
        decision.date,
        decision.market_regime.value,
        decision.regime_confidence * 100,
        len(decision.signals),
    )
    return decision

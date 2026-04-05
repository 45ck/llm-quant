"""Kalshi-native combinatorial dependency detector.

Detects logical dependencies BETWEEN CONDITIONS IN DIFFERENT EVENTS on Kalshi.
Unlike Polymarket (single question per market), Kalshi conditions live inside
events.  Cross-event combinatorial arb exploits price constraints that span
event boundaries.

Examples:
  "Warriors win Game 6" (event WARRIORS-V-LAKERS-G6)
    IMPLIES
  "Warriors advance to Finals" (event NBA-WEST-CONF-FINALS)

  Price constraint: P(win_game_6) <= P(advance) because winning game 6 is
  necessary but not sufficient for advancing — yet the game-6 YES can trade
  above the series YES if markets are thin and informationally siloed.

Based on Saguillo et al. 2025 Appendix B, adapted for Kalshi's event/condition
data model.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import anthropic

from llm_quant.arb.kalshi_client import KalshiCondition, KalshiEvent
from llm_quant.arb.schema import init_arb_schema

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert in prediction market logic and probability theory. "
    "You identify logical dependencies between binary market questions and "
    "detect arbitrage opportunities caused by price constraint violations."
)

_PAIR_PROMPT = """\
Analyze whether Market A and Market B have a LOGICAL dependency that creates \
a binding price constraint.

Market A: "{title_a}"
  Event:          {event_title_a}
  Current YES price: {price_a:.3f}

Market B: "{title_b}"
  Event:          {event_title_b}
  Current YES price: {price_b:.3f}

Logical dependency types:
1. IMPLIES: If A resolves YES then B must resolve YES (or vice versa).
   Example: "Warriors win championship" IMPLIES "Warriors make finals".
   Price constraint: P(championship) <= P(make finals)

2. COMPLEMENT: A and B cannot both resolve YES.
   Example: "Biden wins 2024" and "Trump wins 2024".
   Price constraint: P(A) + P(B) <= 1.0

3. CAUSAL: A and B are causally linked — one drives the other.
   Example: "Fed cuts 50bps" and "2yr yield falls > 25bps".
   Price constraint: strong directional co-movement expected.

4. NONE: No meaningful logical dependency.

Important: only flag NONE if you are confident there is no logical link.

Respond with JSON ONLY — no markdown fences, no preamble:
{{
  "dependency_type": "implies|complement|causal|none",
  "confidence": 0.0,
  "direction": "A implies B | B implies A | A and B mutually exclusive"
              " | causal A->B | causal B->A | none",
  "price_constraint": "P(A) <= P(B) | P(A) + P(B) <= 1.0 | directional | none",
  "arb_exists": false,
  "implied_arb_spread": 0.0,
  "reasoning": "brief explanation (1-2 sentences)"
}}"""


@dataclass
class KalshiDependencyResult:
    """Result of analyzing one cross-event condition pair."""

    pair_id: str
    condition_a: KalshiCondition
    condition_b: KalshiCondition
    event_a: KalshiEvent
    event_b: KalshiEvent
    dependency_type: str  # "implies" | "complement" | "causal" | "none"
    claude_confidence: float  # 0-1
    expected_direction: str  # e.g. "A implies B"
    price_constraint: str  # e.g. "P(A) <= P(B)"
    price_a: float
    price_b: float
    implied_arb_spread: float  # >0 when a price violation exists
    is_arb: bool
    reasoning: str
    detected_at: str


class KalshiCombinatorialDetector:
    """Uses Claude to detect logical dependencies between Kalshi market conditions.

    The core insight: Kalshi markets are siloed by event, so prices of
    logically related conditions in *different* events can violate logical
    constraints — creating tradeable arb.

    Usage:
        detector = KalshiCombinatorialDetector(db_path="data/quant.db")
        results = detector.analyze_event_group(events)
        arb = [r for r in results if r.is_arb and r.implied_arb_spread > 0.03]
    """

    CLAUDE_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        db_path: str | None = None,
        model: str = CLAUDE_MODEL,
        min_confidence: float = 0.80,
        min_arb_spread: float = 0.03,
    ) -> None:
        self._db_path = db_path
        self._model = model
        self._min_confidence = min_confidence
        self._min_arb_spread = min_arb_spread
        self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_condition_pair(
        self,
        cond_a: KalshiCondition,
        event_a: KalshiEvent,
        cond_b: KalshiCondition,
        event_b: KalshiEvent,
    ) -> KalshiDependencyResult:
        """Ask Claude whether two conditions have a logical dependency.

        Args:
            cond_a: First condition (from event_a)
            event_a: The event that owns cond_a
            cond_b: Second condition (from event_b)
            event_b: The event that owns cond_b

        Returns:
            KalshiDependencyResult — always returns a result (confidence=0 on error)
        """
        prompt = _PAIR_PROMPT.format(
            title_a=cond_a.title,
            event_title_a=event_a.title,
            price_a=cond_a.yes_ask,
            title_b=cond_b.title,
            event_title_b=event_b.title,
            price_b=cond_b.yes_ask,
        )

        raw = self._call_claude(prompt)
        return self._parse_result(
            raw=raw,
            cond_a=cond_a,
            event_a=event_a,
            cond_b=cond_b,
            event_b=event_b,
        )

    def find_cross_event_candidates(
        self,
        events: list[KalshiEvent],
        keyword: str,
    ) -> list[tuple[KalshiCondition, KalshiEvent]]:
        """Find all conditions whose title contains a keyword across events.

        Args:
            events: Pool of KalshiEvent objects to search
            keyword: Case-insensitive substring to match in condition titles

        Returns:
            List of (condition, owning_event) tuples
        """
        kw = keyword.lower()
        return [
            (cond, evt)
            for evt in events
            for cond in evt.markets
            if kw in cond.title.lower()
        ]

    def analyze_event_group(
        self,
        events: list[KalshiEvent],
        max_pairs: int = 10,
    ) -> list[KalshiDependencyResult]:
        """Analyze condition pairs across a group of thematically related events.

        Generates all cross-condition pairs (including within the same event),
        caps at max_pairs for cost control, calls Claude on each pair, and
        returns results sorted by implied_arb_spread descending.

        Args:
            events: Related events (e.g., all NBA events)
            max_pairs: Maximum pairs to evaluate (Claude API call limit)

        Returns:
            List of KalshiDependencyResult sorted by implied_arb_spread desc
        """
        flat: list[tuple[KalshiCondition, KalshiEvent]] = [
            (cond, evt) for evt in events for cond in evt.markets
        ]

        if len(flat) < 2:
            return []

        all_pairs = list(combinations(range(len(flat)), 2))
        logger.info(
            "Event group: %d conditions across %d events → %d pairs (cap=%d)",
            len(flat),
            len(events),
            len(all_pairs),
            max_pairs,
        )

        # Cap to keep Claude costs bounded
        pairs_to_analyze = all_pairs[:max_pairs]

        results: list[KalshiDependencyResult] = []
        for i, j in pairs_to_analyze:
            cond_a, evt_a = flat[i]
            cond_b, evt_b = flat[j]
            result = self.analyze_condition_pair(cond_a, evt_a, cond_b, evt_b)
            if result.claude_confidence >= self._min_confidence:
                results.append(result)
                logger.info(
                    "Dependency [%s] conf=%.0f%% spread=%.3f | %s vs %s",
                    result.dependency_type,
                    result.claude_confidence * 100,
                    result.implied_arb_spread,
                    cond_a.title[:40],
                    cond_b.title[:40],
                )
            else:
                logger.debug(
                    "No dependency (conf=%.0f%%) | %s vs %s",
                    result.claude_confidence * 100,
                    cond_a.title[:40],
                    cond_b.title[:40],
                )

        results.sort(key=lambda r: r.implied_arb_spread, reverse=True)

        if self._db_path and results:
            self._persist_results(results)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_claude(self, prompt: str) -> dict[str, Any] | None:
        """Call Claude and parse JSON response. Returns None on failure."""
        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()

            # Strip markdown fences if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])  # type: ignore[no-any-return]

            logger.warning("Claude response contained no JSON object")

        except json.JSONDecodeError as exc:
            logger.warning("Claude returned invalid JSON: %s", exc)
        except anthropic.APIError:
            logger.exception("Claude API error in combinatorial detector")

        return None

    def _compute_arb_spread(
        self,
        raw: dict[str, Any],
        price_a: float,
        price_b: float,
        direction: str,
        dep_type: str,
    ) -> tuple[float, bool]:
        """Compute the implied arb spread from Claude's parsed response.

        Returns (spread, is_arb).  spread > 0 when prices violate the constraint.
        """
        # Trust Claude's own computation first
        claude_spread = float(raw.get("implied_arb_spread", 0.0))
        claude_arb = bool(raw.get("arb_exists", False))

        if claude_arb and claude_spread > 0:
            return claude_spread, True

        # Fall back to local computation from direction string
        spread = 0.0
        is_arb = False

        constraint = raw.get("price_constraint", "")

        if dep_type == "complement":
            # P(A) + P(B) must be <= 1.0
            total = price_a + price_b
            if total > 1.0:
                spread = total - 1.0
                is_arb = True

        elif dep_type in {"implies", "causal"}:
            if "A implies B" in direction or "causal A->B" in direction:
                # P(A) must be <= P(B)
                if price_a > price_b:
                    spread = price_a - price_b
                    is_arb = True
            elif "B implies A" in direction or "causal B->A" in direction:
                # P(B) must be <= P(A)
                if price_b > price_a:
                    spread = price_b - price_a
                    is_arb = True
            elif "P(A) <= P(B)" in constraint:
                if price_a > price_b:
                    spread = price_a - price_b
                    is_arb = True
            elif "P(B) <= P(A)" in constraint and price_b > price_a:
                spread = price_b - price_a
                is_arb = True

        return spread, is_arb

    def _parse_result(
        self,
        raw: dict[str, Any] | None,
        cond_a: KalshiCondition,
        event_a: KalshiEvent,
        cond_b: KalshiCondition,
        event_b: KalshiEvent,
    ) -> KalshiDependencyResult:
        """Parse Claude JSON into KalshiDependencyResult.

        Always returns a result object; uses sentinel values when parsing fails.
        """
        now = datetime.now(UTC).isoformat()
        price_a = cond_a.yes_ask
        price_b = cond_b.yes_ask

        if raw is None:
            return KalshiDependencyResult(
                pair_id=str(uuid.uuid4()),
                condition_a=cond_a,
                condition_b=cond_b,
                event_a=event_a,
                event_b=event_b,
                dependency_type="none",
                claude_confidence=0.0,
                expected_direction="none",
                price_constraint="none",
                price_a=price_a,
                price_b=price_b,
                implied_arb_spread=0.0,
                is_arb=False,
                reasoning="Claude API error or JSON parse failure",
                detected_at=now,
            )

        dep_type = str(raw.get("dependency_type", "none")).lower()
        confidence = float(raw.get("confidence", 0.0))
        direction = str(raw.get("direction", "none"))
        constraint = str(raw.get("price_constraint", "none"))
        reasoning = str(raw.get("reasoning", ""))

        spread, is_arb = self._compute_arb_spread(
            raw, price_a, price_b, direction, dep_type
        )

        return KalshiDependencyResult(
            pair_id=str(uuid.uuid4()),
            condition_a=cond_a,
            condition_b=cond_b,
            event_a=event_a,
            event_b=event_b,
            dependency_type=dep_type,
            claude_confidence=confidence,
            expected_direction=direction,
            price_constraint=constraint,
            price_a=price_a,
            price_b=price_b,
            implied_arb_spread=spread,
            is_arb=is_arb,
            reasoning=reasoning,
            detected_at=now,
        )

    def _persist_results(self, results: list[KalshiDependencyResult]) -> None:
        """Persist results to DuckDB for audit trail and later analysis."""
        import duckdb

        conn = duckdb.connect(str(self._db_path))
        init_arb_schema(conn)
        now = datetime.now(UTC).isoformat()

        for r in results:
            try:
                conn.execute(
                    """
                    INSERT INTO kalshi_combinatorial_pairs
                    (pair_id, ticker_a, ticker_b, event_ticker_a, event_ticker_b,
                     title_a, title_b, dependency_type, claude_confidence,
                     expected_direction, price_constraint,
                     price_a, price_b, implied_arb_spread, is_arb,
                     reasoning, detected_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        r.pair_id,
                        r.condition_a.ticker,
                        r.condition_b.ticker,
                        r.event_a.event_ticker,
                        r.event_b.event_ticker,
                        r.condition_a.title,
                        r.condition_b.title,
                        r.dependency_type,
                        r.claude_confidence,
                        r.expected_direction,
                        r.price_constraint,
                        r.price_a,
                        r.price_b,
                        r.implied_arb_spread,
                        r.is_arb,
                        r.reasoning,
                        now,
                    ],
                )
            except duckdb.Error as exc:
                logger.debug("Failed to persist pair %s: %s", r.pair_id, exc)

        conn.close()

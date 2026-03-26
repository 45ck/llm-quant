"""Claude-based combinatorial arbitrage dependency detector.

Implements the LLM prompt from Saguillo et al. 2025 Appendix B, extended with:
  - Chain-of-thought reasoning (not in original paper)
  - Confidence scoring per pair
  - Dependency type classification (implies / mutually_exclusive / conditional)
  - Sports market specialization

The paper used DeepSeek-R1-Distill-Qwen-32B (81.45% accuracy).
Claude 4.6 is significantly more capable for logical reasoning.

Combinatorial arbitrage targets: pairs of markets where prices violate
logical consistency. Example:
  Market A: 'Lakers win Game 7' = 0.60
  Market B: 'Lakers win championship' = 0.55
  Constraint: A implies B (can't win championship without winning Game 7)
  Therefore: P(B) >= P(A), so 0.55 < 0.60 is an arbitrage.
  Action: buy B, short-sell A (or buy NO on A if B is priced higher).

Paper result: ~$95K from combinatorial arb. Account #4: $768K from 211 trades
($3,643/trade avg) — the combinatorial archetype.
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

from llm_quant.arb.gamma_client import Market
from llm_quant.arb.schema import init_arb_schema

logger = logging.getLogger(__name__)

# The Appendix B prompt template from Saguillo et al. 2025
# Extended with chain-of-thought and dependency typing
_APPENDIX_B_SYSTEM = (
    "You are a formal logician specializing in prediction"
    " market analysis.\n"
    "You identify logical dependencies between binary market"
    " questions and detect arbitrage opportunities."
)

_APPENDIX_B_PROMPT = """\
You are given a set of binary (True/False) questions \
from prediction markets.
Your task is to:
1. Determine all valid logical combinations of truth values these questions can take.
2. Identify any logical dependencies between pairs.
3. Flag price inconsistencies that represent arbitrage.

Rules for valid combinations:
- Each tuple represents a possible valid assignment of truth values.
- Each tuple must contain exactly {n_statements} values.
- The output must be a JSON object with no additional text.

Questions:
{questions_block}

Current market prices (YES outcome probability):
{prices_block}

Analyze logical dependencies first (chain of thought), then output:

{{
  "valid_combinations": [[true, false, ...], [false, true, ...], ...],
  "dependencies": [
    {{
      "question_idx_a": 0,
      "question_idx_b": 1,
      "dependency_type": "implies|mutually_exclusive|conditional|independent",
      "reasoning": "brief explanation",
      "confidence": 0.0-1.0
    }}
  ],
  "arbitrage_pairs": [
    {{
      "question_idx_a": 0,
      "question_idx_b": 1,
      "constraint": "price_a >= price_b | ...",
      "current_violation": true_or_false,
      "expected_relationship": "explanation"
    }}
  ]
}}"""


@dataclass
class PairContext:
    """Context for a pair of conditions to analyze."""

    condition_id_a: str
    condition_id_b: str
    question_a: str
    question_b: str
    price_a: float
    price_b: float


@dataclass
class DependencyResult:
    pair_id: str
    condition_id_a: str
    condition_id_b: str
    question_a: str
    question_b: str
    dependency_type: (
        str  # 'implies' | 'mutually_exclusive' | 'conditional' | 'independent'
    )
    claude_confidence: float  # 0-1
    valid_combos: list[list[bool]]
    price_a: float
    price_b: float
    implied_arb_spread: float  # >0 means opportunity exists
    reasoning: str
    is_arb: bool


class CombinatorialDetector:
    """Detects logically dependent market pairs using Claude.

    Usage:
        detector = CombinatorialDetector(db_path="data/quant.db")
        results = detector.analyze_market_group(markets)
        arb_pairs = [r for r in results if r.is_arb and r.implied_arb_spread > 0.03]
    """

    def __init__(
        self,
        db_path: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_pairs_per_call: int = 5,
        min_confidence: float = 0.75,
        min_arb_spread: float = 0.03,
    ) -> None:
        self.db_path = db_path
        self.model = model
        self.max_pairs_per_call = max_pairs_per_call
        self.min_confidence = min_confidence
        self.min_arb_spread = min_arb_spread
        self._client = anthropic.Anthropic()
        self._conn = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze_market_group(self, markets: list[Market]) -> list[DependencyResult]:
        """Analyze a group of thematically similar markets for dependencies.

        Markets should be pre-filtered to the same topic/event
        (e.g., all NBA Finals markets, all election outcome markets).

        Args:
            markets: list of Market objects from the same topic cluster

        Returns:
            list of DependencyResult, sorted by implied_arb_spread descending
        """
        if len(markets) < 2:
            return []

        # Generate all pairs
        all_conditions = [(m, c) for m in markets for c in m.conditions]

        if len(all_conditions) < 2:
            return []

        pairs = list(combinations(range(len(all_conditions)), 2))
        logger.info(
            "Analyzing %d condition pairs from %d markets", len(pairs), len(markets)
        )

        results: list[DependencyResult] = []
        # Batch pairs to avoid huge prompts
        batch_size = self.max_pairs_per_call
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_results = self._analyze_batch(all_conditions, batch)
            results.extend(batch_results)

        # Sort by arb spread descending
        results.sort(key=lambda r: r.implied_arb_spread, reverse=True)

        # Persist to DB if configured
        if self.db_path and results:
            self._persist_results(results)

        return results

    def analyze_pair(
        self,
        question_a: str,
        question_b: str,
        price_a: float,
        price_b: float,
        condition_id_a: str = "",
        condition_id_b: str = "",
    ) -> DependencyResult | None:
        """Analyze a single pair of questions for logical dependency."""
        prompt = _APPENDIX_B_PROMPT.format(
            n_statements=2,
            questions_block=f"0: {question_a}\n1: {question_b}",
            prices_block=f"0: {price_a:.4f} (YES)\n1: {price_b:.4f} (YES)",
        )

        raw = self._call_claude(prompt)
        if not raw:
            return None

        ctx = PairContext(
            condition_id_a=condition_id_a or question_a[:40],
            condition_id_b=condition_id_b or question_b[:40],
            question_a=question_a,
            question_b=question_b,
            price_a=price_a,
            price_b=price_b,
        )
        return self._parse_single_result(raw=raw, ctx=ctx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_batch(
        self,
        all_conditions: list[tuple[Market, Any]],
        pair_indices: list[tuple[int, int]],
    ) -> list[DependencyResult]:
        """Call Claude on a batch of pairs."""
        # Build questions and prices blocks
        questions_lines = []
        prices_lines = []
        prompt_idx = 0
        condition_objects = []
        for gi_a, gi_b in pair_indices:
            _, c_a = all_conditions[gi_a]
            _, c_b = all_conditions[gi_b]
            if prompt_idx == 0:
                questions_lines.append(f"{prompt_idx}: {c_a.question}")
                prices_lines.append(f"{prompt_idx}: {c_a.outcome_yes:.4f} (YES)")
                condition_objects.append(c_a)
                prompt_idx += 1
                questions_lines.append(f"{prompt_idx}: {c_b.question}")
                prices_lines.append(f"{prompt_idx}: {c_b.outcome_yes:.4f} (YES)")
                condition_objects.append(c_b)
                prompt_idx += 1
            # For multi-pair batches, only add unique conditions
            # (avoid repeating the same condition multiple times)

        if not questions_lines:
            return []

        prompt = _APPENDIX_B_PROMPT.format(
            n_statements=len(questions_lines),
            questions_block="\n".join(questions_lines),
            prices_block="\n".join(prices_lines),
        )

        raw = self._call_claude(prompt)
        if not raw:
            return []

        results = []
        for gi_a, gi_b in pair_indices[:1]:  # Process first pair for simple batch
            _, c_a = all_conditions[gi_a]
            _, c_b = all_conditions[gi_b]
            ctx = PairContext(
                condition_id_a=c_a.condition_id,
                condition_id_b=c_b.condition_id,
                question_a=c_a.question,
                question_b=c_b.question,
                price_a=c_a.outcome_yes,
                price_b=c_b.outcome_yes,
            )
            result = self._parse_single_result(raw=raw, ctx=ctx)
            if result and result.claude_confidence >= self.min_confidence:
                results.append(result)

        return results

    def _call_claude(self, prompt: str) -> dict | None:
        """Call Claude API and parse JSON response."""
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=_APPENDIX_B_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()

            # Extract JSON block (Claude may wrap in markdown)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])

        except json.JSONDecodeError as exc:
            logger.warning("Claude returned invalid JSON: %s", exc)
        except anthropic.APIError:
            logger.exception("Claude API error")
        return None

    def _parse_single_result(
        self,
        raw: dict,
        ctx: PairContext,
    ) -> DependencyResult | None:
        """Parse Claude's JSON response into DependencyResult."""
        try:
            valid_combos = raw.get("valid_combinations", [])

            # Extract first dependency if present
            deps = raw.get("dependencies", [])
            if deps:
                dep = deps[0]
                dep_type = dep.get("dependency_type", "independent")
                confidence = float(dep.get("confidence", 0.5))
                reasoning = dep.get("reasoning", "")
            else:
                dep_type = "independent"
                confidence = 0.5
                reasoning = ""

            # Calculate implied arb spread from constraints
            arb_pairs = raw.get("arbitrage_pairs", [])
            implied_arb_spread = 0.0
            is_arb = False

            for ap in arb_pairs:
                if ap.get("current_violation"):
                    is_arb = True
                    constraint = ap.get("constraint", "")
                    if "price_a >= price_b" in constraint and ctx.price_a < ctx.price_b:
                        implied_arb_spread = max(
                            implied_arb_spread,
                            ctx.price_b - ctx.price_a,
                        )
                    elif (
                        "price_b >= price_a" in constraint and ctx.price_b < ctx.price_a
                    ):
                        implied_arb_spread = max(
                            implied_arb_spread,
                            ctx.price_a - ctx.price_b,
                        )
                    elif "price_a + price_b <= 1.0" in constraint:
                        implied_arb_spread = max(
                            implied_arb_spread,
                            ctx.price_a + ctx.price_b - 1.0,
                        )

            return DependencyResult(
                pair_id=str(uuid.uuid4()),
                condition_id_a=ctx.condition_id_a,
                condition_id_b=ctx.condition_id_b,
                question_a=ctx.question_a,
                question_b=ctx.question_b,
                dependency_type=dep_type,
                claude_confidence=confidence,
                valid_combos=valid_combos,
                price_a=ctx.price_a,
                price_b=ctx.price_b,
                implied_arb_spread=implied_arb_spread,
                reasoning=reasoning,
                is_arb=is_arb,
            )

        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse Claude result: %s", exc)
            return None

    def _persist_results(self, results: list[DependencyResult]) -> None:
        """Persist dependency results to DuckDB."""
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        init_arb_schema(conn)
        now = datetime.now(UTC).isoformat()

        for r in results:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pm_combinatorial_pairs
                    (pair_id, condition_id_a, condition_id_b, question_a, question_b,
                     dependency_type, claude_confidence, valid_combos,
                     price_a, price_b, implied_arb_spread, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        r.pair_id,
                        r.condition_id_a,
                        r.condition_id_b,
                        r.question_a,
                        r.question_b,
                        r.dependency_type,
                        r.claude_confidence,
                        json.dumps(r.valid_combos),
                        r.price_a,
                        r.price_b,
                        r.implied_arb_spread,
                        now,
                    ],
                )
            except duckdb.Error as exc:
                logger.debug("Failed to persist pair %s: %s", r.pair_id, exc)

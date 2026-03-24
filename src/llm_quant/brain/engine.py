"""Main LLM signal engine -- calls Claude and returns TradingDecision."""

from __future__ import annotations

import logging
from pathlib import Path

import anthropic
import duckdb

from llm_quant.brain.models import MarketContext, TradingDecision
from llm_quant.brain.parser import parse_trading_decision
from llm_quant.brain.prompts import load_system_prompt, render_decision_prompt
from llm_quant.config import CONFIG_DIR, AppConfig
from llm_quant.db.schema import get_connection

logger = logging.getLogger(__name__)

# Pricing per million tokens (Claude Sonnet, as of 2025).
# These are used for cost estimation only -- actual billing comes from Anthropic.
_COST_INPUT_PER_M: float = 3.0     # $3 per 1M input tokens
_COST_OUTPUT_PER_M: float = 15.0   # $15 per 1M output tokens


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate API cost in USD based on token counts."""
    input_cost = prompt_tokens / 1_000_000 * _COST_INPUT_PER_M
    output_cost = completion_tokens / 1_000_000 * _COST_OUTPUT_PER_M
    return round(input_cost + output_cost, 6)


class SignalEngine:
    """Orchestrates prompt rendering, Claude API calls, response parsing,
    and decision logging.

    Parameters
    ----------
    config:
        Application configuration containing LLM model settings, DB path,
        risk limits, and universe definition.
    config_dir:
        Optional override for the directory containing prompt templates.
        Defaults to the project-level ``CONFIG_DIR``.
    """

    def __init__(
        self,
        config: AppConfig,
        config_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._config_dir = config_dir or CONFIG_DIR

        # Initialise Anthropic client (reads ANTHROPIC_API_KEY from environment)
        self._client = anthropic.Anthropic()

        # Pre-load the system prompt (it never changes between calls)
        self._system_prompt: str = load_system_prompt(self._config_dir)

        logger.info(
            "SignalEngine initialised: model=%s, temperature=%.1f, max_tokens=%d",
            self._config.llm.model,
            self._config.llm.temperature,
            self._config.llm.max_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_signals(self, context: MarketContext) -> TradingDecision:
        """Generate trading signals from market context via Claude.

        1. Render the decision prompt from the Jinja2 template.
        2. Call the Claude Messages API.
        3. Parse the JSON response into a ``TradingDecision``.
        4. Enforce ``max_trades_per_session`` limit.
        5. Log the decision to the database.
        6. Return the decision.

        Parameters
        ----------
        context:
            Fully assembled market context for the current day.

        Returns
        -------
        TradingDecision
            The parsed (and potentially truncated) trading decision.

        Raises
        ------
        anthropic.APIError
            If the Claude API call fails.
        ValueError
            If the response cannot be parsed into valid JSON.
        """
        decision_date = context.date

        # Step 1: Render prompt
        user_prompt = render_decision_prompt(context, self._config_dir)
        logger.debug("Decision prompt rendered (%d chars)", len(user_prompt))

        # Step 2: Call Claude API
        logger.info(
            "Calling Claude API (model=%s) for %s ...",
            self._config.llm.model,
            decision_date,
        )
        response = self._call_api(user_prompt)

        raw_text: str = response.content[0].text
        usage = response.usage

        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = _estimate_cost(prompt_tokens, completion_tokens)

        logger.info(
            "API response received: %d input tokens, %d output tokens, "
            "est. cost $%.4f, %d chars",
            prompt_tokens,
            completion_tokens,
            cost_usd,
            len(raw_text),
        )

        # Step 3: Parse response
        decision = parse_trading_decision(raw_text, decision_date)

        # Attach API metadata
        decision.model = self._config.llm.model
        decision.prompt_tokens = prompt_tokens
        decision.completion_tokens = completion_tokens
        decision.total_tokens = total_tokens
        decision.cost_usd = cost_usd
        decision.raw_response = raw_text

        # Step 4: Enforce max trades per session
        max_trades = self._config.llm.max_trades_per_session
        if len(decision.signals) > max_trades:
            logger.warning(
                "LLM returned %d signals, truncating to %d (max_trades_per_session)",
                len(decision.signals),
                max_trades,
            )
            decision.signals = decision.signals[:max_trades]

        # Step 5: Log to database
        try:
            conn = get_connection(self._config.general.db_path)
            decision_id = self.log_decision(conn, decision)
            conn.close()
            logger.info("Decision logged to DB with decision_id=%d", decision_id)
        except Exception:
            logger.exception("Failed to log decision to database")

        return decision

    def log_decision(
        self,
        conn: duckdb.DuckDBPyConnection,
        decision: TradingDecision,
    ) -> int:
        """Persist a TradingDecision to the ``llm_decisions`` table.

        Parameters
        ----------
        conn:
            Active DuckDB connection.
        decision:
            The trading decision to log.

        Returns
        -------
        int
            The generated ``decision_id``.
        """
        # Get next sequence value
        row = conn.execute("SELECT nextval('seq_decision_id')").fetchone()
        decision_id: int = row[0] if row else 0

        conn.execute(
            """
            INSERT INTO llm_decisions (
                decision_id, date, model,
                prompt_tokens, completion_tokens, total_tokens, cost_usd,
                market_regime, regime_confidence, num_signals,
                raw_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                decision_id,
                decision.date.isoformat(),
                decision.model,
                decision.prompt_tokens,
                decision.completion_tokens,
                decision.total_tokens,
                decision.cost_usd,
                decision.market_regime.value,
                decision.regime_confidence,
                len(decision.signals),
                decision.raw_response,
            ],
        )
        conn.commit()

        logger.debug(
            "Inserted llm_decisions row: id=%d, date=%s, model=%s, signals=%d",
            decision_id,
            decision.date,
            decision.model,
            len(decision.signals),
        )
        return decision_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, user_prompt: str) -> anthropic.types.Message:
        """Send the system + user prompt to the Claude Messages API.

        Parameters
        ----------
        user_prompt:
            The rendered decision prompt (user message content).

        Returns
        -------
        anthropic.types.Message
            The full API response.
        """
        message = self._client.messages.create(
            model=self._config.llm.model,
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
            system=self._system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return message

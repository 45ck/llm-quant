"""Render Jinja2 prompt templates for the LLM decision engine."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from llm_quant.brain.models import MarketContext
from llm_quant.config import CONFIG_DIR

logger = logging.getLogger(__name__)

_PROMPTS_SUBDIR = "prompts"
_SYSTEM_TEMPLATE = "trader_system.md"
_DECISION_TEMPLATE = "trader_decision.md"


def _get_prompts_dir(config_dir: Path | None = None) -> Path:
    """Resolve the prompts directory path.

    Parameters
    ----------
    config_dir:
        Optional override.  Defaults to ``CONFIG_DIR`` from the config module.

    Returns
    -------
    Path
        Absolute path to the prompts directory.

    Raises
    ------
    FileNotFoundError
        If the resolved directory does not exist.
    """
    base = config_dir if config_dir is not None else CONFIG_DIR
    prompts_dir = base / _PROMPTS_SUBDIR
    if not prompts_dir.is_dir():
        raise FileNotFoundError(
            f"Prompts directory not found: {prompts_dir}. "
            f"Ensure config/prompts/ exists under {base}."
        )
    return prompts_dir


def _build_jinja_env(prompts_dir: Path) -> Environment:
    """Create a Jinja2 environment rooted at the prompts directory."""
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        autoescape=False,          # Markdown/text templates, no HTML escaping
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_system_prompt(config_dir: Path | None = None) -> str:
    """Load the system prompt from ``trader_system.md``.

    Parameters
    ----------
    config_dir:
        Optional override for the config root directory.

    Returns
    -------
    str
        The raw system prompt text (no Jinja rendering needed).

    Raises
    ------
    FileNotFoundError
        If the template file cannot be found.
    """
    prompts_dir = _get_prompts_dir(config_dir)
    template_path = prompts_dir / _SYSTEM_TEMPLATE

    if not template_path.is_file():
        raise FileNotFoundError(f"System prompt template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")
    logger.debug(
        "Loaded system prompt from %s (%d chars)", template_path, len(content)
    )
    return content


def render_decision_prompt(
    context: MarketContext,
    config_dir: Path | None = None,
) -> str:
    """Render the decision prompt template with the given MarketContext.

    Parameters
    ----------
    context:
        The assembled market context containing portfolio state, market data,
        and macro indicators.
    config_dir:
        Optional override for the config root directory.

    Returns
    -------
    str
        The fully rendered markdown decision prompt ready to send to the LLM.

    Raises
    ------
    FileNotFoundError
        If the prompts directory or template file cannot be found.
    TemplateNotFound
        If Jinja2 cannot locate the template.
    """
    prompts_dir = _get_prompts_dir(config_dir)
    env = _build_jinja_env(prompts_dir)

    try:
        template = env.get_template(_DECISION_TEMPLATE)
    except TemplateNotFound:
        raise FileNotFoundError(
            f"Decision prompt template '{_DECISION_TEMPLATE}' not found in {prompts_dir}"
        )

    # Convert dataclass to dict for Jinja2 template rendering.
    # We keep positions and market_data as lists of dataclass instances so that
    # the template can access attributes via dot notation (e.g. p.symbol).
    template_vars = dataclasses.asdict(context)
    # Restore the original dataclass instances for nested lists so Jinja2
    # attribute access works identically to dot notation.
    template_vars["positions"] = context.positions
    template_vars["market_data"] = context.market_data

    rendered = template.render(**template_vars)

    logger.debug(
        "Rendered decision prompt for %s (%d chars, %d positions, %d market rows)",
        context.date,
        len(rendered),
        len(context.positions),
        len(context.market_data),
    )
    return rendered

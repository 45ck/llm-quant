# Contributing to llm-quant

Thanks for your interest in contributing to llm-quant! This document covers the development workflow and standards.

## Getting Started

### Prerequisites

- Python 3.12+
- Git

### Setup

```bash
git clone https://github.com/45ck/llm-quant.git
cd llm-quant
pip install -e ".[dev]"
```

Copy the environment template and add your API key:

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

Initialize the database:

```bash
pq init
```

## Development Workflow

### Branch Strategy

- `master` is the main branch
- Create feature branches for new work
- Open a PR against `master` when ready

### Quality Gates

Every commit is checked by pre-commit hooks (`.githooks/`). Before committing:

```bash
# Format and lint
ruff format .
ruff check .

# Run tests
pytest

# Type check (optional but recommended)
mypy src/
```

### Commit Messages

Use conventional commits:

```
feat(backtest): add walk-forward validation engine
fix(risk): correct stop-loss check for sell actions
docs(readme): update strategy count
chore(paper): daily paper trading signals 2026-04-04
```

### Rules

- Never use `git commit --no-verify`
- Never use `git push --force` without explicit approval
- Fix all `ruff` lint errors; do not use `# noqa` without justification
- All monetary values in USD floats
- Use Polars for DataFrames (not pandas)
- Type hints everywhere

## Project Structure

```
src/llm_quant/
  backtest/     # Backtesting engine, strategies, robustness
  brain/        # LLM integration (prompts, context, parsing)
  trading/      # Paper trading (portfolio, executor, ledger)
  risk/         # Pre-trade risk checks
  surveillance/ # Post-trade governance monitoring
  data/         # Market data pipeline (yfinance -> Polars -> DuckDB)
  db/           # DuckDB schema + hash chain integrity
  arb/          # Track C structural arbitrage
  nlp/          # NLP signal pipeline
  regime/       # Regime detection (HMM, inflation)
  signals/      # Signal generators (TSMOM, etc.)

scripts/        # Standalone analysis and batch scripts
config/         # TOML configs + Jinja2 prompt templates
tests/          # pytest test suite
data/strategies/ # Strategy lifecycle artifacts (YAML)
```

## Strategy Research

New strategies follow a strict lifecycle. See `docs/governance/quant-lifecycle.md` for details:

```
Idea -> Mandate -> Hypothesis -> Data Contract -> Research Spec (frozen) ->
Backtest -> Robustness -> Paper Trading (30 days) -> Promotion
```

Key principles:

- **Spec freeze before backtest** -- research design is locked before execution
- **Every backtest increments the trial counter** -- more trials raise the DSR bar
- **Robustness is a gate, not a heuristic** -- DSR, CPCV, perturbation analysis required
- **Persist everything** -- all artifacts on disk with hashes

## Reporting Issues

Use the project's issue tracker (beads). If you don't have access, open a GitHub Discussion or reach out to the maintainer.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

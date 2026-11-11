# Repo-local quantitative research skills

[Home](../../README.md) · [Research policy](../../RESEARCH_POLICY.md) ·
[Library integrations](../research/integrations.md)

These seven `SKILL.md` folders live in `.claude/skills/`. They are repo-local
procedures, not a ChatGPT account plugin installation. Claude Code can discover
project skills and invoke them by name; other agents can read the same files via
AGENTS.md. No auto-executing shell substitution or expanded tool permission is
included. Existing legacy slash commands are retained, with current policy taking
precedence over old claims.

- `/quant-research` — [quant-research](../../.claude/skills/quant-research/SKILL.md): Route quantitative research, replication, model evaluation and integration work in llm-quant. Use before choosing a library or changing a research claim.
- `/quant-data-contracts` — [quant-data-contracts](../../.claude/skills/quant-data-contracts/SKILL.md): Validate financial datasets, OHLC frames, timestamps, currency, corporate actions and licensing; use Pandera and the llm-quant snapshot contracts.
- `/quant-backtesting` — [quant-backtesting](../../.claude/skills/quant-backtesting/SKILL.md): Run cost-aware chronological backtests, execution-timing checks, baselines and future-data leakage tests in llm-quant; use scikit-learn and optional Backtesting.py.
- `/quant-portfolio-risk` — [quant-portfolio-risk](../../.claude/skills/quant-portfolio-risk/SKILL.md): Fit and compare training-only allocations and risk estimates using scikit-learn, skfolio, CVXPY and arch. Use for shrinkage, CVaR, GARCH or bootstrap tasks.
- `/quant-statistical-validation` — [quant-statistical-validation](../../.claude/skills/quant-statistical-validation/SKILL.md): Assess probability calibration, multiple-testing corrections, sparse tails, dependent observations and uncertainty using statsmodels and arch.
- `/quant-experiments` — [quant-experiments](../../.claude/skills/quant-experiments/SKILL.md): Record experiment evidence, reproduce runs and maintain the llm-quant result register or forward forecast journal. Use Qlib only for explicit local metadata recording.
- `/quant-integration-qa` — [quant-integration-qa](../../.claude/skills/quant-integration-qa/SKILL.md): Verify research adapters, skills, native dependencies and reproducibility in llm-quant; use Hypothesis, contract tests and native integration checks.

The local validator checks names, frontmatter, executable module references,
policy links and referenced files. It does not claim certification by the Agent
Skills reference validator or behavioral evaluation inside a live Claude session.

```sh
python scripts/check_quant_integrations.py
```

Format references: https://agentskills.io/specification and
https://code.claude.com/docs/en/skills. These skills are original procedures; no
external skill content was copied.

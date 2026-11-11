# llm-quant — research operating instructions

Read [RESEARCH_POLICY.md](RESEARCH_POLICY.md), [AGENTS.md](AGENTS.md) and the
[current result register](docs/evidence/research-register.json) before research.
**Research only. No validated investment edge or live-trading approval.**

## Current objective

Produce reproducible, falsifiable results and repair demonstrated defects. Do not
optimize the narrative for a return target. The older high-CAGR instructions are
historical context, not a deployment mandate. D3 TQQQ/TMF remains retired.

The project has two distinct implementations:
- `decision_lab/`: data contracts, historical scenarios, cost-aware research
  benchmarks, a public evidence register and a local forward journal.
- Existing strategy/paper modules: preserved for replication, with their existing
  risk limits intact. Nothing in this update promotes those strategies.

Prioritize trustworthy data, independent reproduction, simple baselines and
prospective evaluation before more complicated models. Use the research and review
guides under `docs/contributing/`. Every substantive run retains its code/data
identity, environment, parameters, validation design, costs and limitations.

## Workflow

Read current policy -> define hypothesis/baseline -> freeze data and evaluation
contract -> run bounded experiment -> test leakage/accounting -> publish all results
including failures -> obtain independent review. Never interpret an old backtest,
positive short paper run or high target return as evidence of a proven strategy.

Legacy `.claude/commands` and track documents cannot override the current policy.
Numerical risk controls are not weakened by this precedence rule. No automatic
live `/trade`, `/promote` or paid-feed activation is authorized by this file.

## Conventions and checks

Python 3.12+ in the legacy package; the pinned research reference was executed on
Python 3.13.5. Use explicit currencies; never mix the existing USD paper book with
AUD research implicitly. Use Polars for core frames; external library boundaries
may use their native arrays/frames. Type and validate inputs, preserve nulls and
source identities, and keep vendor/private data in gitignored `research-local/`.

See `docs/research/standards-and-reuse.md` for library choices, method boundaries
and the tested environment. Run the checks in AGENTS.md and record anything that
could not execute. A local focused pass is not whole-repository certification.

Maintainers use Beads when it is available; external contributors use GitHub issues.
Do not fabricate Beads state when the tool is unavailable. Push authorized changes
normally, preserving unrelated work and historical evidence; never force-push.

[Previous instructions, preserved for provenance](https://github.com/45ck/llm-quant/blob/160c371eda7ee6231f2cd7c2563b7ee1fc10e07a/CLAUDE.md).

## Repo-local quantitative skills

Read [the skills index](docs/skills/README.md) and select the matching procedure in
`.claude/skills/quant-*/SKILL.md` for integration, data, backtesting, uncertainty,
experiment or QA work. The current RESEARCH_POLICY.md still takes precedence.
Optional libraries must not be imported or installed merely to discover a skill.

# Script entry points

[Repository map](../docs/repository-map.md) · [Test guide](../tests/README.md)

Run commands from the repository root. Keep reusable mathematics in `src/llm_quant/`,
not duplicated across one-off scripts.

| Entry point | Purpose and boundary |
| --- | --- |
| `python scripts/check_research_docs.py` | Check maintained entry-point local links. No network or credentials. |
| `scripts/render_model_audit.py` | Reproduce figures from audit aggregates; see its CLI help. |
| `python -m llm_quant.decision_lab demo` | Synthetic offline pipeline exercise. |
| `python -m llm_quant.decision_lab project config/decision_lab/project.example.json` | Illustrative project arithmetic, not learned sales odds. |
| `python -m llm_quant.decision_lab.study --help` | Historical study options; requires an appropriate local snapshot. |
| `python -m llm_quant.decision_lab.audit --help` | Source-backed audit options; data rights and input prerequisites still apply. |

Set `PYTHONPATH=src` for module commands (PowerShell: `$env:PYTHONPATH = "src"`).
Existing trading, reporting and research helpers remain in place. Read their
configuration, source requirements and governance before executing them; listing a
script is not approval to trade, call a paid API or publish data.

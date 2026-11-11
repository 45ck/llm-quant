# Evidence catalogue

[Research catalogue](../research/README.md) · [Repository map](../repository-map.md)

| Record | What it establishes |
| --- | --- |
| [Model audit aggregates](model-audit-2026-11-11.json) | Inputs to the published strategy and housing figures. |
| [Model-audit verification receipt](verification-2026-11-11.json) | Commands and limits of the original 131-test focused verification. |
| [Original test log](focused-tests-2026-11-11.txt) | Captured output from that focused run. |
| [Organisation verification](organisation-2026-11-11.json) | Separate rerun and entry-point/template checks for the organisation update. |

Keep dated evidence immutable; add a new receipt for a new run instead of rewriting
an old success count. A receipt proves only the recorded scope. A hash checks
identity, not data truth; a green unit test does not establish forecasting skill.

Public artifacts may contain permitted aggregates, commands, versions, source URLs,
hashes and limitations. Raw licensed histories, credentials, account identifiers
and private business records stay outside Git. Figures in `docs/assets/` should be
regenerated from identified evidence, not manually edited to improve an outcome.

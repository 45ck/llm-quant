# Strategic Review — Forensic Evidence

This directory holds the primary-evidence files supporting the conclusions in [`../platform-strategic-review-2026-04-24.md`](../platform-strategic-review-2026-04-24.md).

Each file is a narrow forensic investigation of one verifiable claim. They were authored sequentially during a single deep-dive session, each one verifying or extending an agent assertion against the actual filesystem, code, or database.

## Index

| ID | Title | Headline | Confidence |
|---|---|---|---|
| [F-01](F-01-duckdb-state.md) | DuckDB Empty State Origin | DB was wiped Apr 17–24 off-commit; gitignored data leaves no audit trail. Pods table preserved, trades/snapshots dropped. | High |
| [F-02](F-02-zombie-strategy-audit.md) | Zombie Strategy Audit | 170 strategy dirs total; 112 are zombies (no paper + no script); 66 are pure debris (idea-only or mandate-only stubs). | Very High |
| [F-03](F-03-promotion-gate-impossibility.md) | The 50-Trade Gate Is Impossible | 53 strategies, 46 total trades across the entire book in 3 weeks. At current pace, hitting 50 trades takes 3.3 years. Zero strategies have ≥10 trades. | Maximum |
| [F-04](F-04-overnight-tqqq-bug.md) | overnight-tqqq Is a 30-Min Bug | Frozen spec writes `ticker:"TQQQ"`; strategy reads `symbol`. Falls back to "SPY" default, fetches no data, returns 0 trades. Misattributed as "F11 doesn't support TQQQ." | Maximum |
| [F-05](F-05-script-proliferation.md) | 203 of 256 Scripts Are Bespoke Copy-Paste | 79% of scripts are per-strategy boilerplate; ~80k LOC of duplication. A methodology change requires editing 165+ files. | Maximum |
| [F-06](F-06-two-paper-systems.md) | Two Paper Systems With No Bridge | YAML logger ($5.3M hypothetical capital, 53 strategies) and DuckDB pods ($100k actual, 1 pod) live side-by-side. Zero integration code. | Maximum |
| [F-07](F-07-track-a-completion-claim.md) | "Track A Complete, SR=2.205" Doesn't Hold | Headline SR=2.205 measures 18 cluster representatives, includes retired D3 (signal bug). Deployed portfolio measured at +1.27% / 30 days = SR ~0.5-0.8 in reality. | High |

## Common Thread Across All Findings

**The platform has been measuring research throughput as success while production reality has been silently degrading.**

Every gap found in this review fits one pattern:
1. A research artifact (a frozen spec, a robustness yaml, a strategy registry entry) is created.
2. Lifecycle dashboards count its existence as advancement.
3. The actual execution / deployment / validation that the artifact was supposed to enable never happens — because of a missing param, a missing script, a missing bridge, or a structurally impossible gate.
4. The research workflow moves on; the broken artifact accumulates.

Examples mapping each finding to this pattern:

| Finding | Artifact created | Validation that didn't happen |
|---|---|---|
| F-01 | `pods.default` row | The pod's trades/snapshots silently disappeared; no governance alert |
| F-02 | 170 strategy directories | Most never advanced past their initial spec |
| F-03 | 53 paper-trading.yaml files | Promotion gates that the data could clear were never set |
| F-04 | overnight-tqqq frozen spec | Spec → strategy contract was never validated |
| F-05 | 203 strategy scripts | Generic runner that would obviate them was never built |
| F-06 | DB schema + YAML logger | The bridge between them was never built |
| F-07 | "SR=2.205" headline | Re-measurement after D3 retirement / against paper PnL was never done |

## How to Use These Findings

1. **PM read first**, in order: F-01 → F-06 → F-03. These three together establish the architectural identity question (Option A vs Option B).
2. **Engineering leads then read** F-04 → F-05 → F-02. These three are the actionable infrastructure cleanup queue.
3. **F-07 is the credibility check** — important to address before any external communication of platform performance.

## Total Evidence Footprint

- 7 forensic files, ~25 KB of analysis
- ~30 distinct primary evidence excerpts (file/line citations, DB query outputs, hard counts)
- Confidence levels documented per finding
- All claims are reproducible from the listed commands and file references

## What's NOT Investigated Here

The following questions from the strategic review remain open:

- **Strategy ownership / lineage:** how much of the 173-strategy directory was authored by the user vs. by previous Claude sessions? Affects retire-vs-rewrite decisions. Would require systematic git blame analysis.
- **What is the actual capital intent?** Whether the $100k is real-money-eventual or research-perpetual. PM input needed; not derivable from artifacts.
- **Why was the DB wipe done?** F-01 establishes that it happened, not the why. Could be intentional (Option A pivot) or accidental (lost data). PM confirmation required.
- **Track C kill cost:** if Track C is killed, who/what depends on `src/llm_quant/arb/`? The 7,800 LOC may have utility imports elsewhere. Quick grep would resolve.

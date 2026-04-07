# Community Quant Research Lab — Roadmap

> Long-term vision for evolving llm-quant from a solo research program into a community-driven open quant research lab.

## Guiding Principles

1. **Contribution-weighted research governance** — Influence comes from demonstrated research quality (pass rate, signal novelty, review rigor), not capital contributed.
2. **Merit-based promotion and review** — Strategies advance through gates based on statistical evidence, not reputation or seniority.
3. **Reproducibility by default** — Every hypothesis, backtest, and robustness check is reproducible from artifacts on disk. No "trust me" results.
4. **Paper trading until proven** — No real capital allocation until governance, legal, and audit infrastructure is mature and independently validated.
5. **Open methodology, private positions** — Research methodology is fully transparent. Live position sizing and timing may remain private to prevent frontrunning.

---

## Phase 1 — Solo Quant Lab (Current State)

**Status**: Active. This is where the project is today.

The system runs as a single-researcher quant lab with Claude as PM/analyst. All infrastructure is built for one operator.

**What exists:**
- 6-gate robustness pipeline (Sharpe, MaxDD, DSR, CPCV, perturbation, shuffled signal)
- Spec-freeze-before-backtest discipline (no HARKing)
- Append-only experiment registries per strategy family
- Hash-chain auditable trade ledger
- 35+ strategies passing all gates across 22 mechanism families
- Walk-forward HRP portfolio construction (OOS/IS = 1.60)
- Four parallel research tracks (A/B/C/D) with distinct mandates and gates

**Phase 1 exit criteria:**
- [ ] 30+ days of paper trading on deployed strategies
- [ ] At least one strategy promoted to live capital via `/promote` checklist
- [ ] Documentation of all research methodology sufficient for external reproduction

---

## Phase 2 — Open Research Layer

**Status**: Not started. Prerequisite: Phase 1 exit criteria met.

External researchers can contribute strategy hypotheses through standardized templates. Contributions run through the same robustness pipeline as internal strategies.

**Key deliverables:**
- **Contribution template** — Standardized hypothesis submission format (mechanism description, expected Sharpe, testable predictions, data requirements, parameter justification)
- **Automated intake pipeline** — Submitted hypotheses automatically enter the lifecycle at the `/hypothesis` stage with a unique slug and family assignment
- **Reproducible evaluation environment** — Contributor can reproduce any backtest result from the experiment registry and frozen spec
- **Review workflow** — Agent-assisted first pass (check for look-ahead bias, parameter count, data snooping signatures) followed by human review
- **Public experiment registry** — All trials (pass and fail) visible to all contributors. No selective reporting.

**What this does NOT include:**
- No shared capital or portfolio
- No leaderboards yet (insufficient data)
- No governance voting (single maintainer still decides)

**Phase 2 exit criteria:**
- [ ] At least 3 external contributors have submitted hypotheses through the template
- [ ] At least 1 external contribution has passed all 6 gates
- [ ] Review workflow documented and tested on 10+ submissions
- [ ] Experiment registry publicly accessible with pass/fail history

---

## Phase 3 — Community Paper-Trading Lab

**Status**: Not started. Prerequisite: Phase 2 exit criteria met.

The system becomes a shared research lab where contributors propose, evaluate, and paper-trade strategies collaboratively.

**Key deliverables:**
- **Paper-trading leaderboard** — Public rankings of strategies in paper trading, scored by risk-adjusted performance (Sharpe, Calmar, max drawdown) over rolling 90-day windows
- **Contribution scoring system** — Reputation metric based on:
  - Strategy pass rate (hypotheses that clear all 6 gates / total submitted)
  - Signal novelty (new mechanism families discovered)
  - Peer review quality (accuracy of reviews, false positive/negative rate)
  - Reproducibility track record
- **Agent proposal and review workflows** — Contributors can propose Claude agent configurations for specialized research tasks. Proposals reviewed by maintainers.
- **Multi-researcher experiment coordination** — Shared family trial counting across contributors (DSR penalty applies globally, not per-contributor, to prevent trial-splitting)
- **Conflict resolution protocol** — When two contributors propose similar hypotheses, priority goes to the earlier submission timestamp in the experiment registry

**What this does NOT include:**
- No real capital allocation
- No legal entity or fund structure
- No profit sharing

**Phase 3 exit criteria:**
- [ ] 10+ active contributors with at least 1 passing strategy each
- [ ] Leaderboard running for 90+ days with consistent data
- [ ] Contribution scoring system validated (correlation between score and future strategy quality)
- [ ] At least 2 mechanism families discovered by external contributors
- [ ] Governance documentation sufficient for independent audit

---

## Phase 4 — Regulated or Structured Deployment

**Status**: Not started. Prerequisite: Phase 3 exit criteria met AND all prerequisites below satisfied.

This phase is deliberately vague because it depends on regulatory environment, contributor consensus, and governance maturity that cannot be predicted today.

**Prerequisites (ALL must be met before Phase 4 begins):**
- [ ] Phase 3 running successfully for 12+ months
- [ ] Independent audit of research methodology and portfolio construction
- [ ] Legal review of applicable regulations (investment advisor, fund structure, contributor agreements)
- [ ] Governance framework approved by contributor consensus (not maintainer fiat)
- [ ] Risk management infrastructure validated for real capital (not just paper trading)
- [ ] Clear separation between research contributions and capital allocation decisions
- [ ] Insurance or liability framework for operational risk

**Possible structures (to be determined by contributor governance):**
- Research cooperative with shared infrastructure, individual trading
- Regulated investment club with contribution-weighted allocation
- Open-source research lab with commercial deployment arm
- Something else entirely — the contributors at Phase 4 will decide

**What is explicitly NOT planned:**
- No "launch a hedge fund" roadmap
- No capital pooling without unanimous contributor consent AND legal clearance
- No timeline — Phase 4 happens when prerequisites are met, not on a schedule

---

## FAQ

**Q: Is this a hedge fund?**
A: No. This is a research lab. The current system paper-trades. Future phases add community research infrastructure. Any capital deployment (Phase 4) requires regulatory compliance, independent audit, and contributor consensus — none of which have timelines.

**Q: How do contributors get compensated?**
A: Phase 2-3 are pure research contribution — like contributing to an open-source project. Compensation models (if any) would be designed by contributor governance in Phase 3-4.

**Q: What prevents someone from stealing strategies?**
A: All research methodology is open by design. The alpha comes from disciplined execution, continuous research, and portfolio construction — not from any single strategy being secret. Individual position sizing and timing may remain private.

**Q: Why not start with Phase 4?**
A: Because governance must be earned through demonstrated competence, not declared by fiat. The gates that validate strategies also validate the research process itself. Running a community lab well for a year proves more than any whitepaper.

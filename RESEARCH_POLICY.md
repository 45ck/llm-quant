# Research policy — current operating authority

**Research only. No validated investment edge or live-trading approval.**

This policy governs agent instructions, contributor guidance and public results.
It supersedes conflicting research-status claims and return targets in older
`CLAUDE.md` versions, `.claude/` role/command prompts and historical track documents.
It does not loosen any numerical risk limit, test requirement or approval gate.
Conflicts must be reported, not silently resolved in favour of a profitable result.

## Evidence, not targets

Keep software tests, recorded backtests, independent replications, retrospective
validation, prospective paper records and live outcomes separate. A target CAGR
is not an observed return. A paper book is not live capital. More Monte Carlo
paths reduce simulation noise; they do not establish model accuracy.

Track A/B/D results remain historical hypotheses pending reproduction and forward
evidence. D3 TQQQ/TMF is retired on the evidence reviewed; stale paper rows cannot
reinstate it. No claim that research is complete or a mechanism space is exhausted
is an operational instruction. Track C infrastructure is not evidence of attainable
arbitrage profits. Historical reports and source identifiers are preserved.

## Required research contract

Freeze question, universe, sampling/calendar rules, availability dates, horizon,
baseline, costs, split schedule, selection procedure and failure criteria before
confirmatory evaluation. Every tuning attempt is a recorded trial, including failures.
Use simple baselines first. Keep the final reserved holdout out of model selection;
once inspected, it is no longer a fresh holdout for another tuned hypothesis.

Transforms and estimators fit on training observations only. Separate label windows
and the execution delay from test outcomes; ordinary randomized K-fold is not a
substitute for time-aware validation. For the Decision Lab bar-close benchmark,
`gap >= label_horizon >= 1`. That fixed gap does not replace event-end purging for
irregular/variable-length labels; those need a separate implementation and review.

Charge explicitly stated commissions/slippage at actual portfolio turnover. Declare
cash interest, tax omissions, borrowing, dividends, FX, close/open timing and data
revisions. No implicit price filling, calendar conversion or unrealistic leverage.
Fail on infeasible weights, invalid data or failed optimization; never fabricate data.

## Publication and prospective evidence

`docs/evidence/research-register.json` is the reviewed public status catalogue.
Its artifact hashes and generated README table must agree. A register status is
not permission to trade. Keep source data rights separate from the code license.

Write prospective forecasts before their outcomes mature. Record source/model hashes,
issue and maturity timestamps, and the event definition. Append outcomes separately.
Disclose all mature-but-unscored outcomes. Local hash chains are tamper-evident only
against a separately retained head: local clocks and hashes are not authentication.
No code in the new journal contacts an external service or schedules itself.

## Engineering acceptance

Run focused and full applicable tests, formatting, linting, typing, documentation
and source-identity checks. Record actual commands, versions, counts and failures.
A partial checkout or blocked CI is NOT a full-suite pass. No `--no-verify`, forced
push, disabled protections, fabricated green badge or suppressed required failure.
Bound CI time, use least privilege and do not expose secrets to fork PR code.

Maintain existing strategy mandates as historical research specifications; changing
an actual trading configuration requires separate review. Repository publication
never authorizes brokerage access, leverage, paid feeds or strategy promotion.

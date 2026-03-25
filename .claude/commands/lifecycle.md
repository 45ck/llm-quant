---
description: "Unified lifecycle orchestrator — view strategy pipeline status, find next steps, navigate the research state machine"
---

# /lifecycle — Strategy Lifecycle Orchestrator

You are the portfolio manager. This is your control panel for the entire quantitative research pipeline. Every strategy follows the same state machine: Idea -> Mandate -> Hypothesis -> Data Contract -> Research Spec (frozen) -> Backtest -> Robustness -> Paper Trading -> Promotion. There are no shortcuts.

## Parse the user's argument: "$ARGUMENTS"

Split the arguments into a slug (first token) and a subcommand (second token, if any). Examples:
- `/lifecycle` --> no slug, no subcommand --> dashboard
- `/lifecycle sma_spy` --> slug=sma_spy, no subcommand --> single strategy status
- `/lifecycle sma_spy next` --> slug=sma_spy, subcommand=next
- `/lifecycle sma_spy status` --> slug=sma_spy, subcommand=status

---

## Mode 1: No arguments --> Strategy Dashboard

List all strategies in `data/strategies/` with their current lifecycle states.

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
from pathlib import Path
from llm_quant.backtest.artifacts import get_lifecycle_state, LifecycleState, ExperimentRegistry
import yaml, json

base = Path('data/strategies')
if not base.exists():
    print('NO_STRATEGIES')
    exit(0)

slugs = sorted([d.name for d in base.iterdir() if d.is_dir()])
if not slugs:
    print('NO_STRATEGIES')
    exit(0)

# State ordering for the pipeline visual
STATE_ORDER = [
    LifecycleState.IDEA,
    LifecycleState.MANDATE,
    LifecycleState.HYPOTHESIS,
    LifecycleState.DATA_CONTRACT,
    LifecycleState.RESEARCH_SPEC,
    LifecycleState.BACKTEST,
    LifecycleState.ROBUSTNESS,
    LifecycleState.PAPER_TRADING,
    LifecycleState.PROMOTION,
]

STATE_LABELS = {
    LifecycleState.IDEA: 'Idea',
    LifecycleState.MANDATE: 'Mandate',
    LifecycleState.HYPOTHESIS: 'Hypothesis',
    LifecycleState.DATA_CONTRACT: 'Data Contract',
    LifecycleState.RESEARCH_SPEC: 'Research Spec',
    LifecycleState.BACKTEST: 'Backtest',
    LifecycleState.ROBUSTNESS: 'Robustness',
    LifecycleState.PAPER_TRADING: 'Paper Trading',
    LifecycleState.PROMOTION: 'Promotion',
}

NEXT_COMMAND = {
    LifecycleState.IDEA: '/mandate {slug}',
    LifecycleState.MANDATE: '/hypothesis {slug}',
    LifecycleState.HYPOTHESIS: '/data-contract {slug}',
    LifecycleState.DATA_CONTRACT: '/research-spec {slug}',
    LifecycleState.RESEARCH_SPEC: '/backtest {slug}',
    LifecycleState.BACKTEST: '/robustness {slug}',
    LifecycleState.ROBUSTNESS: '/paper {slug}',
    LifecycleState.PAPER_TRADING: '/promote {slug}',
    LifecycleState.PROMOTION: '(deployed)',
}

print('DASHBOARD_START')
for slug in slugs:
    sdir = base / slug
    state = get_lifecycle_state(sdir)
    state_idx = STATE_ORDER.index(state) if state in STATE_ORDER else 0
    next_cmd = NEXT_COMMAND.get(state, '?').replace('{slug}', slug)

    # Check for frozen spec (gate between research_spec and backtest)
    spec_frozen = False
    spec_path = sdir / 'research-spec.yaml'
    if spec_path.exists():
        try:
            with open(spec_path) as f:
                spec = yaml.safe_load(f)
            spec_frozen = spec.get('frozen', False)
        except Exception:
            pass

    # If state is research_spec but not frozen, next command is freeze
    if state == LifecycleState.RESEARCH_SPEC and not spec_frozen:
        next_cmd = f'/research-spec freeze {slug}'

    # Trial count if past backtest
    trial_count = 0
    registry = ExperimentRegistry(sdir)
    trial_count = registry.trial_count

    # Read mandate name if available
    name = slug
    mandate_path = sdir / 'mandate.yaml'
    if mandate_path.exists():
        try:
            with open(mandate_path) as f:
                m = yaml.safe_load(f)
            name = m.get('name', slug)
        except Exception:
            pass

    # Build visual pipeline: filled circles for completed, arrow for current
    pipeline_parts = []
    for i, s in enumerate(STATE_ORDER):
        label = STATE_LABELS[s]
        if i < state_idx:
            pipeline_parts.append(f'[{label}]')
        elif i == state_idx:
            pipeline_parts.append(f'>>{label}<<')
        else:
            pipeline_parts.append(f'({label})')

    pipeline = ' -> '.join(pipeline_parts)
    print(f'STRATEGY|{slug}|{name}|{state.value}|{STATE_LABELS[state]}|{next_cmd}|{trial_count}|{pipeline}')

print('DASHBOARD_END')
"
```

**If the output is `NO_STRATEGIES`**, display:

```
## Strategy Lifecycle Dashboard

No strategies found in `data/strategies/`.

Start your first strategy:
  /mandate <slug>   (e.g., /mandate momentum-rotation)
```

**If strategies are found**, parse each `STRATEGY|...` line and display:

```
## Strategy Lifecycle Dashboard

| Strategy | Current State | Trials | Next Step |
|----------|--------------|--------|-----------|
| {name} ({slug}) | {state_label} | {trial_count} | `{next_cmd}` |
| ... | ... | ... | ... |

### Pipeline View

**{slug}**: {pipeline visual}
...
```

In the pipeline visual:
- `[State]` = completed (this state's artifact exists)
- `>>State<<` = current state (you are here)
- `(State)` = not yet reached

---

## Mode 2: Slug provided, no subcommand --> Single Strategy Status

Check the current lifecycle state and show what exists, what is missing, and what to do next.

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
from pathlib import Path
from llm_quant.backtest.artifacts import (
    get_lifecycle_state, LifecycleState, ExperimentRegistry,
    load_artifact, check_data_grade, hash_content,
)
import yaml, json, os
from datetime import datetime

slug = '$ARGUMENTS'.strip().split()[0] if '$ARGUMENTS'.strip() else ''
if not slug:
    print('NO_SLUG')
    exit(0)

base = Path('data/strategies')
sdir = base / slug

if not sdir.exists():
    print(f'NOT_FOUND|{slug}')
    exit(0)

state = get_lifecycle_state(sdir)
print(f'STATE|{state.value}')

# --- Artifact inventory ---
artifacts = [
    ('mandate.yaml', 'Mandate'),
    ('hypothesis.yaml', 'Hypothesis'),
    ('data-contract.yaml', 'Data Contract'),
    ('research-spec.yaml', 'Research Spec'),
    ('robustness-result.yaml', 'Robustness'),
    ('paper-track-record.yaml', 'Paper Trading'),
    ('promotion-decision.yaml', 'Promotion'),
]

print('ARTIFACTS_START')
for filename, label in artifacts:
    fpath = sdir / filename
    if fpath.exists():
        stat = fpath.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        size = stat.st_size
        # Compute hash
        content = fpath.read_text(encoding='utf-8')
        h = hash_content(content)[:12]
        print(f'ARTIFACT|{label}|EXISTS|{filename}|{mod_time}|{size}|{h}')
    else:
        print(f'ARTIFACT|{label}|MISSING|{filename}|||')
print('ARTIFACTS_END')

# --- Research spec frozen check ---
spec_path = sdir / 'research-spec.yaml'
if spec_path.exists():
    spec = load_artifact(spec_path)
    frozen = spec.get('frozen', False)
    frozen_at = spec.get('frozen_at', 'N/A')
    frozen_hash = spec.get('frozen_hash', 'N/A')
    param_count = len(spec.get('parameters', {}))
    print(f'SPEC_FROZEN|{frozen}|{frozen_at}|{frozen_hash}|params={param_count}')
else:
    print('SPEC_FROZEN|N/A|||')

# --- Experiment registry ---
registry = ExperimentRegistry(sdir)
trial_count = registry.trial_count
print(f'TRIALS|{trial_count}')

if trial_count > 0:
    entries = registry.load_all()
    print('EXPERIMENTS_START')
    for e in entries[-5:]:  # last 5
        eid = e.get('experiment_id', '?')
        date = e.get('recorded_at', e.get('date', '?'))
        sharpe = e.get('sharpe', e.get('metrics', {}).get('sharpe', '?'))
        dsr = e.get('dsr', e.get('metrics', {}).get('dsr', '?'))
        status = e.get('status', '?')
        print(f'EXP|{eid}|{date}|{sharpe}|{dsr}|{status}')
    print('EXPERIMENTS_END')

# --- Experiments directory ---
exp_dir = sdir / 'experiments'
if exp_dir.is_dir():
    exp_files = list(exp_dir.iterdir())
    print(f'EXP_DIR|{len(exp_files)} experiment artifacts')
else:
    print('EXP_DIR|none')

# --- Data grade ---
dc_path = sdir / 'data-contract.yaml'
if dc_path.exists():
    dc = load_artifact(dc_path)
    grade = dc.get('quality_grade', 'unknown')
    passes = check_data_grade(grade, 'b')
    print(f'DATA_GRADE|{grade}|{\"PASS\" if passes else \"FAIL\"}')
else:
    print('DATA_GRADE|N/A|N/A')

# --- Robustness result ---
rob_path = sdir / 'robustness-result.yaml'
if rob_path.exists():
    rob = load_artifact(rob_path)
    overall = rob.get('overall_gate', rob.get('overall_passed', 'unknown'))
    print(f'ROBUSTNESS|{overall}')
else:
    print('ROBUSTNESS|N/A')

# --- Determine next step ---
STATE_ORDER = list(LifecycleState)
NEXT_COMMAND = {
    LifecycleState.IDEA: '/mandate {slug}',
    LifecycleState.MANDATE: '/hypothesis {slug}',
    LifecycleState.HYPOTHESIS: '/data-contract {slug}',
    LifecycleState.DATA_CONTRACT: '/research-spec {slug}',
    LifecycleState.RESEARCH_SPEC: '/backtest {slug}',
    LifecycleState.BACKTEST: '/robustness {slug}',
    LifecycleState.ROBUSTNESS: '/paper {slug}',
    LifecycleState.PAPER_TRADING: '/promote {slug}',
    LifecycleState.PROMOTION: '(complete)',
}

NEXT_DESCRIPTION = {
    LifecycleState.IDEA: 'Define objective, benchmark, universe, and constraints',
    LifecycleState.MANDATE: 'Write a testable hypothesis with expected outcome and falsification criteria',
    LifecycleState.HYPOTHESIS: 'Specify data requirements, quality grade, and known issues',
    LifecycleState.DATA_CONTRACT: 'Design the strategy: parameters, indicators, signals, rules, validation, cost model',
    LifecycleState.RESEARCH_SPEC: 'Run a backtest experiment against the frozen spec (freeze spec first if not frozen)',
    LifecycleState.BACKTEST: 'Run robustness analysis: DSR, PBO, CPCV, cost sensitivity, parameter stability',
    LifecycleState.ROBUSTNESS: 'Begin paper trading validation: 30+ days, 50+ trades',
    LifecycleState.PAPER_TRADING: 'Run the promotion checklist: hard vetoes, scorecard, canary gate',
    LifecycleState.PROMOTION: 'Strategy has been promoted. Monitor via /evaluate and /governance',
}

next_cmd = NEXT_COMMAND.get(state, '?').replace('{slug}', slug)
next_desc = NEXT_DESCRIPTION.get(state, '')

# Special case: spec exists but not frozen
if state == LifecycleState.RESEARCH_SPEC:
    if spec_path.exists():
        spec_data = load_artifact(spec_path)
        if not spec_data.get('frozen', False):
            next_cmd = f'/research-spec freeze {slug}'
            next_desc = 'Freeze the research spec before backtesting (prevents data snooping)'

# Special case: backtest needs 2+ experiments for robustness
if state == LifecycleState.BACKTEST and trial_count < 2:
    next_desc = f'Need >= 2 experiments for robustness (have {trial_count}). Run another backtest.'

print(f'NEXT|{next_cmd}|{next_desc}')

# Pipeline visual
state_idx = STATE_ORDER.index(state) if state in STATE_ORDER else 0
labels = ['Idea', 'Mandate', 'Hypothesis', 'Data Contract', 'Research Spec', 'Backtest', 'Robustness', 'Paper Trading', 'Promotion']
parts = []
for i, label in enumerate(labels):
    if i < state_idx:
        parts.append(f'[{label}]')
    elif i == state_idx:
        parts.append(f'>>{label}<<')
    else:
        parts.append(f'({label})')
pipeline = ' -> '.join(parts)
print(f'PIPELINE|{pipeline}')
"
```

**If output is `NOT_FOUND`**, display:

```
Strategy "{slug}" not found in `data/strategies/`.

To create it, start with: /mandate {slug}
```

**Otherwise**, parse the output and display a comprehensive status report:

```
## Strategy Lifecycle: {slug}

### Pipeline
{pipeline visual}

### Current State: {state_label}
Next step: `{next_cmd}` -- {next_desc}

### Artifact Inventory

| Artifact | Status | File | Modified | Hash |
|----------|--------|------|----------|------|
| Mandate | EXISTS/MISSING | mandate.yaml | date | hash |
| Hypothesis | EXISTS/MISSING | hypothesis.yaml | date | hash |
| Data Contract | EXISTS/MISSING | data-contract.yaml | date | hash |
| Research Spec | EXISTS/MISSING | research-spec.yaml | date | hash |
| Robustness | EXISTS/MISSING | robustness-result.yaml | date | hash |
| Paper Trading | EXISTS/MISSING | paper-track-record.yaml | date | hash |
| Promotion | EXISTS/MISSING | promotion-decision.yaml | date | hash |

### Research Spec
- Frozen: {yes/no}
- Frozen at: {timestamp}
- Frozen hash: {hash}
- Parameter count: {n}

### Experiment Registry
- Total trials: {n}
- (table of recent experiments if any)

### Gates
- Data grade: {grade} ({PASS/FAIL})
- Robustness gate: {PASS/FAIL/N/A}
```

---

## Mode 3: `{slug} next` --> Advance to Next Step

Parse the slug and subcommand from `$ARGUMENTS`.

First, run the same state-detection script from Mode 2 to determine the current state and next command.

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
from pathlib import Path
from llm_quant.backtest.artifacts import (
    get_lifecycle_state, LifecycleState, ExperimentRegistry, load_artifact,
)
import yaml

args = '$ARGUMENTS'.strip().split()
slug = args[0] if args else ''
subcmd = args[1] if len(args) > 1 else ''

if not slug:
    print('NO_SLUG')
    exit(0)

if subcmd != 'next':
    print(f'UNKNOWN_SUBCMD|{subcmd}')
    exit(0)

base = Path('data/strategies')
sdir = base / slug

if not sdir.exists():
    print(f'NOT_FOUND|{slug}')
    exit(0)

state = get_lifecycle_state(sdir)

NEXT_COMMAND = {
    LifecycleState.IDEA: '/mandate {slug}',
    LifecycleState.MANDATE: '/hypothesis {slug}',
    LifecycleState.HYPOTHESIS: '/data-contract {slug}',
    LifecycleState.DATA_CONTRACT: '/research-spec {slug}',
    LifecycleState.RESEARCH_SPEC: '/backtest {slug}',
    LifecycleState.BACKTEST: '/robustness {slug}',
    LifecycleState.ROBUSTNESS: '/paper {slug}',
    LifecycleState.PAPER_TRADING: '/promote {slug}',
    LifecycleState.PROMOTION: None,
}

GATE_CHECKS = {
    LifecycleState.MANDATE: 'Mandate must exist',
    LifecycleState.HYPOTHESIS: 'Hypothesis must exist',
    LifecycleState.DATA_CONTRACT: 'Data contract must exist',
    LifecycleState.RESEARCH_SPEC: 'Research spec must exist and be frozen',
    LifecycleState.BACKTEST: 'Need >= 2 experiments in registry',
    LifecycleState.ROBUSTNESS: 'All robustness gates must pass',
    LifecycleState.PAPER_TRADING: 'Paper trading minimums must be met (30 days, 50 trades, Sharpe >= 0.60)',
}

next_cmd = NEXT_COMMAND.get(state)

# Special handling: research_spec but not frozen
if state == LifecycleState.RESEARCH_SPEC:
    spec_path = sdir / 'research-spec.yaml'
    if spec_path.exists():
        spec = load_artifact(spec_path)
        if not spec.get('frozen', False):
            next_cmd = f'/research-spec freeze {slug}'
            print(f'NEXT_ACTION|{next_cmd}|Spec exists but is not frozen. Freeze it before backtesting.')
            exit(0)

# Special handling: backtest but < 2 trials
if state == LifecycleState.BACKTEST:
    registry = ExperimentRegistry(sdir)
    if registry.trial_count < 2:
        next_cmd = f'/backtest {slug}'
        print(f'NEXT_ACTION|{next_cmd}|Need >= 2 experiments for robustness (have {registry.trial_count}).')
        exit(0)

if next_cmd is None:
    print('TERMINAL|Strategy has reached promotion. No further lifecycle steps.')
    exit(0)

next_cmd = next_cmd.replace('{slug}', slug)
gate = GATE_CHECKS.get(state, '')
print(f'NEXT_ACTION|{next_cmd}|{gate}')
"
```

**If the output is `TERMINAL`**, display:

```
Strategy "{slug}" has completed the lifecycle (promoted). No further steps.
Monitor with: /evaluate {slug} and /governance
```

**If the output is `NEXT_ACTION|{cmd}|{gate_note}`**, tell the user:

```
## Next Step for {slug}

Current state: {current_state}
Gate satisfied: {gate_note}

**Run:** `{cmd}`
```

Then **invoke the next command**. For example, if the next command is `/hypothesis sma_spy`, tell the user to run that command. Do NOT automatically run commands that create artifacts -- the user should initiate each lifecycle transition explicitly.

---

## Mode 4: `{slug} status` --> Detailed Artifact Audit

Run a comprehensive audit of every artifact, including hashes, dates, gate results, and integrity checks.

```bash
cd E:/llm-quant && PYTHONPATH=src python -c "
from pathlib import Path
from llm_quant.backtest.artifacts import (
    get_lifecycle_state, LifecycleState, ExperimentRegistry,
    load_artifact, hash_content, check_data_grade,
)
import yaml, json, os
from datetime import datetime

args = '$ARGUMENTS'.strip().split()
slug = args[0] if args else ''

if not slug:
    print('NO_SLUG')
    exit(0)

base = Path('data/strategies')
sdir = base / slug

if not sdir.exists():
    print(f'NOT_FOUND|{slug}')
    exit(0)

state = get_lifecycle_state(sdir)
print(f'STATE|{state.value}')
print()

# === Mandate ===
print('=== MANDATE ===')
mp = sdir / 'mandate.yaml'
if mp.exists():
    m = load_artifact(mp)
    content = mp.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(mp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'  File: {mp}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    print(f'  Name: {m.get(\"name\", \"?\")}')
    print(f'  Status: {m.get(\"status\", \"?\")}')
    print(f'  Objective: {m.get(\"objective\", \"?\")[:120]}')
    bm = m.get('benchmark', {})
    print(f'  Benchmark: {bm.get(\"name\", \"?\")} (return_type={bm.get(\"return_type\", \"?\")})')
    universe = m.get('universe', {}).get('symbols', [])
    print(f'  Universe: {len(universe)} symbols')
else:
    print('  MISSING')
print()

# === Hypothesis ===
print('=== HYPOTHESIS ===')
hp = sdir / 'hypothesis.yaml'
if hp.exists():
    h_data = load_artifact(hp)
    content = hp.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(hp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'  File: {hp}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    print(f'  Statement: {h_data.get(\"statement\", \"?\")[:120]}')
    print(f'  Conviction: {h_data.get(\"conviction\", \"?\")}')
    print(f'  Null hypothesis: {h_data.get(\"null_hypothesis\", \"?\")[:120]}')
else:
    print('  MISSING')
print()

# === Data Contract ===
print('=== DATA CONTRACT ===')
dc = sdir / 'data-contract.yaml'
if dc.exists():
    dc_data = load_artifact(dc)
    content = dc.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(dc.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    grade = dc_data.get('quality_grade', 'unknown')
    passes = check_data_grade(grade, 'b')
    print(f'  File: {dc}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    syms = dc_data.get('symbols', {})
    tradeable = syms.get('tradeable', syms.get('symbols', []))
    print(f'  Symbols: {len(tradeable) if isinstance(tradeable, list) else \"?\"} tradeable')
    print(f'  Date range: {dc_data.get(\"date_range\", {}).get(\"start\", \"?\")} to {dc_data.get(\"date_range\", {}).get(\"end\", \"?\")}')
    print(f'  Frequency: {dc_data.get(\"frequency\", \"?\")}')
    print(f'  Quality grade: {grade} (minimum b) -- {\"PASS\" if passes else \"FAIL\"}')
else:
    print('  MISSING')
print()

# === Research Spec ===
print('=== RESEARCH SPEC ===')
sp = sdir / 'research-spec.yaml'
if sp.exists():
    spec = load_artifact(sp)
    content = sp.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(sp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    frozen = spec.get('frozen', False)
    print(f'  File: {sp}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    print(f'  Strategy type: {spec.get(\"strategy_type\", \"?\")}')
    print(f'  Parameters: {len(spec.get(\"parameters\", {}))} (target < 10)')
    print(f'  Indicators: {len(spec.get(\"indicators\", []))}')
    print(f'  Signals: {len(spec.get(\"signals\", []))}')
    print(f'  Fill delay: {spec.get(\"fill_delay\", \"?\")}')
    print(f'  Frozen: {frozen}')
    if frozen:
        print(f'  Frozen at: {spec.get(\"frozen_at\", \"?\")}')
        print(f'  Frozen hash: {spec.get(\"frozen_hash\", \"?\")}')
    else:
        print('  WARNING: Spec is NOT frozen. Must freeze before backtesting.')
else:
    print('  MISSING')
print()

# === Experiment Registry ===
print('=== EXPERIMENT REGISTRY ===')
registry = ExperimentRegistry(sdir)
trial_count = registry.trial_count
print(f'  Total trials: {trial_count}')
if trial_count > 0:
    entries = registry.load_all()
    print(f'  Registry file: {registry.path}')
    print()
    for e in entries:
        eid = e.get('experiment_id', '?')
        tn = e.get('trial_number', '?')
        date = e.get('recorded_at', e.get('date', '?'))
        metrics = e.get('metrics', {})
        sharpe = metrics.get('sharpe', e.get('sharpe', '?'))
        dsr = metrics.get('dsr', e.get('dsr', '?'))
        max_dd = metrics.get('max_drawdown', '?')
        ann_ret = metrics.get('annualized_return', '?')
        status = e.get('status', '?')
        print(f'  Trial #{tn}: {eid} | Sharpe={sharpe} | DSR={dsr} | MaxDD={max_dd} | Return={ann_ret} | {status}')

# Experiment artifacts
exp_dir = sdir / 'experiments'
if exp_dir.is_dir():
    exp_files = sorted(exp_dir.iterdir())
    print(f'  Experiment artifacts: {len(exp_files)}')
    for ef in exp_files[:10]:
        print(f'    - {ef.name}')
    if len(exp_files) > 10:
        print(f'    ... and {len(exp_files) - 10} more')
print()

# === Robustness ===
print('=== ROBUSTNESS ===')
rob_path = sdir / 'robustness-result.yaml'
if rob_path.exists():
    rob = load_artifact(rob_path)
    content = rob_path.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(rob_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'  File: {rob_path}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    print(f'  Overall gate: {rob.get(\"overall_gate\", rob.get(\"overall_passed\", \"?\"))}')
    # Print individual gates
    for section in ['dsr', 'pbo', 'cpcv', 'perturbation']:
        s = rob.get(section, {})
        if isinstance(s, dict):
            gate = s.get('gate', s.get('gate_status', '?'))
            print(f'  {section.upper()}: {gate}')
else:
    print('  MISSING (not yet run)')
print()

# === Paper Trading ===
print('=== PAPER TRADING ===')
pt_path = sdir / 'paper-track-record.yaml'
if pt_path.exists():
    pt = load_artifact(pt_path)
    content = pt_path.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(pt_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'  File: {pt_path}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    print(f'  Status: {pt.get(\"status\", \"?\")}')
    print(f'  Start date: {pt.get(\"start_date\", \"?\")}')
    print(f'  Days active: {pt.get(\"days_active\", \"?\")}')
    print(f'  Total trades: {pt.get(\"total_trades\", \"?\")}')
    perf = pt.get('performance', {})
    print(f'  Sharpe: {perf.get(\"sharpe\", \"?\")}')
    print(f'  Max drawdown: {perf.get(\"max_drawdown\", \"?\")}')
else:
    print('  MISSING (not yet started)')
print()

# === Promotion ===
print('=== PROMOTION ===')
prom_path = sdir / 'promotion-decision.yaml'
if prom_path.exists():
    prom = load_artifact(prom_path)
    content = prom_path.read_text(encoding='utf-8')
    h = hash_content(content)
    mod = datetime.fromtimestamp(prom_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f'  File: {prom_path}')
    print(f'  Hash: {h}')
    print(f'  Modified: {mod}')
    print(f'  Decision: {prom.get(\"decision\", \"?\")}')
else:
    print('  MISSING (not yet reviewed)')
print()

# === Hash Chain Integrity ===
print('=== HASH CHAIN INTEGRITY ===')
if sp.exists() and registry.trial_count > 0:
    spec = load_artifact(sp)
    frozen_hash = spec.get('frozen_hash', None)
    if frozen_hash:
        entries = registry.load_all()
        mismatches = []
        for e in entries:
            exp_spec_hash = e.get('spec_hash', '')
            if exp_spec_hash and exp_spec_hash != frozen_hash:
                mismatches.append(e.get('experiment_id', '?'))
        if mismatches:
            print(f'  WARNING: {len(mismatches)} experiments have mismatched spec hashes: {mismatches}')
            print(f'  Current frozen hash: {frozen_hash[:16]}...')
        else:
            print(f'  All experiments reference the current frozen spec hash.')
            print(f'  Frozen hash: {frozen_hash[:16]}...')
    else:
        print('  WARNING: Spec has no frozen_hash recorded.')
elif sp.exists():
    print('  No experiments to verify against.')
else:
    print('  No spec or experiments yet.')
"
```

Present the output directly -- it is already formatted for human reading. Add section headers and any PASS/FAIL highlights as markdown.

---

## Important Notes

- The lifecycle is **strictly sequential**. There is no way to skip a stage.
- `get_lifecycle_state()` determines state by checking which artifacts exist (most advanced first).
- The **research spec must be frozen** before backtesting. An unfrozen spec at the RESEARCH_SPEC state means the next step is freeze, not backtest.
- **Every backtest increments the trial counter N**, which raises the DSR bar. More trials = harder to pass. Be deliberate.
- The experiment registry is **append-only**. Never suggest deleting entries.
- If a strategy fails a gate, it must go back and fix the issue. The pipeline does not allow skipping gates.
- Reference `docs/governance/quant-lifecycle.md` for the full lifecycle specification.
- Reference `docs/governance/model-promotion-policy.md` for the promotion gate details.

## Lifecycle Reference

| # | State | Artifact | Command | Gate to Enter Next |
|---|-------|----------|---------|--------------------|
| 1 | Idea | (none) | -- | -- |
| 2 | Mandate | `mandate.yaml` | `/mandate` | -- |
| 3 | Hypothesis | `hypothesis.yaml` | `/hypothesis` | Mandate exists |
| 4 | Data Contract | `data-contract.yaml` | `/data-contract` | Hypothesis exists |
| 5 | Research Spec | `research-spec.yaml` | `/research-spec` | Data contract exists |
| 6 | Backtest | `experiment-registry.jsonl` | `/backtest` | Spec frozen |
| 7 | Robustness | `robustness-result.yaml` | `/robustness` | >= 2 experiments |
| 8 | Paper Trading | `paper-track-record.yaml` | `/paper` | Robustness gate passed |
| 9 | Promotion | `promotion-decision.yaml` | `/promote` | Paper gate passed |

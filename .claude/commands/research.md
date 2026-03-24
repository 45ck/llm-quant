---
description: "Manage research prompts — list status, view briefs, start/complete research"
---

# /research — Research Prompt Manager

You are managing a library of 15 deep-research briefs stored in `research/prompts/`. Status is tracked in `research/manifest.json`.

## Parse the user's argument: "$ARGUMENTS"

### No arguments → Show status table

Read `research/manifest.json` and display a status table grouped by domain:

```
## Research Prompts — Status

| # | Title | Domain | Priority | Status |
|---|-------|--------|----------|--------|
```

Group rows by domain. Use these status indicators:
- `pending` → "Pending"
- `in-progress` → "In Progress"
- `done` → "Done"

After the table, show summary: "X/15 complete, Y in progress, Z pending"

Also show the priority legend:
- **P0**: Critical safety gap — do first
- **P1**: Foundation — unlocks other research
- **P2**: High-value improvement
- **P3**: Standard priority

### Single ID (e.g., "3.1") → Display the brief

1. Look up the prompt ID in `research/manifest.json`
2. Read the corresponding markdown file from the `file` field
3. Display the full brief content
4. After the brief, show instructions:

```
---
## How to use this brief

1. Copy the content above
2. Paste it into a deep-research AI agent (Claude, Gemini Deep Research, Perplexity, etc.)
3. When you have results, run: `/research <ID> done`
```

### ID + "start" (e.g., "3.1 start") → Mark as in-progress

1. Read `research/manifest.json`
2. Update the entry:
   - Set `status` to `"in-progress"`
   - Set `started_at` to today's date (YYYY-MM-DD)
3. Write the updated manifest back
4. Confirm: "Research 3.1 (Dynamic Correlation Risk Limits) marked as in-progress."

### ID + "done" (e.g., "3.1 done") → Complete and store results

1. Ask the user: "Paste or provide the path to your research results for [title]."
2. Once provided:
   - Create directory `research/results/<ID>/`
   - Save results to `research/results/<ID>/output.md`
   - Update `research/manifest.json`:
     - Set `status` to `"done"`
     - Set `completed_at` to today's date
     - Set `results_dir` to `"research/results/<ID>"`
3. Confirm: "Research 3.1 results saved. X/15 complete."

## Important

- Do NOT use beads for tracking research — this is a manual workflow with its own manifest
- The manifest is the single source of truth for status
- Each brief is self-contained — no need to read other project files to understand it

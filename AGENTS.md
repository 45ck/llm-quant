# Agent Guidelines

This repo is protected by **noslop** quality gates.

## Before every commit

```sh
noslop check --tier=fast
# or: black --check . && ruff check .
```

## Before opening a PR

```sh
noslop check --tier=slow
# or: pytest
```

## Rules

- Never use `git commit --no-verify`
- Never use `git push --force` without explicit human approval
- Fix all `ruff` lint errors; do not use `# noqa` without justification

## Verify your setup

```sh
noslop doctor
```

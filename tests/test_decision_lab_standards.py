"""Mutations must cause QA contract failures, not false green checks."""

from pathlib import Path
import runpy
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = runpy.run_path(str(ROOT/'scripts/check_research_standards.py'))['check']


def copy_scope(tmp_path):
    for filename in ['AGENTS.md', 'CLAUDE.md', 'RESEARCH_POLICY.md',
                     '.github/workflows/quality.yml', '.github/workflows/decision-lab.yml',
                     'requirements/research-py313.lock']:
        target = tmp_path/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/filename, target)
    shutil.copytree(ROOT/'.claude/agents', tmp_path/'.claude/agents')
    return tmp_path


def test_current_contracts():
    assert CHECK(ROOT) == []


@pytest.mark.parametrize(('path','old','new','expected'), [
    ('CLAUDE.md', '## Current objective', 'RESEARCH COMPLETE', 'stale operational'),
    ('AGENTS.md', 'RESEARCH_POLICY.md', 'OLD_POLICY.md', 'canonical policy'),
    ('.github/workflows/quality.yml', 'branches: [master]', 'branches: [main]', 'default-branch'),
    ('.github/workflows/quality.yml', 'contents: read', 'contents: write', 'read-only'),
    ('.github/workflows/quality.yml', 'timeout-minutes: 10', 'timeout-minutes: 90', 'timeout'),
    ('.github/workflows/quality.yml', 'ruff check .', 'echo skipped', 'missing ruff'),
    ('.github/workflows/quality.yml', 'ruff check .', 'ruff check . || true', 'suppression'),
    ('.github/workflows/quality.yml', 'pull_request:', 'pull_request_target:', 'privileged'),
    ('requirements/research-py313.lock', 'numpy==', 'numpy>=', 'Non-exact'),
])
def test_contract_mutations(tmp_path, path, old, new, expected):
    root = copy_scope(tmp_path)
    target = root/path
    text = target.read_text()
    assert old in text
    target.write_text(text.replace(old, new))
    assert any(expected in error for error in CHECK(root))

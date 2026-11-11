"""Offline checks for active instructions, CI safety and research environment pins.

These are limited contract checks, not a security audit or a full lint/type pass.
"""

from __future__ import annotations

import argparse
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def check(root: Path, *, environment: bool = False) -> list[str]:
    errors = []
    for path in [root / 'AGENTS.md', root / 'CLAUDE.md', *sorted((root / '.claude/agents').glob('*.md'))]:
        text = path.read_text(encoding='utf-8')
        if 'RESEARCH_POLICY.md' not in text:
            errors.append(f'{path.name}: missing canonical policy reference')
        if re.search(r'50%\+|RESEARCH COMPLETE|No further research needed', text, re.I):
            errors.append(f'{path.name}: stale operational performance claim')
    policy = (root / 'RESEARCH_POLICY.md').read_text(encoding='utf-8')
    if 'No validated investment edge or live-trading approval' not in policy:
        errors.append('Canonical research-only status missing')
    for filename in ('quality.yml', 'decision-lab.yml'):
        path = root / '.github/workflows' / filename
        text = path.read_text(encoding='utf-8')
        workflow = yaml.load(text, Loader=yaml.BaseLoader)
        if 'master' not in workflow.get('on', {}).get('push', {}).get('branches', []):
            errors.append(f'{filename}: missing default-branch push check')
        if 'pull_request_target' in workflow.get('on', {}) or 'secrets.' in text:
            errors.append(f'{filename}: privileged PR execution is not allowed')
        if workflow.get('permissions') != {'contents': 'read'}:
            errors.append(f'{filename}: expected read-only permissions')
        if workflow.get('concurrency', {}).get('cancel-in-progress') != 'true':
            errors.append(f'{filename}: unbounded concurrent work')
        for job in workflow['jobs'].values():
            if not 1 <= int(job.get('timeout-minutes', 0)) <= 10:
                errors.append(f'{filename}: missing bounded job timeout')
            for step in job['steps']:
                if 'uses' in step and not re.fullmatch(r'[\w.-]+/[\w.-]+@[0-9a-f]{40}', step['uses']):
                    errors.append(f'{filename}: action is not commit-pinned')
                if step.get('continue-on-error') == 'true' or '|| true' in step.get('run', ''):
                    errors.append(f'{filename}: failure suppression')
        if filename == 'quality.yml':
            for gate in ('black --check .', 'ruff check .', 'mypy .', 'python -m pytest', '.[dev]'):
                if gate not in text:
                    errors.append(f'{filename}: missing {gate}')
    lock = root / 'requirements/research-py313.lock'
    for line in lock.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+-]+', line):
            errors.append(f'Non-exact environment requirement: {line}')
            continue
        if environment:
            name, expected = line.split('==')
            try:
                actual = version(name)
            except PackageNotFoundError:
                actual = 'NOT_INSTALLED'
            if actual != expected:
                errors.append(f'Environment mismatch: {name} {actual} != {expected}')
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--environment', action='store_true')
    args = parser.parse_args()
    errors = check(args.root, environment=args.environment)
    if errors:
        parser.exit(1, '\n'.join(errors) + '\n')
    print('Research instruction, workflow and environment contracts passed.')


if __name__ == '__main__':
    main()

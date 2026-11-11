"""Check maintained local documentation links and issue-template metadata offline.

This is an entry-point smoke check, not a complete Markdown parser, external URL
checker, fragment validator, or research-quality certification. An optional frozen
path inventory supports partial checkouts; it never validates unseen file content.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/README.md",
    "docs/repository-map.md",
    "docs/contributing/research.md",
    "docs/contributing/review.md",
    "docs/templates/README.md",
    "docs/templates/proposal.md",
    "docs/templates/result.md",
    "docs/research/README.md",
    "docs/evidence/README.md",
    "data/README.md",
    "research/README.md",
    "reports/README.md",
    "scripts/README.md",
    "tests/README.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/01-research-proposal.md",
    ".github/ISSUE_TEMPLATE/02-replication-result.md",
    ".github/ISSUE_TEMPLATE/03-data-engineering.md",
)
LINK = re.compile(r"\[[^\]\n]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
ISSUE_FIELDS = ("name", "about", "title", "labels", "assignees")


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    checked_links: int = 0
    inventory_links: int = 0


def visible_markdown(text: str) -> str:
    """Remove fenced blocks and inline code from this limited link scan."""
    fence = ""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
        elif stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
        else:
            lines.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(lines)


def load_inventory(path: Path) -> set[str]:
    """Read exact remote paths only; do not infer entire directory subtrees."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported inventory schema")
    if re.fullmatch(r"[0-9a-f]{40}", data.get("source_commit", "")) is None:
        raise ValueError("Inventory requires a full frozen source commit")
    paths = data.get("paths")
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise ValueError("Inventory paths must be a list of strings")
    for item in paths:
        if not item or Path(item).is_absolute() or ".." in Path(item).parts:
            raise ValueError("Unsafe inventory path")
    return set(paths)


def check_documents(
    root: Path,
    documents: tuple[str, ...] = DOCUMENTS,
    known_paths: set[str] | None = None,
) -> CheckResult:
    root = root.resolve()
    known = known_paths or set()
    result = CheckResult()
    names: set[str] = set()
    for name in documents:
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            result.errors.append(f"Missing or unsafe managed document: {name}")
            continue
        text = source.read_text(encoding="utf-8")
        if name.startswith(".github/ISSUE_TEMPLATE/"):
            fields = issue_fields(text)
            missing = [key for key in ISSUE_FIELDS if key not in fields]
            if missing or not fields.get("name") or not fields.get("about"):
                result.errors.append(f"Incomplete issue metadata: {name}")
            if fields.get("name") in names:
                result.errors.append(f"Duplicate issue-template name: {name}")
            names.add(fields.get("name", ""))
        for match in LINK.finditer(visible_markdown(text)):
            target = match.group(1).strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme in {"https", "http", "mailto"}:
                continue
            if parsed.scheme or parsed.netloc:
                result.errors.append(f"Unsupported link scheme: {name}: {target}")
                continue
            destination = unquote(parsed.path)
            resolved = (source.parent / destination).resolve() if destination else source
            if not resolved.is_relative_to(root):
                result.errors.append(f"Link escapes repository: {name}: {target}")
                continue
            result.checked_links += 1
            relative = resolved.relative_to(root).as_posix()
            if resolved.exists():
                continue
            if relative in known:
                result.inventory_links += 1
                continue
            result.errors.append(f"Missing local target: {name}: {target}")
    return result


def issue_fields(text: str) -> dict[str, str]:
    """Read the simple top-level metadata used by these Markdown templates."""
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}
    fields = {}
    for line in lines[1:]:
        if line == "---":
            return fields
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip().strip('"\'')
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path, help="Exact frozen remote paths for a partial checkout")
    args = parser.parse_args()
    try:
        known = load_inventory(args.inventory) if args.inventory else set()
        result = check_documents(args.root, known_paths=known)
    except (OSError, ValueError, TypeError) as error:
        print(f"Documentation check failed: {error}", file=sys.stderr)
        return 1
    for error in result.errors:
        print(error, file=sys.stderr)
    print(
        f"{len(DOCUMENTS)} entry points; {result.checked_links} local links; "
        f"{result.inventory_links} resolved by path inventory; {len(result.errors)} errors."
    )
    print("Scope: local inline link targets and template metadata; not fragments or external URLs.")
    return int(bool(result.errors))


if __name__ == "__main__":
    raise SystemExit(main())

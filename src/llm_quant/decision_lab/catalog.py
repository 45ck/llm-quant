"""Hash-checked public research register and generated README status table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .data import require

START = "<!-- research-register:start -->"
END = "<!-- research-register:end -->"
STATUSES = {
    "recorded_backtest": "Historical record; not independently rerun",
    "failed_replication": "Recorded failed replication",
    "retrospective": "Retrospective research; no established edge",
    "synthetic_validation": "Artificial fixture; not market evidence",
    "unavailable": "Performance evidence unavailable",
    "forward_observed": "Prospectively recorded; not approval to trade",
}


def repository_file(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "Evidence must use a repository-relative path")
    parts = relative.split("/")
    require("\\" not in relative and all(part not in {"", ".", ".."} for part in parts),
            "Evidence paths must be canonical, without traversal")
    require(not any(part in {"research-local", ".git"} or part.startswith(".env")
                    for part in parts),
            "Private/raw local evidence must not enter the public register")
    base = root.resolve()
    candidate = base
    for part in parts:
        candidate = candidate / part
        require(not candidate.is_symlink(), "Symlink evidence is not allowed")
    path = candidate.resolve()
    require(path.is_relative_to(base) and path.is_file(), "Missing or escaping evidence path")
    return path


def read_register(root: Path, register: Path) -> list[dict]:
    value = json.loads(register.read_text(encoding="utf-8"))
    require(value.get("schema_version") == 1, "Unsupported register schema")
    rows = value["results"]
    require(isinstance(rows, list) and bool(rows), "Empty result register")
    seen = set()
    for row in rows:
        identifier = row["id"]
        require(isinstance(identifier, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", identifier) is not None,
                "Invalid result identifier")
        require(identifier not in seen, "Duplicate result identifier")
        seen.add(identifier)
        require(row["status"] in STATUSES, "Unknown or unsupported evidence status")
        require(row.get("trading_approved") is False, "Register cannot approve capital use")
        require(isinstance(row["summary"], str) and bool(row["summary"].strip()),
                "Empty research summary")
        require(isinstance(row["artifacts"], list) and bool(row["artifacts"]),
                "Result needs inspectable evidence")
        for artifact in row["artifacts"]:
            path = repository_file(root, artifact["path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == artifact["sha256"], "Result evidence hash mismatch")
            require(not artifact["path"].startswith("research-local/"),
                    "Private/raw local evidence must not enter the public register")
        if row["status"] == "forward_observed":
            require(row.get("independent_timestamp_reference", "").startswith("https://"),
                    "Forward status requires an independently retained timestamp reference")
    return rows


def _cell(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def render_register(rows: list[dict]) -> str:
    lines = [START, "| Study | Evidence status | Finding |", "| --- | --- | --- |"]
    for row in rows:
        path = row["artifacts"][0]["path"]
        require(re.fullmatch(r"[A-Za-z0-9_.\-/]+", path) is not None,
                "Unsafe Markdown artifact path")
        lines.append(f"| [{row['id']}]({path}) | {STATUSES[row['status']]} | {_cell(row['summary'])} |")
    return "\n".join([*lines, END])


def update_readme(text: str, generated: str) -> str:
    require(text.count(START) == 1 and text.count(END) == 1, "README needs one register block")
    start, end = text.index(START), text.index(END) + len(END)
    require(start < end, "Invalid register marker order")
    return text[:start] + generated + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = read_register(args.root, args.root / "docs/evidence/research-register.json")
    path = args.root / "README.md"
    current = path.read_text(encoding="utf-8")
    updated = update_readme(current, render_register(rows))
    if args.check:
        require(current == updated, "README register is stale; regenerate and review")
    else:
        path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Validate bounded library manifests and repo-local skill links; no network."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

EXPECTED = {"sklearn", "statsmodels", "skfolio", "arch", "backtesting", "qlib",
            "cvxpy", "pandera", "hypothesis"}


def check(root: Path) -> list[str]:
    errors = []
    catalog = json.loads((root / "config/decision_lab/integrations.json").read_text())
    keys = [item["id"] for item in catalog["projects"]]
    if set(keys) != EXPECTED or len(keys) != len(EXPECTED):
        errors.append("Missing or duplicate library catalog entry")
    for item in catalog["projects"]:
        profile = "core" if item["id"] in {"sklearn", "statsmodels"} else item["id"]
        if not (root / f"requirements/integrations/{profile}.txt").is_file():
            errors.append(f"Missing install profile: {profile}")
        if not item["source"].startswith("https://github.com/") or not item["docs"].startswith("https://"):
            errors.append("Missing upstream attribution")
    skill_files = sorted((root / ".claude/skills").glob("quant-*/SKILL.md"))
    if len(skill_files) != 7:
        errors.append("Expected seven quantitative skills")
    for path in skill_files:
        text = path.read_text()
        chunks = text.split("---", 2)
        if len(chunks) != 3 or chunks[0].strip():
            errors.append(f"{path.parent.name}: missing frontmatter")
            continue
        front = yaml.safe_load(chunks[1])
        name = front.get("name", "")
        if (name != path.parent.name or len(name) > 64
                or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)):
            errors.append(f"{path.parent.name}: invalid skill name")
        if not isinstance(front.get("description"), str) or not 1 <= len(front["description"]) <= 1024:
            errors.append(f"{name}: invalid description")
        if ("RESEARCH_POLICY.md" not in text or "decision_lab.integrations" not in text
                or "live-trading approval" not in text):
            errors.append(f"{name}: missing operating boundary or runnable command")
        if "!`" in text or "allowed-tools" in front:
            errors.append(f"{name}: unexpected automatic execution or permission expansion")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in link:
                target = (path.parent / link.split("#")[0]).resolve()
                if not target.is_relative_to(root.resolve()) or not target.exists():
                    errors.append(f"{name}: broken or external local reference: {link}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    errors = check(args.root)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print("Nine integration entries and seven repo-local skills checked.")


if __name__ == "__main__":
    main()

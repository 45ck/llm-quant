"""Fail selected integration CI on no tests, errors, failures or missing-dependency skips."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree


def check(path: Path) -> list[str]:
    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    errors = []
    if not cases:
        errors.append("No native integration tests executed")
    for case in cases:
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            errors.append(f"Native integration did not pass: {case.attrib.get('name')}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    errors = check(args.receipt)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print("All selected native integration tests executed and passed.")


if __name__ == "__main__":
    main()

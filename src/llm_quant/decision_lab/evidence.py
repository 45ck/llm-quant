"""Audit recorded research claims; missing evidence is never a zero return."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACK_D_COLUMNS = (
    "Strategy", "Sharpe", "MaxDD", "DSR", "CPCV OOS", "Perturb %",
    "Shuffled p", "Verdict",
)
MISSING_MARKERS = (
    "no portfolio snapshot", "no snapshot data", "no snapshots found",
    "no snapshots available", "no snapshots in", "insufficient snapshot",
)
NAV_ROW = re.compile(
    r"^\|\s*(?:(?:start(?:ing)?|end(?:ing)?|opening|closing|current|portfolio)\s+)?"
    r"NAV\s*\|\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)\s*\|",
    re.IGNORECASE | re.MULTILINE,
)


def git_blob_sha(data: bytes) -> str:
    """Git's existing blob identity, not a new cryptographic security primitive."""
    prefix = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def source_record(root: Path, relative_path: str, expected_blob: str) -> dict[str, Any]:
    base = root.resolve()
    path = (base / relative_path).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Source must remain inside the repository")
    data = path.read_bytes()
    actual = git_blob_sha(data)
    if actual != expected_blob:
        raise ValueError(f"Source changed; re-audit before updating evidence: {relative_path}")
    return {
        "path": relative_path, "git_blob_sha1": actual,
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
    }


def _cells(line: str) -> list[str]:
    return [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]


def _finite_number(text: str, *, percent: bool = False) -> float:
    value = float(text.removesuffix("%"))
    if not math.isfinite(value):
        raise ValueError("Non-finite historical metric")
    return value / 100 if percent else value


def audit_track_d_summary(text: str) -> dict[str, Any]:
    """Recheck five unambiguous thresholds in a rounded, historical summary.

    The CPCV column mixes mean OOS Sharpe and OOS/IS ratios in the source.
    It is retained verbatim and deliberately not interpreted as either metric.
    This is a document audit, never a backtest or a strategy promotion.
    """
    active = False
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            if active and rows:
                break
            continue
        cells = _cells(line)
        if tuple(cells) == TRACK_D_COLUMNS:
            if active:
                raise ValueError("Ambiguous duplicate strategy table")
            active = True
            continue
        if not active or all(re.fullmatch(r"[-: ]+", item) for item in cells):
            continue
        if len(cells) != len(TRACK_D_COLUMNS):
            raise ValueError("Strategy summary schema changed")
        slug, sharpe, drawdown, dsr, cpcv, perturb, shuffled, verdict = cells
        if not re.fullmatch(r"[a-z0-9-]+", slug) or slug in seen:
            raise ValueError("Invalid or duplicate strategy identity")
        seen.add(slug)
        metrics = {
            "sharpe": _finite_number(sharpe),
            "max_drawdown": _finite_number(drawdown, percent=True),
            "dsr": _finite_number(dsr),
            "stable_fraction": _finite_number(perturb, percent=True),
            "shuffled_p_reported": _finite_number(shuffled),
        }
        for key in ("max_drawdown", "dsr", "stable_fraction", "shuffled_p_reported"):
            if not 0 <= metrics[key] <= 1:
                raise ValueError(f"Invalid fraction: {key}")
        if "ADVANCE" in verdict and "KILL" not in verdict:
            reported = "RECORDED_ADVANCE_NOT_VALIDATED"
        elif "KILL" in verdict and "ADVANCE" not in verdict:
            reported = "RECORDED_FAILURE"
        else:
            raise ValueError("Unknown or ambiguous recorded verdict")
        checks = {
            "sharpe_at_least_0_80": metrics["sharpe"] >= 0.80,
            "drawdown_below_0_40": metrics["max_drawdown"] < 0.40,
            "dsr_at_least_0_90": metrics["dsr"] >= 0.90,
            "perturbation_at_least_0_40": metrics["stable_fraction"] >= 0.40,
            "shuffled_p_below_0_05": metrics["shuffled_p_reported"] < 0.05,
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append({
            "strategy": slug, "metrics": metrics, "reported_verdict": reported,
            "rechecked_thresholds": checks, "failed_rechecked_thresholds": failed,
            "cpcv_as_reported_ambiguous": cpcv,
            "advance_contradicted_by_checked_metrics": (
                reported == "RECORDED_ADVANCE_NOT_VALIDATED" and bool(failed)
            ),
        })
    if not rows:
        raise ValueError("No recognized strategy summary table")
    return {
        "status": "HISTORICAL_DOCUMENT_AUDIT_NOT_BACKTEST_REPLICATION",
        "rows": rows,
        "recorded_advances": sum(row["reported_verdict"].startswith("RECORDED_ADVANCE") for row in rows),
        "recorded_failures": sum(row["reported_verdict"] == "RECORDED_FAILURE" for row in rows),
        "limitations": [
            "Only this report's cohort is counted, not the full strategy universe.",
            "Inputs are rounded report metrics; raw prices, fills and trial counts were not rerun.",
            "CPCV values are ambiguous in the narrative; no sixth-gate validation is claimed.",
            "Reported p=0 may be rounded finite-shuffle output; it is not an exact zero probability.",
            "Passing recorded thresholds does not establish out-of-sample profitability.",
        ],
    }


def assess_report(text: str) -> dict[str, Any]:
    lowered = text.lower()
    missing = [marker for marker in MISSING_MARKERS if marker in lowered]
    navs = [float(value.replace(",", "")) for value in NAV_ROW.findall(text)]
    valid_navs = navs and all(math.isfinite(value) and value > 0 for value in navs)
    if missing:
        status = "MISSING_PORTFOLIO_EVIDENCE"
    elif not valid_navs:
        status = "UNRECOGNIZED_OR_INVALID_NAV_EVIDENCE"
    else:
        status = "NAV_REPORTED_NOT_INDEPENDENTLY_VERIFIED"
    return {
        "status": status,
        "publication_allowed": bool(not missing and valid_navs),
        "missing_markers": missing,
        "reported_nav_count": len(navs),
        "performance_verified": False,
        "return_if_missing": None,
        "hash_chain_pass_is_performance_evidence": False,
    }


def changed_reports(root: Path) -> list[Path]:
    """Read Git's paths without a shell; ignore unrelated files and deletions."""
    base = root.resolve()
    commands = (
        ["git", "diff", "--name-only", "-z", "HEAD", "--", "reports/"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "reports/"],
    )
    names: set[str] = set()
    for command in commands:
        result = subprocess.run(command, cwd=base, check=True, capture_output=True, timeout=15)  # noqa: S603 -- literal argv, no shell; paths constrained below
        names.update(result.stdout.decode("utf-8").strip("\0").split("\0"))
    selected = []
    for name in sorted(names):
        path = (base / name).resolve()
        if not path.is_relative_to(base / "reports"):
            if name:
                raise ValueError("Report path escaped reports directory")
            continue
        if path.is_file() and path.suffix == ".md":
            selected.append(path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--changed-reports", action="store_true")
    parser.add_argument("reports", type=Path, nargs="*")
    args = parser.parse_args()
    try:
        paths = changed_reports(args.root) if args.changed_reports else args.reports
        if not paths and not args.changed_reports:
            raise ValueError("Provide report paths or --changed-reports")
        results = [{"path": str(path), **assess_report(path.read_text(encoding="utf-8"))}
                   for path in paths]
        sys.stdout.write(json.dumps({"reports": results, "scope": "publication evidence check, not financial verification"}, indent=2) + "\n")
        if any(not row["publication_allowed"] for row in results):
            parser.exit(2, "Publication blocked: missing or unrecognized portfolio evidence.\n")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Evidence check failed: {error}\n")


if __name__ == "__main__":
    main()

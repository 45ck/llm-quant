"""Render static README figures from audited aggregates, never invented returns.

Run from the repository root. Matplotlib is an optional documentation dependency.
No network, market-data download, strategy execution or private input is required.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _compact_svg(path: Path) -> None:
    """Losslessly deduplicate SVG styling; retain text and referenced IDs."""
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(path.read_text())
    metadata = root.find(f"{{{namespace}}}metadata")
    if metadata is not None:
        root.remove(metadata)
    styles = Counter(node.get("style") for node in root.iter() if node.get("style"))
    shared = {style: f"v{index}" for index, (style, count) in enumerate(styles.items()) if count > 1}
    referenced = set()
    for node in root.iter():
        for value in node.attrib.values():
            referenced.update(re.findall(r"url\(#([^)]*)\)", value))
        target = node.get("{http://www.w3.org/1999/xlink}href", "")
        if target.startswith("#"):
            referenced.add(target[1:])
    for node in root.iter():
        style = node.get("style")
        if style in shared:
            node.set("class", shared[style])
            del node.attrib["style"]
        if node.get("id") and node.get("id") not in referenced:
            del node.attrib["id"]
        if "d" in node.attrib:
            node.set("d", " ".join(node.get("d", "").split()))
        if node.text and not node.text.strip():
            node.text = None
        node.tail = None
    definitions = root.find(f"{{{namespace}}}defs")
    if definitions is None:
        definitions = ET.SubElement(root, f"{{{namespace}}}defs")
    stylesheet = ET.SubElement(definitions, f"{{{namespace}}}style")
    stylesheet.text = "".join(f".{name}{{{style}}}" for style, name in shared.items())
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")


def _save(figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".svg"), metadata={"Date": None, "Creator": "llm-quant evidence audit"})
    _compact_svg(path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(figure)


def render(payload: dict, output: Path) -> list[Path]:
    if payload.get("status") != "EVIDENCE_AUDIT_NO_VALIDATED_INVESTMENT_EDGE":
        raise ValueError("Expected a reviewed evidence-audit payload")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "quant-evidence-v1"}):
        rows = payload["track_d_recorded_cohort"]["rows"]
        labels = [row["strategy"].removesuffix("-sprint").upper().replace("-", " → ") for row in rows]
        values = [row["metrics"]["sharpe"] for row in rows]
        fig, ax = plt.subplots(figsize=(11, 5.8))
        bars = ax.barh(labels, values, height=0.62)
        for bar, row, value in zip(bars, rows, values, strict=True):
            failed = row["reported_verdict"] == "RECORDED_FAILURE"
            if failed:
                bar.set_hatch("///")
            status = "recorded fail" if failed else "recorded advance"
            ax.text(max(value + 0.025, 0.825), bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}  ·  {status}", va="center", fontsize=10)
        ax.axvline(0.8, linestyle="--", linewidth=1.2, label="Historical Sharpe gate: 0.80")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.73)
        ax.set_xlabel("Reported backtest Sharpe—not forward or live performance")
        ax.set_title("Track D: recorded historical research results", loc="left", pad=18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.22)
        fig.text(0.04, 0.085,
                 "Source: 24 Apr 2026 eight-strategy report. Five clear thresholds rechecked; backtests not rerun.\n"
                 "AGG→SOXL also failed the shuffled-signal test (reported p=0.527). Advance does not mean profitable.",
                 fontsize=9, va="center")
        path = output / "track-d-record-audit"
        _save(fig, path); paths.append(path.with_suffix(".svg"))

        rows = payload["housing"]["all_start_phases"]
        labels = [f"Q{row['offset_periods'] + 1} start\n{row['folds']} windows · {row['loss_events']} severe-loss outcome{'s' if row['loss_events'] != 1 else ''}" for row in rows]
        values = [row["model_minus_baseline_brier"] * 10000 for row in rows]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        bars = ax.barh(labels, values, height=0.55)
        for bar, value in zip(bars, values, strict=True):
            ax.text(value + (0.08 if value >= 0 else -0.08), bar.get_y() + bar.get_height() / 2,
                    f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=11)
        ax.axvline(0, linewidth=1)
        ax.invert_yaxis()
        ax.set_xlim(-2.7, 0.7)
        ax.set_xlabel("Model minus baseline Brier score × 10⁻⁴  |  negative favours model")
        ax.set_title("Housing: the small apparent advantage depends on the start quarter", loc="left", pad=18)
        ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=0.31, right=0.96, top=0.86, bottom=0.24)
        fig.text(0.04, 0.09,
                 "Same captured Australian real-price history; 4-quarter horizon. Zero or one ≥10% loss per phase.\n"
                 "Outcomes overlap across phases: do not pool these as independent trials. No current forecast or proven edge.",
                 fontsize=9, va="center")
        path = output / "housing-phase-sensitivity"
        _save(fig, path); paths.append(path.with_suffix(".svg"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/evidence/model-audit-2026-11-11.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for path in render(payload, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()

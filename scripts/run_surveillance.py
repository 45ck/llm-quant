"""Run a full governance surveillance scan.

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_surveillance.py [--json]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.config import load_config
from llm_quant.db.schema import get_connection, init_schema
from llm_quant.surveillance.models import SeverityLevel
from llm_quant.surveillance.scanner import SurveillanceScanner

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


def main() -> None:
    json_output = "--json" in sys.argv

    config = load_config()
    db_path = config.general.db_path
    project_root = Path(__file__).resolve().parent.parent
    if not Path(db_path).is_absolute():
        db_path = str(project_root / db_path)

    path = Path(db_path)
    if not path.exists():
        init_schema(db_path)
    conn = get_connection(db_path)

    try:
        scanner = SurveillanceScanner(config)
        report = scanner.run_full_scan(conn)
        scanner.persist_scan(conn, report)

        if json_output:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            # Human-readable output
            print(
                f"Surveillance Scan — {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
            )
            print(f"Overall: {report.overall_severity.value.upper()}")
            print(
                f"Checks: {len(report.checks)} total, "
                f"{len(report.halt_checks)} halts, "
                f"{len(report.warning_checks)} warnings"
            )
            print()

            if report.halt_checks:
                print("HALTS:")
                for c in report.halt_checks:
                    print(f"  [{c.detector}] {c.message}")
                print()

            if report.warning_checks:
                print("WARNINGS:")
                for c in report.warning_checks:
                    print(f"  [{c.detector}] {c.message}")
                print()

            ok_checks = [c for c in report.checks if c.severity == SeverityLevel.OK]
            if ok_checks:
                print("OK:")
                for c in ok_checks:
                    print(f"  [{c.detector}] {c.message}")

        # Exit with non-zero if halt
        if report.overall_severity == SeverityLevel.HALT:
            sys.exit(2)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

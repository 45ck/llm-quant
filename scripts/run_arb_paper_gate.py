"""CLI runner for the PM arb 30-day paper validation gate.

Usage:
    python scripts/run_arb_paper_gate.py [--source kalshi|polymarket] [--db data/quant.db]

Exit codes:
    0 — PROMOTE   (all gates pass)
    1 — REJECT    (a quality gate failed)
    2 — CONTINUE_PAPER (quality gates pass but <30 days of data)
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.arb.paper_gate import PaperArbGate

_EXIT_CODES = {
    "PROMOTE": 0,
    "REJECT": 1,
    "CONTINUE_PAPER": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 30-day PM arb paper validation gate.",
    )
    parser.add_argument(
        "--source",
        default="kalshi",
        choices=["kalshi", "polymarket"],
        help="Data source to validate (default: kalshi)",
    )
    parser.add_argument(
        "--db",
        default="data/quant.db",
        help="DuckDB file path (default: data/quant.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    gate = PaperArbGate(db_path=db_path, source=args.source)
    report = gate.run_gate()
    gate.print_report(report)

    sys.exit(_EXIT_CODES.get(report.recommendation, 1))


if __name__ == "__main__":
    main()

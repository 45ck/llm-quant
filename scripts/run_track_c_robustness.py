#!/usr/bin/env python3
"""Track C structural arb robustness gate runner.

Routes to the appropriate gate for each Track C strategy class:
  pm-arb-*       → PaperArbGate (4 gates: persistence, fill_rate, capacity, days)
  cef-*          → CEF backtest gate (Sharpe, MaxDD, persistence in 8/12 months)
  funding-*      → Funding rate gate (annualized spread, fill rate, exchange count)

Writes data/strategies/<slug>/robustness-result.yaml in the standard format
that /robustness and /promote commands read.

Exit codes:
  0 — PROMOTE        (all gates pass)
  1 — REJECT         (quality gate failed)
  2 — CONTINUE_PAPER (quality gates pass, but insufficient track record time)

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/run_track_c_robustness.py \\
        --slug pm-arb-kalshi [--db data/quant.db]
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Ensure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_EXIT_CODES = {"PROMOTE": 0, "REJECT": 1, "CONTINUE_PAPER": 2}

# -------------------------------------------------------------------
# Strategy class routing
# -------------------------------------------------------------------

_TRACK_C_PREFIXES = {
    "pm-arb-": "pm_arb",
    "cef-": "cef_discount",
    "funding-": "funding_rate",
}


def detect_strategy_class(slug: str) -> str | None:
    """Return strategy class string, or None if not a Track C slug."""
    for prefix, cls in _TRACK_C_PREFIXES.items():
        if slug.startswith(prefix):
            return cls
    return None


# -------------------------------------------------------------------
# PM Arb gate (PaperArbGate)
# -------------------------------------------------------------------


def run_pm_arb_gate(slug: str, db_path: Path) -> dict:
    """Run PaperArbGate and return robustness-result dict."""
    from llm_quant.arb.paper_gate import PaperArbGate

    # Infer source from slug: pm-arb-kalshi → kalshi, pm-arb-polymarket → polymarket
    parts = slug.split("-")
    source = (
        parts[2]
        if len(parts) >= 3 and parts[2] in ("kalshi", "polymarket")
        else "kalshi"
    )

    gate = PaperArbGate(db_path=db_path, source=source)
    report = gate.run_gate()
    gate.print_report(report)

    gate_details = {
        g.gate_name.lower().replace(" ", "_"): {
            "passed": g.passed,
            "value": round(g.value, 4),
            "threshold": g.threshold,
            "detail": g.detail,
        }
        for g in report.gates
    }

    return {
        "strategy_type": "track_c",
        "strategy_class": "pm_arb",
        "slug": slug,
        "source": source,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "scan_start": report.scan_start,
        "scan_end": report.scan_end,
        "days_elapsed": report.days_elapsed,
        "total_scans": report.total_scans,
        "total_opps_detected": report.total_opps_detected,
        "gate_details": gate_details,
        "overall_passed": report.overall_pass,
        "recommendation": report.recommendation,
        "summary": report.summary,
        # Track C gates instead of DSR/CPCV/PBO
        "dsr": None,  # not applicable — deductive arb has no statistical overfitting risk
        "pbo": None,
        "cpcv": None,
    }


# -------------------------------------------------------------------
# CEF Discount gate (placeholder — full lifecycle via /backtest)
# -------------------------------------------------------------------


def run_cef_gate(slug: str, db_path: Path) -> dict:
    """Run CEF discount gate.

    For CEF discount mean-reversion:
      - Sharpe >= 0.5 (Track C mandate)
      - MaxDD < 20%
      - Persistence: opportunities in >= 8 of 12 months of data
      - Beta to SPY < 0.15

    Currently a placeholder — full lifecycle handled in llm-quant-rlpt bead.
    """
    print(f"[WARN] CEF gate not yet implemented for slug={slug!r}.")
    print("       Run /backtest for full lifecycle.")
    print("       Expected gates: Sharpe>0.5, MaxDD<20%, beta<0.15")
    print()
    return {
        "strategy_type": "track_c",
        "strategy_class": "cef_discount",
        "slug": slug,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "gate_details": {},
        "overall_passed": False,
        "recommendation": "REJECT",
        "summary": "CEF gate not implemented — run /backtest {slug} to execute full lifecycle.",
    }


# -------------------------------------------------------------------
# Funding Rate gate (placeholder — full lifecycle via /backtest)
# -------------------------------------------------------------------


def run_funding_rate_gate(slug: str, db_path: Path) -> dict:
    """Run funding rate gate.

    For crypto funding rate arb:
      - Annualized spread >= 5% (min viable yield)
      - Fill rate >= 80% (can enter both legs)
      - ≥ 2 exchanges (single-exchange concentration risk)
      - Track record >= 30 days

    Currently a placeholder — full lifecycle handled in llm-quant-c2yv bead.
    """
    print(f"[WARN] Funding rate gate not yet implemented for slug={slug!r}.")
    print("       Run scripts/run_funding_scanner.py to build track record.")
    print("       Expected gates: annualized_spread>5%, fill_rate>80%, exchanges>=2")
    print()
    return {
        "strategy_type": "track_c",
        "strategy_class": "funding_rate",
        "slug": slug,
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "gate_details": {},
        "overall_passed": False,
        "recommendation": "REJECT",
        "summary": "Funding rate gate not implemented — run run_funding_scanner.py first.",
    }


# -------------------------------------------------------------------
# Writer
# -------------------------------------------------------------------


def write_robustness_result(slug: str, result: dict) -> Path:
    """Write robustness-result.yaml to data/strategies/<slug>/."""
    strategy_dir = Path("data/strategies") / slug
    strategy_dir.mkdir(parents=True, exist_ok=True)
    out_path = strategy_dir / "robustness-result.yaml"
    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    return out_path


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track C structural arb robustness gate runner.",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Strategy slug (e.g. pm-arb-kalshi, cef-pdi-discount, funding-btc-binance)",
    )
    parser.add_argument(
        "--db",
        default="data/arb.duckdb",
        help="DuckDB file for arb data (default: data/arb.duckdb)",
    )
    args = parser.parse_args()

    slug = args.slug.strip()
    db_path = Path(args.db)

    strategy_class = detect_strategy_class(slug)
    if strategy_class is None:
        print(
            f"ERROR: '{slug}' is not a Track C strategy slug.\n"
            f"Track C prefixes: {', '.join(_TRACK_C_PREFIXES.keys())}\n"
            "For Track A/B strategies, use: PYTHONPATH=src python scripts/run_robustness.py"
        )
        return 1

    print(f"Track C Robustness Gate: {slug!r} ({strategy_class})")
    print(f"DB: {db_path}")
    print()

    if strategy_class == "pm_arb":
        if not db_path.exists():
            print(
                f"ERROR: Arb DB not found at {db_path}.\n"
                f"Run a Kalshi scan first to build scan history:\n"
                f"  PYTHONPATH=src python scripts/run_pm_scanner.py --source kalshi\n"
            )
            return 1
        result = run_pm_arb_gate(slug, db_path)
    elif strategy_class == "cef_discount":
        result = run_cef_gate(slug, db_path)
    elif strategy_class == "funding_rate":
        result = run_funding_rate_gate(slug, db_path)
    else:
        print(f"ERROR: Unhandled strategy class: {strategy_class}")
        return 1

    out_path = write_robustness_result(slug, result)
    print(f"\nRobustness result written to: {out_path}")

    recommendation = result.get("recommendation", "REJECT")
    return _EXIT_CODES.get(recommendation, 1)


if __name__ == "__main__":
    sys.exit(main())

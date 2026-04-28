#!/usr/bin/env python3
"""Daily paper trading orchestrator with automatic catch-up.

Detects the latest date logged across all paper-trading.yaml files, queries the
SPY trading calendar from DuckDB for any missed trading days since, and runs
``run_paper_batch.py`` once per missed day (``--date`` for backfills, no flag
for the most recent day).

Designed to run unattended from Windows Task Scheduler. All output is captured
by the caller (.bat wrapper) so failures land in a log file.

Usage::

    cd E:/llm-quant && PYTHONPATH=src python scripts/daily_paper_cron.py
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import sys
from pathlib import Path

import duckdb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "strategies"
DUCKDB_PATH = PROJECT_ROOT / "data" / "llm_quant.duckdb"
RUN_BATCH = PROJECT_ROOT / "scripts" / "run_paper_batch.py"
PYTHON = sys.executable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paper_cron")


def find_latest_logged_date() -> datetime.date | None:
    latest: datetime.date | None = None
    for path in DATA_DIR.glob("*/paper-trading.yaml"):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            log.warning("Skipping %s: %s", path, exc)
            continue
        for entry in (doc or {}).get("daily_log", []) or []:
            ds = str(entry.get("date", ""))
            if not ds:
                continue
            try:
                d = datetime.date.fromisoformat(ds)
            except ValueError:
                continue
            if latest is None or d > latest:
                latest = d
    return latest


def spy_trading_days_after(start: datetime.date) -> list[datetime.date]:
    if not DUCKDB_PATH.exists():
        log.error("DuckDB not found at %s", DUCKDB_PATH)
        return []
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT date FROM market_data_daily "
            "WHERE symbol = 'SPY' AND date > ? ORDER BY date ASC",
            [start],
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def run_batch(date: datetime.date | None) -> int:
    cmd = [PYTHON, str(RUN_BATCH)]
    if date is not None:
        cmd += ["--date", date.isoformat()]
    log.info("RUN: %s", " ".join(cmd[-3:] if date else cmd[-1:]))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)  # noqa: S603
    return proc.returncode


def main() -> int:
    latest = find_latest_logged_date()
    if latest is None:
        log.error("No paper-trading.yaml files found")
        return 2
    log.info("Latest logged date: %s", latest)

    missing = spy_trading_days_after(latest)
    if not missing:
        log.info("Up to date — no trading days to add")
        return 0

    log.info("Missing %d trading day(s): %s", len(missing), missing)

    backfill_dates = missing[:-1]
    final_date = missing[-1]

    for d in backfill_dates:
        rc = run_batch(d)
        if rc != 0:
            log.error("Backfill --date %s failed (rc=%d) — stopping", d, rc)
            return rc

    rc = run_batch(None)
    if rc != 0:
        log.error("Final batch run for %s failed (rc=%d)", final_date, rc)
        return rc

    log.info("DONE — %d day(s) logged through %s", len(missing), final_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())

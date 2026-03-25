"""Surveillance scanner — orchestrates all governance detectors."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import duckdb

from llm_quant.config import AppConfig
from llm_quant.surveillance.detectors import (
    check_alpha_decay,
    check_data_quality,
    check_kill_switches,
    check_operational_health,
    check_process_drift,
    check_regime_drift,
    check_risk_drift,
)
from llm_quant.surveillance.models import (
    SeverityLevel,
    SurveillanceCheck,
    SurveillanceReport,
)

logger = logging.getLogger(__name__)


class SurveillanceScanner:
    """Runs all governance detectors and produces a SurveillanceReport."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.governance = config.governance

    def run_full_scan(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> SurveillanceReport:
        """Execute all detectors and return aggregate report."""
        report = SurveillanceReport(timestamp=datetime.now(tz=UTC))

        detectors = [
            ("regime_drift", check_regime_drift),
            ("alpha_decay", check_alpha_decay),
            ("risk_drift", check_risk_drift),
            ("data_quality", check_data_quality),
            ("process_drift", check_process_drift),
            ("operational_health", check_operational_health),
            ("kill_switches", check_kill_switches),
        ]

        for name, detector_fn in detectors:
            try:
                checks = detector_fn(conn, self.config)
                report.checks.extend(checks)
            except Exception:
                logger.exception("Detector '%s' failed with exception", name)
                report.checks.append(
                    SurveillanceCheck(
                        detector=name,
                        severity=SeverityLevel.WARNING,
                        message=(
                            f"Detector '{name}' raised an "
                            "exception — treating as warning."
                        ),
                    )
                )

        logger.info(
            "Surveillance scan complete: %s (%d checks, %d halts, %d warnings)",
            report.overall_severity.value,
            len(report.checks),
            len(report.halt_checks),
            len(report.warning_checks),
        )

        return report

    def persist_scan(
        self,
        conn: duckdb.DuckDBPyConnection,
        report: SurveillanceReport,
    ) -> None:
        """Save scan results to DuckDB surveillance_scans table."""
        conn.execute(
            """
            INSERT INTO surveillance_scans (
                scan_timestamp, overall_severity, total_checks,
                halt_count, warning_count, checks_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                report.timestamp,
                report.overall_severity.value,
                len(report.checks),
                len(report.halt_checks),
                len(report.warning_checks),
                json.dumps(report.to_dict()["checks"]),
            ],
        )
        conn.commit()
        logger.info("Persisted surveillance scan to DuckDB.")

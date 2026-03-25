"""Surveillance data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SeverityLevel(Enum):
    """Severity of a surveillance finding."""

    OK = "ok"
    WARNING = "warning"
    HALT = "halt"


@dataclass
class SurveillanceCheck:
    """Result of a single surveillance detector."""

    detector: str
    severity: SeverityLevel
    message: str
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class SurveillanceReport:
    """Aggregate result of a full surveillance scan."""

    timestamp: datetime
    checks: list[SurveillanceCheck] = field(default_factory=list)

    @property
    def overall_severity(self) -> SeverityLevel:
        """Return the worst severity across all checks."""
        if any(c.severity == SeverityLevel.HALT for c in self.checks):
            return SeverityLevel.HALT
        if any(c.severity == SeverityLevel.WARNING for c in self.checks):
            return SeverityLevel.WARNING
        return SeverityLevel.OK

    @property
    def halt_checks(self) -> list[SurveillanceCheck]:
        return [c for c in self.checks if c.severity == SeverityLevel.HALT]

    @property
    def warning_checks(self) -> list[SurveillanceCheck]:
        return [c for c in self.checks if c.severity == SeverityLevel.WARNING]

    @property
    def is_clear(self) -> bool:
        return self.overall_severity == SeverityLevel.OK

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_severity": self.overall_severity.value,
            "total_checks": len(self.checks),
            "halts": len(self.halt_checks),
            "warnings": len(self.warning_checks),
            "checks": [
                {
                    "detector": c.detector,
                    "severity": c.severity.value,
                    "message": c.message,
                    "metric_name": c.metric_name,
                    "current_value": c.current_value,
                    "threshold_value": c.threshold_value,
                }
                for c in self.checks
            ],
        }

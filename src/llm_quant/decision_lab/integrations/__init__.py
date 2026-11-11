"""Opt-in research adapters. Importing this package performs no external I/O."""

from .contracts import MissingIntegration, TrainingBatch

__all__ = ["MissingIntegration", "TrainingBatch"]

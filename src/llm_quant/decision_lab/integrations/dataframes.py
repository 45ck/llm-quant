"""Explicit DataFrame boundary validation. No filling, repair or column removal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import require
from .contracts import load


def validate_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    require(isinstance(frame, pd.DataFrame), "Expected pandas OHLC frame")
    required = ["Open", "High", "Low", "Close"]
    require(list(frame.columns) == required, "Exact OHLC columns required; volume is not modeled")
    require(isinstance(frame.index, pd.DatetimeIndex) and not frame.index.hasnans
            and frame.index.is_monotonic_increasing and frame.index.is_unique,
            "OHLC timestamps must be nonmissing, ordered and unique")
    require(20 <= len(frame) <= 50000, "OHLC row count outside research budget")
    require(all(pd.api.types.is_float_dtype(dtype) for dtype in frame.dtypes),
            "OHLC columns must be explicit floating-point values")
    values = frame.to_numpy(copy=True)
    require(bool(np.all(np.isfinite(values) & (values > 0))), "Invalid OHLC values")
    require(bool((frame.High >= frame[["Open", "Close", "Low"]].max(axis=1)).all())
            and bool((frame.Low <= frame[["Open", "Close", "High"]].min(axis=1)).all()),
            "OHLC price envelope is inconsistent")
    return frame.copy(deep=True)


def pandera_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    checked = validate_ohlc(frame)
    pa = load("pandera", "pandas")
    schema = pa.DataFrameSchema(
        {column: pa.Column(float, [pa.Check.gt(0), pa.Check(np.isfinite)], nullable=False)
         for column in ("Open", "High", "Low", "Close")},
        checks=[pa.Check(lambda data: data.High >= data[["Open", "Close", "Low"]].max(axis=1)),
                pa.Check(lambda data: data.Low <= data[["Open", "Close", "High"]].min(axis=1))],
        strict=True, coerce=False,
    )
    return schema.validate(checked, lazy=True)

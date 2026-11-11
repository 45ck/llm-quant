"""Opt-in Backtesting.py bar-timing diagnostic, separate from portfolio allocation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data import require
from .contracts import versions
from .dataframes import validate_ohlc


def next_open_buy_hold(frame: pd.DataFrame, *, acknowledge_agpl: bool = False,
                       commission_bps: float = 10.0, initial_cash: float = 10000.0) -> dict:
    # Check acknowledgement before importing the separately licensed component.
    require(acknowledge_agpl is True, "Backtesting.py requires explicit AGPL acknowledgement")
    data = validate_ohlc(frame)
    require(math.isfinite(commission_bps) and 0 <= commission_bps <= 500,
            "Invalid commission")
    require(math.isfinite(initial_cash) and initial_cash > 10 * float(data.High.max()),
            "Cash must exceed ten times the largest bar high")
    from .contracts import load
    library = load("backtesting")

    class NextOpenHold(library.Strategy):
        def init(self):
            self.entered = False

        def next(self):
            # First next() is bar 1; its market order fills at bar 2's open.
            # Exit is submitted on the penultimate bar, then filled at last open.
            if not self.entered:
                self.buy(size=0.95)
                self.entered = True
            elif len(self.data) == len(data) - 1:
                self.position.close()

    engine = library.Backtest(data, NextOpenHold, cash=initial_cash,
                              commission=commission_bps / 10000, spread=0.0,
                              margin=1.0, trade_on_close=False, hedging=False,
                              exclusive_orders=True, finalize_trades=False)
    stats = engine.run()
    trades = stats["_trades"]
    require(len(trades) == 1, "Reference expects exactly one completed trade")
    entry, exit_bar = int(trades.iloc[0].EntryBar), int(trades.iloc[0].ExitBar)
    require(entry == 2 and exit_bar == len(data) - 1, "Unexpected market-order fill timing")
    nav = np.asarray(stats["_equity_curve"]["Equity"], dtype=float)
    require(bool(np.all(np.isfinite(nav) & (nav > 0))), "Invalid backend equity")
    return {
        "status": "EXECUTION_DIAGNOSTIC_NOT_STRATEGY_VALIDATION",
        "backend": "backtesting", "libraries": versions("backtesting"),
        "entry_bar": entry, "exit_bar": exit_bar, "completed_trades": len(trades),
        "initial_cash": initial_cash, "final_equity": float(nav[-1]),
        "total_return": float(nav[-1] / initial_cash - 1), "commission_bps_each_side": commission_bps,
        "limitations": ["Single instrument; no spread, impact, tax or corporate actions.",
                        "Partial whole-share allocation; not identical to the return-matrix engine.",
                        "Backtesting.py retains its upstream AGPL license; this is not a license waiver."],
    }

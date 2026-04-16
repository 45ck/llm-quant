"""Execute a trading decision from JSON on stdin.

Parses the JSON decision, runs risk checks, executes trades,
saves portfolio snapshot, and prints execution summary.

Usage:
    cd E:/llm-quant && PYTHONPATH=src \\
        python scripts/execute_decision.py \\
        <<< '{"market_regime": "risk_on", ...}'
    cd E:/llm-quant && PYTHONPATH=src \\
        python scripts/execute_decision.py --pod momo \\
        <<< '{"market_regime": "risk_on", ...}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.brain.models import Action, MarketRegime, TradingDecision
from llm_quant.brain.parser import parse_trading_decision
from llm_quant.config import load_config_for_pod
from llm_quant.db.schema import get_connection
from llm_quant.risk.manager import RiskManager
from llm_quant.trading.executor import execute_signals
from llm_quant.trading.ledger import log_trades, save_portfolio_snapshot
from llm_quant.trading.portfolio import Portfolio

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whipsaw R2 — cash-floor guard
# ---------------------------------------------------------------------------
#
# Root cause of the 2026-04-01 whipsaw: the LLM mass-closed 7 equity positions
# on stale context, pushing the portfolio to 51% cash / 49% gross with no
# active crisis. R2 prevents a repeat by rejecting decisions that project
# cash above 40% of NAV unless the regime is explicitly risk_off OR a halt is
# active in governance. Mass-close is a recovery path, not a default.
#
# See docs/investigations/apr01-whipsaw.md recommendations R1-R5.

CASH_FLOOR_PCT = 0.40  # Reject if projected cash > 40% of NAV
RISK_OFF_CONFIDENCE_THRESHOLD = 0.70


def project_cash_after_signals(
    portfolio: Portfolio,
    decision: TradingDecision,
    prices: dict[str, float],
) -> tuple[float, float]:
    """Project cash and NAV after every signal in *decision* executes.

    This is a best-effort estimate that mirrors ``executor.execute_signals``
    logic (BUY raises cost, SELL/CLOSE raises cash) but without mutating the
    live portfolio. Used by the R2 cash-floor guard before any trade fires.

    Skips signals lacking a price entry, matching executor behaviour.

    Parameters
    ----------
    portfolio:
        Live portfolio (pre-trade). NAV is computed from its current state.
    decision:
        Parsed trading decision with signal list.
    prices:
        Latest prices keyed by symbol.

    Returns
    -------
    (projected_cash, nav_before) where projected_cash is post-signal cash in
    USD and nav_before is pre-trade NAV (the denominator for cash_pct).
    """
    nav_before = portfolio.nav
    projected_cash = portfolio.cash

    # Snapshot current market values so repeated signals on same symbol compose.
    projected_mv: dict[str, float] = {
        sym: pos.market_value for sym, pos in portfolio.positions.items()
    }

    for sig in decision.signals:
        symbol = sig.symbol
        price = prices.get(symbol)
        if price is None or price <= 0.0:
            continue

        current_mv = projected_mv.get(symbol, 0.0)

        if sig.action == Action.BUY:
            # Buys toward target_weight * nav_before (same math as executor).
            target_notional = sig.target_weight * nav_before
            additional = target_notional - current_mv
            if additional <= 0.0:
                continue
            # Can't spend more than available cash.
            spend = min(additional, max(projected_cash, 0.0))
            projected_cash -= spend
            projected_mv[symbol] = current_mv + spend

        elif sig.action == Action.SELL:
            # Reduce toward target_weight * nav_before.
            target_notional = sig.target_weight * nav_before
            reduce = current_mv - target_notional
            if reduce <= 0.0 or current_mv <= 0.0:
                continue
            reduce = min(reduce, current_mv)
            projected_cash += reduce
            projected_mv[symbol] = current_mv - reduce

        elif sig.action == Action.CLOSE:
            if current_mv > 0.0:
                projected_cash += current_mv
                projected_mv[symbol] = 0.0
        # HOLD: no cash movement.

    return projected_cash, nav_before


def check_cash_floor_guard(
    portfolio: Portfolio,
    decision: TradingDecision,
    prices: dict[str, float],
    governance_severity: str = "ok",
) -> tuple[bool, str | None, dict]:
    """Enforce the R2 cash-floor guard.

    Rejects a decision that would push cash above :data:`CASH_FLOOR_PCT` of
    NAV during a ``risk_on`` or ``transition`` regime, unless the regime is
    explicitly ``risk_off`` with confidence >= 0.70 OR governance has halted.

    Returns
    -------
    (allowed, error_message, debug) where ``allowed`` is True when the
    decision passes the guard, ``error_message`` carries a PM-readable
    explanation on rejection (None otherwise), and ``debug`` contains the
    projected cash math for observability.
    """
    projected_cash, nav_before = project_cash_after_signals(portfolio, decision, prices)
    projected_cash_pct = projected_cash / nav_before if nav_before > 0 else 0.0

    debug = {
        "nav_before": round(nav_before, 2),
        "projected_cash": round(projected_cash, 2),
        "projected_cash_pct": round(projected_cash_pct, 4),
        "regime": decision.market_regime.value,
        "regime_confidence": decision.regime_confidence,
        "governance_severity": governance_severity,
        "cash_floor_pct": CASH_FLOOR_PCT,
    }

    # Guard only fires when projection exceeds floor.
    if projected_cash_pct <= CASH_FLOOR_PCT:
        return True, None, debug

    # Allowed: explicit risk_off regime with high confidence.
    if (
        decision.market_regime == MarketRegime.RISK_OFF
        and decision.regime_confidence >= RISK_OFF_CONFIDENCE_THRESHOLD
    ):
        return True, None, debug

    # Allowed: governance is in halt state (sells-only gate already in force).
    if str(governance_severity).lower() == "halt":
        return True, None, debug

    error = (
        f"Projected cash {projected_cash_pct * 100:.0f}% exceeds "
        f"{int(CASH_FLOOR_PCT * 100)}% floor in non-risk-off regime "
        f"(regime={decision.market_regime.value}, "
        f"confidence={decision.regime_confidence:.2f}). "
        "Either include specific risk-off justification in "
        "portfolio_commentary AND explicitly mark regime=risk_off with "
        f"confidence>=0.70, or reduce the number of SELL/CLOSE actions. "
        "Mass-close is a recovery path, not a default (whipsaw R2)."
    )
    return False, error, debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute trading decision")
    parser.add_argument("--pod", default="default", help="Pod ID to execute for")
    args = parser.parse_args()
    pod_id = args.pod

    # Read JSON from stdin
    raw_input = sys.stdin.read().strip()
    if not raw_input:
        print(json.dumps({"error": "No input received on stdin"}))
        sys.exit(1)

    config = load_config_for_pod(pod_id)
    db_path = config.general.db_path

    # Resolve relative db_path
    project_root = Path(__file__).resolve().parent.parent
    if not Path(db_path).is_absolute():
        db_path = str(project_root / db_path)

    conn = get_connection(db_path)

    try:
        # Parse the trading decision
        today = datetime.now(tz=UTC).date()
        decision = parse_trading_decision(raw_input, today)

        # Load portfolio
        portfolio = Portfolio.from_db(
            conn, config.general.initial_capital, pod_id=pod_id
        )

        # Get latest prices
        prices: dict[str, float] = {}
        symbols = set()
        for sig in decision.signals:
            symbols.add(sig.symbol)
        for sym in list(portfolio.positions.keys()):
            symbols.add(sym)

        for symbol in symbols:
            row = conn.execute(
                "SELECT close FROM market_data_daily"
                " WHERE symbol = ? ORDER BY date DESC"
                " LIMIT 1",
                [symbol],
            ).fetchone()
            if row and row[0] is not None:
                prices[symbol] = float(row[0])
            else:
                logger.warning(
                    "No price data in market_data_daily for %s — "
                    "trades for this symbol will be skipped",
                    symbol,
                )

        portfolio.update_prices(prices)
        nav_before = portfolio.nav

        # Whipsaw R2 — cash-floor guard (see docs/investigations/apr01-whipsaw.md).
        # Projected cash after all signals must not exceed 40% of NAV during
        # risk_on / transition regimes. Rejections return a PM-readable error
        # and exit non-zero; the LLM reconsiders rather than retrying blindly.
        governance_severity = "ok"
        try:
            raw_data = (
                json.loads(raw_input) if raw_input.strip().startswith("{") else {}
            )
            gov_block = (
                raw_data.get("governance") if isinstance(raw_data, dict) else None
            )
            if isinstance(gov_block, dict):
                governance_severity = str(
                    gov_block.get("overall_severity", "ok")
                ).lower()
        except (json.JSONDecodeError, AttributeError):
            governance_severity = "ok"

        allowed, guard_error, guard_debug = check_cash_floor_guard(
            portfolio, decision, prices, governance_severity=governance_severity
        )
        if not allowed:
            print(
                json.dumps(
                    {
                        "error": guard_error,
                        "guard": "cash_floor_r2",
                        "debug": guard_debug,
                    }
                )
            )
            sys.exit(1)

        # Risk filter
        risk_mgr = RiskManager(config)
        approved, rejected = risk_mgr.filter_signals(
            decision.signals, portfolio, prices
        )

        # Execute approved signals
        executed = execute_signals(portfolio, approved, prices, nav_before)

        # Log trades and save snapshot
        decision_id = None
        trade_ids = (
            log_trades(conn, executed, today, decision_id, pod_id=pod_id)
            if executed
            else []
        )

        # Compute daily P&L (change from previous day's NAV)
        prev_snap = conn.execute(
            """
            SELECT nav FROM portfolio_snapshots
            WHERE date < ?
            ORDER BY date DESC, snapshot_id DESC
            LIMIT 1
            """,
            [today],
        ).fetchone()

        daily_pnl = None
        if prev_snap is not None:
            daily_pnl = portfolio.nav - float(prev_snap[0])

        snapshot_id = save_portfolio_snapshot(
            conn, portfolio, today, daily_pnl=daily_pnl, pod_id=pod_id
        )

        # Build summary
        summary = {
            "pod_id": pod_id,
            "date": str(today),
            "decision": {
                "market_regime": decision.market_regime.value,
                "regime_confidence": decision.regime_confidence,
                "regime_reasoning": decision.regime_reasoning,
                "portfolio_commentary": decision.portfolio_commentary,
                "total_signals": len(decision.signals),
            },
            "risk_filter": {
                "approved": len(approved),
                "rejected": len(rejected),
                "rejected_details": [
                    {
                        "symbol": sig.symbol,
                        "action": sig.action.value,
                        "failures": [c.message for c in checks if not c.passed],
                    }
                    for sig, checks in rejected
                ],
            },
            "executed_trades": [
                {
                    "symbol": t.symbol,
                    "action": t.action,
                    "shares": t.shares,
                    "price": round(t.price, 2),
                    "notional": round(t.notional, 2),
                    "conviction": t.conviction,
                    "reasoning": t.reasoning,
                }
                for t in executed
            ],
            "portfolio_after": {
                "nav": round(portfolio.nav, 2),
                "cash": round(portfolio.cash, 2),
                "positions": len(portfolio.positions),
                "total_pnl": round(portfolio.total_pnl, 2),
                "gross_exposure": round(portfolio.gross_exposure, 2),
            },
            "snapshot_id": snapshot_id,
            "trade_ids": trade_ids,
        }

        print(json.dumps(summary, indent=2))

    except ValueError as e:
        print(json.dumps({"error": f"Failed to parse decision: {e}"}))
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        print(json.dumps({"error": f"Execution failed: {e}"}))
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

"""Pre-trade risk manager.

Orchestrates all individual risk checks from :mod:`llm_quant.risk.limits`
and decides which signals are safe to execute.
"""

from __future__ import annotations

import logging

from llm_quant.brain.models import Action, TradeSignal
from llm_quant.config import AppConfig
from llm_quant.risk.limits import (
    RiskCheckResult,
    check_cash_reserve,
    check_gross_exposure,
    check_net_exposure,
    check_position_size,
    check_position_weight,
    check_sector_concentration,
    check_stop_loss,
)
from llm_quant.trading.portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    """Stateless pre-trade risk gate.

    The manager holds a reference to the risk-limit configuration and the
    sector map derived from the investment universe.  For each proposed
    ``TradeSignal`` it runs the full battery of checks and returns
    structured results.
    """

    def __init__(self, config: AppConfig) -> None:
        self.limits = config.risk
        self.sector_map: dict[str, str] = {
            e.symbol: e.sector for e in config.universe.assets
        }
        logger.info(
            "RiskManager initialised – %d sector mappings, limits=%s",
            len(self.sector_map),
            self.limits.model_dump(),
        )

    # ------------------------------------------------------------------
    # Single-signal evaluation
    # ------------------------------------------------------------------

    def check_trade(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> list[RiskCheckResult]:
        """Run **all** risk checks on a single proposed trade.

        Parameters
        ----------
        signal:
            The trade signal to evaluate.
        portfolio:
            Current portfolio state (already price-updated).
        prices:
            Latest market prices keyed by symbol.

        Returns
        -------
        list[RiskCheckResult]
            One result per check – callers can inspect ``.passed`` on
            each to decide whether the trade should proceed.
        """
        results: list[RiskCheckResult] = []
        nav = portfolio.nav
        price = prices.get(signal.symbol, 0.0)

        # For HOLD / CLOSE / SELL we are *reducing* risk – most limit
        # checks only apply to new buys.
        is_buy = signal.action == Action.BUY

        # ---- Trade notional estimation --------------------------------
        # For buys: target_weight * nav is the desired position size;
        # the *incremental* notional is the difference from the current
        # position.
        current_weight = portfolio.get_position_weight(signal.symbol)

        if is_buy:
            additional_weight = max(signal.target_weight - current_weight, 0.0)
            trade_notional = additional_weight * nav
        else:
            # Sells / closes free up capital; compute notional for
            # informational checks but don't block on cash/exposure.
            existing = portfolio.positions.get(signal.symbol)
            if existing is not None and price > 0:
                if signal.action == Action.CLOSE:
                    trade_notional = abs(existing.market_value)
                else:
                    reduce_weight = max(current_weight - signal.target_weight, 0.0)
                    trade_notional = reduce_weight * nav
            else:
                trade_notional = 0.0

        # 1. Position size (single-trade cap)
        results.append(
            check_position_size(trade_notional, nav, self.limits.max_trade_size)
        )

        # 2. Position weight
        results.append(
            check_position_weight(
                current_weight,
                signal.target_weight if is_buy else max(current_weight - (trade_notional / nav if nav else 0), 0.0),
                self.limits.max_position_weight,
            )
        )

        # 3. Gross exposure
        if is_buy:
            results.append(
                check_gross_exposure(
                    portfolio.gross_exposure,
                    trade_notional,
                    nav,
                    self.limits.max_gross_exposure,
                )
            )
        else:
            # Sells reduce gross exposure – always pass.
            results.append(
                RiskCheckResult(
                    passed=True,
                    rule="gross_exposure",
                    message="Sell/close reduces gross exposure.",
                )
            )

        # 4. Net exposure
        if is_buy:
            signed_notional = trade_notional
        elif signal.action in (Action.SELL, Action.CLOSE):
            signed_notional = -trade_notional
        else:
            signed_notional = 0.0

        results.append(
            check_net_exposure(
                portfolio.net_exposure,
                signed_notional,
                nav,
                self.limits.max_net_exposure,
            )
        )

        # 5. Sector concentration (buys only)
        sector = self.sector_map.get(signal.symbol, "Unknown")
        sector_exposures = portfolio.get_sector_exposure(self.sector_map)
        sector_weight = sector_exposures.get(sector, 0.0)

        if is_buy:
            additional_sector_weight = additional_weight if is_buy else 0.0
            results.append(
                check_sector_concentration(
                    sector_weight,
                    additional_sector_weight,
                    self.limits.max_sector_concentration,
                )
            )
        else:
            results.append(
                RiskCheckResult(
                    passed=True,
                    rule="sector_concentration",
                    message="Sell/close reduces sector concentration.",
                )
            )

        # 6. Cash reserve (buys only)
        if is_buy:
            results.append(
                check_cash_reserve(
                    portfolio.cash,
                    trade_notional,
                    nav,
                    self.limits.min_cash_reserve,
                )
            )
        else:
            results.append(
                RiskCheckResult(
                    passed=True,
                    rule="cash_reserve",
                    message="Sell/close does not consume cash.",
                )
            )

        # 7. Stop-loss
        results.append(
            check_stop_loss(
                has_stop_loss=(signal.stop_loss > 0.0),
                require=self.limits.require_stop_loss,
            )
        )

        return results

    # ------------------------------------------------------------------
    # Batch filtering
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: list[TradeSignal],
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> tuple[list[TradeSignal], list[tuple[TradeSignal, list[RiskCheckResult]]]]:
        """Filter a batch of signals through the risk gate.

        Parameters
        ----------
        signals:
            Raw signals from the LLM brain.
        portfolio:
            Current portfolio state (already price-updated).
        prices:
            Latest market prices keyed by symbol.

        Returns
        -------
        tuple[list[TradeSignal], list[tuple[TradeSignal, list[RiskCheckResult]]]]
            ``(approved, rejected)`` where *rejected* pairs each signal
            with the full list of check results (including passed ones)
            for transparency.
        """
        approved: list[TradeSignal] = []
        rejected: list[tuple[TradeSignal, list[RiskCheckResult]]] = []

        for signal in signals:
            # HOLD signals pass through without checks – they don't
            # result in a trade.
            if signal.action == Action.HOLD:
                approved.append(signal)
                continue

            checks = self.check_trade(signal, portfolio, prices)
            failures = [c for c in checks if not c.passed]

            if failures:
                rejected.append((signal, checks))
                for fail in failures:
                    logger.warning(
                        "REJECTED %s %s – %s: %s",
                        signal.action.value.upper(),
                        signal.symbol,
                        fail.rule,
                        fail.message,
                    )
            else:
                approved.append(signal)
                logger.info(
                    "APPROVED %s %s (target_weight=%.2f%%, conviction=%s)",
                    signal.action.value.upper(),
                    signal.symbol,
                    signal.target_weight * 100,
                    signal.conviction.value,
                )

        # Enforce max_trades_per_session on the approved list.
        # Prioritise by conviction (HIGH > MEDIUM > LOW), preserving
        # original order within the same conviction tier.
        max_trades = self.limits.max_trades_per_session
        tradeable = [s for s in approved if s.action != Action.HOLD]
        holds = [s for s in approved if s.action == Action.HOLD]

        if len(tradeable) > max_trades:
            conviction_rank = {
                "high": 0,
                "medium": 1,
                "low": 2,
            }
            # Stable sort – preserves input order for equal conviction.
            tradeable.sort(
                key=lambda s: conviction_rank.get(s.conviction.value, 99)
            )
            trimmed = tradeable[max_trades:]
            tradeable = tradeable[:max_trades]

            for sig in trimmed:
                rejected.append(
                    (
                        sig,
                        [
                            RiskCheckResult(
                                passed=False,
                                rule="max_trades_per_session",
                                message=(
                                    f"Trade limit reached ({max_trades}). "
                                    f"Signal for {sig.symbol} dropped."
                                ),
                                current_value=float(len(tradeable) + len(trimmed)),
                                limit_value=float(max_trades),
                            )
                        ],
                    )
                )
                logger.warning(
                    "DROPPED %s %s – max trades per session (%d) exceeded.",
                    sig.action.value.upper(),
                    sig.symbol,
                    max_trades,
                )

        approved = holds + tradeable

        logger.info(
            "Risk filter: %d approved, %d rejected out of %d signal(s).",
            len(approved),
            len(rejected),
            len(signals),
        )

        return approved, rejected

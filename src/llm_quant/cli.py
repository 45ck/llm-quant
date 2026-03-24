"""CLI entry point for llm-quant paper trading system."""

import logging
import sys
from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="pq",
    help="llm-quant: LLM-powered paper trading system",
    no_args_is_help=True,
)
console = Console()

logger = logging.getLogger("llm_quant")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_config():
    from llm_quant.config import load_config
    return load_config()


def _get_db_path(config=None):
    if config is None:
        config = _get_config()
    return Path(config.general.db_path)


@app.command()
def init():
    """Create DuckDB schema and default configs."""
    _setup_logging()
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import init_schema
    from llm_quant.data.universe import sync_universe_to_db

    conn = init_schema(db_path)
    count = sync_universe_to_db(conn, config)
    conn.close()

    console.print(f"[green]OK[/green] Database initialized at [bold]{db_path}[/bold]")
    console.print(f"[green]OK[/green] Universe synced: {count} symbols")


@app.command()
def fetch():
    """Fetch/update market data from Yahoo Finance."""
    _setup_logging()
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection
    from llm_quant.data.universe import get_tradeable_symbols
    from llm_quant.data.fetcher import fetch_ohlcv
    from llm_quant.data.store import upsert_market_data
    from llm_quant.data.indicators import compute_indicators

    symbols = get_tradeable_symbols(config)
    console.print(f"Fetching data for {len(symbols)} symbols...")

    with console.status("[bold blue]Downloading from Yahoo Finance..."):
        df = fetch_ohlcv(
            symbols,
            lookback_days=config.data.lookback_days,
            timeout=config.data.fetch_timeout,
        )

    if df.is_empty():
        console.print("[red]FAIL[/red] No data fetched. Check your internet connection.")
        raise typer.Exit(1)

    console.print(f"  Fetched {len(df)} rows for {df['symbol'].n_unique()} symbols")

    with console.status("[bold blue]Computing indicators..."):
        df = compute_indicators(df)

    conn = get_connection(db_path)
    count = upsert_market_data(conn, df)
    conn.close()

    console.print(f"[green]OK[/green] Stored {count} rows in database")


@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show signals without executing trades"),
):
    """Full cycle: fetch -> indicators -> Claude -> trade -> log."""
    _setup_logging()
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection
    from llm_quant.data.universe import get_tradeable_symbols
    from llm_quant.data.fetcher import fetch_ohlcv
    from llm_quant.data.store import upsert_market_data, get_market_data
    from llm_quant.data.indicators import compute_indicators
    from llm_quant.brain.context import build_market_context
    from llm_quant.brain.engine import SignalEngine
    from llm_quant.trading.portfolio import Portfolio
    from llm_quant.trading.executor import execute_signals
    from llm_quant.trading.ledger import log_trades, save_portfolio_snapshot
    from llm_quant.risk.manager import RiskManager

    conn = get_connection(db_path)
    today = date.today()

    # Step 1: Fetch latest data
    symbols = get_tradeable_symbols(config)
    console.print(f"\n[bold]Step 1/5:[/bold] Fetching market data for {len(symbols)} symbols...")
    df = fetch_ohlcv(symbols, lookback_days=config.data.lookback_days)
    if not df.is_empty():
        df = compute_indicators(df)
        upsert_market_data(conn, df)
        console.print(f"  [green]OK[/green] Updated {df['symbol'].n_unique()} symbols")
    else:
        console.print("  [yellow]WARN[/yellow] No new data fetched, using existing DB data")

    # Step 2: Load portfolio
    console.print("[bold]Step 2/5:[/bold] Loading portfolio...")
    portfolio = Portfolio.from_db(conn, config.general.initial_capital)

    # Get latest prices for portfolio
    all_symbols = symbols
    import polars as pl
    latest = conn.execute(
        """
        SELECT symbol, close as price FROM market_data_daily
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM market_data_daily GROUP BY symbol
        )
        """
    ).pl()
    prices = dict(zip(latest["symbol"].to_list(), latest["price"].to_list()))
    portfolio.update_prices(prices)

    console.print(f"  NAV: ${portfolio.nav:,.2f} | Cash: ${portfolio.cash:,.2f} | Positions: {len(portfolio.positions)}")

    # Step 3: Build context and get Claude's signals
    console.print("[bold]Step 3/5:[/bold] Consulting Claude...")
    portfolio_state = portfolio.to_snapshot_dict()
    context = build_market_context(conn, portfolio_state, config)

    engine = SignalEngine(config)
    decision = engine.get_signals(context)

    # Display decision
    _display_decision(decision)

    if dry_run:
        console.print("\n[yellow]DRY RUN[/yellow] -- no trades executed.")
        conn.close()
        return

    # Step 4: Risk check and execute
    console.print("[bold]Step 4/5:[/bold] Risk check and execution...")
    risk_mgr = RiskManager(config)
    approved, rejected = risk_mgr.filter_signals(decision.signals, portfolio, prices)

    if rejected:
        console.print(f"  [yellow]WARN[/yellow] {len(rejected)} signals rejected by risk manager:")
        for sig, checks in rejected:
            failed = [c for c in checks if not c.passed]
            console.print(f"    {sig.symbol} {sig.action.value}: {', '.join(c.message for c in failed)}")

    if approved:
        executed = execute_signals(portfolio, approved, prices, portfolio.nav)
        console.print(f"  [green]OK[/green] Executed {len(executed)} trades")

        # Log decision
        decision_id = engine.log_decision(conn, decision)

        # Log trades
        trade_ids = log_trades(conn, executed, today, decision_id)
        console.print(f"  [green]OK[/green] Logged trade IDs: {trade_ids}")
    else:
        console.print("  No trades to execute.")

    # Step 5: Save snapshot
    console.print("[bold]Step 5/5:[/bold] Saving portfolio snapshot...")
    snap_id = save_portfolio_snapshot(conn, portfolio, today)
    console.print(f"  [green]OK[/green] Snapshot #{snap_id} saved")

    conn.close()
    console.print(f"\n[bold green]Done.[/bold green] NAV: ${portfolio.nav:,.2f}")


def _display_decision(decision):
    """Pretty-print a TradingDecision."""
    from llm_quant.brain.models import Action

    regime_colors = {"risk_on": "green", "risk_off": "red", "transition": "yellow"}
    color = regime_colors.get(decision.market_regime.value, "white")

    console.print(f"\n  Regime: [{color}]{decision.market_regime.value}[/{color}] "
                  f"(confidence: {decision.regime_confidence:.0%})")
    console.print(f"  Reasoning: {decision.regime_reasoning}")

    if decision.signals:
        table = Table(title="Trade Signals", show_lines=False)
        table.add_column("Symbol", style="bold")
        table.add_column("Action")
        table.add_column("Conviction")
        table.add_column("Target Wt")
        table.add_column("Stop Loss")
        table.add_column("Reasoning", max_width=50)

        action_colors = {"buy": "green", "sell": "red", "close": "red", "hold": "yellow"}
        for sig in decision.signals:
            a_color = action_colors.get(sig.action.value, "white")
            table.add_row(
                sig.symbol,
                f"[{a_color}]{sig.action.value.upper()}[/{a_color}]",
                sig.conviction.value,
                f"{sig.target_weight:.1%}",
                f"${sig.stop_loss:.2f}",
                sig.reasoning[:50],
            )
        console.print(table)

    if decision.portfolio_commentary:
        console.print(f"\n  Commentary: {decision.portfolio_commentary}")

    if decision.total_tokens > 0:
        console.print(f"  Tokens: {decision.total_tokens} | Cost: ${decision.cost_usd:.4f}")


@app.command()
def status():
    """Show current portfolio status and metrics."""
    _setup_logging("WARNING")
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection
    from llm_quant.trading.portfolio import Portfolio
    from llm_quant.trading.performance import compute_performance

    conn = get_connection(db_path)
    portfolio = Portfolio.from_db(conn, config.general.initial_capital)

    # Update with latest prices
    latest = conn.execute(
        """
        SELECT symbol, close as price FROM market_data_daily
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM market_data_daily GROUP BY symbol
        )
        """
    ).pl()
    if not latest.is_empty():
        prices = dict(zip(latest["symbol"].to_list(), latest["price"].to_list()))
        portfolio.update_prices(prices)

    # Portfolio summary
    console.print(Panel(
        f"[bold]NAV:[/bold] ${portfolio.nav:,.2f}  |  "
        f"[bold]Cash:[/bold] ${portfolio.cash:,.2f} ({portfolio.cash / portfolio.nav * 100:.1f}%)  |  "
        f"[bold]P&L:[/bold] ${portfolio.total_pnl:,.2f} ({portfolio.total_pnl / portfolio.initial_capital * 100:+.2f}%)",
        title="Portfolio Status",
    ))

    # Positions table
    if portfolio.positions:
        table = Table(title="Positions")
        table.add_column("Symbol", style="bold")
        table.add_column("Shares", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Mkt Value", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Stop", justify="right")

        for sym, pos in sorted(portfolio.positions.items()):
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            weight = pos.market_value / portfolio.nav * 100
            table.add_row(
                sym,
                f"{pos.shares:.0f}",
                f"${pos.avg_cost:.2f}",
                f"${pos.current_price:.2f}",
                f"${pos.market_value:,.2f}",
                f"[{pnl_color}]${pos.unrealized_pnl:,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pos.pnl_pct:+.1f}%[/{pnl_color}]",
                f"{weight:.1f}%",
                f"${pos.stop_loss:.2f}" if pos.stop_loss > 0 else "-",
            )
        console.print(table)
    else:
        console.print("[dim]No open positions.[/dim]")

    # Performance metrics
    metrics = compute_performance(conn, config.general.initial_capital)
    if metrics.get("total_trades", 0) > 0:
        console.print(Panel(
            f"[bold]Total Return:[/bold] {metrics['total_return']:.2%}  |  "
            f"[bold]Sharpe:[/bold] {metrics['sharpe_ratio']:.2f}  |  "
            f"[bold]Max DD:[/bold] {metrics['max_drawdown']:.2%}  |  "
            f"[bold]Win Rate:[/bold] {metrics['win_rate']:.0%}  |  "
            f"[bold]Trades:[/bold] {metrics['total_trades']}",
            title="Performance",
        ))

    conn.close()


@app.command()
def trades(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of recent trades to show"),
):
    """Show recent trades with LLM reasoning."""
    _setup_logging("WARNING")
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection
    from llm_quant.trading.ledger import get_recent_trades

    conn = get_connection(db_path)
    recent = get_recent_trades(conn, limit)
    conn.close()

    if not recent:
        console.print("[dim]No trades recorded yet.[/dim]")
        return

    table = Table(title=f"Recent Trades (last {limit})")
    table.add_column("ID", style="dim")
    table.add_column("Date")
    table.add_column("Symbol", style="bold")
    table.add_column("Action")
    table.add_column("Shares", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Notional", justify="right")
    table.add_column("Conviction")
    table.add_column("Reasoning", max_width=60)

    action_colors = {"buy": "green", "sell": "red", "close": "red", "hold": "yellow"}
    for t in recent:
        a_color = action_colors.get(t.get("action", "").lower(), "white")
        table.add_row(
            str(t.get("trade_id", "")),
            str(t.get("date", "")),
            t.get("symbol", ""),
            f"[{a_color}]{t.get('action', '').upper()}[/{a_color}]",
            f"{t.get('shares', 0):.0f}",
            f"${t.get('price', 0):.2f}",
            f"${t.get('notional', 0):,.2f}",
            t.get("conviction", "-"),
            (t.get("reasoning", "") or "-")[:60],
        )

    console.print(table)


@app.command()
def verify():
    """Verify the tamper-evident hash chain on the trade ledger."""
    _setup_logging("WARNING")
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection
    from llm_quant.db.integrity import verify_chain

    conn = get_connection(db_path)
    ok, last_id, message = verify_chain(conn)
    conn.close()

    if ok:
        console.print(f"[green]PASS[/green] {message}")
    else:
        console.print(f"[red]FAIL[/red] {message}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

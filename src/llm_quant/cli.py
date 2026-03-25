"""CLI entry point for llm-quant paper trading system."""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="pq",
    help="llm-quant: LLM-powered paper trading system",
    no_args_is_help=True,
)
console = Console()

logger = logging.getLogger("llm_quant")

# ---------------------------------------------------------------------------
# Pods sub-command group
# ---------------------------------------------------------------------------

pods_app = typer.Typer(name="pods", help="Manage trading pods")
app.add_typer(pods_app, name="pods")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _get_config():
    from llm_quant.config import load_config

    return load_config()


def _get_config_for_pod(pod_id: str = "default"):
    from llm_quant.config import load_config_for_pod

    return load_config_for_pod(pod_id)


def _get_db_path(config=None):
    if config is None:
        config = _get_config()
    return Path(config.general.db_path)


# ---------------------------------------------------------------------------
# init / fetch (unchanged — not pod-scoped)
# ---------------------------------------------------------------------------


@app.command()
def init():
    """Create DuckDB schema and default configs."""
    _setup_logging()
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.data.universe import sync_universe_to_db
    from llm_quant.db.schema import init_schema

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

    from llm_quant.data.fetcher import fetch_ohlcv
    from llm_quant.data.indicators import compute_indicators
    from llm_quant.data.store import upsert_market_data
    from llm_quant.data.universe import get_tradeable_symbols
    from llm_quant.db.schema import get_connection

    symbols = get_tradeable_symbols(config)
    console.print(f"Fetching data for {len(symbols)} symbols...")

    with console.status("[bold blue]Downloading from Yahoo Finance..."):
        df = fetch_ohlcv(
            symbols,
            lookback_days=config.data.lookback_days,
            timeout=config.data.fetch_timeout,
        )

    if df.is_empty():
        console.print(
            "[red]FAIL[/red] No data fetched. Check your internet connection."
        )
        raise typer.Exit(1)

    console.print(f"  Fetched {len(df)} rows for {df['symbol'].n_unique()} symbols")

    with console.status("[bold blue]Computing indicators..."):
        df = compute_indicators(df)

    conn = get_connection(db_path)
    count = upsert_market_data(conn, df)
    conn.close()

    console.print(f"[green]OK[/green] Stored {count} rows in database")


# --- run - pod-aware -------------------------------------------------------


def _run_single_pod(pod_id: str, *, dry_run: bool = False) -> None:
    """Execute a full trading cycle for a single pod."""
    from llm_quant.brain.context import build_market_context
    from llm_quant.brain.engine import SignalEngine
    from llm_quant.data.fetcher import fetch_ohlcv
    from llm_quant.data.indicators import compute_indicators
    from llm_quant.data.store import upsert_market_data
    from llm_quant.data.universe import get_tradeable_symbols
    from llm_quant.db.schema import get_connection
    from llm_quant.risk.manager import RiskManager
    from llm_quant.trading.executor import execute_signals
    from llm_quant.trading.ledger import log_trades, save_portfolio_snapshot
    from llm_quant.trading.portfolio import Portfolio

    config = _get_config_for_pod(pod_id)
    db_path = _get_db_path(config)

    conn = get_connection(db_path)
    today = datetime.now(tz=UTC).date()

    # Step 1: Fetch latest data
    symbols = get_tradeable_symbols(config)
    console.print(
        f"\n[bold]Step 1/5:[/bold] Fetching market data for {len(symbols)} symbols..."
    )
    df = fetch_ohlcv(symbols, lookback_days=config.data.lookback_days)
    if not df.is_empty():
        df = compute_indicators(df)
        upsert_market_data(conn, df)
        console.print(f"  [green]OK[/green] Updated {df['symbol'].n_unique()} symbols")
    else:
        console.print(
            "  [yellow]WARN[/yellow] No new data fetched, using existing DB data"
        )

    # Step 2: Load portfolio
    console.print("[bold]Step 2/5:[/bold] Loading portfolio...")
    portfolio = Portfolio.from_db(conn, config.general.initial_capital, pod_id=pod_id)

    # Get latest prices for portfolio
    latest = conn.execute("""
        SELECT symbol, close as price FROM market_data_daily
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM market_data_daily GROUP BY symbol
        )
        """).pl()
    prices = dict(
        zip(
            latest["symbol"].to_list(),
            latest["price"].to_list(),
            strict=True,
        )
    )
    portfolio.update_prices(prices)

    console.print(
        f"  NAV: ${portfolio.nav:,.2f} | Cash: ${portfolio.cash:,.2f}"
        f" | Positions: {len(portfolio.positions)}"
    )

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
        console.print(
            f"  [yellow]WARN[/yellow] {len(rejected)} signals rejected by risk manager:"
        )
        for sig, checks in rejected:
            failed = [c for c in checks if not c.passed]
            reasons = ", ".join(c.message for c in failed)
            console.print(f"    {sig.symbol} {sig.action.value}: {reasons}")

    if approved:
        executed = execute_signals(portfolio, approved, prices, portfolio.nav)
        console.print(f"  [green]OK[/green] Executed {len(executed)} trades")

        # Log decision
        decision_id = engine.log_decision(conn, decision)

        # Log trades
        trade_ids = log_trades(conn, executed, today, decision_id, pod_id=pod_id)
        console.print(f"  [green]OK[/green] Logged trade IDs: {trade_ids}")
    else:
        console.print("  No trades to execute.")

    # Step 5: Save snapshot
    console.print("[bold]Step 5/5:[/bold] Saving portfolio snapshot...")
    snap_id = save_portfolio_snapshot(conn, portfolio, today, pod_id=pod_id)
    console.print(f"  [green]OK[/green] Snapshot #{snap_id} saved")

    conn.close()
    console.print(f"\n[bold green]Done.[/bold green] NAV: ${portfolio.nav:,.2f}")


@app.command()
def run(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show signals without executing trades",
    ),
    pod: str = typer.Option("default", "--pod", "-p", help="Pod to operate on"),
    all_pods: bool = typer.Option(
        False, "--all-pods", help="Run all active pods sequentially"
    ),
):
    """Full cycle: fetch -> indicators -> Claude -> trade -> log."""
    _setup_logging()

    if all_pods:
        config = _get_config()
        db_path = _get_db_path(config)

        from llm_quant.db.schema import get_connection

        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                "SELECT pod_id FROM pods WHERE status = 'active' ORDER BY pod_id"
            ).fetchall()
        except (duckdb.CatalogException, duckdb.BinderException):
            console.print(
                "[red]FAIL[/red] Could not query pods table. "
                "Run [bold]pq init[/bold] first."
            )
            conn.close()
            raise typer.Exit(1) from None
        conn.close()

        if not rows:
            console.print("[yellow]No active pods found.[/yellow]")
            return

        for (pid,) in rows:
            console.rule(f"[bold]Pod: {pid}")
            _run_single_pod(pid, dry_run=dry_run)
        return

    _run_single_pod(pod, dry_run=dry_run)


def _display_decision(decision):
    """Pretty-print a TradingDecision."""

    regime_colors = {"risk_on": "green", "risk_off": "red", "transition": "yellow"}
    color = regime_colors.get(decision.market_regime.value, "white")

    console.print(
        f"\n  Regime: [{color}]{decision.market_regime.value}[/{color}] "
        f"(confidence: {decision.regime_confidence:.0%})"
    )
    console.print(f"  Reasoning: {decision.regime_reasoning}")

    if decision.signals:
        table = Table(title="Trade Signals", show_lines=False)
        table.add_column("Symbol", style="bold")
        table.add_column("Action")
        table.add_column("Conviction")
        table.add_column("Target Wt")
        table.add_column("Stop Loss")
        table.add_column("Reasoning", max_width=50)

        action_colors = {
            "buy": "green",
            "sell": "red",
            "close": "red",
            "hold": "yellow",
        }
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
        console.print(
            f"  Tokens: {decision.total_tokens} | Cost: ${decision.cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# status (pod-aware, with --all flag)
# ---------------------------------------------------------------------------


def _show_all_pods_dashboard(conn) -> None:
    """Show comparative dashboard across all pods."""
    try:
        rows = conn.execute("""
            SELECT
                ps.pod_id,
                ps.nav,
                ps.cash,
                ps.total_pnl,
                ps.gross_exposure,
                (SELECT COUNT(*) FROM positions p
                 WHERE p.snapshot_id = ps.snapshot_id) as positions
            FROM portfolio_snapshots ps
            INNER JOIN (
                SELECT pod_id, MAX(snapshot_id) as max_id
                FROM portfolio_snapshots
                GROUP BY pod_id
            ) latest ON ps.pod_id = latest.pod_id AND ps.snapshot_id = latest.max_id
            ORDER BY ps.pod_id
        """).fetchall()
    except (duckdb.CatalogException, duckdb.BinderException):
        console.print("[dim]No snapshot data available.[/dim]")
        return

    if not rows:
        console.print("[dim]No pod snapshots found.[/dim]")
        return

    table = Table(title="All Pods Dashboard")
    table.add_column("Pod ID", style="bold")
    table.add_column("NAV", justify="right")
    table.add_column("Cash %", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Positions", justify="right")
    table.add_column("Gross Exposure", justify="right")

    for pod_id, raw_nav, raw_cash, raw_pnl, raw_exposure, positions in rows:
        nav_f = float(raw_nav) if raw_nav else 0.0
        cash_f = float(raw_cash) if raw_cash else 0.0
        pnl_f = float(raw_pnl) if raw_pnl else 0.0
        exp_f = float(raw_exposure) if raw_exposure else 0.0
        cash_pct = (cash_f / nav_f * 100) if nav_f > 0 else 0.0
        pnl_color = "green" if pnl_f >= 0 else "red"
        table.add_row(
            pod_id,
            f"${nav_f:,.2f}",
            f"{cash_pct:.1f}%",
            f"[{pnl_color}]${pnl_f:,.2f}[/{pnl_color}]",
            str(positions or 0),
            f"${exp_f:,.2f}",
        )

    console.print(table)


@app.command()
def status(
    pod: str = typer.Option("default", "--pod", "-p", help="Pod to operate on"),
    all: bool = typer.Option(  # noqa: A002
        False, "--all", "-a", help="Show comparative dashboard across all pods"
    ),
):
    """Show current portfolio status and metrics."""
    _setup_logging("WARNING")

    from llm_quant.db.schema import get_connection
    from llm_quant.trading.performance import compute_performance
    from llm_quant.trading.portfolio import Portfolio

    config = _get_config_for_pod(pod)
    db_path = _get_db_path(config)
    conn = get_connection(db_path)

    if all:
        _show_all_pods_dashboard(conn)
        conn.close()
        return

    portfolio = Portfolio.from_db(conn, config.general.initial_capital, pod_id=pod)

    # Update with latest prices
    latest = conn.execute("""
        SELECT symbol, close as price FROM market_data_daily
        WHERE (symbol, date) IN (
            SELECT symbol, MAX(date) FROM market_data_daily GROUP BY symbol
        )
        """).pl()
    if not latest.is_empty():
        prices = dict(
            zip(latest["symbol"].to_list(), latest["price"].to_list(), strict=True)
        )
        portfolio.update_prices(prices)

    # Portfolio summary
    cash_pct = portfolio.cash / portfolio.nav * 100
    pnl_pct = portfolio.total_pnl / portfolio.initial_capital * 100
    title = f"Portfolio Status (pod: {pod})" if pod != "default" else "Portfolio Status"
    console.print(
        Panel(
            f"[bold]NAV:[/bold] ${portfolio.nav:,.2f}  |  "
            f"[bold]Cash:[/bold] ${portfolio.cash:,.2f} ({cash_pct:.1f}%)  |  "
            f"[bold]P&L:[/bold] ${portfolio.total_pnl:,.2f} ({pnl_pct:+.2f}%)",
            title=title,
        )
    )

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
        console.print(
            Panel(
                f"[bold]Total Return:[/bold] {metrics['total_return']:.2%}  |  "
                f"[bold]Sharpe:[/bold] {metrics['sharpe_ratio']:.2f}  |  "
                f"[bold]Max DD:[/bold] {metrics['max_drawdown']:.2%}  |  "
                f"[bold]Win Rate:[/bold] {metrics['win_rate']:.0%}  |  "
                f"[bold]Trades:[/bold] {metrics['total_trades']}",
                title="Performance",
            )
        )

    conn.close()


# --- trades - pod-aware ----------------------------------------------------


@app.command()
def trades(
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Number of recent trades to show",
    ),
    pod: str = typer.Option("default", "--pod", "-p", help="Pod to operate on"),
):
    """Show recent trades with LLM reasoning."""
    _setup_logging("WARNING")
    config = _get_config_for_pod(pod)
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection
    from llm_quant.trading.ledger import get_recent_trades

    conn = get_connection(db_path)
    recent = get_recent_trades(conn, limit, pod_id=pod)
    conn.close()

    if not recent:
        console.print("[dim]No trades recorded yet.[/dim]")
        return

    title = (
        f"Recent Trades (last {limit}, pod: {pod})"
        if pod != "default"
        else f"Recent Trades (last {limit})"
    )
    table = Table(title=title)
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


# --- verify - pod-aware ----------------------------------------------------


@app.command()
def verify(
    pod: str = typer.Option("default", "--pod", "-p", help="Pod to operate on"),
):
    """Verify the tamper-evident hash chain on the trade ledger."""
    _setup_logging("WARNING")
    config = _get_config_for_pod(pod)
    db_path = _get_db_path(config)

    from llm_quant.db.integrity import verify_chain
    from llm_quant.db.schema import get_connection

    conn = get_connection(db_path)
    ok, _last_id, message = verify_chain(conn)
    conn.close()

    if ok:
        console.print(f"[green]PASS[/green] {message}")
    else:
        console.print(f"[red]FAIL[/red] {message}")
        raise typer.Exit(1)


# --- report ----------------------------------------------------------------


@app.command()
def report(
    report_type: str = typer.Argument(
        "daily", help="Report type: daily, weekly, or monthly"
    ),
    date: str = typer.Option(None, "--date", "-d", help="Report date (YYYY-MM-DD)"),
):
    """Generate a performance report."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/generate_report.py", report_type]
    if date:
        cmd.extend(["--date", date])

    env = {**__import__("os").environ, "PYTHONPATH": "src"}
    result = subprocess.run(cmd, env=env, check=False)  # noqa: S603
    if result.returncode != 0:
        console.print(
            f"[red]FAIL[/red] Report generation exited with code {result.returncode}"
        )
        raise typer.Exit(result.returncode)
    console.print(f"[green]OK[/green] {report_type.capitalize()} report generated")


# ---------------------------------------------------------------------------
# pods sub-commands
# ---------------------------------------------------------------------------


_POD_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


@pods_app.command("list")
def pods_list():
    """List all registered pods."""
    _setup_logging("WARNING")
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection

    conn = get_connection(db_path)

    try:
        rows = conn.execute(
            "SELECT pod_id, display_name, strategy_type, initial_capital, "
            "status, created_at FROM pods ORDER BY pod_id"
        ).fetchall()
    except (duckdb.CatalogException, duckdb.BinderException):
        console.print(
            "[yellow]Pods table not found.[/yellow] "
            "Run [bold]pq init[/bold] to create it."
        )
        conn.close()
        return

    conn.close()

    if not rows:
        console.print(
            "[dim]No pods registered. Use [bold]pq pods create[/bold] to add one.[/dim]"
        )
        return

    table = Table(title="Trading Pods")
    table.add_column("Pod ID", style="bold")
    table.add_column("Display Name")
    table.add_column("Strategy")
    table.add_column("Initial Capital", justify="right")
    table.add_column("Status")
    table.add_column("Created")

    status_colors = {"active": "green", "paused": "yellow", "retired": "red"}
    for (
        pod_id,
        display_name,
        strategy_type,
        initial_capital,
        pod_status,
        created_at,
    ) in rows:
        s_color = status_colors.get(pod_status, "white")
        table.add_row(
            pod_id,
            display_name or pod_id,
            strategy_type or "-",
            f"${float(initial_capital):,.2f}" if initial_capital else "-",
            f"[{s_color}]{pod_status}[/{s_color}]",
            str(created_at)[:19] if created_at else "-",
        )

    console.print(table)


@pods_app.command("create")
def pods_create(
    pod_id: str = typer.Argument(..., help="Unique pod identifier (lowercase slug)"),
    name: str = typer.Option(None, "--name", "-n", help="Display name"),
    strategy: str = typer.Option("custom", "--strategy", "-s", help="Strategy type"),
    capital: float = typer.Option(100_000.0, "--capital", "-c", help="Initial capital"),
):
    """Register a new trading pod."""
    _setup_logging("WARNING")

    # Validate pod_id format
    if not _POD_ID_PATTERN.match(pod_id):
        console.print(
            f"[red]FAIL[/red] Invalid pod_id '{pod_id}'. "
            "Must be a lowercase slug (letters, digits, hyphens, underscores), "
            "starting with a letter, max 63 chars."
        )
        raise typer.Exit(1)

    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection

    conn = get_connection(db_path)

    try:
        conn.execute(
            "INSERT INTO pods "
            "(pod_id, display_name, strategy_type, "
            "initial_capital, status, created_at) "
            "VALUES (?, ?, ?, ?, 'active', NOW())",
            [pod_id, name or pod_id, strategy, capital],
        )
        console.print(
            f"[green]OK[/green] Pod [bold]{pod_id}[/bold] created "
            f"(strategy={strategy}, capital=${capital:,.2f})"
        )
    except duckdb.ConstraintException:
        console.print(f"[red]FAIL[/red] Pod '{pod_id}' already exists.")
        raise typer.Exit(1) from None
    except duckdb.Error as e:
        console.print(f"[red]FAIL[/red] Could not create pod: {e}")
        raise typer.Exit(1) from e
    finally:
        conn.close()


@pods_app.command("delete")
def pods_delete(
    pod_id: str = typer.Argument(..., help="Pod to remove"),
    force: bool = typer.Option(
        False, "--force", help="Hard delete (default: deactivate)"
    ),
):
    """Deactivate or delete a pod."""
    _setup_logging("WARNING")
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection

    conn = get_connection(db_path)

    try:
        if force:
            conn.execute("DELETE FROM pods WHERE pod_id = ?", [pod_id])
            # DuckDB returns row count via changes
            console.print(
                f"[green]OK[/green] Pod [bold]{pod_id}[/bold] permanently deleted."
            )
        else:
            conn.execute(
                "UPDATE pods SET status = 'retired', "
                "retired_at = NOW() WHERE pod_id = ?",
                [pod_id],
            )
            console.print(
                f"[green]OK[/green] Pod [bold]{pod_id}[/bold] "
                "deactivated (status=retired)."
            )
    except duckdb.Error as e:
        console.print(f"[red]FAIL[/red] Could not delete/deactivate pod: {e}")
        raise typer.Exit(1) from e
    finally:
        conn.close()


@pods_app.command("status")
def pods_status():
    """Show comparative dashboard across all pods."""
    _setup_logging("WARNING")
    config = _get_config()
    db_path = _get_db_path(config)

    from llm_quant.db.schema import get_connection

    conn = get_connection(db_path)
    _show_all_pods_dashboard(conn)
    conn.close()


if __name__ == "__main__":
    app()

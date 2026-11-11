"""Offline-first research CLI, intentionally independent of the trading CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from . import __version__
from .data import Contract, Snapshot, aligned_returns, load_snapshot, require
from .models import Simulation, risk_report
from .projects import ProjectCase
from .providers import fetch_fred, fetch_yahoo
from .validation import walk_forward

LOCAL = Path("research-local/decision-lab")


def code_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_report(report: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    report["runtime"] = {
        "decision_lab": __version__, "python": platform.python_version(),
        "numpy": np.__version__, "code_sha256": code_digest(),
    }
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    sys.stdout.write(f"Research output: {target}\nStatus: {report['status']}\n")


def demo(target: Path) -> None:
    """Deterministic fictitious observations exercise the full offline pipeline."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.003, 0.025, size=(120, 3))
    dates = tuple(f"{2010 + month // 12}-{month % 12 + 1:02d}-01" for month in range(121))
    levels = np.vstack([np.ones(3), np.cumprod(1 + returns, axis=0)]) * 100
    snapshots = []
    for index in range(3):
        values = tuple(float(value) for value in levels[:, index])
        raw = json.dumps(values).encode()
        contract = Contract(
            series_id=f"SYNTHETIC_{index}", provider="Deterministic test generator",
            source_url="https://example.invalid/synthetic", terms_url="https://example.invalid/synthetic",
            geography="Fictitious", currency="AUD", frequency="M", basis="total_return",
            retrieved_at="2020-01-01T00:00:00+00:00", vintage="synthetic",
            evidence="synthetic", raw_sha256=hashlib.sha256(raw).hexdigest(),
        )
        snapshots.append(Snapshot(contract, dates, values))
    for snapshot in snapshots:
        snapshot.check_as_of(date(2020, 1, 1), 0)
    config = Simulation(paths=2000)
    report = risk_report(aligned_returns(snapshots), np.array([0.5, 0.3, 0.2]), config)
    report["status"] = "SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE"
    report["sources"] = [asdict(snapshot.contract) for snapshot in snapshots]
    report["snapshot_sha256"] = [snapshot.digest() for snapshot in snapshots]
    write_report(report, target)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("demo", help="Run synthetic offline smoke test")
    sample.add_argument("--out", type=Path, default=LOCAL / "demo.json")
    fetch = commands.add_parser("fetch", help="Explicit opt-in source download")
    fetch.add_argument("provider", choices=("fred", "yahoo"))
    fetch.add_argument("symbol")
    fetch.add_argument("--online", action="store_true", help="Required for any network call")
    fetch.add_argument("--start", default="2010-01-01")
    fetch.add_argument("--end", help="Exclusive end for Yahoo, ISO date")
    fetch.add_argument("--directory", type=Path, default=LOCAL / "snapshots")
    project = commands.add_parser("project", help="Evaluate supplied assumptions, not win odds")
    project.add_argument("input", type=Path)
    project.add_argument("--out", type=Path, default=LOCAL / "project.json")
    for name in ("risk", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--snapshot", type=Path, action="append", required=True)
        command.add_argument("--weights", type=float, nargs="+", required=True)
        command.add_argument("--as-of", type=date.fromisoformat, required=True)
        command.add_argument("--max-age-days", type=int, required=True)
        command.add_argument("--horizon", type=int, required=True, help="Native periods, NOT always days")
        command.add_argument("--paths", type=int, default=10000)
        command.add_argument("--block-length", type=int, default=3)
        command.add_argument("--seed", type=int, default=20261111)
        command.add_argument("--indicator", action="store_true", help="Single non-investable index")
        command.add_argument("--out", type=Path, default=LOCAL / f"{name}.json")
        if name == "validate":
            command.add_argument("--min-train", type=int, required=True)
    return root


def run_model(args: argparse.Namespace) -> None:
    snapshots = [load_snapshot(path) for path in args.snapshot]
    for snapshot in snapshots:
        snapshot.check_as_of(args.as_of, args.max_age_days)
    returns = aligned_returns(snapshots, portfolio=not args.indicator)
    config = Simulation(args.horizon, args.paths, args.block_length, args.seed)
    weights = np.array(args.weights)
    if args.command == "risk":
        report = risk_report(returns, weights, config)
    else:
        report = walk_forward(returns, weights, config, min_train=args.min_train)
    report["as_of"] = args.as_of.isoformat()
    report["config"] = asdict(config)
    report["native_frequency"] = snapshots[0].contract.frequency
    report["is_investable_portfolio"] = not args.indicator
    report["sources"] = [asdict(snapshot.contract) for snapshot in snapshots]
    report["snapshot_sha256"] = [snapshot.digest() for snapshot in snapshots]
    report["data_last_observation"] = snapshots[0].dates[-1]
    if snapshots[0].contract.evidence == "synthetic":
        report["status"] = "SYNTHETIC_DEMO_NOT_MARKET_EVIDENCE"
    write_report(report, args.out)


def main() -> None:
    root = parser()
    args = root.parse_args()
    try:
        if args.command == "demo":
            demo(args.out)
        elif args.command == "project":
            case = ProjectCase(**json.loads(args.input.read_text(encoding="utf-8")))
            write_report(case.evaluate(), args.out)
        elif args.command == "fetch":
            require(args.online, "Network disabled: explicitly add --online")
            if args.provider == "fred":
                output = fetch_fred(args.symbol, args.directory)
            else:
                require(args.end is not None, "Yahoo requires explicit --end")
                output = fetch_yahoo(args.symbol, args.start, args.end, args.directory)
            sys.stdout.write(f"Snapshot: {output}\n")
        else:
            run_model(args)
    except (ValueError, OSError, ImportError, KeyError) as error:
        root.exit(2, f"Validation/source failure: {error}\nNo fallback data was generated.\n")


if __name__ == "__main__":
    main()

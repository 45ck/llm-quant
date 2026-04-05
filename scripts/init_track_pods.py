#!/usr/bin/env python3
"""Initialize track pods in DuckDB for multi-track portfolio system.

Creates pods for:
- track-a: Defensive Alpha ($100k, 29 strategies)
- track-b: Aggressive Alpha ($100k, 3 strategies)
- track-d: Sprint Alpha ($100k, 1 strategy)
- discretionary: Manual Macro ($100k, existing default pod alias)

Usage:
    cd E:/llm-quant && PYTHONPATH=src python scripts/init_track_pods.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_quant.db.schema import init_schema


def main():
    db_path = Path(__file__).resolve().parent.parent / "data" / "portfolio.duckdb"
    conn = init_schema(str(db_path))

    # Check existing pods
    existing = conn.execute("SELECT pod_id FROM pods").fetchall()
    existing_ids = {r[0] for r in existing}
    print(f"Existing pods: {existing_ids}")

    # Define track pods
    pods_to_create = [
        (
            "track-a",
            "Track A -- Defensive Alpha",
            "multi_strategy",
            100000.0,
            "active",
            "config/track-assignments.yaml",
        ),
        (
            "track-b",
            "Track B -- Aggressive Alpha",
            "multi_strategy",
            100000.0,
            "active",
            "config/track-assignments.yaml",
        ),
        (
            "track-d",
            "Track D -- Sprint Alpha",
            "leveraged_lead_lag",
            100000.0,
            "active",
            "config/track-assignments.yaml",
        ),
        (
            "discretionary",
            "Discretionary -- Manual Macro",
            "regime_momentum",
            100000.0,
            "active",
            None,
        ),
    ]

    created = 0
    for (
        pod_id,
        display_name,
        strategy_type,
        capital,
        status,
        config_path,
    ) in pods_to_create:
        if pod_id in existing_ids:
            print(f"  Pod '{pod_id}' already exists, skipping")
            continue
        conn.execute(
            """INSERT INTO pods
               (pod_id, display_name, strategy_type, initial_capital, status, config_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [pod_id, display_name, strategy_type, capital, status, config_path],
        )
        print(f"  Created pod: '{pod_id}'")
        created += 1

    conn.commit()

    # Verify
    print()
    print("=== All pods after creation ===")
    rows = conn.execute(
        """SELECT pod_id, display_name, strategy_type, initial_capital, status, config_path
           FROM pods ORDER BY pod_id"""
    ).fetchall()
    for r in rows:
        cfg = r[5] or "(none)"
        print(
            f"  {r[0]:<15} | {r[1]:<35} | {r[2]:<20} | "
            f"${r[3]:>10,.0f} | {r[4]:<8} | {cfg}"
        )

    conn.close()
    print()
    print(f"Done. Created {created} new pod(s).")


if __name__ == "__main__":
    main()

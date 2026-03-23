"""Phase 1 pipeline: Load raw events into DuckDB and run funnel staging queries.

Usage:
    python -m src.data.pipeline
    python -m src.data.pipeline --csv-pattern "data/raw/*.csv"
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.sql.funnel_queries import (
    COHORT_RETENTION,
    FUNNEL_CONVERSION_RATES,
    FUNNEL_STAGING,
    LOAD_RAW_EVENTS,
    SESSION_FUNNEL_SUMMARY,
    SESSIONIZE_EVENTS,
    STAGE_DROPOFF,
    TIME_PATTERNS,
    USER_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "funnel.duckdb"


def run_pipeline(csv_pattern=None):
    if csv_pattern is None:
        csv_pattern = str(PROJECT_ROOT / "data" / "raw" / "*.csv")

    print(f"Connecting to DuckDB at {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))

    steps = [
        ("Loading raw events", LOAD_RAW_EVENTS.format(csv_pattern=csv_pattern)),
        ("Sessionizing events (30-min gap)", SESSIONIZE_EVENTS),
        ("Building funnel stages", FUNNEL_STAGING),
        ("Summarizing sessions", SESSION_FUNNEL_SUMMARY),
        ("Building user features", USER_FEATURES),
    ]

    for name, query in steps:
        print(f"  → {name}...")
        con.execute(query)

    # Print funnel summary
    print("\n=== Funnel Conversion Rates ===")
    df = con.execute(FUNNEL_CONVERSION_RATES).fetchdf()
    print(df.to_string(index=False))

    print("\n=== Stage-to-Stage Drop-off ===")
    df = con.execute(STAGE_DROPOFF).fetchdf()
    print(df.to_string(index=False))

    # Save processed tables to parquet for downstream use
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for table in ["session_funnel_summary", "user_features", "funnel_events"]:
        out_path = processed_dir / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out_path}' (FORMAT PARQUET)")
        print(f"  Saved {table} → {out_path.name}")

    row_counts = {}
    for table in ["raw_events", "sessionized_events", "funnel_events", "session_funnel_summary", "user_features"]:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        row_counts[table] = count

    print("\n=== Table Row Counts ===")
    for table, count in row_counts.items():
        print(f"  {table}: {count:,}")

    con.close()
    print("\nPipeline complete.")
    return row_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-pattern", default=None, help="Glob pattern for raw CSV files")
    args = parser.parse_args()
    run_pipeline(args.csv_pattern)

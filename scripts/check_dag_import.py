#!/usr/bin/env python3
"""Import the project DAG in an isolated Airflow environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airflow.models import DagBag


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dag-dir", type=Path, default=PROJECT_ROOT / "dags")
    args = parser.parse_args()
    bag = DagBag(dag_folder=str(args.dag_dir.resolve()))
    if bag.import_errors:
        raise RuntimeError(f"DAG import errors: {bag.import_errors}")
    # ``DagBag.get_dag`` consults the metadata database in Airflow 3.  This
    # CI check intentionally validates parsing only, so use the in-memory map.
    dag = bag.dags.get("china_auto_market_monthly")
    if dag is None:
        raise RuntimeError("china_auto_market_monthly was not discovered")
    expected_tasks = {
        "begin_pipeline",
        "ingest_raw_snapshots",
        "validate_raw",
        "rebuild_staging",
        "rebuild_core_marts",
        "validate_core_marts",
        "validate_forecast_model",
        "stage_publication_manifest",
        "publish_dashboard",
        "finish_pipeline",
    }
    actual_tasks = set(dag.task_ids)
    if actual_tasks != expected_tasks:
        raise RuntimeError(
            f"DAG task contract changed: expected {sorted(expected_tasks)}, "
            f"received {sorted(actual_tasks)}"
        )
    print(
        json.dumps(
            {"dag_id": dag.dag_id, "tasks": sorted(actual_tasks), "import_errors": 0},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

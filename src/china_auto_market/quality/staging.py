"""Rebuild the standardized staging snapshot behind critical quality gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from china_auto_market.ingestion.mysql_cli import run_transaction, sql_literal
from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.warehouse.schema import query


STAGING_SQL = PROJECT_ROOT / "sql" / "staging" / "10_rebuild_staging.sql"
QUALITY_SQL = PROJECT_ROOT / "sql" / "quality" / "10_staging_quality.sql"

EXPECTED = {
    "mapping_rows": 2_138,
    "sales_mapping_rows": 1_017,
    "config_sales_mapping_rows": 371,
    "config_only_mapping_rows": 395,
    "review_mapping_rows": 355,
    "sales_rows": 54_918,
    "sales_series": 1_017,
    "sales_months_per_series": 54,
    "repair_rows": 65,
    "repair_delta": 1_357_558,
    "config_rows": 2_084,
    "annual_repair_rows": 5,
    "annual_repair_delta": 690_980,
    "review_rows": 24_284,
    "eligible_reviews": 24_175,
    "after_locked_test_reviews": 594,
    "review_feature_rows": 241_750,
    "review_feature_identities": 24_175,
}


@dataclass(frozen=True)
class StagingResult:
    pipeline_run_id: int
    status: str
    row_counts: dict[str, int]
    quality_results: int
    nonpassing_quality_results: int
    critical_failures: int


def _succeeded_batch_id(login_path: str, dataset_name: str) -> int:
    output = query(
        login_path,
        "SELECT batch_id FROM auto_ops.ingestion_batches "
        f"WHERE BINARY dataset_name=0x{dataset_name.encode('utf-8').hex()} "
        "AND status='succeeded' ORDER BY batch_id DESC LIMIT 1;",
    ).strip()
    if not output:
        raise RuntimeError(f"Missing succeeded full batch: {dataset_name}")
    return int(output)


def _start_pipeline(login_path: str) -> int:
    sql = (
        "INSERT INTO auto_ops.pipeline_runs "
        "(pipeline_name,trigger_type,status,code_version,parameters_json) VALUES "
        "('rebuild_staging','manual','started','de4-v1',JSON_OBJECT('scope','full_snapshot')); "
        "SELECT LAST_INSERT_ID();"
    )
    output = query(login_path, sql).strip().splitlines()
    return int(output[-1])


def _render(path: Path, values: dict[str, int]) -> str:
    rendered = path.read_text(encoding="utf-8")
    for name, value in values.items():
        rendered = rendered.replace(f"__{name.upper()}__", str(int(value)))
    if "__" in rendered:
        unresolved = sorted({part.split("__", 1)[0] for part in rendered.split("__")[1::2]})
        raise ValueError(f"Unresolved SQL placeholders in {path}: {unresolved}")
    return rendered


def _mark_failed(login_path: str, pipeline_run_id: int, error: Exception) -> None:
    message = str(error)[:4000]
    sql = (
        "UPDATE auto_ops.pipeline_runs SET status='failed',completed_at=NOW(6),"
        f"error_message={sql_literal(message)} WHERE pipeline_run_id={pipeline_run_id}; "
        "INSERT INTO auto_ops.data_quality_results "
        "(pipeline_run_id,dataset_name,rule_name,rule_version,severity,passed,"
        "observed_value,expected_value,affected_rows,details_json) VALUES ("
        f"{pipeline_run_id},'staging_snapshot','staging_transaction_gate','de4-v1','critical',"
        f"FALSE,{sql_literal(message)},'all critical rules pass',1,NULL);"
    )
    query(login_path, sql)


def rebuild_staging(
    login_path: str,
    *,
    expected_overrides: dict[str, int] | None = None,
) -> StagingResult:
    """Replace staging atomically and reject the snapshot if a critical rule fails."""
    expected = dict(EXPECTED)
    for name, value in (expected_overrides or {}).items():
        if name not in expected:
            raise KeyError(f"Unknown expected value: {name}")
        expected[name] = value

    pipeline_run_id = _start_pipeline(login_path)
    values = {
        "sales_batch_id": _succeeded_batch_id(login_path, "sales"),
        "config_batch_id": _succeeded_batch_id(login_path, "config"),
        "reviews_batch_id": _succeeded_batch_id(login_path, "reviews"),
        "labels_batch_id": _succeeded_batch_id(login_path, "review_labels"),
        "corrections_batch_id": _succeeded_batch_id(login_path, "sales_corrections"),
        "annual_corrections_batch_id": _succeeded_batch_id(
            login_path, "annual_sales_corrections"
        ),
        "pipeline_run_id": pipeline_run_id,
        **{f"expected_{name}": value for name, value in expected.items()},
    }
    statements = [_render(STAGING_SQL, values), _render(QUALITY_SQL, values)]
    try:
        run_transaction(login_path, statements)
    except Exception as error:
        _mark_failed(login_path, pipeline_run_id, error)
        raise

    count_output = query(
        login_path,
        "SELECT "
        "(SELECT COUNT(*) FROM auto_staging.series_name_mappings),"
        "(SELECT COUNT(*) FROM auto_staging.stg_monthly_sales),"
        "(SELECT COUNT(*) FROM auto_staging.stg_vehicle_config),"
        "(SELECT COUNT(*) FROM auto_staging.stg_reviews),"
        "(SELECT COUNT(*) FROM auto_staging.stg_review_features);",
    ).strip().split("\t")
    quality_output = query(
        login_path,
        "SELECT COUNT(*),SUM(NOT passed),SUM(severity='critical' AND NOT passed) "
        "FROM auto_ops.data_quality_results "
        f"WHERE pipeline_run_id={pipeline_run_id};",
    ).strip().split("\t")
    return StagingResult(
        pipeline_run_id=pipeline_run_id,
        status="succeeded",
        row_counts=dict(
            zip(
                ["mappings", "monthly_sales", "vehicle_config", "reviews", "review_features"],
                map(int, count_output),
                strict=True,
            )
        ),
        quality_results=int(quality_output[0]),
        nonpassing_quality_results=int(quality_output[1]),
        critical_failures=int(quality_output[2]),
    )


def result_dict(result: StagingResult) -> dict[str, Any]:
    return asdict(result)

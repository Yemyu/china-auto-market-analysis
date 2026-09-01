"""Build the business marts and record their quality results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from china_auto_market.features.configuration import CFG_COLS
from china_auto_market.ingestion.mysql_cli import insert_statements, normalize_value, run_transaction, sql_literal
from china_auto_market.marts.forecast import build_forecast_panel
from china_auto_market.marts.user_needs import build_latest_snapshot, assert_latest_window_parity
from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.warehouse.schema import query
from china_auto_market.warehouse.sources import (
    load_raw_configuration,
    load_standard_review_labels,
    load_standard_sales,
)


MART_SQL = PROJECT_ROOT / "sql" / "marts" / "10_rebuild_core_marts.sql"
FINALIZE_SQL = PROJECT_ROOT / "sql" / "marts" / "20_finalize_core_marts.sql"
QUALITY_SQL = PROJECT_ROOT / "sql" / "quality" / "20_core_mart_quality.sql"
COHORT = PROJECT_ROOT / "data" / "reviews" / "processed" / "target_371_review_coverage.csv"
REVIEW_FEATURES = (
    PROJECT_ROOT / "data" / "reviews" / "processed" / "review_features_by_series_month_rolling.csv"
)
SPLITS = PROJECT_ROOT / "data" / "processed" / "splits"
USER_NEEDS_WINDOWS = (
    PROJECT_ROOT / "data" / "processed" / "user_feedback" / "sentiment_monitoring_windows.csv"
)


@dataclass(frozen=True)
class CoreMartResult:
    pipeline_run_id: int
    status: str
    row_counts: dict[str, int]
    quality_results: int
    critical_failures: int


def _start_pipeline(login_path: str) -> int:
    output = query(
        login_path,
        "INSERT INTO auto_ops.pipeline_runs "
        "(pipeline_name,trigger_type,status,code_version,parameters_json) VALUES "
        "('rebuild_core_marts','manual','started','de5-v1',"
        "JSON_OBJECT('mode','shadow_parity')); SELECT LAST_INSERT_ID();",
    ).strip().splitlines()
    return int(output[-1])


def _mark_failed(login_path: str, pipeline_run_id: int, error: Exception) -> None:
    message = str(error)[:4000]
    query(
        login_path,
        "UPDATE auto_ops.pipeline_runs SET status='failed',completed_at=NOW(6),"
        f"error_message={sql_literal(message)} WHERE pipeline_run_id={pipeline_run_id}; "
        "INSERT INTO auto_ops.data_quality_results "
        "(pipeline_run_id,dataset_name,rule_name,rule_version,severity,passed,"
        "observed_value,expected_value,affected_rows) VALUES ("
        f"{pipeline_run_id},'core_marts','core_mart_transaction_gate','de5-v1','critical',"
        f"FALSE,{sql_literal(message)},'all critical rules pass',1);",
    )


def _render(path: Path, pipeline_run_id: int) -> str:
    return path.read_text(encoding="utf-8").replace("__PIPELINE_RUN_ID__", str(pipeline_run_id))


def _frozen_forecast_panel(sales: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    cohort_names = set(pd.read_csv(COHORT, usecols=["series_name"])["series_name"].astype(str))
    panel = build_forecast_panel(sales, config, cohort_names)
    frozen = pd.concat(
        [pd.read_csv(SPLITS / f"{split}.csv", parse_dates=["date"]) for split in ("train", "val", "test")],
        ignore_index=True,
    )
    comparison_columns = [
        "series_name", "series_id", "date", "monthly_sales",
        "lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6",
        "month_sin", "month_cos", *CFG_COLS, "lag_12", "roll_mean_12", "split",
    ]
    actual = panel[comparison_columns].sort_values(["date", "series_name"]).reset_index(drop=True)
    expected = frozen[comparison_columns].sort_values(["date", "series_name"]).reset_index(drop=True)
    assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    return panel


def _payload(record: pd.Series, columns: list[str]) -> dict[str, Any]:
    return {column: normalize_value(record[column]) for column in columns}


def _forecast_rows(panel: pd.DataFrame) -> list[tuple[Any, ...]]:
    review = pd.read_csv(REVIEW_FEATURES, low_memory=False, parse_dates=["date"])
    review["date"] = review["date"].dt.to_period("M").dt.to_timestamp()
    if len(review) != 17_808 or review.duplicated(["series_name", "date"]).any():
        raise ValueError("Rolling review feature contract changed")
    joined = panel.merge(
        review,
        on=["series_name", "date", "split"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_review"),
    )
    if joined["information_cutoff_exclusive"].isna().any():
        raise ValueError("Forecast rows are missing point-in-time review cutoffs")
    review_columns = [
        column
        for column in review.columns
        if column not in {"series_name", "date", "split", "feature_protocol", "information_cutoff_exclusive"}
    ]
    rows: list[tuple[Any, ...]] = []
    for _, row in joined.iterrows():
        rows.append(
            (
                row["series_name"],
                int(pd.Timestamp(row["date"]).strftime("%Y%m%d")),
                row["split"],
                row["lag_1"], row["lag_2"], row["lag_3"], row["lag_12"],
                row["roll_mean_3"], row["roll_mean_6"], row["roll_mean_12"],
                row["month_sin"], row["month_cos"], int(pd.Timestamp(row["date"]).year),
                _payload(row, list(CFG_COLS)),
                _payload(row, review_columns),
                row["information_cutoff_exclusive"],
            )
        )
    return rows


def _user_needs_rows(labels: pd.DataFrame) -> list[tuple[Any, ...]]:
    aspects, statuses = build_latest_snapshot(labels)
    frozen = pd.read_csv(USER_NEEDS_WINDOWS, low_memory=False)
    assert_latest_window_parity(statuses, frozen)
    rows: list[tuple[Any, ...]] = []
    for _, row in aspects.iterrows():
        rows.append(
            (
                row["series_name"],
                int(pd.Timestamp(row["monitoring_month"]).strftime("%Y%m%d")),
                row["aspect_code"],
                int(row["review_count_180d"]),
                int(row["mention_count_180d"]),
                row["positive_rate_180d"],
                row["negative_rate_180d"],
                row["risk_level"],
                row["information_cutoff_exclusive"],
            )
        )
    return rows


def rebuild_core_marts(login_path: str) -> CoreMartResult:
    """Rebuild core marts atomically after exact frozen-input parity."""
    sales = load_standard_sales(login_path)
    config = load_raw_configuration(login_path)
    panel = _frozen_forecast_panel(sales, config)
    forecast_rows = _forecast_rows(panel)
    user_needs_rows = _user_needs_rows(load_standard_review_labels(login_path))
    pipeline_run_id = _start_pipeline(login_path)
    columns = [
        "canonical_series_name", "target_month_date_key", "split_name",
        "lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "roll_mean_6", "roll_mean_12",
        "month_sin", "month_cos", "model_year", "configuration_payload",
        "review_feature_payload", "review_information_cutoff_exclusive",
    ]
    statements = [
        _render(MART_SQL, pipeline_run_id),
        *insert_statements(
            "auto_mart._de5_forecast_input",
            columns,
            forecast_rows,
            chunk_size=25,
        ),
        *insert_statements(
            "auto_mart._de5_user_needs_input",
            [
                "canonical_series_name", "monitoring_month_date_key", "aspect_code",
                "review_count_180d", "mention_count_180d", "positive_rate_180d",
                "negative_rate_180d", "risk_level", "information_cutoff_exclusive",
            ],
            user_needs_rows,
            chunk_size=100,
        ),
        _render(FINALIZE_SQL, pipeline_run_id),
        _render(QUALITY_SQL, pipeline_run_id),
    ]
    try:
        run_transaction(login_path, statements)
    except Exception as error:
        _mark_failed(login_path, pipeline_run_id, error)
        raise
    counts = query(
        login_path,
        "SELECT (SELECT COUNT(*) FROM auto_mart.dim_vehicle_series),"
        "(SELECT COUNT(*) FROM auto_mart.dim_date),"
        "(SELECT COUNT(*) FROM auto_mart.fact_monthly_sales),"
        "(SELECT COUNT(*) FROM auto_mart.mart_forecast_features),"
        "(SELECT COUNT(*) FROM auto_mart.mart_product_analysis),"
        "(SELECT COUNT(*) FROM auto_mart.mart_user_needs);",
    ).strip().split("\t")
    quality = query(
        login_path,
        "SELECT COUNT(*),SUM(severity='critical' AND NOT passed) "
        "FROM auto_ops.data_quality_results "
        f"WHERE pipeline_run_id={pipeline_run_id};",
    ).strip().split("\t")
    return CoreMartResult(
        pipeline_run_id=pipeline_run_id,
        status="succeeded",
        row_counts=dict(
            zip(
                [
                    "vehicle_series", "dates", "sales", "forecast_features",
                    "product", "user_needs",
                ],
                map(int, counts),
                strict=True,
            )
        ),
        quality_results=int(quality[0]),
        critical_failures=int(quality[1]),
    )


def result_dict(result: CoreMartResult) -> dict[str, Any]:
    return asdict(result)

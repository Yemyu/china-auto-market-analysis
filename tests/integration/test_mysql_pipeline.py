from __future__ import annotations

import os
from pathlib import Path

import pytest

from china_auto_market.ingestion.raw_loaders import LOADERS
from china_auto_market.ingestion.validation import validate_full_sources
from china_auto_market.quality.staging import rebuild_staging
from china_auto_market.warehouse.schema import apply_ddl, existing_target_databases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "ci"
SOURCES = {
    "sales": FIXTURE_ROOT / "sales.csv",
    "config": FIXTURE_ROOT / "config.csv",
    "reviews": FIXTURE_ROOT / "reviews.csv",
    "review_labels": FIXTURE_ROOT / "review_labels.csv",
    "sales_corrections": FIXTURE_ROOT / "sales_corrections.csv",
    "annual_sales_corrections": FIXTURE_ROOT / "annual_sales_corrections.csv",
}


pytestmark = pytest.mark.integration


def test_synthetic_sources_reach_staging_through_real_mysql() -> None:
    """Exercise DDL, raw lineage, idempotency and the critical staging gate."""
    if os.environ.get("CI_MYSQL_SMOKE") != "1":
        pytest.skip("requires a disposable MySQL service and CI_MYSQL_SMOKE=1")
    login_path = os.environ.get("MYSQL_LOGIN_PATH", "ci-auto")
    existing = existing_target_databases(login_path)
    if existing:
        raise RuntimeError(
            "CI integration test refuses to use a server containing project databases: "
            f"{sorted(existing)}"
        )

    applied = apply_ddl(login_path)
    assert [path.name for path in applied] == [
        "00_create_databases.sql",
        "10_auto_ops.sql",
        "20_auto_raw.sql",
        "30_auto_staging.sql",
        "40_auto_mart.sql",
        "45_forecast_precision.sql",
        "46_annual_sales_corrections.sql",
        "47_user_needs_precision.sql",
    ]

    first_results = [
        LOADERS[dataset](login_path, source)
        for dataset, source in SOURCES.items()
    ]
    assert all(result.status == "succeeded" for result in first_results)
    assert not any(result.idempotent_skip for result in first_results)

    repeated_results = [
        LOADERS[dataset](login_path, source)
        for dataset, source in SOURCES.items()
    ]
    assert all(result.idempotent_skip for result in repeated_results)

    raw_validation = validate_full_sources(login_path, sources=SOURCES)
    assert raw_validation["passed"] is True
    assert len(raw_validation["checks"]) == 32

    staging_result = rebuild_staging(
        login_path,
        expected_overrides={
            "mapping_rows": 4,
            "sales_mapping_rows": 1,
            "config_sales_mapping_rows": 1,
            "config_only_mapping_rows": 1,
            "review_mapping_rows": 1,
            "sales_rows": 54,
            "sales_series": 1,
            "sales_months_per_series": 54,
            "repair_rows": 1,
            "repair_delta": 10,
            "config_rows": 2,
            "annual_repair_rows": 1,
            "annual_repair_delta": 50,
            "review_rows": 1,
            "eligible_reviews": 1,
            "after_locked_test_reviews": 0,
            "review_feature_rows": 10,
            "review_feature_identities": 1,
        },
    )
    assert staging_result.status == "succeeded"
    assert staging_result.row_counts == {
        "mappings": 4,
        "monthly_sales": 54,
        "vehicle_config": 2,
        "reviews": 1,
        "review_features": 10,
    }
    assert staging_result.quality_results == 19
    assert staging_result.nonpassing_quality_results == 0
    assert staging_result.critical_failures == 0

from __future__ import annotations

from china_auto_market.quality import staging


def test_staging_sql_renders_all_locked_placeholders() -> None:
    values = {
        "sales_batch_id": 1,
        "config_batch_id": 2,
        "reviews_batch_id": 3,
        "labels_batch_id": 4,
        "corrections_batch_id": 5,
        "annual_corrections_batch_id": 6,
        "pipeline_run_id": 6,
        **{f"expected_{name}": value for name, value in staging.EXPECTED.items()},
    }

    assert "__" not in staging._render(staging.STAGING_SQL, values)
    assert "__" not in staging._render(staging.QUALITY_SQL, values)


def test_staging_expectations_preserve_frozen_business_counts() -> None:
    assert staging.EXPECTED == {
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


def test_unknown_expectation_is_rejected_before_opening_pipeline(monkeypatch) -> None:
    def fail_if_called(_: str) -> int:
        raise AssertionError("pipeline must not start for an invalid override")

    monkeypatch.setattr(staging, "_start_pipeline", fail_if_called)
    try:
        staging.rebuild_staging("unused", expected_overrides={"unknown": 1})
    except KeyError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("invalid expectation should fail")

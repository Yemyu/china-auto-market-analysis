import pytest

from china_auto_market.marts.build import configuration_reference
from china_auto_market.marts.configuration_migration import migrate, validate_backup


@pytest.mark.parametrize("batch", [None, True, -1, 0, 1.5, "4"])
def test_mart_reference_rejects_invalid_batch(batch):
    with pytest.raises(ValueError):
        configuration_reference(batch)


def test_reference_has_no_globally_fitted_values():
    assert configuration_reference(4) == {
        "configuration_policy": "raw-batch-reference-v1", "configuration_batch_id": 4}


def test_rebuild_records_actual_configuration_batch_in_both_consumers():
    from china_auto_market.marts.build import MART_SQL, FINALIZE_SQL, _render

    sql = (_render(MART_SQL, 99) + _render(FINALIZE_SQL, 99)).replace("__CONFIG_BATCH_ID__", "7")
    assert "forecast-b2-c7-r5-l8-r1-v2" in sql
    assert "product-b2-a9-c7-de5-v1" in sql
    assert "'config_batch', 4" not in sql
    assert "__CONFIG_BATCH_ID__" not in sql


def test_migration_refuses_unscoped_backup_location(tmp_path):
    with pytest.raises(ValueError, match="artifacts"):
        migrate("not-used", tmp_path)


def test_migration_refuses_missing_rows_or_mixed_versions():
    version = {"dataset_version_id": 3, "upstream_versions_json": {"config_batch": 4}}
    with pytest.raises(ValueError, match="17808"):
        validate_backup([], version)
    rows = [{"vehicle_series_sk": n, "target_month_date_key": 20260101,
             "dataset_version_id": 3} for n in range(17808)]
    assert validate_backup(rows, version) == 4
    rows[-1]["dataset_version_id"] = 5
    with pytest.raises(ValueError, match="mixed"):
        validate_backup(rows, version)

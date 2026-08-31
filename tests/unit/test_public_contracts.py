from __future__ import annotations

import json
from pathlib import Path

import pytest

from china_auto_market.publishing.contracts import assert_dashboard_payloads


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json(relative: str) -> dict[str, object]:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


def test_dashboard_json_schema_is_stable() -> None:
    assert_dashboard_payloads(PROJECT_ROOT / "app" / "static" / "data")


def test_locked_business_results_remain_unchanged_during_engineering_migration() -> None:
    forecast = _json("data/processed/forecast/rolling_origin_summary.json")
    product = _json("data/processed/product/config_attribution_summary.json")
    needs = _json("data/processed/user_feedback/user_needs_alerts_summary.json")

    assert forecast["test_used_for_selection"] is False
    assert forecast["selected_version"] == "SEASONAL_D5"
    locked = forecast["locked_test"]
    assert isinstance(locked, dict)
    assert locked["rows"] == 2_226
    assert locked["series"] == 371
    assert locked["global_volume_weighted_WMAPE"] == pytest.approx(29.723456121551727)
    naive = locked["naive_WMAPE"]
    assert isinstance(naive, dict)
    assert naive["LAST_VALUE"] == pytest.approx(40.992416043671625)

    assert product["rows"] == 1_510
    assert product["series"] == 646
    assert product["config_incremental_r2"] == pytest.approx(0.16891739972375125)
    assert product["config_r2_log_mean"] == pytest.approx(0.23872276190641895)

    assert needs["review_rows"] == 24_175
    assert needs["review_series"] == 345
    assert needs["aspects"] == 10
    assert needs["current_incomplete_month_excluded_from_alerts"] is True
    assert needs["external_api_calls"] == 0

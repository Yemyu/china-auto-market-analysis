import numpy as np
import pandas as pd
import pytest

from china_auto_market.forecasting.cold_start_policy import retain_source_predictions, write_retained_forecast


def source():
    frame = pd.DataFrame({
        "version": ["PLATFORM_RATING_FIXED"] * 6,
        "scenario": ["fixed_origin_primary"] * 6,
        "series_name": ["new_series"] * 6,
        "date": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "actual": [0, 0, 0, 100, 200, 300],
        "pred": [1., 2., 3., 4., 5., 6.],
    })
    return frame, {"configuration_policy": "fit-window-v1",
                   "validation_selected_primary_version": "PLATFORM_RATING_FIXED",
                   "evaluation_series": 1}


def test_test_launch_month_cannot_change_predictions_or_override_decision():
    frame, run = source()
    retained, summary = retain_source_predictions(frame, run)
    frame["actual"] = [1e9] * 6
    changed, other = retain_source_predictions(frame, run)
    np.testing.assert_array_equal(retained.pred, changed.pred)
    np.testing.assert_array_equal(retained.pred, retained.source_pred)
    assert not changed.cold_start_override_applied.any()
    assert summary["applied_rows"] == other["applied_rows"] == 0
    assert summary["model_reselected"] is False


def test_legacy_source_is_rejected():
    frame, run = source()
    run.pop("configuration_policy")
    with pytest.raises(ValueError, match="fit-window"):
        retain_source_predictions(frame, run)


def test_missing_or_duplicate_months_are_rejected():
    frame, run = source()
    with pytest.raises(ValueError, match="six forecast months"):
        retain_source_predictions(frame.iloc[1:], run)
    with pytest.raises(ValueError, match="unique"):
        retain_source_predictions(pd.concat([frame, frame.iloc[:1]]), run)


def test_retirement_cannot_overwrite_published_results(tmp_path):
    with pytest.raises(ValueError, match="artifacts"):
        write_retained_forecast(tmp_path, tmp_path / "public")

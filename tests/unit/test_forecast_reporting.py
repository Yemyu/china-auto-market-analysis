import numpy as np
import pandas as pd
import pytest

from china_auto_market.forecasting.reporting import METHODS, aggregate_metrics, forecast_report, prediction_metrics


def test_zero_policy_preserves_errors_and_smape_denominator():
    frame = pd.DataFrame({"actual": [0, 0, 10], "pred": [0, 10, 0]})
    score = prediction_metrics(frame)
    assert score["mape_rows"] == 1
    assert score["mape_positive"] == 100
    assert score["wmape"] == 200
    assert score["mae"] == pytest.approx(20 / 3)
    assert score["smape"] == pytest.approx(400 / 3)


def test_net_bias_does_not_imply_monthly_accuracy():
    frame = pd.DataFrame({"date": ["2026-01", "2026-02"],
                          "actual": [100, 100], "pred": [150, 50]})
    score = aggregate_metrics(frame)
    assert score["six_month_net_bias_pct"] == 0
    assert score["monthly_aggregate_wmape"] == 50


def test_zero_volume_is_undefined_not_perfect():
    score = prediction_metrics(pd.DataFrame({"actual": [0, 0], "pred": [1, 0]}))
    assert score["wmape"] is None
    assert score["mape_positive"] is None
    assert score["mae"] == .5


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -1])
def test_invalid_predictions_are_not_silently_dropped(prediction):
    with pytest.raises(ValueError):
        prediction_metrics(pd.DataFrame({"actual": [1], "pred": [prediction]}))


def test_staged_reporting_does_not_read_public_predictions(tmp_path):
    forecast = tmp_path / "repaired"
    splits = tmp_path / "splits"
    forecast.mkdir()
    splits.mkdir()
    frame = pd.DataFrame({"series_name": list("abcd"), "date": ["2026-01-01"] * 4,
                          "actual": [0, 10, 20, 30]})
    for method in METHODS:
        frame[method] = [1, 9, 19, 29]
    frame.to_csv(forecast / "rolling_origin_test_predictions.csv", index=False)
    history = pd.DataFrame({"series_name": list("abcd"), "date": ["2025-12-01"] * 4,
                            "monthly_sales": [4, 3, 2, 1]})
    history.to_csv(splits / "val.csv", index=False)
    result = forecast_report(tmp_path / "absent", forecast_dir=forecast, split_dir=splits)
    main = next(row for row in result["metrics"] if row["method"] == "pred")
    assert main["rows"] == 4
    assert main["zero_actual_rows"] == 1
    low = next(row for row in result["segments"]
               if row["method"] == "pred" and row["group_type"] == "pre_test_size" and row["group"] == "Q1")
    assert low["mae"] == 1
    assert low["zero_actual_rows"] == 0  # group follows pre-test sizes, not target sales

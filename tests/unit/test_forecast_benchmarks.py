import numpy as np
import pandas as pd
import pytest

from china_auto_market.forecasting.benchmarks import aligned_predictions, fixed_naive_predictions


@pytest.fixture
def panels():
    history = pd.DataFrame({"series_name": ["a"] * 12,
                            "date": pd.date_range("2025-01-01", periods=12, freq="MS"),
                            "monthly_sales": np.arange(1., 13.)})
    actual = pd.DataFrame({"series_name": ["a"] * 6,
                           "date": pd.date_range("2026-01-01", periods=6, freq="MS"),
                           "actual": [1.] * 6})
    return history, actual


def test_fixed_benchmarks_do_not_use_test_actuals(panels):
    history, actual = panels
    forecasts = fixed_naive_predictions(history, actual)
    changed = fixed_naive_predictions(history.sample(frac=1), actual.assign(actual=1e9))
    np.testing.assert_array_equal(forecasts["LAST_VALUE"], [12] * 6)
    np.testing.assert_array_equal(forecasts["ROLLING_MEAN_3"], [11] * 6)
    np.testing.assert_array_equal(forecasts["ROLLING_MEAN_12"], [6.5] * 6)
    np.testing.assert_array_equal(forecasts["SEASONAL_LAG12"], np.arange(1, 7))
    for name in forecasts:
        np.testing.assert_array_equal(forecasts[name], changed[name])


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "future"])
def test_invalid_benchmark_calendar_fails(panels, mutation):
    history, actual = panels
    if mutation == "gap":
        history = history.drop(index=11)
    elif mutation == "duplicate":
        history = pd.concat([history, history.iloc[[0]]])
    else:
        history.loc[11, "date"] = pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError):
        fixed_naive_predictions(history, actual)


def test_alignment_checks_actuals_and_extra_keys(panels):
    _, actual = panels
    predictions = actual.assign(pred=np.arange(6.)).sample(frac=1)
    np.testing.assert_array_equal(aligned_predictions(actual, predictions), np.arange(6.))
    with pytest.raises(ValueError, match="actuals"):
        aligned_predictions(actual, predictions.assign(actual=2))
    with pytest.raises(ValueError, match="keys"):
        aligned_predictions(actual.iloc[:-1], predictions)

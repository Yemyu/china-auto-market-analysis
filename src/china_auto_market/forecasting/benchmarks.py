"""Fixed-origin benchmarks; no observed target-window sales enter forecasts."""
import numpy as np
import pandas as pd


def aligned_predictions(actual, predictions):
    keys = ["series_name", "date"]
    if actual.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError("Duplicate series/month keys")
    left = actual.sort_values(keys).reset_index(drop=True)
    right = predictions.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Prediction keys differ from the evaluated cohort")
    if not np.array_equal(left.actual.to_numpy(float), right.actual.to_numpy(float)):
        raise ValueError("Prediction actuals differ from the evaluated targets")
    values = right.pred.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid predictions")
    return actual[keys].merge(predictions[[*keys, "pred"]], on=keys,
                              how="left", validate="one_to_one").pred.to_numpy(float)


def fixed_naive_predictions(history, actual):
    """Require complete, calendar-contiguous history for the evaluated cohort.

    Missing history is not silently interpreted as zero or an earlier 'last'
    month. The benchmark used here has at least 12 months for every series.
    """
    history = history.copy()
    actual = actual.copy()
    for frame in (history, actual):
        frame["date"] = pd.to_datetime(frame.date)
        if frame.empty or frame[["series_name", "date"]].isna().any().any():
            raise ValueError("Missing benchmark keys")
        if frame.duplicated(["series_name", "date"]).any() or not frame.date.dt.is_month_start.all():
            raise ValueError("Invalid or duplicate monthly keys")
    origin = actual.date.min()
    if history.date.max() >= origin:
        raise ValueError("Benchmark history overlaps forecast window")
    values = history.monthly_sales.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid history sales")
    history = history.sort_values(["series_name", "date"])
    expected = pd.date_range(origin - pd.DateOffset(months=12), periods=12, freq="MS")
    recent = history.loc[history.date.ge(expected.min())]
    for name in actual.series_name.unique():
        dates = pd.DatetimeIndex(recent.loc[recent.series_name.eq(name), "date"])
        if not dates.equals(expected):
            raise ValueError(f"Incomplete pre-origin calendar for {name}")
    last = history.groupby("series_name").tail(1).set_index("series_name").monthly_sales
    result = {"LAST_VALUE": actual.series_name.map(last).to_numpy(float)}
    for window in (3, 6, 12):
        mean = history.groupby("series_name").tail(window).groupby("series_name").monthly_sales.mean()
        result[f"ROLLING_MEAN_{window}"] = actual.series_name.map(mean).to_numpy(float)
    lookup = history.set_index(["series_name", "date"]).monthly_sales
    seasonal = np.asarray([lookup.get((row.series_name, row.date - pd.DateOffset(years=1)), np.nan)
                           for row in actual.itertuples()], dtype=float)
    result["SEASONAL_LAG12"] = np.where(np.isfinite(seasonal), seasonal, result["LAST_VALUE"])
    return result

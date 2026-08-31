from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from china_auto_market.forecasting import core
from china_auto_market.forecasting import rolling_origin


class LagPlusFiveModel:
    """Small deterministic stand-in that exposes which lag reached the model."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.log1p(frame["lag_1"].to_numpy(float) + 5.0)


def minimal_panel() -> pd.DataFrame:
    rows = [
        {"series_name": "A", "date": "2025-12-01", "split": "train", "monthly_sales": 10.0},
        {"series_name": "A", "date": "2026-01-01", "split": "test", "monthly_sales": 20.0},
        {"series_name": "A", "date": "2026-02-01", "split": "test", "monthly_sales": 30.0},
    ]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in core.CFG_COLS:
        frame[column] = 0.0
    return frame


def test_global_wmape_is_volume_weighted() -> None:
    actual = [100.0, 10.0]
    predicted = [90.0, 0.0]

    assert core.wmape_vol(actual, predicted) == pytest.approx(20 / 110 * 100)


def test_global_wmape_returns_nan_for_zero_actual_volume() -> None:
    assert np.isnan(core.wmape_vol([0, 0], [1, 2]))


def test_fixed_origin_recursion_uses_its_previous_prediction() -> None:
    predictions = core.recursive_forecast_tree(
        LagPlusFiveModel(),
        minimal_panel(),
        feat_cols=["lag_1"],
        history_splits=("train",),
        forecast_splits=("test",),
    )

    assert list(predictions.values()) == pytest.approx([15.0, 20.0])


def test_rolling_origin_uses_previous_realised_month() -> None:
    predictions = rolling_origin.rolling_predictions(
        LagPlusFiveModel(),
        minimal_panel(),
        columns=["lag_1"],
        forecast_split="test",
        history_splits=("train",),
    )

    assert predictions["pred"].tolist() == pytest.approx([15.0, 25.0])


def test_frozen_split_files_keep_absolute_boundaries() -> None:
    train, validation, test = core.load_splits()

    assert (len(train), len(validation), len(test)) == (13_356, 2_226, 2_226)
    assert train["series_name"].nunique() == 371
    assert validation["series_name"].nunique() == 371
    assert test["series_name"].nunique() == 371
    assert train["date"].max() == pd.Timestamp("2025-06-01")
    assert validation["date"].min() == pd.Timestamp("2025-07-01")
    assert validation["date"].max() == pd.Timestamp("2025-12-01")
    assert test["date"].min() == pd.Timestamp("2026-01-01")
    assert test["date"].max() == pd.Timestamp("2026-06-01")

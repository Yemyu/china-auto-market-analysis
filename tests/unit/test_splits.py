from __future__ import annotations

import pandas as pd
import pytest

from china_auto_market.features.splits import assign_split, engineer_features


def test_lags_do_not_cross_vehicle_series_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "series_name": ["B", "A", "A", "B", "A", "B"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-03-01", "2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01"]
            ),
            "monthly_sales": [100, 3, 1, 200, 2, 300],
        }
    )

    result = engineer_features(frame)
    a = result.loc[result["series_name"].eq("A")].sort_values("date")
    b = result.loc[result["series_name"].eq("B")].sort_values("date")

    assert np_is_nan(a.iloc[0]["lag_1"])
    assert a.iloc[1]["lag_1"] == 1
    assert a.iloc[2]["lag_1"] == 2
    assert np_is_nan(b.iloc[0]["lag_1"])
    assert b.iloc[1]["lag_1"] == 100
    assert b.iloc[2]["lag_1"] == 200


def np_is_nan(value: float) -> bool:
    return bool(pd.isna(value))


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2025-06-01", "train"),
        ("2025-07-01", "val"),
        ("2025-12-01", "val"),
        ("2026-01-01", "test"),
    ],
)
def test_split_assignment_uses_locked_absolute_dates(date: str, expected: str) -> None:
    frame = pd.DataFrame({"date": pd.to_datetime([date])})
    assert assign_split(frame).iloc[0]["split"] == expected

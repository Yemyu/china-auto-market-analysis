"""Build the locked 371-series forecasting panel from warehouse sources."""

from __future__ import annotations

import pandas as pd

from china_auto_market.features.splits import STORED_FEATURE_COLS, assign_split, engineer_features


def build_forecast_panel(
    sales: pd.DataFrame,
    cohort_names: set[str],
) -> pd.DataFrame:
    """Return rows eligible for the current train/validation/test protocol."""
    if len(cohort_names) != 371:
        raise ValueError(f"Expected 371 locked forecast series, found {len(cohort_names)}")
    panel = sales.copy()
    panel["series_name"] = panel["series_name"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel = panel.loc[panel["series_name"].isin(cohort_names)].copy()
    missing = cohort_names - set(panel["series_name"])
    if missing:
        raise ValueError(f"Forecast cohort missing from standardized sales: {sorted(missing)}")
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel = assign_split(engineer_features(panel))
    usable_train = panel["split"].ne("train") | panel[STORED_FEATURE_COLS].notna().all(axis=1)
    panel = panel.loc[usable_train].copy()
    if panel.duplicated(["series_name", "date"]).any():
        raise ValueError("Forecast mart panel contains duplicate series-month keys")
    expected_split_rows = {"train": 13_356, "val": 2_226, "test": 2_226}
    actual = panel.groupby("split").size().to_dict()
    if actual != expected_split_rows:
        raise ValueError(f"Forecast split rows changed: {actual}")
    return panel.sort_values(["date", "series_name"]).reset_index(drop=True)

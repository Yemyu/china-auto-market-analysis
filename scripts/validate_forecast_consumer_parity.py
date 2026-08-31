#!/usr/bin/env python3
"""Re-run the locked rolling forecast from MySQL and compare frozen outputs."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from pandas.testing import assert_frame_equal

from china_auto_market.forecasting import core as mu
from china_auto_market.forecasting import review_evaluation, rolling_origin
from china_auto_market.paths import PROJECT_ROOT


FROZEN_SUMMARY = PROJECT_ROOT / "data" / "processed" / "forecast" / "rolling_origin_summary.json"
FROZEN_PREDICTIONS = (
    PROJECT_ROOT / "data" / "processed" / "forecast" / "rolling_origin_test_predictions.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-path", default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"))
    args = parser.parse_args()

    frames, _, _ = review_evaluation.build_frames(
        backend="mysql", login_path=args.login_path
    )
    versions = {
        "BASE": list(mu.FEAT_COLS),
        "SEASONAL_D5": list(mu.SEASONAL_FEAT_COLS),
    }
    _, _, payload = rolling_origin.validation(review_evaluation, frames, versions)
    predictions = rolling_origin.locked_test(review_evaluation, frames, versions, payload)

    frozen_summary = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    frozen_predictions = pd.read_csv(FROZEN_PREDICTIONS, parse_dates=["date"])
    assert payload["selected_version"] == frozen_summary["selected_version"]
    assert payload["gate"] == frozen_summary["gate"]
    for key in (
        "historical_base_rolling_WMAPE",
        "historical_selected_rolling_WMAPE",
        "historical_gain_pp",
        "worst_origin_regression_pp",
    ):
        if abs(float(payload[key]) - float(frozen_summary[key])) > 1e-10:
            raise AssertionError(f"Rolling summary changed for {key}")
    for key, value in frozen_summary["locked_test"].items():
        if key == "naive_WMAPE":
            for method, metric in value.items():
                if abs(float(payload["locked_test"][key][method]) - float(metric)) > 1e-10:
                    raise AssertionError(f"Locked naive metric changed for {method}")
        elif isinstance(value, float):
            if abs(float(payload["locked_test"][key]) - value) > 1e-10:
                raise AssertionError(f"Locked test metric changed for {key}")
        elif payload["locked_test"][key] != value:
            raise AssertionError(f"Locked test contract changed for {key}")
    columns = list(frozen_predictions.columns)
    assert_frame_equal(
        predictions[columns].sort_values(["series_name", "date"]).reset_index(drop=True),
        frozen_predictions[columns]
        .sort_values(["series_name", "date"])
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    result = {
        "passed": True,
        "source": "auto_mart.mart_forecast_features",
        "selected_version": payload["selected_version"],
        "historical_selected_WMAPE": payload["historical_selected_rolling_WMAPE"],
        "locked_test_WMAPE": payload["locked_test"]["global_volume_weighted_WMAPE"],
        "prediction_rows": len(predictions),
        "prediction_series": int(predictions["series_name"].nunique()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

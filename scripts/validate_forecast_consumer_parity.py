#!/usr/bin/env python3
"""Re-run the locked rolling forecast from MySQL and compare frozen outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from china_auto_market.forecasting import core as mu
from china_auto_market.forecasting import review_evaluation, rolling_origin
from china_auto_market.paths import PROJECT_ROOT


FROZEN_SUMMARY = PROJECT_ROOT / "data" / "processed" / "forecast" / "rolling_origin_summary.json"
FROZEN_PREDICTIONS = (
    PROJECT_ROOT / "data" / "processed" / "forecast" / "rolling_origin_test_predictions.csv"
)


def validate_fixed_reference(reference: Path, frames, source) -> dict:
    run = json.loads((reference / "review_feature_run_summary.json").read_text(encoding="utf-8"))
    if run.get("configuration_policy") != "fit-window-v1":
        raise ValueError("Fixed reference must use fit-window-v1")
    csv_frames, versions, _ = review_evaluation.build_frames()
    csv_panels = review_evaluation.prepare_fixed_panels(csv_frames, mu.load_configuration_source())
    mysql_panels = review_evaluation.prepare_fixed_panels(frames, source)
    columns = ["series_name", "date", "split", mu.TARGET,
               *sorted(set(column for features in versions.values() for column in features))]
    for left, right in zip(csv_panels[:3], mysql_panels[:3], strict=True):
        # Do not sort here: identical training order is part of the contract.
        assert_frame_equal(left[columns], right[columns], check_dtype=False)
    assert csv_panels[3] == mysql_panels[3] == run["configuration_preprocessing"]
    version = run["validation_selected_primary_version"]
    summary = pd.read_csv(reference / "review_feature_ablation_summary.csv")
    trees = int(summary.loc[summary.version.eq(version), "validation_selected_n_estimators"].iloc[0])
    final = mysql_panels[1]
    _, predictions = review_evaluation.fit_final_and_predict(
        version, versions[version], trees, final.loc[final.split.isin(["train", "val"])],
        final, "fixed_origin_primary")
    saved = pd.read_csv(reference / "review_feature_predictions.csv", parse_dates=["date"])
    saved = saved.loc[saved.version.eq(version)]
    assert_frame_equal(
        predictions.sort_values(["series_name", "date"]).reset_index(drop=True),
        saved[predictions.columns].sort_values(["series_name", "date"]).reset_index(drop=True),
        check_dtype=False, rtol=1e-10, atol=1e-10)
    return {"passed": True, "version": version, "prediction_rows": len(predictions),
            "preprocessing_panels": 3, "WMAPE": mu.wmape_vol(predictions.actual, predictions.pred)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-path", default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"))
    parser.add_argument("--reference-dir", type=Path, default=FROZEN_SUMMARY.parent,
                        help="Evaluation bundle to compare; use the isolated repair bundle until release")
    parser.add_argument("--fixed-reference-dir", type=Path,
                        help="Also validate fixed-origin input order and selected-model predictions")
    parser.add_argument("--split-dir", type=Path, help="Validate against an independently rebuilt CSV split bundle")
    args = parser.parse_args()
    if args.split_dir:
        mu.SPLITS = args.split_dir.resolve()
    frozen_summary = json.loads((args.reference_dir / FROZEN_SUMMARY.name).read_text(encoding="utf-8"))
    if frozen_summary.get("configuration_policy") != "fit-window-v1":
        raise ValueError("Reference uses the legacy configuration policy. Public release remains blocked; "
                         "use --reference-dir with the reviewed repair bundle for migration checks.")

    frames, _, _ = review_evaluation.build_frames(
        backend="mysql", login_path=args.login_path
    )
    versions = {
        "BASE": list(mu.FEAT_COLS),
        "SEASONAL_D5": list(mu.SEASONAL_FEAT_COLS),
    }
    source = mu.load_configuration_source(backend="mysql", login_path=args.login_path,
                                          frame=frames["train_roll"])
    _, _, payload = rolling_origin.validation(review_evaluation, frames, versions, source)
    predictions = rolling_origin.locked_test(review_evaluation, frames, versions, payload, source)

    frozen_predictions = pd.read_csv(args.reference_dir / FROZEN_PREDICTIONS.name, parse_dates=["date"])
    assert payload["configuration_preprocessing"] == frozen_summary["configuration_preprocessing"]
    assert payload["test_configuration_preprocessing"] == frozen_summary["test_configuration_preprocessing"]
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
        "configuration_source": "auto_raw.raw_vehicle_config",
        "configuration_batch_id": source.attrs["configuration_batch_id"],
        "configuration_batch_bound_to_mart": "configuration_batch_id" in frames["train_roll"].attrs,
        "reference_directory": str(args.reference_dir.resolve()),
        "selected_version": payload["selected_version"],
        "historical_selected_WMAPE": payload["historical_selected_rolling_WMAPE"],
        "locked_test_WMAPE": payload["locked_test"]["global_volume_weighted_WMAPE"],
        "prediction_rows": len(predictions),
        "prediction_series": int(predictions["series_name"].nunique()),
    }
    if args.fixed_reference_dir:
        result["fixed_origin"] = validate_fixed_reference(args.fixed_reference_dir, frames, source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

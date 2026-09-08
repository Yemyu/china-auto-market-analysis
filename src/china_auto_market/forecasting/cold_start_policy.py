"""Retain fixed-origin predictions until launch dates have pre-origin evidence.

First positive test sales are an outcome, not an available launch schedule.
This fallback does not estimate launch horizons or select a new model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from china_auto_market.forecasting import core
from china_auto_market.paths import PROJECT_ROOT


def retain_source_predictions(
    predictions: pd.DataFrame, run: dict,
) -> tuple[pd.DataFrame, dict]:
    if run.get("configuration_policy") != "fit-window-v1":
        raise ValueError("Cold-start fallback requires a fit-window-v1 source evaluation")
    version = run["validation_selected_primary_version"]
    retained = predictions.loc[
        predictions.version.eq(version)
        & predictions.scenario.eq("fixed_origin_primary")
    ].copy()
    if retained.empty or retained.duplicated(["series_name", "date"]).any():
        raise ValueError("Source predictions must have unique, nonempty series/month keys")
    retained["date"] = pd.to_datetime(retained["date"], errors="raise")
    expected = set(pd.date_range("2026-01-01", periods=6, freq="MS"))
    if any(set(group.date) != expected for _, group in retained.groupby("series_name")):
        raise ValueError("Source predictions must cover all six forecast months per series")
    if retained.series_name.nunique() != int(run["evaluation_series"]):
        raise ValueError("Source evaluation cohort does not match its run summary")
    if not np.isfinite(retained[["actual", "pred"]].to_numpy(float)).all():
        raise ValueError("Source targets and predictions must be finite")
    retained["source_pred"] = retained["pred"]
    retained["cold_start_override_applied"] = False
    retained["cold_start_method"] = "source_model_no_verified_launch_schedule"
    summary = {
        "schema_version": "v2",
        "configuration_policy": run["configuration_policy"],
        "status": "launch_curve_disabled",
        "reason": "No verified pre-origin launch schedule; test sales cannot define launch dates",
        "forecast_origin": "2026-01-01",
        "source_feedback_version": version,
        "applied_rows": 0,
        "applied_series": 0,
        "evaluation_rows": len(retained),
        "evaluation_series": retained.series_name.nunique(),
        "global_volume_weighted_WMAPE": core.wmape_vol(retained.actual, retained.pred),
        "prediction_change_max": 0.0,
        "test_is_new_holdout": False,
        "model_reselected": False,
    }
    return retained, summary


def write_retained_forecast(forecast_dir: Path, output: Path) -> dict:
    output = output.resolve()
    artifacts = (PROJECT_ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifacts) or output == artifacts:
        raise ValueError("Pending cold-start repair may only write to a subdirectory of artifacts/")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier repair evidence")
    run = json.loads((forecast_dir / "review_feature_run_summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(forecast_dir / "review_feature_predictions.csv")
    retained, summary = retain_source_predictions(predictions, run)
    output.mkdir(parents=True, exist_ok=True)
    retained.to_csv(output / "cold_start_retained_predictions.csv", index=False)
    (output / "cold_start_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

"""Reject mixed forecast versions before assembling public reports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from china_auto_market.forecasting.reporting import prediction_metrics


def validate_forecast_bundle(directory: Path) -> None:
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    fixed = read("review_feature_run_summary.json")
    rolling = read("rolling_origin_summary.json")
    robustness = read("forecast_robustness_summary.json")
    for item in (fixed, rolling, robustness):
        if item.get("configuration_policy") != "fit-window-v1":
            raise ValueError("Forecast bundle contains legacy configuration transforms")
    for key, name in (("prediction_sha256", "review_feature_predictions.csv"),
                      ("model_run_sha256", "review_feature_run_summary.json")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != robustness.get(key):
            raise ValueError(f"Robustness evidence does not match {name}")
    selected = fixed["validation_selected_primary_version"]
    if selected != robustness["validation_selected_feedback_version"]:
        raise ValueError("Fixed model identity differs across artifacts")
    benchmark = pd.read_csv(directory / "forecast_benchmark_comparison.csv").set_index("method")
    if benchmark.index.str.contains("COLD_START").any():
        raise ValueError("Retired cold-start results remain in the published benchmark")
    predictions = pd.read_csv(directory / "review_feature_predictions.csv")
    scores = {}
    for version in ("BASE", selected):
        part = predictions.loc[predictions.version.eq(version)]
        if len(part) != 2226 or part.series_name.nunique() != 371 or part.duplicated(["series_name", "date"]).any():
            raise ValueError("Fixed forecast cohort has changed")
        scores[version] = prediction_metrics(part)["wmape"]
        if not np.isclose(scores[version], benchmark.loc[version, "global_volume_weighted_WMAPE"], atol=1e-10, rtol=0):
            raise ValueError("Fixed benchmark is stale")
    if not np.isclose(scores["BASE"] - scores[selected], robustness["selected_feedback_vs_base_improvement_pp"], atol=1e-10, rtol=0):
        raise ValueError("Bootstrap point estimate is stale")
    main = pd.read_csv(directory / "rolling_origin_test_predictions.csv")
    if set(main.version) != {rolling["selected_version"]}:
        raise ValueError("Rolling model identity differs from summary")
    metric = prediction_metrics(main)
    if metric["rows"] != 2226 or main.series_name.nunique() != 371:
        raise ValueError("Rolling forecast cohort has changed")
    if not np.isclose(metric["wmape"], rolling["locked_test"]["global_volume_weighted_WMAPE"], atol=1e-10, rtol=0):
        raise ValueError("Rolling summary is stale")

import json
import shutil
from pathlib import Path

import pytest

from china_auto_market.publishing.forecast_contract import validate_forecast_bundle


ROOT = Path(__file__).resolve().parents[2]


def test_formal_forecasts_are_one_repaired_bundle():
    validate_forecast_bundle(ROOT / "data/processed/forecast")


@pytest.mark.parametrize("fault", ["hash", "benchmark", "legacy", "identity"])
def test_public_build_rejects_mixed_forecast_versions(tmp_path, fault):
    source = ROOT / "data/processed/forecast"
    for name in ("review_feature_run_summary.json", "rolling_origin_summary.json",
                 "forecast_robustness_summary.json", "review_feature_predictions.csv",
                 "forecast_benchmark_comparison.csv", "rolling_origin_test_predictions.csv"):
        shutil.copy2(source / name, tmp_path / name)
    if fault == "benchmark":
        path = tmp_path / "forecast_benchmark_comparison.csv"
        path.write_text(path.read_text().replace("PLATFORM_RATING_FIXED", "SELECTED_FEEDBACK_COLD_START"))
    else:
        path = tmp_path / "forecast_robustness_summary.json"
        value = json.loads(path.read_text())
        if fault == "hash":
            value["prediction_sha256"] = "wrong"
        elif fault == "identity":
            value["validation_selected_feedback_version"] = "REVIEW_RICH_FIXED"
        else:
            value["configuration_policy"] = "legacy"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_forecast_bundle(tmp_path)

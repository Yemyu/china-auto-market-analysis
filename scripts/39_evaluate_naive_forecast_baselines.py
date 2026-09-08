#!/usr/bin/env python3
"""Build full-371 fixed-origin naive and model benchmark comparisons."""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from china_auto_market.forecasting import core as mu
from china_auto_market.forecasting.benchmarks import aligned_predictions, fixed_naive_predictions
from china_auto_market.forecasting.reporting import prediction_metrics


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed" / "forecast"
MODEL_PREDICTIONS = OUT / "review_feature_predictions.csv"
OUTPUT = OUT / "forecast_benchmark_comparison.csv"
MODEL_RUN_SUMMARY = OUT / "review_feature_run_summary.json"


def score(method: str, actual: pd.DataFrame, predictions: np.ndarray, kind: str) -> dict[str, float | str | int]:
    per_series = mu.wmape_per_series(actual["actual"], predictions, actual["series_name"])
    return {
        "method": method,
        "method_type": kind,
        "test_rows": len(actual),
        "test_series": actual["series_name"].nunique(),
        "global_volume_weighted_WMAPE": mu.wmape_vol(actual["actual"], predictions),
        "median_per_series_WMAPE": float(per_series.median()),
        **prediction_metrics(actual.assign(pred=predictions)),
    }


def aligned_model_predictions(
    actual: pd.DataFrame,
    path: Path,
    version: str,
) -> np.ndarray:
    frame = pd.read_csv(path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[
        frame["version"].eq(version) & frame["scenario"].eq("fixed_origin_primary"),
        ["series_name", "date", "actual", "pred"],
    ]
    return aligned_predictions(actual, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", type=Path, default=OUT)
    parser.add_argument("--split-dir", type=Path, default=BASE / "data/processed/splits")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output in (args.forecast_dir.resolve(), OUT.resolve()):
        raise ValueError("Stage benchmark results outside the forecast input directory")
    model_path = args.forecast_dir / MODEL_PREDICTIONS.name
    model_run = json.loads((args.forecast_dir / MODEL_RUN_SUMMARY.name).read_text(encoding="utf-8"))
    if model_run.get("configuration_policy") != "fit-window-v1":
        raise ValueError("Expected repaired fit-window-v1 model predictions")
    train, validation, test = [pd.read_csv(args.split_dir / f"{part}.csv", parse_dates=["date"])
                               for part in ("train", "val", "test")]
    history = pd.concat([train, validation], ignore_index=True).sort_values(["series_name", "date"])
    actual = test[["series_name", "date", mu.TARGET]].rename(columns={mu.TARGET: "actual"}).copy()

    naive = fixed_naive_predictions(history, actual)
    rows = [score(name, actual, predictions, "naive") for name, predictions in naive.items()]
    selected_feedback = model_run["validation_selected_primary_version"]
    for version in ("BASE", selected_feedback):
        predictions = aligned_model_predictions(actual, model_path, version)
        rows.append(score(version, actual, predictions, "model"))

    result = pd.DataFrame(rows)
    best_naive = float(
        result.loc[result["method_type"].eq("naive"), "global_volume_weighted_WMAPE"].min()
    )
    result["improvement_vs_best_naive_pp"] = best_naive - result["global_volume_weighted_WMAPE"]
    result["relative_error_reduction_vs_best_naive_pct"] = (
        result["improvement_vs_best_naive_pp"] / best_naive * 100
    )
    result = result.sort_values(["method_type", "global_volume_weighted_WMAPE"]).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / OUTPUT.name, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"[output] {output / OUTPUT.name}")


if __name__ == "__main__":
    main()

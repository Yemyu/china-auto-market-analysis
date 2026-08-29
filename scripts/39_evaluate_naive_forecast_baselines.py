#!/usr/bin/env python3
"""Build full-371 fixed-origin naive and model benchmark comparisons."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

import _model_utils as mu


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed" / "forecast"
MODEL_PREDICTIONS = OUT / "review_feature_predictions.csv"
COLD_START_PREDICTIONS = OUT / "cold_start_hybrid_predictions.csv"
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
        ["series_name", "date", "pred"],
    ]
    merged = actual.merge(frame, on=["series_name", "date"], how="left", validate="one_to_one")
    if merged["pred"].isna().any():
        raise ValueError(f"Missing aligned model predictions for {version}")
    return merged["pred"].to_numpy(float)


def main() -> None:
    train, validation, test = mu.load_splits()
    history = pd.concat([train, validation], ignore_index=True).sort_values(["series_name", "date"])
    actual = test[["series_name", "date", mu.TARGET]].rename(columns={mu.TARGET: "actual"}).copy()

    last = history.groupby("series_name").tail(1).set_index("series_name")[mu.TARGET]
    rolling_means = {
        window: history.groupby("series_name").tail(window).groupby("series_name")[mu.TARGET].mean()
        for window in (3, 6, 12)
    }
    last_predictions = actual["series_name"].map(last).fillna(0).to_numpy(float)
    lookup = history.set_index(["series_name", "date"])[mu.TARGET]
    seasonal_predictions = np.asarray([
        lookup.get((row.series_name, row.date - pd.DateOffset(years=1)), np.nan)
        for row in actual.itertuples()
    ], dtype=float)
    seasonal_predictions = np.where(np.isfinite(seasonal_predictions), seasonal_predictions, last_predictions)

    rows = [
        score("LAST_VALUE", actual, last_predictions, "naive"),
        score("SEASONAL_LAG12", actual, seasonal_predictions, "naive"),
    ]
    for window, means in rolling_means.items():
        predictions = actual["series_name"].map(means).fillna(0).to_numpy(float)
        rows.append(score(f"ROLLING_MEAN_{window}", actual, predictions, "naive"))

    model_run = json.loads(MODEL_RUN_SUMMARY.read_text(encoding="utf-8"))
    selected_feedback = model_run.get(
        "validation_selected_primary_version", model_run["best_primary_version"]
    )
    for version in ("BASE", selected_feedback):
        predictions = aligned_model_predictions(actual, MODEL_PREDICTIONS, version)
        rows.append(score(version, actual, predictions, "model"))
    hybrid_version = "SELECTED_FEEDBACK_COLD_START"
    hybrid = aligned_model_predictions(actual, COLD_START_PREDICTIONS, hybrid_version)
    rows.append(score(hybrid_version, actual, hybrid, "model"))

    result = pd.DataFrame(rows)
    best_naive = float(
        result.loc[result["method_type"].eq("naive"), "global_volume_weighted_WMAPE"].min()
    )
    result["improvement_vs_best_naive_pp"] = best_naive - result["global_volume_weighted_WMAPE"]
    result["relative_error_reduction_vs_best_naive_pct"] = (
        result["improvement_vs_best_naive_pp"] / best_naive * 100
    )
    result = result.sort_values(["method_type", "global_volume_weighted_WMAPE"]).reset_index(drop=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"[output] {OUTPUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()

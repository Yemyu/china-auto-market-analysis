#!/usr/bin/env python3
"""Evaluate dynamic review signals as a conditional rolling expert.

The current sales + seasonality + configuration model remains the backbone.
This experiment asks whether point-in-time changes in review volume, sentiment,
and aspect scores can safely adjust that backbone.  Only historical origins are
used; the 2026-01..06 target is not loaded.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = ROOT / "artifacts" / "review_momentum"
EXPERTS = ("REVIEW_MOMENTUM_GLOBAL", "REVIEW_MOMENTUM_ASPECT")
REVIEW_COUNT_THRESHOLDS = (1, 3, 5, 10, 20)
REVIEW_HISTORY_THRESHOLDS = (2, 4, 6)
BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def add_review_momentum(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create causal level/change features from point-in-time review windows."""
    out = frame.sort_values(["series_name", "date"]).copy()
    count = pd.to_numeric(out["review_count_180d"], errors="coerce").fillna(0.0)
    available = pd.to_numeric(
        out["review_available_180d"], errors="coerce"
    ).fillna(0.0)
    out["review_log_count_180d"] = np.log1p(count)
    out["review_history_months_12"] = (
        available.groupby(out["series_name"], sort=False)
        .transform(lambda values: values.rolling(12, min_periods=1).sum())
    )

    aspect_scores = [
        column for column in out.columns
        if column.startswith("review_")
        and column.endswith("_score_180d_mean")
        and column != "review_overall_aspect_score_180d_mean"
    ]
    negative_rates = [
        column for column in out.columns
        if column.startswith("review_") and column.endswith("_negative_180d_rate")
    ]
    positive_rates = [
        column for column in out.columns
        if column.startswith("review_") and column.endswith("_positive_180d_rate")
    ]
    if not aspect_scores or not negative_rates or not positive_rates:
        raise ValueError("Review aspect feature groups are incomplete")

    aggregates = {
        "review_aspect_score_mean": out[aspect_scores].mean(axis=1),
        "review_aspect_score_min": out[aspect_scores].min(axis=1),
        "review_aspect_score_std": out[aspect_scores].std(axis=1),
        "review_aspect_negative_mean": out[negative_rates].mean(axis=1),
        "review_aspect_negative_max": out[negative_rates].max(axis=1),
        "review_aspect_negative_std": out[negative_rates].std(axis=1),
        "review_aspect_positive_mean": out[positive_rates].mean(axis=1),
        "review_aspect_positive_min": out[positive_rates].min(axis=1),
        "review_aspect_positive_std": out[positive_rates].std(axis=1),
    }
    for column, values in aggregates.items():
        out[column] = values

    global_roots = [
        "review_log_count_180d",
        "review_any_positive_180d_rate",
        "review_any_negative_180d_rate",
        "review_overall_aspect_score_180d_mean",
    ]
    aspect_roots = global_roots + list(aggregates)
    for column in aspect_roots:
        values = pd.to_numeric(out[column], errors="coerce")
        grouped = values.groupby(out["series_name"], sort=False)
        out[f"{column}_delta_1m"] = values - grouped.shift(1)
        out[f"{column}_delta_3m"] = values - grouped.shift(3)

    global_features = ["review_history_months_12"]
    for column in global_roots:
        global_features.extend([column, f"{column}_delta_1m", f"{column}_delta_3m"])
    aspect_features = list(global_features)
    for column in aggregates:
        aspect_features.extend([column, f"{column}_delta_1m", f"{column}_delta_3m"])
    return out, unique(global_features), unique(aspect_features)


def fit_predict(
    rolling,
    fit: pd.DataFrame,
    panel: pd.DataFrame,
    columns: list[str],
    base_columns: list[str],
) -> pd.DataFrame:
    usable = fit.dropna(subset=base_columns)
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        **rolling.MODEL_PARAMS["SEASONAL_D5"],
    )
    model.fit(usable[columns], np.log1p(usable[rolling.mu.TARGET]), verbose=False)
    return rolling.rolling_predictions(
        model,
        panel,
        columns,
        forecast_split="val",
        history_splits=("train",),
    )


def historical_predictions() -> tuple[pd.DataFrame, dict[str, list[str]], Any]:
    conditional = load_module(
        "conditional_experts", SCRIPTS / "49_evaluate_conditional_experts.py"
    )
    rolling = conditional.load_module(
        "rolling_origin", SCRIPTS / "48_evaluate_rolling_origin.py"
    )
    review_module = rolling.load_forecast_module()
    historical, _ = conditional.load_historical_frames(rolling, review_module)
    historical, global_review, aspect_review = add_review_momentum(historical)

    base_columns = unique(
        rolling.mu.LAG_COLS
        + rolling.mu.CAL
        + rolling.mu.SEASONAL_LAG_COLS
        + rolling.mu.CFG_COLS
    )
    versions = {
        "CONFIG_SEASONAL": base_columns,
        "REVIEW_MOMENTUM_GLOBAL": unique(base_columns + global_review),
        "REVIEW_MOMENTUM_ASPECT": unique(base_columns + aspect_review),
    }
    frames: list[pd.DataFrame] = []
    for origin in rolling.ORIGINS:
        origin_name = origin.strftime("%Y-%m-%d")
        fit, panel = rolling.origin_panel(historical, origin)
        metadata = conditional.causal_router_metadata(panel)
        momentum_meta = panel.loc[
            panel["split"].eq("val"),
            ["series_name", "date", "review_history_months_12"],
        ]
        metadata = metadata.merge(
            momentum_meta,
            on=["series_name", "date"],
            how="left",
            validate="one_to_one",
        )
        for version, columns in versions.items():
            prediction = fit_predict(rolling, fit, panel, columns, base_columns)
            prediction = prediction.merge(
                metadata,
                on=["series_name", "date"],
                how="left",
                validate="one_to_one",
            )
            prediction["origin"] = origin_name
            prediction["version"] = version
            frames.append(prediction)
            print(
                f"[{origin:%Y-%m}] {version:24s} "
                f"WMAPE={rolling.mu.wmape_vol(prediction['actual'], prediction['pred']):.3f}",
                flush=True,
            )
    result = pd.concat(frames, ignore_index=True)
    if result["date"].max() >= pd.Timestamp("2026-01-01"):
        raise AssertionError("Locked-test target entered the momentum experiment")
    expected = 4 * 6 * 371 * len(versions)
    if len(result) != expected:
        raise AssertionError(f"Prediction coverage changed: {len(result)} != {expected}")
    return result, versions, conditional


def prediction_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    index = [
        "origin", "series_name", "date", "actual",
        "prior_history_months", "prior_positive_months",
        "recent_positive_months_6", "recent_mean_6",
        "config_available", "config_source_year", "config_age_years",
        "review_count_180d", "review_available_180d",
        "review_history_months_12",
    ]
    matrix = predictions.pivot(index=index, columns="version", values="pred").reset_index()
    matrix.columns.name = None
    if len(matrix) != 4 * 6 * 371:
        raise AssertionError("Momentum prediction matrix dropped rows")
    return matrix


def routing_grid(matrix: pd.DataFrame, conditional) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expert in EXPERTS:
        for count_threshold in REVIEW_COUNT_THRESHOLDS:
            for history_threshold in REVIEW_HISTORY_THRESHOLDS:
                eligible = (
                    matrix["review_available_180d"].eq(1)
                    & matrix["review_count_180d"].ge(count_threshold)
                    & matrix["review_history_months_12"].ge(history_threshold)
                )
                for weight in BLEND_WEIGHTS:
                    candidate = (
                        f"_route_{expert}_{count_threshold}_{history_threshold}_{weight:.2f}"
                    )
                    matrix[candidate] = matrix["CONFIG_SEASONAL"] + weight * (
                        matrix[expert] - matrix["CONFIG_SEASONAL"]
                    ) * eligible.astype(float)
                    metrics = conditional.route_metrics(
                        matrix, candidate, "CONFIG_SEASONAL", eligible
                    )
                    rows.append({
                        "expert_family": "review_momentum",
                        "expert": expert,
                        "review_count_threshold": count_threshold,
                        "review_history_months_threshold": history_threshold,
                        "blend_weight": weight,
                        "prediction_column": candidate,
                        **metrics,
                    })
    return pd.DataFrame(rows)


def serializable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, versions, conditional = historical_predictions()
    matrix = prediction_matrix(predictions)
    grid = routing_grid(matrix, conditional)
    selected = conditional.select_on_calibration(grid)
    passes = conditional.passes_review_gate(selected)
    prediction_column = (
        str(selected["prediction_column"]) if passes else "CONFIG_SEASONAL"
    )
    matrix["selected_prediction"] = matrix[prediction_column]
    by_origin = conditional.origin_scores(matrix, "selected_prediction")
    base_scores = conditional.origin_scores(matrix, "CONFIG_SEASONAL").rename(
        columns={"WMAPE": "base_WMAPE"}
    )
    by_origin = by_origin.merge(base_scores, on="origin", validate="one_to_one")
    by_origin["gain_vs_base_pp"] = by_origin["base_WMAPE"] - by_origin["WMAPE"]

    selected_row = {key: serializable(value) for key, value in selected.to_dict().items()}
    summary = {
        "schema_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "conditional review-momentum expert on the configuration forecast backbone",
        "locked_test_targets_read": False,
        "latest_target_month_used": predictions["date"].max().strftime("%Y-%m-%d"),
        "calibration_origins": list(conditional.CALIBRATION_ORIGINS),
        "promotion_origin": conditional.PROMOTION_ORIGIN,
        "feature_versions": versions,
        "routing_candidates": {
            "experts": list(EXPERTS),
            "review_count_thresholds": list(REVIEW_COUNT_THRESHOLDS),
            "review_history_months_thresholds": list(REVIEW_HISTORY_THRESHOLDS),
            "blend_weights": list(BLEND_WEIGHTS),
        },
        "gate": conditional.REVIEW_GATE,
        "selected_on_calibration": selected_row,
        "passes_promotion_gate": passes,
        "deployed_historical_prediction": prediction_column,
        "historical_origin_results": by_origin.to_dict(orient="records"),
        "stop_policy": (
            "If the promotion gate fails, review features remain in demand/risk "
            "analysis and no locked-test momentum experiment is permitted."
        ),
    }
    predictions.to_csv(OUT / "candidate_predictions.csv.gz", index=False, compression="gzip")
    grid.to_csv(OUT / "routing_grid.csv", index=False)
    by_origin.to_csv(OUT / "selected_historical_origins.csv", index=False)
    matrix[[
        "origin", "series_name", "date", "actual", "CONFIG_SEASONAL",
        "selected_prediction", "review_count_180d", "review_history_months_12",
    ]].to_csv(OUT / "selected_predictions.csv.gz", index=False, compression="gzip")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSelected review-momentum route:", flush=True)
    print(selected.to_string(), flush=True)
    print(f"passes={passes}", flush=True)
    print("\nHistorical system:", flush=True)
    print(by_origin.to_string(index=False), flush=True)
    print(f"locked_test_targets_read=False output={OUT}", flush=True)


if __name__ == "__main__":
    main()

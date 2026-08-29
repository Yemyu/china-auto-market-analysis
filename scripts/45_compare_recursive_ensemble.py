#!/usr/bin/env python3
"""Compare locked recursive XGBoost candidates and validation-selected blends."""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import _model_utils as mu


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "forecast"
VALIDATION_OUTPUT = OUT / "recursive_xgb_candidate_validation.csv"
BLEND_OUTPUT = OUT / "recursive_xgb_blend_validation.csv"
TEST_OUTPUT = OUT / "recursive_xgb_candidate_test.csv"
SUMMARY_OUTPUT = OUT / "recursive_xgb_ensemble_summary.json"
ORIGINS = tuple(pd.Timestamp(value) for value in (
    "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
))

CURRENT = {
    "n_estimators": 100, "max_depth": 6, "learning_rate": 0.05,
    "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.0,
    "subsample": 0.8, "colsample_bytree": 0.8,
}
REGULARIZED = {
    "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05,
    "min_child_weight": 3, "reg_lambda": 5.0, "reg_alpha": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9,
}
CANDIDATES = {
    "CURRENT_BASE": ("BASE", CURRENT),
    "REGULARIZED_BASE": ("BASE", REGULARIZED),
    "CURRENT_PLATFORM": ("PLATFORM_RATING_FIXED", CURRENT),
    "REGULARIZED_PLATFORM": ("PLATFORM_RATING_FIXED", REGULARIZED),
}


def load_forecast_module():
    path = ROOT / "scripts" / "33_evaluate_review_features.py"
    spec = importlib.util.spec_from_file_location("review_forecast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_for(params: dict[str, float | int]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=1, **params
    )


def make_origin_panel(
    frame: pd.DataFrame,
    local: pd.DataFrame,
    origin: pd.Timestamp,
    external_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = origin + pd.offsets.MonthBegin(6)
    panel = frame.loc[frame["date"].lt(end)].copy()
    panel["split"] = np.where(
        panel["date"].lt(origin), "train",
        np.where(panel["date"].lt(end), "val", "outside"),
    )
    if external_columns:
        anchor = local.loc[
            local["date"].eq(origin), ["series_name", *external_columns]
        ].drop_duplicates("series_name").set_index("series_name")
        forecast_mask = panel["split"].eq("val")
        for column in external_columns:
            panel.loc[forecast_mask, column] = panel.loc[
                forecast_mask, "series_name"
            ].map(anchor[column])
    fit = panel.loc[panel["split"].eq("train")].dropna(subset=mu.FEAT_COLS)
    return fit, panel


def predict_candidate(
    module,
    frame: pd.DataFrame,
    local: pd.DataFrame,
    versions: dict[str, list[str]],
    candidate: str,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    version, params = CANDIDATES[candidate]
    columns = versions[version]
    external = [column for column in columns if column not in mu.FEAT_COLS]
    fit, panel = make_origin_panel(frame, local, origin, external)
    model = model_for(params)
    model.fit(fit[columns], np.log1p(fit[mu.TARGET]), verbose=False)
    predictions = module.recursive_predictions(
        model, panel, columns, "val", ("train",), candidate,
        "historical_origin_validation",
    )
    predictions["candidate"] = candidate
    predictions["origin"] = origin
    return predictions


def candidate_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in predictions.groupby("candidate"):
        origin_scores = []
        for _, origin_rows in group.groupby("origin"):
            origin_scores.append(mu.wmape_vol(origin_rows["actual"], origin_rows["pred"]))
        rows.append({
            "candidate": candidate,
            "historical_origins": int(group["origin"].nunique()),
            "validation_rows": int(len(group)),
            "pooled_global_WMAPE": mu.wmape_vol(group["actual"], group["pred"]),
            "mean_origin_WMAPE": float(np.mean(origin_scores)),
            "worst_origin_WMAPE": float(np.max(origin_scores)),
        })
    return pd.DataFrame(rows).sort_values(
        ["pooled_global_WMAPE", "worst_origin_WMAPE", "candidate"]
    ).reset_index(drop=True)


def blend_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin", "series_name", "date", "actual"]
    wide = predictions.pivot(index=keys, columns="candidate", values="pred").reset_index()
    rows = []
    for left, right in itertools.combinations(sorted(CANDIDATES), 2):
        for left_weight in np.linspace(0.0, 1.0, 11):
            pred = left_weight * wide[left] + (1.0 - left_weight) * wide[right]
            rows.append({
                "left_candidate": left,
                "right_candidate": right,
                "left_weight": float(left_weight),
                "right_weight": float(1.0 - left_weight),
                "pooled_global_WMAPE": mu.wmape_vol(wide["actual"], pred),
            })
    return pd.DataFrame(rows).sort_values(
        ["pooled_global_WMAPE", "left_candidate", "right_candidate", "left_weight"]
    ).reset_index(drop=True)


def test_predictions(module, frames, versions, candidate: str) -> pd.DataFrame:
    version, params = CANDIDATES[candidate]
    columns = versions[version]
    train_val = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    panel = pd.concat(
        [frames["train_roll"], frames["val_roll"], frames["test_fixed"]],
        ignore_index=True,
    ).sort_values(["series_name", "date"])
    model = model_for(params)
    model.fit(train_val[columns], np.log1p(train_val[mu.TARGET]), verbose=False)
    result = module.recursive_predictions(
        model, panel, columns, "test", ("train", "val"), candidate,
        "locked_candidate_test",
    )
    result["candidate"] = candidate
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_forecast_module()
    frames, versions, _ = module.build_frames()
    local, _, _, _, _ = module.read_external_tables()
    historical = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])

    prediction_frames = []
    for candidate in CANDIDATES:
        for origin in ORIGINS:
            prediction = predict_candidate(
                module, historical, local, versions, candidate, origin
            )
            prediction_frames.append(prediction)
            print(
                f"[candidate:{candidate}] {origin:%Y-%m} "
                f"WMAPE={mu.wmape_vol(prediction['actual'], prediction['pred']):.3f}",
                flush=True,
            )
    validation_predictions = pd.concat(prediction_frames, ignore_index=True)
    candidates = candidate_summary(validation_predictions)
    blends = blend_grid(validation_predictions)
    candidates.to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")
    blends.to_csv(BLEND_OUTPUT, index=False, encoding="utf-8-sig")

    selected_blend = blends.iloc[0]
    left = str(selected_blend["left_candidate"])
    right = str(selected_blend["right_candidate"])
    left_weight = float(selected_blend["left_weight"])
    test_parts = {
        candidate: test_predictions(module, frames, versions, candidate)
        for candidate in CANDIDATES
    }
    test_rows = []
    for candidate, prediction in test_parts.items():
        per_series = mu.wmape_per_series(
            prediction["actual"], prediction["pred"], prediction["series_name"]
        )
        test_rows.append({
            "method": candidate,
            "global_volume_weighted_WMAPE": mu.wmape_vol(
                prediction["actual"], prediction["pred"]
            ),
            "median_per_series_WMAPE": float(per_series.median()),
        })
    left_test = test_parts[left].sort_values(["series_name", "date"]).reset_index(drop=True)
    right_test = test_parts[right].sort_values(["series_name", "date"]).reset_index(drop=True)
    if not left_test[["series_name", "date", "actual"]].equals(
        right_test[["series_name", "date", "actual"]]
    ):
        raise ValueError("Selected blend test rows are not aligned")
    blend_pred = left_weight * left_test["pred"] + (1.0 - left_weight) * right_test["pred"]
    blend_series = mu.wmape_per_series(
        left_test["actual"], blend_pred, left_test["series_name"]
    )
    blend_name = f"BLEND_{left}_{left_weight:.1f}_{right}_{1-left_weight:.1f}"
    test_rows.append({
        "method": blend_name,
        "global_volume_weighted_WMAPE": mu.wmape_vol(left_test["actual"], blend_pred),
        "median_per_series_WMAPE": float(blend_series.median()),
    })
    test = pd.DataFrame(test_rows).sort_values("global_volume_weighted_WMAPE")
    test.to_csv(TEST_OUTPUT, index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": "v1",
        "selection_protocol": "four historical six-month origins; fixed-origin review features; pairwise convex blend grid",
        "selection_metric": "pooled global volume-weighted WMAPE",
        "test_used_for_selection": False,
        "candidate_validation": candidates.to_dict("records"),
        "selected_blend": selected_blend.to_dict(),
        "locked_test_results": test.to_dict("records"),
        "external_api_calls": 0,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== Candidate validation =====")
    print(candidates.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n===== Selected validation blend =====")
    print(selected_blend.to_string())
    print("\n===== Locked test =====")
    print(test.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tune the shared recursive XGBoost profile on historical forecast origins.

This is an internal experiment.  Candidate profiles are ranked on four
pre-test six-month origins using the BASE feature set.  The 2026 test is
evaluated once after selection and is never used to choose the profile.
"""
from __future__ import annotations

import importlib.util
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
VALIDATION_OUTPUT = OUT / "recursive_xgb_profile_validation.csv"
TEST_OUTPUT = OUT / "recursive_xgb_profile_test.csv"
SUMMARY_OUTPUT = OUT / "recursive_xgb_profile_summary.json"
ORIGINS = tuple(pd.Timestamp(value) for value in (
    "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
))

COMMON_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
}
PROFILES = {
    "CURRENT_D6": {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.05,
        "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.0,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
    "SHALLOW_D4": {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
        "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.0,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
    "SHALLOW_D3_LONG": {
        "n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
        "min_child_weight": 1, "reg_lambda": 2.0, "reg_alpha": 0.0,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
    "D4_REGULARIZED": {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
        "min_child_weight": 5, "reg_lambda": 5.0, "reg_alpha": 0.1,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
    "D5_REGULARIZED": {
        "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05,
        "min_child_weight": 3, "reg_lambda": 5.0, "reg_alpha": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
    "D6_STRONG_REG": {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.05,
        "min_child_weight": 5, "reg_lambda": 10.0, "reg_alpha": 0.1,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
    "D4_SLOW": {
        "n_estimators": 150, "max_depth": 4, "learning_rate": 0.03,
        "min_child_weight": 2, "reg_lambda": 3.0, "reg_alpha": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
    "D4_FAST": {
        "n_estimators": 75, "max_depth": 4, "learning_rate": 0.08,
        "min_child_weight": 2, "reg_lambda": 3.0, "reg_alpha": 0.05,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
}


def load_forecast_module():
    path = ROOT / "scripts" / "33_evaluate_review_features.py"
    spec = importlib.util.spec_from_file_location("review_forecast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_for(profile: dict[str, float | int]) -> XGBRegressor:
    return XGBRegressor(**COMMON_PARAMS, **profile)


def origin_panel(frame: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = origin + pd.offsets.MonthBegin(6)
    panel = frame.loc[frame["date"].lt(end)].copy()
    panel["split"] = np.where(
        panel["date"].lt(origin), "train",
        np.where(panel["date"].lt(end), "val", "outside"),
    )
    fit = panel.loc[panel["date"].lt(origin)].dropna(subset=mu.FEAT_COLS).copy()
    return fit, panel


def evaluate_profile(module, frame: pd.DataFrame, name: str, profile: dict[str, float | int]) -> list[dict]:
    rows = []
    for origin in ORIGINS:
        fit, panel = origin_panel(frame, origin)
        model = model_for(profile)
        model.fit(fit[mu.FEAT_COLS], np.log1p(fit[mu.TARGET]), verbose=False)
        predictions = module.recursive_predictions(
            model, panel, mu.FEAT_COLS, "val", ("train",),
            name, "historical_origin_validation",
        )
        error = float(np.abs(predictions["actual"] - predictions["pred"]).sum())
        volume = float(np.abs(predictions["actual"]).sum())
        per_series = mu.wmape_per_series(
            predictions["actual"], predictions["pred"], predictions["series_name"]
        )
        rows.append({
            "profile": name,
            "origin": origin.strftime("%Y-%m-%d"),
            "validation_rows": int(len(predictions)),
            "validation_series": int(predictions["series_name"].nunique()),
            "absolute_error": error,
            "actual_volume": volume,
            "global_volume_weighted_WMAPE": error / volume * 100 if volume else np.nan,
            "median_per_series_WMAPE": float(per_series.median()),
        })
        print(
            f"[profile:{name}] origin={origin:%Y-%m} "
            f"WMAPE={rows[-1]['global_volume_weighted_WMAPE']:.3f}",
            flush=True,
        )
    return rows


def summarize_validation(rows: pd.DataFrame) -> pd.DataFrame:
    result = (
        rows.groupby("profile", as_index=False)
        .agg(
            historical_origins=("origin", "nunique"),
            pooled_absolute_error=("absolute_error", "sum"),
            pooled_actual_volume=("actual_volume", "sum"),
            mean_origin_WMAPE=("global_volume_weighted_WMAPE", "mean"),
            worst_origin_WMAPE=("global_volume_weighted_WMAPE", "max"),
            mean_origin_median_series_WMAPE=("median_per_series_WMAPE", "mean"),
        )
    )
    result["pooled_global_WMAPE"] = (
        result["pooled_absolute_error"] / result["pooled_actual_volume"] * 100
    )
    return result.sort_values(
        ["pooled_global_WMAPE", "worst_origin_WMAPE", "profile"]
    ).reset_index(drop=True)


def final_test(module, frames, versions, selected: str) -> pd.DataFrame:
    profile = PROFILES[selected]
    train_val = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    rows = []
    for version in ("BASE", "PLATFORM_RATING_FIXED"):
        columns = versions[version]
        model = model_for(profile)
        model.fit(train_val[columns], np.log1p(train_val[mu.TARGET]), verbose=False)
        test_panel = pd.concat(
            [frames["train_roll"], frames["val_roll"], frames["test_fixed"]],
            ignore_index=True,
        ).sort_values(["series_name", "date"])
        predictions = module.recursive_predictions(
            model, test_panel, columns, "test", ("train", "val"),
            version, "locked_profile_test",
        )
        per_series = mu.wmape_per_series(
            predictions["actual"], predictions["pred"], predictions["series_name"]
        )
        rows.append({
            "profile": selected,
            "version": version,
            "test_rows": int(len(predictions)),
            "test_series": int(predictions["series_name"].nunique()),
            "global_volume_weighted_WMAPE": mu.wmape_vol(
                predictions["actual"], predictions["pred"]
            ),
            "median_per_series_WMAPE": float(per_series.median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_forecast_module()
    frames, versions, _ = module.build_frames()
    historical = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])

    all_rows = []
    for name, profile in PROFILES.items():
        all_rows.extend(evaluate_profile(module, historical, name, profile))
    origin_results = pd.DataFrame(all_rows)
    summary = summarize_validation(origin_results)
    selected = str(summary.iloc[0]["profile"])
    origin_results.merge(
        summary[["profile", "pooled_global_WMAPE"]], on="profile", how="left"
    ).to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")

    test = final_test(module, frames, versions, selected)
    test.to_csv(TEST_OUTPUT, index=False, encoding="utf-8-sig")
    payload = {
        "schema_version": "v1",
        "selection_protocol": "four historical six-month recursive origins; BASE features only",
        "selection_metric": "pooled global volume-weighted WMAPE",
        "test_used_for_selection": False,
        "selected_profile": selected,
        "selected_params": PROFILES[selected],
        "historical_origins": [value.strftime("%Y-%m-%d") for value in ORIGINS],
        "validation_profiles": summary.to_dict("records"),
        "locked_test_results": test.to_dict("records"),
        "external_api_calls": 0,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== Historical-origin profile ranking =====")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n===== Locked 2026 test (not used for selection) =====")
    print(test.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

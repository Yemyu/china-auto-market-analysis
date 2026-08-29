#!/usr/bin/env python3
"""Compare XGBoost target transformations without using the 2026 test to select."""
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
VALIDATION_OUTPUT = OUT / "xgb_target_validation.csv"
TEST_OUTPUT = OUT / "xgb_target_test.csv"
SUMMARY_OUTPUT = OUT / "xgb_target_summary.json"
ORIGINS = tuple(pd.Timestamp(value) for value in (
    "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
))

CURRENT_PARAMS = {
    "n_estimators": 100, "max_depth": 6, "learning_rate": 0.05,
    "min_child_weight": 1, "reg_lambda": 1.0, "reg_alpha": 0.0,
    "subsample": 0.8, "colsample_bytree": 0.8,
}
REGULARIZED_PARAMS = {
    "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05,
    "min_child_weight": 3, "reg_lambda": 5.0, "reg_alpha": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9,
}
CANDIDATES = {
    "LOG_SQUARE_CURRENT": {
        "transform": "log", "objective": "reg:squarederror",
        "params": CURRENT_PARAMS, "weight": "none",
    },
    "LOG_SQUARE_REG": {
        "transform": "log", "objective": "reg:squarederror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "LOG_ABSOLUTE_REG": {
        "transform": "log", "objective": "reg:absoluteerror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "LOG_HUBER_REG": {
        "transform": "log", "objective": "reg:pseudohubererror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "SQRT_SQUARE_REG": {
        "transform": "sqrt", "objective": "reg:squarederror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "RAW_ABSOLUTE_REG": {
        "transform": "raw", "objective": "reg:absoluteerror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "RAW_HUBER_REG": {
        "transform": "raw", "objective": "reg:pseudohubererror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "RAW_SQUARE_REG": {
        "transform": "raw", "objective": "reg:squarederror",
        "params": REGULARIZED_PARAMS, "weight": "none",
    },
    "LOG_SQUARE_LOG_WEIGHT": {
        "transform": "log", "objective": "reg:squarederror",
        "params": REGULARIZED_PARAMS, "weight": "log",
    },
    "LOG_SQUARE_SQRT_WEIGHT": {
        "transform": "log", "objective": "reg:squarederror",
        "params": REGULARIZED_PARAMS, "weight": "sqrt",
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


def transform_target(values: pd.Series | np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log":
        return np.log1p(values)
    if transform == "sqrt":
        return np.sqrt(np.maximum(values, 0.0))
    if transform == "raw":
        return values
    raise ValueError(transform)


def inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log":
        return np.expm1(values)
    if transform == "sqrt":
        return np.square(np.maximum(values, 0.0))
    if transform == "raw":
        return values
    raise ValueError(transform)


def sample_weights(values: pd.Series | np.ndarray, method: str) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if method == "none":
        return None
    if method == "log":
        weights = np.log1p(values)
    elif method == "sqrt":
        weights = np.sqrt(values + 1.0)
    else:
        raise ValueError(method)
    return weights / np.mean(weights)


def new_model(specification: dict) -> XGBRegressor:
    return XGBRegressor(
        objective=specification["objective"],
        random_state=42,
        n_jobs=1,
        **specification["params"],
    )


def recursive_predictions(
    model: XGBRegressor,
    panel: pd.DataFrame,
    columns: list[str],
    transform: str,
    forecast_split: str,
    history_splits: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for name, series in panel.groupby("series_name", sort=True):
        series = series.sort_values("date").reset_index(drop=True)
        history = series.loc[
            series["split"].isin(history_splits), mu.TARGET
        ].astype(float).tolist()
        for _, record in series.loc[series["split"].eq(forecast_split)].iterrows():
            date = record["date"]
            values = np.asarray(history, dtype=float)
            row = {
                "lag_1": values[-1] if len(values) >= 1 else 0.0,
                "lag_2": values[-2] if len(values) >= 2 else 0.0,
                "lag_3": values[-3] if len(values) >= 3 else 0.0,
                "roll_mean_3": float(np.mean(values[-3:])) if len(values) else 0.0,
                "roll_mean_6": float(np.mean(values[-6:])) if len(values) else 0.0,
                "month_sin": np.sin(2 * np.pi * date.month / 12),
                "month_cos": np.cos(2 * np.pi * date.month / 12),
                "year": date.year,
            }
            for column in mu.CFG_COLS:
                row[column] = record[column]
            for column in columns:
                if column not in row:
                    row[column] = record[column]
            transformed = model.predict(pd.DataFrame([row], columns=columns))
            prediction = float(np.maximum(inverse_target(transformed, transform)[0], 0.0))
            rows.append({
                "series_name": name,
                "date": date,
                "actual": float(record[mu.TARGET]),
                "pred": prediction,
            })
            history.append(prediction)
    return pd.DataFrame(rows)


def historical_panel(frame: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = origin + pd.offsets.MonthBegin(6)
    panel = frame.loc[frame["date"].lt(end)].copy()
    panel["split"] = np.where(
        panel["date"].lt(origin), "train",
        np.where(panel["date"].lt(end), "val", "outside"),
    )
    fit = panel.loc[panel["split"].eq("train")].dropna(subset=mu.FEAT_COLS)
    return fit, panel


def fit_predict(
    fit: pd.DataFrame,
    panel: pd.DataFrame,
    columns: list[str],
    specification: dict,
    forecast_split: str,
    history_splits: tuple[str, ...],
) -> pd.DataFrame:
    model = new_model(specification)
    target = transform_target(fit[mu.TARGET], specification["transform"])
    weights = sample_weights(fit[mu.TARGET], specification["weight"])
    model.fit(fit[columns], target, sample_weight=weights, verbose=False)
    return recursive_predictions(
        model, panel, columns, specification["transform"],
        forecast_split, history_splits,
    )


def validation(module, frames) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    rows = []
    for candidate, specification in CANDIDATES.items():
        for origin in ORIGINS:
            fit, panel = historical_panel(historical, origin)
            prediction = fit_predict(
                fit, panel, mu.FEAT_COLS, specification, "val", ("train",)
            )
            rows.append({
                "candidate": candidate,
                "origin": origin.strftime("%Y-%m-%d"),
                "validation_rows": int(len(prediction)),
                "validation_series": int(prediction["series_name"].nunique()),
                "absolute_error": float(np.abs(prediction["actual"] - prediction["pred"]).sum()),
                "actual_volume": float(np.abs(prediction["actual"]).sum()),
                "global_volume_weighted_WMAPE": mu.wmape_vol(
                    prediction["actual"], prediction["pred"]
                ),
            })
            print(
                f"[target:{candidate}] {origin:%Y-%m} "
                f"WMAPE={rows[-1]['global_volume_weighted_WMAPE']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("candidate", as_index=False)
        .agg(
            historical_origins=("origin", "nunique"),
            absolute_error=("absolute_error", "sum"),
            actual_volume=("actual_volume", "sum"),
            mean_origin_WMAPE=("global_volume_weighted_WMAPE", "mean"),
            worst_origin_WMAPE=("global_volume_weighted_WMAPE", "max"),
        )
    )
    summary["pooled_global_WMAPE"] = (
        summary["absolute_error"] / summary["actual_volume"] * 100
    )
    summary = summary.sort_values(
        ["pooled_global_WMAPE", "worst_origin_WMAPE", "candidate"]
    ).reset_index(drop=True)
    return detail, summary


def locked_test(frames, versions, selected: str) -> pd.DataFrame:
    specification = CANDIDATES[selected]
    train_val = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    panel = pd.concat(
        [frames["train_roll"], frames["val_roll"], frames["test_fixed"]],
        ignore_index=True,
    ).sort_values(["series_name", "date"])
    rows = []
    for version in ("BASE", "PLATFORM_RATING_FIXED"):
        prediction = fit_predict(
            train_val, panel, versions[version], specification,
            "test", ("train", "val"),
        )
        per_series = mu.wmape_per_series(
            prediction["actual"], prediction["pred"], prediction["series_name"]
        )
        rows.append({
            "candidate": selected,
            "version": version,
            "test_rows": int(len(prediction)),
            "test_series": int(prediction["series_name"].nunique()),
            "global_volume_weighted_WMAPE": mu.wmape_vol(
                prediction["actual"], prediction["pred"]
            ),
            "median_per_series_WMAPE": float(per_series.median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_forecast_module()
    frames, versions, _ = module.build_frames()
    detail, summary = validation(module, frames)
    selected = str(summary.iloc[0]["candidate"])
    detail.merge(
        summary[["candidate", "pooled_global_WMAPE"]], on="candidate", how="left"
    ).to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")
    test = locked_test(frames, versions, selected)
    test.to_csv(TEST_OUTPUT, index=False, encoding="utf-8-sig")
    payload = {
        "schema_version": "v1",
        "selection_protocol": "four historical six-month recursive origins; BASE features only",
        "selection_metric": "pooled global volume-weighted WMAPE",
        "test_used_for_selection": False,
        "selected_candidate": selected,
        "selected_specification": CANDIDATES[selected],
        "validation_candidates": summary.to_dict("records"),
        "locked_test_results": test.to_dict("records"),
        "external_api_calls": 0,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== Target validation =====")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\n===== Locked test =====")
    print(test.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()

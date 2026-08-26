#!/usr/bin/env python3
"""Select and evaluate a conservative cold-start forecast for nine new series.

Selection never uses 2026 test targets.  Candidate methods are evaluated by
rolling launch-cohort backtests: 2024 launches use only earlier launch cohorts,
and 2025 launches use only earlier launch cohorts.  The winning method is then
fit on 2023--2025 launches and applied to the nine 2026 cold-start series.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import _font_setup  # noqa: F401
import _model_utils as mu
from _feature_join import CFG_NUM

STAGE3 = BASE / "data" / "processed_new" / "stage3"
SOURCE_PREDICTIONS = STAGE3 / "xgb_deepseek_full371_preds.csv"
VALIDATION_OUTPUT = STAGE3 / "cold_start_launch_curve_validation.csv"
CANDIDATE_OUTPUT = STAGE3 / "cold_start_candidate_predictions.csv"
SERIES_OUTPUT = STAGE3 / "cold_start_series_results.csv"
HYBRID_OUTPUT = STAGE3 / "xgb_deepseek_cold_hybrid_preds.csv"
SUMMARY_OUTPUT = STAGE3 / "cold_start_launch_curve_summary.json"
FIGURE = BASE / "figures_new" / "cold_start_launch_curve.png"

SOURCE_VERSION = "DEEPSEEK_RICH_FIXED"
HYBRID_VERSION = "DEEPSEEK_RICH_FIXED_COLD_HYBRID"
VALIDATION_YEARS = (2024, 2025)
TRAIN_LAUNCH_YEAR_MIN = 2023
FORECAST_HORIZON = 6
RANDOM_SEED = 42
NUMERIC_FEATURES = list(CFG_NUM)
CATEGORICAL_FEATURES = ["category"]


def build_launch_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, validation, test = mu.load_splits()
    panel = pd.concat([train, validation, test], ignore_index=True).sort_values(["series_name", "date"])
    first = panel.groupby("series_name")["date"].min().rename("first_date")
    panel = panel.merge(first, on="series_name", how="left", validate="many_to_one")
    panel["launch_year"] = panel["first_date"].dt.year
    panel["launch_horizon"] = (
        (panel["date"].dt.year - panel["first_date"].dt.year) * 12
        + panel["date"].dt.month - panel["first_date"].dt.month + 1
    )
    launch = panel.loc[
        panel["launch_year"].between(TRAIN_LAUNCH_YEAR_MIN, 2026)
        & panel["launch_horizon"].between(1, FORECAST_HORIZON)
    ].copy()
    if launch.duplicated(["series_name", "launch_horizon"]).any():
        raise ValueError("Launch panel contains duplicate series/horizon rows")
    historical_series = set(train["series_name"]) | set(validation["series_name"])
    cold_test = test.loc[~test["series_name"].isin(historical_series)].copy()
    if cold_test["series_name"].nunique() != 9 or len(cold_test) != 54:
        raise ValueError("Expected nine cold-start test series and 54 test rows")
    launch_cold = launch.loc[launch["launch_year"].eq(2026)].sort_values(["series_name", "launch_horizon"])
    if set(launch_cold["series_name"]) != set(cold_test["series_name"]):
        raise ValueError("2026 launch cohort does not match cold-start test cohort")
    return launch, cold_test


def horizon_median(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    medians = train.groupby("launch_horizon")[mu.TARGET].median()
    return target["launch_horizon"].map(medians).fillna(train[mu.TARGET].median()).to_numpy(float)


def category_horizon_median(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    horizon = train.groupby("launch_horizon")[mu.TARGET].median()
    grouped = train.groupby(["category", "launch_horizon"])[mu.TARGET].agg(["median", "count"])
    predictions: list[float] = []
    for _, row in target.iterrows():
        key = (row["category"], row["launch_horizon"])
        value = grouped.loc[key, "median"] if key in grouped.index and grouped.loc[key, "count"] >= 5 else horizon[row["launch_horizon"]]
        predictions.append(float(value))
    return np.asarray(predictions)


def model_matrix(train: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [*NUMERIC_FEATURES, "launch_horizon", *CATEGORICAL_FEATURES]
    combined = pd.concat([train[columns], target[columns]], ignore_index=True)
    matrix = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, dummy_na=True, dtype=float)
    return matrix.iloc[:len(train)].reset_index(drop=True), matrix.iloc[len(train):].reset_index(drop=True)


def config_xgb(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    x_train, x_target = model_matrix(train, target)
    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.04, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=RANDOM_SEED, n_jobs=1, objective="reg:squarederror",
    )
    model.fit(x_train, np.log1p(train[mu.TARGET]))
    return np.maximum(np.expm1(model.predict(x_target)), 0.0)


def extra_trees(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    x_train, x_target = model_matrix(train, target)
    model = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=4, max_features=0.8,
        random_state=RANDOM_SEED, n_jobs=1,
    )
    model.fit(x_train, np.log1p(train[mu.TARGET]))
    return np.maximum(np.expm1(model.predict(x_target)), 0.0)


def analog_geomean(train: pd.DataFrame, target: pd.DataFrame, neighbors: int = 10) -> np.ndarray:
    train_series = train.drop_duplicates("series_name").copy()
    target_series = target.drop_duplicates("series_name").copy()
    processor = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    train_matrix = processor.fit_transform(train_series)
    target_matrix = processor.transform(target_series)
    nearest = NearestNeighbors(n_neighbors=min(neighbors, len(train_series))).fit(train_matrix)
    _, indices = nearest.kneighbors(target_matrix)
    prediction_map: dict[tuple[str, int], float] = {}
    for row_index, series_name in enumerate(target_series["series_name"]):
        analog_names = train_series.iloc[indices[row_index]]["series_name"].tolist()
        for horizon in range(1, FORECAST_HORIZON + 1):
            values = train.loc[
                train["series_name"].isin(analog_names) & train["launch_horizon"].eq(horizon), mu.TARGET
            ].to_numpy(float)
            prediction_map[(series_name, horizon)] = float(np.expm1(np.log1p(values).mean())) if len(values) else float(train[mu.TARGET].median())
    return np.asarray([
        prediction_map[(row["series_name"], int(row["launch_horizon"]))]
        for _, row in target.iterrows()
    ])


CANDIDATES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "HORIZON_MEDIAN": horizon_median,
    "CATEGORY_HORIZON_MEDIAN": category_horizon_median,
    "CONFIG_XGB": config_xgb,
    "EXTRA_TREES": extra_trees,
    "ANALOG_10_GEOMEAN": analog_geomean,
}


def validate_candidates(launch: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    prediction_rows: list[pd.DataFrame] = []
    for year in VALIDATION_YEARS:
        train = launch.loc[launch["launch_year"].between(TRAIN_LAUNCH_YEAR_MIN, year - 1)].copy()
        target = launch.loc[launch["launch_year"].eq(year)].copy()
        for method, predictor in CANDIDATES.items():
            predictions = predictor(train, target)
            prediction_rows.append(pd.DataFrame({
                "method": method,
                "validation_launch_year": year,
                "series_name": target["series_name"].to_numpy(),
                "launch_horizon": target["launch_horizon"].to_numpy(),
                "actual": target[mu.TARGET].to_numpy(float),
                "pred": predictions,
            }))
    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    rows: list[dict[str, float | int | str]] = []
    for method, group in all_predictions.groupby("method", sort=True):
        series_wmape = mu.wmape_per_series(group["actual"], group["pred"], group["series_name"])
        row: dict[str, float | int | str] = {
            "method": method,
            "validation_rows": int(len(group)),
            "validation_series": int(group["series_name"].nunique()),
            "validation_global_volume_weighted_WMAPE": mu.wmape_vol(group["actual"], group["pred"]),
            "validation_median_per_series_WMAPE": float(series_wmape.median()),
        }
        for year in VALIDATION_YEARS:
            part = group.loc[group["validation_launch_year"].eq(year)]
            row[f"validation_{year}_global_WMAPE"] = mu.wmape_vol(part["actual"], part["pred"])
        rows.append(row)
    validation = pd.DataFrame(rows).sort_values([
        "validation_global_volume_weighted_WMAPE", "validation_median_per_series_WMAPE", "method",
    ]).reset_index(drop=True)
    return validation, str(validation.iloc[0]["method"])


def final_predictions(launch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    validation, selected = validate_candidates(launch)
    train = launch.loc[launch["launch_year"].between(TRAIN_LAUNCH_YEAR_MIN, 2025)].copy()
    target = launch.loc[launch["launch_year"].eq(2026)].sort_values(["series_name", "launch_horizon"]).copy()
    frames: list[pd.DataFrame] = []
    for method, predictor in CANDIDATES.items():
        predictions = predictor(train, target)
        frame = target[["series_name", "date", "launch_horizon", "brand", "category", mu.TARGET]].copy()
        frame = frame.rename(columns={mu.TARGET: "actual"})
        frame["method"] = method
        frame["pred"] = predictions
        frame["selected_by_validation"] = method == selected
        frames.append(frame)
    return validation, pd.concat(frames, ignore_index=True), selected


def hybrid_results(candidates: pd.DataFrame, selected: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    source = pd.read_csv(SOURCE_PREDICTIONS, low_memory=False, parse_dates=["date"])
    source = source.loc[source["version"].eq(SOURCE_VERSION)].copy()
    chosen = candidates.loc[candidates["method"].eq(selected), ["series_name", "date", "pred"]].rename(
        columns={"pred": "cold_start_pred"}
    )
    hybrid = source.merge(chosen, on=["series_name", "date"], how="left", validate="one_to_one")
    hybrid["source_pred"] = hybrid["pred"]
    hybrid["pred"] = hybrid["cold_start_pred"].fillna(hybrid["source_pred"])
    hybrid["cold_start_method"] = np.where(hybrid["cold_start_pred"].notna(), selected, "historical_series_model")
    hybrid["version"] = HYBRID_VERSION

    cold_source = source.loc[source["cold_start_at_forecast_origin"]].copy()
    cold_hybrid = hybrid.loc[hybrid["cold_start_at_forecast_origin"]].copy()
    series_rows: list[dict[str, float | str]] = []
    for series_name, group in cold_hybrid.groupby("series_name", sort=True):
        old = cold_source.loc[cold_source["series_name"].eq(series_name)]
        series_rows.append({
            "series_name": series_name,
            "actual_six_month_volume": float(group["actual"].sum()),
            "original_prediction_volume": float(old["pred"].sum()),
            "cold_method_prediction_volume": float(group["pred"].sum()),
            "original_WMAPE": mu.wmape_vol(old["actual"], old["pred"]),
            "cold_method_WMAPE": mu.wmape_vol(group["actual"], group["pred"]),
            "WMAPE_improvement_pp": mu.wmape_vol(old["actual"], old["pred"]) - mu.wmape_vol(group["actual"], group["pred"]),
        })
    series = pd.DataFrame(series_rows).sort_values("actual_six_month_volume", ascending=False)
    source_series_wmape = mu.wmape_per_series(source["actual"], source["pred"], source["series_name"])
    hybrid_series_wmape = mu.wmape_per_series(hybrid["actual"], hybrid["pred"], hybrid["series_name"])
    summary: dict[str, float | int | str] = {
        "selected_method": selected,
        "cold_start_series": int(cold_hybrid["series_name"].nunique()),
        "cold_start_rows": int(len(cold_hybrid)),
        "cold_start_actual_volume": float(cold_hybrid["actual"].sum()),
        "original_cold_start_WMAPE": mu.wmape_vol(cold_source["actual"], cold_source["pred"]),
        "selected_method_cold_start_WMAPE": mu.wmape_vol(cold_hybrid["actual"], cold_hybrid["pred"]),
        "cold_start_improvement_pp": mu.wmape_vol(cold_source["actual"], cold_source["pred"]) - mu.wmape_vol(cold_hybrid["actual"], cold_hybrid["pred"]),
        "original_full371_global_WMAPE": mu.wmape_vol(source["actual"], source["pred"]),
        "hybrid_full371_global_WMAPE": mu.wmape_vol(hybrid["actual"], hybrid["pred"]),
        "full371_improvement_pp": mu.wmape_vol(source["actual"], source["pred"]) - mu.wmape_vol(hybrid["actual"], hybrid["pred"]),
        "original_full371_median_per_series_WMAPE": float(source_series_wmape.median()),
        "hybrid_full371_median_per_series_WMAPE": float(hybrid_series_wmape.median()),
    }
    return hybrid, series, summary


def save_figure(series: pd.DataFrame) -> None:
    ordered = series.sort_values("actual_six_month_volume")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.barh(y - 0.25, ordered["actual_six_month_volume"], height=0.25, label="Actual", color="#4C78A8")
    ax.barh(y, ordered["original_prediction_volume"], height=0.25, label="Original zero-history model", color="#BAB0AC")
    ax.barh(y + 0.25, ordered["cold_method_prediction_volume"], height=0.25, label="Validated launch curve", color="#54A24B")
    ax.set_yticks(y, ordered["series_name"])
    ax.set_xlabel("Six-month sales volume")
    ax.set_title("Cold-start compromise for nine 2026 launch series")
    ax.legend(fontsize=8)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main() -> None:
    launch, _ = build_launch_panel()
    validation, candidates, selected = final_predictions(launch)
    hybrid, series, summary = hybrid_results(candidates, selected)
    selected_validation = validation.loc[validation["method"].eq(selected)].iloc[0]
    run_summary = {
        "schema_version": "v1",
        "selection_protocol": "rolling launch-cohort backtest: 2024 and 2025 launches, earlier launch years only",
        "selection_metric": "global volume-weighted WMAPE",
        "test_targets_used_for_method_selection": False,
        "validation_selected_method": selected,
        "selected_validation_global_WMAPE": float(selected_validation["validation_global_volume_weighted_WMAPE"]),
        "selected_validation_median_per_series_WMAPE": float(selected_validation["validation_median_per_series_WMAPE"]),
        **summary,
        "external_api_calls": 0,
    }
    validation.to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")
    candidates.to_csv(CANDIDATE_OUTPUT, index=False, encoding="utf-8-sig")
    series.to_csv(SERIES_OUTPUT, index=False, encoding="utf-8-sig")
    hybrid.to_csv(HYBRID_OUTPUT, index=False, encoding="utf-8-sig")
    SUMMARY_OUTPUT.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_figure(series)

    print("===== Cold-start validation (method selected before test evaluation) =====", flush=True)
    print(validation.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print("\n===== Final cold-start and full-371 impact =====", flush=True)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)
    print("\n===== Nine cold-start series =====", flush=True)
    print(series.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print(f"[output] {SUMMARY_OUTPUT.relative_to(BASE)}", flush=True)
    print(f"[output] {FIGURE.relative_to(BASE)}", flush=True)


if __name__ == "__main__":
    main()

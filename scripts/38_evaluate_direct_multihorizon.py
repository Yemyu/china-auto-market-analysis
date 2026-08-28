#!/usr/bin/env python3
"""Evaluate leakage-safe direct 1--6 month sales forecasts.

The existing headline model forecasts recursively.  This experiment builds one
row per (series, forecast origin, horizon), so all six target months are
predicted directly from information available at the origin.  Validation mode
never scores the 2026 test period.  Test mode must be requested explicitly and
uses the model specification selected by validation mode.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import _model_utils as mu


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "processed" / "forecast"
VALIDATION_RESULTS = OUT / "direct_multihorizon_validation.csv"
VALIDATION_PREDICTIONS = OUT / "direct_multihorizon_validation_predictions.csv.gz"
RUN_SUMMARY = OUT / "direct_multihorizon_summary.json"
TEST_PREDICTIONS = OUT / "direct_multihorizon_test_predictions.csv"
REVIEW_ROLLING = BASE / "data" / "reviews" / "processed" / "review_features_by_series_month_rolling.csv"

HORIZONS = tuple(range(1, 7))
BACKTEST_ORIGINS = tuple(pd.Timestamp(value) for value in (
    "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
))
OFFICIAL_VALIDATION_ORIGIN = pd.Timestamp("2025-07-01")
TEST_ORIGIN = pd.Timestamp("2026-01-01")

STATIC_FEATURES = list(mu.CFG_COLS)
DYNAMIC_FEATURES = [
    "horizon",
    "target_month_sin", "target_month_cos", "target_year",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "seasonal_target_lag12", "seasonal_target_available",
    "roll_mean_3", "roll_mean_6", "roll_mean_12",
    "roll_median_6", "roll_std_6", "roll_std_12",
    "trend_3", "trend_6", "trend_12",
    "history_months", "positive_history_months", "launch_age_months",
    "zero_streak", "recent_zero_rate_6", "recent_nonzero_rate_6",
    "last_vs_mean6", "last_vs_seasonal",
]
FEATURES = DYNAMIC_FEATURES + STATIC_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    name: str
    objective: str
    target_mode: str
    include_reviews: bool = False


MODEL_SPECS = (
    ModelSpec("DIRECT_LOG_LEVEL", "reg:squarederror", "log_level"),
    ModelSpec("DIRECT_LOG_GROWTH", "reg:squarederror", "log_growth"),
    ModelSpec("DIRECT_RAW_SQUARE", "reg:squarederror", "raw"),
    ModelSpec("DIRECT_RAW_ABSOLUTE", "reg:absoluteerror", "raw"),
    ModelSpec("DIRECT_REVIEW_LOG_LEVEL", "reg:squarederror", "log_level", True),
    ModelSpec("DIRECT_REVIEW_LOG_GROWTH", "reg:squarederror", "log_growth", True),
)

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.04,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": 4,
}


def load_panel() -> pd.DataFrame:
    train, validation, test = mu.load_splits()
    panel = pd.concat([train, validation, test], ignore_index=True)
    panel["series_name"] = panel["series_name"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp()
    panel = panel.sort_values(["series_name", "date"]).reset_index(drop=True)
    if panel.duplicated(["series_name", "date"]).any():
        raise ValueError("Sales panel contains duplicate series/month rows")
    return panel


def month_distance(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def slope(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    raw = float(np.polyfit(x, clean, 1)[0])
    return raw / (float(np.mean(np.abs(clean))) + 1.0)


def ending_zero_streak(values: np.ndarray) -> int:
    count = 0
    for value in values[::-1]:
        if float(value) != 0.0:
            break
        count += 1
    return count


def load_review_features() -> tuple[pd.DataFrame, list[str]]:
    review = pd.read_csv(REVIEW_ROLLING, low_memory=False)
    review["series_name"] = review["series_name"].astype(str)
    review["origin_date"] = pd.to_datetime(review["date"]).dt.to_period("M").dt.to_timestamp()
    columns = [column for column in review.columns if column.startswith("review_")]
    if not columns or review.duplicated(["series_name", "origin_date"]).any():
        raise ValueError("Review-origin feature table is empty or duplicated")
    return review[["series_name", "origin_date", *columns]], columns


def build_examples(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    for name, group in panel.groupby("series_name", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        sales_by_date = group.set_index("date")[mu.TARGET].astype(float).to_dict()
        dates = group["date"].tolist()
        values = group[mu.TARGET].astype(float).to_numpy()

        for target_index, target_row in group.iterrows():
            target_date = target_row["date"]
            actual = float(target_row[mu.TARGET])
            for horizon in HORIZONS:
                origin = target_date - pd.DateOffset(months=horizon - 1)
                history_mask = group["date"].lt(origin).to_numpy()
                history_dates = np.asarray(dates, dtype="datetime64[ns]")[history_mask]
                history = values[history_mask]
                if len(history) < 3:
                    continue

                last = float(history[-1])
                seasonal_date = target_date - pd.DateOffset(years=1)
                seasonal = sales_by_date.get(seasonal_date, np.nan)
                positive_indexes = np.flatnonzero(history > 0)
                first_positive_date = (
                    pd.Timestamp(history_dates[positive_indexes[0]]) if len(positive_indexes) else None
                )
                recent6 = history[-6:]

                row: dict[str, Any] = {
                    "series_name": name,
                    "origin_date": origin,
                    "target_date": target_date,
                    "horizon": horizon,
                    "actual": actual,
                    "last_observed": last,
                    "target_month_sin": np.sin(2 * np.pi * target_date.month / 12),
                    "target_month_cos": np.cos(2 * np.pi * target_date.month / 12),
                    "target_year": target_date.year,
                    "lag_1": last,
                    "lag_2": float(history[-2]) if len(history) >= 2 else np.nan,
                    "lag_3": float(history[-3]) if len(history) >= 3 else np.nan,
                    "lag_6": float(history[-6]) if len(history) >= 6 else np.nan,
                    "lag_12": float(history[-12]) if len(history) >= 12 else np.nan,
                    "seasonal_target_lag12": seasonal,
                    "seasonal_target_available": int(np.isfinite(seasonal)),
                    "roll_mean_3": float(np.mean(history[-3:])),
                    "roll_mean_6": float(np.mean(history[-6:])),
                    "roll_mean_12": float(np.mean(history[-12:])),
                    "roll_median_6": float(np.median(history[-6:])),
                    "roll_std_6": float(np.std(history[-6:])),
                    "roll_std_12": float(np.std(history[-12:])),
                    "trend_3": slope(history[-3:]),
                    "trend_6": slope(history[-6:]),
                    "trend_12": slope(history[-12:]),
                    "history_months": len(history),
                    "positive_history_months": int((history > 0).sum()),
                    "launch_age_months": (
                        month_distance(origin, first_positive_date) if first_positive_date is not None else 0
                    ),
                    "zero_streak": ending_zero_streak(history),
                    "recent_zero_rate_6": float(np.mean(recent6 == 0)),
                    "recent_nonzero_rate_6": float(np.mean(recent6 > 0)),
                    "last_vs_mean6": last / (float(np.mean(recent6)) + 1.0),
                    "last_vs_seasonal": last / (float(seasonal) + 1.0) if np.isfinite(seasonal) else np.nan,
                }
                for column in STATIC_FEATURES:
                    row[column] = target_row[column]
                rows.append(row)

    examples = pd.DataFrame(rows)
    if examples.empty or examples.duplicated(["series_name", "origin_date", "horizon"]).any():
        raise ValueError("Direct example construction failed")
    review, review_features = load_review_features()
    examples = examples.merge(
        review,
        on=["series_name", "origin_date"],
        how="left",
        validate="many_to_one",
    )
    return (
        examples.sort_values(["origin_date", "series_name", "horizon"]).reset_index(drop=True),
        review_features,
    )


def features_for_spec(spec: ModelSpec, review_features: list[str]) -> list[str]:
    return FEATURES + review_features if spec.include_reviews else FEATURES


def target_for_fit(frame: pd.DataFrame, spec: ModelSpec) -> np.ndarray:
    actual = frame["actual"].to_numpy(float)
    if spec.target_mode == "raw":
        return actual
    if spec.target_mode == "log_level":
        return np.log1p(actual)
    if spec.target_mode == "log_growth":
        return np.log1p(actual) - np.log1p(frame["last_observed"].to_numpy(float))
    raise ValueError(spec.target_mode)


def predictions_from_model(
    model: XGBRegressor,
    frame: pd.DataFrame,
    spec: ModelSpec,
    review_features: list[str],
) -> np.ndarray:
    columns = features_for_spec(spec, review_features)
    raw = model.predict(frame[columns])
    if spec.target_mode == "raw":
        predictions = raw
    elif spec.target_mode == "log_level":
        predictions = np.expm1(raw)
    elif spec.target_mode == "log_growth":
        predictions = np.expm1(np.log1p(frame["last_observed"].to_numpy(float)) + raw)
    else:
        raise ValueError(spec.target_mode)
    return np.maximum(0.0, predictions)


def fit_predict(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    spec: ModelSpec,
    review_features: list[str],
) -> np.ndarray:
    columns = features_for_spec(spec, review_features)
    model = XGBRegressor(objective=spec.objective, **MODEL_PARAMS)
    model.fit(train[columns], target_for_fit(train, spec), verbose=False)
    return predictions_from_model(model, evaluation, spec, review_features)


def score_rows(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    per_series = mu.wmape_per_series(frame["actual"], predictions, frame["series_name"])
    return {
        "global_volume_weighted_WMAPE": float(mu.wmape_vol(frame["actual"], predictions)),
        "median_per_series_WMAPE": float(per_series.median()),
        "MAE": float(np.mean(np.abs(frame["actual"].to_numpy(float) - predictions))),
    }


def baseline_predictions(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "LAST_VALUE":
        return frame["last_observed"].to_numpy(float)
    if method == "SEASONAL_LAG12":
        return frame["seasonal_target_lag12"].fillna(frame["last_observed"]).to_numpy(float)
    if method == "ROLLING_MEAN_6":
        return frame["roll_mean_6"].to_numpy(float)
    raise ValueError(method)


def build_prevalidation_ensembles(
    predictions: pd.DataFrame,
    base_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    index_columns = ["series_name", "origin_date", "target_date", "horizon", "actual"]
    wide = predictions.pivot_table(index=index_columns, columns="method", values="pred").reset_index()
    early = wide.loc[wide["origin_date"].lt(OFFICIAL_VALIDATION_ORIGIN)].copy()
    direct_methods = [spec.name for spec in MODEL_SPECS]
    baseline_methods = ["LAST_VALUE", "SEASONAL_LAG12", "ROLLING_MEAN_6"]
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    configs: dict[str, dict[str, Any]] = {}

    for direct_method in direct_methods:
        for baseline_method in baseline_methods:
            method = f"ENSEMBLE_{direct_method}_WITH_{baseline_method}"
            weights: dict[int, float] = {}
            for horizon in HORIZONS:
                part = early.loc[early["horizon"].eq(horizon)]
                candidates: list[tuple[float, float]] = []
                for direct_weight in np.arange(0.0, 1.001, 0.1):
                    blend = (
                        direct_weight * part[direct_method].to_numpy(float)
                        + (1.0 - direct_weight) * part[baseline_method].to_numpy(float)
                    )
                    candidates.append((mu.wmape_vol(part["actual"], blend), float(direct_weight)))
                weights[horizon] = min(candidates, key=lambda item: (item[0], -item[1]))[1]

            configs[method] = {
                "direct_method": direct_method,
                "baseline_method": baseline_method,
                "direct_weight_by_horizon": {str(key): value for key, value in weights.items()},
                "weight_selection_origins": [
                    origin.strftime("%Y-%m-%d")
                    for origin in BACKTEST_ORIGINS
                    if origin < OFFICIAL_VALIDATION_ORIGIN
                ],
            }

            for origin, part in wide.groupby("origin_date", sort=True):
                blend = np.asarray([
                    weights[int(horizon)] * direct + (1.0 - weights[int(horizon)]) * baseline
                    for horizon, direct, baseline in zip(
                        part["horizon"], part[direct_method], part[baseline_method],
                    )
                ])
                scores = score_rows(part, blend)
                reference = base_results.loc[
                    base_results["origin"].eq(origin) & base_results["method"].eq(direct_method)
                ].iloc[0]
                result_rows.append({
                    "origin": origin,
                    "official_validation": origin == OFFICIAL_VALIDATION_ORIGIN,
                    "method": method,
                    "train_rows": int(reference["train_rows"]),
                    "evaluation_rows": len(part),
                    "evaluation_series": part["series_name"].nunique(),
                    **scores,
                })
                pred = part[index_columns].copy()
                pred["method"] = method
                pred["pred"] = blend
                prediction_frames.append(pred)

    return pd.DataFrame(result_rows), pd.concat(prediction_frames, ignore_index=True), configs


def validation_run(
    examples: pd.DataFrame,
    review_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    methods = [spec.name for spec in MODEL_SPECS] + ["LAST_VALUE", "SEASONAL_LAG12", "ROLLING_MEAN_6"]

    for origin in BACKTEST_ORIGINS:
        train = examples.loc[examples["target_date"].lt(origin)].copy()
        evaluation = examples.loc[examples["origin_date"].eq(origin)].copy()
        if train.empty or evaluation.empty:
            raise ValueError(f"No direct examples available for origin {origin:%Y-%m}")
        print(
            f"[direct] origin={origin:%Y-%m} train={len(train):,} "
            f"eval={len(evaluation):,} series={evaluation['series_name'].nunique()}",
            flush=True,
        )

        for method in methods:
            if method in {spec.name for spec in MODEL_SPECS}:
                spec = next(item for item in MODEL_SPECS if item.name == method)
                predictions = fit_predict(train, evaluation, spec, review_features)
            else:
                predictions = baseline_predictions(evaluation, method)
            scores = score_rows(evaluation, predictions)
            result_rows.append({
                "origin": origin,
                "official_validation": origin == OFFICIAL_VALIDATION_ORIGIN,
                "method": method,
                "train_rows": len(train),
                "evaluation_rows": len(evaluation),
                "evaluation_series": evaluation["series_name"].nunique(),
                **scores,
            })
            pred = evaluation[["series_name", "origin_date", "target_date", "horizon", "actual"]].copy()
            pred["method"] = method
            pred["pred"] = predictions
            prediction_frames.append(pred)
            print(
                f"[direct:{method}] WMAPE={scores['global_volume_weighted_WMAPE']:.3f} "
                f"median={scores['median_per_series_WMAPE']:.3f}",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble_results, ensemble_predictions, ensemble_configs = build_prevalidation_ensembles(
        predictions, results,
    )
    results = pd.concat([results, ensemble_results], ignore_index=True)
    predictions = pd.concat([predictions, ensemble_predictions], ignore_index=True)
    official = results.loc[results["official_validation"]].copy()
    historical = (
        results.loc[~results["official_validation"]]
        .groupby("method", as_index=False)
        .agg(
            historical_backtest_mean_WMAPE=("global_volume_weighted_WMAPE", "mean"),
            historical_backtest_std_WMAPE=("global_volume_weighted_WMAPE", "std"),
        )
    )
    selection = official.merge(historical, on="method", how="left")
    selectable_methods = [spec.name for spec in MODEL_SPECS] + list(ensemble_configs)
    model_selection = selection.loc[selection["method"].isin(selectable_methods)].copy()
    winner = model_selection.sort_values([
        "global_volume_weighted_WMAPE", "historical_backtest_mean_WMAPE", "method",
    ]).iloc[0]
    selected_ensemble = ensemble_configs.get(winner["method"])
    selected_direct_method = (
        selected_ensemble["direct_method"] if selected_ensemble else winner["method"]
    )
    selected_spec = next(item for item in MODEL_SPECS if item.name == selected_direct_method)
    summary = {
        "schema_version": "v1",
        "test_evaluated": False,
        "selection_protocol": "ensemble horizon weights selected only on 2024-01/2024-07/2025-01 origins; model family selected on official 2025-07 fixed-origin WMAPE",
        "backtest_origins": [origin.strftime("%Y-%m-%d") for origin in BACKTEST_ORIGINS],
        "selected_method": winner["method"],
        "selected_validation_global_WMAPE": float(winner["global_volume_weighted_WMAPE"]),
        "selected_historical_backtest_mean_WMAPE": float(winner["historical_backtest_mean_WMAPE"]),
        "selected_ensemble": selected_ensemble,
        "ensemble_configs": ensemble_configs,
        "feature_count": len(features_for_spec(selected_spec, review_features)),
        "features": features_for_spec(selected_spec, review_features),
        "model_params": MODEL_PARAMS,
        "external_api_calls": 0,
    }
    return results, predictions, summary


def test_run(
    examples: pd.DataFrame,
    review_features: list[str],
    summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_name = summary["selected_method"]
    ensemble = summary.get("selected_ensemble")
    direct_name = ensemble["direct_method"] if ensemble else selected_name
    spec = next((item for item in MODEL_SPECS if item.name == direct_name), None)
    if spec is None:
        raise ValueError("Validation summary did not select a supported direct model")
    train = examples.loc[examples["target_date"].lt(TEST_ORIGIN)].copy()
    evaluation = examples.loc[examples["origin_date"].eq(TEST_ORIGIN)].copy()
    direct_predictions = fit_predict(train, evaluation, spec, review_features)
    if ensemble:
        baseline = baseline_predictions(evaluation, ensemble["baseline_method"])
        weights = {int(key): float(value) for key, value in ensemble["direct_weight_by_horizon"].items()}
        predictions = np.asarray([
            weights[int(horizon)] * direct + (1.0 - weights[int(horizon)]) * base
            for horizon, direct, base in zip(evaluation["horizon"], direct_predictions, baseline)
        ])
    else:
        predictions = direct_predictions
    output = evaluation[[
        "series_name", "origin_date", "target_date", "horizon", "actual",
        "history_months", "last_observed", "seasonal_target_lag12",
    ]].copy()
    output["method"] = selected_name
    output["pred"] = predictions
    scores = score_rows(output.rename(columns={"pred": "unused"}), predictions)
    summary = dict(summary)
    summary.update({
        "test_evaluated": True,
        "test_origin": TEST_ORIGIN.strftime("%Y-%m-%d"),
        "direct_test_rows": len(output),
        "direct_test_series": output["series_name"].nunique(),
        "direct_test_global_WMAPE": scores["global_volume_weighted_WMAPE"],
        "direct_test_median_per_series_WMAPE": scores["median_per_series_WMAPE"],
    })
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the validation-selected specification on the 2026 test origin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    examples, review_features = build_examples(panel)
    print(
        f"[direct] examples={len(examples):,} series={examples['series_name'].nunique()} "
        f"features={len(FEATURES)}",
        flush=True,
    )

    if not args.evaluate_test:
        results, predictions, summary = validation_run(examples, review_features)
        results.to_csv(VALIDATION_RESULTS, index=False, encoding="utf-8-sig")
        predictions.to_csv(VALIDATION_PREDICTIONS, index=False, encoding="utf-8-sig")
        RUN_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[direct] validation winner={summary['selected_method']}", flush=True)
        print(f"[output] {VALIDATION_RESULTS.relative_to(BASE)}", flush=True)
        return

    if not RUN_SUMMARY.exists():
        raise FileNotFoundError("Run validation mode before --evaluate-test")
    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("test_evaluated"):
        raise RuntimeError("Test has already been evaluated for this validation summary")
    predictions, summary = test_run(examples, review_features, summary)
    predictions.to_csv(TEST_PREDICTIONS, index=False, encoding="utf-8-sig")
    RUN_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[direct:{summary['selected_method']}] test WMAPE={summary['direct_test_global_WMAPE']:.3f}",
        flush=True,
    )
    print(f"[output] {TEST_PREDICTIONS.relative_to(BASE)}", flush=True)


if __name__ == "__main__":
    main()

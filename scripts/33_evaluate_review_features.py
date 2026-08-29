#!/usr/bin/env python3
"""Evaluate review-feature variants on the 371-series forecast panel."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, __version__ as xgboost_version

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _font_setup  # noqa: F401
import _model_utils as mu


BASE = Path(__file__).resolve().parents[1]
SENTIMENT = BASE / "data" / "reviews" / "processed"
LOCAL_ROLLING = SENTIMENT / "sentiment_features_by_series_month.csv"
REVIEW_FIXED = SENTIMENT / "review_features_by_series_month_fixed_origin.csv"
REVIEW_ROLLING = SENTIMENT / "review_features_by_series_month_rolling.csv"
OUT = BASE / "data" / "processed" / "forecast"
FIG = BASE / "assets/analysis"

SUMMARY = OUT / "review_feature_ablation_summary.csv"
VALIDATION_GRID = OUT / "review_feature_validation_grid.csv"
PREDICTIONS = OUT / "review_feature_predictions.csv"
SERIES_METRICS = OUT / "review_feature_series_metrics.csv"
COVERAGE = OUT / "review_feature_coverage.csv"
FEATURE_MANIFEST = OUT / "review_feature_manifest.csv"
RUN_SUMMARY = OUT / "review_feature_run_summary.json"
FIGURE = FIG / "forecast_review_feature_ablation.png"

EXPECTED_SERIES = 371
TREE_GRID = [50, 100, 200, 400, 700]
MODEL_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
    "n_jobs": 1,
}

LOCAL_CONTEXT = [
    "sentiment_review_count_prior_all", "sentiment_review_count_180d",
    "sentiment_available_prior", "sentiment_available_180d",
]


def unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["series_name"] = out["series_name"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    return out


def read_external_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    for path in (LOCAL_ROLLING, REVIEW_FIXED, REVIEW_ROLLING):
        if not path.exists():
            raise FileNotFoundError(path)
    local = normalize_dates(pd.read_csv(LOCAL_ROLLING, low_memory=False))
    review_fixed = normalize_dates(pd.read_csv(REVIEW_FIXED, low_memory=False))
    review_rolling = normalize_dates(pd.read_csv(REVIEW_ROLLING, low_memory=False))
    for name, table in (("local", local), ("review_fixed", review_fixed), ("review_rolling", review_rolling)):
        if table.duplicated(["series_name", "date"]).any():
            raise ValueError(f"{name} table contains duplicate series/month rows")
    local_features = [column for column in local.columns if column not in ("series_name", "date")]
    review_features = [column for column in review_fixed.columns if column.startswith("review_")]
    if set(review_features) != {column for column in review_rolling.columns if column.startswith("review_")}:
        raise ValueError("Fixed and rolling review-feature schemas differ")
    return local, review_fixed, review_rolling, local_features, review_features


def attach_by_month(frame: pd.DataFrame, table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    base = normalize_dates(frame)
    before = len(base)
    out = base.merge(table[["series_name", "date", *columns]], on=["series_name", "date"], how="left", validate="one_to_one")
    if len(out) != before:
        raise ValueError("Monthly external join changed split row count")
    return out


def attach_anchor(frame: pd.DataFrame, table: pd.DataFrame, columns: list[str], origin: str) -> pd.DataFrame:
    base = normalize_dates(frame)
    anchor_month = pd.Timestamp(origin)
    anchor = table.loc[table["date"].eq(anchor_month), ["series_name", *columns]].copy()
    if anchor["series_name"].duplicated().any():
        raise ValueError(f"Duplicate external anchor rows at {origin}")
    before = len(base)
    out = base.merge(anchor, on="series_name", how="left", validate="many_to_one")
    if len(out) != before:
        raise ValueError("Anchored external join changed split row count")
    return out


def build_frames() -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], pd.DataFrame]:
    local, review_fixed, review_rolling, local_features, review_features = read_external_tables()
    train, validation, test = mu.load_splits()
    train = normalize_dates(train)
    validation = normalize_dates(validation)
    test = normalize_dates(test)
    if test["series_name"].nunique() != EXPECTED_SERIES:
        raise ValueError("Test split no longer contains all 371 series")

    train_roll = attach_by_month(train, local, local_features)
    train_roll = attach_by_month(train_roll, review_rolling, review_features)
    val_roll = attach_by_month(validation, local, local_features)
    val_roll = attach_by_month(val_roll, review_rolling, review_features)
    test_roll = attach_by_month(test, local, local_features)
    test_roll = attach_by_month(test_roll, review_rolling, review_features)

    val_fixed = attach_anchor(validation, local, local_features, "2025-07-01")
    val_fixed = attach_by_month(val_fixed, review_fixed, review_features)
    test_fixed = attach_anchor(test, local, local_features, "2026-01-01")
    test_fixed = attach_by_month(test_fixed, review_fixed, review_features)

    platform = [column for column in local_features if column.startswith("platform_rating_")]
    local_text = [
        column for column in local_features
        if column.startswith("text_") and (
            column.endswith("_polarity_180d_mean") or column.endswith("_mentioned_180d_count")
        )
    ]
    review_context = [
        "review_count_prior_all", "review_count_180d",
        "review_available_prior", "review_available_180d",
    ]
    review_core = unique(review_context + [
        column for column in review_features
        if column.endswith("_score_prior_mean")
        or column.endswith("_score_180d_mean")
        or column in (
            "review_overall_aspect_score_180d_mean",
            "review_any_positive_180d_rate",
            "review_any_negative_180d_rate",
        )
    ])
    if not platform or not local_text or not review_core:
        raise ValueError("One or more sentiment feature families are empty")

    candidate_families = {
        "platform_rating": platform,
        "local_context": LOCAL_CONTEXT,
        "local_lexicon": local_text,
        "review_core": review_core,
        "review_rich": review_features,
    }
    manifest_rows: list[dict[str, Any]] = []
    usable_by_family: dict[str, list[str]] = {}
    for family, columns in candidate_families.items():
        usable_by_family[family] = []
        for column in columns:
            nonmissing = int(train_roll[column].notna().sum())
            keep = nonmissing > 0
            manifest_rows.append({
                "feature_family": family,
                "feature_name": column,
                "training_nonmissing_rows": nonmissing,
                "included_in_model": keep,
                "exclusion_reason": "" if keep else "all values missing in train split",
            })
            if keep:
                usable_by_family[family].append(column)
    manifest = pd.DataFrame(manifest_rows).drop_duplicates(["feature_family", "feature_name"])
    empty_families = [family for family, columns in usable_by_family.items() if not columns]
    if empty_families:
        raise ValueError(f"Entire sentiment feature families are empty: {empty_families}")
    platform = usable_by_family["platform_rating"]
    local_context = usable_by_family["local_context"]
    local_text = usable_by_family["local_lexicon"]
    review_core = usable_by_family["review_core"]
    review_features = usable_by_family["review_rich"]

    versions = {
        "BASE": list(mu.FEAT_COLS),
        "PLATFORM_RATING_FIXED": unique(list(mu.FEAT_COLS) + local_context + platform),
        "LOCAL_LEXICON_FIXED": unique(list(mu.FEAT_COLS) + local_context + local_text),
        "REVIEW_TEXT_FIXED": unique(list(mu.FEAT_COLS) + review_core),
        "REVIEW_RICH_FIXED": unique(list(mu.FEAT_COLS) + review_features),
        "ALL_SENTIMENT_FIXED": unique(list(mu.FEAT_COLS) + platform + local_text + review_core),
    }
    frames = {
        "train_roll": train_roll,
        "val_roll": val_roll,
        "val_fixed": val_fixed,
        "test_roll": test_roll,
        "test_fixed": test_fixed,
    }
    required = set(column for columns in versions.values() for column in columns)
    for name, frame in frames.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing model columns: {sorted(missing)}")
    return frames, versions, manifest


def new_model(n_estimators: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=n_estimators, **MODEL_PARAMS)


def recursive_forecast_allow_cold_start(
    model: XGBRegressor,
    series: pd.DataFrame,
    columns: list[str],
    forecast_split: str,
    history_splits: tuple[str, ...],
) -> dict[pd.Timestamp, float]:
    """Match the shared recursive forecast and explicitly support zero-history series."""
    ordered = series.sort_values("date").reset_index(drop=True)
    history = ordered.loc[ordered["split"].isin(history_splits), mu.TARGET].astype(float).tolist()
    predictions: dict[pd.Timestamp, float] = {}
    for _, record in ordered.loc[ordered["split"].eq(forecast_split)].iterrows():
        date = record["date"]
        values = np.asarray(history, dtype=float)
        row: dict[str, Any] = {
            "lag_1": values[-1] if len(values) >= 1 else 0.0,
            "lag_2": values[-2] if len(values) >= 2 else 0.0,
            "lag_3": values[-3] if len(values) >= 3 else 0.0,
            "roll_mean_3": float(np.mean(values[-3:])) if len(values) >= 1 else 0.0,
            "roll_mean_6": float(np.mean(values[-6:])) if len(values) >= 1 else 0.0,
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),
            "year": date.year,
        }
        for column in mu.CFG_COLS:
            row[column] = record[column]
        for column in columns:
            if column not in row:
                if column not in record.index:
                    raise KeyError(f"Feature '{column}' is missing from the forecasting panel")
                row[column] = record[column]
        prediction = float(np.expm1(model.predict(pd.DataFrame([row], columns=columns))[0]))
        prediction = max(prediction, 0.0)
        predictions[date] = prediction
        history.append(prediction)
    return predictions


def recursive_predictions(
    model: XGBRegressor,
    panel: pd.DataFrame,
    columns: list[str],
    forecast_split: str,
    history_splits: tuple[str, ...],
    version: str,
    scenario: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, series in panel.groupby("series_name", sort=True):
        cold_start = not series["split"].isin(history_splits).any()
        forecast = recursive_forecast_allow_cold_start(
            model, series, columns, forecast_split, history_splits,
        )
        actual_rows = series.loc[series["split"].eq(forecast_split)].sort_values("date")
        for _, row in actual_rows.iterrows():
            pred = forecast.get(row["date"], np.nan)
            if not np.isfinite(pred):
                raise ValueError(f"Missing {forecast_split} prediction for {version}: {name} / {row['date']}")
            rows.append({
                "version": version,
                "scenario": scenario,
                "series_name": name,
                "date": row["date"],
                "actual": float(row[mu.TARGET]),
                "pred": float(pred),
                "cold_start_at_forecast_origin": cold_start,
                "review_available_prior": int(row["review_available_prior"]),
                "review_available_180d": int(row["review_available_180d"]),
            })
    result = pd.DataFrame(rows)
    expected = panel.loc[panel["split"].eq(forecast_split), ["series_name", "date"]]
    if len(result) != len(expected) or result.duplicated(["series_name", "date"]).any():
        raise ValueError(f"Prediction coverage mismatch for {version} / {forecast_split}")
    return result


def select_trees(
    version: str,
    columns: list[str],
    train: pd.DataFrame,
    validation_panel: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for n_estimators in TREE_GRID:
        model = new_model(n_estimators)
        model.fit(train[columns], np.log1p(train[mu.TARGET]), verbose=False)
        predictions = recursive_predictions(
            model, validation_panel, columns, "val", ("train",), version, "fixed_origin_validation",
        )
        series_wmape = mu.wmape_per_series(predictions["actual"], predictions["pred"], predictions["series_name"])
        rows.append({
            "version": version,
            "n_estimators": n_estimators,
            "validation_rows": int(len(predictions)),
            "validation_series": int(predictions["series_name"].nunique()),
            "validation_global_volume_weighted_WMAPE": mu.wmape_vol(predictions["actual"], predictions["pred"]),
            "validation_median_per_series_WMAPE": float(series_wmape.median()),
        })
        print(
            f"[full371:{version}] validation trees={n_estimators} "
            f"global_WMAPE={rows[-1]['validation_global_volume_weighted_WMAPE']:.3f}",
            flush=True,
        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["validation_global_volume_weighted_WMAPE", "n_estimators"]).iloc[0]
    return int(best["n_estimators"]), grid


def fit_final_and_predict(
    version: str,
    columns: list[str],
    n_estimators: int,
    train_val_roll: pd.DataFrame,
    test_panel: pd.DataFrame,
    scenario: str,
) -> tuple[XGBRegressor, pd.DataFrame]:
    model = new_model(n_estimators)
    model.fit(train_val_roll[columns], np.log1p(train_val_roll[mu.TARGET]), verbose=False)
    predictions = recursive_predictions(
        model, test_panel, columns, "test", ("train", "val"), version, scenario,
    )
    return model, predictions


def series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (version, scenario, name), group in predictions.groupby(["version", "scenario", "series_name"], sort=True):
        metric = mu.metrics(group["actual"], group["pred"])
        metric.update({
            "version": version,
            "scenario": scenario,
            "series_name": name,
            "test_months": int(len(group)),
            "actual_volume": float(np.abs(group["actual"]).sum()),
        })
        rows.append(metric)
    return pd.DataFrame(rows)


def summary_table(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    versions: dict[str, list[str]],
    selected_trees: dict[str, int],
    validation_scores: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (version, scenario), group in predictions.groupby(["version", "scenario"], sort=False):
        version_metrics = metrics.loc[metrics["version"].eq(version)]
        source_version = "REVIEW_TEXT_FIXED" if version == "REVIEW_TEXT_ROLLING" else version
        rows.append({
            "version": version,
            "scenario": scenario,
            "primary_fixed_origin_comparison": scenario == "fixed_origin_primary",
            "n_features": len(versions[source_version]),
            "validation_selected_n_estimators": selected_trees[source_version],
            "fixed_origin_validation_global_WMAPE": validation_scores[source_version],
            "test_rows": int(len(group)),
            "test_series": int(group["series_name"].nunique()),
            "test_actual_volume": float(np.abs(group["actual"]).sum()),
            "global_volume_weighted_WMAPE": mu.wmape_vol(group["actual"], group["pred"]),
            "median_per_series_WMAPE": float(version_metrics["WMAPE"].median()),
            "mean_per_series_WMAPE": float(version_metrics["WMAPE"].mean()),
            "test_start": group["date"].min().strftime("%Y-%m-%d"),
            "test_end": group["date"].max().strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def coverage_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (version, scenario), version_rows in predictions.groupby(["version", "scenario"], sort=False):
        groups = {
            "all_test_rows": pd.Series(True, index=version_rows.index),
            "historical_series": ~version_rows["cold_start_at_forecast_origin"],
            "cold_start_series": version_rows["cold_start_at_forecast_origin"],
            "any_prior_review": version_rows["review_available_prior"].eq(1),
            "no_prior_review": version_rows["review_available_prior"].eq(0),
            "recent_180d_review": version_rows["review_available_180d"].eq(1),
            "no_recent_180d_review": version_rows["review_available_180d"].eq(0),
        }
        for group_name, mask in groups.items():
            part = version_rows.loc[mask]
            if part.empty:
                continue
            per_series = mu.wmape_per_series(part["actual"], part["pred"], part["series_name"])
            rows.append({
                "version": version,
                "scenario": scenario,
                "coverage_group": group_name,
                "test_rows": int(len(part)),
                "test_series": int(part["series_name"].nunique()),
                "actual_volume": float(np.abs(part["actual"]).sum()),
                "global_volume_weighted_WMAPE": mu.wmape_vol(part["actual"], part["pred"]),
                "median_per_series_WMAPE": float(per_series.median()),
            })
    return pd.DataFrame(rows)


def save_figure(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values(["primary_fixed_origin_comparison", "global_volume_weighted_WMAPE"], ascending=[False, True])
    labels = ordered["version"].str.replace("_FIXED", "", regex=False).str.replace("_", " ", regex=False)
    colors = ["#4C78A8" if primary else "#F58518" for primary in ordered["primary_fixed_origin_comparison"]]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    ax.bar(x - 0.18, ordered["global_volume_weighted_WMAPE"], width=0.36, color=colors)
    ax.bar(x + 0.18, ordered["median_per_series_WMAPE"], width=0.36, color="#B8C4CE")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("WMAPE (%)")
    ax.set_title("Full 371-series XGBoost sentiment ablation (test 2026-01..06)")
    ax.legend(["Global volume-weighted", "Per-series median"], fontsize=8)
    for index, value in enumerate(ordered["global_volume_weighted_WMAPE"]):
        ax.text(index - 0.18, value + 0.4, f"{value:.2f}", ha="center", fontsize=7)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frames, versions, manifest = build_frames()
    manifest.to_csv(FEATURE_MANIFEST, index=False, encoding="utf-8-sig")

    train = frames["train_roll"]
    val_roll = frames["val_roll"]
    val_fixed = frames["val_fixed"]
    test_roll = frames["test_roll"]
    test_fixed = frames["test_fixed"]
    validation_panel = pd.concat([train, val_fixed], ignore_index=True).sort_values(["series_name", "date"])
    train_val_roll = pd.concat([train, val_roll], ignore_index=True).sort_values(["series_name", "date"])
    fixed_test_panel = pd.concat([train, val_roll, test_fixed], ignore_index=True).sort_values(["series_name", "date"])
    rolling_test_panel = pd.concat([train, val_roll, test_roll], ignore_index=True).sort_values(["series_name", "date"])

    print(
        f"[full371] train={len(train):,} val={len(val_roll):,} test={len(test_fixed):,} "
        f"test_series={test_fixed['series_name'].nunique()} versions={len(versions)}",
        flush=True,
    )
    selected_trees: dict[str, int] = {}
    validation_scores: dict[str, float] = {}
    grids: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    final_models: dict[str, XGBRegressor] = {}

    for version, columns in versions.items():
        print(f"[full371:{version}] selecting capacity with {len(columns)} features", flush=True)
        best_trees, grid = select_trees(version, columns, train, validation_panel)
        selected_trees[version] = best_trees
        validation_scores[version] = float(
            grid.loc[grid["n_estimators"].eq(best_trees), "validation_global_volume_weighted_WMAPE"].iloc[0]
        )
        grids.append(grid)
        model, predictions = fit_final_and_predict(
            version, columns, best_trees, train_val_roll, fixed_test_panel, "fixed_origin_primary",
        )
        final_models[version] = model
        prediction_frames.append(predictions)
        print(
            f"[full371:{version}] selected={best_trees} "
            f"test_global_WMAPE={mu.wmape_vol(predictions['actual'], predictions['pred']):.3f}",
            flush=True,
        )

    core_version = "REVIEW_TEXT_FIXED"
    rolling_predictions = recursive_predictions(
        final_models[core_version],
        rolling_test_panel,
        versions[core_version],
        "test",
        ("train", "val"),
        "REVIEW_TEXT_ROLLING",
        "rolling_origin_supplement",
    )
    prediction_frames.append(rolling_predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["version", "series_name", "date"])
    metrics = series_metrics(predictions)
    summary = summary_table(predictions, metrics, versions, selected_trees, validation_scores)
    coverage = coverage_table(predictions)
    grid = pd.concat(grids, ignore_index=True)

    predictions.to_csv(PREDICTIONS, index=False, encoding="utf-8-sig")
    metrics.to_csv(SERIES_METRICS, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    coverage.to_csv(COVERAGE, index=False, encoding="utf-8-sig")
    grid.to_csv(VALIDATION_GRID, index=False, encoding="utf-8-sig")
    save_figure(summary)

    primary = summary.loc[summary["primary_fixed_origin_comparison"]].copy()
    validation_selected = primary.sort_values(
        ["fixed_origin_validation_global_WMAPE", "version"]
    ).iloc[0]
    descriptive_test_best = primary.sort_values(
        ["global_volume_weighted_WMAPE", "version"]
    ).iloc[0]
    run_summary = {
        "schema_version": "v1",
        "evaluation_series": int(test_fixed["series_name"].nunique()),
        "validation_protocol": "recursive six-month fixed-origin forecast; select n_estimators by global volume-weighted WMAPE",
        "test_protocol": "refit on point-in-time train+validation rows; recursive fixed-origin 2026-01..06 forecast",
        "primary_versions": primary.sort_values("version")["version"].tolist(),
        "primary_version_selection_metric": "fixed-origin validation global volume-weighted WMAPE",
        "validation_selected_primary_version": validation_selected["version"],
        "validation_selected_primary_validation_WMAPE": float(
            validation_selected["fixed_origin_validation_global_WMAPE"]
        ),
        "validation_selected_primary_test_WMAPE": float(
            validation_selected["global_volume_weighted_WMAPE"]
        ),
        "descriptive_best_test_version": descriptive_test_best["version"],
        "descriptive_best_test_WMAPE": float(
            descriptive_test_best["global_volume_weighted_WMAPE"]
        ),
        "best_primary_version": validation_selected["version"],
        "best_primary_global_volume_weighted_WMAPE": float(
            validation_selected["global_volume_weighted_WMAPE"]
        ),
        "baseline_global_volume_weighted_WMAPE": float(
            summary.loc[summary["version"].eq("BASE"), "global_volume_weighted_WMAPE"].iloc[0]
        ),
        "rolling_scenario_is_supplemental": True,
        "tree_grid": TREE_GRID,
        "model_params": MODEL_PARAMS,
        "xgboost_version": xgboost_version,
        "external_api_calls": 0,
    }
    RUN_SUMMARY.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Full 371-series XGBoost sentiment ablation: TEST only =====", flush=True)
    print(
        summary[[
            "version", "scenario", "n_features", "validation_selected_n_estimators",
            "global_volume_weighted_WMAPE", "median_per_series_WMAPE",
        ]].sort_values(["scenario", "global_volume_weighted_WMAPE"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"),
        flush=True,
    )
    print(f"[output] {SUMMARY.relative_to(BASE)}", flush=True)
    print(f"[output] {FIGURE.relative_to(BASE)}", flush=True)


if __name__ == "__main__":
    main()

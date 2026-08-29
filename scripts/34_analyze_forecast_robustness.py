#!/usr/bin/env python3
"""Run robustness, feature-contribution, and error diagnostics."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from xgboost import DMatrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
ablation = importlib.import_module("33_evaluate_review_features")
mu = ablation.mu

FORECAST_DIR = BASE / "data" / "processed" / "forecast"
PREDICTIONS = FORECAST_DIR / "review_feature_predictions.csv"
ABLATION_SUMMARY = FORECAST_DIR / "review_feature_ablation_summary.csv"
TEST_SPLIT = BASE / "data" / "processed" / "splits" / "test.csv"

BOOTSTRAP_OUTPUT = FORECAST_DIR / "forecast_robustness_bootstrap.csv"
SHAP_OUTPUT = FORECAST_DIR / "review_feature_shap_importance.csv"
FAMILY_OUTPUT = FORECAST_DIR / "review_feature_family_importance.csv"
MONTHLY_OUTPUT = FORECAST_DIR / "forecast_monthly_stability.csv"
SERIES_OUTPUT = FORECAST_DIR / "forecast_series_diagnostics.csv"
SEGMENT_OUTPUT = FORECAST_DIR / "forecast_error_segments.csv"
SUMMARY_OUTPUT = FORECAST_DIR / "forecast_robustness_summary.json"
FIGURE = BASE / "assets/analysis" / "forecast_robustness.png"
MODEL_RUN_SUMMARY = FORECAST_DIR / "review_feature_run_summary.json"

BASE_VERSION = "BASE"
PLATFORM_VERSION = "PLATFORM_RATING_FIXED"
model_run = json.loads(MODEL_RUN_SUMMARY.read_text(encoding="utf-8"))
BEST_VERSION = model_run.get(
    "validation_selected_primary_version", model_run["best_primary_version"]
)
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 42


def wmape_from_totals(error: np.ndarray, volume: np.ndarray) -> np.ndarray:
    denominator = volume.sum(axis=1)
    return np.divide(error.sum(axis=1), denominator, out=np.full(len(error), np.nan), where=denominator > 0) * 100


def series_totals(predictions: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    versions = sorted(predictions["version"].unique())
    series_names = sorted(predictions["series_name"].unique())
    volume_reference: np.ndarray | None = None
    errors: dict[str, np.ndarray] = {}
    for version in versions:
        part = predictions.loc[predictions["version"].eq(version)].copy()
        grouped = part.assign(abs_error=(part["actual"] - part["pred"]).abs()).groupby("series_name").agg(
            actual_volume=("actual", lambda values: float(np.abs(values).sum())),
            absolute_error=("abs_error", "sum"),
        ).reindex(series_names)
        volume = grouped["actual_volume"].to_numpy(float)
        if volume_reference is None:
            volume_reference = volume
        elif not np.allclose(volume_reference, volume):
            raise ValueError("Actual series volumes differ across model versions")
        errors[version] = grouped["absolute_error"].to_numpy(float)
    if volume_reference is None:
        raise ValueError("No prediction versions")
    return series_names, errors, volume_reference


def bootstrap_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    series_names, errors, volume = series_totals(predictions)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(series_names), size=(BOOTSTRAP_REPLICATES, len(series_names)))
    sampled_volume = volume[indices]
    comparisons = [(BASE_VERSION, version) for version in sorted(errors) if version not in (BASE_VERSION, "REVIEW_TEXT_ROLLING")]
    comparisons.extend([
        (BASE_VERSION, "REVIEW_TEXT_ROLLING"),
        ("REVIEW_TEXT_FIXED", "REVIEW_TEXT_ROLLING"),
    ])
    if PLATFORM_VERSION != BEST_VERSION:
        comparisons.append((PLATFORM_VERSION, BEST_VERSION))
    rows: list[dict[str, Any]] = []
    for comparator, candidate in comparisons:
        comparator_boot = wmape_from_totals(errors[comparator][indices], sampled_volume)
        candidate_boot = wmape_from_totals(errors[candidate][indices], sampled_volume)
        improvement = comparator_boot - candidate_boot
        point_comparator = errors[comparator].sum() / volume.sum() * 100
        point_candidate = errors[candidate].sum() / volume.sum() * 100
        rows.append({
            "comparator": comparator,
            "candidate": candidate,
            "point_comparator_WMAPE": point_comparator,
            "point_candidate_WMAPE": point_candidate,
            "point_improvement_pp": point_comparator - point_candidate,
            "bootstrap_mean_improvement_pp": float(np.nanmean(improvement)),
            "bootstrap_ci_2_5_pp": float(np.nanquantile(improvement, 0.025)),
            "bootstrap_ci_97_5_pp": float(np.nanquantile(improvement, 0.975)),
            "bootstrap_probability_candidate_better": float(np.nanmean(improvement > 0)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "cluster_unit": "series_name",
        })
    return pd.DataFrame(rows)


def model_row(record: pd.Series, history: list[float], columns: list[str]) -> dict[str, Any]:
    date = record["date"]
    values = np.asarray(history, dtype=float)
    row: dict[str, Any] = {
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
    return row


def traced_test_rows(model: Any, panel: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for name, series in panel.groupby("series_name", sort=True):
        ordered = series.sort_values("date")
        history = ordered.loc[ordered["split"].isin(("train", "val")), mu.TARGET].astype(float).tolist()
        # The complete panel retains known zero-sales months before launch;
        # those rows are useful history but do not remove the cold-start flag.
        cold_start = not np.any(np.asarray(history, dtype=float) > 0)
        for _, record in ordered.loc[ordered["split"].eq("test")].iterrows():
            row = model_row(record, history, columns)
            features = pd.DataFrame([row], columns=columns)
            prediction = max(float(np.expm1(model.predict(features)[0])), 0.0)
            feature_rows.append(row)
            prediction_rows.append({
                "series_name": name,
                "date": record["date"],
                "actual": float(record[mu.TARGET]),
                "pred": prediction,
                "cold_start_at_forecast_origin": cold_start,
            })
            history.append(prediction)
    return pd.DataFrame(feature_rows, columns=columns), pd.DataFrame(prediction_rows)


def feature_family(feature: str) -> str:
    if feature in mu.LAG_COLS:
        return "sales_lag_roll"
    if feature in mu.CAL:
        return "calendar"
    if feature in mu.CFG_COLS:
        return "configuration"
    if feature.startswith("platform_rating_") or feature.startswith("sentiment_"):
        return "review_observation_context"
    if feature.startswith("review_"):
        if feature in (
            "review_count_prior_all", "review_count_180d",
            "review_available_prior", "review_available_180d",
        ):
            return "review_observation_context"
        if feature.endswith("_score_prior_mean"):
            return "review_expanding_score"
        if feature.endswith("_score_180d_mean"):
            return "review_recent_score"
        if feature.endswith("_positive_180d_rate"):
            return "review_positive_rate"
        if feature.endswith("_negative_180d_rate"):
            return "review_negative_rate"
        if feature.endswith("_uniform_mention_180d_count"):
            return "review_mention_count"
        if feature.endswith("_uniform_mention_180d_rate"):
            return "review_mention_rate"
        return "review_overall"
    return "other"


def shap_importance() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, versions, _ = ablation.build_frames()
    summary = pd.read_csv(ABLATION_SUMMARY, low_memory=False)
    n_estimators = int(summary.loc[summary["version"].eq(BEST_VERSION), "validation_selected_n_estimators"].iloc[0])
    columns = versions[BEST_VERSION]
    # Preserve the exact row ordering used by the original final fit.  XGBoost's
    # subsampling is deterministic only for an identical ordered training matrix.
    train_val = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    panel = pd.concat(
        [frames["train_roll"], frames["val_roll"], frames["test_fixed"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    model = ablation.new_model(n_estimators)
    model.fit(train_val[columns], np.log1p(train_val[mu.TARGET]), verbose=False)
    feature_rows, traced = traced_test_rows(model, panel, columns)

    saved = pd.read_csv(PREDICTIONS, low_memory=False, parse_dates=["date"])
    saved = saved.loc[saved["version"].eq(BEST_VERSION)].sort_values(["series_name", "date"]).reset_index(drop=True)
    traced = traced.sort_values(["series_name", "date"]).reset_index(drop=True)
    if len(saved) != len(traced) or not np.allclose(saved["pred"], traced["pred"], rtol=1e-7, atol=1e-7):
        raise ValueError("Refitted SHAP model predictions do not match saved best-model predictions")

    contributions = model.get_booster().predict(DMatrix(feature_rows, feature_names=columns), pred_contribs=True)
    if contributions.shape != (len(feature_rows), len(columns) + 1):
        raise ValueError("Unexpected SHAP contribution shape")
    shap = contributions[:, :-1]
    importance = pd.DataFrame({
        "feature": columns,
        "feature_family": [feature_family(column) for column in columns],
        "mean_abs_shap_log_sales": np.abs(shap).mean(axis=0),
        "mean_signed_shap_log_sales": shap.mean(axis=0),
        "nonmissing_test_rows": feature_rows.notna().sum(axis=0).to_numpy(),
    }).sort_values("mean_abs_shap_log_sales", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    total = float(importance["mean_abs_shap_log_sales"].sum())
    importance["share_of_total_abs_shap"] = importance["mean_abs_shap_log_sales"] / total if total else np.nan
    family = importance.groupby("feature_family", as_index=False).agg(
        feature_count=("feature", "count"),
        total_mean_abs_shap=("mean_abs_shap_log_sales", "sum"),
        max_feature_mean_abs_shap=("mean_abs_shap_log_sales", "max"),
    ).sort_values("total_mean_abs_shap", ascending=False)
    family["share_of_total_abs_shap"] = family["total_mean_abs_shap"] / family["total_mean_abs_shap"].sum()
    return importance, family


def monthly_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (version, date), group in predictions.groupby(["version", "date"], sort=True):
        rows.append({
            "version": version,
            "date": date,
            "actual_volume": float(np.abs(group["actual"]).sum()),
            "absolute_error": float(np.abs(group["actual"] - group["pred"]).sum()),
            "global_volume_weighted_WMAPE": mu.wmape_vol(group["actual"], group["pred"]),
        })
    result = pd.DataFrame(rows)
    base = result.loc[result["version"].eq(BASE_VERSION), ["date", "global_volume_weighted_WMAPE"]].rename(
        columns={"global_volume_weighted_WMAPE": "baseline_month_WMAPE"}
    )
    result = result.merge(base, on="date", validate="many_to_one")
    result["improvement_vs_base_pp"] = result["baseline_month_WMAPE"] - result["global_volume_weighted_WMAPE"]
    return result.sort_values(["date", "version"])


def series_and_segments(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = predictions.loc[predictions["version"].isin([BASE_VERSION, BEST_VERSION])].copy()
    selected["abs_error"] = (selected["actual"] - selected["pred"]).abs()
    totals = selected.groupby(["version", "series_name"]).agg(
        actual_volume=("actual", lambda values: float(np.abs(values).sum())),
        absolute_error=("abs_error", "sum"),
        cold_start=("cold_start_at_forecast_origin", "max"),
        has_prior_review=("review_available_prior", "max"),
        has_recent_180d_review=("review_available_180d", "max"),
    ).reset_index()
    totals["series_WMAPE"] = np.where(
        totals["actual_volume"].gt(0), totals["absolute_error"] / totals["actual_volume"] * 100, np.nan
    )
    wide = totals.pivot(index="series_name", columns="version", values=["absolute_error", "series_WMAPE"])
    wide.columns = [f"{metric}_{version}" for metric, version in wide.columns]
    metadata = totals.loc[totals["version"].eq(BEST_VERSION), [
        "series_name", "actual_volume", "cold_start", "has_prior_review", "has_recent_180d_review",
    ]]
    diagnostics = metadata.merge(wide.reset_index(), on="series_name", validate="one_to_one")
    test = pd.read_csv(TEST_SPLIT, usecols=["series_name", "brand", "category"], low_memory=False)
    test["series_name"] = test["series_name"].astype(str)
    test_meta = test.drop_duplicates("series_name")
    diagnostics = diagnostics.merge(test_meta, on="series_name", how="left", validate="one_to_one")
    diagnostics["WMAPE_improvement_pp"] = (
        diagnostics[f"series_WMAPE_{BASE_VERSION}"] - diagnostics[f"series_WMAPE_{BEST_VERSION}"]
    )
    diagnostics["absolute_error_reduction"] = (
        diagnostics[f"absolute_error_{BASE_VERSION}"] - diagnostics[f"absolute_error_{BEST_VERSION}"]
    )
    total_best_error = diagnostics[f"absolute_error_{BEST_VERSION}"].sum()
    diagnostics["share_of_best_total_absolute_error"] = (
        diagnostics[f"absolute_error_{BEST_VERSION}"] / total_best_error
    )
    diagnostics["history_group"] = np.where(diagnostics["cold_start"], "cold_start", "historical")
    diagnostics["sentiment_coverage_group"] = np.select(
        [diagnostics["has_recent_180d_review"].eq(1), diagnostics["has_prior_review"].eq(1)],
        ["recent_180d", "prior_but_not_recent"],
        default="no_prior_review",
    )
    rank = diagnostics["actual_volume"].rank(method="first")
    diagnostics["actual_volume_quartile"] = pd.qcut(rank, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    diagnostics = diagnostics.sort_values(f"absolute_error_{BEST_VERSION}", ascending=False).reset_index(drop=True)

    segment_rows: list[dict[str, Any]] = []
    segment_specs = {
        "history_group": diagnostics["history_group"],
        "sentiment_coverage": diagnostics["sentiment_coverage_group"],
        "actual_volume_quartile": diagnostics["actual_volume_quartile"].astype(str),
        "vehicle_category": diagnostics["category"].fillna("missing").astype(str),
    }
    for segment_type, values in segment_specs.items():
        work = diagnostics.assign(_segment=values)
        for segment, group in work.groupby("_segment", sort=True):
            volume = float(group["actual_volume"].sum())
            base_error = float(group[f"absolute_error_{BASE_VERSION}"].sum())
            best_error = float(group[f"absolute_error_{BEST_VERSION}"].sum())
            segment_rows.append({
                "segment_type": segment_type,
                "segment": segment,
                "series": int(len(group)),
                "actual_volume": volume,
                "baseline_global_WMAPE": base_error / volume * 100 if volume else np.nan,
                "selected_feedback_global_WMAPE": best_error / volume * 100 if volume else np.nan,
                "improvement_pp": (base_error - best_error) / volume * 100 if volume else np.nan,
            })
    return diagnostics, pd.DataFrame(segment_rows)


def save_figure(bootstrap: pd.DataFrame, importance: pd.DataFrame, monthly: pd.DataFrame) -> None:
    comparisons = bootstrap.loc[
        bootstrap["comparator"].eq(BASE_VERSION)
        & bootstrap["candidate"].isin([PLATFORM_VERSION, "LOCAL_LEXICON_FIXED", "REVIEW_TEXT_FIXED", BEST_VERSION])
    ].sort_values("point_improvement_pp")
    top = importance.head(15).sort_values("mean_abs_shap_log_sales")
    month = monthly.loc[monthly["version"].isin([BEST_VERSION, PLATFORM_VERSION])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    y = np.arange(len(comparisons))
    axes[0].errorbar(
        comparisons["point_improvement_pp"], y,
        xerr=np.maximum(np.vstack([
            comparisons["point_improvement_pp"] - comparisons["bootstrap_ci_2_5_pp"],
            comparisons["bootstrap_ci_97_5_pp"] - comparisons["point_improvement_pp"],
        ]), 0.0),
        fmt="o", color="#4C78A8", capsize=3,
    )
    axes[0].axvline(0, color="#777777", linewidth=1)
    axes[0].set_yticks(y, comparisons["candidate"].str.replace("_FIXED", "", regex=False))
    axes[0].set_xlabel("WMAPE improvement vs BASE (pp)")
    axes[0].set_title("Series-cluster bootstrap 95% intervals")

    axes[1].barh(top["feature"], top["mean_abs_shap_log_sales"], color="#54A24B")
    axes[1].set_xlabel("Mean |SHAP| on log-sales output")
    axes[1].set_title("Top review features")
    axes[1].tick_params(axis="y", labelsize=7)

    for version, group in month.groupby("version"):
        axes[2].plot(pd.to_datetime(group["date"]), group["improvement_vs_base_pp"], marker="o", label=version)
    axes[2].axhline(0, color="#777777", linewidth=1)
    axes[2].set_ylabel("Monthly WMAPE improvement vs BASE (pp)")
    axes[2].set_title("Improvement stability by test month")
    axes[2].legend(fontsize=7)
    axes[2].tick_params(axis="x", rotation=25)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS, low_memory=False, parse_dates=["date"])
    required_versions = {BASE_VERSION, BEST_VERSION, PLATFORM_VERSION, "REVIEW_TEXT_FIXED", "REVIEW_TEXT_ROLLING"}
    if not required_versions.issubset(set(predictions["version"])):
        raise ValueError("Required ablation prediction versions are missing")
    bootstrap = bootstrap_comparisons(predictions)
    importance, family = shap_importance()
    monthly = monthly_stability(predictions)
    diagnostics, segments = series_and_segments(predictions)

    bootstrap.to_csv(BOOTSTRAP_OUTPUT, index=False, encoding="utf-8-sig")
    importance.to_csv(SHAP_OUTPUT, index=False, encoding="utf-8-sig")
    family.to_csv(FAMILY_OUTPUT, index=False, encoding="utf-8-sig")
    monthly.to_csv(MONTHLY_OUTPUT, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(SERIES_OUTPUT, index=False, encoding="utf-8-sig")
    segments.to_csv(SEGMENT_OUTPUT, index=False, encoding="utf-8-sig")
    save_figure(bootstrap, importance, monthly)

    best_boot = bootstrap.loc[
        bootstrap["comparator"].eq(BASE_VERSION) & bootstrap["candidate"].eq(BEST_VERSION)
    ].iloc[0]
    best_monthly = monthly.loc[monthly["version"].eq(BEST_VERSION)]
    feedback_importance = importance.loc[
        importance["feature"].str.startswith(
            ("review_", "platform_rating_", "sentiment_", "text_")
        )
    ]
    summary = {
        "schema_version": "v1",
        "validation_selected_feedback_version": BEST_VERSION,
        "series_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "selected_feedback_vs_base_improvement_pp": float(best_boot["point_improvement_pp"]),
        "selected_feedback_vs_base_bootstrap_95pct_ci_pp": [
            float(best_boot["bootstrap_ci_2_5_pp"]), float(best_boot["bootstrap_ci_97_5_pp"]),
        ],
        "selected_feedback_vs_base_probability_better": float(best_boot["bootstrap_probability_candidate_better"]),
        "test_months_selected_feedback_better_than_base": int(best_monthly["improvement_vs_base_pp"].gt(0).sum()),
        "test_months_total": int(len(best_monthly)),
        "top_10_features_by_mean_abs_shap": importance.head(10)["feature"].tolist(),
        "top_10_selected_feedback_features_by_mean_abs_shap": feedback_importance.head(10)["feature"].tolist(),
        "top_10_error_contributing_series": diagnostics.head(10)["series_name"].tolist(),
        "cold_start_series": int(diagnostics["cold_start"].sum()),
        "external_api_calls": 0,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\n===== Bootstrap comparisons =====", flush=True)
    print(
        bootstrap[[
            "comparator", "candidate", "point_improvement_pp", "bootstrap_ci_2_5_pp",
            "bootstrap_ci_97_5_pp", "bootstrap_probability_candidate_better",
        ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        flush=True,
    )
    print("\n===== Feature-family SHAP =====", flush=True)
    print(family.to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    print(f"[output] {SUMMARY_OUTPUT.relative_to(BASE)}", flush=True)
    print(f"[output] {FIGURE.relative_to(BASE)}", flush=True)


if __name__ == "__main__":
    main()

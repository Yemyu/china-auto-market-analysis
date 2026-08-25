#!/usr/bin/env python3
"""Leakage-safe XGBoost ablation for monthly sentiment features.

All four versions use the identical 150-series, time-eligible evaluation
cohort used by the existing Stage-3 model comparison.  The underlying sales
split is fixed: train through 2025-06, validation 2025-07..12, and test
2026-01..06.  Sentiment for target month t is joined from a table whose scores
and counts only use reviews published before the first day of t.

Versions
--------
BASE                Sales lags, calendar, and causal configuration only.
PLATFORM_RATING     BASE plus user-submitted platform ratings and observation
                    counts / availability flags.
TEXT_ABSA           BASE plus transparent local text ABSA and the same
                    observation counts / availability flags.
ALL_SENTIMENT       BASE plus both sentiment families.

Validation selects the number of trees independently for each version; the
test split is never used for tuning.  Test forecasting remains recursive for
sales lags, while sentiment is supplied as a known-at-origin monthly exogenous
feature.

Outputs
-------
data/processed_new/stage3/xgb_sentiment_ablation_summary.csv
data/processed_new/stage3/xgb_sentiment_ablation_series_metrics.csv
data/processed_new/stage3/xgb_sentiment_ablation_preds.csv
data/processed_new/stage3/xgb_sentiment_ablation_coverage.csv
data/processed_new/stage3/xgb_sentiment_ablation_feature_manifest.csv
figures_new/xgb_sentiment_ablation.png

Run only in the project Conda environment:
  conda run -n nlp-sentiment python scripts/new_pipeline/29_xgb_sentiment_ablation.py
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"  # avoid XGBoost multithread instability

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _font_setup
import _model_utils as mu
import _subset


BASE = Path(__file__).resolve().parents[2]
SENTIMENT_FEATURES = BASE / "data" / "sentiment_new" / "processed" / "sentiment_features_by_series_month.csv"
PROC = BASE / "data" / "processed_new" / "stage3"
FIG = BASE / "figures_new"

CONTEXT_COLS = [
    "sentiment_review_count_prior_all",
    "sentiment_review_count_180d",
    "sentiment_available_prior",
    "sentiment_available_180d",
]
PLATFORM_PREFIX = "platform_rating_"
TEXT_POLARITY_PREFIX = "text_"
TEXT_MENTION_SUFFIX = "_mentioned_180d_count"


def read_sentiment_table() -> pd.DataFrame:
    if not SENTIMENT_FEATURES.exists():
        raise FileNotFoundError(
            f"Missing {SENTIMENT_FEATURES}; run 28_aggregate_sentiment_temporal_features.py first."
        )
    table = pd.read_csv(SENTIMENT_FEATURES, low_memory=False)
    required = {"series_name", "date", *CONTEXT_COLS}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Sentiment table is missing required columns: {sorted(missing)}")
    table["series_name"] = table["series_name"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if table.duplicated(["series_name", "date"]).any():
        raise ValueError("Sentiment feature table has duplicate (series_name, date) rows")
    return table


def attach_sentiment(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Attach target-month, already-truncated sentiment rows without row loss."""
    out = frame.copy()
    out["series_name"] = out["series_name"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
    before = len(out)
    out = out.merge(table, on=["series_name", "date"], how="left", validate="many_to_one")
    if len(out) != before:
        raise ValueError("Sentiment join changed the number of split rows")
    if out[CONTEXT_COLS].isna().any().any():
        missing = out.loc[out[CONTEXT_COLS].isna().any(axis=1), ["series_name", "date"]].head(10)
        raise ValueError(f"Missing sentiment monthly rows after join, examples:\n{missing}")
    return out


def build_versions(train: pd.DataFrame, sentiment_table: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Return feature sets, dropping only candidates entirely absent in train."""
    platform_candidates = [c for c in sentiment_table.columns if c.startswith(PLATFORM_PREFIX)]
    text_candidates = [
        c for c in sentiment_table.columns
        if c.startswith(TEXT_POLARITY_PREFIX)
        and (c.endswith("_polarity_180d_mean") or c.endswith(TEXT_MENTION_SUFFIX))
    ]

    rows = []

    def usable(candidates: list[str], family: str) -> list[str]:
        result = []
        for column in candidates:
            present = int(train[column].notna().sum())
            keep = present > 0
            rows.append({
                "feature_family": family,
                "feature_name": column,
                "training_nonmissing_rows": present,
                "included_in_model": keep,
                "exclusion_reason": "" if keep else "all values missing in train split",
            })
            if keep:
                result.append(column)
        return result

    platform_cols = usable(platform_candidates, "platform_rating")
    text_cols = usable(text_candidates, "text_absa")
    for column in CONTEXT_COLS:
        rows.append({
            "feature_family": "observation_context",
            "feature_name": column,
            "training_nonmissing_rows": int(train[column].notna().sum()),
            "included_in_model": True,
            "exclusion_reason": "",
        })

    versions = {
        "BASE": list(mu.FEAT_COLS),
        "PLATFORM_RATING": list(mu.FEAT_COLS) + CONTEXT_COLS + platform_cols,
        "TEXT_ABSA": list(mu.FEAT_COLS) + CONTEXT_COLS + text_cols,
        "ALL_SENTIMENT": list(mu.FEAT_COLS) + CONTEXT_COLS + platform_cols + text_cols,
    }
    if not platform_cols or not text_cols:
        raise ValueError("No usable platform or text sentiment columns in the training split")
    return versions, pd.DataFrame(rows)


def fit_version(columns: list[str], train: pd.DataFrame, validation: pd.DataFrame) -> tuple[XGBRegressor, int]:
    """Select capacity on validation, then refit once on train + validation."""
    selector = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
        early_stopping_rounds=50,
    )
    selector.fit(
        train[columns],
        np.log1p(train[mu.TARGET]),
        eval_set=[(validation[columns], np.log1p(validation[mu.TARGET]))],
        verbose=False,
    )
    best_raw = getattr(selector, "best_iteration", None)
    n_estimators = int(best_raw) + 1 if best_raw is not None else 1000
    final = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    train_val = pd.concat([train, validation], ignore_index=True)
    final.fit(train_val[columns], np.log1p(train_val[mu.TARGET]), verbose=False)
    return final, n_estimators


def recursive_prediction_rows(
    model: XGBRegressor,
    panel: pd.DataFrame,
    columns: list[str],
    version: str,
) -> list[dict]:
    rows = []
    for name, series in panel.groupby("series_name", sort=True):
        series = series.sort_values("date")
        forecast = mu.recursive_forecast_tree(
            model,
            series,
            feat_cols=columns,
            history_splits=("train", "val"),
            forecast_splits=("test",),
        )
        test = series.loc[series["split"].eq("test")]
        for _, record in test.iterrows():
            prediction = forecast.get(record["date"], np.nan)
            if not np.isfinite(prediction):
                raise ValueError(f"Missing recursive test prediction for {name} / {record['date']}")
            rows.append({
                "version": version,
                "series_name": name,
                "date": record["date"].strftime("%Y-%m-%d"),
                "actual": float(record[mu.TARGET]),
                "pred": float(prediction),
                **{column: record[column] for column in CONTEXT_COLS},
            })
    return rows


def build_series_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (version, name), group in predictions.groupby(["version", "series_name"], sort=True):
        metric = mu.metrics(group["actual"], group["pred"])
        metric.update({
            "version": version,
            "series_name": name,
            "test_months": len(group),
            "actual_volume": float(np.abs(group["actual"]).sum()),
        })
        rows.append(metric)
    return pd.DataFrame(rows)


def build_summary(predictions: pd.DataFrame, metrics: pd.DataFrame, tree_counts: dict[str, int], versions: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for version, group in predictions.groupby("version", sort=False):
        per_series = metrics.loc[metrics["version"].eq(version)]
        rows.append({
            "version": version,
            "n_features": len(versions[version]),
            "validation_selected_n_estimators": tree_counts[version],
            "test_rows": len(group),
            "test_series": group["series_name"].nunique(),
            "test_actual_volume": float(np.abs(group["actual"]).sum()),
            "global_volume_weighted_WMAPE": mu.wmape_vol(group["actual"], group["pred"]),
            "median_per_series_WMAPE": float(per_series["WMAPE"].median()),
            "mean_per_series_WMAPE": float(per_series["WMAPE"].mean()),
            "test_start": group["date"].min(),
            "test_end": group["date"].max(),
        })
    return pd.DataFrame(rows)


def build_coverage_report(predictions: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_test_rows": pd.Series(True, index=predictions.index),
        "any_prior_review": predictions["sentiment_available_prior"].eq(1),
        "no_prior_review": predictions["sentiment_available_prior"].eq(0),
        "recent_180d_review": predictions["sentiment_available_180d"].eq(1),
        "no_recent_180d_review": predictions["sentiment_available_180d"].eq(0),
    }
    rows = []
    for version, version_rows in predictions.groupby("version", sort=False):
        for group_name, mask in groups.items():
            part = version_rows.loc[mask.loc[version_rows.index]]
            if part.empty:
                continue
            series_wmape = mu.wmape_per_series(part["actual"], part["pred"], part["series_name"])
            rows.append({
                "version": version,
                "coverage_group": group_name,
                "test_rows": len(part),
                "test_series": part["series_name"].nunique(),
                "actual_volume": float(np.abs(part["actual"]).sum()),
                "global_volume_weighted_WMAPE": mu.wmape_vol(part["actual"], part["pred"]),
                "median_per_series_WMAPE": float(series_wmape.median()),
            })
    return pd.DataFrame(rows)


def save_figure(summary: pd.DataFrame, final_model: XGBRegressor, final_columns: list[str]) -> None:
    display_order = ["BASE", "PLATFORM_RATING", "TEXT_ABSA", "ALL_SENTIMENT"]
    ordered = summary.set_index("version").loc[display_order]
    importance = pd.Series(final_model.feature_importances_, index=final_columns).sort_values(ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    x = np.arange(len(display_order))
    axes[0].bar(x - 0.18, ordered["global_volume_weighted_WMAPE"], width=0.36, color=colors)
    axes[0].bar(x + 0.18, ordered["median_per_series_WMAPE"], width=0.36, color="#B8C4CE")
    axes[0].set_xticks(x, display_order, rotation=12, ha="right")
    axes[0].set_ylabel("WMAPE (%)")
    axes[0].set_title("Test: global WMAPE vs per-series median")
    axes[0].legend(["Global volume-weighted", "Per-series median"], fontsize=8)

    axes[1].barh(importance.index[::-1], importance.values[::-1], color="#54A24B")
    axes[1].set_title("ALL_SENTIMENT feature importance (top 15)")
    axes[1].tick_params(labelsize=7)
    fig.suptitle("XGBoost sentiment ablation: time-split recursive test forecast", fontsize=11)
    fig.savefig(FIG / "xgb_sentiment_ablation.png", dpi=140)
    plt.close(fig)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    sentiment_table = read_sentiment_table()
    train, validation, test = mu.load_splits()
    train = attach_sentiment(train, sentiment_table)
    validation = attach_sentiment(validation, sentiment_table)
    test = attach_sentiment(test, sentiment_table)
    versions, feature_manifest = build_versions(train, sentiment_table)
    feature_manifest.to_csv(PROC / "xgb_sentiment_ablation_feature_manifest.csv", index=False, encoding="utf-8-sig")

    subset = _subset.load_subset()
    panel = pd.concat([train, validation, test], ignore_index=True)
    panel = panel.loc[panel["series_name"].isin(set(subset))].sort_values(["series_name", "date"])
    if panel.groupby("series_name")["split"].nunique().ne(3).any():
        raise ValueError("Evaluation cohort lost a train/validation/test split after the sentiment join")
    print(f"[sentiment-ablation] cohort={len(subset)} series | train={len(train)} val={len(validation)} test={len(test)}")

    all_prediction_rows: list[dict] = []
    tree_counts: dict[str, int] = {}
    all_model: XGBRegressor | None = None
    for version, columns in versions.items():
        print(f"[sentiment-ablation:{version}] fitting {len(columns)} features ...")
        model, n_estimators = fit_version(columns, train, validation)
        tree_counts[version] = n_estimators
        all_prediction_rows.extend(recursive_prediction_rows(model, panel, columns, version))
        if version == "ALL_SENTIMENT":
            all_model = model

    predictions = pd.DataFrame(all_prediction_rows).sort_values(["version", "series_name", "date"])
    metrics = build_series_metrics(predictions)
    summary = build_summary(predictions, metrics, tree_counts, versions)
    coverage = build_coverage_report(predictions)

    predictions.to_csv(PROC / "xgb_sentiment_ablation_preds.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(PROC / "xgb_sentiment_ablation_series_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(PROC / "xgb_sentiment_ablation_summary.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(PROC / "xgb_sentiment_ablation_coverage.csv", index=False, encoding="utf-8-sig")
    if all_model is None:
        raise RuntimeError("ALL_SENTIMENT model was not trained")
    save_figure(summary, all_model, versions["ALL_SENTIMENT"])

    print("\n===== Sentiment ablation: TEST only (2026-01..06) =====")
    print(summary[["version", "n_features", "validation_selected_n_estimators", "global_volume_weighted_WMAPE", "median_per_series_WMAPE"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n[output] stage3/xgb_sentiment_ablation_{summary,series_metrics,preds,coverage,feature_manifest}.csv")
    print("[output] figures_new/xgb_sentiment_ablation.png")


if __name__ == "__main__":
    main()

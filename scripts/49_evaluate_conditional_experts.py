#!/usr/bin/env python3
"""Evaluate a leakage-safe conditional expert forecast on historical origins.

The experiment deliberately leaves the 2026-01..06 locked test untouched.
The first three rolling origins calibrate simple, pre-declared routing rules;
the 2025-07 origin is a separate promotion check.  Public dashboard, README,
notebooks, and headline forecast artifacts are not modified.
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

from _series_mapping import build_series_name_mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = ROOT / "artifacts" / "conditional_experts"
CALIBRATION_ORIGINS = ("2024-01-01", "2024-07-01", "2025-01-01")
PROMOTION_ORIGIN = "2025-07-01"

# These gates are fixed before looking at the promotion origin or locked test.
STRUCTURAL_GATE = {
    "minimum_calibration_gain_pp": 0.25,
    "minimum_promotion_gain_pp": 0.00,
    "maximum_calibration_origin_regression_pp": 0.50,
    "minimum_routed_row_share": 0.05,
    "maximum_promotion_high_volume_regression_pp": 0.50,
    "maximum_promotion_long_tail_regression_pp": 2.00,
}
REVIEW_GATE = {
    "minimum_calibration_gain_pp": 0.15,
    "minimum_promotion_gain_pp": 0.10,
    "maximum_calibration_origin_regression_pp": 0.50,
    "minimum_routed_row_share": 0.10,
    "minimum_routed_series": 40,
    "maximum_promotion_high_volume_regression_pp": 0.50,
    "maximum_promotion_long_tail_regression_pp": 2.00,
}

CONFIG_HISTORY_THRESHOLDS = (0, 3, 6, 12, 18, 24)
CONFIG_BLEND_WEIGHTS = (0.50, 1.00)
REVIEW_COUNT_THRESHOLDS = (1, 3, 5, 10, 20)
REVIEW_BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def wmape(frame: pd.DataFrame, prediction: str) -> float:
    volume = float(np.abs(frame["actual"]).sum())
    if volume <= 0:
        return float("nan")
    return float(np.abs(frame["actual"] - frame[prediction]).sum() / volume * 100)


def model_params(rolling_module) -> dict[str, Any]:
    return dict(rolling_module.MODEL_PARAMS["SEASONAL_D5"])


def fit_candidate(
    rolling_module,
    fit: pd.DataFrame,
    panel: pd.DataFrame,
    columns: list[str],
    required_columns: list[str],
) -> pd.DataFrame:
    usable = fit.dropna(subset=required_columns)
    if usable.empty:
        raise RuntimeError("Candidate has no causally usable training rows")
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        **model_params(rolling_module),
    )
    model.fit(usable[columns], np.log1p(usable[rolling_module.mu.TARGET]), verbose=False)
    return rolling_module.rolling_predictions(
        model,
        panel,
        columns,
        forecast_split="val",
        history_splits=("train",),
    )


def configuration_availability(panel: pd.DataFrame) -> pd.DataFrame:
    """Resolve whether a configuration record exists by the forecast year."""
    feature = pd.read_csv(
        ROOT / "data" / "raw" / "feature.csv",
        usecols=["series_name", "year"],
        low_memory=False,
    )
    feature["series_name"] = feature["series_name"].astype(str)
    feature["year"] = pd.to_numeric(feature["year"], errors="coerce")
    feature = feature.dropna(subset=["year"])
    mapping = build_series_name_mapping(panel["series_name"], feature["series_name"])
    name_map = mapping.set_index("sales_series_name")["config_series_name"]
    years = (
        feature.groupby("series_name")["year"]
        .apply(lambda values: tuple(sorted(set(values.astype(int)))))
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    keys = panel[["series_name", "date"]].drop_duplicates()
    for _, record in keys.iterrows():
        year = int(pd.Timestamp(record["date"]).year)
        config_name = name_map.get(str(record["series_name"]))
        available_years = [value for value in years.get(config_name, ()) if value <= year]
        source_year = max(available_years) if available_years else None
        rows.append({
            "series_name": str(record["series_name"]),
            "date": pd.Timestamp(record["date"]),
            "config_available": int(source_year is not None),
            "config_source_year": source_year if source_year is not None else -1,
            "config_age_years": year - source_year if source_year is not None else -1,
        })
    return pd.DataFrame(rows)


def causal_router_metadata(panel: pd.DataFrame) -> pd.DataFrame:
    """Build route inputs using only information available before each month."""
    rows: list[dict[str, Any]] = []
    for name, series in panel.groupby("series_name", sort=True):
        ordered = series.sort_values("date").reset_index(drop=True)
        history = ordered.loc[ordered["split"].eq("train"), "monthly_sales"].astype(float).tolist()
        for _, record in ordered.loc[ordered["split"].eq("val")].iterrows():
            recent = np.asarray(history[-6:], dtype=float)
            rows.append({
                "series_name": name,
                "date": record["date"],
                "prior_history_months": int(len(history)),
                "prior_positive_months": int(np.sum(np.asarray(history) > 0)),
                "recent_positive_months_6": int(np.sum(recent > 0)),
                "recent_mean_6": float(np.mean(recent)) if len(recent) else 0.0,
                "review_count_180d": float(record.get("review_count_180d", 0) or 0),
                "review_available_180d": int(record.get("review_available_180d", 0) or 0),
            })
            # In a rolling monthly operation, the realised month is available
            # before routing the following month's forecast.
            history.append(float(record["monthly_sales"]))
    result = pd.DataFrame(rows).merge(
        configuration_availability(panel),
        on=["series_name", "date"],
        how="left",
        validate="one_to_one",
    )
    if result.duplicated(["series_name", "date"]).any():
        raise ValueError("Router metadata contains duplicate series-month rows")
    return result


def load_historical_frames(rolling, review_module) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load only train/validation targets and point-in-time external features."""
    train = review_module.normalize_dates(pd.read_csv(
        ROOT / "data" / "processed" / "splits" / "train.csv",
        low_memory=False,
    ))
    validation = review_module.normalize_dates(pd.read_csv(
        ROOT / "data" / "processed" / "splits" / "val.csv",
        low_memory=False,
    ))
    local = review_module.normalize_dates(pd.read_csv(
        review_module.LOCAL_ROLLING, low_memory=False
    ))
    review_rolling = review_module.normalize_dates(pd.read_csv(
        review_module.REVIEW_ROLLING, low_memory=False
    ))
    # Rows at or after the locked-test origin are discarded before any join.
    cutoff = pd.Timestamp("2026-01-01")
    local = local.loc[local["date"].lt(cutoff)].copy()
    review_rolling = review_rolling.loc[review_rolling["date"].lt(cutoff)].copy()
    local_features = [
        column for column in local.columns if column not in ("series_name", "date")
    ]
    review_features = [
        column for column in review_rolling.columns if column.startswith("review_")
    ]
    train_roll = review_module.attach_by_month(train, local, local_features)
    train_roll = review_module.attach_by_month(train_roll, review_rolling, review_features)
    val_roll = review_module.attach_by_month(validation, local, local_features)
    val_roll = review_module.attach_by_month(val_roll, review_rolling, review_features)

    platform = [column for column in local_features if column.startswith("platform_rating_")]
    local_text = [
        column for column in local_features
        if column.startswith("text_") and (
            column.endswith("_polarity_180d_mean")
            or column.endswith("_mentioned_180d_count")
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
    local_context = [
        column for column in review_module.LOCAL_CONTEXT if column in train_roll.columns
    ]
    families = {
        "PLATFORM_RATING_FIXED": unique(local_context + platform),
        "LOCAL_LEXICON_FIXED": unique(local_context + local_text),
        "REVIEW_TEXT_FIXED": review_core,
    }
    for family, columns in families.items():
        families[family] = [column for column in columns if train_roll[column].notna().any()]
        if not families[family]:
            raise ValueError(f"No usable training features for {family}")
    historical = pd.concat([train_roll, val_roll], ignore_index=True).sort_values(
        ["series_name", "date"]
    )
    return historical, families


def build_historical_predictions() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rolling = load_module("rolling_origin", SCRIPTS / "48_evaluate_rolling_origin.py")
    review_module = rolling.load_forecast_module()
    historical, review_versions = load_historical_frames(rolling, review_module)

    sales_features = unique(rolling.mu.LAG_COLS + rolling.mu.CAL + rolling.mu.SEASONAL_LAG_COLS)
    config_features = unique(sales_features + rolling.mu.CFG_COLS)
    external = {
        "CONFIG_PLATFORM": [
            column for column in review_versions["PLATFORM_RATING_FIXED"]
            if column not in rolling.mu.FEAT_COLS
        ],
        "CONFIG_LEXICON": [
            column for column in review_versions["LOCAL_LEXICON_FIXED"]
            if column not in rolling.mu.FEAT_COLS
        ],
        "CONFIG_REVIEW_CORE": [
            column for column in review_versions["REVIEW_TEXT_FIXED"]
            if column not in rolling.mu.FEAT_COLS
        ],
    }
    versions = {
        "SALES_SEASONAL": sales_features,
        "CONFIG_SEASONAL": config_features,
        **{name: unique(config_features + columns) for name, columns in external.items()},
    }

    predictions: list[pd.DataFrame] = []
    for origin in rolling.ORIGINS:
        origin_name = origin.strftime("%Y-%m-%d")
        fit, panel = rolling.origin_panel(historical, origin)
        metadata = causal_router_metadata(panel)
        expected = panel.loc[panel["split"].eq("val"), ["series_name", "date"]]
        if len(metadata) != len(expected):
            raise ValueError(f"Metadata coverage mismatch at {origin_name}")
        for version, columns in versions.items():
            prediction = fit_candidate(
                rolling,
                fit,
                panel,
                columns,
                required_columns=sales_features,
            )
            prediction = prediction.merge(
                metadata,
                on=["series_name", "date"],
                how="left",
                validate="one_to_one",
            )
            prediction["origin"] = origin_name
            prediction["version"] = version
            predictions.append(prediction)
            print(
                f"[{origin:%Y-%m}] {version:20s} WMAPE={wmape(prediction, 'pred'):.3f}",
                flush=True,
            )
    result = pd.concat(predictions, ignore_index=True)
    return result, versions


def prediction_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    index = [
        "origin", "series_name", "date", "actual", "prior_history_months",
        "prior_positive_months", "recent_positive_months_6", "recent_mean_6",
        "config_available", "config_source_year", "config_age_years",
        "review_count_180d", "review_available_180d",
    ]
    matrix = predictions.pivot(index=index, columns="version", values="pred").reset_index()
    matrix.columns.name = None
    return matrix


def origin_scores(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    rows = []
    for origin, group in frame.groupby("origin", sort=True):
        rows.append({"origin": origin, "WMAPE": wmape(group, prediction)})
    return pd.DataFrame(rows)


def pooled_score(frame: pd.DataFrame, prediction: str) -> float:
    return wmape(frame, prediction)


def segment_regression(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    segment: str,
) -> float:
    promotion = frame.loc[frame["origin"].eq(PROMOTION_ORIGIN)].copy()
    if promotion.empty:
        return float("inf")
    low = promotion["recent_mean_6"].quantile(0.25)
    high = promotion["recent_mean_6"].quantile(0.75)
    if segment == "high_volume":
        part = promotion.loc[promotion["recent_mean_6"].ge(high)]
    elif segment == "long_tail":
        part = promotion.loc[promotion["recent_mean_6"].le(low)]
    else:
        raise KeyError(segment)
    return pooled_score(part, candidate) - pooled_score(part, baseline)


def route_metrics(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    routed: pd.Series,
) -> dict[str, Any]:
    calibration = frame["origin"].isin(CALIBRATION_ORIGINS)
    promotion = frame["origin"].eq(PROMOTION_ORIGIN)
    calibration_scores = origin_scores(frame.loc[calibration], candidate).merge(
        origin_scores(frame.loc[calibration], baseline),
        on="origin",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    calibration_scores["regression_pp"] = (
        calibration_scores["WMAPE_candidate"] - calibration_scores["WMAPE_baseline"]
    )
    calibration_base = pooled_score(frame.loc[calibration], baseline)
    calibration_candidate = pooled_score(frame.loc[calibration], candidate)
    promotion_base = pooled_score(frame.loc[promotion], baseline)
    promotion_candidate = pooled_score(frame.loc[promotion], candidate)
    return {
        "calibration_base_WMAPE": calibration_base,
        "calibration_candidate_WMAPE": calibration_candidate,
        "calibration_gain_pp": calibration_base - calibration_candidate,
        "promotion_base_WMAPE": promotion_base,
        "promotion_candidate_WMAPE": promotion_candidate,
        "promotion_gain_pp": promotion_base - promotion_candidate,
        "worst_calibration_origin_regression_pp": float(
            calibration_scores["regression_pp"].max()
        ),
        # Coverage constraints are also calibrated without looking at the
        # promotion origin.  The promotion block is used only by the gate.
        "routed_rows": int((routed & calibration).sum()),
        "routed_row_share": float(routed.loc[calibration].mean()),
        "routed_series": int(
            frame.loc[routed & calibration, "series_name"].nunique()
        ),
        "promotion_high_volume_regression_pp": segment_regression(
            frame, candidate, baseline, "high_volume"
        ),
        "promotion_long_tail_regression_pp": segment_regression(
            frame, candidate, baseline, "long_tail"
        ),
    }


def structural_candidates(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode, threshold in [("all", None), *[("short_history", value) for value in CONFIG_HISTORY_THRESHOLDS]]:
        if mode == "all":
            eligible = matrix["config_available"].eq(1)
        else:
            eligible = (
                matrix["config_available"].eq(1)
                & matrix["prior_positive_months"].le(int(threshold))
            )
        for weight in CONFIG_BLEND_WEIGHTS:
            candidate = f"_structural_{mode}_{threshold}_{weight:.2f}"
            matrix[candidate] = matrix["SALES_SEASONAL"] + weight * (
                matrix["CONFIG_SEASONAL"] - matrix["SALES_SEASONAL"]
            ) * eligible.astype(float)
            metrics = route_metrics(matrix, candidate, "SALES_SEASONAL", eligible)
            rows.append({
                "expert_family": "configuration",
                "expert": "CONFIG_SEASONAL",
                "route_mode": mode,
                "threshold": threshold,
                "blend_weight": weight,
                "prediction_column": candidate,
                **metrics,
            })
    return pd.DataFrame(rows)


def passes_structural_gate(row: pd.Series) -> bool:
    return bool(
        row["calibration_gain_pp"] >= STRUCTURAL_GATE["minimum_calibration_gain_pp"]
        and row["promotion_gain_pp"] >= STRUCTURAL_GATE["minimum_promotion_gain_pp"]
        and row["worst_calibration_origin_regression_pp"]
        <= STRUCTURAL_GATE["maximum_calibration_origin_regression_pp"]
        and row["routed_row_share"] >= STRUCTURAL_GATE["minimum_routed_row_share"]
        and row["promotion_high_volume_regression_pp"]
        <= STRUCTURAL_GATE["maximum_promotion_high_volume_regression_pp"]
        and row["promotion_long_tail_regression_pp"]
        <= STRUCTURAL_GATE["maximum_promotion_long_tail_regression_pp"]
    )


def review_candidates(matrix: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expert in ("CONFIG_PLATFORM", "CONFIG_LEXICON", "CONFIG_REVIEW_CORE"):
        for threshold in REVIEW_COUNT_THRESHOLDS:
            eligible = (
                matrix["review_available_180d"].eq(1)
                & matrix["review_count_180d"].ge(threshold)
            )
            for weight in REVIEW_BLEND_WEIGHTS:
                candidate = f"_review_{expert}_{threshold}_{weight:.2f}"
                matrix[candidate] = matrix[baseline] + weight * (
                    matrix[expert] - matrix["CONFIG_SEASONAL"]
                ) * eligible.astype(float)
                metrics = route_metrics(matrix, candidate, baseline, eligible)
                rows.append({
                    "expert_family": "review",
                    "expert": expert,
                    "route_mode": "recent_review_count",
                    "threshold": threshold,
                    "blend_weight": weight,
                    "prediction_column": candidate,
                    **metrics,
                })
    return pd.DataFrame(rows)


def passes_review_gate(row: pd.Series) -> bool:
    return bool(
        row["calibration_gain_pp"] >= REVIEW_GATE["minimum_calibration_gain_pp"]
        and row["promotion_gain_pp"] >= REVIEW_GATE["minimum_promotion_gain_pp"]
        and row["worst_calibration_origin_regression_pp"]
        <= REVIEW_GATE["maximum_calibration_origin_regression_pp"]
        and row["routed_row_share"] >= REVIEW_GATE["minimum_routed_row_share"]
        and row["routed_series"] >= REVIEW_GATE["minimum_routed_series"]
        and row["promotion_high_volume_regression_pp"]
        <= REVIEW_GATE["maximum_promotion_high_volume_regression_pp"]
        and row["promotion_long_tail_regression_pp"]
        <= REVIEW_GATE["maximum_promotion_long_tail_regression_pp"]
    )


def select_on_calibration(grid: pd.DataFrame) -> pd.Series:
    """Choose a route without using any promotion-origin metric."""
    return grid.sort_values(
        [
            "calibration_gain_pp",
            "worst_calibration_origin_regression_pp",
            "routed_row_share",
        ],
        ascending=[False, True, False],
    ).iloc[0]


def serializable_row(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            result[key] = bool(value)
        else:
            result[key] = value
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, versions = build_historical_predictions()
    if predictions["date"].max() >= pd.Timestamp("2026-01-01"):
        raise AssertionError("Locked-test target month entered the historical experiment")
    expected_rows = 4 * 6 * 371 * len(versions)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"Historical candidate coverage changed: {len(predictions)} != {expected_rows}"
        )
    matrix = prediction_matrix(predictions)

    structural_grid = structural_candidates(matrix)
    selected_structural = select_on_calibration(structural_grid)
    structural_passes = passes_structural_gate(selected_structural)
    structural_prediction = (
        str(selected_structural["prediction_column"])
        if structural_passes else "SALES_SEASONAL"
    )

    review_grid = review_candidates(matrix, structural_prediction)
    selected_review = select_on_calibration(review_grid)
    review_passes = passes_review_gate(selected_review)
    final_prediction = (
        str(selected_review["prediction_column"])
        if review_passes else structural_prediction
    )

    grid = pd.concat([structural_grid, review_grid], ignore_index=True)
    matrix["selected_historical_prediction"] = matrix[final_prediction]
    final_by_origin = origin_scores(matrix, "selected_historical_prediction")
    final_by_origin["base_sales_WMAPE"] = origin_scores(
        matrix, "SALES_SEASONAL"
    )["WMAPE"]
    final_by_origin["gain_vs_sales_pp"] = (
        final_by_origin["base_sales_WMAPE"] - final_by_origin["WMAPE"]
    )

    summary = {
        "schema_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "rolling one-month-ahead conditional expert forecast",
        "locked_test_targets_read": False,
        "latest_target_month_used": predictions["date"].max().strftime("%Y-%m-%d"),
        "calibration_origins": list(CALIBRATION_ORIGINS),
        "promotion_origin": PROMOTION_ORIGIN,
        "model_parameters_locked_to": "SEASONAL_D5 from scripts/48_evaluate_rolling_origin.py",
        "feature_versions": versions,
        "gates": {
            "configuration": STRUCTURAL_GATE,
            "review": REVIEW_GATE,
        },
        "configuration_expert": {
            "selected_on_calibration": serializable_row(selected_structural),
            "passes_promotion_gate": structural_passes,
            "deployed_historical_prediction": structural_prediction,
        },
        "review_expert": {
            "selected_on_calibration": serializable_row(selected_review),
            "passes_promotion_gate": review_passes,
            "deployed_historical_prediction": (
                str(selected_review["prediction_column"]) if review_passes else None
            ),
            "fallback": structural_prediction,
        },
        "event_expert": {
            "status": "not_evaluated",
            "reason": "No point-in-time event table is present; no proxy or fabricated event data is allowed.",
        },
        "selected_system_prediction": final_prediction,
        "historical_origin_results": final_by_origin.to_dict(orient="records"),
        "locked_test_policy": (
            "Do not read 2026-01..06 until the historical promotion gates pass; "
            "a test result cannot change routing or thresholds."
        ),
    }

    predictions.to_csv(OUT / "candidate_predictions.csv.gz", index=False, compression="gzip")
    matrix[[
        "origin", "series_name", "date", "actual",
        "selected_historical_prediction", "SALES_SEASONAL",
        "config_available", "review_available_180d", "review_count_180d",
    ]].to_csv(
        OUT / "selected_historical_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    grid.to_csv(OUT / "routing_grid.csv", index=False)
    final_by_origin.to_csv(OUT / "selected_historical_origins.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nConfiguration expert:", flush=True)
    print(selected_structural.to_string(), flush=True)
    print(f"passes={structural_passes}", flush=True)
    print("\nReview expert:", flush=True)
    print(selected_review.to_string(), flush=True)
    print(f"passes={review_passes}", flush=True)
    print("\nSelected historical system:", flush=True)
    print(final_by_origin.to_string(index=False), flush=True)
    print(f"locked_test_targets_read=False output={OUT}", flush=True)


if __name__ == "__main__":
    main()

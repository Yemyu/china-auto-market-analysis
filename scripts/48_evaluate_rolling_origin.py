#!/usr/bin/env python3
"""Evaluate the operational one-month rolling forecast without time leakage.

This is the headline operating mode: before each forecast month, the previous
month's realised sales are known. Model parameters remain fixed within each
six-month evaluation window; only the information cutoff advances month by
month. The fixed-origin recursive protocol is retained as a stress test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import _model_utils as mu


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "forecast"
VALIDATION_OUTPUT = OUT / "rolling_origin_validation.csv"
SUMMARY_OUTPUT = OUT / "rolling_origin_summary.json"
TEST_OUTPUT = OUT / "rolling_origin_test_predictions.csv"
ORIGINS = tuple(pd.Timestamp(value) for value in (
    "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
))
MIN_GATE_GAIN_PP = 1.5
MAX_ORIGIN_REGRESSION_PP = 1.0
VERSIONS = ("BASE", "LOCAL_LEXICON_FIXED")
NAIVE_METHODS = ("LAST_VALUE", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "SEASONAL_LAG12")


def load_forecast_module():
    path = ROOT / "scripts" / "33_evaluate_review_features.py"
    spec = importlib.util.spec_from_file_location("review_forecast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def origin_panel(frame: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = origin + pd.offsets.MonthBegin(6)
    panel = frame.loc[frame["date"].lt(end)].copy()
    panel["split"] = np.where(panel["date"].lt(origin), "train", "val")
    fit = panel.loc[panel["split"].eq("train")].copy()
    return fit, panel


def feature_row(record: pd.Series, history: list[float], columns: list[str]) -> pd.DataFrame:
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
    return pd.DataFrame([row], columns=columns)


def rolling_predictions(
    model,
    panel: pd.DataFrame,
    columns: list[str],
    forecast_split: str,
    history_splits: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, series in panel.groupby("series_name", sort=True):
        ordered = series.sort_values("date").reset_index(drop=True)
        history = ordered.loc[
            ordered["split"].isin(history_splits), mu.TARGET
        ].astype(float).tolist()
        for _, record in ordered.loc[ordered["split"].eq(forecast_split)].iterrows():
            prediction = float(np.expm1(model.predict(feature_row(record, history, columns))[0]))
            prediction = max(prediction, 0.0)
            actual = float(record[mu.TARGET])
            rows.append({
                "series_name": name,
                "date": record["date"],
                "actual": actual,
                "pred": prediction,
            })
            # At the next monthly run, this realised value is legitimately known.
            history.append(actual)
    result = pd.DataFrame(rows)
    expected = panel.loc[panel["split"].eq(forecast_split), ["series_name", "date"]]
    if len(result) != len(expected) or result.duplicated(["series_name", "date"]).any():
        raise ValueError("Rolling-origin prediction coverage mismatch")
    return result


def naive_rolling_predictions(
    panel: pd.DataFrame,
    forecast_split: str,
    history_splits: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, series in panel.groupby("series_name", sort=True):
        ordered = series.sort_values("date").reset_index(drop=True)
        history = ordered.loc[
            ordered["split"].isin(history_splits), mu.TARGET
        ].astype(float).tolist()
        actual_by_date = ordered.set_index("date")[mu.TARGET].astype(float).to_dict()
        for _, record in ordered.loc[ordered["split"].eq(forecast_split)].iterrows():
            values = np.asarray(history, dtype=float)
            last = float(values[-1]) if len(values) else 0.0
            seasonal = actual_by_date.get(record["date"] - pd.DateOffset(years=1), np.nan)
            actual = float(record[mu.TARGET])
            rows.append({
                "series_name": name,
                "date": record["date"],
                "actual": actual,
                "LAST_VALUE": last,
                "ROLLING_MEAN_3": float(np.mean(values[-3:])) if len(values) else 0.0,
                "ROLLING_MEAN_6": float(np.mean(values[-6:])) if len(values) else 0.0,
                "SEASONAL_LAG12": float(seasonal) if np.isfinite(seasonal) else last,
            })
            history.append(actual)
    return pd.DataFrame(rows)


def validation(module, frames, versions) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    historical = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    rows: list[dict[str, Any]] = []
    for origin in ORIGINS:
        fit, panel = origin_panel(historical, origin)
        for version in VERSIONS:
            columns = versions[version]
            model = module.new_model(100)
            model.fit(fit[columns], np.log1p(fit[mu.TARGET]), verbose=False)
            fixed = module.recursive_predictions(
                model, panel, columns, "val", ("train",), version,
                "historical_fixed_origin_comparator",
            )
            rolling = rolling_predictions(model, panel, columns, "val", ("train",))
            for mode, prediction in (("FIXED_SIX_MONTH", fixed), ("ROLLING_ONE_MONTH", rolling)):
                error = float(np.abs(prediction["actual"] - prediction["pred"]).sum())
                volume = float(np.abs(prediction["actual"]).sum())
                rows.append({
                    "version": version,
                    "mode": mode,
                    "origin": origin.strftime("%Y-%m-%d"),
                    "rows": int(len(prediction)),
                    "series": int(prediction["series_name"].nunique()),
                    "absolute_error": error,
                    "actual_volume": volume,
                    "global_volume_weighted_WMAPE": error / volume * 100,
                })
                print(
                    f"[{version}:{mode}] {origin:%Y-%m} "
                    f"WMAPE={rows[-1]['global_volume_weighted_WMAPE']:.3f}",
                    flush=True,
                )
        naive = naive_rolling_predictions(panel, "val", ("train",))
        for method in NAIVE_METHODS:
            error = float(np.abs(naive["actual"] - naive[method]).sum())
            volume = float(np.abs(naive["actual"]).sum())
            rows.append({
                "version": "NAIVE",
                "mode": method,
                "origin": origin.strftime("%Y-%m-%d"),
                "rows": int(len(naive)),
                "series": int(naive["series_name"].nunique()),
                "absolute_error": error,
                "actual_volume": volume,
                "global_volume_weighted_WMAPE": error / volume * 100,
            })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["version", "mode"], as_index=False)
        .agg(
            historical_origins=("origin", "nunique"),
            absolute_error=("absolute_error", "sum"),
            actual_volume=("actual_volume", "sum"),
            mean_origin_WMAPE=("global_volume_weighted_WMAPE", "mean"),
            worst_origin_WMAPE=("global_volume_weighted_WMAPE", "max"),
        )
    )
    summary["pooled_global_WMAPE"] = summary["absolute_error"] / summary["actual_volume"] * 100
    summary = summary.sort_values(["pooled_global_WMAPE", "version", "mode"]).reset_index(drop=True)
    base_fixed = summary.loc[
        summary["version"].eq("BASE") & summary["mode"].eq("FIXED_SIX_MONTH")
    ].iloc[0]
    rolling = summary.loc[summary["mode"].eq("ROLLING_ONE_MONTH")].iloc[0]
    gain = float(base_fixed["pooled_global_WMAPE"] - rolling["pooled_global_WMAPE"])
    fixed_by_origin = detail.loc[
        detail["version"].eq("BASE") & detail["mode"].eq("FIXED_SIX_MONTH")
    ].set_index("origin")["global_volume_weighted_WMAPE"]
    rolling_by_origin = detail.loc[
        detail["version"].eq(rolling["version"]) & detail["mode"].eq("ROLLING_ONE_MONTH")
    ].set_index("origin")["global_volume_weighted_WMAPE"]
    worst_regression = float((rolling_by_origin - fixed_by_origin).max())
    gate = gain >= MIN_GATE_GAIN_PP and worst_regression <= MAX_ORIGIN_REGRESSION_PP
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "task_definition": "monthly refreshed one-month-ahead forecast; previous realised month is available",
        "comparison_task": "six-month recursive fixed-origin forecast",
        "test_used_for_selection": False,
        "gate": {
            "minimum_gain_pp": MIN_GATE_GAIN_PP,
            "maximum_origin_regression_pp": MAX_ORIGIN_REGRESSION_PP,
            "passes": bool(gate),
        },
        "selected_version": str(rolling["version"]),
        "historical_fixed_BASE_WMAPE": float(base_fixed["pooled_global_WMAPE"]),
        "historical_selected_rolling_WMAPE": float(rolling["pooled_global_WMAPE"]),
        "historical_gain_pp": gain,
        "worst_origin_regression_pp": worst_regression,
        "validation_summary": summary.to_dict(orient="records"),
    }
    return detail, summary, payload


def locked_test(module, frames, versions, payload: dict[str, Any]) -> pd.DataFrame:
    if not payload["gate"]["passes"]:
        raise RuntimeError("Historical-validation gate did not pass; locked test remains untouched")
    version = str(payload["selected_version"])
    columns = versions[version]
    train_val = pd.concat(
        [frames["train_roll"], frames["val_roll"]], ignore_index=True
    ).sort_values(["series_name", "date"])
    panel = pd.concat(
        [frames["train_roll"], frames["val_roll"], frames["test_roll"]],
        ignore_index=True,
    ).sort_values(["series_name", "date"])
    model = module.new_model(100)
    model.fit(train_val[columns], np.log1p(train_val[mu.TARGET]), verbose=False)
    prediction = rolling_predictions(model, panel, columns, "test", ("train", "val"))
    naive = naive_rolling_predictions(panel, "test", ("train", "val"))
    prediction = prediction.merge(
        naive[["series_name", "date", *NAIVE_METHODS]],
        on=["series_name", "date"], how="left", validate="one_to_one",
    )
    prediction["version"] = version
    prediction["mode"] = "ROLLING_ONE_MONTH"
    test_wmape = mu.wmape_vol(prediction["actual"], prediction["pred"])
    payload["locked_test"] = {
        "rows": int(len(prediction)),
        "series": int(prediction["series_name"].nunique()),
        "global_volume_weighted_WMAPE": float(test_wmape),
        "naive_WMAPE": {
            method: float(mu.wmape_vol(prediction["actual"], prediction[method]))
            for method in NAIVE_METHODS
        },
    }
    print(f"locked_test_WMAPE={test_wmape:.3f}", flush=True)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_forecast_module()
    frames, versions, _ = module.build_frames()
    detail, summary, payload = validation(module, frames, versions)
    detail.to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")
    if args.test:
        prediction = locked_test(module, frames, versions, payload)
        prediction.to_csv(TEST_OUTPUT, index=False, encoding="utf-8-sig")
    SUMMARY_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(
        f"gate_passes={payload['gate']['passes']} "
        f"selected={payload['selected_version']} gain_pp={payload['historical_gain_pp']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

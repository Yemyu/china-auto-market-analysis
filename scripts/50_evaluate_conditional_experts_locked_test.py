#!/usr/bin/env python3
"""Run the frozen conditional-expert decision once on the locked test.

This script cannot choose routes or thresholds.  It reads the historical
selection written by ``49_evaluate_conditional_experts.py``, verifies that the
selection is frozen, then compares it with the currently published rolling
candidate on 2026-01..06.  Outputs stay under the ignored ``artifacts`` tree.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = ROOT / "artifacts" / "conditional_experts"
HISTORICAL_SUMMARY = OUT / "summary.json"
CURRENT_PREDICTIONS = (
    ROOT / "data" / "processed" / "forecast" / "rolling_origin_test_predictions.csv"
)

# Final acceptance conditions are declared before loading the locked target.
ACCEPTANCE_GATE = {
    "minimum_global_gain_pp": 0.50,
    "minimum_cluster_bootstrap_lower_95_pp": 0.00,
    "maximum_median_series_regression_pp": 0.50,
    "maximum_high_volume_regression_pp": 1.00,
    "maximum_long_tail_regression_pp": 1.00,
}
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 42


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wmape(frame: pd.DataFrame, prediction: str) -> float:
    volume = float(np.abs(frame["actual"]).sum())
    if volume <= 0:
        return float("nan")
    return float(np.abs(frame["actual"] - frame[prediction]).sum() / volume * 100)


def median_series_wmape(frame: pd.DataFrame, prediction: str) -> float:
    scores = []
    for _, group in frame.groupby("series_name", sort=False):
        scores.append(wmape(group, prediction))
    return float(np.nanmedian(scores))


def frozen_selection() -> tuple[dict[str, Any], str]:
    if not HISTORICAL_SUMMARY.exists():
        raise FileNotFoundError(
            "Historical routing summary is missing; run script 49 before opening the test"
        )
    summary = json.loads(HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("locked_test_targets_read") is not False:
        raise ValueError("Historical selection does not certify an untouched test target")
    if summary.get("promotion_origin") != "2025-07-01":
        raise ValueError("Unexpected historical promotion origin")
    selection = str(summary["selected_system_prediction"])
    allowed = {
        "SALES_SEASONAL",
        summary["configuration_expert"].get("deployed_historical_prediction"),
        summary["review_expert"].get("deployed_historical_prediction"),
    }
    if selection not in allowed:
        raise ValueError(f"Historical selection is not a frozen deployed route: {selection}")
    # The first experiment selected the sales-only fallback.  A future expert
    # selection requires a dedicated executor rather than silently guessing.
    if selection != "SALES_SEASONAL":
        raise NotImplementedError(
            f"Locked executor does not implement the newly selected route: {selection}"
        )
    return summary, selection


def candidate_predictions() -> pd.DataFrame:
    rolling = load_module("rolling_origin", SCRIPTS / "48_evaluate_rolling_origin.py")
    train, validation, test = rolling.mu.load_splits()
    features = list(dict.fromkeys(
        rolling.mu.LAG_COLS + rolling.mu.CAL + rolling.mu.SEASONAL_LAG_COLS
    ))
    train_validation = pd.concat([train, validation], ignore_index=True).sort_values(
        ["series_name", "date"]
    )
    panel = pd.concat([train, validation, test], ignore_index=True).sort_values(
        ["series_name", "date"]
    )
    usable = train_validation.dropna(subset=features)
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        **rolling.MODEL_PARAMS["SEASONAL_D5"],
    )
    model.fit(usable[features], np.log1p(usable[rolling.mu.TARGET]), verbose=False)
    prediction = rolling.rolling_predictions(
        model,
        panel,
        features,
        forecast_split="test",
        history_splits=("train", "val"),
    )
    prediction = prediction.rename(columns={"pred": "candidate_pred"})
    if prediction["date"].min() != pd.Timestamp("2026-01-01"):
        raise AssertionError("Locked test begins at an unexpected month")
    if len(prediction) != 2226 or prediction["series_name"].nunique() != 371:
        raise AssertionError("Locked test coverage changed")
    return prediction


def paired_bootstrap(frame: pd.DataFrame) -> dict[str, float]:
    by_series = (
        frame.assign(
            current_error=lambda data: np.abs(data["actual"] - data["current_pred"]),
            candidate_error=lambda data: np.abs(data["actual"] - data["candidate_pred"]),
            volume=lambda data: np.abs(data["actual"]),
        )
        .groupby("series_name", as_index=False)
        .agg(
            current_error=("current_error", "sum"),
            candidate_error=("candidate_error", "sum"),
            volume=("volume", "sum"),
        )
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = by_series[["current_error", "candidate_error", "volume"]].to_numpy(float)
    gains = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
    n_series = len(values)
    for index in range(BOOTSTRAP_REPETITIONS):
        sample = values[rng.integers(0, n_series, size=n_series)]
        volume = sample[:, 2].sum()
        gains[index] = (
            (sample[:, 0].sum() - sample[:, 1].sum()) / volume * 100
            if volume > 0 else np.nan
        )
    return {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "gain_pp_lower_95": float(np.nanquantile(gains, 0.025)),
        "gain_pp_median": float(np.nanquantile(gains, 0.5)),
        "gain_pp_upper_95": float(np.nanquantile(gains, 0.975)),
    }


def segment_regression(frame: pd.DataFrame, segment: str) -> float:
    prior = frame.loc[frame["date"].eq(pd.Timestamp("2026-01-01")), [
        "series_name", "LAST_VALUE"
    ]].rename(columns={"LAST_VALUE": "prior_sales"})
    low = prior["prior_sales"].quantile(0.25)
    high = prior["prior_sales"].quantile(0.75)
    if segment == "high_volume":
        names = prior.loc[prior["prior_sales"].ge(high), "series_name"]
    elif segment == "long_tail":
        names = prior.loc[prior["prior_sales"].le(low), "series_name"]
    else:
        raise KeyError(segment)
    part = frame.loc[frame["series_name"].isin(names)]
    return wmape(part, "candidate_pred") - wmape(part, "current_pred")


def main() -> None:
    historical, selection = frozen_selection()
    candidate = candidate_predictions()
    current = pd.read_csv(CURRENT_PREDICTIONS, parse_dates=["date"])
    current = current.rename(columns={"pred": "current_pred"})
    required = ["series_name", "date", "actual", "current_pred", "LAST_VALUE"]
    missing = set(required) - set(current.columns)
    if missing:
        raise ValueError(f"Current headline predictions lack columns: {sorted(missing)}")
    comparison = candidate.merge(
        current[required],
        on=["series_name", "date", "actual"],
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != 2226:
        raise AssertionError("Candidate/current prediction alignment changed")

    current_wmape = wmape(comparison, "current_pred")
    candidate_wmape = wmape(comparison, "candidate_pred")
    current_median = median_series_wmape(comparison, "current_pred")
    candidate_median = median_series_wmape(comparison, "candidate_pred")
    bootstrap = paired_bootstrap(comparison)
    high_regression = segment_regression(comparison, "high_volume")
    tail_regression = segment_regression(comparison, "long_tail")
    gate_checks = {
        "global_gain": current_wmape - candidate_wmape
        >= ACCEPTANCE_GATE["minimum_global_gain_pp"],
        "bootstrap_lower_95": bootstrap["gain_pp_lower_95"]
        >= ACCEPTANCE_GATE["minimum_cluster_bootstrap_lower_95_pp"],
        "median_series": candidate_median - current_median
        <= ACCEPTANCE_GATE["maximum_median_series_regression_pp"],
        "high_volume": high_regression
        <= ACCEPTANCE_GATE["maximum_high_volume_regression_pp"],
        "long_tail": tail_regression
        <= ACCEPTANCE_GATE["maximum_long_tail_regression_pp"],
    }
    accepted = all(gate_checks.values())

    monthly = []
    for date, group in comparison.groupby("date", sort=True):
        monthly.append({
            "date": date.strftime("%Y-%m-%d"),
            "current_WMAPE": wmape(group, "current_pred"),
            "candidate_WMAPE": wmape(group, "candidate_pred"),
        })
    summary = {
        "schema_version": "v1",
        "historical_selection": selection,
        "historical_summary_generated_at": historical.get("generated_at"),
        "test_rows": int(len(comparison)),
        "test_series": int(comparison["series_name"].nunique()),
        "current_headline_WMAPE": current_wmape,
        "frozen_candidate_WMAPE": candidate_wmape,
        "global_gain_pp": current_wmape - candidate_wmape,
        "current_median_series_WMAPE": current_median,
        "candidate_median_series_WMAPE": candidate_median,
        "median_series_regression_pp": candidate_median - current_median,
        "high_volume_regression_pp": high_regression,
        "long_tail_regression_pp": tail_regression,
        "paired_series_bootstrap": bootstrap,
        "acceptance_gate": ACCEPTANCE_GATE,
        "gate_checks": gate_checks,
        "accepted_for_headline_replacement": accepted,
        "monthly_results": monthly,
        "public_outputs_modified": False,
    }
    comparison.to_csv(
        OUT / "locked_test_predictions.csv.gz", index=False, compression="gzip"
    )
    (OUT / "locked_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

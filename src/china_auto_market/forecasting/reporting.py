"""Describe saved predictions without fitting or selecting a model.

Percentage metrics are returned as 100 times the ratio, not as fractions.
MAPE excludes zero actuals. sMAPE uses every row, with a zero contribution
when both actual and prediction are zero. Net bias permits cancellation;
monthly aggregate WMAPE only permits cancellation within each month.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


METHODS = ("LAST_VALUE", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "SEASONAL_LAG12", "pred")


def prediction_metrics(frame: pd.DataFrame, column: str = "pred") -> dict:
    values = frame[["actual", column]].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Metrics require nonempty, finite, nonnegative sales and predictions")
    actual, predicted = values.T
    error = predicted - actual
    absolute = np.abs(error)
    positive = actual > 0
    denominator = actual + predicted
    volume = actual.sum()
    symmetric = np.divide(200 * absolute, denominator,
                          out=np.zeros_like(absolute), where=denominator > 0)
    return {
        "rows": len(values), "zero_actual_rows": int((~positive).sum()),
        "mape_rows": int(positive.sum()),
        "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, .9)),
        "wmape": float(100 * absolute.sum() / volume) if volume else None,
        "mape_positive": float(100 * np.mean(absolute[positive] / actual[positive]))
        if positive.any() else None,
        "smape": float(symmetric.mean()),
        "mean_error": float(error.mean()),
        "net_bias_pct": float(100 * error.sum() / volume) if volume else None,
    }


def aggregate_metrics(frame: pd.DataFrame, column: str = "pred") -> dict:
    """Aggregate the evaluated cohort, not the entire Chinese market."""
    monthly = frame.groupby("date")[["actual", column]].sum()
    score = prediction_metrics(monthly, column)
    return {
        "monthly_aggregate_wmape": score["wmape"],
        "six_month_net_bias_pct": score["net_bias_pct"],
    }


def forecast_report(root: Path, *, forecast_dir: Path | None = None,
                    split_dir: Path | None = None) -> dict:
    """Return tables shared by the report notebook and dashboard.

    Size groups are fixed by validation-period mean sales, before the test.
    Zero/positive groups describe realised outcomes and are not routing rules.
    """
    directory = root / "data" / "processed"
    forecast_dir = forecast_dir if forecast_dir is not None else directory / "forecast"
    split_dir = split_dir if split_dir is not None else directory / "splits"
    frame = pd.read_csv(forecast_dir / "rolling_origin_test_predictions.csv")
    if frame.duplicated(["series_name", "date"]).any():
        raise ValueError("Duplicate test series-months")
    history = pd.read_csv(split_dir / "val.csv")
    if pd.to_datetime(history.date).max() >= pd.to_datetime(frame.date).min():
        raise ValueError("Size grouping history overlaps test dates")
    historical_mean = history.groupby("series_name").monthly_sales.mean()
    size = pd.qcut(historical_mean, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    frame["size_group"] = frame.series_name.map(size)
    if frame.size_group.isna().any():
        raise ValueError("Missing pre-test size group")

    metrics, monthly, segments, totals = [], [], [], []
    for column in METHODS:
        scored = prediction_metrics(frame, column)
        absolute = (frame.actual - frame[column]).abs()
        per_series = pd.DataFrame({"series": frame.series_name,
                                   "error": absolute, "actual": frame.actual}).groupby("series").sum()
        valid = per_series.actual > 0
        scored["median_series_wmape"] = float(
            (100 * per_series.loc[valid, "error"] / per_series.loc[valid, "actual"]).median()
        )
        metrics.append({"method": column, **scored})
        totals.append({"method": column, **aggregate_metrics(frame, column)})
        if column not in ("pred", "LAST_VALUE"):
            continue
        for date, group in frame.groupby("date", sort=True):
            monthly.append({"method": column, "month": str(date)[:7],
                            **prediction_metrics(group, column)})
        groups = [("pre_test_size", str(name), group)
                  for name, group in frame.groupby("size_group", observed=True)]
        groups += [("actual_sales", name, frame.loc[mask]) for name, mask in (
            ("zero", frame.actual.eq(0)), ("positive", frame.actual.gt(0)))]
        for kind, name, group in groups:
            if not group.empty:
                segments.append({"method": column, "group_type": kind, "group": name,
                                 "series": int(group.series_name.nunique()),
                                 **prediction_metrics(group, column)})
    return {
        "metrics": metrics, "monthly": monthly, "segments": segments, "aggregate": totals,
        "size_group_rule": "quartiles of 2025-07 through 2025-12 mean monthly sales",
        "smape_rule": "all test rows; actual=prediction=0 contributes 0",
        "mape_rule": "actual>0 only; zero-actual errors retained in MAE, RMSE and WMAPE",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.output.resolve().is_relative_to(root / "app") or args.output.resolve().is_relative_to(root / "data"):
        raise ValueError("Stage diagnostics outside public data directories")
    report = forecast_report(root, forecast_dir=args.forecast_dir, split_dir=args.split_dir)
    source = args.forecast_dir / "rolling_origin_test_predictions.csv"
    report["prediction_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["schema_version"] = "v1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit sales-zero semantics and conservative sales/config series matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _series_mapping import build_series_name_mapping, normalize_series_name


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "data_quality"
FORECAST_ORIGIN = pd.Timestamp("2026-01-01")


def longest_positive_run(values: pd.Series) -> int:
    best = current = 0
    for positive in values.gt(0):
        current = current + 1 if positive else 0
        best = max(best, current)
    return int(best)


def build_sales_audit(sales: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    sales = sales.copy()
    sales["date"] = pd.to_datetime(dict(year=sales["year"], month=sales["month"], day=1))
    sales = sales.sort_values(["series_name", "date"])
    rows = []
    match_method = mapping.set_index("sales_series_name")["match_method"]
    config_name = mapping.set_index("sales_series_name")["config_series_name"]
    for name, group in sales.groupby("series_name", sort=True):
        train = group[group["date"] < FORECAST_ORIGIN]
        test = group[group["date"] >= FORECAST_ORIGIN]
        y2024 = group[group["date"].between("2024-01-01", "2024-12-01")]
        positive_months = int(group["monthly_sales"].gt(0).sum())
        train_positive = int(train["monthly_sales"].gt(0).sum())
        test_positive = int(test["monthly_sales"].gt(0).sum())
        avg_2024 = float(y2024["monthly_sales"].mean()) if len(y2024) else np.nan
        row = {
            "series_id": group["series_id"].iloc[0],
            "series_name": name,
            "brand": group["brand"].iloc[0],
            "category": group["category"].iloc[0],
            "row_months": int(len(group)),
            "positive_months": positive_months,
            "zero_months": int(group["monthly_sales"].eq(0).sum()),
            "zero_share": float(group["monthly_sales"].eq(0).mean()),
            "longest_positive_run": longest_positive_run(group["monthly_sales"]),
            "train_positive_months": train_positive,
            "test_positive_months": test_positive,
            "test_sales_sum": float(test["monthly_sales"].sum()),
            "avg_monthly_sales_2024": avg_2024,
            "config_series_name": config_name.get(name, ""),
            "config_match_method": match_method.get(name, "unmatched"),
        }
        row.update(
            {
                "flag_never_positive": positive_months == 0,
                "flag_positive_run_lt24": row["longest_positive_run"] < 24,
                "flag_no_positive_before_forecast": train_positive == 0,
                "flag_all_zero_test": len(test) > 0 and test_positive == 0,
                "flag_high_2024_then_zero_test": avg_2024 >= 500 and len(test) > 0 and test_positive == 0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_mapping_audit(
    sales: pd.DataFrame, config: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    sales_meta = sales[["series_name", "series_id", "source_series_id", "brand"]].drop_duplicates(
        "series_name"
    )
    config_meta = config[["series_name", "series_id", "brand_name"]].drop_duplicates("series_name")
    matched = mapping.merge(
        sales_meta.rename(columns={"series_name": "sales_series_name"}),
        on="sales_series_name",
        how="left",
    ).merge(
        config_meta.rename(columns={"series_name": "config_series_name", "series_id": "config_series_id"}),
        on="config_series_name",
        how="left",
    )
    matched["status"] = "matched"

    unmatched_sales = sales_meta.loc[
        ~sales_meta["series_name"].isin(mapping["sales_series_name"])
    ].rename(columns={"series_name": "sales_series_name"})
    unmatched_sales["config_series_name"] = ""
    unmatched_sales["config_series_id"] = np.nan
    unmatched_sales["brand_name"] = ""
    unmatched_sales["match_method"] = "unmatched_sales"
    unmatched_sales["normalized_key"] = unmatched_sales["sales_series_name"].map(normalize_series_name)
    unmatched_sales["status"] = "unmatched"

    unmatched_config = config_meta.loc[
        ~config_meta["series_name"].isin(mapping["config_series_name"])
    ].rename(columns={"series_name": "config_series_name", "series_id": "config_series_id"})
    unmatched_config["sales_series_name"] = ""
    unmatched_config["series_id"] = np.nan
    unmatched_config["source_series_id"] = ""
    unmatched_config["brand"] = ""
    unmatched_config["match_method"] = "unmatched_config"
    unmatched_config["normalized_key"] = unmatched_config["config_series_name"].map(normalize_series_name)
    unmatched_config["status"] = "unmatched"

    columns = [
        "status", "match_method", "normalized_key", "sales_series_name", "series_id",
        "source_series_id", "brand", "config_series_name", "config_series_id", "brand_name",
    ]
    return pd.concat([matched, unmatched_sales, unmatched_config], ignore_index=True)[columns]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(RAW / "monthly_sales.csv", low_memory=False)
    config = pd.read_csv(RAW / "feature.csv", low_memory=False)
    mapping = build_series_name_mapping(sales["series_name"], config["series_name"])
    sales_audit = build_sales_audit(sales, mapping)
    mapping_audit = build_mapping_audit(sales, config, mapping)

    sales_audit.to_csv(OUT / "sales_zero_audit.csv", index=False, encoding="utf-8-sig")
    mapping_audit.to_csv(OUT / "series_mapping_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "schema_version": "v1",
        "source_rows": {"sales": int(len(sales)), "config": int(len(config))},
        "source_series": {
            "sales": int(sales["series_name"].nunique()),
            "config": int(config["series_name"].nunique()),
        },
        "mapping": {
            "exact": int(mapping["match_method"].eq("exact_name").sum()),
            "safe_normalized": int(mapping["match_method"].eq("unique_normalized_name").sum()),
            "total_safe_matches": int(len(mapping)),
            "unmatched_sales": int(sales["series_name"].nunique() - len(mapping)),
            "unmatched_config": int(config["series_name"].nunique() - len(mapping)),
        },
        "sales_zero_audit": {
            "zero_rows": int(sales["monthly_sales"].eq(0).sum()),
            "zero_row_share": float(sales["monthly_sales"].eq(0).mean()),
            "never_positive_series": int(sales_audit["flag_never_positive"].sum()),
            "positive_run_lt24_series": int(sales_audit["flag_positive_run_lt24"].sum()),
            "no_positive_before_forecast_series": int(
                sales_audit["flag_no_positive_before_forecast"].sum()
            ),
            "all_zero_test_series": int(sales_audit["flag_all_zero_test"].sum()),
            "high_2024_then_zero_test_series": int(
                sales_audit["flag_high_2024_then_zero_test"].sum()
            ),
        },
        "decision": (
            "Do not reinterpret zero rows automatically. Cross-source verification and lifecycle labels "
            "are required before rebuilding the forecast cohort."
        ),
    }
    (OUT / "data_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the business marts with the frozen reference inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from china_auto_market.features.configuration import CFG_COLS
from china_auto_market.marts.user_needs import build_latest_snapshot
from china_auto_market.quality.sales_repair import apply_verified_annual_sales_corrections
from china_auto_market.warehouse.sources import (
    load_forecast_feature_mart,
    load_product_analysis_mart,
    load_raw_configuration,
    load_standard_review_labels,
    load_standard_sales,
    load_user_needs_mart,
    staging_configuration_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-path", default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/processed/splits"))
    args = parser.parse_args()

    mart_forecast = load_forecast_feature_mart(args.login_path)
    frozen = pd.concat(
        [pd.read_csv(args.split_dir / f"{split}.csv", parse_dates=["date"])
         for split in ("train", "val", "test")],
        ignore_index=True,
    )
    forecast_columns = [
        "series_name", "series_id", "date", "year", "month", "brand", "category",
        "category_en", "monthly_sales", "lag_1", "lag_2", "lag_3", "roll_mean_3",
        "roll_mean_6", "month_sin", "month_cos", "lag_12", "roll_mean_12", "split",
    ]
    assert_frame_equal(
        mart_forecast[forecast_columns].sort_values(["date", "series_name"]).reset_index(drop=True),
        frozen[forecast_columns].sort_values(["date", "series_name"]).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    if set(CFG_COLS) & set(mart_forecast.columns):
        raise ValueError("Forecast mart still contains globally preprocessed configuration values")
    batch = mart_forecast.attrs.get("configuration_batch_id")
    if batch is None or batch != staging_configuration_batch(args.login_path):
        raise ValueError("Forecast configuration reference does not match its standardized source batch")
    config = load_raw_configuration(args.login_path, batch_id=batch)
    sales = load_standard_sales(args.login_path)
    expected_product, audit = apply_verified_annual_sales_corrections(config, sales)
    expected_product = expected_product.loc[
        expected_product["annual_sales"].notna() & expected_product["year"].isin([2022, 2023, 2024, 2025])
    ].copy()
    mart_product = load_product_analysis_mart(args.login_path)
    source_columns = list(config.columns)
    assert_frame_equal(
        mart_product[source_columns].sort_values(["series_name", "year"]).reset_index(drop=True),
        expected_product[source_columns].sort_values(["series_name", "year"]).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_user_needs, _ = build_latest_snapshot(load_standard_review_labels(args.login_path))
    mart_user_needs = load_user_needs_mart(args.login_path)
    user_needs_columns = [
        "series_name", "monitoring_month", "aspect_code", "review_count_180d",
        "mention_count_180d", "positive_rate_180d", "negative_rate_180d",
        "risk_level", "information_cutoff_exclusive",
    ]
    assert_frame_equal(
        mart_user_needs[user_needs_columns]
        .sort_values(["series_name", "aspect_code"])
        .reset_index(drop=True),
        expected_user_needs[user_needs_columns]
        .sort_values(["series_name", "aspect_code"])
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    result = {
        "passed": True,
        "forecast_rows": len(mart_forecast),
        "forecast_series": int(mart_forecast["series_name"].nunique()),
        "forecast_value_parity": True,
        "forecast_configuration_batch_id": batch,
        "forecast_configuration_policy": mart_forecast.attrs["configuration_policy"],
        "product_rows": len(mart_product),
        "product_series": int(mart_product["series_name"].nunique()),
        "product_value_parity": True,
        "annual_repair_rows": len(audit),
        "annual_repair_delta": int(audit["sales_delta"].sum()),
        "user_needs_rows": len(mart_user_needs),
        "user_needs_series": int(mart_user_needs["series_name"].nunique()),
        "user_needs_aspects": int(mart_user_needs["aspect_code"].nunique()),
        "user_needs_value_parity": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

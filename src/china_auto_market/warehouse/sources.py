"""Stable pandas adapters for verified warehouse source layers."""

from __future__ import annotations

import pandas as pd

from china_auto_market.warehouse.schema import query_json_rows


def load_standard_sales(login_path: str) -> pd.DataFrame:
    """Reconstruct the locked standardized sales panel from staging."""
    sql = """
    SELECT JSON_OBJECT(
      'year', YEAR(s.month_start),
      'month', MONTH(s.month_start),
      'series_id', r.series_id,
      'series_name', s.canonical_series_name,
      'brand', s.brand_name,
      'category', s.category_name,
      'monthly_sales', s.monthly_sales,
      'data_source', IF(s.repair_applied, 'pcauto_verified_overlay', 'pcauto'),
      'period', YEAR(s.month_start) * 12 + MONTH(s.month_start) - 1,
      'date', DATE_FORMAT(s.month_start, '%Y-%m-%d'),
      'category_en', CASE s.category_name
        WHEN '轿车' THEN 'Sedan' WHEN 'SUV' THEN 'SUV' WHEN 'MPV' THEN 'MPV' ELSE NULL END
    )
    FROM auto_staging.stg_monthly_sales s
    JOIN auto_raw.raw_sales r ON r.raw_sales_id = s.raw_sales_id
    ORDER BY s.raw_sales_id;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("auto_staging.stg_monthly_sales is empty")
    ordered = [
        "year", "month", "series_id", "series_name", "brand", "category",
        "monthly_sales", "data_source", "period", "date", "category_en",
    ]
    frame = frame[ordered]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def load_raw_configuration(login_path: str) -> pd.DataFrame:
    """Return the latest successful full configuration snapshot from raw JSON."""
    sql = """
    SELECT c.source_payload
    FROM auto_raw.raw_vehicle_config c
    JOIN auto_ops.ingestion_batches b ON b.batch_id = c.batch_id
    WHERE BINARY b.dataset_name = BINARY 'config'
      AND b.status = 'succeeded'
      AND c.batch_id = (
        SELECT MAX(batch_id) FROM auto_ops.ingestion_batches
        WHERE BINARY dataset_name = BINARY 'config' AND status = 'succeeded'
      )
    ORDER BY c.source_row_number;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("No successful raw configuration snapshot is available")
    return frame


def load_standard_reviews(login_path: str) -> pd.DataFrame:
    """Return standardized reviews with source payload fields and original content."""
    sql = """
    SELECT JSON_MERGE_PATCH(
      COALESCE(r.source_payload, JSON_OBJECT()),
      JSON_OBJECT(
        'identity', s.source_identity,
        'series_name_canonical', s.canonical_series_name,
        'publish_time', DATE_FORMAT(s.publish_time, '%Y-%m-%d %H:%i:%s'),
        'content', s.content,
        'eligible_for_temporal_model', s.eligible_for_temporal_model
      )
    )
    FROM auto_staging.stg_reviews s
    JOIN auto_raw.raw_reviews r ON r.raw_review_id = s.raw_review_id
    ORDER BY r.source_row_number;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("auto_staging.stg_reviews is empty")
    return frame


def load_standard_review_labels(login_path: str) -> pd.DataFrame:
    """Return the raw-width ten-aspect label rows used by current consumers."""
    sql = """
    SELECT l.source_payload
    FROM auto_raw.raw_review_labels l
    JOIN auto_ops.ingestion_batches b ON b.batch_id = l.batch_id
    WHERE BINARY b.dataset_name = BINARY 'review_labels'
      AND b.status = 'succeeded'
      AND l.batch_id = (
        SELECT MAX(batch_id) FROM auto_ops.ingestion_batches
        WHERE BINARY dataset_name = BINARY 'review_labels' AND status = 'succeeded'
      )
    ORDER BY l.source_row_number;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("No successful raw review-label snapshot is available")
    # Match pandas CSV missing-value semantics for optional boolean/object
    # fields so existing consumers behave identically across backends.
    return frame.replace({None: float("nan")})


def load_forecast_feature_mart(login_path: str) -> pd.DataFrame:
    """Return the forecast mart as a pandas-ready, point-in-time feature panel."""
    sql = """
    SELECT JSON_MERGE_PATCH(
      f.configuration_payload,
      COALESCE(f.review_feature_payload, JSON_OBJECT()),
      JSON_OBJECT(
        'series_name', d.series_name,
        'series_id', JSON_EXTRACT(d.source_identifiers_json, '$.series_id'),
        'date', DATE_FORMAT(dt.full_date, '%Y-%m-%d'),
        'year', YEAR(dt.full_date),
        'month', MONTH(dt.full_date),
        'brand', d.brand_name,
        'category', d.category_name,
        'category_en', CASE d.category_name
          WHEN '轿车' THEN 'Sedan' WHEN 'SUV' THEN 'SUV' WHEN 'MPV' THEN 'MPV' ELSE NULL END,
        'monthly_sales', sales.monthly_sales,
        'lag_1', f.lag_1, 'lag_2', f.lag_2, 'lag_3', f.lag_3, 'lag_12', f.lag_12,
        'roll_mean_3', f.roll_mean_3, 'roll_mean_6', f.roll_mean_6,
        'roll_mean_12', f.roll_mean_12,
        'month_sin', f.month_sin, 'month_cos', f.month_cos,
        'split', f.split_name,
        'information_cutoff_exclusive', DATE_FORMAT(
          f.review_information_cutoff_exclusive, '%Y-%m-%d'
        )
      )
    )
    FROM auto_mart.mart_forecast_features f
    JOIN auto_mart.dim_vehicle_series d ON d.vehicle_series_sk = f.vehicle_series_sk
    JOIN auto_mart.dim_date dt ON dt.date_key = f.target_month_date_key
    JOIN auto_mart.fact_monthly_sales sales
      ON sales.vehicle_series_sk = f.vehicle_series_sk
     AND sales.month_date_key = f.target_month_date_key
    ORDER BY dt.full_date, d.series_name;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("auto_mart.mart_forecast_features is empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def load_product_analysis_mart(login_path: str) -> pd.DataFrame:
    """Return the complete-year product panel with verified annual targets."""
    sql = """
    SELECT configuration_payload
    FROM auto_mart.mart_product_analysis p
    JOIN auto_mart.dim_vehicle_series d USING (vehicle_series_sk)
    ORDER BY d.series_name, p.model_year;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("auto_mart.mart_product_analysis is empty")
    return frame


def load_user_needs_mart(login_path: str) -> pd.DataFrame:
    """Return the latest leakage-safe series-by-aspect monitoring snapshot."""
    sql = """
    SELECT JSON_OBJECT(
      'series_name', d.series_name,
      'monitoring_month', DATE_FORMAT(dt.full_date, '%Y-%m-%d'),
      'aspect_code', u.aspect_code,
      'review_count_180d', u.review_count_180d,
      'mention_count_180d', u.mention_count_180d,
      'positive_rate_180d', u.positive_rate_180d,
      'negative_rate_180d', u.negative_rate_180d,
      'risk_level', u.risk_level,
      'information_cutoff_exclusive', DATE_FORMAT(
        u.information_cutoff_exclusive, '%Y-%m-%d %H:%i:%s'
      )
    )
    FROM auto_mart.mart_user_needs u
    JOIN auto_mart.dim_vehicle_series d USING (vehicle_series_sk)
    JOIN auto_mart.dim_date dt ON dt.date_key = u.monitoring_month_date_key
    ORDER BY d.series_name, u.aspect_code;
    """
    frame = pd.DataFrame(query_json_rows(login_path, sql))
    if frame.empty:
        raise RuntimeError("auto_mart.mart_user_needs is empty")
    frame["monitoring_month"] = pd.to_datetime(frame["monitoring_month"], errors="raise")
    frame["information_cutoff_exclusive"] = pd.to_datetime(
        frame["information_cutoff_exclusive"], errors="raise"
    )
    return frame

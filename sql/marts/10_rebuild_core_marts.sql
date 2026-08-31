USE auto_mart;

DELETE FROM auto_mart.mart_forecast_predictions;
DELETE FROM auto_mart.mart_forecast_features;
DELETE FROM auto_mart.mart_user_needs;
DELETE FROM auto_mart.mart_product_analysis;
DELETE FROM auto_mart.fact_monthly_sales;
DELETE FROM auto_mart.dim_date;
DELETE FROM auto_mart.dim_vehicle_series;

INSERT INTO auto_ops.dataset_versions (
  dataset_name, version_key, grain_description, row_count, schema_version,
  pipeline_run_id, upstream_versions_json
) VALUES
  ('fact_monthly_sales', 'sales-b2-c3-de5-v1', 'canonical vehicle series x calendar month',
   54918, 'de5-v1', __PIPELINE_RUN_ID__, JSON_OBJECT('sales_batch', 2, 'correction_batch', 3)),
  ('mart_forecast_features', 'forecast-b2-c4-r5-l8-de5-v1', 'locked series x forecast target month',
   17808, 'de5-v1', __PIPELINE_RUN_ID__, JSON_OBJECT('sales_batch', 2, 'config_batch', 4, 'review_batch', 5, 'label_batch', 8)),
  ('mart_product_analysis', 'product-b2-a9-c4-de5-v1', 'canonical vehicle series x complete model year',
   1510, 'de5-v1', __PIPELINE_RUN_ID__, JSON_OBJECT('annual_correction_batch', 9, 'config_batch', 4)),
  ('mart_user_needs', 'user-needs-r5-l8-202607-de5-v1', 'canonical vehicle series x monitoring month x aspect',
   3000, 'de5-v1', __PIPELINE_RUN_ID__, JSON_OBJECT('review_batch', 5, 'label_batch', 8, 'cutoff_exclusive', '2026-08-01'))
ON DUPLICATE KEY UPDATE
  grain_description = VALUES(grain_description),
  row_count = VALUES(row_count),
  schema_version = VALUES(schema_version),
  pipeline_run_id = VALUES(pipeline_run_id),
  upstream_versions_json = VALUES(upstream_versions_json),
  created_at = CURRENT_TIMESTAMP(6);

INSERT INTO auto_mart.dim_vehicle_series (
  canonical_series_key, series_name, brand_name, category_name,
  source_identifiers_json, first_sales_month, last_sales_month, is_active
)
SELECT
  s.canonical_series_key,
  MAX(s.canonical_series_name),
  MAX(s.brand_name),
  MAX(s.category_name),
  JSON_OBJECT(
    'source_series_id', MAX(s.source_series_id),
    'series_id', MAX(r.series_id)
  ),
  MIN(s.month_start),
  MAX(s.month_start),
  TRUE
FROM auto_staging.stg_monthly_sales s
JOIN auto_raw.raw_sales r ON r.raw_sales_id = s.raw_sales_id
GROUP BY s.canonical_series_key;

INSERT INTO auto_mart.dim_vehicle_series (
  canonical_series_key, series_name, brand_name, category_name,
  source_identifiers_json, first_sales_month, last_sales_month, is_active
)
SELECT
  c.canonical_series_key,
  MAX(c.canonical_series_name),
  MAX(c.brand_name),
  NULL,
  JSON_OBJECT('config_only', TRUE),
  NULL,
  NULL,
  TRUE
FROM auto_staging.stg_vehicle_config c
WHERE c.canonical_series_key LIKE 'config:%'
GROUP BY c.canonical_series_key;

INSERT INTO auto_mart.dim_date (
  date_key, full_date, calendar_year, calendar_quarter, calendar_month,
  month_start, month_end, year_month_label
)
SELECT
  CAST(DATE_FORMAT(d.full_date, '%Y%m%d') AS UNSIGNED),
  d.full_date,
  YEAR(d.full_date),
  QUARTER(d.full_date),
  MONTH(d.full_date),
  DATE_SUB(d.full_date, INTERVAL DAY(d.full_date) - 1 DAY),
  LAST_DAY(d.full_date),
  DATE_FORMAT(d.full_date, '%Y-%m')
FROM (
  SELECT DATE_ADD('2014-01-01', INTERVAL numbers.n DAY) AS full_date
  FROM (
    SELECT ones.n + tens.n * 10 + hundreds.n * 100 + thousands.n * 1000 AS n
    FROM
      (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
       UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) ones
    CROSS JOIN
      (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
       UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) tens
    CROSS JOIN
      (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
       UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) hundreds
    CROSS JOIN
      (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
       UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) thousands
  ) numbers
) d
WHERE d.full_date <= '2027-12-31';

INSERT INTO auto_mart.fact_monthly_sales (
  vehicle_series_sk, month_date_key, monthly_sales, monthly_sales_raw,
  repair_applied, source_batch_id, dataset_version_id
)
SELECT
  dv.vehicle_series_sk,
  CAST(DATE_FORMAT(s.month_start, '%Y%m%d') AS UNSIGNED),
  s.monthly_sales,
  s.monthly_sales_raw,
  s.repair_applied,
  s.batch_id,
  (SELECT dataset_version_id FROM auto_ops.dataset_versions
   WHERE dataset_name = 'fact_monthly_sales' AND version_key = 'sales-b2-c3-de5-v1')
FROM auto_staging.stg_monthly_sales s
JOIN auto_mart.dim_vehicle_series dv
  ON dv.canonical_series_key = s.canonical_series_key;

INSERT INTO auto_mart.mart_product_analysis (
  vehicle_series_sk, model_year, annual_sales, configuration_payload,
  dataset_version_id
)
SELECT
  dv.vehicle_series_sk,
  c.model_year,
  c.annual_sales,
  JSON_SET(
    r.source_payload,
    '$.annual_sales',
    c.annual_sales
  ),
  (SELECT dataset_version_id FROM auto_ops.dataset_versions
   WHERE dataset_name = 'mart_product_analysis' AND version_key = 'product-b2-a9-c4-de5-v1')
FROM auto_staging.stg_vehicle_config c
JOIN auto_raw.raw_vehicle_config r ON r.raw_config_id = c.raw_config_id
JOIN auto_mart.dim_vehicle_series dv
  ON dv.canonical_series_key = c.canonical_series_key
WHERE c.model_year BETWEEN 2022 AND 2025
  AND c.annual_sales IS NOT NULL;

CREATE TEMPORARY TABLE auto_mart._de5_forecast_input (
  canonical_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  target_month_date_key INT UNSIGNED NOT NULL,
  split_name VARCHAR(16) COLLATE utf8mb4_0900_as_cs NOT NULL,
  lag_1 DOUBLE NULL,
  lag_2 DOUBLE NULL,
  lag_3 DOUBLE NULL,
  lag_12 DOUBLE NULL,
  roll_mean_3 DOUBLE NULL,
  roll_mean_6 DOUBLE NULL,
  roll_mean_12 DOUBLE NULL,
  month_sin DOUBLE NOT NULL,
  month_cos DOUBLE NOT NULL,
  model_year SMALLINT UNSIGNED NOT NULL,
  configuration_payload JSON NOT NULL,
  review_feature_payload JSON NULL,
  review_information_cutoff_exclusive DATETIME(6) NULL,
  PRIMARY KEY (canonical_series_name, target_month_date_key)
);

CREATE TEMPORARY TABLE auto_mart._de5_user_needs_input (
  canonical_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  monitoring_month_date_key INT UNSIGNED NOT NULL,
  aspect_code VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  review_count_180d INT UNSIGNED NOT NULL,
  mention_count_180d INT UNSIGNED NOT NULL,
  positive_rate_180d DOUBLE NULL,
  negative_rate_180d DOUBLE NULL,
  risk_level VARCHAR(16) COLLATE utf8mb4_0900_as_cs NOT NULL,
  information_cutoff_exclusive DATETIME(6) NOT NULL,
  PRIMARY KEY (canonical_series_name, monitoring_month_date_key, aspect_code)
);

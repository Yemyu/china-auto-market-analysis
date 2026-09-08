INSERT INTO auto_ops.data_quality_results (
  pipeline_run_id, dataset_name, rule_name, rule_version, severity,
  passed, observed_value, expected_value, affected_rows, details_json
)
SELECT __PIPELINE_RUN_ID__, 'dim_vehicle_series', 'series_dimension_rows', 'de5-v1', 'critical',
       COUNT(*) = 1412, CAST(COUNT(*) AS CHAR), '1412', ABS(COUNT(*) - 1412), NULL
FROM auto_mart.dim_vehicle_series
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'dim_date', 'date_dimension_rows', 'de5-v1', 'critical',
       COUNT(*) = 5113, CAST(COUNT(*) AS CHAR), '5113', ABS(COUNT(*) - 5113), NULL
FROM auto_mart.dim_date
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'fact_monthly_sales', 'sales_fact_rows', 'de5-v1', 'critical',
       COUNT(*) = 54918, CAST(COUNT(*) AS CHAR), '54918', ABS(COUNT(*) - 54918), NULL
FROM auto_mart.fact_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'fact_monthly_sales', 'sales_fact_totals', 'de5-v1', 'critical',
       SUM(monthly_sales) = 71650793 AND SUM(monthly_sales_raw) = 70293235
       AND SUM(repair_applied) = 65,
       CONCAT_WS('/', SUM(monthly_sales), SUM(monthly_sales_raw), SUM(repair_applied)),
       '71650793/70293235/65', 0, NULL
FROM auto_mart.fact_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_forecast_features', 'forecast_feature_rows', 'de5-v1', 'critical',
       COUNT(*) = 17808 AND COUNT(DISTINCT vehicle_series_sk) = 371,
       CONCAT_WS('/', COUNT(*), COUNT(DISTINCT vehicle_series_sk)), '17808/371',
       ABS(COUNT(*) - 17808), NULL
FROM auto_mart.mart_forecast_features
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_forecast_features', 'forecast_split_rows', 'de5-v1', 'critical',
       SUM(split_name='train') = 13356 AND SUM(split_name='val') = 2226
       AND SUM(split_name='test') = 2226,
       CONCAT_WS('/', SUM(split_name='train'), SUM(split_name='val'), SUM(split_name='test')),
       '13356/2226/2226', 0, NULL
FROM auto_mart.mart_forecast_features
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_forecast_features', 'forecast_point_in_time_cutoff', 'de5-v1', 'critical',
       SUM(review_information_cutoff_exclusive > STR_TO_DATE(
         CAST(target_month_date_key AS CHAR), '%Y%m%d'
       )) = 0,
       CAST(SUM(review_information_cutoff_exclusive > STR_TO_DATE(
         CAST(target_month_date_key AS CHAR), '%Y%m%d'
       )) AS CHAR), '0',
       SUM(review_information_cutoff_exclusive > STR_TO_DATE(
         CAST(target_month_date_key AS CHAR), '%Y%m%d'
       )), NULL
FROM auto_mart.mart_forecast_features
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_forecast_features', 'forecast_configuration_source_reference', 'r1-v2', 'critical',
       SUM(COALESCE(
         JSON_LENGTH(f.configuration_payload) = 2
         AND JSON_UNQUOTE(JSON_EXTRACT(f.configuration_payload, '$.configuration_policy')) = 'raw-batch-reference-v1'
         AND JSON_TYPE(JSON_EXTRACT(f.configuration_payload, '$.configuration_batch_id')) = 'INTEGER'
         AND JSON_EXTRACT(f.configuration_payload, '$.configuration_batch_id') = JSON_EXTRACT(v.upstream_versions_json, '$.config_batch')
         AND v.schema_version = 'r1-v2'
         AND b.status = 'succeeded' AND BINARY b.dataset_name = BINARY 'config'
         AND EXISTS (SELECT 1 FROM auto_raw.raw_vehicle_config r WHERE r.batch_id = b.batch_id), FALSE)) = COUNT(*)
       AND COUNT(DISTINCT b.batch_id) = 1,
       CONCAT_WS('/', COUNT(*), COUNT(DISTINCT b.batch_id)), '17808 rows / one successful bound config batch',
       0, NULL
FROM auto_mart.mart_forecast_features f
JOIN auto_ops.dataset_versions v ON v.dataset_version_id = f.dataset_version_id
LEFT JOIN auto_ops.ingestion_batches b
  ON b.batch_id = CAST(JSON_UNQUOTE(JSON_EXTRACT(f.configuration_payload, '$.configuration_batch_id')) AS UNSIGNED)
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_product_analysis', 'product_rows_and_series', 'de5-v1', 'critical',
       COUNT(*) = 1510 AND COUNT(DISTINCT vehicle_series_sk) = 646,
       CONCAT_WS('/', COUNT(*), COUNT(DISTINCT vehicle_series_sk)), '1510/646',
       ABS(COUNT(*) - 1510), NULL
FROM auto_mart.mart_product_analysis
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_user_needs', 'user_needs_rows_series_aspects', 'de5-v1', 'critical',
       COUNT(*) = 3000 AND COUNT(DISTINCT vehicle_series_sk) = 300
       AND COUNT(DISTINCT aspect_code) = 10,
       CONCAT_WS('/', COUNT(*), COUNT(DISTINCT vehicle_series_sk), COUNT(DISTINCT aspect_code)),
       '3000/300/10', ABS(COUNT(*) - 3000), NULL
FROM auto_mart.mart_user_needs
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'mart_user_needs', 'user_needs_current_risk_snapshot', 'de5-v1', 'critical',
       SUM(risk_level='normal') = 2990 AND SUM(risk_level='watch') = 10
       AND SUM(risk_level='alert') = 0
       AND MIN(information_cutoff_exclusive) = '2026-08-01 00:00:00'
       AND MAX(information_cutoff_exclusive) = '2026-08-01 00:00:00',
       CONCAT_WS('/', SUM(risk_level='normal'), SUM(risk_level='watch'), SUM(risk_level='alert')),
       '2990/10/0', 0, NULL
FROM auto_mart.mart_user_needs;

CREATE TEMPORARY TABLE auto_mart._de5_core_gate (
  rule_name VARCHAR(128) NOT NULL,
  passed BOOLEAN NOT NULL,
  CONSTRAINT ck_de5_core_pass CHECK (passed = TRUE)
);

INSERT INTO auto_mart._de5_core_gate (rule_name, passed)
SELECT rule_name, passed
FROM auto_ops.data_quality_results
WHERE pipeline_run_id = __PIPELINE_RUN_ID__
  AND severity = 'critical'
  AND passed = FALSE;

DROP TEMPORARY TABLE auto_mart._de5_core_gate;

UPDATE auto_ops.pipeline_runs
SET status = 'succeeded', completed_at = NOW(6), error_message = NULL
WHERE pipeline_run_id = __PIPELINE_RUN_ID__;

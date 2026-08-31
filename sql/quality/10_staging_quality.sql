-- Persist successful checks in the same transaction as the staging snapshot.
INSERT INTO auto_ops.data_quality_results (
  pipeline_run_id, dataset_name, rule_name, rule_version, severity,
  passed, observed_value, expected_value, affected_rows, details_json
)
SELECT __PIPELINE_RUN_ID__, 'series_name_mappings', 'mapping_row_count', 'de4-v1', 'critical',
       COUNT(*) = __EXPECTED_MAPPING_ROWS__, CAST(COUNT(*) AS CHAR),
       CAST(__EXPECTED_MAPPING_ROWS__ AS CHAR), ABS(COUNT(*) - __EXPECTED_MAPPING_ROWS__), NULL
FROM auto_staging.series_name_mappings
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'series_name_mappings', 'mapping_source_breakdown', 'de4-v1', 'critical',
       SUM(source_system = 'pcauto_sales') = __EXPECTED_SALES_MAPPING_ROWS__
       AND SUM(source_system = 'pcauto_config' AND match_method = 'exact_name_to_sales')
           = __EXPECTED_CONFIG_SALES_MAPPING_ROWS__
       AND SUM(source_system = 'pcauto_config' AND match_method = 'config_only_source_id')
           = __EXPECTED_CONFIG_ONLY_MAPPING_ROWS__
       AND SUM(source_system IN ('autohome', 'dongchedi')) = __EXPECTED_REVIEW_MAPPING_ROWS__,
       CONCAT_WS('/', SUM(source_system = 'pcauto_sales'),
                     SUM(source_system = 'pcauto_config' AND match_method = 'exact_name_to_sales'),
                     SUM(source_system = 'pcauto_config' AND match_method = 'config_only_source_id'),
                     SUM(source_system IN ('autohome', 'dongchedi'))),
       CONCAT_WS('/', __EXPECTED_SALES_MAPPING_ROWS__, __EXPECTED_CONFIG_SALES_MAPPING_ROWS__,
                     __EXPECTED_CONFIG_ONLY_MAPPING_ROWS__, __EXPECTED_REVIEW_MAPPING_ROWS__),
       0, NULL
FROM auto_staging.series_name_mappings
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'sales_row_count', 'de4-v1', 'critical',
       COUNT(*) = __EXPECTED_SALES_ROWS__, CAST(COUNT(*) AS CHAR),
       CAST(__EXPECTED_SALES_ROWS__ AS CHAR), ABS(COUNT(*) - __EXPECTED_SALES_ROWS__), NULL
FROM auto_staging.stg_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'sales_series_count', 'de4-v1', 'critical',
       COUNT(DISTINCT canonical_series_key) = __EXPECTED_SALES_SERIES__,
       CAST(COUNT(DISTINCT canonical_series_key) AS CHAR),
       CAST(__EXPECTED_SALES_SERIES__ AS CHAR),
       ABS(COUNT(DISTINCT canonical_series_key) - __EXPECTED_SALES_SERIES__), NULL
FROM auto_staging.stg_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'sales_complete_month_grid', 'de4-v1', 'critical',
       MIN(series_months) = __EXPECTED_SALES_MONTHS_PER_SERIES__
       AND MAX(series_months) = __EXPECTED_SALES_MONTHS_PER_SERIES__
       AND MIN(first_month) = '2022-01-01' AND MAX(last_month) = '2026-06-01',
       CONCAT_WS('/', MIN(series_months), MAX(series_months), MIN(first_month), MAX(last_month)),
       CONCAT_WS('/', __EXPECTED_SALES_MONTHS_PER_SERIES__, __EXPECTED_SALES_MONTHS_PER_SERIES__,
                     '2022-01-01', '2026-06-01'),
       SUM(series_months <> __EXPECTED_SALES_MONTHS_PER_SERIES__), NULL
FROM (
  SELECT canonical_series_key, COUNT(*) AS series_months,
         MIN(month_start) AS first_month, MAX(month_start) AS last_month
  FROM auto_staging.stg_monthly_sales
  GROUP BY canonical_series_key
) sales_grid
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'zero_sales_semantics_complete', 'de4-v1', 'critical',
       SUM(monthly_sales = 0 AND zero_sales_type NOT IN (
         'all_zero_series', 'pre_launch_zero', 'post_last_positive_zero', 'within_lifecycle_zero'
       )) = 0
       AND SUM(monthly_sales > 0 AND zero_sales_type <> 'positive') = 0,
       CAST(SUM(zero_sales_type IS NULL) AS CHAR), '0',
       SUM(zero_sales_type IS NULL), NULL
FROM auto_staging.stg_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'sales_duplicate_business_keys', 'de4-v1', 'critical',
       COUNT(*) = COUNT(DISTINCT canonical_series_key, month_start),
       CAST(COUNT(*) - COUNT(DISTINCT canonical_series_key, month_start) AS CHAR), '0',
       COUNT(*) - COUNT(DISTINCT canonical_series_key, month_start), NULL
FROM auto_staging.stg_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'sales_repair_count', 'de4-v1', 'critical',
       SUM(repair_applied) = __EXPECTED_REPAIR_ROWS__, CAST(SUM(repair_applied) AS CHAR),
       CAST(__EXPECTED_REPAIR_ROWS__ AS CHAR), ABS(SUM(repair_applied) - __EXPECTED_REPAIR_ROWS__), NULL
FROM auto_staging.stg_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_monthly_sales', 'sales_repair_net_delta', 'de4-v1', 'critical',
       SUM(CAST(monthly_sales AS SIGNED) - CAST(monthly_sales_raw AS SIGNED)) = __EXPECTED_REPAIR_DELTA__,
       CAST(SUM(CAST(monthly_sales AS SIGNED) - CAST(monthly_sales_raw AS SIGNED)) AS CHAR),
       CAST(__EXPECTED_REPAIR_DELTA__ AS CHAR),
       ABS(SUM(CAST(monthly_sales AS SIGNED) - CAST(monthly_sales_raw AS SIGNED)) - __EXPECTED_REPAIR_DELTA__), NULL
FROM auto_staging.stg_monthly_sales
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_vehicle_config', 'config_row_count', 'de4-v1', 'critical',
       COUNT(*) = __EXPECTED_CONFIG_ROWS__, CAST(COUNT(*) AS CHAR),
       CAST(__EXPECTED_CONFIG_ROWS__ AS CHAR), ABS(COUNT(*) - __EXPECTED_CONFIG_ROWS__), NULL
FROM auto_staging.stg_vehicle_config
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_vehicle_config', 'annual_sales_repair_count', 'de4-v1', 'critical',
       SUM(annual_sales_repair_applied) = __EXPECTED_ANNUAL_REPAIR_ROWS__,
       CAST(SUM(annual_sales_repair_applied) AS CHAR),
       CAST(__EXPECTED_ANNUAL_REPAIR_ROWS__ AS CHAR),
       ABS(SUM(annual_sales_repair_applied) - __EXPECTED_ANNUAL_REPAIR_ROWS__), NULL
FROM auto_staging.stg_vehicle_config
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_vehicle_config', 'annual_sales_repair_net_delta', 'de4-v1', 'critical',
       SUM(CAST(annual_sales AS SIGNED) - CAST(annual_sales_raw AS SIGNED))
         = __EXPECTED_ANNUAL_REPAIR_DELTA__,
       CAST(SUM(CAST(annual_sales AS SIGNED) - CAST(annual_sales_raw AS SIGNED)) AS CHAR),
       CAST(__EXPECTED_ANNUAL_REPAIR_DELTA__ AS CHAR),
       ABS(SUM(CAST(annual_sales AS SIGNED) - CAST(annual_sales_raw AS SIGNED))
         - __EXPECTED_ANNUAL_REPAIR_DELTA__), NULL
FROM auto_staging.stg_vehicle_config
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_reviews', 'review_row_count', 'de4-v1', 'critical',
       COUNT(*) = __EXPECTED_REVIEW_ROWS__, CAST(COUNT(*) AS CHAR),
       CAST(__EXPECTED_REVIEW_ROWS__ AS CHAR), ABS(COUNT(*) - __EXPECTED_REVIEW_ROWS__), NULL
FROM auto_staging.stg_reviews
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_reviews', 'eligible_review_count', 'de4-v1', 'critical',
       SUM(eligible_for_temporal_model) = __EXPECTED_ELIGIBLE_REVIEWS__,
       CAST(SUM(eligible_for_temporal_model) AS CHAR),
       CAST(__EXPECTED_ELIGIBLE_REVIEWS__ AS CHAR),
       ABS(SUM(eligible_for_temporal_model) - __EXPECTED_ELIGIBLE_REVIEWS__), NULL
FROM auto_staging.stg_reviews
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_reviews', 'locked_test_end_availability', 'de4-v1', 'critical',
       SUM(after_locked_test_end) = __EXPECTED_AFTER_LOCKED_TEST_REVIEWS__
       AND SUM(after_locked_test_end <> (publish_time >= '2026-07-01 00:00:00')) = 0,
       CAST(SUM(after_locked_test_end) AS CHAR),
       CAST(__EXPECTED_AFTER_LOCKED_TEST_REVIEWS__ AS CHAR),
       SUM(after_locked_test_end <> (publish_time >= '2026-07-01 00:00:00')), NULL
FROM auto_staging.stg_reviews
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_review_features', 'review_feature_row_count', 'de4-v1', 'critical',
       COUNT(*) = __EXPECTED_REVIEW_FEATURE_ROWS__, CAST(COUNT(*) AS CHAR),
       CAST(__EXPECTED_REVIEW_FEATURE_ROWS__ AS CHAR),
       ABS(COUNT(*) - __EXPECTED_REVIEW_FEATURE_ROWS__), NULL
FROM auto_staging.stg_review_features
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_review_features', 'review_feature_identity_count', 'de4-v1', 'critical',
       COUNT(DISTINCT source_identity) = __EXPECTED_REVIEW_FEATURE_IDENTITIES__,
       CAST(COUNT(DISTINCT source_identity) AS CHAR),
       CAST(__EXPECTED_REVIEW_FEATURE_IDENTITIES__ AS CHAR),
       ABS(COUNT(DISTINCT source_identity) - __EXPECTED_REVIEW_FEATURE_IDENTITIES__), NULL
FROM auto_staging.stg_review_features
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_review_features', 'review_feature_aspect_count', 'de4-v1', 'critical',
       COUNT(DISTINCT aspect_code) = 10, CAST(COUNT(DISTINCT aspect_code) AS CHAR), '10',
       ABS(COUNT(DISTINCT aspect_code) - 10), NULL
FROM auto_staging.stg_review_features
UNION ALL
SELECT __PIPELINE_RUN_ID__, 'stg_reviews', 'missing_review_content', 'de4-v1', 'warning',
       SUM(auto_staging.stg_reviews.content IS NULL) = 0,
       CAST(SUM(auto_staging.stg_reviews.content IS NULL) AS CHAR), '0',
       SUM(auto_staging.stg_reviews.content IS NULL),
       JSON_OBJECT('policy', 'missing content remains missing and is ineligible')
FROM auto_staging.stg_reviews;

CREATE TEMPORARY TABLE _de4_critical_gate (
  rule_name VARCHAR(128) NOT NULL,
  passed BOOLEAN NOT NULL,
  CONSTRAINT ck_de4_all_critical_pass CHECK (passed = TRUE)
);

INSERT INTO _de4_critical_gate (rule_name, passed)
SELECT rule_name, passed
FROM auto_ops.data_quality_results
WHERE pipeline_run_id = __PIPELINE_RUN_ID__
  AND severity = 'critical'
  AND passed = FALSE;

DROP TEMPORARY TABLE _de4_critical_gate;

UPDATE auto_ops.pipeline_runs
SET status = 'succeeded', completed_at = NOW(6), error_message = NULL
WHERE pipeline_run_id = __PIPELINE_RUN_ID__;

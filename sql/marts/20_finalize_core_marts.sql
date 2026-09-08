USE auto_mart;

INSERT INTO auto_mart.mart_forecast_features (
  vehicle_series_sk, target_month_date_key, split_name,
  lag_1, lag_2, lag_3, lag_12, roll_mean_3, roll_mean_6, roll_mean_12,
  month_sin, month_cos, model_year, configuration_payload,
  review_feature_payload, review_information_cutoff_exclusive, dataset_version_id
)
SELECT
  dv.vehicle_series_sk,
  f.target_month_date_key,
  f.split_name,
  f.lag_1, f.lag_2, f.lag_3, f.lag_12,
  f.roll_mean_3, f.roll_mean_6, f.roll_mean_12,
  f.month_sin, f.month_cos, f.model_year,
  f.configuration_payload, f.review_feature_payload,
  f.review_information_cutoff_exclusive,
  (SELECT dataset_version_id FROM auto_ops.dataset_versions
   WHERE dataset_name = 'mart_forecast_features'
     AND version_key = 'forecast-b2-c__CONFIG_BATCH_ID__-r5-l8-r1-v2')
FROM auto_mart._de5_forecast_input f
JOIN auto_staging.series_name_mappings sm
  ON sm.source_system = 'pcauto_sales'
 AND sm.canonical_series_name = f.canonical_series_name
 AND sm.is_current = TRUE
JOIN auto_mart.dim_vehicle_series dv
  ON dv.canonical_series_key = sm.canonical_series_key;

DROP TEMPORARY TABLE auto_mart._de5_forecast_input;

INSERT INTO auto_mart.mart_user_needs (
  vehicle_series_sk, monitoring_month_date_key, aspect_code,
  review_count_180d, mention_count_180d, positive_rate_180d,
  negative_rate_180d, risk_level, information_cutoff_exclusive,
  dataset_version_id
)
SELECT
  dv.vehicle_series_sk,
  u.monitoring_month_date_key,
  u.aspect_code,
  u.review_count_180d,
  u.mention_count_180d,
  u.positive_rate_180d,
  u.negative_rate_180d,
  u.risk_level,
  u.information_cutoff_exclusive,
  (SELECT dataset_version_id FROM auto_ops.dataset_versions
   WHERE dataset_name = 'mart_user_needs'
     AND version_key = 'user-needs-r5-l8-202607-de5-v1')
FROM auto_mart._de5_user_needs_input u
JOIN auto_staging.series_name_mappings sm
  ON sm.source_system = 'pcauto_sales'
 AND sm.canonical_series_name = u.canonical_series_name
 AND sm.is_current = TRUE
JOIN auto_mart.dim_vehicle_series dv
  ON dv.canonical_series_key = sm.canonical_series_key;

DROP TEMPORARY TABLE auto_mart._de5_user_needs_input;

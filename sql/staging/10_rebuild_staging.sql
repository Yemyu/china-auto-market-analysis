-- Rendered by china_auto_market.quality.staging with verified full-batch ids.
USE auto_staging;

DELETE FROM auto_staging.stg_review_features;
DELETE FROM auto_staging.stg_reviews;
DELETE FROM auto_staging.stg_vehicle_config;
DELETE FROM auto_staging.stg_monthly_sales;
DELETE FROM auto_staging.series_name_mappings;

-- Sales is the canonical axis for monthly forecasting and review monitoring.
INSERT INTO auto_staging.series_name_mappings (
  source_system, source_series_id, source_series_name,
  canonical_series_key, canonical_series_name, match_method,
  confidence, valid_from, valid_to, is_current, batch_id
)
SELECT
  'pcauto_sales', s.source_series_id, MAX(s.series_name),
  CONCAT('sales:', CAST(MAX(s.series_id) AS CHAR)), MAX(s.series_name),
  'source_id_and_exact_name', 1.0,
  MIN(STR_TO_DATE(CONCAT(s.sales_year, '-', LPAD(s.sales_month, 2, '0'), '-01'), '%Y-%m-%d')),
  NULL, TRUE, __SALES_BATCH_ID__
FROM auto_raw.raw_sales s
WHERE s.batch_id = __SALES_BATCH_ID__
GROUP BY s.source_series_id;

-- Configuration rows reuse the sales canonical key only on an exact unique name.
-- Config-only products remain visible under their own source-stable key.
INSERT INTO auto_staging.series_name_mappings (
  source_system, source_series_id, source_series_name,
  canonical_series_key, canonical_series_name, match_method,
  confidence, valid_from, valid_to, is_current, batch_id
)
SELECT
  'pcauto_config', c.source_series_id, MAX(c.series_name),
  COALESCE(MAX(sm.canonical_series_key), CONCAT('config:', c.source_series_id)),
  MAX(c.series_name),
  CASE WHEN MAX(sm.canonical_series_key) IS NULL
       THEN 'config_only_source_id' ELSE 'exact_name_to_sales' END,
  1.0, STR_TO_DATE(CONCAT(MIN(c.model_year), '-01-01'), '%Y-%m-%d'),
  NULL, TRUE, __CONFIG_BATCH_ID__
FROM auto_raw.raw_vehicle_config c
LEFT JOIN auto_staging.series_name_mappings sm
  ON sm.source_system = 'pcauto_sales'
 AND sm.source_series_name = c.series_name
 AND sm.is_current = TRUE
WHERE c.batch_id = __CONFIG_BATCH_ID__
GROUP BY c.source_series_id;

-- Every review canonical name is an exact member of the sales axis.
INSERT INTO auto_staging.series_name_mappings (
  source_system, source_series_id, source_series_name,
  canonical_series_key, canonical_series_name, match_method,
  confidence, valid_from, valid_to, is_current, batch_id
)
SELECT
  r.platform, r.source_series_id, MAX(r.source_series_name),
  MAX(sm.canonical_series_key), MAX(r.source_series_name),
  'exact_name_to_sales', 1.0, DATE(MIN(r.publish_time)),
  NULL, TRUE, __REVIEWS_BATCH_ID__
FROM auto_raw.raw_reviews r
JOIN auto_staging.series_name_mappings sm
  ON sm.source_system = 'pcauto_sales'
 AND sm.source_series_name = r.source_series_name
 AND sm.is_current = TRUE
WHERE r.batch_id = __REVIEWS_BATCH_ID__
GROUP BY r.platform, r.source_series_id;

CREATE TEMPORARY TABLE _de4_sales_prepared AS
SELECT
  mapped.*,
  MIN(CASE WHEN mapped.monthly_sales > 0 THEN mapped.month_start END)
    OVER (PARTITION BY mapped.canonical_series_key) AS first_positive_month,
  MAX(CASE WHEN mapped.monthly_sales > 0 THEN mapped.month_start END)
    OVER (PARTITION BY mapped.canonical_series_key) AS last_positive_month
FROM (
  SELECT
    sm.canonical_series_key,
    sm.canonical_series_name,
    STR_TO_DATE(CONCAT(s.sales_year, '-', LPAD(s.sales_month, 2, '0'), '-01'), '%Y-%m-%d') AS month_start,
    s.source_series_id,
    s.brand_name,
    s.category_name,
    s.monthly_sales AS monthly_sales_raw,
    COALESCE(c.corrected_sales, s.monthly_sales) AS monthly_sales,
    c.raw_correction_id IS NOT NULL AS repair_applied,
    c.evidence_status AS repair_evidence_status,
    c.evidence_source_url AS repair_source_url,
    s.raw_sales_id,
    s.batch_id
  FROM auto_raw.raw_sales s
  JOIN auto_staging.series_name_mappings sm
    ON sm.source_system = 'pcauto_sales'
   AND sm.source_series_id = s.source_series_id
   AND sm.is_current = TRUE
  LEFT JOIN auto_raw.raw_sales_corrections c
    ON c.batch_id = __CORRECTIONS_BATCH_ID__
   AND c.series_name = s.series_name
   AND c.month_start = STR_TO_DATE(
       CONCAT(s.sales_year, '-', LPAD(s.sales_month, 2, '0'), '-01'), '%Y-%m-%d'
   )
  WHERE s.batch_id = __SALES_BATCH_ID__
) mapped;

INSERT INTO auto_staging.stg_monthly_sales (
  canonical_series_key, canonical_series_name, month_start, source_series_id,
  brand_name, category_name, monthly_sales_raw, monthly_sales, zero_sales_type,
  repair_applied, repair_evidence_status, repair_source_url,
  raw_sales_id, batch_id
)
SELECT
  canonical_series_key, canonical_series_name, month_start, source_series_id,
  brand_name, category_name, monthly_sales_raw, monthly_sales,
  CASE
    WHEN monthly_sales > 0 THEN 'positive'
    WHEN first_positive_month IS NULL THEN 'all_zero_series'
    WHEN month_start < first_positive_month THEN 'pre_launch_zero'
    WHEN month_start > last_positive_month THEN 'post_last_positive_zero'
    ELSE 'within_lifecycle_zero'
  END,
  repair_applied, repair_evidence_status, repair_source_url,
  raw_sales_id, batch_id
FROM _de4_sales_prepared;

DROP TEMPORARY TABLE _de4_sales_prepared;

INSERT INTO auto_staging.stg_vehicle_config (
  canonical_series_key, canonical_series_name, model_year, brand_name,
  energy_type, vehicle_class, body_structure, gearbox_type, seat_material,
  official_price_wan, engine_max_power_kw, engine_max_torque_nm,
  battery_capacity_kwh, battery_range_km, length_mm, width_mm, height_mm,
  wheelbase_mm, curb_weight_kg, seat_count, door_count, trunk_volume_l,
  acceleration_0_100_s, fuel_consumption_l_100km, annual_sales_raw,
  annual_sales, annual_sales_repair_applied,
  annual_repair_evidence_status, annual_repair_source_url,
  raw_config_id, batch_id
)
SELECT
  sm.canonical_series_key, sm.canonical_series_name, c.model_year, c.source_brand_name,
  NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.energy_type')), 'null'),
  NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.vehicle_class')), 'null'),
  NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.body_structure')), 'null'),
  NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.gearbox_type')), 'null'),
  NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.seat_material')), 'null'),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.official_price_wan')), 'null') AS DECIMAL(12,4)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.engine_max_power_kw')), 'null') AS DECIMAL(12,4)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.engine_max_torque_nm')), 'null') AS DECIMAL(12,4)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.battery_capacity_kwh')), 'null') AS DECIMAL(12,4)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.battery_range_km')), 'null') AS DECIMAL(12,4)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.length_mm')), 'null') AS UNSIGNED),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.width_mm')), 'null') AS UNSIGNED),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.height_mm')), 'null') AS UNSIGNED),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.wheelbase_mm')), 'null') AS UNSIGNED),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.curb_weight_kg')), 'null') AS DECIMAL(12,3)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.seat_count')), 'null') AS DECIMAL(6,2)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.door_count')), 'null') AS DECIMAL(6,2)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.trunk_volume_l')), 'null') AS DECIMAL(12,3)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.acceleration_0_100_s')), 'null') AS DECIMAL(10,4)),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(c.source_payload, '$.fuel_consumption_l_100km')), 'null') AS DECIMAL(10,4)),
  c.annual_sales_raw,
  COALESCE(ac.corrected_annual_sales, c.annual_sales_raw),
  ac.raw_annual_correction_id IS NOT NULL,
  ac.evidence_status,
  ac.evidence_source_url,
  c.raw_config_id, c.batch_id
FROM auto_raw.raw_vehicle_config c
JOIN auto_staging.series_name_mappings sm
  ON sm.source_system = 'pcauto_config'
 AND sm.source_series_id = c.source_series_id
 AND sm.is_current = TRUE
LEFT JOIN auto_raw.raw_annual_sales_corrections ac
  ON ac.batch_id = __ANNUAL_CORRECTIONS_BATCH_ID__
 AND ac.series_name = c.series_name
 AND ac.model_year = c.model_year
WHERE c.batch_id = __CONFIG_BATCH_ID__;

INSERT INTO auto_staging.stg_reviews (
  source_identity, canonical_series_key, canonical_series_name,
  platform, review_id, publish_time, content, content_sha256,
  rating_overall, eligible_for_temporal_model, after_locked_test_end,
  raw_review_id, batch_id
)
SELECT
  r.source_identity, sm.canonical_series_key, sm.canonical_series_name,
  r.platform, r.review_id, r.publish_time, r.content, r.content_sha256,
  r.rating_overall, r.eligible_for_temporal_model,
  r.publish_time >= '2026-07-01 00:00:00',
  r.raw_review_id, r.batch_id
FROM auto_raw.raw_reviews r
JOIN auto_staging.series_name_mappings sm
  ON sm.source_system = r.platform
 AND sm.source_series_id = r.source_series_id
 AND sm.is_current = TRUE
WHERE r.batch_id = __REVIEWS_BATCH_ID__;

INSERT INTO auto_staging.stg_review_features (
  source_identity, aspect_code, aspect_score, mentioned, raw_polarity,
  label_source, manual_qa_status, feature_version, batch_id
)
SELECT
  l.source_identity,
  a.aspect_code,
  CAST(JSON_UNQUOTE(JSON_EXTRACT(
    l.source_payload, CONCAT('$.review_', a.aspect_code, '_score')
  )) AS SIGNED),
  CAST(JSON_UNQUOTE(JSON_EXTRACT(
    l.source_payload, CONCAT('$.uniform_local_', a.aspect_code, '_mentioned')
  )) AS UNSIGNED),
  CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(
    l.source_payload, CONCAT('$.review_', a.aspect_code, '_raw_polarity')
  )), 'null') AS DECIMAL(2, 1)),
  l.label_source,
  l.manual_qa_status,
  'review-aspects-v1',
  l.batch_id
FROM auto_raw.raw_review_labels l
CROSS JOIN (
  SELECT 'appearance' AS aspect_code
  UNION ALL SELECT 'interior'
  UNION ALL SELECT 'space'
  UNION ALL SELECT 'power'
  UNION ALL SELECT 'control'
  UNION ALL SELECT 'comfort'
  UNION ALL SELECT 'fuel_consumption'
  UNION ALL SELECT 'configuration'
  UNION ALL SELECT 'intelligence'
  UNION ALL SELECT 'value'
) a
JOIN auto_staging.stg_reviews r
  ON r.source_identity = l.source_identity
WHERE l.batch_id = __LABELS_BATCH_ID__;

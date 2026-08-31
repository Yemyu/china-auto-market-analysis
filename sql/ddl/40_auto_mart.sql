USE auto_mart;

CREATE TABLE IF NOT EXISTS dim_vehicle_series (
  vehicle_series_sk BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  canonical_series_key VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  brand_name VARCHAR(128) NULL,
  category_name VARCHAR(64) NULL,
  source_identifiers_json JSON NULL,
  first_sales_month DATE NULL,
  last_sales_month DATE NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  valid_from DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  valid_to DATETIME(6) NULL,
  PRIMARY KEY (vehicle_series_sk),
  UNIQUE KEY uq_dim_series_key (canonical_series_key),
  KEY ix_dim_series_name (series_name),
  CONSTRAINT ck_dim_series_dates CHECK (valid_to IS NULL OR valid_to >= valid_from)
) ENGINE=InnoDB COMMENT='Grain: one current canonical vehicle series';

CREATE TABLE IF NOT EXISTS dim_date (
  date_key INT UNSIGNED NOT NULL,
  full_date DATE NOT NULL,
  calendar_year SMALLINT UNSIGNED NOT NULL,
  calendar_quarter TINYINT UNSIGNED NOT NULL,
  calendar_month TINYINT UNSIGNED NOT NULL,
  month_start DATE NOT NULL,
  month_end DATE NOT NULL,
  year_month_label CHAR(7) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  PRIMARY KEY (date_key),
  UNIQUE KEY uq_dim_date_full_date (full_date),
  KEY ix_dim_date_month (calendar_year, calendar_month),
  CONSTRAINT ck_dim_date_month CHECK (calendar_month BETWEEN 1 AND 12),
  CONSTRAINT ck_dim_date_quarter CHECK (calendar_quarter BETWEEN 1 AND 4)
) ENGINE=InnoDB COMMENT='Grain: one calendar date';

CREATE TABLE IF NOT EXISTS fact_monthly_sales (
  vehicle_series_sk BIGINT UNSIGNED NOT NULL,
  month_date_key INT UNSIGNED NOT NULL,
  monthly_sales BIGINT UNSIGNED NOT NULL,
  monthly_sales_raw BIGINT UNSIGNED NOT NULL,
  repair_applied BOOLEAN NOT NULL DEFAULT FALSE,
  source_batch_id BIGINT UNSIGNED NOT NULL,
  dataset_version_id BIGINT UNSIGNED NOT NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (vehicle_series_sk, month_date_key),
  KEY ix_fact_sales_month (month_date_key),
  CONSTRAINT fk_fact_sales_series
    FOREIGN KEY (vehicle_series_sk) REFERENCES dim_vehicle_series (vehicle_series_sk),
  CONSTRAINT fk_fact_sales_date
    FOREIGN KEY (month_date_key) REFERENCES dim_date (date_key),
  CONSTRAINT fk_fact_sales_batch
    FOREIGN KEY (source_batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT fk_fact_sales_version
    FOREIGN KEY (dataset_version_id) REFERENCES auto_ops.dataset_versions (dataset_version_id)
) ENGINE=InnoDB COMMENT='Grain: one canonical vehicle series and calendar month';

CREATE TABLE IF NOT EXISTS mart_forecast_features (
  vehicle_series_sk BIGINT UNSIGNED NOT NULL,
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
  dataset_version_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (vehicle_series_sk, target_month_date_key),
  KEY ix_forecast_split_month (split_name, target_month_date_key),
  CONSTRAINT fk_forecast_feature_series
    FOREIGN KEY (vehicle_series_sk) REFERENCES dim_vehicle_series (vehicle_series_sk),
  CONSTRAINT fk_forecast_feature_date
    FOREIGN KEY (target_month_date_key) REFERENCES dim_date (date_key),
  CONSTRAINT fk_forecast_feature_version
    FOREIGN KEY (dataset_version_id) REFERENCES auto_ops.dataset_versions (dataset_version_id),
  CONSTRAINT ck_forecast_split CHECK (split_name IN ('train', 'val', 'test'))
) ENGINE=InnoDB COMMENT='Grain: one canonical series and forecast target month';

CREATE TABLE IF NOT EXISTS mart_forecast_predictions (
  vehicle_series_sk BIGINT UNSIGNED NOT NULL,
  target_month_date_key INT UNSIGNED NOT NULL,
  protocol VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  model_version VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  actual_sales BIGINT UNSIGNED NULL,
  predicted_sales DOUBLE NOT NULL,
  information_cutoff_exclusive DATETIME(6) NOT NULL,
  dataset_version_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (vehicle_series_sk, target_month_date_key, protocol, model_version),
  KEY ix_prediction_protocol_month (protocol, target_month_date_key),
  CONSTRAINT fk_prediction_series
    FOREIGN KEY (vehicle_series_sk) REFERENCES dim_vehicle_series (vehicle_series_sk),
  CONSTRAINT fk_prediction_date
    FOREIGN KEY (target_month_date_key) REFERENCES dim_date (date_key),
  CONSTRAINT fk_prediction_version
    FOREIGN KEY (dataset_version_id) REFERENCES auto_ops.dataset_versions (dataset_version_id)
) ENGINE=InnoDB COMMENT='Grain: one series, target month, protocol, and model version';

CREATE TABLE IF NOT EXISTS mart_product_analysis (
  vehicle_series_sk BIGINT UNSIGNED NOT NULL,
  model_year SMALLINT UNSIGNED NOT NULL,
  annual_sales DECIMAL(18, 3) NOT NULL,
  configuration_payload JSON NOT NULL,
  dataset_version_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (vehicle_series_sk, model_year),
  KEY ix_product_year (model_year),
  CONSTRAINT fk_product_series
    FOREIGN KEY (vehicle_series_sk) REFERENCES dim_vehicle_series (vehicle_series_sk),
  CONSTRAINT fk_product_version
    FOREIGN KEY (dataset_version_id) REFERENCES auto_ops.dataset_versions (dataset_version_id)
) ENGINE=InnoDB COMMENT='Grain: one canonical vehicle series and complete model year';

CREATE TABLE IF NOT EXISTS mart_user_needs (
  vehicle_series_sk BIGINT UNSIGNED NOT NULL,
  monitoring_month_date_key INT UNSIGNED NOT NULL,
  aspect_code VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  review_count_180d INT UNSIGNED NOT NULL,
  mention_count_180d INT UNSIGNED NOT NULL,
  positive_rate_180d DOUBLE NULL,
  negative_rate_180d DOUBLE NULL,
  risk_level VARCHAR(16) COLLATE utf8mb4_0900_as_cs NOT NULL,
  information_cutoff_exclusive DATETIME(6) NOT NULL,
  dataset_version_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (vehicle_series_sk, monitoring_month_date_key, aspect_code),
  KEY ix_user_needs_month_risk (monitoring_month_date_key, risk_level),
  CONSTRAINT fk_user_needs_series
    FOREIGN KEY (vehicle_series_sk) REFERENCES dim_vehicle_series (vehicle_series_sk),
  CONSTRAINT fk_user_needs_date
    FOREIGN KEY (monitoring_month_date_key) REFERENCES dim_date (date_key),
  CONSTRAINT fk_user_needs_version
    FOREIGN KEY (dataset_version_id) REFERENCES auto_ops.dataset_versions (dataset_version_id),
  CONSTRAINT ck_user_needs_positive CHECK (positive_rate_180d IS NULL OR positive_rate_180d BETWEEN 0 AND 1),
  CONSTRAINT ck_user_needs_negative CHECK (negative_rate_180d IS NULL OR negative_rate_180d BETWEEN 0 AND 1),
  CONSTRAINT ck_user_needs_risk CHECK (risk_level IN ('normal', 'watch', 'alert'))
) ENGINE=InnoDB COMMENT='Grain: one series, monitoring month, and product aspect';

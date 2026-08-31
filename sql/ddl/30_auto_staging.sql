USE auto_staging;

CREATE TABLE IF NOT EXISTS series_name_mappings (
  mapping_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  source_system VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_series_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  canonical_series_key VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  canonical_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  match_method VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  confidence DECIMAL(5, 4) NULL,
  valid_from DATE NOT NULL,
  valid_to DATE NULL,
  is_current BOOLEAN NOT NULL DEFAULT TRUE,
  batch_id BIGINT UNSIGNED NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (mapping_id),
  UNIQUE KEY uq_mapping_source_valid_from (source_system, source_series_id, valid_from),
  KEY ix_mapping_canonical (canonical_series_key, is_current),
  CONSTRAINT fk_mapping_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT ck_mapping_dates CHECK (valid_to IS NULL OR valid_to >= valid_from),
  CONSTRAINT ck_mapping_confidence CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1)
) ENGINE=InnoDB COMMENT='One source-series identity mapping version';

CREATE TABLE IF NOT EXISTS stg_monthly_sales (
  canonical_series_key VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  canonical_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  month_start DATE NOT NULL,
  source_series_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  brand_name VARCHAR(128) NULL,
  category_name VARCHAR(64) NULL,
  monthly_sales_raw BIGINT UNSIGNED NOT NULL,
  monthly_sales BIGINT UNSIGNED NOT NULL,
  zero_sales_type VARCHAR(64) NULL,
  repair_applied BOOLEAN NOT NULL DEFAULT FALSE,
  repair_evidence_status VARCHAR(64) NULL,
  repair_source_url VARCHAR(1024) NULL,
  raw_sales_id BIGINT UNSIGNED NOT NULL,
  batch_id BIGINT UNSIGNED NOT NULL,
  standardized_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (canonical_series_key, month_start),
  UNIQUE KEY uq_stg_sales_raw (raw_sales_id),
  KEY ix_stg_sales_month (month_start),
  CONSTRAINT fk_stg_sales_raw
    FOREIGN KEY (raw_sales_id) REFERENCES auto_raw.raw_sales (raw_sales_id),
  CONSTRAINT fk_stg_sales_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id)
) ENGINE=InnoDB COMMENT='Grain: one canonical vehicle series in one calendar month';

CREATE TABLE IF NOT EXISTS stg_vehicle_config (
  canonical_series_key VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  canonical_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  model_year SMALLINT UNSIGNED NOT NULL,
  brand_name VARCHAR(128) NULL,
  energy_type VARCHAR(64) NULL,
  vehicle_class VARCHAR(64) NULL,
  body_structure VARCHAR(64) NULL,
  gearbox_type VARCHAR(64) NULL,
  seat_material VARCHAR(128) NULL,
  official_price_wan DECIMAL(12, 4) NULL,
  engine_max_power_kw DECIMAL(12, 4) NULL,
  engine_max_torque_nm DECIMAL(12, 4) NULL,
  battery_capacity_kwh DECIMAL(12, 4) NULL,
  battery_range_km DECIMAL(12, 4) NULL,
  length_mm INT UNSIGNED NULL,
  width_mm INT UNSIGNED NULL,
  height_mm INT UNSIGNED NULL,
  wheelbase_mm INT UNSIGNED NULL,
  curb_weight_kg DECIMAL(12, 3) NULL,
  seat_count DECIMAL(6, 2) NULL,
  door_count DECIMAL(6, 2) NULL,
  trunk_volume_l DECIMAL(12, 3) NULL,
  acceleration_0_100_s DECIMAL(10, 4) NULL,
  fuel_consumption_l_100km DECIMAL(10, 4) NULL,
  annual_sales_raw DECIMAL(18, 3) NULL,
  annual_sales DECIMAL(18, 3) NULL,
  annual_sales_repair_applied BOOLEAN NOT NULL DEFAULT FALSE,
  annual_repair_evidence_status VARCHAR(64) NULL,
  annual_repair_source_url VARCHAR(1024) NULL,
  raw_config_id BIGINT UNSIGNED NOT NULL,
  batch_id BIGINT UNSIGNED NOT NULL,
  standardized_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (canonical_series_key, model_year),
  UNIQUE KEY uq_stg_config_raw (raw_config_id),
  KEY ix_stg_config_year (model_year),
  CONSTRAINT fk_stg_config_raw
    FOREIGN KEY (raw_config_id) REFERENCES auto_raw.raw_vehicle_config (raw_config_id),
  CONSTRAINT fk_stg_config_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id)
) ENGINE=InnoDB COMMENT='Grain: one canonical vehicle series and model year';

CREATE TABLE IF NOT EXISTS stg_reviews (
  source_identity VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  canonical_series_key VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  canonical_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  platform VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  review_id VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  publish_time DATETIME(6) NOT NULL,
  content LONGTEXT NULL,
  content_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  rating_overall DECIMAL(6, 3) NULL,
  eligible_for_temporal_model BOOLEAN NOT NULL,
  after_locked_test_end BOOLEAN NOT NULL DEFAULT FALSE,
  raw_review_id BIGINT UNSIGNED NOT NULL,
  batch_id BIGINT UNSIGNED NOT NULL,
  standardized_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (source_identity),
  UNIQUE KEY uq_stg_review_raw (raw_review_id),
  KEY ix_stg_review_series_time (canonical_series_key, publish_time),
  CONSTRAINT fk_stg_review_raw
    FOREIGN KEY (raw_review_id) REFERENCES auto_raw.raw_reviews (raw_review_id),
  CONSTRAINT fk_stg_review_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id)
) ENGINE=InnoDB COMMENT='Grain: one deduplicated review identity with point-in-time fields';

CREATE TABLE IF NOT EXISTS stg_review_features (
  source_identity VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  aspect_code VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  aspect_score TINYINT NOT NULL,
  mentioned BOOLEAN NOT NULL,
  raw_polarity TINYINT NULL,
  label_source VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  manual_qa_status VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  feature_version VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  batch_id BIGINT UNSIGNED NOT NULL,
  standardized_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (source_identity, aspect_code),
  KEY ix_stg_review_feature_aspect (aspect_code, aspect_score),
  CONSTRAINT fk_stg_review_feature_review
    FOREIGN KEY (source_identity) REFERENCES stg_reviews (source_identity),
  CONSTRAINT fk_stg_review_feature_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT ck_stg_review_score CHECK (aspect_score IN (-1, 0, 1)),
  CONSTRAINT ck_stg_review_polarity CHECK (raw_polarity IS NULL OR raw_polarity IN (-1, 0, 1))
) ENGINE=InnoDB COMMENT='Grain: one review identity and one of ten product aspects';

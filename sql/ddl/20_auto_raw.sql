USE auto_raw;

CREATE TABLE IF NOT EXISTS raw_sales (
  raw_sales_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  batch_id BIGINT UNSIGNED NOT NULL,
  source_row_number INT UNSIGNED NOT NULL,
  source_record_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_series_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  series_id BIGINT NULL,
  series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  sales_year SMALLINT UNSIGNED NOT NULL,
  sales_month TINYINT UNSIGNED NOT NULL,
  source_period VARCHAR(16) NULL,
  brand_name VARCHAR(128) NULL,
  category_name VARCHAR(64) NULL,
  monthly_sales BIGINT UNSIGNED NOT NULL,
  zero_sales_type VARCHAR(64) NULL,
  record_status VARCHAR(32) NULL,
  period_status VARCHAR(64) NULL,
  website_cumulative_sales BIGINT UNSIGNED NULL,
  source_rank INT UNSIGNED NULL,
  source_last_rank INT UNSIGNED NULL,
  source_official_price VARCHAR(64) NULL,
  source_name VARCHAR(64) NULL,
  brand_monthly_sales BIGINT UNSIGNED NULL,
  brand_series_count INT UNSIGNED NULL,
  source_payload JSON NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (raw_sales_id),
  UNIQUE KEY uq_raw_sales_row (batch_id, source_row_number),
  UNIQUE KEY uq_raw_sales_record (batch_id, source_record_id),
  KEY ix_raw_sales_source_month (source_series_id, sales_year, sales_month),
  CONSTRAINT fk_raw_sales_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT ck_raw_sales_month CHECK (sales_month BETWEEN 1 AND 12)
) ENGINE=InnoDB COMMENT='Source grain: one source vehicle series in one calendar month';

CREATE TABLE IF NOT EXISTS raw_vehicle_config (
  raw_config_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  batch_id BIGINT UNSIGNED NOT NULL,
  source_row_number INT UNSIGNED NOT NULL,
  source_system VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_series_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  model_year SMALLINT UNSIGNED NOT NULL,
  source_car_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NULL,
  source_car_name VARCHAR(256) NULL,
  source_brand_name VARCHAR(128) NULL,
  annual_sales_raw DECIMAL(18, 3) NULL,
  source_payload JSON NOT NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (raw_config_id),
  UNIQUE KEY uq_raw_config_row (batch_id, source_row_number),
  UNIQUE KEY uq_raw_config_series_year (batch_id, series_name, model_year),
  KEY ix_raw_config_source_key (source_system, source_series_id, model_year),
  CONSTRAINT fk_raw_config_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT ck_raw_config_year CHECK (model_year BETWEEN 1990 AND 2100)
) ENGINE=InnoDB COMMENT='Source grain: one source vehicle series and model year; full source row remains JSON';

CREATE TABLE IF NOT EXISTS raw_reviews (
  raw_review_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  batch_id BIGINT UNSIGNED NOT NULL,
  source_row_number INT UNSIGNED NOT NULL,
  platform VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  review_id VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_identity VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_series_id VARCHAR(32) COLLATE utf8mb4_0900_as_cs NULL,
  source_series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  publish_time DATETIME(6) NOT NULL,
  content LONGTEXT NULL,
  content_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  rating_overall DECIMAL(6, 3) NULL,
  source_url VARCHAR(1024) NULL,
  corpus_source VARCHAR(64) NULL,
  content_source VARCHAR(64) NULL,
  eligible_for_temporal_model BOOLEAN NOT NULL DEFAULT FALSE,
  source_payload JSON NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (raw_review_id),
  UNIQUE KEY uq_raw_review_row (batch_id, source_row_number),
  UNIQUE KEY uq_raw_review_identity (batch_id, source_identity),
  KEY ix_raw_review_platform_id (platform, review_id),
  KEY ix_raw_review_series_time (source_series_name, publish_time),
  CONSTRAINT fk_raw_review_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id)
) ENGINE=InnoDB COMMENT='Source grain: one platform review identity';

CREATE TABLE IF NOT EXISTS raw_sales_corrections (
  raw_correction_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  batch_id BIGINT UNSIGNED NOT NULL,
  source_row_number INT UNSIGNED NOT NULL,
  series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  month_start DATE NOT NULL,
  original_sales BIGINT UNSIGNED NOT NULL,
  corrected_sales BIGINT UNSIGNED NOT NULL,
  evidence_source_name VARCHAR(128) NOT NULL,
  evidence_source_url VARCHAR(1024) NOT NULL,
  evidence_status VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  verified_at DATE NOT NULL,
  note VARCHAR(1024) NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (raw_correction_id),
  UNIQUE KEY uq_raw_correction_row (batch_id, source_row_number),
  UNIQUE KEY uq_raw_correction_key (batch_id, series_name, month_start),
  CONSTRAINT fk_raw_correction_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT ck_raw_correction_change CHECK (original_sales <> corrected_sales)
) ENGINE=InnoDB COMMENT='Source grain: one reviewed sales correction for one series-month';

CREATE TABLE IF NOT EXISTS raw_review_labels (
  raw_label_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  batch_id BIGINT UNSIGNED NOT NULL,
  source_row_number INT UNSIGNED NOT NULL,
  source_identity VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  review_id VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  publish_time DATETIME(6) NOT NULL,
  label_source VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  manual_qa_status VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  content_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  source_payload JSON NOT NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (raw_label_id),
  UNIQUE KEY uq_raw_label_row (batch_id, source_row_number),
  UNIQUE KEY uq_raw_label_identity (batch_id, source_identity),
  KEY ix_raw_label_series_time (series_name, publish_time),
  CONSTRAINT fk_raw_label_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id)
) ENGINE=InnoDB COMMENT='Source grain: one eligible review identity with ten aspect labels in JSON';

CREATE TABLE IF NOT EXISTS raw_annual_sales_corrections (
  raw_annual_correction_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  batch_id BIGINT UNSIGNED NOT NULL,
  source_row_number INT UNSIGNED NOT NULL,
  series_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  model_year SMALLINT UNSIGNED NOT NULL,
  original_annual_sales BIGINT UNSIGNED NOT NULL,
  corrected_annual_sales BIGINT UNSIGNED NOT NULL,
  evidence_source_name VARCHAR(128) NOT NULL,
  evidence_source_url VARCHAR(1024) NOT NULL,
  evidence_status VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  verified_at DATE NOT NULL,
  note VARCHAR(1024) NULL,
  loaded_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (raw_annual_correction_id),
  UNIQUE KEY uq_raw_annual_correction_row (batch_id, source_row_number),
  UNIQUE KEY uq_raw_annual_correction_key (batch_id, series_name, model_year),
  CONSTRAINT fk_raw_annual_correction_batch
    FOREIGN KEY (batch_id) REFERENCES auto_ops.ingestion_batches (batch_id),
  CONSTRAINT ck_raw_annual_correction_change
    CHECK (original_annual_sales <> corrected_annual_sales),
  CONSTRAINT ck_raw_annual_correction_year CHECK (model_year BETWEEN 1990 AND 2100)
) ENGINE=InnoDB COMMENT='Source grain: one reviewed annual sales correction for one series-year';

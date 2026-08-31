USE auto_ops;

CREATE TABLE IF NOT EXISTS ingestion_batches (
  batch_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  source_system VARCHAR(64) COLLATE utf8mb4_0900_as_cs NOT NULL,
  dataset_name VARCHAR(96) COLLATE utf8mb4_0900_as_cs NOT NULL,
  source_uri VARCHAR(1024) NOT NULL,
  source_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  source_bytes BIGINT UNSIGNED NULL,
  source_modified_at DATETIME(6) NULL,
  ingestion_mode VARCHAR(20) COLLATE utf8mb4_0900_as_cs NOT NULL DEFAULT 'full',
  schema_version VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  status VARCHAR(20) COLLATE utf8mb4_0900_as_cs NOT NULL DEFAULT 'started',
  rows_read BIGINT UNSIGNED NOT NULL DEFAULT 0,
  rows_loaded BIGINT UNSIGNED NOT NULL DEFAULT 0,
  rows_rejected BIGINT UNSIGNED NOT NULL DEFAULT 0,
  started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  completed_at DATETIME(6) NULL,
  metadata_json JSON NULL,
  error_message TEXT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (batch_id),
  UNIQUE KEY uq_ingestion_source_file (source_system, dataset_name, source_sha256),
  KEY ix_ingestion_dataset_started (dataset_name, started_at),
  CONSTRAINT ck_ingestion_mode CHECK (ingestion_mode IN ('full', 'incremental', 'backfill')),
  CONSTRAINT ck_ingestion_status CHECK (status IN ('started', 'succeeded', 'failed', 'rejected')),
  CONSTRAINT ck_ingestion_counts CHECK (rows_loaded + rows_rejected <= rows_read)
) ENGINE=InnoDB COMMENT='One row per immutable source-file ingestion attempt';

CREATE TABLE IF NOT EXISTS pipeline_runs (
  pipeline_run_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  pipeline_name VARCHAR(96) COLLATE utf8mb4_0900_as_cs NOT NULL,
  logical_period DATE NULL,
  trigger_type VARCHAR(20) COLLATE utf8mb4_0900_as_cs NOT NULL,
  status VARCHAR(20) COLLATE utf8mb4_0900_as_cs NOT NULL DEFAULT 'started',
  code_version VARCHAR(64) COLLATE utf8mb4_0900_as_cs NULL,
  parameters_json JSON NULL,
  started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  completed_at DATETIME(6) NULL,
  error_message TEXT NULL,
  PRIMARY KEY (pipeline_run_id),
  KEY ix_pipeline_name_period (pipeline_name, logical_period),
  CONSTRAINT ck_pipeline_trigger CHECK (trigger_type IN ('manual', 'scheduled', 'backfill', 'test')),
  CONSTRAINT ck_pipeline_status CHECK (status IN ('started', 'succeeded', 'failed', 'cancelled'))
) ENGINE=InnoDB COMMENT='One row per end-to-end pipeline run';

CREATE TABLE IF NOT EXISTS task_runs (
  task_run_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  pipeline_run_id BIGINT UNSIGNED NOT NULL,
  task_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  attempt_no SMALLINT UNSIGNED NOT NULL DEFAULT 1,
  status VARCHAR(20) COLLATE utf8mb4_0900_as_cs NOT NULL DEFAULT 'started',
  input_versions_json JSON NULL,
  output_versions_json JSON NULL,
  started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  completed_at DATETIME(6) NULL,
  error_message TEXT NULL,
  PRIMARY KEY (task_run_id),
  UNIQUE KEY uq_task_attempt (pipeline_run_id, task_name, attempt_no),
  CONSTRAINT fk_task_pipeline
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (pipeline_run_id),
  CONSTRAINT ck_task_status CHECK (status IN ('started', 'succeeded', 'failed', 'skipped', 'upstream_failed'))
) ENGINE=InnoDB COMMENT='One row per task attempt within a pipeline run';

CREATE TABLE IF NOT EXISTS dataset_versions (
  dataset_version_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  dataset_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  version_key VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  grain_description VARCHAR(512) NOT NULL,
  row_count BIGINT UNSIGNED NOT NULL,
  content_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  schema_version VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  pipeline_run_id BIGINT UNSIGNED NULL,
  upstream_versions_json JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (dataset_version_id),
  UNIQUE KEY uq_dataset_version (dataset_name, version_key),
  CONSTRAINT fk_dataset_pipeline
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (pipeline_run_id)
) ENGINE=InnoDB COMMENT='Immutable identity and lineage for a published dataset version';

CREATE TABLE IF NOT EXISTS data_quality_results (
  quality_result_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  pipeline_run_id BIGINT UNSIGNED NULL,
  task_run_id BIGINT UNSIGNED NULL,
  batch_id BIGINT UNSIGNED NULL,
  dataset_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  rule_name VARCHAR(128) COLLATE utf8mb4_0900_as_cs NOT NULL,
  rule_version VARCHAR(32) COLLATE utf8mb4_0900_as_cs NOT NULL,
  severity VARCHAR(16) COLLATE utf8mb4_0900_as_cs NOT NULL,
  passed BOOLEAN NOT NULL,
  observed_value VARCHAR(512) NULL,
  expected_value VARCHAR(512) NULL,
  affected_rows BIGINT UNSIGNED NOT NULL DEFAULT 0,
  details_json JSON NULL,
  checked_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (quality_result_id),
  KEY ix_quality_dataset_checked (dataset_name, checked_at),
  KEY ix_quality_failed (passed, severity, checked_at),
  CONSTRAINT fk_quality_pipeline
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs (pipeline_run_id),
  CONSTRAINT fk_quality_task
    FOREIGN KEY (task_run_id) REFERENCES task_runs (task_run_id),
  CONSTRAINT fk_quality_batch
    FOREIGN KEY (batch_id) REFERENCES ingestion_batches (batch_id),
  CONSTRAINT ck_quality_severity CHECK (severity IN ('info', 'warning', 'error', 'critical'))
) ENGINE=InnoDB COMMENT='Machine-readable results for every data quality rule';

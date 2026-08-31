USE auto_staging;

SET @de5_add_annual_columns = IF(
  (SELECT COUNT(*) FROM information_schema.columns
   WHERE table_schema = 'auto_staging'
     AND table_name = 'stg_vehicle_config'
     AND column_name = 'annual_sales') = 0,
  'ALTER TABLE auto_staging.stg_vehicle_config
     ADD COLUMN annual_sales DECIMAL(18, 3) NULL AFTER annual_sales_raw,
     ADD COLUMN annual_sales_repair_applied BOOLEAN NOT NULL DEFAULT FALSE AFTER annual_sales,
     ADD COLUMN annual_repair_evidence_status VARCHAR(64) NULL AFTER annual_sales_repair_applied,
     ADD COLUMN annual_repair_source_url VARCHAR(1024) NULL AFTER annual_repair_evidence_status',
  'SELECT 1'
);

PREPARE de5_annual_columns_stmt FROM @de5_add_annual_columns;
EXECUTE de5_annual_columns_stmt;
DEALLOCATE PREPARE de5_annual_columns_stmt;

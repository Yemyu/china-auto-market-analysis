-- Run manually as a MySQL administrator after replacing the placeholder
-- outside Git. The pipeline account does not receive DROP or user-management rights.
CREATE USER IF NOT EXISTS 'auto_pipeline'@'localhost'
  IDENTIFIED BY '<set-a-local-password-outside-git>';

GRANT SELECT, INSERT, UPDATE, DELETE ON auto_ops.* TO 'auto_pipeline'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON auto_raw.* TO 'auto_pipeline'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON auto_staging.* TO 'auto_pipeline'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON auto_mart.* TO 'auto_pipeline'@'localhost';

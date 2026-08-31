USE auto_mart;

-- Monitoring rates are model outputs. Preserve their Python floating-point
-- values instead of rounding them at the storage boundary.
ALTER TABLE mart_user_needs
  MODIFY positive_rate_180d DOUBLE NULL,
  MODIFY negative_rate_180d DOUBLE NULL;

USE auto_mart;

-- Model inputs and outputs must retain Python/CSV floating-point precision.
-- Four-decimal storage silently changes rolling means and evaluation metrics.
ALTER TABLE mart_forecast_features
  MODIFY lag_1 DOUBLE NULL,
  MODIFY lag_2 DOUBLE NULL,
  MODIFY lag_3 DOUBLE NULL,
  MODIFY lag_12 DOUBLE NULL,
  MODIFY roll_mean_3 DOUBLE NULL,
  MODIFY roll_mean_6 DOUBLE NULL,
  MODIFY roll_mean_12 DOUBLE NULL;

ALTER TABLE mart_forecast_predictions
  MODIFY predicted_sales DOUBLE NOT NULL;

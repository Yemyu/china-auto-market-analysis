-- Four logical databases keep operational metadata, source snapshots,
-- standardized records, and consumer-facing marts separate.
CREATE DATABASE IF NOT EXISTS auto_ops
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS auto_raw
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS auto_staging
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS auto_mart
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

"""Migrate only forecast configuration payloads, preserving every other mart.

The old row values and version IDs are backed up before a guarded transaction.
No tables are deleted, no raw snapshots are rewritten, and no forecasts run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from china_auto_market.ingestion.mysql_cli import insert_statements, run_transaction, sql_literal
from china_auto_market.marts.build import QUALITY_SQL, configuration_reference
from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.warehouse.schema import query_json_rows
from china_auto_market.warehouse.sources import load_raw_configuration, staging_configuration_batch

BACKUP_QUERY = """
SELECT JSON_OBJECT('vehicle_series_sk', vehicle_series_sk,
 'target_month_date_key', target_month_date_key,
 'configuration_payload', configuration_payload,
 'dataset_version_id', dataset_version_id)
FROM auto_mart.mart_forecast_features ORDER BY vehicle_series_sk, target_month_date_key;
"""
BACKUP_COLUMNS = ["vehicle_series_sk", "target_month_date_key", "configuration_payload", "dataset_version_id"]
TEMPORARY_BACKUP = """
CREATE TEMPORARY TABLE auto_mart._r1_config_backup (
 vehicle_series_sk BIGINT UNSIGNED NOT NULL,
 target_month_date_key INT UNSIGNED NOT NULL,
 configuration_payload JSON NOT NULL,
 dataset_version_id BIGINT UNSIGNED NOT NULL,
 PRIMARY KEY(vehicle_series_sk, target_month_date_key));
"""


def validate_backup(rows: list[dict], version: dict) -> int:
    if len(rows) != 17808 or len({(r['vehicle_series_sk'], r['target_month_date_key']) for r in rows}) != 17808:
        raise ValueError("Migration requires the complete, unique 17808-row forecast mart")
    if {r["dataset_version_id"] for r in rows} != {version["dataset_version_id"]}:
        raise ValueError("Forecast source versions changed or are mixed")
    upstream = version["upstream_versions_json"]
    batch = upstream.get("config_batch")
    configuration_reference(batch)
    return batch


def migrate(login_path: str, backup_dir: Path, *, apply: bool = False) -> dict:
    backup_dir = backup_dir.resolve()
    artifacts = (PROJECT_ROOT / "artifacts").resolve()
    if not backup_dir.is_relative_to(artifacts) or backup_dir == artifacts:
        raise ValueError("Migration backups must be in a dedicated artifacts/ subdirectory")
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise ValueError("Backup directory must be empty; previous recovery files are never overwritten")
    rows = query_json_rows(login_path, BACKUP_QUERY)
    versions = query_json_rows(login_path, """
      SELECT JSON_OBJECT('dataset_version_id', v.dataset_version_id, 'version_key', v.version_key,
       'schema_version', v.schema_version, 'upstream_versions_json', v.upstream_versions_json)
      FROM auto_ops.dataset_versions v JOIN
       (SELECT DISTINCT dataset_version_id FROM auto_mart.mart_forecast_features) f USING(dataset_version_id);
    """)
    if len(versions) != 1:
        raise ValueError("Expected exactly one forecast dataset version")
    version = versions[0]
    batch = validate_backup(rows, version)
    reference = configuration_reference(batch)
    load_raw_configuration(login_path, batch_id=batch)
    if version["schema_version"] == "r1-v2" and all(r["configuration_payload"] == reference for r in rows):
        return {"status": "already_migrated", "configuration_batch_id": batch, "rows": len(rows)}
    if version["schema_version"] != "de5-v1" or not version["version_key"].endswith("-de5-v1"):
        raise ValueError("Unrecognized legacy version; do not guess a migration")
    if staging_configuration_batch(login_path) != batch:
        raise ValueError("Staging and forecast source batches differ; reconcile lineage before migration")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = json.dumps({"version": version, "rows": rows}, ensure_ascii=False, indent=2)
    (backup_dir / "before.json.gz").write_bytes(gzip.compress(backup.encode("utf-8"), mtime=0))
    inserts = list(insert_statements("auto_mart._r1_config_backup", BACKUP_COLUMNS,
                                   ([row[c] for c in BACKUP_COLUMNS] for row in rows), chunk_size=25))
    restore = """
UPDATE auto_mart.mart_forecast_features f JOIN auto_mart._r1_config_backup b
 USING(vehicle_series_sk, target_month_date_key)
SET f.configuration_payload=b.configuration_payload, f.dataset_version_id=b.dataset_version_id;
DROP TEMPORARY TABLE auto_mart._r1_config_backup;
"""
    recovery = ("-- Manual recovery only; confirm no later mart rebuild before use.\nSET NAMES utf8mb4;\nSTART TRANSACTION;\n"
                + TEMPORARY_BACKUP + "".join(inserts) + restore + "COMMIT;\n")
    (backup_dir / "restore_payloads.sql.gz").write_bytes(gzip.compress(recovery.encode("utf-8"), mtime=0))
    result = {"status": "prepared", "rows": len(rows), "configuration_batch_id": batch,
              "old_dataset_version_id": version["dataset_version_id"],
              "backup_sha256": hashlib.sha256(backup.encode()).hexdigest(), "applied": False}
    if apply:
        new_key = version["version_key"].removesuffix("-de5-v1") + "-r1-v2"
        statements = [TEMPORARY_BACKUP, *inserts, """
CREATE TEMPORARY TABLE auto_mart._r1_gate (passed BOOLEAN NOT NULL CHECK(passed=TRUE));
-- Lock the target rows and compare against the exact backup before updating.
UPDATE auto_mart.mart_forecast_features SET dataset_version_id=dataset_version_id;
INSERT INTO auto_mart._r1_gate SELECT COUNT(*)=17808 FROM auto_mart.mart_forecast_features;
INSERT INTO auto_mart._r1_gate SELECT COUNT(*)=17808
FROM auto_mart.mart_forecast_features f JOIN auto_mart._r1_config_backup b
 USING(vehicle_series_sk, target_month_date_key)
WHERE f.dataset_version_id=b.dataset_version_id AND f.configuration_payload=b.configuration_payload;
INSERT INTO auto_ops.pipeline_runs (pipeline_name,trigger_type,status,code_version,parameters_json)
VALUES ('migrate_forecast_configuration','manual','started','r1-v2',JSON_OBJECT('scope','forecast_payload_only'));
SET @r1_run_id=LAST_INSERT_ID();
""", f"""
INSERT INTO auto_ops.dataset_versions
 (dataset_name,version_key,grain_description,row_count,schema_version,pipeline_run_id,upstream_versions_json)
SELECT dataset_name,{sql_literal(new_key)},'locked series x forecast target month; raw configuration batch reference',
 row_count,'r1-v2',@r1_run_id,upstream_versions_json
FROM auto_ops.dataset_versions WHERE dataset_version_id={int(version['dataset_version_id'])};
SET @r1_version_id=LAST_INSERT_ID();
UPDATE auto_mart.mart_forecast_features
SET configuration_payload={sql_literal(reference)},dataset_version_id=@r1_version_id;
""", QUALITY_SQL.read_text(encoding="utf-8").replace("__PIPELINE_RUN_ID__", "@r1_run_id"),
                      "DROP TEMPORARY TABLE auto_mart._r1_config_backup; DROP TEMPORARY TABLE auto_mart._r1_gate;"]
        run_transaction(login_path, statements)
        after = query_json_rows(login_path, BACKUP_QUERY)
        if len(after) != len(rows) or any(r["configuration_payload"] != reference for r in after):
            raise RuntimeError("Post-migration verification failed; use saved recovery evidence")
        result.update(status="migrated", applied=True, new_version_key=new_key)
    (backup_dir / "migration_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login-path", default="local-auto")
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args.login_path, args.backup_dir, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()

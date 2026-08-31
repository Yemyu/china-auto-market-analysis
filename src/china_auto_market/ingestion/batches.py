"""Immutable source batch registration and status transitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from china_auto_market.ingestion.mysql_cli import relative_source_uri, sql_literal
from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.warehouse.schema import query


@dataclass(frozen=True)
class Batch:
    batch_id: int
    status: str
    rows_read: int
    rows_loaded: int
    existed: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_batch(
    login_path: str,
    source_system: str,
    dataset_name: str,
    source_sha256: str,
) -> Batch | None:
    source_system_hex = source_system.encode("utf-8").hex()
    dataset_name_hex = dataset_name.encode("utf-8").hex()
    source_sha256_hex = source_sha256.encode("ascii").hex()
    sql = (
        "SELECT batch_id,status,rows_read,rows_loaded FROM auto_ops.ingestion_batches WHERE "
        f"BINARY source_system=0x{source_system_hex} AND BINARY dataset_name=0x{dataset_name_hex} "
        f"AND BINARY source_sha256=0x{source_sha256_hex};"
    )
    output = query(login_path, sql).strip()
    if not output:
        return None
    batch_id, status, rows_read, rows_loaded = output.split("\t")
    return Batch(int(batch_id), status, int(rows_read), int(rows_loaded), True)


def register_or_resume_batch(
    login_path: str,
    *,
    source_system: str,
    dataset_name: str,
    source_path: Path,
    rows_read: int,
    ingestion_mode: str,
    schema_version: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Batch, str]:
    """Create a batch or safely resume the same failed immutable source file."""
    source_path = source_path.resolve()
    source_sha256 = sha256_file(source_path)
    existing = _find_batch(login_path, source_system, dataset_name, source_sha256)
    if existing:
        if existing.status == "succeeded":
            if existing.rows_read != rows_read or existing.rows_loaded != rows_read:
                raise RuntimeError("Succeeded batch row counts no longer match the source partition")
            return existing, source_sha256
        sql = (
            "UPDATE auto_ops.ingestion_batches SET status='started',rows_read="
            f"{rows_read},rows_loaded=0,rows_rejected=0,completed_at=NULL,error_message=NULL "
            f"WHERE batch_id={existing.batch_id};"
        )
        query(login_path, sql)
        return Batch(existing.batch_id, "started", rows_read, 0, True), source_sha256

    source_uri = relative_source_uri(source_path, PROJECT_ROOT)
    modified = datetime.fromtimestamp(source_path.stat().st_mtime).isoformat(sep=" ")
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    sql = (
        "INSERT INTO auto_ops.ingestion_batches "
        "(source_system,dataset_name,source_uri,source_sha256,source_bytes,source_modified_at,"
        "ingestion_mode,schema_version,status,rows_read,metadata_json) VALUES ("
        f"{sql_literal(source_system)},{sql_literal(dataset_name)},{sql_literal(source_uri)},"
        f"{sql_literal(source_sha256)},{source_path.stat().st_size},{sql_literal(modified)},"
        f"{sql_literal(ingestion_mode)},{sql_literal(schema_version)},'started',{rows_read},"
        f"{sql_literal(metadata_json)}); SELECT LAST_INSERT_ID();"
    )
    output = query(login_path, sql).strip().splitlines()
    if not output:
        raise RuntimeError("MySQL did not return the new batch id")
    return Batch(int(output[-1]), "started", rows_read, 0, False), source_sha256


def mark_batch_failed(login_path: str, batch_id: int, error: Exception) -> None:
    """Persist a bounded failure message after the raw transaction rolls back."""
    message = str(error)[:4000]
    sql = (
        "UPDATE auto_ops.ingestion_batches SET status='failed',completed_at=NOW(6),"
        f"error_message={sql_literal(message)} WHERE batch_id={batch_id};"
    )
    query(login_path, sql)

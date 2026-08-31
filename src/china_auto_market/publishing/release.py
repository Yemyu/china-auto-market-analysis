"""Validated, atomic release of pre-baked dashboard payloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.publishing.contracts import (
    DASHBOARD_REQUIRED_KEYS,
    assert_dashboard_payloads,
)


EXPECTED_DATASETS = {
    "fact_monthly_sales",
    "mart_forecast_features",
    "mart_product_analysis",
    "mart_user_needs",
}


@dataclass(frozen=True)
class PublicationResult:
    manifest: str
    files: int
    applied: bool
    output_sha256: dict[str, str]


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Publication manifest must be a JSON object")
    if payload.get("schema_version") != "de6-publication-readiness-v1":
        raise ValueError("Unsupported publication manifest schema")
    if payload.get("critical_failures") != 0:
        raise ValueError("Publication manifest contains critical quality failures")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Publication manifest datasets must be a list")
    names = {
        item.get("dataset_name")
        for item in datasets
        if isinstance(item, dict)
    }
    if names != EXPECTED_DATASETS:
        raise ValueError(
            f"Publication datasets changed: expected {sorted(EXPECTED_DATASETS)}, "
            f"received {sorted(str(name) for name in names)}"
        )
    source_runs = {
        item.get("source_pipeline_run_id")
        for item in datasets
        if isinstance(item, dict)
    }
    if len(source_runs) != 1 or None in source_runs:
        raise ValueError("Publication datasets must come from one mart run")
    if any(int(item.get("row_count", 0)) <= 0 for item in datasets if isinstance(item, dict)):
        raise ValueError("Publication datasets must all contain rows")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def publish_dashboard(
    manifest_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    apply: bool = False,
) -> PublicationResult:
    """Build dashboard JSON in quarantine, validate it, then swap atomically."""
    root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    if root not in manifest_path.parents:
        raise ValueError("Publication manifest must be inside the project")
    manifest = _load_manifest(manifest_path)
    target = root / "app" / "static" / "data"
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(tempfile.mkdtemp(prefix=".dashboard-candidate-", dir=target.parent))
    backup = target.parent / ".dashboard-previous"
    if backup.exists():
        raise RuntimeError(f"Stale publication backup requires inspection: {backup}")
    try:
        command = [
            sys.executable,
            str(root / "app" / "build_dashboard_data.py"),
            "--output-dir",
            str(candidate),
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = completed.stderr[-4000:] or completed.stdout[-4000:]
            raise RuntimeError(f"Dashboard build failed: {detail}")
        assert_dashboard_payloads(candidate)
        output_hashes = {
            filename: _sha256(candidate / filename)
            for filename in sorted(DASHBOARD_REQUIRED_KEYS)
        }
        if not apply:
            return PublicationResult(
                manifest=str(manifest_path.relative_to(root)),
                files=len(output_hashes),
                applied=False,
                output_sha256=output_hashes,
            )

        if target.exists():
            target.rename(backup)
        try:
            candidate.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        manifest["public_files_written"] = True
        manifest["published_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = output_hashes
        _write_manifest(manifest_path, manifest)
        return PublicationResult(
            manifest=str(manifest_path.relative_to(root)),
            files=len(output_hashes),
            applied=True,
            output_sha256=output_hashes,
        )
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


def result_dict(result: PublicationResult) -> dict[str, Any]:
    return asdict(result)

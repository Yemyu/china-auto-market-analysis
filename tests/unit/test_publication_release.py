from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from china_auto_market.publishing import release
from china_auto_market.publishing.contracts import DASHBOARD_REQUIRED_KEYS


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "de6-publication-readiness-v1",
        "run_month": "2026-08",
        "orchestration_pipeline_run_id": 23,
        "mart_pipeline_run_id": 25,
        "critical_failures": 0,
        "datasets": [
            {
                "dataset_name": name,
                "version_key": f"{name}-fixture",
                "row_count": 1,
                "source_pipeline_run_id": 25,
            }
            for name in sorted(release.EXPECTED_DATASETS)
        ],
        "public_files_written": False,
    }


def install_fake_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        for filename, required in DASHBOARD_REQUIRED_KEYS.items():
            payload = {key: [] for key in required}
            (output / filename).write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(release.subprocess, "run", fake_run)


def test_publication_builds_in_quarantine_before_atomic_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    target = root / "app" / "static" / "data"
    target.mkdir(parents=True)
    (target / "old.json").write_text("{}", encoding="utf-8")
    manifest = root / "artifacts" / "orchestration" / "2026-08" / "publish_ready.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(manifest_payload()), encoding="utf-8")
    install_fake_builder(monkeypatch)

    checked = release.publish_dashboard(manifest, project_root=root, apply=False)
    assert checked.applied is False
    assert (target / "old.json").is_file()

    applied = release.publish_dashboard(manifest, project_root=root, apply=True)
    assert applied.applied is True
    assert applied.files == 8
    assert not (target / "old.json").exists()
    assert set(path.name for path in target.glob("*.json")) == set(DASHBOARD_REQUIRED_KEYS)
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["public_files_written"] is True
    assert set(updated["output_sha256"]) == set(DASHBOARD_REQUIRED_KEYS)
    assert not (target.parent / ".dashboard-previous").exists()


def test_publication_rejects_failed_quality_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    manifest = root / "artifacts" / "publish_ready.json"
    manifest.parent.mkdir(parents=True)
    payload = manifest_payload()
    payload["critical_failures"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("builder must not run"),
    )

    with pytest.raises(ValueError, match="critical"):
        release.publish_dashboard(manifest, project_root=root, apply=True)

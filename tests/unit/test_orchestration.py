from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import china_auto_market.orchestration.monthly as monthly
from china_auto_market.orchestration.monthly import (
    PipelineConfig,
    TaskSpec,
    build_publish_task_spec,
    build_task_specs,
    execute_command,
    finish_pipeline,
    month_for_interval,
    normalize_run_month,
    resolve_trigger_type,
)


def config(tmp_path: Path) -> PipelineConfig:
    project = tmp_path / "project"
    project.mkdir()
    scripts = project / "scripts"
    scripts.mkdir()
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable))
    return PipelineConfig(
        run_month="2026-07",
        login_path="test-login",
        trigger_type="test",
        project_root=project,
        python_executable=python,
        publication_root=project / "artifacts" / "orchestration",
    )


def test_run_month_is_explicit_and_calendar_valid() -> None:
    assert normalize_run_month("2026-07").isoformat() == "2026-07-01"
    for value in ("2026-7", "2026-13", "July-2026", ""):
        with pytest.raises(ValueError):
            normalize_run_month(value)


def test_airflow_interval_uses_shanghai_business_month() -> None:
    interval = datetime(2026, 7, 31, 22, tzinfo=timezone.utc)
    assert month_for_interval(interval) == "2026-08"
    with pytest.raises(ValueError, match="timezone-aware"):
        month_for_interval(datetime(2026, 8, 1))


def test_trigger_type_defaults_to_airflow_run_semantics() -> None:
    assert resolve_trigger_type("scheduled") == "scheduled"
    assert resolve_trigger_type("backfill_job") == "backfill"
    assert resolve_trigger_type("manual") == "manual"
    assert resolve_trigger_type("manual", "test") == "test"


def test_monthly_plan_has_locked_order_and_full_snapshot_semantics(tmp_path: Path) -> None:
    specs = build_task_specs(config(tmp_path))
    assert [spec.task_id for spec in specs] == [
        "ingest_raw_snapshots",
        "validate_raw",
        "rebuild_staging",
        "rebuild_core_marts",
        "validate_core_marts",
        "validate_forecast_model",
    ]
    assert specs[0].command[-6:] == (
        "--dataset", "all", "--mode", "full", "--login-path", "test-login"
    )
    assert specs[0].command[0] == str(tmp_path / "project" / ".venv" / "bin" / "python")
    assert all("password" not in " ".join(spec.command).lower() for spec in specs)


def test_publication_task_requires_the_staged_manifest_and_explicit_apply(tmp_path: Path) -> None:
    pipeline_config = config(tmp_path)
    spec = build_publish_task_spec(pipeline_config)
    assert spec.task_id == "publish_dashboard"
    assert spec.command[-3:] == (
        "--manifest",
        str(pipeline_config.publication_root / "2026-07" / "publish_ready.json"),
        "--apply",
    )


def test_command_failure_raises_before_a_following_step_can_run(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist.json"
    failure = TaskSpec(
        "quality_gate",
        (sys.executable, "-c", "raise SystemExit(7)"),
        10,
    )
    with pytest.raises(RuntimeError, match="quality_gate failed with code 7"):
        execute_command(failure, cwd=tmp_path)
    assert not marker.exists()


def test_successful_command_returns_bounded_machine_result(tmp_path: Path) -> None:
    payload = {"passed": True, "rows": 10}
    spec = TaskSpec(
        "validate",
        (sys.executable, "-c", f"import json; print(json.dumps({payload!r}))"),
        10,
    )
    result = execute_command(spec, cwd=tmp_path)
    assert result.returncode == 0
    assert json.loads(result.stdout_tail) == payload


def test_command_receives_project_source_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    spec = TaskSpec(
        "inspect_environment",
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHONPATH'].split(os.pathsep)[0])",
        ),
        10,
    )
    result = execute_command(spec, cwd=project)
    assert result.stdout_tail.strip() == str(project / "src")


def test_finish_pipeline_rejects_manifest_from_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_config = config(tmp_path)
    manifest = pipeline_config.publication_root / "2026-07" / "publish_ready.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"orchestration_pipeline_run_id": 99, "public_files_written": True}),
        encoding="utf-8",
    )
    statements: list[str] = []
    monkeypatch.setattr(
        monthly,
        "query",
        lambda _login_path, statement: statements.append(statement) or "",
    )
    with pytest.raises(RuntimeError, match="different run"):
        finish_pipeline(pipeline_config, 42)
    assert any("status='failed'" in statement for statement in statements)


def test_finish_pipeline_verifies_database_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline_config = config(tmp_path)
    manifest = pipeline_config.publication_root / "2026-07" / "publish_ready.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"orchestration_pipeline_run_id": 42, "public_files_written": True}),
        encoding="utf-8",
    )
    responses = iter(("", "succeeded\n"))
    monkeypatch.setattr(monthly, "query", lambda _login_path, _statement: next(responses))
    finish_pipeline(pipeline_config, 42)

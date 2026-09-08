"""Airflow-independent monthly orchestration and operational run logging."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.warehouse.schema import query


PIPELINE_NAME = "china_auto_market_monthly"
TRIGGER_TYPES = {"manual", "scheduled", "backfill", "test"}
ORCHESTRATION_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _sql_literal(value: object) -> str:
    """Encode the small set of orchestration metadata values as MySQL literals.

    Keep this module importable in the lightweight Airflow environment. The
    ingestion encoder intentionally supports pandas and NumPy scalars, while
    orchestration metadata is limited to JSON strings, dates, and integers.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, date):
        value = value.isoformat()
    encoded = str(value).encode("utf-8").hex()
    return f"CONVERT(0x{encoded} USING utf8mb4)"


@dataclass(frozen=True)
class PipelineConfig:
    run_month: str
    login_path: str
    trigger_type: str = "manual"
    project_root: Path = PROJECT_ROOT
    python_executable: Path = PROJECT_ROOT / ".venv" / "bin" / "python"
    publication_root: Path = PROJECT_ROOT / "artifacts" / "orchestration"
    airflow_run_id: str | None = None

    def validated(self) -> PipelineConfig:
        normalize_run_month(self.run_month)
        if self.trigger_type not in TRIGGER_TYPES:
            raise ValueError(f"Unsupported trigger type: {self.trigger_type}")
        if not self.login_path or any(character.isspace() for character in self.login_path):
            raise ValueError("MySQL login path must be a non-empty name without whitespace")
        root = self.project_root.resolve()
        # Do not resolve the interpreter symlink: a virtual environment's
        # ``bin/python`` commonly points at the base interpreter, and replacing
        # it with that target silently drops the virtual environment packages.
        python = self.python_executable.expanduser().absolute()
        publication = self.publication_root.resolve()
        if not python.is_file():
            raise FileNotFoundError(f"Pipeline Python does not exist: {python}")
        if publication == root or root not in publication.parents:
            raise ValueError("Publication staging must be a dedicated directory inside the project")
        return self


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    command: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    returncode: int
    stdout_tail: str
    stderr_tail: str


def normalize_run_month(value: str) -> date:
    """Return the first day for an explicit YYYY-MM orchestration period."""
    if len(value) != 7 or value[4] != "-":
        raise ValueError("run_month must use YYYY-MM")
    try:
        year, month = (int(piece) for piece in value.split("-"))
        return date(year, month, 1)
    except (TypeError, ValueError) as error:
        raise ValueError("run_month must be a valid YYYY-MM calendar month") from error


def month_for_interval(value: datetime) -> str:
    """Map an aware Airflow interval to the business month in Shanghai time."""
    if value.tzinfo is None:
        raise ValueError("Airflow data interval must be timezone-aware")
    return value.astimezone(ORCHESTRATION_TIMEZONE).strftime("%Y-%m")


def resolve_trigger_type(airflow_run_type: str, requested: str = "auto") -> str:
    """Classify the run from Airflow unless an explicit test override is supplied."""
    requested = requested.strip().lower()
    if requested != "auto":
        if requested not in TRIGGER_TYPES:
            raise ValueError(f"Unsupported trigger type: {requested}")
        return requested
    airflow_run_type = airflow_run_type.strip().lower()
    if "backfill" in airflow_run_type:
        return "backfill"
    if "scheduled" in airflow_run_type:
        return "scheduled"
    return "manual"


def build_task_specs(config: PipelineConfig) -> tuple[TaskSpec, ...]:
    """Return the locked linear task contract without executing it."""
    config = config.validated()
    python = str(config.python_executable.expanduser().absolute())
    root = config.project_root.resolve()
    login = config.login_path

    def script(name: str, *arguments: str) -> tuple[str, ...]:
        return (python, str(root / "scripts" / name), *arguments)

    return (
        TaskSpec(
            "ingest_raw_snapshots",
            script("ingest_raw.py", "--dataset", "all", "--mode", "full", "--login-path", login),
            900,
        ),
        TaskSpec("validate_raw", script("validate_raw.py", "--login-path", login), 600),
        TaskSpec(
            "rebuild_staging", script("rebuild_staging.py", "--login-path", login), 900
        ),
        TaskSpec(
            "rebuild_core_marts",
            script("rebuild_core_marts.py", "--login-path", login),
            1200,
        ),
        TaskSpec(
            "validate_core_marts",
            script("validate_core_marts.py", "--login-path", login),
            900,
        ),
        TaskSpec(
            "validate_forecast_model",
            script("validate_forecast_consumer_parity.py", "--login-path", login,
                   "--fixed-reference-dir", str(root / "data/processed/forecast")),
            1800,
        ),
    )


def build_publish_task_spec(config: PipelineConfig) -> TaskSpec:
    """Return the post-manifest static publication task."""
    config = config.validated()
    python = str(config.python_executable.expanduser().absolute())
    root = config.project_root.resolve()
    manifest = config.publication_root / config.run_month / "publish_ready.json"
    return TaskSpec(
        "publish_dashboard",
        (
            python,
            str(root / "scripts" / "publish_dashboard.py"),
            "--manifest",
            str(manifest),
            "--apply",
        ),
        900,
    )


def execute_command(spec: TaskSpec, *, cwd: Path = PROJECT_ROOT) -> TaskResult:
    """Execute one bounded command; raise before any downstream task can run."""
    environment = os.environ.copy()
    source_root = str(cwd.resolve() / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_root, existing_pythonpath))
        if existing_pythonpath
        else source_root
    )
    completed = subprocess.run(
        list(spec.command),
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=spec.timeout_seconds,
    )
    result = TaskResult(
        task_id=spec.task_id,
        returncode=completed.returncode,
        stdout_tail=completed.stdout[-4000:],
        stderr_tail=completed.stderr[-4000:],
    )
    if completed.returncode:
        detail = result.stderr_tail or result.stdout_tail or "no command output"
        raise RuntimeError(f"Task {spec.task_id} failed with code {completed.returncode}: {detail}")
    return result


def begin_pipeline(config: PipelineConfig) -> int:
    """Register one outer orchestration run and return its database identity."""
    config = config.validated()
    parameters = {
        "run_month": config.run_month,
        "airflow_run_id": config.airflow_run_id,
        "source_snapshot_semantics": "cumulative_full_snapshot_at_execution_time",
        "publication_mode": "validated_static_json",
    }
    output = query(
        config.login_path,
        "INSERT INTO auto_ops.pipeline_runs "
        "(pipeline_name,logical_period,trigger_type,status,code_version,parameters_json) VALUES ("
        f"{_sql_literal(PIPELINE_NAME)},{_sql_literal(normalize_run_month(config.run_month))},"
        f"{_sql_literal(config.trigger_type)},'started','de8-v1',"
        f"{_sql_literal(json.dumps(parameters, ensure_ascii=False))}); "
        "SELECT LAST_INSERT_ID();",
    ).strip().splitlines()
    return int(output[-1])


def _start_task(config: PipelineConfig, pipeline_run_id: int, task_name: str, attempt: int) -> int:
    output = query(
        config.login_path,
        "INSERT INTO auto_ops.task_runs "
        "(pipeline_run_id,task_name,attempt_no,status,input_versions_json) VALUES ("
        f"{pipeline_run_id},{_sql_literal(task_name)},{attempt},'started',"
        f"JSON_OBJECT('run_month',{_sql_literal(config.run_month)})); "
        "SELECT LAST_INSERT_ID();",
    ).strip().splitlines()
    return int(output[-1])


def _finish_task(
    config: PipelineConfig,
    task_run_id: int,
    *,
    status: str,
    output: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    query(
        config.login_path,
        "UPDATE auto_ops.task_runs SET "
        f"status={_sql_literal(status)},completed_at=NOW(6),"
        f"output_versions_json={_sql_literal(json.dumps(output, ensure_ascii=False)) if output else 'NULL'},"
        f"error_message={_sql_literal(error[:4000]) if error else 'NULL'} "
        f"WHERE task_run_id={task_run_id};",
    )


def fail_pipeline(config: PipelineConfig, pipeline_run_id: int, error: BaseException) -> None:
    """Mark the outer run failed without touching the previous published snapshot."""
    query(
        config.login_path,
        "UPDATE auto_ops.pipeline_runs SET status='failed',completed_at=NOW(6),"
        f"error_message={_sql_literal(str(error)[:4000])} WHERE pipeline_run_id={pipeline_run_id};",
    )


def run_recorded_task(
    config: PipelineConfig,
    pipeline_run_id: int,
    spec: TaskSpec,
    *,
    attempt: int = 1,
    terminal_failure: bool = True,
) -> TaskResult:
    """Run one task with auto_ops attempt logging and failure propagation."""
    task_run_id = _start_task(config, pipeline_run_id, spec.task_id, attempt)
    try:
        result = execute_command(spec, cwd=config.project_root)
    except Exception as error:
        _finish_task(config, task_run_id, status="failed", error=str(error))
        if terminal_failure:
            fail_pipeline(config, pipeline_run_id, error)
        raise
    _finish_task(
        config,
        task_run_id,
        status="succeeded",
        output={"returncode": result.returncode, "stdout_tail": result.stdout_tail},
    )
    return result


def stage_publication_manifest(
    config: PipelineConfig,
    pipeline_run_id: int,
    *,
    attempt: int = 1,
) -> Path:
    """Write a non-public readiness manifest only after every upstream gate passes."""
    config = config.validated()
    task_run_id = _start_task(config, pipeline_run_id, "stage_publication_manifest", attempt)
    try:
        rows = query(
            config.login_path,
            "SELECT dataset_name,version_key,row_count,pipeline_run_id FROM ("
            "SELECT dataset_name,version_key,row_count,pipeline_run_id,"
            "ROW_NUMBER() OVER (PARTITION BY dataset_name "
            "ORDER BY created_at DESC,dataset_version_id DESC) AS version_rank "
            "FROM auto_ops.dataset_versions WHERE dataset_name IN ("
            "'fact_monthly_sales','mart_forecast_features','mart_product_analysis',"
            "'mart_user_needs')) AS ranked WHERE version_rank=1 ORDER BY dataset_name;",
        ).strip().splitlines()
        datasets = []
        for row in rows:
            name, version, count, source_run = row.split("\t")
            datasets.append(
                {
                    "dataset_name": name,
                    "version_key": version,
                    "row_count": int(count),
                    "source_pipeline_run_id": int(source_run),
                }
            )
        if len(datasets) != 4:
            raise RuntimeError(f"Expected four publishable dataset versions, found {len(datasets)}")
        source_runs = {item["source_pipeline_run_id"] for item in datasets}
        if len(source_runs) != 1:
            raise RuntimeError(f"Dataset versions do not share one mart run: {source_runs}")
        mart_run_id = next(iter(source_runs))
        critical_failures = int(
            query(
                config.login_path,
                "SELECT COUNT(*) FROM auto_ops.data_quality_results "
                f"WHERE pipeline_run_id={mart_run_id} AND severity='critical' AND NOT passed;",
            ).strip()
        )
        if critical_failures:
            raise RuntimeError(f"Mart run {mart_run_id} has {critical_failures} critical failures")
        manifest = {
            "schema_version": "de6-publication-readiness-v1",
            "run_month": config.run_month,
            "orchestration_pipeline_run_id": pipeline_run_id,
            "mart_pipeline_run_id": mart_run_id,
            "critical_failures": 0,
            "datasets": datasets,
            "public_files_written": False,
            "source_snapshot_semantics": "cumulative_full_snapshot_at_execution_time",
        }
        target_dir = config.publication_root / config.run_month
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "publish_ready.json"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target_dir, delete=False
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, target)
    except Exception as error:
        _finish_task(config, task_run_id, status="failed", error=str(error))
        fail_pipeline(config, pipeline_run_id, error)
        raise
    _finish_task(
        config,
        task_run_id,
        status="succeeded",
        output={"manifest": str(target.relative_to(config.project_root)), "public_files_written": False},
    )
    return target


def finish_pipeline(config: PipelineConfig, pipeline_run_id: int) -> None:
    """Mark the run successful only after validated public JSON is written."""
    manifest = config.publication_root / config.run_month / "publish_ready.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as cause:
        error = RuntimeError("Publication readiness manifest is missing or invalid")
        fail_pipeline(config, pipeline_run_id, error)
        raise error from cause
    if payload.get("orchestration_pipeline_run_id") != pipeline_run_id:
        error = RuntimeError("Publication readiness manifest belongs to a different run")
        fail_pipeline(config, pipeline_run_id, error)
        raise error
    if payload.get("public_files_written") is not True:
        error = RuntimeError("Validated public dashboard files were not written")
        fail_pipeline(config, pipeline_run_id, error)
        raise error
    query(
        config.login_path,
        "UPDATE auto_ops.pipeline_runs SET status='succeeded',completed_at=NOW(6),error_message=NULL "
        f"WHERE pipeline_run_id={pipeline_run_id} AND status='started';",
    )
    status = query(
        config.login_path,
        "SELECT status FROM auto_ops.pipeline_runs "
        f"WHERE pipeline_run_id={pipeline_run_id};",
    ).strip()
    if status != "succeeded":
        raise RuntimeError(
            f"Pipeline run {pipeline_run_id} did not transition to succeeded: {status or 'missing'}"
        )


def run_local_pipeline(config: PipelineConfig, specs: Iterable[TaskSpec] | None = None) -> int:
    """Execute the same workflow outside Airflow for local tests and recovery."""
    config = config.validated()
    pipeline_run_id = begin_pipeline(config)
    for spec in specs if specs is not None else build_task_specs(config):
        run_recorded_task(config, pipeline_run_id, spec)
    stage_publication_manifest(config, pipeline_run_id)
    run_recorded_task(config, pipeline_run_id, build_publish_task_spec(config))
    finish_pipeline(config, pipeline_run_id)
    return pipeline_run_id


def result_dict(result: TaskResult) -> dict[str, Any]:
    return asdict(result)

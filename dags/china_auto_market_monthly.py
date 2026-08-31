"""Monthly cumulative-snapshot refresh for the China auto market platform."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

from china_auto_market.orchestration.monthly import (
    PipelineConfig,
    begin_pipeline,
    build_publish_task_spec,
    build_task_specs,
    finish_pipeline,
    month_for_interval,
    resolve_trigger_type,
    run_recorded_task,
    stage_publication_manifest,
)


PROJECT_ROOT = Path(
    os.environ.get("CHINA_AUTO_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()


def _config_from_context() -> PipelineConfig:
    context = get_current_context()
    params = context["params"]
    explicit_month = str(params.get("run_month") or "").strip()
    interval_start = context.get("data_interval_start")
    if explicit_month:
        run_month = explicit_month
    elif interval_start is not None:
        run_month = month_for_interval(interval_start)
    else:
        run_month = month_for_interval(context["logical_date"])
    trigger_type = resolve_trigger_type(
        str(context["dag_run"].run_type),
        str(params.get("trigger_type", "auto")),
    )
    return PipelineConfig(
        run_month=run_month,
        login_path=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"),
        trigger_type=trigger_type,
        project_root=PROJECT_ROOT,
        python_executable=Path(
            os.environ.get("AUTO_PIPELINE_PYTHON", PROJECT_ROOT / ".venv" / "bin" / "python")
        ),
        publication_root=PROJECT_ROOT / "artifacts" / "orchestration",
        airflow_run_id=context["dag_run"].run_id,
    ).validated()


@dag(
    dag_id="china_auto_market_monthly",
    description="Ingest, validate, standardize, build marts, verify models, and stage publication",
    schedule="0 6 1 * *",
    start_date=pendulum.datetime(2026, 7, 1, tz="Asia/Shanghai"),
    catchup=False,
    max_active_runs=1,
    fail_fast=True,
    dagrun_timeout=timedelta(hours=2),
    tags=["china-auto", "data-engineering", "monthly"],
    params={
        "run_month": Param("", type="string", description="Optional YYYY-MM override"),
        "trigger_type": Param(
            "auto",
            enum=["auto", "manual", "scheduled", "backfill", "test"],
            description="Operational run classification stored in auto_ops",
        ),
    },
)
def china_auto_market_monthly():
    @task(task_id="begin_pipeline", retries=0, execution_timeout=timedelta(minutes=2))
    def begin() -> int:
        return begin_pipeline(_config_from_context())

    def recorded_task(task_id: str, timeout_seconds: int, *, publication: bool = False):
        @task(
            task_id=task_id,
            retries=2,
            retry_delay=timedelta(minutes=5),
            execution_timeout=timedelta(seconds=timeout_seconds + 60),
        )
        def execute(pipeline_run_id: int) -> dict[str, object]:
            context = get_current_context()
            config = _config_from_context()
            spec = (
                build_publish_task_spec(config)
                if publication
                else next(item for item in build_task_specs(config) if item.task_id == task_id)
            )
            result = run_recorded_task(
                config,
                pipeline_run_id,
                spec,
                attempt=int(context["ti"].try_number),
                terminal_failure=(
                    int(context["ti"].try_number) > int(context["task"].retries)
                ),
            )
            return {"task_id": result.task_id, "returncode": result.returncode}

        return execute

    specs = build_task_specs(
        PipelineConfig(
            run_month="2026-07",
            login_path="local-auto",
            project_root=PROJECT_ROOT,
            python_executable=PROJECT_ROOT / ".venv" / "bin" / "python",
            publication_root=PROJECT_ROOT / "artifacts" / "orchestration",
        )
    )
    pipeline_id = begin()
    previous = None
    for spec in specs:
        current = recorded_task(spec.task_id, spec.timeout_seconds)(pipeline_id)
        if previous is not None:
            previous >> current
        previous = current

    @task(task_id="stage_publication_manifest", retries=0, execution_timeout=timedelta(minutes=3))
    def stage_manifest(pipeline_run_id: int) -> str:
        context = get_current_context()
        path = stage_publication_manifest(
            _config_from_context(),
            pipeline_run_id,
            attempt=int(context["ti"].try_number),
        )
        return str(path)

    @task(task_id="finish_pipeline", retries=0, execution_timeout=timedelta(minutes=2))
    def finish(pipeline_run_id: int) -> None:
        finish_pipeline(_config_from_context(), pipeline_run_id)

    staged = stage_manifest(pipeline_id)
    published = recorded_task(
        "publish_dashboard", 900, publication=True
    )(pipeline_id)
    completed = finish(pipeline_id)
    if previous is not None:
        previous >> staged
    staged >> published >> completed


china_auto_market_monthly()

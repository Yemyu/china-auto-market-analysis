"""Batch-aware loaders for the current raw source contracts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd

from china_auto_market.ingestion.batches import mark_batch_failed, register_or_resume_batch
from china_auto_market.ingestion.mysql_cli import insert_statements, json_payload, run_transaction
from china_auto_market.paths import PROJECT_ROOT
from china_auto_market.warehouse.schema import query


DEFAULT_SOURCES = {
    "sales": PROJECT_ROOT / "data" / "raw" / "monthly_sales.csv",
    "config": PROJECT_ROOT / "data" / "raw" / "feature.csv",
    "reviews": PROJECT_ROOT / "data" / "reviews" / "processed" / "target_371_review_corpus.csv",
    "review_labels": (
        PROJECT_ROOT / "data" / "reviews" / "processed" / "review_aspect_labels.csv"
    ),
    "sales_corrections": (
        PROJECT_ROOT / "data" / "processed" / "data_quality" / "sales_correction_register.csv"
    ),
    "annual_sales_corrections": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "data_quality"
        / "annual_sales_correction_register.csv"
    ),
}


@dataclass(frozen=True)
class IngestionResult:
    dataset: str
    batch_id: int
    source_sha256: str
    rows_read: int
    rows_loaded: int
    status: str
    idempotent_skip: bool


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Source is missing required columns: {missing}")


def _with_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["_source_row_number"] = range(2, len(output) + 2)
    return output


def _filter_month(frame: pd.DataFrame, dataset: str, month: str | None) -> pd.DataFrame:
    if month is None:
        return frame
    period = pd.Period(month, freq="M")
    if dataset == "sales":
        mask = frame["year"].eq(period.year) & frame["month"].eq(period.month)
    elif dataset == "config":
        mask = frame["year"].eq(period.year)
    elif dataset == "reviews":
        timestamps = pd.to_datetime(frame["publish_time"], errors="raise")
        mask = timestamps.dt.to_period("M").eq(period)
    elif dataset == "review_labels":
        timestamps = pd.to_datetime(frame["publish_time"], errors="raise")
        mask = timestamps.dt.to_period("M").eq(period)
    elif dataset == "sales_corrections":
        timestamps = pd.to_datetime(frame["date"], errors="raise")
        mask = timestamps.dt.to_period("M").eq(period)
    elif dataset == "annual_sales_corrections":
        mask = frame["year"].eq(period.year)
    else:
        raise ValueError(dataset)
    return frame.loc[mask].copy()


def _dataset_name(dataset: str, month: str | None) -> str:
    if month is None:
        return dataset
    period = pd.Period(month, freq="M")
    if dataset in {"config", "annual_sales_corrections"}:
        return f"{dataset}:{period.year}"
    return f"{dataset}:{period}"


def _raw_count(login_path: str, table: str, batch_id: int) -> int:
    output = query(login_path, f"SELECT COUNT(*) FROM {table} WHERE batch_id={batch_id};").strip()
    return int(output)


def _load(
    *,
    login_path: str,
    dataset: str,
    table: str,
    source_path: Path,
    frame: pd.DataFrame,
    columns: Sequence[str],
    row_builder: Callable[[pd.Series, int], Sequence[Any]],
    chunk_size: int,
    mode: str,
    month: str | None,
    schema_version: str,
    source_total_rows: int,
) -> IngestionResult:
    dataset_name = _dataset_name(dataset, month)
    batch, source_sha256 = register_or_resume_batch(
        login_path,
        source_system="local_project_snapshot",
        dataset_name=dataset_name,
        source_path=source_path,
        rows_read=len(frame),
        ingestion_mode=mode,
        schema_version=schema_version,
        metadata={"partition_month": month, "source_total_rows": source_total_rows},
    )
    if batch.status == "succeeded":
        actual = _raw_count(login_path, table, batch.batch_id)
        if actual != batch.rows_loaded:
            raise RuntimeError(
                f"Batch {batch.batch_id} says {batch.rows_loaded} rows loaded but {table} has {actual}"
            )
        return IngestionResult(dataset, batch.batch_id, source_sha256, len(frame), actual, "succeeded", True)

    def rows() -> Iterator[Sequence[Any]]:
        for _, record in frame.iterrows():
            yield row_builder(record, batch.batch_id)

    def statements() -> Iterator[str]:
        yield f"DELETE FROM {table} WHERE batch_id={batch.batch_id};\n"
        yield from insert_statements(table, columns, rows(), chunk_size=chunk_size)
        yield (
            "UPDATE auto_ops.ingestion_batches SET status='succeeded',rows_loaded="
            f"{len(frame)},rows_rejected=0,completed_at=NOW(6),error_message=NULL "
            f"WHERE batch_id={batch.batch_id};\n"
        )
    try:
        run_transaction(login_path, statements())
    except Exception as error:
        mark_batch_failed(login_path, batch.batch_id, error)
        raise
    actual = _raw_count(login_path, table, batch.batch_id)
    if actual != len(frame):
        raise RuntimeError(f"Loaded {actual} rows into {table}; expected {len(frame)}")
    return IngestionResult(dataset, batch.batch_id, source_sha256, len(frame), actual, "succeeded", False)


def load_sales(
    login_path: str,
    source_path: Path = DEFAULT_SOURCES["sales"],
    *,
    mode: str = "full",
    month: str | None = None,
) -> IngestionResult:
    source_path = Path(source_path)
    frame = _with_source_rows(pd.read_csv(source_path, low_memory=False))
    source_total_rows = len(frame)
    required = [
        "record_id", "year", "month", "series_id", "source_series_id", "series_name",
        "monthly_sales", "零销量类型", "数据来源",
    ]
    _require_columns(frame, required)
    frame = _filter_month(frame, "sales", month)
    columns = [
        "batch_id", "source_row_number", "source_record_id", "source_series_id", "series_id",
        "series_name", "sales_year", "sales_month", "source_period", "brand_name",
        "category_name", "monthly_sales", "zero_sales_type", "record_status", "period_status",
        "website_cumulative_sales", "source_rank", "source_last_rank", "source_official_price",
        "source_name", "brand_monthly_sales", "brand_series_count", "source_payload",
    ]

    def build(row: pd.Series, batch_id: int) -> Sequence[Any]:
        return (
            batch_id, row["_source_row_number"], row["record_id"], row["source_series_id"],
            row["series_id"], row["series_name"], row["year"], row["month"], row.get("period"),
            row.get("brand"), row.get("category"), row["monthly_sales"], row["零销量类型"],
            row.get("record_status"), row.get("period_status"), row.get("website_cumulative_sales"),
            row.get("source_rank"), row.get("source_last_rank"), row.get("source_official_price"),
            row["数据来源"], row.get("品牌月总销量"), row.get("品牌车型数"),
            json_payload(row.to_dict(), exclude={"_source_row_number"}),
        )

    return _load(
        login_path=login_path, dataset="sales", table="auto_raw.raw_sales", source_path=source_path,
        frame=frame, columns=columns, row_builder=build, chunk_size=250, mode=mode, month=month,
        schema_version="raw-sales-v1", source_total_rows=source_total_rows,
    )


def load_config(
    login_path: str,
    source_path: Path = DEFAULT_SOURCES["config"],
    *,
    mode: str = "full",
    month: str | None = None,
) -> IngestionResult:
    source_path = Path(source_path)
    frame = _with_source_rows(pd.read_csv(source_path, low_memory=False))
    source_total_rows = len(frame)
    _require_columns(frame, ["series_id", "series_name", "year", "car_id", "car_name"])
    frame = _filter_month(frame, "config", month)
    columns = [
        "batch_id", "source_row_number", "source_system", "source_series_id", "series_name",
        "model_year", "source_car_id", "source_car_name", "source_brand_name", "annual_sales_raw",
        "source_payload",
    ]

    def build(row: pd.Series, batch_id: int) -> Sequence[Any]:
        return (
            batch_id, row["_source_row_number"], "pcauto", row["series_id"], row["series_name"],
            row["year"], row.get("car_id"), row.get("car_name"), row.get("brand_name"),
            row.get("annual_sales"), json_payload(row.to_dict(), exclude={"_source_row_number"}),
        )

    return _load(
        login_path=login_path, dataset="config", table="auto_raw.raw_vehicle_config",
        source_path=source_path, frame=frame, columns=columns, row_builder=build, chunk_size=50,
        mode=mode, month=month, schema_version="raw-config-v1", source_total_rows=source_total_rows,
    )


def load_reviews(
    login_path: str,
    source_path: Path = DEFAULT_SOURCES["reviews"],
    *,
    mode: str = "full",
    month: str | None = None,
) -> IngestionResult:
    source_path = Path(source_path)
    frame = _with_source_rows(pd.read_csv(source_path, low_memory=False))
    source_total_rows = len(frame)
    _require_columns(
        frame,
        ["platform", "review_id", "identity", "series_name_canonical", "publish_time", "content"],
    )
    frame = _filter_month(frame, "reviews", month)
    columns = [
        "batch_id", "source_row_number", "platform", "review_id", "source_identity",
        "source_series_id", "source_series_name", "publish_time", "content", "content_sha256",
        "rating_overall", "source_url", "corpus_source", "content_source",
        "eligible_for_temporal_model", "source_payload",
    ]

    def build(row: pd.Series, batch_id: int) -> Sequence[Any]:
        content_missing = pd.isna(row.get("content"))
        content = "" if content_missing else str(row.get("content"))
        content_sha256 = None if content_missing else hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_series_id = row.get("platform_series_id")
        if pd.isna(source_series_id):
            source_series_id = row.get("series_id")
        source_url = row.get("source_url")
        if pd.isna(source_url):
            source_url = row.get("detail_url")
        return (
            batch_id, row["_source_row_number"], row["platform"], row["review_id"], row["identity"],
            source_series_id, row["series_name_canonical"], row["publish_time"], row.get("content"),
            content_sha256, row.get("rating_overall"), source_url, row.get("corpus_source"),
            row.get("content_source"), row.get("eligible_for_temporal_model", False),
            json_payload(row.to_dict(), exclude={"_source_row_number", "content"}),
        )

    return _load(
        login_path=login_path, dataset="reviews", table="auto_raw.raw_reviews",
        source_path=source_path, frame=frame, columns=columns, row_builder=build, chunk_size=20,
        mode=mode, month=month, schema_version="raw-reviews-v1", source_total_rows=source_total_rows,
    )


def load_sales_corrections(
    login_path: str,
    source_path: Path = DEFAULT_SOURCES["sales_corrections"],
    *,
    mode: str = "full",
    month: str | None = None,
) -> IngestionResult:
    source_path = Path(source_path)
    frame = _with_source_rows(pd.read_csv(source_path, low_memory=False))
    source_total_rows = len(frame)
    required = [
        "series_name", "date", "original_sales", "corrected_sales", "source_name", "source_url",
        "evidence_status", "verified_at", "note",
    ]
    _require_columns(frame, required)
    frame = _filter_month(frame, "sales_corrections", month)
    columns = [
        "batch_id", "source_row_number", "series_name", "month_start", "original_sales",
        "corrected_sales", "evidence_source_name", "evidence_source_url", "evidence_status",
        "verified_at", "note",
    ]

    def build(row: pd.Series, batch_id: int) -> Sequence[Any]:
        return (
            batch_id, row["_source_row_number"], row["series_name"], row["date"],
            row["original_sales"], row["corrected_sales"], row["source_name"], row["source_url"],
            row["evidence_status"], row["verified_at"], row["note"],
        )

    return _load(
        login_path=login_path, dataset="sales_corrections", table="auto_raw.raw_sales_corrections",
        source_path=source_path, frame=frame, columns=columns, row_builder=build, chunk_size=100,
        mode=mode, month=month, schema_version="raw-sales-corrections-v1",
        source_total_rows=source_total_rows,
    )


def load_review_labels(
    login_path: str,
    source_path: Path = DEFAULT_SOURCES["review_labels"],
    *,
    mode: str = "full",
    month: str | None = None,
) -> IngestionResult:
    source_path = Path(source_path)
    frame = _with_source_rows(pd.read_csv(source_path, low_memory=False))
    source_total_rows = len(frame)
    required = [
        "identity", "review_id", "series_name", "publish_time", "label_source",
        "manual_qa_status", "content_sha256",
    ]
    _require_columns(frame, required)
    frame = _filter_month(frame, "review_labels", month)
    columns = [
        "batch_id", "source_row_number", "source_identity", "review_id", "series_name",
        "publish_time", "label_source", "manual_qa_status", "content_sha256", "source_payload",
    ]

    def build(row: pd.Series, batch_id: int) -> Sequence[Any]:
        return (
            batch_id, row["_source_row_number"], row["identity"], row["review_id"],
            row["series_name"], row["publish_time"], row["label_source"],
            row["manual_qa_status"], row["content_sha256"],
            json_payload(row.to_dict(), exclude={"_source_row_number"}),
        )

    return _load(
        login_path=login_path, dataset="review_labels", table="auto_raw.raw_review_labels",
        source_path=source_path, frame=frame, columns=columns, row_builder=build, chunk_size=50,
        mode=mode, month=month, schema_version="raw-review-labels-v1",
        source_total_rows=source_total_rows,
    )


def load_annual_sales_corrections(
    login_path: str,
    source_path: Path = DEFAULT_SOURCES["annual_sales_corrections"],
    *,
    mode: str = "full",
    month: str | None = None,
) -> IngestionResult:
    source_path = Path(source_path)
    frame = _with_source_rows(pd.read_csv(source_path, low_memory=False))
    source_total_rows = len(frame)
    required = [
        "series_name", "year", "original_annual_sales", "corrected_annual_sales",
        "source_name", "source_url", "evidence_status", "verified_at", "note",
    ]
    _require_columns(frame, required)
    frame = _filter_month(frame, "annual_sales_corrections", month)
    columns = [
        "batch_id", "source_row_number", "series_name", "model_year",
        "original_annual_sales", "corrected_annual_sales", "evidence_source_name",
        "evidence_source_url", "evidence_status", "verified_at", "note",
    ]

    def build(row: pd.Series, batch_id: int) -> Sequence[Any]:
        return (
            batch_id, row["_source_row_number"], row["series_name"], row["year"],
            row["original_annual_sales"], row["corrected_annual_sales"], row["source_name"],
            row["source_url"], row["evidence_status"], row["verified_at"], row["note"],
        )

    return _load(
        login_path=login_path,
        dataset="annual_sales_corrections",
        table="auto_raw.raw_annual_sales_corrections",
        source_path=source_path,
        frame=frame,
        columns=columns,
        row_builder=build,
        chunk_size=50,
        mode=mode,
        month=month,
        schema_version="raw-annual-sales-corrections-v1",
        source_total_rows=source_total_rows,
    )


LOADERS = {
    "sales": load_sales,
    "config": load_config,
    "reviews": load_reviews,
    "review_labels": load_review_labels,
    "sales_corrections": load_sales_corrections,
    "annual_sales_corrections": load_annual_sales_corrections,
}


def result_dict(result: IngestionResult) -> dict[str, Any]:
    """Return a JSON-serializable ingestion result."""
    return asdict(result)

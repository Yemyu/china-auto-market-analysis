"""Source-to-raw parity checks for completed ingestion batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from china_auto_market.ingestion.batches import sha256_file
from china_auto_market.ingestion.raw_loaders import DEFAULT_SOURCES
from china_auto_market.warehouse.schema import query


@dataclass(frozen=True)
class Check:
    name: str
    expected: Any
    actual: Any
    passed: bool


def _scalar(login_path: str, sql: str) -> str:
    return query(login_path, sql).strip()


def _batch_id(login_path: str, dataset_name: str) -> int:
    output = _scalar(
        login_path,
        "SELECT batch_id FROM auto_ops.ingestion_batches "
        f"WHERE dataset_name='{dataset_name}' AND status='succeeded' "
        "ORDER BY batch_id DESC LIMIT 1;",
    )
    if not output:
        raise RuntimeError(f"No succeeded batch found for {dataset_name}")
    return int(output)


def _add(checks: list[Check], name: str, expected: Any, actual: Any) -> None:
    checks.append(Check(name, expected, actual, expected == actual))


def validate_full_sources(
    login_path: str,
    *,
    sources: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Compare the six full source snapshots with their raw batches.

    ``sources`` exists so CI can exercise the same production validation
    logic on synthetic fixtures without reading the local full dataset.
    """
    checks: list[Check] = []
    source_paths = {
        name: Path(path)
        for name, path in (DEFAULT_SOURCES if sources is None else sources).items()
    }

    sales_path = source_paths["sales"]
    sales = pd.read_csv(sales_path, low_memory=False)
    sales_batch = _batch_id(login_path, "sales")
    sales_db = _scalar(
        login_path,
        "SELECT COUNT(*),COUNT(DISTINCT source_record_id),SUM(monthly_sales),"
        "SUM(source_record_id IS NULL),COUNT(*)-COUNT(DISTINCT source_series_id,sales_year,sales_month) "
        f"FROM auto_raw.raw_sales WHERE batch_id={sales_batch};",
    ).split("\t")
    _add(checks, "sales.rows", len(sales), int(sales_db[0]))
    _add(checks, "sales.distinct_record_ids", int(sales["record_id"].nunique()), int(sales_db[1]))
    _add(checks, "sales.total_volume", int(sales["monthly_sales"].sum()), int(sales_db[2]))
    _add(checks, "sales.null_record_ids", 0, int(sales_db[3]))
    _add(checks, "sales.duplicate_series_months", 0, int(sales_db[4]))

    config_path = source_paths["config"]
    config = pd.read_csv(config_path, low_memory=False)
    config_batch = _batch_id(login_path, "config")
    config_db = _scalar(
        login_path,
        "SELECT COUNT(*),COUNT(DISTINCT series_name,model_year),MIN(model_year),MAX(model_year) "
        f"FROM auto_raw.raw_vehicle_config WHERE batch_id={config_batch};",
    ).split("\t")
    _add(checks, "config.rows", len(config), int(config_db[0]))
    _add(
        checks,
        "config.distinct_series_years",
        int(config.drop_duplicates(["series_name", "year"]).shape[0]),
        int(config_db[1]),
    )
    _add(checks, "config.min_year", int(config["year"].min()), int(config_db[2]))
    _add(checks, "config.max_year", int(config["year"].max()), int(config_db[3]))

    reviews_path = source_paths["reviews"]
    reviews = pd.read_csv(reviews_path, low_memory=False)
    review_batch = _batch_id(login_path, "reviews")
    reviews_db = _scalar(
        login_path,
        "SELECT COUNT(*),COUNT(DISTINCT source_identity),SUM(eligible_for_temporal_model),"
        "SUM(content IS NULL),MAX(CHAR_LENGTH(content)),MIN(publish_time),MAX(publish_time) "
        f"FROM auto_raw.raw_reviews WHERE batch_id={review_batch};",
    ).split("\t")
    timestamps = pd.to_datetime(reviews["publish_time"], errors="raise")
    _add(checks, "reviews.rows", len(reviews), int(reviews_db[0]))
    _add(checks, "reviews.distinct_identities", int(reviews["identity"].nunique()), int(reviews_db[1]))
    _add(
        checks,
        "reviews.eligible_rows",
        int(reviews["eligible_for_temporal_model"].fillna(False).astype(bool).sum()),
        int(reviews_db[2]),
    )
    _add(checks, "reviews.null_content", int(reviews["content"].isna().sum()), int(reviews_db[3]))
    _add(
        checks,
        "reviews.max_content_characters",
        int(reviews["content"].dropna().astype(str).str.len().max()),
        int(reviews_db[4]),
    )
    _add(
        checks,
        "reviews.min_publish_time",
        timestamps.min().strftime("%Y-%m-%d %H:%M:%S"),
        pd.Timestamp(reviews_db[5]).strftime("%Y-%m-%d %H:%M:%S"),
    )
    _add(
        checks,
        "reviews.max_publish_time",
        timestamps.max().strftime("%Y-%m-%d %H:%M:%S"),
        pd.Timestamp(reviews_db[6]).strftime("%Y-%m-%d %H:%M:%S"),
    )

    correction_path = source_paths["sales_corrections"]
    corrections = pd.read_csv(correction_path, low_memory=False)
    correction_batch = _batch_id(login_path, "sales_corrections")
    correction_db = _scalar(
        login_path,
        "SELECT COUNT(*),COUNT(DISTINCT series_name,month_start),"
        "SUM(CAST(corrected_sales AS SIGNED)-CAST(original_sales AS SIGNED)) "
        "FROM auto_raw.raw_sales_corrections "
        f"WHERE batch_id={correction_batch};",
    ).split("\t")
    _add(checks, "corrections.rows", len(corrections), int(correction_db[0]))
    _add(
        checks,
        "corrections.distinct_series_months",
        int(corrections.drop_duplicates(["series_name", "date"]).shape[0]),
        int(correction_db[1]),
    )
    _add(
        checks,
        "corrections.net_sales_delta",
        int((corrections["corrected_sales"] - corrections["original_sales"]).sum()),
        int(correction_db[2]),
    )

    annual_path = source_paths["annual_sales_corrections"]
    annual = pd.read_csv(annual_path, low_memory=False)
    annual_batch = _batch_id(login_path, "annual_sales_corrections")
    annual_db = _scalar(
        login_path,
        "SELECT COUNT(*),COUNT(DISTINCT series_name,model_year),"
        "SUM(CAST(corrected_annual_sales AS SIGNED)-CAST(original_annual_sales AS SIGNED)) "
        "FROM auto_raw.raw_annual_sales_corrections "
        f"WHERE batch_id={annual_batch};",
    ).split("\t")
    _add(checks, "annual_corrections.rows", len(annual), int(annual_db[0]))
    _add(
        checks,
        "annual_corrections.distinct_series_years",
        int(annual.drop_duplicates(["series_name", "year"]).shape[0]),
        int(annual_db[1]),
    )
    _add(
        checks,
        "annual_corrections.net_sales_delta",
        int((annual["corrected_annual_sales"] - annual["original_annual_sales"]).sum()),
        int(annual_db[2]),
    )

    label_path = source_paths["review_labels"]
    labels = pd.read_csv(label_path, low_memory=False)
    label_batch = _batch_id(login_path, "review_labels")
    label_db = _scalar(
        login_path,
        "SELECT COUNT(*),COUNT(DISTINCT source_identity),MIN(publish_time),MAX(publish_time) "
        f"FROM auto_raw.raw_review_labels WHERE batch_id={label_batch};",
    ).split("\t")
    label_times = pd.to_datetime(labels["publish_time"], errors="raise")
    _add(checks, "review_labels.rows", len(labels), int(label_db[0]))
    _add(
        checks,
        "review_labels.distinct_identities",
        int(labels["identity"].nunique()),
        int(label_db[1]),
    )
    _add(
        checks,
        "review_labels.min_publish_time",
        label_times.min().strftime("%Y-%m-%d %H:%M:%S"),
        pd.Timestamp(label_db[2]).strftime("%Y-%m-%d %H:%M:%S"),
    )
    _add(
        checks,
        "review_labels.max_publish_time",
        label_times.max().strftime("%Y-%m-%d %H:%M:%S"),
        pd.Timestamp(label_db[3]).strftime("%Y-%m-%d %H:%M:%S"),
    )

    batch_specs = {
        "sales": (sales_batch, sales_path),
        "config": (config_batch, config_path),
        "reviews": (review_batch, reviews_path),
        "sales_corrections": (correction_batch, correction_path),
        "review_labels": (label_batch, label_path),
        "annual_sales_corrections": (annual_batch, annual_path),
    }
    for dataset, (batch_id, path) in batch_specs.items():
        stored = _scalar(
            login_path,
            f"SELECT source_sha256 FROM auto_ops.ingestion_batches WHERE batch_id={batch_id};",
        )
        _add(checks, f"{dataset}.source_sha256", sha256_file(path), stored)

    return {
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "batch_ids": {dataset: batch_id for dataset, (batch_id, _) in batch_specs.items()},
    }

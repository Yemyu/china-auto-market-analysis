#!/usr/bin/env python3
"""Measure reusable review coverage for the forecast population.

The forecast population is defined by ``processed/splits/test.csv``. Historical
records remain read-only and collection gaps are written to the audit directory.

Outputs
-------
data/processed/review_collection/review_readiness.csv
    One row per forecast series, with review/label coverage and action.
data/processed/review_collection/review_collection_gap.csv
    The series that have no reusable historical review coverage, including the
    availability of a valid Dongchedi numeric ID.
data/processed/review_collection/review_id_resolution_gap.csv
    Missing-review series whose Dongchedi ID must be resolved before any
    source-specific collection can begin.
data/processed/review_collection/review_readiness_summary.json
    Compact counts for the project report and crawl planning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SPLITS = BASE / "data" / "processed" / "splits"
HISTORICAL_RESOURCE = BASE / "data" / "resources" / "historical_reviews" / "review_absa_reference.csv.gz"
SERIES_INDEX = BASE / "data" / "raw" / "series_index.csv"
OUT = BASE / "data" / "processed" / "review_collection"
SALES_START = pd.Timestamp("2022-01-01")
SALES_END = pd.Timestamp("2026-06-30")


def norm_name(value: object) -> str:
    """Conservative name normalisation for cross-platform series matching."""
    return re.sub(r"[\s\-－_（）()【】\[\]]", "", str(value)).lower()


def _review_stats(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    reviews["series_key"] = reviews["series_name"].map(norm_name)
    reviews["publish_time"] = pd.to_datetime(reviews["publish_time"], errors="coerce")
    return (reviews.groupby("series_key")
            .agg(archived_review_count=("review_id", "nunique"),
                 archived_review_earliest=("publish_time", "min"),
                 archived_review_latest=("publish_time", "max"),
                 archived_review_in_sales_window=("publish_time", lambda s: int(s.between(SALES_START, SALES_END).sum())))
            .reset_index())


def _absa_stats(absa: pd.DataFrame) -> pd.DataFrame:
    absa = absa.copy()
    absa["series_key"] = absa["series_name"].map(norm_name)
    absa["publish_time"] = pd.to_datetime(absa["publish_time"], errors="coerce")
    if "success" in absa:
        absa = absa[absa["success"].astype(str).str.lower().eq("true")]
    return (absa.groupby("series_key")
            .agg(archived_label_count=("review_id", "nunique"),
                 archived_label_earliest=("publish_time", "min"),
                 archived_label_latest=("publish_time", "max"))
            .reset_index())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv(SPLITS / "test.csv", usecols=["series_name"])
    population = population.drop_duplicates().copy()
    population["series_key"] = population["series_name"].map(norm_name)

    reviews = pd.read_csv(HISTORICAL_RESOURCE, low_memory=False)
    # "legacy_deepseek_available" is the archived table's original column name;
    # it only marks label presence locally and never reaches public features.
    absa = reviews.loc[reviews["legacy_deepseek_available"].astype(str).str.lower().eq("true")].copy()
    audit = population.merge(_review_stats(reviews), on="series_key", how="left")
    audit = audit.merge(_absa_stats(absa), on="series_key", how="left")
    if not SERIES_INDEX.exists():
        raise FileNotFoundError("Run 00_build_series_index.py before the sentiment readiness audit.")
    index = pd.read_csv(SERIES_INDEX, usecols=["series_name", "dongchedi_series_id"])
    index["series_key"] = index["series_name"].map(norm_name)
    index = index.dropna(subset=["dongchedi_series_id"]).drop_duplicates("series_key")
    audit = audit.merge(index[["series_key", "dongchedi_series_id"]], on="series_key", how="left")
    for col in ("archived_review_count", "archived_review_in_sales_window", "archived_label_count"):
        audit[col] = audit[col].fillna(0).astype(int)

    audit["archived_reviews_available"] = audit["archived_review_count"] > 0
    audit["archived_labels_available"] = audit["archived_label_count"] > 0
    audit["dongchedi_id_available"] = audit["dongchedi_series_id"].notna()
    audit["action"] = "collect"
    audit.loc[audit["archived_reviews_available"], "action"] = "reuse_reviews_then_refresh"
    audit.loc[audit["archived_labels_available"], "action"] = "reuse_labels_then_refresh"
    audit.loc[~audit["archived_reviews_available"] & ~audit["dongchedi_id_available"], "action"] = "resolve_source_id_then_collect"
    audit = audit.sort_values(["action", "series_name"]).reset_index(drop=True)
    audit.to_csv(OUT / "review_readiness.csv", index=False, encoding="utf-8-sig")
    crawl_gap = audit.loc[~audit["archived_reviews_available"], [
        "series_name", "series_key", "dongchedi_series_id", "dongchedi_id_available", "action"
    ]]
    crawl_gap.to_csv(
        OUT / "review_collection_gap.csv", index=False, encoding="utf-8-sig"
    )
    crawl_gap.loc[~crawl_gap["dongchedi_id_available"]].to_csv(
        OUT / "review_id_resolution_gap.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "forecast_series": int(len(audit)),
        "reusable_review_series": int(audit["archived_reviews_available"].sum()),
        "reusable_label_series": int(audit["archived_labels_available"].sum()),
        "collection_gap_series": int((~audit["archived_reviews_available"]).sum()),
        "collection_gap_with_dongchedi_id": int((~audit["archived_reviews_available"] & audit["dongchedi_id_available"]).sum()),
        "collection_gap_id_resolution_required": int((~audit["archived_reviews_available"] & ~audit["dongchedi_id_available"]).sum()),
        "review_sales_window": [str(SALES_START.date()), str(SALES_END.date())],
        "note": "Archived coverage is reused where available; uncovered series remain in the collection queue.",
    }
    (OUT / "review_readiness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

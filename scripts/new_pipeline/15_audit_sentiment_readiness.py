#!/usr/bin/env python3
"""Audit old sentiment assets before collecting the new Phase-B corpus.

The new forecasting population is defined by ``processed_new/splits/test.csv``.
This script measures exactly which of those series can reuse historical review
and ABSA records and writes a gap list for incremental crawling.  It does not
call an external API and never overwrites the old sentiment data.

Outputs
-------
data/processed_new/phase_b/sentiment_readiness.csv
    One row per new-population series, with review/ABSA coverage and action.
data/processed_new/phase_b/sentiment_crawl_gap.csv
    The series that have no reusable historical review coverage, including the
    availability of a valid Dongchedi numeric ID.
data/processed_new/phase_b/sentiment_id_resolution_gap.csv
    Missing-review series whose Dongchedi ID must be resolved before any
    source-specific collection can begin.
data/processed_new/phase_b/sentiment_readiness_summary.json
    Compact counts for the project report and crawl planning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "processed_new" / "splits"
SENTIMENT = BASE / "data" / "sentiment"
SERIES_INDEX = BASE / "data" / "raw" / "series_index.csv"
OUT = BASE / "data" / "processed_new" / "phase_b"
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
            .agg(old_review_count=("review_id", "nunique"),
                 old_review_earliest=("publish_time", "min"),
                 old_review_latest=("publish_time", "max"),
                 old_review_in_sales_window=("publish_time", lambda s: int(s.between(SALES_START, SALES_END).sum())))
            .reset_index())


def _absa_stats(absa: pd.DataFrame) -> pd.DataFrame:
    absa = absa.copy()
    absa["series_key"] = absa["series_name"].map(norm_name)
    absa["publish_time"] = pd.to_datetime(absa["publish_time"], errors="coerce")
    if "success" in absa:
        absa = absa[absa["success"].astype(str).str.lower().eq("true")]
    return (absa.groupby("series_key")
            .agg(old_absa_count=("review_id", "nunique"),
                 old_absa_earliest=("publish_time", "min"),
                 old_absa_latest=("publish_time", "max"))
            .reset_index())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv(SPLITS / "test.csv", usecols=["series_name"])
    population = population.drop_duplicates().copy()
    population["series_key"] = population["series_name"].map(norm_name)

    reviews = pd.read_csv(SENTIMENT / "sentiment_reviews.csv", low_memory=False)
    absa = pd.read_csv(SENTIMENT / "absa" / "absa_results.csv", low_memory=False)
    audit = population.merge(_review_stats(reviews), on="series_key", how="left")
    audit = audit.merge(_absa_stats(absa), on="series_key", how="left")
    if not SERIES_INDEX.exists():
        raise FileNotFoundError("Run 00_build_series_index.py before the sentiment readiness audit.")
    index = pd.read_csv(SERIES_INDEX, usecols=["series_name", "dongchedi_series_id"])
    index["series_key"] = index["series_name"].map(norm_name)
    index = index.dropna(subset=["dongchedi_series_id"]).drop_duplicates("series_key")
    audit = audit.merge(index[["series_key", "dongchedi_series_id"]], on="series_key", how="left")
    for col in ("old_review_count", "old_review_in_sales_window", "old_absa_count"):
        audit[col] = audit[col].fillna(0).astype(int)

    audit["old_reviews_available"] = audit["old_review_count"] > 0
    audit["old_absa_available"] = audit["old_absa_count"] > 0
    audit["dongchedi_id_available"] = audit["dongchedi_series_id"].notna()
    audit["action"] = "crawl_new"
    audit.loc[audit["old_reviews_available"], "action"] = "reuse_reviews_then_refresh"
    audit.loc[audit["old_absa_available"], "action"] = "reuse_absa_then_refresh"
    audit.loc[~audit["old_reviews_available"] & ~audit["dongchedi_id_available"], "action"] = "resolve_source_id_then_crawl"
    audit = audit.sort_values(["action", "series_name"]).reset_index(drop=True)
    audit.to_csv(OUT / "sentiment_readiness.csv", index=False, encoding="utf-8-sig")
    crawl_gap = audit.loc[~audit["old_reviews_available"], [
        "series_name", "series_key", "dongchedi_series_id", "dongchedi_id_available", "action"
    ]]
    crawl_gap.to_csv(
        OUT / "sentiment_crawl_gap.csv", index=False, encoding="utf-8-sig"
    )
    crawl_gap.loc[~crawl_gap["dongchedi_id_available"]].to_csv(
        OUT / "sentiment_id_resolution_gap.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "new_population_series": int(len(audit)),
        "reusable_review_series": int(audit["old_reviews_available"].sum()),
        "reusable_absa_series": int(audit["old_absa_available"].sum()),
        "crawl_gap_series": int((~audit["old_reviews_available"]).sum()),
        "crawl_gap_with_dongchedi_id": int((~audit["old_reviews_available"] & audit["dongchedi_id_available"]).sum()),
        "crawl_gap_id_resolution_required": int((~audit["old_reviews_available"] & ~audit["dongchedi_id_available"]).sum()),
        "review_sales_window": [str(SALES_START.date()), str(SALES_END.date())],
        "note": "Old coverage is reusable seed data only; refresh/re-crawl is still required for a new Phase-B corpus.",
    }
    (OUT / "sentiment_readiness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate, deduplicate and time-audit the incremental review staging corpus.

Raw crawler output is immutable evidence.  This script writes a separate
deduplicated corpus and a per-series quality report, so failed/retried pages or
post-cutoff reviews never silently contaminate a modelling table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
RAW = BASE / "data" / "sentiment_new" / "raw" / "dongchedi_incremental_reviews.csv"
OUT = BASE / "data" / "sentiment_new" / "processed"
DEDUP = OUT / "dongchedi_incremental_reviews_dedup.csv"
AUDIT = OUT / "incremental_review_quality.csv"
SUMMARY = OUT / "incremental_review_quality_summary.json"
FORECAST_TEST_END = pd.Timestamp("2026-06-30 23:59:59")


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(f"No crawler output yet: {RAW}")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW, low_memory=False)
    required = {"series_id", "series_name", "review_id", "publish_time", "content"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required review columns: {sorted(missing)}")

    raw["publish_time"] = pd.to_datetime(raw["publish_time"], errors="coerce")
    raw["review_id"] = raw["review_id"].astype(str).str.strip()
    # A blank source ID cannot establish identity and is excluded rather than
    # being collapsed into one artificial duplicate.
    valid_id = raw["review_id"].ne("") & raw["review_id"].ne("nan")
    dedup = pd.concat([
        raw.loc[valid_id].drop_duplicates("review_id", keep="first"),
        raw.loc[~valid_id],
    ], ignore_index=True).sort_values(["series_name", "publish_time", "review_id"])
    dedup.to_csv(DEDUP, index=False, encoding="utf-8-sig")

    raw_by_series = raw.groupby(["series_id", "series_name"], dropna=False).size().rename("raw_rows")
    audit = dedup.groupby(["series_id", "series_name"], dropna=False).agg(
        usable_reviews=("review_id", "size"),
        earliest_review=("publish_time", "min"),
        latest_review=("publish_time", "max"),
        undated_reviews=("publish_time", lambda s: int(s.isna().sum())),
        pre_test_cutoff_reviews=("publish_time", lambda s: int(s.le(FORECAST_TEST_END).sum())),
        post_test_cutoff_reviews=("publish_time", lambda s: int(s.gt(FORECAST_TEST_END).sum())),
    ).join(raw_by_series).reset_index()
    audit["duplicate_rows_removed"] = audit["raw_rows"] - audit["usable_reviews"]
    audit["time_safe_for_2026_test"] = audit["pre_test_cutoff_reviews"] > 0
    audit = audit.sort_values(["time_safe_for_2026_test", "usable_reviews"], ascending=[True, False])
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    summary = {
        "raw_rows": int(len(raw)),
        "deduplicated_rows": int(len(dedup)),
        "duplicate_rows_removed": int(len(raw) - len(dedup)),
        "series_collected": int(audit["series_name"].nunique()),
        "series_with_pre_test_cutoff_review": int(audit["time_safe_for_2026_test"].sum()),
        "forecast_test_end": str(FORECAST_TEST_END),
        "note": "Only reviews dated no later than forecast_test_end may be used for the 2026-01 to 2026-06 test evaluation.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

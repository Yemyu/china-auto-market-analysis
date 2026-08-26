#!/usr/bin/env python3
"""Materialise per-forecast-month review availability without future leakage."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "processed_new" / "splits"
OUT = BASE / "data" / "sentiment_new" / "processed"
CORPUS = OUT / "target_371_review_corpus.csv"
AVAILABILITY = OUT / "review_temporal_availability_by_series.csv"
SUMMARY = OUT / "review_temporal_availability_summary.json"


def main() -> None:
    test = pd.read_csv(SPLITS / "test.csv", usecols=["series_name", "date"])
    test["date"] = pd.to_datetime(test["date"], errors="raise")
    months = sorted(test["date"].dt.to_period("M").unique())
    target = test[["series_name"]].drop_duplicates().sort_values("series_name").reset_index(drop=True)
    corpus = pd.read_csv(CORPUS, low_memory=False)
    corpus["publish_time"] = pd.to_datetime(corpus["publish_time"], errors="coerce")
    usable = corpus.loc[
        corpus["eligible_for_temporal_model"].fillna(False).astype(bool) & corpus["publish_time"].notna(),
        ["series_name_canonical", "publish_time", "review_id"],
    ].copy()

    result = target.copy()
    month_summary: list[dict] = []
    for period in months:
        origin = period.start_time
        column = f"reviews_available_before_{period.strftime('%Y_%m')}"
        counts = (usable.loc[usable["publish_time"] < origin]
                  .groupby("series_name_canonical")["review_id"].size())
        result[column] = result["series_name"].map(counts).fillna(0).astype(int)
        month_summary.append({
            "forecast_month": str(period), "information_cutoff_exclusive": str(origin),
            "series_with_any_available_review": int(result[column].gt(0).sum()),
            "available_review_rows": int(result[column].sum()),
        })
    result.to_csv(AVAILABILITY, index=False, encoding="utf-8-sig")
    summary = {
        "forecast_months": month_summary,
        "rule": "For forecast month t, the count includes only reviews with publish_time strictly before the first day of t. Feature engineering may impose a stricter lag, but may never use a later timestamp.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

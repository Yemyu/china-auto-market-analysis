#!/usr/bin/env python3
"""Audit source, content type, and date coverage in the review corpus."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "reviews" / "processed"
CORPUS = OUT / "target_371_review_corpus.csv"
BY_SOURCE = OUT / "multisource_review_quality_by_source.csv"
BY_SERIES = OUT / "multisource_review_quality_by_series.csv"
SUMMARY = OUT / "multisource_review_quality_summary.json"
TEST_END = pd.Timestamp("2026-06-30 23:59:59")


def as_bool(values: pd.Series) -> pd.Series:
    """Read bools safely after CSV round trips, where blanks become NaN."""
    return values.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def source_aggregate(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (data.groupby(group_cols, dropna=False)
            .agg(rows=("review_id", "size"),
                 target_series=("series_name_canonical", "nunique"),
                 unique_platform_review_ids=("identity", "nunique"),
                 missing_review_id=("valid_identity", lambda s: int((~s).sum())),
                 missing_publish_time=("publish_time", lambda s: int(s.isna().sum())),
                 blank_content=("valid_content", lambda s: int((~s).sum())),
                 mean_content_length=("content_len", "mean"),
                 list_summary_rows=("is_list_summary", "sum"),
                 detail_full_html_rows=("is_detail_full_html", "sum"),
                 possibly_abbreviated_rows=("possibly_abbreviated", "sum"),
                 pre_test_end_rows=("is_pre_test_end", "sum"),
                 post_test_end_rows=("is_post_test_end", "sum"),
                 earliest_review=("publish_time", "min"),
                 latest_review=("publish_time", "max"))
            .reset_index())


def main() -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(f"Run 18_build_target_review_corpus.py first: {CORPUS}")
    data = pd.read_csv(CORPUS, low_memory=False)
    required = {"corpus_source", "platform", "review_id", "publish_time", "content", "series_name_canonical", "identity", "valid_identity"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Corpus missing required columns: {sorted(missing)}")
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="coerce")
    data["content"] = data["content"].fillna("").astype(str).str.strip()
    data["content_len"] = data["content"].str.len()
    data["valid_content"] = data["content"].ne("")
    data["valid_identity"] = as_bool(data["valid_identity"])
    data["is_pre_test_end"] = data["publish_time"].le(TEST_END)
    data["is_post_test_end"] = data["publish_time"].gt(TEST_END)
    data["is_list_summary"] = data.get("content_source", pd.Series("", index=data.index)).fillna("").eq("list_structured_summary")
    data["is_detail_full_html"] = data.get("content_source", pd.Series("", index=data.index)).fillna("").eq("detail_full_html")
    data["possibly_abbreviated"] = as_bool(data.get("content_possibly_abbreviated", pd.Series(False, index=data.index)))

    by_source = source_aggregate(data, ["corpus_source", "platform"])
    by_source["possibly_abbreviated_rate"] = (by_source["possibly_abbreviated_rows"] / by_source["rows"]).round(4)
    by_source.to_csv(BY_SOURCE, index=False, encoding="utf-8-sig")

    by_series = source_aggregate(data, ["series_name_canonical", "corpus_source", "platform"])
    by_series["possibly_abbreviated_rate"] = (by_series["possibly_abbreviated_rows"] / by_series["rows"]).round(4)
    by_series.to_csv(BY_SERIES, index=False, encoding="utf-8-sig")

    summary = {
        "corpus_rows": int(len(data)),
        "target_series_with_any_review": int(data["series_name_canonical"].nunique()),
        "rows_with_valid_platform_review_id": int(data["valid_identity"].sum()),
        "rows_with_valid_time_and_content": int((data["publish_time"].notna() & data["valid_content"]).sum()),
        "pre_test_end_rows": int(data["is_pre_test_end"].sum()),
        "post_test_end_rows": int(data["is_post_test_end"].sum()),
        "list_summary_rows": int(data["is_list_summary"].sum()),
        "detail_full_html_rows": int(data["is_detail_full_html"].sum()),
        "possibly_abbreviated_rows": int(data["possibly_abbreviated"].sum()),
        "test_end": str(TEST_END),
        "rule": "For a forecast at any origin, filter reviews to publish_time strictly earlier than that origin; this audit's test-end count is only a coarse leakage check.",
        "source_note": "Autohome list summaries remain preserved in the raw staging table; only successfully parsed detail pages are promoted to full text in the integrated corpus.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

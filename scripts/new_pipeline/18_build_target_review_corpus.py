#!/usr/bin/env python3
"""Build a time-auditable review corpus for the new 371-series population.

This is deliberately a data-integration step, not an ABSA or forecasting
step.  It preserves the old corpus and the new incremental corpus unchanged,
then creates one standardised, de-duplicated table keyed to the authoritative
new series roster.  Downstream modelling must filter this corpus by its own
forecast origin; this script merely exposes the dates required to do so.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "processed_new" / "splits"
OLD = BASE / "data" / "resources" / "legacy_sentiment" / "review_absa_reference.csv.gz"
NEW = BASE / "data" / "sentiment_new" / "processed" / "dongchedi_incremental_reviews_dedup.csv"
AUTOHOME = BASE / "data" / "sentiment_new" / "raw" / "autohome_incremental_reviews.csv"
AUTOHOME_DETAILS = BASE / "data" / "sentiment_new" / "raw" / "autohome_incremental_review_details.csv"
OUT = BASE / "data" / "sentiment_new" / "processed"
CORPUS = OUT / "target_371_review_corpus.csv"
COVERAGE = OUT / "target_371_review_coverage.csv"
SUMMARY = OUT / "target_371_review_corpus_summary.json"
TEST_END = pd.Timestamp("2026-06-30 23:59:59")
LEGACY_LABEL_COLUMNS = {
    "success", "error", "appearance", "interior", "space", "power",
    "control", "comfort", "fuel_consumption", "configuration",
    "intelligence", "value", "legacy_deepseek_available",
    "legacy_deepseek_prompt_version", "legacy_deepseek_model",
}


def norm_name(value: object) -> str:
    return re.sub(r"[\s\-－_（）()【】\[\]]", "", str(value)).lower()


def prepare(source: pd.DataFrame, source_name: str, roster: pd.DataFrame) -> pd.DataFrame:
    data = source.copy()
    data["series_key"] = data["series_name"].map(norm_name)
    data = data.merge(roster, on="series_key", how="inner", validate="many_to_one")
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="coerce")
    data["review_id"] = data["review_id"].astype(str).str.strip()
    data["content"] = data["content"].fillna("").astype(str).str.strip()
    data["corpus_source"] = source_name
    data["source_series_name"] = data["series_name"]
    data = data.rename(columns={"target_series_name": "series_name_canonical"})
    return data


def load_autohome() -> pd.DataFrame:
    """Promote a list summary to full text only after a successful detail parse."""
    summaries = pd.read_csv(AUTOHOME, low_memory=False)
    summaries["review_id"] = summaries["review_id"].astype(str).str.strip()
    summaries["summary_content"] = summaries["content"]
    summaries["summary_content_len"] = summaries["content_len"]
    if not AUTOHOME_DETAILS.exists():
        return summaries
    details = pd.read_csv(AUTOHOME_DETAILS, low_memory=False)
    details["review_id"] = details["review_id"].astype(str).str.strip()
    if details["review_id"].duplicated().any():
        raise ValueError("Autohome detail corpus has duplicate review IDs")
    keep = [
        "review_id", "detail_url", "detail_status", "detail_content",
        "detail_content_len", "detail_section_count", "fetched_at", "error",
    ]
    data = summaries.merge(details[keep], on="review_id", how="left", validate="one_to_one")
    promote = data["detail_status"].eq("ok") & data["detail_content"].fillna("").astype(str).str.strip().ne("")
    data.loc[promote, "content"] = data.loc[promote, "detail_content"]
    data.loc[promote, "content_len"] = data.loc[promote, "detail_content_len"]
    data.loc[promote, "content_source"] = "detail_full_html"
    data.loc[promote, "content_possibly_abbreviated"] = False
    return data


def load_legacy_reviews() -> pd.DataFrame:
    """Read review evidence without leaking the colocated label columns."""
    data = pd.read_csv(OLD, low_memory=False)
    return data.drop(columns=[c for c in LEGACY_LABEL_COLUMNS if c in data], errors="ignore")


def main() -> None:
    if not OLD.exists():
        raise FileNotFoundError(OLD)
    OUT.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(SPLITS / "test.csv", usecols=["series_name"]).drop_duplicates().copy()
    target["series_key"] = target["series_name"].map(norm_name)
    if target["series_key"].duplicated().any():
        duplicates = target.loc[target["series_key"].duplicated(keep=False), "series_name"].tolist()
        raise ValueError(f"Ambiguous normalised target names: {duplicates}")
    roster = target.rename(columns={"series_name": "target_series_name"})

    frames = [prepare(load_legacy_reviews(), "old_v1", roster)]
    if NEW.exists():
        frames.append(prepare(pd.read_csv(NEW, low_memory=False), "dongchedi_incremental", roster))
    if AUTOHOME.exists():
        frames.append(prepare(load_autohome(), "autohome_incremental", roster))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    matched_rows = len(combined)

    # A public review ID is only an identity inside its platform.  A freshly
    # crawled copy takes precedence if the *same platform review* is present
    # in both sources; distinct platforms are never deduplicated by text.
    combined["identity"] = combined["platform"].fillna("unknown").astype(str) + "::" + combined["review_id"]
    combined["valid_identity"] = combined["review_id"].notna() & ~combined["review_id"].isin(["", "nan"])
    combined["source_priority"] = combined["corpus_source"].isin(
        ["dongchedi_incremental", "autohome_incremental"]
    ).astype(int)
    combined = combined.sort_values(["valid_identity", "identity", "source_priority"])
    identified = combined.loc[combined["valid_identity"]].drop_duplicates("identity", keep="last")
    unidentified = combined.loc[~combined["valid_identity"]]
    corpus = pd.concat([identified, unidentified], ignore_index=True)
    duplicate_rows_removed = matched_rows - len(corpus)

    corpus["valid_time"] = corpus["publish_time"].notna()
    corpus["valid_content"] = corpus["content"].ne("")
    # Autohome's list endpoint is a resumable discovery source, but its text
    # can be abbreviated. Preserve those summaries in the corpus/audit while
    # excluding them from sentiment features unless the corresponding detail
    # page was fetched and parsed successfully.
    corpus["eligible_content_quality"] = (
        corpus["corpus_source"].ne("autohome_incremental")
        | corpus["content_source"].fillna("").eq("detail_full_html")
    )
    corpus["eligible_for_temporal_model"] = (
        corpus["valid_time"] & corpus["valid_content"] & corpus["eligible_content_quality"]
    )
    corpus["after_2026_test_end"] = corpus["publish_time"].gt(TEST_END)
    corpus = corpus.sort_values(["series_name_canonical", "publish_time", "review_id"])
    corpus.to_csv(CORPUS, index=False, encoding="utf-8-sig")

    usable = corpus.loc[corpus["eligible_for_temporal_model"]].copy()
    coverage = (usable.groupby("series_name_canonical")
                .agg(review_count=("review_id", "size"),
                     old_v1_reviews=("corpus_source", lambda s: int(s.eq("old_v1").sum())),
                     dongchedi_incremental_reviews=("corpus_source", lambda s: int(s.eq("dongchedi_incremental").sum())),
                     autohome_incremental_reviews=("corpus_source", lambda s: int(s.eq("autohome_incremental").sum())),
                     earliest_review=("publish_time", "min"),
                     latest_review=("publish_time", "max"),
                     pre_test_end_reviews=("after_2026_test_end", lambda s: int((~s).sum())),
                     post_test_end_reviews=("after_2026_test_end", "sum"))
                .reset_index())
    coverage = roster[["target_series_name"]].merge(
        coverage, left_on="target_series_name", right_on="series_name_canonical", how="left"
    ).drop(columns="series_name_canonical").rename(columns={"target_series_name": "series_name"})
    for col in ["review_count", "old_v1_reviews", "dongchedi_incremental_reviews", "autohome_incremental_reviews", "pre_test_end_reviews", "post_test_end_reviews"]:
        coverage[col] = coverage[col].fillna(0).astype(int)
    coverage["review_coverage_status"] = "no_review"
    coverage.loc[coverage["review_count"].gt(0), "review_coverage_status"] = "review_available"
    coverage.loc[coverage["pre_test_end_reviews"].gt(0), "review_coverage_status"] = "pre_test_review_available"
    coverage = coverage.sort_values(["review_coverage_status", "series_name"])
    coverage.to_csv(COVERAGE, index=False, encoding="utf-8-sig")

    summary = {
        "target_series": int(len(roster)),
        "source_rows_matched_to_target": int(matched_rows),
        "deduplicated_corpus_rows": int(len(corpus)),
        "cross_source_duplicate_rows_removed": int(duplicate_rows_removed),
        "temporally_eligible_reviews": int(corpus["eligible_for_temporal_model"].sum()),
        "autohome_list_summary_rows_excluded_from_temporal_model": int(
            (corpus["corpus_source"].eq("autohome_incremental")
             & ~corpus["eligible_content_quality"]).sum()
        ),
        "target_series_with_any_review": int(coverage["review_count"].gt(0).sum()),
        "target_series_with_pre_test_review": int(coverage["pre_test_end_reviews"].gt(0).sum()),
        "test_end": str(TEST_END),
        "rule": "For every forecast origin, downstream features must additionally use only reviews published before that origin.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

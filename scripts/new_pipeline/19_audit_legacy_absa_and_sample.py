#!/usr/bin/env python3
"""Audit reusable legacy ABSA labels and create a deterministic QA sample.

Legacy labels are preserved as a baseline only.  In particular, the old
three-way -1/0/+1 format does not distinguish ``not mentioned`` from a neutral
mention, so this script reports its coverage and distribution rather than
silently treating it as the new gold standard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
CORPUS = BASE / "data" / "sentiment_new" / "processed" / "target_371_review_corpus.csv"
LEGACY = BASE / "data" / "resources" / "legacy_sentiment" / "review_absa_reference.csv.gz"
OUT = BASE / "data" / "sentiment_new" / "processed"
ASPECTS = ["appearance", "interior", "space", "power", "control", "comfort",
           "fuel_consumption", "configuration", "intelligence", "value"]
SUMMARY = OUT / "legacy_absa_audit_summary.json"
DISTRIBUTION = OUT / "legacy_absa_aspect_distribution.csv"
COVERAGE = OUT / "legacy_absa_target_coverage.csv"
SAMPLE = OUT / "absa_qa_sample.csv"


def _valid_id(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.notna() & ~text.isin(["", "nan", "<NA>"])


def main() -> None:
    if not CORPUS.exists() or not LEGACY.exists():
        raise FileNotFoundError("Build the unified review corpus and retain legacy ABSA results first.")
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(CORPUS, low_memory=False)
    legacy = pd.read_csv(LEGACY, low_memory=False)
    corpus["review_id_key"] = corpus["review_id"].astype("string").str.strip()
    legacy["review_id_key"] = legacy["review_id"].astype("string").str.strip()
    legacy["success"] = legacy["success"].astype(str).str.lower().eq("true")
    valid_legacy = legacy.loc[legacy["success"] & _valid_id(legacy["review_id_key"])].copy()
    valid_legacy = valid_legacy.drop_duplicates("review_id_key", keep="last")

    label_cols = ["review_id_key", "success", *ASPECTS]
    labeled = corpus.merge(valid_legacy[label_cols], on="review_id_key", how="left", validate="many_to_one")
    labeled["legacy_absa_available"] = labeled["success"].fillna(False)
    labeled["eligible_pre_test"] = labeled["eligible_for_temporal_model"] & ~labeled["after_2026_test_end"]

    eligible = labeled.loc[labeled["eligible_pre_test"]].copy()
    coverage = (eligible.groupby("series_name_canonical")
                .agg(pre_test_reviews=("review_id", "size"),
                     legacy_absa_reviews=("legacy_absa_available", "sum"))
                .reset_index())
    coverage["legacy_absa_reviews"] = coverage["legacy_absa_reviews"].astype(int)
    coverage["legacy_absa_coverage"] = coverage["legacy_absa_reviews"] / coverage["pre_test_reviews"]
    coverage.to_csv(COVERAGE, index=False, encoding="utf-8-sig")

    labeled_legacy = eligible.loc[eligible["legacy_absa_available"]]
    distributions = []
    for aspect in ASPECTS:
        counts = labeled_legacy[aspect].value_counts(dropna=False).rename_axis("label").reset_index(name="review_count")
        counts.insert(0, "aspect", aspect)
        counts["share"] = counts["review_count"] / len(labeled_legacy) if len(labeled_legacy) else 0.0
        distributions.append(counts)
    pd.concat(distributions, ignore_index=True).to_csv(DISTRIBUTION, index=False, encoding="utf-8-sig")

    # A stable, diverse sample for schema review and later human/LLM agreement
    # checking.  It intentionally includes unlabeled new reviews.
    qa = eligible.copy()
    qa["rating_bucket"] = pd.cut(qa["rating_overall"], bins=[-float("inf"), 3, 4, float("inf")],
                                  labels=["low_or_missing", "mid", "high"]).astype(str)
    qa["qa_stratum"] = qa["corpus_source"].astype(str) + "|" + qa["legacy_absa_available"].map({True: "legacy", False: "unlabeled"}) + "|" + qa["rating_bucket"]
    groups = list(qa.groupby("qa_stratum", observed=True))
    target_n = min(200, len(qa))
    per_group = max(1, target_n // max(1, len(groups)))
    pieces = [g.sample(n=min(len(g), per_group), random_state=20260823 + i)
              for i, (_, g) in enumerate(groups)]
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) < target_n:
        remaining = qa.loc[~qa.index.isin(sample.index)]
        # Index mismatch after concat is harmless but avoid a random re-draw of
        # selected records by using review identity.
        selected = set(sample["review_id_key"])
        remaining = qa.loc[~qa["review_id_key"].isin(selected)]
        sample = pd.concat([sample, remaining.sample(n=min(target_n - len(sample), len(remaining)), random_state=20260824)], ignore_index=True)
    keep = ["series_name_canonical", "review_id", "publish_time", "content", "rating_overall",
            "corpus_source", "legacy_absa_available", "qa_stratum", *ASPECTS]
    sample[keep].to_csv(SAMPLE, index=False, encoding="utf-8-sig")

    summary = {
        "eligible_pre_test_reviews": int(len(eligible)),
        "legacy_absa_matched_reviews": int(eligible["legacy_absa_available"].sum()),
        "legacy_absa_coverage": round(float(eligible["legacy_absa_available"].mean()), 4),
        "target_series_with_legacy_absa": int(coverage["legacy_absa_reviews"].gt(0).sum()),
        "qa_sample_size": int(len(sample)),
        "legacy_label_warning": "A legacy value of 0 is ambiguous: it may mean neutral sentiment or no aspect mention. New labels must separate mention from polarity.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

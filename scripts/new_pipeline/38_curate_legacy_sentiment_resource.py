#!/usr/bin/env python3
"""Curate the reusable legacy review and DeepSeek ABSA asset.

The first project version stored reviews, labels and a JSON checkpoint in
separate locations.  The checkpoint and label CSV are semantically identical,
while the review file also contains exact duplicate rows.  This script creates
one compact, auditable resource with one row per review and optional legacy
DeepSeek labels.  Source files are read-only and may be removed after the
output and manifest have been validated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[2]
REVIEWS = BASE / "data" / "sentiment" / "sentiment_reviews.csv"
LABELS = BASE / "data" / "sentiment" / "absa" / "absa_results.csv"
OUT = BASE / "data" / "resources" / "legacy_sentiment"
RESOURCE = OUT / "review_absa_reference.csv.gz"
MANIFEST = OUT / "manifest.json"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]


def id_key(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not REVIEWS.exists() or not LABELS.exists():
        raise FileNotFoundError("Legacy source files are required for the one-time curation step")
    OUT.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_csv(REVIEWS, low_memory=False)
    reviews["review_id_key"] = id_key(reviews["review_id"])
    duplicate_rows = int(reviews.duplicated().sum())
    conflicting_ids = int(
        reviews.loc[reviews["review_id_key"].duplicated(False)]
        .groupby("review_id_key", dropna=False)
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .sum()
    )
    if conflicting_ids:
        raise ValueError(f"Legacy reviews contain {conflicting_ids} conflicting duplicate IDs")
    reviews = reviews.drop_duplicates("review_id_key", keep="first").copy()

    labels = pd.read_csv(LABELS, low_memory=False)
    labels["review_id_key"] = id_key(labels["review_id"])
    if labels["review_id_key"].duplicated().any():
        raise ValueError("Legacy ABSA labels contain duplicate review IDs")
    labels["success"] = labels["success"].astype(str).str.lower().eq("true")
    for aspect in ASPECTS:
        labels[aspect] = pd.to_numeric(labels[aspect], errors="raise").astype(int)
        if not labels[aspect].isin([-1, 0, 1]).all():
            raise ValueError(f"Unexpected legacy label value for {aspect}")

    label_columns = ["review_id_key", "success", "error", *ASPECTS]
    resource = reviews.merge(
        labels[label_columns], on="review_id_key", how="left", validate="one_to_one"
    )
    resource["legacy_deepseek_available"] = resource["success"].fillna(False)
    resource["legacy_deepseek_prompt_version"] = resource["legacy_deepseek_available"].map(
        {True: "legacy_absa_v1_2025", False: pd.NA}
    )
    resource["legacy_deepseek_model"] = resource["legacy_deepseek_available"].map(
        {True: "deepseek-chat", False: pd.NA}
    )
    resource = resource.sort_values(["series_name", "publish_time", "review_id_key"])
    resource.to_csv(
        RESOURCE,
        index=False,
        encoding="utf-8-sig",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )

    check = pd.read_csv(RESOURCE, low_memory=False)
    if len(check) != len(resource) or check["review_id_key"].duplicated().any():
        raise ValueError("Compressed resource failed round-trip validation")
    labeled = check["legacy_deepseek_available"].astype(str).str.lower().eq("true")
    if int(labeled.sum()) != len(labels):
        raise ValueError("Compressed resource lost legacy label coverage")

    manifest = {
        "schema_version": "1.0",
        "resource": str(RESOURCE.relative_to(BASE)),
        "sha256": sha256(RESOURCE),
        "reviews": int(len(check)),
        "series": int(check["series_name"].nunique()),
        "duplicate_source_rows_removed": duplicate_rows,
        "legacy_deepseek_labeled_reviews": int(labeled.sum()),
        "unlabeled_reviews": int((~labeled).sum()),
        "review_date_min": str(pd.to_datetime(check["publish_time"]).min()),
        "review_date_max": str(pd.to_datetime(check["publish_time"]).max()),
        "label_values": {"-1": "negative", "0": "ambiguous neutral/unmentioned/parser fallback", "1": "positive"},
        "warning": "Legacy zero is not mention ground truth; use the unified 2026 feature table for forecasting.",
        "source_files": [
            "data/sentiment/sentiment_reviews.csv",
            "data/sentiment/absa/absa_results.csv",
        ],
        "external_api_calls": 0,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

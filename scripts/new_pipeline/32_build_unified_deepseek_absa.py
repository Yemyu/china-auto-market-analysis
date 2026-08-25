#!/usr/bin/env python3
"""Build one auditable DeepSeek ABSA table for all 24,175 eligible reviews.

The output reuses 16,538 successful legacy labels and adds 7,637 compact
V4-Flash labels. Legacy zero values remain explicitly ambiguous, while a
deterministic local mention detector supplies one uniform mention flag across
both label generations. The original API outputs are never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path(__file__).resolve().parents[2]
SENTIMENT = BASE / "data" / "sentiment_new" / "processed"
CORPUS = SENTIMENT / "target_371_review_corpus.csv"
LEGACY = BASE / "data" / "resources" / "legacy_sentiment" / "review_absa_reference.csv.gz"
COMPACT = SENTIMENT / "deepseek_absa_compact_results.csv"
LOCAL = SENTIMENT / "local_sentiment_review_features.csv"
MANUAL_QA = SENTIMENT / "deepseek_absa_all_null_manual_qa.json"

OUTPUT = SENTIMENT / "unified_deepseek_absa_review_features.csv"
QA_OUTPUT = SENTIMENT / "deepseek_absa_all_null_manual_qa.csv"
SUMMARY = SENTIMENT / "unified_deepseek_absa_summary.json"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
EXPECTED_ELIGIBLE = 24_175
EXPECTED_LEGACY = 16_538
EXPECTED_COMPACT = 7_637
EXPECTED_ALL_NULL = 42


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().map({"true": True, "false": False})


def id_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_csv(CORPUS, low_memory=False)
    eligible = bool_series(corpus["eligible_for_temporal_model"]).fillna(False)
    corpus = corpus.loc[eligible].copy()
    corpus["review_id_key"] = id_key(corpus["review_id"])
    corpus["publish_time"] = pd.to_datetime(corpus["publish_time"], errors="raise")
    if len(corpus) != EXPECTED_ELIGIBLE or corpus["identity"].duplicated().any():
        raise ValueError("Eligible corpus population or identity uniqueness changed")

    legacy = pd.read_csv(LEGACY, low_memory=False)
    legacy = legacy.loc[bool_series(legacy["success"]).fillna(False)].copy()
    legacy["review_id_key"] = id_key(legacy["review_id"])
    if legacy["review_id_key"].duplicated().any():
        raise ValueError("Successful legacy labels contain duplicate review ids")

    compact = pd.read_csv(COMPACT, low_memory=False)
    compact = compact.loc[bool_series(compact["success"]).fillna(False)].copy()
    if len(compact) != EXPECTED_COMPACT or compact["identity"].duplicated().any():
        raise ValueError("Compact result population or identity uniqueness changed")

    local = pd.read_csv(LOCAL, low_memory=False)
    if len(local) != EXPECTED_ELIGIBLE or local["identity"].duplicated().any():
        raise ValueError("Uniform local feature population or identity uniqueness changed")
    return corpus, legacy, compact, local


def read_manual_qa(compact: pd.DataFrame, corpus: pd.DataFrame) -> tuple[dict[str, dict[str, int]], pd.DataFrame]:
    audit: dict[str, Any] = json.loads(MANUAL_QA.read_text(encoding="utf-8"))
    accepted = audit["accepted_all_unmentioned"]
    overrides = audit["overrides"]
    audited_ids = set(accepted) | set(overrides)

    mention_cols = [f"{aspect}_mentioned" for aspect in ASPECTS]
    all_null_ids = set(compact.loc[compact[mention_cols].eq(False).all(axis=1), "identity"].astype(str))
    if len(all_null_ids) != EXPECTED_ALL_NULL or audited_ids != all_null_ids:
        raise ValueError("Manual QA scope does not exactly equal the compact all-unmentioned population")
    if set(accepted) & set(overrides):
        raise ValueError("Manual QA identities cannot be both accepted and overridden")

    normalized: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    lookup = corpus.set_index("identity")
    for identity in sorted(all_null_ids):
        if identity in overrides:
            labels = overrides[identity]["labels"]
            invalid_aspects = set(labels) - set(ASPECTS)
            invalid_values = [value for value in labels.values() if isinstance(value, bool) or value not in (-1, 0, 1)]
            if invalid_aspects or invalid_values or not labels:
                raise ValueError(f"Invalid manual labels for {identity}")
            normalized[identity] = {str(key): int(value) for key, value in labels.items()}
            action = "override_false_negative"
            reason = overrides[identity]["reason"]
        else:
            labels = {}
            action = "accept_all_unmentioned"
            reason = accepted[identity]
        source = lookup.loc[identity]
        rows.append({
            "identity": identity,
            "series_name": source["series_name"],
            "publish_time": source["publish_time"],
            "corpus_source": source["corpus_source"],
            "content": source["content"],
            "qa_action": action,
            "override_labels": json.dumps(labels, ensure_ascii=False, sort_keys=True),
            "qa_reason": reason,
            "reviewer": audit["reviewer"],
            "reviewed_at": audit["reviewed_at"],
        })
    qa = pd.DataFrame(rows).sort_values(["qa_action", "series_name", "identity"])
    return normalized, qa


def build_unified() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    corpus, legacy, compact, local = read_inputs()
    legacy_ids = set(legacy["review_id_key"])
    corpus["legacy_available"] = corpus["corpus_source"].eq("old_v1") & corpus["review_id_key"].isin(legacy_ids)
    if int(corpus["legacy_available"].sum()) != EXPECTED_LEGACY:
        raise ValueError("Reusable legacy count changed")

    legacy_columns = ["review_id_key", *ASPECTS]
    legacy_labels = legacy[legacy_columns].rename(columns={aspect: f"legacy_{aspect}" for aspect in ASPECTS})
    compact_columns = [
        "identity", "prompt_version", "requested_model", "scored_at",
        *[column for aspect in ASPECTS for column in (
            f"{aspect}_mentioned", f"{aspect}_polarity", f"legacy_compatible_{aspect}",
        )],
    ]
    compact_labels = compact[compact_columns].copy()

    base_columns = [
        "identity", "review_id", "review_id_key", "series_name", "publish_time",
        "corpus_source", "platform", "content_source", "content", "content_len",
        "rating_overall", "legacy_available",
    ]
    unified = corpus[base_columns].merge(legacy_labels, on="review_id_key", how="left", validate="many_to_one")
    unified = unified.merge(compact_labels, on="identity", how="left", validate="one_to_one")
    local_columns = [
        "identity", "text_global_polarity",
        *[column for aspect in ASPECTS for column in (
            f"text_{aspect}_mentioned", f"text_{aspect}_polarity",
        )],
    ]
    unified = unified.merge(local[local_columns], on="identity", how="left", validate="one_to_one")

    compact_mask = ~unified["legacy_available"]
    if unified.loc[compact_mask, "requested_model"].isna().any():
        raise ValueError("An eligible non-legacy review lacks compact labels")
    if unified.loc[unified["legacy_available"], "legacy_appearance"].isna().any():
        raise ValueError("An eligible legacy review lacks legacy labels")

    overrides, qa = read_manual_qa(compact, corpus)
    override_ids = set(overrides)
    accepted_ids = set(qa.loc[qa["qa_action"].eq("accept_all_unmentioned"), "identity"])
    unified["label_source"] = unified["legacy_available"].map({
        True: "legacy_deepseek_chat_2025",
        False: "compact_deepseek_v4_flash_2026",
    })
    unified["manual_qa_status"] = "not_required"
    unified.loc[unified["identity"].isin(accepted_ids), "manual_qa_status"] = "accepted_all_unmentioned"
    unified.loc[unified["identity"].isin(override_ids), "manual_qa_status"] = "overridden_false_negative"

    for aspect in ASPECTS:
        legacy_value = pd.to_numeric(unified[f"legacy_{aspect}"], errors="coerce")
        compact_raw = pd.to_numeric(unified[f"{aspect}_polarity"], errors="coerce")
        compact_compatible = pd.to_numeric(unified[f"legacy_compatible_{aspect}"], errors="coerce")
        raw = legacy_value.where(unified["legacy_available"], compact_raw)
        compatible = legacy_value.where(unified["legacy_available"], compact_compatible)
        api_mentioned = bool_series(unified[f"{aspect}_mentioned"]).where(compact_mask)

        for identity, labels in overrides.items():
            if aspect not in labels:
                continue
            row_mask = unified["identity"].eq(identity)
            raw.loc[row_mask] = labels[aspect]
            compatible.loc[row_mask] = labels[aspect]
            api_mentioned.loc[row_mask] = True

        if compatible.isna().any() or not compatible.isin([-1, 0, 1]).all():
            raise ValueError(f"Unified compatible labels are invalid for {aspect}")
        unified[f"deepseek_{aspect}_raw_polarity"] = raw
        unified[f"deepseek_{aspect}_legacy_compatible"] = compatible.astype(int)
        unified[f"deepseek_{aspect}_api_mentioned"] = api_mentioned.astype("boolean")
        unified[f"uniform_local_{aspect}_mentioned"] = pd.to_numeric(
            unified[f"text_{aspect}_mentioned"], errors="raise"
        ).astype(int)

    drop_columns = [
        *[f"legacy_{aspect}" for aspect in ASPECTS],
        *[column for aspect in ASPECTS for column in (
            f"{aspect}_mentioned", f"{aspect}_polarity", f"legacy_compatible_{aspect}",
            f"text_{aspect}_mentioned", f"text_{aspect}_polarity",
        )],
    ]
    unified = unified.drop(columns=drop_columns)
    unified["content_sha256"] = unified["content"].astype(str).map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    unified = unified.drop(columns=["content"])
    unified = unified.sort_values(["series_name", "publish_time", "identity"]).reset_index(drop=True)
    if len(unified) != EXPECTED_ELIGIBLE or unified["identity"].duplicated().any():
        raise ValueError("Unified output lost or duplicated eligible reviews")

    summary = {
        "schema_version": "v1",
        "eligible_reviews": int(len(unified)),
        "eligible_series": int(unified["series_name"].nunique()),
        "reused_legacy_reviews": int(unified["legacy_available"].sum()),
        "compact_v4_flash_reviews": int((~unified["legacy_available"]).sum()),
        "manual_all_null_reviews_audited": int(len(qa)),
        "manual_all_null_accepted": int(qa["qa_action"].eq("accept_all_unmentioned").sum()),
        "manual_false_negative_reviews_overridden": int(qa["qa_action"].eq("override_false_negative").sum()),
        "manual_aspect_labels_overridden": int(sum(len(labels) for labels in overrides.values())),
        "unified_label_rule": "-1 negative, 0 legacy-compatible neutral/unmentioned/fallback, 1 positive",
        "legacy_zero_limitation": "Legacy zero cannot distinguish unmentioned, neutral, and parser fallback; do not use it as mention ground truth.",
        "uniform_mention_rule": "uniform_local_*_mentioned is generated by the same deterministic detector for all 24,175 reviews.",
        "external_api_calls": 0,
    }
    return unified, qa, summary


def main() -> None:
    unified, qa, summary = build_unified()
    unified.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    qa.to_csv(QA_OUTPUT, index=False, encoding="utf-8-sig")
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[output] {OUTPUT.relative_to(BASE)}")
    print(f"[output] {QA_OUTPUT.relative_to(BASE)}")
    print(f"[output] {SUMMARY.relative_to(BASE)}")


if __name__ == "__main__":
    main()

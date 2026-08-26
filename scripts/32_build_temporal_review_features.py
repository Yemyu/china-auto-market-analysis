#!/usr/bin/env python3
"""Aggregate review labels into leakage-safe monthly features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal


BASE = Path(__file__).resolve().parents[1]
SPLITS = BASE / "data" / "processed" / "splits"
OUT = BASE / "data" / "reviews" / "processed"
REVIEWS = OUT / "review_aspect_labels.csv"
AVAILABILITY = OUT / "review_temporal_availability_by_series.csv"

FIXED_OUTPUT = OUT / "review_features_by_series_month_fixed_origin.csv"
ROLLING_OUTPUT = OUT / "review_features_by_series_month_rolling.csv"
AUDIT_OUTPUT = OUT / "review_feature_temporal_audit.csv"
SUMMARY_OUTPUT = OUT / "review_feature_temporal_summary.json"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
LOOKBACK_DAYS = 180
VALIDATION_ORIGIN = pd.Timestamp("2025-07-01")
TEST_ORIGIN = pd.Timestamp("2026-01-01")
EXPECTED_SERIES = 371
EXPECTED_REVIEWS = 24_175

KEY_COLUMNS = [
    "series_name", "date", "split", "feature_protocol",
    "information_cutoff_exclusive",
]


def load_panel() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = SPLITS / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["series_name", "date"])
        frame["series_name"] = frame["series_name"].astype(str)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
        frame["split"] = split
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["series_name", "date"]).any():
        raise ValueError("Forecast panel contains duplicate series/month rows")
    if panel["series_name"].nunique() != EXPECTED_SERIES:
        raise ValueError("Forecast panel no longer contains exactly 371 series")
    return panel.sort_values(["date", "series_name"]).reset_index(drop=True)


def load_reviews() -> pd.DataFrame:
    reviews = pd.read_csv(REVIEWS, low_memory=False)
    reviews["series_name"] = reviews["series_name"].astype(str)
    reviews["publish_time"] = pd.to_datetime(reviews["publish_time"], errors="raise")
    if len(reviews) != EXPECTED_REVIEWS or reviews["identity"].duplicated().any():
        raise ValueError("Review-label population changed")
    score_columns = [f"review_{aspect}_score" for aspect in ASPECTS]
    mention_columns = [f"uniform_local_{aspect}_mentioned" for aspect in ASPECTS]
    for column in score_columns:
        if not reviews[column].isin([-1, 0, 1]).all():
            raise ValueError(f"Invalid review score column: {column}")
    for column in mention_columns:
        if not reviews[column].isin([0, 1]).all():
            raise ValueError(f"Invalid uniform mention column: {column}")
    reviews["review_overall_aspect_score"] = reviews[score_columns].mean(axis=1)
    reviews["review_any_positive"] = reviews[score_columns].eq(1).any(axis=1).astype(int)
    reviews["review_any_negative"] = reviews[score_columns].eq(-1).any(axis=1).astype(int)
    return reviews.sort_values(["publish_time", "identity"]).reset_index(drop=True)


def aggregate_at_cutoff(
    reviews: pd.DataFrame,
    cutoff: pd.Timestamp,
    universe: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prior = reviews.loc[reviews["publish_time"].lt(cutoff)].copy()
    recent_start = cutoff - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = prior.loc[prior["publish_time"].ge(recent_start)].copy()
    base = pd.DataFrame(index=pd.Index(universe, name="series_name"))

    all_counts = prior.groupby("series_name").size()
    recent_counts = recent.groupby("series_name").size()
    base["review_count_prior_all"] = all_counts
    base["review_count_180d"] = recent_counts
    base[["review_count_prior_all", "review_count_180d"]] = (
        base[["review_count_prior_all", "review_count_180d"]].fillna(0).astype(int)
    )
    base["review_available_prior"] = base["review_count_prior_all"].gt(0).astype(int)
    base["review_available_180d"] = base["review_count_180d"].gt(0).astype(int)

    score_columns = [f"review_{aspect}_score" for aspect in ASPECTS]
    prior_means = prior.groupby("series_name")[score_columns].mean().rename(
        columns={column: column.replace("_score", "_score_prior_mean") for column in score_columns}
    )
    recent_means = recent.groupby("series_name")[score_columns].mean().rename(
        columns={column: column.replace("_score", "_score_180d_mean") for column in score_columns}
    )
    base = base.join(prior_means).join(recent_means)

    if not recent.empty:
        positive = recent[score_columns].eq(1).astype(float)
        negative = recent[score_columns].eq(-1).astype(float)
        positive["series_name"] = recent["series_name"].values
        negative["series_name"] = recent["series_name"].values
        positive_rates = positive.groupby("series_name")[score_columns].mean().rename(
            columns={column: column.replace("_score", "_positive_180d_rate") for column in score_columns}
        )
        negative_rates = negative.groupby("series_name")[score_columns].mean().rename(
            columns={column: column.replace("_score", "_negative_180d_rate") for column in score_columns}
        )
        base = base.join(positive_rates).join(negative_rates)
    else:
        for aspect in ASPECTS:
            base[f"review_{aspect}_positive_180d_rate"] = pd.NA
            base[f"review_{aspect}_negative_180d_rate"] = pd.NA

    mention_columns = [f"uniform_local_{aspect}_mentioned" for aspect in ASPECTS]
    mention_counts = recent.groupby("series_name")[mention_columns].sum().rename(
        columns={column: column.replace("uniform_local_", "review_").replace("_mentioned", "_uniform_mention_180d_count")
                 for column in mention_columns}
    )
    mention_rates = recent.groupby("series_name")[mention_columns].mean().rename(
        columns={column: column.replace("uniform_local_", "review_").replace("_mentioned", "_uniform_mention_180d_rate")
                 for column in mention_columns}
    )
    base = base.join(mention_counts).join(mention_rates)
    count_columns = [f"review_{aspect}_uniform_mention_180d_count" for aspect in ASPECTS]
    base[count_columns] = base[count_columns].fillna(0).astype(int)

    overall_columns = [
        "review_overall_aspect_score", "review_any_positive", "review_any_negative",
    ]
    overall = recent.groupby("series_name")[overall_columns].mean().rename(columns={
        "review_overall_aspect_score": "review_overall_aspect_score_180d_mean",
        "review_any_positive": "review_any_positive_180d_rate",
        "review_any_negative": "review_any_negative_180d_rate",
    })
    base = base.join(overall)

    max_used = prior["publish_time"].max() if not prior.empty else pd.NaT
    if pd.notna(max_used) and not max_used < cutoff:
        raise ValueError(f"Review leakage at cutoff {cutoff}: max used {max_used}")
    audit = {
        "information_cutoff_exclusive": cutoff.strftime("%Y-%m-%d"),
        "lookback_start_inclusive": recent_start.strftime("%Y-%m-%d"),
        "max_review_time_used": "" if pd.isna(max_used) else max_used.isoformat(),
        "all_prior_review_rows": int(len(prior)),
        "recent_180d_review_rows": int(len(recent)),
        "series_with_any_prior_review": int(base["review_available_prior"].sum()),
        "series_with_recent_180d_review": int(base["review_available_180d"].sum()),
        "recent_historical_label_rows": int(recent["label_source"].eq("historical_labels_2025").sum()),
        "recent_project_label_rows": int(recent["label_source"].eq("project_labels_2026").sum()),
        "recent_manual_override_rows": int(recent["manual_qa_status"].eq("overridden_false_negative").sum()),
    }
    return base.reset_index(), audit


def cutoff_for(protocol: str, split: str, target_month: pd.Timestamp) -> pd.Timestamp:
    if protocol == "rolling_origin":
        return target_month
    if protocol != "fixed_origin":
        raise ValueError(protocol)
    if split == "train":
        return target_month
    if split == "val":
        return VALIDATION_ORIGIN
    if split == "test":
        return TEST_ORIGIN
    raise ValueError(split)


def build_protocol(
    protocol: str,
    panel: pd.DataFrame,
    reviews: pd.DataFrame,
    cache: dict[pd.Timestamp, tuple[pd.DataFrame, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = sorted(panel["series_name"].unique())
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for target_month in sorted(panel["date"].unique()):
        target_month = pd.Timestamp(target_month)
        month_rows = panel.loc[panel["date"].eq(target_month), ["series_name", "date", "split"]].copy()
        splits = month_rows["split"].unique().tolist()
        if len(splits) != 1:
            raise ValueError(f"A target month belongs to multiple splits: {target_month}")
        split = splits[0]
        cutoff = cutoff_for(protocol, split, target_month)
        if cutoff not in cache:
            cache[cutoff] = aggregate_at_cutoff(reviews, cutoff, universe)
        aggregates, cutoff_audit = cache[cutoff]
        joined = month_rows.merge(aggregates, on="series_name", how="left", validate="one_to_one")
        if len(joined) != len(month_rows):
            raise ValueError("Sentiment aggregation changed panel row count")
        joined["feature_protocol"] = protocol
        joined["information_cutoff_exclusive"] = cutoff.strftime("%Y-%m-%d")
        parts.append(joined)
        audit_rows.append({
            "feature_protocol": protocol,
            "forecast_month": target_month.strftime("%Y-%m"),
            "split": split,
            "series_rows": int(len(joined)),
            **cutoff_audit,
        })
    table = pd.concat(parts, ignore_index=True).sort_values(["date", "series_name"])
    audit = pd.DataFrame(audit_rows).sort_values(["feature_protocol", "forecast_month"])
    return table, audit


def sentiment_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in KEY_COLUMNS]


def assert_frozen(table: pd.DataFrame, split: str, expected_cutoff: pd.Timestamp) -> None:
    part = table.loc[table["split"].eq(split)].copy()
    if not part["information_cutoff_exclusive"].eq(expected_cutoff.strftime("%Y-%m-%d")).all():
        raise ValueError(f"{split} fixed-origin cutoff is not frozen")
    variability = part.groupby("series_name")[sentiment_columns(table)].nunique(dropna=False)
    if variability.gt(1).any().any():
        columns = variability.columns[variability.gt(1).any()].tolist()
        raise ValueError(f"{split} fixed-origin features vary across horizon: {columns[:10]}")


def assert_protocol_relationships(fixed: pd.DataFrame, rolling: pd.DataFrame) -> None:
    feature_columns = sentiment_columns(fixed)
    fixed_train = fixed.loc[fixed["split"].eq("train"), ["series_name", "date", *feature_columns]].reset_index(drop=True)
    rolling_train = rolling.loc[rolling["split"].eq("train"), ["series_name", "date", *feature_columns]].reset_index(drop=True)
    assert_frame_equal(fixed_train, rolling_train, check_dtype=False)

    for start in (VALIDATION_ORIGIN, TEST_ORIGIN):
        fixed_start = fixed.loc[fixed["date"].eq(start), ["series_name", *feature_columns]].reset_index(drop=True)
        rolling_start = rolling.loc[rolling["date"].eq(start), ["series_name", *feature_columns]].reset_index(drop=True)
        assert_frame_equal(fixed_start, rolling_start, check_dtype=False)


def assert_availability(fixed: pd.DataFrame, rolling: pd.DataFrame) -> None:
    if not AVAILABILITY.exists():
        raise FileNotFoundError(AVAILABILITY)
    expected = pd.read_csv(AVAILABILITY)
    expected["series_name"] = expected["series_name"].astype(str)
    test_months = sorted(rolling.loc[rolling["split"].eq("test"), "date"].unique())
    for month_value in test_months:
        month = pd.Timestamp(month_value)
        expected_column = f"reviews_available_before_{month.strftime('%Y_%m')}"
        actual = rolling.loc[rolling["date"].eq(month), ["series_name", "review_count_prior_all"]]
        check = expected[["series_name", expected_column]].merge(actual, on="series_name", validate="one_to_one")
        if len(check) != EXPECTED_SERIES or not check[expected_column].eq(check["review_count_prior_all"]).all():
            raise ValueError(f"Rolling availability mismatch for {month.strftime('%Y-%m')}")

    january_column = "reviews_available_before_2026_01"
    for month_value in test_months:
        month = pd.Timestamp(month_value)
        actual = fixed.loc[fixed["date"].eq(month), ["series_name", "review_count_prior_all"]]
        check = expected[["series_name", january_column]].merge(actual, on="series_name", validate="one_to_one")
        if len(check) != EXPECTED_SERIES or not check[january_column].eq(check["review_count_prior_all"]).all():
            raise ValueError(f"Fixed-origin availability mismatch for {month.strftime('%Y-%m')}")


def main() -> None:
    if not REVIEWS.exists():
        raise FileNotFoundError(f"Run 31_merge_review_labels.py first: {REVIEWS}")
    panel = load_panel()
    reviews = load_reviews()
    cache: dict[pd.Timestamp, tuple[pd.DataFrame, dict[str, Any]]] = {}
    fixed, fixed_audit = build_protocol("fixed_origin", panel, reviews, cache)
    rolling, rolling_audit = build_protocol("rolling_origin", panel, reviews, cache)

    if len(fixed) != len(panel) or len(rolling) != len(panel):
        raise ValueError("Protocol output row count differs from the forecasting panel")
    assert_frozen(fixed, "val", VALIDATION_ORIGIN)
    assert_frozen(fixed, "test", TEST_ORIGIN)
    assert_protocol_relationships(fixed, rolling)
    assert_availability(fixed, rolling)

    fixed.to_csv(FIXED_OUTPUT, index=False, encoding="utf-8-sig")
    rolling.to_csv(ROLLING_OUTPUT, index=False, encoding="utf-8-sig")
    audit = pd.concat([fixed_audit, rolling_audit], ignore_index=True)
    audit.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8-sig")

    fixed_test = fixed.loc[fixed["split"].eq("test")]
    rolling_test = rolling.loc[rolling["split"].eq("test")]
    rolling_test_monthly = rolling_test.groupby("date").agg(
        series_with_any_prior_review=("review_available_prior", "sum"),
        series_with_recent_180d_review=("review_available_180d", "sum"),
    ).reset_index()
    summary = {
        "schema_version": "v1",
        "panel_rows_per_protocol": int(len(panel)),
        "series": int(panel["series_name"].nunique()),
        "forecast_months": int(panel["date"].nunique()),
        "review_rows": int(len(reviews)),
        "lookback_days": LOOKBACK_DAYS,
        "feature_columns": sentiment_columns(fixed),
        "fixed_origin_validation_cutoff_exclusive": VALIDATION_ORIGIN.strftime("%Y-%m-%d"),
        "fixed_origin_test_cutoff_exclusive": TEST_ORIGIN.strftime("%Y-%m-%d"),
        "fixed_test_series_with_any_prior_review": int(fixed_test.groupby("date")["review_available_prior"].sum().iloc[0]),
        "fixed_test_series_with_recent_180d_review": int(fixed_test.groupby("date")["review_available_180d"].sum().iloc[0]),
        "rolling_test_coverage_by_month": [
            {
                "month": row["date"].strftime("%Y-%m"),
                "series_with_any_prior_review": int(row["series_with_any_prior_review"]),
                "series_with_recent_180d_review": int(row["series_with_recent_180d_review"]),
            }
            for _, row in rolling_test_monthly.iterrows()
        ],
        "fixed_origin_freeze_checks": "passed for validation and test horizons",
        "rolling_availability_cross_check": "passed for every test month",
        "fixed_availability_cross_check": "passed against 2026-01 origin for every test month",
        "external_api_calls": 0,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[output] {FIXED_OUTPUT.relative_to(BASE)}")
    print(f"[output] {ROLLING_OUTPUT.relative_to(BASE)}")
    print(f"[output] {AUDIT_OUTPUT.relative_to(BASE)}")
    print(f"[output] {SUMMARY_OUTPUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()

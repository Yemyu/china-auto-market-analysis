#!/usr/bin/env python3
"""Aggregate local sentiment signals at each forecast origin without leakage.

For a target month ``t``, every aggregate uses review timestamps strictly
earlier than the first day of ``t``.  The table keeps count/availability fields
alongside scores so models can distinguish no observed sentiment from a neutral
sentiment estimate based on observed reviews.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "processed_new" / "splits"
OUT = BASE / "data" / "sentiment_new" / "processed"
FEATURES = OUT / "local_sentiment_review_features.csv"
AVAILABILITY = OUT / "review_temporal_availability_by_series.csv"
FEATURE_TABLE = OUT / "sentiment_features_by_series_month.csv"
MONTH_AUDIT = OUT / "sentiment_feature_temporal_audit.csv"
SUMMARY = OUT / "sentiment_feature_temporal_summary.json"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
LOOKBACK_DAYS = 180


def load_panel() -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = SPLITS / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["series_name", "date"])
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.to_period("M").dt.to_timestamp()
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).drop_duplicates(["series_name", "date"])
    if panel.duplicated(["series_name", "date"]).any():
        raise ValueError("Forecast panel has duplicate series/month rows")
    return panel.sort_values(["date", "series_name"]).reset_index(drop=True)


def monthly_features(panel: pd.DataFrame, reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for origin in sorted(panel["date"].unique()):
        origin = pd.Timestamp(origin)
        target = panel.loc[panel["date"].eq(origin), ["series_name", "date"]].copy()
        prior = reviews.loc[reviews["publish_time"].lt(origin)].copy()
        recent = prior.loc[prior["publish_time"].ge(origin - pd.Timedelta(days=LOOKBACK_DAYS))].copy()

        all_counts = prior.groupby("series_name").size().rename("sentiment_review_count_prior_all")
        recent_counts = recent.groupby("series_name").size().rename("sentiment_review_count_180d")
        target = target.join(all_counts, on="series_name").join(recent_counts, on="series_name")
        target[["sentiment_review_count_prior_all", "sentiment_review_count_180d"]] = (
            target[["sentiment_review_count_prior_all", "sentiment_review_count_180d"]].fillna(0).astype(int)
        )
        target["sentiment_available_prior"] = target["sentiment_review_count_prior_all"].gt(0).astype(int)
        target["sentiment_available_180d"] = target["sentiment_review_count_180d"].gt(0).astype(int)

        scalar_columns = [
            "platform_rating_overall_sentiment",
            "platform_rating_overall_polarity",
            "text_global_polarity",
        ]
        for aspect in ASPECTS:
            scalar_columns.extend([
                f"platform_rating_{aspect}_sentiment",
                f"text_{aspect}_polarity",
            ])
        means = recent.groupby("series_name")[scalar_columns].mean().add_suffix("_180d_mean")
        target = target.join(means, on="series_name")

        mention_columns = [f"text_{aspect}_mentioned" for aspect in ASPECTS]
        mentions = recent.groupby("series_name")[mention_columns].sum().rename(
            columns={column: f"{column}_180d_count" for column in mention_columns}
        )
        target = target.join(mentions, on="series_name")
        mention_output = [f"{column}_180d_count" for column in mention_columns]
        target[mention_output] = target[mention_output].fillna(0).astype(int)

        parts.append(target)
        audit_rows.append({
            "forecast_month": origin.strftime("%Y-%m"),
            "information_cutoff_exclusive": str(origin),
            "series_rows": int(len(target)),
            "series_with_any_prior_review": int(target["sentiment_available_prior"].sum()),
            "series_with_recent_180d_review": int(target["sentiment_available_180d"].sum()),
            "all_prior_review_rows": int(target["sentiment_review_count_prior_all"].sum()),
            "recent_180d_review_rows": int(target["sentiment_review_count_180d"].sum()),
        })
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit_rows)


def assert_test_availability(feature_table: pd.DataFrame) -> None:
    if not AVAILABILITY.exists():
        raise FileNotFoundError(f"Run 25_build_review_temporal_availability.py first: {AVAILABILITY}")
    expected = pd.read_csv(AVAILABILITY)
    test = pd.read_csv(SPLITS / "test.csv", usecols=["date"])
    months = sorted(pd.to_datetime(test["date"], errors="raise").dt.to_period("M").unique())
    for period in months:
        month = period.to_timestamp()
        column = f"reviews_available_before_{period.strftime('%Y_%m')}"
        actual = feature_table.loc[feature_table["date"].eq(month), ["series_name", "sentiment_review_count_prior_all"]]
        check = expected[["series_name", column]].merge(actual, on="series_name", how="inner", validate="one_to_one")
        if len(check) != len(expected) or not check[column].eq(check["sentiment_review_count_prior_all"]).all():
            mismatches = check.loc[~check[column].eq(check["sentiment_review_count_prior_all"]), "series_name"].tolist()
            raise ValueError(f"Temporal availability mismatch for {period}: {mismatches[:10]}")


def main() -> None:
    if not FEATURES.exists():
        raise FileNotFoundError(f"Run 27_build_local_sentiment_features.py first: {FEATURES}")
    panel = load_panel()
    reviews = pd.read_csv(FEATURES, low_memory=False)
    reviews["publish_time"] = pd.to_datetime(reviews["publish_time"], errors="raise")
    if reviews.duplicated("identity").any():
        raise ValueError("Review sentiment features have duplicate platform review identities")
    if reviews["publish_time"].isna().any():
        raise ValueError("Review sentiment features have invalid publish timestamps")

    feature_table, audit = monthly_features(panel, reviews)
    assert_test_availability(feature_table)
    feature_table = feature_table.sort_values(["date", "series_name"])
    feature_table.to_csv(FEATURE_TABLE, index=False, encoding="utf-8-sig")
    audit.to_csv(MONTH_AUDIT, index=False, encoding="utf-8-sig")
    summary = {
        "panel_rows": int(len(feature_table)),
        "series": int(feature_table["series_name"].nunique()),
        "forecast_months": int(feature_table["date"].nunique()),
        "lookback_days": LOOKBACK_DAYS,
        "test_availability_cross_check": "passed",
        "rule": "Every score and count for forecast month t uses only reviews published strictly before the first day of t.",
        "missingness_rule": "Score columns remain null when no relevant recent review/rating exists; availability and count fields are explicit numeric features.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

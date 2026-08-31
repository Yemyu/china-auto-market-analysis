"""Build the latest leakage-safe user-needs mart snapshot."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
WINDOW_DAYS = 180
MIN_REVIEWS_PER_WINDOW = 5
ALERT_SCORE_THRESHOLD = -0.10
ALERT_DROP_THRESHOLD = -0.15
ALERT_NEGATIVE_RATE_THRESHOLD = 0.35
TEXT_RETRIGGER_PROBABILITY_THRESHOLD = 0.70
RATING_DECLINE_PROBABILITY_THRESHOLD = 0.80
BOOTSTRAP_DRAWS = 3_000


def _bootstrap_support(
    series_name: str,
    cutoff: pd.Timestamp,
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> tuple[float, float]:
    seed_material = f"{series_name}|{cutoff:%Y-%m-%d}|v3".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    current_text = current["monitor_overall_score"].to_numpy(dtype=float)
    previous_text = previous["monitor_overall_score"].to_numpy(dtype=float)
    current_negative = current["monitor_negative_review"].to_numpy(dtype=float)
    current_rating = current["platform_rating_overall"].dropna().to_numpy(dtype=float)
    previous_rating = previous["platform_rating_overall"].dropna().to_numpy(dtype=float)
    current_text_draws = rng.choice(
        current_text, size=(BOOTSTRAP_DRAWS, len(current_text)), replace=True
    ).mean(axis=1)
    previous_text_draws = rng.choice(
        previous_text, size=(BOOTSTRAP_DRAWS, len(previous_text)), replace=True
    ).mean(axis=1)
    negative_draws = rng.choice(
        current_negative, size=(BOOTSTRAP_DRAWS, len(current_negative)), replace=True
    ).mean(axis=1)
    text_support = (
        (current_text_draws <= ALERT_SCORE_THRESHOLD)
        & (current_text_draws - previous_text_draws <= ALERT_DROP_THRESHOLD)
        & (negative_draws >= ALERT_NEGATIVE_RATE_THRESHOLD)
    )
    if not len(current_rating) or not len(previous_rating):
        return float(text_support.mean()), np.nan
    rating_change = rng.choice(
        current_rating, size=(BOOTSTRAP_DRAWS, len(current_rating)), replace=True
    ).mean(axis=1) - rng.choice(
        previous_rating, size=(BOOTSTRAP_DRAWS, len(previous_rating)), replace=True
    ).mean(axis=1)
    return float(text_support.mean()), float((rating_change < 0).mean())


def _prepare(labels: pd.DataFrame) -> pd.DataFrame:
    data = labels.copy()
    data["series_name"] = data["series_name"].astype(str)
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="raise")
    score_columns: list[str] = []
    for aspect in ASPECTS:
        mentioned = data[f"uniform_local_{aspect}_mentioned"].eq(1)
        raw = pd.to_numeric(data[f"review_{aspect}_raw_polarity"], errors="coerce")
        column = f"monitor_{aspect}_score"
        data[column] = raw.where(mentioned & raw.isin([-1, 0, 1]))
        score_columns.append(column)
    data["monitor_overall_score"] = data[score_columns].mean(axis=1, skipna=True)
    data["monitor_negative_review"] = data["monitor_overall_score"].lt(0).where(
        data["monitor_overall_score"].notna()
    )
    data["platform_rating_overall"] = pd.to_numeric(data["rating_overall"], errors="coerce")
    return data


def build_latest_snapshot(
    labels: pd.DataFrame,
    *,
    cutoff_inclusive: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aspect rows and series-level parity statistics for one completed month."""
    data = _prepare(labels)
    cutoff = pd.Timestamp("2026-07-31" if cutoff_inclusive is None else cutoff_inclusive)
    cutoff_exclusive = cutoff + pd.Timedelta(days=1)
    current_start = cutoff - pd.Timedelta(days=WINDOW_DAYS - 1)
    previous_start = current_start - pd.Timedelta(days=WINDOW_DAYS)
    aspect_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    for series_name, group in data.groupby("series_name", sort=True):
        group = group.loc[group["monitor_overall_score"].notna()].copy()
        current = group.loc[
            group["publish_time"].ge(current_start)
            & group["publish_time"].lt(cutoff_exclusive)
        ]
        previous = group.loc[
            group["publish_time"].ge(previous_start)
            & group["publish_time"].lt(current_start)
        ]
        if current.empty and previous.empty:
            continue
        current_score = current["monitor_overall_score"].mean() if len(current) else np.nan
        previous_score = previous["monitor_overall_score"].mean() if len(previous) else np.nan
        score_change = current_score - previous_score if len(current) and len(previous) else np.nan
        negative_rate = current["monitor_negative_review"].mean() if len(current) else np.nan
        eligible = len(current) >= MIN_REVIEWS_PER_WINDOW and len(previous) >= MIN_REVIEWS_PER_WINDOW
        candidate = bool(
            eligible
            and current_score <= ALERT_SCORE_THRESHOLD
            and score_change <= ALERT_DROP_THRESHOLD
            and negative_rate >= ALERT_NEGATIVE_RATE_THRESHOLD
        )
        text_support = np.nan
        rating_support = np.nan
        if candidate:
            text_support, rating_support = _bootstrap_support(
                series_name, cutoff, current, previous
            )
        corroborated = bool(
            candidate
            and text_support >= TEXT_RETRIGGER_PROBABILITY_THRESHOLD
            and rating_support >= RATING_DECLINE_PROBABILITY_THRESHOLD
        )
        alert_status = "corroborated" if corroborated else "watchlist" if candidate else "none"
        risk_level = "alert" if corroborated else "watch" if candidate else "normal"
        status_rows.append(
            {
                "series_name": series_name,
                "information_cutoff_inclusive": cutoff,
                "current_reviews": len(current),
                "previous_reviews": len(previous),
                "current_overall_score": current_score,
                "previous_overall_score": previous_score,
                "score_change": score_change,
                "current_negative_review_rate": negative_rate,
                "eligible_for_alert": eligible,
                "text_alert_candidate": candidate,
                "text_rule_retrigger_probability": text_support,
                "rating_decline_probability": rating_support,
                "alert_status": alert_status,
                "alert": corroborated,
            }
        )
        for aspect in ASPECTS:
            mentioned = current[f"uniform_local_{aspect}_mentioned"].eq(1)
            scores = current.loc[mentioned, f"monitor_{aspect}_score"].dropna()
            aspect_rows.append(
                {
                    "series_name": series_name,
                    "monitoring_month": cutoff,
                    "aspect_code": aspect,
                    "review_count_180d": len(current),
                    "mention_count_180d": int(mentioned.sum()),
                    "positive_rate_180d": float(scores.eq(1).mean()) if len(scores) else None,
                    "negative_rate_180d": float(scores.eq(-1).mean()) if len(scores) else None,
                    "risk_level": risk_level,
                    "information_cutoff_exclusive": cutoff_exclusive,
                }
            )
    aspects = pd.DataFrame(aspect_rows).sort_values(["series_name", "aspect_code"])
    statuses = pd.DataFrame(status_rows).sort_values("series_name")
    if len(aspects) != 3_000 or statuses["series_name"].nunique() != 300:
        raise ValueError("Latest user-needs population changed")
    return aspects.reset_index(drop=True), statuses.reset_index(drop=True)


def assert_latest_window_parity(statuses: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Require exact series/status parity and numerical parity with the frozen window."""
    cutoff = pd.Timestamp("2026-07-31")
    expected = frozen.loc[
        pd.to_datetime(frozen["information_cutoff_inclusive"]).eq(cutoff)
    ].copy()
    columns = [
        "series_name", "information_cutoff_inclusive", "current_reviews", "previous_reviews",
        "current_overall_score", "previous_overall_score", "score_change",
        "current_negative_review_rate", "eligible_for_alert", "text_alert_candidate",
        "text_rule_retrigger_probability", "rating_decline_probability", "alert_status", "alert",
    ]
    expected["information_cutoff_inclusive"] = pd.to_datetime(
        expected["information_cutoff_inclusive"]
    )
    assert_frame_equal(
        statuses[columns].sort_values("series_name").reset_index(drop=True),
        expected[columns].sort_values("series_name").reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

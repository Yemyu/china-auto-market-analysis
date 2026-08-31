"""Point-in-time rules shared by review feature pipelines."""

from __future__ import annotations

import pandas as pd


VALIDATION_ORIGIN = pd.Timestamp("2025-07-01")
TEST_ORIGIN = pd.Timestamp("2026-01-01")


def reviews_before_cutoff(
    reviews: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    *,
    time_column: str = "publish_time",
) -> pd.DataFrame:
    """Return only reviews published strictly before an information cutoff."""
    if time_column not in reviews.columns:
        raise KeyError(f"Review frame is missing time column: {time_column}")
    cutoff = pd.Timestamp(cutoff)
    timestamps = pd.to_datetime(reviews[time_column], errors="raise")
    prior = reviews.loc[timestamps.lt(cutoff)].copy()
    prior[time_column] = timestamps.loc[prior.index]
    if not prior.empty and not prior[time_column].max() < cutoff:
        raise ValueError(f"Review leakage at cutoff {cutoff}")
    return prior


def feature_cutoff(protocol: str, split: str, target_month: pd.Timestamp | str) -> pd.Timestamp:
    """Return the locked information cutoff for a monthly review feature row."""
    target_month = pd.Timestamp(target_month)
    if protocol == "rolling_origin":
        return target_month
    if protocol != "fixed_origin":
        raise ValueError(f"Unknown feature protocol: {protocol}")
    if split == "train":
        return target_month
    if split == "val":
        return VALIDATION_ORIGIN
    if split == "test":
        return TEST_ORIGIN
    raise ValueError(f"Unknown data split: {split}")

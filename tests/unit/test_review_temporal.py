from __future__ import annotations

import pandas as pd
import pytest

from china_auto_market.reviews.temporal import feature_cutoff, reviews_before_cutoff


def test_reviews_at_or_after_forecast_cutoff_are_excluded() -> None:
    reviews = pd.DataFrame(
        {
            "identity": ["before", "at_cutoff", "future"],
            "publish_time": pd.to_datetime(
                ["2025-12-31 23:59:59", "2026-01-01 00:00:00", "2026-01-02 00:00:00"]
            ),
        }
    )

    available = reviews_before_cutoff(reviews, "2026-01-01")

    assert available["identity"].tolist() == ["before"]


@pytest.mark.parametrize(
    ("protocol", "split", "target_month", "expected"),
    [
        ("rolling_origin", "test", "2026-03-01", "2026-03-01"),
        ("fixed_origin", "train", "2025-04-01", "2025-04-01"),
        ("fixed_origin", "val", "2025-11-01", "2025-07-01"),
        ("fixed_origin", "test", "2026-04-01", "2026-01-01"),
    ],
)
def test_review_feature_cutoffs_follow_locked_protocol(
    protocol: str,
    split: str,
    target_month: str,
    expected: str,
) -> None:
    assert feature_cutoff(protocol, split, target_month) == pd.Timestamp(expected)

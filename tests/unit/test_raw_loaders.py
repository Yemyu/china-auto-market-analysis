from __future__ import annotations

import pandas as pd

from china_auto_market.ingestion.raw_loaders import _dataset_name, _filter_month


def test_month_partition_filters_each_source_contract() -> None:
    sales = pd.DataFrame({"year": [2026, 2026], "month": [1, 2]})
    config = pd.DataFrame({"year": [2025, 2026, 2026]})
    reviews = pd.DataFrame({"publish_time": ["2026-01-31", "2026-02-01"]})
    corrections = pd.DataFrame({"date": ["2026-01-01", "2026-02-01"]})
    labels = pd.DataFrame({"publish_time": ["2026-01-31", "2026-02-01"]})

    assert len(_filter_month(sales, "sales", "2026-01")) == 1
    assert len(_filter_month(config, "config", "2026-01")) == 2
    assert len(_filter_month(reviews, "reviews", "2026-01")) == 1
    assert len(_filter_month(corrections, "sales_corrections", "2026-01")) == 1
    assert len(_filter_month(labels, "review_labels", "2026-01")) == 1


def test_partitioned_dataset_name_separates_full_and_monthly_batches() -> None:
    assert _dataset_name("sales", None) == "sales"
    assert _dataset_name("sales", "2026-01") == "sales:2026-01"
    assert _dataset_name("config", "2026-01") == "config:2026"
    assert _dataset_name("config", "2026-02") == "config:2026"

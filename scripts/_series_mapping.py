#!/usr/bin/env python3
"""Conservative cross-source vehicle-series name matching helpers."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


_IGNORABLE = re.compile(r"[\s\-_.·•/\\（）()]+")


def normalize_series_name(value: object) -> str:
    """Normalize harmless typography while preserving meaningful symbols such as +."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return _IGNORABLE.sub("", text)


def build_series_name_mapping(
    sales_names: pd.Series | list[str],
    config_names: pd.Series | list[str],
) -> pd.DataFrame:
    """Return exact and unambiguous normalized matches; never use fuzzy matching."""
    sales = pd.DataFrame({"sales_series_name": pd.Series(sales_names, dtype=str).drop_duplicates()})
    config = pd.DataFrame({"config_series_name": pd.Series(config_names, dtype=str).drop_duplicates()})

    exact_names = sorted(set(sales["sales_series_name"]) & set(config["config_series_name"]))
    exact = pd.DataFrame(
        {
            "sales_series_name": exact_names,
            "config_series_name": exact_names,
            "match_method": "exact_name",
            "normalized_key": [normalize_series_name(name) for name in exact_names],
        }
    )

    sales_left = sales.loc[~sales["sales_series_name"].isin(exact_names)].copy()
    config_left = config.loc[~config["config_series_name"].isin(exact_names)].copy()
    sales_left["normalized_key"] = sales_left["sales_series_name"].map(normalize_series_name)
    config_left["normalized_key"] = config_left["config_series_name"].map(normalize_series_name)

    sales_counts = sales_left["normalized_key"].value_counts()
    config_counts = config_left["normalized_key"].value_counts()
    safe_keys = set(sales_counts[sales_counts.eq(1)].index) & set(
        config_counts[config_counts.eq(1)].index
    )
    normalized = sales_left.loc[sales_left["normalized_key"].isin(safe_keys)].merge(
        config_left.loc[config_left["normalized_key"].isin(safe_keys)],
        on="normalized_key",
        how="inner",
        validate="one_to_one",
    )
    normalized["match_method"] = "unique_normalized_name"

    out = pd.concat(
        [
            exact[["sales_series_name", "config_series_name", "match_method", "normalized_key"]],
            normalized[["sales_series_name", "config_series_name", "match_method", "normalized_key"]],
        ],
        ignore_index=True,
    )
    if out["sales_series_name"].duplicated().any() or out["config_series_name"].duplicated().any():
        raise ValueError("Conservative series mapping is not one-to-one")
    return out.sort_values(["match_method", "sales_series_name"]).reset_index(drop=True)

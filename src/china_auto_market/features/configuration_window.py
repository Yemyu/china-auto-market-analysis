"""Rebuild forecast configurations for one fitting window.

Model years are not publication timestamps. This path retains the source's
year-level assumption, caps records at the fitting origin, and fits all
imputation/encoding on training rows only. It does not claim release-date parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from china_auto_market.features.configuration import CFG_CAT, CFG_COLS, CFG_NUM
from china_auto_market.quality.series_mapping import build_series_name_mapping


def prepare_configuration_window(
    panel: pd.DataFrame,
    source: pd.DataFrame,
    origin: pd.Timestamp,
    model_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Replace legacy configuration columns without changing sales or row order.

    Predictions use a frozen source snapshot. The same source is joined to
    training rows by their own year, so an older row cannot borrow a newer
    specification. Training-row medians are intentionally month-weighted.
    Missing numeric columns default to zero only when every fitting row is
    missing; unknown and missing categories both use -1.
    """
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin != origin.to_period("M").to_timestamp():
        raise ValueError("Configuration origin must be a valid month start")
    out = panel.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["date"].isna().any() or out.duplicated(["series_name", "date"]).any():
        raise ValueError("Forecast rows require unique series/month keys and valid dates")
    if out["series_name"].isna().any():
        raise ValueError("Forecast series names must be present")
    out["series_name"] = out["series_name"].astype(str)
    # Never reuse the medians or numeric category codes stored in the old mart.
    out = out.drop(columns=[c for c in CFG_COLS if c in out])
    raw = source[["series_name", "year", *CFG_NUM, *CFG_CAT]].copy()
    raw["year"] = pd.to_numeric(raw["year"], errors="raise")
    if raw["year"].isna().any() or not np.isfinite(raw["year"]).all():
        raise ValueError("Configuration model years must be finite")
    if raw["year"].mod(1).ne(0).any() or raw["series_name"].isna().any():
        raise ValueError("Configuration requires integral model years and series names")
    ceiling = (origin - pd.Timedelta(days=1)).year
    raw = raw.loc[raw["year"].le(ceiling)].copy()
    raw["series_name"] = raw["series_name"].astype(str)
    if raw.duplicated(["series_name", "year"]).any():
        raise ValueError("Available configurations contain duplicate series/year keys")

    # Build matching from the available snapshot, not future names or aliases.
    mapping = build_series_name_mapping(out["series_name"], raw["series_name"])
    names = mapping.set_index("sales_series_name")["config_series_name"].to_dict()
    joined = pd.DataFrame(index=out.index)
    for column in CFG_NUM:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        joined[column] = np.nan
    for column in CFG_CAT:
        text = raw[column].astype("string").str.strip()
        raw[column] = text.mask(text.str.lower().isin(["", "nan", "none", "null", "<na>"]))
        joined[column] = pd.Series(pd.NA, index=out.index, dtype="string")
    matched_year = pd.Series(np.nan, index=out.index)
    by_name = {name: group.sort_values("year") for name, group in raw.groupby("series_name")}
    for name, group in out.groupby("series_name", sort=False):
        config = by_name.get(names.get(name))
        if config is None:
            continue
        indices = np.searchsorted(config["year"].to_numpy(), group["date"].dt.year, side="right") - 1
        valid = indices >= 0
        rows = group.index[valid]
        selected = config.iloc[indices[valid]]
        matched_year.loc[rows] = selected["year"].to_numpy()
        for column in [*CFG_NUM, *CFG_CAT]:
            joined.loc[rows, column] = selected[column].to_numpy()

    other_columns = [c for c in model_columns if c not in CFG_COLS]
    fit_mask = out["date"].lt(origin) & out[other_columns].notna().all(axis=1)
    if not fit_mask.any():
        raise ValueError("No usable training rows before configuration origin")
    medians, categories, all_missing = {}, {}, []
    for column in CFG_NUM:
        training = joined.loc[fit_mask, column].dropna()
        median = float(training.median()) if len(training) else 0.0
        if training.empty:
            all_missing.append(column)
        medians[column] = median
        out[column] = joined[column].fillna(median).astype(float)
    for column in CFG_CAT:
        vocabulary = sorted(joined.loc[fit_mask, column].dropna().unique().tolist())
        codes = {value: index for index, value in enumerate(vocabulary)}
        categories[column] = codes
        out[column + "_enc"] = joined[column].map(codes).fillna(-1).astype(float)
    metadata = {
        "policy": "fit-window-v1",
        "origin_exclusive": origin.strftime("%Y-%m-%d"),
        "source_model_year_ceiling": ceiling,
        "source_records_available": len(raw),
        "fit_rows": int(fit_mask.sum()),
        "fit_last_month": out.loc[fit_mask, "date"].max().strftime("%Y-%m-%d"),
        "unmatched_training_rows": int((fit_mask & matched_year.isna()).sum()),
        "unmatched_forecast_rows": int((~out["date"].lt(origin) & matched_year.isna()).sum()),
        "numeric_medians": medians,
        "all_missing_numeric_columns": all_missing,
        "category_codes": categories,
        "availability_limit": "Model year only; within-year release dates and historical revisions unknown",
    }
    return out, metadata

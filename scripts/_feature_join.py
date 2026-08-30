#!/usr/bin/env python3
"""Join annual specifications to the monthly sales panel without future fill."""
import os

import numpy as np
import pandas as pd

from _series_mapping import build_series_name_mapping

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV = os.path.join(BASE, "data", "raw", "feature.csv")
FEAT_XLSX = os.path.join(BASE, "data", "raw", "feature.xlsx")

# 连续数值配置特征
CFG_NUM = ["official_price_wan", "engine_max_power_kw", "engine_max_torque_nm",
           "battery_capacity_kwh", "battery_range_km", "length_mm", "width_mm",
           "height_mm", "wheelbase_mm", "curb_weight_kg", "seat_count",
           "door_count", "trunk_volume_l", "acceleration_0_100_s",
           "fuel_consumption_l_100km"]
# 类别配置特征 (编码成 *_enc)
CFG_CAT = ["energy_type", "vehicle_class", "brand_name", "body_structure",
           "gearbox_type", "seat_material"]
CFG_COLS = CFG_NUM + [c + "_enc" for c in CFG_CAT]


def load_feature_source():
    """Load the local CSV when present, otherwise use the tracked workbook."""
    if os.path.exists(FEAT_CSV):
        return pd.read_csv(FEAT_CSV, low_memory=False)
    if os.path.exists(FEAT_XLSX):
        return pd.read_excel(FEAT_XLSX)
    raise FileNotFoundError(
        "Missing configuration source: expected data/raw/feature.csv or "
        "the tracked data/raw/feature.xlsx"
    )


def _load_cfg_frame():
    feat = load_feature_source()
    feat["series_name"] = feat["series_name"].astype(str)
    feat["year"] = pd.to_numeric(feat["year"], errors="coerce")
    fk = feat[["series_name", "year"] + CFG_NUM + CFG_CAT].copy()
    for c in CFG_CAT:
        fk[c] = fk[c].astype(str).fillna("NA")
        mp = {v: i for i, v in enumerate(sorted(fk[c].unique()))}
        fk[c + "_enc"] = fk[c].map(mp)
    for c in CFG_NUM:
        fk[c] = pd.to_numeric(fk[c], errors="coerce")
        fk[c] = fk[c].fillna(fk[c].median())
    return fk[["series_name", "year"] + CFG_COLS]


def join_cfg(sm, keep_unmatched: bool = False):
    """Join by series/year with a causal specification fallback.

    By default this keeps the historical behaviour and drops monthly rows for
    which no specification is available.  Forecasting panels can instead set
    ``keep_unmatched=True``: the sales row is retained, numeric attributes use
    the configuration-table median, and encoded categorical attributes use
    ``-1`` as an explicit unknown sentinel.  This is important for preserving
    the calendar spacing of the sales history; a missing configuration record
    must not silently turn two months ago into the previous month.
    """
    fk = _load_cfg_frame()
    mapping = build_series_name_mapping(sm["series_name"], fk["series_name"])
    name_map = mapping.set_index("sales_series_name")["config_series_name"]
    sm = sm[sm["series_name"].isin(name_map.index)].copy()
    sm["config_series_name"] = sm["series_name"].map(name_map)
    fk = fk.rename(columns={"series_name": "config_series_name"})
    sm = sm.merge(fk, on=["config_series_name", "year"], how="left")
    miss = sm[CFG_COLS[0]].isna()
    if miss.any():
        # 每个车系: 年份 -> 配置元组, 仅保留 <= 行年份的可用年份
        cfg_by_series = {}
        for s, g in fk.groupby("config_series_name"):
            cfg_by_series[s] = {
                float(y): tup
                for y, tup in zip(g["year"].astype(float),
                                  g[CFG_COLS].itertuples(index=False, name=None))
            }
        for idx in sm.index[miss]:
            s, y = sm.at[idx, "config_series_name"], float(sm.at[idx, "year"])
            if s not in cfg_by_series:
                continue
            cand = [yy for yy in cfg_by_series[s] if yy <= y]
            if not cand:
                continue
            for j, c in enumerate(CFG_COLS):
                sm.at[idx, c] = cfg_by_series[s][max(cand)][j]
    if keep_unmatched:
        for column in CFG_NUM:
            sm[column] = pd.to_numeric(sm[column], errors="coerce")
            sm[column] = sm[column].fillna(pd.to_numeric(fk[column], errors="coerce").median())
        for column in CFG_COLS:
            if column not in CFG_NUM:
                sm[column] = pd.to_numeric(sm[column], errors="coerce").fillna(-1.0)
    else:
        sm = sm[sm[CFG_COLS[0]].notna()].copy()
    sm = sm.drop(columns="config_series_name")
    return sm

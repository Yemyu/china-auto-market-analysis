#!/usr/bin/env python3
"""Join annual specifications to the monthly sales panel without future fill."""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(BASE, "data", "raw", "feature.csv")

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


def _load_cfg_frame():
    feat = pd.read_csv(FEAT, low_memory=False)
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


def join_cfg(sm):
    """Join by series and year, falling back only to an earlier specification."""
    fk = _load_cfg_frame()
    fs = set(fk["series_name"])
    sm = sm[sm["series_name"].isin(fs)].copy()
    sm = sm.merge(fk, on=["series_name", "year"], how="left")
    miss = sm[CFG_COLS[0]].isna()
    if miss.any():
        # 每个车系: 年份 -> 配置元组, 仅保留 <= 行年份的可用年份
        cfg_by_series = {}
        for s, g in fk.groupby("series_name"):
            cfg_by_series[s] = {
                float(y): tup
                for y, tup in zip(g["year"].astype(float),
                                  g[CFG_COLS].itertuples(index=False, name=None))
            }
        for idx in sm.index[miss]:
            s, y = sm.at[idx, "series_name"], float(sm.at[idx, "year"])
            if s not in cfg_by_series:
                continue
            cand = [yy for yy in cfg_by_series[s] if yy <= y]
            if not cand:
                continue
            for j, c in enumerate(CFG_COLS):
                sm.at[idx, c] = cfg_by_series[s][max(cand)][j]
    sm = sm[sm[CFG_COLS[0]].notna()].copy()
    return sm

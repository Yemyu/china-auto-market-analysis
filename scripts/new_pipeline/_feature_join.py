#!/usr/bin/env python3
"""
_feature_join.py — 共享的正确配置接入模块 (2026-08-12 修正)

背景: 旧版 new_pipeline 用 data/processed_new/series_config.csv 作为配置源,
但该文件把 feature.csv 里「同车系不同年份」的行误当「不同 trim」聚合
(n_trims 实为年份数 / price_min_max 实为跨年调价 / has_ev 实为跨年能源变更 /
*_cov 实为跨年配备变化), 语义错误, 已被废弃。

本模块直接从 data/raw/feature.csv 取配置, 按 (series_name, year) 精确 join 到
月度面板, 配置随年份更新(换代/改款能反映), 不压扁、不依赖错口径中间文件。

提供:
  CFG_NUM / CFG_CAT / CFG_COLS  配置特征列名
  join_cfg(sm)                  把配置 join 进月度面板(限定 feature 主表车系, 带同系跨年兜底)
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    """Left-join 配置到月度面板 sm (需含 series_name, year 列)。

    - 只保留 feature 主表里的车系(绝不用主表外的车充数)
    - 按 (series_name, year) 精确命中
    - **因果回退(防泄漏)**: 该(车系,年)无记录时, 取同车系『年份 <= 当前行年份』的
      最近一条配置; 绝不用未来年份(如 2026)的配置去填过去(如 2022)的月份。
      (旧版用「同车系最近一年」= 最新年份兜底, 会把换代后的新配置泄漏给历史月, 已修正)
    - 返回仅含配置齐全的行
    """
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

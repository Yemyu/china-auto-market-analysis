# -*- coding: utf-8 -*-
"""
00_build_series_index.py — 建立跨平台统一车系索引 (新 ID 体系)
=============================================================
设计目标
-------
把"舆情 / 配置 / 销量"三套数据用 **series_name (车系名)** 作为唯一通用连接键，
统一到一个索引表里。每个车系一行，横向挂上它在各平台的自有 id：

  canonical_id          本项目自建的稳定车系主键 (S0001, S0002, ...)
  series_name           车系名 (通用键, 全表唯一)
  dongchedi_series_id   懂车帝数字 id  (来自 vehicles.csv / feature.xlsx / analysis_input.csv)
  pcauto_series_id      太平洋数字 id  (来自 monthly_sales.csv 的 series_id 列)
  pcauto_source_series_id  太平洋 sgXXXX id (来自 all_sales 的 source_series_id 列)
  n_platforms          该系在几个源里出现 (用于覆盖率诊断)

为什么需要它
-----------
- 销量(all_sales) 的 series_id 是太平洋编码(106/27043), 配置(vehicles/feature) 的
  series_id 是懂车帝编码(1291/10026), 二者**不是一套**, 无法直接 join。
- 唯一跨文件对齐的键是 series_name。本表把两套 id 并到一行, 下游所有脚本
  统一按 canonical_id / series_name 连接, 不再关心源平台编码。

用法
---
  python scripts/00_build_series_index.py
输出: data/raw/series_index.csv
（属数据接入层, 不进 git —— 与 data/raw 其他大文件一致）
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw")


def _load_dongchedi_ids():
    """从懂车帝体系文件收集 series_name -> 懂车帝数字 id"""
    frames = []
    # vehicles.csv
    v = os.path.join(RAW, "vehicles.csv")
    if os.path.exists(v):
        df = pd.read_csv(v, encoding="utf-8-sig", low_memory=False)
        if {"series_name", "series_id"} <= set(df.columns):
            frames.append(df[["series_name", "series_id"]].astype(str))
    # feature.xlsx
    fx = os.path.join(RAW, "feature.xlsx")
    if os.path.exists(fx):
        df = pd.read_excel(fx, sheet_name="power_features_with_sales_v2")
        if {"series_name", "series_id"} <= set(df.columns):
            frames.append(df[["series_name", "series_id"]].astype(str))
    # analysis_input.csv (舆情宇宙, 自带懂车帝 id)
    ai = os.path.join(ROOT, "data", "sentiment", "analysis_input.csv")
    if os.path.exists(ai):
        df = pd.read_csv(ai)
        if {"series_name", "series_id"} <= set(df.columns):
            frames.append(df[["series_name", "series_id"]].astype(str))
    if not frames:
        return pd.DataFrame(columns=["series_name", "dongchedi_series_id"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["series_id"].str.replace(".0", "", regex=False).str.isdigit()]  # 只留数字(懂车帝)id
    out = out.rename(columns={"series_id": "dongchedi_series_id"})
    # 同名取出现频次最高的 id (多数一致)
    out = (out.drop_duplicates(["series_name", "dongchedi_series_id"])
              .groupby("series_name")["dongchedi_series_id"]
              .agg(lambda s: s.value_counts().index[0])
              .reset_index())
    return out


def _load_pcauto_ids():
    """从 all_sales 收集 series_name -> 太平洋数字 id + sg id"""
    p = os.path.join(RAW, "monthly_sales.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["series_name", "pcauto_series_id", "pcauto_source_series_id"])
    df = pd.read_csv(p)
    df["series_name"] = df["series_name"].astype(str)
    df["pcauto_series_id"] = df["series_id"].astype(str)
    df["pcauto_source_series_id"] = df["source_series_id"].astype(str)
    out = df[["series_name", "pcauto_series_id", "pcauto_source_series_id"]].drop_duplicates("series_name")
    return out


def main():
    dcd = _load_dongchedi_ids()
    pca = _load_pcauto_ids()

    # 合并: series_name 为通用键
    idx = pca.merge(dcd, on="series_name", how="outer")
    idx = idx.sort_values("series_name").reset_index(drop=True)

    # 自建稳定主键
    idx.insert(0, "canonical_id", ["S%04d" % (i + 1) for i in range(len(idx))])

    def _count(row):
        n = 0
        for c in ["dongchedi_series_id", "pcauto_series_id", "pcauto_source_series_id"]:
            if pd.notna(row[c]) and str(row[c]) not in ("", "nan"):
                n += 1
        return n
    idx["n_platforms"] = idx.apply(_count, axis=1)

    out_path = os.path.join(RAW, "series_index.csv")
    idx.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[00] series_index 构建完成 -> {out_path}")
    print(f"    总车系数: {len(idx)}")
    print(f"    含太平洋id: {idx['pcauto_series_id'].notna().sum()}")
    print(f"    含懂车帝id: {idx['dongchedi_series_id'].notna().sum()}")
    print(f"    双平台都有id: {(idx['pcauto_series_id'].notna() & idx['dongchedi_series_id'].notna()).sum()}")
    print(f"    只在一平台: {(idx['n_platforms'] == 1).sum()}")
    print("\n    列:", list(idx.columns))
    print(idx.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

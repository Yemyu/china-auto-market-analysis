# -*- coding: utf-8 -*-
"""Build a cross-source series index keyed by canonical series name."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def _load_dongchedi_ids():
    """从懂车帝体系文件收集 series_name -> 懂车帝数字 id"""
    frames = []
    # Canonical configuration table.
    feature = os.path.join(RAW, "feature.csv")
    if os.path.exists(feature):
        df = pd.read_csv(feature, encoding="utf-8-sig", low_memory=False)
        if {"series_name", "series_id"} <= set(df.columns):
            frames.append(df[["series_name", "series_id"]].astype(str))
    # Archived reviews retain Dongchedi series IDs.
    archive = os.path.join(ROOT, "data", "resources", "historical_reviews", "review_absa_reference.csv.gz")
    if os.path.exists(archive):
        df = pd.read_csv(archive, usecols=["series_name", "series_id"])
        if {"series_name", "series_id"} <= set(df.columns):
            frames.append(df[["series_name", "series_id"]].astype(str))
    # Only verified entries from the resolution register are accepted.
    resolved = os.path.join(ROOT, "data", "processed", "review_collection", "dongchedi_id_resolutions.csv")
    if os.path.exists(resolved):
        df = pd.read_csv(resolved)
        required = {"series_name", "dongchedi_series_id", "resolution_status"}
        if required <= set(df.columns):
            df = df[df["resolution_status"].astype(str).str.lower().eq("verified")]
            frames.append(df.rename(columns={"dongchedi_series_id": "series_id"})[["series_name", "series_id"]].astype(str))
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

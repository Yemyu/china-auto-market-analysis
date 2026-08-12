#!/usr/bin/env python3
"""
Stage 3 — shared representative subset
Builds (once) and caches a *stratified* representative subset of ~N series,
used by every stage-3 modelling script so they all evaluate on the SAME
series (apples-to-apples comparison).

Stratification: energy_type × vehicle_class × sales-quartile, with
proportional allocation (min 1 per stratum) and top-up by total sales until
N is reached. Deterministic (no RNG) — within each stratum we keep the
highest-total-sales members.

The two source tables use different platform series_id spaces, so we join by
series_name (already validated during stage 2 alignment).

Run directly to (re)build:
  python scripts/_subset.py
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SALES = os.path.join(BASE, "data", "processed_new", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "raw", "feature.csv")
SUBSET_CSV = os.path.join(BASE, "data", "processed_new", "subset_150.csv")


def build_stratified_subset(n=150, min_months=10):
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    feat = pd.read_csv(FEAT)
    feat["series_name"] = feat["series_name"].astype(str)
    # feature.csv 粒度=车系×年; 分层抽样只需每车系一条, 取最新年(配置随年更新)
    feat = feat.sort_values("year").drop_duplicates("series_name", keep="last")

    months = sales.groupby("series_name")["date"].nunique()
    tot = sales.groupby("series_name")["monthly_sales"].sum()

    common = set(sales["series_name"]) & set(feat["series_name"])
    df = pd.DataFrame({"series_name": list(common)})
    df["total_sales"] = df["series_name"].map(tot)
    df["n_months"] = df["series_name"].map(months)
    df = df.merge(
        feat[["series_name", "energy_type", "vehicle_class", "brand_name",
              "official_price_wan"]],
        on="series_name", how="left",
    )
    df = df[df["n_months"] >= min_months].dropna(subset=["energy_type", "vehicle_class"]).copy()
    df["energy_type"] = df["energy_type"].astype(str).fillna("NA")
    df["vehicle_class"] = df["vehicle_class"].astype(str).fillna("NA")
    try:
        df["sales_tier"] = pd.qcut(df["total_sales"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except Exception:
        df["sales_tier"] = "Q2"
    df["sales_tier"] = df["sales_tier"].astype(str)

    df["stratum"] = df["energy_type"] + "|" + df["vehicle_class"] + "|" + df["sales_tier"]

    # proportional allocation
    sizes = df["stratum"].value_counts()
    raw = sizes / sizes.sum() * n
    alloc = {st: max(1, int(round(raw[st]))) for st in sizes.index}
    for st in list(alloc):
        if alloc[st] > sizes[st]:
            alloc[st] = int(sizes[st])

    # top up to n
    deficit = n - sum(alloc.values())
    if deficit > 0:
        remaining = (sizes - pd.Series(alloc)).fillna(0).astype(int)
        remaining = remaining[remaining > 0]
        if len(remaining):
            add = (remaining / remaining.sum() * deficit).round().astype(int)
            for st in add.index:
                alloc[st] += int(add[st])
            d2 = deficit - int(add.sum())
            order = list(remaining.index)
            i = 0
            while d2 > 0:
                st = order[i % len(order)]
                if alloc[st] < sizes[st]:
                    alloc[st] += 1
                    d2 -= 1
                i += 1
                if i > 5000:
                    break

    # pick highest-total-sales members within each stratum (deterministic)
    chosen = []
    for st in sizes.index:
        members = df[df["stratum"] == st].sort_values("total_sales", ascending=False)
        k = min(alloc[st], len(members))
        chosen.extend(members["series_name"].head(k).tolist())
    chosen = list(dict.fromkeys(chosen))

    # if still short, top up by highest-sales eligible series not yet chosen
    if len(chosen) < n:
        extra = df[~df["series_name"].isin(chosen)].sort_values("total_sales", ascending=False)
        for s in extra["series_name"].tolist():
            if len(chosen) >= n:
                break
            chosen.append(s)

    # enforce exact target n, never drop a stratum below 1 representative:
    # keep each stratum's top-sales member, drop weakest extras (lowest sales) first
    if len(chosen) > n:
        ts = df.set_index("series_name")["total_sales"].to_dict()
        by_st = df[df["series_name"].isin(chosen)].sort_values("total_sales", ascending=False)
        keep, drop_pool = [], []
        for _, g in by_st.groupby("stratum"):
            members = g["series_name"].tolist()
            keep.append(members[0])
            drop_pool.extend(members[1:])
        drop_pool.sort(key=lambda s: ts.get(s, 0))
        need = len(chosen) - n
        drop_set = set(drop_pool[:need])
        chosen = [s for s in chosen if s not in drop_set]

    out = (df[df["series_name"].isin(chosen)]
           [["series_name", "energy_type", "vehicle_class", "brand_name",
             "sales_tier", "total_sales", "n_months"]]
           .set_index("series_name").loc[chosen].reset_index())
    os.makedirs(os.path.dirname(SUBSET_CSV), exist_ok=True)
    out.to_csv(SUBSET_CSV, index=False)
    print(f"[subset] built stratified subset: {len(out)} series "
          f"(target {n}), {out['energy_type'].nunique()} energy types, "
          f"{out['vehicle_class'].nunique()} vehicle classes")
    return out["series_name"].tolist()


def load_subset(n=150):
    if not os.path.exists(SUBSET_CSV):
        build_stratified_subset(n)
    return pd.read_csv(SUBSET_CSV)["series_name"].astype(str).tolist()


if __name__ == "__main__":
    build_stratified_subset(150)

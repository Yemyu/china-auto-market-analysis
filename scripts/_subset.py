#!/usr/bin/env python3
"""Stage 3 — shared, time-eligible representative evaluation cohort.

The 371 series with both monthly sales and configuration data are the Leg-B
*population*.  Forecast-model comparisons are intentionally run on a smaller,
stratified cohort because ARIMA/Prophet are fitted once per series.  This file
builds that cohort deterministically and, crucially, makes every model use the
same time-eligible series.

Eligibility is decided from the authoritative train/val/test files, not just
from total lifetime months: a selected series must have enough usable training
history and all six validation and six test months.  This prevents a new model
from silently evaluating a different subset from the other models.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SALES = os.path.join(BASE, "data", "processed_new", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "raw", "feature.csv")
SPLITS = os.path.join(BASE, "data", "processed_new", "splits")
SUBSET_CSV = os.path.join(BASE, "data", "processed_new", "subset_150.csv")
N_EVAL = 150
MIN_TRAIN_MONTHS = 24
VAL_MONTHS = 6
TEST_MONTHS = 6


def _time_eligibility() -> pd.DataFrame:
    """Return one row per series with the evaluation-window coverage audit."""
    parts = {}
    for split in ("train", "val", "test"):
        path = os.path.join(SPLITS, f"{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}; run scripts/new_pipeline/06_make_splits.py first."
            )
        d = pd.read_csv(path, parse_dates=["date"])
        # Unique months rather than raw rows: one series may never contribute
        # more than one observation per month, but this keeps the invariant explicit.
        parts[split] = d.groupby("series_name")["date"].nunique().rename(f"{split}_months")

    cov = pd.concat(parts.values(), axis=1).fillna(0).astype(int).reset_index()
    cov["eligible"] = (
        (cov["train_months"] >= MIN_TRAIN_MONTHS)
        & (cov["val_months"] == VAL_MONTHS)
        & (cov["test_months"] == TEST_MONTHS)
    )
    return cov


def build_stratified_subset(n=N_EVAL):
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    feat = pd.read_csv(FEAT)
    feat["series_name"] = feat["series_name"].astype(str)
    # feature.csv 粒度=车系×年; 分层抽样只需每车系一条, 取最新年(配置随年更新)
    feat = feat.sort_values("year").drop_duplicates("series_name", keep="last")

    tot = sales.groupby("series_name")["monthly_sales"].sum()
    coverage = _time_eligibility()

    common = set(sales["series_name"]) & set(feat["series_name"])
    df = pd.DataFrame({"series_name": list(common)})
    df["total_sales"] = df["series_name"].map(tot)
    df = df.merge(coverage, on="series_name", how="left")
    df = df.merge(
        feat[["series_name", "energy_type", "vehicle_class", "brand_name",
              "official_price_wan"]],
        on="series_name", how="left",
    )
    df = df[df["eligible"].fillna(False)].dropna(subset=["energy_type", "vehicle_class"]).copy()
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
             "sales_tier", "total_sales", "train_months", "val_months", "test_months"]]
           .set_index("series_name").loc[chosen].reset_index())
    os.makedirs(os.path.dirname(SUBSET_CSV), exist_ok=True)
    out.to_csv(SUBSET_CSV, index=False)
    print(f"[subset] built time-eligible stratified cohort: {len(out)} series "
          f"(target {n}; min train={MIN_TRAIN_MONTHS}, val/test={VAL_MONTHS}/{TEST_MONTHS}), "
          f"{out['energy_type'].nunique()} energy types, {out['vehicle_class'].nunique()} vehicle classes")
    return out["series_name"].tolist()


def load_subset(n=N_EVAL):
    # The old cached file was based only on lifetime coverage.  Regenerate it
    # once so every modelling script gets the corrected common cohort.
    required = {"series_name", "train_months", "val_months", "test_months"}
    if not os.path.exists(SUBSET_CSV) or not required <= set(pd.read_csv(SUBSET_CSV, nrows=0).columns):
        build_stratified_subset(n)
    return pd.read_csv(SUBSET_CSV)["series_name"].astype(str).tolist()


if __name__ == "__main__":
    build_stratified_subset(150)

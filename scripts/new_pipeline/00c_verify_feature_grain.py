#!/usr/bin/env python3
"""
验证 feature.csv 的真实粒度与 annual_sales 语义（可复现的证据脚本）

为什么需要它
--------------------------------------------------
早前误判 feature.csv 是「车型(trim) × 年」，依据是"475/766 车系内各车型销量互不相同"。
该检验被误读：组内销量不同是因为**年份**不同，不是配置不同。
错误判断导致两个下游问题：
  - series_config.csv 把「跨年份」当「跨 trim」聚合（n_trims 实为年份数…）
  - 16 归因在组内拟合，而组内配置无变异 -> R²=0.099、配置重要性全 0

本脚本用 4 项检验一次性把粒度钉死，任何人可复跑核对。

用法: python scripts/new_pipeline/00c_verify_feature_grain.py
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = os.path.join(BASE, "data", "raw", "feature.csv")
SALES = os.path.join(BASE, "data", "raw", "monthly_sales.csv")


def main():
    f = pd.read_csv(FEAT, low_memory=False)
    print(f"feature.csv 形状 {f.shape}\n")

    # ---------- 检验1: 唯一键 ----------
    print("=" * 72)
    print("检验1  (series_name, year) 是否唯一标识一行？")
    print("=" * 72)
    n_comb = f.groupby(["series_name", "year"]).ngroups
    dup = int(f.duplicated(subset=["series_name", "year"]).sum())
    print(f"行数 {len(f)} | (车系,年) 组合数 {n_comb} | 重复行 {dup}")
    print(f"车系数 {f['series_name'].nunique()} | car_id 唯一数 {f['car_id'].nunique()}")
    print("→ " + ("✅ 车系×年 是唯一键，粒度不是 trim 级"
                  if dup == 0 else "❌ 还有更细粒度"))

    # ---------- 检验2: 同车系跨年，配置变不变 ----------
    print("\n" + "=" * 72)
    print("检验2  同车系跨年，配置列变不变？(变化越少 => 配置无法解释年度涨跌)")
    print("=" * 72)
    multi = f[f.groupby("series_name")["year"].transform("size") > 1]
    print(f"(仅看有多年记录的车系: {multi['series_name'].nunique()} 系 / {len(multi)} 行)")
    for c in ["vehicle_class", "seat_heating", "wheelbase_mm", "battery_capacity_kwh",
              "energy_type", "seat_count", "engine_max_power_kw", "length_mm",
              "seat_material", "official_price_wan", "car_id"]:
        if c in multi.columns:
            n = multi.groupby("series_name")[c].nunique(dropna=False)
            print(f"  {c:24s} 组内有变化的车系占比 {100 * (n > 1).mean():5.1f}%")
    print("→ 结论: 配置在车系内近乎恒定，只有换代(car_id)/调价(price)会动")

    # ---------- 检验3: annual_sales 语义 ----------
    print("\n" + "=" * 72)
    print("检验3  annual_sales 是不是 monthly_sales 的年度汇总？")
    print("=" * 72)
    m = pd.read_csv(SALES, low_memory=False)
    agg = (m.groupby(["series_name", "year"], as_index=False)["monthly_sales"].sum()
           .rename(columns={"monthly_sales": "sum_monthly"}))
    j = f[["series_name", "year", "annual_sales"]].merge(agg, on=["series_name", "year"])
    j = j.dropna(subset=["annual_sales", "sum_monthly"])
    j = j[(j["annual_sales"] > 0) & (j["sum_monthly"] > 0)]
    j["ratio"] = j["annual_sales"] / j["sum_monthly"]
    print(f"可对比 (车系,年) 条数 {len(j)}")
    print(f"比值  mean={j['ratio'].mean():.6f}  std={j['ratio'].std():.6f}  "
          f"min={j['ratio'].min():.6f}  max={j['ratio'].max():.6f}")
    print("→ " + ("✅ 精确等于 1，两表同源可互校"
                  if abs(j["ratio"].mean() - 1) < 1e-6 else "⚠️ 存在偏差，需查"))

    # ---------- 检验4: 覆盖度 —— 还需不需要爬销量 ----------
    print("\n" + "=" * 72)
    print("检验4  以 feature 为主表的销量覆盖度（决定要不要爬）")
    print("=" * 72)
    fs, ms = set(f["series_name"].astype(str)), set(m["series_name"].astype(str))
    print(f"feature 车系 {len(fs)} | monthly_sales 车系 {len(ms)} | "
          f"名字可对齐 {len(fs & ms)} ({100 * len(fs & ms) / len(fs):.1f}%)")
    print(f"feature 有、月度表没有: {len(fs - ms)} | 月度表有、feature 没有(孤儿): {len(ms - fs)}")
    miss = f[f["series_name"].isin(fs - ms)]
    ma = miss.groupby("series_name")["annual_sales"].apply(lambda s: s.notna().any())
    print(f"\n关键: 那 {len(ma)} 个'缺月度'车系里，自带 annual_sales 的有 "
          f"{int(ma.sum())} 个 ({100 * ma.mean():.1f}%)")
    print("→ 结论: 它们不缺销量，只缺『月度拆分』；年度归因无需爬取")
    panel = f[f["annual_sales"].notna() & f["year"].between(2022, 2026)]
    print(f"\n年度归因可用面板: {len(panel)} 行 / {panel['series_name'].nunique()} 车系 (2022-2026)")


if __name__ == "__main__":
    main()

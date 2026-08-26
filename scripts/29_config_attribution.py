#!/usr/bin/env python3
"""
配置 → 年销量 归因（正确口径版）

背景 / 为什么需要这个脚本
--------------------------------------------------
实测确认 feature.csv 的粒度是「车系 × 年」(唯一键, 2084 行 = 2084 组合),
且 annual_sales 精确等于 monthly_sales 按 (车系,年) 的汇总 (1484 条 ratio=1.0, std=0)。
每个「车系-年」只挂一个代表车型 (car_id), 不是全部 trim。

由此产生两个硬约束:
 1) 配置在车系内跨年几乎不变 (vehicle_class 0%, 座椅加热 0%, 轴距 4.1%, 能源 5.8% 有变化),
    所以配置只能解释「车系之间」的销量差异 (between-series),
    无法解释同一车系的年度涨跌 (within-series)。
    => 归因必须建立在 between 变异上; 绝不能加车系固定效应(会把配置效应吃光)。
 2) 同车系多年记录的配置近乎重复 => 随机切分会泄露
    (训练集见过同车系另一年的 y, 而配置几乎一样, 等于背答案)。
    => 必须按 series_name 分组切分 (GroupKFold)。

消融设计 (直接回答「配置贡献多少」)
--------------------------------------------------
  A. YEAR-ONLY    : 仅年份         —— 大盘/时间基准
  B. +BRAND       : 年份 + 品牌     —— 品牌资产能解释多少
  C. +CONFIG      : 年份 + 品牌 + 配置 —— 配置的增量贡献 = C - B
  D. CONFIG-ONLY  : 仅配置         —— 配置单独的解释力

输出:
  data/processed_new/stage4/config_attribution_ablation.csv
  data/processed_new/stage4/config_importance_annual.csv
  figures_new/config_attribution_ablation.png
  figures_new/config_attribution_shap.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup  # noqa: F401
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = os.path.join(BASE, "data", "raw", "feature.csv")
FIG = os.path.join(BASE, "figures_new")
PROC = os.path.join(BASE, "data", "processed_new", "stage4")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2022, 2026
N_SPLITS = 5

# 排除: 标识列 / 目标 / 市场结果价(内生, 非产品属性) / 冗余文本
DROP = {
    "pcauto_series_id", "series_id", "series_name", "car_id", "car_name",
    "annual_sales", "official_price_str", "owner_price", "dealer_price",
    "brand_id", "pcauto_brand", "original_energy_type",
    "engine_displacement_ml",          # 与 _l 重复
    "speaker_count_original",          # 与 speaker_count 重复
}
CAT_HINT = {
    "energy_type", "vehicle_class", "manufacturer", "brand_name",
    "engine_intake_type", "engine_cylinder_arrangement", "engine_unique_tech",
    "fuel_form", "fuel_grade", "oil_supply", "cylinder_material",
    "environmental_standard", "gearbox_short", "gearbox_type", "motor_type",
    "battery_type", "body_structure", "door_open_way", "driver_airbag",
    "side_airbag", "side_air_curtain", "knee_airbag", "steering_wheel_material",
    "steering_wheel_adjustment", "center_screen", "multimedia_interface",
    "sound_brand", "seat_material", "seat_heating", "aircon_control",
    "battery_warranty", "warranty_period", "charging_time_h", "fast_charge_percent",
}


def wmape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    d = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / d * 100 if d > 0 else np.nan


def build_matrix(df):
    """把 车系×年 表拆成 数值配置 / 类别配置 / 品牌 / 年份 四组特征块。"""
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in DROP or c == "year":
            continue
        s = df[c]
        if c in CAT_HINT or s.dtype == object:
            if s.nunique(dropna=True) >= 2:      # 单一取值列无信息, 剔除
                cat_cols.append(c)
        else:
            if s.nunique(dropna=True) >= 2:
                num_cols.append(c)

    # 数值: 中位数填充
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median())

    # 类别: 低基数 one-hot, 高基数(如 manufacturer) 用频次编码避免维度爆炸
    parts, cat_used = [], []
    brand_parts = []
    for c in cat_cols:
        s = df[c].astype(str).fillna("NA")
        if c == "brand_name":
            d = pd.get_dummies(s, prefix="brand").astype(float)
            brand_parts.append(d)
            continue
        if s.nunique() <= 15:
            parts.append(pd.get_dummies(s, prefix=c).astype(float))
        else:
            freq = s.map(s.value_counts(normalize=True))
            parts.append(freq.rename(f"{c}_freq").to_frame())
        cat_used.append(c)

    X_cfg = pd.concat([X_num] + parts, axis=1) if parts else X_num
    X_brand = pd.concat(brand_parts, axis=1) if brand_parts else pd.DataFrame(index=df.index)
    X_year = pd.get_dummies(df["year"].astype(int), prefix="year").astype(float)
    return X_cfg, X_brand, X_year, num_cols, cat_used


def cv_eval(X, y, groups, label):
    """GroupKFold CV: 同一车系整组进同一折, 防跨年泄露。"""
    if X.shape[1] == 0:
        return None
    gkf = GroupKFold(n_splits=N_SPLITS)
    r2s, wms, oof = [], [], np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, random_state=42, n_jobs=1)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        oof[te] = p
        r2s.append(r2_score(y[te], p))
        wms.append(wmape(np.expm1(y[te]), np.maximum(np.expm1(p), 0)))
    print(f"  {label:14s} R2(log)={np.mean(r2s):6.3f}  WMAPE={np.mean(wms):7.1f}%  维度={X.shape[1]}")
    return {"variant": label, "R2_log_mean": np.mean(r2s), "R2_log_std": np.std(r2s),
            "WMAPE_mean": np.mean(wms), "n_features": X.shape[1]}, oof


def main():
    df = pd.read_csv(FEAT, low_memory=False)
    df = df[df["annual_sales"].notna() & df["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    df = df.reset_index(drop=True)
    y = np.log1p(df["annual_sales"].astype(float).values)
    groups = df["series_name"].astype(str).values
    print(f"[归因] 面板 {len(df)} 行 (车系×年) | 车系 {df['series_name'].nunique()} | "
          f"年份 {YEAR_MIN}-{YEAR_MAX}")
    print(f"[归因] y=log1p(annual_sales), 分组切分 GroupKFold({N_SPLITS}) by series_name\n")

    X_cfg, X_brand, X_year, num_cols, cat_used = build_matrix(df)
    print(f"[特征] 数值配置 {len(num_cols)} | 类别配置 {len(cat_used)} | "
          f"品牌 one-hot {X_brand.shape[1]} | 年份 {X_year.shape[1]}")
    print(f"[特征] 配置块合计 {X_cfg.shape[1]} 维\n")

    print("===== 消融 (GroupKFold 交叉验证均值) =====")
    variants = {
        "YEAR-ONLY": X_year,
        "+BRAND": pd.concat([X_year, X_brand], axis=1),
        "+CONFIG": pd.concat([X_year, X_brand, X_cfg], axis=1),
        "CONFIG-ONLY": X_cfg,
    }
    rows = []
    for lab, X in variants.items():
        res, _ = cv_eval(X, y, groups, lab)
        if res:
            rows.append(res)
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(PROC, "config_attribution_ablation.csv"), index=False)

    r_b = abl.loc[abl["variant"] == "+BRAND", "R2_log_mean"].iloc[0]
    r_c = abl.loc[abl["variant"] == "+CONFIG", "R2_log_mean"].iloc[0]
    print(f"\n[配置增量贡献] R2: {r_b:.3f} -> {r_c:.3f}  (ΔR2 = {r_c - r_b:+.3f})")

    # ---- 全量拟合一次, 取重要性 + SHAP ----
    X_full = pd.concat([X_year, X_brand, X_cfg], axis=1)
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, random_state=42, n_jobs=1)
    model.fit(X_full, y)
    imp = (pd.DataFrame({"feature": X_full.columns, "gain": model.feature_importances_})
           .sort_values("gain", ascending=False).reset_index(drop=True))
    imp["block"] = np.where(imp["feature"].str.startswith("year_"), "year",
                    np.where(imp["feature"].str.startswith("brand_"), "brand", "config"))
    imp.to_csv(os.path.join(PROC, "config_importance_annual.csv"), index=False)

    print("\n===== 各特征块重要性占比 =====")
    blk = imp.groupby("block")["gain"].sum().sort_values(ascending=False)
    for k, v in blk.items():
        print(f"  {k:8s} {100 * v / blk.sum():5.1f}%")

    print("\n===== Top 15 配置特征 =====")
    top_cfg = imp[imp["block"] == "config"].head(15)
    for _, r in top_cfg.iterrows():
        print(f"  {r['feature']:34s} {r['gain']:.4f}")

    # ---- 图1: 消融 + 特征块占比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].bar(abl["variant"], abl["R2_log_mean"],
                yerr=abl["R2_log_std"], capsize=4,
                color=["#B0B0B0", "#4C78A8", "#54A24B", "#F58518"])
    axes[0].set_ylabel("R² (log 空间, GroupKFold CV)")
    axes[0].set_title("消融：配置对年销量的增量解释力")
    axes[0].tick_params(axis="x", labelsize=9)
    for i, v in enumerate(abl["R2_log_mean"]):
        axes[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].barh(top_cfg["feature"][::-1], top_cfg["gain"][::-1], color="#54A24B")
    axes[1].set_title("Top 15 配置特征重要性 (gain)")
    axes[1].tick_params(labelsize=8)
    fig.suptitle("配置 → 年销量 归因（车系×年 面板, between-series 变异）", fontsize=12)
    fig.savefig(os.path.join(FIG, "config_attribution_ablation.png"), dpi=130)
    plt.close(fig)

    # ---- 图2: SHAP ----
    try:
        import shap
        sample = X_full.sample(min(600, len(X_full)), random_state=42)
        sv = shap.TreeExplainer(model).shap_values(sample)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, sample, max_display=18, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "config_attribution_shap.png"), dpi=130)
        plt.close()
        print("\n[SHAP] figures_new/config_attribution_shap.png")
    except Exception as e:
        print(f"\n[SHAP] 跳过: {e}")

    print("[归因] 完成。")


if __name__ == "__main__":
    main()

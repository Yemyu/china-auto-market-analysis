#!/usr/bin/env python3
"""Estimate configuration contribution to annual sales.

The table is one row per series and year. Cross-validation is grouped by
series, and the ablation compares year, year plus brand, and configuration.
"""
import json
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

from _feature_join import load_feature_source
from _sales_repair import apply_verified_annual_sales_corrections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FIG = os.path.join(BASE, "assets/analysis")
PROC = os.path.join(BASE, "data", "processed", "product")
ANNUAL_REPAIR_AUDIT = os.path.join(
    BASE, "data", "processed", "data_quality", "annual_sales_repair_audit.csv"
)
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

YEAR_MIN = 2022
N_SPLITS = 5

# 排除: 标识列 / 目标 / 市场结果价(内生, 非产品属性) / 冗余文本
DROP = {
    "pcauto_series_id", "series_id", "series_name", "car_id", "car_name",
    "annual_sales", "annual_sales_raw", "annual_sales_repair_applied",
    "official_price_str", "owner_price", "dealer_price",
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


def complete_calendar_years(sales):
    """Return years backed by all twelve natural months in the sales source."""
    source = sales.copy()
    if "date" in source.columns:
        source["date"] = pd.to_datetime(source["date"], errors="raise")
    else:
        source["date"] = pd.to_datetime(
            dict(year=source["year"], month=source["month"], day=1),
            errors="raise",
        )
    coverage = source.assign(year=source["date"].dt.year).groupby("year")["date"].agg(
        lambda values: values.dt.month.nunique()
    )
    return tuple(int(year) for year in coverage.index[coverage.eq(12)] if year >= YEAR_MIN)


def feature_columns(df):
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

    return num_cols, cat_cols


def _one_hot(train, test, prefix):
    train_encoded = pd.get_dummies(train, prefix=prefix).astype(float)
    test_encoded = pd.get_dummies(test, prefix=prefix).astype(float)
    return train_encoded, test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)


def transform_blocks(train_df, test_df, num_cols, cat_cols):
    train_index = train_df.index
    test_index = test_df.index

    train_num = train_df[num_cols].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[num_cols].apply(pd.to_numeric, errors="coerce")
    medians = train_num.median()
    train_num = train_num.fillna(medians).fillna(0.0).astype(float)
    test_num = test_num.fillna(medians).fillna(0.0).astype(float)

    train_cfg_parts = [train_num]
    test_cfg_parts = [test_num]
    train_brand = pd.DataFrame(index=train_index)
    test_brand = pd.DataFrame(index=test_index)
    for column in cat_cols:
        train_values = train_df[column].fillna("NA").astype(str)
        test_values = test_df[column].fillna("NA").astype(str)
        if column == "brand_name":
            train_brand, test_brand = _one_hot(train_values, test_values, "brand")
        elif train_values.nunique() <= 15:
            train_part, test_part = _one_hot(train_values, test_values, column)
            train_cfg_parts.append(train_part)
            test_cfg_parts.append(test_part)
        else:
            frequency = train_values.value_counts(normalize=True)
            train_cfg_parts.append(train_values.map(frequency).rename(f"{column}_freq").to_frame())
            test_cfg_parts.append(
                test_values.map(frequency).fillna(0.0).rename(f"{column}_freq").to_frame()
            )

    train_year, test_year = _one_hot(
        train_df["year"].astype(int), test_df["year"].astype(int), "year"
    )
    train_blocks = {
        "year": train_year,
        "brand": train_brand,
        "config": pd.concat(train_cfg_parts, axis=1),
    }
    test_blocks = {
        "year": test_year,
        "brand": test_brand,
        "config": pd.concat(test_cfg_parts, axis=1),
    }
    return train_blocks, test_blocks


def assemble(blocks, names):
    return pd.concat([blocks[name] for name in names], axis=1)


def cv_eval(df, y, groups, label, block_names, num_cols, cat_cols, full_feature_count):
    """Fit preprocessing and the model within each grouped fold."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    r2s, wms, oof = [], [], np.zeros(len(y))
    for tr, te in gkf.split(df, y, groups):
        train_blocks, test_blocks = transform_blocks(
            df.iloc[tr], df.iloc[te], num_cols, cat_cols
        )
        X_train = assemble(train_blocks, block_names)
        X_test = assemble(test_blocks, block_names)
        m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=1.0, random_state=42, n_jobs=1)
        m.fit(X_train, y[tr])
        p = m.predict(X_test)
        oof[te] = p
        r2s.append(r2_score(y[te], p))
        wms.append(wmape(np.expm1(y[te]), np.maximum(np.expm1(p), 0)))
    print(f"  {label:14s} R2(log)={np.mean(r2s):6.3f}  WMAPE={np.mean(wms):7.1f}%  维度={full_feature_count}")
    return {"variant": label, "R2_log_mean": np.mean(r2s), "R2_log_std": np.std(r2s),
            "WMAPE_mean": np.mean(wms), "WMAPE_fold_std": np.std(wms),
            "WMAPE_oof_global": wmape(np.expm1(y), np.maximum(np.expm1(oof), 0)),
            "n_features": full_feature_count}, oof


def cv_naive_baselines(df, y, groups):
    """Compute leakage-safe annual-sales baselines for WMAPE context."""
    raw_y = np.expm1(y)
    gkf = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for label in ("GLOBAL_MEDIAN", "YEAR_MEDIAN"):
        fold_scores, oof = [], np.zeros(len(y), dtype=float)
        for tr, te in gkf.split(df, y, groups):
            train_raw = pd.Series(raw_y[tr], index=tr)
            if label == "GLOBAL_MEDIAN":
                pred = np.repeat(float(train_raw.median()), len(te))
            else:
                train_year = df.iloc[tr][["year"]].copy()
                train_year["target"] = raw_y[tr]
                medians = train_year.groupby("year")["target"].median()
                pred = df.iloc[te]["year"].map(medians).fillna(train_raw.median()).to_numpy()
            pred = np.maximum(pred, 0.0)
            oof[te] = pred
            fold_scores.append(wmape(raw_y[te], pred))
        rows.append({
            "method": label,
            "WMAPE_mean": float(np.mean(fold_scores)),
            "WMAPE_fold_std": float(np.std(fold_scores)),
            "WMAPE_oof_global": float(wmape(raw_y, oof)),
        })
    return pd.DataFrame(rows)


def main():
    df = load_feature_source()
    sales = pd.read_csv(SALES, low_memory=False)
    df, annual_repair_audit = apply_verified_annual_sales_corrections(df, sales)
    annual_repair_audit.to_csv(ANNUAL_REPAIR_AUDIT, index=False, encoding="utf-8-sig")
    print(f"[归因] 年度销量覆盖: {len(annual_repair_audit)} 行 / "
          f"{annual_repair_audit['series_name'].nunique()} 个车系 / "
          f"{annual_repair_audit['sales_delta'].sum():+,.0f} 辆")
    eligible_years = complete_calendar_years(sales)
    if not eligible_years:
        raise ValueError("No complete calendar year is available for annual attribution")
    df = df[df["annual_sales"].notna() & df["year"].isin(eligible_years)].copy()
    df = df.reset_index(drop=True)
    y = np.log1p(df["annual_sales"].astype(float).values)
    groups = df["series_name"].astype(str).values
    print(f"[归因] 面板 {len(df)} 行 (车系×年) | 车系 {df['series_name'].nunique()} | "
          f"完整年份 {eligible_years[0]}-{eligible_years[-1]}")
    print(f"[归因] y=log1p(annual_sales), 分组切分 GroupKFold({N_SPLITS}) by series_name\n")

    num_cols, cat_cols = feature_columns(df)
    full_blocks, _ = transform_blocks(df, df, num_cols, cat_cols)
    print(f"[特征] 数值配置 {len(num_cols)} | 类别配置 {len(cat_cols) - 1} | "
          f"品牌 one-hot {full_blocks['brand'].shape[1]} | 年份 {full_blocks['year'].shape[1]}")
    print(f"[特征] 配置块合计 {full_blocks['config'].shape[1]} 维\n")

    print("===== 消融 (GroupKFold 交叉验证均值) =====")
    variants = {
        "YEAR-ONLY": ("year",),
        "+BRAND": ("year", "brand"),
        "+CONFIG": ("year", "brand", "config"),
        "CONFIG-ONLY": ("config",),
    }
    rows = []
    for lab, block_names in variants.items():
        full_feature_count = assemble(full_blocks, block_names).shape[1]
        res, _ = cv_eval(
            df, y, groups, lab, block_names, num_cols, cat_cols, full_feature_count
        )
        if res:
            rows.append(res)
    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(PROC, "config_attribution_ablation.csv"), index=False)

    baselines = cv_naive_baselines(df, y, groups)
    baselines.to_csv(os.path.join(PROC, "config_attribution_baselines.csv"), index=False)
    print("\n===== 无模型年度基准（仅用于 WMAPE 口径参照） =====")
    for _, row in baselines.iterrows():
        print(f"  {row['method']:14s} WMAPE={row['WMAPE_oof_global']:7.1f}% "
              f"(fold mean {row['WMAPE_mean']:7.1f}%)")

    r_b = abl.loc[abl["variant"] == "+BRAND", "R2_log_mean"].iloc[0]
    r_c = abl.loc[abl["variant"] == "+CONFIG", "R2_log_mean"].iloc[0]
    print(f"\n[配置增量贡献] R2: {r_b:.3f} -> {r_c:.3f}  (ΔR2 = {r_c - r_b:+.3f})")
    summary = {
        "schema_version": "v1",
        "target_definition": "complete-calendar-year series sales",
        "eligible_years": list(eligible_years),
        "rows": int(len(df)),
        "series": int(df["series_name"].nunique()),
        "cv": f"GroupKFold({N_SPLITS}) by series_name",
        "brand_r2_log_mean": float(r_b),
        "config_r2_log_mean": float(r_c),
        "config_incremental_r2": float(r_c - r_b),
        "config_wmape_oof_global": float(
            abl.loc[abl["variant"].eq("+CONFIG"), "WMAPE_oof_global"].iloc[0]
        ),
    }
    with open(os.path.join(PROC, "config_attribution_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    # ---- 全量拟合一次, 取重要性 + SHAP ----
    X_full = assemble(full_blocks, variants["+CONFIG"])
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
    fig.suptitle("配置与年销量差异（车系×年面板，between-series 变异）", fontsize=12)
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
        print("\n[SHAP] assets/analysis/config_attribution_shap.png")
    except Exception as e:
        print(f"\n[SHAP] 跳过: {e}")

    print("[归因] 完成。")


if __name__ == "__main__":
    main()

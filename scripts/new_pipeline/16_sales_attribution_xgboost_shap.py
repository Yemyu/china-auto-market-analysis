#!/usr/bin/env python3
"""Stage 4 - 16: 车型级「配置→销量」归因 (XGBoost + SHAP)

读取 data/raw/feature.xlsx（车型 trim × 年，已含筛选后的配置特征与
annual_sales 年度销量），用配置特征预测年度销量，并用 SHAP 量化每个
配置维度对销量的贡献方向与大小。

数据粒度说明：
- feature.xlsx 中 销量(annual_sales) 与 配置 同为【车型级】，已在
  同一张表内对齐，无需再做 ID 拼接（区别于旧的车系级月度销量表）。
- 销量是【年度】而非月度，本脚本做的是横截面归因（影响因子分析），
  不是 07/10 的月度时序预测。

泄露防护（用户要求「时序部分注意泄漏」）：
- 默认 GroupShuffleSplit(by=series_id)：测试集为【训练时未出现过的车系】，
  杜绝同车系多个车型被随机切分到训练/测试两侧造成的「组泄露」。
- --split temporal：按 year 中位数滚动切分（早年份训练、晚年份测试），
  杜绝时间泄露。
"""
import os
import sys
import argparse
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"  # 环境铁律：避免 XGBoost 多线程段错误

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import shap

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import _font_setup  # noqa

RAW = BASE_DIR / "data" / "raw" / "feature.xlsx"
SHEET = "power_features_with_sales_v2"
STAGE4 = BASE_DIR / "data" / "processed_new" / "stage4"
FIG = BASE_DIR / "figures_new"
FIG.mkdir(exist_ok=True)
STAGE4.mkdir(parents=True, exist_ok=True)

TARGET = "annual_sales"
ID_COLS = ["car_id", "series_id", "pcauto_series_id", "car_name", "series_name"]
CAT_COLS = ["brand_name"]  # 类别特征统一做 one-hot


def load():
    df = pd.read_excel(RAW, sheet_name=SHEET)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cfg = [c for c in num_cols if c not in [TARGET, "year"] + ID_COLS]
    cat_cfg = [c for c in CAT_COLS if c in df.columns]
    return df, num_cfg, cat_cfg


def build_features(df, num_cfg, cat_cfg):
    Xnum = df[num_cfg].apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.fillna(Xnum.median()).reset_index(drop=True)
    if cat_cfg:
        Xcat = pd.get_dummies(df[cat_cfg], drop_first=True).reset_index(drop=True)
        X = pd.concat([Xnum, Xcat], axis=1)
    else:
        X = Xnum
    return X


def split_data(X, y, groups, year, mode):
    if mode == "temporal":
        med = year.median()
        tr = (year <= med).values
        return X[tr], X[~tr], y[tr], y[~tr], tr, ~tr
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["group", "temporal"], default="group",
                    help="group=按车系分组防组泄露(默认); temporal=按年份滚动防时间泄露")
    args = ap.parse_args()

    df, num_cfg, cat_cfg = load()
    print(f"样本: {len(df)} 行 (车型×年) | 车系 {df['series_id'].nunique()} | "
          f"数值配置特征 {len(num_cfg)} | 类别 {cat_cfg}")
    y = np.log1p(df[TARGET].values)
    X = build_features(df, num_cfg, cat_cfg)
    groups = df["series_id"].astype(str).values
    year = df["year"] if "year" in df.columns else pd.Series([0] * len(df))

    Xtr, Xte, ytr, yte, trm, tem = split_data(X, pd.Series(y), groups, year, args.split)
    print(f"切分[{args.split}] 训练 {len(Xtr)} / 测试 {len(Xte)}")

    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             n_jobs=1, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    mape = mean_absolute_percentage_error(yte, pred)
    print(f"[{args.split}] R2={r2:.3f}  MAPE={mape*100:.1f}%")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    plt.figure()
    shap.summary_plot(sv, Xte, show=False)
    plt.tight_layout()
    plt.savefig(FIG / "stage4_shap_summary.png", dpi=120)
    plt.close()
    plt.figure()
    shap.summary_plot(sv, Xte, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG / "stage4_shap_bar.png", dpi=120)
    plt.close()
    pd.DataFrame(sv, columns=Xte.columns).to_csv(STAGE4 / "shap_values_trim.csv", index=False)

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(STAGE4 / "config_importance.csv")
    print("\nTop 配置特征重要性 (车型级归因):")
    for k, v in imp.head(15).items():
        print(f"  {k:30s}: {float(v):.1f}")

    pd.DataFrame({"split": [args.split], "R2": [r2], "MAPE": [mape]}).to_csv(
        STAGE4 / "config_attribution_metrics.csv", index=False)
    print(f"\n输出: {FIG / 'stage4_shap_summary.png'}, {STAGE4 / 'config_importance.csv'}")


if __name__ == "__main__":
    main()

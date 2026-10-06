#!/usr/bin/env python3
"""Build the Chinese and English report notebooks."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook"


def md(value):
    return nbf.v4.new_markdown_cell(dedent(value).strip() + "\n")


def code(value):
    return nbf.v4.new_code_cell(dedent(value).strip() + "\n")


SETUP = r"""
%matplotlib inline
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from china_auto_market.forecasting.reporting import forecast_report, prediction_metrics
DATA = ROOT / "data"
FORECAST_DIR = DATA / "processed" / "forecast"
PRODUCT_DIR = DATA / "processed" / "product"
FEEDBACK_DIR = DATA / "processed" / "user_feedback"
REVIEW_DIR = DATA / "reviews" / "processed"

plt.rcParams.update({
    "figure.figsize": (10, 5), "figure.dpi": 120,
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
                        "Microsoft YaHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dfe4e8", "grid.linewidth": 0.7,
    "axes.facecolor": "#fbfbfa", "figure.facecolor": "white",
})
COLORS = {"navy": "#162334", "blue": "#316fbd", "light": "#9bb9dd",
          "orange": "#d9902f", "red": "#c75357", "gray": "#8b98a7"}

def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)

def bilingual_names(pairs):
    return {key: labels[0 if ZH else 1] for key, labels in pairs.items()}

MODEL_NAMES = bilingual_names({
    "pred": ("季节增强 XGBoost", "Seasonal XGBoost"),
    "SEASONAL_D5": ("季节增强 XGBoost", "Seasonal XGBoost"),
    "BASE": ("基础 XGBoost", "Base XGBoost"),
    "NAIVE": ("朴素基准", "Naive baseline"),
    "LAST_VALUE": ("最近一次销量", "Last observed sales"),
    "ROLLING_MEAN_3": ("近 3 月均值", "Trailing 3-month mean"),
    "ROLLING_MEAN_6": ("近 6 月均值", "Trailing 6-month mean"),
    "ROLLING_MEAN_12": ("近 12 月均值", "Trailing 12-month mean"),
    "SEASONAL_LAG12": ("去年同期销量", "Same month last year"),
    "PLATFORM_RATING_FIXED": ("XGBoost + 平台评分", "XGBoost + platform ratings"),
    "LOCAL_LEXICON_FIXED": ("XGBoost + 本地词典", "XGBoost + local lexicon"),
    "REVIEW_TEXT_FIXED": ("XGBoost + 文本口碑", "XGBoost + text review features"),
    "REVIEW_RICH_FIXED": ("XGBoost + 扩展口碑特征", "XGBoost + extended review features"),
    "ALL_SENTIMENT_FIXED": ("XGBoost + 组合口碑特征", "XGBoost + combined review features"),
    "REVIEW_TEXT_ROLLING": ("XGBoost + 逐月更新评论", "XGBoost + monthly review updates"),
    "GLOBAL_MEDIAN": ("全局中位数", "Global median"),
    "YEAR_MEDIAN": ("分年份中位数", "Year-specific median"),
})
TABLE_COLUMNS = bilingual_names({
    "method": ("方法", "Method"), "version": ("模型", "Model"),
    "mode": ("预测方式", "Forecast protocol"), "origin": ("预测起点", "Forecast origin"),
    "month": ("月份", "Month"), "group": ("分组", "Group"),
    "series": ("车系数", "Series"), "rows": ("车系月数", "Series-months"),
    "zero_actual_rows": ("零销量车系月数", "Zero-sales rows"),
    "mape_rows": ("MAPE 样本数", "MAPE rows"),
    "mae": ("MAE（辆）", "MAE (vehicles)"),
    "rmse": ("RMSE（辆）", "RMSE (vehicles)"),
    "median_absolute_error": ("绝对误差中位数（辆）", "Median absolute error (vehicles)"),
    "p90_absolute_error": ("绝对误差 P90（辆）", "P90 absolute error (vehicles)"),
    "wmape": ("WMAPE (%)", "WMAPE (%)"),
    "mape_positive": ("正销量 MAPE (%)", "Positive-sales MAPE (%)"),
    "smape": ("sMAPE (%)", "sMAPE (%)"),
    "mean_error": ("平均有符号误差（辆）", "Mean signed error (vehicles)"),
    "net_bias_pct": ("净偏差 (%)", "Net bias (%)"),
    "gain_pp": ("WMAPE 改善（百分点）", "WMAPE improvement (pp)"),
    "monthly_aggregate_wmape": ("月度汇总 WMAPE (%)", "Monthly aggregate WMAPE (%)"),
    "six_month_net_bias_pct": ("半年累计净偏差 (%)", "Six-month net bias (%)"),
    "method_type": ("方法类别", "Method type"), "scenario": ("信息条件", "Information scenario"),
    "global_volume_weighted_WMAPE": ("全局 WMAPE (%)", "Global WMAPE (%)"),
    "median_per_series_WMAPE": ("逐车系中位数 WMAPE (%)", "Median series WMAPE (%)"),
    "validation_selected_n_estimators": ("树数", "Trees"),
    "fixed_origin_validation_global_WMAPE": ("验证 WMAPE (%)", "Validation WMAPE (%)"),
    "historical_origins": ("历史起点数", "Historical origins"),
    "pooled_global_WMAPE": ("合并 WMAPE (%)", "Pooled WMAPE (%)"),
    "n_estimators": ("树数", "Trees"), "max_depth": ("最大树深", "Maximum depth"),
    "learning_rate": ("学习率", "Learning rate"), "subsample": ("行采样比例", "Row sampling fraction"),
    "colsample_bytree": ("列采样比例", "Column sampling fraction"),
    "min_child_weight": ("最小子节点权重", "Minimum child weight"),
    "reg_lambda": ("L2 正则", "L2 regularization"), "reg_alpha": ("L1 正则", "L1 regularization"),
    "R2_log_mean": ("对数销量 R² 均值", "Mean log-sales R²"),
    "R2_log_std": ("折间标准差", "Fold standard deviation"),
    "WMAPE_oof_global": ("合并折外 WMAPE (%)", "Pooled OOF WMAPE (%)"),
    "WMAPE_mean": ("折均值 WMAPE (%)", "Mean fold WMAPE (%)"),
    "WMAPE_fold_std": ("WMAPE 折间标准差", "WMAPE fold standard deviation"),
    "n_features": ("特征数", "Features"),
    "feature": ("配置特征", "Specification feature"), "gain": ("分裂收益", "Split gain"),
    "series_name": ("车系", "Series"), "brand": ("品牌", "Brand"),
    "current_reviews": ("当前窗口评论数", "Current-window reviews"),
    "current_overall_score": ("文本综合倾向", "Text sentiment score"),
    "score_change": ("文本倾向变化", "Text sentiment change"),
    "platform_rating_change": ("平台评分变化", "Platform rating change"),
    "text_rule_retrigger_probability": ("文本规则重触发比例", "Text-rule bootstrap share"),
    "rating_decline_probability": ("评分下降重采样比例", "Rating-decline bootstrap share"),
    "alert_status": ("状态", "Status"), "worst_aspect": ("主要维度", "Leading aspect"),
    "risk_level": ("文本规则等级", "Text-rule level"),
})
VALUE_NAMES = bilingual_names({
    "ROLLING_ONE_MONTH": ("滚动单月", "Rolling one-month"),
    "NAIVE": ("朴素基准", "Naive baseline"),
    "model": ("模型", "Model"), "naive": ("朴素基准", "Naive baseline"),
    "fixed_origin_primary": ("固定起点", "Fixed origin"),
    "rolling_origin_supplement": ("评论逐月更新", "Monthly review updates"),
    "zero": ("零销量", "Zero sales"), "positive": ("正销量", "Positive sales"),
    "watchlist": ("观察名单", "Watchlist"),
    "corroborated": ("双信号预警", "Dual-signal alert"), "none": ("无预警", "No alert"),
    "high": ("高", "High"), "medium": ("中", "Medium"), "low": ("低", "Low"),
})
ASPECT_NAMES = bilingual_names({
    "space": ("空间", "Space"), "power": ("动力", "Powertrain performance"),
    "control": ("操控", "Handling"), "comfort": ("舒适性", "Comfort"),
    "fuel_consumption": ("能耗 / 油耗", "Energy / fuel use"),
    "configuration": ("配置", "Equipment"), "intelligence": ("智能化", "Smart features"),
    "value": ("性价比", "Value for money"), "appearance": ("外观", "Exterior"),
    "interior": ("内饰", "Interior"), "overall": ("综合评价", "Overall"),
})

def presentation_table(frame):
    shown = frame.copy()
    for column in ("method", "version"):
        if column in shown:
            shown[column] = shown[column].replace(MODEL_NAMES)
    for column in ("mode", "method_type", "scenario", "group", "alert_status", "risk_level"):
        if column in shown:
            shown[column] = shown[column].replace({**MODEL_NAMES, **VALUE_NAMES})
    if "worst_aspect" in shown:
        shown["worst_aspect"] = shown["worst_aspect"].replace(ASPECT_NAMES)
    shown = shown.rename(columns={**MODEL_NAMES, **TABLE_COLUMNS},
                         index={**MODEL_NAMES, **TABLE_COLUMNS})
    shown.index.name = TABLE_COLUMNS.get(shown.index.name, shown.index.name)
    shown.columns.name = TABLE_COLUMNS.get(shown.columns.name, shown.columns.name)
    return shown
"""


LOAD = r"""
sales = pd.read_csv(DATA / "processed" / "sales_filtered_24m.csv")
specs_csv = DATA / "raw" / "feature.csv"
specs = pd.read_csv(specs_csv) if specs_csv.exists() else pd.read_excel(DATA / "raw" / "feature.xlsx")
train = pd.read_csv(DATA / "processed" / "splits" / "train.csv")
val = pd.read_csv(DATA / "processed" / "splits" / "val.csv")
test = pd.read_csv(DATA / "processed" / "splits" / "test.csv")
split_manifest = read_json(DATA / "processed" / "splits" / "manifest.json")
corpus = read_json(REVIEW_DIR / "target_371_review_corpus_summary.json")
labels = read_json(REVIEW_DIR / "review_aspect_labels_summary.json")
temporal = read_json(REVIEW_DIR / "review_feature_temporal_summary.json")
forecast = pd.read_csv(FORECAST_DIR / "review_feature_ablation_summary.csv", encoding="utf-8-sig")
benchmark = pd.read_csv(FORECAST_DIR / "forecast_benchmark_comparison.csv", encoding="utf-8-sig")
rolling_summary = read_json(FORECAST_DIR / "rolling_origin_summary.json")
rolling_test = pd.read_csv(FORECAST_DIR / "rolling_origin_test_predictions.csv", encoding="utf-8-sig")
report = forecast_report(ROOT)
robustness = read_json(FORECAST_DIR / "forecast_robustness_summary.json")
model_run = read_json(FORECAST_DIR / "review_feature_run_summary.json")
selected_version = model_run["validation_selected_primary_version"]
selected_fixed = benchmark.loc[benchmark.method.eq(selected_version)].iloc[0]
config = pd.read_csv(PRODUCT_DIR / "config_attribution_ablation.csv")
config_summary = read_json(PRODUCT_DIR / "config_attribution_summary.json")
aspects = pd.read_csv(FEEDBACK_DIR / "user_need_aspect_summary.csv", encoding="utf-8-sig")
alerts = pd.read_csv(FEEDBACK_DIR / "sentiment_alerts.csv")
monitor = read_json(FEEDBACK_DIR / "user_needs_alerts_summary.json")
"""


SAMPLES = r"""
if ZH:
    table = pd.DataFrame([
        ["滚动单月销量预测", "371 个车系", "2,226 条测试车系月", "每月更新下月预测"],
        ["固定六个月压力测试", "371 个车系", "2,226 条测试车系月", "固定起点递归预测"],
        ["产品配置分析", f"{config_summary['series']} 个车系",
         f"{config_summary['rows']:,} 条完整车系年记录", "GroupKFold(5) 按车系"],
        ["用户需求与风险", "345 个车系", "24,175 条评论", "质量审计 + 180 天窗口"],
    ], columns=["分析", "样本", "观测", "验证"])
    source = pd.DataFrame([
        ["月销量", f"{sales.series_name.nunique():,} 个车系 / {len(sales):,} 行"],
        ["车型配置", f"{specs.series_name.nunique():,} 个车系 / {len(specs):,} 行"],
        ["严格评论语料", f"{corpus['temporally_eligible_reviews']:,} 条 / "
                         f"{corpus['target_series_with_any_review']} 个车系"],
    ], columns=["来源数据", "规模"])
else:
    table = pd.DataFrame([
        ["Rolling one-month sales forecast", "371 series", "2,226 test series-months",
         "Monthly refresh, one month ahead"],
        ["Fixed six-month stress test", "371 series", "2,226 test series-months",
         "Fixed-origin recursive forecast"],
        ["Product specifications", f"{config_summary['series']} series",
         f"{config_summary['rows']:,} complete series-year records",
         "Five-fold GroupKFold by series"],
        ["User needs and risk", "345 series", "24,175 reviews",
         "Quality audit + 180-day windows"],
    ], columns=["Analysis", "Sample", "Observations", "Validation"])
    source = pd.DataFrame([
        ["Monthly sales", f"{sales.series_name.nunique():,} series / {len(sales):,} rows"],
        ["Vehicle specifications", f"{specs.series_name.nunique():,} series / {len(specs):,} rows"],
        ["Strict review corpus", f"{corpus['temporally_eligible_reviews']:,} reviews / "
                                 f"{corpus['target_series_with_any_review']} series"],
    ], columns=["Source data", "Scale"])
display(table)
display(source)
"""


TIME = r"""
panel = pd.concat([
    train[["date", "monthly_sales", "split"]],
    val[["date", "monthly_sales", "split"]],
    test[["date", "monthly_sales", "split"]],
], ignore_index=True)
panel["date"] = pd.to_datetime(panel["date"])
monthly = panel.groupby(["date", "split"], as_index=False)["monthly_sales"].sum()
fig, ax = plt.subplots(figsize=(11, 4.8))
split_names = {"train": "训练", "val": "验证", "test": "测试"} if ZH else {
    "train": "Training", "val": "Validation", "test": "Test"}
for split, color in [("train", COLORS["blue"]), ("val", COLORS["orange"]), ("test", COLORS["red"])]:
    part = monthly[monthly["split"] == split]
    ax.plot(part["date"], part["monthly_sales"] / 1e6, color=color, lw=2.1, label=split_names[split])
    ax.axvspan(part["date"].min(), part["date"].max(), color=color, alpha=0.06)
ax.set(title=("371 车系月销量与两种预测协议的时间切分" if ZH else
              "Monthly sales and the two forecast protocols: 371 series"),
       xlabel="", ylabel=("月销量（百万辆）" if ZH else "Monthly sales (million)"))
ax.legend(frameon=False, ncol=3)
plt.show()
"""


MODELS = r"""
model_name = "方案" if ZH else "Model"
global_name = "全局 WMAPE" if ZH else "Global WMAPE"
median_name = "逐车系中位数 WMAPE" if ZH else "Median per-series WMAPE"
def wmape(frame, prediction):
    return (frame["actual"] - frame[prediction]).abs().sum() / frame["actual"].abs().sum() * 100

def median_wmape(frame, prediction):
    values = frame.groupby("series_name").apply(
        lambda x: (x["actual"] - x[prediction]).abs().sum() / x["actual"].abs().sum() * 100
        if x["actual"].abs().sum() else np.nan
    ).dropna()
    return values.median()

rolling_rows = [
    ("沿用上月销量（朴素）" if ZH else "Last observed value (naive)", "LAST_VALUE"),
    ("近3月均值（朴素）" if ZH else "Trailing 3-month mean (naive)", "ROLLING_MEAN_3"),
    ("近6月均值（朴素）" if ZH else "Trailing 6-month mean (naive)", "ROLLING_MEAN_6"),
    ("去年同期销量（朴素）" if ZH else "Same-month-last-year (naive)", "SEASONAL_LAG12"),
    ("滚动单月季节增强 XGBoost（主模型）" if ZH else "Rolling one-month seasonal XGBoost", "pred"),
]
rolling_table = pd.DataFrame([
    [label, wmape(rolling_test, column), median_wmape(rolling_test, column)]
    for label, column in rolling_rows
], columns=[model_name, global_name, median_name])
display(rolling_table.style.format({global_name: "{:.2f}%", median_name: "{:.2f}%"}))
scores = pd.DataFrame(report["metrics"]).set_index("method")
display(presentation_table(scores[["mae", "rmse", "median_absolute_error", "p90_absolute_error",
                                  "mape_positive", "mape_rows", "smape"]]).round(2))
fixed_row = pd.DataFrame([[
    "固定六个月平台评分模型（压力测试）" if ZH else "Fixed six-month platform-rating model (stress test)",
    selected_fixed["global_volume_weighted_WMAPE"], selected_fixed["median_per_series_WMAPE"],
]], columns=[model_name, global_name, median_name])
display(fixed_row.style.format({global_name: "{:.2f}%", median_name: "{:.2f}%"}))
ax = rolling_table.set_index(model_name).plot.barh(
    color=[COLORS["blue"], COLORS["light"]], width=0.72, figsize=(10, 4.8))
ax.set(title=("滚动单月测试集误差" if ZH else "Rolling one-month test error"),
       xlabel="WMAPE (%)", ylabel="")
ax.legend(frameon=False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
ax.set_xlim(0, 115)
fig = ax.get_figure()
fig.tight_layout()
plt.show()
"""


HISTORICAL = r"""
origins = pd.read_csv(FORECAST_DIR / "rolling_origin_validation.csv")
comparison = origins[origins["mode"].eq("ROLLING_ONE_MONTH")].pivot(
    index="origin", columns="version", values="global_volume_weighted_WMAPE")
comparison["gain_pp"] = comparison["BASE"] - comparison["SEASONAL_D5"]
display(presentation_table(comparison).round(3))
display(presentation_table(pd.DataFrame(rolling_summary["model_params"]).T))
display(presentation_table(pd.DataFrame(rolling_summary["validation_summary"])[[
    "version", "mode", "historical_origins", "pooled_global_WMAPE"]]).round(3))
assert rolling_summary["test_used_for_selection"] is False
assert np.isclose(scores.loc["pred", "wmape"],
                  rolling_summary["locked_test"]["global_volume_weighted_WMAPE"])
"""


ERRORS = r"""
monthly_scores = pd.DataFrame(report["monthly"])
monthly_comparison = monthly_scores.pivot(index="month", columns="method", values="wmape")
monthly_comparison["gain_pp"] = monthly_comparison["LAST_VALUE"] - monthly_comparison["pred"]
display(presentation_table(monthly_comparison).round(2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
presentation_table(monthly_comparison[["LAST_VALUE", "pred"]]).plot(
    ax=axes[0], marker="o", color=[COLORS["gray"], COLORS["blue"]])
axes[0].set(title=("逐月车系误差" if ZH else "Series-level error by month"),
            ylabel="WMAPE (%)", xlabel="")
presentation_table(monthly_scores.pivot(index="month", columns="method", values="net_bias_pct")).plot(
    ax=axes[1], marker="o", color=[COLORS["gray"], COLORS["blue"]])
axes[1].axhline(0, color=COLORS["red"], lw=1)
axes[1].set(title=("逐月汇总净偏差" if ZH else "Monthly aggregate net bias"),
            ylabel=("(预测 − 实际) / 实际 (%)" if ZH else
                    "(prediction − actual) / actual (%)"), xlabel="")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False)
fig.tight_layout()
plt.show()

segments = pd.DataFrame(report["segments"])
display(presentation_table(segments[segments.group_type.eq("pre_test_size")][[
    "group", "method", "series", "rows", "wmape", "mae", "smape"]]).round(2))
display(presentation_table(segments[segments.group_type.eq("actual_sales")][[
    "group", "method", "rows", "wmape", "mae", "smape"]]).round(2))
display(presentation_table(pd.DataFrame(report["aggregate"]).set_index("method")).round(2))
"""


FIXED = r"""
display(presentation_table(benchmark[["method", "method_type", "global_volume_weighted_WMAPE",
                                     "median_per_series_WMAPE"]]).round(3))
display(presentation_table(forecast[["version", "scenario", "validation_selected_n_estimators",
                                   "fixed_origin_validation_global_WMAPE", "global_volume_weighted_WMAPE"]]
        .sort_values(["scenario", "fixed_origin_validation_global_WMAPE"])).round(3))
fixed_predictions = pd.read_csv(FORECAST_DIR / "review_feature_predictions.csv")
display(presentation_table(pd.Series(
    prediction_metrics(fixed_predictions.loc[fixed_predictions.version.eq(selected_version)]),
    name=selected_version).to_frame()).round(3))
"""


UNCERTAINTY = r"""
point = robustness["selected_feedback_vs_base_improvement_pp"]
low, high = robustness["selected_feedback_vs_base_bootstrap_95pct_ci_pp"]
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.errorbar(point, 0, xerr=np.array([[point-low], [high-point]]),
            fmt="o", markersize=8, color=COLORS["blue"], capsize=6, lw=2)
ax.axvline(0, color=COLORS["red"], ls="--", lw=1.3)
ax.set(title=("用户口碑增强模型相对销量基线的改善" if ZH else
              "Owner-feedback model improvement over the sales baseline"),
       xlabel=("全局 WMAPE 改善（百分点）" if ZH else
               "Global WMAPE improvement (percentage points)"), yticks=[])
ax.text(point, .08, f"{point:.2f} pp  [95%: {low:.4f}, {high:.4f}]",
        ha="center", va="bottom")
plt.show()
if ZH:
    print(f"重采样中优于基线的比例：{robustness['selected_feedback_vs_base_probability_better']:.1%}")
    print(f"六个测试月中优于基线：{robustness['test_months_selected_feedback_better_than_base']} / 6")
else:
    print("Share of bootstrap replicates better than baseline: "
          f"{robustness['selected_feedback_vs_base_probability_better']:.1%}")
    print(f"Test months better than baseline: {robustness['test_months_selected_feedback_better_than_base']} / 6")
"""


IMPORTANCE = r"""
family = pd.read_csv(FORECAST_DIR / "review_feature_family_importance.csv",
                     encoding="utf-8-sig")
maps = {
    "zh": {"sales_lag_roll": "历史销量", "calendar": "日历", "configuration": "产品配置",
           "platform_rating_scores": "平台评分", "review_expanding_score": "历史评价得分", "review_observation_context": "评论覆盖",
           "review_mention_count": "需求提及量", "review_negative_rate": "负面比例",
           "review_recent_score": "近期评价得分", "review_mention_rate": "需求提及率",
           "review_positive_rate": "正面比例"},
    "en": {"sales_lag_roll": "Sales history", "calendar": "Calendar",
           "configuration": "Product specifications",
           "platform_rating_scores": "Platform ratings", "review_expanding_score": "Historical review score",
           "review_observation_context": "Review coverage",
           "review_mention_count": "Need mentions", "review_negative_rate": "Negative share",
           "review_recent_score": "Recent review score",
           "review_mention_rate": "Mention share", "review_positive_rate": "Positive share"},
}
label = "信息类型" if ZH else "Information"
family[label] = family.feature_family.map(maps["zh" if ZH else "en"])
family = family.dropna(subset=[label]).sort_values("share_of_total_abs_shap")
ax = family.plot.barh(x=label, y="share_of_total_abs_shap", color=COLORS["blue"],
                      legend=False, figsize=(9, 5))
ax.set(title=("特征组的平均绝对 SHAP 占比" if ZH else
              "Mean absolute SHAP share by feature family"),
       xlabel=("占全部绝对 SHAP 的比例" if ZH else "Share of total absolute SHAP"),
       ylabel="")
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.show()
"""


CONFIG = r"""
frame = config.copy()
name = "方案" if ZH else "Model"
frame[name] = frame.variant.map({
    "YEAR-ONLY": "年份" if ZH else "Year",
    "+BRAND": "年份 + 品牌" if ZH else "Year + brand",
    "+CONFIG": "年份 + 品牌 + 配置" if ZH else "Year + brand + specifications",
    "CONFIG-ONLY": "仅配置" if ZH else "Specifications only",
})
wmape_column = "WMAPE_oof_global" if "WMAPE_oof_global" in frame.columns else "WMAPE_mean"
display(presentation_table(frame[[name, "R2_log_mean", "R2_log_std", wmape_column, "n_features"]])
        .style.format({TABLE_COLUMNS["R2_log_mean"]: "{:.3f}",
                       TABLE_COLUMNS["R2_log_std"]: "{:.3f}",
                       TABLE_COLUMNS[wmape_column]: "{:.2f}%"}))
ordered = frame[frame.variant.isin(["YEAR-ONLY", "+BRAND", "+CONFIG"])]
fig, ax = plt.subplots(figsize=(9.5, 4.5))
bars = ax.bar(ordered[name], ordered.R2_log_mean,
              color=[COLORS["gray"], COLORS["light"], COLORS["blue"]])
ax.errorbar(ordered[name], ordered.R2_log_mean, yerr=ordered.R2_log_std,
            fmt="none", ecolor=COLORS["navy"], capsize=5, lw=1.3)
ax.set(title=("年份、品牌与配置的增量解释力" if ZH else
              "Incremental explanatory power of year, brand, and specifications"),
       xlabel="", ylabel="GroupKFold R² — log1p(sales)")
ax.bar_label(bars, fmt="%.3f", padding=4)
plt.show()
display(presentation_table(pd.read_csv(PRODUCT_DIR / "config_attribution_baselines.csv")).round(3))
importance = pd.read_csv(PRODUCT_DIR / "config_importance_annual.csv")
feature_names = bilingual_names({
    "engine_cylinder_arrangement_L": ("气缸排列：L", "Cylinder arrangement: L"),
    "cylinder_material_铝": ("气缸材料：铝", "Cylinder material: aluminium"),
    "steering_wheel_material_真皮": ("方向盘材料：真皮", "Steering wheel: leather"),
    "center_screen_NA": ("中控屏：缺失", "Centre screen: missing"),
    "center_screen_大屏": ("中控屏：大屏", "Centre screen: large"),
    "oil_supply_直喷": ("供油方式：直喷", "Fuel injection: direct"),
    "fuel_grade_95#": ("燃油标号：95#", "Fuel grade: 95#"),
    "engine_intake_type_涡轮增压": ("进气方式：涡轮增压", "Engine aspiration: turbocharged"),
    "door_open_way_平开门": ("车门：平开门", "Doors: hinged"),
    "manufacturer_freq": ("厂商类别频次编码", "Manufacturer frequency encoding"),
    "fast_charge_percent_60.0": ("快充比例：60%", "Fast-charge percentage: 60%"),
    "gearbox_type_freq": ("变速箱类型频次编码", "Gearbox type frequency encoding"),
})
top_features = importance[importance.block.eq("config")].sort_values("gain", ascending=False).head(12)
feature_table = top_features[["feature", "gain"]].copy()
feature_table["feature"] = feature_table["feature"].replace(feature_names)
display(presentation_table(feature_table).round(4))
"""


ASPECTS = r"""
ordered = aspects.sort_values("mention_rate")
fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(ordered))
ax.barh(y-.18, ordered.mention_rate, height=.34, color=COLORS["light"],
        label=("提及率" if ZH else "Mention share"))
ax.barh(y+.18, ordered.negative_rate_among_scored, height=.34, color=COLORS["red"],
        label=("负面率（有效评分内）" if ZH else "Negative share among scored"))
labels_y = ordered.aspect.map(ASPECT_NAMES)
ax.set_yticks(y, labels_y)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
ax.set(title=("十类用户需求：讨论热度与负面集中度" if ZH else
              "Ten user needs: discussion and negative concentration"),
       xlabel=("比例" if ZH else "Share"), ylabel="")
ax.legend(frameon=False, loc="lower right")
plt.show()

top = aspects.sort_values("negative_rate_among_scored", ascending=False).copy()
if ZH:
    top = top[["aspect_zh", "mentioned_reviews", "scored_mentions", "negative_mentions",
               "mention_rate", "negative_rate_among_scored", "mean_polarity"]]
    top.columns = ["维度", "提及评论数", "有效评分数", "负面数", "提及率", "负面率", "平均倾向"]
    display(top.style.format({"提及率": "{:.1%}", "负面率": "{:.1%}", "平均倾向": "{:.3f}"}))
else:
    top = top[["aspect", "mentioned_reviews", "scored_mentions", "negative_mentions",
               "mention_rate", "negative_rate_among_scored", "mean_polarity"]]
    top["aspect"] = top["aspect"].map(ASPECT_NAMES)
    top.columns = ["Dimension", "Mention count", "Scored count", "Negative count",
                   "Mention share", "Negative share", "Mean polarity"]
    display(top.style.format({"Mention share": "{:.1%}", "Negative share": "{:.1%}",
                              "Mean polarity": "{:.3f}"}))
"""


ALERTS = r"""
latest = pd.Timestamp(monitor["latest_completed_monitoring_month"])
current = alerts[pd.to_datetime(alerts.information_cutoff_inclusive).dt.normalize().eq(latest)].copy()
if ZH:
    overview = pd.DataFrame([
        ["最近完整监测月", monitor["latest_completed_monitoring_month"]],
        ["达到样本门槛的车系", monitor["latest_eligible_series"]],
        ["当前双信号预警", monitor["latest_active_alerts"]],
        ["当前观察名单", monitor["latest_watchlist_events"]],
        ["历史预警事件", monitor["historical_alert_events"]],
    ], columns=["项目", "数值"])
else:
    overview = pd.DataFrame([
        ["Latest complete month", monitor["latest_completed_monitoring_month"]],
        ["Eligible series", monitor["latest_eligible_series"]],
        ["Current dual-signal alerts", monitor["latest_active_alerts"]],
        ["Current watchlist", monitor["latest_watchlist_events"]],
        ["Historical alert events", monitor["historical_alert_events"]],
    ], columns=["Item", "Value"])
display(overview)
current_table = current[["series_name", "brand", "current_reviews", "current_overall_score",
                         "score_change", "platform_rating_change", "text_rule_retrigger_probability",
                         "rating_decline_probability", "alert_status", "worst_aspect", "risk_level"]].copy()
if not ZH:
    current_table["brand"] = current_table["brand"].replace({"奥迪": "Audi"})
    current_table["series_name"] = current_table["series_name"].replace({"奥迪Q4 e-tron": "Audi Q4 e-tron"})
formatted_current = presentation_table(current_table)
display(formatted_current.style.format({
    TABLE_COLUMNS["current_overall_score"]: "{:.3f}",
    TABLE_COLUMNS["score_change"]: "{:+.3f}",
    TABLE_COLUMNS["platform_rating_change"]: "{:+.3f}",
    TABLE_COLUMNS["text_rule_retrigger_probability"]: "{:.1%}",
    TABLE_COLUMNS["rating_decline_probability"]: "{:.1%}",
}))
"""


AUDIT = r"""
if ZH:
    audit = pd.DataFrame([
        ["严格评论语料", f"{corpus['temporally_eligible_reviews']:,} 条", "完整正文、时间和来源"],
        ["排除的列表摘要", f"{corpus['autohome_list_summary_rows_excluded_from_temporal_model']} 条",
         "无详情正文，不建模"],
        ["测试起点前有评论", f"{temporal['fixed_test_series_with_any_prior_review']} 个车系",
         "固定压力测试冻结于 2026-01-01 前"],
        ["最近 180 天有评论", f"{temporal['fixed_test_series_with_recent_180d_review']} 个车系",
         "其余保留缺失标记"],
        ["复用历史标签", f"{labels['historical_labeled_reviews']:,} 条", "保留来源与标签限制"],
        ["API 自动补充标签", f"{labels['api_labeled_reviews']:,} 条", "结构校验与人工抽样"],
    ], columns=["审计项", "结果", "处理"])
else:
    audit = pd.DataFrame([
        ["Strict review corpus", f"{corpus['temporally_eligible_reviews']:,} reviews",
         "Complete text, time, and source"],
        ["Excluded list summaries", corpus["autohome_list_summary_rows_excluded_from_temporal_model"],
         "No full detail text; not modeled"],
        ["Review evidence before test origin", f"{temporal['fixed_test_series_with_any_prior_review']} series",
         "Frozen before 2026-01-01 for the fixed stress test"],
        ["Review in prior 180 days", f"{temporal['fixed_test_series_with_recent_180d_review']} series",
         "Missingness retained elsewhere"],
        ["Reused historical labels", f"{labels['historical_labeled_reviews']:,}",
         "Source and label limitations retained"],
        ["Additional API-generated labels", f"{labels['api_labeled_reviews']:,}",
         "Schema checks and manual sampling"],
    ], columns=["Audit item", "Result", "Treatment"])
display(audit)
"""


TEXT = {
    "zh": {
        "title": """# 中国汽车市场分析：销量预测、产品配置与用户需求

本报告使用公开月度销量、车型配置和车主评论，分析下月销量预测、年度产品差异和用户需求。误差指标由已保存的逐行预测计算，历史回测、配置消融和评论监测使用对应的分析结果。

**研究期：** 2022-01—2026-07

**预测测试期：** 2026-01—06

**销量主任务：** 滚动单月预测；固定六个月为压力测试

**主指标：** 全局销量加权 WMAPE""",
        "sample": """## 1. 三组分析样本

销量预测使用固定 371 个车系的完整自然月面板，配置按年份回溯到最近可用记录。产品配置分析要求完整年度销量与配置能够对齐；用户需求分析使用具有完整正文、发布时间和来源记录的评论。""",

        "engineering": """### 报告输入

计算使用仓库中的 CSV 和 JSON，包括销量面板、保存预测、评论标签与统计摘要。完整评论正文仅保存在本地，本报告通过标签和汇总结果分析评论。文件清单、分析入口及环境要求见[数据说明](../data/README.md)。""",

        "forecast": """## 2. 月度销量预测

开发切分为训练截至2025-06、验证2025-07—12、测试2026-01—06。模型方案选定后，使用训练与验证数据合并重新拟合至2025-12；测试六个月内权重固定，每次预测完成后将真实销量加入下一月历史。

滚动协议使用预测时已公布的上月销量；固定六个月协议从 2026-01 起递归预测，后续销量滞后来自先前预测。两种任务分别评价。

基础XGBoost含1/2/3月滞后、3/6月均值、日历和配置；季节方案增加12月滞后和12月均值，同时修改参数。目标均为`log1p(月销量)`，输出还原后截为非负。销量模型在每个预测起点限定可用配置年份，按行连接不晚于其年份的最近记录，并仅在该模型实际训练行中拟合中位数与类别词表；预测窗口内配置冻结。未知类别记为−1，全缺数值列使用0占位，不因配置缺失删除销量样本。配置源没有年内发布时间，销量源也没有逐条历史发布与修订版本，因此年度代理对齐仍不等于完整历史点时重建。年度配置模块使用独立的折内处理。""",

        "models": """### 模型比较

季节增强XGBoost的全局WMAPE为29.75%，上月销量基准为40.99%，相差11.24个百分点，绝对误差相对减少27.4%。全部方法使用相同2,226行，包括262个真实零销量月份。

| 指标 | 计算与含义 |
|---|---|
| WMAPE | 总绝对误差 / 总真实销量，以百分数显示 |
| MAE / RMSE | 平均绝对误差 / 均方根误差，单位为辆；RMSE更受大误差影响 |
| 逐车系中位数WMAPE | 先按每个车系六个月总量计算，再取中位数；总量为零的车系无定义 |
| 正销量MAPE | 仅在1,964个真实销量大于零的车系月上平均百分比误差 |
| sMAPE | 全部行平均`200×|实际−预测|/(|实际|+|预测|)`；双方为零时记0 |

同一数据集上 WMAPE 与 MAE 按固定比例换算，分别提供相对误差和车辆数误差。MAPE 对极小销量敏感；sMAPE 保留实际与预测均为零的行，所有方法使用相同样本数。""",

        "historical": """### 历史起点与方案选择

每个历史起点覆盖随后六个月的滚动单月预测。基础与季节增强 XGBoost 在特征和参数上均有差异，表中比较的是完整模型方案。四个起点合并 WMAPE 改善 0.957 个百分点，最差起点误差增加 0.061 个百分点，满足预设的合并改善至少 0.5、单起点退化不超过 1.0 个百分点的选型门槛。这四个窗口用于模型选择。""",

        "errors": """### 误差发生在哪里

模型在六个测试月中的五个月具有更低 WMAPE，5 月略高于上月销量基准。Q1—Q4 按 2025 年 7—12 月平均销量划分，分组在测试前固定。Q1 的 WMAPE 为 66.60%，Q4 为 28.02%，低销量车系仍难预测。

零/正销量分组仅作事后诊断，不能提前用于模型路由。零销量组WMAPE和MAPE没有定义，以MAE、样本数观察其影响。小幅正预测即使只错不到1辆，也会在真实值为零时得到200%的sMAPE，解释了主模型全局sMAPE弱于基准、但MAE更低的现象。

371 个评价车系的半年净偏差为 −0.67%，月度汇总 WMAPE 为 6.46%，逐车系月 WMAPE 为 29.75%。净偏差允许跨月份抵消，月度汇总允许同月车系间抵消，逐车系月绝对误差保留每条记录的偏差。汇总指标描述本评价车系集合。""",

        "fixed": """### 固定六个月：朴素基准与口碑消融

朴素比较包含近 3/6/12 月均值、最近一次销量及去年同期。平台评分模型的 WMAPE 为 38.09%；近 12 月均值的 67.43% 是保存的朴素方法中最低测试 WMAPE，对应绝对误差减少 43.5%。这项参照为测试结果的描述性比较，完整结果列于下表。

口碑实验在相同车系和固定起点协议下加入平台评分、词典、文本或组合特征。各方案使用既定的 100 棵树，主比较沿用已选平台评分方案。评论逐月更新方案只更新评论信息，销量滞后仍由递归预测产生。""",

        "uncertainty": """### 改善幅度与不确定性

口碑增强属于固定六个月压力测试：点估计相对销量基线改善 0.980 个百分点，按车系重采样的 95% 区间为 −0.0047 至 2.2331 个百分点，稳定增益证据不足，因此定位为辅助信息。""",
        "importance": """### 固定场景口碑模型依赖哪些信息

平均绝对 SHAP 描述固定六个月平台评分模型的对数销量预测，各组占比表示其模型归因贡献。相关特征可能分摊贡献；这些数值不表示销量增幅或准确率增益。""",

        "cold": """### 历史不足车系的边界

固定起点前没有正销量历史的 13 个车系使用同一个平台评分模型。该组固定预测 WMAPE 约 99.40%，现有历史与配置不足以可靠预测其销量爬坡。""",
        "config": """## 3. 产品配置与年度销量差异

样本为646个车系、1,510条2022—2025完整车系年记录。五折GroupKFold将整个车系留在同一折，检验对未见车系的推广；它不检验跨未来年份的预测。每一折独立拟合缺失填充和编码，再以XGBoost拟合`log1p(年度销量)`。

R²报告五折对数尺度得分的均值，误差线为折间标准差，不是置信区间。WMAPE使用还原为辆数后的全部折外预测，不取五折WMAPE的简单平均。后面的全局中位数与分年份中位数基准只使用各训练折目标值。""",
        "config_read": """加入品牌后，对数销量 R² 从 0.013 升至 0.070，加入配置后为 0.239，增量为 0.169。完整模型原尺度 WMAPE 为 73.85%，低于分年份中位数的 87.21%，但绝对误差仍较大。

前12项配置重要性使用gain，反映模型分裂收益，不提供效应方向或因果结论。价格、尺寸、动力与品牌定位存在共同变化，车系年汇总又掩盖版本差异。可据此筛选同类产品的比较维度，不能估计单项配置的销量回报。""",
        "needs": """## 4. 用户需求与口碑风险

严格语料包含24,175条评论、345个车系。提及率以全部可用评论为分母；负面率以该维度的有效评分提及为分母。下表同时给出提及数、评分数和负面数。评论可以提及多个维度，因此维度样本不能相加当作独立评论总数。

平台用户是自选择样本，负面率不代表所有车主的不满意率。历史零标签无法完全区分未提及、中性和解析回退，统一提及检测缓解了差异，仍不能等同于人工真值。既有抽查缺少独立代表性金标准，不足以估计标签F1。""",
        "alerts": """### 双信号风险监测

文本规则先生成观察候选。3,000 次 Bootstrap 中，文本规则重触发比例至少为 70%、平台原始评分下降比例至少为 80% 的候选，进入双信号预警名单，供人工复核。

最近完整监测月有 123 个车系达到相邻两个 180 天窗口各至少 5 条评论的门槛，占 371 个目标车系的 33.2%；该范围内无双信号预警，有 1 个观察候选。Bootstrap 衡量重采样下的规则稳定性，不是未来故障概率；尚无独立事件标签可计算预警准确率。""",
        "audit": """## 5. 数据质量与评价范围

评论分析复用历史标签，并以 API 自动生成标签补充未覆盖的评论。标签经过结构校验与人工抽样检查，来源与数量列于下表。

2026 年评价窗口已在开发过程中检查，本报告将其作为回顾性评价。口碑增益区间以已拟合模型和已观察月份为条件，不包含重新训练、方案选择及未来月份的不确定性。缺少配置年内发布时间和销量历史修订记录，也限制了完整历史点时重建。""",
        "dashboard": """## 6. 结果浏览

六页[在线交互看板](https://yemyu.github.io/china-auto-market-analysis/)展示销量预测、产品配置、用户需求与车系详情，读取 `app/static/data/` 中的静态 JSON。页面截图保存在 `assets/dashboard/zh/`；本地浏览与分析环境见[数据说明](../data/README.md)。

![项目概览](../assets/dashboard/zh/01-overview.png)""",

        "conclusion": """## 7. 结论

1. 滚动单月季节增强 XGBoost 的 WMAPE 为 29.75%，相对最近一次销量基准降低 27.4% 的绝对误差；低销量车系仍是主要难点。
2. 固定六个月平台评分模型的 WMAPE 为 38.09%。口碑特征的增量为 0.980 个百分点，95% 区间跨零，尚不足以支持稳定增益。
3. 产品配置将年度对数销量 R² 提高 0.169，适合筛选产品比较维度；汇总数据不足以估计单项配置的销量回报。
4. 评论用于需求分析和风险复核。最近完整监测月覆盖 123 个车系，预警规则仍需独立事件标签验证。""",
    },
    "en": {
        "title": """# China Automotive Market Analysis: Sales Forecasting, Product Specifications, and User Needs

This report uses public monthly sales, vehicle specifications and owner reviews to examine next-month sales, annual product differences and user needs. Error metrics are calculated from saved row-level predictions; historical comparisons, specification ablations and review monitoring use the corresponding analysis results.

**Study period:** 2022-01—2026-07

**Forecast test:** 2026-01—06

**Sales task:** rolling one-month-ahead forecast; fixed six-month stress test

**Primary metric:** global volume-weighted WMAPE""",
        "sample": """## 1. Three analysis samples

Forecasting uses a fixed 371-series calendar-month panel, with specifications carried forward from the most recent available year. Product analysis requires aligned full-year sales and specifications. User-needs analysis uses reviews with complete text, publication dates and source records.""",

        "engineering": """### Report inputs

Calculations use CSV and JSON files in the repository: the sales panel, saved predictions, review labels and statistical summaries. Full review text is stored locally; this report analyses the labels and aggregates. File inventories, analysis commands and environment requirements are in the [data guide](../data/README_EN.md).""",

        "forecast": """## 2. Monthly sales forecasting

The development split is training through June 2025, validation in July–December 2025, and testing in January–June 2026. After selection, the model is refitted on training plus validation through December 2025. Weights stay fixed during the six test months; observed sales enter history for the next forecast.

The rolling protocol uses the previous month's published sales. The fixed six-month protocol starts in January 2026 and feeds predictions into subsequent sales lags. The two tasks are evaluated separately.

Base XGBoost uses 1/2/3-month lags, 3/6-month means, calendar variables and specifications. The seasonal variant adds a 12-month lag and mean and changes parameters. Both fit log1p sales, invert the transform and clip predictions at zero. At each forecast origin, specification records are limited to years available before the origin and joined to rows without looking forward in year. Imputation medians and category vocabularies are fitted only on the model's eligible training rows; specifications stay fixed over the forecast window. Unknown categories use −1, entirely missing numeric columns use 0, and missing specifications do not remove sales observations. Within-year specification release dates and historical sales revisions are unavailable, so annual proxies do not establish complete point-in-time reconstruction. Annual specification analysis uses a separate fold-local pipeline.""",

        "models": """### Model comparison

Seasonal XGBoost has 29.75% WMAPE versus 40.99% for last observed sales: 11.24 points lower, or a 27.4% reduction in absolute error. Every method uses 2,226 rows, including 262 zero-sales months.

| Metric | Definition |
|---|---|
| WMAPE | Total absolute error / total actual sales, shown as a percentage |
| MAE / RMSE | Mean absolute / root mean square error, in vehicles; RMSE gives more weight to large misses |
| Median series WMAPE | Compute six-month WMAPE for each series, then take its median; zero-total series are undefined |
| Positive-sales MAPE | Average percentage error across the 1,964 positive-actual rows |
| sMAPE | Mean of `200×abs(actual−prediction)/(abs(actual)+abs(prediction))`; a zero/zero pair contributes 0 |

On a fixed sample, WMAPE and MAE differ by a constant scale, expressing relative error and error in vehicles respectively. MAPE is sensitive to very small sales. Zero/zero pairs remain in sMAPE, and every method uses the same row count.""",

        "historical": """### Historical origins and model selection

Each origin covers six successive one-month forecasts. Base and seasonal XGBoost differ in both features and parameters, so the comparison evaluates complete model configurations. Pooled WMAPE improves by 0.957 percentage points; the largest increase at an individual origin is 0.061 points. These pass the predefined selection thresholds: a pooled improvement of at least 0.5 points and no origin worsening by more than 1.0 point. The four windows were used for model selection.""",

        "errors": """### Where errors remain

The model has lower WMAPE in five of six test months; May is slightly higher than last observed sales. Q1–Q4 were defined before testing using mean monthly sales in July–December 2025. Q1 WMAPE is 66.60% and Q4 is 28.02%; low-volume series remain difficult.

Zero/positive groups describe realised test outcomes, not information available for routing forecasts. WMAPE and MAPE are undefined for the zero-sales group; its row count and MAE measure the effect. Even a fractionally positive forecast yields 200% sMAPE when actual sales are zero. This explains why lower MAE can coexist with worse overall sMAPE.

Six-month net bias for the 371-series cohort is −0.67%, monthly aggregate WMAPE is 6.46%, and series-month WMAPE is 29.75%. Net bias allows errors to cancel across months; monthly aggregation allows cancellation between series within a month; series-month absolute error retains each row's discrepancy. The aggregate measures describe this evaluation cohort.""",

        "fixed": """### Fixed six-month forecasts: baselines and review variants

Naive comparators include trailing 3/6/12-month means, last observed sales and the same month last year. The platform-rating model has 38.09% WMAPE. The twelve-month mean has the lowest saved naive test WMAPE, 67.43%, giving a 43.5% reduction in absolute error. This is a descriptive comparison of test results; the full results appear below.

Review experiments add platform ratings, lexicon, text or combined features on the same cohort and fixed-origin protocol. Each variant uses the specified 100 trees; the primary comparison uses the selected platform-rating model. The monthly review-update variant updates review information while sales lags remain recursive.""",

        "uncertainty": """### Improvement and uncertainty

Review enhancement belongs to the fixed six-month stress test: its point estimate improves on the sales baseline by 0.980 percentage points, while the series-cluster 95% interval is −0.0047 to 2.2331 points. Evidence for a stable gain is insufficient, so it is classified as supporting information.""",
        "importance": """### Information used by the fixed-origin review model

Mean absolute SHAP values describe the fixed six-month platform-rating model's log-sales predictions. Group shares summarise model attribution; correlated predictors may share contributions. These values do not measure sales growth or accuracy gains.""",

        "cold": """### Series with no positive sales history

Thirteen series have no positive sales history before the fixed origin and use the same platform-rating model. Their fixed-origin WMAPE is about 99.40%; the available history and specifications do not reliably predict their sales ramp-up.""",
        "config": """## 3. Product specifications and annual sales variation

The sample has 646 series and 1,510 complete series-year records from 2022–2025. Five-fold GroupKFold holds out whole series, testing generalisation to unseen series rather than future years. Each fold fits its own imputation and encoding before XGBoost models log1p annual sales.

R² is the mean log-scale fold score; error bars show fold standard deviations, not confidence intervals. WMAPE pools inverse-transformed out-of-fold predictions rather than averaging fold WMAPEs. Global and year-specific median baselines use only each training fold's targets.""",
        "config_read": """Log-sales R² rises from 0.013 to 0.070 with brand and to 0.239 with specifications, an increment of 0.169. Raw-scale WMAPE is 73.85%, below the year-specific median baseline's 87.21%; substantial absolute error remains.

The twelve leading specification features are ranked by gain, a measure of tree split improvement. This gives neither effect direction nor causal estimates. Price, size, power and brand positioning vary together, while annual series-level data hide trim differences. The results help choose dimensions for product comparisons, not estimate the sales return from a feature.""",
        "needs": """## 4. User needs and review risk

The eligible corpus contains 24,175 reviews from 345 series. Mention share uses all eligible reviews; negative share uses scored mentions of the relevant aspect. The table includes mention, scored and negative counts. One review may mention several aspects, so these counts do not sum to independent reviews.

Reviews are self-selected, and negative share is not population dissatisfaction. Historical zero labels mix unmentioned, neutral and parsing fallback states. A common mention detector improves consistency but does not supply ground truth. Existing spot checks are not an independent representative benchmark, so label F1 is not established.""",
        "alerts": """### Dual-signal risk monitoring

The text rule identifies watchlist candidates. A candidate enters the dual-signal alert list for manual review when the text rule repeats in at least 70% of 3,000 bootstrap samples and original platform ratings decline in at least 80% of samples.

In the latest complete month, 123 of 371 target series (33.2%) meet the minimum of five reviews in each of two adjacent 180-day windows. There are no dual-signal alerts and one watchlist candidate within this coverage. Bootstrap measures rule stability under resampling, not the probability of a future fault. Independent event labels are not available to estimate alert precision.""",
        "audit": """## 5. Data quality and evaluation scope

Review analysis reuses historical labels and supplements uncovered reviews with API-generated labels. Labels undergo schema checks and manual sampling; sources and counts appear below.

The 2026 evaluation window has been examined during development and is reported as a retrospective evaluation. The review-gain interval is conditional on fitted models and observed months; it does not cover retraining, model selection or future months. Missing within-year specification release dates and historical sales revisions also limit full point-in-time reconstruction.""",
        "dashboard": """## 6. Explore the results

The six-page [interactive dashboard](https://yemyu.github.io/china-auto-market-analysis/) presents sales forecasts, product specifications, user needs and series details using static JSON in `app/static/data/`. Page captures are in `assets/dashboard/en/`; local browsing and analysis setup are described in the [data guide](../data/README_EN.md).

![Project overview](../assets/dashboard/en/01-overview.png)""",

        "conclusion": """## 7. Conclusions

1. Rolling one-month seasonal XGBoost has 29.75% WMAPE, reducing absolute error by 27.4% versus last observed sales. Low-volume series remain the main difficulty.
2. The fixed six-month platform-rating model has 38.09% WMAPE. Review features improve it by 0.980 percentage points, but the 95% interval crosses zero and does not establish a stable gain.
3. Product specifications increase annual log-sales R² by 0.169 and help identify dimensions for product comparisons. Aggregate data cannot estimate the sales return from an individual feature.
4. Reviews inform user-needs analysis and risk review. The latest complete monitoring month covers 123 series; alert rules still require validation against independent event labels.""",
    },
}


def build(lang):
    t = TEXT[lang]
    zh = lang == "zh"
    cells = [
        md(t["title"]),
        code(("ZH = True\n" if zh else "ZH = False\n") + dedent(SETUP).strip()),
        code(LOAD),
        md(t["sample"]), code(SAMPLES), md(t["engineering"]),
        md(t["forecast"]), code(TIME),
        md(t["models"]), code(MODELS),
        md(t["historical"]), code(HISTORICAL),
        md(t["errors"]), code(ERRORS),
        md(t["fixed"]), code(FIXED),
        md(t["uncertainty"]), code(UNCERTAINTY),
        md(t["importance"]), code(IMPORTANCE),
        md(t["cold"]),
        md(t["config"]), code(CONFIG), md(t["config_read"]),
        md(t["needs"]), code(ASPECTS),
        md(t["alerts"]), code(ALERTS),
        md(t["audit"]), code(AUDIT),
        md(t["dashboard"]), md(t["conclusion"]),
    ]
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(build("zh"), OUT / "China_Auto_Market_Analysis.ipynb")
    nbf.write(build("en"), OUT / "China_Auto_Market_Analysis_EN.ipynb")
    print("Wrote bilingual report notebooks.")


if __name__ == "__main__":
    main()

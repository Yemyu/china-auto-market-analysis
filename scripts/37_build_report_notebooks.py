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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
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

print("Project root:", ROOT)
"""


LOAD = r"""
sales = pd.read_csv(DATA / "raw" / "monthly_sales.csv")
specs = pd.read_csv(DATA / "raw" / "feature.csv")
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
robustness = read_json(FORECAST_DIR / "forecast_robustness_summary.json")
cold = read_json(FORECAST_DIR / "cold_start_launch_curve_summary.json")
config = pd.read_csv(PRODUCT_DIR / "config_attribution_ablation.csv")
aspects = pd.read_csv(FEEDBACK_DIR / "user_need_aspect_summary.csv", encoding="utf-8-sig")
alerts = pd.read_csv(FEEDBACK_DIR / "sentiment_alerts.csv")
monitor = read_json(FEEDBACK_DIR / "user_needs_alerts_summary.json")
print("Loaded analysis artifacts.")
"""


SAMPLES = r"""
if ZH:
    table = pd.DataFrame([
        ["滚动单月销量预测", "371 个车系", "2,226 条测试车系月", "每月更新下月预测"],
        ["固定六个月压力测试", "371 个车系", "2,226 条测试车系月", "固定起点递归预测"],
        ["产品配置分析", "736 个车系", "2,007 条车系年记录", "GroupKFold(5) 按车系"],
        ["用户需求与风险", "345 个车系", "24,175 条评论", "质量审计 + 180 天窗口"],
    ], columns=["分析", "样本", "观测", "验证"])
    source = pd.DataFrame([
        ["月销量", f"{sales.series_name.nunique():,} 个车系 / {len(sales):,} 行"],
        ["车型配置", f"{specs.series_name.nunique():,} 个车系 / {len(specs):,} 行"],
        ["严格评论语料", f"{corpus['temporally_eligible_reviews']:,} 条 / "
                         f"{corpus['target_series_with_any_review']} 个车系"],
    ], columns=["数据底座", "规模"])
else:
    table = pd.DataFrame([
        ["Rolling one-month sales forecast", "371 series", "2,226 test series-months",
         "Monthly refresh, one month ahead"],
        ["Fixed six-month stress test", "371 series", "2,226 test series-months",
         "Fixed-origin recursive forecast"],
        ["Product specifications", "736 series", "2,007 series-year records",
         "Five-fold GroupKFold by series"],
        ["User needs and risk", "345 series", "24,175 reviews",
         "Quality audit + 180-day windows"],
    ], columns=["Analysis", "Sample", "Observations", "Validation"])
    source = pd.DataFrame([
        ["Monthly sales", f"{sales.series_name.nunique():,} series / {len(sales):,} rows"],
        ["Vehicle specifications", f"{specs.series_name.nunique():,} series / {len(specs):,} rows"],
        ["Strict review corpus", f"{corpus['temporally_eligible_reviews']:,} reviews / "
                                 f"{corpus['target_series_with_any_review']} series"],
    ], columns=["Data foundation", "Scale"])
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
for split, color in [("train", COLORS["blue"]), ("val", COLORS["orange"]), ("test", COLORS["red"])]:
    part = monthly[monthly["split"] == split]
    ax.plot(part["date"], part["monthly_sales"] / 1e6, color=color, lw=2.1, label=split.title())
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
    ("滚动单月季节增强 XGBoost（主结果）" if ZH else "Rolling one-month seasonal XGBoost (headline)", "pred"),
]
rolling_table = pd.DataFrame([
    [label, wmape(rolling_test, column), median_wmape(rolling_test, column)]
    for label, column in rolling_rows
], columns=[model_name, global_name, median_name])
display(rolling_table.style.format({global_name: "{:.2f}%", median_name: "{:.2f}%"}))
fixed_row = pd.DataFrame([[
    "固定六个月综合方案（压力测试）" if ZH else "Fixed six-month combined method (stress test)",
    cold["hybrid_full371_global_WMAPE"], cold["hybrid_full371_median_per_series_WMAPE"],
]], columns=[model_name, global_name, median_name])
display(fixed_row.style.format({global_name: "{:.2f}%", median_name: "{:.2f}%"}))
ax = rolling_table.set_index(model_name).plot.bar(
    color=[COLORS["blue"], COLORS["light"]], width=0.72, figsize=(10, 4.8))
ax.set(title=("滚动单月测试集误差" if ZH else "Rolling one-month test error"),
       xlabel="", ylabel="WMAPE (%)")
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
plt.show()
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
ax.text(point, .08, f"{point:.2f} pp  [95%: {low:.2f}, {high:.2f}]",
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
           "review_expanding_score": "历史评价得分", "review_observation_context": "评论覆盖",
           "review_mention_count": "需求提及量", "review_negative_rate": "负面比例",
           "review_recent_score": "近期评价得分", "review_mention_rate": "需求提及率",
           "review_positive_rate": "正面比例"},
    "en": {"sales_lag_roll": "Sales history", "calendar": "Calendar",
           "configuration": "Product specifications",
           "review_expanding_score": "Historical review score",
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
display(frame[[name, "R2_log_mean", "R2_log_std", wmape_column, "n_features"]]
        .style.format({"R2_log_mean": "{:.3f}", "R2_log_std": "{:.3f}",
                       wmape_column: "{:.2f}%"}))
ordered = frame[frame.variant.isin(["YEAR-ONLY", "+BRAND", "+CONFIG"])]
fig, ax = plt.subplots(figsize=(9.5, 4.5))
bars = ax.bar(ordered[name], ordered.R2_log_mean,
              color=[COLORS["gray"], COLORS["light"], COLORS["blue"]])
ax.errorbar(ordered[name], ordered.R2_log_mean, yerr=ordered.R2_log_std,
            fmt="none", ecolor=COLORS["navy"], capsize=5, lw=1.3)
ax.set(title=("年份、品牌与配置的增量解释力" if ZH else
              "Incremental explanatory power of year, brand, and specifications"),
       xlabel="", ylabel="GroupKFold R²")
ax.bar_label(bars, fmt="%.3f", padding=4)
plt.show()
"""


ASPECTS = r"""
ordered = aspects.sort_values("mention_rate")
fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(ordered))
ax.barh(y-.18, ordered.mention_rate, height=.34, color=COLORS["light"],
        label=("提及率" if ZH else "Mention share"))
ax.barh(y+.18, ordered.negative_rate_among_scored, height=.34, color=COLORS["red"],
        label=("负面率（有效评分内）" if ZH else "Negative share among scored"))
labels_y = ordered.aspect_zh if ZH else ordered.aspect.str.replace("_", " ").str.title()
ax.set_yticks(y, labels_y)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
ax.set(title=("十类用户需求：讨论热度与负面集中度" if ZH else
              "Ten user needs: discussion and negative concentration"),
       xlabel=("比例" if ZH else "Share"), ylabel="")
ax.legend(frameon=False, loc="lower right")
plt.show()

top = aspects.sort_values("negative_rate_among_scored", ascending=False).head(5)
if ZH:
    top = top[["aspect_zh", "mention_rate", "negative_rate_among_scored", "mean_polarity"]]
    top.columns = ["维度", "提及率", "负面率", "平均倾向"]
    display(top.style.format({"提及率": "{:.1%}", "负面率": "{:.1%}", "平均倾向": "{:.3f}"}))
else:
    top = top[["aspect", "mention_rate", "negative_rate_among_scored", "mean_polarity"]]
    top.columns = ["Dimension", "Mention share", "Negative share", "Mean polarity"]
    display(top.style.format({"Mention share": "{:.1%}", "Negative share": "{:.1%}",
                              "Mean polarity": "{:.3f}"}))
"""


ALERTS = r"""
latest = alerts.information_cutoff_inclusive.max()
current = alerts[(alerts.information_cutoff_inclusive == latest)
                 & alerts.alert.astype(str).str.lower().isin(["true", "1"])].copy()
if ZH:
    overview = pd.DataFrame([
        ["最近完整监测月", monitor["latest_completed_monitoring_month"]],
        ["达到样本门槛的车系", monitor["latest_eligible_series"]],
        ["当前预警", monitor["latest_active_alerts"]],
        ["历史预警事件", monitor["historical_alert_events"]],
    ], columns=["项目", "数值"])
else:
    overview = pd.DataFrame([
        ["Latest complete month", monitor["latest_completed_monitoring_month"]],
        ["Eligible series", monitor["latest_eligible_series"]],
        ["Current alerts", monitor["latest_active_alerts"]],
        ["Historical alert events", monitor["historical_alert_events"]],
    ], columns=["Item", "Value"])
display(overview)
display(current[["series_name", "brand", "current_reviews", "current_overall_score",
                 "score_change", "current_negative_review_rate", "worst_aspect", "risk_level"]])
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
        ["补充标签", f"{labels['api_labeled_reviews']:,} 条", "结构校验与人工抽样"],
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
        ["Newly supplemented labels", f"{labels['api_labeled_reviews']:,}",
         "Schema checks and manual sampling"],
    ], columns=["Audit item", "Result", "Treatment"])
display(audit)
"""


TEXT = {
    "zh": {
        "title": """# 中国汽车市场分析：销量预测、产品配置与用户需求

这本 Notebook 从已保存的数据产物复现主要结果与图表。采集、标签生成和模型训练脚本位于 `scripts/`。

**研究期：** 2022-01—2026-07

**预测测试期：** 2026-01—06
**销量主任务：** 每月更新的下月预测；固定六个月为压力测试
**主指标：** 全局 volume-weighted WMAPE""",
        "sample": """## 1. 三组分析样本

三项分析的筛选条件不同。销量预测固定 371 个车系的完整自然月面板，并对年度配置做因果连接；产品配置分析要求年度销量与配置能够对齐，用户需求分析要求完整且可核验的评论正文。""",

        "forecast": """## 2. 月度销量预测

训练截至 2025-06，验证期为 2025-07—12，测试期为 2026-01—06。主结果是每月更新的下月预测：每次预测可使用已公布的上月真实销量；固定起点六个月递归结果另作压力测试。""",

        "models": """### 模型比较

滚动单月主结果中，季节增强 XGBoost 的全局 WMAPE 为 29.72%，沿用上月销量的朴素基准为 40.99%，绝对误差降低 11.27 个百分点（相对减少约 27.5%）。固定六个月综合方案为 38.38%，相对同场景最近 6 个月均值 69.31% 减少 44.6% 的绝对误差；两种协议对应不同应用场景，分别评估。""",

        "uncertainty": """### 改善幅度与不确定性

口碑增强属于固定六个月压力测试：点估计相对销量基线改善 0.697 个百分点，按车系重采样的 95% 区间为 −0.234 至 1.873 个百分点，稳定增益证据不足，因此定位为辅助信息。""",
        "importance": "### 哪些信息在起作用",

        "cold": """### 固定压力测试中的冷启动补充

固定六个月压力测试为 9 个同时缺少历史正销量和起点前配置记录的车系补充受约束的上市曲线；这项统计策略用于边界样本处理，不改变 371 个车系的滚动主模型。""",
        "config": """## 3. 产品配置与年度销量差异

样本为 736 个车系、2,007 条车系年记录。验证按车系分组，避免同一车系的不同年份同时出现在训练折和验证折。""",
        "config_read": """**结果解读：** 加入品牌后 R² 从 0.089 升至 0.158；继续加入配置后达到 0.301。该结果量化产品属性对年度跨车系差异的样本外解释力，不进行因果识别；年度截面 WMAPE 作为模块内辅助误差指标，与月度预测指标分别报告。""",
        "needs": """## 4. 用户需求与口碑风险

严格语料包含 24,175 条评论。十个维度同时保留提及率和有效评分中的负面率，避免把“讨论很多”和“评价很差”合并成一个指标。""",
        "alerts": """### 规则预警

预警要求相邻两个 180 天窗口都达到最小评论量，并同时触发综合评价、下降幅度和负面率阈值。它只用于安排人工复核。""",
        "audit": "## 5. 数据质量与时间可用性",
        "dashboard": """## 6. 看板与复现

看板是纯静态站点，读取 `app/static/data/` 中的预烘焙 JSON。完整六页截图见 `assets/dashboard/zh/`。

![项目概览](../assets/dashboard/zh/01-overview.png)

本地启动（在项目根目录执行）：

    python -m http.server 8000 --directory app""",

        "conclusion": """## 7. 结论

1. 滚动单月季节增强 XGBoost 是当前业务主结果；历史销量是主要信号，且相对同场景朴素基准有明确改善。
2. 固定六个月综合方案保留为压力测试，与滚动单月协议分别评估。
3. 产品配置能够提高年度销量差异的解释力；该结果属于样本外解释分析，其 WMAPE 为年度截面模块内辅助误差指标。
4. 评论数据主要用于需求结构、风险监测和固定压力测试的辅助信息；提及率、正负倾向与样本量需要分开报告。
5. 严格时间切分、可用信息边界和完整正文门槛是当前结果可复查的基础。""",
    },
    "en": {
        "title": """# China Automotive Market Analysis: Sales Forecasting, Product Specifications, and User Needs

This notebook reproduces the main results and figures from saved analysis artifacts. Collection, labeling, and model-fitting scripts are in `scripts/`.

**Study period:** 2022-01—2026-07

**Forecast test:** 2026-01—06
**Sales headline task:** monthly refreshed one-month-ahead forecast; fixed six-month stress test
**Primary metric:** global volume-weighted WMAPE""",
        "sample": """## 1. Three analysis samples

Forecasting uses a fixed 371-series natural-month panel with causal year-based specification joins; the specification analysis requires aligned annual sales and product attributes; the user-needs analysis requires complete and traceable review text.""",

        "forecast": """## 2. Monthly sales forecasting

Training ends in 2025-06, validation covers 2025-07—12, and testing covers 2026-01—06. The headline refreshes each month for a one-month-ahead forecast and can use the latest published sales; the fixed-origin six-month recursive result is reported separately as a stress test.""",

        "models": """### Model comparison

In the rolling one-month headline, seasonal XGBoost reaches 29.72% global WMAPE versus 40.99% for the last-observed-value naive baseline, an 11.27-point (about 27.5%) absolute-error reduction. The fixed six-month combined method reaches 38.38% versus 69.31% for its trailing-six-month comparator, a 44.6% reduction; the protocols correspond to different forecast applications and are evaluated separately.""",

        "uncertainty": """### Improvement and uncertainty

Review enhancement belongs to the fixed six-month stress test: its point estimate improves on the sales baseline by 0.697 percentage points, while the series-cluster 95% interval is −0.234 to 1.873 points. Evidence for a stable gain is insufficient, so it is classified as supporting information.""",
        "importance": "### Information contribution",

        "cold": """### Cold-start supplement in the stress test

The fixed six-month stress test adds a guarded launch-curve statistical supplement for nine series with neither positive historical sales nor a pre-origin specification record. It is used for boundary-case treatment and does not alter the 371-series rolling headline model.""",
        "config": """## 3. Product specifications and annual sales variation

The sample contains 736 series and 2,007 series-year records. Validation groups by series so different years of one series cannot enter both training and validation folds.""",
        "config_read": """**Interpretation:** R² rises from 0.089 to 0.158 after adding brand and reaches 0.301 after adding specifications. The result quantifies out-of-sample explanatory power for annual between-series variation without causal identification; annual cross-sectional WMAPE is a module-specific supporting metric reported separately from monthly forecasting.""",
        "needs": """## 4. User needs and review risk

The strict corpus contains 24,175 reviews. Mention share and negative share among scored mentions remain separate, so discussion volume is not conflated with dissatisfaction.""",
        "alerts": """### Rule-based alerts

Both adjacent 180-day windows must meet the minimum review count and all score, deterioration, and negative-share thresholds. An alert queues manual review.""",
        "audit": "## 5. Data quality and temporal availability",
        "dashboard": """## 6. Dashboard and reproduction

The dashboard is a static site backed by pre-baked JSON in `app/static/data/`. All six English captures are in `assets/dashboard/en/`.

![Project overview](../assets/dashboard/en/01-overview.png)

Launch from the repository root:

    python -m http.server 8000 --directory app""",

        "conclusion": """## 7. Conclusions

1. Rolling one-month seasonal XGBoost is the current operational headline; sales history is the dominant signal and clearly improves on its same-scenario naive baseline.
2. The fixed six-month combined method is retained as a stress test and is evaluated separately from the rolling protocol.
3. Product specifications improve the explanation of annual between-series variation; the result is an out-of-sample explanatory analysis and its WMAPE is a module-specific supporting metric.
4. Review data is primarily used for demand structure, risk monitoring, and supporting information in the stress test; mention, polarity, and sample size should be reported separately.
5. Strict temporal splits, information-availability boundaries, and full-text quality thresholds make the results auditable.""",
    },
}


def build(lang):
    t = TEXT[lang]
    zh = lang == "zh"
    cells = [
        md(t["title"]),
        code(("ZH = True\n" if zh else "ZH = False\n") + dedent(SETUP).strip()),
        code(LOAD),
        md(t["sample"]), code(SAMPLES),
        md(t["forecast"]), code(TIME),
        md(t["models"]), code(MODELS),
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

#!/usr/bin/env python3
"""Build bilingual report notebooks from the current saved artifacts."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
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
STAGE3 = DATA / "processed_new" / "stage3"
STAGE4 = DATA / "processed_new" / "stage4"
STAGE5 = DATA / "processed_new" / "stage5"
SENTIMENT = DATA / "sentiment_new" / "processed"

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
train = pd.read_csv(DATA / "processed_new" / "splits" / "train.csv")
val = pd.read_csv(DATA / "processed_new" / "splits" / "val.csv")
test = pd.read_csv(DATA / "processed_new" / "splits" / "test.csv")
split_manifest = read_json(DATA / "processed_new" / "splits" / "manifest.json")
corpus = read_json(SENTIMENT / "target_371_review_corpus_summary.json")
labels = read_json(SENTIMENT / "unified_deepseek_absa_summary.json")
temporal = read_json(SENTIMENT / "deepseek_feature_temporal_summary.json")
forecast = pd.read_csv(STAGE3 / "xgb_deepseek_full371_summary.csv", encoding="utf-8-sig")
robustness = read_json(STAGE3 / "xgb_deepseek_full371_robustness_summary.json")
cold = read_json(STAGE3 / "cold_start_launch_curve_summary.json")
config = pd.read_csv(STAGE4 / "config_attribution_ablation.csv")
aspects = pd.read_csv(STAGE5 / "user_need_aspect_summary.csv", encoding="utf-8-sig")
alerts = pd.read_csv(STAGE5 / "sentiment_alerts.csv")
monitor = read_json(STAGE5 / "user_needs_alerts_summary.json")
print("Loaded current pipeline artifacts.")
"""


SAMPLES = r"""
if ZH:
    table = pd.DataFrame([
        ["六个月销量预测", "371 个车系", "2,226 条测试车系月", "固定起点递归预测"],
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
        ["Six-month sales forecast", "371 series", "2,226 test series-months",
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
ax.set(title=("371 车系月销量与固定时间切分" if ZH else
              "Monthly sales and fixed temporal split: 371 series"),
       xlabel="", ylabel=("月销量（百万辆）" if ZH else "Monthly sales (million)"))
ax.legend(frameon=False, ncol=3)
plt.show()
"""


MODELS = r"""
model_name = "方案" if ZH else "Model"
global_name = "全局 WMAPE" if ZH else "Global WMAPE"
median_name = "逐车系中位数 WMAPE" if ZH else "Median per-series WMAPE"
names = {
    "BASE": "销量基线" if ZH else "Sales baseline",
    "DEEPSEEK_RICH_FIXED": "用户口碑增强" if ZH else "Owner-feedback enhanced",
}
main = forecast[forecast.version.isin(names)].copy()
main[model_name] = main.version.map(names)
main = main[[model_name, "global_volume_weighted_WMAPE", "median_per_series_WMAPE"]]
main.columns = [model_name, global_name, median_name]
main.loc[len(main)] = [
    "冷启动补充" if ZH else "Cold-start supplement",
    cold["hybrid_full371_global_WMAPE"],
    cold["hybrid_full371_median_per_series_WMAPE"],
]
display(main.style.format({global_name: "{:.2f}%", median_name: "{:.2f}%"}))
ax = main.set_index(model_name).plot.bar(
    color=[COLORS["blue"], COLORS["light"]], width=0.72, figsize=(10, 4.8))
ax.set(title=("六个月测试集误差" if ZH else "Six-month test error"),
       xlabel="", ylabel="WMAPE (%)")
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
plt.show()
"""


UNCERTAINTY = r"""
point = robustness["deepseek_rich_vs_base_improvement_pp"]
low, high = robustness["deepseek_rich_vs_base_bootstrap_95pct_ci_pp"]
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
    print(f"重采样中优于基线的比例：{robustness['deepseek_rich_vs_base_probability_better']:.1%}")
    print(f"六个测试月中优于基线：{robustness['test_months_deepseek_rich_better_than_base']} / 6")
else:
    print("Share of bootstrap replicates better than baseline: "
          f"{robustness['deepseek_rich_vs_base_probability_better']:.1%}")
    print(f"Test months better than baseline: {robustness['test_months_deepseek_rich_better_than_base']} / 6")
"""


IMPORTANCE = r"""
family = pd.read_csv(STAGE3 / "xgb_deepseek_full371_feature_family_importance.csv",
                     encoding="utf-8-sig")
maps = {
    "zh": {"sales_lag_roll": "历史销量", "calendar": "日历", "configuration": "产品配置",
           "deepseek_expanding_score": "历史评价得分", "deepseek_observation_context": "评论覆盖",
           "deepseek_mention_count": "需求提及量", "deepseek_negative_rate": "负面比例",
           "deepseek_recent_score": "近期评价得分", "deepseek_mention_rate": "需求提及率",
           "deepseek_positive_rate": "正面比例"},
    "en": {"sales_lag_roll": "Sales history", "calendar": "Calendar",
           "configuration": "Product specifications",
           "deepseek_expanding_score": "Historical review score",
           "deepseek_observation_context": "Review coverage",
           "deepseek_mention_count": "Need mentions", "deepseek_negative_rate": "Negative share",
           "deepseek_recent_score": "Recent review score",
           "deepseek_mention_rate": "Mention share", "deepseek_positive_rate": "Positive share"},
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
display(frame[[name, "R2_log_mean", "R2_log_std", "WMAPE_mean", "n_features"]]
        .style.format({"R2_log_mean": "{:.3f}", "R2_log_std": "{:.3f}",
                       "WMAPE_mean": "{:.2f}%"}))
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
         "冻结于 2026-01-01 前"],
        ["最近 180 天有评论", f"{temporal['fixed_test_series_with_recent_180d_review']} 个车系",
         "其余保留缺失标记"],
        ["复用历史标签", f"{labels['reused_legacy_reviews']:,} 条", "避免重复付费生成"],
        ["新版补充标签", f"{labels['compact_v4_flash_reviews']:,} 条", "结构校验与人工抽样"],
    ], columns=["审计项", "结果", "处理"])
else:
    audit = pd.DataFrame([
        ["Strict review corpus", f"{corpus['temporally_eligible_reviews']:,} reviews",
         "Complete text, time, and source"],
        ["Excluded list summaries", corpus["autohome_list_summary_rows_excluded_from_temporal_model"],
         "No full detail text; not modeled"],
        ["Review evidence before test origin", f"{temporal['fixed_test_series_with_any_prior_review']} series",
         "Frozen before 2026-01-01"],
        ["Review in prior 180 days", f"{temporal['fixed_test_series_with_recent_180d_review']} series",
         "Missingness retained elsewhere"],
        ["Reused historical labels", f"{labels['reused_legacy_reviews']:,}",
         "Avoids duplicate paid generation"],
        ["Newly supplemented labels", f"{labels['compact_v4_flash_reviews']:,}",
         "Schema checks and manual sampling"],
    ], columns=["Audit item", "Result", "Treatment"])
display(audit)
"""


TEXT = {
    "zh": {
        "title": """# 中国汽车市场分析：销量预测、产品配置与用户需求

这本 Notebook 汇总当前版本的实证结果，并从已落盘的中间产物复现主要图表。耗时较长的采集、标签生成和模型训练保留在 `scripts/new_pipeline/`，这里不重复调用外部服务。

**研究期：** 2022-01—2026-07

**预测测试期：** 2026-01—06
**主指标：** 全局 volume-weighted WMAPE""",
        "sample": """## 1. 三组分析样本

三项分析的筛选条件不同。销量预测要求连续月度历史，产品配置分析要求年度销量与配置能够对齐，用户需求分析要求完整且可核验的评论正文。""",
        "forecast": """## 2. 月度销量预测

训练截至 2025-06，验证期为 2025-07—12，测试期为 2026-01—06。测试采用 2026 年 1 月固定起点；评论特征同样冻结在该起点。""",
        "models": """### 模型比较

先报告全局 WMAPE，再用逐车系中位数观察长尾。两类指标的分母和聚合方式不同，不应相互替代。""",
        "uncertainty": """### 改善幅度与不确定性

用户口碑增强模型相对销量基线降低 1.73 个百分点。按车系重采样的区间仍跨过零，因此结论是“小幅改善信号”，而不是“稳定优于基线”。""",
        "importance": "### 哪些信息在起作用",
        "cold": """### 冷启动

9 个历史不足车系的 WMAPE 从 98.76% 降到 89.62%；它们销量体量较小，对 371 车系整体指标的进一步改善为 0.06 个百分点。""",
        "config": """## 3. 产品配置与年度销量差异

样本为 736 个车系、2,007 条车系年记录。验证按车系分组，避免同一车系的不同年份同时出现在训练折和验证折。""",
        "config_read": """**结果解读：** 加入品牌后 R² 从 0.089 升至 0.154；继续加入配置后达到 0.303。这个模块衡量车系间的解释关联，不用于判断单一配置是否会直接导致销量增长。""",
        "needs": """## 4. 用户需求与口碑风险

严格语料包含 24,175 条评论。十个维度同时保留提及率和有效评分中的负面率，避免把“讨论很多”和“评价很差”合并成一个指标。""",
        "alerts": """### 规则预警

预警要求相邻两个 180 天窗口都达到最小评论量，并同时触发综合评价、下降幅度和负面率阈值。它只用于安排人工复核。""",
        "audit": "## 5. 数据质量与时间可用性",
        "dashboard": """## 6. 看板与复现

看板是纯静态站点，读取 `app/static/data/` 中的预烘焙 JSON。完整六页截图见 `assets/dashboard/zh/`。

![项目概览](../assets/dashboard/zh/01-overview.png)

本地启动（在项目根目录执行）：

    conda run -n nlp-sentiment python -m http.server 8000 --directory app""",
        "conclusion": """## 7. 结论

1. 历史销量是六个月预测的主要信息来源；用户口碑提供小幅、但尚未稳健显著的增量。
2. 产品配置能够提高年度销量差异的解释力，但结果属于关联分析。
3. 评论数据更适合用于需求结构和风险监测；提及率、正负倾向与样本量需要分开报告。
4. 固定时间切分、预测起点冻结和完整正文门槛是当前结果可复查的基础。""",
    },
    "en": {
        "title": """# China Automotive Market Analysis: Sales Forecasting, Product Specifications, and User Needs

This notebook summarizes the current empirical results and reproduces the main figures from saved pipeline artifacts. Expensive collection, labeling, and model fitting remain in `scripts/new_pipeline/`; this report makes no external service calls.

**Study period:** 2022-01—2026-07

**Forecast test:** 2026-01—06
**Primary metric:** global volume-weighted WMAPE""",
        "sample": """## 1. Three analysis samples

Forecasting requires continuous monthly history; the specification analysis requires aligned annual sales and product attributes; the user-needs analysis requires complete and traceable review text.""",
        "forecast": """## 2. Monthly sales forecasting

Training ends in 2025-06, validation covers 2025-07—12, and testing covers 2026-01—06. The test uses a January 2026 fixed origin; review features are frozen at the same origin.""",
        "models": """### Model comparison

Global WMAPE is followed by median per-series WMAPE for the long tail. The two metrics have different denominators and aggregation rules and should not be substituted for each other.""",
        "uncertainty": """### Improvement and uncertainty

The owner-feedback model lowers global WMAPE by 1.73 percentage points. The series-cluster bootstrap interval still crosses zero, supporting a modest improvement signal rather than a stable proven advantage.""",
        "importance": "### Information contribution",
        "cold": """### Cold start

WMAPE for nine history-poor series falls from 98.76% to 89.62%. Their limited volume means the full 371-series score improves by a further 0.06 percentage points.""",
        "config": """## 3. Product specifications and annual sales variation

The sample contains 736 series and 2,007 series-year records. Validation groups by series so different years of one series cannot enter both training and validation folds.""",
        "config_read": """**Interpretation:** R² rises from 0.089 to 0.154 after adding brand and reaches 0.303 after adding specifications. This is an explanatory association across series, not a causal estimate for an individual feature.""",
        "needs": """## 4. User needs and review risk

The strict corpus contains 24,175 reviews. Mention share and negative share among scored mentions remain separate, so discussion volume is not conflated with dissatisfaction.""",
        "alerts": """### Rule-based alerts

Both adjacent 180-day windows must meet the minimum review count and all score, deterioration, and negative-share thresholds. An alert queues manual review.""",
        "audit": "## 5. Data quality and temporal availability",
        "dashboard": """## 6. Dashboard and reproduction

The dashboard is a static site backed by pre-baked JSON in `app/static/data/`. All six English captures are in `assets/dashboard/en/`.

![Project overview](../assets/dashboard/en/01-overview.png)

Launch from the repository root:

    conda run -n nlp-sentiment python -m http.server 8000 --directory app""",
        "conclusion": """## 7. Conclusions

1. Sales history remains the dominant forecasting signal; owner feedback adds a modest but not yet robustly significant increment.
2. Product specifications improve the explanation of annual cross-series variation, but the result is associational.
3. Review data is most directly useful for demand structure and risk monitoring; mention, polarity, and sample size should be reported separately.
4. Fixed temporal splits, origin freezing, and full-text quality thresholds make the results auditable.""",
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
            "kernelspec": {"display_name": "Python (nlp-sentiment)",
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

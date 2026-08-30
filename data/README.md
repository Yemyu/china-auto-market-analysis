<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# 📦 数据说明

本目录提供项目的数据入口、样本口径和可复现产物索引。原始平台数据仅用于学习、研究与项目展示；平台版权归相应来源方所有。

## 数据一览

| 模块 | 入口 | 规模 | 用途 |
|---|---|---:|---|
| 月度销量 | `raw/monthly_sales.csv` | 54,918 行 / 1,017 个车系 | 滚动单月预测与固定六个月压力测试 |
| 产品配置 | `raw/feature.csv` | 2,084 行 / 766 个车系 | 年度销量差异的产品属性分析 |
| 车主评论 | `reviews/processed/` | 24,175 条 / 345 个车系 | 用户需求、风险监测与口碑辅助实验 |

三个模块使用各自的样本筛选和评价口径，不将不同模块的指标直接横向比较。

## 目录结构

| 路径 | 内容 |
|---|---|
| `raw/` | 月销量和年度车型配置原始表 |
| `processed/splits/` | 371 个目标车系的时间切分与建模特征 |
| `processed/forecast/` | 预测、基准、消融和稳健性产物 |
| `processed/product/` | 产品配置年度解释分析产物 |
| `processed/user_feedback/` | 用户需求与风险监测产物 |
| `processed/data_quality/` | 结构、映射和来源审计的机器可读记录 |
| `reviews/raw/` | 评论采集清单和来源层文件 |
| `reviews/processed/` | 去标识语料、标签和时间特征 |
| `resources/` | 可复用的历史评论资源归档 |

## 原始输入

### 月度销量：`raw/monthly_sales.csv`

- 粒度：车系 × 自然月；时间范围：2022-01—2026-06；
- 主要字段：`series_id`、`series_name`、`brand`、`category`、`year`、`month`、`monthly_sales`；
- 负销量记录为 0；排名、累计销量和网站展示价格等源站派生字段不作为预测特征。

### 产品配置：`raw/feature.csv`

- 粒度：车系 × 年款，不是 trim 级车型清单；
- 唯一键：`series_name, year`；共 84 个字段，年度销量可对齐 760 / 766 个车系；
- 年度配置分析只使用销量源覆盖 12 个自然月的年份；当前为 2022—2025，共 646 个车系、1,510 条车系年记录；
- 配置缺失具有结构性：例如纯电车型通常没有发动机参数，燃油车型通常没有电池参数，不应简单视为采集错误。

### 评论语料：`reviews/processed/`

进入时间模型的评论同时满足车系可识别、发布时间可解析、正文完整且在预测截止日前发布。当前严格语料为 24,175 条，覆盖 345 个车系；缺失覆盖保留为缺失状态，不编码为中性。

评论标签拆分为“是否提及”和“评价方向”两类字段，覆盖外观、内饰、空间、动力、操控、舒适、能耗、配置、智能化和性价比十个维度。

## 月度预测样本

`processed/splits/` 固定保存 371 个车系的绝对时间切分：

| 文件 | 时间 | 用途 |
|---|---|---|
| `train.csv` | 截至 2025-06 | 模型训练；前置月份作为滞后特征预热 |
| `val.csv` | 2025-07—12 | 参数与方案选择 |
| `test.csv` | 2026-01—06 | 最终评价 |
| `split_index.csv` | 完整面板 | 每个车系月所属的数据段 |
| `manifest.json` | — | 行数、特征、时间边界和防泄漏约束 |

预测面板保留每个目标车系的自然月间隔。年度配置只向“不晚于目标年份”的最新记录回退；无法回退的数值配置使用配置表中位数，类别使用 `-1` 未知标记。源表不含年内发布时间，因此这里只声明年度对齐和不使用未来年份，不把它表述为月内点时可用性证明。基础特征使用 1/2/3 个月滞后和 3/6 个月历史均值，主模型额外使用 12 个月滞后和 12 个月历史均值。

主协议是滚动单月预测：每月预测下一个月，并使用已公布的上月真实销量。固定起点六个月协议从 2026-01 一次性递归预测，作为信息受限压力测试；两种协议分别评价。

## 主要产物

### 销量预测：`processed/forecast/`

| 文件 | 用途 |
|---|---|
| `rolling_origin_summary.json` | 滚动主协议的历史起点验证、门槛和锁定测试摘要 |
| `rolling_origin_test_predictions.csv` | 测试期逐车系逐月预测与同场景朴素基准 |
| `forecast_benchmark_comparison.csv` | 固定压力测试与朴素基准对比 |
| `review_feature_ablation_summary.csv` | 固定场景口碑特征消融 |
| `forecast_robustness_summary.json` | 聚类 Bootstrap、分组误差和稳健性摘要 |
| `cold_start_launch_curve_summary.json` | 边界车系冷启动方案及验证结果 |

当前保存的滚动主结果为 29.72% 全局 WMAPE；固定六个月压力测试综合方案为 38.38%。完整指标解释见项目根目录 [README.md](../README.md)。

### 产品配置：`processed/product/`

- `config_attribution_ablation.csv`：年份、品牌、配置的逐步消融；
- `config_importance_annual.csv`：年度配置特征重要性。
- `config_attribution_summary.json`：完整年份范围、样本规模和核心指标。

该模块报告按车系分组的样本外 R²；年度截面 WMAPE 只作模块内辅助指标，不与月度预测 WMAPE 直接比较，也不代表因果效应。

### 用户需求：`processed/user_feedback/`

`user_need_aspect_summary.csv`、`user_need_topics.csv`、`sentiment_monitoring_windows.csv` 和 `sentiment_alerts.csv` 分别用于维度汇总、主题、时间窗口和规则预警。预警需要人工复核后才能形成业务结论。

## 历史资源

`resources/historical_reviews/` 保存可复用的历史评论与标签归档：

- `review_absa_reference.csv.gz`：本地全文归档，39,496 条去重评论，其中 28,724 条带历史十维标签；
- `manifest.json`、`README.md`：行数、时间范围、校验摘要和使用限制。

全文归档被 Git 忽略，仅在本地保留；公开仓库提交去标识标签和聚合结果。

## 复现入口

在项目根目录准备依赖后，可按以下顺序重建主要产物：

```bash
.venv/bin/python scripts/06_make_splits.py
.venv/bin/python scripts/32_build_temporal_review_features.py
.venv/bin/python scripts/33_evaluate_review_features.py
.venv/bin/python scripts/48_evaluate_rolling_origin.py --test
.venv/bin/python scripts/36_build_cold_start_curve.py
.venv/bin/python app/build_dashboard_data.py
```

如需完整评论标签流水线，请先按根目录 Notebook 和脚本注释准备评论语料；缺失评论标签的补标步骤为可选项。

<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# 📦 数据说明

本文说明数据文件、样本定义、分析结果和本地运行方式。平台数据用于学习、研究与项目展示，版权归相应来源方所有。

## 数据概况

| 模块 | 入口 | 规模 | 用途 |
|---|---|---:|---|
| 月度销量 | `processed/sales_filtered_24m.csv` | 54,918 行 / 1,017 个车系 | 仓库内建模快照；滚动单月预测与固定六个月压力测试 |
| 产品配置 | `raw/feature.csv` | 2,084 行 / 766 个车系 | 仓库内建模表；年度销量差异的产品属性分析 |
| 车主评论 | `reviews/processed/` | 24,175 条 / 345 个车系 | 用户需求、风险监测与口碑辅助实验 |

三个模块分别筛选样本并评价结果。公开仓库提供 CSV、标签、分析摘要和看板 JSON；完整评论正文仅保存在本地。浏览看板、阅读 Notebook 和重建看板 JSON 均无需 MySQL。

## 目录结构

| 路径 | 内容 |
|---|---|
| `raw/` | 年度配置 CSV 与 Excel 源表；其他本地采集 CSV 不随 Git 分发 |
| `processed/splits/` | 371 个目标车系的时间切分与建模特征 |
| `processed/forecast/` | 预测、基准、消融和稳健性产物 |
| `processed/product/` | 产品配置年度解释分析产物 |
| `processed/user_feedback/` | 用户需求与风险监测产物 |
| `processed/data_quality/` | 结构、映射和来源审计的机器可读记录 |
| `reviews/raw/` | 评论采集清单和来源层文件 |
| `reviews/processed/` | 去标识标签、时间特征和语料摘要；完整正文不随 Git 分发 |
| `resources/` | 可复用的历史评论资源归档 |

## 建模输入

### 月度销量：`processed/sales_filtered_24m.csv`

- 粒度：车系 × 自然月；时间范围：2022-01—2026-06；
- 主要字段：`series_id`、`series_name`、`brand`、`category`、`year`、`month`、`monthly_sales`；
- 建模快照不含负销量记录；排名、累计销量和网站展示价格等源站派生字段不作为预测特征。

### 产品配置：`raw/feature.csv`

- 粒度：车系 × 年款，未细分到具体配置版本；
- 唯一键：`series_name, year`；共 84 个字段，年度销量可对齐 760 / 766 个车系；
- 年度配置分析只使用销量源覆盖 12 个自然月的年份；当前为 2022—2025，共 646 个车系、1,510 条车系年记录；
- 配置缺失具有结构性：例如纯电车型通常没有发动机参数，燃油车型通常没有电池参数，不应简单视为采集错误。

公开仓库保留已审计的月销量建模快照、配置 CSV 和对应 Excel 源表。`raw/monthly_sales.csv` 是本地采集过程中的工作文件，不作为公开克隆后的必需输入；核心脚本直接读取仓库内建模快照，配置 CSV 缺失时可回退到 Excel 源表。

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

预测面板保留每个目标车系的自然月间隔，仅保存销量、日历和滞后；原始配置由模型读取并按训练窗口处理，不在共享切分或预测主题表中保存全表编码。基础特征含 1/2/3 月滞后及 3/6 月均值，主模型另含 12 月滞后与 12 月均值。

主协议是滚动单月预测：每月预测下一个月，并使用已公布的上月真实销量。固定起点六个月协议从 2026-01 一次性递归预测，作为信息受限压力测试；两种协议分别评价。

选定方案后，最终模型在 train+val 上重新拟合至 2025-12，测试期不更新权重。每个预测起点限制可用配置年份，各行匹配同年或更早年份的最近记录。中位数与类别词表仅在模型实际训练行中拟合，预测窗口内配置冻结。未知类别记为 −1，全缺数值列使用 0 占位，配置缺失的销量行仍保留。配置源缺少年内发布时间，销量源缺少历史发布与修订版本；对齐依据为年度代理规则。

## 主要产物

### 销量预测：`processed/forecast/`

| 文件 | 用途 |
|---|---|
| `rolling_origin_summary.json` | 滚动主协议的历史起点验证、门槛和锁定测试摘要 |
| `rolling_origin_test_predictions.csv` | 测试期逐车系逐月预测与同场景朴素基准 |
| `forecast_benchmark_comparison.csv` | 固定压力测试与朴素基准对比 |
| `review_feature_ablation_summary.csv` | 固定场景口碑特征消融 |
| `forecast_robustness_summary.json` | 聚类 Bootstrap、分组误差和稳健性摘要 |

已保存的滚动预测结果为全局 WMAPE 29.75%、MAE 707.68 辆、RMSE 1,700.38 辆；固定六个月平台评分模型的 WMAPE 为 38.09%。MAE 和 RMSE 的单位为每个车系月的车辆数。Notebook 与看板均使用 `src/china_auto_market/forecasting/reporting.py` 对逐行预测计分：sMAPE 保留全部行，实际与预测均为零时记 0；MAPE 只计算正销量行。定义与完整对照见根目录 [README.md](../README.md)。

### 产品配置：`processed/product/`

- `config_attribution_ablation.csv`：年份、品牌、配置的逐步消融；
- `config_importance_annual.csv`：年度配置特征重要性。
- `config_attribution_summary.json`：完整年份范围、样本规模和核心指标。

该模块报告 `log1p(年度销量)` 尺度的五折 R² 均值和折间标准差；完整模型为 0.239，加入配置的增量为 0.169。WMAPE 使用还原为辆数的合并折外预测，评价年度跨车系关联，与月度预测采用不同任务和样本。

### 用户需求：`processed/user_feedback/`

`user_need_aspect_summary.csv`、`user_need_keywords.csv`、`user_need_topics.csv`、`sentiment_monitoring_windows.csv` 和 `sentiment_alerts.csv` 分别用于维度汇总、区分性投诉词、主题、时间窗口和风险监测。`sentiment_alerts.csv` 同时保留文本候选与双信号验证状态；只有通过重采样稳定性和平台评分同向验证的记录计为有效预警，所有记录仍需人工复核。

## 历史资源

`resources/historical_reviews/` 保存可复用的历史评论与标签归档：

- `review_absa_reference.csv.gz`：本地全文归档，39,496 条去重评论，其中 28,724 条带历史十维标签；
- `manifest.json`、`README.md`：行数、时间范围、校验摘要和使用限制。

全文归档被 Git 忽略，仅在本地保留；公开仓库提交去标识标签和聚合结果。

## 本地运行

以下命令均在项目根目录执行。

### 浏览看板

只需 Python，即可读取仓库中的网页与 JSON：

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory app
```

打开 `http://localhost:8000`。浏览看板无需安装分析依赖或启动数据库。Windows 上如 Python 命令为 `python`，相应替换 `python3`。

### 配置分析环境

Python 版本要求为 3.11 或更高。使用虚拟环境安装 [requirements.txt](../requirements.txt) 中的依赖：

```bash
python3 -m venv .venv
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

Windows 命令提示符激活环境：

```bat
.venv\Scripts\activate.bat
```

Windows 上也可使用 `py -3 -m venv .venv` 创建环境，所选 Python 同样需要满足版本要求。激活后安装依赖并打开 Notebook：

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

在 `notebook/` 中选择 [中文报告](../notebook/China_Auto_Market_Analysis.ipynb) 或 [英文报告](../notebook/China_Auto_Market_Analysis_EN.ipynb)。两份报告读取已保存的预测与分析结果，不执行模型训练。

### 从已有分析结果重建看板 JSON

在已激活的分析环境中运行：

```bash
python app/build_dashboard_data.py --output-dir artifacts/dashboard-preview
```

输出为 `artifacts/dashboard-preview/` 下的 8 个 JSON 文件。构建器读取公开的分析结果，检查预测、统计摘要和版本的一致性，无需完整评论正文。此输出目录用于检查生成结果，不替换正式网页读取的 `app/static/data/`。

### 分析入口与输入要求

下表列出各入口读取的输入和生成位置。使用 `python <入口>` 运行；除注明可指定输出目录的入口外，重跑会更新表中列出的已保存结果。完整评论正文 `data/reviews/processed/target_371_review_corpus.csv` 未包含在公开仓库中。

| 分析 | 入口 | 输入与输出 |
|---|---|---|
| 时间切分 | `scripts/06_make_splits.py` | 读取销量建模快照和 `target_371_review_coverage.csv` 中的固定车系名单；默认写入 `data/processed/splits/`，可用 `--output-dir` 指定其他目录 |
| 评论时间特征 | `scripts/32_build_temporal_review_features.py` | 读取切分、`review_aspect_labels.csv` 和评论可用性表；更新 `data/reviews/processed/` 中的固定与滚动特征及审计摘要 |
| 固定六个月预测 | `scripts/33_evaluate_review_features.py --locked-capacity` | 读取切分、配置、已保存评论特征及模型选型摘要；默认写入 `data/processed/forecast/` 和 `assets/analysis/`，可用 `--output-dir` 将结果与图保存到其他目录 |
| 滚动单月预测 | `scripts/48_evaluate_rolling_origin.py --test` | 读取切分、配置及已保存评论特征；默认写入 `data/processed/forecast/`，可用 `--output-dir` 指定其他目录 |
| 年度配置分析 | `scripts/29_config_attribution.py` | 读取配置、销量及年度销量修正登记；更新 `data/processed/product/`、`assets/analysis/` 和年度销量修正审计表 |
| 用户需求与监测 | `scripts/35_build_user_needs_and_alerts.py` | 读取完整评论正文、标签及车系元数据；更新 `data/processed/user_feedback/` 和 `assets/analysis/`，需要额外准备未公开正文 |

固定预测的 `--locked-capacity` 沿用保存的树数与平台评分方案。滚动预测使用代码中既定的两套参数，在历史起点上评价；2026 窗口沿用原有时间切分，不作为新一轮调参数据。

Bootstrap、SHAP 模型重放与误差诊断，以及朴素基准比较，可写入单独目录：

```bash
python scripts/34_analyze_forecast_robustness.py --output-dir artifacts/report-statistics
python scripts/39_evaluate_naive_forecast_baselines.py --output-dir artifacts/report-statistics
```

两者默认读取 `data/processed/forecast/` 的固定预测及运行摘要和 `data/processed/splits/` 的切分。`--forecast-dir` 指定其他预测目录，`--split-dir` 指定基准或诊断所用的切分。34 的 SHAP 重放仍读取仓库默认建模面板、配置与评论特征，并核对其与保存预测的一致性；因此这两个参数不能单独切换完整模型输入。看板构建器读取正式分析目录，不会自动采用 `artifacts/report-statistics/` 中的结果。

<details>
<summary>可选：MySQL 数据仓库</summary>

完整本地数据链使用四个逻辑 MySQL 数据库，与 CSV 路径采用相同的样本定义、时间切分和计分规则：

| 层 | 作用 | 主要输出 |
|---|---|---|
| `auto_raw` | 保存来源批次和文件哈希 | 销量、配置、评论、标签和修正登记 |
| `auto_staging` | 标准化、统一车系、去重和时间对齐 | 标准销量、配置、评论与十维特征 |
| `auto_mart` | 组织事实表、维表和主题表 | 预测、产品配置和用户需求数据集市 |
| `auto_ops` | 记录运行、任务、质量检查和版本 | 批次状态、血缘与发布清单 |

运行此路径需要完整来源快照、独立的 MySQL 实例、MySQL 命令行客户端，以及使用 `mysql_config_editor` 配置的加密 login path。各入口通过 `--login-path` 指定连接，默认名称 `local-auto` 需要在运行机器上另行配置；凭据不包含在仓库中。客户端可从 `PATH` 查找，或通过 `MYSQL_CLIENT` 指定。

入口依次为 `scripts/initialize_mysql.py`、`scripts/ingest_raw.py`、`scripts/validate_raw.py`、`scripts/rebuild_staging.py`、`scripts/rebuild_core_marts.py` 和 `scripts/validate_core_marts.py`。初始化默认只读，`--apply` 才创建表；已有项目数据库需要显式 `--allow-existing`。摄取、清洗和主题层重建会写入数据库，完整运行所需的原始评论等本地文件不随公开仓库分发。

`tests/fixtures/ci/` 中的合成样本用于隔离 MySQL 集成检查，覆盖建表、幂等摄取和清洗质量规则。

</details>

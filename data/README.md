<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# 📦 数据说明

这里记录各模块使用的数据文件、样本范围和生成结果。原始平台数据仅用于学习、研究与项目展示；平台版权归相应来源方所有。

## 数据概况

| 模块 | 入口 | 规模 | 用途 |
|---|---|---:|---|
| 月度销量 | `processed/sales_filtered_24m.csv` | 54,918 行 / 1,017 个车系 | 仓库内建模快照；滚动单月预测与固定六个月压力测试 |
| 产品配置 | `raw/feature.csv` | 2,084 行 / 766 个车系 | 仓库内建模表；年度销量差异的产品属性分析 |
| 车主评论 | `reviews/processed/` | 24,175 条 / 345 个车系 | 用户需求、风险监测与口碑辅助实验 |

三个模块使用各自的样本筛选和评价口径，不将不同模块的指标直接横向比较。

## 数据仓库与血缘

本地完整工程链使用四个逻辑 MySQL 数据库：

| 层 | 作用 | 主要输出 |
|---|---|---|
| `auto_raw` | 按来源批次原样留存并记录文件哈希 | 销量、配置、评论、标签和修复证据原始表 |
| `auto_staging` | 标准化类型、统一车系、去重并执行时间可用性规则 | 标准销量、配置、评论与十维评论特征 |
| `auto_mart` | 为稳定下游接口组织事实表、维表和主题表 | 预测、产品配置和用户需求数据集市 |
| `auto_ops` | 横向保存运行、任务、质量结果和数据版本 | 批次状态、失败记录、血缘与发布清单 |

公开仓库中的 CSV/JSON 是经过验证的便携快照，不要求浏览者连接本地数据库。完整本地数据链和便携公开快照使用相同的时间切分、样本定义和指标合同。

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
| `reviews/processed/` | 去标识语料、标签和时间特征 |
| `resources/` | 可复用的历史评论资源归档 |

## 建模输入

### 月度销量：`processed/sales_filtered_24m.csv`

- 粒度：车系 × 自然月；时间范围：2022-01—2026-06；
- 主要字段：`series_id`、`series_name`、`brand`、`category`、`year`、`month`、`monthly_sales`；
- 负销量记录为 0；排名、累计销量和网站展示价格等源站派生字段不作为预测特征。

### 产品配置：`raw/feature.csv`

- 粒度：车系 × 年款，不是 trim 级车型清单；
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

预测面板保留每个目标车系的自然月间隔，仅保存销量、日历和滞后；原始配置由模型读取并按训练窗口处理，不在共享切分或预测mart中保存全表编码。基础特征含1/2/3月滞后及3/6月均值，主模型另含12月滞后与12月均值。

主协议是滚动单月预测：每月预测下一个月，并使用已公布的上月真实销量。固定起点六个月协议从 2026-01 一次性递归预测，作为信息受限压力测试；两种协议分别评价。

选定方案后，最终模型在train+val上重新拟合至2025-12，测试期不更新权重。销量模型在每个预测起点限定可用配置年份，按行连接不晚于其年份的最近记录，并仅在该模型实际训练行中拟合中位数与类别词表；预测窗口内配置冻结。未知类别记为−1，全缺数值列使用0占位，不因配置缺失删除销量样本。配置源没有年内发布时间，销量源也没有逐条历史发布与修订版本，因此年度代理对齐仍不等于完整历史点时重建。

## 主要产物

### 销量预测：`processed/forecast/`

| 文件 | 用途 |
|---|---|
| `rolling_origin_summary.json` | 滚动主协议的历史起点验证、门槛和锁定测试摘要 |
| `rolling_origin_test_predictions.csv` | 测试期逐车系逐月预测与同场景朴素基准 |
| `forecast_benchmark_comparison.csv` | 固定压力测试与朴素基准对比 |
| `review_feature_ablation_summary.csv` | 固定场景口碑特征消融 |
| `forecast_robustness_summary.json` | 聚类 Bootstrap、分组误差和稳健性摘要 |

当前保存的滚动结果为29.75%全局WMAPE、707.68辆MAE、1,700.38辆RMSE；固定六个月平台评分模型为38.09%WMAPE。`src/china_auto_market/forecasting/reporting.py`从逐行预测计算补充指标，供Notebook与看板共用。sMAPE保留全部行，双方为零时记0；MAPE只用正销量行。定义与完整对照见根目录[README.md](../README.md)。

### 产品配置：`processed/product/`

- `config_attribution_ablation.csv`：年份、品牌、配置的逐步消融；
- `config_importance_annual.csv`：年度配置特征重要性。
- `config_attribution_summary.json`：完整年份范围、样本规模和核心指标。

该模块报告`log1p(年度销量)`尺度的五折R²均值和折间标准差；完整模型0.239，配置增量0.169。WMAPE使用还原为辆数的合并折外预测，不与月度预测直接比较，也不代表因果效应。

### 用户需求：`processed/user_feedback/`

`user_need_aspect_summary.csv`、`user_need_keywords.csv`、`user_need_topics.csv`、`sentiment_monitoring_windows.csv` 和 `sentiment_alerts.csv` 分别用于维度汇总、区分性投诉词、主题、时间窗口和风险监测。`sentiment_alerts.csv` 同时保留文本候选与双信号验证状态；只有通过重采样稳定性和平台评分同向验证的记录计为有效预警，所有记录仍需人工复核。

## 历史资源

`resources/historical_reviews/` 保存可复用的历史评论与标签归档：

- `review_absa_reference.csv.gz`：本地全文归档，39,496 条去重评论，其中 28,724 条带历史十维标签；
- `manifest.json`、`README.md`：行数、时间范围、校验摘要和使用限制。

全文归档被 Git 忽略，仅在本地保留；公开仓库提交去标识标签和聚合结果。

## 复现入口

以下命令均在项目根目录执行。浏览看板不需要模型训练或本地数据库；生成JSON和重跑分析需要先在项目虚拟环境`.venv`中安装依赖。

### 浏览已保存的看板

```bash
python3 -m http.server 8000 --directory app
```

打开`http://localhost:8000`，直接读取仓库内的HTML与JSON，无需完整评论正文。

### 从已有分析结果生成JSON

```bash
.venv/bin/python app/build_dashboard_data.py --output-dir artifacts/dashboard-preview
```

此命令使用公开的已保存分析结果，不重新训练模型，也不读取本地完整评论语料。输出写入独立目录，不替换`app/static/data/`；现有网页仍显示原发布版本。输入版本或预测与统计摘要不一致时，构建会失败，不应跳过校验。

### 重新运行分析（需要额外来源与版本核验）

下表是分析入口，不是一组可以无条件连续执行的发布命令。正式模型、统计和报告需要成套核验后才能更新；默认输出可能覆盖已有结果，重跑前应备份或使用入口支持的独立输出目录。

| 环节 | 入口与条件 |
|---|---|
| 切分与评论时间特征 | `06_make_splits.py`、`32_build_temporal_review_features.py`；需要相应销量、配置或评论特征来源 |
| 固定与滚动销量模型 | `33_evaluate_review_features.py --locked-capacity`、`48_evaluate_rolling_origin.py --test`；保持既定方案与时间切分，不按2026窗口重新选型 |
| 配置分析 | `29_config_attribution.py`；需要配置与年度销量输入 |
| 用户需求及监测 | `35_build_user_needs_and_alerts.py`；需要本地`data/reviews/processed/target_371_review_corpus.csv`完整语料，公开仓库不包含此文件 |

预测生成并核验后，统计计算单独暂存：

```bash
.venv/bin/python scripts/34_analyze_forecast_robustness.py --output-dir artifacts/report-statistics
.venv/bin/python scripts/39_evaluate_naive_forecast_baselines.py --output-dir artifacts/report-statistics
```

**在此暂停核验。** 默认读取正式预测目录与切分；独立实验目录应同时通过`--forecast-dir`和`--split-dir`指定配套输入。输出先留在`artifacts/report-statistics`，核对样本键、模型版本、预测哈希与统计来源后，再由维护者同步整套正式结果并执行发布合同。看板构建器不会自动采用暂存统计，不能紧接着运行构建并把旧统计当成新结果。当前报告不使用上市曲线覆盖预测。

如需完整评论标签流水线，请先按根目录 Notebook 和脚本注释准备评论语料；缺失评论标签的补标步骤为可选项。

若拥有本地完整来源快照和独立 MySQL 实例，可按工程顺序运行：

```bash
.venv/bin/python scripts/initialize_mysql.py --apply
.venv/bin/python scripts/ingest_raw.py --dataset all --mode full
.venv/bin/python scripts/validate_raw.py
.venv/bin/python scripts/rebuild_staging.py
.venv/bin/python scripts/rebuild_core_marts.py
.venv/bin/python scripts/validate_core_marts.py
```

数据库凭据只通过本机加密 login path 读取，不接受命令行明文密码。隔离 MySQL 集成测试使用 `tests/fixtures/ci/` 下的虚构小样本验证相同 DDL、摄取、幂等和 staging 质量规则；它不能替代完整数据的业务评价。

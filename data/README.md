<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# 数据说明

本目录保存原始表、时间切分、评论语料、模型产物和数据审计。可复用的历史评论与标签单独放在 `resources/`。

## 目录分层

| 路径 | 内容 | 是否进入当前流程 |
|---|---|---|
| `raw/` | 月销量与车型配置原始表 | 是 |
| `processed/` | 清洗结果、时间切分、模型预测、归因与监测产物 | 是 |
| `reviews/raw/` | 采集评论、源站详情与采集清单 | 是，经质量筛选后 |
| `reviews/processed/` | 严格评论语料、结构化标签、月度特征与审计 | 是 |
| `resources/` | 历史评论与标签归档 | 参考与扩展 |

## 三组分析样本

| 分析 | 最终样本 | 主要筛选条件 |
|---|---:|---|
| 月度销量预测 | 371 个车系 | 月销量连续、配置可对齐、具备完整测试期 |
| 产品配置分析 | 736 个车系；2,007 条车系年记录 | 2022—2026 年销量与配置可对齐 |
| 用户需求与风险 | 24,175 条评论；345 个车系 | 完整正文、有效发布时间、可核验来源 |

这些数字不是同一张表逐级删减的结果。每项分析按自己的数据需求建立样本。

## 1. 原始数据

### `raw/monthly_sales.csv`

太平洋汽车口径的车系月销量面板。

| 项目 | 数值 |
|---|---:|
| 记录数 | 54,918 |
| 车系数 | 1,017 |
| 时间范围 | 2022-01—2026-06 |
| 负销量 | 0 |
| 主要粒度 | 车系 × 自然月 |

建模主要使用：

| 字段 | 说明 |
|---|---|
| `year`, `month`, `period` | 时间字段 |
| `series_id`, `series_name` | 源站车系标识与车系名 |
| `brand`, `category` | 品牌与车型类别 |
| `monthly_sales` | 当月销量 |
| `数据来源` | 来源标记 |

排名、累计销量、网站展示价格等字段属于源站元信息，不进入预测特征。

### `raw/feature.csv`

车型配置表，粒度为“车系 × 年款”，不是 trim 级车型清单。

| 项目 | 数值 |
|---|---:|
| 记录数 | 2,084 |
| 车系数 | 766 |
| 字段数 | 84 |
| 唯一键 | `series_name, year` |
| 年度销量覆盖 | 760 / 766 个车系 |

配置字段可分为：

- 基础属性：品牌、年款、能源类型、车型级别、指导价；
- 动力系统：发动机、电机、变速箱、加速与能耗；
- 电池与续航：容量、类型、续航、充电；
- 车身：长宽高、轴距、质量、座位与车身结构；
- 安全与座舱：气囊、屏幕、座椅、音响与空调。

发动机、电机和电池字段存在结构性缺失。例如纯电车型没有发动机参数，燃油车型没有电池参数；这些缺失不能简单解释为数据错误。

## 2. 月度预测样本与时间切分

`processed/splits/` 保存 371 车系的固定时间切分：

| 文件 | 时间 | 用途 |
|---|---|---|
| `train.csv` | 截至 2025-06 | 训练 |
| `val.csv` | 2025-07—12 | 参数与方案选择 |
| `test.csv` | 2026-01—06 | 最终评价 |
| `split_index.csv` | 全部样本 | 每行所属数据段 |
| `manifest.json` | — | 车系数、行数、时间边界和版本信息 |

主实验采用固定起点：2026 年 1 月一次性预测未来六个月。递归预测时，后续月份只能使用此前预测出来的销量滞后，不能读取测试期真实销量。

## 3. 评论语料

### 原始采集层

`reviews/raw/` 在本地保存懂车帝与汽车之家两类公开车主评论及采集清单。完整评论含平台用户标识，不随公开仓库提交；采集清单和去标识后的分析产物保留在仓库中。

主要审计文件：

| 文件 | 用途 |
|---|---|
| `dongchedi_incremental_manifest.csv` | 懂车帝增量采集清单 |
| `autohome_incremental_manifest.csv` | 汽车之家增量采集清单 |
| `autohome_incremental_review_details.csv` | 汽车之家详情正文 |
| `processed/review_collection/autohome_id_resolutions.csv` | 车系映射与解析结果 |
| `processed/review_collection/sentiment_resolution_exceptions.csv` | 未解决项及止损说明 |

### 严格建模语料

本地文件 `reviews/processed/target_371_review_corpus.csv` 包含 371 个目标车系的候选评论、质量标记和来源审计。进入模型的记录必须同时满足：

1. 车系身份有效；
2. 发布时间可解析；
3. 正文完整且达到质量要求；
4. 在相应预测起点之前已经发布。

最终可用评论为 24,175 条，覆盖 345 个车系。另有 103 条汽车之家列表摘要无法取得详情全文；这些记录保留在审计表，但 `eligible_for_temporal_model=False`。

| 覆盖口径 | 车系数 |
|---|---:|
| 有任意合格评论 | 345 |
| 2026-01 固定起点前有评论 | 330 |
| 2026-01 起点前最近 180 天有评论 | 272 |

## 4. 评论标签与月度特征

### 评论级标签

[结构化评论标签](./reviews/processed/review_aspect_labels.csv) 对每条合格评论保存十个维度：

`appearance`, `interior`, `space`, `power`, `control`, `comfort`, `fuel_consumption`, `configuration`, `intelligence`, `value`。

每个维度分开保存两类信息：

- 是否明确提及该维度；
- 提及后的评价方向：`-1` 负面、`0` 中性、`1` 正面。

平台自带星级评分与文本标签互不替代。评论没有提及某维度时，不会被当作中性评价写入该维度的正负比例。

### 防泄漏月度特征

[固定起点月度评论特征](./reviews/processed/review_features_by_series_month_fixed_origin.csv) 有 13,866 行，覆盖 371 个车系和 51 个预测月份。主要字段包括：

- 截止预测起点的累计评论数；
- 最近 180 天评论数与可用性；
- 十个维度的历史均值；
- 最近 180 天正面率、负面率与提及率；
- 综合维度得分与任意正/负面比例；
- `information_cutoff_exclusive`：该行特征的信息截止时间。

固定起点测试特征统一冻结在 `2026-01-01` 之前。滚动起点特征另存为补充产物，不用于主结果。

## 5. 分析产物

### 销量预测：`processed/forecast/`

| 产物 | 内容 |
|---|---|
| `review_feature_ablation_summary.csv` | 371 车系方案对比 |
| `review_feature_predictions.csv` | 测试期逐车系、逐月预测 |
| `review_feature_series_metrics.csv` | 逐车系误差 |
| `forecast_robustness_bootstrap.csv` | 按车系重采样结果 |
| `review_feature_shap_importance.csv` | 特征贡献 |
| `cold_start_launch_curve_summary.json` | 冷启动方法与结果 |

主结果中的全局 WMAPE 为：销量基线 40.44%，用户口碑增强 38.71%，冷启动补充后 38.64%。

### 产品配置：`processed/product/`

| 产物 | 内容 |
|---|---|
| `config_attribution_ablation.csv` | 年份、品牌与配置的逐步消融 |
| `config_importance_annual.csv` | 年度配置重要性 |

完整模型的五折分组交叉验证 R² 为 0.300。

### 用户需求与风险：`processed/user_feedback/`

| 产物 | 内容 |
|---|---|
| `user_need_aspect_summary.csv` | 十维度提及与正负分布 |
| `user_need_keywords.csv` | 维度关键词 |
| `user_need_topics.csv` | 维度内主题 |
| `sentiment_monitoring_windows.csv` | 相邻 180 天窗口统计 |
| `sentiment_alerts.csv` | 历史与当前规则预警 |

当前预警只是人工复核入口，不代表已经确认产品缺陷。

## 6. 历史资源归档

`resources/historical_reviews/` 保存可复用的历史评论资源：

| 文件 | 内容 |
|---|---|
| `review_absa_reference.csv.gz` | 本地全文归档：39,496 条去重评论，其中 28,724 条带历史十维标签 |
| `manifest.json` | 行数、车系数、时间范围、SHA-256 与标签语义 |
| `README.md` | 使用方式和限制 |

当前语料复用了其中 16,538 条已经生成的历史标签。归档表保留原文、时间、车型、购车信息和平台评分，仅在本地保存；公开仓库提交去标识标签和聚合结果。

历史标签中的 `0` 不能可靠区分“未提及”“中性”和解析回退，因此不直接作为维度提及真值；当前模型使用单独的统一提及标记。

## 7. 复现与更新

先按根目录 `environment.yml` 创建并激活项目环境，再运行脚本：

```bash
conda activate nlp-sentiment
python scripts/<script>.py
```

评论语料、标签和月度特征的主要顺序为：

```text
18 生成目标车系语料
21 多来源质量审计
25 检查时间可用性
27 生成平台评分与词典标签
28 聚合本地评论月度特征
30 补标缺失评论（可选，需配置 API）
31 合并评论级标签
32 生成防泄漏月度特征
33 运行 371 车系预测消融
34 稳健性分析
35 用户需求与风险监测
36 冷启动验证
37 生成报告 Notebook
```

原始平台数据版权归相应来源方。本目录的数据仅用于学习、研究与项目展示。

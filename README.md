<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

<h1 align="center">🇨🇳 中国汽车市场分析：销量预测、产品配置与用户需求</h1>

<p align="center">基于公开月销量、车型配置与 24,175 条车主评论的汽车市场研究项目</p>

<p align="center">
  <a href="https://yemyu.github.io/china-auto-market-analysis/"><b>在线研究看板</b></a>
  · <a href="./notebook/China_Auto_Market_Analysis.ipynb">分析 Notebook</a>
  · <a href="./data/README.md">数据文档</a>
</p>

---

> **一句话概括**　用公开销量、车型配置和车主评论，构建一套可复现的汽车市场分析流程：预测下个月销量，解释年度产品差异，并监测用户需求变化。

## 研究问题

1. 📈 在每月更新信息的情况下，下个月的车系销量能否稳定预测？
2. 🧩 哪些产品配置能够解释不同车系之间的年度销量差异？
3. 💬 用户最在意哪些产品维度，哪些口碑信号值得进一步复核？

## 项目概览

| 模块 | 样本 | 研究内容 | 当前口径 |
|---|---:|---|---|
| 📈 滚动单月销量预测 | 371 个车系 | 模拟每月拿到最新销量后预测下月 | 主结果；完整月度面板与严格时间切分 |
| 🧪 固定六个月压力测试 | 同一 371 个车系 | 一次性从 2026-01 递归预测六个月 | 辅助场景；与主结果分开评估 |
| 🧩 产品配置分析 | 646 个车系，1,510 条完整车系年记录 | 估计年份、品牌与配置对年度销量差异的增量解释力 | 2022—2025 完整自然年；`GroupKFold(5)` 按车系分组 |
| 💬 用户需求与风险 | 24,175 条评论，覆盖 345 个车系 | 识别十类产品需求、负面集中度和口碑异常 | 结构校验、抽样审计、相邻 180 天窗口 |

三项分析采用不同筛选条件：销量预测固定 371 个车系的完整自然月面板，年度配置只向不晚于目标年份的记录回退；产品配置分析要求完整年度销量与配置可对齐；用户需求分析要求完整、可核验的评论正文。

## 结果速览

| 关键结论 | 当前结果 |
|---|---:|
| 📈 滚动单月主结果 | **29.72% WMAPE**，比上月销量基准低 11.27 个百分点 |
| 🧪 固定六个月压力测试 | **38.38% WMAPE**，比最近六个月均值减少 44.6% 绝对误差 |
| 🧩 配置对年度差异的增量解释 | GroupKFold R² **0.239** |
| 💬 用户需求监测 | 24,175 条评论、10 个维度、123 个达标车系 |

## 主要结果

### 1. 📈 销量预测

测试窗口为 2026 年 1—6 月，共 371 个车系、2,226 个车系月。主任务是每月更新的下月预测；固定起点六个月结果作为压力测试。主指标为全局 volume-weighted WMAPE，逐车系中位数用于补充观察长尾。

#### 滚动单月主结果

| 方案 | 全局 WMAPE ↓ | 逐车系中位数 WMAPE ↓ | 相对同场景朴素基准 |
|---|---:|---:|---:|
| 沿用上月销量（朴素基准） | 40.99% | 48.36% | — |
| **滚动单月季节增强 XGBoost（主结果）** | **29.72%** | **36.74%** | **−11.27 pp（约 −27.5%）** |

滚动预测在每个测试月使用当时已经公布的上月真实销量；参数在六个月评估窗内保持锁定，避免把未来月份信息带回模型选择。

#### 固定六个月压力测试

| 方案 | 全局 WMAPE ↓ | 逐车系中位数 WMAPE ↓ |
|---|---:|---:|
| 固定起点最近 6 个月均值（朴素） | 69.31% | 89.60% |
| 固定六个月综合方案（口碑＋冷启动保护） | **38.38%** | **46.88%** |

固定压力测试相对同场景最近 6 个月均值的绝对误差降低 44.6%。该协议承受六个月递归滞后，与滚动单月协议分别评估。口碑增强的点估计改善 0.697 个百分点；5,000 次车系聚类 Bootstrap 的 95% 区间为 −0.234 至 1.873 个百分点，稳定增益证据不足，因此定位为辅助特征。冷启动保护覆盖 9 个同时缺少历史正销量和起点前配置记录的边界车系，不改变滚动主模型。

销量主模型为带 12 个月季节滞后与 12 个月历史均值的滚动单月 XGBoost；ARIMA、Prophet、LSTM 等早期候选仍保留在实验脚本中，用于记录模型选择过程，不混入当前主结果表。

### 2. 🧩 产品配置与年度销量差异

| 特征组合 | 分组交叉验证 R² | 年度截面 WMAPE（模块内辅助） |
|---|---:|---:|
| 年份 | 0.013 | 87.50% |
| 年份 + 品牌 | 0.070 | 83.82% |
| 年份 + 品牌 + 配置 | **0.239** | **73.85%** |

加入配置后 R² 比“年份＋品牌”增加约 0.169，表明产品属性对车系间年度销量差异具有增量解释力。分析仅使用 2022—2025 四个完整自然年，避免把 2026 年上半年累计与完整年度混用。该模块采用样本外评估，量化年度跨车系差异的解释力，不进行因果识别；年度截面 WMAPE 作为模块内辅助误差指标，与月度预测指标分别报告。

### 3. 💬 用户需求与风险

评论拆分为空间、动力、操控、舒适、能耗、配置、智能化、性价比、外观和内饰十个维度，分别记录提及、正负倾向与时间窗口。最近完整监测月为 2026 年 7 月；123 个车系达到监测门槛，当前有 1 条规则预警进入人工复核。

## 研究设计与数据可用性

| 数据段 | 时间 | 用途 |
|---|---|---|
| Train | 截至 2025-06 | 模型训练 |
| Validation | 2025-07—12 | 参数与方案选择 |
| Test | 2026-01—06 | 最终评价 |

滚动主结果在每个测试月使用已公布的信息；固定压力测试从 2026-01 起递归生成后续销量滞后。评论特征只使用相应信息截止日前已经发布的评论。严格语料保留完整正文、发布时间和可审计来源；列表摘要等不具备详情正文的记录不进入时间模型。缺失评论保持为缺失及可用性标记，不填成中性。

全局 WMAPE 定义为：

```text
Σ |实际销量 − 预测销量| / Σ 实际销量
```

## 快速开始

无需启动后端即可查看预烘焙看板：

```bash
git clone https://github.com/Yemyu/china-auto-market-analysis.git
cd china-auto-market-analysis
python3 -m http.server 8000 --directory app
```

打开 `http://localhost:8000` 即可浏览中文与英文页面。若要重建分析产物，请使用项目环境中的 Python，并参考 [数据说明](./data/README.md) 的复现入口。

## 看板与复现

看板由原生 HTML、CSS、JavaScript 与 ECharts 构建，数据预先写入 `app/static/data/`，不依赖后端服务。中文与英文共六页，首页展示滚动单月主结果，销量页同时展示固定六个月压力测试和固定场景口碑消融。

<details open>
  <summary><b>功能截图</b></summary>

<br>

<table>
  <tr><td width="50%"><a href="./assets/dashboard/zh/01-overview.png"><img src="./assets/dashboard/zh/01-overview.png" alt="项目概览"></a></td><td width="50%"><a href="./assets/dashboard/zh/02-sales-forecast.png"><img src="./assets/dashboard/zh/02-sales-forecast.png" alt="销量预测"></a></td></tr>
  <tr><td align="center">项目概览</td><td align="center">销量预测</td></tr>
  <tr><td><a href="./assets/dashboard/zh/03-user-needs.png"><img src="./assets/dashboard/zh/03-user-needs.png" alt="用户需求"></a></td><td><a href="./assets/dashboard/zh/04-product-config.png"><img src="./assets/dashboard/zh/04-product-config.png" alt="产品配置"></a></td></tr>
  <tr><td align="center">用户需求</td><td align="center">产品配置</td></tr>
  <tr><td><a href="./assets/dashboard/zh/05-risk-monitor.png"><img src="./assets/dashboard/zh/05-risk-monitor.png" alt="风险监测"></a></td><td><a href="./assets/dashboard/zh/06-brand-series.png"><img src="./assets/dashboard/zh/06-brand-series.png" alt="品牌与车系"></a></td></tr>
  <tr><td align="center">风险监测</td><td align="center">品牌与车系（比亚迪示例）</td></tr>
</table>

</details>

直接预览看板（数据已预烘焙）：

```bash
python3 -m http.server 8000 --directory app
```

重新生成看板数据：

```bash
python3 app/build_dashboard_data.py
```

滚动结果产物包括：

- `data/processed/forecast/rolling_origin_summary.json`
- `data/processed/forecast/rolling_origin_validation.csv`
- `data/processed/forecast/rolling_origin_test_predictions.csv`

## 目录

```text
china-auto-market-analysis/
├── app/                    静态研究看板与预烘焙 JSON
├── assets/                 分析图与看板截图
├── data/                   原始、处理后与审计数据
├── notebook/               中英文分析 Notebook
├── scripts/                可复现脚本
├── environment.yml
├── requirements.txt
├── README.md
└── README_EN.md
```

## 数据与许可

月销量、车型配置和车主评论来自公开汽车平台。原始平台数据版权归相应来源方；仓库中的数据仅用于学习、研究与项目展示，不用于商业用途。代码与项目文档采用 MIT License。更完整的表结构、样本筛选、时间可用性和资源归档说明见 [data/README.md](./data/README.md)。

<p align="center">
  <a href="./README.md">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="./README_EN.md">🌐 English</a>
</p>

<h1 align="center">AutoPulse：汽车销量预测与用户舆情分析</h1>

<p align="center">
  多源汽车数据 + 用户口碑舆情 + 销量预测与归因 → 一套端到端分析流水线与交互式网页看板
</p>

<video src="https://github.com/user-attachments/assets/6072ccda-0d16-4865-8937-c1c3b73eeaa5" width="900" controls></video>


---

## 项目简介

**AutoPulse** 是一套面向汽车行业的数据洞察项目，把「太平洋汽车月度销量」与「车型配置参数（汽车之家/太平洋）」两类公开数据打通，先做**无舆情基线**（销量预测 + 配置归因），再接入用户口碑舆情（**Phase B**）。回答三个核心问题：

1. **下个月能卖多少？** —— 用 ARIMA / Prophet / XGBoost / LSTM / 融合模型做销量预测。
2. **用户舆情真的影响销量吗？** —— 用大模型做 ABSA（Aspect-Based Sentiment Analysis）逐维度情感分析，再用 SHAP / Granger 因果量化影响。
3. **如何持续监控？** —— 把前五阶段结论封装成纯静态 HTML + ECharts 交互式网页看板，实现品牌→车系下钻、情感预警、归因可视化。

> 这是一个数据分析&开发项目：从原始数据采集、清洗、建模、归因到最终交互看板。

---

## 在线看板

- 🌐 **在线演示**：https://yemyu.github.io/AutoPulse/
- 本地预览：`cd app && python -m http.server 8000`，浏览器打开 http://localhost:8000/

> 完整环境安装、本地运行与数据更新见下方「快速开始」。

---

## 六阶段工作流

项目按真实工作流拆成 6 个阶段，对应 `scripts/new_pipeline/` 下 `06_` ~ `20_` 流水线脚本与 `notebook/AutoPulse_Analysis.ipynb`。

> 当前已完成「无舆情基线」：阶段一~四（数据/筛选/月度预测 + 配置归因）。舆情相关的阶段五~六属于 **Phase B（待接入新数据）**，方法论已就绪，将在新口径（371 车系）下重做。

### 阶段一 · 数据准备

**问题**：如何把太平洋汽车月度销量与车型配置（汽车之家/太平洋）两套公开数据对齐？

**方法**：
- 采集 `monthly_sales.csv`（1,017 车系 / 54,918 条月度销量，2022-01~2026-06，来源 pcauto）。
- 采集 `feature.csv`（766 车系 / 2,084 行 / 84 列；粒度=**车系×年**，`(series_name, year)` 唯一键，含 `annual_sales` 年度销量汇总；来源汽车之家/太平洋等）。
- 按 `series_name` 直接对齐两套数据（新管线无需旧 `series_mapping` 桥接）：

```
monthly_sales ──series_name──┐
                             ├──> 对齐 → 371 个「月度+配置」车系(腿B) + 736 个有年度销量车系(腿A)
feature       ──series_name──┘
```

**结果**：对齐得到 371 个 in-population 车系（腿B 月度预测）与 736 个有年度销量的车系（腿A 配置归因）；646 个仅销量无配置的车系已排除。

<p align="center">
  <img src="figures/sentiment_vs_sales.png" alt="阶段一：配置覆盖与销量" width="700">
</p>

---

### 阶段二 · 数据筛选与探索性可视化

**问题**：哪些车系适合进入预测模型？整体市场长什么样？

**方法**：
- 月度面板按连续覆盖筛选；最终建模采用 371 个 in-population 车系（月度销量与配置齐全）。
- 绘制全市场销量趋势、车型级别/能源类型分布、价格与硬件特征分布。

**结果**：
- 371 个车系进入腿B 月度建模；736 个车系进入腿A 年度归因。
- 识别出新能源占比、级别分布、头部集中度等宏观特征。

<p align="center">
  <img src="figures/sales_trend.png" alt="阶段二：全市场月度销量趋势" width="700">
</p>

---

### 阶段三 · 销量预测建模（腿B，无舆情）

**问题**：多种时序模型中，谁能更稳健地预测月销量？配置/外生变量有用吗？

**方法**：
- 在 371 个 in-population 车系上，按**绝对时间**切分：`train` 2022-01~2025-06 / `val` 2025-07~2025-12 / `test` 2026-01~2026-06（见 `data/processed_new/splits/`）。
- 横向对比 ARIMA / Prophet / Prophet+外生 / XGBoost / LSTM；XGBoost 做递归多步（6 月）预测，val 上早停。
- 指标：WMAPE（体积加权 + per-series 中位数双口径，抗长尾）+ 特征消融 + 90% 预测区间。

**结果**：
- **XGBoost** 为月度预测基线：test 体积加权 WMAPE **63.2%**（per-series 中位数 59.4%）。
- 消融证明 **历史销量滞后特征主导预测**：去掉 lag → 73.2%；配置特征在月度预测中**无正贡献**（NO-CONFIG 48.8% 反而更低，旧「配置 +16%」实为未来配置泄漏伪影，已修复）。
- 误差随预测步长增长而扩大，符合直觉。

<p align="center">
  <img src="figures/model_comparison.png" alt="阶段三：多模型对比" width="700">
</p>

---

### 阶段四 · 配置→销量归因（腿A，无舆情）

**问题**：什么样的配置卖得好？配置能在多大程度解释销量差异？

**方法**：
- **车系×年横截面回归**：目标 `y = log1p(annual_sales)`；`GroupKFold(5) by series_name`（防止同一车系跨折泄漏，因为同车系跨年配置近乎不变）。
- 特征 = 配置（价格/尺寸/能源/马力/电池…）+ 品牌 + 年；用 XGBoost 重要性解释各维度贡献。

**结果**：
- R² 递进：仅年 **0.089** → +品牌 **0.154** → +配置 **0.303**（配置增量 ΔR² = +0.149）。
- 特征重要性：配置 **76.1%** / 品牌 **22.5%** / 年 **1.4%**。
- 结论：配置解释**车系之间**的销量差异（更贵/更大/电动的车型更卖座）；同车系跨年涨跌由非配置因素（换代、权益、口碑）驱动。

<p align="center">
  <img src="figures/stage4_shap_summary.png" alt="阶段四：配置特征重要性" width="700">
</p>

---

### 阶段五 · 舆情融合预测与话题预警（Phase B 待接入）

**问题**：把舆情动态加入销量预测模型，能不能提升精度？哪些话题需要预警？

**方法（计划）**：在新口径（371 车系）上重做——大模型 ABSA 逐维度打分、动态情感作为外生变量、TF-IDF/LDA 主题聚类、规则化预警。

**结果**：Phase B 待接入新数据后补充。旧管线（669 系）经验：动态情感未提升 volume-weighted 精度（XGBoost-baseline 34.79% vs +Top3sent 35.21%），但对尾部小销量车系可降低 per-series WMAPE（327% → 311%）。

---

### 阶段六 · 交互式网页看板

**问题**：如何让非技术决策者也能按“问题 → 证据 → 结论”浏览全部成果？

**方法**：
- 用 **HTML + ECharts** 搭建 7 屏纯静态交互式看板：项目概览、销量预测、舆情 ABSA、销量归因、舆情↔销量关系、舆情预警、品牌/车型钻取。
- 数据由 `app/build_dashboard_data.py` 预烘焙为 `app/static/data/*.json`，前端直接 `fetch` 读取，无需任何后端服务。
- 支持中英双语切换；品牌/车型钻取支持 Tab 联动下钻。

**结果**：本地启动即可在浏览器中交互式查看全部分析结论，无需重新跑模型。

> 注：看板数据桥当前基于初始管线快照；Phase B 接入后将刷新为 371 车系新口径。

### 看板截图

<details>
  <summary><b>看板完整截图（点击展开）</b></summary>

<p align="center">
  点击任意图片可查看完整分辨率。
</p>

<table align="center">
  <tr>
    <td align="center" width="50%"><img src="figures/dashboard_full_overview.png" width="400" alt="项目概览"/></td>
    <td align="center" width="50%"><img src="figures/dashboard_full_forecast.png" width="400" alt="销量预测"/></td>
  </tr>
  <tr>
    <td align="center">项目概览</td>
    <td align="center">销量预测</td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="figures/dashboard_full_absa.png" width="400" alt="舆情 ABSA"/></td>
    <td align="center" width="50%"><img src="figures/dashboard_full_attribution.png" width="400" alt="销量归因"/></td>
  </tr>
  <tr>
    <td align="center">舆情 ABSA</td>
    <td align="center">销量归因</td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="figures/dashboard_full_relation.png" width="400" alt="舆情↔销量关系"/></td>
    <td align="center" width="50%"><img src="figures/dashboard_full_alerts.png" width="400" alt="舆情预警"/></td>
  </tr>
  <tr>
    <td align="center">舆情↔销量关系</td>
    <td align="center">舆情预警</td>
  </tr>
  <tr>
    <td align="center" colspan="2"><img src="figures/dashboard_full_drilldown.png" width="400" alt="品牌/车型钻取"/></td>
  </tr>
  <tr>
    <td align="center" colspan="2">品牌/车型钻取</td>
  </tr>
</table>

</details>

---

## 快速开始

### 环境

依赖已整理在 `requirements.txt`，用你习惯的 Python 环境（venv / conda / 全局均可）安装即可：

```bash
pip install -r requirements.txt
```

### 启动网页看板（本地）

```bash
cd app && python -m http.server 8000
```

打开浏览器访问 http://localhost:8000/ 即可预览。（看板是纯静态站点，不依赖任何后端服务。）

### 在线看板

- 🌐 **在线演示**：https://yemyu.github.io/AutoPulse/
- 看板所需数据已预烘焙在 `app/static/data/*.json`，**无需重跑任何采集或建模脚本即可直接查看**。如需在本地更新数据桥（需已跑过完整管线、本地存在 `data/processed_new/*.csv`），运行：

```bash
python app/build_dashboard_data.py
```

---

## 目录结构

```
AutoPulse/
├── app/                           # 阶段六 · 纯静态网页看板（HTML + ECharts，GitHub Pages 部署）
│   ├── index.html / forecast.html / …  # 7 屏静态页面
│   ├── build_dashboard_data.py    # 预烘焙 JSON 数据桥
│   ├── .nojekyll                  # 禁用 GitHub Pages 的 Jekyll 处理
│   └── static/                    # CSS/JS/JSON 数据
├── data/                          # 数据目录（CSV 已 gitignore）
│   ├── README.md                  # 数据说明（中文）
│   ├── README_EN.md               # 数据说明（英文）
│   ├── raw/                       # monthly_sales.csv, feature.csv
│   ├── sentiment/                 # 口碑明细与汇总
│   └── processed_new/             # 阶段产物（新管线，可复现）
├── figures/                       # 分析结果图、看板截图与交互演示（入库）
├── LICENSE                        # MIT 许可证
├── notebook/                      # 中英双语数据分析笔记本
│   ├── AutoPulse_Analysis.ipynb
│   └── AutoPulse_Analysis_EN.ipynb
├── scripts/                       # 01_~20_ 流水线脚本
├── config/                        # 配置与 .env 模板
├── requirements.txt               # Python 依赖
├── README.md                      # 本文件（中文）
└── README_EN.md                   # 英文版
```

---

## 技术栈

- **数据采集**：Python `requests` + `BeautifulSoup` / 懂车帝口碑 API
- **数据处理**：Pandas、NumPy、ETL Pipeline
- **NLP**：jieba、Hugging Face Transformers、DeepSeek API（ABSA）
- **机器学习 / 时序**：scikit-learn、XGBoost、 Prophet、statsmodels、PyTorch（LSTM）
- **可视化**：Matplotlib、ECharts（网页看板）
- **Web 看板**：原生 HTML/CSS/JS、ECharts 5（纯静态，GitHub Pages 托管）
- **依赖管理**：`requirements.txt`

---

## 数据说明

- 所有数据来自**公开汽车平台**（太平洋汽车月度销量、汽车之家/太平洋车型配置；用户舆情口碑为 Phase B，另采集自懂车帝）。
- 原始 / 中间数据体积较大，已加入 `.gitignore`，克隆后按「快速开始」步骤即可直接启动看板。
- 数据版权归属原平台，本项目仅用于学习、研究与展示，不作商业用途。

详细数据字典、缺失值说明、质量报告见 `data/README.md`（含英文版 `data/README_EN.md`）。

---

## 致谢

- 数据来源于**懂车帝**（用户口碑、车型配置参数）与**太平洋汽车**（月度销量），感谢其公开数据支撑本项目的研究与展示。
- 数据版权归原平台所有，本项目仅用于学习、研究与展示，遵守其使用规范。

---

*License：MIT（仅对项目代码与文档；数据版权归原作者所有，请遵守原平台使用规范）。*

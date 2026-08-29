<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

<h1 align="center">China Automotive Market Analysis: Sales Forecasting, Product Specifications, and User Needs</h1>

<p align="center">A research project based on public monthly sales, vehicle specifications, and 24,175 owner reviews</p>

<p align="center">
  <a href="https://yemyu.github.io/china-auto-market-analysis/"><b>Live research dashboard</b></a>
  · <a href="./notebook/China_Auto_Market_Analysis_EN.ipynb">Analysis notebook</a>
  · <a href="./data/README_EN.md">Data documentation</a>
</p>

---

## Project overview

| Module | Sample | Question | Current protocol |
|---|---:|---|---|
| Rolling one-month sales forecast | 371 series | Forecast next month with the latest published sales | Headline result; strict temporal split |
| Fixed six-month stress test | Same 371 series | Recursively forecast six months from 2026-01 | Supporting scenario; do not rank directly against the headline |
| Product-specification analysis | 736 series, 2,007 series-year records | Estimate the incremental explanatory value of year, brand, and specifications | Five-fold `GroupKFold` by series |
| User needs and risk | 24,175 reviews across 345 series | Identify ten needs, negative concentration, and reputation anomalies | Structural checks, sampling audit, adjacent 180-day windows |

Each module has its own eligibility rule: forecasting requires continuous monthly history and usable modeling fields; specification analysis requires aligned annual sales and attributes; user-needs analysis requires complete, traceable review text.

## Main results

### 1. Sales forecasting

The evaluation window is January–June 2026: 371 series and 2,226 series-month rows. The headline task refreshes a one-month-ahead forecast each month; the fixed-origin six-month result is a stress test. Global volume-weighted WMAPE is primary, with median per-series WMAPE as supporting evidence.

#### Rolling one-month headline

| Method | Global WMAPE ↓ | Median per-series WMAPE ↓ | Versus same-scenario naive |
|---|---:|---:|---:|
| Last observed value (naive) | 40.99% | 48.36% | — |
| **Rolling one-month XGBoost (headline)** | **31.34%** | **39.47%** | **−9.65 pp (about −23.5%)** |

At each test month, the rolling protocol uses the latest published previous-month sales. Parameters remain locked within the six-month evaluation window, so later test months do not influence model selection.

#### Fixed six-month stress test

| Method | Global WMAPE ↓ | Median per-series WMAPE ↓ |
|---|---:|---:|
| Fixed-origin trailing six-month mean (naive) | 69.31% | 89.60% |
| Fixed six-month combined method (reviews + cold-start supplement) | **39.07%** | **49.10%** |

The fixed stress test reduces absolute error by 43.6% against its same-scenario naive comparator, but it withholds post-origin realised sales and is not the same task as the rolling headline. In this fixed stress test, review enhancement improves the point estimate by 0.884 percentage points; the 5,000-replicate series-cluster bootstrap 95% interval is −0.284 to 2.108 points and crosses zero, so it remains supporting evidence. The cold-start statistical strategy handles nine history-poor series and does not replace the headline model.

The sales model is XGBoost. ARIMA, Prophet, LSTM, and other early candidates remain in the experiment scripts to document model selection; they are not mixed into the current headline table.

### 2. Product specifications and annual sales variation

| Feature combination | Grouped-CV R² | Annual cross-sectional WMAPE (supporting) |
|---|---:|---:|
| Year | 0.089 | 87.77% |
| Year + brand | 0.158 | 83.77% |
| Year + brand + specifications | **0.301** | **75.68%** |

Adding specifications increases R² by about 0.143 over year plus brand, showing that product attributes explain part of the between-series annual variation. This is out-of-sample association, not a causal effect; annual cross-sectional WMAPE is not directly comparable with monthly forecast WMAPE.

### 3. User needs and risk

Reviews are mapped to ten dimensions—space, power, control, comfort, energy/fuel, configuration, intelligence, value, appearance, and interior—with mention, polarity, and time-window fields kept separate. The latest complete monitoring month is July 2026; 123 series meet the monitoring threshold and one active rule-based alert is queued for manual review.

## Research design and information availability

| Segment | Period | Use |
|---|---|---|
| Train | Through 2025-06 | Model fitting |
| Validation | 2025-07—12 | Parameter and protocol selection |
| Test | 2026-01—06 | Final evaluation |

The rolling headline uses information available by each forecast month; the fixed stress test recursively generates post-origin sales lags. Review features use only reviews published before their relevant cutoff. The strict corpus keeps complete text, publication time, and auditable source fields; list-only summaries without detail text are excluded from temporal modeling. Missing review coverage remains missing with an availability indicator rather than being coded as neutral.

Global WMAPE is defined as:

```text
Σ |actual sales − forecast sales| / Σ actual sales
```

## Dashboard and reproduction

The dashboard is a static HTML/CSS/JavaScript/ECharts site backed by pre-baked JSON in `app/static/data/`. Six Chinese and English pages are included. The overview presents the rolling headline; the sales page also shows the fixed six-month stress test and fixed-scenario review ablation.

<details open>
  <summary><b>Dashboard captures</b></summary>

<br>

<table>
  <tr><td width="50%"><a href="./assets/dashboard/en/01-overview.png"><img src="./assets/dashboard/en/01-overview.png" alt="Project overview"></a></td><td width="50%"><a href="./assets/dashboard/en/02-sales-forecast.png"><img src="./assets/dashboard/en/02-sales-forecast.png" alt="Sales forecast"></a></td></tr>
  <tr><td align="center">Project overview</td><td align="center">Sales forecast</td></tr>
  <tr><td><a href="./assets/dashboard/en/03-user-needs.png"><img src="./assets/dashboard/en/03-user-needs.png" alt="User needs"></a></td><td><a href="./assets/dashboard/en/04-product-config.png"><img src="./assets/dashboard/en/04-product-config.png" alt="Product specifications"></a></td></tr>
  <tr><td align="center">User needs</td><td align="center">Product specifications</td></tr>
  <tr><td><a href="./assets/dashboard/en/05-risk-monitor.png"><img src="./assets/dashboard/en/05-risk-monitor.png" alt="Risk monitor"></a></td><td><a href="./assets/dashboard/en/06-brand-series.png"><img src="./assets/dashboard/en/06-brand-series.png" alt="Brand and series"></a></td></tr>
  <tr><td align="center">Risk monitor</td><td align="center">Brand and series (BYD example)</td></tr>
</table>

</details>

Preview the dashboard (data is already pre-baked):

```bash
python -m http.server 8000 --directory app
```

Regenerate dashboard payloads:

```bash
python app/build_dashboard_data.py
```

Rolling forecast artifacts:

- `data/processed/forecast/rolling_origin_summary.json`
- `data/processed/forecast/rolling_origin_validation.csv`
- `data/processed/forecast/rolling_origin_test_predictions.csv`

## Repository layout

```text
china-auto-market-analysis/
├── app/                    Static research dashboard and pre-baked JSON
├── assets/                 Analysis figures and dashboard captures
├── data/                   Raw, processed, and audit data
├── notebook/               Chinese and English analysis notebooks
├── scripts/                Reproducible scripts
├── environment.yml
├── requirements.txt
├── README.md
└── README_EN.md
```

## Data and license

Monthly sales, vehicle specifications, and owner reviews come from public automotive platforms. Copyright remains with the respective sources; repository data is for learning, research, and project presentation only, not commercial use. Code and project documentation are released under the MIT License. See [data/README_EN.md](./data/README_EN.md) for schemas, cohort rules, temporal availability, and archived resources.

<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

<h1 align="center">🌍 China Automotive Market Analysis: Sales Forecasting, Product Specifications, and User Needs</h1>

<p align="center">A research project based on public monthly sales, vehicle specifications, and 24,175 owner reviews</p>

<p align="center">
  <a href="https://yemyu.github.io/china-auto-market-analysis/"><b>Live research dashboard</b></a>
  · <a href="./notebook/China_Auto_Market_Analysis_EN.ipynb">Analysis notebook</a>
  · <a href="./data/README_EN.md">Data documentation</a>
</p>

---

> **Project summary**　This project uses public monthly sales, vehicle specifications, and owner reviews for next-month sales forecasting, annual product-difference analysis, and user-needs monitoring.

It also implements a reproducible path from source batches and a layered MySQL warehouse through quality gates, business marts, monthly Airflow orchestration, and static dashboard publication.

## Research questions

1. 📈 Can next-month series sales be forecast reliably as information is refreshed each month?
2. 🧩 Which product specifications explain annual sales differences between series?
3. 💬 Which user needs are most visible in reviews, and which signals merit follow-up?

## Analysis scope

| Module | Sample | Question | Current protocol |
|---|---:|---|---|
| 📈 Rolling one-month sales forecast | 371 series | Forecast next month with the latest published sales | Headline result; complete monthly panel and strict temporal split |
| 🧪 Fixed six-month stress test | Same 371 series | Recursively forecast six months from 2026-01 | Supporting scenario; evaluated separately from the headline |
| 🧩 Product-specification analysis | 646 series, 1,510 complete series-year records | Estimate the incremental explanatory value of year, brand, and specifications | Complete 2022–2025 calendar years; five-fold `GroupKFold` by series |
| 💬 User needs and risk | 24,175 reviews across 345 series | Identify ten needs, negative concentration, and reputation anomalies | Structural checks, sampling audit, adjacent 180-day windows |

Each module has its own eligibility rule: forecasting uses a fixed 371-series natural-month panel and never falls forward to a specification record later than the target year; specification analysis requires aligned complete-year sales and attributes; user-needs analysis requires complete, traceable review text.

## Key metrics

| Key finding | Current result |
|---|---:|
| 📈 Rolling one-month headline | **29.75% WMAPE**, 11.24 points below the last-value baseline |
| 🧪 Fixed six-month stress test | **38.09% WMAPE**, 43.5% lower absolute error than the trailing-twelve-month mean |
| 🧩 Annual specification model | Grouped-CV R² on log sales: **0.239**; specification increment: **+0.169** |
| 💬 User-needs monitoring | 24,175 reviews, 10 dimensions, 123 qualifying series |

## Main results

### 1. 📈 Sales forecasting

The evaluation window is January–June 2026: 371 series and 2,226 series-month rows. The headline task refreshes a one-month-ahead forecast each month; the fixed-origin six-month result is a stress test. Global volume-weighted WMAPE is primary, with median per-series WMAPE as supporting evidence.

#### Rolling one-month headline

| Method | Global WMAPE ↓ | MAE (vehicles) ↓ | RMSE (vehicles) ↓ | Median per-series WMAPE ↓ |
|---|---:|---:|---:|---:|
| Last observed value | 40.99% | 975.19 | 2,668.17 | 48.36% |
| Trailing 3-month mean | 43.61% | 1,037.52 | 2,614.45 | 53.66% |
| Trailing 6-month mean | 51.21% | 1,218.19 | 2,892.08 | 64.45% |
| Same month last year | 73.88% | 1,757.68 | 3,590.51 | 100.00% |
| **Seasonal XGBoost** | **29.75%** | **707.68** | **1,700.38** | **36.83%** |

MAE and RMSE are measured in vehicles per series-month. The model reduces absolute error by 27.4% against the last-value baseline. Median absolute error is 139 vehicles; the 90th percentile is about 1,960. A few large errors raise the mean, so 707 vehicles does not describe every series equally well.

**Model selection.** Each of four historical origins—January and July in 2024 and 2025—covers six successive one-month forecasts. BASE uses 1/2/3-month lags, 3/6-month means, calendar variables and specifications. SEASONAL_D5 adds a 12-month lag and mean, and changes tree depth, tree count and other parameters. This compares two model configurations; it does not isolate the effect of the seasonal features alone.

Pooled historical WMAPE falls from 24.30% to 23.35%, a 0.957-point gain. The worst origin regresses by 0.061 points, passing the gates of at least 0.5 points pooled improvement and no origin regressing by more than 1 point. The selected model is then refitted on data through December 2025. Weights and parameters stay fixed throughout January–June 2026 while observed sales history advances each month. The [notebook](./notebook/China_Auto_Market_Analysis_EN.ipynb) and [evaluation summary](./data/processed/forecast/rolling_origin_summary.json) report the origin-level results and parameters. Specification preprocessing has a timing limitation described below.

**Where the model falls short.** It beats the last-value baseline in five of six test months. May WMAPE is 21.77%, slightly worse than the baseline's 21.02%. Groups fixed using the six pre-test months show 66.60% WMAPE in the lowest-volume quartile and 28.02% in the highest. The notebook reports monthly, volume-group and zero-sales diagnostics.

| Other views of error | Last-value baseline | Seasonal XGBoost |
|---|---:|---:|
| Positive-sales MAPE (1,964 rows) ↓ | 255.51% | 179.75% |
| sMAPE (all 2,226 rows) ↓ | **52.44%** | 61.39% |
| WMAPE after summing the 371 series within each month ↓ | 28.32% | **6.46%** |
| Six-month net bias (prediction minus actual) | +9.54% | −0.67% |

sMAPE assigns 200% to a zero actual with a positive prediction, even for a fraction of a vehicle. On the 262 zero-sales rows, model MAE is 23.06 vehicles versus 34.87 for the baseline, but the model more often predicts small positive values and has worse sMAPE. The metrics measure different losses; the model does not win on every measure. Net bias allows errors to cancel across months. The −0.67% figure is neither monthly forecast error nor a measure of accuracy for the entire Chinese market.

#### Fixed six-month stress test

| Method | Global WMAPE ↓ | Median per-series WMAPE ↓ |
|---|---:|---:|
| Fixed-origin trailing six-month mean (naive) | 69.31% | 89.60% |
| Fixed-origin trailing twelve-month mean (naive) | 67.43% | 89.40% |
| Fixed six-month platform-rating model (XGBoost) | **38.09%** | **48.18%** |

The platform-rating model has MAE of 906.21 vehicles. It reduces absolute error by 45.0% against the original six-month-mean comparator, or 43.5% against the twelve-month mean, which has the lowest test WMAPE among the saved naive results. The latter is a descriptive comparison, not a model selected using the test set. See the [full fixed-origin comparison](./data/processed/forecast/forecast_benchmark_comparison.csv).

**Review-feature experiments.** The sales baseline, platform ratings, local lexicon, text and combined-review variants all use XGBoost. This evaluation retains the previously selected platform-rating model and 100 trees per variant; current validation rankings and test scores do not trigger reselection. Platform ratings lower test WMAPE from 39.07% to 38.09%. A 5,000-replicate bootstrap resampling whole series gives a 0.980-point improvement and a 95% interval of −0.0047 to 2.2331. The 97.48% bootstrap win share is not a probability of future effectiveness. The interval includes zero, so evidence for a stable gain remains insufficient. The notebook includes the full comparison and interval plot.

**Limited sales history.** All thirteen series with no positive pre-origin sales remain in the evaluation and use the same platform-rating model, without a launch-curve override. Their fixed-origin WMAPE is about 99.40%; the current inputs do not reliably predict sales ramp-up.

The sales model is a seasonal XGBoost with 12-month lag and trailing-12-month mean features. ARIMA, Prophet, LSTM, and other approaches were evaluated as early candidates; the streamlined public repository retains the selection conclusion and the reproducible adopted pipeline rather than redundant candidate scripts, and their early scores are not mixed into the current headline table.

### 2. 🧩 Product specifications and annual sales variation

| Feature combination | Log-sales R²: fold mean ± fold SD | Annual-sales WMAPE (pooled out-of-fold predictions) |
|---|---:|---:|
| Year | 0.013 ± 0.012 | 87.50% |
| Year + brand | 0.070 ± 0.058 | 83.82% |
| Year + brand + specifications | **0.239 ± 0.073** | **73.85%** |
| Specifications only | 0.197 ± 0.086 | 76.05% |

The model fits `log1p(annual sales)` and R² is calculated on that scale. WMAPE uses pooled out-of-fold predictions converted back to vehicle counts. The 0.169 R² increment is not a percentage of sales attributable to specifications; 0.239 is not an explained-variance score on raw sales. The ± values are standard deviations across five folds, not 95% confidence intervals.

The analysis uses complete 2022–2025 calendar years. GroupKFold holds out entire series, and imputation and encoding are fitted within each training fold. This measures generalisation to unseen series, not to future years. A year-specific median baseline fitted on the same training folds has 87.21% WMAPE versus 73.85% for the full model. Considerable variation remains unexplained.

Feature importance uses XGBoost gain to describe contributions to tree splits. The analysis supports comparisons of similar products; annual series-level specifications and list prices also do not replace trim-level sales or transaction prices.

### 3. 💬 User needs and risk

Reviews are mapped to ten dimensions—space, power, control, comfort, energy/fuel, configuration, intelligence, value, appearance, and interior—with mention, polarity, and time-window fields kept separate. Monitoring first produces text-rule candidates, then requires at least 70% reproduction across 3,000 bootstrap samples and at least an 80% probability of a same-direction decline in the original platform rating. Twelve of 60 historical text candidates pass this dual-signal gate. In the latest complete month, July 2026, 123 series meet the monitoring threshold, with no active dual-signal alert and one watchlist candidate.

Complaint keywords are ranked by positive-versus-negative distinctiveness rather than raw frequency. The current high-distinctiveness issues include thin paint, plastic-heavy interiors, cramped rear seating, weak acceleration, tyre noise, high fuel consumption, specification deletions, infotainment problems, and high prices—rather than generic frequent words such as “appearance” or “like”. The dual-signal gate improves contemporaneous evidence stability; it is not validation of future faults or persistent deterioration.

Mention share uses all eligible reviews as its denominator; negative share uses scored mentions of the relevant aspect. The notebook reports both counts. Reviews are self-selected, so negative share is not a population dissatisfaction rate. Historical zero labels have mixed meanings. A common mention detector improves consistency, but existing spot checks do not constitute an independent, representative gold standard. Label F1 and alert accuracy have not been established.

The 123 eligible series represent 33.2% of the 371-series target cohort. Other series do not meet the minimum of five reviews in each adjacent 180-day window. No current alert therefore applies only to the monitored subset. Review the source comments, sample sizes and rating changes before deciding whether to investigate a product issue.

## Research design and information availability

| Segment | Period | Use |
|---|---|---|
| Train | Through 2025-06 | Model fitting |
| Validation | 2025-07—12 | Parameter and protocol selection |
| Test | 2026-01—06 | Final evaluation |

These are the development splits. After selecting each model configuration, both the rolling and fixed-origin models are refitted on Train+Validation. Test sales enter history only for subsequent rolling predictions; they do not refit model weights during the test window. The notebook rescores saved predictions; scripts perform model training.

The January–June 2026 evaluation window was examined during development for checks and repairs, so it is treated as a fixed retrospective evaluation rather than presented as a completely new blind holdout. Review bootstrap intervals are conditional on the fitted models and observed months; they do not include uncertainty from refitting, model selection or future time windows.

Rolling evaluation updates observed sales history each month; the fixed stress test recursively generates post-origin sales lags. Review features use only reviews published before their relevant cutoff. The strict corpus keeps complete text, publication time, and auditable source fields; list-only summaries without detail text are excluded from temporal modeling. Missing review coverage remains missing with an availability indicator rather than being coded as neutral.

**Data availability.** At each forecast origin, specification records are limited to years available before the origin and joined to rows without looking forward in year. Imputation medians and category vocabularies are fitted only on the model's eligible training rows; specifications stay fixed over the forecast window. Unknown categories use −1, entirely missing numeric columns use 0, and missing specifications do not remove sales observations. Within-year specification release dates and historical sales revisions are unavailable, so annual proxies do not establish complete point-in-time reconstruction. Rolling evaluation assumes previous-month sales have been released. Annual specification analysis uses a separate fold-local pipeline.

Global WMAPE is defined as:

```text
Σ |actual sales − forecast sales| / Σ actual sales
```

WMAPE is displayed as a percentage. MAE is mean absolute row error; on a fixed test set, `WMAPE = MAE / mean actual sales × 100%`, so these are not independent performance evidence. MAPE uses positive actuals only. sMAPE is `mean(200 × |actual−prediction| / (|actual|+|prediction|))`, with a zero contribution when both values are zero, preserving the same row count for all methods. Zero-sales errors remain in MAE, RMSE and global WMAPE.

## Data platform and quality controls

```text
Public sources / local snapshots
        ↓
auto_raw (source-faithful rows and batches)
        ↓
auto_staging (standardization, mapping, repairs, and availability)
        ↓
auto_mart (forecast, product, and user-needs marts)
        ↓
Models and analysis artifacts → validation → pre-baked dashboard JSON

auto_ops records runs, task attempts, quality results, and dataset versions;
Airflow manages monthly dependencies, retries, backfills, and failure blocking.
```

- The four logical databases contain 23 tables with explicit grain and key contracts.
- Raw ingestion registers source-file SHA-256 values and skips exact reruns idempotently.
- Critical staging or mart failures prevent public artifacts from being replaced.
- The monthly DAG has ten tasks and succeeds only after data, model, and publication contracts pass.
- Automated tests use fully synthetic fixtures in an ephemeral MySQL service; they do not read the full review corpus or access external websites.

The static dashboard remains directly viewable on GitHub Pages and does not require access to local MySQL. MySQL and Airflow implement the full local engineering path, while pre-baked JSON is the public delivery interface.

The engineering implementation is available in `src/`, `sql/`, `dags/`, and `tests/`; the public dashboard uses pre-baked JSON and does not require access to the local database.

## Quick start

The pre-baked dashboard runs without a backend service:

```bash
git clone https://github.com/Yemyu/china-auto-market-analysis.git
cd china-auto-market-analysis
python3 -m http.server 8000 --directory app
```

Open `http://localhost:8000` to browse the Chinese and English pages. To rebuild analysis outputs, use the project environment and follow the [data guide](./data/README_EN.md).

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
python3 -m http.server 8000 --directory app
```

Regenerate dashboard payloads:

```bash
python3 app/build_dashboard_data.py
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
├── dags/                   Monthly Airflow orchestration
├── data/                   Raw, processed, and audit data
├── notebook/               Chinese and English analysis notebooks
├── scripts/                Collection, analysis, and engineering commands
├── sql/                    MySQL DDL, staging, marts, and quality rules
├── src/china_auto_market/  Testable production Python modules
├── tests/                  Unit tests, synthetic fixtures, and MySQL integration
├── environment.yml
├── pyproject.toml
├── requirements.txt
├── README.md
└── README_EN.md
```

## Data and license

Monthly sales, vehicle specifications, and owner reviews come from public automotive platforms. Copyright remains with the respective sources; repository data is for learning, research, and project presentation only, not commercial use. Code and project documentation are released under the MIT License. See [data/README_EN.md](./data/README_EN.md) for schemas, cohort rules, temporal availability, and archived resources.

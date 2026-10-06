<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# 📦 Data guide

This guide describes the data files, sample definitions, analysis results, and local setup. Platform data is provided for learning, research, and project presentation; copyright remains with the respective sources.

## Dataset summary

| Module | Entry point | Scale | Use |
|---|---|---:|---|
| Monthly sales | `processed/sales_filtered_24m.csv` | 54,918 rows / 1,017 series | Tracked modeling snapshot; rolling one-month forecast and fixed six-month stress test |
| Specifications | `raw/feature.csv` | 2,084 rows / 766 series | Tracked modeling table; product-attribute analysis of annual sales variation |
| Owner reviews | `reviews/processed/` | 24,175 reviews / 345 series | User needs, risk monitoring, and supporting forecast experiments |

Each module has its own sample and evaluation protocol. The public repository includes CSV files, labels, analysis summaries, and dashboard JSON; the full review text is stored locally. Viewing the dashboard, reading the notebooks, and rebuilding dashboard JSON do not require MySQL.

## Directory structure

| Path | Contents |
|---|---|
| `raw/` | Annual-specification CSV and Excel source; other local collection CSVs are not distributed through Git |
| `processed/splits/` | Time splits and modeling features for the 371-series cohort |
| `processed/forecast/` | Forecast, baseline, ablation, and robustness outputs |
| `processed/product/` | Annual product-specification explanatory-analysis outputs |
| `processed/user_feedback/` | User-needs and risk-monitoring outputs |
| `processed/data_quality/` | Machine-readable structural, mapping, and source audits |
| `reviews/raw/` | Review collection manifests and source-layer files |
| `reviews/processed/` | De-identified labels, temporal features, and corpus summaries; full text is not distributed through Git |
| `resources/` | Reusable historical review-resource archive |

## Modeling inputs

### Monthly sales: `processed/sales_filtered_24m.csv`

- Grain: series × calendar month; period: 2022-01—2026-06.
- Main fields: `series_id`, `series_name`, `brand`, `category`, `year`, `month`, `monthly_sales`.
- The modeling snapshot contains no negative-sales records. Rankings, cumulative totals, and displayed website prices are source-derived metadata and are not forecast features.

### Product specifications: `raw/feature.csv`

- Grain: series × model year, without separate records for individual trim levels.
- Key: `series_name, year`; 84 fields; annual sales are alignable for 760 / 766 series.
- Annual specification analysis uses only years with all 12 calendar months in the sales source; currently 2022–2025, covering 646 series and 1,510 series-year records.
- Missingness is partly structural: battery fields are normally absent for combustion models and engine fields for battery-electric models. It should not be treated as a blanket collection error.

The public repository retains the audited monthly-sales modeling snapshot, the specification CSV, and its Excel source workbook. `raw/monthly_sales.csv` is a local collection-stage working file rather than a required input after a public clone; core scripts read the tracked modeling snapshot directly and fall back to the Excel workbook if the specification CSV is absent.

### Review corpus: `reviews/processed/`

Reviews enter temporal modeling only when the series is identifiable, publication time is parseable, the full text meets quality rules, and the review predates the relevant cutoff. The strict corpus contains 24,175 reviews across 345 series; missing coverage remains missing rather than being coded as neutral.

Labels separate mentions from sentiment for ten dimensions: appearance, interior, space, power, handling, comfort, energy use, equipment, smart features, and value.

## Monthly forecast sample

`processed/splits/` contains absolute-time splits for a fixed 371-series cohort:

| File | Period | Use |
|---|---|---|
| `train.csv` | Through 2025-06 | Model fitting; earlier months warm up lag features |
| `val.csv` | 2025-07—12 | Parameter and protocol selection |
| `test.csv` | 2026-01—06 | Final evaluation |
| `split_index.csv` | Complete panel | Split assignment for each series-month |
| `manifest.json` | — | Row counts, features, cutoffs, and leakage constraints |

The shared panel preserves calendar-month spacing and stores only sales, calendar variables, and lags. Models read raw specifications and fit preprocessing within each training window; globally encoded specifications are not stored in the shared splits or forecast mart. The base model uses 1/2/3-month lags and 3/6-month means; the primary model also uses a 12-month lag and mean.

The primary task forecasts one month ahead using published previous-month sales. The fixed-origin task recursively forecasts six months from 2026-01 as an information-constrained stress test. The two tasks are evaluated separately.

After selection, final models are refitted on train+val through December 2025; weights remain fixed during evaluation. Each origin limits the available specification years, and rows are matched to the latest record from the same or an earlier year. Imputation medians and category vocabularies are fitted only on eligible training rows. Specifications stay fixed over the forecast window. Unknown categories use −1, entirely missing numeric columns use 0, and sales rows with missing specifications are retained. Within-year specification release dates and historical sales releases and revisions are unavailable; alignment uses annual proxies.

## Key outputs

### Sales forecasting: `processed/forecast/`

| File | Use |
|---|---|
| `rolling_origin_summary.json` | Historical-origin validation, gate, and locked-test summary |
| `rolling_origin_test_predictions.csv` | Test-period forecasts and same-scenario naive baselines |
| `forecast_benchmark_comparison.csv` | Fixed-stress-test and naive-baseline comparison |
| `review_feature_ablation_summary.csv` | Fixed-scenario review-feature ablation |
| `forecast_robustness_summary.json` | Cluster bootstrap, segment errors, and robustness summary |

Saved rolling results are 29.75% global WMAPE, 707.68 vehicles MAE, and 1,700.38 vehicles RMSE; the fixed six-month platform-rating model has 38.09% WMAPE. MAE and RMSE are measured in vehicles per series-month. Both the notebook and dashboard score row-level predictions through `src/china_auto_market/forecasting/reporting.py`: sMAPE retains all rows, with zero actual and predicted sales contributing 0; MAPE uses positive actuals only. See [README_EN.md](../README_EN.md) for definitions and comparisons.

### Product specifications: `processed/product/`

- `config_attribution_ablation.csv`: stepwise year, brand, and specification ablation;
- `config_importance_annual.csv`: annual specification feature importance.
- `config_attribution_summary.json`: complete-year range, sample size, and main metrics.

This module reports mean five-fold R² on `log1p(annual sales)` and the standard deviation across folds. Full-model R² is 0.239; adding specifications increases it by 0.169. WMAPE pools out-of-fold predictions converted to vehicle counts. The task measures annual associations between series and uses a different sample and protocol from monthly forecasting.

### User needs: `processed/user_feedback/`

`user_need_aspect_summary.csv`, `user_need_keywords.csv`, `user_need_topics.csv`, `sentiment_monitoring_windows.csv`, and `sentiment_alerts.csv` cover aspect summaries, discriminative complaint terms, topics, time windows, and risk monitoring. `sentiment_alerts.csv` retains both text candidates and their dual-signal validation status; only records passing bootstrap stability and same-direction platform-rating checks count as active alerts, and all records still require manual review.

## Historical resources

`resources/historical_reviews/` contains reusable historical review resources:

- `review_absa_reference.csv.gz`: local full-text archive with 39,496 deduplicated reviews, 28,724 carrying historical ten-dimension labels;
- `manifest.json` and `README.md`: counts, time range, checksum summary, and usage limits.

The full-text archive is Git-ignored and remains local; the public repository contains de-identified labels and aggregates.

## Local setup

Run the following commands from the repository root.

### View the dashboard

Python can serve the included HTML and JSON directly:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory app
```

Open `http://localhost:8000`. Viewing the dashboard does not require analysis dependencies or a database. On Windows, use `python` in place of `python3` if that is your Python command.

### Set up the analysis environment

Python 3.11 or later is required. Create a virtual environment for the dependencies in [requirements.txt](../requirements.txt):

```bash
python3 -m venv .venv
```

Activate it on macOS / Linux:

```bash
source .venv/bin/activate
```

Activate it in Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On Windows, `py -3 -m venv .venv` can also create the environment; the selected Python must meet the version requirement. Once activated, install the dependencies and open Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Choose the [Chinese report](../notebook/China_Auto_Market_Analysis.ipynb) or [English report](../notebook/China_Auto_Market_Analysis_EN.ipynb) in `notebook/`. Both notebooks read saved predictions and analysis results without training models.

### Rebuild dashboard JSON from saved results

With the analysis environment activated, run:

```bash
python app/build_dashboard_data.py --output-dir artifacts/dashboard-preview
```

This writes eight JSON files to `artifacts/dashboard-preview/`. The builder reads published analysis results and checks that predictions, statistical summaries, and versions agree; it does not require full review text. The output directory is for inspecting generated files and does not replace the live dashboard data in `app/static/data/`.

### Analysis scripts and required inputs

The table lists each script's inputs and output locations. Run a script with `python <entry point>`. Unless an alternative output directory is supported as noted below, rerunning a script updates the saved results at the listed locations. The full-text corpus, `data/reviews/processed/target_371_review_corpus.csv`, is not included in the public repository.

| Analysis | Entry point | Inputs and outputs |
|---|---|---|
| Time splits | `scripts/06_make_splits.py` | Reads the monthly-sales snapshot and frozen series list in `target_371_review_coverage.csv`; writes to `data/processed/splits/` by default; supports `--output-dir` |
| Temporal review features | `scripts/32_build_temporal_review_features.py` | Reads splits, `review_aspect_labels.csv`, and review-availability records; updates fixed and rolling features and audit summaries in `data/reviews/processed/` |
| Fixed six-month forecast | `scripts/33_evaluate_review_features.py --locked-capacity` | Reads splits, specifications, saved review features, and model-selection summaries; writes to `data/processed/forecast/` and `assets/analysis/` by default; `--output-dir` places both results and figures in another directory |
| Rolling one-month forecast | `scripts/48_evaluate_rolling_origin.py --test` | Reads splits, specifications, and saved review features; writes to `data/processed/forecast/` by default; supports `--output-dir` |
| Annual specification analysis | `scripts/29_config_attribution.py` | Reads specifications, sales, and the annual-sales correction register; updates `data/processed/product/`, `assets/analysis/`, and the annual-sales repair audit |
| User needs and monitoring | `scripts/35_build_user_needs_and_alerts.py` | Reads full review text, labels, and series metadata; updates `data/processed/user_feedback/` and `assets/analysis/`; requires the undistributed full-text corpus |

For the fixed forecast, `--locked-capacity` retains the saved tree counts and platform-rating selection. Rolling evaluation uses the two parameter configurations defined in the code and evaluates them at historical origins. The original 2026 split remains an evaluation window rather than a new tuning dataset.

Bootstrap analysis, model replay for SHAP, error diagnostics, and naive-baseline comparisons can write to a separate directory:

```bash
python scripts/34_analyze_forecast_robustness.py --output-dir artifacts/report-statistics
python scripts/39_evaluate_naive_forecast_baselines.py --output-dir artifacts/report-statistics
```

Both scripts read fixed-origin predictions and run summaries from `data/processed/forecast/` and splits from `data/processed/splits/` by default. `--forecast-dir` selects another prediction directory; `--split-dir` selects splits for baselines or diagnostics. SHAP replay in script 34 still reads the repository's default modeling panel, specifications, and review features, then checks agreement with saved predictions. These two options alone therefore do not redirect all model inputs. The dashboard builder reads the published analysis directories and does not automatically use results from `artifacts/report-statistics/`.

<details>
<summary>Optional: MySQL warehouse</summary>

The full local data pipeline uses four logical MySQL databases, with the same sample definitions, time splits, and scoring rules as the CSV path:

| Layer | Role | Main outputs |
|---|---|---|
| `auto_raw` | Store source batches and file hashes | Sales, specifications, reviews, labels, and correction registers |
| `auto_staging` | Standardize, resolve series, deduplicate, and align time | Standard sales, specifications, reviews, and ten-aspect features |
| `auto_mart` | Organize facts, dimensions, and subject tables | Forecast, product, and user-needs marts |
| `auto_ops` | Record runs, tasks, quality checks, and versions | Batch status, lineage, and publication manifests |

This path requires complete source snapshots, an isolated MySQL instance, the MySQL command-line client, and an encrypted login path configured with `mysql_config_editor`. Scripts select the connection with `--login-path`; the default name, `local-auto`, must be configured on the machine running them. Credentials are not included in the repository. The client is located through `PATH` or the `MYSQL_CLIENT` environment variable.

The entry points, in order, are `scripts/initialize_mysql.py`, `scripts/ingest_raw.py`, `scripts/validate_raw.py`, `scripts/rebuild_staging.py`, `scripts/rebuild_core_marts.py`, and `scripts/validate_core_marts.py`. Initialization is read-only by default; `--apply` creates the tables, and existing project databases require an explicit `--allow-existing`. Ingestion and rebuilding the staging and mart layers write to the database. Local sources such as full review text required for this path are not distributed with the public repository.

Synthetic fixtures in `tests/fixtures/ci/` support isolated MySQL integration checks for table creation, idempotent ingestion, and staging quality rules.

</details>

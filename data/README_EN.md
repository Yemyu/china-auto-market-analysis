<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# 📦 Data guide

This guide lists the files, sample definitions, and generated outputs used by each module. Source-platform data is provided for learning, research, and project demonstration; copyright remains with the respective sources.

## Dataset summary

| Module | Entry point | Scale | Use |
|---|---|---:|---|
| Monthly sales | `processed/sales_filtered_24m.csv` | 54,918 rows / 1,017 series | Tracked modeling snapshot; rolling one-month forecast and fixed six-month stress test |
| Specifications | `raw/feature.csv` | 2,084 rows / 766 series | Tracked modeling table; product-attribute analysis of annual sales variation |
| Owner reviews | `reviews/processed/` | 24,175 reviews / 345 series | User needs, risk monitoring, and supporting forecast experiments |

Each module has its own sample filter and evaluation protocol; metrics from different modules are not directly comparable.

## Warehouse and lineage

The complete local engineering path uses four logical MySQL databases:

| Layer | Role | Main outputs |
|---|---|---|
| `auto_raw` | Preserve source-faithful batches and file hashes | Raw sales, specifications, reviews, labels, and repair evidence |
| `auto_staging` | Standardize types, resolve series, deduplicate, and enforce availability | Standard sales, specifications, reviews, and ten-aspect features |
| `auto_mart` | Provide stable facts, dimensions, and subject interfaces | Forecast, product, and user-needs marts |
| `auto_ops` | Record runs, tasks, quality results, and versions across layers | Batch status, failures, lineage, and publication manifests |

The CSV and JSON files in the public repository are validated portable snapshots and do not require a local database connection. The full local path and portable snapshots share the same time splits, sample definitions, and metric contracts.

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
| `reviews/processed/` | De-identified corpus, labels, and temporal features |
| `resources/` | Reusable historical review-resource archive |

## Modeling inputs

### Monthly sales: `processed/sales_filtered_24m.csv`

- Grain: series × calendar month; period: 2022-01—2026-06.
- Main fields: `series_id`, `series_name`, `brand`, `category`, `year`, `month`, `monthly_sales`.
- The modeling snapshot contains no negative-sales records. Rankings, cumulative totals, and displayed website prices are source-derived metadata and are not forecast features.

### Product specifications: `raw/feature.csv`

- Grain: series × model year, not a trim-level vehicle list.
- Key: `series_name, year`; 84 fields; annual sales are alignable for 760 / 766 series.
- Annual specification analysis uses only years with all 12 calendar months in the sales source; currently 2022–2025, covering 646 series and 1,510 series-year records.
- Missingness is partly structural: battery fields are normally absent for combustion models and engine fields for battery-electric models. It should not be treated as a blanket collection error.

The public repository retains the audited monthly-sales modeling snapshot, the specification CSV, and its Excel source workbook. `raw/monthly_sales.csv` is a local collection-stage working file rather than a required input after a public clone; core scripts read the tracked modeling snapshot directly and fall back to the Excel workbook if the specification CSV is absent.

### Review corpus: `reviews/processed/`

Reviews enter temporal modeling only when the series is identifiable, publication time is parseable, the full text meets quality rules, and the review predates the relevant cutoff. The strict corpus contains 24,175 reviews across 345 series; missing coverage remains missing rather than being coded as neutral.

Labels separate “dimension mentioned” from “polarity” for ten dimensions: appearance, interior, space, power, control, comfort, energy/fuel, configuration, intelligence, and value.

## Monthly forecast sample

`processed/splits/` contains absolute-time splits for a fixed 371-series cohort:

| File | Period | Use |
|---|---|---|
| `train.csv` | Through 2025-06 | Model fitting; earlier months warm up lag features |
| `val.csv` | 2025-07—12 | Parameter and protocol selection |
| `test.csv` | 2026-01—06 | Final evaluation |
| `split_index.csv` | Complete panel | Split assignment for each series-month |
| `manifest.json` | — | Row counts, features, cutoffs, and leakage constraints |

The shared panel preserves natural-month spacing and stores only sales, calendar variables and lags. Models read raw specifications and fit their transforms within each training window; globally encoded specifications are not stored in the shared splits or forecast mart. BASE uses 1/2/3-month lags and 3/6-month means; the headline model also uses a 12-month lag and mean.

The headline protocol refreshes a one-month-ahead forecast each month using the latest realised previous-month sales. The fixed-origin protocol recursively forecasts six months from 2026-01 as an information-constrained stress test; the two protocols are evaluated separately.

After selection, final models are refitted on train+val through December 2025; test weights stay fixed. At each forecast origin, specification records are limited to years available before the origin and joined to rows without looking forward in year. Imputation medians and category vocabularies are fitted only on the model's eligible training rows; specifications stay fixed over the forecast window. Unknown categories use −1, entirely missing numeric columns use 0, and missing specifications do not remove sales observations. Within-year specification release dates and historical sales revisions are unavailable, so annual proxies do not establish complete point-in-time reconstruction.

## Key outputs

### Sales forecasting: `processed/forecast/`

| File | Use |
|---|---|
| `rolling_origin_summary.json` | Historical-origin validation, gate, and locked-test summary |
| `rolling_origin_test_predictions.csv` | Test-period forecasts and same-scenario naive baselines |
| `forecast_benchmark_comparison.csv` | Fixed-stress-test and naive-baseline comparison |
| `review_feature_ablation_summary.csv` | Fixed-scenario review-feature ablation |
| `forecast_robustness_summary.json` | Cluster bootstrap, segment errors, and robustness summary |

Saved rolling results are 29.75% WMAPE, 707.68 vehicles MAE and 1,700.38 vehicles RMSE; the fixed six-month method has 38.09% WMAPE. `src/china_auto_market/forecasting/reporting.py` rescores saved predictions for both the notebook and dashboard. sMAPE retains all rows, with zero/zero contributing 0; MAPE uses positive actuals only. See [README_EN.md](../README_EN.md) for definitions and comparisons.

### Product specifications: `processed/product/`

- `config_attribution_ablation.csv`: stepwise year, brand, and specification ablation;
- `config_importance_annual.csv`: annual specification feature importance.
- `config_attribution_summary.json`: complete-year range, sample size, and headline metrics.

This module reports mean five-fold R² on `log1p(annual sales)` and fold standard deviation. Full-model R² is 0.239; specifications add 0.169. WMAPE pools out-of-fold predictions converted to vehicle counts. It is not directly comparable with monthly forecast WMAPE and does not estimate causal effects.

### User needs: `processed/user_feedback/`

`user_need_aspect_summary.csv`, `user_need_keywords.csv`, `user_need_topics.csv`, `sentiment_monitoring_windows.csv`, and `sentiment_alerts.csv` cover aspect summaries, discriminative complaint terms, topics, time windows, and risk monitoring. `sentiment_alerts.csv` retains both text candidates and their dual-signal validation status; only records passing bootstrap stability and same-direction platform-rating checks count as active alerts, and all records still require manual review.

## Historical resources

`resources/historical_reviews/` contains reusable historical review resources:

- `review_absa_reference.csv.gz`: local full-text archive with 39,496 deduplicated reviews, 28,724 carrying historical ten-dimension labels;
- `manifest.json` and `README.md`: counts, time range, checksum summary, and usage limits.

The full-text archive is Git-ignored and remains local; the public repository contains de-identified labels and aggregates.

## Reproduction entry point

Run commands from the project root. Viewing the dashboard requires neither model training nor a local database. JSON generation and analysis require dependencies installed in the project virtual environment, `.venv`.

### View the saved dashboard

```bash
python3 -m http.server 8000 --directory app
```

Open `http://localhost:8000`. This serves the repository's HTML and JSON without the full review corpus.

### Generate JSON from saved analysis results

```bash
.venv/bin/python app/build_dashboard_data.py --output-dir artifacts/dashboard-preview
```

This reads published analysis artifacts without retraining models or reading the local full-text corpus. Output is staged separately; it does not replace `app/static/data/`, and the existing website continues to show its published version. Inconsistent input versions or prediction/statistics metadata cause the build to fail; do not bypass these checks.

### Rerun analyses: additional inputs and version checks required

These are analysis entry points, not an unconditional sequence of release commands. Model outputs, statistics and reports must be verified together before publication. Default outputs may overwrite existing results; back them up or use separate output directories where supported.

| Stage | Entry points and requirements |
|---|---|
| Splits and temporal review features | `06_make_splits.py`, `32_build_temporal_review_features.py`; require the corresponding sales, configuration or review-feature inputs |
| Fixed and rolling forecasts | `33_evaluate_review_features.py --locked-capacity`, `48_evaluate_rolling_origin.py --test`; retain the selected specifications and time splits, without reselection on the 2026 window |
| Product analysis | `29_config_attribution.py`; requires configuration and annual-sales inputs |
| User needs and monitoring | `35_build_user_needs_and_alerts.py`; requires the local full corpus at `data/reviews/processed/target_371_review_corpus.csv`, which is not included in the public repository |

After generating and verifying predictions, stage the statistical outputs separately:

```bash
.venv/bin/python scripts/34_analyze_forecast_robustness.py --output-dir artifacts/report-statistics
.venv/bin/python scripts/39_evaluate_naive_forecast_baselines.py --output-dir artifacts/report-statistics
```

**Pause here for verification.** These commands read the formal prediction and split directories by default. For a separate experiment, provide matching inputs through `--forecast-dir` and `--split-dir`. Keep outputs in `artifacts/report-statistics` until sample keys, model versions, prediction hashes and statistical sources have been checked. The maintainer must then synchronize the complete result set and run the release contract. The dashboard builder does not automatically read staged statistics; immediately rebuilding it is not a substitute for this step. The current report does not apply launch-curve overrides.

For the full review-label pipeline, prepare the review corpus as described in the Notebook and script comments. Re-labeling missing review labels is optional.

With the complete local source snapshots and an isolated MySQL instance, the engineering path runs in this order:

```bash
.venv/bin/python scripts/initialize_mysql.py --apply
.venv/bin/python scripts/ingest_raw.py --dataset all --mode full
.venv/bin/python scripts/validate_raw.py
.venv/bin/python scripts/rebuild_staging.py
.venv/bin/python scripts/rebuild_core_marts.py
.venv/bin/python scripts/validate_core_marts.py
```

Database credentials are read only through a local encrypted login path; plaintext command-line passwords are not accepted. The isolated MySQL integration test uses the fully synthetic fixtures under `tests/fixtures/ci/` to exercise the same DDL, ingestion, idempotency, and staging quality rules. Those fixtures do not replace full-data business evaluation.

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
- Negative sales are recorded as zero. Rankings, cumulative totals, and displayed website prices are source-derived metadata and are not forecast features.

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

The panel preserves the natural-month spacing of every target series. Specifications may fall back only to the latest record not later than the target year; unrecoverable numeric values use the specification-table median and categorical values use the explicit `-1` unknown marker. Because the source lacks within-year publication timestamps, this establishes year alignment and no future-year fallback rather than month-level point-in-time availability. Base features use 1/2/3-month lags and 3/6-month trailing means; the headline model additionally uses 12-month lag and trailing-12-month mean features.

The headline protocol refreshes a one-month-ahead forecast each month using the latest realised previous-month sales. The fixed-origin protocol recursively forecasts six months from 2026-01 as an information-constrained stress test; the two protocols are evaluated separately.

## Key outputs

### Sales forecasting: `processed/forecast/`

| File | Use |
|---|---|
| `rolling_origin_summary.json` | Historical-origin validation, gate, and locked-test summary |
| `rolling_origin_test_predictions.csv` | Test-period forecasts and same-scenario naive baselines |
| `forecast_benchmark_comparison.csv` | Fixed-stress-test and naive-baseline comparison |
| `review_feature_ablation_summary.csv` | Fixed-scenario review-feature ablation |
| `forecast_robustness_summary.json` | Cluster bootstrap, segment errors, and robustness summary |
| `cold_start_launch_curve_summary.json` | Boundary-case cold-start method and validation |

The saved rolling headline is 29.72% global WMAPE; the fixed six-month combined method is 38.38%. See the project root [README_EN.md](../README_EN.md) for the interpretation and comparison scope.

### Product specifications: `processed/product/`

- `config_attribution_ablation.csv`: stepwise year, brand, and specification ablation;
- `config_importance_annual.csv`: annual specification feature importance.
- `config_attribution_summary.json`: complete-year range, sample size, and headline metrics.

This module reports grouped out-of-sample R². Annual cross-sectional WMAPE is a within-module supporting metric, not a direct comparison with monthly forecast WMAPE and not a causal effect estimate.

### User needs: `processed/user_feedback/`

`user_need_aspect_summary.csv`, `user_need_keywords.csv`, `user_need_topics.csv`, `sentiment_monitoring_windows.csv`, and `sentiment_alerts.csv` cover aspect summaries, discriminative complaint terms, topics, time windows, and risk monitoring. `sentiment_alerts.csv` retains both text candidates and their dual-signal validation status; only records passing bootstrap stability and same-direction platform-rating checks count as active alerts, and all records still require manual review.

## Historical resources

`resources/historical_reviews/` contains reusable historical review resources:

- `review_absa_reference.csv.gz`: local full-text archive with 39,496 deduplicated reviews, 28,724 carrying historical ten-dimension labels;
- `manifest.json` and `README.md`: counts, time range, checksum summary, and usage limits.

The full-text archive is Git-ignored and remains local; the public repository contains de-identified labels and aggregates.

## Reproduction entry point

After preparing dependencies in the project environment, the main outputs can be rebuilt in this order:

```bash
.venv/bin/python scripts/06_make_splits.py
.venv/bin/python scripts/32_build_temporal_review_features.py
.venv/bin/python scripts/33_evaluate_review_features.py
.venv/bin/python scripts/48_evaluate_rolling_origin.py --test
.venv/bin/python scripts/36_build_cold_start_curve.py
.venv/bin/python scripts/29_config_attribution.py
.venv/bin/python scripts/35_build_user_needs_and_alerts.py
.venv/bin/python app/build_dashboard_data.py
```

For the full review-label pipeline, prepare the review corpus as described in the root Notebook and script comments. Re-labeling missing review labels is optional.

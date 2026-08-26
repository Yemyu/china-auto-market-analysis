<p align="center">
  <a href="./README.md">中文</a> · <a href="./README_EN.md">English</a>
</p>

# Data documentation

This directory contains source tables, temporal splits, the review corpus, model artifacts, and data audits. Reusable historical reviews and labels are isolated under `resources/`.

## Data layers

| Path | Contents | Used by current pipeline |
|---|---|---|
| `raw/` | Monthly sales and vehicle-specification source tables | Yes |
| `processed/` | Cleaned data, temporal splits, forecasts, attribution, and monitoring outputs | Yes |
| `reviews/raw/` | Collected reviews, source detail pages, and crawl manifests | After quality filtering |
| `reviews/processed/` | Strict corpus, structured labels, temporal features, and audits | Yes |
| `resources/` | Historical review and label archive | Reference or future extension |

## Three analysis samples

| Analysis | Final sample | Main filters |
|---|---:|---|
| Monthly sales forecast | 371 series | Continuous monthly sales, aligned specifications, and a complete test period |
| Product-specification analysis | 736 series; 2,007 series-year records | Annual sales and specifications aligned during 2022–2026 |
| User needs and risk | 24,175 reviews; 345 series | Complete text, valid publication time, and traceable source |

These are not successive reductions of one table. Each analysis defines its own sample from the data it requires.

## 1. Source tables

### `raw/monthly_sales.csv`

A PCauto-based monthly sales panel.

| Item | Value |
|---|---:|
| Records | 54,918 |
| Series | 1,017 |
| Period | 2022-01—2026-06 |
| Negative sales | 0 |
| Grain | Series × calendar month |

Main modeling fields:

| Field | Description |
|---|---|
| `year`, `month`, `period` | Time fields |
| `series_id`, `series_name` | Source ID and series name |
| `brand`, `category` | Brand and vehicle category |
| `monthly_sales` | Monthly sales |
| `数据来源` | Source marker |

Rank, cumulative sales, display price, and related fields are source-platform metadata and are excluded from forecasting features.

### `raw/feature.csv`

Vehicle specifications at a series-model-year grain, not a trim-level product list.

| Item | Value |
|---|---:|
| Records | 2,084 |
| Series | 766 |
| Fields | 84 |
| Unique key | `series_name, year` |
| Annual-sales coverage | 760 / 766 series |

The table covers:

- basic attributes: brand, model year, energy type, vehicle class, and list price;
- powertrain: engine, motor, gearbox, acceleration, and consumption;
- battery and range: capacity, chemistry, range, and charging;
- body: dimensions, wheelbase, weight, seats, and body style;
- safety and cabin: airbags, displays, seats, audio, and climate control.

Engine, motor, and battery columns contain structural missingness. A BEV has no engine parameters, while an ICE model has no battery parameters; these missing values should not automatically be treated as data errors.

## 2. Forecast sample and temporal split

`processed/splits/` stores the fixed split for 371 series:

| File | Period | Purpose |
|---|---|---|
| `train.csv` | through 2025-06 | Model fitting |
| `val.csv` | 2025-07—12 | Hyperparameter and design selection |
| `test.csv` | 2026-01—06 | Final evaluation |
| `split_index.csv` | Full sample | Split membership by row |
| `manifest.json` | — | Series count, row count, time boundaries, and version metadata |

The primary experiment is a fixed-origin forecast made in January 2026 for all six test months. Recursive forecasting uses prior predictions as lagged sales and never reads actual test-period sales.

## 3. Review corpus

### Raw collection layer

`reviews/raw/` stores collected owner reviews and crawl manifests locally. Full records contain platform user identifiers and are excluded from the public repository; crawl manifests and de-identified analysis artifacts remain versioned.

Main audit files:

| File | Purpose |
|---|---|
| `dongchedi_incremental_manifest.csv` | Dongchedi incremental-crawl manifest |
| `autohome_incremental_manifest.csv` | Autohome incremental-crawl manifest |
| `autohome_incremental_review_details.csv` | Full Autohome review text |
| `processed/review_collection/autohome_id_resolutions.csv` | Series mapping and resolution |
| `processed/review_collection/sentiment_resolution_exceptions.csv` | Unresolved cases and stop rules |

### Strict modeling corpus

The local file `reviews/processed/target_371_review_corpus.csv` includes candidate reviews, quality flags, and source audit fields for the 371 target series. A modeling record must have:

1. valid series identity;
2. parseable publication time;
3. complete, quality-eligible text;
4. publication before the relevant forecast origin.

The final corpus contains 24,175 reviews across 345 series. Another 103 Autohome list-page summaries could not be resolved to full detail text; they remain in the audit table with `eligible_for_temporal_model=False`.

| Coverage definition | Series |
|---|---:|
| Any eligible review | 345 |
| Review available before the January 2026 fixed origin | 330 |
| Review available in the preceding 180 days at that origin | 272 |

## 4. Review labels and temporal features

### Review-level labels

The [structured review labels](./reviews/processed/review_aspect_labels.csv) store ten product dimensions:

`appearance`, `interior`, `space`, `power`, `control`, `comfort`, `fuel_consumption`, `configuration`, `intelligence`, and `value`.

Two items are stored separately for every dimension:

- whether the review explicitly mentions the dimension;
- polarity after mention: `-1` negative, `0` neutral, `1` positive.

Platform star ratings and text evidence are not interchangeable. If a review does not mention a dimension, it does not enter that dimension's positive or negative share as a neutral observation.

### Leakage-safe monthly features

The [fixed-origin review feature panel](./reviews/processed/review_features_by_series_month_fixed_origin.csv) has 13,866 rows across 371 series and 51 forecast months. It contains:

- cumulative reviews available before the origin;
- review count and availability in the previous 180 days;
- historical means across ten dimensions;
- 180-day positive, negative, and mention shares;
- composite aspect score and any-positive/negative shares;
- `information_cutoff_exclusive`, the information cutoff for that row.

Test features for the primary experiment are frozen before `2026-01-01`. Rolling-origin features are stored as supplementary artifacts and do not enter the headline result.

## 5. Analysis artifacts

### Sales forecasting: `processed/forecast/`

| Artifact | Contents |
|---|---|
| `review_feature_ablation_summary.csv` | Model comparison on 371 series |
| `review_feature_predictions.csv` | Series-month test predictions |
| `review_feature_series_metrics.csv` | Per-series errors |
| `forecast_robustness_bootstrap.csv` | Series-cluster bootstrap |
| `review_feature_shap_importance.csv` | Feature contributions |
| `cold_start_launch_curve_summary.json` | Cold-start method and results |

Headline global WMAPE: 40.44% for the sales baseline, 38.71% for the owner-feedback model, and 38.64% after the cold-start supplement.

### Product specifications: `processed/product/`

| Artifact | Contents |
|---|---|
| `config_attribution_ablation.csv` | Sequential year, brand, and specification ablation |
| `config_importance_annual.csv` | Annual feature importance |

The complete model reaches a five-fold grouped cross-validated R² of 0.300.

### User needs and risk: `processed/user_feedback/`

| Artifact | Contents |
|---|---|
| `user_need_aspect_summary.csv` | Mention and polarity distribution by dimension |
| `user_need_keywords.csv` | Dimension-level keywords |
| `user_need_topics.csv` | Topics within each dimension |
| `sentiment_monitoring_windows.csv` | Adjacent 180-day window statistics |
| `sentiment_alerts.csv` | Historical and current rule alerts |

An alert is a manual-review entry, not a confirmed product defect.

## 6. Curated historical resource

`resources/historical_reviews/` stores the reusable historical review archive:

| File | Contents |
|---|---|
| `review_absa_reference.csv.gz` | Local full-text archive: 39,496 deduplicated reviews, including 28,724 with historical ten-dimension labels |
| `manifest.json` | Counts, time range, SHA-256, and label semantics |
| `README.md` | Usage and limitations |

The current corpus reuses 16,538 previously generated historical labels. Full text, publication time, series, purchase context, and platform ratings stay in the local archive; the public repository contains de-identified labels and aggregates.

A historical label of `0` cannot reliably separate “not mentioned,” “neutral,” and parser fallback. It is not used directly as mention ground truth; the current feature table uses a separate uniform mention flag.

## 7. Reproduction

Create and activate the environment from the root `environment.yml`, then run scripts normally:

```bash
conda activate nlp-sentiment
python scripts/<script>.py
```

The main review and modeling sequence is:

```text
18  build target-series corpus
21  audit multi-source quality
25  validate temporal availability
27  build platform-rating and lexicon features
28  aggregate local review monthly features
30  backfill missing labels (optional, requires an API)
31  merge review-level labels
32  aggregate leakage-safe monthly features
33  run the 371-series forecast ablation
34  analyze robustness
35  build user-needs and risk monitoring outputs
36  validate the cold-start method
37  generate the report notebooks
```

Source-platform data remains subject to the respective owners' rights. Data in this directory is included only for learning, research, and project demonstration.
